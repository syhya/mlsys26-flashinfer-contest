#include <stdio.h>
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <device_launch_parameters.h>
#include <cstdint>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

struct FusedGatingData
{
    // input
    void* routing_logits; // [seq_len, 256]
    void* routing_bias; // [256]
    float routing_scaling_factor;

    // output
    void* routing_idx; // [seq_len, 8]
    void* routing_weights; // [seq_len, 8]
};

static constexpr int NUM_EXPERTS = 256;
static constexpr int NUM_SELECTED_EXPERTS = 8;
static constexpr int NUM_EXPERT_GROUPS = 8;
static constexpr int NUM_SELECTED_GROUPS = 4;
static constexpr int ROW_KERNEL_LONG_SEQ_CUTOVER = 8192;
static constexpr int ROW_KERNEL_MID_SEQ_SPECIALIZATION = 901;


// Grid: one block per token.
// Block: one thread per expert candidate.
__global__ void fusedGatingKernel(
    FusedGatingData data
) {
    __shared__ __nv_bfloat16 smem_bias[NUM_EXPERTS];
    __shared__ float smem_logits_with_sigmoid_bias[NUM_EXPERTS];
    __shared__ float smem_group_sums[NUM_EXPERT_GROUPS]; // Cross-warp reduction buffer.

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;


    // 1. sigmoid + bias
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x; // One expert score for the current token.
    smem_bias[threadIdx.x] = ((__nv_bfloat16*)data.routing_bias)[threadIdx.x];
    float logit = ((float*)data.routing_logits)[global_idx];
    logit = 1.0f / (1.0f + expf(-logit));
    logit += __bfloat162float(smem_bias[threadIdx.x]);
    smem_logits_with_sigmoid_bias[threadIdx.x] = logit;

    // 2. Compute the warp-local top-2 values inside each expert group.
    float top2_m1 = logit;
    float top2_m2 = -FLT_MAX;
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_m1 = __shfl_xor_sync(0xffffffff, top2_m1, mask);
        float other_m2 = __shfl_xor_sync(0xffffffff, top2_m2, mask);


        if (other_m1 > top2_m1) {
            // top2_m1 = other_m1;
            top2_m2 = max(top2_m1, other_m2);
            top2_m1 = other_m1; // Update order matters so top2_m2 still sees the old top-1.
        } else if (other_m1 > top2_m2) {
            top2_m2 = other_m1;
        }

    }

    // sum
    float top2_sum = top2_m1 + top2_m2;
    if (lane_id == 0) {
        smem_group_sums[warp_id] = top2_sum;
    }

    __syncthreads();

    int selected_groups_idx[NUM_SELECTED_GROUPS];

    int selected_group_expert_idx[NUM_SELECTED_GROUPS];
    float selected_group_expert_score[NUM_SELECTED_GROUPS];

    int top_expert_idx[NUM_SELECTED_EXPERTS];
    float top_expert_score[NUM_SELECTED_EXPERTS];

    if (warp_id == 0) {
        // 3. Let lane 0 of warp 0 select the top-4 expert groups.
        if (lane_id == 0) {
            float selected_groups_sums[NUM_SELECTED_GROUPS];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                selected_groups_idx[i] = -1;
                selected_groups_sums[i] = -FLT_MAX;
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float cur_idx = i;
                float cur_sum = smem_group_sums[i];

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (cur_sum > selected_groups_sums[j]) {
                        for (int k = 3; k > j; k--) {
                            selected_groups_idx[k] = selected_groups_idx[k - 1];
                            selected_groups_sums[k] = selected_groups_sums[k - 1];
                        }
                        selected_groups_idx[j] = cur_idx;
                        selected_groups_sums[j] = cur_sum;
                        break;
                    }
                }
            }
        }
        // Broadcast the selected group ids from lane 0.
        #pragma unroll
        for (int i = 0; i < NUM_SELECTED_GROUPS; i++) {
            selected_groups_idx[i] = __shfl_sync(0xffffffff, selected_groups_idx[i], 0);
        }

        // One warp covers 4 candidate groups x 32 experts = 128 experts.
        #pragma unroll
        for (int i = 0; i < NUM_SELECTED_GROUPS; i++) {  // bound of params.mNumLimitedGroups
            auto groupIdx= selected_groups_idx[i];
            selected_group_expert_idx[i] = groupIdx * NUM_EXPERTS / NUM_EXPERT_GROUPS + lane_id;
            selected_group_expert_score[i] = smem_logits_with_sigmoid_bias[selected_group_expert_idx[i]];
        }


        // 4. Pick the top-8 experts from the 128 candidates.
        // 4.1. Sort the 4 local candidates inside each lane.
        #pragma unroll
        for (int i = 0; i < NUM_SELECTED_GROUPS; i++) {
            for (int j = i + 1; j < NUM_SELECTED_GROUPS; j++) {
                if (selected_group_expert_score[j] > selected_group_expert_score[i]) {
                    float tmp_score = selected_group_expert_score[i];
                    selected_group_expert_score[i] = selected_group_expert_score[j];
                    selected_group_expert_score[j] = tmp_score;

                    int tmp_idx = selected_group_expert_idx[i];
                    selected_group_expert_idx[i] = selected_group_expert_idx[j];
                    selected_group_expert_idx[j] = tmp_idx;
                }
            }
        }
        // 4.2. Initialize the lane-local top-8 buffer and pad unused slots with -FLT_MAX.
        float thread_scores[NUM_SELECTED_EXPERTS];
        int thread_indices[NUM_SELECTED_EXPERTS];
        #pragma unroll
        for (int i = 0; i < NUM_SELECTED_GROUPS; ++i) {
            thread_scores[i] = selected_group_expert_score[i];
            thread_indices[i] = selected_group_expert_idx[i];
        }
        #pragma unroll
        for (int i = NUM_SELECTED_GROUPS; i < NUM_SELECTED_EXPERTS; ++i) {
            thread_scores[i] = -FLT_MAX;
            thread_indices[i] = -1;
        }
        // 4.3. Merge lane-local candidates into a warp-wide top-8 list.
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_scores[NUM_SELECTED_EXPERTS];
            int other_indices[NUM_SELECTED_EXPERTS];

            #pragma unroll
            for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
                other_scores[i] = __shfl_down_sync(0xffffffff, thread_scores[i], offset);
                other_indices[i] = __shfl_down_sync(0xffffffff, thread_indices[i], offset);
            }

            // Merge two sorted top-8 lists.
            float merged_scores[NUM_SELECTED_EXPERTS];
            int merged_indices[NUM_SELECTED_EXPERTS];
            int p1 = 0, p2 = 0;
            #pragma unroll
            for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
                if (thread_scores[p1] >= other_scores[p2]) {
                    merged_scores[i] = thread_scores[p1];
                    merged_indices[i] = thread_indices[p1];
                    p1++;
                } else {
                    merged_scores[i] = other_scores[p2];
                    merged_indices[i] = other_indices[p2];
                    p2++;
                }
            }

            // Write the merged state back into the current lane registers.
            #pragma unroll
            for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
                thread_scores[i] = merged_scores[i];
                thread_indices[i] = merged_indices[i];
            }
        }
        // 4.4. Broadcast the final top-8 expert ids and scores.
        #pragma unroll
        for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
            top_expert_score[i] = __shfl_sync(0xffffffff, thread_scores[i], 0);
            top_expert_idx[i] = __shfl_sync(0xffffffff, thread_indices[i], 0);
        }

        // 5. The first 8 lanes normalize routing weights and write the outputs.
        if (lane_id < NUM_SELECTED_EXPERTS) {
            auto selected_expert = top_expert_idx[lane_id];
            auto selected_score = top_expert_score[lane_id] - __bfloat162float(smem_bias[selected_expert]);

            float score_sum = selected_score;
            score_sum += __shfl_xor_sync(0xff, score_sum, 1);
            score_sum += __shfl_xor_sync(0xff, score_sum, 2);
            score_sum += __shfl_xor_sync(0xff, score_sum, 4);

            auto final_score = selected_score * data.routing_scaling_factor / score_sum;

            int write_idx = blockIdx.x * NUM_SELECTED_EXPERTS + lane_id;
            ((int*)data.routing_idx)[write_idx] = selected_expert;
            ((float*)data.routing_weights)[write_idx] = final_score;
        }
    }

}


void launchFusedGatingKernel(
    void* routing_logits,
    void* routing_bias,
    float routing_scaling_factor,
    void* routing_idx,
    void* routing_weights,
    int seq_len
) {
    FusedGatingData data;
    data.routing_logits = routing_logits;
    data.routing_bias = routing_bias;
    data.routing_scaling_factor = routing_scaling_factor;
    data.routing_idx = routing_idx;
    data.routing_weights = routing_weights;

    int threads_per_block = NUM_EXPERTS; // 256
    int num_blocks = seq_len; // One block per token.

    fusedGatingKernel<<<num_blocks, threads_per_block>>>(data);
}


__global__ void countExpertKernel(
    const int* __restrict__ routing_idx, // [seq_len, 8]
    int* __restrict__ expert_counts,    // [32]
    int seq_len,
    int local_expert_offset
) {
    __shared__ int smem_counts[32];

    int tid = threadIdx.x;
    if (tid < 32) {
        smem_counts[tid] = 0;
    }
    __syncthreads();

    // Each thread processes one token.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int e_id = routing_idx[idx * 8 + k];
            if (e_id >= local_expert_offset && e_id < local_expert_offset + 32) {
                atomicAdd(&smem_counts[e_id - local_expert_offset], 1);
            }
        }
    }
    __syncthreads();

    // Accumulate block-local counts into global memory.
    if (tid < 32) {
        if (smem_counts[tid] > 0) {
            atomicAdd(&expert_counts[tid], smem_counts[tid]);
        }
    }
}

__global__ void exclusiveScan32Kernel(
    int* __restrict__ expert_counts, // [32]
    int* __restrict__ expert_offsets,      // [32]
    int* __restrict__ total_tokens         // [1]
) {
    __shared__ int temp[32];
    const int tid = threadIdx.x;

    if (tid < 32) {
        temp[tid] = expert_counts[tid];
    }
    __syncthreads();

    // In-place inclusive scan for fixed size 32.
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int add_val = 0;
        if (tid >= offset) {
            add_val = temp[tid - offset];
        }
        __syncthreads();
        if (tid < 32) {
            temp[tid] += add_val;
        }
        __syncthreads();
    }

    if (tid < 32) {
        expert_offsets[tid] = (tid == 0) ? 0 : temp[tid - 1];
        expert_counts[tid] = expert_offsets[tid];
    }
    if (tid == 31) {
        expert_offsets[32] = temp[31];
        total_tokens[0] = temp[31];
    }
}


__global__ void permuteKernel(
    const int* __restrict__ routing_idx,     // [seq_len, 8]
    const float* __restrict__ routing_weight,   // [seq_len, 8]
    int* __restrict__ expert_offsets,       // [32] - already converted to exclusive offsets before entry.
    int* __restrict__ out_token_idx,        // [total_tokens]
    float* __restrict__ out_weights,        // [total_tokens]
    int* __restrict__ token2permuted_idx,     // [seq_len * 8]
    int* __restrict__ token_counts,
    int seq_len,
    int local_expert_offset
) {
    // Each thread processes one token.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;

    int expert_cnt = 0;
    for (int k = 0; k < 8; ++k) {
        int e_id = routing_idx[idx * 8 + k];
        if (e_id >= local_expert_offset && e_id < local_expert_offset + 32) {
            float w = routing_weight[idx * 8 + k];
            int rel_id = e_id - local_expert_offset;

            // Reserve a global slot for this routed token.
            int write_pos = atomicAdd(&expert_offsets[rel_id], 1);

            // Write the permutation mapping tables.
            out_token_idx[write_pos] = idx;
            out_weights[write_pos] = w;

            token2permuted_idx[idx * 8 + expert_cnt] = write_pos;
            expert_cnt++;
        }
    }
    token_counts[idx] = expert_cnt;
}

__global__ void countScanPermuteKernel(
    const int* __restrict__ routing_idx,      // [seq_len, 8]
    const float* __restrict__ routing_weight, // [seq_len, 8]
    int* __restrict__ expert_counts,          // [32] (kept as prefix offsets for compatibility)
    int* __restrict__ expert_offsets,         // [33]
    int* __restrict__ total_tokens,           // [1]
    int* __restrict__ out_token_idx,          // [total_tokens]
    float* __restrict__ out_weights,          // [total_tokens]
    int* __restrict__ token2permuted_idx,     // [seq_len * 8]
    int* __restrict__ token_counts,
    int seq_len,
    int local_expert_offset
) {
    __shared__ int smem_counts[32];
    __shared__ int smem_cursors[32];

    const int tid = threadIdx.x;
    if (tid < 32) {
        smem_counts[tid] = 0;
    }
    __syncthreads();

    // Pass-1: count local expert tokens.
    for (int idx = tid; idx < seq_len; idx += blockDim.x) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int e_id = routing_idx[idx * 8 + k];
            if (e_id >= local_expert_offset && e_id < local_expert_offset + 32) {
                atomicAdd(&smem_counts[e_id - local_expert_offset], 1);
            }
        }
    }
    __syncthreads();

    // Build exclusive offsets and initialize per-expert cursors.
    if (tid == 0) {
        int prefix = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            expert_offsets[i] = prefix;
            expert_counts[i] = prefix;
            smem_cursors[i] = prefix;
            prefix += smem_counts[i];
        }
        expert_offsets[32] = prefix;
        total_tokens[0] = prefix;
    }
    __syncthreads();

    // Pass-2: permute into compact buffers.
    for (int idx = tid; idx < seq_len; idx += blockDim.x) {
        int expert_cnt = 0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int e_id = routing_idx[idx * 8 + k];
            if (e_id >= local_expert_offset && e_id < local_expert_offset + 32) {
                const int rel_id = e_id - local_expert_offset;
                const int write_pos = atomicAdd(&smem_cursors[rel_id], 1);
                out_token_idx[write_pos] = idx;
                out_weights[write_pos] = routing_weight[idx * 8 + k];
                token2permuted_idx[idx * 8 + expert_cnt] = write_pos;
                expert_cnt++;
            }
        }
        token_counts[idx] = expert_cnt;
    }
}


__global__ void moe_permute_copy_fp8_with_scale_kernel(
    const __nv_fp8_e4m3* __restrict__ input,        // [S, 7168]
    const float* __restrict__ input_scale,          // [56, S]
    const int* __restrict__ permuted_token_idx,     // [TotalValidTokens]
    __nv_fp8_e4m3* __restrict__ output,             // [TotalValidTokens, 7168]
    float* __restrict__ output_scale,               // [56, TotalValidTokens]
    int* __restrict__ offset,                     // [33]
    int input_seq_len
) {
    const int HIDDEN_DIM = 7168;
    const int NUM_HIDDEN_BLOCKS = 56;
    const int VEC_SIZE = 16; // 128 bit / 8 bit = 16 elements per uint4

    const int total_valid = offset[32];
    const int scale_stride = input_seq_len * 8;

    for (int out_row_idx = blockIdx.x; out_row_idx < total_valid; out_row_idx += gridDim.x) {
        int src_row_idx = permuted_token_idx[out_row_idx];

        const uint4* src_ptr4 = reinterpret_cast<const uint4*>(input + src_row_idx * HIDDEN_DIM);
        uint4* dst_ptr4 = reinterpret_cast<uint4*>(output + out_row_idx * HIDDEN_DIM);

        for (int v = threadIdx.x; v < HIDDEN_DIM / VEC_SIZE; v += blockDim.x) {
            dst_ptr4[v] = src_ptr4[v];
        }

        for (int hb = threadIdx.x; hb < NUM_HIDDEN_BLOCKS; hb += blockDim.x) {
            output_scale[hb * scale_stride + out_row_idx] =
                input_scale[hb * input_seq_len + src_row_idx];
        }
    }
}

// Long-band copy uses one seq_len-sized grid and walks only the valid compacted
// rows, avoiding the 8x overlaunch from the generic route-slot kernel.
__global__ void moe_permute_copy_fp8_with_scale_row_kernel(
    const __nv_fp8_e4m3* __restrict__ input,        // [S, 7168]
    const float* __restrict__ input_scale,          // [56, S]
    const int* __restrict__ permuted_token_idx,     // [TotalValidTokens]
    __nv_fp8_e4m3* __restrict__ output,             // [TotalValidTokens, 7168]
    float* __restrict__ output_scale,               // [56, TotalValidTokens]
    int* __restrict__ offset,                       // [33]
    int input_seq_len
) {
    constexpr int HIDDEN_DIM = 7168;
    constexpr int NUM_HIDDEN_BLOCKS = 56;
    constexpr int VEC_SIZE = 16; // 128 bit / 8 bit = 16 elements per uint4

    const int total_valid = offset[32];
    const int scale_stride = input_seq_len * 8;

    for (int out_row_idx = blockIdx.x; out_row_idx < total_valid; out_row_idx += gridDim.x) {
        const int src_row_idx = permuted_token_idx[out_row_idx];

        const uint4* src_ptr4 = reinterpret_cast<const uint4*>(input + src_row_idx * HIDDEN_DIM);
        uint4* dst_ptr4 = reinterpret_cast<uint4*>(output + out_row_idx * HIDDEN_DIM);

        for (int v = threadIdx.x; v < HIDDEN_DIM / VEC_SIZE; v += blockDim.x) {
            dst_ptr4[v] = src_ptr4[v];
        }

        for (int hb = threadIdx.x; hb < NUM_HIDDEN_BLOCKS; hb += blockDim.x) {
            output_scale[hb * scale_stride + out_row_idx] =
                input_scale[hb * input_seq_len + src_row_idx];
        }
    }
}


void launchCountExpertAndOffsetsKernel(
    void* routing_idx,
    void* expert_counts,
    void* expert_offsets,
    void* total_tokens,
    int seq_len,
    int local_expert_offset
) {
    cudaMemsetAsync(expert_counts, 0, 32 * sizeof(int));
    constexpr int threads_per_block = 256;
    const int num_blocks = (seq_len + threads_per_block - 1) / threads_per_block;
    countExpertKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const int*>(routing_idx),
        static_cast<int*>(expert_counts),
        seq_len,
        local_expert_offset
    );

    exclusiveScan32Kernel<<<1, 32>>>(
        static_cast<int*>(expert_counts),
        static_cast<int*>(expert_offsets),
        static_cast<int*>(total_tokens)
    );
}

void launchPermuteKernel(
    void* routing_idx,
    void* routing_weight,
    void* expert_offsets,
    void* out_token_idx,
    void* out_weights,
    void* token2permuted_idx,     // [seq_len * 8]
    void* token_counts,
    int seq_len,
    int local_expert_offset
) {
    constexpr int threads_per_block = 256;
    const int num_blocks = (seq_len + threads_per_block - 1) / threads_per_block;
    permuteKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const int*>(routing_idx),
        static_cast<const float*>(routing_weight),
        static_cast<int*>(expert_offsets),
        static_cast<int*>(out_token_idx),
        static_cast<float*>(out_weights),
        static_cast<int*>(token2permuted_idx),
        static_cast<int*>(token_counts),
        seq_len,
        local_expert_offset
    );
}

void launchCountScanPermuteKernel(
    void* routing_idx,
    void* routing_weight,
    void* expert_counts,
    void* expert_offsets,
    void* total_tokens,
    void* out_token_idx,
    void* out_weights,
    void* token2permuted_idx,     // [seq_len * 8]
    void* token_counts,
    int seq_len,
    int local_expert_offset
) {
    constexpr int threads_per_block = 256;
    countScanPermuteKernel<<<1, threads_per_block>>>(
        static_cast<const int*>(routing_idx),
        static_cast<const float*>(routing_weight),
        static_cast<int*>(expert_counts),
        static_cast<int*>(expert_offsets),
        static_cast<int*>(total_tokens),
        static_cast<int*>(out_token_idx),
        static_cast<float*>(out_weights),
        static_cast<int*>(token2permuted_idx),
        static_cast<int*>(token_counts),
        seq_len,
        local_expert_offset
    );
}

void launchMoePermuteCopyFp8WithScaleKernel(
    void* input,
    void* input_scale,
    void* permuted_token_idx,
    void* output,
    void* output_scale,
    void* offset,
    int input_seq_len
) {
    constexpr int threads_per_block = 256;
    int num_blocks = input_seq_len;
    if (input_seq_len <= 2048) {
        num_blocks = input_seq_len * 8;
    }
    moe_permute_copy_fp8_with_scale_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const __nv_fp8_e4m3*>(input),
        static_cast<const float*>(input_scale),
        static_cast<const int*>(permuted_token_idx),
        static_cast<__nv_fp8_e4m3*>(output),
        static_cast<float*>(output_scale),
        static_cast<int*>(offset),
        input_seq_len
    );
}


__global__ void build_padded_offsets_kernel(
    const int* __restrict__ offsets,
    int* __restrict__ padded_offsets
) {
    __shared__ int padded_counts[32];
    const int tid = threadIdx.x;
    if (tid < 32) {
        const int count = offsets[tid + 1] - offsets[tid];
        // FlashInfer's SM100 grouped CUTLASS binding requires 4-aligned M
        // offsets and is not validated for empty groups. Give zero-token
        // experts four zero rows so every problem has a launchable M.
        padded_counts[tid] = (count == 0) ? 4 : ((count + 3) & ~3);
    }
    __syncthreads();

    if (tid < 33) {
        int prefix = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (tid == i) {
                padded_offsets[i] = prefix;
            }
            prefix += padded_counts[i];
        }
        if (tid == 32) {
            padded_offsets[32] = prefix;
        }
    }
}

__global__ void pad_fp8_hidden_scale_kernel(
    const __nv_fp8_e4m3* __restrict__ src_hidden,
    const float* __restrict__ src_scale,
    const int* __restrict__ offsets,
    const int* __restrict__ padded_offsets,
    __nv_fp8_e4m3* __restrict__ dst_hidden,
    float* __restrict__ dst_scale,
    int src_scale_stride,
    int dst_scale_stride,
    int dst_capacity
) {
    constexpr int HIDDEN_DIM = 7168;
    constexpr int NUM_HIDDEN_BLOCKS = 56;
    constexpr int VEC_SIZE = 16;

    const int dst_row = blockIdx.x;
    if (dst_row >= dst_capacity || dst_row >= padded_offsets[32]) {
        return;
    }

    int expert = 0;
    #pragma unroll
    for (int e = 0; e < 32; ++e) {
        if (dst_row >= padded_offsets[e] && dst_row < padded_offsets[e + 1]) {
            expert = e;
        }
    }

    const int rel = dst_row - padded_offsets[expert];
    const int count = offsets[expert + 1] - offsets[expert];
    const bool valid = rel < count;
    const int src_row = offsets[expert] + rel;

    const uint4* src_ptr4 = reinterpret_cast<const uint4*>(src_hidden + src_row * HIDDEN_DIM);
    uint4* dst_ptr4 = reinterpret_cast<uint4*>(dst_hidden + dst_row * HIDDEN_DIM);
    const uint4 zero4 = make_uint4(0, 0, 0, 0);

    for (int v = threadIdx.x; v < HIDDEN_DIM / VEC_SIZE; v += blockDim.x) {
        dst_ptr4[v] = valid ? src_ptr4[v] : zero4;
    }

    for (int hb = threadIdx.x; hb < NUM_HIDDEN_BLOCKS; hb += blockDim.x) {
        dst_scale[hb * dst_scale_stride + dst_row] =
            valid ? src_scale[hb * src_scale_stride + src_row] : 1.0f;
    }
}

__global__ void transpose_scale_mn_to_k_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int rows,
    int scale_blocks
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * scale_blocks;
    if (idx >= total) {
        return;
    }

    const int row = idx / scale_blocks;
    const int block = idx - row * scale_blocks;
    dst[row * scale_blocks + block] = src[block * rows + row];
}

__global__ void unpad_bf16_rows_kernel(
    const __nv_bfloat16* __restrict__ src,
    const int* __restrict__ offsets,
    const int* __restrict__ padded_offsets,
    __nv_bfloat16* __restrict__ dst,
    int dst_capacity
) {
    constexpr int HIDDEN_DIM = 7168;
    constexpr int VEC_SIZE_BF16 = 8; // 16 bytes / 2 bytes

    const int dst_row = blockIdx.x;
    if (dst_row >= dst_capacity || dst_row >= offsets[32]) {
        return;
    }

    int expert = 0;
    #pragma unroll
    for (int e = 0; e < 32; ++e) {
        if (dst_row >= offsets[e] && dst_row < offsets[e + 1]) {
            expert = e;
        }
    }

    const int rel = dst_row - offsets[expert];
    const int src_row = padded_offsets[expert] + rel;

    const uint4* src_ptr4 = reinterpret_cast<const uint4*>(src + src_row * HIDDEN_DIM);
    uint4* dst_ptr4 = reinterpret_cast<uint4*>(dst + dst_row * HIDDEN_DIM);
    for (int v = threadIdx.x; v < HIDDEN_DIM / VEC_SIZE_BF16; v += blockDim.x) {
        dst_ptr4[v] = src_ptr4[v];
    }
}

__global__ void scale_bf16_rows_by_weight_kernel(
    __nv_bfloat16* __restrict__ data,
    const float* __restrict__ weights,
    int rows
) {
    constexpr int HIDDEN_DIM = 7168;

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float weight = weights[row];
    __nv_bfloat16* row_ptr = data + row * HIDDEN_DIM;
    for (int col = threadIdx.x; col < HIDDEN_DIM; col += blockDim.x) {
        const float value = __bfloat162float(row_ptr[col]) * weight;
        row_ptr[col] = __float2bfloat16(value);
    }
}

void launchBuildPaddedOffsetsKernel(
    void* offsets,
    void* padded_offsets
) {
    build_padded_offsets_kernel<<<1, 64>>>(
        static_cast<const int*>(offsets),
        static_cast<int*>(padded_offsets)
    );
}

void launchPadFp8HiddenScaleKernel(
    void* src_hidden,
    void* src_scale,
    void* offsets,
    void* padded_offsets,
    void* dst_hidden,
    void* dst_scale,
    int src_scale_stride,
    int dst_scale_stride,
    int dst_capacity
) {
    constexpr int threads_per_block = 256;
    pad_fp8_hidden_scale_kernel<<<dst_capacity, threads_per_block>>>(
        static_cast<const __nv_fp8_e4m3*>(src_hidden),
        static_cast<const float*>(src_scale),
        static_cast<const int*>(offsets),
        static_cast<const int*>(padded_offsets),
        static_cast<__nv_fp8_e4m3*>(dst_hidden),
        static_cast<float*>(dst_scale),
        src_scale_stride,
        dst_scale_stride,
        dst_capacity
    );
}

void launchTransposeScaleMnToKKernel(
    void* src,
    void* dst,
    int rows,
    int scale_blocks
) {
    constexpr int threads_per_block = 256;
    const int total = rows * scale_blocks;
    const int blocks = (total + threads_per_block - 1) / threads_per_block;
    transpose_scale_mn_to_k_kernel<<<blocks, threads_per_block>>>(
        static_cast<const float*>(src),
        static_cast<float*>(dst),
        rows,
        scale_blocks
    );
}

void launchUnpadBf16RowsKernel(
    void* src,
    void* offsets,
    void* padded_offsets,
    void* dst,
    int dst_capacity
) {
    constexpr int threads_per_block = 256;
    unpad_bf16_rows_kernel<<<dst_capacity, threads_per_block>>>(
        static_cast<const __nv_bfloat16*>(src),
        static_cast<const int*>(offsets),
        static_cast<const int*>(padded_offsets),
        static_cast<__nv_bfloat16*>(dst),
        dst_capacity
    );
}

void launchScaleBf16RowsByWeightKernel(
    void* data,
    void* weights,
    int rows
) {
    constexpr int threads_per_block = 256;
    scale_bf16_rows_by_weight_kernel<<<rows, threads_per_block>>>(
        static_cast<__nv_bfloat16*>(data),
        static_cast<const float*>(weights),
        rows
    );
}



__global__ void scatter_add_kernel(
    const float* __restrict__ tmp_output, // [seq*8, 7168]
    const int* __restrict__ token_idx,    // [seq*8]
    float* __restrict__ output,           // [seq, 7168]
    int* offset
) {
    int num_valid = offset[32];
    int row_idx = blockIdx.x;
    if (row_idx >= num_valid) return;

    int target_row = token_idx[row_idx];

    int col_offset = threadIdx.x * 4;

    for (; col_offset < 7168; col_offset += blockDim.x * 4) {
        float4 val = reinterpret_cast<const float4*>(&tmp_output[row_idx * 7168 + col_offset])[0];

        float* target_ptr = &output[target_row * 7168 + col_offset];

        atomicAdd(&target_ptr[0], val.x);
        atomicAdd(&target_ptr[1], val.y);
        atomicAdd(&target_ptr[2], val.z);
        atomicAdd(&target_ptr[3], val.w);
    }
}

void launchScatterAddKernel(
    void* tmp_output,
    void* token_idx,
    void* output,
    void* offset,
    int seq_len
) {
    size_t zero_size = seq_len * 7168 * sizeof(float);
    cudaMemsetAsync(output, 0, zero_size);

    constexpr int threads_per_block = 256;
    const int num_blocks = seq_len * 8;
    scatter_add_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float*>(tmp_output),
        static_cast<const int*>(token_idx),
        static_cast<float*>(output),
        static_cast<int*>(offset)
    );
}

__global__ void act_quant_kernel(
    const __half* __restrict__ input, // [valid, 4096], fp16
    __nv_fp8_e4m3* __restrict__ output,      // [valid, 2048]
    float* __restrict__ scale,       // [2048 / 128, valid]
    int* __restrict__ offset,        // [33]
    int seq_len
) {
    constexpr int HIDDEN = 2048;
    constexpr int GROUP = 128;
    constexpr float FP8_E4M3_MAX = 448.0f;

    const int row = blockIdx.x;
    const int group_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int num_valid = offset[32];
    if (row >= num_valid || group_idx >= (HIDDEN / GROUP) || tid >= GROUP) {
        return;
    }

    const int col = group_idx * GROUP + tid;
    const int in_row_base = row * (HIDDEN * 2);
    const int out_row_base = row * HIDDEN;

    // Split the input into gate/value halves and compute input1 * SiLU(input2).
    const float x1 = __half2float(input[in_row_base + col]);
    const float x2 = __half2float(input[in_row_base + HIDDEN + col]);
    const float silu2 = x2 / (1.0f + expf(-x2));
    const float res = x1 * silu2;

    __shared__ float smem_abs_max[GROUP];
    smem_abs_max[tid] = fabsf(res);
    __syncthreads();

    for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_abs_max[tid] = fmaxf(smem_abs_max[tid], smem_abs_max[tid + stride]);
        }
        __syncthreads();
    }

    const float max_abs = smem_abs_max[0];
    const float group_scale = (max_abs > 0.0f) ? (max_abs / FP8_E4M3_MAX) : 1.0f;
    if (tid == 0) {
        scale[group_idx * seq_len * 8 + row] = group_scale;
    }

    float q = res / group_scale;
    q = fminf(fmaxf(q, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    output[out_row_base + col] = static_cast<__nv_fp8_e4m3>(q);
}

// Long-band activation quantization reuses one CTA across many valid rows.
// This avoids launching seq_len * 8 mostly empty CTAs once the routing stage
// has already compacted tokens, which was the dominant long-shape overhead.
__global__ void act_quant_row_kernel(
    const __half* __restrict__ input, // [valid, 4096], fp16
    __nv_fp8_e4m3* __restrict__ output,      // [valid, 2048]
    float* __restrict__ scale,       // [2048 / 128, valid]
    int* __restrict__ offset,        // [33]
    int seq_len
) {
    constexpr int HIDDEN = 2048;
    constexpr int GROUP = 128;
    constexpr int GROUPS_PER_BLOCK = 2;
    constexpr int BLOCK_COLS = GROUP * GROUPS_PER_BLOCK; // 256 output columns per CTA iteration.
    constexpr float FP8_E4M3_MAX = 448.0f;

    const int tid = threadIdx.x;
    const int num_valid = offset[32];
    if (tid >= GROUP) {
        return;
    }
    const int scale_stride = seq_len * 8;

    __shared__ float smem_res[BLOCK_COLS];
    __shared__ float smem_warp_max0[4];
    __shared__ float smem_warp_max1[4];
    __shared__ float smem_group_scale[GROUPS_PER_BLOCK];

    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    // The strided row loop lets a fixed seq_len-sized grid cover only the
    // valid compacted rows instead of materializing one CTA per route slot.
    for (int row = blockIdx.x; row < num_valid; row += gridDim.x) {
        const int in_row_base = row * (HIDDEN * 2);
        const int out_row_base = row * HIDDEN;

        #pragma unroll
        for (int pair_idx = 0; pair_idx < HIDDEN / BLOCK_COLS; ++pair_idx) {
            const int col0 = pair_idx * BLOCK_COLS + tid;
            const int col1 = col0 + GROUP;

            const float x10 = __half2float(input[in_row_base + col0]);
            const float x20 = __half2float(input[in_row_base + HIDDEN + col0]);
            const float x11 = __half2float(input[in_row_base + col1]);
            const float x21 = __half2float(input[in_row_base + HIDDEN + col1]);

            const float silu20 = x20 / (1.0f + __expf(-x20));
            const float silu21 = x21 / (1.0f + __expf(-x21));
            const float res0 = x10 * silu20;
            const float res1 = x11 * silu21;

            smem_res[tid] = res0;
            smem_res[GROUP + tid] = res1;

            float warp_max0 = fabsf(res0);
            float warp_max1 = fabsf(res1);
            #pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                warp_max0 = fmaxf(warp_max0, __shfl_down_sync(0xffffffff, warp_max0, delta));
                warp_max1 = fmaxf(warp_max1, __shfl_down_sync(0xffffffff, warp_max1, delta));
            }
            if (lane_id == 0) {
                smem_warp_max0[warp_id] = warp_max0;
                smem_warp_max1[warp_id] = warp_max1;
            }
            __syncthreads();

            if (tid < 4) {
                float max_group0 = smem_warp_max0[tid];
                float max_group1 = smem_warp_max1[tid];
                #pragma unroll
                for (int delta = 2; delta > 0; delta >>= 1) {
                    max_group0 = fmaxf(max_group0, __shfl_down_sync(0xF, max_group0, delta, 4));
                    max_group1 = fmaxf(max_group1, __shfl_down_sync(0xF, max_group1, delta, 4));
                }
                if (tid == 0) {
                    const float scale0 = (max_group0 > 0.0f) ? (max_group0 / FP8_E4M3_MAX) : 1.0f;
                    const float scale1 = (max_group1 > 0.0f) ? (max_group1 / FP8_E4M3_MAX) : 1.0f;
                    smem_group_scale[0] = scale0;
                    smem_group_scale[1] = scale1;
                    scale[(pair_idx * GROUPS_PER_BLOCK + 0) * scale_stride + row] = scale0;
                    scale[(pair_idx * GROUPS_PER_BLOCK + 1) * scale_stride + row] = scale1;
                }
            }
            __syncthreads();

            float q0 = smem_res[tid] / smem_group_scale[0];
            float q1 = smem_res[GROUP + tid] / smem_group_scale[1];
            q0 = fminf(fmaxf(q0, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            q1 = fminf(fmaxf(q1, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            output[out_row_base + col0] = static_cast<__nv_fp8_e4m3>(q0);
            output[out_row_base + col1] = static_cast<__nv_fp8_e4m3>(q1);
            __syncthreads();
        }
    }
}

__global__ void act_quant_warp_small_kernel(
    const __half* __restrict__ input,       // [valid, 4096]
    __nv_fp8_e4m3* __restrict__ output,     // [valid, 2048]
    float* __restrict__ scale,              // [16, valid-capacity]
    const int* __restrict__ offset,         // [33]
    int seq_len
) {
    constexpr int HIDDEN = 2048;
    constexpr int GROUP = 128;
    constexpr int GROUPS = HIDDEN / GROUP;
    constexpr float FP8_E4M3_MAX = 448.0f;

    const int row = blockIdx.x >> 2;
    const int group_base = (blockIdx.x & 3) << 2;
    const int num_valid = offset[32];
    if (row >= num_valid) {
        return;
    }

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int group_idx = group_base + warp_id;
    if (group_idx >= GROUPS) {
        return;
    }

    const int in_base0 = row * (HIDDEN * 2) + group_idx * GROUP;
    const int in_base1 = in_base0 + HIDDEN;
    const int2 val0 = reinterpret_cast<const int2*>(input + in_base0)[lane_id];
    const int2 val1 = reinterpret_cast<const int2*>(input + in_base1)[lane_id];
    const __half* h0 = reinterpret_cast<const __half*>(&val0);
    const __half* h1 = reinterpret_cast<const __half*>(&val1);

    float res[4];
    float local_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = __half2float(h0[i]);
        const float x1 = __half2float(h1[i]);
        const float silu = x1 / (1.0f + expf(-x1));
        res[i] = x0 * silu;
        local_max = fmaxf(local_max, fabsf(res[i]));
    }

    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, delta));
    }
    const float max_abs = __shfl_sync(0xffffffff, local_max, 0);
    const float group_scale = (max_abs > 0.0f) ? (max_abs / FP8_E4M3_MAX) : 1.0f;

    const int scale_stride = seq_len * 8;
    if (lane_id == 0) {
        scale[group_idx * scale_stride + row] = group_scale;
    }

    __nv_fp8_e4m3* out_ptr = output + row * HIDDEN + group_idx * GROUP + lane_id * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float q = res[i] / group_scale;
        q = fminf(fmaxf(q, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        out_ptr[i] = static_cast<__nv_fp8_e4m3>(q);
    }
}

__global__ void act_quant_warp_row_kernel(
    const __half* __restrict__ input,       // [valid, 4096]
    __nv_fp8_e4m3* __restrict__ output,     // [valid, 2048]
    float* __restrict__ scale,              // [16, valid-capacity]
    const int* __restrict__ offset,         // [33]
    int seq_len
) {
    constexpr int HIDDEN = 2048;
    constexpr int GROUP = 128;
    constexpr int GROUPS = HIDDEN / GROUP;
    constexpr float FP8_E4M3_MAX = 448.0f;

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    const int num_valid = offset[32];
    const int scale_stride = seq_len * 8;

    for (int row = blockIdx.x; row < num_valid; row += gridDim.x) {
        #pragma unroll
        for (int group_idx = warp_id; group_idx < GROUPS; group_idx += num_warps) {
            const int in_base0 = row * (HIDDEN * 2) + group_idx * GROUP;
            const int in_base1 = in_base0 + HIDDEN;
            const int2 val0 = reinterpret_cast<const int2*>(input + in_base0)[lane_id];
            const int2 val1 = reinterpret_cast<const int2*>(input + in_base1)[lane_id];
            const __half* h0 = reinterpret_cast<const __half*>(&val0);
            const __half* h1 = reinterpret_cast<const __half*>(&val1);

            float res[4];
            float local_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float x0 = __half2float(h0[i]);
                const float x1 = __half2float(h1[i]);
                const float silu = x1 / (1.0f + expf(-x1));
                res[i] = x0 * silu;
                local_max = fmaxf(local_max, fabsf(res[i]));
            }

            #pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, delta));
            }
            const float max_abs = __shfl_sync(0xffffffff, local_max, 0);
            const float group_scale = (max_abs > 0.0f) ? (max_abs / FP8_E4M3_MAX) : 1.0f;

            if (lane_id == 0) {
                scale[group_idx * scale_stride + row] = group_scale;
            }

            __nv_fp8_e4m3* out_ptr = output + row * HIDDEN + group_idx * GROUP + lane_id * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float q = res[i] / group_scale;
                q = fminf(fmaxf(q, -FP8_E4M3_MAX), FP8_E4M3_MAX);
                out_ptr[i] = static_cast<__nv_fp8_e4m3>(q);
            }
        }
    }
}

void launchActQuantKernel(
    void* input,
    void* output,
    void* scale,
    void* offset,
    int seq_len
) {
    constexpr int threads_per_block = 128;
    if (seq_len <= 128) {
        const dim3 grid_dim(seq_len * 8 * 4);
        act_quant_warp_small_kernel<<<grid_dim, threads_per_block>>>(
            static_cast<const __half*>(input),
            static_cast<__nv_fp8_e4m3*>(output),
            static_cast<float*>(scale),
            static_cast<const int*>(offset),
            seq_len
        );
        return;
    }

    int num_blocks = seq_len;
    if (seq_len <= 2048) {
        num_blocks = seq_len * 8;
    }
    const dim3 grid_dim(num_blocks);
    act_quant_warp_row_kernel<<<grid_dim, threads_per_block>>>(
        static_cast<const __half*>(input),
        static_cast<__nv_fp8_e4m3*>(output),
        static_cast<float*>(scale),
        static_cast<const int*>(offset),
        seq_len
    );
}

__global__ void reduce_add_kernel(
    const __nv_bfloat16* __restrict__ tmp_output, // [seq*8, 7168]
    const int* __restrict__ token2permuted_idx,     // [seq_len * 8]
    const int* __restrict__ token_counts,           // [seq_len]
    __nv_bfloat16* __restrict__ output           // [seq, 7168]
) {
    uint32_t token_id = blockIdx.x >> 3; // blockIdx.x / 8
    uint32_t inter_id = blockIdx.x & 7; // blockIdx.x % 8

    // Explicit 128-bit vectorized path: 1 x uint4 = 8 x bf16 = 4 x bf16x2.
    int col_bf16 = threadIdx.x * 8 + inter_id * 1024;
    // Each thread accumulates 8 bf16 values, so 256 threads cover 1024 bf16 values.
    if (col_bf16 < 7168) {
        __nv_bfloat162 val2[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            val2[j] = __halves2bfloat162(__float2bfloat16(0.0f), __float2bfloat16(0.0f));
        }
        for (int i = 0; i < token_counts[token_id]; ++i) {
            int permuted_idx = token2permuted_idx[token_id * 8 + i];
            uint4 tmp_pack = reinterpret_cast<const uint4*>(
                &tmp_output[permuted_idx * 7168 + col_bf16]
            )[0];
            const __nv_bfloat162* tmp_vals2 = reinterpret_cast<const __nv_bfloat162*>(&tmp_pack);

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                val2[j] = __hadd2(val2[j], tmp_vals2[j]);
            }
        }

        uint4 out_pack;
        __nv_bfloat162* out_vals2 = reinterpret_cast<__nv_bfloat162*>(&out_pack);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            out_vals2[j] = val2[j];
        }

        reinterpret_cast<uint4*>(&output[token_id * 7168 + col_bf16])[0] = out_pack;
    }
}

__global__ void reduce_add_dynamic_row_kernel(
    const __nv_bfloat16* __restrict__ tmp_output, // [seq*8, 7168]
    const int* __restrict__ token2permuted_idx,   // [seq_len * 8]
    const int* __restrict__ token_counts,         // [seq_len]
    __nv_bfloat16* __restrict__ output            // [seq, 7168]
) {
    constexpr int HIDDEN = 7168;
    constexpr int TOPK = 8;
    constexpr int VEC_BF16 = 8;

    const int token_id = blockIdx.x;
    const int count = token_counts[token_id];

    for (int col_bf16 = threadIdx.x * VEC_BF16;
         col_bf16 < HIDDEN;
         col_bf16 += blockDim.x * VEC_BF16) {
        __nv_bfloat162 acc[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[j] = __halves2bfloat162(__float2bfloat16(0.0f), __float2bfloat16(0.0f));
        }

        #pragma unroll
        for (int i = 0; i < TOPK; ++i) {
            if (i < count) {
                const int row = token2permuted_idx[token_id * TOPK + i];
                const uint4 pack = reinterpret_cast<const uint4*>(
                    &tmp_output[row * HIDDEN + col_bf16]
                )[0];
                const __nv_bfloat162* vals = reinterpret_cast<const __nv_bfloat162*>(&pack);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[j] = __hadd2(acc[j], vals[j]);
                }
            }
        }

        uint4 out_pack;
        __nv_bfloat162* out_vals = reinterpret_cast<__nv_bfloat162*>(&out_pack);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            out_vals[j] = acc[j];
        }
        reinterpret_cast<uint4*>(&output[token_id * HIDDEN + col_bf16])[0] = out_pack;
    }
}

// Specialize the common small route counts so the long-band reduction avoids
// the generic counted loop for the hot 0..4 cases.
template <int ROUTES>
__device__ __forceinline__ uint4 reduce_add_row_fixed_count(
    const __nv_bfloat16* __restrict__ tmp_output,
    const int* __restrict__ permuted_idx,
    int col_bf16
) {
    constexpr int HIDDEN = 7168;
    if constexpr (ROUTES == 0) {
        return make_uint4(0u, 0u, 0u, 0u);
    }

    const uint4 first_pack = reinterpret_cast<const uint4*>(
        &tmp_output[permuted_idx[0] * HIDDEN + col_bf16]
    )[0];
    if constexpr (ROUTES == 1) {
        return first_pack;
    }

    __nv_bfloat162 val2[4];
    const __nv_bfloat162* first_vals2 = reinterpret_cast<const __nv_bfloat162*>(&first_pack);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        val2[j] = first_vals2[j];
    }

    #pragma unroll
    for (int i = 1; i < ROUTES; ++i) {
        const uint4 tmp_pack = reinterpret_cast<const uint4*>(
            &tmp_output[permuted_idx[i] * HIDDEN + col_bf16]
        )[0];
        const __nv_bfloat162* tmp_vals2 = reinterpret_cast<const __nv_bfloat162*>(&tmp_pack);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            val2[j] = __hadd2(val2[j], tmp_vals2[j]);
        }
    }

    uint4 out_pack;
    __nv_bfloat162* out_vals2 = reinterpret_cast<__nv_bfloat162*>(&out_pack);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        out_vals2[j] = val2[j];
    }
    return out_pack;
}

__device__ __forceinline__ uint4 reduce_add_row_dynamic_count(
    const __nv_bfloat16* __restrict__ tmp_output,
    const int* __restrict__ permuted_idx,
    int count,
    int col_bf16
) {
    constexpr int HIDDEN = 7168;
    __nv_bfloat162 val2[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        val2[j] = __halves2bfloat162(__float2bfloat16(0.0f), __float2bfloat16(0.0f));
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i < count) {
            const uint4 tmp_pack = reinterpret_cast<const uint4*>(
                &tmp_output[permuted_idx[i] * HIDDEN + col_bf16]
            )[0];
            const __nv_bfloat162* tmp_vals2 = reinterpret_cast<const __nv_bfloat162*>(&tmp_pack);

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                val2[j] = __hadd2(val2[j], tmp_vals2[j]);
            }
        }
    }

    uint4 out_pack;
    __nv_bfloat162* out_vals2 = reinterpret_cast<__nv_bfloat162*>(&out_pack);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        out_vals2[j] = val2[j];
    }
    return out_pack;
}

// The long-band reduction switches on the routed count once per token and then
// reuses the specialized vectorized accumulation path across the hidden axis.
__global__ void reduce_add_row_kernel(
    const __nv_bfloat16* __restrict__ tmp_output, // [seq*8, 7168]
    const int* __restrict__ token2permuted_idx,   // [seq_len * 8]
    const int* __restrict__ token_counts,         // [seq_len]
    __nv_bfloat16* __restrict__ output            // [seq, 7168]
) {
    constexpr int HIDDEN = 7168;
    constexpr int MAX_TOPK = 8;
    constexpr int VEC_BF16 = 8;

    const int token_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int base_col = tid * VEC_BF16;

    __shared__ int smem_count;
    __shared__ int smem_permuted_idx[MAX_TOPK];

    if (tid == 0) {
        smem_count = token_counts[token_id];
    }
    if (tid < MAX_TOPK) {
        smem_permuted_idx[tid] = token2permuted_idx[token_id * MAX_TOPK + tid];
    }
    __syncthreads();

    const int count = smem_count;
    for (int col_bf16 = base_col; col_bf16 < HIDDEN; col_bf16 += blockDim.x * VEC_BF16) {
        uint4 out_pack;
        switch (count) {
            case 0:
                out_pack = reduce_add_row_fixed_count<0>(tmp_output, smem_permuted_idx, col_bf16);
                break;
            case 1:
                out_pack = reduce_add_row_fixed_count<1>(tmp_output, smem_permuted_idx, col_bf16);
                break;
            case 2:
                out_pack = reduce_add_row_fixed_count<2>(tmp_output, smem_permuted_idx, col_bf16);
                break;
            case 3:
                out_pack = reduce_add_row_fixed_count<3>(tmp_output, smem_permuted_idx, col_bf16);
                break;
            case 4:
                out_pack = reduce_add_row_fixed_count<4>(tmp_output, smem_permuted_idx, col_bf16);
                break;
            default:
                out_pack = reduce_add_row_dynamic_count(tmp_output, smem_permuted_idx, count, col_bf16);
                break;
        }
        reinterpret_cast<uint4*>(&output[token_id * HIDDEN + col_bf16])[0] = out_pack;
    }
}

void launchReduceAddKernel(
    void* tmp_output,
    void* token2permuted_idx,
    void* token_counts,
    void* output,
    int seq_len
) {
    if (seq_len > 128) {
        constexpr int threads_per_block = 256;
        reduce_add_dynamic_row_kernel<<<seq_len, threads_per_block>>>(
            static_cast<const __nv_bfloat16*>(tmp_output),
            static_cast<const int*>(token2permuted_idx),
            static_cast<const int*>(token_counts),
            static_cast<__nv_bfloat16*>(output)
        );
        return;
    }

    constexpr int threads_per_block = 256;
    const int num_blocks = seq_len * 8;
    reduce_add_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const __nv_bfloat16*>(tmp_output),
        static_cast<const int*>(token2permuted_idx),
        static_cast<const int*>(token_counts),
        static_cast<__nv_bfloat16*>(output)
    );
}


__global__ void cooperative_kernel(
    FusedGatingData gating_data,
    const __nv_fp8_e4m3* __restrict__ input,        // [S, 7168]
    const float* __restrict__ input_scale,          // [56, S]
    int* __restrict__ expert_cnts,                // [32]
    int* __restrict__ expert_offsets,               // [33]
    int* __restrict__ out_token_idx,                // [total_tokens]
    float* __restrict__ out_weights,                // [total_tokens]
    __nv_fp8_e4m3* __restrict__ output,             // [total_tokens, 7168]
    float* __restrict__ output_scale,               // [56, total_tokens]
    int* __restrict__ token2permuted_idx,     // [seq_len * 8]
    int* __restrict__ token_counts,           // [seq_len] - number of experts per token for later reduce bookkeeping.
    int seq_len,
    int local_expert_offset
) {
    constexpr int HIDDEN_DIM = 7168;
    constexpr int NUM_HIDDEN_BLOCKS = 56;
    constexpr int VEC_SIZE = 16; // 128 bit / 8 bit = 16 fp8 values per uint4

    cg::grid_group grid = cg::this_grid();

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    // Stage-1: fused gating for all tokens assigned to this block in grid-stride order.
    for (int token_idx = blockIdx.x; token_idx < seq_len; token_idx += gridDim.x) {
        __shared__ __nv_bfloat16 smem_bias[NUM_EXPERTS];
        __shared__ float smem_logits_with_sigmoid_bias[NUM_EXPERTS];
        __shared__ float smem_group_sums[NUM_EXPERT_GROUPS];

        const int global_idx = token_idx * blockDim.x + tid;
        smem_bias[tid] = ((__nv_bfloat16*)gating_data.routing_bias)[tid];
        float logit = ((float*)gating_data.routing_logits)[global_idx];
        logit = 1.0f / (1.0f + expf(-logit));
        logit += __bfloat162float(smem_bias[tid]);
        smem_logits_with_sigmoid_bias[tid] = logit;

        float top2_m1 = logit;
        float top2_m2 = -FLT_MAX;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float other_m1 = __shfl_xor_sync(0xffffffff, top2_m1, mask);
            const float other_m2 = __shfl_xor_sync(0xffffffff, top2_m2, mask);

            if (other_m1 > top2_m1) {
                top2_m2 = max(top2_m1, other_m2);
                top2_m1 = other_m1;
            } else if (other_m1 > top2_m2) {
                top2_m2 = other_m1;
            }
        }

        if (lane_id == 0) {
            smem_group_sums[warp_id] = top2_m1 + top2_m2;
        }
        __syncthreads();

        int selected_groups_idx[NUM_SELECTED_GROUPS];
        int selected_group_expert_idx[NUM_SELECTED_GROUPS];
        float selected_group_expert_score[NUM_SELECTED_GROUPS];
        int top_expert_idx[NUM_SELECTED_EXPERTS];
        float top_expert_score[NUM_SELECTED_EXPERTS];

        if (warp_id == 0) {
            if (lane_id == 0) {
                float selected_groups_sums[NUM_SELECTED_GROUPS];
                #pragma unroll
                for (int i = 0; i < NUM_SELECTED_GROUPS; ++i) {
                    selected_groups_idx[i] = -1;
                    selected_groups_sums[i] = -FLT_MAX;
                }

                #pragma unroll
                for (int i = 0; i < NUM_EXPERT_GROUPS; ++i) {
                    const int cur_idx = i;
                    const float cur_sum = smem_group_sums[i];

                    #pragma unroll
                    for (int j = 0; j < NUM_SELECTED_GROUPS; ++j) {
                        if (cur_sum > selected_groups_sums[j]) {
                            for (int k = NUM_SELECTED_GROUPS - 1; k > j; --k) {
                                selected_groups_idx[k] = selected_groups_idx[k - 1];
                                selected_groups_sums[k] = selected_groups_sums[k - 1];
                            }
                            selected_groups_idx[j] = cur_idx;
                            selected_groups_sums[j] = cur_sum;
                            break;
                        }
                    }
                }
            }

            #pragma unroll
            for (int i = 0; i < NUM_SELECTED_GROUPS; ++i) {
                selected_groups_idx[i] = __shfl_sync(0xffffffff, selected_groups_idx[i], 0);
            }

            #pragma unroll
            for (int i = 0; i < NUM_SELECTED_GROUPS; ++i) {
                const int group_idx = selected_groups_idx[i];
                selected_group_expert_idx[i] = group_idx * NUM_EXPERTS / NUM_EXPERT_GROUPS + lane_id;
                selected_group_expert_score[i] = smem_logits_with_sigmoid_bias[selected_group_expert_idx[i]];
            }

            #pragma unroll
            for (int i = 0; i < NUM_SELECTED_GROUPS; ++i) {
                for (int j = i + 1; j < NUM_SELECTED_GROUPS; ++j) {
                    if (selected_group_expert_score[j] > selected_group_expert_score[i]) {
                        const float tmp_score = selected_group_expert_score[i];
                        selected_group_expert_score[i] = selected_group_expert_score[j];
                        selected_group_expert_score[j] = tmp_score;

                        const int tmp_idx = selected_group_expert_idx[i];
                        selected_group_expert_idx[i] = selected_group_expert_idx[j];
                        selected_group_expert_idx[j] = tmp_idx;
                    }
                }
            }

            float thread_scores[NUM_SELECTED_EXPERTS];
            int thread_indices[NUM_SELECTED_EXPERTS];
            #pragma unroll
            for (int i = 0; i < NUM_SELECTED_GROUPS; ++i) {
                thread_scores[i] = selected_group_expert_score[i];
                thread_indices[i] = selected_group_expert_idx[i];
            }
            #pragma unroll
            for (int i = NUM_SELECTED_GROUPS; i < NUM_SELECTED_EXPERTS; ++i) {
                thread_scores[i] = -FLT_MAX;
                thread_indices[i] = -1;
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other_scores[NUM_SELECTED_EXPERTS];
                int other_indices[NUM_SELECTED_EXPERTS];

                #pragma unroll
                for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
                    other_scores[i] = __shfl_down_sync(0xffffffff, thread_scores[i], offset);
                    other_indices[i] = __shfl_down_sync(0xffffffff, thread_indices[i], offset);
                }

                float merged_scores[NUM_SELECTED_EXPERTS];
                int merged_indices[NUM_SELECTED_EXPERTS];
                int p1 = 0;
                int p2 = 0;
                #pragma unroll
                for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
                    if (thread_scores[p1] >= other_scores[p2]) {
                        merged_scores[i] = thread_scores[p1];
                        merged_indices[i] = thread_indices[p1];
                        ++p1;
                    } else {
                        merged_scores[i] = other_scores[p2];
                        merged_indices[i] = other_indices[p2];
                        ++p2;
                    }
                }

                #pragma unroll
                for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
                    thread_scores[i] = merged_scores[i];
                    thread_indices[i] = merged_indices[i];
                }
            }

            #pragma unroll
            for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
                top_expert_score[i] = __shfl_sync(0xffffffff, thread_scores[i], 0);
                top_expert_idx[i] = __shfl_sync(0xffffffff, thread_indices[i], 0);
            }

            if (lane_id < NUM_SELECTED_EXPERTS) {
                const int selected_expert = top_expert_idx[lane_id];
                const float selected_score = top_expert_score[lane_id] - __bfloat162float(smem_bias[selected_expert]);

                float score_sum = selected_score;
                score_sum += __shfl_xor_sync(0xff, score_sum, 1);
                score_sum += __shfl_xor_sync(0xff, score_sum, 2);
                score_sum += __shfl_xor_sync(0xff, score_sum, 4);

                const float final_score = selected_score * gating_data.routing_scaling_factor / score_sum;
                const int write_idx = token_idx * NUM_SELECTED_EXPERTS + lane_id;
                ((int*)gating_data.routing_idx)[write_idx] = selected_expert;
                ((float*)gating_data.routing_weights)[write_idx] = final_score;
            }
        }

        __syncthreads();
    }

    grid.sync();

    // Stage-2: count + exclusive scan, done by block0 for seq_len <= 64.
    // `expert_offsets[i]` stores the exclusive prefix sum up to expert `i`.
    if (blockIdx.x == 0) {
        __shared__ int smem_cnts[32];
        if (tid < 32) {
            smem_cnts[tid] = 0;
        }
        __syncthreads();

        for (int idx = tid; idx < seq_len * 8; idx += blockDim.x) {
            // #pragma unroll
            // for (int k = 0; k < NUM_SELECTED_EXPERTS; ++k) {
            //     const int e_id = ((int*)gating_data.routing_idx)[idx * NUM_SELECTED_EXPERTS + k];
            //     if (e_id >= local_expert_offset && e_id < local_expert_offset + 32) {
            //         atomicAdd(&smem_cnts[e_id - local_expert_offset], 1);
            //     }
            // }
            int token_id = idx >> 3; // idx / 8
            int inter_id = idx & 7; // idx % 8
            int e_id = ((int*)gating_data.routing_idx)[token_id * 8 + inter_id];
            if (e_id >= local_expert_offset && e_id < local_expert_offset + 32) {
                atomicAdd(&smem_cnts[e_id - local_expert_offset], 1);
            }
        }
        __syncthreads();

        // if (tid == 0) {
        //     int prefix = 0;
        //     #pragma unroll
        //     for (int i = 0; i < 32; ++i) {
        //         int tmp = prefix;
        //         prefix += smem_cnts[i];
        //         smem_cnts[i] = tmp; // reuse smem for scan result
        //     }
        //     expert_offsets[32] = prefix;
        // }
        // __syncthreads();
        // if (tid < 32) {
        //     expert_offsets[tid] = smem_cnts[tid];
        //     expert_cnts[tid] = smem_cnts[tid];
        // }
        if (tid < 32) {
            int val = smem_cnts[tid];
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int remote = __shfl_up_sync(0xFFFFFFFF, val, offset);
                if (tid >= offset) {
                    val += remote;
                }
            }

            int exclusive_val = __shfl_up_sync(0xFFFFFFFF, val, 1);
            if (tid == 0) exclusive_val = 0;

            smem_cnts[tid] = exclusive_val;
            expert_offsets[tid] = exclusive_val;
            if (tid == 31) {
                expert_offsets[32] = val; // total count
            }
        }
        __syncthreads();

        for (int idx = tid; idx < seq_len; idx += blockDim.x) {
            int expert_cnt = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int e_id = ((int*)gating_data.routing_idx)[idx * 8 + k];
                if (e_id >= local_expert_offset && e_id < local_expert_offset + 32) {
                    const int rel_id = e_id - local_expert_offset;
                    const int write_pos = atomicAdd(&smem_cnts[rel_id], 1);
                    out_token_idx[write_pos] = idx;
                    out_weights[write_pos] = ((float*)gating_data.routing_weights)[idx * 8 + k];
                    token2permuted_idx[idx * 8 + expert_cnt] = write_pos;
                    expert_cnt++;
                }
            }
            token_counts[idx] = expert_cnt;
        }
    }


    grid.sync();

    // Stage-3: copy fp8 activations and scales using permuted order.
    const int total_valid = expert_offsets[32];
    for (int out_row_idx = blockIdx.x; out_row_idx < total_valid; out_row_idx += gridDim.x) {
        const int src_row_idx = out_token_idx[out_row_idx];

        const uint4* src_ptr4 = reinterpret_cast<const uint4*>(input + src_row_idx * HIDDEN_DIM);
        uint4* dst_ptr4 = reinterpret_cast<uint4*>(output + out_row_idx * HIDDEN_DIM);

        for (int v = tid; v < HIDDEN_DIM / VEC_SIZE; v += blockDim.x) {
            dst_ptr4[v] = src_ptr4[v];
        }

        for (int hb = tid; hb < NUM_HIDDEN_BLOCKS; hb += blockDim.x) {
            output_scale[hb * seq_len * NUM_SELECTED_EXPERTS + out_row_idx] =
                input_scale[hb * seq_len + src_row_idx];
        }
    }
}


bool launchCooperativeKernel(
    void* routing_logits,
    void* routing_bias,
    float routing_scaling_factor,
    void* routing_idx,
    void* routing_weights,
    void* input,
    void* input_scale,
    void* expert_counts,
    void* expert_offsets,
    // void* total_tokens,
    void* out_token_idx,
    void* out_weights,
    void* output,
    void* output_scale,
    void* token2permuted_idx,     // [seq_len * 8]
    void* token_counts,
    int seq_len,
    int local_expert_offset
) {
    constexpr int threads_per_block = 256;
    int num_blocks = 128;

    FusedGatingData data;
    data.routing_logits = routing_logits;
    data.routing_bias = routing_bias;
    data.routing_scaling_factor = routing_scaling_factor;
    data.routing_idx = routing_idx;
    data.routing_weights = routing_weights;

    void* kernel_args[] = {
        &data,
        &input,
        &input_scale,
        &expert_counts,
        &expert_offsets,
        // &total_tokens,
        &out_token_idx,
        &out_weights,
        &output,
        &output_scale,
        &token2permuted_idx,
        &token_counts,
        &seq_len,
        &local_expert_offset
    };

    cudaError_t st = cudaLaunchCooperativeKernel(
        (void*)cooperative_kernel,
        num_blocks,
        threads_per_block,
        kernel_args,
        0,
        0
    );
    return st == cudaSuccess;
}
