#include <torch/extension.h>
#include <cuda_fp8.h>
#include <tuple>
#include <cub/cub.cuh>
#include <cub/block/block_radix_sort.cuh>

constexpr int TOPK_CONST = 2048;
constexpr int L_MED_MAX  = 6144;
constexpr int BT_THREADS = 256;
constexpr int BT_ITEMS   = 24;
static_assert(BT_THREADS * BT_ITEMS >= L_MED_MAX, "TIER-M buffer too small");

inline __device__ float4 cvt_fp8x4_to_float4(uint32_t val) {
    float4 res;
    union { uint32_t u32; __nv_fp8_e4m3 v[4]; } u;
    u.u32 = val;
    res.x = float(u.v[0]);
    res.y = float(u.v[1]);
    res.z = float(u.v[2]);
    res.w = float(u.v[3]);
    return res;
}

// TIER-S: For sequences with L <= TOPK, directly emit indices
__global__ void emit_short_topk_kernel(
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    int32_t* __restrict__ topk_indices,
    int max_num_pages,
    int topk
) {
    int b = blockIdx.x;
    int L = seq_lens[b];
    if (L > topk) return;

    for (int tid = threadIdx.x; tid < topk; tid += blockDim.x) {
        int v = -1;
        if (tid < L) {
            int p = tid >> 6;
            int t = tid & 63;
            v = block_table[b * max_num_pages + p] * 64 + t;
        }
        topk_indices[b * topk + tid] = v;
    }
}

__global__ void compute_scores_kernel(
    const uint8_t* __restrict__ q_index_fp8,
    const uint8_t* __restrict__ k_index_cache_fp8,
    const float* __restrict__ weights,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    float* __restrict__ scores_buf,
    int32_t* __restrict__ indices_buf,
    int max_num_pages,
    int topk
) {
    int p = blockIdx.x;
    int b = blockIdx.y;

    int seq_len = seq_lens[b];
    if (seq_len <= topk) return;

    int N = max_num_pages * 64;

    if (p * 64 >= seq_len) {
        for (int i = threadIdx.x; i < 64; i += blockDim.x) {
            scores_buf[b * N + p * 64 + i] = -1e20f;
            indices_buf[b * N + p * 64 + i] = -1;
        }
        return;
    }

    __shared__ uint32_t q_smem_u32[64 * 32];
    __shared__ float weights_smem[64];

    const uint32_t* q_global_u32 = (const uint32_t*)(q_index_fp8 + b * 64 * 128);
    for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x) {
        q_smem_u32[i] = q_global_u32[i];
    }

    if (threadIdx.x < 64) {
        weights_smem[threadIdx.x] = weights[b * 64 + threadIdx.x];
    }

    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int global_page_idx = block_table[b * max_num_pages + p];
    const uint8_t* k_page_base = k_index_cache_fp8 + global_page_idx * (64 * 132);

    for (int iter = 0; iter < 8; ++iter) {
        int t = iter * 8 + warp_id;
        if (t >= 64) continue;

        bool valid = (p * 64 + t < seq_len);
        float total_score = 0.0f;

        if (valid) {
            uint32_t k_val = *(const uint32_t*)(k_page_base + t * 128 + lane_id * 4);
            float4 k_f = cvt_fp8x4_to_float4(k_val);
            float scale = *(const float*)(k_page_base + 8192 + t * 4);

            k_f.x *= scale; k_f.y *= scale; k_f.z *= scale; k_f.w *= scale;

            for (int h = 0; h < 64; ++h) {
                uint32_t q_val = q_smem_u32[h * 32 + lane_id];
                float4 q_f = cvt_fp8x4_to_float4(q_val);

                float val = q_f.x * k_f.x + q_f.y * k_f.y + q_f.z * k_f.z + q_f.w * k_f.w;

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }

                if (lane_id == 0) {
                    if (val > 0.0f) {
                        total_score += val * weights_smem[h];
                    }
                }
            }
            if (lane_id == 0 && isnan(total_score)) {
                total_score = -1e20f;
            }
        }

        if (lane_id == 0) {
            int out_idx = b * N + p * 64 + t;
            if (valid) {
                scores_buf[out_idx] = total_score;
                indices_buf[out_idx] = global_page_idx * 64 + t;
            } else {
                scores_buf[out_idx] = -1e20f;
                indices_buf[out_idx] = -1;
            }
        }
    }
}

// TIER-M: Block-level radix sort for medium sequences
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void block_topk_kernel(
    const float* __restrict__ scores_buf,
    const int32_t* __restrict__ indices_buf,
    const int32_t* __restrict__ seq_lens,
    int32_t* __restrict__ topk_indices,
    int N, int topk, int l_med_max
) {
    int b = blockIdx.x;
    int L = seq_lens[b];
    if (L <= topk) return;        // TIER-S
    if (L > l_med_max) return;    // TIER-L

    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int32_t>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    float keys[ITEMS_PER_THREAD];
    int32_t vals[ITEMS_PER_THREAD];
    int base = b * N;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = i * BLOCK_THREADS + threadIdx.x;
        if (idx < L) {
            keys[i] = scores_buf[base + idx];
            vals[i] = indices_buf[base + idx];
        } else {
            keys[i] = -1e30f;
            vals[i] = -1;
        }
    }
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(keys, vals);
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int out_pos = i * BLOCK_THREADS + threadIdx.x;
        if (out_pos < topk) topk_indices[b * topk + out_pos] = vals[i];
    }
}

__global__ void generate_offsets_begin_end(
    int32_t* begin_offsets,
    int32_t* end_offsets,
    const int32_t* __restrict__ seq_lens,
    int batch_size,
    int seg,
    int l_med_max
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        int L = seq_lens[b];
        begin_offsets[b] = b * seg;
        end_offsets[b] = (L > l_med_max) ? (b * seg + L) : (b * seg);
    }
}

__global__ void extract_topk_kernel(
    const int32_t* __restrict__ indices_sorted,
    int32_t* __restrict__ topk_indices,
    const int32_t* __restrict__ seq_lens,
    int batch_size,
    int seg,
    int topk,
    int l_med_max
) {
    int b = blockIdx.x;
    if (seq_lens[b] <= l_med_max) return;
    int actual = min(topk, seq_lens[b]);
    for (int tid = threadIdx.x; tid < topk; tid += blockDim.x) {
        topk_indices[b * topk + tid] = (tid < actual) ? indices_sorted[b * seg + tid] : -1;
    }
}

torch::Tensor topk_indexer_forward(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table
) {
    int batch_size = q_index_fp8.size(0);
    int max_num_pages = block_table.size(1);
    int topk = TOPK_CONST;

    auto q = q_index_fp8.contiguous();
    auto k = k_index_cache_fp8.contiguous();
    auto w = weights.contiguous();
    auto sl = seq_lens.contiguous();
    auto bt = block_table.contiguous();

    auto device = q.device();

    int N = max_num_pages * 64;
    auto scores_buf = torch::empty({batch_size, N}, torch::dtype(torch::kFloat32).device(device));
    auto indices_buf = torch::empty({batch_size, N}, torch::dtype(torch::kInt32).device(device));
    auto topk_indices = torch::empty({batch_size, topk}, torch::dtype(torch::kInt32).device(device));

    // TIER-S
    emit_short_topk_kernel<<<batch_size, 256>>>(
        sl.data_ptr<int32_t>(),
        bt.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        max_num_pages,
        topk
    );

    // Compute scores for batches with L > topk
    dim3 grid(max_num_pages, batch_size);
    compute_scores_kernel<<<grid, 256>>>(
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k.data_ptr()),
        w.data_ptr<float>(),
        sl.data_ptr<int32_t>(),
        bt.data_ptr<int32_t>(),
        scores_buf.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(),
        max_num_pages,
        topk
    );

    // TIER-M: block radix sort
    static bool smem_set = false;
    if (!smem_set) {
        cudaFuncSetAttribute((const void*)block_topk_kernel<BT_THREADS, BT_ITEMS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        smem_set = true;
    }
    block_topk_kernel<BT_THREADS, BT_ITEMS><<<batch_size, BT_THREADS>>>(
        scores_buf.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(),
        sl.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        N, topk, L_MED_MAX
    );

    // TIER-L: device segmented radix sort
    auto scores_sorted = torch::empty_like(scores_buf);
    auto indices_sorted = torch::empty_like(indices_buf);
    auto begin_offsets = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
    auto end_offsets = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));

    int num_blocks = (batch_size + 255) / 256;
    generate_offsets_begin_end<<<num_blocks, 256>>>(
        begin_offsets.data_ptr<int32_t>(),
        end_offsets.data_ptr<int32_t>(),
        sl.data_ptr<int32_t>(),
        batch_size, N, L_MED_MAX
    );

    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        scores_buf.data_ptr<float>(), scores_sorted.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(), indices_sorted.data_ptr<int32_t>(),
        batch_size * N, batch_size,
        begin_offsets.data_ptr<int32_t>(), end_offsets.data_ptr<int32_t>()
    );

    auto temp_storage = torch::empty({(long)temp_storage_bytes}, torch::dtype(torch::kUInt8).device(device));

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data_ptr(), temp_storage_bytes,
        scores_buf.data_ptr<float>(), scores_sorted.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(), indices_sorted.data_ptr<int32_t>(),
        batch_size * N, batch_size,
        begin_offsets.data_ptr<int32_t>(), end_offsets.data_ptr<int32_t>()
    );

    extract_topk_kernel<<<batch_size, 1024>>>(
        indices_sorted.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        sl.data_ptr<int32_t>(),
        batch_size, N, topk, L_MED_MAX
    );

    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
    m.def("dsa_forward", &topk_indexer_forward, "DSA TopK Indexer Forward Alias");
}
