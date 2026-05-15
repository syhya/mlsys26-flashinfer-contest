#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <tuple>

// Helper to compute dot product of 8 bfloat16 elements packed in a float4
__device__ __forceinline__ float dot_bf16(float4 a, float4 b) {
    nv_bfloat162* a_bf = (nv_bfloat162*)&a;
    nv_bfloat162* b_bf = (nv_bfloat162*)&b;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 fa = __bfloat1622float2(a_bf[i]);
        float2 fb = __bfloat1622float2(b_bf[i]);
        sum += fa.x * fb.x + fa.y * fb.y;
    }
    return sum;
}

// Helper to compute dot product of 2 bfloat16 elements packed in a float
__device__ __forceinline__ float dot_bf16_2(float a, float b) {
    nv_bfloat162* a_bf = (nv_bfloat162*)&a;
    nv_bfloat162* b_bf = (nv_bfloat162*)&b;
    float2 fa = __bfloat1622float2(*a_bf);
    float2 fb = __bfloat1622float2(*b_bf);
    return fa.x * fb.x + fa.y * fb.y;
}

__global__ void dsa_forward_kernel(
    const nv_bfloat16* __restrict__ q_nope,
    const nv_bfloat16* __restrict__ q_pe,
    const nv_bfloat16* __restrict__ ckv_cache,
    const nv_bfloat16* __restrict__ kpe_cache,
    const int32_t* __restrict__ sparse_indices,
    nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    float sm_scale,
    int num_tokens,
    int topk
) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    // Shared Memory Allocations
    __shared__ __align__(16) nv_bfloat16 smem_Kc[32][512];
    __shared__ __align__(16) nv_bfloat16 smem_qn[512];
    __shared__ __align__(16) nv_bfloat16 smem_qp[64];
    
    __shared__ int smem_indices[32];
    __shared__ float smem_logits[32];
    __shared__ float smem_attn[32];

    // Load q_nope into shared memory (128 threads load 512 elements -> 4 elements per thread -> float2)
    if (tid < 128) {
        float2 qn_vec = *(float2*)(&q_nope[token_idx * 16 * 512 + head_idx * 512 + tid * 4]);
        *(float2*)(&smem_qn[tid * 4]) = qn_vec;
    }
    
    // Load q_pe into shared memory (16 threads load 64 elements -> 4 elements per thread -> float2)
    if (tid < 16) {
        float2 qp_vec = *(float2*)(&q_pe[token_idx * 16 * 64 + head_idx * 64 + tid * 4]);
        *(float2*)(&smem_qp[tid * 4]) = qp_vec;
    }
    __syncthreads();

    // Accumulators for 4 output elements per thread
    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m_val = -INFINITY;
    float d_val = 0.0f;

    int B_k = 32;
    int num_tiles = (topk + B_k - 1) / B_k;

    int wid = tid / 32;
    int lane = tid % 32;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int tile_start = tile * B_k;
        
        // Load sparse indices for the current tile
        if (tid < 32) {
            if (tile_start + tid < topk) {
                smem_indices[tid] = sparse_indices[token_idx * topk + tile_start + tid];
            } else {
                smem_indices[tid] = -1;
            }
        }
        __syncthreads();

        // Cooperatively load Kc for the 32 selected valid indices into shared memory
        // 32 rows * 512 cols = 16384 elements = 2048 float4s
        for (int i = tid; i < 2048; i += 128) {
            int row = i / 64; 
            int col = (i % 64) * 8; 
            int k_idx = smem_indices[row];
            if (k_idx != -1) {
                float4 vec = *(float4*)(&ckv_cache[k_idx * 512 + col]);
                *(float4*)(&smem_Kc[row][col]) = vec;
            } else {
                *(float4*)(&smem_Kc[row][col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        // Phase 1: Compute attention logits
        for (int i = 0; i < 8; ++i) {
            int idx = wid * 8 + i;
            int k_idx = smem_indices[idx];
            float logit = -INFINITY;
            
            if (k_idx != -1) {
                float sum = 0.0f;
                // Dot product for q_nope and ckv_cache (512 elements)
                for (int k = 0; k < 2; ++k) {
                    int offset = k * 256 + lane * 8;
                    float4 qn_vec = *(float4*)(&smem_qn[offset]);
                    float4 kc_vec = *(float4*)(&smem_Kc[idx][offset]);
                    sum += dot_bf16(qn_vec, kc_vec);
                }
                
                // Dot product for q_pe and kpe_cache (64 elements)
                int offset_p = lane * 2;
                float qp_vec = *(float*)(&smem_qp[offset_p]);
                float kp_vec = *(float*)(&kpe_cache[k_idx * 64 + offset_p]);
                sum += dot_bf16_2(qp_vec, kp_vec);
                
                // Warp reduction
                #pragma unroll
                for (int mask = 16; mask > 0; mask /= 2) {
                    sum += __shfl_down_sync(0xffffffff, sum, mask);
                }
                if (lane == 0) {
                    logit = sum * sm_scale;
                }
            }
            if (lane == 0) {
                smem_logits[idx] = logit;
            }
        }
        __syncthreads();

        // Phase 2: Online Softmax Max Update
        float local_max = -INFINITY;
        if (tid < 32) {
            local_max = smem_logits[tid];
        }
        
        float warp_max = local_max;
        #pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, mask));
        }
        warp_max = __shfl_sync(0xffffffff, warp_max, 0);

        float m_new = m_val;
        if (tid < 32) {
            m_new = fmaxf(m_val, warp_max);
        }
        __shared__ float smem_m_new;
        if (tid == 0) smem_m_new = m_new;
        __syncthreads();
        m_new = smem_m_new;

        // Phase 3: Online Softmax Normalizer Update
        float exp_diff = (m_val == -INFINITY) ? 0.0f : expf(m_val - m_new);
        d_val = d_val * exp_diff;

        float local_exp = 0.0f;
        if (tid < 32) {
            float logit = smem_logits[tid];
            local_exp = (logit == -INFINITY) ? 0.0f : expf(logit - m_new);
            smem_attn[tid] = local_exp;
        }

        float warp_sum = local_exp;
        #pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, mask);
        }
        warp_sum = __shfl_sync(0xffffffff, warp_sum, 0);

        __shared__ float smem_warp_sum;
        if (tid == 0) smem_warp_sum = warp_sum;
        __syncthreads();

        d_val += smem_warp_sum;
        m_val = m_new;

        for (int i = 0; i < 4; ++i) {
            out_acc[i] *= exp_diff;
        }
        __syncthreads();

        // Phase 4: Compute Output Attn Weighted Sum
        for (int idx = 0; idx < 32; ++idx) {
            float attn_w = smem_attn[idx];
            if (attn_w > 0.0f) {
                // Read 4 elements (8 bytes) = float2
                float2 kc_vec = *(float2*)(&smem_Kc[idx][tid * 4]);
                nv_bfloat162* kc_bf = (nv_bfloat162*)&kc_vec;
                
                float2 f0 = __bfloat1622float2(kc_bf[0]);
                float2 f1 = __bfloat1622float2(kc_bf[1]);
                
                out_acc[0] += attn_w * f0.x;
                out_acc[1] += attn_w * f0.y;
                out_acc[2] += attn_w * f1.x;
                out_acc[3] += attn_w * f1.y;
            }
        }
        __syncthreads();
    }

    // Phase 5: Final Normalization and Store
    float inv_d = (d_val > 0.0f) ? (1.0f / d_val) : 0.0f;
    for (int i = 0; i < 4; ++i) {
        out_acc[i] *= inv_d;
    }

    int out_base = token_idx * 16 * 512 + head_idx * 512 + tid * 4;
    nv_bfloat162 res0 = __floats2bfloat162_rn(out_acc[0], out_acc[1]);
    nv_bfloat162 res1 = __floats2bfloat162_rn(out_acc[2], out_acc[3]);

    float2 out_vec;
    out_vec.x = *(float*)&res0;
    out_vec.y = *(float*)&res1;

    *(float2*)(&output[out_base]) = out_vec;

    // Log-sum-exp Base-2 Calculation (only thread 0 handles the store)
    if (tid == 0) {
        float lse_val = -INFINITY;
        if (d_val > 0.0f) {
            lse_val = m_val * 1.44269504f + log2f(d_val);
        }
        lse[token_idx * 16 + head_idx] = lse_val;
    }
}

std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
    torch::Tensor q_nope,         
    torch::Tensor q_pe,           
    torch::Tensor ckv_cache,      
    torch::Tensor kpe_cache,      
    torch::Tensor sparse_indices, 
    float sm_scale                
) {
    // Ensure contiguous memory footprint to safely rely on index mapping
    q_nope = q_nope.contiguous();
    q_pe = q_pe.contiguous();
    ckv_cache = ckv_cache.contiguous();
    kpe_cache = kpe_cache.contiguous();
    sparse_indices = sparse_indices.contiguous();

    int num_tokens = q_nope.size(0);
    int topk = sparse_indices.size(1);
    int num_qo_heads = 16;

    auto options = q_nope.options();
    torch::Tensor output = torch::empty({num_tokens, num_qo_heads, 512}, options);
    torch::Tensor lse = torch::empty({num_tokens, num_qo_heads}, options.dtype(torch::kFloat32));

    if (num_tokens == 0) {
        return std::make_tuple(output, lse);
    }

    dim3 grid(num_tokens, num_qo_heads);
    dim3 block(128); // Standard stable tile thread block

    dsa_forward_kernel<<<grid, block>>>(
        (const nv_bfloat16*)q_nope.data_ptr<at::BFloat16>(),
        (const nv_bfloat16*)q_pe.data_ptr<at::BFloat16>(),
        (const nv_bfloat16*)ckv_cache.data_ptr<at::BFloat16>(),
        (const nv_bfloat16*)kpe_cache.data_ptr<at::BFloat16>(),
        sparse_indices.data_ptr<int32_t>(),
        (nv_bfloat16*)output.data_ptr<at::BFloat16>(),
        lse.data_ptr<float>(),
        sm_scale,
        num_tokens,
        topk
    );

    return std::make_tuple(output, lse);
}

// Ensure the module entry point is bound correctly
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward Implementation");
}