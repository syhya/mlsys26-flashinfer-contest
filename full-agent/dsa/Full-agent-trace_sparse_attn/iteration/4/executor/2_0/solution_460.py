#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// SINGLE-PASS KERNEL (For High Workloads)
// ============================================================================
__global__ void dsa_forward_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    int t = blockIdx.x;
    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = threadIdx.y * 32 + threadIdx.x;

    __shared__ alignas(16) int idx_shared[32];
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * 512];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * 64];

    __nv_bfloat162 q_n_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        q_n_reg[j] = *reinterpret_cast<const __nv_bfloat162*>(&q_nope_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]);
    }

    __nv_bfloat162 q_p_reg[1];
    q_p_reg[0] = *reinterpret_cast<const __nv_bfloat162*>(&q_pe_ptr[t * 16 * 64 + h * 64 + lane * 2]);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        O_reg[j] = make_float2(0.0f, 0.0f);
    }

    float m = -INFINITY;
    float l = 0.0f;

    int num_tiles = 2048 / 32;
    for (int tile = 0; tile < num_tiles; ++tile) {
        
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * 2048 + tile * 32 + tid];
        }
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid / 64);
            int load_col = tid % 64; 
            int idx = idx_shared[load_row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(&ckv_cache_ptr[idx * 512]);
                float4* dst = reinterpret_cast<float4*>(&smem_Kc[load_row * 512]);
                dst[load_col] = src[load_col];
            }
        }
        
        if (tid < 256) {
            int load_row = tid / 8; 
            int load_col = tid % 8;
            int idx = idx_shared[load_row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(&kpe_cache_ptr[idx * 64]);
                float4* dst = reinterpret_cast<float4*>(&smem_Kp[load_row * 64]);
                dst[load_col] = src[load_col];
            }
        }
        __syncthreads();

        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] == -1) continue;

            float local_dot = 0.0f;
            
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
                float2 q_f = __bfloat1622float2(q_n_reg[j]);
                float2 k_f = __bfloat1622float2(k_n);
                local_dot = fmaf(q_f.x, k_f.x, local_dot);
                local_dot = fmaf(q_f.y, k_f.y, local_dot);
            }
            
            {
                __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * 64 + lane * 2]);
                float2 q_f = __bfloat1622float2(q_p_reg[0]);
                float2 k_f = __bfloat1622float2(k_p);
                local_dot = fmaf(q_f.x, k_f.x, local_dot);
                local_dot = fmaf(q_f.y, k_f.y, local_dot);
            }

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0);
            
            logit *= sm_scale;

            float m_new = fmaxf(m, logit);
            float exp_diff = (m == -INFINITY) ? 0.0f : expf(m - m_new);
            float exp_logit = expf(logit - m_new);

            l = fmaf(l, exp_diff, exp_logit);
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
                float2 k_f = __bfloat1622float2(k_n);
                float O_x_scaled = O_reg[j].x * exp_diff;
                float O_y_scaled = O_reg[j].y * exp_diff;
                O_reg[j].x = fmaf(exp_logit, k_f.x, O_x_scaled);
                O_reg[j].y = fmaf(exp_logit, k_f.y, O_y_scaled);
            }
        }
        __syncthreads(); 
    }

    if (l > 0.0f) {
        float inv_l = 1.0f / l;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x *= inv_l;
            O_reg[j].y *= inv_l;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x = 0.0f;
            O_reg[j].y = 0.0f;
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 out_bf16 = __floats2bfloat162_rn(O_reg[j].x, O_reg[j].y);
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]) = out_bf16;
    }

    if (lane == 0) {
        float lse_val = (l > 0.0f) ? (m * 1.4426950408889634f + log2f(l)) : -INFINITY;
        lse_ptr[t * 16 + h] = lse_val;
    }
}

// ============================================================================
// FUSED SPLIT-K SEQUENCE-LEVEL PARALLELISM (For Low Workloads / SM Saturation)
// ============================================================================
__global__ void fused_split_k_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float* __restrict__ O_tmp,
    float* __restrict__ m_tmp,
    float* __restrict__ l_tmp,
    int32_t* __restrict__ sync_counters,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    int s = blockIdx.x; // chunk index (0 to 63)
    int t = blockIdx.y; // token index
    int h = threadIdx.y; // head index
    int lane = threadIdx.x;
    int tid = threadIdx.y * 32 + threadIdx.x;

    __shared__ alignas(16) int idx_shared[32];
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * 512];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * 64];

    __nv_bfloat162 q_n_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        q_n_reg[j] = *reinterpret_cast<const __nv_bfloat162*>(&q_nope_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]);
    }

    __nv_bfloat162 q_p_reg[1];
    q_p_reg[0] = *reinterpret_cast<const __nv_bfloat162*>(&q_pe_ptr[t * 16 * 64 + h * 64 + lane * 2]);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        O_reg[j] = make_float2(0.0f, 0.0f);
    }

    float m = -INFINITY;
    float l = 0.0f;

    // S=64 chunks. 2048 keys total / 64 chunks = 32 keys per chunk (exactly 1 tile).
    int tile = s; 
    
    // 1. Cooperative load of Sparse Indices
    if (tid < 32) {
        idx_shared[tid] = sparse_indices_ptr[t * 2048 + tile * 32 + tid];
    }
    __syncthreads();

    // 2. Cooperative Vectorized Load of Key Caches
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        int load_row = step * 8 + (tid / 64);
        int load_col = tid % 64; 
        int idx = idx_shared[load_row];
        if (idx != -1) {
            const float4* src = reinterpret_cast<const float4*>(&ckv_cache_ptr[idx * 512]);
            float4* dst = reinterpret_cast<float4*>(&smem_Kc[load_row * 512]);
            dst[load_col] = src[load_col];
        }
    }
    
    if (tid < 256) {
        int load_row = tid / 8; 
        int load_col = tid % 8;
        int idx = idx_shared[load_row];
        if (idx != -1) {
            const float4* src = reinterpret_cast<const float4*>(&kpe_cache_ptr[idx * 64]);
            float4* dst = reinterpret_cast<float4*>(&smem_Kp[load_row * 64]);
            dst[load_col] = src[load_col];
        }
    }
    __syncthreads();

    // 3. Compute Attention for this specific chunk
    for (int i = 0; i < 32; ++i) {
        if (idx_shared[i] == -1) continue;

        float local_dot = 0.0f;
        
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
            float2 q_f = __bfloat1622float2(q_n_reg[j]);
            float2 k_f = __bfloat1622float2(k_n);
            local_dot = fmaf(q_f.x, k_f.x, local_dot);
            local_dot = fmaf(q_f.y, k_f.y, local_dot);
        }
        
        {
            __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * 64 + lane * 2]);
            float2 q_f = __bfloat1622float2(q_p_reg[0]);
            float2 k_f = __bfloat1622float2(k_p);
            local_dot = fmaf(q_f.x, k_f.x, local_dot);
            local_dot = fmaf(q_f.y, k_f.y, local_dot);
        }

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0); 
        logit *= sm_scale;

        float m_new = fmaxf(m, logit);
        float exp_diff = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        float exp_logit = expf(logit - m_new);

        l = fmaf(l, exp_diff, exp_logit);
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
            float2 k_f = __bfloat1622float2(k_n);
            float O_x_scaled = O_reg[j].x * exp_diff;
            float O_y_scaled = O_reg[j].y * exp_diff;
            O_reg[j].x = fmaf(exp_logit, k_f.x, O_x_scaled);
            O_reg[j].y = fmaf(exp_logit, k_f.y, O_y_scaled);
        }
    }

    // 4. Dump intermediate unnormalized states directly to FP32 workspace
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int base_idx = t * 64 * 16 * 512 + s * 16 * 512 + h * 512 + j * 64 + lane * 2;
        *reinterpret_cast<float2*>(&O_tmp[base_idx]) = make_float2(O_reg[j].x, O_reg[j].y);
    }
    
    if (lane == 0) {
        // Contiguous memory write for s layout [num_tokens, 16, S]
        m_tmp[t * 16 * 64 + h * 64 + s] = m;
        l_tmp[t * 16 * 64 + h * 64 + s] = l;
    }

    // 5. Cross-block synchronization for Split-K reduction
    __threadfence();
    __shared__ bool is_last_block;
    if (tid == 0) {
        int old = atomicAdd(&sync_counters[t], 1);
        is_last_block = (old == 63);
    }
    __syncthreads();

    if (is_last_block) {
        // We are the last block for token t. Reduce across 64 chunks for all 16 heads.
        // Each thread (out of 512) corresponds to a specific head (h = threadIdx.y) and lane (threadIdx.x).
        
        float m_max = -INFINITY;
        float m_s_local[2];
        float l_s_local[2];
        
        #pragma unroll
        for(int k=0; k<2; ++k) {
            int chunk_idx = lane + k * 32;
            m_s_local[k] = m_tmp[t * 16 * 64 + h * 64 + chunk_idx];
            l_s_local[k] = l_tmp[t * 16 * 64 + h * 64 + chunk_idx];
            m_max = fmaxf(m_max, m_s_local[k]);
        }
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            m_max = fmaxf(m_max, __shfl_down_sync(0xffffffff, m_max, offset));
        }
        m_max = __shfl_sync(0xffffffff, m_max, 0);
        
        float l_sum = 0.0f;
        #pragma unroll
        for(int k=0; k<2; ++k) {
            if (m_s_local[k] != -INFINITY) {
                l_sum = fmaf(l_s_local[k], expf(m_s_local[k] - m_max), l_sum);
            }
        }
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            l_sum += __shfl_down_sync(0xffffffff, l_sum, offset);
        }
        l_sum = __shfl_sync(0xffffffff, l_sum, 0);
        
        if (lane == 0) {
            if (m_max == -INFINITY) {
                lse_ptr[t * 16 + h] = -INFINITY;
            } else {
                lse_ptr[t * 16 + h] = fmaf(m_max, 1.4426950408889634f, log2f(l_sum));
            }
        }

        float2 out[8];
        #pragma unroll
        for(int j=0; j<8; ++j) {
            out[j] = make_float2(0.0f, 0.0f);
        }

        if (m_max != -INFINITY && l_sum > 0.0f) {
            float inv_l = 1.0f / l_sum;
            for (int chunk = 0; chunk < 64; ++chunk) {
                float m_s = m_tmp[t * 16 * 64 + h * 64 + chunk];
                if (m_s != -INFINITY) {
                    float scale = expf(m_s - m_max) * inv_l;
                    #pragma unroll
                    for(int j=0; j<8; ++j) {
                        int base_idx = t * 64 * 16 * 512 + chunk * 16 * 512 + h * 512 + j * 64 + lane * 2;
                        float2 val = *reinterpret_cast<const float2*>(&O_tmp[base_idx]);
                        out[j].x = fmaf(val.x, scale, out[j].x);
                        out[j].y = fmaf(val.y, scale, out[j].y);
                    }
                }
            }
        }

        #pragma unroll
        for(int j=0; j<8; ++j) {
            __nv_bfloat162 out_bf16 = __floats2bfloat162_rn(out[j].x, out[j].y);
            *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]) = out_bf16;
        }
    }
}

// ============================================================================
// MAIN DISPATCHER ENTRY POINT
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
    torch::Tensor q_nope,         // [num_tokens, 16, 512]  bfloat16
    torch::Tensor q_pe,           // [num_tokens, 16, 64]   bfloat16
    torch::Tensor ckv_cache,      // [num_pages, 64, 512]   bfloat16
    torch::Tensor kpe_cache,      // [num_pages, 64, 64]    bfloat16
    torch::Tensor sparse_indices, // [num_tokens, 2048]     int32
    float sm_scale                // scalar: 1/sqrt(192)
) {
    int num_tokens = q_nope.size(0);
    
    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16}, torch::dtype(torch::kFloat32).device(q_nope.device()));

    if (num_tokens == 0) {
        return {output, lse};
    }

    if (num_tokens < 128) {
        int S = 64;
        
        auto O_tmp = torch::empty({num_tokens, S, 16, 512}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto m_tmp = torch::empty({num_tokens, 16, S}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto l_tmp = torch::empty({num_tokens, 16, S}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto sync_counters = torch::zeros({num_tokens}, torch::dtype(torch::kInt32).device(q_nope.device()));

        dim3 compute_grid(S, num_tokens, 1);
        dim3 compute_block(32, 16, 1);
        fused_split_k_kernel<<<compute_grid, compute_block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            sync_counters.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>()
        );
    } 
    else {
        dim3 grid(num_tokens);
        dim3 block(32, 16); 

        dsa_forward_kernel<<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>()
        );
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward Kernel with Fused Split-K Dispatch");
}
