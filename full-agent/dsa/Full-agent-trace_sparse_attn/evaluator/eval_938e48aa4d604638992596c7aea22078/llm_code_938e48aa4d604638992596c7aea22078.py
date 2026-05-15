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

        int row = tid / 16;
        int col_group = tid % 16;
        if (row < 32) {
            int idx = idx_shared[row];
            if (idx != -1) {
                const float4* src_c = reinterpret_cast<const float4*>(&ckv_cache_ptr[idx * 512 + col_group * 32]);
                float4* dst_c = reinterpret_cast<float4*>(&smem_Kc[row * 512 + col_group * 32]);
                dst_c[0] = src_c[0];
                dst_c[1] = src_c[1];
                dst_c[2] = src_c[2];
                dst_c[3] = src_c[3];

                const float2* src_p = reinterpret_cast<const float2*>(&kpe_cache_ptr[idx * 64 + col_group * 4]);
                float2* dst_p = reinterpret_cast<float2*>(&smem_Kp[row * 64 + col_group * 4]);
                dst_p[0] = src_p[0];
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
                local_dot += q_f.x * k_f.x + q_f.y * k_f.y;
            }
            
            {
                __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * 64 + lane * 2]);
                float2 q_f = __bfloat1622float2(q_p_reg[0]);
                float2 k_f = __bfloat1622float2(k_p);
                local_dot += q_f.x * k_f.x + q_f.y * k_f.y;
            }

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0);
            
            logit *= sm_scale;

            float m_new = fmaxf(m, logit);
            float exp_diff = (m == -INFINITY) ? 0.0f : expf(m - m_new);
            float exp_logit = expf(logit - m_new);

            l = l * exp_diff + exp_logit;
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
                float2 k_f = __bfloat1622float2(k_n);
                O_reg[j].x = O_reg[j].x * exp_diff + exp_logit * k_f.x;
                O_reg[j].y = O_reg[j].y * exp_diff + exp_logit * k_f.y;
            }
        }
        __syncthreads(); 
    }

    if (l > 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x /= l;
            O_reg[j].y /= l;
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
// SPLIT-K SEQUENCE-LEVEL PARALLELISM (For Low Workloads / SM Saturation)
// ============================================================================
__global__ void split_k_compute_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float* __restrict__ O_tmp,
    float* __restrict__ m_tmp,
    float* __restrict__ l_tmp
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
    int row = tid / 16;
    int col_group = tid % 16;
    if (row < 32) {
        int idx = idx_shared[row];
        if (idx != -1) {
            const float4* src_c = reinterpret_cast<const float4*>(&ckv_cache_ptr[idx * 512 + col_group * 32]);
            float4* dst_c = reinterpret_cast<float4*>(&smem_Kc[row * 512 + col_group * 32]);
            dst_c[0] = src_c[0];
            dst_c[1] = src_c[1];
            dst_c[2] = src_c[2];
            dst_c[3] = src_c[3];

            const float2* src_p = reinterpret_cast<const float2*>(&kpe_cache_ptr[idx * 64 + col_group * 4]);
            float2* dst_p = reinterpret_cast<float2*>(&smem_Kp[row * 64 + col_group * 4]);
            dst_p[0] = src_p[0];
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
            local_dot += q_f.x * k_f.x + q_f.y * k_f.y;
        }
        
        {
            __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * 64 + lane * 2]);
            float2 q_f = __bfloat1622float2(q_p_reg[0]);
            float2 k_f = __bfloat1622float2(k_p);
            local_dot += q_f.x * k_f.x + q_f.y * k_f.y;
        }

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0); 
        logit *= sm_scale;

        float m_new = fmaxf(m, logit);
        float exp_diff = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        float exp_logit = expf(logit - m_new);

        l = l * exp_diff + exp_logit;
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
            float2 k_f = __bfloat1622float2(k_n);
            O_reg[j].x = O_reg[j].x * exp_diff + exp_logit * k_f.x;
            O_reg[j].y = O_reg[j].y * exp_diff + exp_logit * k_f.y;
        }
    }

    // 4. Dump intermediate unnormalized states directly to FP32 workspace
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int base_idx = t * 64 * 16 * 512 + s * 16 * 512 + h * 512 + j * 64 + lane * 2;
        *reinterpret_cast<float2*>(&O_tmp[base_idx]) = make_float2(O_reg[j].x, O_reg[j].y);
    }
    
    if (lane == 0) {
        m_tmp[t * 64 * 16 + s * 16 + h] = m;
        l_tmp[t * 64 * 16 + s * 16 + h] = l;
    }
}

__global__ void split_k_reduce_kernel(
    const float* __restrict__ O_tmp,
    const float* __restrict__ m_tmp,
    const float* __restrict__ l_tmp,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    int t = blockIdx.x; // token index
    int h = blockIdx.y; // head index
    int lane = threadIdx.x; // 0..255 (256 threads to handle 512 dimensions)

    __shared__ float m_global;
    __shared__ float l_global;
    __shared__ float m_s_smem[64];
    __shared__ float l_s_smem[64];

    // Load chunk metadata (64 chunks)
    if (lane < 64) {
        m_s_smem[lane] = m_tmp[t * 64 * 16 + lane * 16 + h];
        l_s_smem[lane] = l_tmp[t * 64 * 16 + lane * 16 + h];
    }
    __syncthreads();

    // Compute Global M and L
    if (lane == 0) {
        float m_max = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            m_max = fmaxf(m_max, m_s_smem[i]);
        }
        
        float l_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            if (m_s_smem[i] != -INFINITY) {
                l_sum += l_s_smem[i] * expf(m_s_smem[i] - m_max);
            }
        }
        
        m_global = m_max;
        l_global = l_sum;

        // Base-2 LSE mathematical enforcement
        if (m_max == -INFINITY) {
            lse_ptr[t * 16 + h] = -INFINITY;
        } else {
            lse_ptr[t * 16 + h] = m_max * 1.4426950408889634f + log2f(l_sum);
        }
    }
    __syncthreads();

    // Accumulate final normalized Output Projection
    int d_idx = lane * 2;
    if (d_idx < 512) {
        float out0 = 0.0f;
        float out1 = 0.0f;
        
        if (m_global != -INFINITY && l_global > 0.0f) {
            #pragma unroll 4
            for (int s = 0; s < 64; ++s) {
                float m_s = m_s_smem[s];
                if (m_s != -INFINITY) {
                    float scale = expf(m_s - m_global) / l_global;
                    int base_idx = t * 64 * 16 * 512 + s * 16 * 512 + h * 512 + d_idx;
                    float2 val = *reinterpret_cast<const float2*>(&O_tmp[base_idx]);
                    out0 += val.x * scale;
                    out1 += val.y * scale;
                }
            }
        }

        // Store back to BF16 accurately
        __nv_bfloat162 out_bf16 = __floats2bfloat162_rn(out0, out1);
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * 16 * 512 + h * 512 + d_idx]) = out_bf16;
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

    // Dynamic Multi-path Dispatch:
    // If the grid is extremely small, Single-Pass heavily under-utilizes SMs.
    // By splitting the 2048-key sequence across 64 blocks, we inject massive Sequence-Level Parallelism
    if (num_tokens < 128) {
        int S = 64;
        
        // Exact mathematically lossless FP32 workspace allocation
        auto O_tmp = torch::empty({num_tokens, S, 16, 512}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto m_tmp = torch::empty({num_tokens, S, 16}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto l_tmp = torch::empty({num_tokens, S, 16}, torch::dtype(torch::kFloat32).device(q_nope.device()));

        // Phase 1: Distributed Chunk Compute
        dim3 compute_grid(S, num_tokens, 1);
        dim3 compute_block(32, 16, 1);
        split_k_compute_kernel<<<compute_grid, compute_block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>()
        );

        // Phase 2: Global Assembly & Normalization
        dim3 reduce_grid(num_tokens, 16, 1);
        dim3 reduce_block(256, 1, 1);
        split_k_reduce_kernel<<<reduce_grid, reduce_block>>>(
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>()
        );
    } 
    // High Throughput Single-Pass Execution for Large Batches
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
    m.def("dsa_forward", &dsa_forward, "DSA Forward Kernel with Dynamic Multi-path Dispatch");
}