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

    float2 q_n_f32[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 q_val = *reinterpret_cast<const __nv_bfloat162*>(&q_nope_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]);
        q_n_f32[j] = __bfloat1622float2(q_val);
    }

    float2 q_p_f32;
    {
        __nv_bfloat162 q_val = *reinterpret_cast<const __nv_bfloat162*>(&q_pe_ptr[t * 16 * 64 + h * 64 + lane * 2]);
        q_p_f32 = __bfloat1622float2(q_val);
    }

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
            float2 k_f_reg[8];
            
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
                float2 k_f = __bfloat1622float2(k_n);
                k_f_reg[j] = k_f;
                local_dot = fmaf(q_n_f32[j].x, k_f.x, local_dot);
                local_dot = fmaf(q_n_f32[j].y, k_f.y, local_dot);
            }
            
            {
                __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * 64 + lane * 2]);
                float2 k_f = __bfloat1622float2(k_p);
                local_dot = fmaf(q_p_f32.x, k_f.x, local_dot);
                local_dot = fmaf(q_p_f32.y, k_f.y, local_dot);
            }

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0);
            
            logit *= sm_scale;

            float m_new = fmaxf(m, logit);
            float exp_diff = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
            float exp_logit = __expf(logit - m_new);

            l = fmaf(l, exp_diff, exp_logit);
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float O_x_scaled = O_reg[j].x * exp_diff;
                float O_y_scaled = O_reg[j].y * exp_diff;
                O_reg[j].x = fmaf(exp_logit, k_f_reg[j].x, O_x_scaled);
                O_reg[j].y = fmaf(exp_logit, k_f_reg[j].y, O_y_scaled);
            }
        }
        __syncthreads(); 
    }

    if (l > 0.0f) {
        float inv_l = __fdividef(1.0f, l);
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

    // Phase 3: Extending the "Safety Valve" to dsa_forward_kernel
    // Write to smem_O to do a transpose, so we can write out using 128-bit stores
    __shared__ float smem_O[16 * 512]; // 32KB
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        smem_O[h * 512 + j * 64 + lane * 2] = O_reg[j].x;
        smem_O[h * 512 + j * 64 + lane * 2 + 1] = O_reg[j].y;
    }
    __syncthreads();

    // Now, each thread reads 16 contiguous float values, converts to 8 bfloat16 (4 uint32_t),
    // and writes 2 uint4 to output_ptr.
    // However, lane is 0..31. We have 512 dimensions. 32 * 16 = 512.
    // So each thread processes 16 dimensions.
    #pragma unroll
    for (int j = 0; j < 2; ++j) { // 2 uint4 writes per thread
        int offset = lane * 16 + j * 8;
        float out0 = smem_O[h * 512 + offset];
        float out1 = smem_O[h * 512 + offset + 1];
        float out2 = smem_O[h * 512 + offset + 2];
        float out3 = smem_O[h * 512 + offset + 3];
        float out4 = smem_O[h * 512 + offset + 4];
        float out5 = smem_O[h * 512 + offset + 5];
        float out6 = smem_O[h * 512 + offset + 6];
        float out7 = smem_O[h * 512 + offset + 7];
        
        __nv_bfloat162 b0 = __floats2bfloat162_rn(out0, out1);
        __nv_bfloat162 b1 = __floats2bfloat162_rn(out2, out3);
        __nv_bfloat162 b2 = __floats2bfloat162_rn(out4, out5);
        __nv_bfloat162 b3 = __floats2bfloat162_rn(out6, out7);
        
        uint32_t u0 = *reinterpret_cast<uint32_t*>(&b0);
        uint32_t u1 = *reinterpret_cast<uint32_t*>(&b1);
        uint32_t u2 = *reinterpret_cast<uint32_t*>(&b2);
        uint32_t u3 = *reinterpret_cast<uint32_t*>(&b3);
        
        uint4 out_vec = make_uint4(u0, u1, u2, u3);
        *reinterpret_cast<uint4*>(&output_ptr[t * 16 * 512 + h * 512 + offset]) = out_vec;
    }

    if (lane == 0) {
        float lse_val = (l > 0.0f) ? fmaf(m, 1.4426950408889634f, __log2f(l)) : -INFINITY;
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

    float2 q_n_f32[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 q_val = *reinterpret_cast<const __nv_bfloat162*>(&q_nope_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]);
        q_n_f32[j] = __bfloat1622float2(q_val);
    }

    float2 q_p_f32;
    {
        __nv_bfloat162 q_val = *reinterpret_cast<const __nv_bfloat162*>(&q_pe_ptr[t * 16 * 64 + h * 64 + lane * 2]);
        q_p_f32 = __bfloat1622float2(q_val);
    }

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
        float2 k_f_reg[8];
        
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
            float2 k_f = __bfloat1622float2(k_n);
            k_f_reg[j] = k_f;
            local_dot = fmaf(q_n_f32[j].x, k_f.x, local_dot);
            local_dot = fmaf(q_n_f32[j].y, k_f.y, local_dot);
        }
        
        {
            __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * 64 + lane * 2]);
            float2 k_f = __bfloat1622float2(k_p);
            local_dot = fmaf(q_p_f32.x, k_f.x, local_dot);
            local_dot = fmaf(q_p_f32.y, k_f.y, local_dot);
        }

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0); 
        logit *= sm_scale;

        float m_new = fmaxf(m, logit);
        float exp_diff = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        float exp_logit = __expf(logit - m_new);

        l = fmaf(l, exp_diff, exp_logit);
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float O_x_scaled = O_reg[j].x * exp_diff;
            float O_y_scaled = O_reg[j].y * exp_diff;
            O_reg[j].x = fmaf(exp_logit, k_f_reg[j].x, O_x_scaled);
            O_reg[j].y = fmaf(exp_logit, k_f_reg[j].y, O_y_scaled);
        }
    }
    __syncthreads(); // Ensure reading smem_Kc is done

    // Phase 2: Transpose Implementation
    float* smem_O = reinterpret_cast<float*>(smem_Kc);
    
    // Write to SMEM
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        smem_O[h * 512 + j * 64 + lane * 2] = O_reg[j].x;
        smem_O[h * 512 + j * 64 + lane * 2 + 1] = O_reg[j].y;
    }
    
    __syncthreads();

    // Vectorized HBM Store: each thread reads 4 float4 values
    // lane ranges 0..31. We need to write 512 values per head.
    // 32 threads * 4 * 4 floats = 512 floats. Perfect.
    // offset ranges from 0 to 512, step 16 per j.
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int offset = lane * 16 + j * 4; 
        // wait, j=0: 0..3; j=1: 4..7; j=2: 8..11; j=3: 12..15 within each thread's 16 floats.
        float4 val;
        val.x = smem_O[h * 512 + lane * 16 + j * 4];
        val.y = smem_O[h * 512 + lane * 16 + j * 4 + 1];
        val.z = smem_O[h * 512 + lane * 16 + j * 4 + 2];
        val.w = smem_O[h * 512 + lane * 16 + j * 4 + 3];
        
        int base_idx = t * 64 * 16 * 512 + s * 16 * 512 + h * 512 + lane * 16 + j * 4;
        *reinterpret_cast<float4*>(&O_tmp[base_idx]) = val;
    }
    
    if (lane == 0) {
        // Contiguous memory write for s layout [num_tokens, 16, S]
        m_tmp[t * 16 * 64 + h * 64 + s] = m;
        l_tmp[t * 16 * 64 + h * 64 + s] = l;
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
    int lane = threadIdx.x; // 0..127 (128 threads to handle 512 dimensions)

    __shared__ float m_global;
    __shared__ float l_global;
    __shared__ float m_s_smem[64];
    __shared__ float l_s_smem[64];
    __shared__ float scale_smem[64];

    // Load chunk metadata (64 chunks)
    if (lane < 64) {
        m_s_smem[lane] = m_tmp[t * 16 * 64 + h * 64 + lane];
        l_s_smem[lane] = l_tmp[t * 16 * 64 + h * 64 + lane];
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
                l_sum = fmaf(l_s_smem[i], __expf(m_s_smem[i] - m_max), l_sum);
            }
        }
        
        m_global = m_max;
        l_global = l_sum;

        // Base-2 LSE mathematical enforcement
        if (m_max == -INFINITY) {
            lse_ptr[t * 16 + h] = -INFINITY;
        } else {
            lse_ptr[t * 16 + h] = fmaf(m_max, 1.4426950408889634f, __log2f(l_sum));
        }
    }
    __syncthreads();

    // Cache the scales for all 64 chunks
    if (lane < 64) {
        if (m_global != -INFINITY && l_global > 0.0f && m_s_smem[lane] != -INFINITY) {
            scale_smem[lane] = __fdividef(__expf(m_s_smem[lane] - m_global), l_global);
        } else {
            scale_smem[lane] = 0.0f;
        }
    }
    __syncthreads();

    // Accumulate final normalized Output Projection
    int d_idx = lane * 4;
    if (d_idx < 512) {
        float out0 = 0.0f;
        float out1 = 0.0f;
        float out2 = 0.0f;
        float out3 = 0.0f;
        
        if (m_global != -INFINITY && l_global > 0.0f) {
            #pragma unroll 16
            for (int s = 0; s < 64; ++s) {
                float scale = scale_smem[s];
                if (scale > 0.0f) {
                    int base_idx = t * 64 * 16 * 512 + s * 16 * 512 + h * 512 + d_idx;
                    float4 val = *reinterpret_cast<const float4*>(&O_tmp[base_idx]);
                    out0 = fmaf(val.x, scale, out0);
                    out1 = fmaf(val.y, scale, out1);
                    out2 = fmaf(val.z, scale, out2);
                    out3 = fmaf(val.w, scale, out3);
                }
            }
        }

        // Store back to BF16 accurately
        __nv_bfloat162 out_bf16_0 = __floats2bfloat162_rn(out0, out1);
        __nv_bfloat162 out_bf16_1 = __floats2bfloat162_rn(out2, out3);
        
        uint32_t val0 = *reinterpret_cast<uint32_t*>(&out_bf16_0);
        uint32_t val1 = *reinterpret_cast<uint32_t*>(&out_bf16_1);
        uint2 out_vec = make_uint2(val0, val1);
        *reinterpret_cast<uint2*>(&output_ptr[t * 16 * 512 + h * 512 + d_idx]) = out_vec;
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

        dim3 reduce_grid(num_tokens, 16, 1);
        dim3 reduce_block(128, 1, 1); // Optimized block size (4 warps instead of 8)
        split_k_reduce_kernel<<<reduce_grid, reduce_block>>>(
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
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
    m.def("dsa_forward", &dsa_forward, "DSA Forward Kernel with Register Caching and FMA");
}