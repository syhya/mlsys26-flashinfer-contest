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
    // Grid maps to tokens, Y-block maps to heads (1 warp per head), X-block maps to lanes
    int t = blockIdx.x;
    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = threadIdx.y * 32 + threadIdx.x;

    // Shared Memory declarations for cooperative tile loading
    __shared__ alignas(16) int idx_shared[32];
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * 512];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * 64];

    // Thread-local registers for Query Nope (512-dim -> 8 x float4 equiv per thread)
    // Distributed as 2 elements (1 __nv_bfloat162) per group of 64 to avoid SMEM bank conflicts
    __nv_bfloat162 q_n_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        q_n_reg[j] = *reinterpret_cast<const __nv_bfloat162*>(&q_nope_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]);
    }

    // Thread-local registers for Query PE (64-dim)
    __nv_bfloat162 q_p_reg[1];
    q_p_reg[0] = *reinterpret_cast<const __nv_bfloat162*>(&q_pe_ptr[t * 16 * 64 + h * 64 + lane * 2]);

    // Thread-local accumulators for Output Attention (512-dim)
    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        O_reg[j] = make_float2(0.0f, 0.0f);
    }

    // Online Softmax State
    float m = -INFINITY;
    float l = 0.0f;

    // Process topk = 2048 keys in tiles of 32
    int num_tiles = 2048 / 32;
    for (int tile = 0; tile < num_tiles; ++tile) {
        
        // 1. Cooperative load of Sparse Indices
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * 2048 + tile * 32 + tid];
        }
        __syncthreads();

        // 2. Cooperative Vectorized Load of Key Caches into Shared Memory
        int row = tid / 16;
        int col_group = tid % 16;
        if (row < 32) {
            int idx = idx_shared[row];
            if (idx != -1) {
                // Kc: 512 elements per row. 16 threads load 32 elements each (4 float4s). Perfectly coalesced 128-byte transactions.
                const float4* src_c = reinterpret_cast<const float4*>(&ckv_cache_ptr[idx * 512 + col_group * 32]);
                float4* dst_c = reinterpret_cast<float4*>(&smem_Kc[row * 512 + col_group * 32]);
                dst_c[0] = src_c[0];
                dst_c[1] = src_c[1];
                dst_c[2] = src_c[2];
                dst_c[3] = src_c[3];

                // Kp: 64 elements per row. 16 threads load 4 elements each (1 float2).
                const float2* src_p = reinterpret_cast<const float2*>(&kpe_cache_ptr[idx * 64 + col_group * 4]);
                float2* dst_p = reinterpret_cast<float2*>(&smem_Kp[row * 64 + col_group * 4]);
                dst_p[0] = src_p[0];
            }
        }
        __syncthreads();

        // 3. Compute Attention for the Tile (1 Warp per Head)
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] == -1) continue; // Skip strictly padded indices

            float local_dot = 0.0f;
            
            // Query * Key (Nope portion)
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
                float2 q_f = __bfloat1622float2(q_n_reg[j]);
                float2 k_f = __bfloat1622float2(k_n);
                local_dot += q_f.x * k_f.x + q_f.y * k_f.y;
            }
            
            // Query * Key (PE portion)
            {
                __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * 64 + lane * 2]);
                float2 q_f = __bfloat1622float2(q_p_reg[0]);
                float2 k_f = __bfloat1622float2(k_p);
                local_dot += q_f.x * k_f.x + q_f.y * k_f.y;
            }

            // Fast Warp Reduction
            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0); // Broadcast sum from lane 0
            
            logit *= sm_scale;

            // Online Softmax State Update (Exact FP32 Base-e internally)
            float m_new = fmaxf(m, logit);
            float exp_diff = (m == -INFINITY) ? 0.0f : expf(m - m_new);
            float exp_logit = expf(logit - m_new);

            l = l * exp_diff + exp_logit;
            m = m_new;

            // Accumulate Output Projection (Value component is same as Key Nope component)
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
                float2 k_f = __bfloat1622float2(k_n);
                O_reg[j].x = O_reg[j].x * exp_diff + exp_logit * k_f.x;
                O_reg[j].y = O_reg[j].y * exp_diff + exp_logit * k_f.y;
            }
        }
        __syncthreads(); // Ensure tile compute finishes before next cooperative load overwrites SMEM
    }

    // 4. Finalize and Normalize Outputs
    if (l > 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x /= l;
            O_reg[j].y /= l;
        }
    } else {
        // Edge case: All padding (fallback matches PyTorch zeroing exactly)
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x = 0.0f;
            O_reg[j].y = 0.0f;
        }
    }

    // Write Coalesced Output Vectorized
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 out_bf16 = __floats2bfloat162_rn(O_reg[j].x, O_reg[j].y);
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]) = out_bf16;
    }

    // Write Log-Sum-Exp (Base 2 mathematically enforced: LSE = m/log(2) + log2(l))
    if (lane == 0) {
        float lse_val = (l > 0.0f) ? (m * 1.4426950408889634f + log2f(l)) : -INFINITY;
        lse_ptr[t * 16 + h] = lse_val;
    }
}

std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
    torch::Tensor q_nope,         // [num_tokens, 16, 512]  bfloat16
    torch::Tensor q_pe,           // [num_tokens, 16, 64]   bfloat16
    torch::Tensor ckv_cache,      // [num_pages, 64, 512]   bfloat16
    torch::Tensor kpe_cache,      // [num_pages, 64, 64]    bfloat16
    torch::Tensor sparse_indices, // [num_tokens, 2048]     int32
    float sm_scale                // scalar: 1/sqrt(192)
) {
    int num_tokens = q_nope.size(0);
    
    // Allocate outputs directly on appropriate device and match baseline exactly
    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16}, torch::dtype(torch::kFloat32).device(q_nope.device()));

    if (num_tokens == 0) {
        return {output, lse};
    }

    // Grid Topology: 1 Block per Token (Batch), handling all 16 Heads via 16 Warps
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

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward Kernel");
}