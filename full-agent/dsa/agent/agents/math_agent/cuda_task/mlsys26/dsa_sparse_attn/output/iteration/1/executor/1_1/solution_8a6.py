#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

__device__ __forceinline__ float2 bf1622float2(const __nv_bfloat162& val) {
    return __bfloat1622float2(val);
}

__device__ __forceinline__ __nv_bfloat162 floats2bf162(float x, float y) {
    return __floats2bfloat162_rn(x, y);
}

__global__ void dsa_forward_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr,
    int topk
) {
    int64_t t_idx = blockIdx.x;
    int64_t h_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    // Allocate statically as the total size is ~36KB which easily fits within the 48KB limit
    __shared__ __nv_bfloat16 Kc_tile[32][512];
    __shared__ __nv_bfloat16 Kp_tile[32][64];
    __shared__ int indices_tile[32];

    __nv_bfloat162 q_n_frag[8];
    __nv_bfloat162 q_p_frag[1];
    float out_frag_0[8];
    float out_frag_1[8];
    float m_val = -INFINITY;
    float d_val = 0.0f;

    // Initialize local registers (perfectly coalesced load)
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int col = k * 64 + lane * 2;
        int64_t qn_offset = t_idx * 16 * 512 + h_idx * 512 + col;
        uint32_t val = *reinterpret_cast<const uint32_t*>(&q_nope_ptr[qn_offset]);
        q_n_frag[k] = *reinterpret_cast<const __nv_bfloat162*>(&val);
        out_frag_0[k] = 0.0f;
        out_frag_1[k] = 0.0f;
    }
    {
        int col = lane * 2;
        int64_t qp_offset = t_idx * 16 * 64 + h_idx * 64 + col;
        uint32_t val = *reinterpret_cast<const uint32_t*>(&q_pe_ptr[qp_offset]);
        q_p_frag[0] = *reinterpret_cast<const __nv_bfloat162*>(&val);
    }

    int num_chunks = (topk + 31) / 32;

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        // Load sparse indices
        if (threadIdx.x < 32) {
            int idx_idx = chunk * 32 + threadIdx.x;
            indices_tile[threadIdx.x] = (idx_idx < topk) ? sparse_indices_ptr[t_idx * topk + idx_idx] : -1;
        }
        __syncthreads();

        // Cooperatively load Kc_tile using 128-bit memory instructions
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int job_idx = step * 512 + threadIdx.x;
            int row = job_idx / 64;
            int col_chunk = job_idx % 64;
            int kv_idx = indices_tile[row];
            if (kv_idx != -1) {
                int64_t ckv_offset = (int64_t)kv_idx * 512;
                const uint4* ptr = reinterpret_cast<const uint4*>(ckv_cache_ptr + ckv_offset);
                reinterpret_cast<uint4*>(Kc_tile[row])[col_chunk] = ptr[col_chunk];
            } else {
                reinterpret_cast<uint4*>(Kc_tile[row])[col_chunk] = make_uint4(0, 0, 0, 0);
            }
        }
        
        // Cooperatively load Kp_tile using 128-bit memory instructions
        if (threadIdx.x < 256) {
            int job_idx = threadIdx.x;
            int row = job_idx / 8;
            int col_chunk = job_idx % 8;
            int kv_idx = indices_tile[row];
            if (kv_idx != -1) {
                int64_t kpe_offset = (int64_t)kv_idx * 64;
                const uint4* ptr = reinterpret_cast<const uint4*>(kpe_cache_ptr + kpe_offset);
                reinterpret_cast<uint4*>(Kp_tile[row])[col_chunk] = ptr[col_chunk];
            } else {
                reinterpret_cast<uint4*>(Kp_tile[row])[col_chunk] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        int limit = (chunk == num_chunks - 1) ? (topk - chunk * 32) : 32;

        for (int i = 0; i < limit; ++i) {
            int kv_idx = indices_tile[i];
            float sum = 0.0f;

            // Dot product avoiding bank conflicts by reading 32-bit (2x bf16) at once
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int col = k * 64 + lane * 2;
                uint32_t kc_val_u32 = *reinterpret_cast<const uint32_t*>(&Kc_tile[i][col]);
                __nv_bfloat162 kc_val = *reinterpret_cast<const __nv_bfloat162*>(&kc_val_u32);
                float2 qn_f2 = bf1622float2(q_n_frag[k]);
                float2 kc_f2 = bf1622float2(kc_val);
                sum += qn_f2.x * kc_f2.x + qn_f2.y * kc_f2.y;
            }
            {
                int col = lane * 2;
                uint32_t kp_val_u32 = *reinterpret_cast<const uint32_t*>(&Kp_tile[i][col]);
                __nv_bfloat162 kp_val = *reinterpret_cast<const __nv_bfloat162*>(&kp_val_u32);
                float2 qp_f2 = bf1622float2(q_p_frag[0]);
                float2 kp_f2 = bf1622float2(kp_val);
                sum += qp_f2.x * kp_f2.x + qp_f2.y * kp_f2.y;
            }

            // Warp level parallel reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            // Broadcast final logit to all threads in the warp
            sum = __shfl_sync(0xffffffff, sum, 0);
            float logit = sum * sm_scale;
            if (kv_idx == -1) {
                logit = -INFINITY;
            }

            // Online Softmax Logic
            float m_new = fmaxf(m_val, logit);
            float exp_diff = (m_val == -INFINITY) ? 0.0f : expf(m_val - m_new);
            float exp_logit = (logit == -INFINITY) ? 0.0f : expf(logit - m_new);

            d_val = d_val * exp_diff + exp_logit;
            m_val = m_new;

            // Output state update
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int col = k * 64 + lane * 2;
                uint32_t kc_val_u32 = *reinterpret_cast<const uint32_t*>(&Kc_tile[i][col]);
                __nv_bfloat162 kc_val = *reinterpret_cast<const __nv_bfloat162*>(&kc_val_u32);
                float2 kc_f2 = bf1622float2(kc_val);
                
                out_frag_0[k] = out_frag_0[k] * exp_diff + exp_logit * kc_f2.x;
                out_frag_1[k] = out_frag_1[k] * exp_diff + exp_logit * kc_f2.y;
            }
        }
        __syncthreads();
    }

    // Write final output
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float final_out_0 = (d_val > 0.0f) ? (out_frag_0[k] / d_val) : 0.0f;
        float final_out_1 = (d_val > 0.0f) ? (out_frag_1[k] / d_val) : 0.0f;
        
        __nv_bfloat162 final_bf2 = floats2bf162(final_out_0, final_out_1);
        int col = k * 64 + lane * 2;
        int64_t out_offset = t_idx * 16 * 512 + h_idx * 512 + col;
        *reinterpret_cast<uint32_t*>(&output_ptr[out_offset]) = *reinterpret_cast<uint32_t*>(&final_bf2);
    }

    // Write Log-Sum-Exp
    if (lane == 0) {
        float lse_val;
        if (d_val > 0.0f) {
            // log2(e) ≈ 1.44269504f to compute base-2 LSE strictly
            lse_val = m_val * 1.44269504f + log2f(d_val);
        } else {
            lse_val = -INFINITY;
        }
        lse_ptr[t_idx * 16 + h_idx] = lse_val;
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
    int topk = sparse_indices.size(1);

    auto output = torch::zeros_like(q_nope);
    auto lse = torch::empty({num_tokens, 16}, torch::dtype(torch::kFloat32).device(q_nope.device()));

    if (num_tokens > 0) {
        dim3 grid(num_tokens);
        dim3 block(512);

        dsa_forward_kernel<<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            topk
        );
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &dsa_forward, "Batched Native Sparse Attention Forward");
}