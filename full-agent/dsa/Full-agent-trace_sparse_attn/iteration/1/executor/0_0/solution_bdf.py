#include <torch/extension.h>
#include <tuple>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

// Vectorized utility for converting 8 bfloat16 elements to 8 fp32 elements
__device__ __forceinline__ void unpack_bf16x8_to_f32(const float4& src, float* dst) {
    const __nv_bfloat162* bf162_ptr = reinterpret_cast<const __nv_bfloat162*>(&src);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f2 = __bfloat1622float2(bf162_ptr[i]);
        dst[i*2] = f2.x;
        dst[i*2+1] = f2.y;
    }
}

// Highly Optimized CUDA Kernel for Native Sparse Attention
__global__ void dsa_forward_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int32_t* __restrict__ sparse_indices,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    float sm_scale,
    int topk
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int w = tid / 32;       // Warp index (0-15) -> maps to 16 Query Heads
    int lane = tid % 32;    // Thread index within warp

    // Shared Memory declarations: strictly bounded to fit within fast static SMEM limits (~37KB)
    __shared__ __align__(16) __nv_bfloat16 smem_ckv[32][512];
    __shared__ __align__(16) __nv_bfloat16 smem_kpe[32][64];
    __shared__ int smem_idx[32];

    // Thread-local Registers for FP32 Accumulations & Computations
    float qn_f32[16];
    float qp_f32[2];
    float m_i = -INFINITY;
    float d_i = 0.0f;
    float out_f32[16] = {0.0f};

    // Pre-Load Q_nope & Q_pe globally into thread-local FP32 registers (Once per block)
    const float4* qn_src = reinterpret_cast<const float4*>(q_nope + token_idx * 16 * 512 + w * 512 + lane * 16);
    unpack_bf16x8_to_f32(qn_src[0], qn_f32);
    unpack_bf16x8_to_f32(qn_src[1], qn_f32 + 8);

    const __nv_bfloat162* qp_src = reinterpret_cast<const __nv_bfloat162*>(q_pe + token_idx * 16 * 64 + w * 64 + lane * 2);
    float2 qp_f2 = __bfloat1622float2(*qp_src);
    qp_f32[0] = qp_f2.x;
    qp_f32[1] = qp_f2.y;

    // Iterate over the dynamically gathered TopK Cache in optimal tiles of 32
    int num_steps = topk / 32;
    for (int step = 0; step < num_steps; ++step) {
        // Asynchronous/Cooperative Loading Phase
        if (tid < 32) {
            smem_idx[tid] = sparse_indices[token_idx * topk + step * 32 + tid];
        }
        __syncthreads();

        int k_load = tid / 16;
        int col_ckv = (tid % 16) * 32;
        int idx_load = smem_idx[k_load];
        
        if (idx_load != -1) {
            // Coalesced 128-bit vectorized loading of CKV Cache
            const float4* src_ckv = reinterpret_cast<const float4*>(ckv_cache + idx_load * 512 + col_ckv);
            float4* dst_ckv = reinterpret_cast<float4*>(&smem_ckv[k_load][col_ckv]);
            dst_ckv[0] = src_ckv[0];
            dst_ckv[1] = src_ckv[1];
            dst_ckv[2] = src_ckv[2];
            dst_ckv[3] = src_ckv[3];

            // Coalesced 64-bit loading of KPE Cache
            int col_kpe = (tid % 16) * 4;
            const float2* src_kpe = reinterpret_cast<const float2*>(kpe_cache + idx_load * 64 + col_kpe);
            float2* dst_kpe = reinterpret_cast<float2*>(&smem_kpe[k_load][col_kpe]);
            dst_kpe[0] = src_kpe[0];
        }
        __syncthreads();

        // Warp-Level Math Compute Phase
        #pragma unroll 4
        for (int k = 0; k < 32; ++k) {
            int idx = smem_idx[k];
            float logit = -INFINITY;
            float ckv_f32[16];

            if (idx != -1) {
                float sum = 0.0f;
                // Dot Product over CKV
                const float4* ckv_src = reinterpret_cast<const float4*>(&smem_ckv[k][lane * 16]);
                unpack_bf16x8_to_f32(ckv_src[0], ckv_f32);
                unpack_bf16x8_to_f32(ckv_src[1], ckv_f32 + 8);
                #pragma unroll
                for(int i=0; i<16; ++i) {
                    sum += qn_f32[i] * ckv_f32[i];
                }

                // Dot Product over KPE
                const __nv_bfloat162* kpe_src = reinterpret_cast<const __nv_bfloat162*>(&smem_kpe[k][lane * 2]);
                float2 kpe_f2 = __bfloat1622float2(*kpe_src);
                sum += qp_f32[0] * kpe_f2.x + qp_f32[1] * kpe_f2.y;

                // Logarithmic Warp-Level Reduction
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    sum += __shfl_down_sync(0xffffffff, sum, offset);
                }
                
                // Broadcast final dot-product sum
                logit = __shfl_sync(0xffffffff, sum, 0);
                logit *= sm_scale;
            }

            // Online Safe Softmax with Local State Update (Matched dynamically to FP32 Precision)
            float m_prev = m_i;
            m_i = fmaxf(m_i, logit);
            
            // Defend mathematically against -INF handling
            float exp_diff = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_i);
            d_i = d_i * exp_diff;
            float p = (logit == -INFINITY) ? 0.0f : expf(logit - m_i);
            d_i += p;

            // Update local value components in FP32 explicitly
            #pragma unroll
            for(int i=0; i<16; ++i) {
                float v = (idx != -1) ? ckv_f32[i] : 0.0f;
                out_f32[i] = out_f32[i] * exp_diff + p * v;
            }
        }
        __syncthreads();
    }

    // Output Finalization & Normalization
    if (d_i > 0.0f) {
        #pragma unroll
        for(int i=0; i<16; ++i) {
            out_f32[i] /= d_i;
        }
    }

    // Vectorized precision collapse back down to BF16
    __nv_bfloat162 out_bf162[8];
    #pragma unroll
    for(int i=0; i<8; ++i) {
        out_bf162[i] = __floats2bfloat162_rn(out_f32[i*2], out_f32[i*2+1]);
    }
    
    // Store exact memory blocks
    float4* out_dst = reinterpret_cast<float4*>(output + token_idx * 16 * 512 + w * 512 + lane * 16);
    out_dst[0] = reinterpret_cast<float4*>(out_bf162)[0];
    out_dst[1] = reinterpret_cast<float4*>(out_bf162)[1];

    // Compute Base-2 LSE exclusively for Head 0
    if (lane == 0) {
        // Fast Log2 Transformation derived from Base e limits directly maps to PyTorch standard limits
        float lse_val = (m_i == -INFINITY) ? -INFINITY : (m_i * 1.4426950408889634f) + log2f(d_i);
        lse[token_idx * 16 + w] = lse_val;
    }
}

// Entry point – must have this exact name and signature
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

    auto options = torch::TensorOptions().dtype(q_nope.dtype()).device(q_nope.device());
    torch::Tensor output = torch::zeros({num_tokens, 16, 512}, options);
    
    auto lse_options = torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device());
    torch::Tensor lse = torch::full({num_tokens, 16}, -INFINITY, lse_options);

    if (num_tokens > 0) {
        // 1 Token maps 1 Block. 512 Threads equals 16 Warps dynamically parallelized 
        dim3 grid(num_tokens);
        dim3 block(512);

        dsa_forward_kernel<<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            sm_scale,
            topk
        );
    }

    return {output, lse};
}