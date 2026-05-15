#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>

// CUDA Kernel for Batched Native Sparse Attention (DSA)
__global__ void dsa_kernel(
    const nv_bfloat16* __restrict__ q_nope,
    const nv_bfloat16* __restrict__ q_pe,
    const nv_bfloat16* __restrict__ ckv_cache,
    const nv_bfloat16* __restrict__ kpe_cache,
    const int* __restrict__ sparse_indices,
    nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    float sm_scale
) {
    int bx = blockIdx.x; // Token ID
    int tx = threadIdx.x; // 0..31 (Thread in warp)
    int ty = threadIdx.y; // 0..15 (Warp ID / Head ID)

    int tid = ty * 32 + tx; // 0..511 (Thread in block)

    // 1. Load Q_nope [16 elements per thread] into registers
    uint4 qn_u4[2];
    size_t qn_idx = bx * 16 * 512 + ty * 512 + tx * 16;
    const uint4* qn_src_u4 = reinterpret_cast<const uint4*>(q_nope + qn_idx);
    qn_u4[0] = qn_src_u4[0];
    qn_u4[1] = qn_src_u4[1];
    
    nv_bfloat162 qn_2[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        qn_2[i] = reinterpret_cast<nv_bfloat162*>(qn_u4)[i];
    }

    // 2. Load Q_pe [2 elements per thread] into registers
    size_t qp_idx = bx * 16 * 64 + ty * 64 + tx * 2;
    const uint32_t* qp_src_u32 = reinterpret_cast<const uint32_t*>(q_pe + qp_idx);
    nv_bfloat162 qp_2[1];
    *reinterpret_cast<uint32_t*>(&qp_2[0]) = *qp_src_u32;

    // 3. Setup dynamic shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    nv_bfloat16* smem_kc = reinterpret_cast<nv_bfloat16*>(smem);                             // 64 * 512 * 2 = 65536 bytes
    nv_bfloat16* smem_kpe = reinterpret_cast<nv_bfloat16*>(smem + 65536);                    // 64 * 64 * 2 = 8192 bytes
    int* smem_idx = reinterpret_cast<int*>(smem + 73728);                                    // 64 * 4 = 256 bytes

    uint4* smem_kc_uint4 = reinterpret_cast<uint4*>(smem_kc);
    uint4* smem_kpe_uint4 = reinterpret_cast<uint4*>(smem_kpe);

    const uint4* ckv_cache_u4 = reinterpret_cast<const uint4*>(ckv_cache);
    const uint4* kpe_cache_u4 = reinterpret_cast<const uint4*>(kpe_cache);
    const int* sparse_idx_ptr = sparse_indices + bx * 2048;

    // 4. Initialize Online Softmax State
    float m_val = -INFINITY;
    float d_val = 0.0f;
    float O[16] = {0.0f}; // Accumulators for Output

    // 5. Tiled Processing over Top-K Keys
    for (int k_start = 0; k_start < 2048; k_start += 64) {
        // Load sparse indices for the current tile
        if (tid < 64) {
            smem_idx[tid] = sparse_idx_ptr[k_start + tid];
        }
        __syncthreads();

        // Cooperatively load K_c tile (64 keys * 512 dim)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int task_idx = i * 512 + tid; // 512 threads * 8 = 4096 uint4s
            int row = task_idx / 64;
            int col_u4 = task_idx % 64;
            int idx = smem_idx[row];
            if (idx != -1) {
                smem_kc_uint4[task_idx] = ckv_cache_u4[idx * 64 + col_u4];
            } else {
                smem_kc_uint4[task_idx] = make_uint4(0, 0, 0, 0);
            }
        }

        // Cooperatively load K_pe tile (64 keys * 64 dim)
        {
            int row = tid / 8;
            int col_u4 = tid % 8;
            int idx = smem_idx[row];
            if (idx != -1) {
                smem_kpe_uint4[tid] = kpe_cache_u4[idx * 8 + col_u4];
            } else {
                smem_kpe_uint4[tid] = make_uint4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        // Compute Attention for the current tile
        for (int k = 0; k < 64; ++k) {
            int idx = smem_idx[k];
            
            // Vectorized Dot Product
            float partial_sum = 0.0f;
            const nv_bfloat162* kc_ptr = reinterpret_cast<const nv_bfloat162*>(smem_kc + k * 512 + tx * 16);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float2 q = __bfloat1622float2(qn_2[i]);
                float2 k_val = __bfloat1622float2(kc_ptr[i]);
                partial_sum += q.x * k_val.x + q.y * k_val.y;
            }
            
            const nv_bfloat162* kpe_ptr = reinterpret_cast<const nv_bfloat162*>(smem_kpe + k * 64 + tx * 2);
            float2 q_p = __bfloat1622float2(qp_2[0]);
            float2 k_p = __bfloat1622float2(kpe_ptr[0]);
            partial_sum += q_p.x * k_p.x + q_p.y * k_p.y;

            // Fast Warp Reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
            }
            float logit = __shfl_sync(0xffffffff, partial_sum, 0);

            // Safety Guard & Masking
            if (idx == -1) {
                logit = -INFINITY;
            } else {
                logit *= sm_scale;
            }

            // FlashAttention Online Softmax Recurrence
            float m_prev = m_val;
            m_val = fmaxf(m_prev, logit);
            
            float exp_diff = (m_val == -INFINITY) ? 0.0f : expf(m_prev - m_val);
            float exp_logit = (logit == -INFINITY) ? 0.0f : expf(logit - m_val);

            d_val = d_val * exp_diff + exp_logit;

            // Accumulate V (K_c)
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float2 k_val = __bfloat1622float2(kc_ptr[i]);
                O[i * 2 + 0] = O[i * 2 + 0] * exp_diff + exp_logit * k_val.x;
                O[i * 2 + 1] = O[i * 2 + 1] * exp_diff + exp_logit * k_val.y;
            }
        }

        __syncthreads();
    }

    // 6. Normalize and Store Output
    float inv_d = (d_val > 0.0f) ? (1.0f / d_val) : 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        O[i] *= inv_d;
    }

    // Coalesced Vectorized Global Write
    uint4 out_u4_0;
    uint4 out_u4_1;
    uint32_t* u0 = reinterpret_cast<uint32_t*>(&out_u4_0);
    uint32_t* u1 = reinterpret_cast<uint32_t*>(&out_u4_1);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        nv_bfloat162 bf2 = __floats2bfloat162_rn(O[i * 2 + 0], O[i * 2 + 1]);
        u0[i] = *reinterpret_cast<uint32_t*>(&bf2);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        nv_bfloat162 bf2 = __floats2bfloat162_rn(O[8 + i * 2 + 0], O[8 + i * 2 + 1]);
        u1[i] = *reinterpret_cast<uint32_t*>(&bf2);
    }
    size_t out_idx = bx * 16 * 512 + ty * 512 + tx * 16;
    uint4* global_out = reinterpret_cast<uint4*>(output + out_idx);
    global_out[0] = out_u4_0;
    global_out[1] = out_u4_1;

    // 7. Store Exact Base-2 LSE
    if (tx == 0) {
        float lse_val;
        if (m_val == -INFINITY) {
            lse_val = -INFINITY;
        } else {
            // log_2(e) = 1.4426950408889634
            lse_val = (m_val + logf(d_val)) * 1.4426950408889634f;
        }
        lse[bx * 16 + ty] = lse_val;
    }
}

// C++ / PyBind11 Entry Point
std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
    torch::Tensor q_nope,         
    torch::Tensor q_pe,           
    torch::Tensor ckv_cache,      
    torch::Tensor kpe_cache,      
    torch::Tensor sparse_indices, 
    float sm_scale                
) {
    // Ensure tensors are contiguous in memory
    auto q_nope_c = q_nope.contiguous();
    auto q_pe_c = q_pe.contiguous();
    auto ckv_cache_c = ckv_cache.contiguous();
    auto kpe_cache_c = kpe_cache.contiguous();
    auto sparse_indices_c = sparse_indices.contiguous();

    int num_tokens = q_nope_c.size(0);

    // Output allocation
    auto options = q_nope_c.options();
    auto output = torch::empty({num_tokens, 16, 512}, options);
    auto lse = torch::empty({num_tokens, 16}, options.dtype(torch::kFloat32));

    // Topology configuration
    dim3 grid(num_tokens);
    dim3 block(32, 16); // 512 Threads: 16 Warps * 32 Threads
    int shared_mem_size = 73984; // SMEM Required: 64*512*2 + 64*64*2 + 64*4

    cudaFuncSetAttribute(dsa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    dsa_kernel<<<grid, block, shared_mem_size>>>(
        reinterpret_cast<const nv_bfloat16*>(q_nope_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(q_pe_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(ckv_cache_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(kpe_cache_c.data_ptr<at::BFloat16>()),
        sparse_indices_c.data_ptr<int>(),
        reinterpret_cast<nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        sm_scale
    );

    return std::make_tuple(output, lse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward Kernel");
}