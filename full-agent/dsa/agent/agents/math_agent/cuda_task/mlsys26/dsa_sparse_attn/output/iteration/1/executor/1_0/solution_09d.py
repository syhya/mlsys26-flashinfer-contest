#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <limits>

// Define NEG_INF for fallback and masking
const float NEG_INF = -std::numeric_limits<float>::infinity();

__global__ void dsa_forward_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int32_t* __restrict__ sparse_indices,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    float sm_scale,
    int num_tokens,
    int topk
) {
    int t = blockIdx.x;
    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = h * 32 + lane; // 512 threads total per block

    if (t >= num_tokens) return;

    // Phase 1: Load Q tensors entirely into thread-local registers
    // Each warp handles 1 query head completely. Thread `lane` processes 16 elements of q_nope and 2 elements of q_pe.
    float4 q_n_f4_0 = reinterpret_cast<const float4*>(q_nope + t * 16 * 512 + h * 512)[lane * 2];
    float4 q_n_f4_1 = reinterpret_cast<const float4*>(q_nope + t * 16 * 512 + h * 512)[lane * 2 + 1];

    __nv_bfloat162 q_n[8];
    reinterpret_cast<float4*>(q_n)[0] = q_n_f4_0;
    reinterpret_cast<float4*>(q_n + 4)[0] = q_n_f4_1;

    int32_t q_p_i32 = reinterpret_cast<const int32_t*>(q_pe + t * 16 * 64 + h * 64)[lane];
    __nv_bfloat162 q_p_bf2 = reinterpret_cast<__nv_bfloat162&>(q_p_i32);

    // Initialize Online Softmax and output accumulators in registers
    float m_val = NEG_INF;
    float d_val = 0.0f;
    float out_acc[16] = {0.0f};

    // Shared memory layouts for optimal coalesced streaming
    extern __shared__ int8_t shared_mem[];
    __nv_bfloat16* smem_ckv = reinterpret_cast<__nv_bfloat16*>(shared_mem);
    __nv_bfloat16* smem_kpe = smem_ckv + 32 * 512;
    int32_t* smem_indices = reinterpret_cast<int32_t*>(smem_kpe + 32 * 64);

    // Phase 2: Iterate over KV cache in dynamic tiles of size 32
    for (int start_k = 0; start_k < topk; start_k += 32) {
        int valid_k = min(32, topk - start_k);

        // Step 2.1: Gather dynamic indices into shared memory
        if (tid < 32) {
            if (tid < valid_k) {
                smem_indices[tid] = sparse_indices[t * topk + start_k + tid];
            } else {
                smem_indices[tid] = -1; // Pad remaining to avoid out-of-bounds
            }
        }
        __syncthreads();

        // Step 2.2: Vectorized Cooperative Load of KV into Shared Memory
        // CKV Cache: 32 rows * 512 elements mapped gracefully across 512 threads using float4
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int row = step * 8 + (tid / 64);
            int col = tid % 64; // float4 index
            int idx = smem_indices[row];
            if (idx != -1) {
                float4 val = reinterpret_cast<const float4*>(ckv_cache + idx * 512)[col];
                reinterpret_cast<float4*>(smem_ckv + row * 512)[col] = val;
            } else {
                // mathematically zero out bfloat16 bytes if padding
                reinterpret_cast<float4*>(smem_ckv + row * 512)[col] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        // KPE Cache: 32 rows * 64 elements
        if (tid < 256) {
            int row = tid / 8;
            int col = tid % 8;
            int idx = smem_indices[row];
            if (idx != -1) {
                float4 val = reinterpret_cast<const float4*>(kpe_cache + idx * 64)[col];
                reinterpret_cast<float4*>(smem_kpe + row * 64)[col] = val;
            } else {
                reinterpret_cast<float4*>(smem_kpe + row * 64)[col] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads(); // Synchronize before inner warp compute phase

        // Phase 3: Inner Math Kernel
        for (int k = 0; k < valid_k; ++k) {
            int idx = smem_indices[k];

            // Load mapped slices of KV efficiently avoiding bank conflicts
            float4 k_c_f4_0 = reinterpret_cast<float4*>(smem_ckv + k * 512)[lane * 2];
            float4 k_c_f4_1 = reinterpret_cast<float4*>(smem_ckv + k * 512)[lane * 2 + 1];
            int32_t k_p_i32 = reinterpret_cast<int32_t*>(smem_kpe + k * 64)[lane];

            __nv_bfloat162 k_c[8];
            reinterpret_cast<float4*>(k_c)[0] = k_c_f4_0;
            reinterpret_cast<float4*>(k_c + 4)[0] = k_c_f4_1;

            // Dot Product computation in high-precision float32
            float sum = 0.0f;
            #pragma unroll
            for(int i = 0; i < 8; ++i) {
                float2 q_f2 = __bfloat1622float2(q_n[i]);
                float2 k_f2 = __bfloat1622float2(k_c[i]);
                sum += q_f2.x * k_f2.x;
                sum += q_f2.y * k_f2.y;
            }

            __nv_bfloat162 k_p_bf2 = reinterpret_cast<__nv_bfloat162&>(k_p_i32);
            float2 q_p_f2 = __bfloat1622float2(q_p_bf2);
            float2 k_p_f2 = __bfloat1622float2(k_p_bf2);
            sum += q_p_f2.x * k_p_f2.x;
            sum += q_p_f2.y * k_p_f2.y;

            // Fast Warp Reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            // Phase 4: Online Softmax Safety Valve Check
            float logit = __shfl_sync(0xffffffff, sum, 0);
            logit *= sm_scale;

            if (idx == -1) {
                logit = NEG_INF; // Mask out invalid tokens (-1 indices)
            }

            float m_prev = m_val;
            m_val = fmaxf(m_prev, logit);

            // Compute Base-e differences robustly preventing NaNs
            float exp_diff = (m_prev == NEG_INF) ? 0.0f : expf(m_prev - m_val);
            float p = expf(logit - m_val);
            if (logit == NEG_INF) p = 0.0f;

            d_val = d_val * exp_diff + p;

            // Accumulate fractional components into registers
            #pragma unroll
            for(int i = 0; i < 8; ++i) {
                float2 k_f2 = __bfloat1622float2(k_c[i]);
                out_acc[i * 2]     = out_acc[i * 2] * exp_diff + p * k_f2.x;
                out_acc[i * 2 + 1] = out_acc[i * 2 + 1] * exp_diff + p * k_f2.y;
            }
        }
        __syncthreads(); // ensure trailing execution drops before fetching new shared mem
    }

    // Phase 5: Normalization and Write-back
    if (d_val > 0.0f) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            out_acc[j] /= d_val;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            out_acc[j] = 0.0f; // All-padding perfect fallback
        }
    }

    __nv_bfloat162 out_bf2[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_bf2[i] = __floats2bfloat162_rn(out_acc[i * 2], out_acc[i * 2 + 1]);
    }

    float4 out_f4_0 = reinterpret_cast<float4*>(out_bf2)[0];
    float4 out_f4_1 = reinterpret_cast<float4*>(out_bf2 + 4)[0];

    reinterpret_cast<float4*>(output + t * 16 * 512 + h * 512)[lane * 2] = out_f4_0;
    reinterpret_cast<float4*>(output + t * 16 * 512 + h * 512)[lane * 2 + 1] = out_f4_1;

    // Strict mathematical adjustment to Base-2 LSE
    if (lane == 0) {
        float lse_out;
        if (d_val > 0.0f) {
            lse_out = m_val * 1.4426950408889634f + log2f(d_val);
        } else {
            lse_out = NEG_INF;
        }
        lse[t * 16 + h] = lse_out;
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
    // Ensures fully contiguous layout mapping
    q_nope = q_nope.contiguous();
    q_pe = q_pe.contiguous();
    ckv_cache = ckv_cache.contiguous();
    kpe_cache = kpe_cache.contiguous();
    sparse_indices = sparse_indices.contiguous();

    int num_tokens = q_nope.size(0);
    int topk = sparse_indices.size(1);

    auto output = torch::empty_like(q_nope);
    auto lse = torch::empty({num_tokens, 16}, q_nope.options().dtype(torch::kFloat32));

    if (num_tokens == 0) {
        return {output, lse};
    }

    dim3 grid(num_tokens);
    dim3 block(32, 16); // 16 heads assigned dynamically to 16 warps. 512 threads

    // 32 tiles * 512 * 2 (ckv) + 32 tiles * 64 * 2 (kpe) + 32 tiles * 4 (indices) = 36992 bytes
    int smem_size = 32 * 512 * 2 + 32 * 64 * 2 + 32 * 4;

    auto kernel = dsa_forward_kernel;
    // Strictly bounds dynamic shared memory context to prevent launch failures
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<grid, block, smem_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
        sparse_indices.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        sm_scale,
        num_tokens,
        topk
    );

    return {output, lse};
}

// Module export definitions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &dsa_forward, "DSA Forward Kernel");
}