#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>
#include <cmath>

namespace {

constexpr int HEADS = 16;
constexpr int D_NOPE = 512;
constexpr int D_PE = 64;
constexpr int TOPK = 2048;
constexpr int PAGE_SIZE = 64;
constexpr float LOG2E_F = 1.4426950408889634f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Devised from plan: remove split-K temporary HBM workspaces entirely.
// Use a persistent fused warp-specialized kernel for small workloads.
__global__ void dsa_persistent_fused_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    int num_tokens,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr) {

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int warps_per_block = blockDim.y;
    const int global_warp = blockIdx.x * warps_per_block + warp_id;
    const int total_warps = gridDim.x * warps_per_block;
    const int total_jobs = num_tokens * HEADS;

    for (int job = global_warp; job < total_jobs; job += total_warps) {
        const int t = job / HEADS;
        const int h = job - t * HEADS;

        float2 q_n[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
                q_nope_ptr + ((t * HEADS + h) * D_NOPE + j * 64 + lane * 2));
            q_n[j] = __bfloat1622float2(qv);
        }

        float2 q_p;
        {
            const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
                q_pe_ptr + ((t * HEADS + h) * D_PE + lane * 2));
            q_p = __bfloat1622float2(qv);
        }

        float2 acc[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j] = make_float2(0.0f, 0.0f);
        }

        float m = -CUDART_INF_F;
        float l = 0.0f;

        #pragma unroll 64
        for (int s = 0; s < TOPK; ++s) {
            const int idx = sparse_indices_ptr[t * TOPK + s];
            if (idx == -1) continue;

            float dot = 0.0f;
            float2 v_reg[8];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                    ckv_cache_ptr + idx * D_NOPE + j * 64 + lane * 2);
                const float2 kf = __bfloat1622float2(kv);
                v_reg[j] = kf;
                dot = fmaf(q_n[j].x, kf.x, dot);
                dot = fmaf(q_n[j].y, kf.y, dot);
            }

            {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                    kpe_cache_ptr + idx * D_PE + lane * 2);
                const float2 kf = __bfloat1622float2(kv);
                dot = fmaf(q_p.x, kf.x, dot);
                dot = fmaf(q_p.y, kf.y, dot);
            }

            float z = warp_reduce_sum(dot);
            z = __shfl_sync(0xffffffff, z, 0) * sm_scale;

            const float m_new = fmaxf(m, z);
            const float alpha = (m == -CUDART_INF_F) ? 0.0f : __expf(m - m_new);
            const float beta = __expf(z - m_new);
            l = l * alpha + beta;
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j].x = fmaf(beta, v_reg[j].x, acc[j].x * alpha);
                acc[j].y = fmaf(beta, v_reg[j].y, acc[j].y * alpha);
            }
        }

        if (l > 0.0f) {
            const float inv_l = __fdividef(1.0f, l);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j].x *= inv_l;
                acc[j].y *= inv_l;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j].x = 0.0f;
                acc[j].y = 0.0f;
            }
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 out = __floats2bfloat162_rn(acc[j].x, acc[j].y);
            *reinterpret_cast<__nv_bfloat162*>(output_ptr + ((t * HEADS + h) * D_NOPE + j * 64 + lane * 2)) = out;
        }

        if (lane == 0) {
            lse_ptr[t * HEADS + h] = (l > 0.0f) ? fmaf(m, LOG2E_F, __log2f(l)) : -CUDART_INF_F;
        }
    }
}

__global__ void dsa_tiled_fused_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr) {

    const int t = blockIdx.x;
    const int h = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = h * 32 + lane;

    __shared__ alignas(16) int idx_shared[32];
    __shared__ alignas(16) __nv_bfloat16 smem_kc[32 * D_NOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_kp[32 * D_PE];

    float2 q_n[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            q_nope_ptr + ((t * HEADS + h) * D_NOPE + j * 64 + lane * 2));
        q_n[j] = __bfloat1622float2(qv);
    }

    float2 q_p;
    {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            q_pe_ptr + ((t * HEADS + h) * D_PE + lane * 2));
        q_p = __bfloat1622float2(qv);
    }

    float2 acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.0f, 0.0f);

    float m = -CUDART_INF_F;
    float l = 0.0f;

    #pragma unroll 64
    for (int tile = 0; tile < TOPK / 32; ++tile) {
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * 32 + tid];
        }
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int row = step * 8 + (tid >> 6);
            const int col = tid & 63;
            const int idx = idx_shared[row];
            if (idx != -1) {
                const int4 v = *reinterpret_cast<const int4*>(ckv_cache_ptr + idx * D_NOPE + col * 8);
                *reinterpret_cast<int4*>(smem_kc + row * D_NOPE + col * 8) = v;
            }
        }

        if (tid < 256) {
            const int row = tid >> 3;
            const int col = tid & 7;
            const int idx = idx_shared[row];
            if (idx != -1) {
                const int4 v = *reinterpret_cast<const int4*>(kpe_cache_ptr + idx * D_PE + col * 8);
                *reinterpret_cast<int4*>(smem_kp + row * D_PE + col * 8) = v;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] == -1) continue;

            float dot = 0.0f;
            float2 v_reg[8];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                    smem_kc + i * D_NOPE + j * 64 + lane * 2);
                const float2 kf = __bfloat1622float2(kv);
                v_reg[j] = kf;
                dot = fmaf(q_n[j].x, kf.x, dot);
                dot = fmaf(q_n[j].y, kf.y, dot);
            }

            {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                    smem_kp + i * D_PE + lane * 2);
                const float2 kf = __bfloat1622float2(kv);
                dot = fmaf(q_p.x, kf.x, dot);
                dot = fmaf(q_p.y, kf.y, dot);
            }

            float z = warp_reduce_sum(dot);
            z = __shfl_sync(0xffffffff, z, 0) * sm_scale;

            const float m_new = fmaxf(m, z);
            const float alpha = (m == -CUDART_INF_F) ? 0.0f : __expf(m - m_new);
            const float beta = __expf(z - m_new);
            l = l * alpha + beta;
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j].x = fmaf(beta, v_reg[j].x, acc[j].x * alpha);
                acc[j].y = fmaf(beta, v_reg[j].y, acc[j].y * alpha);
            }
        }
        __syncthreads();
    }

    if (l > 0.0f) {
        const float inv_l = __fdividef(1.0f, l);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j].x *= inv_l;
            acc[j].y *= inv_l;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j] = make_float2(0.0f, 0.0f);
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 out = __floats2bfloat162_rn(acc[j].x, acc[j].y);
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + ((t * HEADS + h) * D_NOPE + j * 64 + lane * 2)) = out;
    }

    if (lane == 0) {
        lse_ptr[t * HEADS + h] = (l > 0.0f) ? fmaf(m, LOG2E_F, __log2f(l)) : -CUDART_INF_F;
    }
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    float sm_scale) {

    TORCH_CHECK(q_nope.is_cuda(), "q_nope must be CUDA tensor");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be CUDA tensor");
    TORCH_CHECK(ckv_cache.is_cuda(), "ckv_cache must be CUDA tensor");
    TORCH_CHECK(kpe_cache.is_cuda(), "kpe_cache must be CUDA tensor");
    TORCH_CHECK(sparse_indices.is_cuda(), "sparse_indices must be CUDA tensor");

    c10::cuda::CUDAGuard device_guard(q_nope.device());

    TORCH_CHECK(q_nope.scalar_type() == torch::kBFloat16, "q_nope must be bfloat16");
    TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16, "q_pe must be bfloat16");
    TORCH_CHECK(ckv_cache.scalar_type() == torch::kBFloat16, "ckv_cache must be bfloat16");
    TORCH_CHECK(kpe_cache.scalar_type() == torch::kBFloat16, "kpe_cache must be bfloat16");
    TORCH_CHECK(sparse_indices.scalar_type() == torch::kInt32, "sparse_indices must be int32");

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == HEADS && q_nope.size(2) == D_NOPE,
                "q_nope must have shape [T,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == HEADS && q_pe.size(2) == D_PE,
                "q_pe must have shape [T,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == PAGE_SIZE && ckv_cache.size(2) == D_NOPE,
                "ckv_cache must have shape [P,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == PAGE_SIZE && kpe_cache.size(2) == D_PE,
                "kpe_cache must have shape [P,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == TOPK,
                "sparse_indices must have shape [T,2048]");
    TORCH_CHECK(q_pe.size(0) == q_nope.size(0), "q_pe batch mismatch");
    TORCH_CHECK(sparse_indices.size(0) == q_nope.size(0), "sparse_indices batch mismatch");
    TORCH_CHECK(kpe_cache.size(0) == ckv_cache.size(0), "cache page mismatch");

    TORCH_CHECK(q_nope.is_contiguous(), "q_nope must be contiguous");
    TORCH_CHECK(q_pe.is_contiguous(), "q_pe must be contiguous");
    TORCH_CHECK(ckv_cache.is_contiguous(), "ckv_cache must be contiguous");
    TORCH_CHECK(kpe_cache.is_contiguous(), "kpe_cache must be contiguous");
    TORCH_CHECK(sparse_indices.is_contiguous(), "sparse_indices must be contiguous");

    const int num_tokens = static_cast<int>(q_nope.size(0));
    auto output = torch::empty({num_tokens, HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, HEADS}, torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));

    if (num_tokens == 0) {
        return {output, lse};
    }

    auto stream = at::cuda::getDefaultCUDAStream();

    if (num_tokens <= 192) {
        const int warps_per_block = 8;
        int grid = (num_tokens * HEADS + warps_per_block - 1) / warps_per_block;
        int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        if (grid < sm_count * 2) grid = sm_count * 2;
        dsa_persistent_fused_kernel<<<grid, dim3(32, warps_per_block, 1), 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            num_tokens,
            sm_scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>());
    } else {
        dsa_tiled_fused_kernel<<<num_tokens, dim3(32, HEADS, 1), 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>());
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}
