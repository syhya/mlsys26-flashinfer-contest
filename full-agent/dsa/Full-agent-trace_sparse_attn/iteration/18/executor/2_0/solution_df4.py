#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>
#include <cmath>

namespace {

constexpr int NUM_HEADS = 16;
constexpr int D_NOPE = 512;
constexpr int D_PE = 64;
constexpr int TOPK = 2048;
constexpr int PAGE_SIZE = 64;
constexpr int TILE = 32;
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ int warp_reduce_max_int(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(0xffffffff, val, offset);
        val = max(val, other);
    }
    return val;
}

// Small-workload path: persistent warp scheduler over (token, head) pairs.
// Devised from Plan: we avoid the parent's split-K temporary HBM workspaces entirely.
__global__ void dsa_persistent_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    int num_tokens,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int warp_stride = gridDim.x * WARPS_PER_BLOCK;
    int jobs = num_tokens * NUM_HEADS;

    __shared__ int32_t idx_smem[WARPS_PER_BLOCK][TILE];

    float qn[16];
    float qp[2];
    float acc[16];

    for (int job = global_warp; job < jobs; job += warp_stride) {
        int t = job / NUM_HEADS;
        int h = job - t * NUM_HEADS;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int base = t * NUM_HEADS * D_NOPE + h * D_NOPE + lane * 16 + i;
            qn[i] = __bfloat162float(q_nope_ptr[base]);
            acc[i] = 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int base = t * NUM_HEADS * D_PE + h * D_PE + lane * 2 + i;
            qp[i] = __bfloat162float(q_pe_ptr[base]);
        }

        float m = -INFINITY;
        float l = 0.0f;

        #pragma unroll 1
        for (int tile = 0; tile < TOPK / TILE; ++tile) {
            if (lane < TILE) {
                idx_smem[warp_id][lane] = sparse_indices_ptr[t * TOPK + tile * TILE + lane];
            }
            __syncwarp();

            #pragma unroll
            for (int i = 0; i < TILE; ++i) {
                int idx = idx_smem[warp_id][i];
                if (idx == -1) continue;

                float local_dot = 0.0f;
                int kc_base = idx * D_NOPE + lane * 16;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    float kv = __bfloat162float(ckv_cache_ptr[kc_base + j]);
                    local_dot = fmaf(qn[j], kv, local_dot);
                }

                int kp_base = idx * D_PE + lane * 2;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    float kv = __bfloat162float(kpe_cache_ptr[kp_base + j]);
                    local_dot = fmaf(qp[j], kv, local_dot);
                }

                float logit = warp_reduce_sum(local_dot);
                logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

                float m_new = fmaxf(m, logit);
                float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
                float beta = __expf(logit - m_new);
                l = fmaf(l, alpha, beta);
                m = m_new;

                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    float v = __bfloat162float(ckv_cache_ptr[kc_base + j]);
                    acc[j] = fmaf(acc[j], alpha, beta * v);
                }
            }
        }

        float inv_l = (l > 0.0f) ? __fdividef(1.0f, l) : 0.0f;
        int out_base = t * NUM_HEADS * D_NOPE + h * D_NOPE + lane * 16;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            float out = (l > 0.0f) ? acc[j] * inv_l : 0.0f;
            output_ptr[out_base + j] = __float2bfloat16(out);
        }

        if (lane == 0) {
            lse_ptr[t * NUM_HEADS + h] = (l > 0.0f) ? fmaf(m, LOG2E, __log2f(l)) : -INFINITY;
        }
    }
}

// Large-workload path: CTA per token, one warp per head, shared-memory staging by 32 sparse entries.
__global__ void dsa_tiled_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    int num_tokens,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    int lane = threadIdx.x;
    int h = threadIdx.y;
    int tid = h * 32 + lane;

    __shared__ alignas(16) int32_t idx_shared[TILE];
    __shared__ alignas(16) __nv_bfloat16 smem_kc[TILE * D_NOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_kp[TILE * D_PE];

    float qn[16];
    float qp[2];
    float acc[16];

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int base = t * NUM_HEADS * D_NOPE + h * D_NOPE + lane * 16 + i;
        qn[i] = __bfloat162float(q_nope_ptr[base]);
        acc[i] = 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int base = t * NUM_HEADS * D_PE + h * D_PE + lane * 2 + i;
        qp[i] = __bfloat162float(q_pe_ptr[base]);
    }

    float m = -INFINITY;
    float l = 0.0f;

    #pragma unroll 1
    for (int tile = 0; tile < TOPK / TILE; ++tile) {
        if (tid < TILE) {
            idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * TILE + tid];
        }
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int row = step * 8 + (tid >> 6);
            int col = tid & 63;
            int idx = idx_shared[row];
            if (idx != -1) {
                const int4* src = reinterpret_cast<const int4*>(&ckv_cache_ptr[idx * D_NOPE]);
                int4* dst = reinterpret_cast<int4*>(&smem_kc[row * D_NOPE]);
                dst[col] = src[col];
            }
        }
        if (tid < 256) {
            int row = tid >> 3;
            int col = tid & 7;
            int idx = idx_shared[row];
            if (idx != -1) {
                const int4* src = reinterpret_cast<const int4*>(&kpe_cache_ptr[idx * D_PE]);
                int4* dst = reinterpret_cast<int4*>(&smem_kp[row * D_PE]);
                dst[col] = src[col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            if (idx_shared[i] == -1) continue;

            float local_dot = 0.0f;
            int base_kc = i * D_NOPE + lane * 16;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                float kv = __bfloat162float(smem_kc[base_kc + j]);
                local_dot = fmaf(qn[j], kv, local_dot);
            }
            int base_kp = i * D_PE + lane * 2;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                float kv = __bfloat162float(smem_kp[base_kp + j]);
                local_dot = fmaf(qp[j], kv, local_dot);
            }

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            float m_new = fmaxf(m, logit);
            float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
            float beta = __expf(logit - m_new);
            l = fmaf(l, alpha, beta);
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                float v = __bfloat162float(smem_kc[base_kc + j]);
                acc[j] = fmaf(acc[j], alpha, beta * v);
            }
        }
        __syncthreads();
    }

    float inv_l = (l > 0.0f) ? __fdividef(1.0f, l) : 0.0f;
    int out_base = t * NUM_HEADS * D_NOPE + h * D_NOPE + lane * 16;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        float out = (l > 0.0f) ? acc[j] * inv_l : 0.0f;
        output_ptr[out_base + j] = __float2bfloat16(out);
    }
    if (lane == 0) {
        lse_ptr[t * NUM_HEADS + h] = (l > 0.0f) ? fmaf(m, LOG2E, __log2f(l)) : -INFINITY;
    }
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    float sm_scale
) {
    TORCH_CHECK(q_nope.is_cuda(), "q_nope must be CUDA");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be CUDA");
    TORCH_CHECK(ckv_cache.is_cuda(), "ckv_cache must be CUDA");
    TORCH_CHECK(kpe_cache.is_cuda(), "kpe_cache must be CUDA");
    TORCH_CHECK(sparse_indices.is_cuda(), "sparse_indices must be CUDA");

    TORCH_CHECK(q_nope.scalar_type() == torch::kBFloat16, "q_nope must be bfloat16");
    TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16, "q_pe must be bfloat16");
    TORCH_CHECK(ckv_cache.scalar_type() == torch::kBFloat16, "ckv_cache must be bfloat16");
    TORCH_CHECK(kpe_cache.scalar_type() == torch::kBFloat16, "kpe_cache must be bfloat16");
    TORCH_CHECK(sparse_indices.scalar_type() == torch::kInt32, "sparse_indices must be int32");

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == NUM_HEADS && q_nope.size(2) == D_NOPE, "q_nope shape must be [T,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == NUM_HEADS && q_pe.size(2) == D_PE, "q_pe shape must be [T,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == PAGE_SIZE && ckv_cache.size(2) == D_NOPE, "ckv_cache shape must be [P,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == PAGE_SIZE && kpe_cache.size(2) == D_PE, "kpe_cache shape must be [P,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == TOPK, "sparse_indices shape must be [T,2048]");
    TORCH_CHECK(q_pe.size(0) == q_nope.size(0), "q_pe token dim mismatch");
    TORCH_CHECK(sparse_indices.size(0) == q_nope.size(0), "sparse_indices token dim mismatch");
    TORCH_CHECK(kpe_cache.size(0) == ckv_cache.size(0), "cache page dim mismatch");

    TORCH_CHECK(q_nope.is_contiguous(), "q_nope must be contiguous");
    TORCH_CHECK(q_pe.is_contiguous(), "q_pe must be contiguous");
    TORCH_CHECK(ckv_cache.is_contiguous(), "ckv_cache must be contiguous");
    TORCH_CHECK(kpe_cache.is_contiguous(), "kpe_cache must be contiguous");
    TORCH_CHECK(sparse_indices.is_contiguous(), "sparse_indices must be contiguous");

    const at::cuda::CUDAGuard device_guard(q_nope.device());
    int num_tokens = static_cast<int>(q_nope.size(0));

    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) {
        return {output, lse};
    }

    auto q_nope_c = q_nope.view({num_tokens, NUM_HEADS, D_NOPE});
    auto q_pe_c = q_pe.view({num_tokens, NUM_HEADS, D_PE});
    auto ckv_flat = ckv_cache.view({ckv_cache.size(0) * PAGE_SIZE, D_NOPE});
    auto kpe_flat = kpe_cache.view({kpe_cache.size(0) * PAGE_SIZE, D_PE});

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (num_tokens <= 96) {
        int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        int blocks = sm_count * 2;
        dsa_persistent_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope_c.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe_c.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_flat.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_flat.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            num_tokens,
            sm_scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>()
        );
    } else {
        dim3 grid(num_tokens);
        dim3 block(32, NUM_HEADS);
        dsa_tiled_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope_c.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe_c.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_flat.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_flat.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            num_tokens,
            sm_scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>()
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}
