#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == torch::kBFloat16, #x " must be bfloat16")
#define CHECK_I32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

static constexpr int NUM_HEADS = 16;
static constexpr int D_NOPE = 512;
static constexpr int D_PE = 64;
static constexpr int TOPK = 2048;
static constexpr int TILE = 32;
static constexpr float LOG2E_F = 1.4426950408889634f;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// Large-throughput path: one CTA per token, 16 warps per CTA, one warp per head.
__global__ void dsa_large_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int32_t* __restrict__ sparse_indices,
    float sm_scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    int num_tokens
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    int lane = threadIdx.x;
    int h = threadIdx.y;
    int tid = h * 32 + lane;

    __shared__ alignas(16) int idx_shared[TILE];
    __shared__ alignas(16) __nv_bfloat16 smem_kc[TILE * D_NOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_kp[TILE * D_PE];

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        auto qv = *reinterpret_cast<const __nv_bfloat162*>(q_nope + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2);
        qn[j] = __bfloat1622float2(qv);
    }
    float2 qp;
    {
        auto qv = *reinterpret_cast<const __nv_bfloat162*>(q_pe + t * NUM_HEADS * D_PE + h * D_PE + lane * 2);
        qp = __bfloat1622float2(qv);
    }

    float2 acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.f, 0.f);
    float m = -INFINITY;
    float l = 0.f;

    #pragma unroll
    for (int tile = 0; tile < TOPK / TILE; ++tile) {
        if (tid < TILE) idx_shared[tid] = sparse_indices[t * TOPK + tile * TILE + tid];
        __syncthreads();

        // 128-bit cooperative staging of Kc
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int row = step * 8 + (tid >> 6);
            int col = tid & 63;
            int idx = idx_shared[row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(ckv_cache + idx * D_NOPE);
                float4* dst = reinterpret_cast<float4*>(smem_kc + row * D_NOPE);
                dst[col] = src[col];
            }
        }
        if (tid < 256) {
            int row = tid >> 3;
            int col = tid & 7;
            int idx = idx_shared[row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(kpe_cache + idx * D_PE);
                float4* dst = reinterpret_cast<float4*>(smem_kp + row * D_PE);
                dst[col] = src[col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            if (idx_shared[i] == -1) continue;

            float local_dot = 0.f;
            float2 kvals[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                auto kv = *reinterpret_cast<const __nv_bfloat162*>(smem_kc + i * D_NOPE + j * 64 + lane * 2);
                float2 kf = __bfloat1622float2(kv);
                kvals[j] = kf;
                local_dot = fmaf(qn[j].x, kf.x, local_dot);
                local_dot = fmaf(qn[j].y, kf.y, local_dot);
            }
            {
                auto kv = *reinterpret_cast<const __nv_bfloat162*>(smem_kp + i * D_PE + lane * 2);
                float2 kf = __bfloat1622float2(kv);
                local_dot = fmaf(qp.x, kf.x, local_dot);
                local_dot = fmaf(qp.y, kf.y, local_dot);
            }

            float z = warp_reduce_sum(local_dot);
            z = __shfl_sync(0xffffffff, z, 0) * sm_scale;

            float m_new = fmaxf(m, z);
            float alpha = (m == -INFINITY) ? 0.f : __expf(m - m_new);
            float beta = __expf(z - m_new);
            l = l * alpha + beta;
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j].x = fmaf(beta, kvals[j].x, acc[j].x * alpha);
                acc[j].y = fmaf(beta, kvals[j].y, acc[j].y * alpha);
            }
        }
        __syncthreads();
    }

    float inv_l = (l > 0.f) ? __fdividef(1.f, l) : 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float ox = acc[j].x * inv_l;
        float oy = acc[j].y * inv_l;
        auto outv = __floats2bfloat162_rn(ox, oy);
        *reinterpret_cast<__nv_bfloat162*>(output + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2) = outv;
    }
    if (lane == 0) {
        lse[t * NUM_HEADS + h] = (l > 0.f) ? fmaf(m, LOG2E_F, __log2f(l)) : -INFINITY;
    }
}

// Low-latency persistent path. Devised from plan: remove split-K FP32 temporaries and second reduction kernel.
// Each warp owns one (token, head) job and processes the full online softmax recurrence in one pass.
__global__ void dsa_small_persistent_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int32_t* __restrict__ sparse_indices,
    float sm_scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    int num_tokens
) {
    int lane = threadIdx.x;
    int warp_id = threadIdx.y;
    int warps_per_block = blockDim.y;
    int global_warp = blockIdx.x * warps_per_block + warp_id;
    int total_warps = gridDim.x * warps_per_block;
    int jobs = num_tokens * NUM_HEADS;

    extern __shared__ unsigned char smem_raw[];
    int* idx_smem = reinterpret_cast<int*>(smem_raw);
    __nv_bfloat16* kc_smem = reinterpret_cast<__nv_bfloat16*>(idx_smem + warps_per_block * TILE);
    __nv_bfloat16* kp_smem = kc_smem + warps_per_block * TILE * D_NOPE;

    for (int job = global_warp; job < jobs; job += total_warps) {
        int t = job / NUM_HEADS;
        int h = job % NUM_HEADS;

        int* my_idx = idx_smem + warp_id * TILE;
        __nv_bfloat16* my_kc = kc_smem + warp_id * TILE * D_NOPE;
        __nv_bfloat16* my_kp = kp_smem + warp_id * TILE * D_PE;

        float2 qn[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            auto qv = *reinterpret_cast<const __nv_bfloat162*>(q_nope + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2);
            qn[j] = __bfloat1622float2(qv);
        }
        float2 qp;
        {
            auto qv = *reinterpret_cast<const __nv_bfloat162*>(q_pe + t * NUM_HEADS * D_PE + h * D_PE + lane * 2);
            qp = __bfloat1622float2(qv);
        }

        float2 acc[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.f, 0.f);
        float m = -INFINITY;
        float l = 0.f;

        #pragma unroll
        for (int tile = 0; tile < TOPK / TILE; ++tile) {
            my_idx[lane] = sparse_indices[t * TOPK + tile * TILE + lane];

            // Stage current tile into warp-private shared memory.
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                int col = lane * 16 + j;
                int idx = my_idx[j];
                if (idx != -1) {
                    my_kc[j * D_NOPE + col] = ckv_cache[idx * D_NOPE + col];
                }
            }
            if (lane < 2) {
                #pragma unroll
                for (int row = 0; row < TILE; ++row) {
                    int idx = my_idx[row];
                    if (idx != -1) {
                        #pragma unroll
                        for (int k = 0; k < 32; ++k) {
                            my_kp[row * D_PE + lane * 32 + k] = kpe_cache[idx * D_PE + lane * 32 + k];
                        }
                    }
                }
            }
            __syncwarp();

            #pragma unroll
            for (int i = 0; i < TILE; ++i) {
                int idx = my_idx[i];
                if (idx == -1) continue;

                float local_dot = 0.f;
                float2 kvals[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    auto kv = *reinterpret_cast<const __nv_bfloat162*>(my_kc + i * D_NOPE + j * 64 + lane * 2);
                    float2 kf = __bfloat1622float2(kv);
                    kvals[j] = kf;
                    local_dot = fmaf(qn[j].x, kf.x, local_dot);
                    local_dot = fmaf(qn[j].y, kf.y, local_dot);
                }
                {
                    auto kv = *reinterpret_cast<const __nv_bfloat162*>(my_kp + i * D_PE + lane * 2);
                    float2 kf = __bfloat1622float2(kv);
                    local_dot = fmaf(qp.x, kf.x, local_dot);
                    local_dot = fmaf(qp.y, kf.y, local_dot);
                }

                float z = warp_reduce_sum(local_dot);
                z = __shfl_sync(0xffffffff, z, 0) * sm_scale;
                float m_new = fmaxf(m, z);
                float alpha = (m == -INFINITY) ? 0.f : __expf(m - m_new);
                float beta = __expf(z - m_new);
                l = l * alpha + beta;
                m = m_new;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[j].x = fmaf(beta, kvals[j].x, acc[j].x * alpha);
                    acc[j].y = fmaf(beta, kvals[j].y, acc[j].y * alpha);
                }
            }
            __syncwarp();
        }

        float inv_l = (l > 0.f) ? __fdividef(1.f, l) : 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float ox = acc[j].x * inv_l;
            float oy = acc[j].y * inv_l;
            auto outv = __floats2bfloat162_rn(ox, oy);
            *reinterpret_cast<__nv_bfloat162*>(output + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2) = outv;
        }
        if (lane == 0) {
            lse[t * NUM_HEADS + h] = (l > 0.f) ? fmaf(m, LOG2E_F, __log2f(l)) : -INFINITY;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    float sm_scale
) {
    CHECK_CUDA(q_nope);
    CHECK_CUDA(q_pe);
    CHECK_CUDA(ckv_cache);
    CHECK_CUDA(kpe_cache);
    CHECK_CUDA(sparse_indices);
    CHECK_CONTIGUOUS(q_nope);
    CHECK_CONTIGUOUS(q_pe);
    CHECK_CONTIGUOUS(ckv_cache);
    CHECK_CONTIGUOUS(kpe_cache);
    CHECK_CONTIGUOUS(sparse_indices);
    CHECK_BF16(q_nope);
    CHECK_BF16(q_pe);
    CHECK_BF16(ckv_cache);
    CHECK_BF16(kpe_cache);
    CHECK_I32(sparse_indices);

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == NUM_HEADS && q_nope.size(2) == D_NOPE, "q_nope shape must be [T,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == NUM_HEADS && q_pe.size(2) == D_PE, "q_pe shape must be [T,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == 64 && ckv_cache.size(2) == D_NOPE, "ckv_cache shape must be [P,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == 64 && kpe_cache.size(2) == D_PE, "kpe_cache shape must be [P,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == TOPK, "sparse_indices shape must be [T,2048]");
    TORCH_CHECK(q_nope.size(0) == q_pe.size(0) && q_nope.size(0) == sparse_indices.size(0), "token dimension mismatch");
    TORCH_CHECK(ckv_cache.size(0) == kpe_cache.size(0), "cache page dimension mismatch");

    c10::cuda::CUDAGuard device_guard(q_nope.device());

    int num_tokens = static_cast<int>(q_nope.size(0));
    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    auto stream = at::cuda::getDefaultCUDAStream();

    const __nv_bfloat16* qn_ptr = reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>());
    const __nv_bfloat16* qp_ptr = reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>());
    const __nv_bfloat16* kc_ptr = reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>());
    const __nv_bfloat16* kp_ptr = reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>());
    const int32_t* si_ptr = sparse_indices.data_ptr<int32_t>();
    __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    float* lse_ptr = lse.data_ptr<float>();

    // Dynamic multi-path dispatch.
    if (num_tokens <= 128) {
        constexpr int WARPS_PER_BLOCK = 4;
        dim3 block(32, WARPS_PER_BLOCK, 1);
        int jobs = num_tokens * NUM_HEADS;
        int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        int grid_x = sm_count > 0 ? sm_count : 1;
        size_t smem = WARPS_PER_BLOCK * (TILE * sizeof(int) + TILE * D_NOPE * sizeof(__nv_bfloat16) + TILE * D_PE * sizeof(__nv_bfloat16));
        dsa_small_persistent_kernel<<<grid_x, block, smem, stream>>>(qn_ptr, qp_ptr, kc_ptr, kp_ptr, si_ptr, sm_scale, out_ptr, lse_ptr, num_tokens);
    } else {
        dim3 grid(num_tokens, 1, 1);
        dim3 block(32, NUM_HEADS, 1);
        dsa_large_kernel<<<grid, block, 0, stream>>>(qn_ptr, qp_ptr, kc_ptr, kp_ptr, si_ptr, sm_scale, out_ptr, lse_ptr, num_tokens);
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}
