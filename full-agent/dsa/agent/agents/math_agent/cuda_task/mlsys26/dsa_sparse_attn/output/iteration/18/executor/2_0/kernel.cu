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
    int tid = h * 32 + lane;

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

    int num_tiles = TOPK / TILE;
    for (int tile = 0; tile < num_tiles; ++tile) {
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * TILE + tid];
        }
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid / 64);
            int load_col = tid % 64;
            int idx = idx_shared[load_row];
            if (idx != -1) {
                const int4* src = reinterpret_cast<const int4*>(&ckv_cache_ptr[idx * 512]);
                int4* dst = reinterpret_cast<int4*>(&smem_Kc[load_row * 512]);
                dst[load_col] = src[load_col];
            }
        }

        if (tid < 256) {
            int load_row = tid / 8;
            int load_col = tid % 8;
            int idx = idx_shared[load_row];
            if (idx != -1) {
                const int4* src = reinterpret_cast<const int4*>(&kpe_cache_ptr[idx * 64]);
                int4* dst = reinterpret_cast<int4*>(&smem_Kp[load_row * 64]);
                dst[load_col] = src[load_col];
            }
        }
        __syncthreads();

        #pragma unroll
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

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 out_bf16 = __floats2bfloat162_rn(O_reg[j].x, O_reg[j].y);
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]) = out_bf16;
    }

    if (lane == 0) {
        lse_ptr[t * 16 + h] = (l > 0.0f) ? fmaf(m, LOG2E, __log2f(l)) : -INFINITY;
    }
}

// Improved low-workload path: keep split-K idea for occupancy, but reduce HBM transactions
// by storing O_tmp as float4 to match reduction access granularity.
__global__ void split_k_compute_kernel_vec4(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float4* __restrict__ O_tmp4,
    float* __restrict__ m_tmp,
    float* __restrict__ l_tmp
) {
    int s = blockIdx.x;
    int t = blockIdx.y;
    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = h * 32 + lane;

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

    float4 O4_reg[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        O4_reg[j] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    float m = -INFINITY;
    float l = 0.0f;

    if (tid < 32) {
        idx_shared[tid] = sparse_indices_ptr[t * TOPK + s * TILE + tid];
    }
    __syncthreads();

    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        int load_row = step * 8 + (tid / 64);
        int load_col = tid % 64;
        int idx = idx_shared[load_row];
        if (idx != -1) {
            const int4* src = reinterpret_cast<const int4*>(&ckv_cache_ptr[idx * 512]);
            int4* dst = reinterpret_cast<int4*>(&smem_Kc[load_row * 512]);
            dst[load_col] = src[load_col];
        }
    }

    if (tid < 256) {
        int load_row = tid / 8;
        int load_col = tid % 8;
        int idx = idx_shared[load_row];
        if (idx != -1) {
            const int4* src = reinterpret_cast<const int4*>(&kpe_cache_ptr[idx * 64]);
            int4* dst = reinterpret_cast<int4*>(&smem_Kp[load_row * 64]);
            dst[load_col] = src[col];
        }
    }
    __syncthreads();

    #pragma unroll
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
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float m_new = fmaxf(m, logit);
        float exp_diff = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        float exp_logit = __expf(logit - m_new);
        l = fmaf(l, exp_diff, exp_logit);
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 a0 = k_f_reg[2 * j + 0];
            float2 a1 = k_f_reg[2 * j + 1];
            O4_reg[j].x = fmaf(exp_logit, a0.x, O4_reg[j].x * exp_diff);
            O4_reg[j].y = fmaf(exp_logit, a0.y, O4_reg[j].y * exp_diff);
            O4_reg[j].z = fmaf(exp_logit, a1.x, O4_reg[j].z * exp_diff);
            O4_reg[j].w = fmaf(exp_logit, a1.y, O4_reg[j].w * exp_diff);
        }
    }

    int vec_base = t * 64 * 16 * 128 + s * 16 * 128 + h * 128 + lane * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        O_tmp4[vec_base + j] = O4_reg[j];
    }

    if (lane == 0) {
        m_tmp[t * 16 * 64 + h * 64 + s] = m;
        l_tmp[t * 16 * 64 + h * 64 + s] = l;
    }
}

__global__ void split_k_reduce_kernel(
    const float4* __restrict__ O_tmp4,
    const float* __restrict__ m_tmp,
    const float* __restrict__ l_tmp,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int lane = threadIdx.x;

    __shared__ float m_global;
    __shared__ float l_global;
    __shared__ float m_s[64];
    __shared__ float l_s[64];
    __shared__ float scale_s[64];

    if (lane < 64) {
        m_s[lane] = m_tmp[t * 16 * 64 + h * 64 + lane];
        l_s[lane] = l_tmp[t * 16 * 64 + h * 64 + lane];
    }
    __syncthreads();

    if (lane == 0) {
        float m_max = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 64; ++i) m_max = fmaxf(m_max, m_s[i]);
        float l_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            if (m_s[i] != -INFINITY) {
                l_sum = fmaf(l_s[i], __expf(m_s[i] - m_max), l_sum);
            }
        }
        m_global = m_max;
        l_global = l_sum;
        lse_ptr[t * 16 + h] = (m_max == -INFINITY) ? -INFINITY : fmaf(m_max, LOG2E, __log2f(l_sum));
    }
    __syncthreads();

    if (lane < 64) {
        scale_s[lane] = (m_global != -INFINITY && l_global > 0.0f && m_s[lane] != -INFINITY)
                        ? __fdividef(__expf(m_s[lane] - m_global), l_global)
                        : 0.0f;
    }
    __syncthreads();

    int d_idx = lane * 4;
    if (d_idx < 512) {
        float4 out = make_float4(0.f, 0.f, 0.f, 0.f);
        if (m_global != -INFINITY && l_global > 0.0f) {
            #pragma unroll 16
            for (int s = 0; s < 64; ++s) {
                float scale = scale_s[s];
                if (scale > 0.0f) {
                    int idx = t * 64 * 16 * 128 + s * 16 * 128 + h * 128 + lane;
                    float4 val = O_tmp4[idx];
                    out.x = fmaf(val.x, scale, out.x);
                    out.y = fmaf(val.y, scale, out.y);
                    out.z = fmaf(val.z, scale, out.z);
                    out.w = fmaf(val.w, scale, out.w);
                }
            }
        }

        __nv_bfloat162 out01 = __floats2bfloat162_rn(out.x, out.y);
        __nv_bfloat162 out23 = __floats2bfloat162_rn(out.z, out.w);
        uint32_t v0 = *reinterpret_cast<uint32_t*>(&out01);
        uint32_t v1 = *reinterpret_cast<uint32_t*>(&out23);
        uint2 packed = make_uint2(v0, v1);
        *reinterpret_cast<uint2*>(&output_ptr[t * 16 * 512 + h * 512 + d_idx]) = packed;
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

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == 16 && q_nope.size(2) == 512, "q_nope shape must be [T,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == 16 && q_pe.size(2) == 64, "q_pe shape must be [T,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == 64 && ckv_cache.size(2) == 512, "ckv_cache shape must be [P,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == 64 && kpe_cache.size(2) == 64, "kpe_cache shape must be [P,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == 2048, "sparse_indices shape must be [T,2048]");
    TORCH_CHECK(q_pe.size(0) == q_nope.size(0), "token mismatch");
    TORCH_CHECK(sparse_indices.size(0) == q_nope.size(0), "token mismatch");
    TORCH_CHECK(kpe_cache.size(0) == ckv_cache.size(0), "page mismatch");

    TORCH_CHECK(q_nope.is_contiguous(), "q_nope must be contiguous");
    TORCH_CHECK(q_pe.is_contiguous(), "q_pe must be contiguous");
    TORCH_CHECK(ckv_cache.is_contiguous(), "ckv_cache must be contiguous");
    TORCH_CHECK(kpe_cache.is_contiguous(), "kpe_cache must be contiguous");
    TORCH_CHECK(sparse_indices.is_contiguous(), "sparse_indices must be contiguous");

    const c10::cuda::CUDAGuard device_guard(q_nope.device());
    int num_tokens = q_nope.size(0);
    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    auto ckv_flat = ckv_cache.view({ckv_cache.size(0) * 64, 512});
    auto kpe_flat = kpe_cache.view({kpe_cache.size(0) * 64, 64});
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (num_tokens < 128) {
        int S = 64;
        auto O_tmp = torch::empty({num_tokens, S, 16, 128, 4}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto m_tmp = torch::empty({num_tokens, 16, S}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto l_tmp = torch::empty({num_tokens, 16, S}, torch::dtype(torch::kFloat32).device(q_nope.device()));

        dim3 compute_grid(S, num_tokens, 1);
        dim3 compute_block(32, 16, 1);
        split_k_compute_kernel_vec4<<<compute_grid, compute_block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_flat.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_flat.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            reinterpret_cast<float4*>(O_tmp.data_ptr<float>()),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>()
        );

        dim3 reduce_grid(num_tokens, 16, 1);
        dim3 reduce_block(128, 1, 1);
        split_k_reduce_kernel<<<reduce_grid, reduce_block, 0, stream>>>(
            reinterpret_cast<const float4*>(O_tmp.data_ptr<float>()),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>()
        );
    } else {
        dim3 grid(num_tokens);
        dim3 block(32, 16);
        dsa_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_flat.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_flat.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
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
