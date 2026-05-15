cpp
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).scalar_type() == (dt), #x " has wrong dtype")

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
    int t = blockIdx.x;
    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = threadIdx.y * 32 + threadIdx.x;

    __shared__ alignas(16) int idx_shared[32];
    __shared__ int tile_has_valid;
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
    #pragma unroll 1
    for (int tile = 0; tile < num_tiles; ++tile) {
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * 2048 + tile * 32 + tid];
        }
        if (tid == 0) tile_has_valid = 0;
        __syncthreads();

        if (tid < 32 && idx_shared[tid] != -1) {
            atomicOr(&tile_has_valid, 1);
        }
        __syncthreads();

        if (!tile_has_valid) continue;

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

        #pragma unroll 1
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
                O_reg[j].x = fmaf(exp_logit, k_f_reg[j].x, O_reg[j].x * exp_diff);
                O_reg[j].y = fmaf(exp_logit, k_f_reg[j].y, O_reg[j].y * exp_diff);
            }
        }
        __syncthreads();
    }

    if (l <= 0.0f || m == -INFINITY) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat162 out_bf16 = __floats2bfloat162_rn(0.0f, 0.0f);
            *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]) = out_bf16;
        }
        if (lane == 0) lse_ptr[t * 16 + h] = -INFINITY;
        return;
    }

    float inv_l = __fdividef(1.0f, l);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        O_reg[j].x *= inv_l;
        O_reg[j].y *= inv_l;
        __nv_bfloat162 out_bf16 = __floats2bfloat162_rn(O_reg[j].x, O_reg[j].y);
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]) = out_bf16;
    }

    if (lane == 0) {
        lse_ptr[t * 16 + h] = fmaf(m, 1.4426950408889634f, __log2f(l));
    }
}

// ============== SPLIT-K PATH for small num_tokens ==============
// S=64 partitions, each handles 32 sparse slots.

constexpr int SPLIT_S = 64;
constexpr int SPLIT_TILE = 32;  // 64*32 = 2048

__global__ void split_k_compute_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float* __restrict__ O_tmp,    // [T, S, 16, 512]
    float* __restrict__ m_tmp,    // [T, 16, S]
    float* __restrict__ l_tmp     // [T, 16, S]
) {
    int s = blockIdx.x;
    int t = blockIdx.y;
    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = threadIdx.y * 32 + threadIdx.x;

    __shared__ alignas(16) int idx_shared[32];
    __shared__ int tile_has_valid;
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
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;

    // Load tile of 32 indices for this partition
    if (tid < 32) {
        idx_shared[tid] = sparse_indices_ptr[t * 2048 + s * SPLIT_TILE + tid];
    }
    if (tid == 0) tile_has_valid = 0;
    __syncthreads();

    if (tid < 32 && idx_shared[tid] != -1) atomicOr(&tile_has_valid, 1);
    __syncthreads();

    if (tile_has_valid) {
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

        #pragma unroll 1
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
                O_reg[j].x = fmaf(exp_logit, k_f_reg[j].x, O_reg[j].x * exp_diff);
                O_reg[j].y = fmaf(exp_logit, k_f_reg[j].y, O_reg[j].y * exp_diff);
            }
        }
    }

    // Store partials (UNNORMALIZED)
    // O_tmp layout: [T, S, 16, 512]
    long base_O = ((long)t * SPLIT_S + s) * 16 * 512 + h * 512;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        O_tmp[base_O + j * 64 + lane * 2 + 0] = O_reg[j].x;
        O_tmp[base_O + j * 64 + lane * 2 + 1] = O_reg[j].y;
    }
    if (lane == 0) {
        m_tmp[(t * 16 + h) * SPLIT_S + s] = m;
        l_tmp[(t * 16 + h) * SPLIT_S + s] = l;
    }
}

__global__ void split_k_reduce_kernel(
    const float* __restrict__ O_tmp,
    const float* __restrict__ m_tmp,
    const float* __restrict__ l_tmp,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;  // 128 threads

    __shared__ float m_s_smem[SPLIT_S];
    __shared__ float l_s_smem[SPLIT_S];
    __shared__ float m_global;
    __shared__ float inv_l;
    __shared__ int valid_flag;

    if (tid < SPLIT_S) {
        m_s_smem[tid] = m_tmp[(t * 16 + h) * SPLIT_S + tid];
        l_s_smem[tid] = l_tmp[(t * 16 + h) * SPLIT_S + tid];
    }
    if (tid == 0) valid_flag = 0;
    __syncthreads();

    if (tid == 0) {
        float mm = -INFINITY;
        for (int s = 0; s < SPLIT_S; ++s) {
            if (l_s_smem[s] > 0.0f && m_s_smem[s] > mm) mm = m_s_smem[s];
        }
        m_global = mm;
        if (mm == -INFINITY) {
            inv_l = 0.0f;
            lse_ptr[t * 16 + h] = -INFINITY;
        } else {
            float lsum = 0.0f;
            for (int s = 0; s < SPLIT_S; ++s) {
                if (l_s_smem[s] > 0.0f) {
                    lsum += l_s_smem[s] * __expf(m_s_smem[s] - mm);
                }
            }
            if (lsum > 0.0f) {
                inv_l = __fdividef(1.0f, lsum);
                lse_ptr[t * 16 + h] = fmaf(mm, 1.4426950408889634f, __log2f(lsum));
                valid_flag = 1;
            } else {
                inv_l = 0.0f;
                lse_ptr[t * 16 + h] = -INFINITY;
            }
        }
    }
    __syncthreads();

    // Each thread handles 4 dims: total 128 threads * 4 = 512
    int d = tid * 4;
    if (!valid_flag) {
        __nv_bfloat162 zero = __floats2bfloat162_rn(0.0f, 0.0f);
        long out_base = (long)(t * 16 + h) * 512 + d;
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[out_base + 0]) = zero;
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[out_base + 2]) = zero;
        return;
    }

    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    float mg = m_global;
    float invl = inv_l;
    for (int s = 0; s < SPLIT_S; ++s) {
        float ms = m_s_smem[s];
        float ls = l_s_smem[s];
        if (ls <= 0.0f) continue;
        float coeff = __expf(ms - mg) * invl;
        long base = ((long)t * SPLIT_S + s) * 16 * 512 + h * 512 + d;
        v0 = fmaf(coeff, O_tmp[base + 0], v0);
        v1 = fmaf(coeff, O_tmp[base + 1], v1);
        v2 = fmaf(coeff, O_tmp[base + 2], v2);
        v3 = fmaf(coeff, O_tmp[base + 3], v3);
    }

    long out_base = (long)(t * 16 + h) * 512 + d;
    *reinterpret_cast<__nv_bfloat162*>(&output_ptr[out_base + 0]) = __floats2bfloat162_rn(v0, v1);
    *reinterpret_cast<__nv_bfloat162*>(&output_ptr[out_base + 2]) = __floats2bfloat162_rn(v2, v3);
}

std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    float sm_scale
) {
    CHECK_CUDA(q_nope); CHECK_CUDA(q_pe); CHECK_CUDA(ckv_cache); CHECK_CUDA(kpe_cache); CHECK_CUDA(sparse_indices);
    CHECK_CONTIGUOUS(q_nope); CHECK_CONTIGUOUS(q_pe); CHECK_CONTIGUOUS(ckv_cache); CHECK_CONTIGUOUS(kpe_cache); CHECK_CONTIGUOUS(sparse_indices);
    CHECK_DTYPE(q_nope, torch::kBFloat16);
    CHECK_DTYPE(q_pe, torch::kBFloat16);
    CHECK_DTYPE(ckv_cache, torch::kBFloat16);
    CHECK_DTYPE(kpe_cache, torch::kBFloat16);
    CHECK_DTYPE(sparse_indices, torch::kInt32);

    int num_tokens = q_nope.size(0);
    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16}, torch::dtype(torch::kFloat32).device(q_nope.device()));

    if (num_tokens == 0) return {output, lse};

    if (num_tokens < 128) {
        auto fp32_opts = torch::dtype(torch::kFloat32).device(q_nope.device());
        auto O_tmp = torch::empty({num_tokens, SPLIT_S, 16, 512}, fp32_opts);
        auto m_tmp = torch::empty({num_tokens, 16, SPLIT_S}, fp32_opts);
        auto l_tmp = torch::empty({num_tokens, 16, SPLIT_S}, fp32_opts);

        dim3 grid_c(SPLIT_S, num_tokens);
        dim3 block_c(32, 16);
        split_k_compute_kernel<<<grid_c, block_c>>>(
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

        dim3 grid_r(num_tokens, 16);
        dim3 block_r(128);
        split_k_reduce_kernel<<<grid_r, block_r>>>(
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>()
        );
    } else {
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
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}