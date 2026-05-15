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

// =============================================================
// Large-workload fused kernel: one block per token, 16 heads * 32 lanes
// =============================================================
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
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * 512];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * 64];

    float2 q_n_f32[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 q_val = *reinterpret_cast<const __nv_bfloat162*>(
            &q_nope_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]);
        q_n_f32[j] = __bfloat1622float2(q_val);
    }
    float2 q_p_f32;
    {
        __nv_bfloat162 q_val = *reinterpret_cast<const __nv_bfloat162*>(
            &q_pe_ptr[t * 16 * 64 + h * 64 + lane * 2]);
        q_p_f32 = __bfloat1622float2(q_val);
    }

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    int num_tiles = 2048 / 32;
    #pragma unroll 1
    for (int tile = 0; tile < num_tiles; ++tile) {
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * 2048 + tile * 32 + tid];
        }
        __syncthreads();

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

        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] == -1) continue;
            ++valid_count;

            float local_dot = 0.0f;
            float2 k_f_reg[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(
                    &smem_Kc[i * 512 + j * 64 + lane * 2]);
                float2 k_f = __bfloat1622float2(k_n);
                k_f_reg[j] = k_f;
                local_dot = fmaf(q_n_f32[j].x, k_f.x, local_dot);
                local_dot = fmaf(q_n_f32[j].y, k_f.y, local_dot);
            }
            {
                __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(
                    &smem_Kp[i * 64 + lane * 2]);
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

    if (valid_count == 0 || l <= 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat162 out_bf16 = __floats2bfloat162_rn(0.0f, 0.0f);
            *reinterpret_cast<__nv_bfloat162*>(
                &output_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]) = out_bf16;
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
        *reinterpret_cast<__nv_bfloat162*>(
            &output_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]) = out_bf16;
    }
    if (lane == 0) {
        lse_ptr[t * 16 + h] = fmaf(m, 1.4426950408889634f, __log2f(l));
    }
}

// =============================================================
// Split-K compute kernel: each block handles one (token, partition s)
// processes 32 sparse keys (tile s of 32 keys), all 16 heads
// =============================================================
__global__ void split_k_compute_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float* __restrict__ O_tmp_ptr, // [T, S, 16, 512]
    float* __restrict__ m_tmp_ptr, // [T, 16, S]
    float* __restrict__ l_tmp_ptr  // [T, 16, S]
) {
    constexpr int S = 64;
    int s = blockIdx.x;
    int t = blockIdx.y;
    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = threadIdx.y * 32 + threadIdx.x;

    __shared__ alignas(16) int idx_shared[32];
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * 512];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * 64];

    float2 q_n_f32[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 q_val = *reinterpret_cast<const __nv_bfloat162*>(
            &q_nope_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]);
        q_n_f32[j] = __bfloat1622float2(q_val);
    }
    float2 q_p_f32;
    {
        __nv_bfloat162 q_val = *reinterpret_cast<const __nv_bfloat162*>(
            &q_pe_ptr[t * 16 * 64 + h * 64 + lane * 2]);
        q_p_f32 = __bfloat1622float2(q_val);
    }

    if (tid < 32) {
        idx_shared[tid] = sparse_indices_ptr[t * 2048 + s * 32 + tid];
    }
    __syncthreads();

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

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    for (int i = 0; i < 32; ++i) {
        if (idx_shared[i] == -1) continue;
        ++valid_count;

        float local_dot = 0.0f;
        float2 k_f_reg[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(
                &smem_Kc[i * 512 + j * 64 + lane * 2]);
            float2 k_f = __bfloat1622float2(k_n);
            k_f_reg[j] = k_f;
            local_dot = fmaf(q_n_f32[j].x, k_f.x, local_dot);
            local_dot = fmaf(q_n_f32[j].y, k_f.y, local_dot);
        }
        {
            __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(
                &smem_Kp[i * 64 + lane * 2]);
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

    // Store partial unnormalized O, m, l
    // O_tmp layout: [T, S, 16, 512]
    size_t O_base = ((size_t)t * S + s) * 16 * 512 + (size_t)h * 512;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int d = j * 64 + lane * 2;
        O_tmp_ptr[O_base + d]     = O_reg[j].x;
        O_tmp_ptr[O_base + d + 1] = O_reg[j].y;
    }
    if (lane == 0) {
        size_t ml_base = ((size_t)t * 16 + h) * S + s;
        m_tmp_ptr[ml_base] = (valid_count == 0) ? -INFINITY : m;
        l_tmp_ptr[ml_base] = (valid_count == 0) ? 0.0f : l;
    }
}

// =============================================================
// Split-K reduce kernel: combine S partials into final output + lse
// grid: (num_tokens, 16), block: 128
// =============================================================
__global__ void split_k_reduce_kernel(
    const float* __restrict__ O_tmp_ptr, // [T, S, 16, 512]
    const float* __restrict__ m_tmp_ptr, // [T, 16, S]
    const float* __restrict__ l_tmp_ptr, // [T, 16, S]
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    constexpr int S = 64;
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    __shared__ float m_smem[S];
    __shared__ float l_smem[S];
    __shared__ float m_global;
    __shared__ float l_global;

    size_t ml_base = ((size_t)t * 16 + h) * S;

    if (tid < S) {
        m_smem[tid] = m_tmp_ptr[ml_base + tid];
        l_smem[tid] = l_tmp_ptr[ml_base + tid];
    }
    __syncthreads();

    if (tid == 0) {
        float mm = -INFINITY;
        #pragma unroll
        for (int i = 0; i < S; ++i) {
            if (m_smem[i] > mm) mm = m_smem[i];
        }
        float ll = 0.0f;
        if (mm != -INFINITY) {
            #pragma unroll
            for (int i = 0; i < S; ++i) {
                if (m_smem[i] != -INFINITY) {
                    ll += l_smem[i] * __expf(m_smem[i] - mm);
                }
            }
        }
        m_global = mm;
        l_global = ll;
        if (ll > 0.0f) {
            lse_ptr[t * 16 + h] = fmaf(mm, 1.4426950408889634f, __log2f(ll));
        } else {
            lse_ptr[t * 16 + h] = -INFINITY;
        }
    }
    __syncthreads();

    float mm = m_global;
    float ll = l_global;

    // Each thread handles 4 dims: d = tid*4
    int d = tid * 4;
    if (d >= 512) return;

    bool empty = (mm == -INFINITY) || (ll <= 0.0f);
    float inv_l = empty ? 0.0f : __fdividef(1.0f, ll);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    if (!empty) {
        for (int s = 0; s < S; ++s) {
            float ms = m_smem[s];
            if (ms == -INFINITY) continue;
            float coeff = __expf(ms - mm) * inv_l;
            size_t O_base = ((size_t)t * S + s) * 16 * 512 + (size_t)h * 512 + d;
            acc0 = fmaf(coeff, O_tmp_ptr[O_base + 0], acc0);
            acc1 = fmaf(coeff, O_tmp_ptr[O_base + 1], acc1);
            acc2 = fmaf(coeff, O_tmp_ptr[O_base + 2], acc2);
            acc3 = fmaf(coeff, O_tmp_ptr[O_base + 3], acc3);
        }
    }

    size_t out_base = (size_t)t * 16 * 512 + (size_t)h * 512 + d;
    __nv_bfloat162 a = __floats2bfloat162_rn(acc0, acc1);
    __nv_bfloat162 b = __floats2bfloat162_rn(acc2, acc3);
    *reinterpret_cast<__nv_bfloat162*>(&output_ptr[out_base])     = a;
    *reinterpret_cast<__nv_bfloat162*>(&output_ptr[out_base + 2]) = b;
}

std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    float sm_scale
) {
    CHECK_CUDA(q_nope); CHECK_CUDA(q_pe); CHECK_CUDA(ckv_cache);
    CHECK_CUDA(kpe_cache); CHECK_CUDA(sparse_indices);
    CHECK_CONTIGUOUS(q_nope); CHECK_CONTIGUOUS(q_pe); CHECK_CONTIGUOUS(ckv_cache);
    CHECK_CONTIGUOUS(kpe_cache); CHECK_CONTIGUOUS(sparse_indices);
    CHECK_DTYPE(q_nope, torch::kBFloat16);
    CHECK_DTYPE(q_pe, torch::kBFloat16);
    CHECK_DTYPE(ckv_cache, torch::kBFloat16);
    CHECK_DTYPE(kpe_cache, torch::kBFloat16);
    CHECK_DTYPE(sparse_indices, torch::kInt32);

    int num_tokens = q_nope.size(0);
    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16},
        torch::dtype(torch::kFloat32).device(q_nope.device()));

    if (num_tokens == 0) return {output, lse};

    auto q_nope_p = reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>());
    auto q_pe_p = reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>());
    auto ckv_p = reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>());
    auto kpe_p = reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>());
    auto idx_p = sparse_indices.data_ptr<int32_t>();
    auto out_p = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    auto lse_p = lse.data_ptr<float>();

    if (num_tokens < 128) {
        constexpr int S = 64;
        auto fp32_opts = torch::dtype(torch::kFloat32).device(q_nope.device());
        auto O_tmp = torch::empty({num_tokens, S, 16, 512}, fp32_opts);
        auto m_tmp = torch::empty({num_tokens, 16, S}, fp32_opts);
        auto l_tmp = torch::empty({num_tokens, 16, S}, fp32_opts);

        dim3 grid_c(S, num_tokens);
        dim3 block_c(32, 16);
        split_k_compute_kernel<<<grid_c, block_c>>>(
            q_nope_p, q_pe_p, ckv_p, kpe_p, idx_p, sm_scale,
            O_tmp.data_ptr<float>(), m_tmp.data_ptr<float>(), l_tmp.data_ptr<float>()
        );

        dim3 grid_r(num_tokens, 16);
        dim3 block_r(128);
        split_k_reduce_kernel<<<grid_r, block_r>>>(
            O_tmp.data_ptr<float>(), m_tmp.data_ptr<float>(), l_tmp.data_ptr<float>(),
            out_p, lse_p
        );
    } else {
        dim3 grid(num_tokens);
        dim3 block(32, 16);
        dsa_forward_kernel<<<grid, block>>>(
            q_nope_p, q_pe_p, ckv_p, kpe_p, idx_p, sm_scale, out_p, lse_p
        );
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}