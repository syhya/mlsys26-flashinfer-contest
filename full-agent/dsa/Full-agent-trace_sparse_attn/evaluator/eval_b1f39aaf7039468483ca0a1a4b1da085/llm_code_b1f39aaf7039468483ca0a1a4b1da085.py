#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

// Devised from plan: implement the exact small-workload split-K redesign with
// deterministic FP32 partial-state merge, while preserving the proven large fused path.
// We also replace the fragile shared-index mutation used for tile-validity encoding
// with an orthogonal shared boolean flag.

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 x) {
    return __bfloat1622float2(x);
}

__device__ __forceinline__ __nv_bfloat162 float2_to_bf162(float a, float b) {
    return __floats2bfloat162_rn(a, b);
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
        q_n_f32[j] = bf162_to_float2(q_val);
    }

    __nv_bfloat162 q_pe_v = *reinterpret_cast<const __nv_bfloat162*>(&q_pe_ptr[t * 16 * 64 + h * 64 + lane * 2]);
    float2 q_p_f32 = bf162_to_float2(q_pe_v);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;

    #pragma unroll 1
    for (int tile = 0; tile < 64; ++tile) {
        if (tid < 32) idx_shared[tid] = sparse_indices_ptr[t * 2048 + tile * 32 + tid];
        if (tid == 0) tile_has_valid = 0;
        __syncthreads();

        if (tid < 32 && idx_shared[tid] != -1) atomicExch(&tile_has_valid, 1);
        __syncthreads();
        if (!tile_has_valid) {
            __syncthreads();
            continue;
        }

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

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] == -1) continue;

            float local_dot = 0.0f;
            float2 k_f_reg[8];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
                float2 k_f = bf162_to_float2(k_n);
                k_f_reg[j] = k_f;
                local_dot = fmaf(q_n_f32[j].x, k_f.x, local_dot);
                local_dot = fmaf(q_n_f32[j].y, k_f.y, local_dot);
            }

            __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * 64 + lane * 2]);
            float2 k_pf = bf162_to_float2(k_p);
            local_dot = fmaf(q_p_f32.x, k_pf.x, local_dot);
            local_dot = fmaf(q_p_f32.y, k_pf.y, local_dot);

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

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

    float inv_l = (l > 0.0f) ? __fdividef(1.0f, l) : 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float ox = O_reg[j].x * inv_l;
        float oy = O_reg[j].y * inv_l;
        __nv_bfloat162 out_bf16 = float2_to_bf162(ox, oy);
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]) = out_bf16;
    }

    if (lane == 0) {
        lse_ptr[t * 16 + h] = (l > 0.0f) ? fmaf(m, 1.4426950408889634f, __log2f(l)) : -INFINITY;
    }
}

constexpr int SMALL_P = 64;
constexpr int SMALL_HG = 4;

__global__ void small_partial_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float* __restrict__ O_tmp,
    float* __restrict__ m_tmp,
    float* __restrict__ l_tmp
) {
    int t = blockIdx.x;
    int h = blockIdx.y * SMALL_HG + threadIdx.y;
    int p = blockIdx.z;
    int lane = threadIdx.x;

    if (h >= 16) return;

    float2 q_n_f32[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 q_val = *reinterpret_cast<const __nv_bfloat162*>(&q_nope_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]);
        q_n_f32[j] = bf162_to_float2(q_val);
    }
    __nv_bfloat162 q_pe_v = *reinterpret_cast<const __nv_bfloat162*>(&q_pe_ptr[t * 16 * 64 + h * 64 + lane * 2]);
    float2 q_p_f32 = bf162_to_float2(q_pe_v);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;

    int slot_begin = p * 32;
    int base_sparse = t * 2048;

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int raw = sparse_indices_ptr[base_sparse + slot_begin + i];
        if (raw == -1) continue;

        float local_dot = 0.0f;
        float2 k_f_reg[8];
        const __nv_bfloat16* k_nope_ptr = ckv_cache_ptr + raw * 512;
        const __nv_bfloat16* k_pe_ptr = kpe_cache_ptr + raw * 64;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&k_nope_ptr[j * 64 + lane * 2]);
            float2 k_f = bf162_to_float2(k_n);
            k_f_reg[j] = k_f;
            local_dot = fmaf(q_n_f32[j].x, k_f.x, local_dot);
            local_dot = fmaf(q_n_f32[j].y, k_f.y, local_dot);
        }

        __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(&k_pe_ptr[lane * 2]);
        float2 k_pf = bf162_to_float2(k_p);
        local_dot = fmaf(q_p_f32.x, k_pf.x, local_dot);
        local_dot = fmaf(q_p_f32.y, k_pf.y, local_dot);

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

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

    size_t state_base = ((size_t)t * 16 + h) * SMALL_P + p;
    if (lane == 0) {
        m_tmp[state_base] = m;
        l_tmp[state_base] = l;
    }

    size_t out_base = (((size_t)t * 16 + h) * SMALL_P + p) * 512;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<float2*>(&O_tmp[out_base + j * 64 + lane * 2]) = O_reg[j];
    }
}

__global__ void small_reduce_kernel(
    const float* __restrict__ O_tmp,
    const float* __restrict__ m_tmp,
    const float* __restrict__ l_tmp,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    __shared__ float m_global;
    __shared__ float l_global;
    __shared__ float m_smem[SMALL_P];
    __shared__ float scale_smem[SMALL_P];

    if (tid < SMALL_P) {
        size_t state_base = ((size_t)t * 16 + h) * SMALL_P + tid;
        m_smem[tid] = m_tmp[state_base];
    }
    __syncthreads();

    if (tid == 0) {
        float m_max = -INFINITY;
        #pragma unroll
        for (int p = 0; p < SMALL_P; ++p) {
            m_max = fmaxf(m_max, m_smem[p]);
        }

        float l_sum = 0.0f;
        #pragma unroll
        for (int p = 0; p < SMALL_P; ++p) {
            float mp = m_smem[p];
            float scale = (mp == -INFINITY) ? 0.0f : __expf(mp - m_max);
            scale_smem[p] = scale;
            if (scale > 0.0f) {
                size_t state_base = ((size_t)t * 16 + h) * SMALL_P + p;
                l_sum = fmaf(l_tmp[state_base], scale, l_sum);
            }
        }
        m_global = m_max;
        l_global = l_sum;
        lse_ptr[t * 16 + h] = (m_max == -INFINITY || l_sum <= 0.0f) ? -INFINITY : fmaf(m_max, 1.4426950408889634f, __log2f(l_sum));
    }
    __syncthreads();

    int d = tid * 4;
    if (d < 512) {
        float4 out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (m_global != -INFINITY && l_global > 0.0f) {
            #pragma unroll 16
            for (int p = 0; p < SMALL_P; ++p) {
                float scale = scale_smem[p];
                if (scale > 0.0f) {
                    size_t base = ((((size_t)t * 16 + h) * SMALL_P + p) * 512) + d;
                    float4 v = *reinterpret_cast<const float4*>(&O_tmp[base]);
                    out.x = fmaf(v.x, scale, out.x);
                    out.y = fmaf(v.y, scale, out.y);
                    out.z = fmaf(v.z, scale, out.z);
                    out.w = fmaf(v.w, scale, out.w);
                }
            }
            float inv_l = __fdividef(1.0f, l_global);
            out.x *= inv_l;
            out.y *= inv_l;
            out.z *= inv_l;
            out.w *= inv_l;
        }
        __nv_bfloat162 out0 = float2_to_bf162(out.x, out.y);
        __nv_bfloat162 out1 = float2_to_bf162(out.z, out.w);
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * 16 * 512 + h * 512 + d + 0]) = out0;
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * 16 * 512 + h * 512 + d + 2]) = out1;
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
    int num_tokens = q_nope.size(0);

    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    constexpr int TAU_SMALL = 96;
    if (num_tokens < TAU_SMALL) {
        auto O_tmp = torch::empty({num_tokens, 16, SMALL_P, 512}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto m_tmp = torch::empty({num_tokens, 16, SMALL_P}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto l_tmp = torch::empty({num_tokens, 16, SMALL_P}, torch::dtype(torch::kFloat32).device(q_nope.device()));

        dim3 compute_grid(num_tokens, 16 / SMALL_HG, SMALL_P);
        dim3 compute_block(32, SMALL_HG, 1);
        small_partial_kernel<<<compute_grid, compute_block>>>(
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

        dim3 reduce_grid(num_tokens, 16, 1);
        dim3 reduce_block(128, 1, 1);
        small_reduce_kernel<<<reduce_grid, reduce_block>>>(
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
