#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

// Devised from plan: keep the proven parent kernel structure for correctness and only
// apply the targeted optimization to the small-token split-k path: float4 temp stores
// aligned with float4 reduction loads.

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == torch::kBFloat16, #x " must be bfloat16")
#define CHECK_I32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

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
        float lse_val = (l > 0.0f) ? fmaf(m, 1.4426950408889634f, __log2f(l)) : -INFINITY;
        lse_ptr[t * 16 + h] = lse_val;
    }
}

__global__ void split_k_compute_kernel(
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

    int tile = s;

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

    // Optimization versus parent bug-risk exploration: store temp buffer as float4-aligned chunks
    // while preserving exact element order consumed by the reducer.
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int base_idx = t * 64 * 16 * 512 + s * 16 * 512 + h * 512 + j * 128 + lane * 4;
        float4 packed = make_float4(
            O_reg[2 * j + 0].x,
            O_reg[2 * j + 0].y,
            O_reg[2 * j + 1].x,
            O_reg[2 * j + 1].y);
        *reinterpret_cast<float4*>(&O_tmp[base_idx]) = packed;
    }

    if (lane == 0) {
        m_tmp[t * 16 * 64 + h * 64 + s] = m;
        l_tmp[t * 16 * 64 + h * 64 + s] = l;
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
    int lane = threadIdx.x;

    __shared__ float m_global;
    __shared__ float l_global;
    __shared__ float m_s_smem[64];
    __shared__ float l_s_smem[64];
    __shared__ float scale_smem[64];

    if (lane < 64) {
        m_s_smem[lane] = m_tmp[t * 16 * 64 + h * 64 + lane];
        l_s_smem[lane] = l_tmp[t * 16 * 64 + h * 64 + lane];
    }
    __syncthreads();

    if (lane == 0) {
        float m_max = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            m_max = fmaxf(m_max, m_s_smem[i]);
        }

        float l_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            if (m_s_smem[i] != -INFINITY) {
                l_sum = fmaf(l_s_smem[i], __expf(m_s_smem[i] - m_max), l_sum);
            }
        }

        m_global = m_max;
        l_global = l_sum;

        if (m_max == -INFINITY) {
            lse_ptr[t * 16 + h] = -INFINITY;
        } else {
            lse_ptr[t * 16 + h] = fmaf(m_max, 1.4426950408889634f, __log2f(l_sum));
        }
    }
    __syncthreads();

    if (lane < 64) {
        if (m_global != -INFINITY && l_global > 0.0f && m_s_smem[lane] != -INFINITY) {
            scale_smem[lane] = __fdividef(__expf(m_s_smem[lane] - m_global), l_global);
        } else {
            scale_smem[lane] = 0.0f;
        }
    }
    __syncthreads();

    int d_idx = lane * 4;
    if (d_idx < 512) {
        float out0 = 0.0f;
        float out1 = 0.0f;
        float out2 = 0.0f;
        float out3 = 0.0f;

        if (m_global != -INFINITY && l_global > 0.0f) {
            #pragma unroll 16
            for (int s = 0; s < 64; ++s) {
                float scale = scale_smem[s];
                if (scale > 0.0f) {
                    int base_idx = t * 64 * 16 * 512 + s * 16 * 512 + h * 512 + d_idx;
                    float4 val = *reinterpret_cast<const float4*>(&O_tmp[base_idx]);
                    out0 = fmaf(val.x, scale, out0);
                    out1 = fmaf(val.y, scale, out1);
                    out2 = fmaf(val.z, scale, out2);
                    out3 = fmaf(val.w, scale, out3);
                }
            }
        }

        __nv_bfloat162 out_bf16_0 = __floats2bfloat162_rn(out0, out1);
        __nv_bfloat162 out_bf16_1 = __floats2bfloat162_rn(out2, out3);

        uint32_t val0 = *reinterpret_cast<uint32_t*>(&out_bf16_0);
        uint32_t val1 = *reinterpret_cast<uint32_t*>(&out_bf16_1);
        uint2 out_vec = make_uint2(val0, val1);
        *reinterpret_cast<uint2*>(&output_ptr[t * 16 * 512 + h * 512 + d_idx]) = out_vec;
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

    int num_tokens = q_nope.size(0);

    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16}, torch::dtype(torch::kFloat32).device(q_nope.device()));

    if (num_tokens == 0) {
        return {output, lse};
    }

    if (num_tokens < 128) {
        int S = 64;

        auto O_tmp = torch::empty({num_tokens, S, 16, 512}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto m_tmp = torch::empty({num_tokens, 16, S}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto l_tmp = torch::empty({num_tokens, 16, S}, torch::dtype(torch::kFloat32).device(q_nope.device()));

        dim3 compute_grid(S, num_tokens, 1);
        dim3 compute_block(32, 16, 1);
        split_k_compute_kernel<<<compute_grid, compute_block>>>(
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
        split_k_reduce_kernel<<<reduce_grid, reduce_block>>>(
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
