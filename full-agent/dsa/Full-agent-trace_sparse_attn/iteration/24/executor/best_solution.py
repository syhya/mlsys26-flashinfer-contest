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
    for (int tile = 0; tile < num_tiles; ++tile) {
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * 2048 + tile * 32 + tid];
        }
        __syncthreads();

        if (tid == 0) {
            int any_valid = 0;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                any_valid |= (idx_shared[i] != -1);
            }
            tile_has_valid = any_valid;
        }
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

    float inv_l = (l > 0.0f) ? __fdividef(1.0f, l) : 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float ox = O_reg[j].x * inv_l;
        float oy = O_reg[j].y * inv_l;
        __nv_bfloat162 out_bf16 = __floats2bfloat162_rn(ox, oy);
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * 16 * 512 + h * 512 + j * 64 + lane * 2]) = out_bf16;
    }

    if (lane == 0) {
        float lse_val = (l > 0.0f) ? fmaf(m, 1.4426950408889634f, __log2f(l)) : -INFINITY;
        lse_ptr[t * 16 + h] = lse_val;
    }
}

// Split-K compute kernel: one block per (partition, token), computes partial state for all 16 heads
__global__ void split_k_compute_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float* __restrict__ O_tmp,    // [T, 64, 16, 512]
    float* __restrict__ m_tmp,    // [T, 16, 64]
    float* __restrict__ l_tmp     // [T, 16, 64]
) {
    int s = blockIdx.x;  // partition index in [0,64)
    int t = blockIdx.y;  // token
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

    if (tid < 32) {
        idx_shared[tid] = sparse_indices_ptr[t * 2048 + s * 32 + tid];
    }
    __syncthreads();

    if (tid == 0) {
        int any_valid = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            any_valid |= (idx_shared[i] != -1);
        }
        tile_has_valid = any_valid;
    }
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
    }

    // Write partial state
    // O_tmp layout: [T, 64, 16, 512]
    size_t o_base = ((size_t)t * 64 + s) * 16 * 512 + h * 512;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<float2*>(&O_tmp[o_base + j * 64 + lane * 2]) = O_reg[j];
    }

    if (lane == 0) {
        // m_tmp, l_tmp layout: [T, 16, 64]
        m_tmp[t * 16 * 64 + h * 64 + s] = m;
        l_tmp[t * 16 * 64 + h * 64 + s] = l;
    }
}

__global__ void split_k_reduce_kernel(
    const float* __restrict__ O_tmp,  // [T, 64, 16, 512]
    const float* __restrict__ m_tmp,  // [T, 16, 64]
    const float* __restrict__ l_tmp,  // [T, 16, 64]
    __nv_bfloat16* __restrict__ output_ptr,  // [T, 16, 512]
    float* __restrict__ lse_ptr              // [T, 16]
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int lane = threadIdx.x;  // 0..127

    __shared__ float m_global;
    __shared__ float l_global;
    __shared__ float m_s_smem[64];

    if (lane < 64) {
        m_s_smem[lane] = m_tmp[t * 16 * 64 + h * 64 + lane];
    }
    __syncthreads();

    if (lane == 0) {
        float m_max = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            float ms = m_s_smem[i];
            if (ms > m_max) m_max = ms;
        }
        float l_sum = 0.0f;
        if (m_max != -INFINITY) {
            for (int i = 0; i < 64; ++i) {
                float ms = m_s_smem[i];
                if (ms != -INFINITY) {
                    float ls = l_tmp[t * 16 * 64 + h * 64 + i];
                    l_sum += ls * __expf(ms - m_max);
                }
            }
        }
        m_global = m_max;
        l_global = l_sum;
        lse_ptr[t * 16 + h] = (m_max != -INFINITY && l_sum > 0.0f) ?
            fmaf(m_max, 1.4426950408889634f, __log2f(l_sum)) : -INFINITY;
    }
    __syncthreads();

    int d_idx = lane * 4;
    if (d_idx < 512) {
        float m_g = m_global;
        float l_g = l_global;
        bool valid = (m_g != -INFINITY && l_g > 0.0f);
        float inv_l = valid ? __fdividef(1.0f, l_g) : 0.0f;

        float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
        if (valid) {
            for (int s = 0; s < 64; ++s) {
                float ms = m_s_smem[s];
                if (ms == -INFINITY) continue;
                float coeff = __expf(ms - m_g) * inv_l;
                size_t o_base = ((size_t)t * 64 + s) * 16 * 512 + h * 512 + d_idx;
                float4 v = *reinterpret_cast<const float4*>(&O_tmp[o_base]);
                // Note: O_tmp stores unnormalized partial accumulator (sum over partition)
                // To reconstruct contribution: we need ls partial. But coeff above uses inv_l of global l.
                // Correct merge: O_global = sum_s O_s * exp(m_s - m_g), then divide by l_g.
                // So coeff per partition = exp(m_s - m_g) / l_g. And O_s here is unnormalized.
                out0 = fmaf(v.x, coeff, out0);
                out1 = fmaf(v.y, coeff, out1);
                out2 = fmaf(v.z, coeff, out2);
                out3 = fmaf(v.w, coeff, out3);
            }
        }

        __nv_bfloat162 p01 = __floats2bfloat162_rn(out0, out1);
        __nv_bfloat162 p23 = __floats2bfloat162_rn(out2, out3);
        uint2 packed;
        packed.x = *reinterpret_cast<unsigned int*>(&p01);
        packed.y = *reinterpret_cast<unsigned int*>(&p23);
        *reinterpret_cast<uint2*>(&output_ptr[t * 16 * 512 + h * 512 + d_idx]) = packed;
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
    CHECK_CUDA(q_nope); CHECK_CUDA(q_pe);
    CHECK_CUDA(ckv_cache); CHECK_CUDA(kpe_cache);
    CHECK_CUDA(sparse_indices);
    CHECK_CONTIGUOUS(q_nope); CHECK_CONTIGUOUS(q_pe);
    CHECK_CONTIGUOUS(ckv_cache); CHECK_CONTIGUOUS(kpe_cache);
    CHECK_CONTIGUOUS(sparse_indices);
    CHECK_DTYPE(q_nope, torch::kBFloat16);
    CHECK_DTYPE(q_pe, torch::kBFloat16);
    CHECK_DTYPE(ckv_cache, torch::kBFloat16);
    CHECK_DTYPE(kpe_cache, torch::kBFloat16);
    CHECK_DTYPE(sparse_indices, torch::kInt32);

    int num_tokens = q_nope.size(0);
    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16}, torch::dtype(torch::kFloat32).device(q_nope.device()));

    if (num_tokens == 0) {
        return {output, lse};
    }

    auto q_nope_p = reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>());
    auto q_pe_p = reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>());
    auto ckv_p = reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>());
    auto kpe_p = reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>());
    auto si_p = sparse_indices.data_ptr<int32_t>();
    auto out_p = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    auto lse_p = lse.data_ptr<float>();

    if (num_tokens < 128) {
        auto opts_f32 = torch::dtype(torch::kFloat32).device(q_nope.device());
        auto O_tmp = torch::empty({num_tokens, 64, 16, 512}, opts_f32);
        auto m_tmp = torch::empty({num_tokens, 16, 64}, opts_f32);
        auto l_tmp = torch::empty({num_tokens, 16, 64}, opts_f32);

        dim3 grid_c(64, num_tokens, 1);
        dim3 block_c(32, 16, 1);
        split_k_compute_kernel<<<grid_c, block_c>>>(
            q_nope_p, q_pe_p, ckv_p, kpe_p, si_p, sm_scale,
            O_tmp.data_ptr<float>(), m_tmp.data_ptr<float>(), l_tmp.data_ptr<float>()
        );

        dim3 grid_r(num_tokens, 16, 1);
        dim3 block_r(128, 1, 1);
        split_k_reduce_kernel<<<grid_r, block_r>>>(
            O_tmp.data_ptr<float>(), m_tmp.data_ptr<float>(), l_tmp.data_ptr<float>(),
            out_p, lse_p
        );
    } else {
        dim3 grid(num_tokens);
        dim3 block(32, 16);
        dsa_forward_kernel<<<grid, block>>>(
            q_nope_p, q_pe_p, ckv_p, kpe_p, si_p, sm_scale, out_p, lse_p
        );
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}