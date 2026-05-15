#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <int K_VEC>
__global__ void __launch_bounds__(128) gdn_decode_kernel_fast(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state,
    int num_v_heads, int num_q_heads, int V, int K
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int v_group_idx = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sh_g;
    __shared__ float sh_beta;

    if (tx == 0 && ty == 0) {
        float a_val = __bfloat162float(a[b_idx * num_v_heads + h_idx]);
        float b_val = __bfloat162float(b[b_idx * num_v_heads + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : logf(1.0f + expf(x));
        sh_g = expf(-expf(A_log_val) * sp);
        sh_beta = 1.0f / (1.0f + expf(-b_val));
    }
    __syncthreads();

    float g = sh_g;
    float beta = sh_beta;

    int qk_h_idx = h_idx * num_q_heads / num_v_heads;
    const __nv_bfloat16* q_ptr = q + b_idx * num_q_heads * K + qk_h_idx * K;
    const __nv_bfloat16* k_ptr = k + b_idx * num_q_heads * K + qk_h_idx * K;

    float q_f[K_VEC];
    float k_f[K_VEC];

    if constexpr (K_VEC == 2) {
        float q_vec = reinterpret_cast<const float*>(q_ptr)[tx];
        __nv_bfloat16* q_h_ptr = reinterpret_cast<__nv_bfloat16*>(&q_vec);
        q_f[0] = __bfloat162float(q_h_ptr[0]);
        q_f[1] = __bfloat162float(q_h_ptr[1]);

        float k_vec = reinterpret_cast<const float*>(k_ptr)[tx];
        __nv_bfloat16* k_h_ptr = reinterpret_cast<__nv_bfloat16*>(&k_vec);
        k_f[0] = __bfloat162float(k_h_ptr[0]);
        k_f[1] = __bfloat162float(k_h_ptr[1]);
    } else if constexpr (K_VEC == 4) {
        float2 q_vec = reinterpret_cast<const float2*>(q_ptr)[tx];
        __nv_bfloat16* q_h_ptr = reinterpret_cast<__nv_bfloat16*>(&q_vec);
        #pragma unroll
        for(int i=0; i<4; ++i) q_f[i] = __bfloat162float(q_h_ptr[i]);

        float2 k_vec = reinterpret_cast<const float2*>(k_ptr)[tx];
        __nv_bfloat16* k_h_ptr = reinterpret_cast<__nv_bfloat16*>(&k_vec);
        #pragma unroll
        for(int i=0; i<4; ++i) k_f[i] = __bfloat162float(k_h_ptr[i]);
    } else if constexpr (K_VEC == 8) {
        float4 q_vec = reinterpret_cast<const float4*>(q_ptr)[tx];
        __nv_bfloat16* q_h_ptr = reinterpret_cast<__nv_bfloat16*>(&q_vec);
        #pragma unroll
        for(int i=0; i<8; ++i) q_f[i] = __bfloat162float(q_h_ptr[i]);

        float4 k_vec = reinterpret_cast<const float4*>(k_ptr)[tx];
        __nv_bfloat16* k_h_ptr = reinterpret_cast<__nv_bfloat16*>(&k_vec);
        #pragma unroll
        for(int i=0; i<8; ++i) k_f[i] = __bfloat162float(k_h_ptr[i]);
    }

    float qk_local = 0.0f;
    #pragma unroll
    for (int i = 0; i < K_VEC; ++i) {
        qk_local += q_f[i] * k_f[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);

    const float* state_base = state != nullptr ? state + b_idx * num_v_heads * V * K + h_idx * V * K : nullptr;
    float* new_state_base = new_state + b_idx * num_v_heads * V * K + h_idx * V * K;
    const __nv_bfloat16* v_ptr = v + b_idx * num_v_heads * V + h_idx * V;
    __nv_bfloat16* out_ptr = output + b_idx * num_v_heads * V + h_idx * V;

    int v_idx = v_group_idx * blockDim.y + ty;
    if (v_idx < V) {
        float st[K_VEC];
        if constexpr (K_VEC == 2) {
            float2 st_vec = make_float2(0.0f, 0.0f);
            if (state_base != nullptr) {
                st_vec = reinterpret_cast<const float2*>(state_base + v_idx * K)[tx];
            }
            st[0] = st_vec.x; st[1] = st_vec.y;
        } else if constexpr (K_VEC == 4) {
            float4 st_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (state_base != nullptr) {
                st_vec = reinterpret_cast<const float4*>(state_base + v_idx * K)[tx];
            }
            st[0] = st_vec.x; st[1] = st_vec.y; st[2] = st_vec.z; st[3] = st_vec.w;
        } else if constexpr (K_VEC == 8) {
            float4 st_vec1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 st_vec2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (state_base != nullptr) {
                st_vec1 = reinterpret_cast<const float4*>(state_base + v_idx * K)[tx * 2];
                st_vec2 = reinterpret_cast<const float4*>(state_base + v_idx * K)[tx * 2 + 1];
            }
            st[0] = st_vec1.x; st[1] = st_vec1.y; st[2] = st_vec1.z; st[3] = st_vec1.w;
            st[4] = st_vec2.x; st[5] = st_vec2.y; st[6] = st_vec2.z; st[7] = st_vec2.w;
        }

        float qh_local = 0.0f;
        float kh_local = 0.0f;
        #pragma unroll
        for (int i = 0; i < K_VEC; ++i) {
            qh_local += q_f[i] * st[i];
            kh_local += k_f[i] * st[i];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
            kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        }
        float qh_v = __shfl_sync(0xffffffff, qh_local, 0);
        float kh_v = __shfl_sync(0xffffffff, kh_local, 0);

        float v_val = 0.0f;
        if (tx == 0) {
            v_val = __bfloat162float(v_ptr[v_idx]);
        }
        v_val = __shfl_sync(0xffffffff, v_val, 0);

        float old_v_v = g * kh_v;
        float dv_v = beta * (v_val - old_v_v);
        float out_v = scale * (g * qh_v + dv_v * qk_dot);

        if (tx == 0) {
            out_ptr[v_idx] = __float2bfloat16(out_v);
        }

        #pragma unroll
        for (int i = 0; i < K_VEC; ++i) {
            st[i] = g * st[i] + dv_v * k_f[i];
        }

        if constexpr (K_VEC == 2) {
            reinterpret_cast<float2*>(new_state_base + v_idx * K)[tx] = make_float2(st[0], st[1]);
        } else if constexpr (K_VEC == 4) {
            reinterpret_cast<float4*>(new_state_base + v_idx * K)[tx] = make_float4(st[0], st[1], st[2], st[3]);
        } else if constexpr (K_VEC == 8) {
            reinterpret_cast<float4*>(new_state_base + v_idx * K)[tx * 2] = make_float4(st[0], st[1], st[2], st[3]);
            reinterpret_cast<float4*>(new_state_base + v_idx * K)[tx * 2 + 1] = make_float4(st[4], st[5], st[6], st[7]);
        }
    }
}

__global__ void gdn_decode_kernel_generic(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state,
    int B, int num_v_heads, int num_q_heads, int V, int K
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int v_group_idx = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sh_g;
    __shared__ float sh_beta;

    if (tx == 0 && ty == 0) {
        float a_val = __bfloat162float(a[b_idx * num_v_heads + h_idx]);
        float b_val = __bfloat162float(b[b_idx * num_v_heads + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : logf(1.0f + expf(x));
        sh_g = expf(-expf(A_log_val) * sp);
        sh_beta = 1.0f / (1.0f + expf(-b_val));
    }
    __syncthreads();

    float g = sh_g;
    float beta = sh_beta;

    int qk_h_idx = h_idx * num_q_heads / num_v_heads;
    const __nv_bfloat16* q_ptr = q + b_idx * num_q_heads * K + qk_h_idx * K;
    const __nv_bfloat16* k_ptr = k + b_idx * num_q_heads * K + qk_h_idx * K;

    float qk_local = 0.0f;
    for (int i = tx; i < K; i += blockDim.x) {
        qk_local += __bfloat162float(q_ptr[i]) * __bfloat162float(k_ptr[i]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);

    const float* state_base = state != nullptr ? state + b_idx * num_v_heads * V * K + h_idx * V * K : nullptr;
    float* new_state_base = new_state + b_idx * num_v_heads * V * K + h_idx * V * K;
    const __nv_bfloat16* v_ptr = v + b_idx * num_v_heads * V + h_idx * V;
    __nv_bfloat16* out_ptr = output + b_idx * num_v_heads * V + h_idx * V;

    int v_idx = v_group_idx * blockDim.y + ty;
    if (v_idx < V) {
        float qh_local = 0.0f;
        float kh_local = 0.0f;

        for (int i = tx; i < K; i += blockDim.x) {
            float st = state_base != nullptr ? state_base[v_idx * K + i] : 0.0f;
            float q_val = __bfloat162float(q_ptr[i]);
            float k_val = __bfloat162float(k_ptr[i]);

            qh_local += q_val * st;
            kh_local += k_val * st;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
            kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        }
        float qh_v = __shfl_sync(0xffffffff, qh_local, 0);
        float kh_v = __shfl_sync(0xffffffff, kh_local, 0);

        float v_val = 0.0f;
        if (tx == 0) {
            v_val = __bfloat162float(v_ptr[v_idx]);
        }
        v_val = __shfl_sync(0xffffffff, v_val, 0);

        float old_v_v = g * kh_v;
        float dv_v = beta * (v_val - old_v_v);
        float out_v = scale * (g * qh_v + dv_v * qk_dot);

        if (tx == 0) {
            out_ptr[v_idx] = __float2bfloat16(out_v);
        }

        for (int i = tx; i < K; i += blockDim.x) {
            float st = state_base != nullptr ? state_base[v_idx * K + i] : 0.0f;
            float k_val = __bfloat162float(k_ptr[i]);
            float new_st = g * st + dv_v * k_val;
            new_state_base[v_idx * K + i] = new_st;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> gdn_forward(
    torch::Tensor q,       // [batch_size, 1, num_q_heads, K]     bfloat16
    torch::Tensor k,       // [batch_size, 1, num_q_heads, K]     bfloat16
    torch::Tensor v,       // [batch_size, 1, num_v_heads, V]     bfloat16
    torch::Tensor state,   // [batch_size, num_v_heads, V, K]   float32  (k-last layout)
    torch::Tensor A_log,   // [num_v_heads]                          float32
    torch::Tensor a,       // [batch_size, 1, num_v_heads]           bfloat16
    torch::Tensor dt_bias, // [num_v_heads]                          float32
    torch::Tensor b,       // [batch_size, 1, num_v_heads]           bfloat16
    float scale            // scalar
) {
    int B = q.size(0);
    int num_q_heads = q.size(2);
    int num_v_heads = v.size(2);
    int V = v.size(3);
    int K = q.size(3);

    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(K));
    }

    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    A_log = A_log.contiguous();
    a = a.contiguous();
    dt_bias = dt_bias.contiguous();
    b = b.contiguous();

    const float* state_ptr = nullptr;
    if (state.defined() && state.numel() > 0) {
        state = state.contiguous();
        state_ptr = state.data_ptr<float>();
    }

    auto options_bf16 = q.options();
    auto options_f32 = options_bf16.dtype(torch::kFloat32);

    torch::Tensor new_state = torch::empty({B, num_v_heads, V, K}, options_f32);
    torch::Tensor output = torch::empty({B, 1, num_v_heads, V}, options_bf16);

    dim3 block(32, 4);
    dim3 grid(B, num_v_heads, (V + 3) / 4);

    if (K == 64) {
        gdn_decode_kernel_fast<2><<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
            state_ptr, A_log.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
            dt_bias.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
            scale, reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            new_state.data_ptr<float>(), num_v_heads, num_q_heads, V, K
        );
    } else if (K == 128) {
        gdn_decode_kernel_fast<4><<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
            state_ptr, A_log.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
            dt_bias.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
            scale, reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            new_state.data_ptr<float>(), num_v_heads, num_q_heads, V, K
        );
    } else if (K == 256) {
        gdn_decode_kernel_fast<8><<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
            state_ptr, A_log.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
            dt_bias.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
            scale, reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            new_state.data_ptr<float>(), num_v_heads, num_q_heads, V, K
        );
    } else {
        gdn_decode_kernel_generic<<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
            state_ptr, A_log.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
            dt_bias.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
            scale, reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            new_state.data_ptr<float>(), B, num_v_heads, num_q_heads, V, K
        );
    }

    return std::make_tuple(output, new_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_forward", &gdn_forward, "GDN Forward");
}
