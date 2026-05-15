#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <tuple>
#include <cmath>

template <int NUM_V_HEADS, int NUM_Q_HEADS, int V, int K>
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
    float* __restrict__ new_state
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int v_group_idx = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sh_g;
    __shared__ float sh_beta;

    // Compute head-level scalars exactly once per block
    if (tx == 0 && ty == 0) {
        float a_val = __bfloat162float(a[b_idx * NUM_V_HEADS + h_idx]);
        float b_val = __bfloat162float(b[b_idx * NUM_V_HEADS + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : __logf(1.0f + __expf(x));
        sh_g = __expf(-__expf(A_log_val) * sp);
        sh_beta = 1.0f / (1.0f + __expf(-b_val));
    }
    __syncthreads();

    float g = sh_g;
    float beta = sh_beta;

    // GVA mapping
    int qk_h_idx = h_idx * NUM_Q_HEADS / NUM_V_HEADS;
    const __nv_bfloat16* q_ptr = q + b_idx * NUM_Q_HEADS * K + qk_h_idx * K;
    const __nv_bfloat16* k_ptr = k + b_idx * NUM_Q_HEADS * K + qk_h_idx * K;

    // Vectorized read of 4 bfloat16 elements into registers using float2
    float2 q_vec = reinterpret_cast<const float2*>(q_ptr)[tx];
    __nv_bfloat16* q_h_ptr = reinterpret_cast<__nv_bfloat16*>(&q_vec);
    float q_f[4];
    q_f[0] = __bfloat162float(q_h_ptr[0]);
    q_f[1] = __bfloat162float(q_h_ptr[1]);
    q_f[2] = __bfloat162float(q_h_ptr[2]);
    q_f[3] = __bfloat162float(q_h_ptr[3]);

    float2 k_vec = reinterpret_cast<const float2*>(k_ptr)[tx];
    __nv_bfloat16* k_h_ptr = reinterpret_cast<__nv_bfloat16*>(&k_vec);
    float k_f[4];
    k_f[0] = __bfloat162float(k_h_ptr[0]);
    k_f[1] = __bfloat162float(k_h_ptr[1]);
    k_f[2] = __bfloat162float(k_h_ptr[2]);
    k_f[3] = __bfloat162float(k_h_ptr[3]);

    // Precompute dot product of q and k for this block
    float qk_local = q_f[0] * k_f[0] + q_f[1] * k_f[1] + q_f[2] * k_f[2] + q_f[3] * k_f[3];
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);

    const float* state_base = state != nullptr ? state + b_idx * NUM_V_HEADS * V * K + h_idx * V * K : nullptr;
    float* new_state_base = new_state + b_idx * NUM_V_HEADS * V * K + h_idx * V * K;
    const __nv_bfloat16* v_ptr = v + b_idx * NUM_V_HEADS * V + h_idx * V;
    __nv_bfloat16* out_ptr = output + b_idx * NUM_V_HEADS * V + h_idx * V;

    int v_idx = v_group_idx * blockDim.y + ty;
    if (v_idx < V) {
        // Coalesced 128-byte read of the state tensor using float4
        float4 st = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (state_base != nullptr) {
            st = reinterpret_cast<const float4*>(state_base + v_idx * K)[tx];
        }

        // Compute local dot products for qh and kh
        float qh_local = q_f[0] * st.x + q_f[1] * st.y + q_f[2] * st.z + q_f[3] * st.w;
        float kh_local = k_f[0] * st.x + k_f[1] * st.y + k_f[2] * st.z + k_f[3] * st.w;

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
            kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        }
        float qh_v = __shfl_sync(0xffffffff, qh_local, 0);
        float kh_v = __shfl_sync(0xffffffff, kh_local, 0);

        // Load the single element from v_h
        float v_val = 0.0f;
        if (tx == 0) {
            v_val = __bfloat162float(v_ptr[v_idx]);
        }
        v_val = __shfl_sync(0xffffffff, v_val, 0);

        // Compute exactly equivalent decoupled update dynamics
        float old_v_v = g * kh_v;
        float dv_v = beta * (v_val - old_v_v);
        float out_v = scale * (g * qh_v + dv_v * qk_dot);

        // Store to global output array
        if (tx == 0) {
            out_ptr[v_idx] = __float2bfloat16(out_v);
        }

        // Update local state registers
        st.x = g * st.x + dv_v * k_f[0];
        st.y = g * st.y + dv_v * k_f[1];
        st.z = g * st.z + dv_v * k_f[2];
        st.w = g * st.w + dv_v * k_f[3];

        // Coalesced 128-byte write back to state tensor
        reinterpret_cast<float4*>(new_state_base + v_idx * K)[tx] = st;
    }
}

template <int K>
__global__ void __launch_bounds__(128) gdn_decode_kernel_semi(
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
    int B, int num_v_heads, int num_q_heads, int V
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int v_group_idx = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sh_g;
    __shared__ float sh_beta;

    // Compute head-level scalars exactly once per block
    if (tx == 0 && ty == 0) {
        float a_val = __bfloat162float(a[b_idx * num_v_heads + h_idx]);
        float b_val = __bfloat162float(b[b_idx * num_v_heads + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : __logf(1.0f + __expf(x));
        sh_g = __expf(-__expf(A_log_val) * sp);
        sh_beta = 1.0f / (1.0f + __expf(-b_val));
    }
    __syncthreads();

    float g = sh_g;
    float beta = sh_beta;

    int qk_h_idx = h_idx * num_q_heads / num_v_heads;
    const __nv_bfloat16* q_ptr = q + b_idx * num_q_heads * K + qk_h_idx * K;
    const __nv_bfloat16* k_ptr = k + b_idx * num_q_heads * K + qk_h_idx * K;

    constexpr int K_vecs = K / 4;
    constexpr int VECS_PER_THREAD = (K_vecs + 31) / 32;

    float q_f[VECS_PER_THREAD][4];
    float k_f[VECS_PER_THREAD][4];

    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        q_f[i][0] = 0.0f; q_f[i][1] = 0.0f; q_f[i][2] = 0.0f; q_f[i][3] = 0.0f;
        k_f[i][0] = 0.0f; k_f[i][1] = 0.0f; k_f[i][2] = 0.0f; k_f[i][3] = 0.0f;
    }

    float qk_local = 0.0f;

    // Vectorized load of q and k into registers
    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        int vec_idx = i * 32 + tx;
        if (vec_idx < K_vecs) {
            float2 q_vec = reinterpret_cast<const float2*>(q_ptr)[vec_idx];
            __nv_bfloat16* q_h_ptr = reinterpret_cast<__nv_bfloat16*>(&q_vec);
            q_f[i][0] = __bfloat162float(q_h_ptr[0]);
            q_f[i][1] = __bfloat162float(q_h_ptr[1]);
            q_f[i][2] = __bfloat162float(q_h_ptr[2]);
            q_f[i][3] = __bfloat162float(q_h_ptr[3]);

            float2 k_vec = reinterpret_cast<const float2*>(k_ptr)[vec_idx];
            __nv_bfloat16* k_h_ptr = reinterpret_cast<__nv_bfloat16*>(&k_vec);
            k_f[i][0] = __bfloat162float(k_h_ptr[0]);
            k_f[i][1] = __bfloat162float(k_h_ptr[1]);
            k_f[i][2] = __bfloat162float(k_h_ptr[2]);
            k_f[i][3] = __bfloat162float(k_h_ptr[3]);

            qk_local += q_f[i][0] * k_f[i][0] + q_f[i][1] * k_f[i][1] + q_f[i][2] * k_f[i][2] + q_f[i][3] * k_f[i][3];
        }
    }

    // Fast intra-warp reduction for the dot product
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
    
    // Each warp processes exactly one v_idx.
    if (v_idx < V) {
        float qh_local = 0.0f;
        float kh_local = 0.0f;

        float4 st[VECS_PER_THREAD];

        // Coalesced reads of the state tensor using float4
        #pragma unroll
        for (int i = 0; i < VECS_PER_THREAD; ++i) {
            st[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            int vec_idx = i * 32 + tx;
            if (vec_idx < K_vecs && state_base != nullptr) {
                st[i] = reinterpret_cast<const float4*>(state_base + v_idx * K)[vec_idx];
            }

            qh_local += q_f[i][0] * st[i].x + q_f[i][1] * st[i].y + q_f[i][2] * st[i].z + q_f[i][3] * st[i].w;
            kh_local += k_f[i][0] * st[i].x + k_f[i][1] * st[i].y + k_f[i][2] * st[i].z + k_f[i][3] * st[i].w;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
            kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        }
        float qh_v = __shfl_sync(0xffffffff, qh_local, 0);
        float kh_v = __shfl_sync(0xffffffff, kh_local, 0);

        // Load the single element from v
        float v_val = 0.0f;
        if (tx == 0) {
            v_val = __bfloat162float(v_ptr[v_idx]);
        }
        v_val = __shfl_sync(0xffffffff, v_val, 0);

        // Compute exactly equivalent decoupled update dynamics
        float old_v_v = g * kh_v;
        float dv_v = beta * (v_val - old_v_v);
        float out_v = scale * (g * qh_v + dv_v * qk_dot);

        // Store to global output array
        if (tx == 0) {
            out_ptr[v_idx] = __float2bfloat16(out_v);
        }

        // Update local state registers and perform coalesced writes back to state
        #pragma unroll
        for (int i = 0; i < VECS_PER_THREAD; ++i) {
            int vec_idx = i * 32 + tx;
            if (vec_idx < K_vecs) {
                st[i].x = g * st[i].x + dv_v * k_f[i][0];
                st[i].y = g * st[i].y + dv_v * k_f[i][1];
                st[i].z = g * st[i].z + dv_v * k_f[i][2];
                st[i].w = g * st[i].w + dv_v * k_f[i][3];

                reinterpret_cast<float4*>(new_state_base + v_idx * K)[vec_idx] = st[i];
            }
        }
    }
}

// Generic fallback kernel for arbitrary constraints where K % 4 != 0
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
        float sp = x > 20.0f ? x : __logf(1.0f + __expf(x));
        sh_g = __expf(-__expf(A_log_val) * sp);
        sh_beta = 1.0f / (1.0f + __expf(-b_val));
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

// C++ Entry Point
std::tuple<torch::Tensor, torch::Tensor> gdn_forward(
    torch::Tensor q,       // [batch_size, 1, num_q_heads, K]     bfloat16
    torch::Tensor k,       // [batch_size, 1, num_q_heads, K]     bfloat16
    torch::Tensor v,       // [batch_size, 1, num_v_heads, V]     bfloat16
    torch::Tensor state,   // [batch_size, num_v_heads, V, K]     float32  (k-last layout)
    torch::Tensor A_log,   // [num_v_heads]                       float32
    torch::Tensor a,       // [batch_size, 1, num_v_heads]        bfloat16
    torch::Tensor dt_bias, // [num_v_heads]                       float32
    torch::Tensor b,       // [batch_size, 1, num_v_heads]        bfloat16
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

    // Ensure contiguous memory for safe casting and offset logic
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

    if (K == 128 && num_v_heads == 8 && num_q_heads == 4 && V == 128) {
        dim3 grid(B, 8, 128 / 4);
        gdn_decode_kernel_fast<8, 4, 128, 128><<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
            state_ptr,
            A_log.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
            dt_bias.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
            scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            new_state.data_ptr<float>()
        );
    } else {
        dim3 grid(B, num_v_heads, (V + 3) / 4);
        if (K == 64) {
            gdn_decode_kernel_semi<64><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
                state_ptr,
                A_log.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
                dt_bias.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
                scale,
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                new_state.data_ptr<float>(),
                B, num_v_heads, num_q_heads, V
            );
        } else if (K == 128) {
            gdn_decode_kernel_semi<128><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
                state_ptr,
                A_log.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
                dt_bias.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
                scale,
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                new_state.data_ptr<float>(),
                B, num_v_heads, num_q_heads, V
            );
        } else if (K == 256) {
            gdn_decode_kernel_semi<256><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
                state_ptr,
                A_log.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
                dt_bias.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
                scale,
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                new_state.data_ptr<float>(),
                B, num_v_heads, num_q_heads, V
            );
        } else {
            gdn_decode_kernel_generic<<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
                state_ptr,
                A_log.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
                dt_bias.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
                scale,
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                new_state.data_ptr<float>(),
                B, num_v_heads, num_q_heads, V, K
            );
        }
    }

    return std::make_tuple(output, new_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_forward", &gdn_forward, "GDN Forward");
}
