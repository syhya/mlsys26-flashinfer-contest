#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void __launch_bounds__(512) gdn_decode_kernel_splitK(
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
    int tx = threadIdx.x; // 0..127
    int ty = threadIdx.y; // 0..3

    int warp_id = tx / 32;
    int lane_id = tx % 32;

    __shared__ float sh_q[128];
    __shared__ float sh_k[128];
    __shared__ float sh_g;
    __shared__ float sh_beta;

    if (ty == 0) {
        int qk_h_idx = h_idx / 2;
        sh_q[tx] = __bfloat162float(q[b_idx * 4 * 128 + qk_h_idx * 128 + tx]);
        sh_k[tx] = __bfloat162float(k[b_idx * 4 * 128 + qk_h_idx * 128 + tx]);

        if (tx == 0) {
            float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
            float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
            float dt_bias_val = dt_bias[h_idx];
            float A_log_val = A_log[h_idx];

            float x = a_val + dt_bias_val;
            float sp = x > 20.0f ? x : logf(1.0f + expf(x));
            sh_g = expf(-expf(A_log_val) * sp);
            sh_beta = 1.0f / (1.0f + expf(-b_val));
        }
    }
    __syncthreads();

    float q_val = sh_q[tx];
    float k_val = sh_k[tx];
    float g = sh_g;
    float beta = sh_beta;

    __shared__ float sh_qk[4];
    if (ty == 0) {
        float qk_local = q_val * k_val;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
        }
        if (lane_id == 0) {
            sh_qk[warp_id] = qk_local;
        }
    }
    __syncthreads();

    float qk_dot = 0.0f;
    if (ty == 0 && tx == 0) {
        qk_dot = sh_qk[0] + sh_qk[1] + sh_qk[2] + sh_qk[3];
        sh_qk[0] = qk_dot;
    }
    __syncthreads();
    qk_dot = sh_qk[0];

    int v_idx = v_group_idx * 4 + ty;
    if (v_idx < 128) {
        const float* state_base = state != nullptr ? state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 : nullptr;
        float st_val = 0.0f;
        if (state_base != nullptr) {
            st_val = state_base[v_idx * 128 + tx];
        }

        float qh_local = q_val * st_val;
        float kh_local = k_val * st_val;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
            kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        }

        __shared__ float sh_qh[4][4];
        __shared__ float sh_kh[4][4];

        if (lane_id == 0) {
            sh_qh[ty][warp_id] = qh_local;
            sh_kh[ty][warp_id] = kh_local;
        }
        __syncthreads();

        float qh_v, kh_v;
        if (tx == 0) {
            qh_v = sh_qh[ty][0] + sh_qh[ty][1] + sh_qh[ty][2] + sh_qh[ty][3];
            kh_v = sh_kh[ty][0] + sh_kh[ty][1] + sh_kh[ty][2] + sh_kh[ty][3];
            
            sh_qh[ty][0] = qh_v;
            sh_kh[ty][0] = kh_v;
        }
        __syncthreads();

        qh_v = sh_qh[ty][0];
        kh_v = sh_kh[ty][0];

        float dv_v = 0.0f;
        if (tx == 0) {
            float v_val = __bfloat162float(v[b_idx * 8 * 128 + h_idx * 128 + v_idx]);
            float old_v_v = g * kh_v;
            dv_v = beta * (v_val - old_v_v);
            float out_v = scale * (g * qh_v + dv_v * qk_dot);
            output[b_idx * 8 * 128 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
            sh_qh[ty][1] = dv_v;
        }
        __syncthreads();
        dv_v = sh_qh[ty][1];

        st_val = g * st_val + dv_v * k_val;

        float* new_state_base = new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128;
        new_state_base[v_idx * 128 + tx] = st_val;
    }
}

__global__ void __launch_bounds__(128) gdn_decode_kernel_1warp(
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

    float a_val, b_val, dt_bias_val, A_log_val;
    if (tx == 0) {
        a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        dt_bias_val = dt_bias[h_idx];
        A_log_val = A_log[h_idx];
    }
    a_val = __shfl_sync(0xffffffff, a_val, 0);
    b_val = __shfl_sync(0xffffffff, b_val, 0);
    dt_bias_val = __shfl_sync(0xffffffff, dt_bias_val, 0);
    A_log_val = __shfl_sync(0xffffffff, A_log_val, 0);

    float x = a_val + dt_bias_val;
    float sp = x > 20.0f ? x : logf(1.0f + expf(x));
    float g = expf(-expf(A_log_val) * sp);
    float beta = 1.0f / (1.0f + expf(-b_val));

    int qk_h_idx = h_idx / 2;
    const __nv_bfloat16* q_ptr = q + b_idx * 4 * 128 + qk_h_idx * 128;
    const __nv_bfloat16* k_ptr = k + b_idx * 4 * 128 + qk_h_idx * 128;

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

    float qk_local = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        qk_local += q_f[i] * k_f[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);

    const float* state_base = state != nullptr ? state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 : nullptr;
    float* new_state_base = new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128;
    const __nv_bfloat16* v_ptr = v + b_idx * 8 * 128 + h_idx * 128;
    __nv_bfloat16* out_ptr = output + b_idx * 8 * 128 + h_idx * 128;

    int v_idx = v_group_idx * 4 + ty;
    if (v_idx < 128) {
        float4 st = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (state_base != nullptr) {
            st = reinterpret_cast<const float4*>(state_base + v_idx * 128)[tx];
        }

        float qh_local = q_f[0] * st.x + q_f[1] * st.y + q_f[2] * st.z + q_f[3] * st.w;
        float kh_local = k_f[0] * st.x + k_f[1] * st.y + k_f[2] * st.z + k_f[3] * st.w;

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

        st.x = g * st.x + dv_v * k_f[0];
        st.y = g * st.y + dv_v * k_f[1];
        st.z = g * st.z + dv_v * k_f[2];
        st.w = g * st.w + dv_v * k_f[3];

        reinterpret_cast<float4*>(new_state_base + v_idx * 128)[tx] = st;
    }
}

std::tuple<torch::Tensor, torch::Tensor> gdn_forward(
    torch::Tensor q,       // [batch_size, 1, 4, 128]     bfloat16
    torch::Tensor k,       // [batch_size, 1, 4, 128]     bfloat16
    torch::Tensor v,       // [batch_size, 1, 8, 128]     bfloat16
    torch::Tensor state,   // [batch_size, 8, 128, 128]   float32  (k-last layout: [B, HV, V, K])
    torch::Tensor A_log,   // [8]                          float32
    torch::Tensor a,       // [batch_size, 1, 8]           bfloat16
    torch::Tensor dt_bias, // [8]                          float32
    torch::Tensor b,       // [batch_size, 1, 8]           bfloat16
    float scale            // scalar
) {
    int B = q.size(0);
    int num_v_heads = 8;
    int K = 128;
    int V = 128;

    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(128.0f);
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

    if (B <= 8) {
        dim3 grid(B, num_v_heads, V / 4);
        dim3 block(128, 4);
        gdn_decode_kernel_splitK<<<grid, block>>>(
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
        dim3 grid(B, num_v_heads, V / 4);
        dim3 block(32, 4);
        gdn_decode_kernel_1warp<<<grid, block>>>(
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
    }

    return std::make_tuple(output, new_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_forward", &gdn_forward, "GDN Forward");
}
