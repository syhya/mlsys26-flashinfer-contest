#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void __launch_bounds__(64) gdn_decode_kernel(
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

    __shared__ float smem_g;
    __shared__ float smem_beta;
    __shared__ float smem_qk_dot;

    int qk_h_idx = h_idx / 2;
    const __nv_bfloat16* q_ptr = q + b_idx * 4 * 128 + qk_h_idx * 128;
    const __nv_bfloat16* k_ptr = k + b_idx * 4 * 128 + qk_h_idx * 128;

    float2 q_vec = reinterpret_cast<const float2*>(q_ptr)[tx];
    float2 k_vec = reinterpret_cast<const float2*>(k_ptr)[tx];
    
    const __nv_bfloat162* q_bf2 = reinterpret_cast<const __nv_bfloat162*>(&q_vec);
    const __nv_bfloat162* k_bf2 = reinterpret_cast<const __nv_bfloat162*>(&k_vec);

    float q0 = __bfloat162float(q_bf2[0].x);
    float q1 = __bfloat162float(q_bf2[0].y);
    float q2 = __bfloat162float(q_bf2[1].x);
    float q3 = __bfloat162float(q_bf2[1].y);

    float k0 = __bfloat162float(k_bf2[0].x);
    float k1 = __bfloat162float(k_bf2[0].y);
    float k2 = __bfloat162float(k_bf2[1].x);
    float k3 = __bfloat162float(k_bf2[1].y);

    if (ty == 0) {
        if (tx == 0) {
            float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
            float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
            float dt_bias_val = dt_bias[h_idx];
            float A_log_val = A_log[h_idx];

            float x = a_val + dt_bias_val;
            float sp = x > 20.0f ? x : logf(1.0f + expf(x));
            smem_g = expf(-expf(A_log_val) * sp);
            smem_beta = 1.0f / (1.0f + expf(-b_val));
        }

        float qk_local = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
        }
        if (tx == 0) {
            smem_qk_dot = qk_local;
        }
    }
    __syncthreads();

    float g_val = smem_g;
    float beta_val = smem_beta;
    float qk_dot_val = smem_qk_dot;

    const float* state_base = state != nullptr ? state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 : nullptr;
    float* new_state_base = new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128;
    const __nv_bfloat16* v_ptr = v + b_idx * 8 * 128 + h_idx * 128;
    __nv_bfloat16* out_ptr = output + b_idx * 8 * 128 + h_idx * 128;

    int v_idx = v_group_idx * 2 + ty;
    if (v_idx < 128) {
        float4 st = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (state_base != nullptr) {
            st = reinterpret_cast<const float4*>(state_base + v_idx * 128)[tx];
        }

        float qh_local = q0 * st.x + q1 * st.y + q2 * st.z + q3 * st.w;
        float kh_local = k0 * st.x + k1 * st.y + k2 * st.z + k3 * st.w;

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

        float old_v_v = g_val * kh_v;
        float dv_v = beta_val * (v_val - old_v_v);
        float out_v = scale * (g_val * qh_v + dv_v * qk_dot_val);

        if (tx == 0) {
            out_ptr[v_idx] = __float2bfloat16(out_v);
        }

        st.x = fmaf(g_val, st.x, dv_v * k0);
        st.y = fmaf(g_val, st.y, dv_v * k1);
        st.z = fmaf(g_val, st.z, dv_v * k2);
        st.w = fmaf(g_val, st.w, dv_v * k3);

        reinterpret_cast<float4*>(new_state_base + v_idx * 128)[tx] = st;
    }
}

std::tuple<torch::Tensor, torch::Tensor> gdn_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    float scale
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

    dim3 grid(B, num_v_heads, V / 2);
    dim3 block(32, 2);

    gdn_decode_kernel<<<grid, block>>>(
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

    return std::make_tuple(output, new_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_forward", &gdn_forward, "GDN Forward");
}
