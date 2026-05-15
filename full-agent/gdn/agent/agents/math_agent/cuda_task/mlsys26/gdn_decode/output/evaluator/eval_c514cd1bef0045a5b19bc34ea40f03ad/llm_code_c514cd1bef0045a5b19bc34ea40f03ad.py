#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void __launch_bounds__(128) gdn_decode_kernel_vec(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_tensor,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int ty = threadIdx.y; // 0..3 (warp index)
    int tx = threadIdx.x; // 0..31 (lane index)
    int v_idx = blockIdx.z * 4 + ty;

    if (v_idx >= 128) return;

    int qk_h_idx = h_idx / 2;
    const __nv_bfloat16* q_ptr = q + b_idx * 4 * 128 + qk_h_idx * 128;
    const __nv_bfloat16* k_ptr = k + b_idx * 4 * 128 + qk_h_idx * 128;

    float q0, q1, q2, q3;
    float k0, k1, k2, k3;
    {
        const float2* q_f2_ptr = reinterpret_cast<const float2*>(q_ptr);
        const float2* k_f2_ptr = reinterpret_cast<const float2*>(k_ptr);
        
        float2 q_vec = q_f2_ptr[tx];
        __nv_bfloat162* q_bf2 = reinterpret_cast<__nv_bfloat162*>(&q_vec);
        float2 q_01 = __bfloat1622float2(q_bf2[0]);
        float2 q_23 = __bfloat1622float2(q_bf2[1]);
        q0 = q_01.x; q1 = q_01.y; q2 = q_23.x; q3 = q_23.y;

        float2 k_vec = k_f2_ptr[tx];
        __nv_bfloat162* k_bf2 = reinterpret_cast<__nv_bfloat162*>(&k_vec);
        float2 k_01 = __bfloat1622float2(k_bf2[0]);
        float2 k_23 = __bfloat1622float2(k_bf2[1]);
        k0 = k_01.x; k1 = k_01.y; k2 = k_23.x; k3 = k_23.y;
    }

    float qk_local = q0*k0 + q1*k1 + q2*k2 + q3*k3;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);

    float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
    float b_val = __bfloat162float(b_tensor[b_idx * 8 + h_idx]);
    float dt_bias_val = dt_bias[h_idx];
    float A_log_val = A_log[h_idx];

    float x = a_val + dt_bias_val;
    float sp = x > 20.0f ? x : __logf(1.0f + __expf(x));
    float g = __expf(-__expf(A_log_val) * sp);
    float beta = 1.0f / (1.0f + __expf(-b_val));

    float v_val = __bfloat162float(v[b_idx * 8 * 128 + h_idx * 128 + v_idx]);

    const float* state_base = state != nullptr ? state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 + v_idx * 128 : nullptr;
    float* new_state_base = new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 + v_idx * 128;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    if (state_base != nullptr) {
        const float4* s_ptr = reinterpret_cast<const float4*>(state_base);
        float4 s_vec = s_ptr[tx];
        s0 = s_vec.x; s1 = s_vec.y; s2 = s_vec.z; s3 = s_vec.w;
    }

    float qh_local = q0*s0 + q1*s1 + q2*s2 + q3*s3;
    float kh_local = k0*s0 + k1*s1 + k2*s2 + k3*s3;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
        kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
    }
    
    float qh_v = __shfl_sync(0xffffffff, qh_local, 0);
    float kh_v = __shfl_sync(0xffffffff, kh_local, 0);

    float old_v_v = g * kh_v;
    float dv_v = beta * (v_val - old_v_v);

    if (tx == 0) {
        float out_v = scale * (g * qh_v + dv_v * qk_dot);
        output[b_idx * 8 * 128 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
    }

    s0 = g * s0 + dv_v * k0;
    s1 = g * s1 + dv_v * k1;
    s2 = g * s2 + dv_v * k2;
    s3 = g * s3 + dv_v * k3;

    float4* new_s_ptr = reinterpret_cast<float4*>(new_state_base);
    new_s_ptr[tx] = make_float4(s0, s1, s2, s3);
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

    dim3 grid(B, num_v_heads, V / 4);
    dim3 block(32, 4);

    gdn_decode_kernel_vec<<<grid, block>>>(
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
