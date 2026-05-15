#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void __launch_bounds__(32) gdn_decode_kernel(
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
    int V
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int v_idx = blockIdx.z;
    int tx = threadIdx.x;

    if (v_idx >= V) return;

    // Issue state load early
    long long state_offset = (long long)b_idx * 8 * V * 128 + (long long)h_idx * V * 128 + (long long)v_idx * 128;
    float4 st = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (state != nullptr) {
        st = reinterpret_cast<const float4*>(state + state_offset)[tx];
    }

    // Prepare q and k pointers
    int qk_h_idx = h_idx / 2;
    const __nv_bfloat16* q_ptr = q + b_idx * 4 * 128 + qk_h_idx * 128;
    const __nv_bfloat16* k_ptr = k + b_idx * 4 * 128 + qk_h_idx * 128;

    // Issue q and k loads
    float2 q_vec = reinterpret_cast<const float2*>(q_ptr)[tx];
    float2 k_vec = reinterpret_cast<const float2*>(k_ptr)[tx];

    // Compute head-level scalars while waiting for memory
    float g, beta;
    if (tx == 0) {
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : logf(1.0f + expf(x));
        g = expf(-expf(A_log_val) * sp);
        beta = 1.0f / (1.0f + expf(-b_val));
    }
    // Broadcast scalars to all threads in warp
    g = __shfl_sync(0xffffffff, g, 0);
    beta = __shfl_sync(0xffffffff, beta, 0);

    // Convert q and k to float
    const __nv_bfloat162* q_h_ptr2 = reinterpret_cast<const __nv_bfloat162*>(&q_vec);
    float2 q_f01 = __bfloat1622float2(q_h_ptr2[0]);
    float2 q_f23 = __bfloat1622float2(q_h_ptr2[1]);
    float q_f[4] = {q_f01.x, q_f01.y, q_f23.x, q_f23.y};

    const __nv_bfloat162* k_h_ptr2 = reinterpret_cast<const __nv_bfloat162*>(&k_vec);
    float2 k_f01 = __bfloat1622float2(k_h_ptr2[0]);
    float2 k_f23 = __bfloat1622float2(k_h_ptr2[1]);
    float k_f[4] = {k_f01.x, k_f01.y, k_f23.x, k_f23.y};

    // Compute qk_dot
    float qk_local = q_f[0] * k_f[0];
    qk_local = fmaf(q_f[1], k_f[1], qk_local);
    qk_local = fmaf(q_f[2], k_f[2], qk_local);
    qk_local = fmaf(q_f[3], k_f[3], qk_local);

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);

    // Precompute combined scalars
    float beta_g = beta * g;
    float scale_g = scale * g;
    float scale_qk_dot = scale * qk_dot;

    // Use the loaded state `st`
    float qh_local = q_f[0] * st.x;
    qh_local = fmaf(q_f[1], st.y, qh_local);
    qh_local = fmaf(q_f[2], st.z, qh_local);
    qh_local = fmaf(q_f[3], st.w, qh_local);

    float kh_local = k_f[0] * st.x;
    kh_local = fmaf(k_f[1], st.y, kh_local);
    kh_local = fmaf(k_f[2], st.z, kh_local);
    kh_local = fmaf(k_f[3], st.w, kh_local);

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
        kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
    }
    float qh_v = __shfl_sync(0xffffffff, qh_local, 0);
    float kh_v = __shfl_sync(0xffffffff, kh_local, 0);

    // Load v_val
    float v_val = 0.0f;
    if (tx == 0) {
        const __nv_bfloat16* v_ptr = v + b_idx * 8 * V + h_idx * V;
        v_val = __bfloat162float(v_ptr[v_idx]);
    }
    v_val = __shfl_sync(0xffffffff, v_val, 0);

    // Final dynamics
    float dv_v = fmaf(-beta_g, kh_v, beta * v_val);
    float out_v = fmaf(scale_qk_dot, dv_v, scale_g * qh_v);

    if (tx == 0) {
        __nv_bfloat16* out_ptr = output + b_idx * 8 * V + h_idx * V;
        out_ptr[v_idx] = __float2bfloat16(out_v);
    }

    // Update state registers
    st.x = fmaf(dv_v, k_f[0], g * st.x);
    st.y = fmaf(dv_v, k_f[1], g * st.y);
    st.z = fmaf(dv_v, k_f[2], g * st.z);
    st.w = fmaf(dv_v, k_f[3], g * st.w);

    // Write back state
    reinterpret_cast<float4*>(new_state + state_offset)[tx] = st;
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
    int V = v.size(3);
    int K = 128;

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

    dim3 grid(B, num_v_heads, V);
    dim3 block(32);

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
        new_state.data_ptr<float>(),
        V
    );

    return std::make_tuple(output, new_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_forward", &gdn_forward, "GDN Forward");
}
