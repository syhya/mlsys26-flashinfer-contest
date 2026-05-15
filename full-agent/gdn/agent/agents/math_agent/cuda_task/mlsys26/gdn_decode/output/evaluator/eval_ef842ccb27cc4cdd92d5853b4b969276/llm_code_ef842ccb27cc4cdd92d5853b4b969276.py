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
    int v_group_idx = blockIdx.z; // V/2 blocks
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float shared_g;
    __shared__ float shared_beta;
    __shared__ float shared_qk_dot;
    __shared__ float4 shared_k[32];
    __shared__ float4 shared_q[32];

    int qk_h_idx = h_idx / 2;

    if (ty == 0) {
        if (tx == 0) {
            float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
            float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
            float dt_bias_val = dt_bias[h_idx];
            float A_log_val = A_log[h_idx];

            float x = a_val + dt_bias_val;
            float sp = x > 20.0f ? x : logf(1.0f + expf(x));
            shared_g = expf(-expf(A_log_val) * sp);
            shared_beta = 1.0f / (1.0f + expf(-b_val));
        }

        const __nv_bfloat16* q_ptr = q + b_idx * 4 * 128 + qk_h_idx * 128;
        const __nv_bfloat16* k_ptr = k + b_idx * 4 * 128 + qk_h_idx * 128;

        union {
            float2 f2;
            __nv_bfloat16 bf16[4];
        } q_load, k_load;

        q_load.f2 = reinterpret_cast<const float2*>(q_ptr)[tx];
        k_load.f2 = reinterpret_cast<const float2*>(k_ptr)[tx];

        float q_f[4];
        float k_f[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            q_f[i] = __bfloat162float(q_load.bf16[i]);
            k_f[i] = __bfloat162float(k_load.bf16[i]);
        }

        shared_q[tx] = make_float4(q_f[0], q_f[1], q_f[2], q_f[3]);
        shared_k[tx] = make_float4(k_f[0], k_f[1], k_f[2], k_f[3]);

        float qk_local = q_f[0] * k_f[0];
        qk_local = fmaf(q_f[1], k_f[1], qk_local);
        qk_local = fmaf(q_f[2], k_f[2], qk_local);
        qk_local = fmaf(q_f[3], k_f[3], qk_local);
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
        }
        if (tx == 0) {
            shared_qk_dot = qk_local;
        }
    }

    __syncthreads();

    float g = shared_g;
    float beta = shared_beta;
    float qk_dot = shared_qk_dot;
    float4 q_f4 = shared_q[tx];
    float4 k_f4 = shared_k[tx];

    int v_idx = v_group_idx * 2 + ty; // V/2 groups, 2 threads in y
    if (v_idx < 128) {
        const float* state_base = state != nullptr ? state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 : nullptr;
        float* new_state_base = new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128;
        const __nv_bfloat16* v_ptr = v + b_idx * 8 * 128 + h_idx * 128;
        __nv_bfloat16* out_ptr = output + b_idx * 8 * 128 + h_idx * 128;

        float4 st = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (state_base != nullptr) {
            st = reinterpret_cast<const float4*>(state_base + v_idx * 128)[tx];
        }

        float qh_local = q_f4.x * st.x;
        qh_local = fmaf(q_f4.y, st.y, qh_local);
        qh_local = fmaf(q_f4.z, st.z, qh_local);
        qh_local = fmaf(q_f4.w, st.w, qh_local);

        float kh_local = k_f4.x * st.x;
        kh_local = fmaf(k_f4.y, st.y, kh_local);
        kh_local = fmaf(k_f4.z, st.z, kh_local);
        kh_local = fmaf(k_f4.w, st.w, kh_local);

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

        float dv_v = beta * fmaf(-g, kh_v, v_val);
        float out_v = scale * fmaf(dv_v, qk_dot, g * qh_v);

        if (tx == 0) {
            out_ptr[v_idx] = __float2bfloat16(out_v);
        }

        st.x = fmaf(g, st.x, dv_v * k_f4.x);
        st.y = fmaf(g, st.y, dv_v * k_f4.y);
        st.z = fmaf(g, st.z, dv_v * k_f4.z);
        st.w = fmaf(g, st.w, dv_v * k_f4.w);

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