#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// CUDA Kernel for GDN Decode step
__global__ void __launch_bounds__(128) gdn_decode_kernel_k_split(
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
    int v_idx = blockIdx.z; // 0 to 127
    int tx = threadIdx.x;   // 0 to 127

    // GVA mapping: num_v_heads (8) -> num_q_heads (4)
    int qk_h_idx = h_idx / 2;

    const __nv_bfloat16* q_ptr = q + b_idx * 4 * 128 + qk_h_idx * 128;
    const __nv_bfloat16* k_ptr = k + b_idx * 4 * 128 + qk_h_idx * 128;
    const float* state_base = state != nullptr ? state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 : nullptr;
    float* new_state_base = new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128;

    // Load data
    float q_val = __bfloat162float(q_ptr[tx]);
    float k_val = __bfloat162float(k_ptr[tx]);
    float state_val = 0.0f;
    if (state_base != nullptr) {
        state_val = state_base[v_idx * 128 + tx];
    }

    // Multiply
    float qh_val = q_val * state_val;
    float kh_val = k_val * state_val;
    float qk_val = q_val * k_val;

    // Intra-warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qh_val += __shfl_down_sync(0xffffffff, qh_val, offset);
        kh_val += __shfl_down_sync(0xffffffff, kh_val, offset);
        qk_val += __shfl_down_sync(0xffffffff, qk_val, offset);
    }

    // Inter-warp reduction
    __shared__ float sh_qh[4];
    __shared__ float sh_kh[4];
    __shared__ float sh_qk[4];

    int warp_id = tx / 32;
    int lane_id = tx % 32;

    if (lane_id == 0) {
        sh_qh[warp_id] = qh_val;
        sh_kh[warp_id] = kh_val;
        sh_qk[warp_id] = qk_val;
    }
    __syncthreads();

    float qh_total = 0.0f;
    float kh_total = 0.0f;
    float qk_total = 0.0f;

    if (tx < 4) {
        qh_total = sh_qh[tx];
        kh_total = sh_kh[tx];
        qk_total = sh_qk[tx];
    }

    #pragma unroll
    for (int offset = 2; offset > 0; offset /= 2) {
        qh_total += __shfl_down_sync(0x0000000f, qh_total, offset);
        kh_total += __shfl_down_sync(0x0000000f, kh_total, offset);
        qk_total += __shfl_down_sync(0x0000000f, qk_total, offset);
    }

    __shared__ float sh_g;
    __shared__ float sh_dv_v;

    if (tx == 0) {
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : logf(1.0f + expf(x));
        float g = expf(-expf(A_log_val) * sp);
        float beta = 1.0f / (1.0f + expf(-b_val));

        float v_val = __bfloat162float(v[b_idx * 8 * 128 + h_idx * 128 + v_idx]);
        float old_v_v = g * kh_total;
        float dv_v = beta * (v_val - old_v_v);
        float out_v = scale * (g * qh_total + dv_v * qk_total);

        output[b_idx * 8 * 128 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);

        sh_g = g;
        sh_dv_v = dv_v;
    }
    __syncthreads();

    float g = sh_g;
    float dv_v = sh_dv_v;

    // Update state
    new_state_base[v_idx * 128 + tx] = g * state_val + dv_v * k_val;
}

// C++ Entry Point
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

    // Launch configuration: 128 threads per block (1 block per V-row)
    dim3 grid(B, num_v_heads, V);
    dim3 block(128);

    gdn_decode_kernel_k_split<<<grid, block>>>(
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
