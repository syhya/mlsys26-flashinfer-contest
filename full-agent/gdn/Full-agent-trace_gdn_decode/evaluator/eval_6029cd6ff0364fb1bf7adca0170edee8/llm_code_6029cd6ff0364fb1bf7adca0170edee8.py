#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void __launch_bounds__(128) gdn_decode_kernel(
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
    int v_idx = blockIdx.z;
    int tx = threadIdx.x;

    // 1. Calculate flat offsets
    int qk_h_idx = h_idx / 2;
    int state_offset = b_idx * 131072 + h_idx * 16384 + v_idx * 128 + tx;
    int q_offset = b_idx * 512 + qk_h_idx * 128 + tx;
    int k_offset = b_idx * 512 + qk_h_idx * 128 + tx;
    
    // 2. Initiate global memory loads
    float state_val = 0.0f;
    if (state != nullptr) {
        state_val = state[state_offset];
    }
    float q_val = __bfloat162float(q[q_offset]);
    float k_val = __bfloat162float(k[k_offset]);

    // 3. Compute scalars while waiting for global loads (Latency hiding)
    float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
    float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
    float dt_bias_val = dt_bias[h_idx];
    float A_log_val = A_log[h_idx];

    float x = a_val + dt_bias_val;
    float sp = x > 20.0f ? x : logf(1.0f + expf(x));
    float g = expf(-expf(A_log_val) * sp);
    float beta = 1.0f / (1.0f + expf(-b_val));
    float v_val = __bfloat162float(v[b_idx * 1024 + h_idx * 128 + v_idx]);

    // 4. Warp reductions for qk, qh, kh
    float qk_prod = q_val * k_val;
    float qh_prod = q_val * state_val;
    float kh_prod = k_val * state_val;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_prod += __shfl_down_sync(0xffffffff, qk_prod, offset);
        qh_prod += __shfl_down_sync(0xffffffff, qh_prod, offset);
        kh_prod += __shfl_down_sync(0xffffffff, kh_prod, offset);
    }

    // 5. Store warp sums to shared memory
    __shared__ float sh_qk[4];
    __shared__ float sh_qh[4];
    __shared__ float sh_kh[4];

    if ((tx % 32) == 0) {
        int warp_id = tx / 32;
        sh_qk[warp_id] = qk_prod;
        sh_qh[warp_id] = qh_prod;
        sh_kh[warp_id] = kh_prod;
    }

    __syncthreads(); // 1st sync

    // 6. Final reduction in warp 0
    float qk_dot = 0.0f, qh_v = 0.0f, kh_v = 0.0f;
    if (tx < 4) {
        qk_dot = sh_qk[tx];
        qh_v = sh_qh[tx];
        kh_v = sh_kh[tx];
    }
    
    #pragma unroll
    for (int offset = 2; offset > 0; offset /= 2) {
        qk_dot += __shfl_down_sync(0xffffffff, qk_dot, offset);
        qh_v += __shfl_down_sync(0xffffffff, qh_v, offset);
        kh_v += __shfl_down_sync(0xffffffff, kh_v, offset);
    }
    
    if (tx == 0) {
        sh_qk[0] = qk_dot;
        sh_qh[0] = qh_v;
        sh_kh[0] = kh_v;
    }
    
    __syncthreads(); // 2nd sync
    
    // 7. Load broadcasted values
    qk_dot = sh_qk[0];
    qh_v = sh_qh[0];
    kh_v = sh_kh[0];

    // 8. Compute final values
    float old_v_v = g * kh_v;
    float dv_v = beta * (v_val - old_v_v);

    if (tx == 0) {
        float out_v = scale * (g * qh_v + dv_v * qk_dot);
        output[b_idx * 1024 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
    }

    float new_state_val = g * state_val + dv_v * k_val;
    new_state[state_offset] = new_state_val;
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

    dim3 grid(B, num_v_heads, V);
    dim3 block(128);

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
