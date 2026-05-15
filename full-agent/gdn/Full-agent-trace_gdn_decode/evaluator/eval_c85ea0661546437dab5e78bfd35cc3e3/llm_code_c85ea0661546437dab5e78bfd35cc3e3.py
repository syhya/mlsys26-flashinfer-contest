#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// CUDA Kernel for GDN Decode step
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
    int tx = threadIdx.x; // 0 to 127

    __shared__ float sh_qh[4];
    __shared__ float sh_kh[4];
    __shared__ float sh_qk[4];
    __shared__ float sh_g;
    __shared__ float sh_beta;
    __shared__ float sh_v_val;
    __shared__ float final_qh;
    __shared__ float final_kh;
    __shared__ float final_qk;

    // Thread 0 computes block-wide scalars
    if (tx == 0) {
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : __logf(1.0f + __expf(x));
        sh_g = __expf(-__expf(A_log_val) * sp);
        sh_beta = 1.0f / (1.0f + __expf(-b_val));
        
        sh_v_val = __bfloat162float(v[b_idx * 8 * 128 + h_idx * 128 + v_idx]);
    }

    // GVA mapping: num_v_heads (8) -> num_q_heads (4)
    int qk_h_idx = h_idx / 2;
    __nv_bfloat16 q_val_bf = q[b_idx * 4 * 128 + qk_h_idx * 128 + tx];
    __nv_bfloat16 k_val_bf = k[b_idx * 4 * 128 + qk_h_idx * 128 + tx];
    float q_val = __bfloat162float(q_val_bf);
    float k_val = __bfloat162float(k_val_bf);

    float state_val = 0.0f;
    if (state != nullptr) {
        state_val = state[b_idx * 8 * 128 * 128 + h_idx * 128 * 128 + v_idx * 128 + tx];
    }

    // Compute local products
    float qh_local = q_val * state_val;
    float kh_local = k_val * state_val;
    float qk_local = q_val * k_val;

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
        kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }

    int warp_id = tx / 32;
    int lane_id = tx % 32;

    if (lane_id == 0) {
        sh_qh[warp_id] = qh_local;
        sh_kh[warp_id] = kh_local;
        sh_qk[warp_id] = qk_local;
    }

    __syncthreads();

    // Warp 0 does the final reduction
    if (warp_id == 0) {
        float qh_v = (lane_id < 4) ? sh_qh[lane_id] : 0.0f;
        float kh_v = (lane_id < 4) ? sh_kh[lane_id] : 0.0f;
        float qk_v = (lane_id < 4) ? sh_qk[lane_id] : 0.0f;

        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            qh_v += __shfl_down_sync(0xffffffff, qh_v, offset);
            kh_v += __shfl_down_sync(0xffffffff, kh_v, offset);
            qk_v += __shfl_down_sync(0xffffffff, qk_v, offset);
        }

        if (lane_id == 0) {
            final_qh = qh_v;
            final_kh = kh_v;
            final_qk = qk_v;
        }
    }

    __syncthreads();

    // Fetch broadcasted values
    float g = sh_g;
    float beta = sh_beta;
    float v_val = sh_v_val;
    float qh_sum = final_qh;
    float kh_sum = final_kh;
    float qk_sum = final_qk;

    // Compute state update dynamics
    float old_v_v = g * kh_sum;
    float dv_v = beta * (v_val - old_v_v);
    
    // Store output (only tx == 0)
    if (tx == 0) {
        float out_v = scale * (g * qh_sum + dv_v * qk_sum);
        output[b_idx * 8 * 128 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
    }

    // Update state and write back
    float new_state_val = g * state_val + dv_v * k_val;
    new_state[b_idx * 8 * 128 * 128 + h_idx * 128 * 128 + v_idx * 128 + tx] = new_state_val;
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

    // Launch configuration: 128 threads per block (4 warps)
    // Grid: (B, 8, 128) -> Total blocks: B * 1024
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
