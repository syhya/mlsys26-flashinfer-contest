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
    int tx = threadIdx.x;
    int warp_id = tx / 32;
    int lane_id = tx % 32;

    __shared__ float shm_qk_arr[4];
    __shared__ float shm_qh_arr[4];
    __shared__ float shm_kh_arr[4];
    __shared__ float shm_g;
    __shared__ float shm_beta;
    __shared__ float shm_v_val;

    // 1. Initiate state load to overlap with math
    float state_val = 0.0f;
    if (state != nullptr) {
        state_val = state[b_idx * 131072 + h_idx * 16384 + v_idx * 128 + tx];
    }

    // 2. Load q and k
    int qk_h_idx = h_idx / 2;
    int qk_offset = b_idx * 512 + qk_h_idx * 128 + tx;
    __nv_bfloat16 q_val = q[qk_offset];
    __nv_bfloat16 k_val = k[qk_offset];

    float q_f = __bfloat162float(q_val);
    float k_f = __bfloat162float(k_val);

    // 3. Compute qk dot product (warp reduction)
    float qk_local = q_f * k_f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    if (lane_id == 0) {
        shm_qk_arr[warp_id] = qk_local;
    }

    // 4. Thread 0 computes scalar values
    if (tx == 0) {
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : logf(1.0f + expf(x));
        shm_g = expf(-expf(A_log_val) * sp);
        shm_beta = 1.0f / (1.0f + expf(-b_val));
    }

    // 5. Barrier 1: ensure shared memory is visible
    __syncthreads();

    float g = shm_g;
    float beta = shm_beta;
    float qk_dot = shm_qk_arr[0] + shm_qk_arr[1] + shm_qk_arr[2] + shm_qk_arr[3];

    // 6. Compute qh and kh using state_val
    float qh_local = q_f * state_val;
    float kh_local = k_f * state_val;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
        kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
    }

    if (lane_id == 0) {
        shm_qh_arr[warp_id] = qh_local;
        shm_kh_arr[warp_id] = kh_local;
    }
    
    if (tx == 0) {
        shm_v_val = __bfloat162float(v[b_idx * 1024 + h_idx * 128 + v_idx]);
    }

    // 7. Barrier 2: ensure shared memory is visible
    __syncthreads();

    float qh_v = shm_qh_arr[0] + shm_qh_arr[1] + shm_qh_arr[2] + shm_qh_arr[3];
    float kh_v = shm_kh_arr[0] + shm_kh_arr[1] + shm_kh_arr[2] + shm_kh_arr[3];
    float v_val = shm_v_val;

    // 9. State update math
    float old_v_v = g * kh_v;
    float dv_v = beta * (v_val - old_v_v);

    // 10. Thread 0 computes and writes output
    if (tx == 0) {
        float out_v = scale * (g * qh_v + dv_v * qk_dot);
        output[b_idx * 1024 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
    }

    // 11. All threads compute and write new state
    float new_state_val = g * state_val + dv_v * k_f;
    new_state[b_idx * 131072 + h_idx * 16384 + v_idx * 128 + tx] = new_state_val;
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

    // Ensure contiguous memory
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

    // Launch configuration: 128 threads per block (1 thread per K element)
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
