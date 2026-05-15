#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void __launch_bounds__(128) gdn_decode_kernel_128(
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

    __shared__ float shm_qk[4];
    __shared__ float shm_g;
    __shared__ float shm_beta;

    // 1. Initiate state load
    float state_val = 0.0f;
    if (state != nullptr) {
        int state_offset = b_idx * 131072 + h_idx * 16384 + v_idx * 128 + tx;
        state_val = state[state_offset];
    }

    // 2. Load q and k
    int qk_h_idx = h_idx / 2;
    int qk_offset = b_idx * 512 + qk_h_idx * 128 + tx;
    float q_val = __bfloat162float(q[qk_offset]);
    float k_val = __bfloat162float(k[qk_offset]);

    // 3. Thread 0 computes g and beta
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

    // 4. Compute qk_local
    float qk_local = q_val * k_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    
    int warp_id = tx >> 5;
    int lane_id = tx & 31;
    if (lane_id == 0) shm_qk[warp_id] = qk_local;

    __syncthreads();

    float g = shm_g;
    float beta = shm_beta;
    float qk_dot = shm_qk[0] + shm_qk[1] + shm_qk[2] + shm_qk[3];

    // 5. Use state_val
    float qh_local = q_val * state_val;
    float kh_local = k_val * state_val;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
        kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
    }

    __shared__ float shm_qh[4];
    __shared__ float shm_kh[4];
    if (lane_id == 0) {
        shm_qh[warp_id] = qh_local;
        shm_kh[warp_id] = kh_local;
    }

    __syncthreads();

    float qh_v = shm_qh[0] + shm_qh[1] + shm_qh[2] + shm_qh[3];
    float kh_v = shm_kh[0] + shm_kh[1] + shm_kh[2] + shm_kh[3];

    // 6. Update
    float old_v_v = g * kh_v;
    float dv_v = 0.0f;
    float out_v = 0.0f;
    
    if (tx == 0) {
        float v_val = __bfloat162float(v[b_idx * 1024 + h_idx * 128 + v_idx]);
        dv_v = beta * (v_val - old_v_v);
        out_v = scale * (g * qh_v + dv_v * qk_dot);
        output[b_idx * 1024 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
    }
    
    // Broadcast dv_v to all threads in the block to update state
    __shared__ float shm_dv;
    if (tx == 0) {
        shm_dv = dv_v;
    }
    __syncthreads();
    dv_v = shm_dv;

    float new_state_val = g * state_val + dv_v * k_val;
    int state_offset = b_idx * 131072 + h_idx * 16384 + v_idx * 128 + tx;
    new_state[state_offset] = new_state_val;
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

    dim3 grid(B, num_v_heads, V);
    dim3 block(128, 1);

    gdn_decode_kernel_128<<<grid, block>>>(
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
