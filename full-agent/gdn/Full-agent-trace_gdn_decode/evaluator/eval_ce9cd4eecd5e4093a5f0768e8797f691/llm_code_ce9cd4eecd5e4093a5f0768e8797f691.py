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
    
    // 1. Initiate memory loads (GVA mapping: num_v_heads (8) -> num_q_heads (4))
    int qk_h_idx = h_idx / 2;
    float q_val = __bfloat162float(q[b_idx * 4 * 128 + qk_h_idx * 128 + tx]);
    float k_val = __bfloat162float(k[b_idx * 4 * 128 + qk_h_idx * 128 + tx]);
    
    float state_val = 0.0f;
    if (state != nullptr) {
        state_val = state[b_idx * 8 * 128 * 128 + h_idx * 128 * 128 + v_idx * 128 + tx];
    }
    
    // 2. Compute scalars in thread 0 while waiting for memory
    __shared__ float sh_g;
    __shared__ float sh_beta;
    __shared__ float sh_v_val;
    __shared__ float shm_qk[4];
    __shared__ float shm_qh[4];
    __shared__ float shm_kh[4];
    
    if (tx == 0) {
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : __logf(1.0f + __expf(x));
        sh_g = __expf(-__expf(A_log_val) * sp);
        sh_beta = __frcp_rn(1.0f + __expf(-b_val));
        
        sh_v_val = __bfloat162float(v[b_idx * 8 * 128 + h_idx * 128 + v_idx]);
    }
    
    // 3. Compute local dot products
    float qk_local = q_val * k_val;
    float qh_local = q_val * state_val;
    float kh_local = k_val * state_val;
    
    // 4. First Warp Reduction (32 threads)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
        qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
        kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
    }
    
    int warp_id = tx / 32;
    int lane_id = tx % 32;
    
    if (lane_id == 0) {
        shm_qk[warp_id] = qk_local;
        shm_qh[warp_id] = qh_local;
        shm_kh[warp_id] = kh_local;
    }
    
    // Barrier 1: ensure sh_g, sh_beta, sh_v_val, and shm arrays are written
    __syncthreads();
    
    // 5. Final reduction in warp 0
    if (tx < 4) {
        qk_local = shm_qk[tx];
        qh_local = shm_qh[tx];
        kh_local = shm_kh[tx];
    } else {
        qk_local = 0.0f;
        qh_local = 0.0f;
        kh_local = 0.0f;
    }
    
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
            qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
            kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        }
        if (lane_id == 0) {
            shm_qk[0] = qk_local;
            shm_qh[0] = qh_local;
            shm_kh[0] = kh_local;
        }
    }
    
    // Barrier 2: ensure final reductions are visible to all threads
    __syncthreads();
    
    // 6. Read broadcasted values
    float g = sh_g;
    float beta = sh_beta;
    float v_val = sh_v_val;
    float qk_dot = shm_qk[0];
    float qh_v = shm_qh[0];
    float kh_v = shm_kh[0];
    
    // 7. Compute exact decoupled update dynamics
    float old_v_v = g * kh_v;
    float dv_v = beta * (v_val - old_v_v);
    
    // Store to global output array (only thread 0)
    if (tx == 0) {
        float out_v = scale * fmaf(dv_v, qk_dot, g * qh_v);
        output[b_idx * 8 * 128 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
    }
    
    // Update state and write back
    float new_st = fmaf(dv_v, k_val, g * state_val);
    new_state[b_idx * 8 * 128 * 128 + h_idx * 128 * 128 + v_idx * 128 + tx] = new_st;
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

    // Launch configuration: exactly 1 V-row per block, 128 threads to perfectly map to K=128
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
