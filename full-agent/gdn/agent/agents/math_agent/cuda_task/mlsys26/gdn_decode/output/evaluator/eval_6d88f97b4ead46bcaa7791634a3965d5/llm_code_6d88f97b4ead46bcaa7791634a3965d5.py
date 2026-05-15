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
    int warp_id = tx / 32;
    int lane_id = tx % 32;

    int qk_h_idx = h_idx / 2;
    
    // Load q, k
    const __nv_bfloat16* q_ptr = q + b_idx * 4 * 128 + qk_h_idx * 128;
    const __nv_bfloat16* k_ptr = k + b_idx * 4 * 128 + qk_h_idx * 128;
    
    float q_val = __bfloat162float(q_ptr[tx]);
    float k_val = __bfloat162float(k_ptr[tx]);
    
    // Load state
    const float* state_base = state != nullptr ? state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 : nullptr;
    float state_val = 0.0f;
    if (state_base != nullptr) {
        state_val = state_base[v_idx * 128 + tx];
    }
    
    // Compute local values
    float qh_local = q_val * state_val;
    float kh_local = k_val * state_val;
    float qk_local = q_val * k_val;
    
    // Intra-warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
        kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    
    __shared__ float sh_qh[4];
    __shared__ float sh_kh[4];
    __shared__ float sh_qk[4];
    
    if (lane_id == 0) {
        sh_qh[warp_id] = qh_local;
        sh_kh[warp_id] = kh_local;
        sh_qk[warp_id] = qk_local;
    }
    
    __syncthreads();
    
    // Inter-warp reduction
    float qh_v = 0.0f;
    float kh_v = 0.0f;
    float qk_dot = 0.0f;
    
    if (warp_id == 0) {
        qh_v = (lane_id < 4) ? sh_qh[lane_id] : 0.0f;
        kh_v = (lane_id < 4) ? sh_kh[lane_id] : 0.0f;
        qk_dot = (lane_id < 4) ? sh_qk[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            qh_v += __shfl_down_sync(0xffffffff, qh_v, offset);
            kh_v += __shfl_down_sync(0xffffffff, kh_v, offset);
            qk_dot += __shfl_down_sync(0xffffffff, qk_dot, offset);
        }
    }
    
    __shared__ float sh_g;
    __shared__ float sh_dv_v;
    
    if (tx == 0) {
        // Evaluate non-linear gates
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];
        
        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : logf(1.0f + expf(x));
        float g = expf(-expf(A_log_val) * sp);
        float beta = 1.0f / (1.0f + expf(-b_val));
        
        // Load v_val
        const __nv_bfloat16* v_ptr = v + b_idx * 8 * 128 + h_idx * 128;
        float v_val = __bfloat162float(v_ptr[v_idx]);
        
        float old_v_v = g * kh_v;
        float dv_v = beta * (v_val - old_v_v);
        float out_v = scale * (g * qh_v + dv_v * qk_dot);
        
        // Store output
        __nv_bfloat16* out_ptr = output + b_idx * 8 * 128 + h_idx * 128;
        out_ptr[v_idx] = __float2bfloat16(out_v);
        
        sh_g = g;
        sh_dv_v = dv_v;
    }
    
    __syncthreads();
    
    float g = sh_g;
    float dv_v = sh_dv_v;
    
    // Update state
    float* new_state_base = new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128;
    new_state_base[v_idx * 128 + tx] = g * state_val + dv_v * k_val;
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
    dim3 block(128, 1, 1);

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
