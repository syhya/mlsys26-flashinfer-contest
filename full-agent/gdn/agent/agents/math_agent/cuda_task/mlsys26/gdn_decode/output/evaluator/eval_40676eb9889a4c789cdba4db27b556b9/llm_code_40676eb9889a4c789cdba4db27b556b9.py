#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void __launch_bounds__(128) gdn_decode_kernel_k128(
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

    __shared__ float sh_g;
    __shared__ float sh_beta;
    __shared__ float sh_qk[4];
    __shared__ float sh_qh[4];
    __shared__ float sh_kh[4];
    __shared__ float sh_v_val;

    if (tx == 0) {
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : logf(1.0f + expf(x));
        sh_g = expf(-expf(A_log_val) * sp);
        sh_beta = 1.0f / (1.0f + expf(-b_val));
        sh_v_val = __bfloat162float(v[b_idx * 8 * 128 + h_idx * 128 + v_idx]);
    }

    int qk_h_idx = h_idx / 2;
    float q_val = __bfloat162float(q[b_idx * 4 * 128 + qk_h_idx * 128 + tx]);
    float k_val = __bfloat162float(k[b_idx * 4 * 128 + qk_h_idx * 128 + tx]);
    
    float st_val = 0.0f;
    if (state != nullptr) {
        st_val = state[b_idx * 8 * 128 * 128 + h_idx * 128 * 128 + v_idx * 128 + tx];
    }

    float qk_val = q_val * k_val;
    float qh_val = q_val * st_val;
    float kh_val = k_val * st_val;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_val += __shfl_down_sync(0xffffffff, qk_val, offset);
        qh_val += __shfl_down_sync(0xffffffff, qh_val, offset);
        kh_val += __shfl_down_sync(0xffffffff, kh_val, offset);
    }

    int lane = tx % 32;
    int warp = tx / 32;

    if (lane == 0) {
        sh_qk[warp] = qk_val;
        sh_qh[warp] = qh_val;
        sh_kh[warp] = kh_val;
    }
    __syncthreads();

    if (warp == 0) {
        qk_val = (lane < 4) ? sh_qk[lane] : 0.0f;
        qh_val = (lane < 4) ? sh_qh[lane] : 0.0f;
        kh_val = (lane < 4) ? sh_kh[lane] : 0.0f;
        
        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            qk_val += __shfl_down_sync(0xffffffff, qk_val, offset);
            qh_val += __shfl_down_sync(0xffffffff, qh_val, offset);
            kh_val += __shfl_down_sync(0xffffffff, kh_val, offset);
        }
        
        if (lane == 0) {
            sh_qk[0] = qk_val;
            sh_qh[0] = qh_val;
            sh_kh[0] = kh_val;
        }
    }
    __syncthreads();

    float g = sh_g;
    float beta = sh_beta;
    float v_val = sh_v_val;
    float qk_dot = sh_qk[0];
    float qh_sum = sh_qh[0];
    float kh_sum = sh_kh[0];

    float old_v_v = g * kh_sum;
    float dv_v = beta * (v_val - old_v_v);
    
    if (tx == 0) {
        float out_v = scale * (g * qh_sum + dv_v * qk_dot);
        output[b_idx * 8 * 128 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
    }

    float new_st_val = g * st_val + dv_v * k_val;
    new_state[b_idx * 8 * 128 * 128 + h_idx * 128 * 128 + v_idx * 128 + tx] = new_st_val;
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
    dim3 block(128);

    gdn_decode_kernel_k128<<<grid, block>>>(
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
