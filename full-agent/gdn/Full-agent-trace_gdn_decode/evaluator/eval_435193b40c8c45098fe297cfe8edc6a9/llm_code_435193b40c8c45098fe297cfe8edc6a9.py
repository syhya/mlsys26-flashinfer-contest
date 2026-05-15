import json

def generate_code():
    return """#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Optimized CUDA Kernel for GDN Decode step
__global__ void __launch_bounds__(128) gdn_decode_kernel_opt(
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
    int warp_id = threadIdx.y;
    int tx = threadIdx.x;

    int qk_h_idx = h_idx / 2;
    
    // Load Q and K into registers (4 elements per thread)
    const uint2* q_ptr = reinterpret_cast<const uint2*>(q + b_idx * 4 * 128 + qk_h_idx * 128);
    const uint2* k_ptr = reinterpret_cast<const uint2*>(k + b_idx * 4 * 128 + qk_h_idx * 128);
    
    uint2 q_vec = q_ptr[tx];
    uint2 k_vec = k_ptr[tx];
    
    __nv_bfloat162* q_bf2 = reinterpret_cast<__nv_bfloat162*>(&q_vec);
    __nv_bfloat162* k_bf2 = reinterpret_cast<__nv_bfloat162*>(&k_vec);
    
    float qf[4], kf[4];
    qf[0] = __bfloat162float(q_bf2[0].x);
    qf[1] = __bfloat162float(q_bf2[0].y);
    qf[2] = __bfloat162float(q_bf2[1].x);
    qf[3] = __bfloat162float(q_bf2[1].y);
    
    kf[0] = __bfloat162float(k_bf2[0].x);
    kf[1] = __bfloat162float(k_bf2[0].y);
    kf[2] = __bfloat162float(k_bf2[1].x);
    kf[3] = __bfloat162float(k_bf2[1].y);
    
    // Compute qk_dot once per warp
    float qk_local = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        qk_local += qf[i] * kf[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);
    
    // Compute block-uniform scalars
    float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
    float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
    float dt_bias_val = dt_bias[h_idx];
    float A_log_val = A_log[h_idx];
    
    float x = a_val + dt_bias_val;
    float sp = x > 20.0f ? x : logf(1.0f + expf(x));
    float g = expf(-expf(A_log_val) * sp);
    float beta = 1.0f / (1.0f + expf(-b_val));
    
    // Loop over V elements assigned to this warp
    // V = 128, blockDim.y = 4, so each warp does 32 iterations
    for (int v_iter = 0; v_iter < 32; ++v_iter) {
        int v_idx = warp_id * 32 + v_iter;
        
        float4 st_vec = make_float4(0, 0, 0, 0);
        if (state != nullptr) {
            const float4* state_ptr = reinterpret_cast<const float4*>(state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 + v_idx * 128);
            st_vec = state_ptr[tx];
        }
        
        float qh_local = 0;
        float kh_local = 0;
        
        qh_local += qf[0] * st_vec.x;
        kh_local += kf[0] * st_vec.x;
        qh_local += qf[1] * st_vec.y;
        kh_local += kf[1] * st_vec.y;
        qh_local += qf[2] * st_vec.z;
        kh_local += kf[2] * st_vec.z;
        qh_local += qf[3] * st_vec.w;
        kh_local += kf[3] * st_vec.w;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
            kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        }
        
        float qh_v = __shfl_sync(0xffffffff, qh_local, 0);
        float kh_v = __shfl_sync(0xffffffff, kh_local, 0);
        
        float v_val_f = __bfloat162float(v[b_idx * 8 * 128 + h_idx * 128 + v_idx]);
        float dv_v = beta * (v_val_f - g * kh_v);
        
        if (tx == 0) {
            float out_v = scale * (g * qh_v + dv_v * qk_dot);
            output[b_idx * 8 * 128 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
        }
        
        st_vec.x = g * st_vec.x + dv_v * kf[0];
        st_vec.y = g * st_vec.y + dv_v * kf[1];
        st_vec.z = g * st_vec.z + dv_v * kf[2];
        st_vec.w = g * st_vec.w + dv_v * kf[3];
        
        float4* new_state_ptr = reinterpret_cast<float4*>(new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 + v_idx * 128);
        new_state_ptr[tx] = st_vec;
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

    dim3 grid(B, num_v_heads);
    dim3 block(32, 4);

    gdn_decode_kernel_opt<<<grid, block>>>(
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
"""

with open("test_kernel.cu", "w") as f:
    f.write(generate_code())
