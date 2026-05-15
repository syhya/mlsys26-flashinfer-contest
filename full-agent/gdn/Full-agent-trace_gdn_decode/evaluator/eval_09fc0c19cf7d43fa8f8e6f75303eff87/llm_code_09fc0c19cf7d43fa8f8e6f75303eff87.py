#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float2 bfloat162_to_float2(const __nv_bfloat162& val) {
#if __CUDA_ARCH__ >= 800
    return __bfloat1622float2(val);
#else
    float2 res;
    res.x = __bfloat162float(val.x);
    res.y = __bfloat162float(val.y);
    return res;
#endif
}

__global__ void __launch_bounds__(128) gdn_decode_kernel_optimized(
    const uint64_t* __restrict__ q,
    const uint64_t* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float4* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float4* __restrict__ new_state
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int warp_id = threadIdx.y;
    int tx = threadIdx.x;

    int qk_h_idx = h_idx / 2;
    
    // Each thread processes 4 elements. K=128, so 32 threads * 4 elements.
    int k_offset = tx; 
    
    // Load q and k
    int qk_base = b_idx * 4 * 32 + qk_h_idx * 32 + tx;
    uint64_t q_u64 = q[qk_base];
    uint64_t k_u64 = k[qk_base];
    
    __nv_bfloat162* q_h2 = (__nv_bfloat162*)&q_u64;
    __nv_bfloat162* k_h2 = (__nv_bfloat162*)&k_u64;
    
    float2 q_f2_0 = bfloat162_to_float2(q_h2[0]);
    float2 q_f2_1 = bfloat162_to_float2(q_h2[1]);
    
    float2 k_f2_0 = bfloat162_to_float2(k_h2[0]);
    float2 k_f2_1 = bfloat162_to_float2(k_h2[1]);
    
    float q_vec[4] = {q_f2_0.x, q_f2_0.y, q_f2_1.x, q_f2_1.y};
    float k_vec[4] = {k_f2_0.x, k_f2_0.y, k_f2_1.x, k_f2_1.y};
    
    float qk_local = q_vec[0]*k_vec[0] + q_vec[1]*k_vec[1] + q_vec[2]*k_vec[2] + q_vec[3]*k_vec[3];
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);
    
    float g, beta;
    if (tx == 0) {
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : logf(1.0f + expf(x));
        g = expf(-expf(A_log_val) * sp);
        beta = 1.0f / (1.0f + expf(-b_val));
    }
    g = __shfl_sync(0xffffffff, g, 0);
    beta = __shfl_sync(0xffffffff, beta, 0);
    
    // V-loop
    // 4 warps per block, V=128. Warp i processes v_idx = i, i+4, ...
    for (int v_idx = warp_id; v_idx < 128; v_idx += 4) {
        int st_idx = b_idx * 8 * 128 * 32 + h_idx * 128 * 32 + v_idx * 32 + tx;
        float4 st_vec = make_float4(0, 0, 0, 0);
        if (state != nullptr) {
            st_vec = state[st_idx];
        }
        
        float qh_local = q_vec[0]*st_vec.x + q_vec[1]*st_vec.y + q_vec[2]*st_vec.z + q_vec[3]*st_vec.w;
        float kh_local = k_vec[0]*st_vec.x + k_vec[1]*st_vec.y + k_vec[2]*st_vec.z + k_vec[3]*st_vec.w;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
            kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        }
        
        float qh_v = __shfl_sync(0xffffffff, qh_local, 0);
        float kh_v = __shfl_sync(0xffffffff, kh_local, 0);
        
        float dv_v;
        if (tx == 0) {
            float v_val = __bfloat162float(v[b_idx * 8 * 128 + h_idx * 128 + v_idx]);
            float old_v_v = g * kh_v;
            dv_v = beta * (v_val - old_v_v);
            float out_v = scale * (g * qh_v + dv_v * qk_dot);
            output[b_idx * 8 * 128 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
        }
        dv_v = __shfl_sync(0xffffffff, dv_v, 0);
        
        st_vec.x = g * st_vec.x + dv_v * k_vec[0];
        st_vec.y = g * st_vec.y + dv_v * k_vec[1];
        st_vec.z = g * st_vec.z + dv_v * k_vec[2];
        st_vec.w = g * st_vec.w + dv_v * k_vec[3];
        
        new_state[st_idx] = st_vec;
    }
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

    const float4* state_ptr = nullptr;
    if (state.defined() && state.numel() > 0) {
        state = state.contiguous();
        state_ptr = reinterpret_cast<const float4*>(state.data_ptr<float>());
    }

    auto options_bf16 = q.options();
    auto options_f32 = options_bf16.dtype(torch::kFloat32);

    torch::Tensor new_state = torch::empty({B, num_v_heads, V, K}, options_f32);
    torch::Tensor output = torch::empty({B, 1, num_v_heads, V}, options_bf16);

    dim3 grid(B, num_v_heads);
    dim3 block(32, 4); // 4 warps per block

    gdn_decode_kernel_optimized<<<grid, block>>>(
        reinterpret_cast<const uint64_t*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint64_t*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        state_ptr,
        A_log.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        dt_bias.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
        scale,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<float4*>(new_state.data_ptr<float>())
    );

    return std::make_tuple(output, new_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_forward", &gdn_forward, "GDN Forward");
}
