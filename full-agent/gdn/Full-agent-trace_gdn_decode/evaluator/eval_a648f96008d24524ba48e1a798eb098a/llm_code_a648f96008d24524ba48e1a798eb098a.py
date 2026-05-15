#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// CUDA Kernel for GDN Decode step
__global__ void __launch_bounds__(512) gdn_decode_kernel(
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
    int v_group_idx = blockIdx.z;
    int tx = threadIdx.x; // 0..127
    int ty = threadIdx.y; // 0..3

    __shared__ float sh_g;
    __shared__ float sh_beta;
    __shared__ float sh_qk_dot;
    __shared__ float sh_q[128];
    __shared__ float sh_k[128];
    __shared__ float sh_v[4];
    __shared__ float sh_qh[4][4];
    __shared__ float sh_kh[4][4];
    __shared__ float sh_dv[4];
    __shared__ float2 sh_out_f2;

    int qk_h_idx = h_idx / 2;
    const __nv_bfloat16* q_ptr = q + b_idx * 4 * 128 + qk_h_idx * 128;
    const __nv_bfloat16* k_ptr = k + b_idx * 4 * 128 + qk_h_idx * 128;
    const __nv_bfloat16* v_ptr = v + b_idx * 8 * 128 + h_idx * 128;

    // Load scalar gates and v values cooperatively
    if (ty == 1 && tx == 0) {
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : __logf(1.0f + __expf(x));
        sh_g = __expf(-__expf(A_log_val) * sp);
        sh_beta = __fdividef(1.0f, 1.0f + __expf(-b_val));
        
        float2 v_vec = *reinterpret_cast<const float2*>(v_ptr + v_group_idx * 4);
        __nv_bfloat16* v_h_ptr = reinterpret_cast<__nv_bfloat16*>(&v_vec);
        sh_v[0] = __bfloat162float(v_h_ptr[0]);
        sh_v[1] = __bfloat162float(v_h_ptr[1]);
        sh_v[2] = __bfloat162float(v_h_ptr[2]);
        sh_v[3] = __bfloat162float(v_h_ptr[3]);
    }

    // Load q, k and compute qk_dot cooperatively
    if (ty == 0 && tx < 32) {
        float2 q_vec = reinterpret_cast<const float2*>(q_ptr)[tx];
        float2 k_vec = reinterpret_cast<const float2*>(k_ptr)[tx];
        
        __nv_bfloat16* q_h = reinterpret_cast<__nv_bfloat16*>(&q_vec);
        __nv_bfloat16* k_h = reinterpret_cast<__nv_bfloat16*>(&k_vec);
        
        float q0 = __bfloat162float(q_h[0]);
        float q1 = __bfloat162float(q_h[1]);
        float q2 = __bfloat162float(q_h[2]);
        float q3 = __bfloat162float(q_h[3]);
        
        float k0 = __bfloat162float(k_h[0]);
        float k1 = __bfloat162float(k_h[1]);
        float k2 = __bfloat162float(k_h[2]);
        float k3 = __bfloat162float(k_h[3]);
        
        sh_q[tx * 4 + 0] = q0;
        sh_q[tx * 4 + 1] = q1;
        sh_q[tx * 4 + 2] = q2;
        sh_q[tx * 4 + 3] = q3;
        
        sh_k[tx * 4 + 0] = k0;
        sh_k[tx * 4 + 1] = k1;
        sh_k[tx * 4 + 2] = k2;
        sh_k[tx * 4 + 3] = k3;
        
        float qk = q0*k0 + q1*k1 + q2*k2 + q3*k3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qk += __shfl_down_sync(0xffffffff, qk, offset);
        }
        if (tx == 0) {
            sh_qk_dot = qk;
        }
    }

    __syncthreads();

    float g = sh_g;
    float beta = sh_beta;
    float qk_dot = sh_qk_dot;
    float q_val = sh_q[tx];
    float k_val = sh_k[tx];

    int v_idx = v_group_idx * 4 + ty;
    
    const float* state_base = state != nullptr ? state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 : nullptr;
    float* new_state_base = new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128;
    
    float st_val = 0.0f;
    if (state_base != nullptr) {
        st_val = state_base[v_idx * 128 + tx];
    }
    
    float qh_local = q_val * st_val;
    float kh_local = k_val * st_val;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
        kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
    }
    
    int warp_id = tx / 32;
    int lane_id = tx % 32;
    
    if (lane_id == 0) {
        sh_qh[ty][warp_id] = qh_local;
        sh_kh[ty][warp_id] = kh_local;
    }
    
    __syncthreads();
    
    if (tx == 0) {
        float qh_v = sh_qh[ty][0] + sh_qh[ty][1] + sh_qh[ty][2] + sh_qh[ty][3];
        float kh_v = sh_kh[ty][0] + sh_kh[ty][1] + sh_kh[ty][2] + sh_kh[ty][3];
        
        float v_val = sh_v[ty];
        float old_v_v = g * kh_v;
        float dv_v = beta * (v_val - old_v_v);
        float out_v = scale * (g * qh_v + dv_v * qk_dot);
        
        sh_dv[ty] = dv_v;
        __nv_bfloat16* sh_out = reinterpret_cast<__nv_bfloat16*>(&sh_out_f2);
        sh_out[ty] = __float2bfloat16(out_v);
    }
    
    __syncthreads();
    
    float dv_v = sh_dv[ty];
    st_val = g * st_val + dv_v * k_val;
    new_state_base[v_idx * 128 + tx] = st_val;
    
    if (ty == 0 && tx == 0) {
        __nv_bfloat16* out_ptr = output + b_idx * 8 * 128 + h_idx * 128;
        *reinterpret_cast<float2*>(out_ptr + v_group_idx * 4) = sh_out_f2;
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

    dim3 grid(B, num_v_heads, V / 4);
    dim3 block(128, 4);

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
