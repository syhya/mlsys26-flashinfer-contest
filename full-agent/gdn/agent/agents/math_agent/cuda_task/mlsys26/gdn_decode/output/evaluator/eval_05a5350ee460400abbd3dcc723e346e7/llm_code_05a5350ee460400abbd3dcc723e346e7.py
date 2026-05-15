#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

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
    int v_group_idx = blockIdx.z; // 0 to 31 (since V=128, V/4=32)
    int tx = threadIdx.x; // 0 to 127
    int ty = threadIdx.y; // 0 to 3

    __shared__ float sh_g;
    __shared__ float sh_beta;
    __shared__ float sh_q[128];
    __shared__ float sh_k[128];
    __shared__ float sh_qk_dot;
    __shared__ float sh_v[4]; // to hold v elements for the 4 rows
    __shared__ __nv_bfloat16 sh_out[4];

    int tid = ty * 128 + tx;

    // Head-level scalars
    if (tid == 0) {
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : logf(1.0f + expf(x));
        sh_g = expf(-expf(A_log_val) * sp);
        sh_beta = 1.0f / (1.0f + expf(-b_val));
    }

    // GVA mapping: num_v_heads (8) -> num_q_heads (4)
    int qk_h_idx = h_idx / 2;
    const __nv_bfloat16* q_ptr = q + b_idx * 4 * 128 + qk_h_idx * 128;
    const __nv_bfloat16* k_ptr = k + b_idx * 4 * 128 + qk_h_idx * 128;

    // Load q and k cooperatively. 128 elements total. 
    // A warp of 32 threads can load 256 bytes using 8-byte loads (float2) per thread!
    if (tid < 32) {
        float2 q_vec = reinterpret_cast<const float2*>(q_ptr)[tid];
        __nv_bfloat16* q_h = reinterpret_cast<__nv_bfloat16*>(&q_vec);
        float q_f[4];
        for (int i=0; i<4; ++i) q_f[i] = __bfloat162float(q_h[i]);

        float2 k_vec = reinterpret_cast<const float2*>(k_ptr)[tid];
        __nv_bfloat16* k_h = reinterpret_cast<__nv_bfloat16*>(&k_vec);
        float k_f[4];
        for (int i=0; i<4; ++i) k_f[i] = __bfloat162float(k_h[i]);

        float qk_sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            sh_q[tid * 4 + i] = q_f[i];
            sh_k[tid * 4 + i] = k_f[i];
            qk_sum += q_f[i] * k_f[i];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qk_sum += __shfl_down_sync(0xffffffff, qk_sum, offset);
        }
        if (tid == 0) {
            sh_qk_dot = qk_sum;
        }
    }

    const __nv_bfloat16* v_ptr = v + b_idx * 8 * 128 + h_idx * 128;
    if (tid == 0) {
        int v_base = v_group_idx * 4;
        float2 v_vec = *reinterpret_cast<const float2*>(&v_ptr[v_base]);
        __nv_bfloat16* v_h = reinterpret_cast<__nv_bfloat16*>(&v_vec);
        sh_v[0] = __bfloat162float(v_h[0]);
        sh_v[1] = __bfloat162float(v_h[1]);
        sh_v[2] = __bfloat162float(v_h[2]);
        sh_v[3] = __bfloat162float(v_h[3]);
    }

    __syncthreads();

    float g = sh_g;
    float beta = sh_beta;
    float qk_dot = sh_qk_dot;
    float q_val = sh_q[tx];
    float k_val = sh_k[tx];

    int v_idx = v_group_idx * 4 + ty;
    float v_val = sh_v[ty];

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

    __shared__ float sh_qh[4][4];
    __shared__ float sh_kh[4][4];

    int warp_id = tx / 32;
    int lane_id = tx % 32;

    if (lane_id == 0) {
        sh_qh[ty][warp_id] = qh_local;
        sh_kh[ty][warp_id] = kh_local;
    }

    __syncthreads();

    float qh_v = 0.0f;
    float kh_v = 0.0f;
    if (lane_id < 4) {
        qh_v = sh_qh[ty][lane_id];
        kh_v = sh_kh[ty][lane_id];
    }
    #pragma unroll
    for (int offset = 2; offset > 0; offset /= 2) {
        qh_v += __shfl_down_sync(0xffffffff, qh_v, offset);
        kh_v += __shfl_down_sync(0xffffffff, kh_v, offset);
    }
    qh_v = __shfl_sync(0xffffffff, qh_v, 0);
    kh_v = __shfl_sync(0xffffffff, kh_v, 0);

    float old_v_v = g * kh_v;
    float dv_v = beta * (v_val - old_v_v);

    if (tx == 0) {
        float out_v = scale * (g * qh_v + dv_v * qk_dot);
        sh_out[ty] = __float2bfloat16(out_v);
    }

    st_val = g * st_val + dv_v * k_val;
    new_state_base[v_idx * 128 + tx] = st_val;

    __syncthreads();

    if (tid == 0) {
        __nv_bfloat16* out_ptr = output + b_idx * 8 * 128 + h_idx * 128;
        int v_base = v_group_idx * 4;
        float2 out_vec = *reinterpret_cast<const float2*>(&sh_out[0]);
        *reinterpret_cast<float2*>(&out_ptr[v_base]) = out_vec;
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
