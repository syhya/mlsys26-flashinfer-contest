import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
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
    int v_group_idx = blockIdx.z;
    int tx = threadIdx.x; // 0..127
    int ty = threadIdx.y; // 0..3

    __shared__ float sh_g;
    __shared__ float sh_beta;
    __shared__ float sh_qh[4][4];
    __shared__ float sh_kh[4][4];
    __shared__ float sh_qk[4][4];
    __shared__ float sh_dv[4];

    if (tx == 0 && ty == 0) {
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        float sp = x > 20.0f ? x : logf(1.0f + expf(x));
        sh_g = expf(-expf(A_log_val) * sp);
        sh_beta = 1.0f / (1.0f + expf(-b_val));
    }
    __syncthreads();

    float g = sh_g;
    float beta = sh_beta;

    int qk_h_idx = h_idx / 2;
    const __nv_bfloat16* q_ptr = q + b_idx * 4 * 128 + qk_h_idx * 128;
    const __nv_bfloat16* k_ptr = k + b_idx * 4 * 128 + qk_h_idx * 128;

    float q_val = __bfloat162float(q_ptr[tx]);
    float k_val = __bfloat162float(k_ptr[tx]);

    int v_idx = v_group_idx * 4 + ty;
    
    float st_val = 0.0f;
    const float* state_base = nullptr;
    if (state != nullptr) {
        state_base = state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128;
        if (v_idx < 128) {
            st_val = state_base[v_idx * 128 + tx];
        }
    }

    float qh_local = q_val * st_val;
    float kh_local = k_val * st_val;
    float qk_local = q_val * k_val;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
        kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }

    int warp_idx = tx / 32;
    int lane_idx = tx % 32;

    if (lane_idx == 0) {
        sh_qh[ty][warp_idx] = qh_local;
        sh_kh[ty][warp_idx] = kh_local;
        sh_qk[ty][warp_idx] = qk_local;
    }
    __syncthreads();

    if (v_idx < 128) {
        if (tx == 0) {
            float qh_v = sh_qh[ty][0] + sh_qh[ty][1] + sh_qh[ty][2] + sh_qh[ty][3];
            float kh_v = sh_kh[ty][0] + sh_kh[ty][1] + sh_kh[ty][2] + sh_kh[ty][3];
            float qk_dot = sh_qk[ty][0] + sh_qk[ty][1] + sh_qk[ty][2] + sh_qk[ty][3];

            const __nv_bfloat16* v_ptr = v + b_idx * 8 * 128 + h_idx * 128;
            float v_val = __bfloat162float(v_ptr[v_idx]);

            float old_v_v = g * kh_v;
            float dv_v = beta * (v_val - old_v_v);
            float out_v = scale * (g * qh_v + dv_v * qk_dot);

            __nv_bfloat16* out_ptr = output + b_idx * 8 * 128 + h_idx * 128;
            out_ptr[v_idx] = __float2bfloat16(out_v);

            sh_dv[ty] = dv_v;
        }
    }
    __syncthreads();

    if (v_idx < 128) {
        float dv_v = sh_dv[ty];
        st_val = g * st_val + dv_v * k_val;
        float* new_state_base = new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128;
        new_state_base[v_idx * 128 + tx] = st_val;
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

    if (scale == 0.0f) scale = 1.0f / std::sqrt(128.0f);

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
"""

module = load_inline(
    name='gdn_decode',
    cpp_sources=[],
    cuda_sources=[cuda_source],
    functions=['gdn_forward'],
    with_cuda=True,
    extra_cuda_cflags=['-O3', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__']
)

def ref_forward(q, k, v, state, A_log, a, dt_bias, b, scale):
    B = q.shape[0]
    V = v.shape[3]
    K = q.shape[3]
    if scale == 0.0:
        scale = K ** -0.5
    
    out = torch.zeros_like(v)
    new_state = state.clone() if state is not None else torch.zeros(B, 8, V, K, dtype=torch.float32, device=q.device)
    
    for b_idx in range(B):
        for h_idx in range(8):
            qk_h_idx = h_idx // 2
            
            a_val = a[b_idx, 0, h_idx].float()
            dt_bias_val = dt_bias[h_idx].float()
            A_log_val = A_log[h_idx].float()
            b_val = b[b_idx, 0, h_idx].float()
            
            x = a_val + dt_bias_val
            sp = x if x > 20.0 else torch.log(1.0 + torch.exp(x))
            g = torch.exp(-torch.exp(A_log_val) * sp)
            beta = 1.0 / (1.0 + torch.exp(-b_val))
            
            q_vec = q[b_idx, 0, qk_h_idx].float()
            k_vec = k[b_idx, 0, qk_h_idx].float()
            qk_dot = torch.sum(q_vec * k_vec)
            
            for v_idx in range(V):
                st = new_state[b_idx, h_idx, v_idx].float()
                
                qh = torch.sum(q_vec * st)
                kh = torch.sum(k_vec * st)
                
                v_val = v[b_idx, 0, h_idx, v_idx].float()
                
                old_v = g * kh
                dv = beta * (v_val - old_v)
                out_val = scale * (g * qh + dv * qk_dot)
                
                out[b_idx, 0, h_idx, v_idx] = out_val.to(torch.bfloat16)
                
                new_state[b_idx, h_idx, v_idx] = g * st + dv * k_vec
                
    return out, new_state

B = 2
q = torch.randn(B, 1, 4, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(B, 1, 4, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(B, 1, 8, 128, dtype=torch.bfloat16, device='cuda')
state = torch.randn(B, 8, 128, 128, dtype=torch.float32, device='cuda')
A_log = torch.randn(8, dtype=torch.float32, device='cuda')
a = torch.randn(B, 1, 8, dtype=torch.bfloat16, device='cuda')
dt_bias = torch.randn(8, dtype=torch.float32, device='cuda')
b = torch.randn(B, 1, 8, dtype=torch.bfloat16, device='cuda')
scale = 0.0

out_ref, state_ref = ref_forward(q, k, v, state, A_log, a, dt_bias, b, scale)
out_my, state_my = module.gdn_forward(q, k, v, state, A_log, a, dt_bias, b, scale)

print("out max diff:", (out_ref - out_my).abs().max().item())
print("state max diff:", (state_ref - state_my).abs().max().item())
