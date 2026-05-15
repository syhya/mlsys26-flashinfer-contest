#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define ENSURE_CONTIGUOUS(t) if (!t.is_contiguous()) t = t.contiguous()

// CUDA Kernel for GDN Decode step, fully K-parallelized
// By parallelizing across the K dimension, we increase the number of active warps by 4x,
// allowing the hardware scheduler to seamlessly hide L1TEX memory access latency.
__global__ void __launch_bounds__(128) gdn_decode_kernel_k_parallel(
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
    // 1 Block = 1 V-row
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int v_idx = blockIdx.z;

    int tx = threadIdx.x; // 0 to 31 (Warp lane)
    int ty = threadIdx.y; // 0 to 3  (Warp ID)
    int k_idx = ty * 32 + tx; // 0 to 127

    __shared__ float sh_g;
    __shared__ float sh_beta;
    __shared__ float sh_v_val;

    // Load and compute loop-invariant head-level scalars exactly once
    if (tx == 0 && ty == 0) {
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

    // GVA mapping: num_v_heads (8) -> num_q_heads (4)
    int qk_h_idx = h_idx / 2;
    
    // Each thread reads 1 element from K dimension (32 threads perfectly span a 128-byte/64-byte transaction)
    float q_val = __bfloat162float(q[b_idx * 4 * 128 + qk_h_idx * 128 + k_idx]);
    float k_val = __bfloat162float(k[b_idx * 4 * 128 + qk_h_idx * 128 + k_idx]);

    const float* state_base = state != nullptr ? state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 : nullptr;
    float st_val = 0.0f;
    if (state_base != nullptr) {
        st_val = state_base[v_idx * 128 + k_idx];
    }

    // Compute partial dot products
    float qk_prod = q_val * k_val;
    float qh_prod = q_val * st_val;
    float kh_prod = k_val * st_val;

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_prod += __shfl_down_sync(0xffffffff, qk_prod, offset);
        qh_prod += __shfl_down_sync(0xffffffff, qh_prod, offset);
        kh_prod += __shfl_down_sync(0xffffffff, kh_prod, offset);
    }

    __shared__ float sh_qk[4];
    __shared__ float sh_qh[4];
    __shared__ float sh_kh[4];

    // Thread 0 of each warp commits partial sum to shared memory
    if (tx == 0) {
        sh_qk[ty] = qk_prod;
        sh_qh[ty] = qh_prod;
        sh_kh[ty] = kh_prod;
    }

    // Ensure shared memory writes and head-level scalar calculations are visible
    __syncthreads();

    // All threads read the 4 partial sums and compute the final sum
    float qk_dot = sh_qk[0] + sh_qk[1] + sh_qk[2] + sh_qk[3];
    float qh_v   = sh_qh[0] + sh_qh[1] + sh_qh[2] + sh_qh[3];
    float kh_v   = sh_kh[0] + sh_kh[1] + sh_kh[2] + sh_kh[3];

    float g = sh_g;
    float beta = sh_beta;
    float v_val = sh_v_val;

    // Compute decoupled update dynamics
    float old_v_v = g * kh_v;
    float dv_v = beta * (v_val - old_v_v);

    // Global output write (done exactly once per V-row)
    if (tx == 0 && ty == 0) {
        float out_v = scale * (g * qh_v + dv_v * qk_dot);
        output[b_idx * 8 * 128 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
    }

    // State update and global store (fully coalesced over K dimension)
    st_val = g * st_val + dv_v * k_val;
    float* new_state_base = new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128;
    new_state_base[v_idx * 128 + k_idx] = st_val;
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

    // Minimized dispatch overhead via fast is_contiguous() inline check
    ENSURE_CONTIGUOUS(q);
    ENSURE_CONTIGUOUS(k);
    ENSURE_CONTIGUOUS(v);
    ENSURE_CONTIGUOUS(A_log);
    ENSURE_CONTIGUOUS(a);
    ENSURE_CONTIGUOUS(dt_bias);
    ENSURE_CONTIGUOUS(b);

    const float* state_ptr = nullptr;
    if (state.defined() && state.numel() > 0) {
        ENSURE_CONTIGUOUS(state);
        state_ptr = state.data_ptr<float>();
    }

    auto options_bf16 = q.options();
    auto options_f32 = options_bf16.dtype(torch::kFloat32);

    torch::Tensor new_state = torch::empty({B, num_v_heads, V, K}, options_f32);
    torch::Tensor output = torch::empty({B, 1, num_v_heads, V}, options_bf16);

    // Grid configuration: (B, HV, V). Launching 128 blocks per head instead of 32.
    // 4 warps per block (128 threads) will process the 128-element K dimension,
    // thereby maintaining 100% bandwidth utilization while driving a 4x higher warp occupancy.
    dim3 grid(B, num_v_heads, V);
    dim3 block(32, 4);

    gdn_decode_kernel_k_parallel<<<grid, block>>>(
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
