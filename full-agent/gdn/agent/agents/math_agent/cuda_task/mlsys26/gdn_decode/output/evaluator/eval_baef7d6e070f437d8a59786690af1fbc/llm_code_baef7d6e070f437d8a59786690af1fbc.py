#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// CUDA Kernel for GDN Decode step
// Thread layout: 128 threads map directly to the K=128 dimension.
// Grid layout: (B, num_v_heads, V) spawns an independent block per V-row to maximize SM occupancy.
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

    __shared__ float sh_g;
    __shared__ float sh_beta;
    __shared__ float sh_dv;

    __shared__ float sh_qk[4];
    __shared__ float sh_qh[4];
    __shared__ float sh_kh[4];

    // Compute head-level scalars exactly once per block
    if (tx == 0) {
        float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
        float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
        float dt_bias_val = dt_bias[h_idx];
        float A_log_val = A_log[h_idx];

        float x = a_val + dt_bias_val;
        // stable softplus
        float sp = x > 20.0f ? x : logf(1.0f + expf(x));
        sh_g = expf(-expf(A_log_val) * sp);
        sh_beta = 1.0f / (1.0f + expf(-b_val));
    }
    __syncthreads();

    float g = sh_g;
    float beta = sh_beta;

    // GVA mapping: num_v_heads (8) -> num_q_heads (4)
    int qk_h_idx = h_idx / 2;
    const __nv_bfloat16* q_ptr = q + b_idx * 4 * 128 + qk_h_idx * 128;
    const __nv_bfloat16* k_ptr = k + b_idx * 4 * 128 + qk_h_idx * 128;

    // Load 1 element per thread perfectly matched to 128 K-dimension
    float q_val = __bfloat162float(q_ptr[tx]);
    float k_val = __bfloat162float(k_ptr[tx]);

    // Enforce safety fallback for optional state tensor
    const float* state_base = state != nullptr ? state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128 : nullptr;
    float st_val = 0.0f;
    if (state_base != nullptr) {
        st_val = state_base[v_idx * 128 + tx];
    }

    // Compute local dot products
    float qk_warp = q_val * k_val;
    float qh_warp = q_val * st_val;
    float kh_warp = k_val * st_val;

    // First stage intra-warp reduction via exact mathematical shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_warp += __shfl_down_sync(0xffffffff, qk_warp, offset);
        qh_warp += __shfl_down_sync(0xffffffff, qh_warp, offset);
        kh_warp += __shfl_down_sync(0xffffffff, kh_warp, offset);
    }

    int warp_id = tx / 32;
    int lane_id = tx % 32;

    if (lane_id == 0) {
        sh_qk[warp_id] = qk_warp;
        sh_qh[warp_id] = qh_warp;
        sh_kh[warp_id] = kh_warp;
    }
    __syncthreads();
    
    // Pre-multiply st_val by g to hide latency
    st_val = g * st_val;

    // Second stage reduction: Warp 0 aggregates the 4 warp sums
    if (warp_id == 0) {
        float qk_dot = 0.0f, qh_v = 0.0f, kh_v = 0.0f;
        if (tx < 4) {
            qk_dot = sh_qk[tx];
            qh_v = sh_qh[tx];
            kh_v = sh_kh[tx];
        }
        
        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            qk_dot += __shfl_down_sync(0xffffffff, qk_dot, offset);
            qh_v += __shfl_down_sync(0xffffffff, qh_v, offset);
            kh_v += __shfl_down_sync(0xffffffff, kh_v, offset);
        }
        
        if (tx == 0) {
            // Compute decoupled dynamics only on thread 0
            float v_val = __bfloat162float(v[b_idx * 8 * 128 + h_idx * 128 + v_idx]);
            float old_v_v = g * kh_v;
            float dv_v = beta * (v_val - old_v_v);
            float out_v = scale * (g * qh_v + dv_v * qk_dot);

            sh_dv = dv_v;
            output[b_idx * 8 * 128 + h_idx * 128 + v_idx] = __float2bfloat16(out_v);
        }
    }
    __syncthreads();

    // Broadcast dv_v and finish mathematically equivalent state update
    float dv_v = sh_dv;
    st_val += dv_v * k_val;

    // Coalesced 128-byte write-back
    float* new_state_base = new_state + b_idx * 8 * 128 * 128 + h_idx * 128 * 128;
    new_state_base[v_idx * 128 + tx] = st_val;
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

    // Force strict memory contiguity before pointer extraction
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    A_log = A_log.contiguous();
    a = a.contiguous();
    dt_bias = dt_bias.contiguous();
    b = b.contiguous();

    // Deterministic state validation to prevent segfaults
    const float* state_ptr = nullptr;
    if (state.defined() && state.numel() > 0) {
        state = state.contiguous();
        state_ptr = state.data_ptr<float>();
    }

    auto options_bf16 = q.options();
    auto options_f32 = options_bf16.dtype(torch::kFloat32);

    torch::Tensor new_state = torch::empty({B, num_v_heads, V, K}, options_f32);
    torch::Tensor output = torch::empty({B, 1, num_v_heads, V}, options_bf16);

    // Grid topology mathematically forces >40% occupancy Multi-Start scaling
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
