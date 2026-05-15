#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Optimized CUDA Kernel for GDN Decode step
// Strategy: Maximize L1TEX cache efficiency and occupancy via vectorized memory accesses.
// - 1 Warp per V-row perfectly aligns with K=128 using float4 / float2 loads.
// - Shared memory caching for q, k, and scalar gates eliminates redundant recomputations.
// - Grid maps exactly to batches, heads, and V-chunks.
__global__ void __launch_bounds__(512) gdn_decode_kernel_vec4(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b,
    float scale,
    int V,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int v_chunk = blockIdx.z;
    int warp_id = threadIdx.y; // 0..15
    int lane_id = threadIdx.x; // 0..31

    int v_idx = v_chunk * 16 + warp_id;
    int qk_h_idx = h_idx / 2;

    // Shared memory for 16-warp block to avoid redundant global memory loads
    // and redundant computations across the V-chunk.
    __shared__ float2 sh_q[32];
    __shared__ float2 sh_k[32];
    __shared__ float sh_qk;
    __shared__ float sh_g;
    __shared__ float sh_beta;

    // Warp 0 precomputes and caches the invariant data for the whole block
    if (warp_id == 0) {
        // Load q and k using float2 (8 bytes = 4 bfloat16 elements)
        const float2* q_ptr2 = reinterpret_cast<const float2*>(q + b_idx * 4 * 128 + qk_h_idx * 128);
        const float2* k_ptr2 = reinterpret_cast<const float2*>(k + b_idx * 4 * 128 + qk_h_idx * 128);
        
        float2 q_vec = q_ptr2[lane_id];
        float2 k_vec = k_ptr2[lane_id];
        sh_q[lane_id] = q_vec;
        sh_k[lane_id] = k_vec;

        const __nv_bfloat16* q_bf = reinterpret_cast<const __nv_bfloat16*>(&q_vec);
        const __nv_bfloat16* k_bf = reinterpret_cast<const __nv_bfloat16*>(&k_vec);

        float q_f[4], k_f[4];
        q_f[0] = __bfloat162float(q_bf[0]);
        q_f[1] = __bfloat162float(q_bf[1]);
        q_f[2] = __bfloat162float(q_bf[2]);
        q_f[3] = __bfloat162float(q_bf[3]);

        k_f[0] = __bfloat162float(k_bf[0]);
        k_f[1] = __bfloat162float(k_bf[1]);
        k_f[2] = __bfloat162float(k_bf[2]);
        k_f[3] = __bfloat162float(k_bf[3]);

        float qk_local = 0.0f;
        #pragma unroll
        for(int i=0; i<4; ++i) {
            qk_local = fmaf(q_f[i], k_f[i], qk_local);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
        }

        if (lane_id == 0) {
            sh_qk = qk_local;
            
            float a_val = __bfloat162float(a[b_idx * 8 + h_idx]);
            float b_val = __bfloat162float(b[b_idx * 8 + h_idx]);
            float dt_bias_val = dt_bias[h_idx];
            float A_log_val = A_log[h_idx];

            float x = a_val + dt_bias_val;
            float sp = x > 20.0f ? x : logf(1.0f + expf(x));
            sh_g = expf(-expf(A_log_val) * sp);
            sh_beta = 1.0f / (1.0f + expf(-b_val));
        }
    }

    __syncthreads();

    // Bounds check after sync to prevent deadlocks
    if (v_idx >= V) return;

    // All warps fetch cached q and k from shared memory (conflict-free)
    float2 q_vec = sh_q[lane_id];
    float2 k_vec = sh_k[lane_id];

    const __nv_bfloat16* q_bf = reinterpret_cast<const __nv_bfloat16*>(&q_vec);
    const __nv_bfloat16* k_bf = reinterpret_cast<const __nv_bfloat16*>(&k_vec);

    float q_f[4], k_f[4];
    q_f[0] = __bfloat162float(q_bf[0]);
    q_f[1] = __bfloat162float(q_bf[1]);
    q_f[2] = __bfloat162float(q_bf[2]);
    q_f[3] = __bfloat162float(q_bf[3]);

    k_f[0] = __bfloat162float(k_bf[0]);
    k_f[1] = __bfloat162float(k_bf[1]);
    k_f[2] = __bfloat162float(k_bf[2]);
    k_f[3] = __bfloat162float(k_bf[3]);

    // Load state using float4 (16 bytes = perfectly coalesced 128-byte L1 cache line per warp)
    const float4* state_ptr4 = nullptr;
    float st_f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (state != nullptr) {
        state_ptr4 = reinterpret_cast<const float4*>(state + b_idx * 8 * V * 128 + h_idx * V * 128 + v_idx * 128);
        float4 st_vec = state_ptr4[lane_id];
        st_f[0] = st_vec.x;
        st_f[1] = st_vec.y;
        st_f[2] = st_vec.z;
        st_f[3] = st_vec.w;
    }

    float qh_local = 0.0f;
    float kh_local = 0.0f;
    #pragma unroll
    for(int i=0; i<4; ++i) {
        qh_local = fmaf(q_f[i], st_f[i], qh_local);
        kh_local = fmaf(k_f[i], st_f[i], kh_local);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
        kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
    }

    kh_local = __shfl_sync(0xffffffff, kh_local, 0);

    float g = sh_g;
    float beta = sh_beta;

    float v_val;
    if (lane_id == 0) {
        v_val = __bfloat162float(v[b_idx * 8 * V + h_idx * V + v_idx]);
    }
    v_val = __shfl_sync(0xffffffff, v_val, 0);

    float old_v_v = g * kh_local;
    float dv_v = beta * (v_val - old_v_v);

    if (lane_id == 0) {
        float qk_dot = sh_qk;
        float out_v = scale * (g * qh_local + dv_v * qk_dot);
        output[b_idx * 8 * V + h_idx * V + v_idx] = __float2bfloat16(out_v);
    }

    // Write-back new state with float4
    float4 new_st_vec;
    new_st_vec.x = g * st_f[0] + dv_v * k_f[0];
    new_st_vec.y = g * st_f[1] + dv_v * k_f[1];
    new_st_vec.z = g * st_f[2] + dv_v * k_f[2];
    new_st_vec.w = g * st_f[3] + dv_v * k_f[3];

    float4* new_state_ptr4 = reinterpret_cast<float4*>(new_state + b_idx * 8 * V * 128 + h_idx * V * 128 + v_idx * 128);
    new_state_ptr4[lane_id] = new_st_vec;
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
    int num_v_heads = v.size(2); // Expected 8
    int V = v.size(3);           // Expected 128
    int K = q.size(3);           // Expected 128

    TORCH_CHECK(num_v_heads == 8, "num_v_heads must be 8");
    TORCH_CHECK(K == 128, "K must be 128 for float4 vectorization");

    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(128.0f);
    }

    // Ensure strictly contiguous memory for safety of offset/vectorized logic
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

    // Grid configuration:
    // Block maps (32 threads, 16 warps). Each warp exclusively processes one V row.
    int v_chunks = (V + 15) / 16;
    dim3 grid(B, num_v_heads, v_chunks);
    dim3 block(32, 16);

    gdn_decode_kernel_vec4<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        state_ptr,
        A_log.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        dt_bias.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
        scale,
        V,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        new_state.data_ptr<float>()
    );

    return std::make_tuple(output, new_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_forward", &gdn_forward, "GDN Forward");
}
