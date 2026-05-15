#include <torch/extension.h>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <int K>
struct KTraits;

template <>
struct KTraits<64> {
    static constexpr int VECS = 2;
    static __device__ __forceinline__ void load_qk(const __nv_bfloat16* ptr, int tx, float* vals) {
        float vec = reinterpret_cast<const float*>(ptr)[tx];
        __nv_bfloat16* h_ptr = reinterpret_cast<__nv_bfloat16*>(&vec);
        vals[0] = __bfloat162float(h_ptr[0]);
        vals[1] = __bfloat162float(h_ptr[1]);
    }
    static __device__ __forceinline__ void load_state(const float* ptr, int tx, float* vals) {
        float2 vec = reinterpret_cast<const float2*>(ptr)[tx];
        vals[0] = vec.x;
        vals[1] = vec.y;
    }
    static __device__ __forceinline__ void store_state(float* ptr, int tx, const float* vals) {
        float2 vec = make_float2(vals[0], vals[1]);
        reinterpret_cast<float2*>(ptr)[tx] = vec;
    }
};

template <>
struct KTraits<128> {
    static constexpr int VECS = 4;
    static __device__ __forceinline__ void load_qk(const __nv_bfloat16* ptr, int tx, float* vals) {
        float2 vec = reinterpret_cast<const float2*>(ptr)[tx];
        __nv_bfloat16* h_ptr = reinterpret_cast<__nv_bfloat16*>(&vec);
        vals[0] = __bfloat162float(h_ptr[0]);
        vals[1] = __bfloat162float(h_ptr[1]);
        vals[2] = __bfloat162float(h_ptr[2]);
        vals[3] = __bfloat162float(h_ptr[3]);
    }
    static __device__ __forceinline__ void load_state(const float* ptr, int tx, float* vals) {
        float4 vec = reinterpret_cast<const float4*>(ptr)[tx];
        vals[0] = vec.x;
        vals[1] = vec.y;
        vals[2] = vec.z;
        vals[3] = vec.w;
    }
    static __device__ __forceinline__ void store_state(float* ptr, int tx, const float* vals) {
        float4 vec = make_float4(vals[0], vals[1], vals[2], vals[3]);
        reinterpret_cast<float4*>(ptr)[tx] = vec;
    }
};

template <>
struct KTraits<256> {
    static constexpr int VECS = 8;
    static __device__ __forceinline__ void load_qk(const __nv_bfloat16* ptr, int tx, float* vals) {
        float4 vec = reinterpret_cast<const float4*>(ptr)[tx];
        __nv_bfloat16* h_ptr = reinterpret_cast<__nv_bfloat16*>(&vec);
        #pragma unroll
        for(int i=0; i<8; ++i) {
            vals[i] = __bfloat162float(h_ptr[i]);
        }
    }
    static __device__ __forceinline__ void load_state(const float* ptr, int tx, float* vals) {
        float4 vec1 = reinterpret_cast<const float4*>(ptr)[tx * 2];
        float4 vec2 = reinterpret_cast<const float4*>(ptr)[tx * 2 + 1];
        vals[0] = vec1.x; vals[1] = vec1.y; vals[2] = vec1.z; vals[3] = vec1.w;
        vals[4] = vec2.x; vals[5] = vec2.y; vals[6] = vec2.z; vals[7] = vec2.w;
    }
    static __device__ __forceinline__ void store_state(float* ptr, int tx, const float* vals) {
        float4 vec1 = make_float4(vals[0], vals[1], vals[2], vals[3]);
        float4 vec2 = make_float4(vals[4], vals[5], vals[6], vals[7]);
        reinterpret_cast<float4*>(ptr)[tx * 2] = vec1;
        reinterpret_cast<float4*>(ptr)[tx * 2 + 1] = vec2;
    }
};

// Fast-path kernel using templates for known dimensions
template <int NUM_V_HEADS, int NUM_Q_HEADS, int V, int K>
__global__ void __launch_bounds__(128) gdn_decode_kernel_specialized(
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
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sh_g;
    __shared__ float sh_beta;

    // Compute head-level scalars exactly once per block
    if (tx == 0 && ty == 0) {
        float a_val = __bfloat162float(a[b_idx * NUM_V_HEADS + h_idx]);
        float b_val = __bfloat162float(b[b_idx * NUM_V_HEADS + h_idx]);
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

    // GVA mapping
    int qk_h_idx = h_idx * NUM_Q_HEADS / NUM_V_HEADS;
    const __nv_bfloat16* q_ptr = q + b_idx * NUM_Q_HEADS * K + qk_h_idx * K;
    const __nv_bfloat16* k_ptr = k + b_idx * NUM_Q_HEADS * K + qk_h_idx * K;

    // Vectorized read of 4 bfloat16 elements into registers using float2
    float2 q_vec = reinterpret_cast<const float2*>(q_ptr)[tx];
    __nv_bfloat16* q_h_ptr = reinterpret_cast<__nv_bfloat16*>(&q_vec);
    float q_f[4];
    q_f[0] = __bfloat162float(q_h_ptr[0]);
    q_f[1] = __bfloat162float(q_h_ptr[1]);
    q_f[2] = __bfloat162float(q_h_ptr[2]);
    q_f[3] = __bfloat162float(q_h_ptr[3]);

    float2 k_vec = reinterpret_cast<const float2*>(k_ptr)[tx];
    __nv_bfloat16* k_h_ptr = reinterpret_cast<__nv_bfloat16*>(&k_vec);
    float k_f[4];
    k_f[0] = __bfloat162float(k_h_ptr[0]);
    k_f[1] = __bfloat162float(k_h_ptr[1]);
    k_f[2] = __bfloat162float(k_h_ptr[2]);
    k_f[3] = __bfloat162float(k_h_ptr[3]);

    // Precompute dot product of q and k for this block
    float qk_local = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        qk_local += q_f[i] * k_f[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);

    const float* state_base = state != nullptr ? state + b_idx * NUM_V_HEADS * V * K + h_idx * V * K : nullptr;
    float* new_state_base = new_state + b_idx * NUM_V_HEADS * V * K + h_idx * V * K;
    const __nv_bfloat16* v_ptr = v + b_idx * NUM_V_HEADS * V + h_idx * V;
    __nv_bfloat16* out_ptr = output + b_idx * NUM_V_HEADS * V + h_idx * V;

    int v_idx = v_group_idx * 4 + ty;
    if (v_idx < V) {
        // Coalesced 128-byte read of the state tensor using float4
        float4 st = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (state_base != nullptr) {
            st = reinterpret_cast<const float4*>(state_base + v_idx * K)[tx];
        }

        // Compute local dot products for qh and kh
        float qh_local = q_f[0] * st.x + q_f[1] * st.y + q_f[2] * st.z + q_f[3] * st.w;
        float kh_local = k_f[0] * st.x + k_f[1] * st.y + k_f[2] * st.z + k_f[3] * st.w;

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
            kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        }
        float qh_v = __shfl_sync(0xffffffff, qh_local, 0);
        float kh_v = __shfl_sync(0xffffffff, kh_local, 0);

        // Load the single element from v_h
        float v_val = 0.0f;
        if (tx == 0) {
            v_val = __bfloat162float(v_ptr[v_idx]);
        }
        v_val = __shfl_sync(0xffffffff, v_val, 0);

        // Compute exactly equivalent decoupled update dynamics
        float old_v_v = g * kh_v;
        float dv_v = beta * (v_val - old_v_v);
        float out_v = scale * (g * qh_v + dv_v * qk_dot);

        // Store to global output array
        if (tx == 0) {
            out_ptr[v_idx] = __float2bfloat16(out_v);
        }

        // Update local state registers
        st.x = g * st.x + dv_v * k_f[0];
        st.y = g * st.y + dv_v * k_f[1];
        st.z = g * st.z + dv_v * k_f[2];
        st.w = g * st.w + dv_v * k_f[3];

        // Coalesced 128-byte write back to state tensor
        reinterpret_cast<float4*>(new_state_base + v_idx * K)[tx] = st;
    }
}

template <int K>
__global__ void __launch_bounds__(128) gdn_decode_kernel_fast(
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
    float* __restrict__ new_state,
    int B, int num_v_heads, int num_q_heads, int V
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int v_group_idx = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sh_g;
    __shared__ float sh_beta;

    // Compute head-level scalars exactly once per block
    if (tx == 0 && ty == 0) {
        float a_val = __bfloat162float(a[b_idx * num_v_heads + h_idx]);
        float b_val = __bfloat162float(b[b_idx * num_v_heads + h_idx]);
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

    // GVA mapping
    int qk_h_idx = h_idx * num_q_heads / num_v_heads;
    const __nv_bfloat16* q_ptr = q + b_idx * num_q_heads * K + qk_h_idx * K;
    const __nv_bfloat16* k_ptr = k + b_idx * num_q_heads * K + qk_h_idx * K;

    constexpr int VECS = KTraits<K>::VECS;
    float q_f[VECS];
    float k_f[VECS];

    KTraits<K>::load_qk(q_ptr, tx, q_f);
    KTraits<K>::load_qk(k_ptr, tx, k_f);

    // Precompute dot product of q and k for this block
    float qk_local = 0.0f;
    #pragma unroll
    for (int i = 0; i < VECS; ++i) {
        qk_local += q_f[i] * k_f[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);

    const float* state_base = state != nullptr ? state + b_idx * num_v_heads * V * K + h_idx * V * K : nullptr;
    float* new_state_base = new_state + b_idx * num_v_heads * V * K + h_idx * V * K;
    const __nv_bfloat16* v_ptr = v + b_idx * num_v_heads * V + h_idx * V;
    __nv_bfloat16* out_ptr = output + b_idx * num_v_heads * V + h_idx * V;

    int v_idx = v_group_idx * blockDim.y + ty;
    if (v_idx < V) {
        float st_f[VECS];
        #pragma unroll
        for(int i=0; i<VECS; ++i) st_f[i] = 0.0f;

        if (state_base != nullptr) {
            KTraits<K>::load_state(state_base + v_idx * K, tx, st_f);
        }

        // Compute local dot products for qh and kh
        float qh_local = 0.0f;
        float kh_local = 0.0f;
        #pragma unroll
        for (int i = 0; i < VECS; ++i) {
            qh_local += q_f[i] * st_f[i];
            kh_local += k_f[i] * st_f[i];
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
            kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        }
        float qh_v = __shfl_sync(0xffffffff, qh_local, 0);
        float kh_v = __shfl_sync(0xffffffff, kh_local, 0);

        // Load the single element from v_h
        float v_val = 0.0f;
        if (tx == 0) {
            v_val = __bfloat162float(v_ptr[v_idx]);
        }
        v_val = __shfl_sync(0xffffffff, v_val, 0);

        // Compute exactly equivalent decoupled update dynamics
        float old_v_v = g * kh_v;
        float dv_v = beta * (v_val - old_v_v);
        float out_v = scale * (g * qh_v + dv_v * qk_dot);

        // Store to global output array
        if (tx == 0) {
            out_ptr[v_idx] = __float2bfloat16(out_v);
        }

        // Update local state registers
        #pragma unroll
        for(int i=0; i<VECS; ++i) {
            st_f[i] = g * st_f[i] + dv_v * k_f[i];
        }

        // Coalesced write back to state tensor
        KTraits<K>::store_state(new_state_base + v_idx * K, tx, st_f);
    }
}

// Generic fallback kernel for arbitrary K
__global__ void gdn_decode_kernel_generic(
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
    float* __restrict__ new_state,
    int B, int num_v_heads, int num_q_heads, int V, int K
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int v_group_idx = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sh_g;
    __shared__ float sh_beta;

    if (tx == 0 && ty == 0) {
        float a_val = __bfloat162float(a[b_idx * num_v_heads + h_idx]);
        float b_val = __bfloat162float(b[b_idx * num_v_heads + h_idx]);
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

    int qk_h_idx = h_idx * num_q_heads / num_v_heads;
    const __nv_bfloat16* q_ptr = q + b_idx * num_q_heads * K + qk_h_idx * K;
    const __nv_bfloat16* k_ptr = k + b_idx * num_q_heads * K + qk_h_idx * K;

    float qk_local = 0.0f;
    for (int i = tx; i < K; i += blockDim.x) {
        qk_local += __bfloat162float(q_ptr[i]) * __bfloat162float(k_ptr[i]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    }
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);

    const float* state_base = state != nullptr ? state + b_idx * num_v_heads * V * K + h_idx * V * K : nullptr;
    float* new_state_base = new_state + b_idx * num_v_heads * V * K + h_idx * V * K;
    const __nv_bfloat16* v_ptr = v + b_idx * num_v_heads * V + h_idx * V;
    __nv_bfloat16* out_ptr = output + b_idx * num_v_heads * V + h_idx * V;

    int v_idx = v_group_idx * blockDim.y + ty;
    if (v_idx < V) {
        float qh_local = 0.0f;
        float kh_local = 0.0f;

        for (int i = tx; i < K; i += blockDim.x) {
            float st = state_base != nullptr ? state_base[v_idx * K + i] : 0.0f;
            float q_val = __bfloat162float(q_ptr[i]);
            float k_val = __bfloat162float(k_ptr[i]);

            qh_local += q_val * st;
            kh_local += k_val * st;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qh_local += __shfl_down_sync(0xffffffff, qh_local, offset);
            kh_local += __shfl_down_sync(0xffffffff, kh_local, offset);
        }
        float qh_v = __shfl_sync(0xffffffff, qh_local, 0);
        float kh_v = __shfl_sync(0xffffffff, kh_local, 0);

        float v_val = 0.0f;
        if (tx == 0) {
            v_val = __bfloat162float(v_ptr[v_idx]);
        }
        v_val = __shfl_sync(0xffffffff, v_val, 0);

        float old_v_v = g * kh_v;
        float dv_v = beta * (v_val - old_v_v);
        float out_v = scale * (g * qh_v + dv_v * qk_dot);

        if (tx == 0) {
            out_ptr[v_idx] = __float2bfloat16(out_v);
        }

        for (int i = tx; i < K; i += blockDim.x) {
            float st = state_base != nullptr ? state_base[v_idx * K + i] : 0.0f;
            float k_val = __bfloat162float(k_ptr[i]);
            float new_st = g * st + dv_v * k_val;
            new_state_base[v_idx * K + i] = new_st;
        }
    }
}

// C++ Entry Point
std::tuple<torch::Tensor, torch::Tensor> gdn_forward(
    torch::Tensor q,       // [batch_size, 1, num_q_heads, K]     bfloat16
    torch::Tensor k,       // [batch_size, 1, num_q_heads, K]     bfloat16
    torch::Tensor v,       // [batch_size, 1, num_v_heads, V]     bfloat16
    torch::Tensor state,   // [batch_size, num_v_heads, V, K]   float32  (k-last layout)
    torch::Tensor A_log,   // [num_v_heads]                          float32
    torch::Tensor a,       // [batch_size, 1, num_v_heads]           bfloat16
    torch::Tensor dt_bias, // [num_v_heads]                          float32
    torch::Tensor b,       // [batch_size, 1, num_v_heads]           bfloat16
    float scale            // scalar
) {
    int B = q.size(0);
    int num_q_heads = q.size(2);
    int num_v_heads = v.size(2);
    int V = v.size(3);
    int K = q.size(3);

    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(K));
    }

    // Ensure contiguous memory for safe casting and offset logic
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

    if (K == 128 && num_v_heads == 8 && num_q_heads == 4 && V == 128) {
        dim3 grid(B, 8, 128 / 4);
        dim3 block(32, 4);

        gdn_decode_kernel_specialized<8, 4, 128, 128><<<grid, block>>>(
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
    } else {
        dim3 block(32, 4);
        dim3 grid(B, num_v_heads, (V + 3) / 4);

        if (K == 64) {
            gdn_decode_kernel_fast<64><<<grid, block>>>(
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
                new_state.data_ptr<float>(),
                B, num_v_heads, num_q_heads, V
            );
        } else if (K == 128) {
            gdn_decode_kernel_fast<128><<<grid, block>>>(
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
                new_state.data_ptr<float>(),
                B, num_v_heads, num_q_heads, V
            );
        } else if (K == 256) {
            gdn_decode_kernel_fast<256><<<grid, block>>>(
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
                new_state.data_ptr<float>(),
                B, num_v_heads, num_q_heads, V
            );
        } else {
            gdn_decode_kernel_generic<<<grid, block>>>(
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
                new_state.data_ptr<float>(),
                B, num_v_heads, num_q_heads, V, K
            );
        }
    }

    return std::make_tuple(output, new_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_forward", &gdn_forward, "GDN Forward");
}
