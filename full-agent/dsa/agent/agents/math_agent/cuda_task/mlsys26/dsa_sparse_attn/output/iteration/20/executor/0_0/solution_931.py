#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

// Devised from Plan: implement the planned exact small-workload split-K redesign
// with partition-native layout and exact FP32 merge, while preserving the proven
// large fused kernel structure for larger workloads.

static constexpr int NUM_HEADS = 16;
static constexpr int D_NOPE = 512;
static constexpr int D_PE = 64;
static constexpr int TOPK = 2048;
static constexpr int SMALL_P = 64;
static constexpr int SMALL_HG = 4;
static constexpr int SMALL_SLOTS = TOPK / SMALL_P; // 32
static constexpr float LOG2E_F = 1.4426950408889634f;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_broadcast0(float v) {
    return __shfl_sync(0xffffffff, v, 0);
}

__global__ void large_fused_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    int t = blockIdx.x;
    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = h * 32 + lane;

    __shared__ alignas(16) int idx_shared[32];
    __shared__ int any_valid_shared;
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * D_NOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * D_PE];

    float2 q_n_f32[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            &q_nope_ptr[(t * NUM_HEADS + h) * D_NOPE + j * 64 + lane * 2]);
        q_n_f32[j] = __bfloat1622float2(qv);
    }
    const __nv_bfloat162 qpe = *reinterpret_cast<const __nv_bfloat162*>(
        &q_pe_ptr[(t * NUM_HEADS + h) * D_PE + lane * 2]);
    const float2 q_p_f32 = __bfloat1622float2(qpe);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);
    float m = -INFINITY;
    float l = 0.0f;

    #pragma unroll 1
    for (int tile = 0; tile < SMALL_P; ++tile) {
        if (tid < 32) idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * 32 + tid];
        __syncthreads();

        if (tid == 0) {
            int any_valid = 0;
            #pragma unroll
            for (int i = 0; i < 32; ++i) any_valid |= (idx_shared[i] != -1);
            any_valid_shared = any_valid;
        }
        __syncthreads();
        if (!any_valid_shared) {
            __syncthreads();
            continue;
        }

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid >> 6);
            int load_col = tid & 63;
            int raw = idx_shared[load_row];
            if (raw != -1) {
                int page = raw >> 6;
                int offset = raw & 63;
                int src_base = ((page * 64 + offset) * D_NOPE);
                const float4* src = reinterpret_cast<const float4*>(&ckv_cache_ptr[src_base]);
                float4* dst = reinterpret_cast<float4*>(&smem_Kc[load_row * D_NOPE]);
                dst[load_col] = src[load_col];
            }
        }
        if (tid < 256) {
            int load_row = tid >> 3;
            int load_col = tid & 7;
            int raw = idx_shared[load_row];
            if (raw != -1) {
                int page = raw >> 6;
                int offset = raw & 63;
                int src_base = ((page * 64 + offset) * D_PE);
                const float4* src = reinterpret_cast<const float4*>(&kpe_cache_ptr[src_base]);
                float4* dst = reinterpret_cast<float4*>(&smem_Kp[load_row * D_PE]);
                dst[load_col] = src[load_col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] == -1) continue;
            float local_dot = 0.0f;
            float2 k_f_reg[8];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_bfloat162 kn = *reinterpret_cast<const __nv_bfloat162*>(
                    &smem_Kc[i * D_NOPE + j * 64 + lane * 2]);
                const float2 kf = __bfloat1622float2(kn);
                k_f_reg[j] = kf;
                local_dot = fmaf(q_n_f32[j].x, kf.x, local_dot);
                local_dot = fmaf(q_n_f32[j].y, kf.y, local_dot);
            }
            const __nv_bfloat162 kp = *reinterpret_cast<const __nv_bfloat162*>(
                &smem_Kp[i * D_PE + lane * 2]);
            const float2 kpf = __bfloat1622float2(kp);
            local_dot = fmaf(q_p_f32.x, kpf.x, local_dot);
            local_dot = fmaf(q_p_f32.y, kpf.y, local_dot);

            float logit = warp_reduce_sum(local_dot);
            logit = warp_broadcast0(logit) * sm_scale;

            float m_new = fmaxf(m, logit);
            float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
            float beta = __expf(logit - m_new);
            l = fmaf(l, alpha, beta);
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                O_reg[j].x = fmaf(beta, k_f_reg[j].x, O_reg[j].x * alpha);
                O_reg[j].y = fmaf(beta, k_f_reg[j].y, O_reg[j].y * alpha);
            }
        }
        __syncthreads();
    }

    float inv_l = (l > 0.0f) ? __fdividef(1.0f, l) : 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float ox = O_reg[j].x * inv_l;
        float oy = O_reg[j].y * inv_l;
        const __nv_bfloat162 outv = __floats2bfloat162_rn(ox, oy);
        *reinterpret_cast<__nv_bfloat162*>(
            &output_ptr[(t * NUM_HEADS + h) * D_NOPE + j * 64 + lane * 2]) = outv;
    }
    if (lane == 0) {
        lse_ptr[t * NUM_HEADS + h] = (l > 0.0f) ? fmaf(m, LOG2E_F, __log2f(l)) : -INFINITY;
    }
}

__global__ void small_partial_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float* __restrict__ O_tmp,
    float* __restrict__ m_tmp,
    float* __restrict__ l_tmp
) {
    const int t = blockIdx.x;
    const int hg_idx = blockIdx.y;
    const int p = blockIdx.z;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int h = hg_idx * SMALL_HG + warp;

    if (h >= NUM_HEADS) return;

    const int slot_start = p * SMALL_SLOTS;

    float2 q_n_f32[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            &q_nope_ptr[(t * NUM_HEADS + h) * D_NOPE + j * 64 + lane * 2]);
        q_n_f32[j] = __bfloat1622float2(qv);
    }
    const __nv_bfloat162 qpe = *reinterpret_cast<const __nv_bfloat162*>(
        &q_pe_ptr[(t * NUM_HEADS + h) * D_PE + lane * 2]);
    const float2 q_p_f32 = __bfloat1622float2(qpe);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);
    float m = -INFINITY;
    float l = 0.0f;

    #pragma unroll
    for (int ii = 0; ii < SMALL_SLOTS; ++ii) {
        int raw = sparse_indices_ptr[t * TOPK + slot_start + ii];
        if (raw == -1) continue;
        int page = raw >> 6;
        int offset = raw & 63;

        float local_dot = 0.0f;
        float2 k_f_reg[8];
        const int ckv_base = ((page * 64 + offset) * D_NOPE);
        const int kpe_base = ((page * 64 + offset) * D_PE);

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 kn = *reinterpret_cast<const __nv_bfloat162*>(
                &ckv_cache_ptr[ckv_base + j * 64 + lane * 2]);
            const float2 kf = __bfloat1622float2(kn);
            k_f_reg[j] = kf;
            local_dot = fmaf(q_n_f32[j].x, kf.x, local_dot);
            local_dot = fmaf(q_n_f32[j].y, kf.y, local_dot);
        }
        const __nv_bfloat162 kp = *reinterpret_cast<const __nv_bfloat162*>(
            &kpe_cache_ptr[kpe_base + lane * 2]);
        const float2 kpf = __bfloat1622float2(kp);
        local_dot = fmaf(q_p_f32.x, kpf.x, local_dot);
        local_dot = fmaf(q_p_f32.y, kpf.y, local_dot);

        float logit = warp_reduce_sum(local_dot);
        logit = warp_broadcast0(logit) * sm_scale;

        float m_new = fmaxf(m, logit);
        float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        float beta = __expf(logit - m_new);
        l = fmaf(l, alpha, beta);
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x = fmaf(beta, k_f_reg[j].x, O_reg[j].x * alpha);
            O_reg[j].y = fmaf(beta, k_f_reg[j].y, O_reg[j].y * alpha);
        }
    }

    const int state_idx = ((t * NUM_HEADS + h) * SMALL_P + p);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int vec = j * 32 + lane;
        int out_idx = (state_idx * 128 + vec) * 4;
        O_tmp[out_idx + 0] = O_reg[j].x;
        O_tmp[out_idx + 1] = O_reg[j].y;
        O_tmp[out_idx + 2] = 0.0f;
        O_tmp[out_idx + 3] = 0.0f;
    }
    if (lane == 0) {
        m_tmp[state_idx] = m;
        l_tmp[state_idx] = l;
    }
}

__global__ void small_reduce_kernel(
    const float* __restrict__ O_tmp,
    const float* __restrict__ m_tmp,
    const float* __restrict__ l_tmp,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;

    __shared__ float m_parts[SMALL_P];
    __shared__ float scale_parts[SMALL_P];
    __shared__ float m_global;
    __shared__ float l_global;

    if (lane < SMALL_P) {
        int idx = ((t * NUM_HEADS + h) * SMALL_P + lane);
        m_parts[lane] = m_tmp[idx];
    }
    __syncthreads();

    if (lane == 0) {
        float m = -INFINITY;
        #pragma unroll
        for (int p = 0; p < SMALL_P; ++p) m = fmaxf(m, m_parts[p]);
        float l = 0.0f;
        #pragma unroll
        for (int p = 0; p < SMALL_P; ++p) {
            if (m_parts[p] != -INFINITY) {
                int idx = ((t * NUM_HEADS + h) * SMALL_P + p);
                float s = __expf(m_parts[p] - m);
                scale_parts[p] = s;
                l = fmaf(l_tmp[idx], s, l);
            } else {
                scale_parts[p] = 0.0f;
            }
        }
        m_global = m;
        l_global = l;
        lse_ptr[t * NUM_HEADS + h] = (m == -INFINITY || l <= 0.0f) ? -INFINITY : fmaf(m, LOG2E_F, __log2f(l));
    }
    __syncthreads();

    int vec = lane;
    if (vec < 128) {
        float out0 = 0.0f, out1 = 0.0f;
        if (m_global != -INFINITY && l_global > 0.0f) {
            float inv_l = __fdividef(1.0f, l_global);
            #pragma unroll
            for (int p = 0; p < SMALL_P; ++p) {
                float coeff = scale_parts[p] * inv_l;
                if (coeff > 0.0f) {
                    int in_idx = ((((t * NUM_HEADS + h) * SMALL_P + p) * 128) + vec) * 4;
                    out0 = fmaf(O_tmp[in_idx + 0], coeff, out0);
                    out1 = fmaf(O_tmp[in_idx + 1], coeff, out1);
                }
            }
        }
        const __nv_bfloat162 outv = __floats2bfloat162_rn(out0, out1);
        *reinterpret_cast<__nv_bfloat162*>(
            &output_ptr[(t * NUM_HEADS + h) * D_NOPE + vec * 2]) = outv;
    }
}

std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    float sm_scale
) {
    const int num_tokens = q_nope.size(0);

    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) {
        return {output, lse};
    }

    // Small-workload exact split-K path; large path preserves the proven fused kernel.
    if (num_tokens < 64) {
        auto opts_f32 = torch::dtype(torch::kFloat32).device(q_nope.device());
        auto O_tmp = torch::empty({num_tokens, NUM_HEADS, SMALL_P, 128, 4}, opts_f32);
        auto m_tmp = torch::empty({num_tokens, NUM_HEADS, SMALL_P}, opts_f32);
        auto l_tmp = torch::empty({num_tokens, NUM_HEADS, SMALL_P}, opts_f32);

        dim3 grid_partial(num_tokens, NUM_HEADS / SMALL_HG, SMALL_P);
        dim3 block_partial(32, SMALL_HG, 1);
        small_partial_kernel<<<grid_partial, block_partial>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>()
        );

        dim3 grid_reduce(num_tokens, NUM_HEADS, 1);
        dim3 block_reduce(128, 1, 1);
        small_reduce_kernel<<<grid_reduce, block_reduce>>>(
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>()
        );
    } else {
        dim3 grid(num_tokens, 1, 1);
        dim3 block(32, NUM_HEADS, 1);
        large_fused_kernel<<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>()
        );
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}
