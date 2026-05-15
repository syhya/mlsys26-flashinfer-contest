#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

// Devised from plan: implement the exact small-workload partition+merge path,
// while preserving the already strong fused large-workload path. We also replace
// the parent's shared-index mutation used for tile-validity with a separate flag.

static constexpr int NUM_HEADS = 16;
static constexpr int D_NOPE = 512;
static constexpr int D_PE = 64;
static constexpr int TOPK = 2048;
static constexpr int PAGE = 64;
static constexpr int SMALL_P = 64;
static constexpr int SMALL_HG = 4;
static constexpr int SMALL_THRESHOLD = 96;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 x) {
    return __bfloat1622float2(x);
}

__device__ __forceinline__ __nv_bfloat162 float2_to_bf162(float a, float b) {
    return __floats2bfloat162_rn(a, b);
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
    __shared__ int tile_has_valid;
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * D_NOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * D_PE];

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            &q_nope_ptr[(t * NUM_HEADS + h) * D_NOPE + j * 64 + lane * 2]);
        qn[j] = bf162_to_float2(qv);
    }
    __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(
        &q_pe_ptr[(t * NUM_HEADS + h) * D_PE + lane * 2]);
    float2 qp = bf162_to_float2(qpv);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);
    float m = -INFINITY;
    float l = 0.0f;

    #pragma unroll 1
    for (int tile = 0; tile < 64; ++tile) {
        if (tid < 32) idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * 32 + tid];
        if (tid == 0) tile_has_valid = 0;
        __syncthreads();

        if (tid < 32 && idx_shared[tid] != -1) atomicExch(&tile_has_valid, 1);
        __syncthreads();
        if (!tile_has_valid) {
            __syncthreads();
            continue;
        }

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid >> 6);
            int load_col = tid & 63;
            int idx = idx_shared[load_row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(&ckv_cache_ptr[idx * D_NOPE]);
                float4* dst = reinterpret_cast<float4*>(&smem_Kc[load_row * D_NOPE]);
                dst[load_col] = src[load_col];
            }
        }
        if (tid < 256) {
            int load_row = tid >> 3;
            int load_col = tid & 7;
            int idx = idx_shared[load_row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(&kpe_cache_ptr[idx * D_PE]);
                float4* dst = reinterpret_cast<float4*>(&smem_Kp[load_row * D_PE]);
                dst[load_col] = src[load_col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] == -1) continue;

            float local_dot = 0.0f;
            float2 kreg[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * D_NOPE + j * 64 + lane * 2]);
                float2 kf = bf162_to_float2(kv);
                kreg[j] = kf;
                local_dot = fmaf(qn[j].x, kf.x, local_dot);
                local_dot = fmaf(qn[j].y, kf.y, local_dot);
            }
            __nv_bfloat162 kpv = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * D_PE + lane * 2]);
            float2 kpf = bf162_to_float2(kpv);
            local_dot = fmaf(qp.x, kpf.x, local_dot);
            local_dot = fmaf(qp.y, kpf.y, local_dot);

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            float m_new = fmaxf(m, logit);
            float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
            float beta = __expf(logit - m_new);
            l = fmaf(l, alpha, beta);
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                O_reg[j].x = fmaf(beta, kreg[j].x, O_reg[j].x * alpha);
                O_reg[j].y = fmaf(beta, kreg[j].y, O_reg[j].y * alpha);
            }
        }
        __syncthreads();
    }

    float inv_l = (l > 0.0f) ? __fdividef(1.0f, l) : 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 outv = float2_to_bf162(O_reg[j].x * inv_l, O_reg[j].y * inv_l);
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[(t * NUM_HEADS + h) * D_NOPE + j * 64 + lane * 2]) = outv;
    }
    if (lane == 0) {
        lse_ptr[t * NUM_HEADS + h] = (l > 0.0f) ? fmaf(m, 1.4426950408889634f, __log2f(l)) : -INFINITY;
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
    int t = blockIdx.x;
    int h = blockIdx.y * SMALL_HG + threadIdx.y;
    int p = blockIdx.z;
    int lane = threadIdx.x;
    if (h >= NUM_HEADS) return;

    const int slot_begin = p * 32;
    const int base_qn = (t * NUM_HEADS + h) * D_NOPE;
    const int base_qp = (t * NUM_HEADS + h) * D_PE;

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(&q_nope_ptr[base_qn + j * 64 + lane * 2]);
        qn[j] = bf162_to_float2(qv);
    }
    __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(&q_pe_ptr[base_qp + lane * 2]);
    float2 qp = bf162_to_float2(qpv);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);
    float m = -INFINITY;
    float l = 0.0f;

    #pragma unroll
    for (int ii = 0; ii < 32; ++ii) {
        int raw = sparse_indices_ptr[t * TOPK + slot_begin + ii];
        if (raw == -1) continue;

        float local_dot = 0.0f;
        float2 kreg[8];
        const int kc_base = raw * D_NOPE;
        const int kp_base = raw * D_PE;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(&ckv_cache_ptr[kc_base + j * 64 + lane * 2]);
            float2 kf = bf162_to_float2(kv);
            kreg[j] = kf;
            local_dot = fmaf(qn[j].x, kf.x, local_dot);
            local_dot = fmaf(qn[j].y, kf.y, local_dot);
        }
        __nv_bfloat162 kpv2 = *reinterpret_cast<const __nv_bfloat162*>(&kpe_cache_ptr[kp_base + lane * 2]);
        float2 kpf = bf162_to_float2(kpv2);
        local_dot = fmaf(qp.x, kpf.x, local_dot);
        local_dot = fmaf(qp.y, kpf.y, local_dot);

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float m_new = fmaxf(m, logit);
        float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        float beta = __expf(logit - m_new);
        l = fmaf(l, alpha, beta);
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x = fmaf(beta, kreg[j].x, O_reg[j].x * alpha);
            O_reg[j].y = fmaf(beta, kreg[j].y, O_reg[j].y * alpha);
        }
    }

    const int row_base = ((t * NUM_HEADS + h) * SMALL_P + p) * D_NOPE;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float2* out_ptr = reinterpret_cast<float2*>(&O_tmp[row_base + j * 64 + lane * 2]);
        *out_ptr = make_float2(O_reg[j].x, O_reg[j].y);
    }
    if (lane == 0) {
        m_tmp[(t * NUM_HEADS + h) * SMALL_P + p] = m;
        l_tmp[(t * NUM_HEADS + h) * SMALL_P + p] = l;
    }
}

__global__ void small_reduce_kernel(
    const float* __restrict__ O_tmp,
    const float* __restrict__ m_tmp,
    const float* __restrict__ l_tmp,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int lane = threadIdx.x;

    __shared__ float m_part[SMALL_P];
    __shared__ float scale_part[SMALL_P];
    __shared__ float m_global;
    __shared__ float l_global;

    if (lane < SMALL_P) {
        m_part[lane] = m_tmp[(t * NUM_HEADS + h) * SMALL_P + lane];
    }
    __syncthreads();

    if (lane == 0) {
        float mg = -INFINITY;
        #pragma unroll
        for (int p = 0; p < SMALL_P; ++p) mg = fmaxf(mg, m_part[p]);
        float lg = 0.0f;
        if (mg != -INFINITY) {
            #pragma unroll
            for (int p = 0; p < SMALL_P; ++p) {
                float mp = m_part[p];
                if (mp != -INFINITY) {
                    lg = fmaf(l_tmp[(t * NUM_HEADS + h) * SMALL_P + p], __expf(mp - mg), lg);
                }
            }
        }
        m_global = mg;
        l_global = lg;
        lse_ptr[t * NUM_HEADS + h] = (mg == -INFINITY || lg <= 0.0f) ? -INFINITY : fmaf(mg, 1.4426950408889634f, __log2f(lg));
    }
    __syncthreads();

    if (lane < SMALL_P) {
        float mp = m_part[lane];
        if (m_global != -INFINITY && l_global > 0.0f && mp != -INFINITY) {
            scale_part[lane] = __fdividef(__expf(mp - m_global), l_global);
        } else {
            scale_part[lane] = 0.0f;
        }
    }
    __syncthreads();

    int d = lane * 4;
    if (d < D_NOPE) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        if (m_global != -INFINITY && l_global > 0.0f) {
            #pragma unroll 16
            for (int p = 0; p < SMALL_P; ++p) {
                float s = scale_part[p];
                if (s > 0.0f) {
                    int base = (((t * NUM_HEADS + h) * SMALL_P + p) * D_NOPE + d);
                    float4 v = *reinterpret_cast<const float4*>(&O_tmp[base]);
                    acc.x = fmaf(v.x, s, acc.x);
                    acc.y = fmaf(v.y, s, acc.y);
                    acc.z = fmaf(v.z, s, acc.z);
                    acc.w = fmaf(v.w, s, acc.w);
                }
            }
        }
        __nv_bfloat162 o0 = float2_to_bf162(acc.x, acc.y);
        __nv_bfloat162 o1 = float2_to_bf162(acc.z, acc.w);
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[(t * NUM_HEADS + h) * D_NOPE + d + 0]) = o0;
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[(t * NUM_HEADS + h) * D_NOPE + d + 2]) = o1;
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
    int num_tokens = q_nope.size(0);

    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    if (num_tokens < SMALL_THRESHOLD) {
        auto O_tmp = torch::empty({num_tokens, NUM_HEADS, SMALL_P, D_NOPE}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto m_tmp = torch::empty({num_tokens, NUM_HEADS, SMALL_P}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto l_tmp = torch::empty({num_tokens, NUM_HEADS, SMALL_P}, torch::dtype(torch::kFloat32).device(q_nope.device()));

        dim3 grid_partial(num_tokens, (NUM_HEADS + SMALL_HG - 1) / SMALL_HG, SMALL_P);
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
