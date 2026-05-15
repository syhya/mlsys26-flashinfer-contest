#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>
#include <ATen/cuda/CUDAContext.h>

// Child solution: materially changes the small-workload path from the parent.
// We keep the parent's strong fused large-token path, but redesign the small path
// into an over-decomposed split-head/chunk scheme with exact online-softmax state merge
// and float4-packed workspace stores/loads that match reduction access.

constexpr int NUM_HEADS = 16;
constexpr int D_NOPE = 512;
constexpr int D_PE = 64;
constexpr int TOPK = 2048;
constexpr int CHUNK = 32;
constexpr int NUM_CHUNKS = TOPK / CHUNK;
constexpr int HEADS_PER_BLOCK = 4;
constexpr int HEAD_GROUPS = NUM_HEADS / HEADS_PER_BLOCK;
constexpr int PARTS = NUM_CHUNKS * HEAD_GROUPS;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_INF_F = -INFINITY;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float2 bf162_to_float2(const __nv_bfloat162 x) {
    return __bfloat1622float2(x);
}

__global__ void dsa_forward_large_kernel(
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
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * D_NOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * D_PE];

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            q_nope_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2);
        qn[j] = bf162_to_float2(qv);
    }
    const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(
        q_pe_ptr + t * NUM_HEADS * D_PE + h * D_PE + lane * 2);
    float2 qp = bf162_to_float2(qpv);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    float m = NEG_INF_F;
    float l = 0.0f;

    #pragma unroll 1
    for (int tile = 0; tile < NUM_CHUNKS; ++tile) {
        if (tid < 32) idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * 32 + tid];
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid >> 6);
            int load_col = tid & 63;
            int idx = idx_shared[load_row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(ckv_cache_ptr + idx * D_NOPE);
                float4* dst = reinterpret_cast<float4*>(smem_Kc + load_row * D_NOPE);
                dst[load_col] = src[load_col];
            }
        }
        if (tid < 256) {
            int load_row = tid >> 3;
            int load_col = tid & 7;
            int idx = idx_shared[load_row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(kpe_cache_ptr + idx * D_PE);
                float4* dst = reinterpret_cast<float4*>(smem_Kp + load_row * D_PE);
                dst[load_col] = src[load_col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] == -1) continue;

            float local_dot = 0.0f;
            float2 kvals[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(smem_Kc + i * D_NOPE + j * 64 + lane * 2);
                float2 kf = bf162_to_float2(kv);
                kvals[j] = kf;
                local_dot = fmaf(qn[j].x, kf.x, local_dot);
                local_dot = fmaf(qn[j].y, kf.y, local_dot);
            }
            {
                __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(smem_Kp + i * D_PE + lane * 2);
                float2 kf = bf162_to_float2(kv);
                local_dot = fmaf(qp.x, kf.x, local_dot);
                local_dot = fmaf(qp.y, kf.y, local_dot);
            }

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            float m_new = fmaxf(m, logit);
            float alpha = (m == NEG_INF_F) ? 0.0f : __expf(m - m_new);
            float beta = __expf(logit - m_new);
            l = fmaf(l, alpha, beta);
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                O_reg[j].x = fmaf(beta, kvals[j].x, O_reg[j].x * alpha);
                O_reg[j].y = fmaf(beta, kvals[j].y, O_reg[j].y * alpha);
            }
        }
        __syncthreads();
    }

    if (l > 0.0f) {
        float inv_l = __fdividef(1.0f, l);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x *= inv_l;
            O_reg[j].y *= inv_l;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 outv = __floats2bfloat162_rn(O_reg[j].x, O_reg[j].y);
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2) = outv;
    }
    if (lane == 0) {
        lse_ptr[t * NUM_HEADS + h] = (l > 0.0f) ? fmaf(m, LOG2E, __log2f(l)) : NEG_INF_F;
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
    int hg = blockIdx.y;
    int chunk = blockIdx.z;
    int lane = threadIdx.x;
    int local_head = threadIdx.y;
    int h = hg * HEADS_PER_BLOCK + local_head;
    int part = hg * NUM_CHUNKS + chunk;

    __shared__ int idx_shared[CHUNK];
    if (local_head == 0 && lane < CHUNK) {
        idx_shared[lane] = sparse_indices_ptr[t * TOPK + chunk * CHUNK + lane];
    }
    __syncthreads();

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            q_nope_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2);
        qn[j] = bf162_to_float2(qv);
    }
    const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(
        q_pe_ptr + t * NUM_HEADS * D_PE + h * D_PE + lane * 2);
    float2 qp = bf162_to_float2(qpv);

    float2 Oreg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) Oreg[j] = make_float2(0.0f, 0.0f);
    float m = NEG_INF_F;
    float l = 0.0f;

    #pragma unroll
    for (int i = 0; i < CHUNK; ++i) {
        int idx = idx_shared[i];
        if (idx == -1) continue;

        float local_dot = 0.0f;
        float2 kvals[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                ckv_cache_ptr + idx * D_NOPE + j * 64 + lane * 2);
            float2 kf = bf162_to_float2(kv);
            kvals[j] = kf;
            local_dot = fmaf(qn[j].x, kf.x, local_dot);
            local_dot = fmaf(qn[j].y, kf.y, local_dot);
        }
        {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                kpe_cache_ptr + idx * D_PE + lane * 2);
            float2 kf = bf162_to_float2(kv);
            local_dot = fmaf(qp.x, kf.x, local_dot);
            local_dot = fmaf(qp.y, kf.y, local_dot);
        }

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float m_new = fmaxf(m, logit);
        float alpha = (m == NEG_INF_F) ? 0.0f : __expf(m - m_new);
        float beta = __expf(logit - m_new);
        l = fmaf(l, alpha, beta);
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            Oreg[j].x = fmaf(beta, kvals[j].x, Oreg[j].x * alpha);
            Oreg[j].y = fmaf(beta, kvals[j].y, Oreg[j].y * alpha);
        }
    }

    int vec_base = (((t * NUM_HEADS + h) * PARTS + part) * 128 + lane) * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float4 v;
        v.x = Oreg[j * 2 + 0].x;
        v.y = Oreg[j * 2 + 0].y;
        v.z = Oreg[j * 2 + 1].x;
        v.w = Oreg[j * 2 + 1].y;
        *reinterpret_cast<float4*>(O_tmp + vec_base + j * 128) = v;
    }
    if (lane == 0) {
        m_tmp[(t * NUM_HEADS + h) * PARTS + part] = m;
        l_tmp[(t * NUM_HEADS + h) * PARTS + part] = l;
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

    __shared__ float m_global;
    __shared__ float l_global;
    __shared__ float scale[PARTS];

    if (lane == 0) {
        float m = NEG_INF_F;
        #pragma unroll
        for (int p = 0; p < PARTS; ++p) {
            float mp = m_tmp[(t * NUM_HEADS + h) * PARTS + p];
            m = fmaxf(m, mp);
        }
        float l = 0.0f;
        if (m != NEG_INF_F) {
            #pragma unroll
            for (int p = 0; p < PARTS; ++p) {
                float mp = m_tmp[(t * NUM_HEADS + h) * PARTS + p];
                float lp = l_tmp[(t * NUM_HEADS + h) * PARTS + p];
                if (mp != NEG_INF_F) l = fmaf(lp, __expf(mp - m), l);
            }
        }
        m_global = m;
        l_global = l;
        lse_ptr[t * NUM_HEADS + h] = (l > 0.0f) ? fmaf(m, LOG2E, __log2f(l)) : NEG_INF_F;
    }
    __syncthreads();

    if (lane < PARTS) {
        float mp = m_tmp[(t * NUM_HEADS + h) * PARTS + lane];
        scale[lane] = (m_global != NEG_INF_F && l_global > 0.0f && mp != NEG_INF_F)
            ? __fdividef(__expf(mp - m_global), l_global)
            : 0.0f;
    }
    __syncthreads();

    if (lane < 128) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        if (l_global > 0.0f) {
            #pragma unroll
            for (int p = 0; p < PARTS; ++p) {
                float s = scale[p];
                if (s > 0.0f) {
                    int off = (((t * NUM_HEADS + h) * PARTS + p) * 128 + lane) * 4;
                    float4 v = *reinterpret_cast<const float4*>(O_tmp + off);
                    acc.x = fmaf(v.x, s, acc.x);
                    acc.y = fmaf(v.y, s, acc.y);
                    acc.z = fmaf(v.z, s, acc.z);
                    acc.w = fmaf(v.w, s, acc.w);
                }
            }
        }
        int out_d = lane * 4;
        __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y);
        __nv_bfloat162 o1 = __floats2bfloat162_rn(acc.z, acc.w);
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + out_d + 0) = o0;
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + out_d + 2) = o1;
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
    TORCH_CHECK(q_nope.is_cuda(), "q_nope must be CUDA");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be CUDA");
    TORCH_CHECK(ckv_cache.is_cuda(), "ckv_cache must be CUDA");
    TORCH_CHECK(kpe_cache.is_cuda(), "kpe_cache must be CUDA");
    TORCH_CHECK(sparse_indices.is_cuda(), "sparse_indices must be CUDA");
    TORCH_CHECK(q_nope.scalar_type() == torch::kBFloat16, "q_nope must be bf16");
    TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16, "q_pe must be bf16");
    TORCH_CHECK(ckv_cache.scalar_type() == torch::kBFloat16, "ckv_cache must be bf16");
    TORCH_CHECK(kpe_cache.scalar_type() == torch::kBFloat16, "kpe_cache must be bf16");
    TORCH_CHECK(sparse_indices.scalar_type() == torch::kInt32, "sparse_indices must be int32");

    int num_tokens = (int)q_nope.size(0);
    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (num_tokens < 96) {
        auto opts_f = torch::dtype(torch::kFloat32).device(q_nope.device());
        auto O_tmp = torch::empty({num_tokens, NUM_HEADS, PARTS, 128, 4}, opts_f);
        auto m_tmp = torch::empty({num_tokens, NUM_HEADS, PARTS}, opts_f);
        auto l_tmp = torch::empty({num_tokens, NUM_HEADS, PARTS}, opts_f);

        dim3 grid(num_tokens, HEAD_GROUPS, NUM_CHUNKS);
        dim3 block(32, HEADS_PER_BLOCK, 1);
        small_partial_kernel<<<grid, block, 0, stream>>>(
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

        dim3 rgrid(num_tokens, NUM_HEADS, 1);
        dim3 rblock(128, 1, 1);
        small_reduce_kernel<<<rgrid, rblock, 0, stream>>>(
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>()
        );
    } else {
        dim3 grid(num_tokens, 1, 1);
        dim3 block(32, NUM_HEADS, 1);
        dsa_forward_large_kernel<<<grid, block, 0, stream>>>(
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
