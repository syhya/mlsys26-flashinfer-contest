#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == torch::kBFloat16, #x " must be bfloat16")
#define CHECK_I32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

static constexpr int NUM_HEADS = 16;
static constexpr int D_NOPE = 512;
static constexpr int D_PE = 64;
static constexpr int TOPK = 2048;
static constexpr int PAGE_SIZE = 64;
static constexpr int CHUNK_KEYS = 32;
static constexpr int NUM_CHUNKS = TOPK / CHUNK_KEYS;
static constexpr float NEG_INF_F = -1.0e20f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void dsa_forward_kernel_main(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr,
    int num_tokens
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    int lane = threadIdx.x;
    int h = threadIdx.y;
    int tid = h * 32 + lane;

    __shared__ int idx_shared[CHUNK_KEYS];
    __shared__ __align__(16) __nv_bfloat16 smem_kc[CHUNK_KEYS * D_NOPE];
    __shared__ __align__(16) __nv_bfloat16 smem_kp[CHUNK_KEYS * D_PE];

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        auto v = *reinterpret_cast<const __nv_bfloat162*>(
            q_nope_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2);
        qn[j] = __bfloat1622float2(v);
    }
    float2 qp = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
        q_pe_ptr + t * NUM_HEADS * D_PE + h * D_PE + lane * 2));

    float2 acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.f, 0.f);

    float m = NEG_INF_F;
    float l = 0.f;

    #pragma unroll 1
    for (int tile = 0; tile < NUM_CHUNKS; ++tile) {
        if (tid < CHUNK_KEYS) idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * CHUNK_KEYS + tid];
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int row = step * 8 + (tid >> 6);
            int col = tid & 63;
            int idx = idx_shared[row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(ckv_cache_ptr + idx * D_NOPE);
                float4* dst = reinterpret_cast<float4*>(smem_kc + row * D_NOPE);
                dst[col] = src[col];
            }
        }
        if (tid < 256) {
            int row = tid >> 3;
            int col = tid & 7;
            int idx = idx_shared[row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(kpe_cache_ptr + idx * D_PE);
                float4* dst = reinterpret_cast<float4*>(smem_kp + row * D_PE);
                dst[col] = src[col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < CHUNK_KEYS; ++i) {
            if (idx_shared[i] == -1) continue;

            float dot = 0.f;
            float2 kval[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                auto kv = *reinterpret_cast<const __nv_bfloat162*>(smem_kc + i * D_NOPE + j * 64 + lane * 2);
                kval[j] = __bfloat1622float2(kv);
                dot = fmaf(qn[j].x, kval[j].x, dot);
                dot = fmaf(qn[j].y, kval[j].y, dot);
            }
            float2 kp = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(smem_kp + i * D_PE + lane * 2));
            dot = fmaf(qp.x, kp.x, dot);
            dot = fmaf(qp.y, kp.y, dot);

            float logit = warp_reduce_sum(dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            float m_new = fmaxf(m, logit);
            float alpha = (m <= -1.0e19f) ? 0.f : __expf(m - m_new);
            float p = __expf(logit - m_new);
            l = l * alpha + p;
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j].x = acc[j].x * alpha + p * kval[j].x;
                acc[j].y = acc[j].y * alpha + p * kval[j].y;
            }
        }
        __syncthreads();
    }

    float inv_l = l > 0.f ? __fdividef(1.f, l) : 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j].x *= inv_l;
        acc[j].y *= inv_l;
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2) =
            __floats2bfloat162_rn(acc[j].x, acc[j].y);
    }
    if (lane == 0) {
        lse_ptr[t * NUM_HEADS + h] = (l > 0.f) ? (m * 1.4426950408889634f + __log2f(l)) : NEG_INF_F;
    }
}

// Devised from parent guidance: small-workload path keeps split-K, but fixes temporary output writes
// to contiguous float4 vectors so they match reduction loads and reduce store inefficiency.
__global__ void split_k_compute_kernel_v2(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float4* __restrict__ o_tmp4,
    float* __restrict__ m_tmp,
    float* __restrict__ l_tmp,
    int num_tokens
) {
    int s = blockIdx.x;
    int t = blockIdx.y;
    if (t >= num_tokens) return;

    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = h * 32 + lane;

    __shared__ int idx_shared[CHUNK_KEYS];
    __shared__ __align__(16) __nv_bfloat16 smem_kc[CHUNK_KEYS * D_NOPE];
    __shared__ __align__(16) __nv_bfloat16 smem_kp[CHUNK_KEYS * D_PE];

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        auto v = *reinterpret_cast<const __nv_bfloat162*>(
            q_nope_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2);
        qn[j] = __bfloat1622float2(v);
    }
    float2 qp = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
        q_pe_ptr + t * NUM_HEADS * D_PE + h * D_PE + lane * 2));

    if (tid < CHUNK_KEYS) idx_shared[tid] = sparse_indices_ptr[t * TOPK + s * CHUNK_KEYS + tid];
    __syncthreads();

    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        int row = step * 8 + (tid >> 6);
        int col = tid & 63;
        int idx = idx_shared[row];
        if (idx != -1) {
            const float4* src = reinterpret_cast<const float4*>(ckv_cache_ptr + idx * D_NOPE);
            float4* dst = reinterpret_cast<float4*>(smem_kc + row * D_NOPE);
            dst[col] = src[col];
        }
    }
    if (tid < 256) {
        int row = tid >> 3;
        int col = tid & 7;
        int idx = idx_shared[row];
        if (idx != -1) {
            const float4* src = reinterpret_cast<const float4*>(kpe_cache_ptr + idx * D_PE);
            float4* dst = reinterpret_cast<float4*>(smem_kp + row * D_PE);
            dst[col] = src[col];
        }
    }
    __syncthreads();

    float2 acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.f, 0.f);
    float m = NEG_INF_F;
    float l = 0.f;

    #pragma unroll
    for (int i = 0; i < CHUNK_KEYS; ++i) {
        if (idx_shared[i] == -1) continue;

        float dot = 0.f;
        float2 kval[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            auto kv = *reinterpret_cast<const __nv_bfloat162*>(smem_kc + i * D_NOPE + j * 64 + lane * 2);
            kval[j] = __bfloat1622float2(kv);
            dot = fmaf(qn[j].x, kval[j].x, dot);
            dot = fmaf(qn[j].y, kval[j].y, dot);
        }
        float2 kp = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(smem_kp + i * D_PE + lane * 2));
        dot = fmaf(qp.x, kp.x, dot);
        dot = fmaf(qp.y, kp.y, dot);

        float logit = warp_reduce_sum(dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float m_new = fmaxf(m, logit);
        float alpha = (m <= -1.0e19f) ? 0.f : __expf(m - m_new);
        float p = __expf(logit - m_new);
        l = l * alpha + p;
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j].x = acc[j].x * alpha + p * kval[j].x;
            acc[j].y = acc[j].y * alpha + p * kval[j].y;
        }
    }

    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        int j = group * 2;
        size_t idx = (((size_t)t * NUM_CHUNKS + s) * NUM_HEADS + h) * 128 + group * 32 + lane;
        o_tmp4[idx] = make_float4(acc[j].x, acc[j].y, acc[j + 1].x, acc[j + 1].y);
    }

    if (lane == 0) {
        m_tmp[(t * NUM_HEADS + h) * NUM_CHUNKS + s] = m;
        l_tmp[(t * NUM_HEADS + h) * NUM_CHUNKS + s] = l;
    }
}

__global__ void split_k_reduce_kernel_v2(
    const float4* __restrict__ o_tmp4,
    const float* __restrict__ m_tmp,
    const float* __restrict__ l_tmp,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr,
    int num_tokens
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= num_tokens) return;

    int lane = threadIdx.x;

    __shared__ float ms[NUM_CHUNKS];
    __shared__ float ls[NUM_CHUNKS];
    __shared__ float scales[NUM_CHUNKS];
    __shared__ float mglob;
    __shared__ float lglob;

    if (lane < NUM_CHUNKS) {
        ms[lane] = m_tmp[(t * NUM_HEADS + h) * NUM_CHUNKS + lane];
        ls[lane] = l_tmp[(t * NUM_HEADS + h) * NUM_CHUNKS + lane];
    }
    __syncthreads();

    if (lane == 0) {
        float m = NEG_INF_F;
        #pragma unroll
        for (int i = 0; i < NUM_CHUNKS; ++i) m = fmaxf(m, ms[i]);
        float l = 0.f;
        #pragma unroll
        for (int i = 0; i < NUM_CHUNKS; ++i) {
            if (ms[i] > -1.0e19f) l = fmaf(ls[i], __expf(ms[i] - m), l);
        }
        mglob = m;
        lglob = l;
        lse_ptr[t * NUM_HEADS + h] = (l > 0.f) ? (m * 1.4426950408889634f + __log2f(l)) : NEG_INF_F;
    }
    __syncthreads();

    if (lane < NUM_CHUNKS) {
        float scale = 0.f;
        if (lglob > 0.f && ms[lane] > -1.0e19f) scale = __expf(ms[lane] - mglob) / lglob;
        scales[lane] = scale;
    }
    __syncthreads();

    if (lane < 128) {
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll 16
        for (int s = 0; s < NUM_CHUNKS; ++s) {
            float scale = scales[s];
            if (scale > 0.f) {
                size_t idx = (((size_t)t * NUM_CHUNKS + s) * NUM_HEADS + h) * 128 + lane;
                float4 v = o_tmp4[idx];
                sum.x = fmaf(v.x, scale, sum.x);
                sum.y = fmaf(v.y, scale, sum.y);
                sum.z = fmaf(v.z, scale, sum.z);
                sum.w = fmaf(v.w, scale, sum.w);
            }
        }

        int d = lane * 4;
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + d + 0) =
            __floats2bfloat162_rn(sum.x, sum.y);
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + d + 2) =
            __floats2bfloat162_rn(sum.z, sum.w);
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
    CHECK_CUDA(q_nope);
    CHECK_CUDA(q_pe);
    CHECK_CUDA(ckv_cache);
    CHECK_CUDA(kpe_cache);
    CHECK_CUDA(sparse_indices);
    CHECK_CONTIGUOUS(q_nope);
    CHECK_CONTIGUOUS(q_pe);
    CHECK_CONTIGUOUS(ckv_cache);
    CHECK_CONTIGUOUS(kpe_cache);
    CHECK_CONTIGUOUS(sparse_indices);
    CHECK_BF16(q_nope);
    CHECK_BF16(q_pe);
    CHECK_BF16(ckv_cache);
    CHECK_BF16(kpe_cache);
    CHECK_I32(sparse_indices);

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == NUM_HEADS && q_nope.size(2) == D_NOPE, "q_nope shape mismatch");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == NUM_HEADS && q_pe.size(2) == D_PE, "q_pe shape mismatch");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == PAGE_SIZE && ckv_cache.size(2) == D_NOPE, "ckv_cache shape mismatch");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == PAGE_SIZE && kpe_cache.size(2) == D_PE, "kpe_cache shape mismatch");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == TOPK, "sparse_indices shape mismatch");
    TORCH_CHECK(q_nope.size(0) == q_pe.size(0) && q_nope.size(0) == sparse_indices.size(0), "num_tokens mismatch");

    int num_tokens = q_nope.size(0);
    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    if (num_tokens < 128) {
        auto o_tmp = torch::empty({num_tokens, NUM_CHUNKS, NUM_HEADS, 128, 4}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto m_tmp = torch::empty({num_tokens, NUM_HEADS, NUM_CHUNKS}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto l_tmp = torch::empty({num_tokens, NUM_HEADS, NUM_CHUNKS}, torch::dtype(torch::kFloat32).device(q_nope.device()));

        dim3 grid_compute(NUM_CHUNKS, num_tokens, 1);
        dim3 block_compute(32, NUM_HEADS, 1);
        split_k_compute_kernel_v2<<<grid_compute, block_compute>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            reinterpret_cast<float4*>(o_tmp.data_ptr<float>()),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            num_tokens);

        dim3 grid_reduce(num_tokens, NUM_HEADS, 1);
        dim3 block_reduce(128, 1, 1);
        split_k_reduce_kernel_v2<<<grid_reduce, block_reduce>>>(
            reinterpret_cast<const float4*>(o_tmp.data_ptr<float>()),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            num_tokens);
    } else {
        dim3 grid(num_tokens, 1, 1);
        dim3 block(32, NUM_HEADS, 1);
        dsa_forward_kernel_main<<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            num_tokens);
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}
