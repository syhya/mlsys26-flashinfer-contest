#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK(x.scalar_type() == dt, #x " has wrong dtype")

static constexpr int NUM_HEADS = 16;
static constexpr int D_NOPE = 512;
static constexpr int D_PE = 64;
static constexpr int TOPK = 2048;
static constexpr int TILE_KEYS = 32;
static constexpr float LOG2E_F = 1.4426950408889634f;
static constexpr float NEG_INF_F = -1.0e30f;

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

__global__ void dsa_forward_fused_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    const int t = blockIdx.x;
    const int h = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = h * 32 + lane;

    __shared__ alignas(16) int idx_shared[TILE_KEYS];
    __shared__ alignas(16) __nv_bfloat16 smem_kc[TILE_KEYS * D_NOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_kp[TILE_KEYS * D_PE];

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const auto v = *reinterpret_cast<const __nv_bfloat162*>(
            q_nope_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2);
        qn[j] = bf162_to_float2(v);
    }
    const float2 qp = bf162_to_float2(*reinterpret_cast<const __nv_bfloat162*>(
        q_pe_ptr + t * NUM_HEADS * D_PE + h * D_PE + lane * 2));

    float2 oreg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) oreg[j] = make_float2(0.f, 0.f);

    float m = NEG_INF_F;
    float l = 0.f;

    #pragma unroll 1
    for (int tile = 0; tile < TOPK / TILE_KEYS; ++tile) {
        if (tid < TILE_KEYS) idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * TILE_KEYS + tid];
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int row = step * 8 + (tid >> 6);
            int col = tid & 63;
            int idx = idx_shared[row];
            if (idx != -1) {
                reinterpret_cast<float4*>(smem_kc + row * D_NOPE)[col] =
                    reinterpret_cast<const float4*>(ckv_cache_ptr + idx * D_NOPE)[col];
            }
        }
        if (tid < 256) {
            int row = tid >> 3;
            int col = tid & 7;
            int idx = idx_shared[row];
            if (idx != -1) {
                reinterpret_cast<float4*>(smem_kp + row * D_PE)[col] =
                    reinterpret_cast<const float4*>(kpe_cache_ptr + idx * D_PE)[col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_KEYS; ++i) {
            if (idx_shared[i] == -1) continue;

            float dot = 0.f;
            float2 kval[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const auto kn = *reinterpret_cast<const __nv_bfloat162*>(smem_kc + i * D_NOPE + j * 64 + lane * 2);
                const float2 kf = bf162_to_float2(kn);
                kval[j] = kf;
                dot = fmaf(qn[j].x, kf.x, dot);
                dot = fmaf(qn[j].y, kf.y, dot);
            }
            {
                const auto kp = *reinterpret_cast<const __nv_bfloat162*>(smem_kp + i * D_PE + lane * 2);
                const float2 kf = bf162_to_float2(kp);
                dot = fmaf(qp.x, kf.x, dot);
                dot = fmaf(qp.y, kf.y, dot);
            }

            float logit = warp_reduce_sum(dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            float m_new = fmaxf(m, logit);
            float alpha = (m < -1.0e20f) ? 0.f : __expf(m - m_new);
            float beta = __expf(logit - m_new);
            l = fmaf(l, alpha, beta);
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                oreg[j].x = fmaf(beta, kval[j].x, oreg[j].x * alpha);
                oreg[j].y = fmaf(beta, kval[j].y, oreg[j].y * alpha);
            }
        }
        __syncthreads();
    }

    float inv_l = (l > 0.f) ? __fdividef(1.f, l) : 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        oreg[j].x *= inv_l;
        oreg[j].y *= inv_l;
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2) =
            __floats2bfloat162_rn(oreg[j].x, oreg[j].y);
    }
    if (lane == 0) lse_ptr[t * NUM_HEADS + h] = (l > 0.f) ? (m * LOG2E_F + __log2f(l)) : NEG_INF_F;
}

__global__ void split_k_compute_kernel_vec4(
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
    const int s = blockIdx.x;
    const int t = blockIdx.y;
    const int h = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = h * 32 + lane;

    __shared__ alignas(16) int idx_shared[TILE_KEYS];
    __shared__ alignas(16) __nv_bfloat16 smem_kc[TILE_KEYS * D_NOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_kp[TILE_KEYS * D_PE];

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const auto v = *reinterpret_cast<const __nv_bfloat162*>(
            q_nope_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2);
        qn[j] = bf162_to_float2(v);
    }
    const float2 qp = bf162_to_float2(*reinterpret_cast<const __nv_bfloat162*>(
        q_pe_ptr + t * NUM_HEADS * D_PE + h * D_PE + lane * 2));

    float2 oreg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) oreg[j] = make_float2(0.f, 0.f);
    float m = NEG_INF_F;
    float l = 0.f;

    if (tid < TILE_KEYS) idx_shared[tid] = sparse_indices_ptr[t * TOPK + s * TILE_KEYS + tid];
    __syncthreads();

    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        int row = step * 8 + (tid >> 6);
        int col = tid & 63;
        int idx = idx_shared[row];
        if (idx != -1) {
            reinterpret_cast<float4*>(smem_kc + row * D_NOPE)[col] =
                reinterpret_cast<const float4*>(ckv_cache_ptr + idx * D_NOPE)[col];
        }
    }
    if (tid < 256) {
        int row = tid >> 3;
        int col = tid & 7;
        int idx = idx_shared[row];
        if (idx != -1) {
            reinterpret_cast<float4*>(smem_kp + row * D_PE)[col] =
                reinterpret_cast<const float4*>(kpe_cache_ptr + idx * D_PE)[col];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TILE_KEYS; ++i) {
        if (idx_shared[i] == -1) continue;

        float dot = 0.f;
        float2 kval[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const auto kn = *reinterpret_cast<const __nv_bfloat162*>(smem_kc + i * D_NOPE + j * 64 + lane * 2);
            const float2 kf = bf162_to_float2(kn);
            kval[j] = kf;
            dot = fmaf(qn[j].x, kf.x, dot);
            dot = fmaf(qn[j].y, kf.y, dot);
        }
        {
            const auto kp = *reinterpret_cast<const __nv_bfloat162*>(smem_kp + i * D_PE + lane * 2);
            const float2 kf = bf162_to_float2(kp);
            dot = fmaf(qp.x, kf.x, dot);
            dot = fmaf(qp.y, kf.y, dot);
        }

        float logit = warp_reduce_sum(dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float m_new = fmaxf(m, logit);
        float alpha = (m < -1.0e20f) ? 0.f : __expf(m - m_new);
        float beta = __expf(logit - m_new);
        l = fmaf(l, alpha, beta);
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            oreg[j].x = fmaf(beta, kval[j].x, oreg[j].x * alpha);
            oreg[j].y = fmaf(beta, kval[j].y, oreg[j].y * alpha);
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int base_idx = (((t * 64 + s) * NUM_HEADS + h) * D_NOPE) + j * 128 + lane * 4;
        float4 v = make_float4(oreg[2 * j + 0].x, oreg[2 * j + 0].y, oreg[2 * j + 1].x, oreg[2 * j + 1].y);
        *reinterpret_cast<float4*>(O_tmp + base_idx) = v;
    }
    if (lane == 0) {
        m_tmp[(t * NUM_HEADS + h) * 64 + s] = m;
        l_tmp[(t * NUM_HEADS + h) * 64 + s] = l;
    }
}

__global__ void split_k_reduce_kernel(
    const float* __restrict__ O_tmp,
    const float* __restrict__ m_tmp,
    const float* __restrict__ l_tmp,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;

    __shared__ float mvals[64];
    __shared__ float lvals[64];
    __shared__ float scales[64];
    __shared__ float m_global;
    __shared__ float l_global;

    if (lane < 64) {
        mvals[lane] = m_tmp[(t * NUM_HEADS + h) * 64 + lane];
        lvals[lane] = l_tmp[(t * NUM_HEADS + h) * 64 + lane];
    }
    __syncthreads();

    if (lane == 0) {
        float mmax = NEG_INF_F;
        #pragma unroll
        for (int i = 0; i < 64; ++i) mmax = fmaxf(mmax, mvals[i]);
        float lsum = 0.f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            if (mvals[i] > -1.0e20f) lsum = fmaf(lvals[i], __expf(mvals[i] - mmax), lsum);
        }
        m_global = mmax;
        l_global = lsum;
        lse_ptr[t * NUM_HEADS + h] = (mmax > -1.0e20f && lsum > 0.f) ? (mmax * LOG2E_F + __log2f(lsum)) : NEG_INF_F;
    }
    __syncthreads();

    if (lane < 64) {
        scales[lane] = (m_global > -1.0e20f && l_global > 0.f && mvals[lane] > -1.0e20f)
            ? __fdividef(__expf(mvals[lane] - m_global), l_global) : 0.f;
    }
    __syncthreads();

    int d = lane * 4;
    if (d < D_NOPE) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll 16
        for (int s = 0; s < 64; ++s) {
            float scale = scales[s];
            if (scale > 0.f) {
                int base_idx = (((t * 64 + s) * NUM_HEADS + h) * D_NOPE) + d;
                float4 v = *reinterpret_cast<const float4*>(O_tmp + base_idx);
                acc.x = fmaf(v.x, scale, acc.x);
                acc.y = fmaf(v.y, scale, acc.y);
                acc.z = fmaf(v.z, scale, acc.z);
                acc.w = fmaf(v.w, scale, acc.w);
            }
        }
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + d + 0) =
            __floats2bfloat162_rn(acc.x, acc.y);
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + t * NUM_HEADS * D_NOPE + h * D_NOPE + d + 2) =
            __floats2bfloat162_rn(acc.z, acc.w);
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
    CHECK_DTYPE(q_nope, torch::kBFloat16);
    CHECK_DTYPE(q_pe, torch::kBFloat16);
    CHECK_DTYPE(ckv_cache, torch::kBFloat16);
    CHECK_DTYPE(kpe_cache, torch::kBFloat16);
    CHECK_DTYPE(sparse_indices, torch::kInt32);

    const int num_tokens = q_nope.size(0);
    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    if (num_tokens < 128) {
        auto O_tmp = torch::empty({num_tokens, 64, NUM_HEADS, D_NOPE}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto m_tmp = torch::empty({num_tokens, NUM_HEADS, 64}, torch::dtype(torch::kFloat32).device(q_nope.device()));
        auto l_tmp = torch::empty({num_tokens, NUM_HEADS, 64}, torch::dtype(torch::kFloat32).device(q_nope.device()));

        dim3 grid_compute(64, num_tokens, 1);
        dim3 block_compute(32, NUM_HEADS, 1);
        split_k_compute_kernel_vec4<<<grid_compute, block_compute>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>());

        dim3 grid_reduce(num_tokens, NUM_HEADS, 1);
        dim3 block_reduce(128, 1, 1);
        split_k_reduce_kernel<<<grid_reduce, block_reduce>>>(
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>());
    } else {
        dim3 grid(num_tokens, 1, 1);
        dim3 block(32, NUM_HEADS, 1);
        dsa_forward_fused_kernel<<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>());
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}
