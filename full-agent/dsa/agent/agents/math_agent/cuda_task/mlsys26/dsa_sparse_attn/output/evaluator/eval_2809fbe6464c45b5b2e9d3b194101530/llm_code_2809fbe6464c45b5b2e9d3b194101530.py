#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).scalar_type() == (dt), #x " has wrong dtype")

static constexpr int HEADS = 16;
static constexpr int DNOPE = 512;
static constexpr int DPE = 64;
static constexpr int TOPK = 2048;
static constexpr int PAGE = 64;
static constexpr float LOG2E_F = 1.4426950408889634f;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ int decode_row_index(int idx) {
    int page = idx >> 6;
    int offset = idx & 63;
    return page * 64 + offset;
}

// Low-latency path: one warp computes one (token, head), no shared-memory staging.
__global__ void dsa_forward_small_kernel(
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
    int h = blockIdx.y;
    int lane = threadIdx.x;
    if (t >= num_tokens || h >= HEADS) return;

    const __nv_bfloat16* qn = q_nope_ptr + ((int64_t)t * HEADS + h) * DNOPE;
    const __nv_bfloat16* qp = q_pe_ptr + ((int64_t)t * HEADS + h) * DPE;
    const int32_t* idx_ptr = sparse_indices_ptr + (int64_t)t * TOPK;

    float2 qn_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(qn + j * 64 + lane * 2);
        qn_reg[j] = __bfloat1622float2(v);
    }
    const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(qp + lane * 2);
    const float2 qp_reg = __bfloat1622float2(qpv);

    float2 acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    #pragma unroll 1
    for (int i = 0; i < TOPK; ++i) {
        int idx = idx_ptr[i];
        if (idx < 0) continue;
        ++valid_count;
        int row = decode_row_index(idx);
        const __nv_bfloat16* kn = ckv_cache_ptr + (int64_t)row * DNOPE;
        const __nv_bfloat16* kp = kpe_cache_ptr + (int64_t)row * DPE;

        float local_dot = 0.0f;
        float2 kv_reg[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(kn + j * 64 + lane * 2);
            const float2 kf = __bfloat1622float2(kv);
            kv_reg[j] = kf;
            local_dot = fmaf(qn_reg[j].x, kf.x, local_dot);
            local_dot = fmaf(qn_reg[j].y, kf.y, local_dot);
        }
        const __nv_bfloat162 kpv = *reinterpret_cast<const __nv_bfloat162*>(kp + lane * 2);
        const float2 kpf = __bfloat1622float2(kpv);
        local_dot = fmaf(qp_reg.x, kpf.x, local_dot);
        local_dot = fmaf(qp_reg.y, kpf.y, local_dot);

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float m_new = fmaxf(m, logit);
        float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        float beta = expf(logit - m_new);
        l = l * alpha + beta;
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j].x = acc[j].x * alpha + beta * kv_reg[j].x;
            acc[j].y = acc[j].y * alpha + beta * kv_reg[j].y;
        }
    }

    __nv_bfloat16* out = output_ptr + ((int64_t)t * HEADS + h) * DNOPE;
    if (valid_count == 0 || l <= 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(out + j * 64 + lane * 2) = __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse_ptr[t * HEADS + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(out + j * 64 + lane * 2) =
            __floats2bfloat162_rn(acc[j].x * inv_l, acc[j].y * inv_l);
    }
    if (lane == 0) lse_ptr[t * HEADS + h] = m * LOG2E_F + log2f(l);
}

// Throughput path from parent: shared staging for 32 sparse rows across all 16 heads of a token.
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
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * DNOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * DPE];

    const __nv_bfloat16* qn = q_nope_ptr + ((int64_t)t * HEADS + h) * DNOPE;
    const __nv_bfloat16* qp = q_pe_ptr + ((int64_t)t * HEADS + h) * DPE;

    float2 qn_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(qn + j * 64 + lane * 2);
        qn_reg[j] = __bfloat1622float2(v);
    }
    const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(qp + lane * 2);
    const float2 qp_reg = __bfloat1622float2(qpv);

    float2 acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    #pragma unroll 1
    for (int tile = 0; tile < TOPK / 32; ++tile) {
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[(int64_t)t * TOPK + tile * 32 + tid];
        }
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid >> 6);
            int load_col = tid & 63;
            int idx = idx_shared[load_row];
            if (idx >= 0) {
                int row = decode_row_index(idx);
                const float4* src = reinterpret_cast<const float4*>(ckv_cache_ptr + (int64_t)row * DNOPE);
                float4* dst = reinterpret_cast<float4*>(smem_Kc + load_row * DNOPE);
                dst[load_col] = src[load_col];
            }
        }

        if (tid < 256) {
            int load_row = tid >> 3;
            int load_col = tid & 7;
            int idx = idx_shared[load_row];
            if (idx >= 0) {
                int row = decode_row_index(idx);
                const float4* src = reinterpret_cast<const float4*>(kpe_cache_ptr + (int64_t)row * DPE);
                float4* dst = reinterpret_cast<float4*>(smem_Kp + load_row * DPE);
                dst[load_col] = src[load_col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] < 0) continue;
            ++valid_count;

            float local_dot = 0.0f;
            float2 kv_reg[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(smem_Kc + i * DNOPE + j * 64 + lane * 2);
                const float2 kf = __bfloat1622float2(kv);
                kv_reg[j] = kf;
                local_dot = fmaf(qn_reg[j].x, kf.x, local_dot);
                local_dot = fmaf(qn_reg[j].y, kf.y, local_dot);
            }
            const __nv_bfloat162 kpv2 = *reinterpret_cast<const __nv_bfloat162*>(smem_Kp + i * DPE + lane * 2);
            const float2 kpf = __bfloat1622float2(kpv2);
            local_dot = fmaf(qp_reg.x, kpf.x, local_dot);
            local_dot = fmaf(qp_reg.y, kpf.y, local_dot);

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            float m_new = fmaxf(m, logit);
            float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
            float beta = expf(logit - m_new);
            l = l * alpha + beta;
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j].x = acc[j].x * alpha + beta * kv_reg[j].x;
                acc[j].y = acc[j].y * alpha + beta * kv_reg[j].y;
            }
        }
        __syncthreads();
    }

    __nv_bfloat16* out = output_ptr + ((int64_t)t * HEADS + h) * DNOPE;
    if (valid_count == 0 || l <= 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(out + j * 64 + lane * 2) = __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse_ptr[t * HEADS + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(out + j * 64 + lane * 2) =
            __floats2bfloat162_rn(acc[j].x * inv_l, acc[j].y * inv_l);
    }
    if (lane == 0) lse_ptr[t * HEADS + h] = m * LOG2E_F + log2f(l);
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

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == HEADS && q_nope.size(2) == DNOPE, "q_nope shape must be [T,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == HEADS && q_pe.size(2) == DPE, "q_pe shape must be [T,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == PAGE && ckv_cache.size(2) == DNOPE, "ckv_cache shape must be [P,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == PAGE && kpe_cache.size(2) == DPE, "kpe_cache shape must be [P,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == TOPK, "sparse_indices shape must be [T,2048]");
    TORCH_CHECK(q_nope.size(0) == q_pe.size(0) && q_nope.size(0) == sparse_indices.size(0), "token dimensions must match");
    TORCH_CHECK(ckv_cache.size(0) == kpe_cache.size(0), "cache page dimensions must match");

    int num_tokens = (int)q_nope.size(0);
    auto output = torch::empty({num_tokens, HEADS, DNOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, HEADS}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    // Devised from plan: remove parent split-K small-token path because its temporary
    // O/m/l global workspace can dominate tiny-workload latency; use fused warp-per-head.
    if (num_tokens <= 32) {
        dim3 grid(num_tokens, HEADS, 1);
        dim3 block(32, 1, 1);
        dsa_forward_small_kernel<<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            num_tokens
        );
    } else {
        dim3 grid(num_tokens, 1, 1);
        dim3 block(32, HEADS, 1);
        dsa_forward_large_kernel<<<grid, block>>>(
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
