#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <cmath>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).scalar_type() == (dt), #x " has wrong dtype")

static constexpr int NUM_HEADS = 16;
static constexpr int D_NOPE = 512;
static constexpr int D_PE = 64;
static constexpr int TOPK = 2048;
static constexpr int PAGE_SIZE = 64;
static constexpr int SPLIT = 4;
static constexpr int CHUNK = TOPK / SPLIT;  // 512
static constexpr float LOG2E_F = 1.4426950408889634f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ int decode_row_index(int idx) {
    int page = idx >> 6;
    int offset = idx & 63;
    return page * PAGE_SIZE + offset;
}

// Split-K partial kernel: one warp computes one (token, head, split).
// Produces partial online-softmax state: m, l, and unnormalized weighted value sum O.
__global__ void dsa_splitk_stage1_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float* __restrict__ partial_m_ptr,   // [T,16,SPLIT]
    float* __restrict__ partial_l_ptr,   // [T,16,SPLIT]
    float* __restrict__ partial_o_ptr,   // [T,16,SPLIT,512]
    int num_tokens
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z;
    int lane = threadIdx.x;
    if (t >= num_tokens || h >= NUM_HEADS || s >= SPLIT) return;

    float2 q_n_f32[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            q_nope_ptr + ((t * NUM_HEADS + h) * D_NOPE + j * 64 + lane * 2));
        q_n_f32[j] = __bfloat1622float2(qv);
    }
    const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(
        q_pe_ptr + ((t * NUM_HEADS + h) * D_PE + lane * 2));
    const float2 q_p_f32 = __bfloat1622float2(qpv);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    int start = s * CHUNK;
    int end = start + CHUNK;
    int base_sparse = t * TOPK;

    #pragma unroll 1
    for (int i = start; i < end; ++i) {
        int idx = sparse_indices_ptr[base_sparse + i];
        if (idx < 0) continue;
        ++valid_count;
        int row = decode_row_index(idx);

        float local_dot = 0.0f;
        float2 k_f_reg[8];

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                ckv_cache_ptr + row * D_NOPE + j * 64 + lane * 2);
            const float2 kf = __bfloat1622float2(kv);
            k_f_reg[j] = kf;
            local_dot = fmaf(q_n_f32[j].x, kf.x, local_dot);
            local_dot = fmaf(q_n_f32[j].y, kf.y, local_dot);
        }
        {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                kpe_cache_ptr + row * D_PE + lane * 2);
            const float2 kf = __bfloat1622float2(kv);
            local_dot = fmaf(q_p_f32.x, kf.x, local_dot);
            local_dot = fmaf(q_p_f32.y, kf.y, local_dot);
        }

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float m_new = fmaxf(m, logit);
        float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        float beta = expf(logit - m_new);
        l = l * alpha + beta;
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x = O_reg[j].x * alpha + beta * k_f_reg[j].x;
            O_reg[j].y = O_reg[j].y * alpha + beta * k_f_reg[j].y;
        }
    }

    int partial_base = ((t * NUM_HEADS + h) * SPLIT + s);
    if (valid_count == 0 || l <= 0.0f) {
        if (lane == 0) {
            partial_m_ptr[partial_base] = -INFINITY;
            partial_l_ptr[partial_base] = 0.0f;
        }
        float* o_dst = partial_o_ptr + partial_base * D_NOPE;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            o_dst[j * 64 + lane * 2 + 0] = 0.0f;
            o_dst[j * 64 + lane * 2 + 1] = 0.0f;
        }
        return;
    }

    if (lane == 0) {
        partial_m_ptr[partial_base] = m;
        partial_l_ptr[partial_base] = l;
    }
    float* o_dst = partial_o_ptr + partial_base * D_NOPE;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        o_dst[j * 64 + lane * 2 + 0] = O_reg[j].x;
        o_dst[j * 64 + lane * 2 + 1] = O_reg[j].y;
    }
}

// Reduction kernel: merge SPLIT partial online-softmax states exactly.
__global__ void dsa_splitk_stage2_kernel(
    const float* __restrict__ partial_m_ptr,   // [T,16,SPLIT]
    const float* __restrict__ partial_l_ptr,   // [T,16,SPLIT]
    const float* __restrict__ partial_o_ptr,   // [T,16,SPLIT,512]
    __nv_bfloat16* __restrict__ output_ptr,    // [T,16,512]
    float* __restrict__ lse_ptr,               // [T,16]
    int num_tokens
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int lane = threadIdx.x;
    if (t >= num_tokens || h >= NUM_HEADS) return;

    int base_th = (t * NUM_HEADS + h) * SPLIT;

    float m = -INFINITY;
    #pragma unroll
    for (int s = 0; s < SPLIT; ++s) {
        m = fmaxf(m, partial_m_ptr[base_th + s]);
    }

    if (m == -INFINITY) {
        int out_base = (t * NUM_HEADS + h) * D_NOPE;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base + j * 64 + lane * 2) =
                __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse_ptr[t * NUM_HEADS + h] = -INFINITY;
        return;
    }

    float l = 0.0f;
    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    #pragma unroll
    for (int s = 0; s < SPLIT; ++s) {
        float ms = partial_m_ptr[base_th + s];
        float ls = partial_l_ptr[base_th + s];
        if (ls <= 0.0f || ms == -INFINITY) continue;
        float scale = expf(ms - m);
        l += ls * scale;

        const float* o_src = partial_o_ptr + (base_th + s) * D_NOPE;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x += o_src[j * 64 + lane * 2 + 0] * scale;
            O_reg[j].y += o_src[j * 64 + lane * 2 + 1] * scale;
        }
    }

    int out_base = (t * NUM_HEADS + h) * D_NOPE;
    if (l <= 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base + j * 64 + lane * 2) =
                __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse_ptr[t * NUM_HEADS + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base + j * 64 + lane * 2) =
            __floats2bfloat162_rn(O_reg[j].x * inv_l, O_reg[j].y * inv_l);
    }
    if (lane == 0) lse_ptr[t * NUM_HEADS + h] = m * LOG2E_F + log2f(l);
}

// Fused warp-per-head kernel for very small workloads.
__global__ void dsa_fused_small_kernel(
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
    if (t >= num_tokens || h >= NUM_HEADS) return;

    float2 q_n_f32[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            q_nope_ptr + ((t * NUM_HEADS + h) * D_NOPE + j * 64 + lane * 2));
        q_n_f32[j] = __bfloat1622float2(qv);
    }
    const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(
        q_pe_ptr + ((t * NUM_HEADS + h) * D_PE + lane * 2));
    const float2 q_p_f32 = __bfloat1622float2(qpv);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;
    int base_sparse = t * TOPK;

    #pragma unroll 1
    for (int i = 0; i < TOPK; ++i) {
        int idx = sparse_indices_ptr[base_sparse + i];
        if (idx < 0) continue;
        ++valid_count;
        int row = decode_row_index(idx);

        float local_dot = 0.0f;
        float2 k_f_reg[8];

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                ckv_cache_ptr + row * D_NOPE + j * 64 + lane * 2);
            const float2 kf = __bfloat1622float2(kv);
            k_f_reg[j] = kf;
            local_dot = fmaf(q_n_f32[j].x, kf.x, local_dot);
            local_dot = fmaf(q_n_f32[j].y, kf.y, local_dot);
        }
        {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                kpe_cache_ptr + row * D_PE + lane * 2);
            const float2 kf = __bfloat1622float2(kv);
            local_dot = fmaf(q_p_f32.x, kf.x, local_dot);
            local_dot = fmaf(q_p_f32.y, kf.y, local_dot);
        }

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float m_new = fmaxf(m, logit);
        float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        float beta = expf(logit - m_new);
        l = l * alpha + beta;
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x = O_reg[j].x * alpha + beta * k_f_reg[j].x;
            O_reg[j].y = O_reg[j].y * alpha + beta * k_f_reg[j].y;
        }
    }

    int out_base = (t * NUM_HEADS + h) * D_NOPE;
    if (valid_count == 0 || l <= 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base + j * 64 + lane * 2) =
                __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse_ptr[t * NUM_HEADS + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base + j * 64 + lane * 2) =
            __floats2bfloat162_rn(O_reg[j].x * inv_l, O_reg[j].y * inv_l);
    }
    if (lane == 0) lse_ptr[t * NUM_HEADS + h] = m * LOG2E_F + log2f(l);
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

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == NUM_HEADS && q_nope.size(2) == D_NOPE,
                "q_nope shape must be [num_tokens,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == NUM_HEADS && q_pe.size(2) == D_PE,
                "q_pe shape must be [num_tokens,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == PAGE_SIZE && ckv_cache.size(2) == D_NOPE,
                "ckv_cache shape must be [num_pages,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == PAGE_SIZE && kpe_cache.size(2) == D_PE,
                "kpe_cache shape must be [num_pages,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == TOPK,
                "sparse_indices shape must be [num_tokens,2048]");
    TORCH_CHECK(q_nope.size(0) == q_pe.size(0) && q_nope.size(0) == sparse_indices.size(0),
                "token dimensions must match");
    TORCH_CHECK(ckv_cache.size(0) == kpe_cache.size(0),
                "cache page dimensions must match");

    const int num_tokens = static_cast<int>(q_nope.size(0));
    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS},
                            torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));

    if (num_tokens == 0) {
        return {output, lse};
    }

    // Very small batches: avoid workspace and favor low launch overhead.
    if (num_tokens <= 2) {
        dim3 grid(num_tokens, NUM_HEADS, 1);
        dim3 block(32, 1, 1);
        dsa_fused_small_kernel<<<grid, block>>>(
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
        return {output, lse};
    }

    // Main high-parallelism path: split-k partials + exact reduction.
    auto partial_m = torch::empty({num_tokens, NUM_HEADS, SPLIT},
                                  torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));
    auto partial_l = torch::empty({num_tokens, NUM_HEADS, SPLIT},
                                  torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));
    auto partial_o = torch::empty({num_tokens, NUM_HEADS, SPLIT, D_NOPE},
                                  torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));

    {
        dim3 grid(num_tokens, NUM_HEADS, SPLIT);
        dim3 block(32, 1, 1);
        dsa_splitk_stage1_kernel<<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            partial_m.data_ptr<float>(),
            partial_l.data_ptr<float>(),
            partial_o.data_ptr<float>(),
            num_tokens
        );
    }

    {
        dim3 grid(num_tokens, NUM_HEADS, 1);
        dim3 block(32, 1, 1);
        dsa_splitk_stage2_kernel<<<grid, block>>>(
            partial_m.data_ptr<float>(),
            partial_l.data_ptr<float>(),
            partial_o.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            num_tokens
        );
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}