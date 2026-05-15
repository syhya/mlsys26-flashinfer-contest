#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).scalar_type() == (dt), #x " has wrong dtype")

static constexpr int kNumHeads = 16;
static constexpr int kQNopeDim = 512;
static constexpr int kQPeDim = 64;
static constexpr int kTopK = 2048;
static constexpr int kPageSize = 64;
static constexpr float kLog2e = 1.4426950408889634f;

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

__device__ __forceinline__ void load_query_fragments(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    int t,
    int h,
    int lane,
    float2 (&q_n_f32)[8],
    float2 &q_p_f32) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            &q_nope_ptr[t * kNumHeads * kQNopeDim + h * kQNopeDim + j * 64 + lane * 2]);
        q_n_f32[j] = __bfloat1622float2(qv);
    }
    const __nv_bfloat162 qpe = *reinterpret_cast<const __nv_bfloat162*>(
        &q_pe_ptr[t * kNumHeads * kQPeDim + h * kQPeDim + lane * 2]);
    q_p_f32 = __bfloat1622float2(qpe);
}

// Low-latency path: one warp per (token, head), direct gather, no shared memory staging.
__global__ void dsa_forward_small_kernel(
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
    int h = blockIdx.y;
    int lane = threadIdx.x;

    float2 q_n_f32[8];
    float2 q_p_f32;
    load_query_fragments(q_nope_ptr, q_pe_ptr, t, h, lane, q_n_f32, q_p_f32);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    #pragma unroll 1
    for (int i = 0; i < kTopK; ++i) {
        int idx = sparse_indices_ptr[t * kTopK + i];
        if (idx < 0) continue;
        ++valid_count;

        int row = decode_row_index(idx);
        const __nv_bfloat16* k_nope_row = &ckv_cache_ptr[row * kQNopeDim];
        const __nv_bfloat16* k_pe_row = &kpe_cache_ptr[row * kQPeDim];

        float local_dot = 0.0f;
        float2 k_f_reg[8];

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(&k_nope_row[j * 64 + lane * 2]);
            float2 kf = __bfloat1622float2(kv);
            k_f_reg[j] = kf;
            local_dot = fmaf(q_n_f32[j].x, kf.x, local_dot);
            local_dot = fmaf(q_n_f32[j].y, kf.y, local_dot);
        }
        {
            const __nv_bfloat162 kvp = *reinterpret_cast<const __nv_bfloat162*>(&k_pe_row[lane * 2]);
            float2 kp = __bfloat1622float2(kvp);
            local_dot = fmaf(q_p_f32.x, kp.x, local_dot);
            local_dot = fmaf(q_p_f32.y, kp.y, local_dot);
        }

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float m_new = fmaxf(m, logit);
        float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        float beta = expf(logit - m_new);
        l = fmaf(l, alpha, beta);

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x = fmaf(beta, k_f_reg[j].x, O_reg[j].x * alpha);
            O_reg[j].y = fmaf(beta, k_f_reg[j].y, O_reg[j].y * alpha);
        }
        m = m_new;
    }

    if (valid_count == 0 || l <= 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(
                &output_ptr[t * kNumHeads * kQNopeDim + h * kQNopeDim + j * 64 + lane * 2]) =
                __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse_ptr[t * kNumHeads + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float ox = O_reg[j].x * inv_l;
        float oy = O_reg[j].y * inv_l;
        *reinterpret_cast<__nv_bfloat162*>(
            &output_ptr[t * kNumHeads * kQNopeDim + h * kQNopeDim + j * 64 + lane * 2]) =
            __floats2bfloat162_rn(ox, oy);
    }
    if (lane == 0) lse_ptr[t * kNumHeads + h] = m * kLog2e + log2f(l);
}

// Throughput path: parent-style fused staging over 32 sparse rows per tile.
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
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * 512];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * 64];

    float2 q_n_f32[8];
    float2 q_p_f32;
    load_query_fragments(q_nope_ptr, q_pe_ptr, t, h, lane, q_n_f32, q_p_f32);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    constexpr int kTileRows = 32;
    constexpr int num_tiles = kTopK / kTileRows;

    for (int tile = 0; tile < num_tiles; ++tile) {
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * kTopK + tile * kTileRows + tid];
        }
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid / 64);
            int load_col = tid % 64;
            int idx = idx_shared[load_row];
            if (idx >= 0) {
                int row = decode_row_index(idx);
                const float4* src = reinterpret_cast<const float4*>(&ckv_cache_ptr[row * 512]);
                float4* dst = reinterpret_cast<float4*>(&smem_Kc[load_row * 512]);
                dst[load_col] = src[load_col];
            }
        }

        if (tid < 256) {
            int load_row = tid / 8;
            int load_col = tid % 8;
            int idx = idx_shared[load_row];
            if (idx >= 0) {
                int row = decode_row_index(idx);
                const float4* src = reinterpret_cast<const float4*>(&kpe_cache_ptr[row * 64]);
                float4* dst = reinterpret_cast<float4*>(&smem_Kp[load_row * 64]);
                dst[load_col] = src[load_col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < kTileRows; ++i) {
            if (idx_shared[i] < 0) continue;
            ++valid_count;

            float local_dot = 0.0f;
            float2 k_f_reg[8];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * 512 + j * 64 + lane * 2]);
                float2 kf = __bfloat1622float2(kv);
                k_f_reg[j] = kf;
                local_dot = fmaf(q_n_f32[j].x, kf.x, local_dot);
                local_dot = fmaf(q_n_f32[j].y, kf.y, local_dot);
            }
            {
                const __nv_bfloat162 kvp = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * 64 + lane * 2]);
                float2 kp = __bfloat1622float2(kvp);
                local_dot = fmaf(q_p_f32.x, kp.x, local_dot);
                local_dot = fmaf(q_p_f32.y, kp.y, local_dot);
            }

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            float m_new = fmaxf(m, logit);
            float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
            float beta = expf(logit - m_new);
            l = fmaf(l, alpha, beta);

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                O_reg[j].x = fmaf(beta, k_f_reg[j].x, O_reg[j].x * alpha);
                O_reg[j].y = fmaf(beta, k_f_reg[j].y, O_reg[j].y * alpha);
            }
            m = m_new;
        }
        __syncthreads();
    }

    if (valid_count == 0 || l <= 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(
                &output_ptr[t * kNumHeads * kQNopeDim + h * kQNopeDim + j * 64 + lane * 2]) =
                __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse_ptr[t * kNumHeads + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float ox = O_reg[j].x * inv_l;
        float oy = O_reg[j].y * inv_l;
        *reinterpret_cast<__nv_bfloat162*>(
            &output_ptr[t * kNumHeads * kQNopeDim + h * kQNopeDim + j * 64 + lane * 2]) =
            __floats2bfloat162_rn(ox, oy);
    }
    if (lane == 0) lse_ptr[t * kNumHeads + h] = m * kLog2e + log2f(l);
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

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == 16 && q_nope.size(2) == 512, "q_nope shape must be [T,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == 16 && q_pe.size(2) == 64, "q_pe shape must be [T,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == 64 && ckv_cache.size(2) == 512, "ckv_cache shape must be [P,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == 64 && kpe_cache.size(2) == 64, "kpe_cache shape must be [P,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == 2048, "sparse_indices shape must be [T,2048]");
    TORCH_CHECK(q_nope.size(0) == q_pe.size(0) && q_nope.size(0) == sparse_indices.size(0), "token dimensions must match");
    TORCH_CHECK(ckv_cache.size(0) == kpe_cache.size(0), "cache page dimensions must match");

    const int num_tokens = static_cast<int>(q_nope.size(0));
    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16}, torch::dtype(torch::kFloat32).device(q_nope.device()));

    if (num_tokens == 0) {
        return {output, lse};
    }

    // Devised from plan: replace the parent split-K path with a single-pass warp-per-head kernel
    // for small workloads to avoid temporary HBM traffic; keep the proven fused staged kernel for larger T.
    if (num_tokens <= 32) {
        dim3 grid(num_tokens, 16, 1);
        dim3 block(32, 1, 1);
        dsa_forward_small_kernel<<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>());
    } else {
        dim3 grid(num_tokens, 1, 1);
        dim3 block(32, 16, 1);
        dsa_forward_large_kernel<<<grid, block>>>(
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
