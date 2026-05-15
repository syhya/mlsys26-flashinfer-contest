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
static constexpr int ROWS_PER_PAGE_SHIFT = 6;
static constexpr float LOG2E_F = 1.4426950408889634f;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ int decode_row_index(int idx) {
    int page = idx >> ROWS_PER_PAGE_SHIFT;
    int offset = idx & (PAGE_SIZE - 1);
    return page * PAGE_SIZE + offset;
}

__device__ __forceinline__ void zero_store_output(
    __nv_bfloat16* __restrict__ output_ptr,
    int out_base,
    int lane
) {
    const __nv_bfloat162 z = __floats2bfloat162_rn(0.0f, 0.0f);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base + j * 64 + lane * 2) = z;
    }
}

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
    if (t >= num_tokens || h >= NUM_HEADS) return;

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            q_nope_ptr + ((t * NUM_HEADS + h) * D_NOPE + j * 64 + lane * 2));
        qn[j] = __bfloat1622float2(qv);
    }
    const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(
        q_pe_ptr + ((t * NUM_HEADS + h) * D_PE + lane * 2));
    const float2 qp = __bfloat1622float2(qpv);

    float2 acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;
    int sparse_base = t * TOPK;

    #pragma unroll 1
    for (int i = 0; i < TOPK; ++i) {
        int idx = sparse_indices_ptr[sparse_base + i];
        if (idx < 0) continue;
        ++valid_count;

        int row = decode_row_index(idx);
        float local_dot = 0.0f;
        float2 kvals[8];

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                ckv_cache_ptr + row * D_NOPE + j * 64 + lane * 2);
            const float2 kf = __bfloat1622float2(kv);
            kvals[j] = kf;
            local_dot = fmaf(qn[j].x, kf.x, local_dot);
            local_dot = fmaf(qn[j].y, kf.y, local_dot);
        }
        {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                kpe_cache_ptr + row * D_PE + lane * 2);
            const float2 kf = __bfloat1622float2(kv);
            local_dot = fmaf(qp.x, kf.x, local_dot);
            local_dot = fmaf(qp.y, kf.y, local_dot);
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
            acc[j].x = acc[j].x * alpha + beta * kvals[j].x;
            acc[j].y = acc[j].y * alpha + beta * kvals[j].y;
        }
    }

    int out_base = (t * NUM_HEADS + h) * D_NOPE;
    if (valid_count == 0 || l <= 0.0f) {
        zero_store_output(output_ptr, out_base, lane);
        if (lane == 0) lse_ptr[t * NUM_HEADS + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base + j * 64 + lane * 2) =
            __floats2bfloat162_rn(acc[j].x * inv_l, acc[j].y * inv_l);
    }
    if (lane == 0) {
        lse_ptr[t * NUM_HEADS + h] = m * LOG2E_F + log2f(l);
    }
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
            q_nope_ptr + ((t * NUM_HEADS + h) * D_NOPE + j * 64 + lane * 2));
        qn[j] = __bfloat1622float2(qv);
    }
    const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(
        q_pe_ptr + ((t * NUM_HEADS + h) * D_PE + lane * 2));
    const float2 qp = __bfloat1622float2(qpv);

    float2 acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    #pragma unroll 1
    for (int tile = 0; tile < TOPK / 32; ++tile) {
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * 32 + tid];
        }
        __syncthreads();

        if (tid < 1024) {
            int row_in_tile = tid >> 5;
            int vec = tid & 31;
            int idx = idx_shared[row_in_tile];
            __nv_bfloat16* dst = smem_Kc + row_in_tile * D_NOPE + vec * 16;
            if (idx >= 0) {
                int row = decode_row_index(idx);
                const int4* src4 = reinterpret_cast<const int4*>(ckv_cache_ptr + row * D_NOPE + vec * 16);
                *reinterpret_cast<int4*>(dst) = *src4;
            } else {
                int4 z = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(dst) = z;
            }
        }

        if (tid < 128) {
            int row_in_tile = tid >> 2;
            int vec = tid & 3;
            int idx = idx_shared[row_in_tile];
            __nv_bfloat16* dst = smem_Kp + row_in_tile * D_PE + vec * 16;
            if (idx >= 0) {
                int row = decode_row_index(idx);
                const int4* src4 = reinterpret_cast<const int4*>(kpe_cache_ptr + row * D_PE + vec * 16);
                *reinterpret_cast<int4*>(dst) = *src4;
            } else {
                int4 z = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(dst) = z;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] < 0) continue;
            ++valid_count;

            float local_dot = 0.0f;
            float2 kvals[8];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                    smem_Kc + i * D_NOPE + j * 64 + lane * 2);
                const float2 kf = __bfloat1622float2(kv);
                kvals[j] = kf;
                local_dot = fmaf(qn[j].x, kf.x, local_dot);
                local_dot = fmaf(qn[j].y, kf.y, local_dot);
            }
            {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                    smem_Kp + i * D_PE + lane * 2);
                const float2 kf = __bfloat1622float2(kv);
                local_dot = fmaf(qp.x, kf.x, local_dot);
                local_dot = fmaf(qp.y, kf.y, local_dot);
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
                acc[j].x = acc[j].x * alpha + beta * kvals[j].x;
                acc[j].y = acc[j].y * alpha + beta * kvals[j].y;
            }
        }
        __syncthreads();
    }

    int out_base = (t * NUM_HEADS + h) * D_NOPE;
    if (valid_count == 0 || l <= 0.0f) {
        zero_store_output(output_ptr, out_base, lane);
        if (lane == 0) lse_ptr[t * NUM_HEADS + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base + j * 64 + lane * 2) =
            __floats2bfloat162_rn(acc[j].x * inv_l, acc[j].y * inv_l);
    }
    if (lane == 0) {
        lse_ptr[t * NUM_HEADS + h] = m * LOG2E_F + log2f(l);
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

    int num_tokens = static_cast<int>(q_nope.size(0));

    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS},
                            torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));

    if (num_tokens == 0) {
        return {output, lse};
    }

    // Restore strong dynamic dispatch: tiny workloads use the lower-overhead warp kernel,
    // everything else uses the higher-parallel block kernel.
    if (num_tokens <= 4) {
        dim3 grid(num_tokens, NUM_HEADS, 1);
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
        dim3 block(32, NUM_HEADS, 1);
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