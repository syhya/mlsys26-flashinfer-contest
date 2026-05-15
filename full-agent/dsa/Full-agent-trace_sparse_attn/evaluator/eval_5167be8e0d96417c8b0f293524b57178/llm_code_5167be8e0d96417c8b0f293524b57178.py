#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <vector>
#include <cmath>
#include <cfloat>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).scalar_type() == (dt), #x " has wrong dtype")

static constexpr int NUM_HEADS = 16;
static constexpr int D_NOPE = 512;
static constexpr int D_PE = 64;
static constexpr int TOPK = 2048;
static constexpr int PAGE_SIZE = 64;
static constexpr int SPLIT_SIZE = 128;   // 16 chunks over topk=2048
static constexpr int NUM_SPLITS = TOPK / SPLIT_SIZE;
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
    return page * PAGE_SIZE + offset;
}

// Split-K partial kernel: one warp computes one (token, head, split).
__global__ void dsa_splitk_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float* __restrict__ partial_out_ptr,   // [T,16,S,512]
    float* __restrict__ partial_m_ptr,     // [T,16,S]
    float* __restrict__ partial_l_ptr,     // [T,16,S]
    int* __restrict__ partial_valid_ptr,   // [T,16,S]
    int num_tokens
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z;
    int lane = threadIdx.x;
    if (t >= num_tokens || h >= NUM_HEADS || s >= NUM_SPLITS) return;

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

    int start = s * SPLIT_SIZE;
    int end = start + SPLIT_SIZE;
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

    int part_idx = ((t * NUM_HEADS + h) * NUM_SPLITS + s);
    if (lane == 0) {
        partial_m_ptr[part_idx] = m;
        partial_l_ptr[part_idx] = l;
        partial_valid_ptr[part_idx] = valid_count;
    }

    float* out_base = partial_out_ptr + (((t * NUM_HEADS + h) * NUM_SPLITS + s) * D_NOPE);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        out_base[j * 64 + lane * 2 + 0] = O_reg[j].x;
        out_base[j * 64 + lane * 2 + 1] = O_reg[j].y;
    }
}

// Reduce split-K partials into final output.
__global__ void dsa_splitk_reduce_kernel(
    const float* __restrict__ partial_out_ptr,
    const float* __restrict__ partial_m_ptr,
    const float* __restrict__ partial_l_ptr,
    const int* __restrict__ partial_valid_ptr,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr,
    int num_tokens
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int lane = threadIdx.x;
    if (t >= num_tokens || h >= NUM_HEADS) return;

    int base_idx = (t * NUM_HEADS + h) * NUM_SPLITS;

    float M = -INFINITY;
    int total_valid = 0;
    #pragma unroll
    for (int s = 0; s < NUM_SPLITS; ++s) {
        int v = partial_valid_ptr[base_idx + s];
        total_valid += v;
        if (v > 0) {
            float ms = partial_m_ptr[base_idx + s];
            M = fmaxf(M, ms);
        }
    }

    int out_base = (t * NUM_HEADS + h) * D_NOPE;
    if (total_valid == 0 || M == -INFINITY) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base + j * 64 + lane * 2) =
                __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse_ptr[t * NUM_HEADS + h] = -INFINITY;
        return;
    }

    float L = 0.0f;
    #pragma unroll
    for (int s = 0; s < NUM_SPLITS; ++s) {
        int v = partial_valid_ptr[base_idx + s];
        if (v > 0) {
            float ms = partial_m_ptr[base_idx + s];
            float ls = partial_l_ptr[base_idx + s];
            L += ls * expf(ms - M);
        }
    }

    float2 out_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) out_reg[j] = make_float2(0.0f, 0.0f);

    #pragma unroll
    for (int s = 0; s < NUM_SPLITS; ++s) {
        int v = partial_valid_ptr[base_idx + s];
        if (v <= 0) continue;
        float weight = expf(partial_m_ptr[base_idx + s] - M);
        const float* part_base = partial_out_ptr + ((base_idx + s) * D_NOPE);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_reg[j].x = fmaf(weight, part_base[j * 64 + lane * 2 + 0], out_reg[j].x);
            out_reg[j].y = fmaf(weight, part_base[j * 64 + lane * 2 + 1], out_reg[j].y);
        }
    }

    float inv_L = 1.0f / L;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base + j * 64 + lane * 2) =
            __floats2bfloat162_rn(out_reg[j].x * inv_L, out_reg[j].y * inv_L);
    }
    if (lane == 0) {
        lse_ptr[t * NUM_HEADS + h] = M * LOG2E_F + log2f(L);
    }
}

// Large throughput path: one block per token, 16 warps for 16 heads, shared-memory staged gather.
__global__ void dsa_forward_large_kernel(
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

    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = h * 32 + lane;

    __shared__ alignas(16) int idx_shared[32];
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * D_NOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * D_PE];

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

    #pragma unroll 1
    for (int tile = 0; tile < TOPK / 32; ++tile) {
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * 32 + tid];
        }
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid >> 6); // 0..31
            int load_col = tid & 63;              // float4 index in 512 bf16 row
            int idx = idx_shared[load_row];
            if (idx >= 0) {
                int row = decode_row_index(idx);
                const float4* src = reinterpret_cast<const float4*>(ckv_cache_ptr + row * D_NOPE);
                float4* dst = reinterpret_cast<float4*>(smem_Kc + load_row * D_NOPE);
                dst[load_col] = src[load_col];
            }
        }

        if (tid < 256) {
            int load_row = tid >> 3; // 0..31
            int load_col = tid & 7;  // float4 index in 64 bf16 row
            int idx = idx_shared[load_row];
            if (idx >= 0) {
                int row = decode_row_index(idx);
                const float4* src = reinterpret_cast<const float4*>(kpe_cache_ptr + row * D_PE);
                float4* dst = reinterpret_cast<float4*>(smem_Kp + load_row * D_PE);
                dst[load_col] = src[load_col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] < 0) continue;
            ++valid_count;

            float local_dot = 0.0f;
            float2 k_f_reg[8];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                    smem_Kc + i * D_NOPE + j * 64 + lane * 2);
                const float2 kf = __bfloat1622float2(kv);
                k_f_reg[j] = kf;
                local_dot = fmaf(q_n_f32[j].x, kf.x, local_dot);
                local_dot = fmaf(q_n_f32[j].y, kf.y, local_dot);
            }
            {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                    smem_Kp + i * D_PE + lane * 2);
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
        __syncthreads();
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
    TORCH_CHECK(ckv_cache.size(0) == kpe_cache.size(0), "cache page dimensions must match");

    const int num_tokens = static_cast<int>(q_nope.size(0));
    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS},
                            torch::dtype(torch::kFloat32).device(q_nope.device()));

    if (num_tokens == 0) {
        return {output, lse};
    }

    // Restore strong small-workload path: split-K partials + reduce.
    if (num_tokens <= 32) {
        auto float_opts = torch::dtype(torch::kFloat32).device(q_nope.device());
        auto int_opts = torch::dtype(torch::kInt32).device(q_nope.device());

        auto partial_out = torch::empty({num_tokens, NUM_HEADS, NUM_SPLITS, D_NOPE}, float_opts);
        auto partial_m = torch::empty({num_tokens, NUM_HEADS, NUM_SPLITS}, float_opts);
        auto partial_l = torch::empty({num_tokens, NUM_HEADS, NUM_SPLITS}, float_opts);
        auto partial_valid = torch::empty({num_tokens, NUM_HEADS, NUM_SPLITS}, int_opts);

        dim3 grid1(num_tokens, NUM_HEADS, NUM_SPLITS);
        dim3 block1(32, 1, 1);
        dsa_splitk_kernel<<<grid1, block1>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            partial_out.data_ptr<float>(),
            partial_m.data_ptr<float>(),
            partial_l.data_ptr<float>(),
            partial_valid.data_ptr<int>(),
            num_tokens
        );

        dim3 grid2(num_tokens, NUM_HEADS, 1);
        dim3 block2(32, 1, 1);
        dsa_splitk_reduce_kernel<<<grid2, block2>>>(
            partial_out.data_ptr<float>(),
            partial_m.data_ptr<float>(),
            partial_l.data_ptr<float>(),
            partial_valid.data_ptr<int>(),
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
            lse.data_ptr<float>(),
            num_tokens
        );
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}