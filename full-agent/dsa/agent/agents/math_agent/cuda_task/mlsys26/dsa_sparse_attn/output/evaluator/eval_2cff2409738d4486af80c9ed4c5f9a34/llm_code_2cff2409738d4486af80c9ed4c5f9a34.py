#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).scalar_type() == (dt), #x " has wrong dtype")

static constexpr int NUM_HEADS = 16;
static constexpr int D_NOPE = 512;
static constexpr int D_PE = 64;
static constexpr int TOPK = 2048;
static constexpr int PAGE_SIZE = 64;
static constexpr float LOG2E_F = 1.4426950408889634f;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ void decode_sparse_index(int idx, int &page, int &offset) {
    page = idx >> 6;
    offset = idx & 63;
}

__device__ __forceinline__ float compute_lse_base2(float m, float l) {
    return (l > 0.0f && m != -INFINITY) ? (m * LOG2E_F + log2f(l)) : -INFINITY;
}

__device__ __forceinline__ void online_update_scalar(float x, float &m, float &l, float &acc, float v) {
    float m_new = fmaxf(m, x);
    float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
    float beta = expf(x - m_new);
    l = l * alpha + beta;
    acc = acc * alpha + beta * v;
    m = m_new;
}

__global__ void dsa_small_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int32_t* __restrict__ sparse_indices,
    float sm_scale,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int num_tokens
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int lane = threadIdx.x;
    if (t >= num_tokens) return;

    const __nv_bfloat16* qn_ptr = q_nope + ((t * NUM_HEADS + h) * D_NOPE);
    const __nv_bfloat16* qp_ptr = q_pe + ((t * NUM_HEADS + h) * D_PE);
    const int32_t* idx_ptr = sparse_indices + t * TOPK;

    float2 qn_frag[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(qn_ptr + j * 64 + lane * 2);
        qn_frag[j] = __bfloat1622float2(*p);
    }
    float2 qp_frag = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qp_ptr + lane * 2));

    float2 o_frag[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) o_frag[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    #pragma unroll 1
    for (int kk = 0; kk < TOPK; ++kk) {
        int idx = idx_ptr[kk];
        if (idx < 0) continue;
        ++valid_count;
        int page, offset;
        decode_sparse_index(idx, page, offset);
        int row = page * PAGE_SIZE + offset;

        const __nv_bfloat16* kc_ptr = ckv_cache + row * D_NOPE;
        const __nv_bfloat16* kp_ptr = kpe_cache + row * D_PE;

        float local_dot = 0.0f;
        float2 k_frag[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(kc_ptr + j * 64 + lane * 2);
            k_frag[j] = __bfloat1622float2(*p);
            local_dot = fmaf(qn_frag[j].x, k_frag[j].x, local_dot);
            local_dot = fmaf(qn_frag[j].y, k_frag[j].y, local_dot);
        }
        float2 kp_frag = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(kp_ptr + lane * 2));
        local_dot = fmaf(qp_frag.x, kp_frag.x, local_dot);
        local_dot = fmaf(qp_frag.y, kp_frag.y, local_dot);

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float m_new = fmaxf(m, logit);
        float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        float beta = expf(logit - m_new);
        l = l * alpha + beta;
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            o_frag[j].x = o_frag[j].x * alpha + beta * k_frag[j].x;
            o_frag[j].y = o_frag[j].y * alpha + beta * k_frag[j].y;
        }
    }

    __nv_bfloat16* out_ptr = out + ((t * NUM_HEADS + h) * D_NOPE);
    if (valid_count == 0 || l <= 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(out_ptr + j * 64 + lane * 2) = __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse[t * NUM_HEADS + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(out_ptr + j * 64 + lane * 2) =
            __floats2bfloat162_rn(o_frag[j].x * inv_l, o_frag[j].y * inv_l);
    }
    if (lane == 0) lse[t * NUM_HEADS + h] = compute_lse_base2(m, l);
}

__global__ void dsa_large_kernel(
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
    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = threadIdx.y * 32 + threadIdx.x;
    if (t >= num_tokens) return;

    __shared__ alignas(16) int idx_shared[32];
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * 512];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * 64];

    float2 q_n_f32[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&q_nope_ptr[t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2]);
        q_n_f32[j] = __bfloat1622float2(*p);
    }
    float2 q_p_f32 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&q_pe_ptr[t * NUM_HEADS * D_PE + h * D_PE + lane * 2]));

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    #pragma unroll 1
    for (int tile = 0; tile < TOPK / 32; ++tile) {
        if (tid < 32) idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * 32 + tid];
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid >> 6);
            int load_col = tid & 63;
            int idx = idx_shared[load_row];
            if (idx >= 0) {
                int page = idx >> 6;
                int offset = idx & 63;
                int row = page * PAGE_SIZE + offset;
                const float4* src = reinterpret_cast<const float4*>(ckv_cache_ptr + row * D_NOPE);
                float4* dst = reinterpret_cast<float4*>(smem_Kc + load_row * D_NOPE);
                dst[load_col] = src[load_col];
            }
        }

        if (tid < 256) {
            int load_row = tid >> 3;
            int load_col = tid & 7;
            int idx = idx_shared[load_row];
            if (idx >= 0) {
                int page = idx >> 6;
                int offset = idx & 63;
                int row = page * PAGE_SIZE + offset;
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
                const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * D_NOPE + j * 64 + lane * 2]);
                k_f_reg[j] = __bfloat1622float2(*p);
                local_dot = fmaf(q_n_f32[j].x, k_f_reg[j].x, local_dot);
                local_dot = fmaf(q_n_f32[j].y, k_f_reg[j].y, local_dot);
            }
            float2 k_p = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * D_PE + lane * 2]));
            local_dot = fmaf(q_p_f32.x, k_p.x, local_dot);
            local_dot = fmaf(q_p_f32.y, k_p.y, local_dot);

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

    if (valid_count == 0 || l <= 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2]) =
                __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse_ptr[t * NUM_HEADS + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[t * NUM_HEADS * D_NOPE + h * D_NOPE + j * 64 + lane * 2]) =
            __floats2bfloat162_rn(O_reg[j].x * inv_l, O_reg[j].y * inv_l);
    }
    if (lane == 0) lse_ptr[t * NUM_HEADS + h] = compute_lse_base2(m, l);
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

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == NUM_HEADS && q_nope.size(2) == D_NOPE, "q_nope shape must be [T,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == NUM_HEADS && q_pe.size(2) == D_PE, "q_pe shape must be [T,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == PAGE_SIZE && ckv_cache.size(2) == D_NOPE, "ckv_cache shape must be [P,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == PAGE_SIZE && kpe_cache.size(2) == D_PE, "kpe_cache shape must be [P,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == TOPK, "sparse_indices shape must be [T,2048]");
    TORCH_CHECK(q_nope.size(0) == q_pe.size(0) && q_nope.size(0) == sparse_indices.size(0), "token dimensions must match");
    TORCH_CHECK(ckv_cache.size(0) == kpe_cache.size(0), "cache page dimensions must match");

    const int num_tokens = static_cast<int>(q_nope.size(0));
    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    // Devised from Plan: remove split-K workspace-heavy small path; use a direct warp-per-(token,head) kernel.
    if (num_tokens <= 64) {
        dim3 grid(num_tokens, NUM_HEADS, 1);
        dim3 block(32, 1, 1);
        dsa_small_kernel<<<grid, block>>>(
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
        dsa_large_kernel<<<grid, block>>>(
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
