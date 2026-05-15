#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).scalar_type() == (dt), #x " has wrong dtype")

constexpr int kNumHeads = 16;
constexpr int kQNope = 512;
constexpr int kQPe = 64;
constexpr int kTopK = 2048;
constexpr int kPage = 64;
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void online_update(float x, float &m, float &l, float2* acc) {
    float m_new = fmaxf(m, x);
    float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
    float beta = expf(x - m_new);
    l = l * alpha + beta;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i].x = acc[i].x * alpha;
        acc[i].y = acc[i].y * alpha;
    }
    m = m_new;
}

__device__ __forceinline__ float finalize_lse_base2(float m, float l) {
    return (l > 0.0f) ? (m * kLog2e + log2f(l)) : -INFINITY;
}

__device__ __forceinline__ int decode_row_index(int idx) {
    int page = idx >> 6;
    int offset = idx & 63;
    return page * 64 + offset;
}

// Devised from plan: avoid the parent's split-K small-token path because its temporary
// global-memory workspace can dominate latency at low token count. Use a single-pass
// warp-per-(token,head) kernel instead.
__global__ void dsa_small_kernel(
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

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(
            &q_nope_ptr[(t * kNumHeads + h) * kQNope + j * 64 + lane * 2]);
        qn[j] = __bfloat1622float2(v);
    }
    const __nv_bfloat162 qp_v = *reinterpret_cast<const __nv_bfloat162*>(
        &q_pe_ptr[(t * kNumHeads + h) * kQPe + lane * 2]);
    const float2 qp = __bfloat1622float2(qp_v);

    float2 acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    const int* idx_row = sparse_indices_ptr + t * kTopK;
    #pragma unroll 1
    for (int s = 0; s < kTopK; ++s) {
        int idx = idx_row[s];
        if (idx < 0) continue;
        ++valid_count;
        int row = decode_row_index(idx);

        float local_dot = 0.0f;
        float2 kv_pair[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                &ckv_cache_ptr[row * kQNope + j * 64 + lane * 2]);
            kv_pair[j] = __bfloat1622float2(kv);
            local_dot = fmaf(qn[j].x, kv_pair[j].x, local_dot);
            local_dot = fmaf(qn[j].y, kv_pair[j].y, local_dot);
        }
        const __nv_bfloat162 kp = *reinterpret_cast<const __nv_bfloat162*>(
            &kpe_cache_ptr[row * kQPe + lane * 2]);
        const float2 kp_f = __bfloat1622float2(kp);
        local_dot = fmaf(qp.x, kp_f.x, local_dot);
        local_dot = fmaf(qp.y, kp_f.y, local_dot);

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float old_m = m;
        float m_new = fmaxf(old_m, logit);
        float alpha = (old_m == -INFINITY) ? 0.0f : expf(old_m - m_new);
        float beta = expf(logit - m_new);
        l = l * alpha + beta;
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j].x = fmaf(beta, kv_pair[j].x, acc[j].x * alpha);
            acc[j].y = fmaf(beta, kv_pair[j].y, acc[j].y * alpha);
        }
    }

    __nv_bfloat16* out_base = output_ptr + (t * kNumHeads + h) * kQNope;
    if (valid_count == 0 || l <= 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(&out_base[j * 64 + lane * 2]) =
                __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse_ptr[t * kNumHeads + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(&out_base[j * 64 + lane * 2]) =
            __floats2bfloat162_rn(acc[j].x * inv_l, acc[j].y * inv_l);
    }
    if (lane == 0) lse_ptr[t * kNumHeads + h] = finalize_lse_base2(m, l);
}

__global__ void dsa_large_kernel(
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
    __shared__ alignas(16) __nv_bfloat16 smem_kv[32 * 512];
    __shared__ alignas(16) __nv_bfloat16 smem_kp[32 * 64];

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(
            &q_nope_ptr[(t * kNumHeads + h) * kQNope + j * 64 + lane * 2]);
        qn[j] = __bfloat1622float2(v);
    }
    const __nv_bfloat162 qp_v = *reinterpret_cast<const __nv_bfloat162*>(
        &q_pe_ptr[(t * kNumHeads + h) * kQPe + lane * 2]);
    const float2 qp = __bfloat1622float2(qp_v);

    float2 acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.0f, 0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    int valid_count = 0;

    #pragma unroll 1
    for (int tile = 0; tile < kTopK / 32; ++tile) {
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * kTopK + tile * 32 + tid];
        }
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid >> 6);
            int load_col = tid & 63;
            int idx = idx_shared[load_row];
            if (idx >= 0) {
                int row = decode_row_index(idx);
                const float4* src = reinterpret_cast<const float4*>(&ckv_cache_ptr[row * kQNope]);
                float4* dst = reinterpret_cast<float4*>(&smem_kv[load_row * kQNope]);
                dst[load_col] = src[load_col];
            }
        }
        if (tid < 256) {
            int load_row = tid >> 3;
            int load_col = tid & 7;
            int idx = idx_shared[load_row];
            if (idx >= 0) {
                int row = decode_row_index(idx);
                const float4* src = reinterpret_cast<const float4*>(&kpe_cache_ptr[row * kQPe]);
                float4* dst = reinterpret_cast<float4*>(&smem_kp[load_row * kQPe]);
                dst[load_col] = src[load_col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] < 0) continue;
            ++valid_count;
            float local_dot = 0.0f;
            float2 kv_pair[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                    &smem_kv[i * kQNope + j * 64 + lane * 2]);
                kv_pair[j] = __bfloat1622float2(kv);
                local_dot = fmaf(qn[j].x, kv_pair[j].x, local_dot);
                local_dot = fmaf(qn[j].y, kv_pair[j].y, local_dot);
            }
            const __nv_bfloat162 kp = *reinterpret_cast<const __nv_bfloat162*>(
                &smem_kp[i * kQPe + lane * 2]);
            const float2 kp_f = __bfloat1622float2(kp);
            local_dot = fmaf(qp.x, kp_f.x, local_dot);
            local_dot = fmaf(qp.y, kp_f.y, local_dot);

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            float old_m = m;
            float m_new = fmaxf(old_m, logit);
            float alpha = (old_m == -INFINITY) ? 0.0f : expf(old_m - m_new);
            float beta = expf(logit - m_new);
            l = l * alpha + beta;
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j].x = fmaf(beta, kv_pair[j].x, acc[j].x * alpha);
                acc[j].y = fmaf(beta, kv_pair[j].y, acc[j].y * alpha);
            }
        }
        __syncthreads();
    }

    __nv_bfloat16* out_base = output_ptr + (t * kNumHeads + h) * kQNope;
    if (valid_count == 0 || l <= 0.0f) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(&out_base[j * 64 + lane * 2]) =
                __floats2bfloat162_rn(0.0f, 0.0f);
        }
        if (lane == 0) lse_ptr[t * kNumHeads + h] = -INFINITY;
        return;
    }

    float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(&out_base[j * 64 + lane * 2]) =
            __floats2bfloat162_rn(acc[j].x * inv_l, acc[j].y * inv_l);
    }
    if (lane == 0) lse_ptr[t * kNumHeads + h] = finalize_lse_base2(m, l);
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

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == kNumHeads && q_nope.size(2) == kQNope, "q_nope shape must be [T,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == kNumHeads && q_pe.size(2) == kQPe, "q_pe shape must be [T,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == kPage && ckv_cache.size(2) == kQNope, "ckv_cache shape must be [P,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == kPage && kpe_cache.size(2) == kQPe, "kpe_cache shape must be [P,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == kTopK, "sparse_indices shape must be [T,2048]");
    TORCH_CHECK(q_nope.size(0) == q_pe.size(0) && q_nope.size(0) == sparse_indices.size(0), "token dimensions must match");
    TORCH_CHECK(ckv_cache.size(0) == kpe_cache.size(0), "cache page dimensions must match");

    const int64_t num_tokens = q_nope.size(0);
    auto output = torch::empty({num_tokens, kNumHeads, kQNope}, q_nope.options());
    auto lse = torch::empty({num_tokens, kNumHeads}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    constexpr int kSmallThreshold = 64;
    if (num_tokens <= kSmallThreshold) {
        dim3 grid(num_tokens, kNumHeads, 1);
        dim3 block(32, 1, 1);
        dsa_small_kernel<<<grid, block>>>(
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
        dim3 block(32, kNumHeads, 1);
        dsa_large_kernel<<<grid, block>>>(
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
