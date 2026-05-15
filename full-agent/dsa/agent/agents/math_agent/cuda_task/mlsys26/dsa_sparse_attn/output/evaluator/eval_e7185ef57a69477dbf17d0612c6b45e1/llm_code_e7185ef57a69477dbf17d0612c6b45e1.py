#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <tuple>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == at::kBFloat16, #x " must be bfloat16")
#define CHECK_I32(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")

static constexpr int NUM_HEADS = 16;
static constexpr int D_NOPE = 512;
static constexpr int D_PE = 64;
static constexpr int TOPK = 2048;
static constexpr int PAGE_SIZE = 64;
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

__global__ void dsa_forward_streaming_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr,
    int num_tokens) {

    int t = blockIdx.x;
    if (t >= num_tokens) return;

    int h = threadIdx.y;
    int lane = threadIdx.x;

    const __nv_bfloat16* qn_head = q_nope_ptr + ((size_t)t * NUM_HEADS + h) * D_NOPE;
    const __nv_bfloat16* qp_head = q_pe_ptr   + ((size_t)t * NUM_HEADS + h) * D_PE;
    const int32_t* sparse_row = sparse_indices_ptr + (size_t)t * TOPK;

    float2 qn_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        qn_reg[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qn_head + j * 64 + lane * 2));
    }
    float2 qp_reg = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qp_head + lane * 2));

    float2 o_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) o_reg[j] = make_float2(0.0f, 0.0f);

    float m = NEG_INF_F;
    float l = 0.0f;

    #pragma unroll 1
    for (int kk = 0; kk < TOPK; ++kk) {
        int idx = sparse_row[kk];
        if (idx == -1) continue;

        const __nv_bfloat16* kc = ckv_cache_ptr + (size_t)idx * D_NOPE;
        const __nv_bfloat16* kp = kpe_cache_ptr + (size_t)idx * D_PE;

        float local_dot = 0.0f;
        float2 k_cache[8];

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(kc + j * 64 + lane * 2);
            float2 kf = __bfloat1622float2(kv);
            k_cache[j] = kf;
            local_dot = fmaf(qn_reg[j].x, kf.x, local_dot);
            local_dot = fmaf(qn_reg[j].y, kf.y, local_dot);
        }

        {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(kp + lane * 2);
            float2 kf = __bfloat1622float2(kv);
            local_dot = fmaf(qp_reg.x, kf.x, local_dot);
            local_dot = fmaf(qp_reg.y, kf.y, local_dot);
        }

        float score = warp_reduce_sum(local_dot);
        score = __shfl_sync(0xffffffff, score, 0) * sm_scale;

        float m_new = fmaxf(m, score);
        float alpha = (m <= NEG_INF_F * 0.5f) ? 0.0f : __expf(m - m_new);
        float beta  = __expf(score - m_new);
        l = l * alpha + beta;
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            o_reg[j].x = o_reg[j].x * alpha + beta * k_cache[j].x;
            o_reg[j].y = o_reg[j].y * alpha + beta * k_cache[j].y;
        }
    }

    float inv_l = (l > 0.0f) ? __fdividef(1.0f, l) : 0.0f;
    __nv_bfloat16* out_head = output_ptr + ((size_t)t * NUM_HEADS + h) * D_NOPE;

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<__nv_bfloat162*>(out_head + j * 64 + lane * 2) =
            __floats2bfloat162_rn(o_reg[j].x * inv_l, o_reg[j].y * inv_l);
    }

    if (lane == 0) {
        lse_ptr[t * NUM_HEADS + h] = (l > 0.0f) ? (m * LOG2E_F + __log2f(l)) : NEG_INF_F;
    }
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
    float* __restrict__ l_tmp,
    int num_tokens) {

    int s = blockIdx.x;
    int t = blockIdx.y;
    if (t >= num_tokens) return;

    int h = threadIdx.y;
    int lane = threadIdx.x;

    const __nv_bfloat16* qn_head = q_nope_ptr + ((size_t)t * NUM_HEADS + h) * D_NOPE;
    const __nv_bfloat16* qp_head = q_pe_ptr   + ((size_t)t * NUM_HEADS + h) * D_PE;
    const int32_t* sparse_row = sparse_indices_ptr + (size_t)t * TOPK + s * 32;

    float2 qn_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        qn_reg[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qn_head + j * 64 + lane * 2));
    }
    float2 qp_reg = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qp_head + lane * 2));

    float2 o_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) o_reg[j] = make_float2(0.0f, 0.0f);

    float m = NEG_INF_F;
    float l = 0.0f;

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int idx = sparse_row[i];
        if (idx == -1) continue;

        const __nv_bfloat16* kc = ckv_cache_ptr + (size_t)idx * D_NOPE;
        const __nv_bfloat16* kp = kpe_cache_ptr + (size_t)idx * D_PE;

        float local_dot = 0.0f;
        float2 k_cache[8];

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(kc + j * 64 + lane * 2);
            float2 kf = __bfloat1622float2(kv);
            k_cache[j] = kf;
            local_dot = fmaf(qn_reg[j].x, kf.x, local_dot);
            local_dot = fmaf(qn_reg[j].y, kf.y, local_dot);
        }

        {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(kp + lane * 2);
            float2 kf = __bfloat1622float2(kv);
            local_dot = fmaf(qp_reg.x, kf.x, local_dot);
            local_dot = fmaf(qp_reg.y, kf.y, local_dot);
        }

        float score = warp_reduce_sum(local_dot);
        score = __shfl_sync(0xffffffff, score, 0) * sm_scale;

        float m_new = fmaxf(m, score);
        float alpha = (m <= NEG_INF_F * 0.5f) ? 0.0f : __expf(m - m_new);
        float beta  = __expf(score - m_new);
        l = l * alpha + beta;
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            o_reg[j].x = o_reg[j].x * alpha + beta * k_cache[j].x;
            o_reg[j].y = o_reg[j].y * alpha + beta * k_cache[j].y;
        }
    }

    int out_base = (((t * 64 + s) * NUM_HEADS + h) * D_NOPE);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float4 v;
        v.x = o_reg[2 * j + 0].x;
        v.y = o_reg[2 * j + 0].y;
        v.z = o_reg[2 * j + 1].x;
        v.w = o_reg[2 * j + 1].y;
        *reinterpret_cast<float4*>(O_tmp + out_base + j * 128 + lane * 4) = v;
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
    float* __restrict__ lse_ptr,
    int num_tokens) {

    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= num_tokens) return;

    int lane = threadIdx.x;

    __shared__ float m_s[64];
    __shared__ float l_s[64];
    __shared__ float scale_s[64];
    __shared__ float m_global;
    __shared__ float l_global;

    if (lane < 64) {
        m_s[lane] = m_tmp[(t * NUM_HEADS + h) * 64 + lane];
        l_s[lane] = l_tmp[(t * NUM_HEADS + h) * 64 + lane];
    }
    __syncthreads();

    if (lane == 0) {
        float mmax = NEG_INF_F;
        #pragma unroll
        for (int i = 0; i < 64; ++i) mmax = fmaxf(mmax, m_s[i]);
        float lsum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            if (m_s[i] > NEG_INF_F * 0.5f) lsum = fmaf(l_s[i], __expf(m_s[i] - mmax), lsum);
        }
        m_global = mmax;
        l_global = lsum;
        lse_ptr[t * NUM_HEADS + h] = (lsum > 0.0f) ? (mmax * LOG2E_F + __log2f(lsum)) : NEG_INF_F;
    }
    __syncthreads();

    if (lane < 64) {
        scale_s[lane] = (l_global > 0.0f && m_s[lane] > NEG_INF_F * 0.5f)
            ? __fdividef(__expf(m_s[lane] - m_global), l_global)
            : 0.0f;
    }
    __syncthreads();

    int d = lane * 4;
    if (d < D_NOPE) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll 16
        for (int s = 0; s < 64; ++s) {
            float scale = scale_s[s];
            if (scale == 0.0f) continue;
            int base = (((t * 64 + s) * NUM_HEADS + h) * D_NOPE) + d;
            float4 v = *reinterpret_cast<const float4*>(O_tmp + base);
            acc.x = fmaf(v.x, scale, acc.x);
            acc.y = fmaf(v.y, scale, acc.y);
            acc.z = fmaf(v.z, scale, acc.z);
            acc.w = fmaf(v.w, scale, acc.w);
        }
        __nv_bfloat16* out = output_ptr + ((size_t)t * NUM_HEADS + h) * D_NOPE + d;
        *reinterpret_cast<__nv_bfloat162*>(out + 0) = __floats2bfloat162_rn(acc.x, acc.y);
        *reinterpret_cast<__nv_bfloat162*>(out + 2) = __floats2bfloat162_rn(acc.z, acc.w);
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
    TORCH_CHECK(q_pe.size(0) == q_nope.size(0) && sparse_indices.size(0) == q_nope.size(0), "token count mismatch");

    int num_tokens = static_cast<int>(q_nope.size(0));
    auto output = torch::empty({num_tokens, NUM_HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, NUM_HEADS}, torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    auto stream = at::cuda::getDefaultCUDAStream();

    if (num_tokens <= 8) {
        auto O_tmp = torch::empty({num_tokens, 64, NUM_HEADS, D_NOPE}, torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device()));
        auto m_tmp = torch::empty({num_tokens, NUM_HEADS, 64}, torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device()));
        auto l_tmp = torch::empty({num_tokens, NUM_HEADS, 64}, torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device()));

        dim3 grid0(64, num_tokens, 1);
        dim3 block0(32, NUM_HEADS, 1);
        split_k_compute_kernel_vec4<<<grid0, block0, 0, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            num_tokens);

        dim3 grid1(num_tokens, NUM_HEADS, 1);
        dim3 block1(128, 1, 1);
        split_k_reduce_kernel<<<grid1, block1, 0, stream.stream()>>>(
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            num_tokens);
    } else {
        dim3 grid(num_tokens, 1, 1);
        dim3 block(32, NUM_HEADS, 1);
        dsa_forward_streaming_kernel<<<grid, block, 0, stream.stream()>>>(
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
