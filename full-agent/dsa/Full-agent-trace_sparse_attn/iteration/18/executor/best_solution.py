#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).scalar_type() == (dt), #x " has wrong dtype")

static constexpr int kNumHeads = 16;
static constexpr int kQNopeDim = 512;
static constexpr int kQPeDim = 64;
static constexpr int kTopK = 2048;
static constexpr int kTile = 32;
static constexpr float kLog2e = 1.4426950408889634f;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void dsa_fused_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    const int num_tokens,
    const float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr) {

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int warps_per_block = blockDim.y;
    const int global_warp = blockIdx.x * warps_per_block + warp_id;
    const int total_jobs = num_tokens * kNumHeads;
    if (global_warp >= total_jobs) return;

    const int t = global_warp / kNumHeads;
    const int h = global_warp - t * kNumHeads;

    __shared__ alignas(16) int idx_shared[8][kTile];
    int* my_idx = idx_shared[warp_id];

    const __nv_bfloat16* qn_base = q_nope_ptr + ((t * kNumHeads + h) * kQNopeDim);
    const __nv_bfloat16* qp_base = q_pe_ptr + ((t * kNumHeads + h) * kQPeDim);

    float2 qn_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(qn_base + j * 64 + lane * 2);
        qn_reg[j] = __bfloat1622float2(v);
    }
    const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(qp_base + lane * 2);
    const float2 qp_reg = __bfloat1622float2(qpv);

    float2 out_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) out_reg[j] = make_float2(0.f, 0.f);

    float m = -INFINITY;
    float l = 0.f;

    const int* sparse_base = sparse_indices_ptr + t * kTopK;

    #pragma unroll 1
    for (int tile = 0; tile < kTopK / kTile; ++tile) {
        my_idx[lane] = sparse_base[tile * kTile + lane];
        __syncwarp();

        #pragma unroll
        for (int i = 0; i < kTile; ++i) {
            const int idx = my_idx[i];
            if (idx == -1) continue;

            const __nv_bfloat16* kc_base = ckv_cache_ptr + idx * kQNopeDim;
            const __nv_bfloat16* kp_base = kpe_cache_ptr + idx * kQPeDim;

            float local_dot = 0.f;
            float2 kv_reg[8];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(kc_base + j * 64 + lane * 2);
                const float2 kf = __bfloat1622float2(kv);
                kv_reg[j] = kf;
                local_dot = fmaf(qn_reg[j].x, kf.x, local_dot);
                local_dot = fmaf(qn_reg[j].y, kf.y, local_dot);
            }

            const __nv_bfloat162 kpv = *reinterpret_cast<const __nv_bfloat162*>(kp_base + lane * 2);
            const float2 kpf = __bfloat1622float2(kpv);
            local_dot = fmaf(qp_reg.x, kpf.x, local_dot);
            local_dot = fmaf(qp_reg.y, kpf.y, local_dot);

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            const float m_new = fmaxf(m, logit);
            const float alpha = (m == -INFINITY) ? 0.f : __expf(m - m_new);
            const float beta = __expf(logit - m_new);
            l = fmaf(l, alpha, beta);
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                out_reg[j].x = fmaf(beta, kv_reg[j].x, out_reg[j].x * alpha);
                out_reg[j].y = fmaf(beta, kv_reg[j].y, out_reg[j].y * alpha);
            }
        }
    }

    const __nv_bfloat16 zero_bf16 = __float2bfloat16(0.f);
    __nv_bfloat16* out_base = output_ptr + ((t * kNumHeads + h) * kQNopeDim);
    if (l > 0.f) {
        const float inv_l = __fdividef(1.f, l);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_reg[j].x *= inv_l;
            out_reg[j].y *= inv_l;
            const __nv_bfloat162 outv = __floats2bfloat162_rn(out_reg[j].x, out_reg[j].y);
            *reinterpret_cast<__nv_bfloat162*>(out_base + j * 64 + lane * 2) = outv;
        }
        if (lane == 0) {
            lse_ptr[t * kNumHeads + h] = fmaf(m, kLog2e, __log2f(l));
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_base[j * 64 + lane * 2] = zero_bf16;
            out_base[j * 64 + lane * 2 + 1] = zero_bf16;
        }
        if (lane == 0) {
            lse_ptr[t * kNumHeads + h] = -INFINITY;
        }
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

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == kNumHeads && q_nope.size(2) == kQNopeDim,
                "q_nope must have shape [num_tokens, 16, 512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(0) == q_nope.size(0) && q_pe.size(1) == kNumHeads && q_pe.size(2) == kQPeDim,
                "q_pe must have shape [num_tokens, 16, 64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == 64 && ckv_cache.size(2) == kQNopeDim,
                "ckv_cache must have shape [num_pages, 64, 512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(0) == ckv_cache.size(0) && kpe_cache.size(1) == 64 && kpe_cache.size(2) == kQPeDim,
                "kpe_cache must have shape [num_pages, 64, 64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(0) == q_nope.size(0) && sparse_indices.size(1) == kTopK,
                "sparse_indices must have shape [num_tokens, 2048]");

    const c10::cuda::CUDAGuard device_guard(q_nope.device());

    const int num_tokens = static_cast<int>(q_nope.size(0));
    auto output = torch::empty({num_tokens, kNumHeads, kQNopeDim}, q_nope.options());
    auto lse = torch::empty({num_tokens, kNumHeads}, torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));

    if (num_tokens == 0) {
        return {output, lse};
    }

    // Deviation from plan: use a single fused kernel for all workloads.
    // This removes split-K temporary HBM traffic while preserving the parent's
    // fast one-pass online softmax structure and low register footprint.
    constexpr int warps_per_block = 8;
    dim3 block(32, warps_per_block, 1);
    dim3 grid((num_tokens * kNumHeads + warps_per_block - 1) / warps_per_block, 1, 1);
    auto stream = at::cuda::getDefaultCUDAStream();

    dsa_fused_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
        sparse_indices.data_ptr<int32_t>(),
        num_tokens,
        sm_scale,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>());

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}
