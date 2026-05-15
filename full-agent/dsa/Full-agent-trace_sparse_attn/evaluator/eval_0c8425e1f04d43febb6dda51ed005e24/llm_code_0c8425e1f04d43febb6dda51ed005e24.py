#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == torch::kBFloat16, #x " must be bfloat16")
#define CHECK_I32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused one-pass online-softmax kernel.
// Devised from Plan: replace split-K temporary workspaces with a persistent fused path
// to avoid O_tmp/m_tmp/l_tmp HBM traffic while preserving exact FP32 online recurrence.
__global__ void dsa_fused_persistent_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    int num_tokens,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    constexpr int HEADS = 16;
    constexpr int DNOPE = 512;
    constexpr int DPE = 64;
    constexpr int TOPK = 2048;
    constexpr int JOBS_PER_TOKEN = HEADS;

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int warps_per_block = blockDim.y;
    const int global_warp = blockIdx.x * warps_per_block + warp_id;
    const int total_warps = gridDim.x * warps_per_block;
    const int total_jobs = num_tokens * JOBS_PER_TOKEN;

    __shared__ int s_idx[8][32];
    int* warp_idx = &s_idx[warp_id][0];

    for (int job = global_warp; job < total_jobs; job += total_warps) {
        const int t = job >> 4;
        const int h = job & 15;

        const __nv_bfloat16* qn_ptr = q_nope_ptr + ((int64_t)t * HEADS + h) * DNOPE;
        const __nv_bfloat16* qp_ptr = q_pe_ptr + ((int64_t)t * HEADS + h) * DPE;
        __nv_bfloat16* out_ptr = output_ptr + ((int64_t)t * HEADS + h) * DNOPE;
        const int32_t* idx_ptr = sparse_indices_ptr + (int64_t)t * TOPK;

        float2 qn_reg[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(qn_ptr + j * 64 + lane * 2);
            qn_reg[j] = __bfloat1622float2(v);
        }
        const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(qp_ptr + lane * 2);
        const float2 qp_reg = __bfloat1622float2(qpv);

        float2 acc[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.0f, 0.0f);
        float m = -CUDART_INF_F;
        float l = 0.0f;

        #pragma unroll 1
        for (int tile = 0; tile < TOPK; tile += 32) {
            warp_idx[lane] = idx_ptr[tile + lane];
            __syncwarp();

            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int idx = warp_idx[i];
                if (idx == -1) continue;

                const __nv_bfloat16* kc_ptr = ckv_cache_ptr + (int64_t)idx * DNOPE;
                const __nv_bfloat16* kp_ptr = kpe_cache_ptr + (int64_t)idx * DPE;

                float local_dot = 0.0f;
                float2 vreg[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(kc_ptr + j * 64 + lane * 2);
                    const float2 kf = __bfloat1622float2(kv);
                    vreg[j] = kf;
                    local_dot = fmaf(qn_reg[j].x, kf.x, local_dot);
                    local_dot = fmaf(qn_reg[j].y, kf.y, local_dot);
                }
                const __nv_bfloat162 kpv = *reinterpret_cast<const __nv_bfloat162*>(kp_ptr + lane * 2);
                const float2 kpf = __bfloat1622float2(kpv);
                local_dot = fmaf(qp_reg.x, kpf.x, local_dot);
                local_dot = fmaf(qp_reg.y, kpf.y, local_dot);

                float z = warp_reduce_sum(local_dot);
                z = __shfl_sync(0xffffffff, z, 0) * sm_scale;

                const float m_new = fmaxf(m, z);
                const float alpha = (m == -CUDART_INF_F) ? 0.0f : __expf(m - m_new);
                const float beta = __expf(z - m_new);
                l = fmaf(l, alpha, beta);
                m = m_new;

                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[j].x = fmaf(beta, vreg[j].x, acc[j].x * alpha);
                    acc[j].y = fmaf(beta, vreg[j].y, acc[j].y * alpha);
                }
            }
            __syncwarp();
        }

        if (l > 0.0f) {
            const float inv_l = __fdividef(1.0f, l);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j].x *= inv_l;
                acc[j].y *= inv_l;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j] = make_float2(0.0f, 0.0f);
            }
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 outv = __floats2bfloat162_rn(acc[j].x, acc[j].y);
            *reinterpret_cast<__nv_bfloat162*>(out_ptr + j * 64 + lane * 2) = outv;
        }
        if (lane == 0) {
            lse_ptr[t * HEADS + h] = (l > 0.0f) ? fmaf(m, 1.4426950408889634f, __log2f(l)) : -CUDART_INF_F;
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
    CHECK_BF16(q_nope);
    CHECK_BF16(q_pe);
    CHECK_BF16(ckv_cache);
    CHECK_BF16(kpe_cache);
    CHECK_I32(sparse_indices);
    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == 16 && q_nope.size(2) == 512, "q_nope must be [T,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == 16 && q_pe.size(2) == 64, "q_pe must be [T,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == 64 && ckv_cache.size(2) == 512, "ckv_cache must be [P,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == 64 && kpe_cache.size(2) == 64, "kpe_cache must be [P,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == 2048, "sparse_indices must be [T,2048]");
    TORCH_CHECK(q_pe.size(0) == q_nope.size(0), "token count mismatch between q_nope and q_pe");
    TORCH_CHECK(sparse_indices.size(0) == q_nope.size(0), "token count mismatch with sparse_indices");
    TORCH_CHECK(kpe_cache.size(0) == ckv_cache.size(0), "page count mismatch between caches");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q_nope));
    const int num_tokens = static_cast<int>(q_nope.size(0));

    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16}, torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));
    if (num_tokens == 0) return {output, lse};

    int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int warps_per_block = (num_tokens < 192) ? 8 : 4;
    int target_warps = (num_tokens < 192) ? sm_count * 4 : num_tokens * 16;
    int blocks = (target_warps + warps_per_block - 1) / warps_per_block;
    if (blocks < 1) blocks = 1;

    dim3 block(32, warps_per_block, 1);
    dim3 grid(blocks, 1, 1);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    dsa_fused_persistent_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
        sparse_indices.data_ptr<int32_t>(),
        num_tokens,
        sm_scale,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>()
    );

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}
