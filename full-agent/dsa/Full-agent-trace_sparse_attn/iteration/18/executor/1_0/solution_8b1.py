#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == torch::kBFloat16, #x " must be bfloat16")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

static constexpr int HEADS = 16;
static constexpr int D_NOPE = 512;
static constexpr int D_PE = 64;
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

__device__ __forceinline__ int load_int4_row_bf16(
    const __nv_bfloat16* src,
    __nv_bfloat16* dst,
    int elems,
    int lane
) {
    int vecs = elems / 8; // int4 = 16B = 8 bf16
    for (int v = lane; v < vecs; v += 32) {
        reinterpret_cast<int4*>(dst)[v] = reinterpret_cast<const int4*>(src)[v];
    }
    return vecs;
}

// Small-workload path: persistent scheduler over (token, head) jobs.
// Devised from Plan: keep fused online softmax, but avoid split-K temporary HBM workspaces.
__global__ void dsa_small_persistent_kernel(
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
    int lane = threadIdx.x;
    int warp_id = threadIdx.y;
    int warps_per_block = blockDim.y;
    int global_warp = blockIdx.x * warps_per_block + warp_id;
    int warp_stride = gridDim.x * warps_per_block;

    __shared__ alignas(16) int idx_shared[8][32];
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[8][32 * D_NOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[8][32 * D_PE];

    for (int job = global_warp; job < num_tokens * HEADS; job += warp_stride) {
        int t = job / HEADS;
        int h = job - t * HEADS;

        const __nv_bfloat16* qn = q_nope_ptr + (static_cast<long long>(t) * HEADS + h) * D_NOPE;
        const __nv_bfloat16* qp = q_pe_ptr + (static_cast<long long>(t) * HEADS + h) * D_PE;
        __nv_bfloat16* out = output_ptr + (static_cast<long long>(t) * HEADS + h) * D_NOPE;

        float2 qn_reg[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(qn + j * 64 + lane * 2);
            qn_reg[j] = __bfloat1622float2(v);
        }
        __nv_bfloat162 vp = *reinterpret_cast<const __nv_bfloat162*>(qp + lane * 2);
        float2 qp_reg = __bfloat1622float2(vp);

        float2 acc[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = make_float2(0.0f, 0.0f);
        float m = -INFINITY;
        float l = 0.0f;

        for (int tile = 0; tile < TOPK / 32; ++tile) {
            if (lane < 32) {
                idx_shared[warp_id][lane] = sparse_indices_ptr[t * TOPK + tile * 32 + lane];
            }
            __syncwarp();

            #pragma unroll
            for (int row = 0; row < 32; ++row) {
                int idx = idx_shared[warp_id][row];
                if (idx != -1) {
                    const __nv_bfloat16* kc_src = ckv_cache_ptr + static_cast<long long>(idx) * D_NOPE;
                    const __nv_bfloat16* kp_src = kpe_cache_ptr + static_cast<long long>(idx) * D_PE;
                    __nv_bfloat16* kc_dst = &smem_Kc[warp_id][row * D_NOPE];
                    __nv_bfloat16* kp_dst = &smem_Kp[warp_id][row * D_PE];
                    load_int4_row_bf16(kc_src, kc_dst, D_NOPE, lane);
                    load_int4_row_bf16(kp_src, kp_dst, D_PE, lane);
                }
            }
            __syncwarp();

            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                int idx = idx_shared[warp_id][i];
                if (idx == -1) continue;

                float local_dot = 0.0f;
                float2 kreg[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[warp_id][i * D_NOPE + j * 64 + lane * 2]);
                    float2 kf = __bfloat1622float2(kv);
                    kreg[j] = kf;
                    local_dot = fmaf(qn_reg[j].x, kf.x, local_dot);
                    local_dot = fmaf(qn_reg[j].y, kf.y, local_dot);
                }
                __nv_bfloat162 pkv = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[warp_id][i * D_PE + lane * 2]);
                float2 pkf = __bfloat1622float2(pkv);
                local_dot = fmaf(qp_reg.x, pkf.x, local_dot);
                local_dot = fmaf(qp_reg.y, pkf.y, local_dot);

                float logit = warp_reduce_sum(local_dot);
                logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

                float m_new = fmaxf(m, logit);
                float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
                float beta = __expf(logit - m_new);
                l = l * alpha + beta;
                m = m_new;

                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[j].x = fmaf(beta, kreg[j].x, acc[j].x * alpha);
                    acc[j].y = fmaf(beta, kreg[j].y, acc[j].y * alpha);
                }
            }
            __syncwarp();
        }

        if (l > 0.0f) {
            float inv_l = __fdividef(1.0f, l);
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
            __nv_bfloat162 outv = __floats2bfloat162_rn(acc[j].x, acc[j].y);
            *reinterpret_cast<__nv_bfloat162*>(out + j * 64 + lane * 2) = outv;
        }
        if (lane == 0) {
            lse_ptr[t * HEADS + h] = (l > 0.0f) ? (m * LOG2E_F + __log2f(l)) : -INFINITY;
        }
    }
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
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * D_NOPE];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * D_PE];

    float2 q_n_f32[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 q_val = *reinterpret_cast<const __nv_bfloat162*>(&q_nope_ptr[(static_cast<long long>(t) * HEADS + h) * D_NOPE + j * 64 + lane * 2]);
        q_n_f32[j] = __bfloat1622float2(q_val);
    }

    __nv_bfloat162 q_pe_v = *reinterpret_cast<const __nv_bfloat162*>(&q_pe_ptr[(static_cast<long long>(t) * HEADS + h) * D_PE + lane * 2]);
    float2 q_p_f32 = __bfloat1622float2(q_pe_v);

    float2 O_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);
    float m = -INFINITY;
    float l = 0.0f;

    for (int tile = 0; tile < TOPK / 32; ++tile) {
        if (tid < 32) idx_shared[tid] = sparse_indices_ptr[t * TOPK + tile * 32 + tid];
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid >> 6);
            int load_col = tid & 63;
            int idx = idx_shared[load_row];
            if (idx != -1) {
                reinterpret_cast<int4*>(&smem_Kc[load_row * D_NOPE])[load_col] =
                    reinterpret_cast<const int4*>(&ckv_cache_ptr[static_cast<long long>(idx) * D_NOPE])[load_col];
            }
        }
        if (tid < 256) {
            int load_row = tid >> 3;
            int load_col = tid & 7;
            int idx = idx_shared[load_row];
            if (idx != -1) {
                reinterpret_cast<int4*>(&smem_Kp[load_row * D_PE])[load_col] =
                    reinterpret_cast<const int4*>(&kpe_cache_ptr[static_cast<long long>(idx) * D_PE])[load_col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] == -1) continue;
            float local_dot = 0.0f;
            float2 k_f_reg[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 k_n = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kc[i * D_NOPE + j * 64 + lane * 2]);
                float2 k_f = __bfloat1622float2(k_n);
                k_f_reg[j] = k_f;
                local_dot = fmaf(q_n_f32[j].x, k_f.x, local_dot);
                local_dot = fmaf(q_n_f32[j].y, k_f.y, local_dot);
            }
            __nv_bfloat162 k_p = *reinterpret_cast<const __nv_bfloat162*>(&smem_Kp[i * D_PE + lane * 2]);
            float2 kp_f = __bfloat1622float2(k_p);
            local_dot = fmaf(q_p_f32.x, kp_f.x, local_dot);
            local_dot = fmaf(q_p_f32.y, kp_f.y, local_dot);

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            float m_new = fmaxf(m, logit);
            float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
            float beta = __expf(logit - m_new);
            l = l * alpha + beta;
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                O_reg[j].x = fmaf(beta, k_f_reg[j].x, O_reg[j].x * alpha);
                O_reg[j].y = fmaf(beta, k_f_reg[j].y, O_reg[j].y * alpha);
            }
        }
        __syncthreads();
    }

    if (l > 0.0f) {
        float inv_l = __fdividef(1.0f, l);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            O_reg[j].x *= inv_l;
            O_reg[j].y *= inv_l;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) O_reg[j] = make_float2(0.0f, 0.0f);
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 out_bf16 = __floats2bfloat162_rn(O_reg[j].x, O_reg[j].y);
        *reinterpret_cast<__nv_bfloat162*>(&output_ptr[(static_cast<long long>(t) * HEADS + h) * D_NOPE + j * 64 + lane * 2]) = out_bf16;
    }
    if (lane == 0) {
        lse_ptr[t * HEADS + h] = (l > 0.0f) ? (m * LOG2E_F + __log2f(l)) : -INFINITY;
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
    CHECK_INT32(sparse_indices);

    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == HEADS && q_nope.size(2) == D_NOPE, "q_nope shape must be [T,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == HEADS && q_pe.size(2) == D_PE, "q_pe shape must be [T,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == PAGE && ckv_cache.size(2) == D_NOPE, "ckv_cache shape must be [P,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == PAGE && kpe_cache.size(2) == D_PE, "kpe_cache shape must be [P,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == TOPK, "sparse_indices shape must be [T,2048]");
    TORCH_CHECK(q_pe.size(0) == q_nope.size(0), "token count mismatch between q_nope and q_pe");
    TORCH_CHECK(sparse_indices.size(0) == q_nope.size(0), "token count mismatch between q_nope and sparse_indices");
    TORCH_CHECK(kpe_cache.size(0) == ckv_cache.size(0), "page count mismatch between caches");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q_nope));
    int num_tokens = q_nope.size(0);
    auto output = torch::empty({num_tokens, HEADS, D_NOPE}, q_nope.options());
    auto lse = torch::empty({num_tokens, HEADS}, torch::dtype(torch::kFloat32).device(q_nope.device()));
    if (num_tokens == 0) return {output, lse};

    auto stream = at::cuda::getDefaultCUDAStream();

    auto* qn_ptr = reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>());
    auto* qp_ptr = reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>());
    auto* kc_ptr = reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>());
    auto* kp_ptr = reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>());
    auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    auto* idx_ptr = sparse_indices.data_ptr<int32_t>();
    auto* lse_ptr = lse.data_ptr<float>();

    if (num_tokens <= 128) {
        int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        int blocks = sm_count;
        dim3 block(32, 8, 1);
        dim3 grid(blocks, 1, 1);
        dsa_small_persistent_kernel<<<grid, block, 0, stream>>>(
            qn_ptr, qp_ptr, kc_ptr, kp_ptr, idx_ptr, num_tokens, sm_scale, out_ptr, lse_ptr);
    } else {
        dim3 block(32, HEADS, 1);
        dim3 grid(num_tokens, 1, 1);
        dsa_large_kernel<<<grid, block, 0, stream>>>(
            qn_ptr, qp_ptr, kc_ptr, kp_ptr, idx_ptr, sm_scale, out_ptr, lse_ptr);
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA Forward");
}
