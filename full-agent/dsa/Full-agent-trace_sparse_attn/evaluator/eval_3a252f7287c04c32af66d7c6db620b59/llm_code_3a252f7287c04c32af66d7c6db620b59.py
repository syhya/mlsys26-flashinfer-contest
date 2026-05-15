#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == at::kBFloat16, #x " must be bfloat16")
#define CHECK_I32(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Single-pass online softmax kernel. One block processes one token, 16 warps process 16 heads.
// Devised from parent: keep the stronger single-pass path, but reduce register pressure by
// streaming output accumulation directly in float scalars and using read-only cache loads.
__global__ void dsa_forward_kernel_opt(
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

    __shared__ int idx_shared[32];
    __shared__ __align__(16) __nv_bfloat16 smem_kc[32 * 512];
    __shared__ __align__(16) __nv_bfloat16 smem_kp[32 * 64];

    const __nv_bfloat16* qn_base = q_nope_ptr + t * 16 * 512 + h * 512;
    const __nv_bfloat16* qp_base = q_pe_ptr + t * 16 * 64 + h * 64;

    float qn[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        __nv_bfloat16 v = qn_base[lane * 16 + j];
        qn[j] = __bfloat162float(v);
    }

    float qp0 = __bfloat162float(qp_base[lane * 2 + 0]);
    float qp1 = __bfloat162float(qp_base[lane * 2 + 1]);

    float out[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = 0.0f;

    float m = -CUDART_INF_F;
    float l = 0.0f;

    #pragma unroll 1
    for (int tile = 0; tile < 64; ++tile) {
        if (tid < 32) {
            idx_shared[tid] = sparse_indices_ptr[t * 2048 + tile * 32 + tid];
        }
        __syncthreads();

        // 32 rows x 512 bf16 = 32KB, loaded as float4 vectors (128-bit)
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_row = step * 8 + (tid >> 6);
            int load_col = tid & 63;
            int idx = idx_shared[load_row];
            float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(ckv_cache_ptr + idx * 512);
                val = __ldg(src + load_col);
            }
            float4* dst = reinterpret_cast<float4*>(smem_kc + load_row * 512);
            dst[load_col] = val;
        }

        // 32 rows x 64 bf16 = 4KB
        if (tid < 256) {
            int load_row = tid >> 3;
            int load_col = tid & 7;
            int idx = idx_shared[load_row];
            float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(kpe_cache_ptr + idx * 64);
                val = __ldg(src + load_col);
            }
            float4* dst = reinterpret_cast<float4*>(smem_kp + load_row * 64);
            dst[load_col] = val;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] == -1) continue;

            const __nv_bfloat16* kc_row = smem_kc + i * 512 + lane * 16;
            const __nv_bfloat16* kp_row = smem_kp + i * 64 + lane * 2;

            float local_dot = 0.0f;
            float kvals[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                float kv = __bfloat162float(kc_row[j]);
                kvals[j] = kv;
                local_dot = fmaf(qn[j], kv, local_dot);
            }
            float kpv0 = __bfloat162float(kp_row[0]);
            float kpv1 = __bfloat162float(kp_row[1]);
            local_dot = fmaf(qp0, kpv0, local_dot);
            local_dot = fmaf(qp1, kpv1, local_dot);

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            float m_new = fmaxf(m, logit);
            float alpha = (m == -CUDART_INF_F) ? 0.0f : __expf(m - m_new);
            float beta = __expf(logit - m_new);
            l = l * alpha + beta;
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                out[j] = out[j] * alpha + beta * kvals[j];
            }
        }
        __syncthreads();
    }

    float inv_l = (l > 0.0f) ? __fdividef(1.0f, l) : 0.0f;
    __nv_bfloat16* out_base = output_ptr + t * 16 * 512 + h * 512;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        out_base[lane * 16 + j] = __float2bfloat16(out[j] * inv_l);
    }

    if (lane == 0) {
        lse_ptr[t * 16 + h] = (l > 0.0f) ? (m * 1.4426950408889634f + __log2f(l)) : -CUDART_INF_F;
    }
}

// Small-workload path: increase grid size by splitting heads over tiles but avoid temporary O buffer.
// Each block handles one (token, head, chunk), writes local stats, then a compact reduction finalizes.
__global__ void split_k_stats_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float* __restrict__ m_tmp,
    float* __restrict__ l_tmp,
    float* __restrict__ o_tmp,
    int num_tokens
) {
    int s = blockIdx.x;
    int t = blockIdx.y;
    if (t >= num_tokens) return;

    int h = threadIdx.y;
    int lane = threadIdx.x;
    int tid = h * 32 + lane;

    __shared__ int idx_shared[32];
    __shared__ __align__(16) __nv_bfloat16 smem_kc[32 * 512];
    __shared__ __align__(16) __nv_bfloat16 smem_kp[32 * 64];

    const __nv_bfloat16* qn_base = q_nope_ptr + t * 16 * 512 + h * 512;
    const __nv_bfloat16* qp_base = q_pe_ptr + t * 16 * 64 + h * 64;

    float qn[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) qn[j] = __bfloat162float(qn_base[lane * 16 + j]);
    float qp0 = __bfloat162float(qp_base[lane * 2 + 0]);
    float qp1 = __bfloat162float(qp_base[lane * 2 + 1]);

    float out[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = 0.0f;
    float m = -CUDART_INF_F;
    float l = 0.0f;

    if (tid < 32) idx_shared[tid] = sparse_indices_ptr[t * 2048 + s * 32 + tid];
    __syncthreads();

    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        int load_row = step * 8 + (tid >> 6);
        int load_col = tid & 63;
        int idx = idx_shared[load_row];
        float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
        if (idx != -1) {
            const float4* src = reinterpret_cast<const float4*>(ckv_cache_ptr + idx * 512);
            val = __ldg(src + load_col);
        }
        float4* dst = reinterpret_cast<float4*>(smem_kc + load_row * 512);
        dst[load_col] = val;
    }
    if (tid < 256) {
        int load_row = tid >> 3;
        int load_col = tid & 7;
        int idx = idx_shared[load_row];
        float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
        if (idx != -1) {
            const float4* src = reinterpret_cast<const float4*>(kpe_cache_ptr + idx * 64);
            val = __ldg(src + load_col);
        }
        float4* dst = reinterpret_cast<float4*>(smem_kp + load_row * 64);
        dst[load_col] = val;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (idx_shared[i] == -1) continue;
        const __nv_bfloat16* kc_row = smem_kc + i * 512 + lane * 16;
        const __nv_bfloat16* kp_row = smem_kp + i * 64 + lane * 2;
        float local_dot = 0.0f;
        float kvals[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            float kv = __bfloat162float(kc_row[j]);
            kvals[j] = kv;
            local_dot = fmaf(qn[j], kv, local_dot);
        }
        local_dot = fmaf(qp0, __bfloat162float(kp_row[0]), local_dot);
        local_dot = fmaf(qp1, __bfloat162float(kp_row[1]), local_dot);

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        float m_new = fmaxf(m, logit);
        float alpha = (m == -CUDART_INF_F) ? 0.0f : __expf(m - m_new);
        float beta = __expf(logit - m_new);
        l = l * alpha + beta;
        m = m_new;
        #pragma unroll
        for (int j = 0; j < 16; ++j) out[j] = out[j] * alpha + beta * kvals[j];
    }

    int stats_idx = t * 16 * 64 + h * 64 + s;
    if (lane == 0) {
        m_tmp[stats_idx] = m;
        l_tmp[stats_idx] = l;
    }
    int out_idx = (((t * 64 + s) * 16 + h) * 512) + lane * 16;
    #pragma unroll
    for (int j = 0; j < 16; ++j) o_tmp[out_idx + j] = out[j];
}

__global__ void split_k_finalize_kernel(
    const float* __restrict__ m_tmp,
    const float* __restrict__ l_tmp,
    const float* __restrict__ o_tmp,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr,
    int num_tokens
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int lane = threadIdx.x;
    if (t >= num_tokens || lane >= 32) return;

    float m = -CUDART_INF_F;
    #pragma unroll
    for (int s = 0; s < 64; ++s) {
        float ms = m_tmp[t * 16 * 64 + h * 64 + s];
        m = fmaxf(m, ms);
    }

    float l = 0.0f;
    #pragma unroll
    for (int s = 0; s < 64; ++s) {
        float ms = m_tmp[t * 16 * 64 + h * 64 + s];
        float ls = l_tmp[t * 16 * 64 + h * 64 + s];
        if (ms != -CUDART_INF_F) l = fmaf(ls, __expf(ms - m), l);
    }

    float accum[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) accum[j] = 0.0f;

    if (l > 0.0f) {
        float inv_l = __fdividef(1.0f, l);
        #pragma unroll
        for (int s = 0; s < 64; ++s) {
            float ms = m_tmp[t * 16 * 64 + h * 64 + s];
            if (ms == -CUDART_INF_F) continue;
            float scale = __expf(ms - m) * inv_l;
            int out_idx = (((t * 64 + s) * 16 + h) * 512) + lane * 16;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                accum[j] = fmaf(o_tmp[out_idx + j], scale, accum[j]);
            }
        }
    }

    __nv_bfloat16* out_base = output_ptr + t * 16 * 512 + h * 512 + lane * 16;
    #pragma unroll
    for (int j = 0; j < 16; ++j) out_base[j] = __float2bfloat16(accum[j]);

    if (lane == 0) {
        lse_ptr[t * 16 + h] = (l > 0.0f) ? (m * 1.4426950408889634f + __log2f(l)) : -CUDART_INF_F;
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

    int num_tokens = static_cast<int>(q_nope.size(0));
    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16}, torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));
    if (num_tokens == 0) return {output, lse};

    auto stream = at::cuda::getDefaultCUDAStream();

    if (num_tokens < 8) {
        auto m_tmp = torch::empty({num_tokens, 16, 64}, lse.options());
        auto l_tmp = torch::empty({num_tokens, 16, 64}, lse.options());
        auto o_tmp = torch::empty({num_tokens, 64, 16, 512}, lse.options());

        dim3 grid_stats(64, num_tokens, 1);
        dim3 block_stats(32, 16, 1);
        split_k_stats_kernel<<<grid_stats, block_stats, 0, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            o_tmp.data_ptr<float>(),
            num_tokens
        );

        dim3 grid_fin(num_tokens, 16, 1);
        dim3 block_fin(32, 1, 1);
        split_k_finalize_kernel<<<grid_fin, block_fin, 0, stream.stream()>>>(
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            o_tmp.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            num_tokens
        );
    } else {
        dim3 grid(num_tokens, 1, 1);
        dim3 block(32, 16, 1);
        dsa_forward_kernel_opt<<<grid, block, 0, stream.stream()>>>(
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
