#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == torch::kBFloat16, #x " must be bfloat16")
#define CHECK_I32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void dsa_forward_main_kernel(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    const int t = blockIdx.x;
    const int h = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = h * 32 + lane;

    __shared__ alignas(16) int idx_shared[32];
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * 512];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * 64];

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            &q_nope_ptr[t * (16 * 512) + h * 512 + j * 64 + lane * 2]);
        qn[j] = __bfloat1622float2(qv);
    }
    const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(
        &q_pe_ptr[t * (16 * 64) + h * 64 + lane * 2]);
    const float2 qp = __bfloat1622float2(qpv);

    float2 out_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) out_reg[j] = make_float2(0.f, 0.f);

    float m = -INFINITY;
    float l = 0.f;

    #pragma unroll 1
    for (int tile = 0; tile < 64; ++tile) {
        if (tid < 32) idx_shared[tid] = sparse_indices_ptr[t * 2048 + tile * 32 + tid];
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int row = step * 8 + (tid >> 6);
            const int col = tid & 63;
            const int idx = idx_shared[row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(&ckv_cache_ptr[idx * 512]);
                float4* dst = reinterpret_cast<float4*>(&smem_Kc[row * 512]);
                dst[col] = src[col];
            }
        }
        if (tid < 256) {
            const int row = tid >> 3;
            const int col = tid & 7;
            const int idx = idx_shared[row];
            if (idx != -1) {
                const float4* src = reinterpret_cast<const float4*>(&kpe_cache_ptr[idx * 64]);
                float4* dst = reinterpret_cast<float4*>(&smem_Kp[row * 64]);
                dst[col] = src[col];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (idx_shared[i] == -1) continue;

            float local_dot = 0.f;
            float2 kc_frag[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                    &smem_Kc[i * 512 + j * 64 + lane * 2]);
                const float2 kf = __bfloat1622float2(kv);
                kc_frag[j] = kf;
                local_dot = fmaf(qn[j].x, kf.x, local_dot);
                local_dot = fmaf(qn[j].y, kf.y, local_dot);
            }
            const __nv_bfloat162 kpv2 = *reinterpret_cast<const __nv_bfloat162*>(
                &smem_Kp[i * 64 + lane * 2]);
            const float2 kpf = __bfloat1622float2(kpv2);
            local_dot = fmaf(qp.x, kpf.x, local_dot);
            local_dot = fmaf(qp.y, kpf.y, local_dot);

            float logit = warp_reduce_sum(local_dot);
            logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

            const float m_new = fmaxf(m, logit);
            const float alpha = isinf(m) ? 0.f : __expf(m - m_new);
            const float p = __expf(logit - m_new);
            l = fmaf(l, alpha, p);
            m = m_new;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                out_reg[j].x = fmaf(p, kc_frag[j].x, out_reg[j].x * alpha);
                out_reg[j].y = fmaf(p, kc_frag[j].y, out_reg[j].y * alpha);
            }
        }
        __syncthreads();
    }

    const float inv_l = (l > 0.f) ? __fdividef(1.f, l) : 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        out_reg[j].x *= inv_l;
        out_reg[j].y *= inv_l;
        const __nv_bfloat162 outv = __floats2bfloat162_rn(out_reg[j].x, out_reg[j].y);
        *reinterpret_cast<__nv_bfloat162*>(
            &output_ptr[t * (16 * 512) + h * 512 + j * 64 + lane * 2]) = outv;
    }
    if (lane == 0) {
        lse_ptr[t * 16 + h] = (l > 0.f) ? (m * 1.4426950408889634f + __log2f(l)) : -INFINITY;
    }
}

__global__ void split_k_compute_kernel_v4(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache_ptr,
    const __nv_bfloat16* __restrict__ kpe_cache_ptr,
    const int32_t* __restrict__ sparse_indices_ptr,
    float sm_scale,
    float* __restrict__ O_tmp,
    float* __restrict__ m_tmp,
    float* __restrict__ l_tmp
) {
    const int s = blockIdx.x;
    const int t = blockIdx.y;
    const int h = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = h * 32 + lane;

    __shared__ alignas(16) int idx_shared[32];
    __shared__ alignas(16) __nv_bfloat16 smem_Kc[32 * 512];
    __shared__ alignas(16) __nv_bfloat16 smem_Kp[32 * 64];

    float2 qn[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            &q_nope_ptr[t * (16 * 512) + h * 512 + j * 64 + lane * 2]);
        qn[j] = __bfloat1622float2(qv);
    }
    const __nv_bfloat162 qpv = *reinterpret_cast<const __nv_bfloat162*>(
        &q_pe_ptr[t * (16 * 64) + h * 64 + lane * 2]);
    const float2 qp = __bfloat1622float2(qpv);

    float2 out_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) out_reg[j] = make_float2(0.f, 0.f);
    float m = -INFINITY;
    float l = 0.f;

    if (tid < 32) idx_shared[tid] = sparse_indices_ptr[t * 2048 + s * 32 + tid];
    __syncthreads();

    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int row = step * 8 + (tid >> 6);
        const int col = tid & 63;
        const int idx = idx_shared[row];
        if (idx != -1) {
            const float4* src = reinterpret_cast<const float4*>(&ckv_cache_ptr[idx * 512]);
            float4* dst = reinterpret_cast<float4*>(&smem_Kc[row * 512]);
            dst[col] = src[col];
        }
    }
    if (tid < 256) {
        const int row = tid >> 3;
        const int col = tid & 7;
        const int idx = idx_shared[row];
        if (idx != -1) {
            const float4* src = reinterpret_cast<const float4*>(&kpe_cache_ptr[idx * 64]);
            float4* dst = reinterpret_cast<float4*>(&smem_Kp[row * 64]);
            dst[col] = src[col];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (idx_shared[i] == -1) continue;
        float local_dot = 0.f;
        float2 kc_frag[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(
                &smem_Kc[i * 512 + j * 64 + lane * 2]);
            const float2 kf = __bfloat1622float2(kv);
            kc_frag[j] = kf;
            local_dot = fmaf(qn[j].x, kf.x, local_dot);
            local_dot = fmaf(qn[j].y, kf.y, local_dot);
        }
        const __nv_bfloat162 kpv2 = *reinterpret_cast<const __nv_bfloat162*>(
            &smem_Kp[i * 64 + lane * 2]);
        const float2 kpf = __bfloat1622float2(kpv2);
        local_dot = fmaf(qp.x, kpf.x, local_dot);
        local_dot = fmaf(qp.y, kpf.y, local_dot);

        float logit = warp_reduce_sum(local_dot);
        logit = __shfl_sync(0xffffffff, logit, 0) * sm_scale;

        const float m_new = fmaxf(m, logit);
        const float alpha = isinf(m) ? 0.f : __expf(m - m_new);
        const float p = __expf(logit - m_new);
        l = fmaf(l, alpha, p);
        m = m_new;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_reg[j].x = fmaf(p, kc_frag[j].x, out_reg[j].x * alpha);
            out_reg[j].y = fmaf(p, kc_frag[j].y, out_reg[j].y * alpha);
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int base = (((t * 64 + s) * 16 + h) * 512) + j * 128 + lane * 4;
        const float4 v = make_float4(
            out_reg[2 * j + 0].x,
            out_reg[2 * j + 0].y,
            out_reg[2 * j + 1].x,
            out_reg[2 * j + 1].y);
        *reinterpret_cast<float4*>(&O_tmp[base]) = v;
    }

    if (lane == 0) {
        m_tmp[t * 16 * 64 + h * 64 + s] = m;
        l_tmp[t * 16 * 64 + h * 64 + s] = l;
    }
}

__global__ void split_k_reduce_kernel(
    const float* __restrict__ O_tmp,
    const float* __restrict__ m_tmp,
    const float* __restrict__ l_tmp,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ lse_ptr
) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;

    __shared__ float m_global;
    __shared__ float l_global;
    __shared__ float m_smem[64];
    __shared__ float l_smem[64];
    __shared__ float scale_smem[64];

    if (lane < 64) {
        m_smem[lane] = m_tmp[t * 16 * 64 + h * 64 + lane];
        l_smem[lane] = l_tmp[t * 16 * 64 + h * 64 + lane];
    }
    __syncthreads();

    if (lane == 0) {
        float mg = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 64; ++i) mg = fmaxf(mg, m_smem[i]);
        float lg = 0.f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            if (!isinf(m_smem[i])) lg = fmaf(l_smem[i], __expf(m_smem[i] - mg), lg);
        }
        m_global = mg;
        l_global = lg;
        lse_ptr[t * 16 + h] = (lg > 0.f) ? (mg * 1.4426950408889634f + __log2f(lg)) : -INFINITY;
    }
    __syncthreads();

    if (lane < 64) {
        scale_smem[lane] = (l_global > 0.f && !isinf(m_smem[lane]))
            ? __fdividef(__expf(m_smem[lane] - m_global), l_global)
            : 0.f;
    }
    __syncthreads();

    const int d = lane * 4;
    if (d < 512) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll 16
        for (int s = 0; s < 64; ++s) {
            const float scale = scale_smem[s];
            if (scale > 0.f) {
                const int base = (((t * 64 + s) * 16 + h) * 512) + d;
                const float4 v = *reinterpret_cast<const float4*>(&O_tmp[base]);
                acc.x = fmaf(v.x, scale, acc.x);
                acc.y = fmaf(v.y, scale, acc.y);
                acc.z = fmaf(v.z, scale, acc.z);
                acc.w = fmaf(v.w, scale, acc.w);
            }
        }
        const __nv_bfloat162 b0 = __floats2bfloat162_rn(acc.x, acc.y);
        const __nv_bfloat162 b1 = __floats2bfloat162_rn(acc.z, acc.w);
        const uint32_t u0 = *reinterpret_cast<const uint32_t*>(&b0);
        const uint32_t u1 = *reinterpret_cast<const uint32_t*>(&b1);
        *reinterpret_cast<uint2*>(&output_ptr[t * (16 * 512) + h * 512 + d]) = make_uint2(u0, u1);
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
    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == 16 && q_nope.size(2) == 512, "q_nope shape must be [T,16,512]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == 16 && q_pe.size(2) == 64, "q_pe shape must be [T,16,64]");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == 64 && ckv_cache.size(2) == 512, "ckv_cache shape must be [P,64,512]");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == 64 && kpe_cache.size(2) == 64, "kpe_cache shape must be [P,64,64]");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(1) == 2048, "sparse_indices shape must be [T,2048]");
    TORCH_CHECK(q_nope.size(0) == q_pe.size(0) && q_nope.size(0) == sparse_indices.size(0), "token dims must match");

    const int num_tokens = static_cast<int>(q_nope.size(0));
    auto output = torch::empty({num_tokens, 16, 512}, q_nope.options());
    auto lse = torch::empty({num_tokens, 16}, torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));
    if (num_tokens == 0) return {output, lse};

    if (num_tokens < 128) {
        constexpr int S = 64;
        auto O_tmp = torch::empty({num_tokens, S, 16, 512}, torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));
        auto m_tmp = torch::empty({num_tokens, 16, S}, torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));
        auto l_tmp = torch::empty({num_tokens, 16, S}, torch::TensorOptions().device(q_nope.device()).dtype(torch::kFloat32));

        dim3 grid1(S, num_tokens, 1);
        dim3 block1(32, 16, 1);
        split_k_compute_kernel_v4<<<grid1, block1>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            sm_scale,
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>()
        );

        dim3 grid2(num_tokens, 16, 1);
        dim3 block2(128, 1, 1);
        split_k_reduce_kernel<<<grid2, block2>>>(
            O_tmp.data_ptr<float>(),
            m_tmp.data_ptr<float>(),
            l_tmp.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>()
        );
    } else {
        dim3 grid(num_tokens, 1, 1);
        dim3 block(32, 16, 1);
        dsa_forward_main_kernel<<<grid, block>>>(
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
