#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <tuple>
#include <cmath>

__device__ __forceinline__ void unpack_bf16x8_to_float4x2(const float4& packed, float4& out0, float4& out1) {
    __nv_bfloat162 bf0 = reinterpret_cast<const __nv_bfloat162&>(packed.x);
    __nv_bfloat162 bf1 = reinterpret_cast<const __nv_bfloat162&>(packed.y);
    __nv_bfloat162 bf2 = reinterpret_cast<const __nv_bfloat162&>(packed.z);
    __nv_bfloat162 bf3 = reinterpret_cast<const __nv_bfloat162&>(packed.w);
    
    float2 f0 = __bfloat1622float2(bf0);
    float2 f1 = __bfloat1622float2(bf1);
    float2 f2 = __bfloat1622float2(bf2);
    float2 f3 = __bfloat1622float2(bf3);
    
    out0 = make_float4(f0.x, f0.y, f1.x, f1.y);
    out1 = make_float4(f2.x, f2.y, f3.x, f3.y);
}

__device__ __forceinline__ float4 pack_float4x2_to_bf16x8(const float4& in0, const float4& in1) {
    __nv_bfloat162 bf0 = __floats2bfloat162_rn(in0.x, in0.y);
    __nv_bfloat162 bf1 = __floats2bfloat162_rn(in0.z, in0.w);
    __nv_bfloat162 bf2 = __floats2bfloat162_rn(in1.x, in1.y);
    __nv_bfloat162 bf3 = __floats2bfloat162_rn(in1.z, in1.w);
    
    return make_float4(
        reinterpret_cast<const float&>(bf0),
        reinterpret_cast<const float&>(bf1),
        reinterpret_cast<const float&>(bf2),
        reinterpret_cast<const float&>(bf3)
    );
}

__global__ void __launch_bounds__(128) gdn_forward_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b,
    const int64_t* __restrict__ cu_seqlens,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state,
    float scale
) {
    extern __shared__ char smem_buf[];
    
    // We want T=32
    // float q[32][128], k[32][128], v_out[32][128]
    // float g[32], beta[32]
    // float state_transpose[128][33]
    
    float* smem_q = reinterpret_cast<float*>(smem_buf);
    float* smem_k = smem_q + 32 * 128;
    float* smem_v_out = smem_k + 32 * 128;
    float* smem_g = smem_v_out + 32 * 128;
    float* smem_beta = smem_g + 32;
    
    float* state_transpose = reinterpret_cast<float*>(smem_buf); // reuse

    int seq_idx = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x; 

    long long seq_start = cu_seqlens[seq_idx];
    long long seq_end = cu_seqlens[seq_idx + 1];
    long long seq_len = seq_end - seq_start;

    float4 state_reg_f4[32];

    if (state != nullptr) {
        size_t state_base = ((size_t)seq_idx * 8 + h) * 16384; 
        const float* block_state = state + state_base;
        const float4* block_state_f4 = reinterpret_cast<const float4*>(block_state);
        
        for (int k_blk = 0; k_blk < 128; k_blk += 32) {
            for (int i = 0; i < 8; ++i) {
                int idx = i * 128 + tid; 
                int r = idx / 8;
                int c_f4 = idx % 8;
                
                float4 val = block_state_f4[r * 32 + (k_blk / 4) + c_f4];
                state_transpose[r * 33 + c_f4 * 4 + 0] = val.x;
                state_transpose[r * 33 + c_f4 * 4 + 1] = val.y;
                state_transpose[r * 33 + c_f4 * 4 + 2] = val.z;
                state_transpose[r * 33 + c_f4 * 4 + 3] = val.w;
            }
            __syncthreads();
            
            #pragma unroll 8
            for (int k_idx = 0; k_idx < 8; k_idx++) {
                float4 val;
                val.x = state_transpose[tid * 33 + k_idx * 4 + 0];
                val.y = state_transpose[tid * 33 + k_idx * 4 + 1];
                val.z = state_transpose[tid * 33 + k_idx * 4 + 2];
                val.w = state_transpose[tid * 33 + k_idx * 4 + 3];
                state_reg_f4[(k_blk / 4) + k_idx] = val;
            }
            __syncthreads();
        }
    } else {
        #pragma unroll 32
        for (int k = 0; k < 32; k++) {
            state_reg_f4[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    float neg_exp_A_log = -__expf(A_log[h]);
    float dt_bias_val = dt_bias[h];

    if (seq_len > 0) {
        int qk_head = h / 2; 
        for (long long t_base = seq_start; t_base < seq_end; t_base += 32) {
            int t_chunk = (int)((seq_end - t_base) < 32LL ? (seq_end - t_base) : 32LL);
            int num_float4s = t_chunk * 16;
            
            for (int i = tid; i < num_float4s; i += 128) {
                int t_rel = i / 16;
                int c_f4 = i % 16;
                int c = c_f4 * 8;
                long long t = t_base + t_rel;
                
                size_t qk_idx = ((size_t)t * 4 + qk_head) * 128 + c;
                float4 q_val = *reinterpret_cast<const float4*>(&q[qk_idx]);
                float4 q_out0, q_out1;
                unpack_bf16x8_to_float4x2(q_val, q_out0, q_out1);
                float4* smem_q_ptr = reinterpret_cast<float4*>(&smem_q[t_rel * 128 + c]);
                smem_q_ptr[0] = q_out0;
                smem_q_ptr[1] = q_out1;
                
                float4 k_val = *reinterpret_cast<const float4*>(&k[qk_idx]);
                float4 k_out0, k_out1;
                unpack_bf16x8_to_float4x2(k_val, k_out0, k_out1);
                float4* smem_k_ptr = reinterpret_cast<float4*>(&smem_k[t_rel * 128 + c]);
                smem_k_ptr[0] = k_out0;
                smem_k_ptr[1] = k_out1;
                
                size_t v_idx = ((size_t)t * 8 + h) * 128 + c;
                float4 v_val = *reinterpret_cast<const float4*>(&v[v_idx]);
                float4 v_out0, v_out1;
                unpack_bf16x8_to_float4x2(v_val, v_out0, v_out1);
                float4* smem_v_ptr = reinterpret_cast<float4*>(&smem_v_out[t_rel * 128 + c]);
                smem_v_ptr[0] = v_out0;
                smem_v_ptr[1] = v_out1;
            }
            
            if (tid < t_chunk) {
                long long t = t_base + tid;
                size_t ab_idx = (size_t)t * 8 + h;
                float a_val = __bfloat162float(a[ab_idx]);
                float b_val = __bfloat162float(b[ab_idx]);
                
                float x = a_val + dt_bias_val;
                float sp_x = (x > 20.0f) ? x : __logf(1.0f + __expf(x));
                smem_g[tid] = __expf(neg_exp_A_log * sp_x);
                smem_beta[tid] = __frcp_rn(1.0f + __expf(-b_val));
            }
            
            __syncthreads(); 
            
            for (int t_rel = 0; t_rel < t_chunk; ++t_rel) {
                float g = smem_g[t_rel];
                float beta = smem_beta[t_rel];
                float v_val = smem_v_out[t_rel * 128 + tid]; 
                
                float old_v_unscaled0 = 0.0f;
                float old_v_unscaled1 = 0.0f;
                float old_v_unscaled2 = 0.0f;
                float old_v_unscaled3 = 0.0f;
                const float4* smem_k_f4 = reinterpret_cast<const float4*>(&smem_k[t_rel * 128]);
                
                #pragma unroll 32
                for (int k_idx_4 = 0; k_idx_4 < 32; k_idx_4++) {
                    float4 k_vec = smem_k_f4[k_idx_4];
                    float4 s_vec = state_reg_f4[k_idx_4];
                    old_v_unscaled0 = fmaf(k_vec.x, s_vec.x, old_v_unscaled0);
                    old_v_unscaled1 = fmaf(k_vec.y, s_vec.y, old_v_unscaled1);
                    old_v_unscaled2 = fmaf(k_vec.z, s_vec.z, old_v_unscaled2);
                    old_v_unscaled3 = fmaf(k_vec.w, s_vec.w, old_v_unscaled3);
                }
                float old_v = g * (old_v_unscaled0 + old_v_unscaled1 + old_v_unscaled2 + old_v_unscaled3);
                
                float delta_v = beta * (v_val - old_v);
                
                float o_val0 = 0.0f;
                float o_val1 = 0.0f;
                float o_val2 = 0.0f;
                float o_val3 = 0.0f;
                const float4* smem_q_f4 = reinterpret_cast<const float4*>(&smem_q[t_rel * 128]);
                
                #pragma unroll 32
                for (int k_idx_4 = 0; k_idx_4 < 32; k_idx_4++) {
                    float4 k_vec = smem_k_f4[k_idx_4];
                    float4 q_vec = smem_q_f4[k_idx_4];
                    float4 s_vec = state_reg_f4[k_idx_4];
                    
                    float s_new_x = fmaf(k_vec.x, delta_v, g * s_vec.x);
                    float s_new_y = fmaf(k_vec.y, delta_v, g * s_vec.y);
                    float s_new_z = fmaf(k_vec.z, delta_v, g * s_vec.z);
                    float s_new_w = fmaf(k_vec.w, delta_v, g * s_vec.w);
                    
                    o_val0 = fmaf(q_vec.x, s_new_x, o_val0);
                    o_val1 = fmaf(q_vec.y, s_new_y, o_val1);
                    o_val2 = fmaf(q_vec.z, s_new_z, o_val2);
                    o_val3 = fmaf(q_vec.w, s_new_w, o_val3);
                    
                    s_vec.x = s_new_x;
                    s_vec.y = s_new_y;
                    s_vec.z = s_new_z;
                    s_vec.w = s_new_w;
                    state_reg_f4[k_idx_4] = s_vec;
                }
                
                float o_val = (o_val0 + o_val1 + o_val2 + o_val3) * scale;
                smem_v_out[t_rel * 128 + tid] = o_val;
            }
            
            __syncthreads(); 
            
            for (int i = tid; i < num_float4s; i += 128) {
                int t_rel = i / 16;
                int c_f4 = i % 16;
                int c = c_f4 * 8;
                long long t = t_base + t_rel;
                
                const float4* smem_v_ptr = reinterpret_cast<const float4*>(&smem_v_out[t_rel * 128 + c]);
                float4 o_in0 = smem_v_ptr[0];
                float4 o_in1 = smem_v_ptr[1];
                float4 out_val = pack_float4x2_to_bf16x8(o_in0, o_in1);
                
                size_t v_idx = ((size_t)t * 8 + h) * 128 + c;
                *reinterpret_cast<float4*>(&output[v_idx]) = out_val;
            }
            
            __syncthreads();
        }
    }

    if (new_state != nullptr) {
        size_t state_base = ((size_t)seq_idx * 8 + h) * 16384;
        float* block_new_state = new_state + state_base;
        float4* block_new_state_f4 = reinterpret_cast<float4*>(block_new_state);
        
        for (int k_blk = 0; k_blk < 128; k_blk += 32) {
            #pragma unroll 8
            for (int k_idx = 0; k_idx < 8; k_idx++) {
                float4 val = state_reg_f4[(k_blk / 4) + k_idx];
                state_transpose[tid * 33 + k_idx * 4 + 0] = val.x;
                state_transpose[tid * 33 + k_idx * 4 + 1] = val.y;
                state_transpose[tid * 33 + k_idx * 4 + 2] = val.z;
                state_transpose[tid * 33 + k_idx * 4 + 3] = val.w;
            }
            __syncthreads();
            
            for (int i = 0; i < 8; ++i) {
                int idx = i * 128 + tid;
                int r = idx / 8;
                int c_f4 = idx % 8;
                
                float4 val;
                val.x = state_transpose[r * 33 + c_f4 * 4 + 0];
                val.y = state_transpose[r * 33 + c_f4 * 4 + 1];
                val.z = state_transpose[r * 33 + c_f4 * 4 + 2];
                val.w = state_transpose[r * 33 + c_f4 * 4 + 3];
                
                block_new_state_f4[r * 32 + (k_blk / 4) + c_f4] = val;
            }
            __syncthreads();
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> gdn_forward(
    torch::Tensor q,           
    torch::Tensor k,           
    torch::Tensor v,           
    torch::Tensor state,       
    torch::Tensor A_log,       
    torch::Tensor a,           
    torch::Tensor dt_bias,     
    torch::Tensor b,           
    torch::Tensor cu_seqlens,  
    float scale                
) {
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(128.0f);
    }
    
    int total_seq_len = q.size(0);
    int num_seqs = cu_seqlens.size(0) - 1;
    
    auto options_bf16 = q.options();
    auto options_fp32 = q.options().dtype(torch::kFloat32);
    
    torch::Tensor output = torch::empty({total_seq_len, 8, 128}, options_bf16);
    torch::Tensor new_state = torch::empty({num_seqs, 8, 128, 128}, options_fp32);
    
    if (num_seqs <= 0) {
        return {output, new_state};
    }
    
    torch::Tensor q_c = q.contiguous();
    torch::Tensor k_c = k.contiguous();
    torch::Tensor v_c = v.contiguous();
    torch::Tensor A_log_c = A_log.contiguous();
    torch::Tensor a_c = a.contiguous();
    torch::Tensor dt_bias_c = dt_bias.contiguous();
    torch::Tensor b_c = b.contiguous();
    torch::Tensor cu_seqlens_c = cu_seqlens.contiguous();
    
    torch::Tensor state_c;
    const float* state_ptr = nullptr;
    if (state.defined() && state.numel() > 0) {
        state_c = state.contiguous();
        state_ptr = state_c.data_ptr<float>();
    }
    
    dim3 grid(8, num_seqs);
    dim3 block(128);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // T=32
    // q: 32 * 128 * 4 = 16384 bytes
    // k: 32 * 128 * 4 = 16384 bytes
    // v_out: 32 * 128 * 4 = 16384 bytes
    // g: 32 * 4 = 128 bytes
    // beta: 32 * 4 = 128 bytes
    // Total = 49408 bytes
    // state_transpose needs 128 * 33 * 4 = 16896 bytes (can reuse part of q/k/v_out buffer)
    
    int smem_size = 49408;
    cudaFuncSetAttribute(gdn_forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    gdn_forward_kernel<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v_c.data_ptr<at::BFloat16>()),
        state_ptr,
        A_log_c.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(a_c.data_ptr<at::BFloat16>()),
        dt_bias_c.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(b_c.data_ptr<at::BFloat16>()),
        cu_seqlens_c.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        new_state.data_ptr<float>(),
        scale
    );
    
    return {output, new_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_forward", &gdn_forward, "GDN Prefill Forward");
}
