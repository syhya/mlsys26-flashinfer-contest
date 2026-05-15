#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <tuple>
#include <cmath>

// Kernel to compute the GDN Prefill Forward pass
// Block size: 128 threads. Each block processes one sequence and one value head.
// Grid size: (8, num_seqs)
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
    int seq_idx = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x; // maps to v_idx in the 128-dim head

    long long seq_start = cu_seqlens[seq_idx];
    long long seq_end = cu_seqlens[seq_idx + 1];
    long long seq_len = seq_end - seq_start;

    // Thread-local registers to hold the entire row of the state matrix
    float state_reg[128];

    // Shared memory for coalesced state loading/saving and broadcasting q/k
    union SharedMem {
        float state_transpose[128][33]; // 33 avoids bank conflicts
        struct {
            float q[2][128];
            float k[2][128];
        } qk;
    };
    __shared__ SharedMem smem;

    // 1. Load or Initialize State
    if (state != nullptr) {
        size_t state_base = ((size_t)seq_idx * 8 + h) * 16384; // 16384 = 128 * 128
        const float* block_state = state + state_base;
        const float4* block_state_f4 = reinterpret_cast<const float4*>(block_state);
        
        // Cooperatively load 128x128 state into registers via shared memory using float4
        for (int k_blk = 0; k_blk < 128; k_blk += 32) {
            for (int i = 0; i < 8; ++i) {
                int idx = i * 128 + tid; 
                int r = idx / 8;
                int c_f4 = idx % 8;
                
                float4 val = block_state_f4[r * 32 + (k_blk / 4) + c_f4];
                smem.state_transpose[r][c_f4 * 4 + 0] = val.x;
                smem.state_transpose[r][c_f4 * 4 + 1] = val.y;
                smem.state_transpose[r][c_f4 * 4 + 2] = val.z;
                smem.state_transpose[r][c_f4 * 4 + 3] = val.w;
            }
            __syncthreads();
            
            for (int k_idx = 0; k_idx < 32; k_idx++) {
                state_reg[k_blk + k_idx] = smem.state_transpose[tid][k_idx];
            }
            __syncthreads();
        }
    } else {
        #pragma unroll 8
        for (int k = 0; k < 128; k++) {
            state_reg[k] = 0.0f;
        }
    }

    // Precompute constant head parameters
    float exp_A_log = expf(A_log[h]);
    float dt_bias_val = dt_bias[h];

    // 2. Double-Buffered Sequence Loop
    if (seq_len > 0) {
        int qk_head = h / 2; // 4 Q/K heads map to 8 V heads
        long long t = seq_start;
        int buffer_idx = 0;
        
        // Prologue: Load first token
        size_t qk_idx = ((size_t)t * 4 + qk_head) * 128 + tid;
        smem.qk.q[buffer_idx][tid] = __bfloat162float(q[qk_idx]);
        smem.qk.k[buffer_idx][tid] = __bfloat162float(k[qk_idx]);
        
        size_t v_idx = ((size_t)t * 8 + h) * 128 + tid;
        float v_val = __bfloat162float(v[v_idx]);
        
        size_t ab_idx = (size_t)t * 8 + h;
        float a_val = __bfloat162float(a[ab_idx]);
        float b_val = __bfloat162float(b[ab_idx]);
        
        __syncthreads(); // Wait for prologue to be visible
        
        for (; t < seq_end; t++) {
            int next_buffer_idx = buffer_idx ^ 1;
            float next_v_val, next_a_val, next_b_val;
            
            // Prefetch next token
            if (t + 1 < seq_end) {
                long long next_t = t + 1;
                size_t next_qk_idx = ((size_t)next_t * 4 + qk_head) * 128 + tid;
                smem.qk.q[next_buffer_idx][tid] = __bfloat162float(q[next_qk_idx]);
                smem.qk.k[next_buffer_idx][tid] = __bfloat162float(k[next_qk_idx]);
                
                size_t next_v_idx = ((size_t)next_t * 8 + h) * 128 + tid;
                next_v_val = __bfloat162float(v[next_v_idx]);
                
                size_t next_ab_idx = (size_t)next_t * 8 + h;
                next_a_val = __bfloat162float(a[next_ab_idx]);
                next_b_val = __bfloat162float(b[next_ab_idx]);
            }
            
            // Compute gates for current token
            float x = a_val + dt_bias_val;
            float sp_x = (x > 20.0f) ? x : log1pf(expf(x));
            float g = expf(-exp_A_log * sp_x);
            float beta = 1.0f / (1.0f + expf(-b_val));
            
            // Compute old_v = g * sum_k (k[k] * state[v, k])
            float old_v_unscaled = 0.0f;
            #pragma unroll 16
            for (int k_idx = 0; k_idx < 128; k_idx++) {
                old_v_unscaled = fmaf(smem.qk.k[buffer_idx][k_idx], state_reg[k_idx], old_v_unscaled);
            }
            float old_v = g * old_v_unscaled;
            
            // Compute delta
            float new_v = beta * v_val + (1.0f - beta) * old_v;
            float delta_v = new_v - old_v;
            
            // Update state and compute output element
            float o_val = 0.0f;
            #pragma unroll 16
            for (int k_idx = 0; k_idx < 128; k_idx++) {
                float k_val = smem.qk.k[buffer_idx][k_idx];
                float q_val = smem.qk.q[buffer_idx][k_idx];
                float s = fmaf(k_val, delta_v, g * state_reg[k_idx]);
                state_reg[k_idx] = s;
                o_val = fmaf(q_val, s, o_val);
            }
            
            // Write scaled output
            o_val *= scale;
            size_t curr_v_idx = ((size_t)t * 8 + h) * 128 + tid;
            output[curr_v_idx] = __float2bfloat16(o_val);
            
            // Advance pipeline
            if (t + 1 < seq_end) {
                v_val = next_v_val;
                a_val = next_a_val;
                b_val = next_b_val;
                buffer_idx = next_buffer_idx;
            }
            
            __syncthreads(); // Synchronize before next iteration overwrites buffers
        }
    }

    // 3. Write Back State
    if (new_state != nullptr) {
        size_t state_base = ((size_t)seq_idx * 8 + h) * 16384;
        float* block_new_state = new_state + state_base;
        float4* block_new_state_f4 = reinterpret_cast<float4*>(block_new_state);
        
        // Cooperatively write registers back to global memory using float4
        for (int k_blk = 0; k_blk < 128; k_blk += 32) {
            for (int k_idx = 0; k_idx < 32; k_idx++) {
                smem.state_transpose[tid][k_idx] = state_reg[k_blk + k_idx];
            }
            __syncthreads();
            
            for (int i = 0; i < 8; ++i) {
                int idx = i * 128 + tid;
                int r = idx / 8;
                int c_f4 = idx % 8;
                
                float4 val;
                val.x = smem.state_transpose[r][c_f4 * 4 + 0];
                val.y = smem.state_transpose[r][c_f4 * 4 + 1];
                val.z = smem.state_transpose[r][c_f4 * 4 + 2];
                val.w = smem.state_transpose[r][c_f4 * 4 + 3];
                
                block_new_state_f4[r * 32 + (k_blk / 4) + c_f4] = val;
            }
            __syncthreads();
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> gdn_forward(
    torch::Tensor q,           // [total_seq_len, 4, 128]          bfloat16
    torch::Tensor k,           // [total_seq_len, 4, 128]          bfloat16
    torch::Tensor v,           // [total_seq_len, 8, 128]          bfloat16
    torch::Tensor state,       // [num_seqs, 8, 128, 128]          float32  (k-last: [N,H,V,K]), may be undefined/empty
    torch::Tensor A_log,       // [8]                               float32
    torch::Tensor a,           // [total_seq_len, 8]                bfloat16
    torch::Tensor dt_bias,     // [8]                               float32
    torch::Tensor b,           // [total_seq_len, 8]                bfloat16
    torch::Tensor cu_seqlens,  // [num_seqs + 1]                   int64
    float scale                // scalar float32 value
) {
    // Correctly fallback on scale
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
    
    // Safety check: ensure contiguous memory layout
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
    
    gdn_forward_kernel<<<grid, block, 0, stream>>>(
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
