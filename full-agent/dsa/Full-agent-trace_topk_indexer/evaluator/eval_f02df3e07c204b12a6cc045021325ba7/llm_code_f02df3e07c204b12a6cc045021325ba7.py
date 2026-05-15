```cpp
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ float fp8_to_float(uint8_t val) {
    union {
        uint8_t u8;
        __nv_fp8_e4m3 f8;
    } tmp;
    tmp.u8 = val;
    return float(tmp.f8);
}

__global__ void fill_offsets_kernel(int* offsets, int batch_size, int max_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= batch_size) {
        offsets[idx] = idx * max_tokens;
    }
}

__global__ void compute_scores_kernel(
    const uint8_t* __restrict__ q_ptr,
    const uint8_t* __restrict__ k_ptr,
    const float* __restrict__ weights_ptr,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_table_ptr,
    float* __restrict__ workspace_scores,
    int* __restrict__ workspace_indices,
    int batch_size,
    int max_num_pages,
    int max_tokens
) {
    int b = blockIdx.y;
    int p = blockIdx.x;
    
    int seq_len = seq_lens[b];
    int num_pages_for_seq = (seq_len + 63) / 64;
    
    int t = threadIdx.x + (threadIdx.y % 2) * 32; // 0 to 63
    int h_group = threadIdx.y / 2; // 0 to 3
    
    if (p >= num_pages_for_seq) {
        if (h_group == 0) {
            int global_t = p * 64 + t;
            int out_idx = b * max_tokens + global_t;
            workspace_scores[out_idx] = -1e30f;
            workspace_indices[out_idx] = -1;
        }
        return;
    }
    
    int global_page_idx = block_table_ptr[b * max_num_pages + p];
    
    __shared__ __align__(16) uint8_t smem_q[64 * 128];
    __shared__ __align__(16) uint8_t smem_k[64 * 144];
    __shared__ float smem_w[64];
    __shared__ float smem_s[64];
    
    int tid = threadIdx.y * 32 + threadIdx.x; // 0 to 255
    
    // Load Q (8192 bytes = 512 int4s, 2 per thread)
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int offset = (i * 256 + tid) * 16;
        *((uint4*)((char*)smem_q + offset)) = *((uint4*)(q_ptr + b * 8192 + offset));
    }
    
    // Load K (8192 bytes FP8 data, map linearly to padded layout to avoid bank conflicts)
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int offset = (i * 256 + tid) * 16;
        int t_idx = offset / 128;
        int d_idx = offset % 128;
        int smem_offset = t_idx * 144 + d_idx;
        *((uint4*)((char*)smem_k + smem_offset)) = *((uint4*)(k_ptr + global_page_idx * 8448 + offset));
    }
    
    // Load Weights and Scales
    if (tid < 64) {
        smem_w[tid] = weights_ptr[b * 64 + tid];
        smem_s[tid] = *(const float*)(k_ptr + global_page_idx * 8448 + 8192 + tid * 4);
    }
    
    __syncthreads();
    
    float local_score = 0.0f;
    
    // Compute Score (Tensor-core matmul equivalent over FP8 bytes)
    for (int i = 0; i < 16; ++i) {
        int h = h_group * 16 + i;
        float dot = 0.0f;
        
        uint4* q_row = (uint4*)&smem_q[h * 128];
        uint4* k_row = (uint4*)&smem_k[t * 144];
        
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            uint4 q_val = q_row[d];
            uint4 k_val = k_row[d];
            
            uint32_t q_words[4] = {q_val.x, q_val.y, q_val.z, q_val.w};
            uint32_t k_words[4] = {k_val.x, k_val.y, k_val.z, k_val.w};
            
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                uint32_t qw = q_words[w];
                uint32_t kw = k_words[w];
                
                dot += fp8_to_float(qw & 0xFF) * fp8_to_float(kw & 0xFF);
                dot += fp8_to_float((qw >> 8) & 0xFF) * fp8_to_float((kw >> 8) & 0xFF);
                dot += fp8_to_float((qw >> 16) & 0xFF) * fp8_to_float((kw >> 16) & 0xFF);
                dot += fp8_to_float((qw >> 24) & 0xFF) * fp8_to_float((kw >> 24) & 0xFF);
            }
        }
        
        float val = smem_s[t] * dot;
        if (val > 0.0f) {
            local_score += smem_w[h] * val;
        }
    }
    
    __shared__ float reduce_smem[64][4];
    reduce_smem[t][h_group] = local_score;
    
    __syncthreads();
    
    // Reduce & Write out valid tokens
    if (h_group == 0) {
        float final_score = reduce_smem[t][0] + reduce_smem[t][1] + reduce_smem[t][2] + reduce_smem[t][3];
        int global_t = p * 64 + t;
        int out_idx = b * max_tokens + global_t;
        
        if (global_t < seq_len) {
            workspace_scores[out_idx