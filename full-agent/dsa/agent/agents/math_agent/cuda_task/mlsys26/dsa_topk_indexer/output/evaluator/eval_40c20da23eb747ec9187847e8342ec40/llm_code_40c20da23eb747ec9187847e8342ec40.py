```cpp
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cub/cub.cuh>

using namespace nvcuda;

__global__ void compute_scores_kernel(
    const uint8_t* __restrict__ q_index,
    const uint8_t* __restrict__ k_cache,
    const float* __restrict__ weights,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_table,
    float* __restrict__ global_scores,
    int* __restrict__ global_indices,
    int max_num_pages,
    int max_seq_len
) {
    int page_seq_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    int seq_len = seq_lens[batch_idx];
    
    // If the entire page is beyond the sequence length, fill with padding values and exit
    if (page_seq_idx * 64 >= seq_len) {
        if (tid < 64) {
            int global_token_idx = page_seq_idx * 64 + tid;
            global_scores[batch_idx * max_seq_len + global_token_idx] = -1e20f;
            global_indices[batch_idx * max_seq_len + global_token_idx] = -1;
        }
        return;
    }

    int global_page_idx = block_table[batch_idx * max_num_pages + page_seq_idx];

    // Shared memory layout carefully designed to avoid bank conflicts and fit in standard 48KB
    __shared__ union {
        struct {
            half Q[64][136];
            half K[64][136];
        } in;
        float Out[64][68];
    } smem;
    __shared__ float W_smem[64];
    __shared__ float Scale_smem[64];
    
    // Coalesced load Q: 128 threads load 64x128 FP8 values (2048 uint32_t total -> 16 per thread)
    for (int i = 0; i < 16; ++i) {
        int idx = tid + i * 128; 
        int r = idx / 32;        
        int c = (idx % 32) * 4;  
        
        uint32_t val = *reinterpret_cast<const uint32_t*>(q_index + (batch_idx * 64 * 128 + r * 128 + c));
        
        uint8_t bytes[4];
        *reinterpret_cast<uint32_t*>(bytes) = val;
        half half_vals[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_fp8_e4m3 f8;
            *reinterpret_cast<uint8_t*>(&f8) = bytes[j];
            half_vals[j] = static_cast<half>(static_cast<float>(f8));
        }
        *reinterpret_cast<float2*>(&smem.in.Q[r][c]) = *reinterpret_cast<float2*>(half_vals);
    }

    // Coalesced load K: 128 threads load 64x128 FP8 values
    // K format has 128 bytes FP8 + 4 bytes scale per row.
    for (int i = 0; i < 16; ++i) {
        int idx = tid + i * 128; 
        int r = idx / 32;        
        int c = (idx % 32) * 4;  
        
        uint32_t val = *reinterpret_cast<const uint32_t*>(k_cache + (global_page_idx * 64 * 132 + r * 132 + c));
        
        uint8_t bytes[4];
        *reinterpret_cast<uint32_t*>(bytes) = val;
        half half_vals[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_fp8_e4m3 f8;
            *reinterpret_cast<uint8_t*>(&f8) = bytes[j];
            half_vals[j] = static_cast<half>(static_cast<float>(f8));
        }
        *reinterpret_cast<float2*>(&smem.in.K[r][c]) = *reinterpret_cast<float2*>(half_vals);
    }

    // Load Weights and Scales
    if (tid < 64) {
        W_smem[tid] = weights[batch_idx * 64 + tid];
        Scale_smem[tid] = *reinterpret_cast<const float*>(k_cache + (global_page_idx * 64 * 132 + tid * 132 + 128));
    }

    __syncthreads();

    // Setup 4 Warps to compute 64x64 output block
    int warp_id = tid / 32;
    int w_r = warp_id / 2;
    int w_c = warp_id