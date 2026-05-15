#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <tuple>
#include <cmath>

// Helper to multiply two FP8 values by directly casting the bits
__device__ __forceinline__ float mul_fp8(uint8_t a, uint8_t b) {
    return (float)reinterpret_cast<const __nv_fp8_e4m3&>(a) * (float)reinterpret_cast<const __nv_fp8_e4m3&>(b);
}

// Simple kernel to initialize segment offsets for CUB segmented sort
__global__ void init_offsets(int* offsets, int batch_size, int max_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= batch_size) {
        offsets[idx] = idx * max_tokens;
    }
}

// Main scoring kernel: computes dot products, applies scaling, ReLU, and weighted sum
__global__ void scoring_kernel(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ k_cache,
    const float* __restrict__ weights,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_table,
    float* __restrict__ scores_out,
    int* __restrict__ indices_out,
    int batch_size,
    int max_num_pages,
    int page_size,
    int head_dim_with_scale
) {
    int b = blockIdx.x;
    int page_idx = blockIdx.y; 
    
    if (b >= batch_size || page_idx >= max_num_pages) return;
    
    // Shared memory padded to 132 bytes to completely eliminate bank conflicts
    __shared__ uint8_t sq[64][132];
    __shared__ uint8_t sk[64][132];
    __shared__ float sw[64];
    __shared__ float sscale[64];
    
    int tid = threadIdx.x; // 0..255
    
    // Load Q using 128-bit vectorized loads
    const int4* q_int4 = reinterpret_cast<const int4*>(q_fp8 + b * 64 * 128);
    for (int i = tid; i < 512; i += 256) {
        int row = i / 8;
        int col = i % 8;
        reinterpret_cast<int4*>(sq[row])[col] = q_int4[i];
    }
    
    // Load Weights
    if (tid < 64) {
        sw[tid] = weights[b * 64 + tid];
    }
    
    int seq_len = seq_lens[b];
    int num_pages_for_seq = (seq_len + page_size - 1) / page_size;
    int global_page_idx = -1;
    if (page_idx < num_pages_for_seq) {
        global_page_idx = block_table[b * max_num_pages + page_idx];
    }
    
    // Load K Cache and Scales
    if (global_page_idx != -1) {
        const uint8_t* k_page_ptr = k_cache + global_page_idx * page_size * head_dim_with_scale;
        const int4* k_int4 = reinterpret_cast<const int4*>(k_page_ptr);
        for (int i = tid; i < 512; i += 256) {
            int row = i / 8;
            int col = i % 8;
            reinterpret_cast<int4*>(sk[row])[col] = k_int4[i];
        }
        if (tid < 64) {
            int scale_offset = 64 * 128 + tid * 4;
            float scale;
            memcpy(&scale, k_page_ptr + scale_offset, 4);
            sscale[tid] = scale;
        }
    }
    
    __syncthreads();
    
    // 4 threads cooperate per token
    int token = tid / 4;
    int thread_in_token = tid % 4;
    
    float thread_score = 0.0f;
    
    if (global_page_idx != -1 && (page_idx * 64 + token < seq_len)) {
        float scale = sscale[token];
        
        // Each thread computes 16 heads, interleaved to ensure 0 bank conflicts with 132-byte padding
        for (int i = 0; i < 16; ++i) {
            int h = thread_in_token + i * 4;
            const uint32_t* q_h = reinterpret_cast<const uint32_t*>(sq[h]);
            const uint32_t* k_t = reinterpret_cast<const uint32_t*>(sk[token]);
            
            float dp = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < 32; ++j) {
                uint32_t q_val = q_h[j];
                uint32_t k_val = k_t[j];
                
                dp += mul_fp8(q_val & 0xFF, k_val & 0xFF);
                dp += mul_fp8((q_val >> 8) & 0xFF, (k_val >> 8) & 0xFF);
                dp += mul_fp8((q_val >> 16) & 0xFF, (k_val >> 16) & 0xFF);
                dp += mul_fp8((q_val >> 24) & 0xFF, (k_val >> 24) & 0xFF);
            }
            
            dp *= scale;
            if (dp > 0.0f) { // ReLU activation
                thread_score += dp * sw[h];
            }
        }
    }
    
    // Warp-level reduction across the 4 threads in a token
    #pragma unroll
    for (int offset = 2; offset > 0; offset /= 2) {
        thread_score += __shfl_down_sync(0xFFFFFFFF, thread_score, offset);
    }
    
    // Thread 0 of each token writes out the result
    if (thread_in_token == 0) {
        int out_idx = b * (max_num_pages * 64) + page_idx * 64 + token;
        if (global_page_idx != -1 && (page_idx * 64 + token < seq_len)) {
            scores_out[out_idx] = thread_score;
            indices_out[out_idx] = global_page_idx * 64 + token;
        } else {
            // Unused/padded slots receive -INFINITY to naturally sort to the bottom
            scores_out[out_idx] = -INFINITY;
            indices_out[out_idx] = -1;
        }
    }
}

// Truncates the globally sorted token indices to the top-K constraint
__global__ void gather_topk(
    const int* __restrict__ indices_sorted,
    int* __restrict__ topk_indices,
    int topk,
    int max_tokens
) {
    int b = blockIdx.x;
    int k = blockIdx.y * blockDim.x + threadIdx.x;
    if (k < topk) {
        topk_indices[b * topk + k] = indices_sorted[b * max_tokens + k];
    }
}

// Main C++ Entry Point
torch::Tensor topk_indexer_forward(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table
) {
    int batch_size = q_index_fp8.size(0);
    int max_num_pages = block_table.size(1);
    int topk = 2048;
    
    auto options = torch::TensorOptions().device(q_index_fp8.device());
    torch::Tensor topk_indices = torch::full({batch_size, topk}, -1, options.dtype(torch::kInt32));
    
    if (batch_size == 0 || max_num_pages == 0) {
        return topk_indices;
    }
    
    int max_tokens = max_num_pages * 64;
    int total_tokens = batch_size * max_tokens;
    
    // Temporary workspaces for sorting
    torch::Tensor scores_in = torch::empty({total_tokens}, options.dtype(torch::kFloat32));
    torch::Tensor scores_out = torch::empty({total_tokens}, options.dtype(torch::kFloat32));
    torch::Tensor indices_in = torch::empty({total_tokens}, options.dtype(torch::kInt32));
    torch::Tensor indices_out = torch::empty({total_tokens}, options.dtype(torch::kInt32));
    torch::Tensor offsets = torch::empty({batch_size + 1}, options.dtype(torch::kInt32));
    
    // 1. Initialize segment boundaries for CUB
    int threads = 256;
    int blocks = (batch_size + 1 + threads - 1) / threads;
    init_offsets<<<blocks, threads>>>(offsets.data_ptr<int>(), batch_size, max_tokens);
    
    // 2. Score FP8 query against paged KV cache natively
    dim3 score_grid(batch_size, max_num_pages);
    dim3 score_block(256);
    scoring_kernel<<<score_grid, score_block>>>(
        reinterpret_cast<const uint8_t*>(q_index_fp8.data_ptr()),
        reinterpret_cast<const uint8_t*>(k_index_cache_fp8.data_ptr()),
        weights.data_ptr<float>(),
        seq_lens.data_ptr<int>(),
        block_table.data_ptr<int>(),
        scores_in.data_ptr<float>(),
        indices_in.data_ptr<int>(),
        batch_size,
        max_num_pages,
        64,
        132
    );
    
    // 3. Exact batched Segmented Radix Sort leveraging standard CUB primitives (Safe Execution)
    int* offsets_ptr = offsets.data_ptr<int>();
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        scores_in.data_ptr<float>(), scores_out.data_ptr<float>(),
        indices_in.data_ptr<int>(), indices_out.data_ptr<int>(),
        total_tokens, batch_size,
        offsets_ptr, offsets_ptr + 1
    );
    
    torch::Tensor temp_storage = torch::empty({(int64_t)temp_storage_bytes}, options.dtype(torch::kUInt8));
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data_ptr(), temp_storage_bytes,
        scores_in.data_ptr<float>(), scores_out.data_ptr<float>(),
        indices_in.data_ptr<int>(), indices_out.data_ptr<int>(),
        total_tokens, batch_size,
        offsets_ptr, offsets_ptr + 1
    );
    
    // 4. Truncate correctly sorted index mapping
    dim3 gather_grid(batch_size, (topk + 255) / 256);
    dim3 gather_block(256);
    gather_topk<<<gather_grid, gather_block>>>(
        indices_out.data_ptr<int>(),
        topk_indices.data_ptr<int>(),
        topk,
        max_tokens
    );
    
    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
}