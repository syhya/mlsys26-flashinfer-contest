#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <algorithm>

// Vectorized FP8 to Float32 Dot Product (4 elements)
inline __device__ float dot_fp8_x4(uint32_t a, uint32_t b) {
    float sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        __nv_fp8_e4m3 a_fp8, b_fp8;
        *((uint8_t*)&a_fp8) = (a >> (i * 8)) & 0xFF;
        *((uint8_t*)&b_fp8) = (b >> (i * 8)) & 0xFF;
        sum += float(a_fp8) * float(b_fp8);
    }
    return sum;
}

// Vectorized FP8 to Float32 Dot Product (16 elements via uint4)
inline __device__ float dot_fp8_uint4(uint4 q, uint4 k) {
    return dot_fp8_x4(q.x, k.x) + dot_fp8_x4(q.y, k.y) + dot_fp8_x4(q.z, k.z) + dot_fp8_x4(q.w, k.w);
}

__global__ void init_buffers_kernel(
    float* __restrict__ scores_buf,
    int32_t* __restrict__ indices_buf,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    int batch_size,
    int max_num_pages)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * max_num_pages * 64;
    if (tid >= total_elements) return;
    
    int batch_idx = tid / (max_num_pages * 64);
    int token_in_batch = tid % (max_num_pages * 64);
    
    int logical_page_idx = token_in_batch / 64;
    int offset = token_in_batch % 64;
    
    // Mathematically strictly -INFINITY so invalid tokens sink to the bottom during sort
    float score = __int_as_float(0xff800000); 
    int32_t global_idx = -1;
    
    if (token_in_batch < seq_lens[batch_idx]) {
        int global_page_idx = block_table[batch_idx * max_num_pages + logical_page_idx];
        global_idx = global_page_idx * 64 + offset;
    }
    
    scores_buf[tid] = score;
    indices_buf[tid] = global_idx;
}

__global__ void compute_scores_kernel(
    const uint8_t* __restrict__ q_index_fp8,
    const uint8_t* __restrict__ k_index_cache_fp8,
    const float* __restrict__ weights,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    float* __restrict__ scores_buf,
    int batch_size,
    int max_num_pages)
{
    int batch_idx = blockIdx.x;
    int page_idx = blockIdx.y;
    
    int seq_len = seq_lens[batch_idx];
    if (page_idx * 64 >= seq_len) return;
    
    int page_id = block_table[batch_idx * max_num_pages + page_idx];
    if (page_id < 0) return; 
    
    // Shared memory layout with exactly 16 bytes of padding per row to eliminate bank conflicts
    __align__(16) __shared__ uint8_t s_q[64][144];
    __align__(16) __shared__ uint8_t s_k[64][144];
    __shared__ float s_k_scale[64];
    __shared__ float s_weights[64];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Vectorized 16-byte Loads for FP8 Query
    const uint4* q_ptr_vec = (const uint4*)(q_index_fp8 + batch_idx * 64 * 128);
    for(int i = tid; i < 512; i += 256) {
        int row = i / 8;
        int col = i % 8;
        ((uint4*)s_q[row])[col] = q_ptr_vec[i];
    }
    
    // Vectorized 16-byte Loads for FP8 Key (deep_gemm packing: fp8 part is contiguous at start of page)
    const uint4* k_ptr_vec = (const uint4*)(k_index_cache_fp8 + page_id * 64 * 132);
    for(int i = tid; i < 512; i += 256) {
        int row = i / 8;
        int col = i % 8;
        ((uint4*)s_k[row])[col] = k_ptr_vec[i];
    }
    
    // Load Token Scales (last 256 bytes of the page) and Weights
    if (tid < 64) {
        const float* k_scale_ptr = (const float*)(k_index_cache_fp8 + page_id * 64 * 132 + 8192);
        s_k_scale[tid] = k_scale_ptr[tid];
        s_weights[tid] = weights[batch_idx * 64 + tid];
    }
    
    __syncthreads();
    
    // 256 threads map to 64 tokens (4 threads per token)
    int token_idx = tid / 4;
    int thread_idx_in_token = tid % 4; // 0..3
    
    float token_score = 0.0f;
    float k_scale = s_k_scale[token_idx];
    
    const uint4* k_t_vec = (const uint4*)s_k[token_idx];
    
    // Each thread processes 16 heads
    for(int h = thread_idx_in_token; h < 64; h += 4) {
        const uint4* q_h_vec = (const uint4*)s_q[h];
        
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            dot += dot_fp8_uint4(q_h_vec[d], k_t_vec[d]);
        }
        
        dot *= k_scale;
        if (dot > 0.0f) { // ReLU Activation
            token_score += dot * s_weights[h];
        }
    }
    
    // Warp-level synchronous butterfly reduction for the 4 threads of the token
    token_score += __shfl_down_sync(0xFFFFFFFF, token_score, 2);
    token_score += __shfl_down_sync(0xFFFFFFFF, token_score, 1);
    
    // Lead thread writes the scalar score to global memory
    if (thread_idx_in_token == 0) {
        int global_token_idx = page_idx * 64 + token_idx;
        if (global_token_idx < seq_len) {
            scores_buf[batch_idx * max_num_pages * 64 + global_token_idx] = token_score;
        }
    }
}

__global__ void set_offsets_kernel(int* offsets, int batch_size, int segment_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid <= batch_size) {
        offsets[tid] = tid * segment_size;
    }
}

__global__ void extract_topk_kernel(
    const int32_t* __restrict__ sorted_indices_buf,
    const int32_t* __restrict__ seq_lens,
    int32_t* __restrict__ out_indices,
    int batch_size,
    int max_num_pages,
    int topk)
{
    int batch_idx = blockIdx.x;
    int seq_len = seq_lens[batch_idx];
    int actual_topk = min(topk, seq_len);
    
    for (int tid = threadIdx.x; tid < topk; tid += blockDim.x) {
        if (tid < actual_topk) {
            out_indices[batch_idx * topk + tid] = sorted_indices_buf[batch_idx * max_num_pages * 64 + tid];
        } else {
            out_indices[batch_idx * topk + tid] = -1; // Strict padding constraint
        }
    }
}

torch::Tensor topk_indexer_forward(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table
) {
    auto q_c = q_index_fp8.contiguous();
    auto k_c = k_index_cache_fp8.contiguous();
    auto w_c = weights.contiguous();
    auto s_c = seq_lens.contiguous();
    auto b_c = block_table.contiguous();

    int batch_size = q_c.size(0);
    int max_num_pages = b_c.size(1);
    int topk = 2048;
    
    auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(q_c.device());
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(q_c.device());
    
    // Allocate intermediate sorting buffers
    auto scores_buf = torch::empty({batch_size, max_num_pages * 64}, options_fp32);
    auto indices_buf = torch::empty({batch_size, max_num_pages * 64}, options_int32);
    auto scores_out = torch::empty({batch_size, max_num_pages * 64}, options_fp32);
    auto indices_out = torch::empty({batch_size, max_num_pages * 64}, options_int32);
    
    auto d_offsets = torch::empty({batch_size + 1}, options_int32);
    
    int total_elements = batch_size * max_num_pages * 64;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    // Phase 1: Initialize values ensuring invalid tokens fall to -INFINITY
    init_buffers_kernel<<<grid_size, block_size>>>(
        scores_buf.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(),
        s_c.data_ptr<int32_t>(),
        b_c.data_ptr<int32_t>(),
        batch_size,
        max_num_pages
    );
    
    // Phase 2: Compute Sparse Attn Scores via Shared Memory Tiled Matmul
    dim3 compute_grid(batch_size, max_num_pages);
    dim3 compute_block(32, 8); // 256 threads
    
    compute_scores_kernel<<<compute_grid, compute_block>>>(
        (const uint8_t*)q_c.data_ptr(),
        (const uint8_t*)k_c.data_ptr(),
        w_c.data_ptr<float>(),
        s_c.data_ptr<int32_t>(),
        b_c.data_ptr<int32_t>(),
        scores_buf.data_ptr<float>(),
        batch_size,
        max_num_pages
    );
    
    // Phase 3: Segmented Batched Radix Sort via CUB for deterministic Top-K selection
    int offset_grid = (batch_size + 1 + 255) / 256;
    set_offsets_kernel<<<offset_grid, 256>>>(
        d_offsets.data_ptr<int>(),
        batch_size,
        max_num_pages * 64
    );
    
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        scores_buf.data_ptr<float>(), scores_out.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(), indices_out.data_ptr<int32_t>(),
        total_elements, batch_size,
        d_offsets.data_ptr<int>(), d_offsets.data_ptr<int>() + 1
    );
    
    auto d_temp_storage = torch::empty({(int64_t)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kInt8).device(q_c.device()));
    
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage.data_ptr(), temp_storage_bytes,
        scores_buf.data_ptr<float>(), scores_out.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(), indices_out.data_ptr<int32_t>(),
        total_elements, batch_size,
        d_offsets.data_ptr<int>(), d_offsets.data_ptr<int>() + 1
    );
    
    // Phase 4: Final Output Extraction with Safe Padding
    auto topk_indices = torch::empty({batch_size, topk}, options_int32);
    
    dim3 extract_grid(batch_size);
    dim3 extract_block(std::min(topk, 1024));
    
    extract_topk_kernel<<<extract_grid, extract_block>>>(
        indices_out.data_ptr<int32_t>(),
        s_c.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        batch_size,
        max_num_pages,
        topk
    );
    
    return topk_indices;
}

// MANDATORY PyBind11 Module Registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
}