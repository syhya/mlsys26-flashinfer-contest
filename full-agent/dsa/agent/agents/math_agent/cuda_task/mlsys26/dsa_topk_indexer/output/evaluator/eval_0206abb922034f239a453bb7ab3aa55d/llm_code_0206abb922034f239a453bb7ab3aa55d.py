#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <tuple>
#include <cstdint>

__global__ void fill_offsets_kernel(int32_t* offsets, int batch_size, int max_seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid <= batch_size) {
        offsets[tid] = tid * max_seq_len;
    }
}

__global__ void compute_scores_kernel(
    const uint8_t* __restrict__ q_index_fp8,       
    const uint8_t* __restrict__ k_index_cache_fp8, 
    const float* __restrict__ weights,             
    const int32_t* __restrict__ seq_lens,          
    const int32_t* __restrict__ block_table,       
    float* __restrict__ workspace_scores,          
    int32_t* __restrict__ workspace_indices,       
    int max_num_pages,
    int max_seq_len
) {
    int b = blockIdx.y;
    int p_idx = blockIdx.x; 
    
    if (p_idx >= max_num_pages) return;
    
    int seq_len = seq_lens[b];
    int tid = threadIdx.x;
    
    // If the entire page is beyond seq_len, just output padding
    if (p_idx * 64 >= seq_len) {
        if (tid < 64) {
            int global_t = p_idx * 64 + tid;
            workspace_scores[b * max_seq_len + global_t] = -1e9f;
            workspace_indices[b * max_seq_len + global_t] = -1;
        }
        return;
    }
    
    int actual_page = block_table[b * max_num_pages + p_idx];
    
    const uint8_t* q_ptr = q_index_fp8 + b * 64 * 128;
    const uint8_t* k_ptr = k_index_cache_fp8 + actual_page * 64 * 132;
    const float* weights_ptr = weights + b * 64;
    
    // Shared memory with padding to avoid bank conflicts
    __shared__ uint8_t shm_q[64][144];
    __shared__ uint8_t shm_k[64][144];
    __shared__ float shm_weights[64];
    
    // Load Q into shared memory
    for(int i = tid; i < 2048; i += 256) {
        int row = i >> 5; 
        int col = i & 31; 
        ((int*)shm_q[row])[col] = ((const int*)q_ptr)[i];
    }
    
    // Load K into shared memory
    for(int i = tid; i < 2112; i += 256) { 
        int row = i / 33;
        int col = i % 33;
        ((int*)shm_k[row])[col] = ((const int*)k_ptr)[i];
    }
    
    // Load Weights into shared memory
    if (tid < 64) {
        shm_weights[tid] = weights_ptr[tid];
    }
    
    __syncthreads();
    
    // 2D register tiling: each thread computes a 4x4 tile
    int tx = tid % 16;
    int ty = tid / 16;
    
    int h_start = tx * 4;
    int t_start = ty * 4;
    
    float acc[4][4] = {0};
    
    // Compute dot products
    for (int d = 0; d < 128; d += 4) {
        uint32_t q_vec[4];
        uint32_t k_vec[4];
        
        #pragma unroll
        for(int i=0; i<4; ++i) {
            q_vec[i] = *(uint32_t*)&shm_q[h_start + i][d];
            k_vec[i] = *(uint32_t*)&shm_k[t_start + i][d];
        }
        
        #pragma unroll
        for(int k_d=0; k_d<4; ++k_d) {
            float q_val[4];
            float k_val[4];
            
            #pragma unroll
            for(int i=0; i<4; ++i) {
                uint8_t q_b = (q_vec[i] >> (k_d * 8)) & 0xFF;
                uint8_t k_b = (k_vec[i] >> (k_d * 8)) & 0xFF;
                q_val[i] = (float)reinterpret_cast<__nv_fp8_e4m3&>(q_b);
                k_val[i] = (float)reinterpret_cast<__nv_fp8_e4m3&>(k_b);
            }
            
            #pragma unroll
            for(int i=0; i<4; ++i) {
                #pragma unroll
                for(int j=0; j<4; ++j) {
                    acc[i][j] += k_val[i] * q_val[j];
                }
            }
        }
    }
    
    // Apply scale, ReLU, and multiply by weight
    float token_head_sum[4] = {0};
    for(int i=0; i<4; ++i) {
        float scale;
        // The scale is precisely at offset 128 in the 132-byte layout
        memcpy(&scale, &shm_k[t_start + i][128], 4);
        
        for(int j=0; j<4; ++j) {
            float val = acc[i][j] * scale;
            if (val > 0.0f) { // ReLU
                token_head_sum[i] += val * shm_weights[h_start + j];
            }
        }
    }
    
    // Warp-level reduction across heads (tx from 0 to 15 covers the 64 heads)
    for (int i=0; i<4; ++i) {
        float sum = token_head_sum[i];
        sum += __shfl_down_sync(0xffffffff, sum, 8, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 4, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 2, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 1, 16);
        
        if (tx == 0) {
            int global_t = p_idx * 64 + (t_start + i);
            if (global_t < seq_len) {
                workspace_scores[b * max_seq_len + global_t] = sum;
                // Index is the absolute token index across the actual allocated pages
                workspace_indices[b * max_seq_len + global_t] = actual_page * 64 + (t_start + i);
            } else {
                workspace_scores[b * max_seq_len + global_t] = -1e9f;
                workspace_indices[b * max_seq_len + global_t] = -1;
            }
        }
    }
}

__global__ void extract_topk_kernel(
    const int32_t* __restrict__ sorted_indices,
    int32_t* __restrict__ topk_indices,
    const int32_t* __restrict__ seq_lens,
    int max_seq_len,
    int topk
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int seq_len = seq_lens[b];
    
    int actual_topk = (seq_len < topk) ? seq_len : topk;
    
    for (int i = tid; i < topk; i += blockDim.x) {
        if (i < actual_topk) {
            topk_indices[b * topk + i] = sorted_indices[b * max_seq_len + i];
        } else {
            topk_indices[b * topk + i] = -1;
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
    int max_seq_len = max_num_pages * 64;
    int topk = 2048;

    auto device = q_c.device();
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);

    torch::Tensor workspace_scores = torch::empty({batch_size * max_seq_len}, options_float);
    torch::Tensor workspace_indices = torch::empty({batch_size * max_seq_len}, options_int32);
    
    torch::Tensor workspace_scores_sorted = torch::empty({batch_size * max_seq_len}, options_float);
    torch::Tensor workspace_indices_sorted = torch::empty({batch_size * max_seq_len}, options_int32);
    
    torch::Tensor offsets = torch::empty({batch_size + 1}, options_int32);
    
    dim3 block_offsets(256);
    dim3 grid_offsets((batch_size + 256) / 256);
    if (batch_size > 0) {
        fill_offsets_kernel<<<grid_offsets, block_offsets>>>(offsets.data_ptr<int32_t>(), batch_size, max_seq_len);
    }
    
    if (batch_size > 0 && max_num_pages > 0) {
        dim3 block_scores(256);
        dim3 grid_scores(max_num_pages, batch_size);
        
        compute_scores_kernel<<<grid_scores, block_scores>>>(
            reinterpret_cast<const uint8_t*>(q_c.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_c.data_ptr()),
            w_c.data_ptr<float>(),
            s_c.data_ptr<int32_t>(),
            b_c.data_ptr<int32_t>(),
            workspace_scores.data_ptr<float>(),
            workspace_indices.data_ptr<int32_t>(),
            max_num_pages,
            max_seq_len
        );
        
        size_t temp_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            nullptr, temp_storage_bytes,
            workspace_scores.data_ptr<float>(), workspace_scores_sorted.data_ptr<float>(),
            workspace_indices.data_ptr<int32_t>(), workspace_indices_sorted.data_ptr<int32_t>(),
            batch_size * max_seq_len, batch_size,
            offsets.data_ptr<int32_t>(), offsets.data_ptr<int32_t>() + 1
        );
        
        torch::Tensor temp_storage = torch::empty({(long long)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
        
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp_storage.data_ptr(), temp_storage_bytes,
            workspace_scores.data_ptr<float>(), workspace_scores_sorted.data_ptr<float>(),
            workspace_indices.data_ptr<int32_t>(), workspace_indices_sorted.data_ptr<int32_t>(),
            batch_size * max_seq_len, batch_size,
            offsets.data_ptr<int32_t>(), offsets.data_ptr<int32_t>() + 1
        );
    }
    
    torch::Tensor topk_indices = torch::empty({batch_size, topk}, options_int32);
    
    if (batch_size > 0) {
        dim3 block_extract(256);
        dim3 grid_extract(batch_size);
        extract_topk_kernel<<<grid_extract, block_extract>>>(
            workspace_indices_sorted.data_ptr<int32_t>(),
            topk_indices.data_ptr<int32_t>(),
            s_c.data_ptr<int32_t>(),
            max_seq_len,
            topk
        );
    }
    
    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
    m.def("dsa_forward", &topk_indexer_forward, "DSA TopK Indexer Forward (alias to prevent benchmark errors)");
}