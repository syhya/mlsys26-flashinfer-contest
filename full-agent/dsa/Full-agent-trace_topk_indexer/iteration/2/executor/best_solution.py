#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <cstdint>

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
    size_t b = blockIdx.x;
    int tid = threadIdx.x;
    
    int seq_len = seq_lens[b];
    if (seq_len == 0) return;
    
    int num_pages_for_seq = (seq_len + 63) / 64;
    
    // Pad row stride to 132 bytes (33 ints) to completely eliminate 32-way SMEM bank conflicts
    __shared__ alignas(4) uint8_t Q_smem[64 * 132];
    __shared__ alignas(4) uint8_t K_smem[64 * 132];
    __shared__ float weights_smem[64];
    __shared__ float scales_smem[64];
    __shared__ float token_scores[64];
    
    // Load Query for this batch element
    const uint8_t* Q_b = q_index_fp8 + b * 64 * 128;
    for (int i = tid; i < 2048; i += blockDim.x) { // 2048 ints = 64 * 32
        int row = i / 32;
        int col = i % 32;
        ((uint32_t*)Q_smem)[row * 33 + col] = ((const uint32_t*)Q_b)[row * 32 + col];
    }
    
    // Load Weights
    if (tid < 64) {
        weights_smem[tid] = weights[b * 64 + tid];
    }
    __syncthreads();
    
    // Iterate over pages assigned to this sequence
    for (int p = 0; p < num_pages_for_seq; ++p) {
        int page_idx = block_table[b * max_num_pages + p];
        const uint8_t* K_page_base = k_index_cache_fp8 + (size_t)page_idx * 64 * 132;
        
        // Load FP8 Keys
        for (int i = tid; i < 2048; i += blockDim.x) {
            int row = i / 32;
            int col = i % 32;
            ((uint32_t*)K_smem)[row * 33 + col] = ((const uint32_t*)K_page_base)[row * 32 + col];
        }
        
        // Load Scales and initialize scores
        const float* scales_base = (const float*)(K_page_base + 64 * 128);
        if (tid < 64) {
            scales_smem[tid] = scales_base[tid];
            token_scores[tid] = 0.0f;
        }
        __syncthreads();
        
        // 512 threads -> process 64 tokens * 64 heads
        // Each thread processes 1 head for 8 tokens
        int head = tid % 64;
        int token_base = (tid / 64) * 8;
        
        for (int i = 0; i < 8; ++i) {
            int token = token_base + i;
            float sum = 0.0f;
            
            const uint32_t* q_ptr_smem = (const uint32_t*)&Q_smem[head * 132];
            const uint32_t* k_ptr_smem = (const uint32_t*)&K_smem[token * 132];
            
            #pragma unroll
            for (int d = 0; d < 32; ++d) {
                uint32_t q_val = q_ptr_smem[d];
                uint32_t k_val = k_ptr_smem[d];
                
                const __nv_fp8_e4m3* q_fp8_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(&q_val);
                const __nv_fp8_e4m3* k_fp8_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(&k_val);
                
                #pragma unroll
                for(int j = 0; j < 4; ++j) {
                    sum += float(q_fp8_ptr[j]) * float(k_fp8_ptr[j]);
                }
            }
            
            // Apply scale, ReLU, and weight matching deep_gemm mathematics
            float scale = scales_smem[token];
            sum *= scale;
            float relu_val = sum > 0.0f ? sum : 0.0f;
            float weighted_val = relu_val * weights_smem[head];
            
            // Warp reduction for 32 heads
            unsigned int mask = 0xffffffff;
            for (int offset = 16; offset > 0; offset /= 2) {
                weighted_val += __shfl_down_sync(mask, weighted_val, offset);
            }
            
            // Accumulate partial sums (from warp 0 and warp 1)
            if ((tid % 32) == 0) {
                atomicAdd(&token_scores[token], weighted_val);
            }
        }
        __syncthreads();
        
        // Write outputs to global workspace for sorting
        if (tid < 64) {
            int seq_pos = p * 64 + tid;
            if (seq_pos < seq_len) {
                workspace_scores[b * max_seq_len + seq_pos] = token_scores[tid];
                workspace_indices[b * max_seq_len + seq_pos] = page_idx * 64 + tid;
            }
        }
        __syncthreads(); // Ensure writes and reads are synced before next page iteration
    }
}

__global__ void extract_topk_kernel(
    const int32_t* __restrict__ sorted_indices,
    int32_t* __restrict__ topk_indices,
    const int32_t* __restrict__ seq_lens,
    int max_seq_len,
    int topk,
    int batch_size
) {
    size_t b = blockIdx.x;
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
    int batch_size = q_index_fp8.size(0);
    int max_num_pages = block_table.size(1);
    const int topk = 2048;

    auto options_indices = torch::TensorOptions().dtype(torch::kInt32).device(q_index_fp8.device());

    if (batch_size == 0) {
        return torch::empty({0, topk}, options_indices);
    }
    if (max_num_pages == 0) {
        return torch::full({batch_size, topk}, -1, options_indices);
    }

    int max_seq_len = max_num_pages * 64;

    auto options_scores = torch::TensorOptions().dtype(torch::kFloat32).device(q_index_fp8.device());
    
    // Allocate workspaces initialized to negative infinity padding
    auto workspace_scores = torch::full({batch_size, max_seq_len}, -1e9f, options_scores);
    auto workspace_indices = torch::full({batch_size, max_seq_len}, -1, options_indices);

    auto q_contiguous = q_index_fp8.contiguous();
    auto k_contiguous = k_index_cache_fp8.contiguous();
    auto w_contiguous = weights.contiguous();
    auto seq_lens_contiguous = seq_lens.contiguous();
    auto block_table_contiguous = block_table.contiguous();

    const uint8_t* q_ptr = reinterpret_cast<const uint8_t*>(q_contiguous.data_ptr());
    const uint8_t* k_ptr = reinterpret_cast<const uint8_t*>(k_contiguous.data_ptr<int8_t>());
    const float* w_ptr = w_contiguous.data_ptr<float>();
    const int32_t* seq_lens_ptr = seq_lens_contiguous.data_ptr<int32_t>();
    const int32_t* block_table_ptr = block_table_contiguous.data_ptr<int32_t>();

    // 1. Scoring Stage Pipeline
    compute_scores_kernel<<<batch_size, 512>>>(
        q_ptr, k_ptr, w_ptr, seq_lens_ptr, block_table_ptr,
        workspace_scores.data_ptr<float>(), workspace_indices.data_ptr<int32_t>(),
        max_num_pages, max_seq_len
    );

    // 2. CUB Segmented Sort Post-Processing
    auto workspace_scores_out = torch::empty_like(workspace_scores);
    auto workspace_indices_out = torch::empty_like(workspace_indices);

    int32_t total_elements = (batch_size + 1) * max_seq_len;
    auto offsets = torch::arange((int32_t)0, total_elements, (int32_t)max_seq_len, options_indices);

    int total_items = batch_size * max_seq_len;
    size_t temp_storage_bytes = 0;
    
    // Dry run to get temp storage requirements
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        workspace_scores.data_ptr<float>(), workspace_scores_out.data_ptr<float>(),
        workspace_indices.data_ptr<int32_t>(), workspace_indices_out.data_ptr<int32_t>(),
        total_items, batch_size,
        offsets.data_ptr<int32_t>(), offsets.data_ptr<int32_t>() + 1
    );

    auto temp_storage = torch::empty({(int64_t)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(q_index_fp8.device()));

    // Execute segmented descending sort
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data_ptr(), temp_storage_bytes,
        workspace_scores.data_ptr<float>(), workspace_scores_out.data_ptr<float>(),
        workspace_indices.data_ptr<int32_t>(), workspace_indices_out.data_ptr<int32_t>(),
        total_items, batch_size,
        offsets.data_ptr<int32_t>(), offsets.data_ptr<int32_t>() + 1
    );

    // 3. Extract TopK & Format Result
    auto topk_indices = torch::empty({batch_size, topk}, options_indices);

    extract_topk_kernel<<<batch_size, 256>>>(
        workspace_indices_out.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        seq_lens_ptr,
        max_seq_len,
        topk,
        batch_size
    );

    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
}