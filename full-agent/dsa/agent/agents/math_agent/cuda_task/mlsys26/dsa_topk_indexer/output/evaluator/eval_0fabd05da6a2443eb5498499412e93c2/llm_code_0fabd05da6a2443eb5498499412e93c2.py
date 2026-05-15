#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

__global__ void compute_scores_kernel(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const float* __restrict__ weights,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    float* __restrict__ scores_buf,
    int32_t* __restrict__ indices_buf,
    int max_num_pages,
    int max_seq_len_padded
) {
    int b = blockIdx.x;
    int p = blockIdx.y;

    int seq_len = seq_lens[b];
    if (seq_len > max_seq_len_padded) {
        seq_len = max_seq_len_padded;
    }

    int num_pages_for_seq = (seq_len + 63) / 64;
    
    // Early exit at the block level for unused pages
    if (p >= num_pages_for_seq) {
        return; 
    }

    int t = threadIdx.x;       // token offset within the page (0..63)
    int h_group = threadIdx.y; // head group index (0..3)
    
    // Mask for tokens that are within sequence length
    bool is_valid = (p * 64 + t < seq_len);

    int global_page_idx = block_table[b * max_num_pages + p];
    
    // Base pointer for the page. Layout: 64*128 bytes FP8, then 64*4 bytes float32 scales
    const uint8_t* k_page = k_cache + global_page_idx * 64 * 132;
    const uint8_t* k_token_fp8 = k_page + t * 128;
    float scale = *(const float*)(k_page + 64 * 128 + t * 4);

    const uint8_t* q_b = q + b * 64 * 128;
    const float* w_b = weights + b * 64;

    float partial_score_sum = 0.0f;

    // Load queries and weights into shared memory for fast broadcast
    __shared__ alignas(16) uint8_t sq[64 * 128];
    __shared__ float sw[64];
    
    int tid = h_group * 64 + t; 
    int q_offset = tid * 32;
    const float4* q_b_f4 = (const float4*)(q_b + q_offset);
    float4* sq_f4 = (float4*)(sq + q_offset);
    
    // Vectorized 16-byte aligned load
    sq_f4[0] = q_b_f4[0];
    sq_f4[1] = q_b_f4[1];

    if (tid < 64) {
        sw[tid] = w_b[tid];
    }
    
    __syncthreads();

    int h_start = h_group * 16;
    int h_end = h_start + 16;
    
    // Compute dot product for the assigned 16 heads
    for (int h = h_start; h < h_end; ++h) {
        const uint8_t* q_h = sq + h * 128;
        
        float dot = 0.0f;
        const uint4* q_vec = (const uint4*)q_h;
        const uint4* k_vec = (const uint4*)k_token_fp8;
        
        #pragma unroll
        for (int i = 0; i < 8; ++i) { 
            uint4 qv = q_vec[i];
            uint4 kv = k_vec[i];
            
            uint32_t* q_w = (uint32_t*)&qv;
            uint32_t* k_w = (uint32_t*)&kv;
            
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                uint32_t qw = q_w[w];
                uint32_t kw = k_w[w];
                
                #pragma unroll
                for (int b_idx = 0; b_idx < 4; ++b_idx) {
                    uint8_t q0 = (qw >> (b_idx * 8)) & 0xFF;
                    uint8_t k0 = (kw >> (b_idx * 8)) & 0xFF;
                    
                    __nv_fp8_e4m3 fq0, fk0;
                    *(uint8_t*)&fq0 = q0;
                    *(uint8_t*)&fk0 = k0;
                    
                    // Accumulate in FP32 to match exact reference mathematical semantics
                    dot += float(fq0) * float(fk0);
                }
            }
        }
        
        dot *= scale;
        float relu_dot = dot > 0.0f ? dot : 0.0f;
        partial_score_sum += relu_dot * sw[h];
    }

    // Reduce over head groups
    // 64x5 layout prevents 4-way bank conflicts
    __shared__ float smem_reduce[64][5];
    smem_reduce[t][h_group] = partial_score_sum;
    __syncthreads();

    // Write final results
    if (h_group == 0 && is_valid) {
        float final_score = smem_reduce[t][0] + smem_reduce[t][1] + smem_reduce[t][2] + smem_reduce[t][3];
        int global_out_idx = b * max_seq_len_padded + p * 64 + t;
        scores_buf[global_out_idx] = final_score;
        indices_buf[global_out_idx] = global_page_idx * 64 + t;
    }
}

__global__ void fill_offsets_var_len(
    int32_t* begin_offsets, 
    int32_t* end_offsets, 
    const int32_t* seq_lens, 
    int batch_size, 
    int segment_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int seq_len = seq_lens[idx];
        if (seq_len > segment_length) {
            seq_len = segment_length;
        }
        // Define exact segments for CUB to avoid sorting padding elements
        begin_offsets[idx] = idx * segment_length;
        end_offsets[idx] = idx * segment_length + seq_len;
    }
}

__global__ void extract_topk_kernel(
    const int32_t* __restrict__ indices_sorted,
    int32_t* __restrict__ topk_indices,
    const int32_t* __restrict__ seq_lens,
    int batch_size,
    int topk,
    int segment_length
) {
    int b = blockIdx.x;
    int seq_len = seq_lens[b];
    int actual_topk = min(topk, seq_len);

    for (int tid = threadIdx.x; tid < topk; tid += blockDim.x) {
        if (tid < actual_topk) {
            topk_indices[b * topk + tid] = indices_sorted[b * segment_length + tid];
        } else {
            // Strict padding enforcement
            topk_indices[b * topk + tid] = -1;
        }
    }
}

torch::Tensor topk_indexer_forward(
    torch::Tensor q_index_fp8,        // [batch_size, 64, 128]  float8_e4m3fn
    torch::Tensor k_index_cache_fp8,  // [num_pages, 64, 1, 132] int8 interpreted as uint8
    torch::Tensor weights,            // [batch_size, 64]        float32
    torch::Tensor seq_lens,           // [batch_size]            int32
    torch::Tensor block_table         // [batch_size, max_num_pages] int32
) {
    // Ensure contiguous memory mapping to prevent memory faults
    auto q_index_fp8_c = q_index_fp8.contiguous();
    auto k_index_cache_fp8_c = k_index_cache_fp8.contiguous();
    auto weights_c = weights.contiguous();
    auto seq_lens_c = seq_lens.contiguous();
    auto block_table_c = block_table.contiguous();

    int batch_size = q_index_fp8_c.size(0);
    int max_num_pages = block_table_c.size(1);
    int topk = 2048; 
    
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(q_index_fp8.device());
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(q_index_fp8.device());

    if (batch_size == 0) {
        return torch::empty({0, topk}, options_int);
    }
    if (max_num_pages == 0) {
        return torch::full({batch_size, topk}, -1, options_int);
    }

    int segment_length = max_num_pages * 64;
    int total_elements = batch_size * segment_length;

    torch::Tensor scores_buf = torch::empty({total_elements}, options_float);
    torch::Tensor indices_buf = torch::empty({total_elements}, options_int);

    // Compute sparse attention scores natively in registers
    dim3 grid(batch_size, max_num_pages);
    dim3 block(64, 4);

    compute_scores_kernel<<<grid, block>>>(
        (const uint8_t*)q_index_fp8_c.data_ptr(),
        (const uint8_t*)k_index_cache_fp8_c.data_ptr(),
        weights_c.data_ptr<float>(),
        seq_lens_c.data_ptr<int32_t>(),
        block_table_c.data_ptr<int32_t>(),
        scores_buf.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(),
        max_num_pages,
        segment_length
    );

    // Segment mappings to instruct CUB precisely which subsets to sort
    torch::Tensor begin_offsets = torch::empty({batch_size}, options_int);
    torch::Tensor end_offsets = torch::empty({batch_size}, options_int);

    int threads_offset = 256;
    int blocks_offset = (batch_size + threads_offset - 1) / threads_offset;
    fill_offsets_var_len<<<blocks_offset, threads_offset>>>(
        begin_offsets.data_ptr<int32_t>(),
        end_offsets.data_ptr<int32_t>(),
        seq_lens_c.data_ptr<int32_t>(),
        batch_size,
        segment_length
    );

    torch::Tensor scores_sorted = torch::empty({total_elements}, options_float);
    torch::Tensor indices_sorted = torch::empty({total_elements}, options_int);

    // Obtain required workspace size for CUB
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        scores_buf.data_ptr<float>(), scores_sorted.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(), indices_sorted.data_ptr<int32_t>(),
        total_elements, batch_size,
        begin_offsets.data_ptr<int32_t>(), end_offsets.data_ptr<int32_t>()
    );

    torch::Tensor temp_storage = torch::empty({(int64_t)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(q_index_fp8.device()));

    // Execute highly optimized batched segmented radix sort
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data_ptr(), temp_storage_bytes,
        scores_buf.data_ptr<float>(), scores_sorted.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(), indices_sorted.data_ptr<int32_t>(),
        total_elements, batch_size,
        begin_offsets.data_ptr<int32_t>(), end_offsets.data_ptr<int32_t>()
    );

    torch::Tensor topk_indices = torch::empty({batch_size, topk}, options_int);

    // Extract exact top-K matches ensuring strictly mapped output shapes
    extract_topk_kernel<<<batch_size, 256>>>(
        indices_sorted.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        seq_lens_c.data_ptr<int32_t>(),
        batch_size,
        topk,
        segment_length
    );

    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
}