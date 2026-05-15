#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <tuple>
#include <cmath>

// Functor for providing segment offsets to CUB's segmented radix sort
struct OffsetOp {
    int segment_size;
    
    __host__ __device__ __forceinline__
    OffsetOp(int size) : segment_size(size) {}
    
    __host__ __device__ __forceinline__
    int operator()(int i) const {
        return i * segment_size;
    }
};

// Helper to convert 16 FP8 elements (1 uint4) into 16 floats safely using hardware intrinsics
__device__ __forceinline__ void convert_16fp8_to_float4_store(uint4 vec, float* out) {
    // Treat the 16 bytes directly as __nv_fp8_e4m3 array
    const __nv_fp8_e4m3* fp8_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(&vec);
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float4 f4;
        f4.x = (float)fp8_ptr[i * 4 + 0];
        f4.y = (float)fp8_ptr[i * 4 + 1];
        f4.z = (float)fp8_ptr[i * 4 + 2];
        f4.w = (float)fp8_ptr[i * 4 + 3];
        *(float4*)(out + i * 4) = f4;
    }
}

// Compute attention scores for each token and prepare them for sorting
__global__ void scoring_kernel(
    const uint8_t* __restrict__ q_index_fp8,
    const uint8_t* __restrict__ k_index_cache_fp8,
    const float* __restrict__ weights,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_table,
    float* __restrict__ scores_out,
    int* __restrict__ indices_out,
    int max_num_pages,
    int batch_size
) {
    int page_logical_idx = blockIdx.x;
    int b = blockIdx.y;

    if (b >= batch_size) return;

    int seq_len = seq_lens[b];
    // Number of active pages for this sequence element
    int num_pages_for_seq = (seq_len + 63) / 64;

    // Fast path: if the assigned logical page exceeds sequence bounds, write -INF and return
    if (page_logical_idx >= num_pages_for_seq) {
        for (int i = threadIdx.x; i < 64; i += blockDim.x) {
            int out_idx = b * (max_num_pages * 64) + page_logical_idx * 64 + i;
            scores_out[out_idx] = -1e20f;
            indices_out[out_idx] = -1;
        }
        return;
    }

    // Dynamic shared memory allocations
    extern __shared__ float smem[];
    float* smem_Q = smem;                                  // 8192 floats = 32768 bytes
    float* smem_K = smem + 8192;                           // 8192 floats = 32768 bytes
    float* smem_weights = smem + 16384;                    // 64 floats   = 256 bytes
    float* smem_K_scale = smem + 16448;                    // 64 floats   = 256 bytes

    int tid = threadIdx.x;

    // 1. Cooperative Load Q (64 heads x 128 dim = 8192 FP8 elements)
    const uint8_t* q_ptr = q_index_fp8 + b * 8192;
    uint4 q_vec = ((const uint4*)q_ptr)[tid];
    convert_16fp8_to_float4_store(q_vec, smem_Q + tid * 16);

    // 2. Cooperative Load K page (64 tokens x 128 dim = 8192 FP8 elements)
    int page_idx = block_table[b * max_num_pages + page_logical_idx];
    const uint8_t* k_ptr = k_index_cache_fp8 + page_idx * 8448; // 64 * 132 bytes per page
    uint4 k_vec = ((const uint4*)k_ptr)[tid];
    convert_16fp8_to_float4_store(k_vec, smem_K + tid * 16);

    // 3. Load scale factors and weights for current batch
    if (tid < 64) {
        // Scales reside immediately after the 8192 bytes of FP8 data
        smem_K_scale[tid] = *(const float*)(k_ptr + 8192 + tid * 4);
        smem_weights[tid] = weights[b * 64 + tid];
    }

    __syncthreads();

    // 16 Warps in the block, 64 tokens total. Each warp covers 4 consecutive tokens.
    int wid = tid / 32;
    int lane = tid % 32;
    int t_start = wid * 4;

    float score[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float scale[4] = {
        smem_K_scale[t_start], 
        smem_K_scale[t_start + 1], 
        smem_K_scale[t_start + 2], 
        smem_K_scale[t_start + 3]
    };

    // Calculate sum_heads(relu(Q @ K.T * scale) * weight)
    for (int h = 0; h < 64; ++h) {
        // Each thread processes 4 elements along the embedding dim (128)
        int d_start = lane * 4;
        float4 q_f4 = *(float4*)(smem_Q + h * 128 + d_start);
        
        float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        // Inner dot-product accumulation
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 k_f4 = *(float4*)(smem_K + (t_start + i) * 128 + d_start);
            dot[i] += q_f4.x * k_f4.x;
            dot[i] += q_f4.y * k_f4.y;
            dot[i] += q_f4.z * k_f4.z;
            dot[i] += q_f4.w * k_f4.w;
        }
        
        // Parallel warp reduction across 32 threads for the 4 tokens simultaneously
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                dot[i] += __shfl_down_sync(0xffffffff, dot[i], offset);
            }
        }
        
        // Apply scaling, ReLU, and weighted sum on lane 0
        if (lane == 0) {
            float w = smem_weights[h];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float val = dot[i] * scale[i];
                if (val > 0.0f) { // ReLU
                    score[i] += val * w;
                }
            }
        }
    }

    // Write-back scores and absolute token indices
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int t = t_start + i;
            int global_token_idx = page_logical_idx * 64 + t;
            int out_idx = b * (max_num_pages * 64) + global_token_idx;
            
            // Strictly enforce padding masking for sequence bounds mathematically
            if (global_token_idx < seq_len) {
                scores_out[out_idx] = score[i];
                indices_out[out_idx] = page_idx * 64 + t;
            } else {
                scores_out[out_idx] = -1e20f;
                indices_out[out_idx] = -1;
            }
        }
    }
}

// Extract top-2048 from strictly sorted arrays
__global__ void extract_topk_kernel(
    const int* __restrict__ indices_sorted,
    int* __restrict__ topk_indices,
    int batch_size,
    int max_tokens,
    int topk
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * topk;
    
    if (idx < total_elements) {
        int b = idx / topk;
        int k = idx % topk;
        topk_indices[idx] = indices_sorted[b * max_tokens + k];
    }
}

torch::Tensor topk_indexer_forward(
    torch::Tensor q_index_fp8,        // [batch_size, 64, 128]  float8_e4m3fn
    torch::Tensor k_index_cache_fp8,  // [num_pages, 64, 1, 132] int8 (uint8 interpreted)
    torch::Tensor weights,            // [batch_size, 64]        float32
    torch::Tensor seq_lens,           // [batch_size]            int32
    torch::Tensor block_table         // [batch_size, max_num_pages] int32
) {
    int batch_size = q_index_fp8.size(0);
    int max_num_pages = block_table.size(1);
    int topk = 2048; // Architectural limit requirement for selection

    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(q_index_fp8.device());
    
    // Safety handle for empty tasks
    if (batch_size == 0 || max_num_pages == 0) {
        return torch::full({batch_size, topk}, -1, options_i32);
    }

    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q_index_fp8.device());

    int num_items = batch_size * max_num_pages * 64;
    
    // Intermediate allocations for scoring & sorting
    torch::Tensor scores_out = torch::empty({num_items}, options_f32);
    torch::Tensor indices_out = torch::empty({num_items}, options_i32);
    torch::Tensor scores_sorted = torch::empty({num_items}, options_f32);
    torch::Tensor indices_sorted = torch::empty({num_items}, options_i32);

    // Scoring stage Kernel launch definitions
    dim3 grid(max_num_pages, batch_size);
    dim3 block(512); // 16 warps
    int smem_size = 66048; // Required Dynamic Size (64.5 KB)
    
    // Elevate dynamic shared memory cap for Hopper architectures
    cudaFuncSetAttribute(scoring_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    scoring_kernel<<<grid, block, smem_size>>>(
        (const uint8_t*)q_index_fp8.data_ptr(),
        (const uint8_t*)k_index_cache_fp8.data_ptr(),
        weights.data_ptr<float>(),
        seq_lens.data_ptr<int>(),
        block_table.data_ptr<int>(),
        scores_out.data_ptr<float>(),
        indices_out.data_ptr<int>(),
        max_num_pages,
        batch_size
    );

    // Setup segmented array pointers using CUB Transform Iterators to avoid explicit memory allocation
    auto d_begin_offsets = cub::MakeTransformIterator(cub::CountingInputIterator<int>(0), OffsetOp(max_num_pages * 64));
    auto d_end_offsets = cub::MakeTransformIterator(cub::CountingInputIterator<int>(1), OffsetOp(max_num_pages * 64));

    size_t temp_storage_bytes = 0;
    
    // Dry-run to identify workspace requirement
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        scores_out.data_ptr<float>(), scores_sorted.data_ptr<float>(),
        indices_out.data_ptr<int>(), indices_sorted.data_ptr<int>(),
        num_items, batch_size,
        d_begin_offsets, d_end_offsets
    );

    // Leverage PyTorch's caching allocator for fast zero-overhead workspace provision
    torch::Tensor temp_storage = torch::empty({(long)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(q_index_fp8.device()));

    // Exact Mathematical Sorted Top-K Enforcer
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data_ptr(), temp_storage_bytes,
        scores_out.data_ptr<float>(), scores_sorted.data_ptr<float>(),
        indices_out.data_ptr<int>(), indices_sorted.data_ptr<int>(),
        num_items, batch_size,
        d_begin_offsets, d_end_offsets
    );

    // Extract the K tokens
    torch::Tensor topk_indices = torch::empty({batch_size, topk}, options_i32);
    int total_topk_elements = batch_size * topk;
    int threads = 256;
    int blocks = (total_topk_elements + threads - 1) / threads;

    extract_topk_kernel<<<blocks, threads>>>(
        indices_sorted.data_ptr<int>(),
        topk_indices.data_ptr<int>(),
        batch_size,
        max_num_pages * 64,
        topk
    );

    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
}