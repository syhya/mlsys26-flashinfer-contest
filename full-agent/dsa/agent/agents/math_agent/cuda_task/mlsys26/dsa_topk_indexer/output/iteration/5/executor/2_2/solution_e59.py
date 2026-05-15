#include <torch/extension.h>
#include <cuda_fp8.h>
#include <tuple>
#include <cub/cub.cuh>

// Helper to efficiently convert 4 packed FP8 (e4m3) bytes into 4 FP32 floats
inline __device__ float4 cvt_fp8x4_to_float4(uint32_t val) {
    float4 res;
    uint8_t bytes[4];
    *reinterpret_cast<uint32_t*>(bytes) = val;
    __nv_fp8_e4m3 v[4];
    v[0] = *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[0]);
    v[1] = *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[1]);
    v[2] = *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[2]);
    v[3] = *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[3]);
    res.x = float(v[0]);
    res.y = float(v[1]);
    res.z = float(v[2]);
    res.w = float(v[3]);
    return res;
}

// Fused kernel: Dequantizes, computes dot products, applies ReLU, applies weights, and reduces across heads
__global__ void compute_scores_kernel(
    const uint8_t* __restrict__ q_index_fp8,        
    const uint8_t* __restrict__ k_index_cache_fp8,  
    const float* __restrict__ weights,              
    const int32_t* __restrict__ seq_lens,           
    const int32_t* __restrict__ block_table,        
    float* __restrict__ scores_buf,                 
    int32_t* __restrict__ indices_buf,              
    int max_num_pages
) {
    int p = blockIdx.x; // page index within sequence
    int b = blockIdx.y; // batch index
    
    int seq_len = seq_lens[b];
    int N = max_num_pages * 64;
    
    // If the entire page is beyond the sequence length, fill with padding values and exit
    if (p * 64 >= seq_len) {
        for (int i = threadIdx.x; i < 64; i += blockDim.x) {
            scores_buf[b * N + p * 64 + i] = -1e20f;
            indices_buf[b * N + p * 64 + i] = -1;
        }
        return;
    }

    __shared__ uint32_t q_smem_u32[64 * 32]; // 64 heads * 32 uint32_t (128 bytes) per head
    __shared__ float weights_smem[64];

    // Load queries into shared memory for fast broadcast using uint4
    const uint4* q_global_u4 = (const uint4*)(q_index_fp8 + b * 64 * 128);
    uint4* q_smem_u4 = (uint4*)q_smem_u32;
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        q_smem_u4[i] = q_global_u4[i];
    }

    // Load weights into shared memory
    if (threadIdx.x < 64) {
        weights_smem[threadIdx.x] = weights[b * 64 + threadIdx.x];
    }

    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    int global_page_idx = block_table[b * max_num_pages + p];
    // Deep_gemm KV cache format: [page_size, 128] FP8 data followed by [page_size, 1] FP32 scale
    const uint8_t* k_page_base = k_index_cache_fp8 + global_page_idx * (64 * 132);

    // Each warp processes 8 tokens in the page (256 threads = 8 warps => 64 tokens)
    for (int iter = 0; iter < 8; ++iter) {
        int t = iter * 8 + warp_id;
        if (t >= 64) continue;

        bool valid = (p * 64 + t < seq_len);
        float total_score = 0.0f;

        if (valid) {
            // Load 4 FP8 key elements for the current token
            uint32_t k_val = *(const uint32_t*)(k_page_base + t * 128 + lane_id * 4);
            float4 k_f = cvt_fp8x4_to_float4(k_val);
            // Load scalar scale for the current token (located after the 8192 bytes of FP8 data)
            float scale = *(const float*)(k_page_base + 8192 + t * 4);
            
            // Apply scale explicitly to K
            k_f.x *= scale; k_f.y *= scale; k_f.z *= scale; k_f.w *= scale;

            // Compute dot product and accumulate score across all 64 heads
            for (int h = 0; h < 64; ++h) {
                uint32_t q_val = q_smem_u32[h * 32 + lane_id];
                float4 q_f = cvt_fp8x4_to_float4(q_val);
                
                // Element-wise multiply-add
                float val = q_f.x * k_f.x + q_f.y * k_f.y + q_f.z * k_f.z + q_f.w * k_f.w;
                
                // Warp reduction
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }
                
                // Head reduction with ReLU and learned weights
                if (lane_id == 0) {
                    if (val > 0.0f) {
                        total_score += val * weights_smem[h];
                    }
                }
            }
            if (lane_id == 0 && isnan(total_score)) {
                total_score = -1e20f; // Robustness against upstream NaNs
            }
        }

        // Write scores and global token indices to global memory buffers for batched sorting
        if (lane_id == 0) {
            int out_idx = b * N + p * 64 + t;
            if (valid) {
                scores_buf[out_idx] = total_score;
                indices_buf[out_idx] = global_page_idx * 64 + t;
            } else {
                scores_buf[out_idx] = -1e20f;
                indices_buf[out_idx] = -1;
            }
        }
    }
}

// Generate deterministic segment boundaries for SegmentedRadixSort
__global__ void generate_offsets(int32_t* offsets, int batch_size, int segment_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= batch_size) {
        offsets[idx] = idx * segment_size;
    }
}

// Safely extract strictly the required Top-K elements (applying padding explicitly where requested)
__global__ void extract_topk_kernel(
    const int32_t* __restrict__ indices_sorted,
    int32_t* __restrict__ topk_indices,
    const int32_t* __restrict__ seq_lens,
    int batch_size,
    int segment_size,
    int topk
) {
    int b = blockIdx.x;
    int seq_len = seq_lens[b];
    int actual_topk = min(topk, seq_len);

    for (int tid = threadIdx.x; tid < topk; tid += blockDim.x) {
        if (tid < actual_topk) {
            topk_indices[b * topk + tid] = indices_sorted[b * segment_size + tid];
        } else {
            topk_indices[b * topk + tid] = -1;
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
    int topk = 2048;

    auto q = q_index_fp8.contiguous();
    auto k = k_index_cache_fp8.contiguous();
    auto w = weights.contiguous();
    auto sl = seq_lens.contiguous();
    auto bt = block_table.contiguous();

    auto device = q.device();

    // Max capacity elements per batch sequence
    int N = max_num_pages * 64; 
    auto scores_buf = torch::empty({batch_size, N}, torch::dtype(torch::kFloat32).device(device));
    auto indices_buf = torch::empty({batch_size, N}, torch::dtype(torch::kInt32).device(device));

    // Phase 1: High Bandwidth Warp-Tiled Score Computation 
    dim3 grid(max_num_pages, batch_size);
    dim3 block(256);
    
    compute_scores_kernel<<<grid, block>>>(
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k.data_ptr()),
        w.data_ptr<float>(),
        sl.data_ptr<int32_t>(),
        bt.data_ptr<int32_t>(),
        scores_buf.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(),
        max_num_pages
    );

    auto scores_sorted = torch::empty_like(scores_buf);
    auto indices_sorted = torch::empty_like(indices_buf);

    auto offsets = torch::empty({batch_size + 1}, torch::dtype(torch::kInt32).device(device));
    int num_blocks = (batch_size + 1 + 255) / 256;
    generate_offsets<<<num_blocks, 256>>>(offsets.data_ptr<int32_t>(), batch_size, N);

    // Phase 2: Deterministic CUB Segmented Descending Sorting
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        scores_buf.data_ptr<float>(), scores_sorted.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(), indices_sorted.data_ptr<int32_t>(),
        batch_size * N, batch_size,
        offsets.data_ptr<int32_t>(), offsets.data_ptr<int32_t>() + 1
    );

    auto temp_storage = torch::empty({(long)temp_storage_bytes}, torch::dtype(torch::kUInt8).device(device));

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data_ptr(), temp_storage_bytes,
        scores_buf.data_ptr<float>(), scores_sorted.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(), indices_sorted.data_ptr<int32_t>(),
        batch_size * N, batch_size,
        offsets.data_ptr<int32_t>(), offsets.data_ptr<int32_t>() + 1
    );

    // Phase 3: Extraction and Padding
    auto topk_indices = torch::empty({batch_size, topk}, torch::dtype(torch::kInt32).device(device));

    extract_topk_kernel<<<batch_size, 1024>>>(
        indices_sorted.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        sl.data_ptr<int32_t>(),
        batch_size,
        N,
        topk
    );

    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
    m.def("dsa_forward", &topk_indexer_forward, "DSA TopK Indexer Forward Alias");
}
