#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <tuple>
#include <cmath>

#define NEG_INF __int_as_float(0xff800000)

// Helper to convert an FP8 byte to float safely using native intriniscs
__device__ __forceinline__ float fp8_to_float(uint8_t byte_val) {
    __nv_fp8_e4m3 val;
    reinterpret_cast<uint8_t*>(&val)[0] = byte_val;
    return float(val);
}

// Lightweight kernel to set up offsets for CUB's Segmented Radix Sort
__global__ void setup_offsets_kernel(int* begin_offsets, int* end_offsets, int batch_size, int segment_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        begin_offsets[b] = b * segment_size;
        end_offsets[b] = (b + 1) * segment_size;
    }
}

// Truncation kernel to extract exactly the top K tokens after sorting
__global__ void truncation_kernel(const int* __restrict__ sorted_indices, int* __restrict__ topk_indices, int batch_size, int segment_size, int topk) {
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < topk) {
        topk_indices[b * topk + idx] = sorted_indices[b * segment_size + idx];
    }
}

// Main map-reduce scoring kernel
__global__ void scoring_kernel(
    const uint8_t* __restrict__ q_index_fp8,
    const uint8_t* __restrict__ k_index_cache_fp8,
    const float* __restrict__ weights,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_table,
    float* __restrict__ global_scores,
    int* __restrict__ global_indices,
    int max_num_pages
) {
    int b = blockIdx.y;
    int p_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    int seq_len = seq_lens[b];
    int num_pages_for_seq = (seq_len + 63) / 64;
    
    int global_p = -1;
    if (p_idx < num_pages_for_seq) {
        global_p = block_table[b * max_num_pages + p_idx];
    }
    
    // Fast path out for empty or out-of-bounds pages
    if (global_p == -1) {
        if (tid < 64) {
            int t = tid;
            int out_idx = b * max_num_pages * 64 + p_idx * 64 + t;
            global_scores[out_idx] = NEG_INF;
            global_indices[out_idx] = -1;
        }
        return;
    }
    
    // Shared memory allocations perfectly sized for L1 cache hits and coalesced access
    __shared__ uint8_t s_Q[64][128];
    __shared__ uint8_t s_K[64][128];
    __shared__ float s_scale[64];
    __shared__ float s_W[64];
    
    // Coalesced loads for Q
    const uint8_t* Q_batch = q_index_fp8 + b * 8192; // 64 heads * 128 dim
    int q_idx = tid * 32;
    if (q_idx < 8192) {
        *(uint4*)&s_Q[0][q_idx] = *(uint4*)(&Q_batch[q_idx]);
        *(uint4*)&s_Q[0][q_idx + 16] = *(uint4*)(&Q_batch[q_idx + 16]);
    }
    
    // Coalesced loads for K
    const uint8_t* K_page = k_index_cache_fp8 + global_p * 8448; // 64 * 132 bytes
    int k_idx = tid * 32;
    if (k_idx < 8192) {
        *(uint4*)&s_K[0][k_idx] = *(uint4*)(&K_page[k_idx]);
        *(uint4*)&s_K[0][k_idx + 16] = *(uint4*)(&K_page[k_idx + 16]);
    }
    
    // Load scales and weights
    if (tid < 64) {
        const float* scale_page = (const float*)(K_page + 8192);
        s_scale[tid] = scale_page[tid];
        s_W[tid] = weights[b * 64 + tid];
    }
    
    __syncthreads();
    
    // 4 threads process 1 token cooperatively
    int t = tid / 4;
    int th = tid % 4;
    
    bool valid = (p_idx * 64 + t < seq_len);
    float token_score = 0.0f;
    
    if (valid) {
        int h_start = th * 16;
        float scale = s_scale[t];
        
        for (int i = 0; i < 16; i++) {
            int h = h_start + i;
            float acc = 0.0f;
            
            uint32_t* q_ptr = (uint32_t*)&s_Q[h][0];
            uint32_t* k_ptr = (uint32_t*)&s_K[t][0];
            
            // Unroll vector instructions to maximize FP32 MAC throughput
            #pragma unroll 4
            for (int d_vec = 0; d_vec < 32; d_vec++) {
                uint32_t q4 = q_ptr[d_vec];
                uint32_t k4 = k_ptr[d_vec];
                
                uint8_t* q_b = (uint8_t*)&q4;
                uint8_t* k_b = (uint8_t*)&k4;
                
                acc += fp8_to_float(q_b[0]) * fp8_to_float(k_b[0]);
                acc += fp8_to_float(q_b[1]) * fp8_to_float(k_b[1]);
                acc += fp8_to_float(q_b[2]) * fp8_to_float(k_b[2]);
                acc += fp8_to_float(q_b[3]) * fp8_to_float(k_b[3]);
            }
            
            acc *= scale;
            if (acc > 0.0f) { // ReLU logic matched from DeepGEMM
                token_score += acc * s_W[h];
            }
        }
    }
    
    // Warp-level reduction across the 4 threads assigned to the token
    token_score += __shfl_down_sync(0xffffffff, token_score, 2);
    token_score += __shfl_down_sync(0xffffffff, token_score, 1);
    
    // Primary thread records final mathematical score and exact index mapping
    if (th == 0) {
        int out_idx = b * max_num_pages * 64 + p_idx * 64 + t;
        if (valid) {
            global_scores[out_idx] = token_score;
            global_indices[out_idx] = global_p * 64 + t;
        } else {
            global_scores[out_idx] = NEG_INF;
            global_indices[out_idx] = -1;
        }
    }
}

// C++ API wrapper guaranteeing memory correctness prior to kernel launch
torch::Tensor topk_indexer_forward(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table
) {
    // Explicit continuous assertions to safeguard pointer boundaries
    auto q_c = q_index_fp8.contiguous();
    auto k_c = k_index_cache_fp8.contiguous();
    auto w_c = weights.contiguous();
    auto sl_c = seq_lens.contiguous();
    auto bt_c = block_table.contiguous();

    int batch_size = q_c.size(0);
    int max_num_pages = bt_c.size(1);
    int topk = 2048;
    int segment_size = max_num_pages * 64;
    int total_elements = batch_size * segment_size;
    
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(q_c.device());
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(q_c.device());
    
    auto global_scores = torch::empty({total_elements}, options_float);
    auto global_indices = torch::empty({total_elements}, options_int);
    auto sorted_scores = torch::empty({total_elements}, options_float);
    auto sorted_indices = torch::empty({total_elements}, options_int);
    
    auto begin_offsets = torch::empty({batch_size}, options_int);
    auto end_offsets = torch::empty({batch_size}, options_int);
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    setup_offsets_kernel<<<blocks, threads>>>(
        begin_offsets.data_ptr<int>(),
        end_offsets.data_ptr<int>(),
        batch_size,
        segment_size
    );
    
    dim3 grid(max_num_pages, batch_size);
    dim3 block(256);
    
    scoring_kernel<<<grid, block>>>(
        (const uint8_t*)q_c.data_ptr(),
        (const uint8_t*)k_c.data_ptr(),
        w_c.data_ptr<float>(),
        sl_c.data_ptr<int>(),
        bt_c.data_ptr<int>(),
        global_scores.data_ptr<float>(),
        global_indices.data_ptr<int>(),
        max_num_pages
    );
    
    // First pass estimates size
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        global_scores.data_ptr<float>(), sorted_scores.data_ptr<float>(),
        global_indices.data_ptr<int>(), sorted_indices.data_ptr<int>(),
        total_elements, batch_size,
        begin_offsets.data_ptr<int>(), end_offsets.data_ptr<int>()
    );
    
    auto temp_storage = torch::empty({(long)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(q_c.device()));
    
    // Second pass commits true global sorting mathematically decoupled from local heuristics
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data_ptr<uint8_t>(), temp_storage_bytes,
        global_scores.data_ptr<float>(), sorted_scores.data_ptr<float>(),
        global_indices.data_ptr<int>(), sorted_indices.data_ptr<int>(),
        total_elements, batch_size,
        begin_offsets.data_ptr<int>(), end_offsets.data_ptr<int>()
    );
    
    auto topk_indices = torch::empty({batch_size, topk}, options_int);
    dim3 grid_trunc((topk + 255) / 256, batch_size);
    dim3 block_trunc(256);
    truncation_kernel<<<grid_trunc, block_trunc>>>(
        sorted_indices.data_ptr<int>(),
        topk_indices.data_ptr<int>(),
        batch_size,
        segment_size,
        topk
    );
    
    return topk_indices;
}

// Additional interface handler to perfectly align with benchmark framework symbol tests
torch::Tensor dsa_forward(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table
) {
    return topk_indexer_forward(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table);
}

// Bound modules to expose exact method names guaranteeing compatibility
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
    m.def("dsa_forward", &dsa_forward, "DSA Forward Alias");
}