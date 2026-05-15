#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>

__device__ __forceinline__ float fp8_to_float(uint8_t val) {
    return static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3*>(&val));
}

__global__ void fill_offsets(int* offsets, int B, int N) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b <= B) {
        offsets[b] = b * N;
    }
}

__global__ void scoring_kernel(
    const uint8_t* __restrict__ q_index_fp8,
    const uint8_t* __restrict__ k_index_cache_fp8,
    const float* __restrict__ weights,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_table,
    float* __restrict__ scores_out,
    int* __restrict__ indices_out,
    int B, int max_num_pages) 
{
    int b = blockIdx.y;
    int p = blockIdx.x;
    int seq_len = seq_lens[b];

    int tid = threadIdx.x;
    int t_in_page = tid / 4;
    int lane_in_token = tid % 4;
    
    int global_t_start = p * 64;

    // Early exit if the entire page is out of sequence length bounds
    if (global_t_start >= seq_len) {
        if (lane_in_token == 0 && t_in_page < 64) {
            int out_idx = b * max_num_pages * 64 + global_t_start + t_in_page;
            scores_out[out_idx] = -INFINITY;
            indices_out[out_idx] = -1;
        }
        return;
    }

    __shared__ float4 smem_q[512]; // 8192 bytes
    __shared__ uint32_t smem_k[64][33]; // Padded to avoid bank conflicts
    __shared__ float smem_w[64];
    __shared__ float smem_scale[64];

    // Load queries for the batch element
    const float4* q_ptr_g = (const float4*)(q_index_fp8 + b * 64 * 128);
    smem_q[tid] = q_ptr_g[tid];
    smem_q[tid + 256] = q_ptr_g[tid + 256];

    // Load keys for the specific page
    int64_t phys_p = block_table[b * max_num_pages + p];
    const uint32_t* k_ptr_g_32 = (const uint32_t*)(k_index_cache_fp8 + phys_p * 64 * 132);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 256;
        int row = idx >> 5;
        int col = idx & 31;
        smem_k[row][col] = k_ptr_g_32[idx];
    }

    // Load scales
    if (tid < 16) {
        ((float4*)smem_scale)[tid] = ((const float4*)(k_index_cache_fp8 + phys_p * 64 * 132 + 8192))[tid];
    }

    // Load weights
    if (tid < 16) {
        ((float4*)smem_w)[tid] = ((const float4*)(weights + b * 64))[tid];
    }

    __syncthreads();

    float token_score = 0;
    float k_scale = smem_scale[t_in_page];
    int global_t = global_t_start + t_in_page;
    bool valid_token = (global_t < seq_len);

    for (int h = 0; h < 64; h++) {
        float dot = 0;
        uint32_t* q_ptr = (uint32_t*)(&smem_q[h * 8]);
        uint32_t* k_ptr = (uint32_t*)smem_k[t_in_page];
        
        #pragma unroll
        for (int i = lane_in_token; i < 32; i += 4) {
            uint32_t q_val_32 = q_ptr[i];
            uint32_t k_val_32 = k_ptr[i];
            
            uint8_t q0 = q_val_32 & 0xff;
            uint8_t q1 = (q_val_32 >> 8) & 0xff;
            uint8_t q2 = (q_val_32 >> 16) & 0xff;
            uint8_t q3 = (q_val_32 >> 24) & 0xff;
            
            uint8_t k0 = k_val_32 & 0xff;
            uint8_t k1 = (k_val_32 >> 8) & 0xff;
            uint8_t k2 = (k_val_32 >> 16) & 0xff;
            uint8_t k3 = (k_val_32 >> 24) & 0xff;
            
            dot += fp8_to_float(q0) * fp8_to_float(k0) +
                   fp8_to_float(q1) * fp8_to_float(k1) +
                   fp8_to_float(q2) * fp8_to_float(k2) +
                   fp8_to_float(q3) * fp8_to_float(k3);
        }
        
        // Warp-level reduction across the 4 threads assigned to this token
        dot += __shfl_down_sync(0xffffffff, dot, 2);
        dot += __shfl_down_sync(0xffffffff, dot, 1);
        
        if (lane_in_token == 0) {
            dot *= k_scale;
            if (dot > 0.0f) { // ReLU Activation
                token_score += dot * smem_w[h];
            }
        }
    }

    if (lane_in_token == 0) {
        int out_idx = b * max_num_pages * 64 + global_t;
        if (valid_token) {
            scores_out[out_idx] = token_score;
            int global_page_idx = phys_p;
            indices_out[out_idx] = global_page_idx * 64 + t_in_page;
        } else {
            scores_out[out_idx] = -INFINITY;
            indices_out[out_idx] = -1;
        }
    }
}

__global__ void extract_topk(const int* sorted_indices, int* topk_indices, int B, int N, int K) {
    int b = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B && k < K) {
        topk_indices[b * K + k] = sorted_indices[b * N + k];
    }
}

torch::Tensor topk_indexer_forward(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table
) {
    int B = q_index_fp8.size(0);
    int max_num_pages = block_table.size(1);
    int N = max_num_pages * 64;
    int K_top = 2048;

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(q_index_fp8.device());
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(q_index_fp8.device());

    if (B == 0 || max_num_pages == 0) {
        return torch::empty({B, K_top}, options_int);
    }

    torch::Tensor scores_out = torch::empty({B, N}, options_float);
    torch::Tensor indices_out = torch::empty({B, N}, options_int);
    torch::Tensor offsets = torch::empty({B + 1}, options_int);

    // 1. Fill offsets for CUB Segmented Sort
    int threads = 256;
    int blocks = (B + threads) / threads;
    fill_offsets<<<blocks, threads>>>(offsets.data_ptr<int>(), B, N);

    // 2. Compute Attention Scores Map-Reduce
    dim3 grid(max_num_pages, B);
    scoring_kernel<<<grid, 256>>>(
        static_cast<const uint8_t*>(q_index_fp8.data_ptr()),
        static_cast<const uint8_t*>(k_index_cache_fp8.data_ptr()),
        weights.data_ptr<float>(),
        seq_lens.data_ptr<int>(),
        block_table.data_ptr<int>(),
        scores_out.data_ptr<float>(),
        indices_out.data_ptr<int>(),
        B, max_num_pages
    );

    // 3. Exact Segmented Radix Sort
    torch::Tensor sorted_scores = torch::empty({B, N}, options_float);
    torch::Tensor sorted_indices = torch::empty({B, N}, options_int);

    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        scores_out.data_ptr<float>(), sorted_scores.data_ptr<float>(),
        indices_out.data_ptr<int>(), sorted_indices.data_ptr<int>(),
        B * N, B,
        offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1
    );

    torch::Tensor temp_storage = torch::empty({(int64_t)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(q_index_fp8.device()));

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data_ptr<uint8_t>(), temp_storage_bytes,
        scores_out.data_ptr<float>(), sorted_scores.data_ptr<float>(),
        indices_out.data_ptr<int>(), sorted_indices.data_ptr<int>(),
        B * N, B,
        offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1
    );

    // 4. Extract global Top-K tokens
    torch::Tensor topk_indices = torch::empty({B, K_top}, options_int);
    dim3 grid_ext((K_top + 255) / 256, B);
    extract_topk<<<grid_ext, 256>>>(
        sorted_indices.data_ptr<int>(),
        topk_indices.data_ptr<int>(),
        B, N, K_top
    );

    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
    m.def("dsa_forward", &topk_indexer_forward, "DSA TopK Indexer Forward alias");
    m.def("run", &topk_indexer_forward, "DSA TopK Indexer Forward alias");
}