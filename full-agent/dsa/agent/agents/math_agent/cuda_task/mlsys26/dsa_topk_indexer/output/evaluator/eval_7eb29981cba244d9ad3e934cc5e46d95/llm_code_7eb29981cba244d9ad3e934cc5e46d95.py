#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>

inline __device__ float4 cvt_fp8x4_to_float4(uint32_t fp8x4) {
    union {
        uint32_t u32;
        __nv_fp8_e4m3 fp8[4];
    } u;
    u.u32 = fp8x4;
    return make_float4((float)u.fp8[0], (float)u.fp8[1], (float)u.fp8[2], (float)u.fp8[3]);
}

__global__ void init_offsets_kernel(int* offsets, int batch_size, int max_tokens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= batch_size) {
        offsets[i] = i * max_tokens;
    }
}

__global__ void copy_topk_kernel(const int* sorted_indices, int* topk_indices, int batch_size, int max_tokens, int topk) {
    int b = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size && k < topk) {
        topk_indices[b * topk + k] = sorted_indices[b * max_tokens + k];
    }
}

__global__ void compute_scores_kernel(
    const uint8_t* __restrict__ q_index_fp8,
    const uint8_t* __restrict__ k_index_cache_fp8,
    const float* __restrict__ weights,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_table,
    float* __restrict__ global_scores,
    int* __restrict__ global_indices,
    int batch_size,
    int max_num_pages)
{
    int b = blockIdx.y;
    int p = blockIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0 to 255
    int hx = threadIdx.x; // 0 to 63
    int ty = threadIdx.y; // 0 to 3

    int seq_len = seq_lens[b];
    if (p * 64 >= seq_len) {
        if (tid < 64) {
            int global_t = p * 64 + tid;
            global_scores[b * max_num_pages * 64 + global_t] = -INFINITY;
            global_indices[b * max_num_pages * 64 + global_t] = -1;
        }
        return;
    }

    __shared__ uint8_t shared_Q[64][132];
    __shared__ uint8_t shared_K[64][132];
    __shared__ float shared_K_scale[64];
    __shared__ float shared_weights[64];
    __shared__ float shared_partial_sum[4][2];

    // Load Q (Avoiding bank conflicts by padding to 132 bytes)
    const uint32_t* q_global = (const uint32_t*)(q_index_fp8 + b * 8192);
    for (int i = tid; i < 2048; i += 256) {
        int row = i / 32;
        int col = i % 32;
        ((uint32_t*)shared_Q[row])[col] = q_global[i];
    }

    // Load Weights
    if (tid < 64) {
        shared_weights[tid] = weights[b * 64 + tid];
    }

    // Load K (Broadcasting access later, padding to 132 bytes for consistent loop access)
    int page_idx = block_table[b * max_num_pages + p];
    const uint8_t* k_global_page = k_index_cache_fp8 + page_idx * 8448;
    const uint32_t* k_fp8_global = (const uint32_t*)k_global_page;
    for (int i = tid; i < 2048; i += 256) {
        int row = i / 32;
        int col = i % 32;
        ((uint32_t*)shared_K[row])[col] = k_fp8_global[i];
    }

    const float* k_scale_global = (const float*)(k_global_page + 8192);
    if (tid < 64) {
        shared_K_scale[tid] = k_scale_global[tid];
    }

    __syncthreads();

    // Compute Dot Product
    for (int t = ty; t < 64; t += 4) {
        float sum = 0.0f;
        uint32_t* q_ptr = (uint32_t*) &shared_Q[hx][0];
        uint32_t* k_ptr = (uint32_t*) &shared_K[t][0];
        
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            uint32_t q4 = q_ptr[i];
            uint32_t k4 = k_ptr[i];
            float4 qf = cvt_fp8x4_to_float4(q4);
            float4 kf = cvt_fp8x4_to_float4(k4);
            sum += qf.x * kf.x + qf.y * kf.y + qf.z * kf.z + qf.w * kf.w;
        }

        sum *= shared_K_scale[t];
        sum = fmaxf(sum, 0.0f) * shared_weights[hx];

        // Warp reduction
        float val = sum;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (hx % 32 == 0) {
            shared_partial_sum[ty][hx / 32] = val;
        }
        __syncthreads();

        // Write Final Value
        if (hx == 0) {
            float final_s = shared_partial_sum[ty][0] + shared_partial_sum[ty][1];
            int global_t = p * 64 + t;
            if (global_t < seq_len) {
                global_scores[b * max_num_pages * 64 + global_t] = final_s;
                global_indices[b * max_num_pages * 64 + global_t] = page_idx * 64 + t;
            } else {
                global_scores[b * max_num_pages * 64 + global_t] = -INFINITY;
                global_indices[b * max_num_pages * 64 + global_t] = -1;
            }
        }
        __syncthreads();
    }
}

torch::Tensor topk_indexer_forward(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table
) {
    torch::Tensor q_c = q_index_fp8.contiguous();
    torch::Tensor k_c = k_index_cache_fp8.contiguous();
    torch::Tensor w_c = weights.contiguous();
    torch::Tensor s_c = seq_lens.contiguous();
    torch::Tensor b_c = block_table.contiguous();

    int batch_size = q_c.size(0);
    int max_num_pages = b_c.size(1);
    int max_tokens = max_num_pages * 64;

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(q_c.device());
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(q_c.device());

    if (batch_size == 0) return torch::empty({0, 2048}, options_int);

    torch::Tensor global_scores = torch::empty({batch_size, max_tokens}, options_float);
    torch::Tensor global_indices = torch::empty({batch_size, max_tokens}, options_int);

    dim3 grid(max_num_pages, batch_size);
    dim3 block(64, 4);

    const uint8_t* q_ptr = static_cast<const uint8_t*>(q_c.data_ptr());
    const uint8_t* k_ptr = static_cast<const uint8_t*>(k_c.data_ptr());
    const float* w_ptr = static_cast<const float*>(w_c.data_ptr());
    const int* seq_lens_ptr = static_cast<const int*>(s_c.data_ptr());
    const int* block_table_ptr = static_cast<const int*>(b_c.data_ptr());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    compute_scores_kernel<<<grid, block, 0, stream>>>(
        q_ptr, k_ptr, w_ptr, seq_lens_ptr, block_table_ptr,
        static_cast<float*>(global_scores.data_ptr()), static_cast<int*>(global_indices.data_ptr()),
        batch_size, max_num_pages
    );

    torch::Tensor d_offsets = torch::empty({batch_size + 1}, options_int);
    init_offsets_kernel<<<(batch_size + 255) / 256, 256, 0, stream>>>(static_cast<int*>(d_offsets.data_ptr()), batch_size, max_tokens);

    torch::Tensor sorted_scores = torch::empty_like(global_scores);
    torch::Tensor sorted_indices = torch::empty_like(global_indices);

    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        static_cast<float*>(global_scores.data_ptr()), static_cast<float*>(sorted_scores.data_ptr()),
        static_cast<int*>(global_indices.data_ptr()), static_cast<int*>(sorted_indices.data_ptr()),
        batch_size * max_tokens, batch_size,
        static_cast<int*>(d_offsets.data_ptr()), static_cast<int*>(d_offsets.data_ptr()) + 1,
        0, sizeof(float)*8, stream
    );

    torch::Tensor temp_storage = torch::empty({(int64_t)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(q_c.device()));

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        static_cast<uint8_t*>(temp_storage.data_ptr()), temp_storage_bytes,
        static_cast<float*>(global_scores.data_ptr()), static_cast<float*>(sorted_scores.data_ptr()),
        static_cast<int*>(global_indices.data_ptr()), static_cast<int*>(sorted_indices.data_ptr()),
        batch_size * max_tokens, batch_size,
        static_cast<int*>(d_offsets.data_ptr()), static_cast<int*>(d_offsets.data_ptr()) + 1,
        0, sizeof(float)*8, stream
    );

    torch::Tensor topk_indices = torch::empty({batch_size, 2048}, options_int);
    dim3 copy_grid( (2048 + 255) / 256, batch_size );
    copy_topk_kernel<<<copy_grid, 256, 0, stream>>>(
        static_cast<const int*>(sorted_indices.data_ptr()), 
        static_cast<int*>(topk_indices.data_ptr()), 
        batch_size, max_tokens, 2048
    );

    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
}