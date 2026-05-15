#include <torch/extension.h>
#include <cuda_fp8.h>
#include <tuple>
#include <cub/cub.cuh>

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
    
    if (p * 64 >= seq_len) {
        for (int i = threadIdx.x; i < 64; i += blockDim.x) {
            scores_buf[b * N + p * 64 + i] = -1e20f;
            indices_buf[b * N + p * 64 + i] = -1;
        }
        return;
    }

    __shared__ uint4 q_smem_u4[64][8]; // 64 heads, 8 uint4 per head
    __shared__ float weights_smem[64];

    const uint4* q_global_u4 = (const uint4*)(q_index_fp8 + b * 64 * 128);
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        ((uint4*)q_smem_u4)[i] = q_global_u4[i];
    }

    if (threadIdx.x < 64) {
        weights_smem[threadIdx.x] = weights[b * 64 + threadIdx.x];
    }

    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int group_id = warp_id * 4 + lane_id / 8; // 32 groups total
    int group_lane = lane_id % 8;
    
    int global_page_idx = block_table[b * max_num_pages + p];
    const uint8_t* k_page_base = k_index_cache_fp8 + global_page_idx * (64 * 132);

    for (int iter = 0; iter < 2; ++iter) {
        int t = iter * 32 + group_id;
        
        bool valid = (p * 64 + t < seq_len);
        float total_score = 0.0f;

        if (valid) {
            uint4 k_val_u4 = *(const uint4*)(k_page_base + t * 128 + group_lane * 16);
            float scale = *(const float*)(k_page_base + 8192 + t * 4);
            
            float4 k_f[4];
            k_f[0] = cvt_fp8x4_to_float4(k_val_u4.x);
            k_f[1] = cvt_fp8x4_to_float4(k_val_u4.y);
            k_f[2] = cvt_fp8x4_to_float4(k_val_u4.z);
            k_f[3] = cvt_fp8x4_to_float4(k_val_u4.w);

            #pragma unroll
            for(int i=0; i<4; ++i){
                k_f[i].x *= scale;
                k_f[i].y *= scale;
                k_f[i].z *= scale;
                k_f[i].w *= scale;
            }

            for (int h = 0; h < 64; ++h) {
                uint4 q_val_u4 = q_smem_u4[h][group_lane];
                
                float4 q_f0 = cvt_fp8x4_to_float4(q_val_u4.x);
                float4 q_f1 = cvt_fp8x4_to_float4(q_val_u4.y);
                float4 q_f2 = cvt_fp8x4_to_float4(q_val_u4.z);
                float4 q_f3 = cvt_fp8x4_to_float4(q_val_u4.w);

                float val = 0.0f;
                val += q_f0.x * k_f[0].x + q_f0.y * k_f[0].y + q_f0.z * k_f[0].z + q_f0.w * k_f[0].w;
                val += q_f1.x * k_f[1].x + q_f1.y * k_f[1].y + q_f1.z * k_f[1].z + q_f1.w * k_f[1].w;
                val += q_f2.x * k_f[2].x + q_f2.y * k_f[2].y + q_f2.z * k_f[2].z + q_f2.w * k_f[2].w;
                val += q_f3.x * k_f[3].x + q_f3.y * k_f[3].y + q_f3.z * k_f[3].z + q_f3.w * k_f[3].w;
                
                val += __shfl_xor_sync(0xffffffff, val, 4);
                val += __shfl_xor_sync(0xffffffff, val, 2);
                val += __shfl_xor_sync(0xffffffff, val, 1);
                
                if (group_lane == 0) {
                    if (val > 0.0f) {
                        total_score += val * weights_smem[h];
                    }
                }
            }
            if (group_lane == 0 && isnan(total_score)) {
                total_score = -1e20f; 
            }
        }

        if (group_lane == 0) {
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

__global__ void generate_offsets(int32_t* offsets, int batch_size, int segment_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= batch_size) {
        offsets[idx] = idx * segment_size;
    }
}

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

    int N = max_num_pages * 64; 
    auto scores_buf = torch::empty({batch_size, N}, torch::dtype(torch::kFloat32).device(device));
    auto indices_buf = torch::empty({batch_size, N}, torch::dtype(torch::kInt32).device(device));

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
}
