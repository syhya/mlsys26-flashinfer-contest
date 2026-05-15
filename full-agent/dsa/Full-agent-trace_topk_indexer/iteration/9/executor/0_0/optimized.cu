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
    int p = blockIdx.x; 
    int b = blockIdx.y; 
    
    int seq_len = seq_lens[b];
    int N = max_num_pages * 64;
    
    if (p * 64 >= seq_len) {
        return;
    }

    __shared__ float4 q_smem_f4[2048]; // 64 heads * 32 float4
    __shared__ float weights_smem[64];

    const uint4* q_global_u4 = (const uint4*)(q_index_fp8 + b * 8192);
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        uint4 q_val_u4 = q_global_u4[i];
        q_smem_f4[i * 4 + 0] = cvt_fp8x4_to_float4(q_val_u4.x);
        q_smem_f4[i * 4 + 1] = cvt_fp8x4_to_float4(q_val_u4.y);
        q_smem_f4[i * 4 + 2] = cvt_fp8x4_to_float4(q_val_u4.z);
        q_smem_f4[i * 4 + 3] = cvt_fp8x4_to_float4(q_val_u4.w);
    }

    if (threadIdx.x < 64) {
        weights_smem[threadIdx.x] = weights[b * 64 + threadIdx.x];
    }

    __syncthreads();

    int group_id = threadIdx.x / 8;
    int lane_in_group = threadIdx.x % 8;
    
    int global_page_idx = block_table[b * max_num_pages + p];
    const uint8_t* k_page_base = k_index_cache_fp8 + global_page_idx * (64 * 132);

    for (int iter = 0; iter < 2; ++iter) {
        int t = iter * 32 + group_id;
        
        uint4 k_val_u4 = *(const uint4*)(k_page_base + t * 128 + lane_in_group * 16);
        float4 k_f0 = cvt_fp8x4_to_float4(k_val_u4.x);
        float4 k_f1 = cvt_fp8x4_to_float4(k_val_u4.y);
        float4 k_f2 = cvt_fp8x4_to_float4(k_val_u4.z);
        float4 k_f3 = cvt_fp8x4_to_float4(k_val_u4.w);
        
        float scale = 0.0f;
        if (lane_in_group == 0) {
            scale = *(const float*)(k_page_base + 8192 + t * 4);
        }
        scale = __shfl_sync(0xffffffff, scale, (threadIdx.x / 8) * 8);
        
        k_f0.x *= scale; k_f0.y *= scale; k_f0.z *= scale; k_f0.w *= scale;
        k_f1.x *= scale; k_f1.y *= scale; k_f1.z *= scale; k_f1.w *= scale;
        k_f2.x *= scale; k_f2.y *= scale; k_f2.z *= scale; k_f2.w *= scale;
        k_f3.x *= scale; k_f3.y *= scale; k_f3.z *= scale; k_f3.w *= scale;

        float total_score = 0.0f;

        #pragma unroll
        for (int h = 0; h < 64; h += 8) {
            float val0 = 0.0f, val1 = 0.0f, val2 = 0.0f, val3 = 0.0f;
            float val4 = 0.0f, val5 = 0.0f, val6 = 0.0f, val7 = 0.0f;
            float4 q;
            
            int q_base = h * 32 + lane_in_group * 4;
            
            // h+0
            q = q_smem_f4[q_base + 0]; val0 += q.x*k_f0.x + q.y*k_f0.y + q.z*k_f0.z + q.w*k_f0.w;
            q = q_smem_f4[q_base + 1]; val0 += q.x*k_f1.x + q.y*k_f1.y + q.z*k_f1.z + q.w*k_f1.w;
            q = q_smem_f4[q_base + 2]; val0 += q.x*k_f2.x + q.y*k_f2.y + q.z*k_f2.z + q.w*k_f2.w;
            q = q_smem_f4[q_base + 3]; val0 += q.x*k_f3.x + q.y*k_f3.y + q.z*k_f3.z + q.w*k_f3.w;

            // h+1
            q_base += 32;
            q = q_smem_f4[q_base + 0]; val1 += q.x*k_f0.x + q.y*k_f0.y + q.z*k_f0.z + q.w*k_f0.w;
            q = q_smem_f4[q_base + 1]; val1 += q.x*k_f1.x + q.y*k_f1.y + q.z*k_f1.z + q.w*k_f1.w;
            q = q_smem_f4[q_base + 2]; val1 += q.x*k_f2.x + q.y*k_f2.y + q.z*k_f2.z + q.w*k_f2.w;
            q = q_smem_f4[q_base + 3]; val1 += q.x*k_f3.x + q.y*k_f3.y + q.z*k_f3.z + q.w*k_f3.w;

            // h+2
            q_base += 32;
            q = q_smem_f4[q_base + 0]; val2 += q.x*k_f0.x + q.y*k_f0.y + q.z*k_f0.z + q.w*k_f0.w;
            q = q_smem_f4[q_base + 1]; val2 += q.x*k_f1.x + q.y*k_f1.y + q.z*k_f1.z + q.w*k_f1.w;
            q = q_smem_f4[q_base + 2]; val2 += q.x*k_f2.x + q.y*k_f2.y + q.z*k_f2.z + q.w*k_f2.w;
            q = q_smem_f4[q_base + 3]; val2 += q.x*k_f3.x + q.y*k_f3.y + q.z*k_f3.z + q.w*k_f3.w;

            // h+3
            q_base += 32;
            q = q_smem_f4[q_base + 0]; val3 += q.x*k_f0.x + q.y*k_f0.y + q.z*k_f0.z + q.w*k_f0.w;
            q = q_smem_f4[q_base + 1]; val3 += q.x*k_f1.x + q.y*k_f1.y + q.z*k_f1.z + q.w*k_f1.w;
            q = q_smem_f4[q_base + 2]; val3 += q.x*k_f2.x + q.y*k_f2.y + q.z*k_f2.z + q.w*k_f2.w;
            q = q_smem_f4[q_base + 3]; val3 += q.x*k_f3.x + q.y*k_f3.y + q.z*k_f3.z + q.w*k_f3.w;

            // h+4
            q_base += 32;
            q = q_smem_f4[q_base + 0]; val4 += q.x*k_f0.x + q.y*k_f0.y + q.z*k_f0.z + q.w*k_f0.w;
            q = q_smem_f4[q_base + 1]; val4 += q.x*k_f1.x + q.y*k_f1.y + q.z*k_f1.z + q.w*k_f1.w;
            q = q_smem_f4[q_base + 2]; val4 += q.x*k_f2.x + q.y*k_f2.y + q.z*k_f2.z + q.w*k_f2.w;
            q = q_smem_f4[q_base + 3]; val4 += q.x*k_f3.x + q.y*k_f3.y + q.z*k_f3.z + q.w*k_f3.w;

            // h+5
            q_base += 32;
            q = q_smem_f4[q_base + 0]; val5 += q.x*k_f0.x + q.y*k_f0.y + q.z*k_f0.z + q.w*k_f0.w;
            q = q_smem_f4[q_base + 1]; val5 += q.x*k_f1.x + q.y*k_f1.y + q.z*k_f1.z + q.w*k_f1.w;
            q = q_smem_f4[q_base + 2]; val5 += q.x*k_f2.x + q.y*k_f2.y + q.z*k_f2.z + q.w*k_f2.w;
            q = q_smem_f4[q_base + 3]; val5 += q.x*k_f3.x + q.y*k_f3.y + q.z*k_f3.z + q.w*k_f3.w;

            // h+6
            q_base += 32;
            q = q_smem_f4[q_base + 0]; val6 += q.x*k_f0.x + q.y*k_f0.y + q.z*k_f0.z + q.w*k_f0.w;
            q = q_smem_f4[q_base + 1]; val6 += q.x*k_f1.x + q.y*k_f1.y + q.z*k_f1.z + q.w*k_f1.w;
            q = q_smem_f4[q_base + 2]; val6 += q.x*k_f2.x + q.y*k_f2.y + q.z*k_f2.z + q.w*k_f2.w;
            q = q_smem_f4[q_base + 3]; val6 += q.x*k_f3.x + q.y*k_f3.y + q.z*k_f3.z + q.w*k_f3.w;

            // h+7
            q_base += 32;
            q = q_smem_f4[q_base + 0]; val7 += q.x*k_f0.x + q.y*k_f0.y + q.z*k_f0.z + q.w*k_f0.w;
            q = q_smem_f4[q_base + 1]; val7 += q.x*k_f1.x + q.y*k_f1.y + q.z*k_f1.z + q.w*k_f1.w;
            q = q_smem_f4[q_base + 2]; val7 += q.x*k_f2.x + q.y*k_f2.y + q.z*k_f2.z + q.w*k_f2.w;
            q = q_smem_f4[q_base + 3]; val7 += q.x*k_f3.x + q.y*k_f3.y + q.z*k_f3.z + q.w*k_f3.w;

            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                val0 += __shfl_xor_sync(0xffffffff, val0, offset);
                val1 += __shfl_xor_sync(0xffffffff, val1, offset);
                val2 += __shfl_xor_sync(0xffffffff, val2, offset);
                val3 += __shfl_xor_sync(0xffffffff, val3, offset);
                val4 += __shfl_xor_sync(0xffffffff, val4, offset);
                val5 += __shfl_xor_sync(0xffffffff, val5, offset);
                val6 += __shfl_xor_sync(0xffffffff, val6, offset);
                val7 += __shfl_xor_sync(0xffffffff, val7, offset);
            }

            if (lane_in_group == 0) {
                if (val0 > 0.0f) total_score += val0 * weights_smem[h+0];
                if (val1 > 0.0f) total_score += val1 * weights_smem[h+1];
                if (val2 > 0.0f) total_score += val2 * weights_smem[h+2];
                if (val3 > 0.0f) total_score += val3 * weights_smem[h+3];
                if (val4 > 0.0f) total_score += val4 * weights_smem[h+4];
                if (val5 > 0.0f) total_score += val5 * weights_smem[h+5];
                if (val6 > 0.0f) total_score += val6 * weights_smem[h+6];
                if (val7 > 0.0f) total_score += val7 * weights_smem[h+7];
            }
        }

        bool valid = (p * 64 + t < seq_len);
        if (lane_in_group == 0) {
            if (!valid || isnan(total_score)) {
                total_score = -1e20f;
            }
            int out_idx = b * N + p * 64 + t;
            scores_buf[out_idx] = total_score;
            indices_buf[out_idx] = global_page_idx * 64 + t;
        }
    }
}

__global__ void generate_offsets_begin_end(
    int32_t* begin_offsets, 
    int32_t* end_offsets, 
    const int32_t* seq_lens, 
    int batch_size, 
    int segment_size
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        begin_offsets[b] = b * segment_size;
        end_offsets[b] = b * segment_size + seq_lens[b];
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

    auto begin_offsets = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
    auto end_offsets = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
    int num_blocks = (batch_size + 255) / 256;
    generate_offsets_begin_end<<<num_blocks, 256>>>(
        begin_offsets.data_ptr<int32_t>(), 
        end_offsets.data_ptr<int32_t>(), 
        sl.data_ptr<int32_t>(), 
        batch_size, 
        N
    );

    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        scores_buf.data_ptr<float>(), scores_sorted.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(), indices_sorted.data_ptr<int32_t>(),
        batch_size * N, batch_size,
        begin_offsets.data_ptr<int32_t>(), end_offsets.data_ptr<int32_t>()
    );

    auto temp_storage = torch::empty({(long)temp_storage_bytes}, torch::dtype(torch::kUInt8).device(device));

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data_ptr(), temp_storage_bytes,
        scores_buf.data_ptr<float>(), scores_sorted.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(), indices_sorted.data_ptr<int32_t>(),
        batch_size * N, batch_size,
        begin_offsets.data_ptr<int32_t>(), end_offsets.data_ptr<int32_t>()
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
    m.def("dsa_forward", &topk_indexer_forward, "DSA TopK Indexer Forward Alias");
}
