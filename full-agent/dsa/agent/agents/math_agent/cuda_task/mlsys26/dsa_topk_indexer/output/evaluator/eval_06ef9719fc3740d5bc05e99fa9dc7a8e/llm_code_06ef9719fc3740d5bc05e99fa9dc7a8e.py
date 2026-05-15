#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <tuple>

constexpr int TOPK_CONST = 2048;
constexpr int L_MED_MAX  = 6144;
constexpr int BT_THREADS = 256;
constexpr int BT_ITEMS   = 24;
static_assert(BT_THREADS * BT_ITEMS >= L_MED_MAX, "buffer sufficiency");

inline __device__ float4 cvt_fp8x4_to_float4(uint32_t val) {
    float4 res;
    union {
        uint32_t u32;
        __nv_fp8_e4m3 v[4];
    } tmp;
    tmp.u32 = val;
    res.x = float(tmp.v[0]);
    res.y = float(tmp.v[1]);
    res.z = float(tmp.v[2]);
    res.w = float(tmp.v[3]);
    return res;
}

// Fast-path short: L <= topk
__global__ void emit_short_topk_kernel(
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    int32_t* __restrict__ topk_indices,
    int max_num_pages,
    int topk
) {
    int b = blockIdx.x;
    int L = seq_lens[b];
    if (L > topk) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    for (int i = tid; i < topk; i += bs) {
        int out_val;
        if (i < L) {
            int p = i >> 6;
            int t = i & 63;
            int page = block_table[b * max_num_pages + p];
            out_val = page * 64 + t;
        } else {
            out_val = -1;
        }
        topk_indices[b * topk + i] = out_val;
    }
}

// Long path: L > L_MED_MAX -> compute scores_buf/indices_buf
__global__ void compute_scores_kernel(
    const uint8_t* __restrict__ q_index_fp8,
    const uint8_t* __restrict__ k_index_cache_fp8,
    const float* __restrict__ weights,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    float* __restrict__ scores_buf,
    int32_t* __restrict__ indices_buf,
    int max_num_pages,
    int topk
) {
    int p = blockIdx.x;
    int b = blockIdx.y;

    int seq_len = seq_lens[b];
    int N = max_num_pages * 64;

    if (seq_len <= L_MED_MAX) return;
    if (p * 64 >= seq_len) return;

    __shared__ float q_smem_f[128][65];
    __shared__ float k_smem[8][128];
    __shared__ float weights_smem[64];

    const uint32_t* q_global_u32 = (const uint32_t*)(q_index_fp8 + b * 8192);
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
        uint32_t q_val = q_global_u32[i];
        float4 q_f = cvt_fp8x4_to_float4(q_val);
        int head = i / 32;
        int cg = i % 32;
        q_smem_f[cg * 4 + 0][head] = q_f.x;
        q_smem_f[cg * 4 + 1][head] = q_f.y;
        q_smem_f[cg * 4 + 2][head] = q_f.z;
        q_smem_f[cg * 4 + 3][head] = q_f.w;
    }

    if (threadIdx.x < 64) {
        weights_smem[threadIdx.x] = weights[b * 64 + threadIdx.x];
    }

    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int global_page_idx = block_table[b * max_num_pages + p];
    const uint8_t* k_page_base = k_index_cache_fp8 + global_page_idx * (64 * 132);

    for (int iter = 0; iter < 8; ++iter) {
        int t = iter * 8 + warp_id;
        bool valid = (p * 64 + t < seq_len);

        uint32_t k_val = *(const uint32_t*)(k_page_base + t * 128 + lane_id * 4);
        float4 k_f = cvt_fp8x4_to_float4(k_val);
        ((float4*)(k_smem[warp_id]))[lane_id] = k_f;

        __syncwarp();

        if (valid) {
            float val0 = 0.0f;
            float val1 = 0.0f;

            #pragma unroll
            for (int i = 0; i < 128; i++) {
                float k_v = k_smem[warp_id][i];
                val0 += q_smem_f[i][lane_id] * k_v;
                val1 += q_smem_f[i][lane_id + 32] * k_v;
            }

            float scale;
            if (lane_id == 0) scale = *(const float*)(k_page_base + 8192 + t * 4);
            scale = __shfl_sync(0xffffffff, scale, 0);

            val0 *= scale;
            val1 *= scale;

            float head_score = 0.0f;
            if (val0 > 0.0f) head_score += val0 * weights_smem[lane_id];
            if (val1 > 0.0f) head_score += val1 * weights_smem[lane_id + 32];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                head_score += __shfl_down_sync(0xffffffff, head_score, offset);
            }

            if (lane_id == 0) {
                if (isnan(head_score)) {
                    head_score = -1e20f;
                }
                int out_idx = b * N + p * 64 + t;
                scores_buf[out_idx] = head_score;
                indices_buf[out_idx] = global_page_idx * 64 + t;
            }
        }
        __syncwarp();
    }
}

// Fused score + block radix sort for TIER-M: 2048 < L <= 6144
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void fused_score_topk_kernel(
    const uint8_t* __restrict__ q_index_fp8,
    const uint8_t* __restrict__ k_index_cache_fp8,
    const float* __restrict__ weights,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    int32_t* __restrict__ topk_indices,
    int max_num_pages,
    int topk
) {
    int b = blockIdx.x;
    int L = seq_lens[b];
    if (L <= topk) return;           // TIER-S
    if (L > L_MED_MAX) return;       // TIER-L

    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int32_t>;

    // Static shared memory
    __shared__ float   q_smem_f[128][65];   // 33.28 KB
    __shared__ float   weights_smem[64];
    __shared__ float   k_smem[8][128];      // 4 KB

    // Dynamic shared memory: overlay scores+indices (48KB) with CUB temp storage
    extern __shared__ unsigned char smem_raw[];
    float*   scores_shmem  = reinterpret_cast<float*>(smem_raw);
    int32_t* indices_shmem = reinterpret_cast<int32_t*>(smem_raw + sizeof(float) * L_MED_MAX);

    // Step 1: Load Q
    const uint32_t* q_global_u32 = (const uint32_t*)(q_index_fp8 + b * 8192);
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
        uint32_t q_val = q_global_u32[i];
        float4 q_f = cvt_fp8x4_to_float4(q_val);
        int head = i / 32;
        int cg = i % 32;
        q_smem_f[cg * 4 + 0][head] = q_f.x;
        q_smem_f[cg * 4 + 1][head] = q_f.y;
        q_smem_f[cg * 4 + 2][head] = q_f.z;
        q_smem_f[cg * 4 + 3][head] = q_f.w;
    }
    if (threadIdx.x < 64) weights_smem[threadIdx.x] = weights[b * 64 + threadIdx.x];

    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;  // 8

    int num_pages_b = (L + 63) >> 6;

    // Step 2: Each warp processes entire pages
    for (int p = warp_id; p < num_pages_b; p += NUM_WARPS) {
        int global_page_idx = block_table[b * max_num_pages + p];
        const uint8_t* k_page_base = k_index_cache_fp8 + global_page_idx * (64 * 132);

        for (int iter = 0; iter < 8; ++iter) {
            // Reuse 8-tokens-per-iter pattern: warp processes 8 tokens together via lanes?
            // Actually simpler: one warp does one token at a time, 32 lanes compute 128 dims (4 per lane).
            // But that's slow. Use parent pattern: within a warp handling one page, process 8 iterations
            // and within each iter, ONE warp handles ONE token (not 8). Better: handle 1 token per iter,
            // 64 tokens per page = 64 iterations per page per warp. Too many.
            //
            // Alternative: mimic parent exactly - one warp does 8 tokens simultaneously is wrong;
            // in parent one block(page) has 8 warps, each warp does one token at iter via warp_id.
            // Here one warp owns a whole page, so serialize: 64 tokens over 8 iter groups with 8 tokens each,
            // but we only have 1 warp. So do 64 tokens sequentially.
            //
            // Simpler and still fast: lane computes 2 heads via dot product with shared K row.
            break; // placeholder, use nested loop below
        }

        // Process all 64 tokens of this page
        for (int t = 0; t < 64; ++t) {
            int global_pos = p * 64 + t;
            bool valid = (global_pos < L);

            // Cooperative K-load: 32 lanes each load 4 FP8 bytes -> float4
            uint32_t k_val = *(const uint32_t*)(k_page_base + t * 128 + lane_id * 4);
            float4 k_f = cvt_fp8x4_to_float4(k_val);
            ((float4*)(k_smem[warp_id]))[lane_id] = k_f;

            __syncwarp();

            float head_score = 0.0f;
            if (valid) {
                float val0 = 0.0f;
                float val1 = 0.0f;

                #pragma unroll
                for (int i = 0; i < 128; i++) {
                    float k_v = k_smem[warp_id][i];
                    val0 += q_smem_f[i][lane_id] * k_v;
                    val1 += q_smem_f[i][lane_id + 32] * k_v;
                }

                float scale;
                if (lane_id == 0) scale = *(const float*)(k_page_base + 8192 + t * 4);
                scale = __shfl_sync(0xffffffff, scale, 0);

                val0 *= scale;
                val1 *= scale;

                if (val0 > 0.0f) head_score += val0 * weights_smem[lane_id];
                if (val1 > 0.0f) head_score += val1 * weights_smem[lane_id + 32];

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    head_score += __shfl_down_sync(0xffffffff, head_score, offset);
                }
            }

            if (lane_id == 0) {
                if (valid) {
                    if (isnan(head_score)) head_score = -1e20f;
                    scores_shmem[global_pos] = head_score;
                    indices_shmem[global_pos] = global_page_idx * 64 + t;
                } else {
                    scores_shmem[global_pos] = -1e30f;
                    indices_shmem[global_pos] = -1;
                }
            }
            __syncwarp();
        }
    }

    __syncthreads();

    // Pad [L, L_MED_MAX) with sentinel
    // But we also must pad page-aligned regions beyond num_pages_b*64 too
    int pad_start = num_pages_b * 64;
    // Actually the loop above only wrote positions 0..num_pages_b*64-1. Positions from num_pages_b*64..L_MED_MAX need padding.
    for (int i = pad_start + threadIdx.x; i < L_MED_MAX; i += BLOCK_THREADS) {
        scores_shmem[i] = -1e30f;
        indices_shmem[i] = -1;
    }
    __syncthreads();

    // Step 3: Strided load into registers
    float   keys[ITEMS_PER_THREAD];
    int32_t vals[ITEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = i * BLOCK_THREADS + threadIdx.x;
        if (idx < L_MED_MAX) {
            keys[i] = scores_shmem[idx];
            vals[i] = indices_shmem[idx];
        } else {
            keys[i] = -1e30f;
            vals[i] = -1;
        }
    }

    __syncthreads();

    // Overlay: scores/indices region is now dead; reuse for CUB temp storage
    // We need a separate declaration since CUB temp_storage needs a typed __shared__.
    // Trick: reinterpret_cast the dynamic smem.
    auto& temp_storage = *reinterpret_cast<typename BlockRadixSort::TempStorage*>(smem_raw);

    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(keys, vals);

    // Step 4: Strided write
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int out_pos = i * BLOCK_THREADS + threadIdx.x;
        if (out_pos < topk) {
            topk_indices[b * topk + out_pos] = vals[i];
        }
    }
}

__global__ void generate_offsets_begin_end(
    int32_t* begin_offsets,
    int32_t* end_offsets,
    const int32_t* seq_lens,
    int batch_size,
    int segment_size,
    int topk
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        int L = seq_lens[b];
        begin_offsets[b] = b * segment_size;
        // empty segments for batches handled by short/medium paths
        end_offsets[b] = (L <= L_MED_MAX) ? (b * segment_size) : (b * segment_size + L);
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
    if (seq_len <= L_MED_MAX) return;

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
    int topk = TOPK_CONST;

    auto q = q_index_fp8.contiguous();
    auto k = k_index_cache_fp8.contiguous();
    auto w = weights.contiguous();
    auto sl = seq_lens.contiguous();
    auto bt = block_table.contiguous();

    auto device = q.device();

    int N = max_num_pages * 64;
    auto scores_buf = torch::empty({batch_size, N}, torch::dtype(torch::kFloat32).device(device));
    auto indices_buf = torch::empty({batch_size, N}, torch::dtype(torch::kInt32).device(device));

    auto topk_indices = torch::empty({batch_size, topk}, torch::dtype(torch::kInt32).device(device));

    // Step A: TIER-S fast-path
    emit_short_topk_kernel<<<batch_size, 256>>>(
        sl.data_ptr<int32_t>(),
        bt.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        max_num_pages,
        topk
    );

    // Step B: TIER-L compute scores (only for L > L_MED_MAX)
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
        max_num_pages,
        topk
    );

    // Step C: TIER-M fused kernel
    using BRS = cub::BlockRadixSort<float, BT_THREADS, BT_ITEMS, int32_t>;
    size_t cub_temp_bytes = sizeof(typename BRS::TempStorage);
    size_t scores_bytes = sizeof(float) * L_MED_MAX + sizeof(int32_t) * L_MED_MAX;
    size_t dyn_smem = cub_temp_bytes > scores_bytes ? cub_temp_bytes : scores_bytes;

    static bool smem_set = false;
    if (!smem_set) {
        cudaFuncSetAttribute(
            (const void*)fused_score_topk_kernel<BT_THREADS, BT_ITEMS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            98304
        );
        smem_set = true;
    }

    fused_score_topk_kernel<BT_THREADS, BT_ITEMS><<<batch_size, BT_THREADS, dyn_smem>>>(
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k.data_ptr()),
        w.data_ptr<float>(),
        sl.data_ptr<int32_t>(),
        bt.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        max_num_pages,
        topk
    );

    // Step D: TIER-L offsets + CUB segmented radix sort
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
        N,
        topk
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
