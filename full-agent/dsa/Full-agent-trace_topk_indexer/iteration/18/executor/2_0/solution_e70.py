#include <torch/extension.h>
#include <cuda_fp8.h>
#include <tuple>
#include <cub/cub.cuh>
#include <cub/block/block_radix_sort.cuh>

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

// TIER-L only: L > L_MED_MAX
__global__ void compute_scores_kernel(
    const uint8_t* __restrict__ q_index_fp8,
    const uint8_t* __restrict__ k_index_cache_fp8,
    const float* __restrict__ weights,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    float* __restrict__ scores_buf,
    int32_t* __restrict__ indices_buf,
    int max_num_pages,
    int topk,
    int l_med_max
) {
    int p = blockIdx.x;
    int b = blockIdx.y;

    int seq_len = seq_lens[b];
    int N = max_num_pages * 64;

    if (seq_len <= l_med_max) return;
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

// TIER-M FUSED: 2048 < L <= L_MED_MAX (6144)
// Uses dynamic shared memory for scores+indices buffers (overlay with cub temp storage).
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void fused_score_topk_kernel(
    const uint8_t* __restrict__ q_index_fp8,
    const uint8_t* __restrict__ k_index_cache_fp8,
    const float* __restrict__ weights,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    int32_t* __restrict__ topk_indices,
    int max_num_pages,
    int topk,
    int l_med_max
) {
    int b = blockIdx.x;
    int L = seq_lens[b];
    if (L <= topk) return;        // TIER-S
    if (L > l_med_max) return;    // TIER-L

    constexpr int SCORE_BUF = BLOCK_THREADS * ITEMS_PER_THREAD; // 6144

    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int32_t>;

    // Dynamic shmem layout (overlay):
    // Region A (Phase 1): scores [SCORE_BUF*4] + indices [SCORE_BUF*4] = 48KB
    // Region B (Phase 2): cub temp_storage
    // Use max of both sizes as dynamic smem allocation.
    extern __shared__ unsigned char smem_dyn[];
    float*   scores_shmem  = reinterpret_cast<float*>(smem_dyn);
    int32_t* indices_shmem = reinterpret_cast<int32_t*>(smem_dyn + SCORE_BUF * sizeof(float));

    // Static shared (smaller stuff): Q + weights + per-warp K rows
    __shared__ float q_smem_f[128][65];      // 33.28 KB
    __shared__ float weights_smem[64];        // 0.25 KB
    __shared__ float k_smem[8][128];          // 4 KB

    // Load Q
    const uint32_t* q_global_u32 = (const uint32_t*)(q_index_fp8 + b * 8192);
    for (int i = threadIdx.x; i < 2048; i += BLOCK_THREADS) {
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

    int num_pages_b = (L + 63) >> 6;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;  // 8

    __syncthreads();

    // Score pages: each warp handles one page (all 64 tokens serially).
    for (int p = warp_id; p < num_pages_b; p += NUM_WARPS) {
        int global_page_idx = block_table[b * max_num_pages + p];
        const uint8_t* k_page_base = k_index_cache_fp8 + global_page_idx * (64 * 132);

        for (int t = 0; t < 64; ++t) {
            int n = p * 64 + t;
            bool valid = (n < L);

            uint32_t k_val = *(const uint32_t*)(k_page_base + t * 128 + lane_id * 4);
            float4 k_f = cvt_fp8x4_to_float4(k_val);
            ((float4*)(k_smem[warp_id]))[lane_id] = k_f;
            __syncwarp();

            float val0 = 0.0f, val1 = 0.0f;
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
                if (isnan(head_score)) head_score = -1e20f;
                if (!valid) head_score = -1e30f;
                scores_shmem[n] = head_score;
                indices_shmem[n] = valid ? (global_page_idx * 64 + t) : -1;
            }
            __syncwarp();
        }
    }

    // Pad rest
    int pad_start = num_pages_b * 64;
    for (int i = pad_start + threadIdx.x; i < SCORE_BUF; i += BLOCK_THREADS) {
        scores_shmem[i] = -1e30f;
        indices_shmem[i] = -1;
    }
    __syncthreads();

    // Load into registers, strided
    float   keys[ITEMS_PER_THREAD];
    int32_t vals[ITEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = i * BLOCK_THREADS + threadIdx.x;
        keys[i] = scores_shmem[idx];
        vals[i] = indices_shmem[idx];
    }
    __syncthreads();   // ensure all threads finished reading before reusing region

    // Now overlay temp_storage onto the same dynamic smem region.
    auto& temp_storage = *reinterpret_cast<typename BlockRadixSort::TempStorage*>(smem_dyn);

    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(keys, vals);

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
    int topk,
    int l_med_max
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        int L = seq_lens[b];
        begin_offsets[b] = b * segment_size;
        end_offsets[b] = (L <= l_med_max) ? (b * segment_size) : (b * segment_size + L);
    }
}

__global__ void extract_topk_kernel(
    const int32_t* __restrict__ indices_sorted,
    int32_t* __restrict__ topk_indices,
    const int32_t* __restrict__ seq_lens,
    int batch_size,
    int segment_size,
    int topk,
    int l_med_max
) {
    int b = blockIdx.x;
    int seq_len = seq_lens[b];
    if (seq_len <= l_med_max) return;

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
    int l_med_max = L_MED_MAX;

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

    // TIER-S
    emit_short_topk_kernel<<<batch_size, 256>>>(
        sl.data_ptr<int32_t>(),
        bt.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        max_num_pages, topk
    );

    // TIER-L scoring
    dim3 grid(max_num_pages, batch_size);
    compute_scores_kernel<<<grid, 256>>>(
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k.data_ptr()),
        w.data_ptr<float>(),
        sl.data_ptr<int32_t>(),
        bt.data_ptr<int32_t>(),
        scores_buf.data_ptr<float>(),
        indices_buf.data_ptr<int32_t>(),
        max_num_pages, topk, l_med_max
    );

    // TIER-M fused
    {
        using BlockRadixSort = cub::BlockRadixSort<float, BT_THREADS, BT_ITEMS, int32_t>;
        constexpr size_t SCORE_BYTES = (size_t)BT_THREADS * BT_ITEMS * (sizeof(float) + sizeof(int32_t));
        constexpr size_t TEMP_BYTES  = sizeof(typename BlockRadixSort::TempStorage);
        size_t dyn_smem = SCORE_BYTES > TEMP_BYTES ? SCORE_BYTES : TEMP_BYTES;

        static bool smem_set = false;
        if (!smem_set) {
            cudaFuncSetAttribute(
                (const void*)fused_score_topk_kernel<BT_THREADS, BT_ITEMS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)dyn_smem
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
            max_num_pages, topk, l_med_max
        );
    }

    // TIER-L sort + extract
    auto scores_sorted = torch::empty_like(scores_buf);
    auto indices_sorted = torch::empty_like(indices_buf);

    auto begin_offsets = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
    auto end_offsets = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
    int num_blocks = (batch_size + 255) / 256;
    generate_offsets_begin_end<<<num_blocks, 256>>>(
        begin_offsets.data_ptr<int32_t>(),
        end_offsets.data_ptr<int32_t>(),
        sl.data_ptr<int32_t>(),
        batch_size, N, topk, l_med_max
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
        batch_size, N, topk, l_med_max
    );

    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_indexer_forward", &topk_indexer_forward, "DSA TopK Indexer Forward");
    m.def("dsa_forward", &topk_indexer_forward, "DSA TopK Indexer Forward Alias");
}
