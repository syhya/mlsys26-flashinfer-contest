// CuTe/CUTLASS long-row scorer — v50: tile N=16, 8 warps, 256 threads.
//
// v50 (promoted from v49c) key changes over v48:
// - Double token tile from 8→16 (LONG_ROW_TOKEN_TILE=16).
// - TiledMMA layout: Shape<_4,_2,_1> giving M=64 (heads), N=16 (tokens),
//   K=16 with 8 warps / 256 threads.
// - Halves scorer CTAs (~280→~140 for medium band), better Q reuse per CTA.
// - Parallel epilogue: each warp independently reduces 2 tokens (32 lanes
//   cover 64 heads at 2 heads/lane), no inter-warp merge needed.
// - launch_bounds(256, 2) targets 2 CTAs/SM so reduced grid fits in 1 wave.
//
// Retained: vectorized uint4 Q/K loads, FP8→FP16 conversion, PDL, swizzle.
#include <cuda_runtime.h>

#include <cstdint>

#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

namespace cute_scorer {

using namespace cute;

using fp8_t = cutlass::float_e4m3_t;
using mma_t = cutlass::half_t;

constexpr int NUM_HEADS = 64;
constexpr int HEAD_DIM = 128;
constexpr int PAGE_SIZE = 64;
constexpr int HEAD_DIM_WITH_SCALE = 132;
constexpr int TOPK = 2048;
constexpr int LONG_ROW_TOKEN_TILE = 16;
constexpr int CUTE_SCORE_WARPS = 8;    // 4 in M (heads) × 2 in N (tokens)
constexpr int CUTE_SCORE_THREADS = CUTE_SCORE_WARPS * 32;  // 256
constexpr int CUTE_K_TILE = 16;

// Fast FP8 E4M3 pair → half2 using hardware PTX instruction.
// Converts 2 packed E4M3 bytes into a packed half2 in one instruction.
__device__ __forceinline__ uint32_t cvt_fp8x2_to_half2(uint16_t packed_fp8) {
  uint32_t result;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(packed_fp8));
  return result;
}

__device__ __forceinline__ float WarpReduceSum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ int32_t LocalToGlobalToken(
    const int32_t* block_table,
    int row,
    int block_table_stride,
    int32_t local_idx) {
  if (local_idx < 0) {
    return -1;
  }
  const int32_t page_idx =
      block_table[row * block_table_stride + (local_idx >> 6)];
  return static_cast<int32_t>((page_idx << 6) | (local_idx & (PAGE_SIZE - 1)));
}

__launch_bounds__(CUTE_SCORE_THREADS, 2)
__global__ void compute_logits_single_row_cute_tensor_kernel(
    const fp8_t* __restrict__ q_index_fp8,
    const uint8_t* __restrict__ k_index_cache_fp8,
    const float* __restrict__ weights,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    int32_t* __restrict__ long_row_meta,
    float* __restrict__ logits,
    int32_t* __restrict__ topk_indices,
    int batch_size,
    int max_num_pages,
    int num_long_row_tiles) {
  if (blockIdx.x >= static_cast<unsigned>(num_long_row_tiles)) {
    const int row = blockIdx.x - num_long_row_tiles;
    if (row >= batch_size) {
      return;
    }
    const int32_t seq_len = seq_lens[row];
    if (seq_len > TOPK) {
      return;
    }
    int32_t* row_output = topk_indices + row * TOPK;
    for (int idx = threadIdx.x; idx < TOPK; idx += blockDim.x) {
      row_output[idx] =
          (idx < seq_len)
              ? LocalToGlobalToken(block_table, row, max_num_pages, idx)
              : -1;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  // ---- Scorer tiles [0, num_long_row_tiles) ----

  using Swz = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(_8{}, _64{}), make_stride(_64{}, _1{}))));
  using SQ = decltype(tile_to_shape(Swz{}, Shape<Int<NUM_HEADS>, Int<HEAD_DIM>>{}));
  using SK = decltype(tile_to_shape(Swz{}, Shape<Int<LONG_ROW_TOKEN_TILE>, Int<HEAD_DIM>>{}));
  using LS = Layout<Shape<Int<NUM_HEADS>, Int<LONG_ROW_TOKEN_TILE>>,
                    Stride<Int<LONG_ROW_TOKEN_TILE>, _1>>;

  // CuTe arrays FIRST to preserve LDSM alignment requirements.
  __shared__ mma_t q_row_t[cosize(SQ{})];
  __shared__ mma_t k_tile[cosize(SK{})];
  __shared__ float scores[NUM_HEADS * LONG_ROW_TOKEN_TILE];
  __shared__ float token_scales[LONG_ROW_TOKEN_TILE];
  // Long-row discovery cache — placed AFTER CuTe arrays to preserve alignment.
  __shared__ int s_long_row;
  __shared__ int s_long_row_len;

  // Each CTA discovers the long row by scanning seq_lens inline.
  // batch_size <= 31, so this is at most 31 L2-cached int32 reads.
  if (threadIdx.x == 0) {
    int found = -1;
    int found_len = 0;
    for (int r = 0; r < batch_size; ++r) {
      int32_t sl = seq_lens[r];
      if (sl > TOPK) {
        found = r;
        found_len = sl;
        break;
      }
    }
    s_long_row = found;
    s_long_row_len = found_len;
    // First scorer CTA also writes long_row_meta for downstream histogram.
    if (blockIdx.x == 0) {
      long_row_meta[0] = found;
      long_row_meta[1] = found_len;
    }
  }
  __syncthreads();

  const int row = s_long_row;
  if (row < 0 || row >= batch_size) {
    return;
  }

  const int32_t seq_len = s_long_row_len;
  const int token_tile_start = blockIdx.x * LONG_ROW_TOKEN_TILE;
  if (token_tile_start >= seq_len) {
    return;
  }

  Tensor sQ = make_tensor(make_smem_ptr(q_row_t), SQ{});
  Tensor sK = make_tensor(make_smem_ptr(k_tile), SK{});
  Tensor sScore = make_tensor(make_smem_ptr(scores), LS{});

  const fp8_t* q_row =
      q_index_fp8 + static_cast<long long>(row) * NUM_HEADS * HEAD_DIM;
  const float* row_weight_ptr = weights + static_cast<long long>(row) * NUM_HEADS;
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // Load Q with vectorized 128-bit loads + paired FP8-to-FP16 conversion.
  // Each uint4 fetches 16 FP8 bytes; 8 cvt pairs produce 16 half values.
  // Q is at row*8192 offset from a CUDA-allocated base, so 16-byte aligned.
  {
    constexpr int Q_ELEMS = NUM_HEADS * HEAD_DIM;  // 8192
    constexpr int Q_VEC = 16;                       // 16 FP8 per uint4
    constexpr int Q_VECS = Q_ELEMS / Q_VEC;         // 512
    const auto* q_vec = reinterpret_cast<const uint4*>(q_row);
    for (int vi = threadIdx.x; vi < Q_VECS; vi += blockDim.x) {
      const uint4 packed16 = q_vec[vi];
      const uint16_t* pairs = reinterpret_cast<const uint16_t*>(&packed16);
      const int base = vi * Q_VEC;
      #pragma unroll
      for (int p = 0; p < 8; ++p) {
        uint32_t h2 = cvt_fp8x2_to_half2(pairs[p]);
        const mma_t* halves = reinterpret_cast<const mma_t*>(&h2);
        const int i0 = base + p * 2;
        sQ(i0 / HEAD_DIM, i0 % HEAD_DIM) = halves[0];
        sQ((i0 + 1) / HEAD_DIM, (i0 + 1) % HEAD_DIM) = halves[1];
      }
    }
  }

  const int32_t page_idx =
      block_table[row * max_num_pages + (token_tile_start / PAGE_SIZE)];
  const int32_t tile_token_off = token_tile_start % PAGE_SIZE;
  const uint8_t* page_base =
      k_index_cache_fp8 +
      static_cast<long long>(page_idx) * PAGE_SIZE * HEAD_DIM_WITH_SCALE;
  const int remaining_tokens = static_cast<int>(seq_len) - token_tile_start;
  const int tile_tokens =
      remaining_tokens < LONG_ROW_TOKEN_TILE ? remaining_tokens : LONG_ROW_TOKEN_TILE;

  // Load K with vectorized 128-bit loads + paired FP8-to-FP16 conversion.
  // Each token's 128-byte K row is 128-byte aligned, so uint4 loads are safe.
  {
    constexpr int K_VEC = 16;
    constexpr int K_VECS_PER_TOKEN = HEAD_DIM / K_VEC;  // 8
    for (int idx = threadIdx.x; idx < tile_tokens * K_VECS_PER_TOKEN; idx += blockDim.x) {
      const int lt = idx / K_VECS_PER_TOKEN;
      const int vi = idx % K_VECS_PER_TOKEN;
      const auto* token_vec = reinterpret_cast<const uint4*>(
          page_base + static_cast<long long>(tile_token_off + lt) * HEAD_DIM);
      const uint4 packed16 = token_vec[vi];
      const uint16_t* pairs = reinterpret_cast<const uint16_t*>(&packed16);
      const int d_base = vi * K_VEC;
      #pragma unroll
      for (int p = 0; p < 8; ++p) {
        uint32_t h2 = cvt_fp8x2_to_half2(pairs[p]);
        const mma_t* halves = reinterpret_cast<const mma_t*>(&h2);
        sK(lt, d_base + p * 2) = halves[0];
        sK(lt, d_base + p * 2 + 1) = halves[1];
      }
    }
  }
  for (int idx = threadIdx.x + tile_tokens * (HEAD_DIM / 2);
       idx < LONG_ROW_TOKEN_TILE * (HEAD_DIM / 2);
       idx += blockDim.x) {
    const int local_token = idx / (HEAD_DIM / 2);
    const int d_pair = idx % (HEAD_DIM / 2);
    const int d = d_pair * 2;
    sK(local_token, d) = mma_t(0.0f);
    sK(local_token, d + 1) = mma_t(0.0f);
  }
  for (int local_token = threadIdx.x; local_token < tile_tokens; local_token += blockDim.x) {
    token_scales[local_token] = *reinterpret_cast<const float*>(
        page_base + PAGE_SIZE * HEAD_DIM + (tile_token_off + local_token) * 4);
  }
  __syncthreads();

  using TiledMMA = decltype(make_tiled_mma(
      MMA_Atom<MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>>{},
      make_layout(Shape<_4, _2, _1>{}, LayoutRight{})));
  CUTE_STATIC_ASSERT_V(tile_size<0>(TiledMMA{}) == Int<NUM_HEADS>{});
  CUTE_STATIC_ASSERT_V(tile_size<1>(TiledMMA{}) == Int<LONG_ROW_TOKEN_TILE>{});
  CUTE_STATIC_ASSERT_V(tile_size<2>(TiledMMA{}) == Int<CUTE_K_TILE>{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tCsC = thr_mma.partition_C(sScore);
  auto accum = thr_mma.make_fragment_C(tCsC);
  clear(accum);
  cute::detail::cooperative_gemm_no_predication(
      threadIdx.x,
      thr_mma,
      sQ,
      sK,
      accum,
      cute::identity{},
      cute::identity{},
      SM75_U32x4_LDSM_N{},
      SM75_U32x2_LDSM_N{});
  cute::detail::epilogue_predication(
      thr_mma,
      1.0f,
      accum,
      0.0f,
      sScore,
      tCsC,
      cute::identity{},
      cute::identity{});
  __syncthreads();

  // Parallel epilogue: each warp reduces 2 tokens independently.
  // 8 warps × 2 tokens/warp = 16 tokens. 32 lanes cover 64 heads (2 per lane).
  {
    const int my_token_base = warp_id * 2;
    const int h0 = lane_id * 2;
    const int h1 = h0 + 1;
    // Pre-load per-head weights (broadcast across all warps).
    const float w0 = (h0 < NUM_HEADS) ? row_weight_ptr[h0] : 0.0f;
    const float w1 = (h1 < NUM_HEADS) ? row_weight_ptr[h1] : 0.0f;

    #pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int tile_idx = my_token_base + t;
      if (tile_idx >= tile_tokens) break;
      const float scale = token_scales[tile_idx];
      float partial = 0.0f;
      if (h0 < NUM_HEADS) {
        float s0 = sScore(h0, tile_idx) * scale;
        partial += (s0 < 0.0f ? 0.0f : s0) * w0;
      }
      if (h1 < NUM_HEADS) {
        float s1 = sScore(h1, tile_idx) * scale;
        partial += (s1 < 0.0f ? 0.0f : s1) * w1;
      }
      float sum = WarpReduceSum(partial);
      if (lane_id == 0) {
        logits[token_tile_start + tile_idx] = sum;
      }
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

}  // namespace cute_scorer

cudaError_t compute_logits_single_row_cute_tensor(
    const cutlass::float_e4m3_t* q_index_fp8,
    const uint8_t* k_index_cache_fp8,
    const float* weights,
    const int32_t* seq_lens,
    const int32_t* block_table,
    int32_t* long_row_meta,
    float* logits,
    int32_t* topk_indices,
    int batch_size,
    int max_num_pages,
    int num_long_row_tiles,
    cudaStream_t stream) {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(num_long_row_tiles + batch_size, 1, 1);
  config.blockDim = dim3(cute_scorer::CUTE_SCORE_THREADS, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;

  return cudaLaunchKernelEx(
      &config,
      cute_scorer::compute_logits_single_row_cute_tensor_kernel,
      q_index_fp8,
      k_index_cache_fp8,
      weights,
      seq_lens,
      block_table,
      long_row_meta,
      logits,
      topk_indices,
      batch_size,
      max_num_pages,
      num_long_row_tiles);
}
