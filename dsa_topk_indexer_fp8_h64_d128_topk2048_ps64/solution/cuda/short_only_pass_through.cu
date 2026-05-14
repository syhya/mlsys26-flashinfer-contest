// Short-only fast path for candidate v48b.
//
// Single-pass kernel: each warp handles one page's 64-entry output region,
// writing valid global token ids or -1 padding in a single sweep.
// Eliminates the previous two-pass approach (fill -1 + overwrite valid)
// and removes the __syncthreads barrier between them.
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kTopK = 2048;
constexpr int kPageSize = 64;
constexpr int kPassThroughThreads = 1024;
constexpr int kMaxShortPages = kTopK / kPageSize;

static_assert(kMaxShortPages == 32);

__launch_bounds__(kPassThroughThreads, 1)
__global__ void fill_short_only_rows_pass_through_kernel(
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    int32_t* __restrict__ topk_indices,
    int batch_size,
    int block_table_stride) {
  const int row = blockIdx.x;
  if (row >= batch_size) {
    return;
  }

  const int tid = threadIdx.x;
  const int warp_idx = tid >> 5;
  const int lane_idx = tid & 31;

  __shared__ int32_t s_pages[kMaxShortPages];

  const int32_t seq_len = seq_lens[row];
  int32_t* row_output = topk_indices + row * kTopK;

  // Load block_table into shared memory (only first 32 threads needed).
  for (int idx = tid; idx < kMaxShortPages; idx += kPassThroughThreads) {
    s_pages[idx] =
        (idx < block_table_stride) ? block_table[row * block_table_stride + idx] : -1;
  }
  __syncthreads();

  // Single pass: each warp writes one page's 64-entry region.
  // Warps beyond block_table_stride write all -1 (padding pages).
  // Warps within block_table_stride write valid tokens and -1 padding.
  if (warp_idx < kMaxShortPages) {
    const int page_start = warp_idx * kPageSize;
    if (warp_idx < block_table_stride) {
      const int32_t page_idx = s_pages[warp_idx];
      for (int offset = lane_idx; offset < kPageSize; offset += 32) {
        const int global_idx = page_start + offset;
        row_output[global_idx] = (global_idx < seq_len)
            ? static_cast<int32_t>((page_idx << 6) | offset)
            : -1;
      }
    } else {
      // Beyond block_table_stride: all -1 padding.
      for (int offset = lane_idx; offset < kPageSize; offset += 32) {
        row_output[page_start + offset] = -1;
      }
    }
  }
}

}  // namespace

cudaError_t fill_short_only_rows_pass_through(
    const int32_t* seq_lens,
    const int32_t* block_table,
    int32_t* topk_indices,
    int batch_size,
    int block_table_stride,
    cudaStream_t stream) {
  if (!seq_lens || !block_table || !topk_indices) {
    return cudaErrorInvalidValue;
  }
  if (batch_size <= 0 || block_table_stride <= 0 || block_table_stride > kMaxShortPages) {
    return cudaErrorInvalidValue;
  }

  fill_short_only_rows_pass_through_kernel<<<batch_size, kPassThroughThreads, 0, stream>>>(
      seq_lens,
      block_table,
      topk_indices,
      batch_size,
      block_table_stride);
  return cudaGetLastError();
}
