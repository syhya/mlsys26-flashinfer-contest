// Candidate v48b (`filtered_vec4_loads`).
//
// v48b change over v47: vectorized float4 loads in the filtered selector's
// coarse histogram pass and second-pass filter. Uses the same load_float4 /
// load_float4_predicated helpers proven in persistent_topk.cuh::histogram_2048.
// The filtered algorithm (256-bin coarse + multi-round FP32 refinement) is
// unchanged — only the global-memory scan loops are 4x fewer loads.
//
// Retained: hist2048 fallback, filtered selector for [33,91], pre-PDL init,
//           persistent_topk generic fallback.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "persistent_topk.cuh"

namespace {

constexpr size_t kWorkspaceBytes = 1 << 20;
constexpr int kPageSize = 64;
constexpr size_t kHist2048SmemBytes = 8192 * sizeof(int);
constexpr int kFilteredTopK = 2048;
constexpr int kFilteredTopKThreads = 1024;
constexpr int kFilteredRadix = 256;
constexpr int kFilteredInputBufferSize = 16 * 1024;
constexpr int kFilteredInputHalfBufferSize = kFilteredInputBufferSize / 2;
constexpr size_t kFilteredTopkSmemBytes =
    sizeof(int) * 2 * kFilteredInputBufferSize;
constexpr int kFilteredSelectorMinPages = 33;
constexpr int kFilteredSelectorMaxPages = 91;

struct FilteredTopKFloatTraits {
  using OrderedType = uint32_t;
  static constexpr int kNumRefineRounds = 4;
  static constexpr int kFirstRefineShift = 24;

  __device__ __forceinline__ static uint8_t ToCoarseKey(float x) {
    __half h = __float2half_rn(x);
    const uint16_t bits = __half_as_ushort(h);
    const uint16_t key =
        (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                        : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
  }

  __device__ __forceinline__ static OrderedType ToOrdered(float x) {
    const uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  }
};

__device__ __forceinline__ int32_t LocalToGlobalSingleRow(
    const int32_t* block_table_row,
    int32_t local_idx) {
  if (local_idx < 0) {
    return -1;
  }
  const int32_t page_idx = block_table_row[local_idx >> 6];
  return static_cast<int32_t>((page_idx << 6) | (local_idx & (kPageSize - 1)));
}

// Exact selector for the official single-long-row regime. It reuses the
// histogram_2048 refinement logic from persistent_topk.cuh, but writes final
// global token ids in place so the long-row path avoids an extra remap kernel.
__global__ void single_long_row_topk_hist2048_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ long_row_meta,
    const int32_t* __restrict__ block_table,
    int32_t* __restrict__ topk_indices,
    int batch_size,
    int block_table_stride) {
  namespace P = vllm::persistent;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  const int row = long_row_meta[0];
  if (row < 0 || row >= batch_size) {
    return;
  }

  const int32_t seq_len = long_row_meta[1];
  if (seq_len <= 0) {
    return;
  }

  int32_t* row_output = topk_indices + row * P::TopK;
  const int32_t* row_block_table = block_table + row * block_table_stride;

  if (seq_len <= P::TopK) {
    for (int idx = threadIdx.x; idx < P::TopK; idx += blockDim.x) {
      row_output[idx] =
          (idx < seq_len) ? LocalToGlobalSingleRow(row_block_table, idx) : -1;
    }
    return;
  }

  P::histogram_2048_topk(logits, row_output, seq_len);
  __syncthreads();

  for (int idx = threadIdx.x; idx < P::TopK; idx += blockDim.x) {
    row_output[idx] = LocalToGlobalSingleRow(row_block_table, row_output[idx]);
  }
}

// FlashInfer-style filtered selector specialized for the long-page tail.
// It fuses exact top-k selection with the block-table transform so the
// 82-91 page regime avoids the extra histogram/remap tail.
__global__ void single_long_row_topk_filtered_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ long_row_meta,
    const int32_t* __restrict__ block_table,
    int32_t* __restrict__ topk_indices,
    int batch_size,
    int block_table_stride) {
  using Traits = FilteredTopKFloatTraits;

  const int tx = threadIdx.x;
  alignas(128) __shared__ int s_histogram_buf[2][kFilteredRadix + 128];
  __shared__ int s_counter;
  __shared__ int s_threshold_bin_id;
  __shared__ int s_num_input[2];
  __shared__ int s_last_remain;
  alignas(128) __shared__ int s_indices[kFilteredTopK];
  extern __shared__ int s_input_idx[][kFilteredInputBufferSize];

  auto& s_histogram = s_histogram_buf[0];
  const float* score = logits;
  int remaining = kFilteredTopK;

  if (tx < kFilteredRadix + 1) {
    s_histogram[tx] = 0;
  }
  __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  const int row = long_row_meta[0];
  if (row < 0 || row >= batch_size) {
    return;
  }

  const int seq_len = long_row_meta[1];
  if (seq_len <= 0) {
    return;
  }

  int32_t* row_output = topk_indices + row * kFilteredTopK;
  const int32_t* row_block_table = block_table + row * block_table_stride;
  if (seq_len <= kFilteredTopK) {
    for (int idx = tx; idx < kFilteredTopK; idx += blockDim.x) {
      row_output[idx] =
          (idx < seq_len) ? LocalToGlobalSingleRow(row_block_table, idx) : -1;
    }
    return;
  }

  // Vectorized scan setup for coarse histogram pass and second-pass filter.
  const int n_vec = (seq_len + 3) >> 2;
  const bool score_aligned = ((reinterpret_cast<uintptr_t>(score) & 15) == 0);
  constexpr int kVecScanStride = kFilteredTopKThreads << 2;

  for (int i = tx; i < n_vec; i += kFilteredTopKThreads) {
    const int base = i << 2;
    float v0, v1, v2, v3;
    if (score_aligned && base + 3 < seq_len) {
      if (base + 2 * kVecScanStride < seq_len) {
        asm volatile("prefetch.global.L2 [%0];"
                     :
                     : "l"(score + base + 2 * kVecScanStride));
      }
      vllm::persistent::load_float4(score + base, v0, v1, v2, v3);
    } else {
      vllm::persistent::load_float4_predicated(score + base, base, seq_len,
                                                v0, v1, v2, v3);
    }
    atomicAdd(&s_histogram[Traits::ToCoarseKey(v0)], 1);
    atomicAdd(&s_histogram[Traits::ToCoarseKey(v1)], 1);
    atomicAdd(&s_histogram[Traits::ToCoarseKey(v2)], 1);
    atomicAdd(&s_histogram[Traits::ToCoarseKey(v3)], 1);
  }
  __syncthreads();

  auto run_suffix_scan = [&]() {
    int value = 0;
    const unsigned lane = tx & 31;
    const unsigned warp_id = tx >> 5;

    if (tx < kFilteredRadix) {
      value = s_histogram[tx];
      #pragma unroll
      for (int stride = 1; stride < 32; stride *= 2) {
        const int n = __shfl_down_sync(0xFFFFFFFFu, value, stride);
        if (lane + stride < 32) {
          value += n;
        }
      }
      if (lane == 0) {
        s_histogram_buf[1][warp_id] = value;
      }
    }
    __syncthreads();

    if (tx == 0) {
      for (int i = 6; i >= 0; --i) {
        s_histogram_buf[1][i] += s_histogram_buf[1][i + 1];
      }
    }
    __syncthreads();

    if (tx < kFilteredRadix) {
      if (warp_id < 7) {
        value += s_histogram_buf[1][warp_id + 1];
      }
      s_histogram[tx] = value;
    }
    __syncthreads();
  };

  run_suffix_scan();
  if (tx < kFilteredRadix &&
      s_histogram[tx] > remaining &&
      s_histogram[tx + 1] <= remaining) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const int coarse_threshold = s_threshold_bin_id;
  remaining -= s_histogram[coarse_threshold + 1];
  if (remaining == 0) {
    for (int i = tx; i < n_vec; i += kFilteredTopKThreads) {
      const int base = i << 2;
      float v0, v1, v2, v3;
      if (score_aligned && base + 3 < seq_len) {
        if (base + 2 * kVecScanStride < seq_len) {
          asm volatile("prefetch.global.L2 [%0];"
                       :
                       : "l"(score + base + 2 * kVecScanStride));
        }
        vllm::persistent::load_float4(score + base, v0, v1, v2, v3);
      } else {
        vllm::persistent::load_float4_predicated(score + base, base, seq_len,
                                                  v0, v1, v2, v3);
      }
      #pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        const float val = (sub == 0) ? v0 : (sub == 1) ? v1 : (sub == 2) ? v2 : v3;
        const int idx = base + sub;
        if (__builtin_expect(idx < seq_len &&
            static_cast<int>(Traits::ToCoarseKey(val)) > coarse_threshold, 0)) {
          const int pos = atomicAdd(&s_counter, 1);
          s_indices[pos] = idx;
        }
      }
    }
    __syncthreads();
  } else {
    if (tx < kFilteredRadix + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int i = tx; i < n_vec; i += kFilteredTopKThreads) {
      const int base = i << 2;
      float v0, v1, v2, v3;
      if (score_aligned && base + 3 < seq_len) {
        if (base + 2 * kVecScanStride < seq_len) {
          asm volatile("prefetch.global.L2 [%0];"
                       :
                       : "l"(score + base + 2 * kVecScanStride));
        }
        vllm::persistent::load_float4(score + base, v0, v1, v2, v3);
      } else {
        vllm::persistent::load_float4_predicated(score + base, base, seq_len,
                                                  v0, v1, v2, v3);
      }
      #pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        const float val = (sub == 0) ? v0 : (sub == 1) ? v1 : (sub == 2) ? v2 : v3;
        const int idx = base + sub;
        if (idx >= seq_len) continue;
        const int coarse_bin = static_cast<int>(Traits::ToCoarseKey(val));
        if (__builtin_expect(coarse_bin >= coarse_threshold, 0)) {
          if (coarse_bin > coarse_threshold) {
            const int pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          } else {
            const int pos = atomicAdd(&s_num_input[0], 1);
            const auto ordered = Traits::ToOrdered(val);
            s_input_idx[0][pos] = idx;
            s_input_idx[0][kFilteredInputHalfBufferSize + pos] =
                static_cast<int>(ordered);
            atomicAdd(&s_histogram[(ordered >> (Traits::kFirstRefineShift - 8)) & 0xFF], 1);
          }
        }
      }
    }
    __syncthreads();

    constexpr int kSkippedRefineRounds =
        (Traits::kFirstRefineShift >= 16) ? 1 : 0;
    constexpr int kEffectiveRefineRounds =
        Traits::kNumRefineRounds - kSkippedRefineRounds;

    #pragma unroll
    for (int round = 0; round < kEffectiveRefineRounds; ++round) {
      const int r_idx = round & 1;
      const int next_r_idx = r_idx ^ 1;
      const int offset =
          Traits::kFirstRefineShift - 8 * kSkippedRefineRounds - round * 8;
      const bool last_round = (round == kEffectiveRefineRounds - 1);
      const int num_input = s_num_input[r_idx];

      run_suffix_scan();
      if (tx < kFilteredRadix &&
          s_histogram[tx] > remaining &&
          s_histogram[tx + 1] <= remaining) {
        s_threshold_bin_id = tx;
        s_num_input[next_r_idx] = 0;
        s_last_remain = remaining - s_histogram[tx + 1];
      }
      __syncthreads();

      const int threshold = s_threshold_bin_id;
      remaining = s_last_remain;
      if (remaining == 0) {
        for (int i = tx; i < num_input; i += kFilteredTopKThreads) {
          const int idx = s_input_idx[r_idx][i];
          const auto ordered = static_cast<uint32_t>(
              s_input_idx[r_idx][kFilteredInputHalfBufferSize + i]);
          const int bin =
              static_cast<int>((ordered >> offset) & 0xFF);
          if (bin > threshold) {
            const int pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          }
        }
        __syncthreads();
        break;
      }

      if (last_round) {
        for (int i = tx; i < num_input; i += kFilteredTopKThreads) {
          const int idx = s_input_idx[r_idx][i];
          const auto ordered = static_cast<uint32_t>(
              s_input_idx[r_idx][kFilteredInputHalfBufferSize + i]);
          const int bin =
              static_cast<int>((ordered >> offset) & 0xFF);
          if (bin > threshold) {
            const int pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          } else if (bin == threshold) {
            const int pos = atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              s_indices[kFilteredTopK - pos] = idx;
            }
          }
        }
        __syncthreads();
      } else {
        if (tx < kFilteredRadix + 1) {
          s_histogram[tx] = 0;
        }
        __syncthreads();
        for (int i = tx; i < num_input; i += kFilteredTopKThreads) {
          const int idx = s_input_idx[r_idx][i];
          const auto ordered = static_cast<uint32_t>(
              s_input_idx[r_idx][kFilteredInputHalfBufferSize + i]);
          const int bin = static_cast<int>((ordered >> offset) & 0xFF);
          if (bin > threshold) {
            const int pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          } else if (bin == threshold) {
            const int pos = atomicAdd(&s_num_input[next_r_idx], 1);
            s_input_idx[next_r_idx][pos] = idx;
            s_input_idx[next_r_idx][kFilteredInputHalfBufferSize + pos] =
                static_cast<int>(ordered);
            atomicAdd(&s_histogram[(ordered >> (offset - 8)) & 0xFF], 1);
          }
        }
        __syncthreads();
      }
    }
  }

  for (int i = tx; i < kFilteredTopK; i += kFilteredTopKThreads) {
    row_output[i] = LocalToGlobalSingleRow(row_block_table, s_indices[i]);
  }
}

}  // namespace

cudaError_t single_long_row_topk_direct(const float* logits,
                                        const int32_t* long_row_meta,
                                        const int32_t* block_table,
                                        int32_t* topk_indices,
                                        int batch_size,
                                        int block_table_stride,
                                        cudaStream_t stream) {
  namespace P = vllm::persistent;

  if (!logits || !long_row_meta || !block_table || !topk_indices) {
    return cudaErrorInvalidValue;
  }

  cudaLaunchAttribute pdl_attr[1];
  pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl_attr[0].val.programmaticStreamSerializationAllowed = 1;

  if (block_table_stride < kFilteredSelectorMinPages ||
      block_table_stride > kFilteredSelectorMaxPages) {
    auto kernel = &single_long_row_topk_hist2048_kernel;
    cudaError_t set_err = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kHist2048SmemBytes);
    if (set_err != cudaSuccess) {
      return set_err;
    }

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(1, 1, 1);
    config.blockDim = dim3(P::kThreadsPerBlock, 1, 1);
    config.dynamicSmemBytes = kHist2048SmemBytes;
    config.stream = stream;
    config.attrs = pdl_attr;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(
        &config,
        kernel,
        logits,
        long_row_meta,
        block_table,
        topk_indices,
        batch_size,
        block_table_stride);
  }

  auto kernel = &single_long_row_topk_filtered_kernel;
  cudaError_t set_err = cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kFilteredTopkSmemBytes);
  if (set_err != cudaSuccess) {
    return set_err;
  }
  set_err = cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
  if (set_err != cudaSuccess) {
    return set_err;
  }

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(kFilteredTopKThreads, 1, 1);
  config.dynamicSmemBytes = kFilteredTopkSmemBytes;
  config.stream = stream;
  config.attrs = pdl_attr;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(
      &config,
      kernel,
      logits,
      long_row_meta,
      block_table,
      topk_indices,
      batch_size,
      block_table_stride);
}

cudaError_t persistent_topk(const float* logits,
                            const int32_t* lengths,
                            int32_t* output,
                            uint8_t* workspace,
                            size_t workspace_bytes,
                            uint32_t num_rows,
                            uint32_t stride,
                            uint32_t max_seq_len,
                            cudaStream_t stream) {
  namespace P = vllm::persistent;

  if (!logits || !lengths || !output || !workspace) {
    return cudaErrorInvalidValue;
  }
  if (num_rows == 0) {
    return cudaSuccess;
  }
  if (stride < P::TopK || workspace_bytes < kWorkspaceBytes) {
    return cudaErrorInvalidValue;
  }

  // Cache simple device properties after the first launch. They are invariant
  // for a fixed GPU and only affect the launch-shape heuristic below.
  static int num_sms = 0;
  static int max_smem_per_block = 0;
  if (num_sms == 0) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
      return err;
    }
    err = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    if (err != cudaSuccess) {
      return err;
    }
    err = cudaDeviceGetAttribute(
        &max_smem_per_block,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device);
    if (err != cudaSuccess) {
      return err;
    }
  }

  int effective_max_smem = max_smem_per_block;
  if (num_rows <= 4) {
    effective_max_smem = std::min(max_smem_per_block, static_cast<int>(P::kSmemMedium));
  } else if (num_rows <= 8) {
    effective_max_smem = std::min(max_smem_per_block, 48 * 1024);
  }

  size_t available_for_ordered =
      static_cast<size_t>(effective_max_smem) - P::kFixedSmemLarge;
  uint32_t max_chunk_elements =
      static_cast<uint32_t>(available_for_ordered / sizeof(uint32_t));

  uint32_t vec_size = 1;
  if (stride % 4 == 0) {
    vec_size = 4;
  } else if (stride % 2 == 0) {
    vec_size = 2;
  }

  // Derive a chunk size that fits in dynamic shared memory while keeping the
  // launch shape aligned to the vectorized load width expected by the kernel.
  max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;
  uint32_t min_chunk = vec_size * P::kThreadsPerBlock;
  if (max_chunk_elements < min_chunk) {
    max_chunk_elements = min_chunk;
  }

  // A "group" is the set of CTAs that cooperatively processes one row.
  // The current submission uses one row, but the upstream launcher logic still
  // computes group sizing generically.
  uint32_t ctas_per_group =
      (stride + max_chunk_elements - 1) / max_chunk_elements;
  uint32_t chunk_size = (stride + ctas_per_group - 1) / ctas_per_group;
  chunk_size = ((chunk_size + vec_size - 1) / vec_size) * vec_size;
  if (chunk_size > max_chunk_elements) {
    chunk_size = max_chunk_elements;
  }

  size_t smem_size = P::kFixedSmemLarge + chunk_size * sizeof(uint32_t);
  if (smem_size < P::kSmemMedium) {
    smem_size = P::kSmemMedium;
  }

  int occupancy = 1;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, P::persistent_topk_kernel<4>, P::kThreadsPerBlock, smem_size);
  if (err != cudaSuccess) {
    return err;
  }
  if (occupancy < 1) {
    occupancy = 1;
  }

  uint32_t max_resident_ctas = static_cast<uint32_t>(num_sms) * occupancy;
  uint32_t num_groups =
      std::min(max_resident_ctas / ctas_per_group, num_rows);
  if (num_groups == 0) {
    num_groups = 1;
  }
  uint32_t total_ctas = num_groups * ctas_per_group;

  size_t state_bytes = num_groups * sizeof(P::RadixRowState);
  if (workspace_bytes < state_bytes) {
    return cudaErrorInvalidValue;
  }

  P::PersistentTopKParams params;
  params.input = logits;
  params.output = output;
  params.lengths = const_cast<int32_t*>(lengths);
  params.num_rows = num_rows;
  params.stride = stride;
  params.chunk_size = chunk_size;
  params.row_states = reinterpret_cast<P::RadixRowState*>(workspace);
  params.ctas_per_group = ctas_per_group;
  params.max_seq_len = max_seq_len;

#define LAUNCH_PERSISTENT(VS)                                                 \
  do {                                                                        \
    auto kernel = &P::persistent_topk_kernel<VS>;                             \
    cudaError_t set_err = cudaFuncSetAttribute(                               \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);      \
    if (set_err != cudaSuccess) {                                             \
      return set_err;                                                         \
    }                                                                         \
    kernel<<<total_ctas, P::kThreadsPerBlock, smem_size, stream>>>(params);   \
  } while (0)

  if (vec_size == 4) {
    LAUNCH_PERSISTENT(4);
  } else if (vec_size == 2) {
    LAUNCH_PERSISTENT(2);
  } else {
    LAUNCH_PERSISTENT(1);
  }

#undef LAUNCH_PERSISTENT

  return cudaGetLastError();
}
