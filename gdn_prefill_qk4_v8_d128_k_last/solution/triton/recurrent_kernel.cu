/*
 * CUDA sidecar for the active GDN prefill submission path
 * =======================================================
 *
 * Active Python route today:
 *   tiny measured-regression shapes -> kernel_cuda_prefill_trace_v1
 *   all other shapes                -> compute_gates -> CuTe chunked kernel
 *
 * Only two TVM FFI exports remain live:
 * 1. kernel_cuda_prefill_trace_v1
 *    Recovered from submission-v25 as a narrow fallback for the very small
 *    shapes that regressed after removing the historical short-path dispatch.
 * 2. compute_gates
 *    Scalar fused gate helper that materializes log_gate and beta for the
 *    PR #3001-derived Blackwell chunked path.
 *
 * The vec2 long-band gate helper is intentionally not restored.
 */

#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace {

constexpr int kHeadSize = 128;
constexpr int kNumQHeads = 4;
constexpr int kNumKHeads = 4;
constexpr int kNumVHeads = 8;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = 4;
constexpr int kThreads = kWarpSize * kNumWarps;   // 128
constexpr int kKVec = kHeadSize / kWarpSize;      // 4

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ void warp_reduce_sum_pair(float& a, float& b) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    a += __shfl_xor_sync(0xffffffff, a, offset);
    b += __shfl_xor_sync(0xffffffff, b, offset);
  }
}

__device__ __forceinline__ float sigmoid_fast(float x) {
  return __frcp_rn(1.0f + __expf(-x));
}

__device__ __forceinline__ float softplus_fast(float x) {
  if (x > 20.0f) {
    return x;
  }
  return __logf(1.0f + __expf(x));
}

__global__ __launch_bounds__(kThreads, 4) void gdn_prefill_trace_kernel_rpw1(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a_gate,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_gate,
    const int64_t* __restrict__ cu_seqlens,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state) {
  constexpr int kRowsPerWarp = 1;
  constexpr int kRowsPerBlock = kNumWarps * kRowsPerWarp;

  const int row_tile = blockIdx.x;
  const int seq_head = blockIdx.y;
  const int seq_idx = seq_head / kNumVHeads;
  const int hv = seq_head % kNumVHeads;
  const int qk_head = hv >> 1;

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int k_base = lane * kKVec;
  const int row_base = row_tile * kRowsPerBlock;
  const int v_idx = row_base + warp_id;

  const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
  const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
  if (seq_start >= seq_end) {
    return;
  }

  const float A_exp = __expf(A_log[hv]);
  const float dt_bias_val = dt_bias[hv];

  float h[kKVec];
  {
    const int64_t state_offset =
        ((((int64_t)seq_idx * kNumVHeads + hv) * kHeadSize + v_idx) * kHeadSize) + k_base;
    if (state != nullptr) {
      const float4 packed = *reinterpret_cast<const float4*>(state + state_offset);
      h[0] = packed.x;
      h[1] = packed.y;
      h[2] = packed.z;
      h[3] = packed.w;
    } else {
      h[0] = h[1] = h[2] = h[3] = 0.0f;
    }
  }

  const int64_t qk_stride = (int64_t)kNumQHeads * kHeadSize;
  int64_t q_offset = (((int64_t)seq_start * kNumQHeads + qk_head) * kHeadSize) + k_base;
  int64_t k_offset = (((int64_t)seq_start * kNumKHeads + qk_head) * kHeadSize) + k_base;
  uint2 next_q_raw = *reinterpret_cast<const uint2*>(q + q_offset);
  uint2 next_k_raw = *reinterpret_cast<const uint2*>(k + k_offset);
  __nv_bfloat16 next_a_gate = a_gate[seq_start * kNumVHeads + hv];
  __nv_bfloat16 next_b_gate = b_gate[seq_start * kNumVHeads + hv];

  for (int t = seq_start; t < seq_end; ++t) {
    const uint2 cur_q_raw = next_q_raw;
    const uint2 cur_k_raw = next_k_raw;
    const __nv_bfloat16 cur_a_gate = next_a_gate;
    const __nv_bfloat16 cur_b_gate = next_b_gate;

    float q_reg[kKVec], k_reg[kKVec];
    {
      const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&cur_q_raw.x);
      const __nv_bfloat162 q23 = *reinterpret_cast<const __nv_bfloat162*>(&cur_q_raw.y);
      const __nv_bfloat162 k01 = *reinterpret_cast<const __nv_bfloat162*>(&cur_k_raw.x);
      const __nv_bfloat162 k23 = *reinterpret_cast<const __nv_bfloat162*>(&cur_k_raw.y);
      q_reg[0] = __bfloat162float(q01.x);
      q_reg[1] = __bfloat162float(q01.y);
      q_reg[2] = __bfloat162float(q23.x);
      q_reg[3] = __bfloat162float(q23.y);
      k_reg[0] = __bfloat162float(k01.x);
      k_reg[1] = __bfloat162float(k01.y);
      k_reg[2] = __bfloat162float(k23.x);
      k_reg[3] = __bfloat162float(k23.y);
    }

    float qk_partial = 0.0f;
#pragma unroll
    for (int i = 0; i < kKVec; ++i) {
      qk_partial = fmaf(q_reg[i], k_reg[i], qk_partial);
    }
    const float qk_dot = warp_reduce_sum(qk_partial);

    const int64_t v_base = ((int64_t)t * kNumVHeads + hv) * kHeadSize;
    const float v_val = __bfloat162float(v[v_base + v_idx]);
    const float x = __bfloat162float(cur_a_gate) + dt_bias_val;
    const float g = __expf(-A_exp * softplus_fast(x));
    const float beta = sigmoid_fast(__bfloat162float(cur_b_gate));
    const float scale_qk = scale * qk_dot;

    if (t + 1 < seq_end) {
      q_offset += qk_stride;
      k_offset += qk_stride;
      next_q_raw = *reinterpret_cast<const uint2*>(q + q_offset);
      next_k_raw = *reinterpret_cast<const uint2*>(k + k_offset);
      next_a_gate = a_gate[(t + 1) * kNumVHeads + hv];
      next_b_gate = b_gate[(t + 1) * kNumVHeads + hv];
    }

    float kh = 0.0f, qh = 0.0f;
#pragma unroll
    for (int i = 0; i < kKVec; ++i) {
      const float h_scaled = h[i] * g;
      h[i] = h_scaled;
      kh = fmaf(k_reg[i], h_scaled, kh);
      qh = fmaf(q_reg[i], h_scaled, qh);
    }
    warp_reduce_sum_pair(kh, qh);
    const float dv = beta * (v_val - kh);
#pragma unroll
    for (int i = 0; i < kKVec; ++i) {
      h[i] = fmaf(dv, k_reg[i], h[i]);
    }
    if (lane == 0) {
      output[v_base + v_idx] = __float2bfloat16(fmaf(dv, scale_qk, qh * scale));
    }
  }

  {
    const int64_t state_offset =
        ((((int64_t)seq_idx * kNumVHeads + hv) * kHeadSize + v_idx) * kHeadSize) + k_base;
    float4 packed;
    packed.x = h[0];
    packed.y = h[1];
    packed.z = h[2];
    packed.w = h[3];
    __stcs(reinterpret_cast<float4*>(new_state + state_offset), packed);
  }
}

__global__ __launch_bounds__(kThreads, 4) void gdn_prefill_trace_kernel_rpw2(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a_gate,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_gate,
    const int64_t* __restrict__ cu_seqlens,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state) {
  constexpr int kRowsPerWarp = 2;
  constexpr int kRowsPerBlock = kNumWarps * kRowsPerWarp;

  const int row_tile = blockIdx.x;
  const int seq_head = blockIdx.y;
  const int seq_idx = seq_head / kNumVHeads;
  const int hv = seq_head % kNumVHeads;
  const int qk_head = hv >> 1;

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int k_base = lane * kKVec;
  const int row_base = row_tile * kRowsPerBlock;

  const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
  const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
  if (seq_start >= seq_end) {
    return;
  }

  const float A_exp = __expf(A_log[hv]);
  const float dt_bias_val = dt_bias[hv];

  const int v_idx0 = row_base + warp_id * kRowsPerWarp;
  const int v_idx1 = v_idx0 + 1;

  float h0[kKVec], h1[kKVec];
  {
    const int64_t base = (((int64_t)seq_idx * kNumVHeads + hv) * kHeadSize) * kHeadSize;
    if (state != nullptr) {
      const float4 p0 = *reinterpret_cast<const float4*>(
          state + base + (int64_t)v_idx0 * kHeadSize + k_base);
      h0[0] = p0.x;
      h0[1] = p0.y;
      h0[2] = p0.z;
      h0[3] = p0.w;
      const float4 p1 = *reinterpret_cast<const float4*>(
          state + base + (int64_t)v_idx1 * kHeadSize + k_base);
      h1[0] = p1.x;
      h1[1] = p1.y;
      h1[2] = p1.z;
      h1[3] = p1.w;
    } else {
      h0[0] = h0[1] = h0[2] = h0[3] = 0.0f;
      h1[0] = h1[1] = h1[2] = h1[3] = 0.0f;
    }
  }

  const int64_t qk_stride = (int64_t)kNumQHeads * kHeadSize;
  int64_t q_offset = (((int64_t)seq_start * kNumQHeads + qk_head) * kHeadSize) + k_base;
  int64_t k_offset = (((int64_t)seq_start * kNumKHeads + qk_head) * kHeadSize) + k_base;
  uint2 next_q_raw = *reinterpret_cast<const uint2*>(q + q_offset);
  uint2 next_k_raw = *reinterpret_cast<const uint2*>(k + k_offset);
  __nv_bfloat16 next_a_gate = a_gate[seq_start * kNumVHeads + hv];
  __nv_bfloat16 next_b_gate = b_gate[seq_start * kNumVHeads + hv];

  for (int t = seq_start; t < seq_end; ++t) {
    const uint2 cur_q_raw = next_q_raw;
    const uint2 cur_k_raw = next_k_raw;
    const __nv_bfloat16 cur_a_gate = next_a_gate;
    const __nv_bfloat16 cur_b_gate = next_b_gate;

    float q_reg[kKVec], k_reg[kKVec];
    {
      const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&cur_q_raw.x);
      const __nv_bfloat162 q23 = *reinterpret_cast<const __nv_bfloat162*>(&cur_q_raw.y);
      const __nv_bfloat162 k01 = *reinterpret_cast<const __nv_bfloat162*>(&cur_k_raw.x);
      const __nv_bfloat162 k23 = *reinterpret_cast<const __nv_bfloat162*>(&cur_k_raw.y);
      q_reg[0] = __bfloat162float(q01.x);
      q_reg[1] = __bfloat162float(q01.y);
      q_reg[2] = __bfloat162float(q23.x);
      q_reg[3] = __bfloat162float(q23.y);
      k_reg[0] = __bfloat162float(k01.x);
      k_reg[1] = __bfloat162float(k01.y);
      k_reg[2] = __bfloat162float(k23.x);
      k_reg[3] = __bfloat162float(k23.y);
    }

    float qk_partial = 0.0f;
#pragma unroll
    for (int i = 0; i < kKVec; ++i) {
      qk_partial = fmaf(q_reg[i], k_reg[i], qk_partial);
    }
    const float qk_dot = warp_reduce_sum(qk_partial);

    const int64_t v_base = ((int64_t)t * kNumVHeads + hv) * kHeadSize;
    const float v_val0 = __bfloat162float(v[v_base + v_idx0]);
    const float v_val1 = __bfloat162float(v[v_base + v_idx1]);
    const float x = __bfloat162float(cur_a_gate) + dt_bias_val;
    const float g = __expf(-A_exp * softplus_fast(x));
    const float beta = sigmoid_fast(__bfloat162float(cur_b_gate));
    const float scale_qk = scale * qk_dot;

    if (t + 1 < seq_end) {
      q_offset += qk_stride;
      k_offset += qk_stride;
      next_q_raw = *reinterpret_cast<const uint2*>(q + q_offset);
      next_k_raw = *reinterpret_cast<const uint2*>(k + k_offset);
      next_a_gate = a_gate[(t + 1) * kNumVHeads + hv];
      next_b_gate = b_gate[(t + 1) * kNumVHeads + hv];
    }

    float kh0 = 0.0f, qh0 = 0.0f;
#pragma unroll
    for (int i = 0; i < kKVec; ++i) {
      const float h_scaled = h0[i] * g;
      h0[i] = h_scaled;
      kh0 = fmaf(k_reg[i], h_scaled, kh0);
      qh0 = fmaf(q_reg[i], h_scaled, qh0);
    }
    warp_reduce_sum_pair(kh0, qh0);
    const float dv0 = beta * (v_val0 - kh0);
#pragma unroll
    for (int i = 0; i < kKVec; ++i) {
      h0[i] = fmaf(dv0, k_reg[i], h0[i]);
    }

    float kh1 = 0.0f, qh1 = 0.0f;
#pragma unroll
    for (int i = 0; i < kKVec; ++i) {
      const float h_scaled = h1[i] * g;
      h1[i] = h_scaled;
      kh1 = fmaf(k_reg[i], h_scaled, kh1);
      qh1 = fmaf(q_reg[i], h_scaled, qh1);
    }
    warp_reduce_sum_pair(kh1, qh1);
    const float dv1 = beta * (v_val1 - kh1);
#pragma unroll
    for (int i = 0; i < kKVec; ++i) {
      h1[i] = fmaf(dv1, k_reg[i], h1[i]);
    }

    if (lane == 0) {
      output[v_base + v_idx0] = __float2bfloat16(fmaf(dv0, scale_qk, qh0 * scale));
      output[v_base + v_idx1] = __float2bfloat16(fmaf(dv1, scale_qk, qh1 * scale));
    }
  }

  {
    const int64_t base = (((int64_t)seq_idx * kNumVHeads + hv) * kHeadSize) * kHeadSize;
    float4 p0, p1;
    p0.x = h0[0];
    p0.y = h0[1];
    p0.z = h0[2];
    p0.w = h0[3];
    p1.x = h1[0];
    p1.y = h1[1];
    p1.z = h1[2];
    p1.w = h1[3];
    __stcs(reinterpret_cast<float4*>(
               new_state + base + (int64_t)v_idx0 * kHeadSize + k_base),
           p0);
    __stcs(reinterpret_cast<float4*>(
               new_state + base + (int64_t)v_idx1 * kHeadSize + k_base),
           p1);
  }
}

void KernelCudaPrefillTraceV1(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView A_log,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias,
    tvm::ffi::TensorView b_gate,
    tvm::ffi::TensorView cu_seqlens,
    double scale,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView new_state) {
  const int num_seqs = static_cast<int>(cu_seqlens.size(0)) - 1;
  DLDevice dev = q.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  const int grid_y = num_seqs * kNumVHeads;

  if (num_seqs >= 3) {
    constexpr int kRowTiles2 = 16;
    dim3 grid(kRowTiles2, grid_y);
    dim3 block(kThreads);
    gdn_prefill_trace_kernel_rpw2<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const __nv_bfloat16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const __nv_bfloat16*>(b_gate.data_ptr()),
        static_cast<const int64_t*>(cu_seqlens.data_ptr()),
        static_cast<float>(scale),
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()));
  } else {
    constexpr int kRowTiles1 = 32;
    dim3 grid(kRowTiles1, grid_y);
    dim3 block(kThreads);
    gdn_prefill_trace_kernel_rpw1<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const __nv_bfloat16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const __nv_bfloat16*>(b_gate.data_ptr()),
        static_cast<const int64_t*>(cu_seqlens.data_ptr()),
        static_cast<float>(scale),
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()));
  }
}

__global__ void fused_gate_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ log_gate_out,
    float* __restrict__ beta_out,
    int total_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }

  const int h = idx % kNumVHeads;
  const float a_val = __bfloat162float(a[idx]);
  const float b_val = __bfloat162float(b[idx]);
  const float x = a_val + dt_bias[h];

  log_gate_out[idx] = -__expf(A_log[h]) * softplus_fast(x);
  beta_out[idx] = sigmoid_fast(b_val);
}

void ComputeGates(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView A_log,
    tvm::ffi::TensorView dt_bias,
    tvm::ffi::TensorView log_gate_out,
    tvm::ffi::TensorView beta_out) {
  const int total = static_cast<int>(a.shape()[0]) * kNumVHeads;
  DLDevice dev = a.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  constexpr int kBlock = 256;
  const int grid = (total + kBlock - 1) / kBlock;
  fused_gate_kernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(a.data_ptr()),
      static_cast<const __nv_bfloat16*>(b.data_ptr()),
      static_cast<const float*>(A_log.data_ptr()),
      static_cast<const float*>(dt_bias.data_ptr()),
      static_cast<float*>(log_gate_out.data_ptr()),
      static_cast<float*>(beta_out.data_ptr()),
      total);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda_prefill_trace_v1, KernelCudaPrefillTraceV1);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(compute_gates, ComputeGates);
