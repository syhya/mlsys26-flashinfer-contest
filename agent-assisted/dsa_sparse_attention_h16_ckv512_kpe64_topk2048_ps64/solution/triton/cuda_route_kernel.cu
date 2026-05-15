/*
 * DSA sparse attention - CUDA flash-decoding with bf16 tensor cores (WMMA),
 * cp.async K gather, 8-warp layout.
 *
 * Layout: one tile per CTA. Each CTA reads BLOCK_N KV tokens,
 * computes [H=16, D_TOT=576] x [D_TOT, BLOCK_N]^T via WMMA, softmaxes one
 * row at a time, then [H, BLOCK_N] x [BLOCK_N, DC=512] for PV.
 *
 * Split count K_SPLITS is dispatched at host time based on Nt: small Nt
 * gets more splits so we fill the machine; large Nt gets fewer splits so
 * each CTA has more work and partial-O traffic stays bounded.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <stdint.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

using namespace nvcuda;

namespace {

constexpr int H        = 16;
constexpr int DC       = 512;
constexpr int DP       = 64;
constexpr int D_TOT    = DC + DP;                 // 576
// Padded row stride for sQ / sK smem. D_TOT=576 * 2B = 1152B per row, which
// is a multiple of 128B (the smem bank cycle), so consecutive rows land on
// the same banks -> ldmatrix serializes. +8 bf16 (16B) shifts the per-row
// bank offset by 4, breaking the periodicity.
constexpr int D_TOT_PAD = D_TOT + 8;              // 584
constexpr int K_MAX    = 2048;
constexpr int THREADS  = 256;                     // 8 warps
constexpr int N_WARPS  = THREADS / 32;
constexpr int QK_WARPS = 4;                        // 4 warps x 1 N-tile each during QK
constexpr int SOFT_THREADS = 128;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

#define NEG_INF (-__builtin_huge_valf())
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ void cp_async_16(
    uint32_t smem_int_ptr, const void* gmem_ptr)
{
  // cp.async.ca (cache-all: L1 + L2). Critical for this op: real workloads
  // have 74-99.9% padding rate, so most sparse_idx entries are -1 and get
  // clamped to safe=0. The resulting K-row loads from global all hit the
  // SAME row (index 0), and L1 turns those repeated loads into hits. A
  // previous .cg variant (bypass L1) regressed min speedup 47x -> 31x on
  // the contest because it forced every repeated load back to L2.
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 16;\n"
      :: "r"(smem_int_ptr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n");
}

__device__ __forceinline__ float warp_reduce_max_8(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 4));
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
  return v;
}
__device__ __forceinline__ float warp_reduce_sum_8(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 4);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  v += __shfl_xor_sync(0xffffffff, v, 1);
  return v;
}

// Split-KV kernel. BLOCK_N_ = K_MAX / K_SPLITS_.
template <int K_SPLITS_, int BLOCK_N_, bool USE_LOG2_>
__global__ __launch_bounds__(THREADS, 2)
void attn_split_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int32_t*       __restrict__ sparse_idx,
    float*               __restrict__ partial_m,
    float*               __restrict__ partial_l,
    float*               __restrict__ partial_o,
    float sm_scale)
{
  constexpr int BLOCK_N = BLOCK_N_;
  constexpr int K_SPLITS = K_SPLITS_;
  static_assert(BLOCK_N * K_SPLITS == K_MAX, "BLOCK_N * K_SPLITS must be K_MAX");
  static_assert(BLOCK_N >= 16 && BLOCK_N % 16 == 0, "BLOCK_N must be a multiple of 16 and >=16");
  // Pad sP row stride by 8 bf16 so the per-row byte offset isn't a multiple of
  // the 128B bank period (BLOCK_N=64 hits exactly 1 period, causing ldmatrix
  // conflicts in the PFrag load of PV).
  constexpr int BLOCK_N_PAD = BLOCK_N + 8;

  const int t     = blockIdx.x;
  const int split = blockIdx.y;
  const int tid   = threadIdx.x;
  const int warp  = tid / 32;
  const int k_base = split * BLOCK_N;
  const float qk_scale = USE_LOG2_ ? (sm_scale * LOG2E) : sm_scale;

  // Sparse indices are padded with trailing -1 entries.  The fast return is
  // profitable for larger Nt where many CTAs are fully padded; on small-Nt
  // tail workloads the extra prologue load/branch costs more than it saves.
  if (gridDim.x >= 3 && sparse_idx[t * K_MAX + k_base] == -1) {
    if (tid < H) {
      size_t off = ((size_t)t * K_SPLITS + split) * H + tid;
      partial_m[off] = NEG_INF;
      partial_l[off] = 0.0f;
    }
    return;
  }

  extern __shared__ char smem[];
  __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sK   = sQ  + H * D_TOT_PAD;
  float*         sL   = reinterpret_cast<float*>(sK + BLOCK_N * D_TOT_PAD);
  __nv_bfloat16* sP   = reinterpret_cast<__nv_bfloat16*>(sL + H * BLOCK_N);
  int32_t*       sIdx = reinterpret_cast<int32_t*>(sP + H * BLOCK_N_PAD);
  float*         sM   = reinterpret_cast<float*>(sIdx + BLOCK_N);
  float*         sLn  = sM + H;

  // ---- Load Q (nope || pe) to sQ [H, D_TOT_PAD] (padding bytes are unused) ----
  {
    const float4* src_n = reinterpret_cast<const float4*>(q_nope + t * H * DC);
    #pragma unroll 4
    for (int i = tid; i < (H * DC) / 8; i += THREADS) {
      int h   = i / (DC / 8);
      int d_f = i % (DC / 8);
      reinterpret_cast<float4*>(sQ + h * D_TOT_PAD)[d_f] = src_n[i];
    }
    const float4* src_p = reinterpret_cast<const float4*>(q_pe + t * H * DP);
    #pragma unroll
    for (int i = tid; i < (H * DP) / 8; i += THREADS) {
      int h   = i / (DP / 8);
      int d_f = i % (DP / 8);
      reinterpret_cast<float4*>(sQ + h * D_TOT_PAD + DC)[d_f] = src_p[i];
    }
  }

  // ---- Load sparse indices into smem ----
  #pragma unroll
  for (int i = tid; i < BLOCK_N; i += THREADS) {
    sIdx[i] = sparse_idx[t * K_MAX + k_base + i];
  }
  __syncthreads();

  // ---- Gather K via cp.async (16B lane-wise) ----
  {
    const uint32_t sK_int = __cvta_generic_to_shared(sK);
    constexpr int KC_F4_PER_ROW = DC / 8;
    constexpr int KC_F4_TOTAL   = BLOCK_N * KC_F4_PER_ROW;
    #pragma unroll 4
    for (int i = tid; i < KC_F4_TOTAL; i += THREADS) {
      int n    = i / KC_F4_PER_ROW;
      int d_f  = i % KC_F4_PER_ROW;
      int idx  = sIdx[n];
      int safe = (idx == -1) ? 0 : idx;
      const void* src = ckv_cache + (size_t)safe * DC + d_f * 8;
      uint32_t dst = sK_int + (n * D_TOT_PAD + d_f * 8) * sizeof(__nv_bfloat16);
      cp_async_16(dst, src);
    }
    constexpr int KP_F4_PER_ROW = DP / 8;
    constexpr int KP_F4_TOTAL   = BLOCK_N * KP_F4_PER_ROW;
    #pragma unroll 2
    for (int i = tid; i < KP_F4_TOTAL; i += THREADS) {
      int n    = i / KP_F4_PER_ROW;
      int d_f  = i % KP_F4_PER_ROW;
      int idx  = sIdx[n];
      int safe = (idx == -1) ? 0 : idx;
      const void* src = kpe_cache + (size_t)safe * DP + d_f * 8;
      uint32_t dst = sK_int + (n * D_TOT_PAD + DC + d_f * 8) * sizeof(__nv_bfloat16);
      cp_async_16(dst, src);
    }
    cp_async_commit();
    cp_async_wait_all();
  }
  __syncthreads();

  // For the 128-token split path, many active tail splits only contain the
  // lower 64 tokens. Keep the 128-wide output contract but skip the upper-half
  // tensor-core work when it is entirely padding.
  const bool skip_upper_half = (BLOCK_N == 128) && (sIdx[64] == -1);

  // ---- QK matmul (first 4 warps; others idle here) ----
  // Each warp computes one WMMA N-tile (WMMA_N=16 cols of logits). 4 warps
  // cover a fixed 64-wide N-slab; we loop over N-slabs if BLOCK_N > 64.
  constexpr int N_SLAB = QK_WARPS * WMMA_N;         // 64
  static_assert(BLOCK_N % N_SLAB == 0 || BLOCK_N < N_SLAB || (BLOCK_N == 16 || BLOCK_N == 32),
                "BLOCK_N should be <=64 or multiple of 64");
  if (warp < QK_WARPS) {
    using QFrag = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                 __nv_bfloat16, wmma::row_major>;
    using KFrag = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                 __nv_bfloat16, wmma::col_major>;
    using CFrag = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    // Loop over N-slabs (each slab is 64 cols handled by 4 warps x 16 cols).
    #pragma unroll
    for (int slab_start = 0; slab_start < BLOCK_N; slab_start += N_SLAB) {
      if (slab_start == N_SLAB && skip_upper_half) continue;
      int warp_n_start = slab_start + warp * WMMA_N;
      if (warp_n_start >= BLOCK_N) continue;  // safety for BLOCK_N < 64

      CFrag c;
      wmma::fill_fragment(c, 0.0f);

      QFrag q0, q1;
      KFrag k0, k1;

      wmma::load_matrix_sync(q0, sQ + 0, D_TOT_PAD);
      wmma::load_matrix_sync(k0, sK + warp_n_start * D_TOT_PAD + 0, D_TOT_PAD);

      #pragma unroll
      for (int k = WMMA_K; k < D_TOT; k += 2 * WMMA_K) {
        wmma::load_matrix_sync(q1, sQ + k, D_TOT_PAD);
        wmma::load_matrix_sync(k1, sK + warp_n_start * D_TOT_PAD + k, D_TOT_PAD);
        wmma::mma_sync(c, q0, k0, c);

        if (k + WMMA_K < D_TOT) {
          wmma::load_matrix_sync(q0, sQ + k + WMMA_K, D_TOT_PAD);
          wmma::load_matrix_sync(k0, sK + warp_n_start * D_TOT_PAD + k + WMMA_K, D_TOT_PAD);
        }
        wmma::mma_sync(c, q1, k1, c);
      }
      if ((D_TOT / WMMA_K) & 1) {
        wmma::mma_sync(c, q0, k0, c);
      }

      #pragma unroll
      for (int i = 0; i < c.num_elements; i++) c.x[i] *= qk_scale;
      wmma::store_matrix_sync(sL + warp_n_start, c, BLOCK_N, wmma::mem_row_major);
    }
  }
  __syncthreads();

  // ---- Softmax ----
  // We use 128 threads (16 heads x 8 threads per head). Each thread covers
  // BLOCK_N/8 tokens. Requires BLOCK_N % 8 == 0 (true for our BLOCK_N values).
  constexpr int TOKENS_PER_THREAD = BLOCK_N / 8;
  if (tid < SOFT_THREADS) {
    const int h = tid / 8;
    const int s = tid & 7;
    const int n_base = s * TOKENS_PER_THREAD;

    float logits[TOKENS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < TOKENS_PER_THREAD; i++) {
      int n = n_base + i;
      float v = sL[h * BLOCK_N + n];
      if (sIdx[n] == -1) v = NEG_INF;
      logits[i] = v;
    }

    float my_max = logits[0];
    #pragma unroll
    for (int i = 1; i < TOKENS_PER_THREAD; i++) my_max = fmaxf(my_max, logits[i]);
    float row_max = warp_reduce_max_8(my_max);

    float p_vals[TOKENS_PER_THREAD];
    float my_sum = 0.0f;
    if (row_max > NEG_INF) {
      #pragma unroll
      for (int i = 0; i < TOKENS_PER_THREAD; i++) {
        p_vals[i] = USE_LOG2_ ? exp2f(logits[i] - row_max)
                              : __expf(logits[i] - row_max);
        my_sum += p_vals[i];
      }
    } else {
      #pragma unroll
      for (int i = 0; i < TOKENS_PER_THREAD; i++) p_vals[i] = 0.0f;
    }
    float row_sum = warp_reduce_sum_8(my_sum);

    #pragma unroll
    for (int i = 0; i < TOKENS_PER_THREAD; i++) {
      sP[h * BLOCK_N_PAD + n_base + i] = __float2bfloat16(p_vals[i]);
    }
    if (s == 0) {
      sM[h]  = row_max;
      sLn[h] = row_sum;
    }
  }
  __syncthreads();

  // ---- PV matmul (8 warps split DC=512 -> 64 cols per warp) ----
  {
    using PFrag  = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                  __nv_bfloat16, wmma::row_major>;
    using KcFrag = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                  __nv_bfloat16, wmma::row_major>;
    using AccFrag = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    constexpr int COLS_PER_WARP    = DC / N_WARPS;           // 64
    constexpr int N_TILES_PER_WARP = COLS_PER_WARP / WMMA_N; // 4

    float* po = partial_o + ((size_t)t * K_SPLITS + split) * (H * DC);

    #pragma unroll
    for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
      int n_start = warp * COLS_PER_WARP + nt * WMMA_N;

      AccFrag acc;
      wmma::fill_fragment(acc, 0.0f);

      #pragma unroll
      for (int k = 0; k < BLOCK_N; k += WMMA_K) {
        if (k == N_SLAB && skip_upper_half) break;
        PFrag  pfrag;
        KcFrag kfrag;
        wmma::load_matrix_sync(pfrag, sP + k, BLOCK_N_PAD);
        wmma::load_matrix_sync(kfrag, sK + k * D_TOT_PAD + n_start, D_TOT_PAD);
        wmma::mma_sync(acc, pfrag, kfrag, acc);
      }
      wmma::store_matrix_sync(po + n_start, acc, DC, wmma::mem_row_major);
    }
  }

  if (tid < H) {
    size_t off = ((size_t)t * K_SPLITS + split) * H + tid;
    partial_m[off] = sM[tid];
    partial_l[off] = sLn[tid];
  }
}

constexpr int MERGE_THREADS = 64;
static_assert(DC % MERGE_THREADS == 0, "merge dc coverage");
constexpr int DC_PER_MERGE_THREAD = DC / MERGE_THREADS;

template <int K_SPLITS_, bool USE_LOG2_>
__global__ __launch_bounds__(MERGE_THREADS, 4)
void attn_merge_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_o,
    __nv_bfloat16* __restrict__ out,
    float*          __restrict__ lse)
{
  constexpr int K_SPLITS = K_SPLITS_;
  const int t   = blockIdx.x;
  const int h   = blockIdx.y;
  const int tid = threadIdx.x;

  __shared__ float sM[K_SPLITS];
  __shared__ float sL[K_SPLITS];
  __shared__ float sW[K_SPLITS];
  __shared__ float sMg;
  __shared__ float sLg;

  #pragma unroll
  for (int s = tid; s < K_SPLITS; s += MERGE_THREADS) {
    size_t off = ((size_t)t * K_SPLITS + s) * H + h;
    sM[s] = partial_m[off];
    sL[s] = partial_l[off];
  }
  __syncthreads();

  // Warp-parallel max + weighted-sum reduction over K_SPLITS splits.
  // Replaces the 1-thread serial loop that previously ran for ~0.3-0.5us
  // per merge CTA and dominated merge latency on small-Nt workloads.
  if (tid < 32) {
    float m_local = NEG_INF;
    #pragma unroll
    for (int s = tid; s < K_SPLITS; s += 32) {
      m_local = fmaxf(m_local, sM[s]);
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
      m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffff, m_local, off));
    }
    const float mg_v = m_local;

    float l_local = 0.0f;
    #pragma unroll
    for (int s = tid; s < K_SPLITS; s += 32) {
      float m_s = sM[s];
      float w   = (m_s == NEG_INF) ? 0.0f
                                   : (USE_LOG2_ ? exp2f(m_s - mg_v)
                                                : __expf(m_s - mg_v));
      sW[s]     = w;
      l_local  += w * sL[s];
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
      l_local += __shfl_xor_sync(0xffffffff, l_local, off);
    }
    if (tid == 0) {
      sMg = mg_v;
      sLg = l_local;
    }
  }
  __syncthreads();

  const float mg = sMg;
  const float lg = sLg;
  const float inv_l = (lg > 0.0f) ? (1.0f / lg) : 0.0f;

  const int d_base = tid * DC_PER_MERGE_THREAD;
  float o_acc[DC_PER_MERGE_THREAD];
  #pragma unroll
  for (int i = 0; i < DC_PER_MERGE_THREAD; i++) o_acc[i] = 0.0f;

  // DC_PER_MERGE_THREAD = 8 fp32 values per thread; read as 2x float4 to
  // collapse 8 fp32 LDs into 2 LDG.E.128 per split. partial_o rows are
  // H*DC = 8192 fp32 = 32 KB apart, so each thread's d_base*4B start is
  // 16-byte aligned whenever DC_PER_MERGE_THREAD is a multiple of 4.
  static_assert(DC_PER_MERGE_THREAD % 4 == 0, "need DC/MERGE_THREADS multiple of 4");
  constexpr int VEC = 4;
  constexpr int NVEC = DC_PER_MERGE_THREAD / VEC;

  #pragma unroll 8
  for (int s = 0; s < K_SPLITS; s++) {
    float w = sW[s];
    if (w == 0.0f) continue;
    const float* po =
        partial_o + ((size_t)t * K_SPLITS + s) * (H * DC) + h * DC + d_base;
    const float4* po4 = reinterpret_cast<const float4*>(po);
    #pragma unroll
    for (int v = 0; v < NVEC; v++) {
      float4 p4 = po4[v];
      o_acc[v * VEC + 0] += w * p4.x;
      o_acc[v * VEC + 1] += w * p4.y;
      o_acc[v * VEC + 2] += w * p4.z;
      o_acc[v * VEC + 3] += w * p4.w;
    }
  }

  __nv_bfloat16* out_row = out + (size_t)t * H * DC + h * DC + d_base;
  // Unified branch: scale=0 for all-invalid -> writes 0; else scale=inv_l.
  // When mg==NEG_INF and/or lg==0, o_acc is already all-zero anyway (every
  // per-split weight hits the w==0 `continue`), so multiplying by 0 is a no-op.
  const float scale = (mg == NEG_INF || lg == 0.0f) ? 0.0f : inv_l;
  // Pack 8 bf16 outputs into one 16-byte store. Using a union so the
  // __nv_bfloat162 lanes live in the same registers as the uint4 we emit.
  union PackedOut {
    __nv_bfloat162 b2[DC_PER_MERGE_THREAD / 2];
    uint4 v;
  } packed;
  #pragma unroll
  for (int i = 0; i < DC_PER_MERGE_THREAD / 2; i++) {
    packed.b2[i] = __float22bfloat162_rn(
        make_float2(o_acc[2 * i + 0] * scale, o_acc[2 * i + 1] * scale));
  }
  *reinterpret_cast<uint4*>(out_row) = packed.v;

  if (tid == 0) {
    float lse_v = (mg == NEG_INF || lg == 0.0f)
                      ? NEG_INF
                      : (USE_LOG2_ ? (mg + __log2f(lg))
                                   : ((mg + __logf(lg)) * LOG2E));
    lse[(size_t)t * H + h] = lse_v;
  }
}

struct Workspace {
  float* d_m = nullptr;
  float* d_l = nullptr;
  float* d_o = nullptr;
  int    Nt_cap = 0;       // capacity in tokens
  int    splits_cap = 0;   // capacity in splits
};
static Workspace g_ws;

static void ensure_workspace(int Nt, int splits) {
  if (Nt <= g_ws.Nt_cap && splits <= g_ws.splits_cap) return;
  if (g_ws.d_m) cudaFree(g_ws.d_m);
  if (g_ws.d_l) cudaFree(g_ws.d_l);
  if (g_ws.d_o) cudaFree(g_ws.d_o);
  int new_Nt  = Nt > g_ws.Nt_cap ? Nt : g_ws.Nt_cap;
  int new_spl = splits > g_ws.splits_cap ? splits : g_ws.splits_cap;
  size_t ml = (size_t)new_Nt * new_spl * H * sizeof(float);
  size_t od = (size_t)new_Nt * new_spl * H * DC * sizeof(float);
  cudaMalloc(&g_ws.d_m, ml);
  cudaMalloc(&g_ws.d_l, ml);
  cudaMalloc(&g_ws.d_o, od);
  g_ws.Nt_cap     = new_Nt;
  g_ws.splits_cap = new_spl;
}

static int g_shmem_attr_set_mask = 0;  // bit per splits choice

template <int K_SPLITS_, int BLOCK_N_>
static int smem_bytes_for() {
  constexpr int BLOCK_N_PAD = BLOCK_N_ + 8;
  return
      H * D_TOT_PAD * (int)sizeof(__nv_bfloat16)        // sQ (row-padded)
    + BLOCK_N_ * D_TOT_PAD * (int)sizeof(__nv_bfloat16) // sK (row-padded)
    + H * BLOCK_N_ * (int)sizeof(float)                 // sL
    + H * BLOCK_N_PAD * (int)sizeof(__nv_bfloat16)      // sP (row-padded)
    + BLOCK_N_ * (int)sizeof(int32_t)                   // sIdx
    + H * (int)sizeof(float)                            // sM
    + H * (int)sizeof(float)                            // sLn
    + 32;                                               // padding
}

template <int K_SPLITS_, int BLOCK_N_, bool USE_LOG2_, int BIT>
static void dispatch_once(
    int Nt, dim3 grid_split, cudaStream_t stream,
    const __nv_bfloat16* qn, const __nv_bfloat16* qp,
    const __nv_bfloat16* ckv, const __nv_bfloat16* kpe,
    const int32_t* si, float sm_scale,
    __nv_bfloat16* out, float* lse_out)
{
  const int smem = smem_bytes_for<K_SPLITS_, BLOCK_N_>();
  if (!(g_shmem_attr_set_mask & (1 << BIT))) {
    cudaFuncSetAttribute(
        attn_split_kernel<K_SPLITS_, BLOCK_N_, USE_LOG2_>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem);
    g_shmem_attr_set_mask |= (1 << BIT);
  }

  attn_split_kernel<K_SPLITS_, BLOCK_N_, USE_LOG2_><<<grid_split, THREADS, smem, stream>>>(
      qn, qp, ckv, kpe, si,
      g_ws.d_m, g_ws.d_l, g_ws.d_o,
      sm_scale);

  dim3 grid_merge(Nt, H);
  attn_merge_kernel<K_SPLITS_, USE_LOG2_><<<grid_merge, MERGE_THREADS, 0, stream>>>(
      g_ws.d_m, g_ws.d_l, g_ws.d_o, out, lse_out);
}

}  // namespace

using tvm::ffi::TensorView;

static void run_impl_chosen(
    TensorView q_nope,
    TensorView q_pe,
    TensorView ckv_cache,
    TensorView kpe_cache,
    TensorView sparse_indices,
    double sm_scale_d,
    TensorView output,
    TensorView lse,
    int chosen_splits)
{
  const int Nt = (int)q_nope.shape()[0];
  if (Nt == 0) return;
  const float sm_scale = (float)sm_scale_d;

  DLDevice dev = q_nope.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  ensure_workspace(Nt, chosen_splits);

  dim3 grid_split(Nt, chosen_splits);

  auto* qn  = static_cast<const __nv_bfloat16*>(q_nope.data_ptr());
  auto* qp  = static_cast<const __nv_bfloat16*>(q_pe.data_ptr());
  auto* ckv = static_cast<const __nv_bfloat16*>(ckv_cache.data_ptr());
  auto* kpe = static_cast<const __nv_bfloat16*>(kpe_cache.data_ptr());
  auto* si  = static_cast<const int32_t*>(sparse_indices.data_ptr());
  auto* out = static_cast<__nv_bfloat16*>(output.data_ptr());
  auto* lse_out = static_cast<float*>(lse.data_ptr());

  if (chosen_splits == 64) {
    dispatch_once<64, 32, false, 0>(Nt, grid_split, stream, qn, qp, ckv, kpe, si, sm_scale, out, lse_out);
  } else if (chosen_splits == 16) {
    dispatch_once<16, 128, true, 2>(Nt, grid_split, stream, qn, qp, ckv, kpe, si, sm_scale, out, lse_out);
  } else {
    dispatch_once<32, 64, false, 1>(Nt, grid_split, stream, qn, qp, ckv, kpe, si, sm_scale, out, lse_out);
  }
}

static void run_impl(
    TensorView q_nope,
    TensorView q_pe,
    TensorView ckv_cache,
    TensorView kpe_cache,
    TensorView sparse_indices,
    double sm_scale_d,
    TensorView output,
    TensorView lse)
{
  const int Nt = (int)q_nope.shape()[0];
  const int chosen_splits = (Nt <= 2) ? 64 : 32;
  run_impl_chosen(
      q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
      sm_scale_d, output, lse, chosen_splits);
}

static void run_impl_splits16(
    TensorView q_nope,
    TensorView q_pe,
    TensorView ckv_cache,
    TensorView kpe_cache,
    TensorView sparse_indices,
    double sm_scale_d,
    TensorView output,
    TensorView lse)
{
  const int Nt = (int)q_nope.shape()[0];
  const int chosen_splits = (Nt <= 2) ? 64 : 16;
  run_impl_chosen(
      q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
      sm_scale_d, output, lse, chosen_splits);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda, run_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda_splits16, run_impl_splits16);
