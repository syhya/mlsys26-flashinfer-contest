An analysis of the parent solution reveals that while the custom Triton Grouped GEMMs successfully eliminated CPU-GPU synchronization and achieved impressive speedups, they suffered from algorithmic edge cases related to pointer arithmetic and floating-point logic, resulting in a severe correctness failure (max absolute error of 8192.0). 

Specifically, two major issues were addressed:
1. **Int32 Pointer Overflow**: The indexing arithmetic for large weight tensors with strides exceeding millions can easily overflow 32-bit integer limits, corrupting memory access. We resolve this algorithmically by enforcing strict `tl.int64` casting on all pointers and strides before multiplication.
2. **Missing Tensor Core Utilization for GEMM2**: The intermediate SwiGLU operation produced `FP32` activations that were directly fed into the second GEMM. This bypassed `FP8` Tensor Cores and relied on slower/less precise TF32 operations. We now extract SwiGLU into an optimized PyTorch routine, dynamically quantize the results back to `float8_e4m3fn` (with 128-element block scales), and utilize `FP8` Tensor Cores for both matrix multiplications.

Here is the complete and correct child solution:

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['T']
)
@triton.jit
def grouped_gemm1_kernel(
    A_ptr, A_scale_ptr,
    W_ptr, W_scale_ptr,
    Out_ptr,
    sorted_tokens_ptr, expert_offsets_ptr,
    T, H, N_dim,
    stride_am, stride_ak,
    stride_ascale_k, stride_ascale_m,
    stride_we, stride_wn, stride_wk,
    stride_wscale_e, stride_wscale_n, stride_wscale_k,
    stride_outm, stride_outn,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    n_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    e_idx = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + e_idx)
    end_idx   = tl.load(expert_offsets_ptr + e_idx + 1)
    Tk = end_idx - start_idx

    if m_idx * BLOCK_M >= Tk:
        return

    m_offs = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < Tk

    token_idx_ptr = sorted_tokens_ptr + start_idx + m_offs
    token_ids = tl.load(token_idx_ptr, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_offs = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    
    A_ptrs = A_ptr + token_ids[:, None] * stride_am.to(tl.int64) + tl.arange(0, BLOCK_K)[None, :].to(tl.int64) * stride_ak.to(tl.int64)
    W_ptrs = W_ptr + e_idx.to(tl.int64) * stride_we.to(tl.int64) + n_offs[None, :].to(tl.int64) * stride_wn.to(tl.int64) + tl.arange(0, BLOCK_K)[:, None].to(tl.int64) * stride_wk.to(tl.int64)
    
    A_scale_ptrs = A_scale_ptr + token_ids[:, None] * stride_ascale_m.to(tl.int64)
    scale_n_offs = n_offs // 128
    W_scale_ptrs = W_scale_ptr + e_idx.to(tl.int64) * stride_wscale_e.to(tl.int64) + scale_n_offs[None, :].to(tl.int64) * stride_wscale_n.to(tl.int64)

    for k in range(0, H, BLOCK_K):
        k_idx = k // BLOCK_K
        a = tl.load(A_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(W_ptrs)
        
        a_scale = tl.load(A_scale_ptrs + k_idx * stride_ascale_k.to(tl.int64), mask=m_mask[:, None], other=0.0)
        w_scale = tl.load(W_scale_ptrs + k_idx * stride_wscale_k.to(tl.int64))
        
        dot_res = tl.dot(