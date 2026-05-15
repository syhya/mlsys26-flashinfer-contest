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
    T,
    stride_am, stride_ak,
    stride_ascale_k, stride_ascale_m,
    stride_we, stride_wn, stride_wk,
    stride_wscale_e, stride_wscale_n, stride_wscale_k,
    stride_outm, stride_outn,
    H: tl.constexpr, N_dim: tl.constexpr,
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
    
    n_offs = (n_idx * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    k_offs = tl.arange(0, BLOCK_K).to(tl.int64)
    
    A_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
    W_ptrs = W_ptr + e_idx.to(tl.int64) * stride_we + n_offs[None, :] * stride_wn + k_offs[:, None] * stride_wk
    
    scale_n_offs = n_offs // 128
    W_scale_ptrs = W_scale_ptr + e_idx.to(tl.int64) * stride_wscale_e + scale_n_offs[None, :] * stride_wscale_n
    A_scale_ptrs = A_scale_ptr + token_ids[:, None] * stride_ascale_m

    for k in range(0, H, BLOCK_K):
        k_idx = k // BLOCK_K
        
        a = tl.load(A_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(W_ptrs)
        
        a_scale = tl.load(A_scale_ptrs + k_idx * stride_ascale_k, mask=m_mask[:, None], other=0.0)
        w_scale = tl.load(W_scale_ptrs + k_idx * stride_wscale_k)
        
        # FP8 Tensor Core dot
        dot_res = tl.dot(a, w, out_dtype=tl.float32)
        dot_res = dot_res * a_scale * w_scale
        acc += dot_res
        
        A_ptrs += BLOCK_K * stride_ak
        W_ptrs += BLOCK_K * stride_wk

    Out_ptrs = Out_ptr + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm + n_offs[None, :] * stride_outn
    tl.store(Out_ptrs, acc, mask=m_mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['T']
)
@triton.jit
def grouped_gemm2_kernel(
    G1_ptr, 
    W2_ptr, W2_scale_ptr,
    Routing_weights_ptr,
    Out_ptr,
    expert_offsets_ptr,
    T,