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
    
    # Fully cast all pointer arithmetic components to int64 to prevent overflow
    A_ptrs = A_ptr + (token_ids[:, None] * stride_am.to(tl.int64) + tl.arange(0, BLOCK_K)[None, :].to(tl.int64) * stride_ak.to(tl.int64))
    W_ptrs = W_ptr + e_idx.to(tl.int64) * stride_we.to(tl.int64) + (n_offs[None, :].to(tl.int64) * stride_wn.to(tl.int64) + tl.arange(0, BLOCK_K)[:, None].to(tl.int64) * stride_wk.to(tl.int64))
    
    A_scale_ptrs = A_scale_ptr + token_ids[:, None] * stride_ascale_m.to(tl.int64)
    scale_n_offs = n_offs // 128
    W_scale_ptrs = W_scale_ptr + e_idx.to(tl.int64) * stride_wscale_e.to(tl.int64) + scale_n_offs[None, :].to(tl.int64) * stride_wscale_n.to(tl.int64)

    for k in range(0, H, BLOCK_K):
        k_idx = k // BLOCK_K
        a = tl.load(A_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(W_ptrs)
        
        a_scale = tl.load(A_scale_ptrs + k_idx * stride_ascale_k.to(tl.int64), mask=m_mask[:, None], other=0.0)
        w_scale = tl.load(W_scale_ptrs + k_idx * stride_wscale_k.to(tl.int64))
        
        dot_res = tl.dot(a, w, out_dtype=tl.float32)
        dot_res = dot_res * a_scale * w_scale
        acc += dot_res
        
        A_ptrs += BLOCK_K * stride_ak.to(tl.int64)
        W_ptrs += BLOCK_K * stride_wk.to(tl.int64)

    Out_ptrs = Out_ptr + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm.to(tl.int64) + n_offs[None, :].to(tl.int64) * stride_outn.to(tl.int64)
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
def grouped_gemm2_kernel_fp8(
    C_ptr, C_scale_ptr,
    W2_ptr, W2_scale_ptr,
    Routing_weights_ptr,
    Out_ptr,
    expert_offsets_ptr,
    T, I, H,
    stride_cm, stride_ck,
    stride_cscale_m, stride_cscale_k,
    stride_w2e, stride_w2n, stride_w2k,
    stride_w2scale_e, stride_w2scale_n, stride_w2scale_k,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_offs = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    
    W2_ptrs = W2_ptr + e_idx.to(tl.int64) * stride_w2e.to(tl.int64) + (n_offs[None, :].to(tl.int64) * stride_w2n.to(tl.int64)) + (tl.arange(0, BLOCK_K)[:, None].to(tl.int64) * stride_w2k.to(tl.int64))
    
    scale_n_offs = n_offs // 128
    W2_scale_ptrs = W2_scale_ptr + e_idx.to(tl.int64) * stride_w2scale_e.to(tl.int64) + scale_n_offs[None, :].to(tl.int64) * stride_w2scale_n.to(tl.int64)

    c_row = start_idx + m_offs
    C_ptrs = C_ptr + c_row[:, None].to(tl.int64) * stride_cm.to(tl.int64) + tl.arange(0, BLOCK_K)[None, :].to(tl.int64) * stride_ck.to(tl.int64)
    C_scale_ptrs = C_scale_ptr + c_row[:, None].to(tl.int64) * stride_cscale_m.to(tl.int64)

    for k in range(0, I, BLOCK_K):
        k_idx = k // BLOCK_K
        
        # Load FP8 activation and its corresponding block-scale
        c = tl.load(C_ptrs, mask=m_mask[:, None], other=0.0)
        c_scale = tl.load(C_scale_ptrs + k_idx * stride_cscale_k.to(tl.int64), mask=m_mask[:, None], other=0.0)
        
        # Load FP8 weight and its corresponding block-scale
        w = tl.load(W2_ptrs) 
        w_scale = tl.load(W2_scale_ptrs + k_idx * stride_w2scale_k.to(tl.int64))
        
        # Perform FP8 Tensor Core dot, scaling back to FP32 correctly
        dot_res = tl.dot(c, w, out_dtype=tl.float32)
        dot_res = dot_res * c_scale * w_scale
        acc += dot_res
        
        C_ptrs += BLOCK_K * stride_ck.to(tl.int64)
        W2_ptrs += BLOCK_K * stride_w2k.to(tl.int64)

    weight_ptrs = Routing_weights_ptr + start_idx + m_offs
    w_routing = tl.load(weight_ptrs, mask=m_mask, other=0.0)
    
    acc = acc * w_routing[:, None]
    
    O_ptrs = Out_ptr + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm.to(tl.int64) + n_offs[None, :].to(tl.int64) * stride_outn.to(tl.int64)
    tl.store(O_ptrs, acc, mask=m_mask[:, None])


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm