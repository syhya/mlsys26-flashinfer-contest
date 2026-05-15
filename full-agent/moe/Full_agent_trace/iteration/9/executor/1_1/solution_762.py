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
    
    n_offs = (n_idx * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    k_offs = tl.arange(0, BLOCK_K).to(tl.int64)
    
    A_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
    W_ptrs = W_ptr + e_idx.to(tl.int64) * stride_we + n_offs[None, :] * stride_wn + k_offs[:, None] * stride_wk
    
    scale_n_offs = n_offs // 128
    W_scale_ptrs = W_scale_ptr + e_idx.to(tl.int64) * stride_wscale_e + scale_n_offs[None, :] * stride_wscale_n
    A_scale_ptrs = A_scale_ptr + token_ids[:, None] * stride_ascale_m

    for k in range(0, H, BLOCK_K):
        k_idx = k // BLOCK_K
        
        w = tl.load(W_ptrs)
        a = tl.load(A_ptrs, mask=m_mask[:, None], other=0.0).to(w.dtype)
        
        a_scale = tl.load(A_scale_ptrs + k_idx * stride_ascale_k, mask=m_mask[:, None], other=0.0)
        w_scale = tl.load(W_scale_ptrs + k_idx * stride_wscale_k)
        
        # FP8 Tensor Core dot, accumulation in float32
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
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
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
    T, I, H,
    stride_g1m, stride_g1k,
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
    
    n_offs = (n_idx * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    k_offs = tl.arange(0, BLOCK_K).to(tl.int64)
    
    W2_ptrs = W2_ptr + e_idx.to(tl.int64) * stride_w2e + n_offs[None, :] * stride_w2n + k_offs[:, None] * stride_w2k
    
    scale_n_offs = n_offs // 128
    W2_scale_ptrs = W2_scale_ptr + e_idx.to(tl.int64) * stride_w2scale_e + scale_n_offs[None, :] * stride_w2scale_n

    g1_row = (start_idx + m_offs).to(tl.int64) * stride_g1m
    
    g1_x1_ptrs = G1_ptr + g1_row[:, None] + k_offs[None, :] * stride_g1k
    g1_x2_ptrs = G1_ptr + g1_row[:, None] + I * stride_g1k + k_offs[None, :] * stride_g1k
    
    for k in range(0, I, BLOCK_K):
        k_idx = k // BLOCK_K
        
        x1 = tl.load(g1_x1_ptrs, mask=m_mask[:, None], other=0.0)
        x2 = tl.load(g1_x2_ptrs, mask=m_mask[:, None], other=0.0)
        
        # SwiGLU: silu(x2) * x1
        c = x1 * x2 * tl.sigmoid(x2)
        c_bf16 = c.to(tl.bfloat16)
        
        w = tl.load(W2_ptrs) 
        w_scale = tl.load(W2_scale_ptrs + k_idx * stride_w2scale_k)
        
        # Exact BF16 dequantization for numerical stability
        w_bf16 = (w.to(tl.float32) * w_scale).to(tl.bfloat16)
        
        # BF16 Tensor Core dot
        dot_res = tl.dot(c_bf16, w_bf16, out_dtype=tl.float32)
        acc += dot_res
        
        W2_ptrs += BLOCK_K * stride_w2k
        g1_x1_ptrs += BLOCK_K * stride_g1k
        g1_x2_ptrs += BLOCK_K * stride_g1k

    weight_ptrs = Routing_weights_ptr + (start_idx + m_offs).to(tl.int64)
    w_routing = tl.load(weight_ptrs, mask=m_mask, other=0.0)
    
    acc = acc * w_routing[:, None]
    
    O_ptrs = Out_ptr + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm + n_offs[None, :] * stride_outn
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
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    device = hidden_states.device

    # 1) DeepSeek-V3 No-Aux Routing (Vectorized with sorted=True for exact deterministic matching)
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias

    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    group_size = E_global // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)

    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=True)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=True)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    
    # Crucial Correction: DeepSeek routing does NOT sum-normalize weights.
    weights = s * M * routed_scaling_factor

    # 2) Token Sorting and CSR Grouping
    local_start = int(local_expert_offset)
    expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + E_local)
    
    flat_token_idx, flat_k_idx = torch.nonzero(expert_mask, as_tuple=True)
    Tk_total = flat_token_idx.numel()

    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    local_expert_idx = topk_idx[flat_token_idx, flat_k_idx] - local_start

    sorted_indices = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[sorted_indices].to(torch.int32)
    sorted_experts = local_expert_idx[sorted_indices].to(torch.int32)

    global_expert_idx = sorted_experts + local_start
    sorted_weights = weights[sorted_tokens.long(), global_expert_idx.long()].to(torch.float32)
    
    expert_counts = torch.bincount(sorted_experts, minlength=E_local)
    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)
    
    max_Tk = expert_counts.max().item()

    # 3) Grouped GEMM 1 (FP8 Tensor Cores + Block Scale Integration)
    G1 = torch.zeros((Tk_total, 2 * I), dtype=torch.float32, device=device)
    grid_gemm1 = lambda META: (triton.cdiv(2 * I, META['BLOCK_N']), triton.cdiv(max_Tk, META['BLOCK_M']), E_local)
    
    grouped_gemm1_kernel[grid_gemm1](
        A_ptr=hidden_states, A_scale_ptr=hidden_states_scale,
        W_ptr=gemm1_weights, W_scale_ptr=gemm1_weights_scale,
        Out_ptr=G1,
        sorted_tokens_ptr=sorted_tokens, expert_offsets_ptr=expert_offsets,
        T=T, H=H, N_dim=2*I,
        stride_am=hidden_states.stride(0), stride_ak=hidden_states.stride(1),
        stride_ascale_k=hidden_states_scale.stride(0), stride_ascale_m=hidden_states_scale.stride(1),
        stride_we=gemm1_weights.stride(0), stride_wn=gemm1_weights.stride(1), stride_wk=gemm1_weights.stride(2),
        stride_wscale_e=gemm1_weights_scale.stride(0), stride_wscale_n=gemm1_weights_scale.stride(1), stride_wscale_k=gemm1_weights_scale.stride(2),
        stride_outm=G1.stride(0), stride_outn=G1.stride(1),
        BLOCK_K=128
    )

    # 4) Grouped GEMM 2 (Fused SwiGLU + Exact BF16 Tensor Cores)
    O = torch.zeros((Tk_total, H), dtype=torch.float32, device=device)
    grid_gemm2 = lambda META: (triton.cdiv(H, META['BLOCK_N']), triton.cdiv(max_Tk, META['BLOCK_M']), E_local)
    
    grouped_gemm2_kernel[grid_gemm2](
        G1_ptr=G1,
        W2_ptr=gemm2_weights, W2_scale_ptr=gemm2_weights_scale,
        Routing_weights_ptr=sorted_weights,
        Out_ptr=O,
        expert_offsets_ptr=expert_offsets,
        T=T, I=I, H=H,
        stride_g1m=G1.stride(0), stride_g1k=G1.stride(1),
        stride_w2e=gemm2_weights.stride(0), stride_w2n=gemm2_weights.stride(1), stride_w2k=gemm2_weights.stride(2),
        stride_w2scale_e=gemm2_weights_scale.stride(0), stride_w2scale_n=gemm2_weights_scale.stride(1), stride_w2scale_k=gemm2_weights_scale.stride(2),
        stride_outm=O.stride(0), stride_outn=O.stride(1),
        BLOCK_K=128
    )

    # 5) Scatter Add Deterministic Accumulation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    output.index_add_(0, sorted_tokens.long(), O)

    return output.to(torch.bfloat16)