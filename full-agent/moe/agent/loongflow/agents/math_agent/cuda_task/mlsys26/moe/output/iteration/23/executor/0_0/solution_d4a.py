import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    ],
    key=["Tk", "H", "I"],
)
@triton.jit
def fused_swiglu_gemm2_kernel(
    G1_ptr, W2_ptr, W2_scale_ptr, Out_ptr, weights_ptr,
    Tk, H, I,
    stride_g1_m, stride_g1_n,
    stride_w2_h, stride_w2_i,
    stride_w2scale_h, stride_w2scale_i,
    stride_out_m, stride_out_h,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    
    m_mask = m_offs < Tk
    n_mask = n_offs < H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, I, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < I
        
        # Load G1 gate and up projections
        g1_gate_ptrs = G1_ptr + m_offs[:, None] * stride_g1_m + k_offs[None, :] * stride_g1_n
        g1_up_ptrs = G1_ptr + m_offs[:, None] * stride_g1_m + (I + k_offs[None, :]) * stride_g1_n
        
        gate = tl.load(g1_gate_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        up = tl.load(g1_up_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # SwiGLU: silu(up) * gate
        up_silu = up * tl.sigmoid(up)
        swiglu_out = up_silu * gate
        
        # Load W2 weights (FP8)
        w2_ptrs = W2_ptr + n_offs[:, None] * stride_w2_h + k_offs[None, :] * stride_w2_i
        w2 = tl.load(w2_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load W2 scales
        k_block = k_start // 128
        n_block = n_offs // 128
        w2_scale_ptrs = W2_scale_ptr + n_block[:, None] * stride_w2scale_h + k_block * stride_w2scale_i
        w2_scale = tl.load(w2_scale_ptrs, mask=n_mask[:, None], other=1.0)
        
        # FP8 dot with scale
        dot = tl.dot(swiglu_out, tl.trans(w2), out_dtype=tl.float32)
        acc += dot * w2_scale
    
    # Apply routing weights
    routing_weights = tl.load(weights_ptr + m_offs, mask=m_mask, other=0.0)
    acc = acc * routing_weights[:, None]
    
    # Store output
    out_ptrs = Out_ptr + m_offs[:, None] * stride_out_m + n_offs[None, :] * stride_out_h
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _route_tokens(routing_logits, routing_bias, local_expert_offset, e_local, routed_scaling_factor):
    device = routing_logits.device
    T, E_global = routing_logits.shape
    N_GROUP = 8
    TOPK_GROUP = 4
    TOP_K = 8
    GROUP_SIZE = E_global // N_GROUP

    q = torch.sigmoid(routing_logits.float())
    rank_scores = q + routing_bias.float().view(1, E_global)

    grouped = rank_scores.view(T, N_GROUP, GROUP_SIZE)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)
    expert_allowed = group_mask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_global)

    pruned = torch.where(expert_allowed, rank_scores, torch.full_like(rank_scores, float("-inf")))
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    sel = torch.zeros((T, E_global), dtype=torch.bool, device=device)
    sel.scatter_(1, topk_idx, True)

    raw_w = q * sel
    denom = raw_w.sum(dim=1, keepdim=True) + 1e-20
    weights = raw_w * (routed_scaling_factor / denom)

    local_start = int(local_expert_offset)
    expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)

    flat_token_idx, flat_k_idx = torch.nonzero(expert_mask, as_tuple=True)
    if flat_token_idx.numel() == 0:
        return None

    chosen_experts = topk_idx[flat_token_idx, flat_k_idx]
    local_expert_idx = chosen_experts - local_start
    chosen_weights = weights[flat_token_idx, chosen_experts]

    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order].to(torch.int32)
    sorted_weights = chosen_weights[order].to(torch.float32)

    expert_counts = torch.bincount(sorted_experts.to(torch.int64), minlength=e_local)
    expert_offsets = torch.zeros(e_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts.to(torch.int32), dim=0)

    return sorted_tokens, sorted_experts, sorted_weights, expert_counts.to(torch.int32), expert_offsets


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
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]

    routed = _route_tokens(routing_logits, routing_bias, local_expert_offset, E_local, routed_scaling_factor)
    if routed is None:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets = routed
    Tk_total = sorted_tokens.numel()

    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    # Process each expert
    for e in range(E_local):
        s = int(expert_offsets[e].item())
        t = int(expert_offsets[e + 1].item())
        if s == t:
            continue

        Tk_e = t - s
        tok_ids = sorted_tokens[s:t].long()
        rw = sorted_weights[s:t]

        # Gather hidden states for this expert's tokens
        h_e = hidden_states[tok_ids]  # [Tk_e, H] fp8
        
        # Prepare scales for torch._scaled_mm
        # hidden_states_scale is [56, T], we need [Tk_e, 56] then flatten to per-token scales
        h_scale_e = hidden_states_scale[:, tok_ids].t()  # [Tk_e, 56]
        
        # For torch._scaled_mm, we need per-row scale (one scale per token)
        # Average across K-blocks as approximation, or use max
        h_scale_vec = h_scale_e.amax(dim=1, keepdim=True)  # [Tk_e, 1]
        
        # gemm1_weights[e] is [4096, 7168] fp8
        # gemm1_weights_scale[e] is [32, 56]
        w1_e = gemm1_weights[e]  # [4096, H]
        w1_scale_e = gemm1_weights_scale[e]  # [32, 56]
        
        # Use per-output-block scale (average or max across K-blocks)
        w1_scale_vec = w1_scale_e.amax(dim=1, keepdim=True)  # [32, 1]
        
        # Dequantize for GEMM1 (torch._scaled_mm requires specific scale format)
        # Fallback to explicit dequant + matmul for correctness
        h_scale_expanded = h_scale_e.repeat_interleave(128, dim=1)  # [Tk_e, H]
        h_dequant = h_e.float() * h_scale_expanded
        
        w1_scale_expanded = w1_scale_e.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)  # [4096, H]
        w1_dequant = w1_e.float() * w1_scale_expanded
        
        # GEMM1: [Tk_e, H] @ [H, 4096] -> [Tk_e, 4096]
        G1_e = h_dequant @ w1_dequant.t()
        
        # SwiGLU + GEMM2 with fused Triton kernel
        # G1_e is [Tk_e, 4096] where first 2048 is gate, second 2048 is up
        
        # Allocate output for this expert
        out_e = torch.zeros((Tk_e, H), dtype=torch.float32, device=device)
        
        grid = lambda META: (
            triton.cdiv(Tk_e, META["BLOCK_M"]),
            triton.cdiv(H, META["BLOCK_N"]),
        )
        
        fused_swiglu_gemm2_kernel[grid](
            G1_e, gemm2_weights[e], gemm2_weights_scale[e], out_e, rw,
            Tk_e, H, I,
            G1_e.stride(0), G1_e.stride(1),
            gemm2_weights[e].stride(0), gemm2_weights[e].stride(1),
            gemm2_weights_scale[e].stride(0), gemm2_weights_scale[e].stride(1),
            out_e.stride(0), out_e.stride(1),
        )
        
        # Scatter-add to output
        output.index_add_(0, tok_ids, out_e)

    return output.to(torch.bfloat16)