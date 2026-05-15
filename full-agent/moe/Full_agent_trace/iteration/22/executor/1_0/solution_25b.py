import torch
import torch.nn.functional as F


HIDDEN = 7168
INTERMEDIATE = 2048
BLOCK = 128
N_GROUP = 8
TOPK_GROUP = 4
TOPK_EXPERT = 8


def _route(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    t, e_global = routing_logits.shape
    s = torch.sigmoid(routing_logits.to(torch.float32))
    sb = s + routing_bias.to(torch.float32).view(1, e_global)

    group_size = e_global // N_GROUP
    sbg = sb.view(t, N_GROUP, group_size)
    top2_vals = torch.topk(sbg, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((t, N_GROUP), device=routing_logits.device, dtype=torch.bool)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(t, N_GROUP, group_size).reshape(t, e_global)

    pruned = sb.masked_fill(~expert_mask, float("-inf"))
    topk_idx = torch.topk(pruned, k=TOPK_EXPERT, dim=1, largest=True, sorted=True).indices

    topk_w = s.gather(1, topk_idx)
    topk_w = topk_w / (topk_w.sum(dim=1, keepdim=True) + 1e-20)
    topk_w = topk_w * routed_scaling_factor
    return topk_idx, topk_w


def _build_dispatch(
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
    local_expert_offset: int,
    e_local: int,
):
    local_mask = (topk_idx >= local_expert_offset) & (topk_idx < local_expert_offset + e_local)
    tok, slot = torch.nonzero(local_mask, as_tuple=True)
    if tok.numel() == 0:
        device = topk_idx.device
        return (
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.float32),
            torch.zeros(e_local + 1, device=device, dtype=torch.long),
        )

    local_exp = (topk_idx[tok, slot] - local_expert_offset).to(torch.long)
    w = topk_w[tok, slot].to(torch.float32)

    order = torch.argsort(local_exp)
    tok = tok[order]
    local_exp = local_exp[order]
    w = w[order]

    counts = torch.bincount(local_exp, minlength=e_local)
    offsets = torch.empty(e_local + 1, device=topk_idx.device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return tok, local_exp, w, offsets


def _dequant_and_gemm(
    A_fp8: torch.Tensor,
    A_scale: torch.Tensor,
    W_fp8: torch.Tensor,
    W_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Perform FP8 GEMM with block-scale dequantization.
    A_fp8: [M, K] fp8
    A_scale: [M, K//128] or [K//128, M] (will be transposed if needed)
    W_fp8: [N, K] fp8
    W_scale: [N//128, K//128]
    Returns: [M, N] in out_dtype
    """
    M, K = A_fp8.shape
    N = W_fp8.shape[0]
    
    # Expand A scales: [M, K//128] -> [M, K]
    if A_scale.shape[0] == K // 128:
        A_scale = A_scale.t()  # [K//128, M] -> [M, K//128]
    A_scale_expanded = A_scale.repeat_interleave(BLOCK, dim=1)[:, :K]
    
    # Expand W scales: [N//128, K//128] -> [N, K]
    W_scale_expanded = W_scale.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)[:N, :K]
    
    # Dequantize and compute
    A_fp32 = A_fp8.to(torch.float32) * A_scale_expanded
    W_fp32 = W_fp8.to(torch.float32) * W_scale_expanded
    
    result = torch.mm(A_fp32, W_fp32.t())
    return result.to(out_dtype)


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
    device = hidden_states.device
    t = hidden_states.shape[0]
    e_local = gemm1_weights.shape[0]

    # Phase 1: Routing
    topk_idx, topk_w = _route(routing_logits, routing_bias, routed_scaling_factor)
    
    # Phase 2: Dispatch
    sorted_tokens, _, sorted_weights, expert_offsets = _build_dispatch(
        topk_idx, topk_w, local_expert_offset, e_local
    )

    # Phase 3: Initialize output
    output = torch.zeros((t, HIDDEN), device=device, dtype=torch.float32)
    
    if sorted_tokens.numel() == 0:
        return output.to(torch.bfloat16)

    # Phase 4: Per-expert execution
    for e in range(e_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        
        if start == end:
            continue
        
        tok_e = sorted_tokens[start:end]
        w_e = sorted_weights[start:end]
        
        # Gather activations
        A_fp8 = hidden_states[tok_e]  # [T_e, 7168]
        A_scale = hidden_states_scale[:, tok_e].t().contiguous()  # [T_e, 56]
        
        # GEMM1: [T_e, 7168] x [4096, 7168]^T -> [T_e, 4096]
        W1_fp8 = gemm1_weights[e]  # [4096, 7168]
        W1_scale = gemm1_weights_scale[e]  # [32, 56]
        
        G1 = _dequant_and_gemm(A_fp8, A_scale, W1_fp8, W1_scale, torch.float32)
        
        # SwiGLU
        x1 = G1[:, :INTERMEDIATE]
        x2 = G1[:, INTERMEDIATE:]
        S = F.silu(x2) * x1
        
        # GEMM2: [T_e, 2048] x [7168, 2048]^T -> [T_e, 7168]
        W2_fp8 = gemm2_weights[e]  # [7168, 2048]
        W2_scale = gemm2_weights_scale[e]  # [56, 16]
        
        # For GEMM2, we need to treat S as if it has block scales
        # Since S is computed in fp32, we create uniform scales
        S_scale = torch.ones((S.shape[0], S.shape[1] // BLOCK), device=device, dtype=torch.float32)
        
        Y = _dequant_and_gemm(S, S_scale, W2_fp8, W2_scale, torch.float32)
        
        # Apply routing weights
        Y = Y * w_e.unsqueeze(1)
        
        # Phase 5: Accumulate
        output.index_add_(0, tok_e, Y)
    
    # Phase 6: Final conversion
    output = torch.nan_to_num(output, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return output.to(torch.bfloat16)