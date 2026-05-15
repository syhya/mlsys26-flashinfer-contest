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
    top2_vals = torch.topk(sbg, k=2, dim=2, largest=True, sorted=False).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False).indices
    group_mask = torch.zeros((t, N_GROUP), device=routing_logits.device, dtype=torch.bool)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(t, N_GROUP, group_size).reshape(t, e_global)

    pruned = sb.masked_fill(~expert_mask, float("-inf"))
    topk_idx = torch.topk(pruned, k=TOPK_EXPERT, dim=1, largest=True, sorted=False).indices

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

    order = torch.argsort(local_exp, stable=True)
    tok = tok[order]
    local_exp = local_exp[order]
    w = w[order]

    counts = torch.bincount(local_exp, minlength=e_local)
    offsets = torch.empty(e_local + 1, device=topk_idx.device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return tok, local_exp, w, offsets


def _expand_block_scales(scale_2d, block_size, target_shape):
    """Expand block scales to full tensor shape."""
    expanded = scale_2d.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    return expanded[:target_shape[0], :target_shape[1]]


def _dequant_and_matmul_fp32(a_fp8, a_scale_2d, b_fp8, b_scale_2d, a_shape, b_shape):
    """Fallback: exact dequantization + FP32 matmul."""
    a_scale = _expand_block_scales(a_scale_2d, BLOCK, a_shape)
    b_scale = _expand_block_scales(b_scale_2d, BLOCK, b_shape)
    
    a_fp32 = a_fp8.to(torch.float32) * a_scale
    b_fp32 = b_fp8.to(torch.float32) * b_scale
    
    return torch.mm(a_fp32, b_fp32.t())


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

    # Phase 1: Exact routing
    topk_idx, topk_w = _route(routing_logits, routing_bias, routed_scaling_factor)
    
    # Phase 2: Build dispatch
    sorted_tokens, _, sorted_weights, expert_offsets = _build_dispatch(
        topk_idx, topk_w, local_expert_offset, e_local
    )

    # Phase 3: Initialize output
    output = torch.zeros((t, HIDDEN), device=device, dtype=torch.float32)
    
    if sorted_tokens.numel() == 0:
        return output.to(torch.bfloat16)

    # Phase 4: Per-expert execution with exact dequantization
    for e in range(e_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        
        if start == end:
            continue

        tok_e = sorted_tokens[start:end]
        w_e = sorted_weights[start:end]
        t_e = tok_e.shape[0]

        # Gather activations
        a_fp8 = hidden_states[tok_e]  # [t_e, 7168]
        
        # Gather activation scales: [56, t] -> [56, t_e] -> [t_e, 56]
        a_scale_2d = hidden_states_scale[:, tok_e].t().contiguous()
        
        # GEMM1: [t_e, 7168] @ [4096, 7168].T -> [t_e, 4096]
        w1_fp8 = gemm1_weights[e]  # [4096, 7168]
        w1_scale_2d = gemm1_weights_scale[e]  # [32, 56]
        
        g1 = _dequant_and_matmul_fp32(
            a_fp8, a_scale_2d,
            w1_fp8, w1_scale_2d,
            (t_e, HIDDEN), (2 * INTERMEDIATE, HIDDEN)
        )
        
        # SwiGLU
        x1 = g1[:, :INTERMEDIATE]
        x2 = g1[:, INTERMEDIATE:]
        s = F.silu(x2) * x1
        
        # GEMM2: [t_e, 2048] @ [7168, 2048].T -> [t_e, 7168]
        w2_fp8 = gemm2_weights[e]  # [7168, 2048]
        w2_scale_2d = gemm2_weights_scale[e]  # [56, 16]
        
        # For GEMM2, we need to handle s (fp32) with w2 (fp8)
        # Expand w2 scales and dequantize
        w2_scale = _expand_block_scales(w2_scale_2d, BLOCK, (HIDDEN, INTERMEDIATE))
        w2_fp32 = w2_fp8.to(torch.float32) * w2_scale
        y = torch.mm(s, w2_fp32.t())
        
        # Apply routing weights
        y = y * w_e.unsqueeze(1)
        
        # Phase 5: Accumulate
        output.index_add_(0, tok_e, y)

    # Phase 6: Final conversion with safety
    output = torch.nan_to_num(output, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return output.to(torch.bfloat16)