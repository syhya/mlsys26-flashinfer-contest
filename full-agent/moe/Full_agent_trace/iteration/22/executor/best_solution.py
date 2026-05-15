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


def _dequant_activation_rows(hidden_states, hidden_states_scale, token_ids):
    """Dequantize activation rows for given token IDs."""
    a_fp8 = hidden_states.index_select(0, token_ids)
    a_scale = hidden_states_scale.index_select(1, token_ids).t().contiguous()
    a_scale_expanded = a_scale.repeat_interleave(BLOCK, dim=1)[:, :HIDDEN]
    return a_fp8.to(torch.float32) * a_scale_expanded


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

    # Phase 4: Dequantize unique tokens once
    unique_tokens, inverse = torch.unique(sorted_tokens, sorted=True, return_inverse=True)
    hidden_deq = _dequant_activation_rows(hidden_states, hidden_states_scale, unique_tokens)

    # Phase 5: Per-expert execution
    for e in range(e_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        
        if start == end:
            continue

        # Gather tokens for this expert
        tok_e = sorted_tokens[start:end]
        inv_e = inverse[start:end]
        w_e = sorted_weights[start:end]
        
        # Get dequantized activations
        a_e = hidden_deq.index_select(0, inv_e)
        
        # GEMM1: Dequantize weights and compute
        w1_fp8 = gemm1_weights[e]
        w1_scale = gemm1_weights_scale[e]
        w1_scale_expanded = _expand_block_scales(w1_scale, BLOCK, (2 * INTERMEDIATE, HIDDEN))
        w1_fp32 = w1_fp8.to(torch.float32) * w1_scale_expanded
        
        g1 = torch.mm(a_e, w1_fp32.t())
        
        # SwiGLU
        x1 = g1[:, :INTERMEDIATE]
        x2 = g1[:, INTERMEDIATE:]
        s = F.silu(x2) * x1
        
        # GEMM2: Dequantize weights and compute
        w2_fp8 = gemm2_weights[e]
        w2_scale = gemm2_weights_scale[e]
        w2_scale_expanded = _expand_block_scales(w2_scale, BLOCK, (HIDDEN, INTERMEDIATE))
        w2_fp32 = w2_fp8.to(torch.float32) * w2_scale_expanded
        
        y = torch.mm(s, w2_fp32.t())
        
        # Apply routing weights
        y = y * w_e.unsqueeze(1)
        
        # Accumulate
        output.index_add_(0, tok_e, y)

    # Phase 6: Final conversion with safety
    output = torch.nan_to_num(output, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return output.to(torch.bfloat16)