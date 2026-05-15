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
    offsets = torch.zeros(e_local + 1, device=topk_idx.device, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return tok, local_exp, w, offsets


def _expand_activation_scales(hidden_states_scale: torch.Tensor, tok_indices: torch.Tensor) -> torch.Tensor:
    """Expand activation scales from [56, T] to [T_e, 7168] for selected tokens."""
    # hidden_states_scale is [56, T], we need [T_e, 56] then expand to [T_e, 7168]
    scales = hidden_states_scale[:, tok_indices].t()  # [T_e, 56]
    scales_expanded = scales.repeat_interleave(BLOCK, dim=1)  # [T_e, 7168]
    return scales_expanded


def _expand_weight_scales(weight_scale: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
    """Expand weight scales from block format to full tensor."""
    # weight_scale is [out_blocks, in_blocks]
    scales_expanded = weight_scale.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return scales_expanded[:out_dim, :in_dim]


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
        start = expert_offsets[e].item()
        end = expert_offsets[e + 1].item()
        
        if start == end:
            continue
        
        tok_e = sorted_tokens[start:end]
        w_e = sorted_weights[start:end]
        
        # Gather activations
        A_fp8 = hidden_states[tok_e]  # [T_e, 7168] fp8
        A_scale = _expand_activation_scales(hidden_states_scale, tok_e)  # [T_e, 7168]
        
        # Dequantize activations to FP32
        A_fp32 = A_fp8.float() * A_scale
        
        # GEMM1: [T_e, 7168] @ [7168, 4096] -> [T_e, 4096]
        W1_fp8 = gemm1_weights[e]  # [4096, 7168] fp8
        W1_scale = _expand_weight_scales(gemm1_weights_scale[e], 2 * INTERMEDIATE, HIDDEN)  # [4096, 7168]
        W1_fp32 = W1_fp8.float() * W1_scale
        
        G1 = torch.mm(A_fp32, W1_fp32.t())  # [T_e, 4096] in FP32
        
        # SwiGLU
        x1 = G1[:, :INTERMEDIATE]
        x2 = G1[:, INTERMEDIATE:]
        S = F.silu(x2) * x1  # [T_e, 2048] in FP32
        
        # GEMM2: [T_e, 2048] @ [2048, 7168] -> [T_e, 7168]
        W2_fp8 = gemm2_weights[e]  # [7168, 2048] fp8
        W2_scale = _expand_weight_scales(gemm2_weights_scale[e], HIDDEN, INTERMEDIATE)  # [7168, 2048]
        W2_fp32 = W2_fp8.float() * W2_scale
        
        Y = torch.mm(S, W2_fp32.t())  # [T_e, 7168] in FP32
        
        # Apply routing weights
        Y = Y * w_e.unsqueeze(1)
        
        # Phase 5: Accumulate
        output.index_add_(0, tok_e, Y)
    
    # Phase 6: Final conversion
    output = torch.nan_to_num(output, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return output.to(torch.bfloat16)