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


def _dequant_blockwise(fp8_tensor, scale_2d, block_size, target_shape):
    """Dequantize FP8 tensor using block scales."""
    fp32 = fp8_tensor.to(torch.float32)
    scale_expanded = scale_2d.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[:target_shape[0], :target_shape[1]]
    return fp32 * scale_expanded


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
        
        # Gather activations (FP8)
        A_fp8 = hidden_states[tok_e]  # [T_e, 7168]
        
        # Gather activation scales and transpose: [56, T] -> [T_e, 56]
        A_scale = hidden_states_scale[:, tok_e].t().contiguous()
        
        # Expand activation scales to [T_e, 7168]
        A_scale_expanded = A_scale.repeat_interleave(BLOCK, dim=1)[:, :HIDDEN]
        
        # Dequantize activations to FP32
        A_fp32 = A_fp8.to(torch.float32) * A_scale_expanded
        
        # GEMM1: [T_e, 7168] @ [7168, 4096] -> [T_e, 4096]
        W1_fp8 = gemm1_weights[e]  # [4096, 7168]
        W1_scale = gemm1_weights_scale[e]  # [32, 56]
        
        # Dequantize GEMM1 weights
        W1_fp32 = _dequant_blockwise(W1_fp8, W1_scale, BLOCK, (2 * INTERMEDIATE, HIDDEN))
        
        # Compute GEMM1 in FP32
        G1 = torch.mm(A_fp32, W1_fp32.t())
        
        # SwiGLU
        x1 = G1[:, :INTERMEDIATE]
        x2 = G1[:, INTERMEDIATE:]
        S = F.silu(x2) * x1
        
        # GEMM2: [T_e, 2048] @ [2048, 7168] -> [T_e, 7168]
        W2_fp8 = gemm2_weights[e]  # [7168, 2048]
        W2_scale = gemm2_weights_scale[e]  # [56, 16]
        
        # Dequantize GEMM2 weights
        W2_fp32 = _dequant_blockwise(W2_fp8, W2_scale, BLOCK, (HIDDEN, INTERMEDIATE))
        
        # Compute GEMM2 in FP32
        Y = torch.mm(S, W2_fp32.t())
        
        # Apply routing weights
        Y = Y * w_e.unsqueeze(1)
        
        # Accumulate
        output.index_add_(0, tok_e, Y)

    # Phase 5: Convert to BF16
    output = torch.nan_to_num(output, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return output.to(torch.bfloat16)