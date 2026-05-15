import torch
import torch.nn.functional as F


HIDDEN = 7168
INTERMEDIATE = 2048
BLOCK = 128
N_GROUP = 8
TOPK_GROUP = 4
TOPK_EXPERT = 8


def _expand_scales_2d(scale_2d: torch.Tensor, dim0_block: int, dim1_block: int, dim0: int, dim1: int) -> torch.Tensor:
    return scale_2d.repeat_interleave(dim0_block, dim=0).repeat_interleave(dim1_block, dim=1)[:dim0, :dim1]


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

    topk_idx, topk_w = _route(routing_logits, routing_bias, routed_scaling_factor)

    local_mask = (topk_idx >= local_expert_offset) & (topk_idx < local_expert_offset + e_local)
    tok, slot = torch.nonzero(local_mask, as_tuple=True)

    out = torch.zeros((t, HIDDEN), device=device, dtype=torch.float32)
    if tok.numel() == 0:
        return out.to(torch.bfloat16)

    local_exp = (topk_idx[tok, slot] - local_expert_offset).to(torch.long)
    routed_w = topk_w[tok, slot].to(torch.float32)

    # Sort once by expert to keep same-expert work contiguous.
    order = torch.argsort(local_exp)
    tok = tok[order]
    local_exp = local_exp[order]
    routed_w = routed_w[order]

    counts = torch.bincount(local_exp, minlength=e_local)
    expert_offsets = torch.empty(e_local + 1, device=device, dtype=torch.long)
    expert_offsets[0] = 0
    expert_offsets[1:] = torch.cumsum(counts, dim=0)

    # Cache dequantized activations for unique locally-routed tokens.
    unique_tokens, inverse = torch.unique(tok, sorted=True, return_inverse=True)
    a_fp32 = hidden_states.index_select(0, unique_tokens).to(torch.float32)
    a_scales = hidden_states_scale.index_select(1, unique_tokens).transpose(0, 1).contiguous()
    a_scales = a_scales.repeat_interleave(BLOCK, dim=1)[:, :HIDDEN]
    hidden_deq = a_fp32 * a_scales

    # Preexpand scales for all local experts once to avoid repeated repeat_interleave in the hot loop.
    w1_scale_full = gemm1_weights_scale.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    w1_scale_full = w1_scale_full[:, : 2 * INTERMEDIATE, :HIDDEN]

    w2_scale_full = gemm2_weights_scale.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    w2_scale_full = w2_scale_full[:, :HIDDEN, :INTERMEDIATE]

    for e in range(e_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        inv_e = inverse[start:end]
        tok_e = tok[start:end]
        w_e = routed_w[start:end]

        a_e = hidden_deq.index_select(0, inv_e)

        w1 = gemm1_weights[e].to(torch.float32) * w1_scale_full[e]
        g1 = a_e @ w1.t()

        x1 = g1[:, :INTERMEDIATE]
        x2 = g1[:, INTERMEDIATE:]
        s = F.silu(x2) * x1

        w2 = gemm2_weights[e].to(torch.float32) * w2_scale_full[e]
        y = s @ w2.t()
        y.mul_(w_e.unsqueeze(1))

        out.index_add_(0, tok_e, y)

    return out.to(torch.bfloat16)