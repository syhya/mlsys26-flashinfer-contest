import torch
import torch.nn.functional as F


HIDDEN = 7168
INTERMEDIATE = 2048
BLOCK = 128
N_GROUP = 8
TOPK_GROUP = 4
TOPK_EXPERT = 8
E_LOCAL = 32


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
    topk = torch.topk(pruned, k=TOPK_EXPERT, dim=1, largest=True, sorted=False)
    topk_idx = topk.indices

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
    device = topk_idx.device
    if tok.numel() == 0:
        return (
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.float32),
            torch.zeros(e_local + 1, device=device, dtype=torch.long),
        )

    local_exp = (topk_idx[tok, slot] - local_expert_offset).to(torch.long)
    w = topk_w[tok, slot].to(torch.float32)

    # group by expert to maximize weight cache reuse
    order = torch.argsort(local_exp)
    tok = tok[order]
    local_exp = local_exp[order]
    w = w[order]

    counts = torch.bincount(local_exp, minlength=e_local)
    offsets = torch.empty(e_local + 1, device=device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return tok, local_exp, w, offsets


def _dequant_hidden_all(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
) -> torch.Tensor:
    a = hidden_states.to(torch.float32)
    s = hidden_states_scale.transpose(0, 1).contiguous()
    s = s.repeat_interleave(BLOCK, dim=1)[:, :HIDDEN]
    return a * s


def _dequant_w1_all(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
) -> torch.Tensor:
    # [32, 4096, 7168]
    w = gemm1_weights.to(torch.float32)
    s = gemm1_weights_scale.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    s = s[:, : (2 * INTERMEDIATE), :HIDDEN]
    return w * s


def _dequant_w2_all(
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
) -> torch.Tensor:
    # [32, 7168, 2048]
    w = gemm2_weights.to(torch.float32)
    s = gemm2_weights_scale.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    s = s[:, :HIDDEN, :INTERMEDIATE]
    return w * s


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
    sorted_tokens, sorted_local_exp, sorted_weights, expert_offsets = _build_dispatch(
        topk_idx, topk_w, local_expert_offset, e_local
    )

    out = torch.zeros((t, HIDDEN), device=device, dtype=torch.float32)
    if sorted_tokens.numel() == 0:
        return out.to(torch.bfloat16)

    # High-throughput exact path:
    # 1) dequant all token activations once
    # 2) dequant all local expert weights once
    # 3) use batched dense matmuls per expert batch
    hidden_all = _dequant_hidden_all(hidden_states, hidden_states_scale)
    w1_all = _dequant_w1_all(gemm1_weights, gemm1_weights_scale)
    w2_all = _dequant_w2_all(gemm2_weights, gemm2_weights_scale)

    for e in range(e_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        tok_e = sorted_tokens[start:end]
        a_e = hidden_all.index_select(0, tok_e)

        g1 = torch.mm(a_e, w1_all[e].transpose(0, 1))
        x1 = g1[:, :INTERMEDIATE]
        x2 = g1[:, INTERMEDIATE:]
        s = F.silu(x2) * x1
        y = torch.mm(s, w2_all[e].transpose(0, 1))
        y.mul_(sorted_weights[start:end].unsqueeze(1))
        out.index_add_(0, tok_e, y)

    return out.to(torch.bfloat16)