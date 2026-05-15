import torch
import torch.nn.functional as F


HIDDEN = 7168
INTER = 2048
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLOCK = 128


def _route_noaux(routing_logits, routing_bias, routed_scaling_factor):
    logits = routing_logits.float()
    bias = routing_bias.float().view(1, -1)

    s = torch.sigmoid(logits)
    s_bias = s + bias

    group_size = s.shape[1] // N_GROUP
    grouped = s_bias.view(s.shape[0], N_GROUP, group_size)
    top2 = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2.sum(dim=2)

    topg = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, topg, True)

    expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, group_size).reshape(s.shape[0], s.shape[1])
    neg_inf = torch.full_like(s_bias, float("-inf"))
    pruned = torch.where(expert_mask, s_bias, neg_inf)

    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    sel = torch.zeros_like(s)
    sel.scatter_(1, topk_idx, 1.0)
    weights = s * sel
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return topk_idx, weights


def _build_local_dispatch(topk_idx, weights, local_expert_offset, e_local):
    local_mask = (topk_idx >= local_expert_offset) & (topk_idx < local_expert_offset + e_local)
    token_idx, slot_idx = torch.nonzero(local_mask, as_tuple=True)

    if token_idx.numel() == 0:
        device = topk_idx.device
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            torch.zeros((e_local + 1,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    local_experts = (topk_idx[token_idx, slot_idx] - local_expert_offset).long()
    route_w = weights[token_idx, topk_idx[token_idx, slot_idx]].float()

    order = torch.argsort(local_experts, stable=True)
    sorted_tokens = token_idx[order].long()
    sorted_local_experts = local_experts[order]
    sorted_route_w = route_w[order]

    counts = torch.bincount(sorted_local_experts, minlength=e_local)
    offsets = torch.zeros((e_local + 1,), dtype=torch.long, device=topk_idx.device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    active = torch.nonzero(counts, as_tuple=False).flatten()
    return sorted_tokens, sorted_local_experts, sorted_route_w, offsets, active


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    a = hidden_states.index_select(0, token_ids)
    tk = a.shape[0]
    a = a.view(tk, HIDDEN // BLOCK, BLOCK).float()
    s = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous().view(tk, HIDDEN // BLOCK, 1)
    return (a * s).reshape(tk, HIDDEN)


def _dequant_w1_expert(gemm1_weights, gemm1_weights_scale, e):
    w = gemm1_weights[e].float().view((2 * INTER) // BLOCK, BLOCK, HIDDEN // BLOCK, BLOCK)
    s = gemm1_weights_scale[e].float().view((2 * INTER) // BLOCK, HIDDEN // BLOCK, 1, 1)
    return (w * s).permute(0, 2, 1, 3).reshape(2 * INTER, HIDDEN)


def _dequant_w2_expert(gemm2_weights, gemm2_weights_scale, e):
    w = gemm2_weights[e].float().view(HIDDEN // BLOCK, BLOCK, INTER // BLOCK, BLOCK)
    s = gemm2_weights_scale[e].float().view(HIDDEN // BLOCK, INTER // BLOCK, 1, 1)
    return (w * s).permute(0, 2, 1, 3).reshape(HIDDEN, INTER)


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
    T = routing_logits.shape[0]
    e_local = gemm1_weights.shape[0]

    topk_idx, weights = _route_noaux(routing_logits, routing_bias, routed_scaling_factor)
    sorted_tokens, _, sorted_route_w, offsets, active_experts = _build_local_dispatch(
        topk_idx, weights, int(local_expert_offset), e_local
    )

    if sorted_tokens.numel() == 0:
        return torch.zeros((T, HIDDEN), dtype=torch.bfloat16, device=device)

    output = torch.zeros((T, HIDDEN), dtype=torch.float32, device=device)

    hidden_cache = {}
    w1_cache = {}
    w2_cache = {}

    for e in active_experts.tolist():
        start = int(offsets[e].item())
        end = int(offsets[e + 1].item())
        token_ids = sorted_tokens[start:end]
        route_w = sorted_route_w[start:end]

        tk = token_ids.numel()
        if tk == 0:
            continue

        key = token_ids
        a = _dequant_hidden_rows(hidden_states, hidden_states_scale, key)

        if e not in w1_cache:
            w1_cache[e] = _dequant_w1_expert(gemm1_weights, gemm1_weights_scale, e)
        if e not in w2_cache:
            w2_cache[e] = _dequant_w2_expert(gemm2_weights, gemm2_weights_scale, e)

        w1 = w1_cache[e]
        w2 = w2_cache[e]

        g1 = torch.mm(a, w1.t())
        x1 = g1[:, :INTER]
        x2 = g1[:, INTER:]
        swiglu = F.silu(x2) * x1
        o = torch.mm(swiglu, w2.t())
        o.mul_(route_w.unsqueeze(1))
        output.index_add_(0, token_ids, o)

    return output.to(torch.bfloat16)