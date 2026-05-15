import torch
import torch.nn.functional as F

HIDDEN = 7168
INTER = 2048
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLOCK = 128
E_LOCAL = 32
GROUP_SIZE = 256 // N_GROUP


def _route_noaux(routing_logits, routing_bias, routed_scaling_factor):
    logits = routing_logits.float()
    bias = routing_bias.float().view(1, -1)

    s = torch.sigmoid(logits)
    s_bias = s + bias

    grouped = s_bias.view(s_bias.shape[0], N_GROUP, GROUP_SIZE)
    top2 = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2.sum(dim=2)

    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_idx, True)

    expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, GROUP_SIZE).reshape(s.shape[0], -1)
    neg_inf = torch.full((), float("-inf"), device=s.device, dtype=s.dtype)
    pruned = torch.where(expert_mask, s_bias, neg_inf)

    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    sel = torch.zeros_like(s)
    sel.scatter_(1, topk_idx, 1.0)
    weights = s * sel
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * float(routed_scaling_factor)
    return topk_idx, weights


def _build_local_dispatch(topk_idx, weights, local_expert_offset, e_local):
    local_start = int(local_expert_offset)
    local_end = local_start + int(e_local)

    local_mask = (topk_idx >= local_start) & (topk_idx < local_end)
    token_ids, slot_ids = torch.nonzero(local_mask, as_tuple=True)

    if token_ids.numel() == 0:
        device = topk_idx.device
        empty_i = torch.empty((0,), device=device, dtype=torch.int64)
        offsets = torch.zeros((e_local + 1,), device=device, dtype=torch.int64)
        return empty_i, empty_i, torch.empty((0,), device=device, dtype=torch.float32), offsets, empty_i

    local_experts = (topk_idx[token_ids, slot_ids] - local_start).to(torch.int64)
    route_w = weights[token_ids, topk_idx[token_ids, slot_ids]].float()

    order = torch.argsort(local_experts, stable=True)
    sorted_tokens = token_ids[order].to(torch.int64)
    sorted_experts = local_experts[order]
    sorted_weights = route_w[order]

    counts = torch.bincount(sorted_experts, minlength=e_local)
    offsets = torch.zeros((e_local + 1,), device=topk_idx.device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    active_experts = torch.nonzero(counts > 0, as_tuple=False).flatten().to(torch.int64)
    return sorted_tokens, sorted_experts, sorted_weights, offsets, active_experts


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    a = hidden_states.index_select(0, token_ids)
    a = a.view(-1, HIDDEN // BLOCK, BLOCK).float()
    s = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous()
    return (a * s.unsqueeze(-1)).reshape(-1, HIDDEN)


def _dequant_w1_expert(gemm1_weights, gemm1_weights_scale, e):
    w = gemm1_weights[e].view((2 * INTER) // BLOCK, BLOCK, HIDDEN // BLOCK, BLOCK).float()
    s = gemm1_weights_scale[e].float()
    return (w * s[:, None, :, None]).reshape(2 * INTER, HIDDEN)


def _dequant_w2_expert(gemm2_weights, gemm2_weights_scale, e):
    w = gemm2_weights[e].view(HIDDEN // BLOCK, BLOCK, INTER // BLOCK, BLOCK).float()
    s = gemm2_weights_scale[e].float()
    return (w * s[:, None, :, None]).reshape(HIDDEN, INTER)


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

    topk_idx, weights = _route_noaux(routing_logits, routing_bias, routed_scaling_factor)
    sorted_tokens, _, sorted_weights, expert_offsets, active_experts = _build_local_dispatch(
        topk_idx, weights, local_expert_offset, gemm1_weights.shape[0]
    )

    if sorted_tokens.numel() == 0:
        return torch.zeros((T, HIDDEN), device=device, dtype=torch.bfloat16)

    output = torch.zeros((T, HIDDEN), device=device, dtype=torch.float32)

    w1_cache = {}
    w2_cache = {}

    for e in active_experts.tolist():
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        token_ids = sorted_tokens[start:end]
        route_w = sorted_weights[start:end]

        a = _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids)

        if e not in w1_cache:
            w1_cache[e] = _dequant_w1_expert(gemm1_weights, gemm1_weights_scale, e)
        if e not in w2_cache:
            w2_cache[e] = _dequant_w2_expert(gemm2_weights, gemm2_weights_scale, e)

        w1 = w1_cache[e]
        w2 = w2_cache[e]

        g1 = a @ w1.t()
        x1 = g1[:, :INTER]
        x2 = g1[:, INTER:]
        swiglu = F.silu(x2) * x1
        o = swiglu @ w2.t()
        o.mul_(route_w.unsqueeze(1))
        output.index_add_(0, token_ids, o)

    return output.to(torch.bfloat16)