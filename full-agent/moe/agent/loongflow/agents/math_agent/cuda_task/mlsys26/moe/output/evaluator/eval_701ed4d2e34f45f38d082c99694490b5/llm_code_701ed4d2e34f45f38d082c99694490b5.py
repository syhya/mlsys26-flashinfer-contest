import torch
import torch.nn.functional as F

H_DIM = 7168
I_DIM = 2048
BLOCK = 128
N_GROUP = 8
TOPK_GROUP = 4
TOP_K = 8


def _route_noaux(routing_logits, routing_bias, routed_scaling_factor):
    logits = routing_logits.float()
    bias = routing_bias.float().view(1, -1)

    s = torch.sigmoid(logits)
    s_bias = s + bias

    t, e_global = s.shape
    group_size = e_global // N_GROUP

    grouped = s_bias.view(t, N_GROUP, group_size)
    top2 = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((t, N_GROUP), dtype=torch.bool, device=s.device)
    group_mask.scatter_(1, top_groups, True)

    score_mask = group_mask.unsqueeze(-1).expand(t, N_GROUP, group_size).reshape(t, e_global)
    pruned = s_bias.masked_fill(~score_mask, float("-inf"))
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    selected = torch.zeros_like(s)
    selected.scatter_(1, topk_idx, 1.0)
    weights = s * selected
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * float(routed_scaling_factor)
    return topk_idx, weights


def _build_local_dispatch(topk_idx, weights, local_expert_offset, e_local):
    local_start = int(local_expert_offset)
    local_end = local_start + int(e_local)

    mask = (topk_idx >= local_start) & (topk_idx < local_end)
    token_ids, slot_ids = torch.nonzero(mask, as_tuple=True)

    if token_ids.numel() == 0:
        device = topk_idx.device
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        empty_f32 = torch.empty((0,), dtype=torch.float32, device=device)
        offsets = torch.zeros((e_local + 1,), dtype=torch.int32, device=device)
        active = torch.empty((0,), dtype=torch.int64, device=device)
        return empty_i32, empty_i32, empty_f32, offsets, active

    local_experts = (topk_idx[token_ids, slot_ids] - local_start).to(torch.int32)
    route_weights = weights[token_ids, topk_idx[token_ids, slot_ids]].float()

    order = torch.argsort(local_experts)
    sorted_tokens = token_ids[order].to(torch.int32)
    sorted_local_experts = local_experts[order]
    sorted_route_weights = route_weights[order]

    counts = torch.bincount(sorted_local_experts.to(torch.int64), minlength=e_local)
    offsets = torch.zeros((e_local + 1,), dtype=torch.int32, device=topk_idx.device)
    offsets[1:] = torch.cumsum(counts.to(torch.int32), dim=0)
    active = torch.nonzero(counts > 0, as_tuple=False).flatten()

    return sorted_tokens, sorted_local_experts, sorted_route_weights, offsets, active


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    if token_ids.numel() == 0:
        return torch.empty((0, H_DIM), dtype=torch.float32, device=hidden_states.device)
    rows = hidden_states[token_ids.long()]
    scales = hidden_states_scale[:, token_ids.long()].transpose(0, 1).contiguous()
    return (rows.float().view(-1, H_DIM // BLOCK, BLOCK) * scales.unsqueeze(-1)).reshape(-1, H_DIM)


def _dequant_w1_expert(gemm1_weights, gemm1_weights_scale, e):
    w = gemm1_weights[e].float().view((2 * I_DIM) // BLOCK, BLOCK, H_DIM // BLOCK, BLOCK)
    s = gemm1_weights_scale[e].float().view((2 * I_DIM) // BLOCK, H_DIM // BLOCK)
    return (w * s[:, None, :, None]).reshape(2 * I_DIM, H_DIM)


def _dequant_w2_expert(gemm2_weights, gemm2_weights_scale, e):
    w = gemm2_weights[e].float().view(H_DIM // BLOCK, BLOCK, I_DIM // BLOCK, BLOCK)
    s = gemm2_weights_scale[e].float().view(H_DIM // BLOCK, I_DIM // BLOCK)
    return (w * s[:, None, :, None]).reshape(H_DIM, I_DIM)


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
    t = routing_logits.shape[0]
    e_local = gemm1_weights.shape[0]

    topk_idx, weights = _route_noaux(routing_logits, routing_bias, routed_scaling_factor)
    sorted_tokens, _, sorted_route_weights, expert_offsets, active_experts = _build_local_dispatch(
        topk_idx, weights, local_expert_offset, e_local
    )

    if sorted_tokens.numel() == 0:
        return torch.zeros((t, H_DIM), dtype=torch.bfloat16, device=device)

    output = torch.zeros((t, H_DIM), dtype=torch.float32, device=device)

    deq_a_cache = {}
    w1_cache = {}
    w2_cache = {}

    for e in active_experts.tolist():
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        token_ids = sorted_tokens[start:end].long()
        route_w = sorted_route_weights[start:end]

        key = (int(token_ids[0].item()), int(token_ids.numel()), start, end)
        if key in deq_a_cache:
            a = deq_a_cache[key]
        else:
            a = _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids)
            deq_a_cache[key] = a

        if e not in w1_cache:
            w1_cache[e] = _dequant_w1_expert(gemm1_weights, gemm1_weights_scale, e)
        if e not in w2_cache:
            w2_cache[e] = _dequant_w2_expert(gemm2_weights, gemm2_weights_scale, e)

        w1 = w1_cache[e]
        w2 = w2_cache[e]

        g1 = torch.mm(a, w1.t())
        x1 = g1[:, :I_DIM]
        x2 = g1[:, I_DIM:]
        swiglu = F.silu(x2) * x1
        o = torch.mm(swiglu, w2.t())
        o.mul_(route_w.unsqueeze(1))
        output.index_add_(0, token_ids, o)

    return output.to(torch.bfloat16)