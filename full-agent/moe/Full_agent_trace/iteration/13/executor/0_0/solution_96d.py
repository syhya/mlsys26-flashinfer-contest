import torch
import torch.nn.functional as F


H = 7168
I = 2048
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = 256 // N_GROUP
BLOCK = 128


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    # hidden_states: [T, 7168] fp8
    # hidden_states_scale: [56, T] fp32
    rows = hidden_states.index_select(0, token_ids).float().view(-1, H // BLOCK, BLOCK)
    scales = hidden_states_scale[:, token_ids].transpose(0, 1).contiguous().view(-1, H // BLOCK, 1)
    return (rows * scales).view(-1, H)


def _dequant_gemm1_weight_one(gemm1_weights, gemm1_weights_scale, expert_idx):
    # gemm1_weights[e]: [4096, 7168] fp8
    # gemm1_weights_scale[e]: [32, 56] fp32
    w = gemm1_weights[expert_idx].float().view((2 * I) // BLOCK, BLOCK, H // BLOCK, BLOCK)
    s = gemm1_weights_scale[expert_idx].float().view((2 * I) // BLOCK, H // BLOCK, 1, 1)
    return (w * s).permute(0, 2, 1, 3).reshape(2 * I, H)


def _dequant_gemm2_weight_one(gemm2_weights, gemm2_weights_scale, expert_idx):
    # gemm2_weights[e]: [7168, 2048] fp8
    # gemm2_weights_scale[e]: [56, 16] fp32
    w = gemm2_weights[expert_idx].float().view(H // BLOCK, BLOCK, I // BLOCK, BLOCK)
    s = gemm2_weights_scale[expert_idx].float().view(H // BLOCK, I // BLOCK, 1, 1)
    return (w * s).permute(0, 2, 1, 3).reshape(H, I)


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
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]

    # 1) Exact DeepSeek routing
    scores = torch.sigmoid(routing_logits.float())
    scores_bias = scores + routing_bias.float().view(1, -1)

    grouped = scores_bias.view(T, N_GROUP, GROUP_SIZE)
    top2 = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2.sum(dim=2)
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, group_idx, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_global)

    neg_inf = torch.full((), float("-inf"), device=device, dtype=torch.float32)
    pruned = torch.where(expert_mask, scores_bias, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    selected_scores = torch.gather(scores, 1, topk_idx)
    selected_weights = selected_scores / (selected_scores.sum(dim=1, keepdim=True) + 1e-20)
    selected_weights = selected_weights * float(routed_scaling_factor)

    # 2) Keep only local experts and build CSR-like grouping
    local_start = int(local_expert_offset)
    local_end = local_start + E_local
    is_local = (topk_idx >= local_start) & (topk_idx < local_end)

    flat_token_idx, flat_slot_idx = torch.nonzero(is_local, as_tuple=True)
    Tk_total = flat_token_idx.numel()
    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    flat_global_experts = topk_idx[flat_token_idx, flat_slot_idx]
    flat_local_experts = (flat_global_experts - local_start).to(torch.int64)
    flat_weights = selected_weights[flat_token_idx, flat_slot_idx].float()

    order = torch.argsort(flat_local_experts)
    sorted_tokens = flat_token_idx[order].to(torch.int64)
    sorted_experts = flat_local_experts[order]
    sorted_weights = flat_weights[order]

    counts = torch.bincount(sorted_experts, minlength=E_local)
    offsets = torch.zeros(E_local + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)

    # 3) Exact active-slice MoE compute
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    # Lazy weight cache only for active experts
    w1_cache = [None] * E_local
    w2_cache = [None] * E_local

    for e in range(E_local):
        start = int(offsets[e].item())
        end = int(offsets[e + 1].item())
        if start == end:
            continue

        tok = sorted_tokens[start:end]
        rw = sorted_weights[start:end]

        a = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok)

        w1 = w1_cache[e]
        if w1 is None:
            w1 = _dequant_gemm1_weight_one(gemm1_weights, gemm1_weights_scale, e)
            w1_cache[e] = w1

        g1 = torch.mm(a, w1.t())
        x1 = g1[:, :I]
        x2 = g1[:, I:]
        c = F.silu(x2) * x1

        w2 = w2_cache[e]
        if w2 is None:
            w2 = _dequant_gemm2_weight_one(gemm2_weights, gemm2_weights_scale, e)
            w2_cache[e] = w2

        o = torch.mm(c, w2.t())
        o.mul_(rw.view(-1, 1))
        output.index_add_(0, tok, o)

    return output.to(torch.bfloat16)