import torch
import torch.nn.functional as F

H = 7168
I = 2048
E_LOCAL = 32
BLOCK = 128
H_BLKS = H // BLOCK          # 56
I_BLKS = I // BLOCK          # 16
O_BLKS = (2 * I) // BLOCK    # 32


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    # hidden_states: [T, H] fp8
    # hidden_states_scale: [H//128, T]
    x = hidden_states.index_select(0, token_ids).to(torch.float32).view(-1, H_BLKS, BLOCK)
    s = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous().view(-1, H_BLKS, 1)
    return (x * s).reshape(-1, H)


def _dequant_gemm1_weight_one(gemm1_weights, gemm1_weights_scale, e):
    # gemm1_weights[e]: [2I, H] fp8
    # gemm1_weights_scale[e]: [2I//128, H//128]
    # scale applies to [128,128] tiles
    w = gemm1_weights[e].to(torch.float32).view(O_BLKS, BLOCK, H_BLKS, BLOCK)
    s = gemm1_weights_scale[e].to(torch.float32).view(O_BLKS, 1, H_BLKS, 1)
    return (w * s).reshape(2 * I, H)


def _dequant_gemm2_weight_one(gemm2_weights, gemm2_weights_scale, e):
    # gemm2_weights[e]: [H, I] fp8
    # gemm2_weights_scale[e]: [H//128, I//128]
    w = gemm2_weights[e].to(torch.float32).view(H_BLKS, BLOCK, I_BLKS, BLOCK)
    s = gemm2_weights_scale[e].to(torch.float32).view(H_BLKS, 1, I_BLKS, 1)
    return (w * s).reshape(H, I)


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
    E_global = routing_logits.shape[1]
    E_local = gemm1_weights.shape[0]

    # 1) Exact DeepSeek no-aux routing
    logits = routing_logits.to(torch.float32)
    base_scores = torch.sigmoid(logits)
    biased_scores = base_scores + routing_bias.to(torch.float32).view(1, E_global)

    n_group = 8
    topk_group = 4
    topk_expert = 8
    group_size = E_global // n_group

    grouped = biased_scores.view(T, n_group, group_size)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)
    selected_groups = torch.topk(group_scores, k=topk_group, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, n_group), dtype=torch.bool, device=device)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, n_group, group_size).reshape(T, E_global)

    neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
    pruned = torch.where(expert_mask, biased_scores, neg_inf)
    topk_idx = torch.topk(pruned, k=topk_expert, dim=1, largest=True, sorted=True).indices

    selected_base = torch.gather(base_scores, 1, topk_idx)
    selected_weights = selected_base / (selected_base.sum(dim=1, keepdim=True) + 1e-20)
    selected_weights = selected_weights * float(routed_scaling_factor)

    # 2) Local expert dispatch
    local_start = int(local_expert_offset)
    local_end = local_start + E_local
    is_local = (topk_idx >= local_start) & (topk_idx < local_end)

    flat_token_idx, flat_slot_idx = torch.nonzero(is_local, as_tuple=True)
    if flat_token_idx.numel() == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    flat_global_experts = topk_idx[flat_token_idx, flat_slot_idx]
    flat_local_experts = (flat_global_experts - local_start).to(torch.int64)
    flat_weights = selected_weights[flat_token_idx, flat_slot_idx].to(torch.float32)

    order = torch.argsort(flat_local_experts)
    sorted_tokens = flat_token_idx[order].to(torch.int64)
    sorted_experts = flat_local_experts[order]
    sorted_weights = flat_weights[order]

    counts = torch.bincount(sorted_experts, minlength=E_local)
    offsets = torch.zeros(E_local + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)

    # 3) Per-expert exact active-slice computation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    w1_cache = [None] * E_local
    w2_cache = [None] * E_local

    for e in range(E_local):
        start = int(offsets[e].item())
        end = int(offsets[e + 1].item())
        if start == end:
            continue

        tok = sorted_tokens[start:end]
        rw = sorted_weights[start:end]

        a_e = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok)

        w1_e = w1_cache[e]
        if w1_e is None:
            w1_e = _dequant_gemm1_weight_one(gemm1_weights, gemm1_weights_scale, e)
            w1_cache[e] = w1_e

        g1 = torch.mm(a_e, w1_e.t())
        x1 = g1[:, :I]
        x2 = g1[:, I:]
        c = F.silu(x2) * x1

        w2_e = w2_cache[e]
        if w2_e is None:
            w2_e = _dequant_gemm2_weight_one(gemm2_weights, gemm2_weights_scale, e)
            w2_cache[e] = w2_e

        o_e = torch.mm(c, w2_e.t())
        o_e.mul_(rw.view(-1, 1))
        output.index_add_(0, tok, o_e)

    return output.to(torch.bfloat16)