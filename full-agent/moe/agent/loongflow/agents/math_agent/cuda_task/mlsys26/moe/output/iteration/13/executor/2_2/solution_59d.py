import torch
import torch.nn.functional as F


H = 7168
I = 2048
BLOCK = 128
H_BLKS = H // BLOCK          # 56
I2_BLKS = (2 * I) // BLOCK   # 32
I_BLKS = I // BLOCK          # 16
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = 256 // N_GROUP


def _dequant_hidden_rows(hidden_states: torch.Tensor,
                         hidden_states_scale: torch.Tensor,
                         token_ids: torch.Tensor) -> torch.Tensor:
    # hidden_states: [T, 7168] fp8
    # hidden_states_scale: [56, T]
    x = hidden_states.index_select(0, token_ids).float().view(-1, H_BLKS, BLOCK)
    s = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous().view(-1, H_BLKS, 1)
    return (x * s).view(-1, H)


def _dequant_gemm1_weight_one(gemm1_weights: torch.Tensor,
                              gemm1_weights_scale: torch.Tensor,
                              e: int) -> torch.Tensor:
    # gemm1_weights[e]: [4096, 7168] fp8
    # gemm1_weights_scale[e]: [32, 56]
    # scale[n_blk, k_blk]
    w = gemm1_weights[e].float().view(I2_BLKS, BLOCK, H_BLKS, BLOCK)
    s = gemm1_weights_scale[e].float().view(I2_BLKS, 1, H_BLKS, 1)
    return (w * s).reshape(2 * I, H)


def _dequant_gemm2_weight_one(gemm2_weights: torch.Tensor,
                              gemm2_weights_scale: torch.Tensor,
                              e: int) -> torch.Tensor:
    # gemm2_weights[e]: [7168, 2048] fp8
    # gemm2_weights_scale[e]: [56, 16]
    # scale[h_blk, i_blk]
    w = gemm2_weights[e].float().view(H_BLKS, BLOCK, I_BLKS, BLOCK)
    s = gemm2_weights_scale[e].float().view(H_BLKS, 1, I_BLKS, 1)
    return (w * s).reshape(H, I)


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,        # [T, 256] float32
    routing_bias: torch.Tensor,          # [256]    bfloat16
    hidden_states: torch.Tensor,         # [T, 7168] float8_e4m3fn
    hidden_states_scale: torch.Tensor,   # [56, T]  float32
    gemm1_weights: torch.Tensor,         # [32, 4096, 7168] float8_e4m3fn
    gemm1_weights_scale: torch.Tensor,   # [32, 32, 56] float32
    gemm2_weights: torch.Tensor,         # [32, 7168, 2048] float8_e4m3fn
    gemm2_weights_scale: torch.Tensor,   # [32, 56, 16] float32
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]

    # 1) Exact DeepSeek no-aux routing
    scores = torch.sigmoid(routing_logits.float())
    scores_bias = scores + routing_bias.float().view(1, -1)

    grouped = scores_bias.view(T, N_GROUP, GROUP_SIZE)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, group_idx, True)
    expert_mask_groups = group_mask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_global)

    neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
    pruned = torch.where(expert_mask_groups, scores_bias, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    select_mask = torch.zeros((T, E_global), dtype=torch.bool, device=device)
    select_mask.scatter_(1, topk_idx, True)

    weights = torch.where(select_mask, scores, torch.zeros((), dtype=torch.float32, device=device))
    weights_sum = weights.sum(dim=1, keepdim=True)
    weights = weights / (weights_sum + 1e-20)
    weights = weights * float(routed_scaling_factor)

    # 2) Build local dispatch on GPU
    local_start = int(local_expert_offset)
    local_end = local_start + E_local

    local_mask = (topk_idx >= local_start) & (topk_idx < local_end)
    flat_token_idx, flat_slot_idx = torch.nonzero(local_mask, as_tuple=True)
    Tk_total = flat_token_idx.numel()

    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    local_expert_idx = (topk_idx[flat_token_idx, flat_slot_idx] - local_start).to(torch.int64)
    route_w = weights[flat_token_idx, topk_idx[flat_token_idx, flat_slot_idx]].float()

    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int64)
    sorted_experts = local_expert_idx[order]
    sorted_route_w = route_w[order]

    expert_counts = torch.bincount(sorted_experts, minlength=E_local)
    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    # 3) Exact active-slice computation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    # Tiny caches to avoid re-dequantizing experts if reused
    w1_cache = [None] * E_local
    w2_cache = [None] * E_local

    for e in range(E_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        tok = sorted_tokens[start:end]
        rw = sorted_route_w[start:end]

        a_e = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok)

        w1 = w1_cache[e]
        if w1 is None:
            w1 = _dequant_gemm1_weight_one(gemm1_weights, gemm1_weights_scale, e)
            w1_cache[e] = w1

        g1 = torch.mm(a_e, w1.t())
        x1 = g1[:, :I]
        x2 = g1[:, I:]
        c = F.silu(x2) * x1

        w2 = w2_cache[e]
        if w2 is None:
            w2 = _dequant_gemm2_weight_one(gemm2_weights, gemm2_weights_scale, e)
            w2_cache[e] = w2

        o = torch.mm(c, w2.t())
        o.mul_(rw[:, None])
        output.index_add_(0, tok, o)

        # reduce peak memory pressure
        del a_e, g1, x1, x2, c, o

    return output.to(torch.bfloat16)