import torch
import torch.nn.functional as F

# Problem constants
_H = 7168
_I = 2048
_E_LOCAL = 32
_BLOCK = 128
_HB = _H // _BLOCK          # 56
_I2 = 2 * _I                # 4096
_I2B = _I2 // _BLOCK        # 32
_IB = _I // _BLOCK          # 16


def _dequant_hidden_rows(hidden_states: torch.Tensor,
                         hidden_states_scale: torch.Tensor,
                         token_ids: torch.Tensor) -> torch.Tensor:
    # hidden_states: [T, 7168] fp8
    # hidden_states_scale: [56, T] fp32
    rows = hidden_states.index_select(0, token_ids).to(torch.float32).view(-1, _HB, _BLOCK)
    scales = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous().unsqueeze(-1)
    return (rows * scales).view(-1, _H)


def _dequant_gemm1_weight_one(gemm1_weights: torch.Tensor,
                              gemm1_weights_scale: torch.Tensor,
                              e: int) -> torch.Tensor:
    # weights[e]: [4096, 7168], scales[e]: [32, 56]
    # Block layout: [32, 128, 56, 128] -> apply [32, 1, 56, 1] -> reshape [4096, 7168]
    w = gemm1_weights[e].to(torch.float32).view(_I2B, _BLOCK, _HB, _BLOCK)
    s = gemm1_weights_scale[e].to(torch.float32).view(_I2B, 1, _HB, 1)
    return (w * s).reshape(_I2, _H)


def _dequant_gemm2_weight_one(gemm2_weights: torch.Tensor,
                              gemm2_weights_scale: torch.Tensor,
                              e: int) -> torch.Tensor:
    # weights[e]: [7168, 2048], scales[e]: [56, 16]
    # Block layout: [56, 128, 16, 128] -> apply [56, 1, 16, 1] -> reshape [7168, 2048]
    w = gemm2_weights[e].to(torch.float32).view(_HB, _BLOCK, _IB, _BLOCK)
    s = gemm2_weights_scale[e].to(torch.float32).view(_HB, 1, _IB, 1)
    return (w * s).reshape(_H, _I)


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,        # [T, 256] float32
    routing_bias: torch.Tensor,          # [256]    bfloat16
    hidden_states: torch.Tensor,         # [T, 7168] fp8
    hidden_states_scale: torch.Tensor,   # [56, T]  float32
    gemm1_weights: torch.Tensor,         # [32, 4096, 7168] fp8
    gemm1_weights_scale: torch.Tensor,   # [32, 32, 56] float32
    gemm2_weights: torch.Tensor,         # [32, 7168, 2048] fp8
    gemm2_weights_scale: torch.Tensor,   # [32, 56, 16] float32
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_global = routing_logits.shape[1]
    E_local = gemm1_weights.shape[0]

    # ------------------------------------------------------------------
    # 1) Exact DeepSeek no-aux routing
    # ------------------------------------------------------------------
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32)

    scores = torch.sigmoid(logits)
    scores_bias = scores + bias

    N_GROUP = 8
    GROUP_SIZE = E_global // N_GROUP
    TOPK_GROUP = 4
    TOP_K = 8

    grouped = scores_bias.view(T, N_GROUP, GROUP_SIZE)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)
    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)
    expert_mask_allowed = group_mask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    pruned = scores_bias.masked_fill(~expert_mask_allowed, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    selected = torch.zeros_like(scores)
    selected.scatter_(1, topk_idx, scores.gather(1, topk_idx))
    denom = selected.sum(dim=1, keepdim=True).clamp_min(1e-20)
    weights = selected / denom
    weights.mul_(float(routed_scaling_factor))

    # ------------------------------------------------------------------
    # 2) Dispatch local experts on GPU
    # ------------------------------------------------------------------
    local_start = int(local_expert_offset)
    local_end = local_start + E_local
    is_local = (topk_idx >= local_start) & (topk_idx < local_end)

    flat_token_idx, flat_slot_idx = torch.nonzero(is_local, as_tuple=True)
    Tk_total = flat_token_idx.numel()
    if Tk_total == 0:
        return torch.zeros((T, _H), dtype=torch.bfloat16, device=device)

    local_expert_idx = topk_idx[flat_token_idx, flat_slot_idx] - local_start
    route_w = weights[flat_token_idx, topk_idx[flat_token_idx, flat_slot_idx]].to(torch.float32)

    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.long)
    sorted_experts = local_expert_idx[order].to(torch.long)
    sorted_route_w = route_w[order]

    counts = torch.bincount(sorted_experts, minlength=E_local)
    offsets = torch.empty(E_local + 1, dtype=torch.long, device=device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)

    # ------------------------------------------------------------------
    # 3) Exact active-slice MoE compute in FP32
    #    Cache dequantized weights lazily per expert.
    # ------------------------------------------------------------------
    output = torch.zeros((T, _H), dtype=torch.float32, device=device)
    w1_cache = [None] * E_local
    w2_cache = [None] * E_local

    for e in range(E_local):
        start = int(offsets[e].item())
        end = int(offsets[e + 1].item())
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
        x1 = g1[:, :_I]
        x2 = g1[:, _I:]
        c = F.silu(x2) * x1

        w2 = w2_cache[e]
        if w2 is None:
            w2 = _dequant_gemm2_weight_one(gemm2_weights, gemm2_weights_scale, e)
            w2_cache[e] = w2

        o = torch.mm(c, w2.t())
        o.mul_(rw.unsqueeze(1))
        output.index_add_(0, tok, o)

    return output.to(torch.bfloat16)