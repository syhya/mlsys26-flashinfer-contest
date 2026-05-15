import torch
import torch.nn.functional as F

HIDDEN = 7168
INTER = 2048
BLOCK = 128
N_GROUP = 8
TOPK_GROUP = 4
TOP_K = 8
LOCAL_EXPERTS = 32


def _expand_scales_2d(scale_2d: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return scale_2d.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)[:rows, :cols]


def _dequant_hidden_rows(hidden_states: torch.Tensor, hidden_states_scale: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    a_fp32 = hidden_states.index_select(0, token_ids).to(torch.float32)
    scales = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous()
    scales = scales.repeat_interleave(BLOCK, dim=1)[:, :HIDDEN]
    return a_fp32 * scales


def _dequant_gemm1_weight_full(gemm1_w_e: torch.Tensor, gemm1_s_e: torch.Tensor) -> torch.Tensor:
    w_fp32 = gemm1_w_e.to(torch.float32)
    scales = _expand_scales_2d(gemm1_s_e, 2 * INTER, HIDDEN)
    return w_fp32 * scales


def _dequant_gemm2_weight_full(gemm2_w_e: torch.Tensor, gemm2_s_e: torch.Tensor) -> torch.Tensor:
    w_fp32 = gemm2_w_e.to(torch.float32)
    scales = _expand_scales_2d(gemm2_s_e, HIDDEN, INTER)
    return w_fp32 * scales


def _route(routing_logits: torch.Tensor, routing_bias: torch.Tensor, routed_scaling_factor: float):
    T, E = routing_logits.shape
    logits = routing_logits.to(torch.float32)
    s = torch.sigmoid(logits)
    s_bias = s + routing_bias.to(torch.float32).view(1, E)

    group_size = E // N_GROUP
    grouped = s_bias.view(T, N_GROUP, group_size)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)
    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=routing_logits.device)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, N_GROUP, group_size).reshape(T, E)

    neg_inf = torch.full((), float("-inf"), device=routing_logits.device, dtype=torch.float32)
    pruned = torch.where(expert_mask, s_bias, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    topk_scores = s.gather(1, topk_idx)
    weights = topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return topk_idx, weights


def _build_dispatch(topk_idx: torch.Tensor, topk_weights: torch.Tensor, local_expert_offset: int):
    local_mask = (topk_idx >= local_expert_offset) & (topk_idx < local_expert_offset + LOCAL_EXPERTS)
    flat_tok, flat_slot = torch.nonzero(local_mask, as_tuple=True)
    if flat_tok.numel() == 0:
        return None

    local_experts = (topk_idx[flat_tok, flat_slot] - local_expert_offset).to(torch.int64)
    route_weights = topk_weights[flat_tok, flat_slot].to(torch.float32)

    order = torch.argsort(local_experts)
    sorted_tok = flat_tok[order]
    sorted_exp = local_experts[order]
    sorted_w = route_weights[order]

    counts = torch.bincount(sorted_exp, minlength=LOCAL_EXPERTS)
    offsets = torch.empty(LOCAL_EXPERTS + 1, dtype=torch.int64, device=topk_idx.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return sorted_tok, sorted_exp, sorted_w, offsets


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
    T = hidden_states.shape[0]

    topk_idx, topk_weights = _route(routing_logits, routing_bias, routed_scaling_factor)
    dispatch = _build_dispatch(topk_idx, topk_weights, int(local_expert_offset))

    if dispatch is None:
        return torch.zeros((T, HIDDEN), device=device, dtype=torch.bfloat16)

    sorted_tok, _, sorted_w, offsets = dispatch
    out = torch.zeros((T, HIDDEN), device=device, dtype=torch.float32)

    # cache dequantized weights per expert lazily
    w1_cache = {}
    w2_cache = {}

    for e in range(LOCAL_EXPERTS):
        start = int(offsets[e].item())
        end = int(offsets[e + 1].item())
        if start == end:
            continue

        tok_e = sorted_tok[start:end]
        w_e = sorted_w[start:end]

        a_e = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok_e)

        if e not in w1_cache:
            w1_cache[e] = _dequant_gemm1_weight_full(gemm1_weights[e], gemm1_weights_scale[e])
        g1 = a_e @ w1_cache[e].transpose(0, 1)

        x1 = g1[:, :INTER]
        x2 = g1[:, INTER:]
        swiglu = F.silu(x2) * x1

        if e not in w2_cache:
            w2_cache[e] = _dequant_gemm2_weight_full(gemm2_weights[e], gemm2_weights_scale[e])
        y_e = swiglu @ w2_cache[e].transpose(0, 1)

        y_e = y_e * w_e.unsqueeze(1)
        out.index_add_(0, tok_e, y_e)

    return out.to(torch.bfloat16)