import torch
import torch.nn.functional as F

H_DIM = 7168
I_DIM = 2048
BLOCK = 128
N_GROUP = 8
TOPK_GROUP = 4
TOP_K = 8
E_LOCAL = 32


def _expand_scales_2d(scales: torch.Tensor, rep0: int = BLOCK, rep1: int = BLOCK) -> torch.Tensor:
    return scales.repeat_interleave(rep0, dim=0).repeat_interleave(rep1, dim=1)


def _route_tokens(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    logits = routing_logits.float()
    s = torch.sigmoid(logits)
    s_with_bias = s + routing_bias.float().view(1, -1)

    t, e_global = s.shape
    group_size = e_global // N_GROUP

    grouped = s_with_bias.view(t, N_GROUP, group_size)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((t, N_GROUP), device=s.device, dtype=torch.bool)
    group_mask.scatter_(1, top_groups, True)

    expert_mask = group_mask.unsqueeze(-1).expand(t, N_GROUP, group_size).reshape(t, e_global)
    neg_inf = torch.full((), float("-inf"), device=s.device, dtype=s.dtype)
    pruned = torch.where(expert_mask, s_with_bias, neg_inf)

    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    topk_scores = torch.gather(s, 1, topk_idx)
    topk_scores = topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20)
    topk_scores = topk_scores * routed_scaling_factor
    return topk_idx, topk_scores


def _build_dispatch(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
):
    local_mask = (topk_idx >= local_expert_offset) & (topk_idx < local_expert_offset + E_LOCAL)
    token_idx, slot_idx = torch.nonzero(local_mask, as_tuple=True)
    if token_idx.numel() == 0:
        return None

    local_expert_idx = (topk_idx[token_idx, slot_idx] - local_expert_offset).to(torch.int64)
    weights = topk_weights[token_idx, slot_idx].float()

    order = torch.argsort(local_expert_idx)
    token_idx = token_idx[order]
    local_expert_idx = local_expert_idx[order]
    weights = weights[order]

    counts = torch.bincount(local_expert_idx, minlength=E_LOCAL)
    offsets = torch.empty(E_LOCAL + 1, device=topk_idx.device, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return token_idx, local_expert_idx, weights, offsets


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    a_fp32 = hidden_states.index_select(0, token_ids).float()
    scales = hidden_states_scale.index_select(1, token_ids).t().contiguous()
    scales = scales.repeat_interleave(BLOCK, dim=1)
    return a_fp32 * scales


def _dequant_gemm1_weight_full(w_fp8, w_scale):
    scales = _expand_scales_2d(w_scale)
    return w_fp8.float() * scales


def _dequant_gemm2_weight_rows(w_fp8_rows, w_scale_rows):
    scales = w_scale_rows.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return w_fp8_rows.float() * scales


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

    topk_idx, topk_weights = _route_tokens(
        routing_logits,
        routing_bias,
        routed_scaling_factor,
    )

    dispatch = _build_dispatch(topk_idx, topk_weights, int(local_expert_offset))
    if dispatch is None:
        return torch.zeros((t, H_DIM), device=device, dtype=torch.bfloat16)

    sorted_tokens, _, sorted_weights, expert_offsets = dispatch
    output = torch.zeros((t, H_DIM), device=device, dtype=torch.float32)

    # correctness-first implementation:
    # exact routing/dispatch, exact block dequant, fp32 GEMMs, bf16 cast only at the end.
    for e in range(E_LOCAL):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        tok_e = sorted_tokens[start:end]
        w_route = sorted_weights[start:end]

        a = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok_e)

        w1 = _dequant_gemm1_weight_full(gemm1_weights[e], gemm1_weights_scale[e])
        g1 = a @ w1.t()

        x1 = g1[:, :I_DIM]
        x2 = g1[:, I_DIM:]
        swiglu = F.silu(x2) * x1

        y = torch.zeros((tok_e.numel(), H_DIM), device=device, dtype=torch.float32)

        # Chunk GEMM2 over output rows to reduce peak memory.
        for h0 in range(0, H_DIM, 512):
            h1 = min(h0 + 512, H_DIM)
            row_blk0 = h0 // BLOCK
            row_blk1 = (h1 + BLOCK - 1) // BLOCK

            w2_rows_fp8 = gemm2_weights[e, h0:h1, :]
            w2_scales_rows = gemm2_weights_scale[e, row_blk0:row_blk1, :]
            w2_rows = _dequant_gemm2_weight_rows(w2_rows_fp8, w2_scales_rows)[: h1 - h0, :]
            y[:, h0:h1] = swiglu @ w2_rows.t()

        y.mul_(w_route.unsqueeze(1))
        output.index_add_(0, tok_e, y)

    return output.to(torch.bfloat16)