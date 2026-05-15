import torch
import torch.nn.functional as F


HIDDEN = 7168
INTERMEDIATE = 2048
LOCAL_EXPERTS = 32
GLOBAL_EXPERTS = 256
GROUPS = 8
GROUP_SIZE = GLOBAL_EXPERTS // GROUPS
TOP_GROUPS = 4
TOP_EXPERTS = 8
BLOCK = 128


def _route_tokens(routing_logits, routing_bias, routed_scaling_factor):
    logits = routing_logits.float()
    s = torch.sigmoid(logits)
    s_bias = s + routing_bias.float().view(1, -1)

    grouped = s_bias.view(-1, GROUPS, GROUP_SIZE)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOP_GROUPS, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, top_groups, True)

    expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, GROUP_SIZE).reshape(-1, GLOBAL_EXPERTS)
    neg_inf = torch.full((), float("-inf"), device=logits.device, dtype=torch.float32)
    pruned = torch.where(expert_mask, s_bias, neg_inf)

    topk_idx = torch.topk(pruned, k=TOP_EXPERTS, dim=1, largest=True, sorted=True).indices
    topk_scores = s.gather(1, topk_idx)
    weights = topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * float(routed_scaling_factor)
    return topk_idx, weights


def _build_dispatch(topk_idx, weights, local_expert_offset):
    local_mask = (topk_idx >= local_expert_offset) & (topk_idx < local_expert_offset + LOCAL_EXPERTS)
    tok, slot = torch.nonzero(local_mask, as_tuple=True)
    if tok.numel() == 0:
        return None

    local_experts = (topk_idx[tok, slot] - local_expert_offset).to(torch.int64)
    dispatch_weights = weights[tok, slot].float()

    order = torch.argsort(local_experts)
    sorted_tokens = tok[order]
    sorted_experts = local_experts[order]
    sorted_weights = dispatch_weights[order]

    counts = torch.bincount(sorted_experts, minlength=LOCAL_EXPERTS)
    offsets = torch.empty(LOCAL_EXPERTS + 1, device=topk_idx.device, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return sorted_tokens, sorted_experts, sorted_weights, offsets


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    a_fp32 = hidden_states.index_select(0, token_ids).float()
    scales = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous()
    scales = scales.repeat_interleave(BLOCK, dim=1)
    return a_fp32 * scales


def _dequant_w1_full(w_fp8, w_scale):
    w_fp32 = w_fp8.float()
    scales = w_scale.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return w_fp32 * scales


def _dequant_w2_rows_chunk(w_fp8_rows, w_scale_rows):
    w_fp32 = w_fp8_rows.float()
    scales = w_scale_rows.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return w_fp32 * scales


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

    topk_idx, weights = _route_tokens(routing_logits, routing_bias, routed_scaling_factor)
    dispatch = _build_dispatch(topk_idx, weights, int(local_expert_offset))

    output = torch.zeros((T, HIDDEN), device=device, dtype=torch.float32)
    if dispatch is None:
        return output.to(torch.bfloat16)

    sorted_tokens, _, sorted_weights, offsets = dispatch

    for e in range(LOCAL_EXPERTS):
        start = int(offsets[e].item())
        end = int(offsets[e + 1].item())
        if start == end:
            continue

        tok_e = sorted_tokens[start:end]
        w_e = sorted_weights[start:end]

        a_e = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok_e)

        w1_e = _dequant_w1_full(gemm1_weights[e], gemm1_weights_scale[e])
        g1 = a_e @ w1_e.t()
        x1 = g1[:, :INTERMEDIATE]
        x2 = g1[:, INTERMEDIATE:]
        act = F.silu(x2) * x1

        row_chunk_blocks = 8
        row_chunk = row_chunk_blocks * BLOCK
        for h0 in range(0, HIDDEN, row_chunk):
            h1 = min(h0 + row_chunk, HIDDEN)
            sb0 = h0 // BLOCK
            sb1 = (h1 + BLOCK - 1) // BLOCK
            w2_chunk = _dequant_w2_rows_chunk(
                gemm2_weights[e, h0:h1, :],
                gemm2_weights_scale[e, sb0:sb1, :],
            )
            y_chunk = act @ w2_chunk.t()
            y_chunk.mul_(w_e[:, None])
            output[:, h0:h1].index_add_(0, tok_e, y_chunk)

    return output.to(torch.bfloat16)