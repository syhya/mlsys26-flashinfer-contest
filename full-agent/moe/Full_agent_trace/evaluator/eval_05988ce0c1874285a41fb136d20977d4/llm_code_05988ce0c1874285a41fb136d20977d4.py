import torch
import torch.nn.functional as F


H_DIM = 7168
I_DIM = 2048
BLOCK = 128
N_GROUP = 8
TOPK_GROUP = 4
TOP_K = 8


def _route_tokens(routing_logits, routing_bias, routed_scaling_factor):
    logits = routing_logits.float()
    s = torch.sigmoid(logits)
    sb = s + routing_bias.float().view(1, -1)

    T, E = sb.shape
    group_size = E // N_GROUP

    sb_group = sb.view(T, N_GROUP, group_size)
    top2_vals = torch.topk(sb_group, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=sb.device)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, N_GROUP, group_size).reshape(T, E)

    pruned = sb.masked_fill(~expert_mask, float("-inf"))
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    topk_scores = torch.gather(s, 1, topk_idx)
    topk_weights = topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_idx, topk_weights


def _build_dispatch(topk_idx, topk_weights, local_expert_offset, e_local):
    mask = (topk_idx >= local_expert_offset) & (topk_idx < local_expert_offset + e_local)
    tok, slot = torch.nonzero(mask, as_tuple=True)
    if tok.numel() == 0:
        return None

    local_e = (topk_idx[tok, slot] - local_expert_offset).to(torch.int64)
    w = topk_weights[tok, slot].float()

    order = torch.argsort(local_e)
    tok = tok[order]
    local_e = local_e[order]
    w = w[order]

    counts = torch.bincount(local_e, minlength=e_local)
    offsets = torch.zeros(e_local + 1, dtype=torch.int64, device=topk_idx.device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return tok, local_e, w, offsets


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    a_fp32 = hidden_states.index_select(0, token_ids).float()
    scales = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous()
    scales = scales.repeat_interleave(BLOCK, dim=1)
    return a_fp32 * scales


def _dequant_w1_chunk(w_fp8_chunk, w_scale_chunk):
    s = w_scale_chunk.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return w_fp8_chunk.float() * s


def _dequant_w2_chunk(w_fp8_chunk, w_scale_chunk):
    s = w_scale_chunk.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return w_fp8_chunk.float() * s


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
    e_local = gemm1_weights.shape[0]

    topk_idx, topk_weights = _route_tokens(
        routing_logits, routing_bias, routed_scaling_factor
    )

    dispatch = _build_dispatch(topk_idx, topk_weights, local_expert_offset, e_local)
    if dispatch is None:
        return torch.zeros((T, H_DIM), dtype=torch.bfloat16, device=device)

    sorted_tokens, _, sorted_weights, expert_offsets = dispatch
    output = torch.zeros((T, H_DIM), dtype=torch.float32, device=device)

    n_chunk = 512
    h_chunk = 512

    for e in range(e_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        tok_e = sorted_tokens[start:end]
        w_route = sorted_weights[start:end]
        a = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok_e)
        tk = a.shape[0]

        g1 = torch.empty((tk, 2 * I_DIM), dtype=torch.float32, device=device)
        w1_e = gemm1_weights[e]
        w1s_e = gemm1_weights_scale[e]

        for n0 in range(0, 2 * I_DIM, n_chunk):
            n1 = min(n0 + n_chunk, 2 * I_DIM)
            bn0 = n0 // BLOCK
            bn1 = n1 // BLOCK
            w1_chunk = _dequant_w1_chunk(
                w1_e[n0:n1, :],
                w1s_e[bn0:bn1, :],
            )
            g1[:, n0:n1] = a @ w1_chunk.t()

        x1 = g1[:, :I_DIM]
        x2 = g1[:, I_DIM:]
        s = F.silu(x2) * x1

        w2_e = gemm2_weights[e]
        w2s_e = gemm2_weights_scale[e]

        for h0 in range(0, H_DIM, h_chunk):
            h1 = min(h0 + h_chunk, H_DIM)
            bh0 = h0 // BLOCK
            bh1 = h1 // BLOCK
            w2_chunk = _dequant_w2_chunk(
                w2_e[h0:h1, :],
                w2s_e[bh0:bh1, :],
            )
            y = s @ w2_chunk.t()
            y.mul_(w_route[:, None])
            output[:, h0:h1].index_add_(0, tok_e, y)

    return output.to(torch.bfloat16)