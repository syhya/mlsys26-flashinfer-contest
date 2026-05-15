import torch
import torch.nn.functional as F

HIDDEN = 7168
INTERMEDIATE = 2048
BLOCK = 128
N_GROUP = 8
TOPK_GROUP = 4
TOPK_EXPERT = 8
E_LOCAL = 32


def _route(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    t, e_global = routing_logits.shape
    s = torch.sigmoid(routing_logits.to(torch.float32))
    sb = s + routing_bias.to(torch.float32).view(1, e_global)

    group_size = e_global // N_GROUP
    sbg = sb.view(t, N_GROUP, group_size)
    top2_vals = torch.topk(sbg, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((t, N_GROUP), device=routing_logits.device, dtype=torch.bool)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(t, N_GROUP, group_size).reshape(t, e_global)

    pruned = sb.masked_fill(~expert_mask, float("-inf"))
    topk_idx = torch.topk(pruned, k=TOPK_EXPERT, dim=1, largest=True, sorted=True).indices

    topk_w = s.gather(1, topk_idx)
    topk_w = topk_w / (topk_w.sum(dim=1, keepdim=True) + 1e-20)
    topk_w = topk_w * routed_scaling_factor
    return topk_idx, topk_w


def _build_dispatch(
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
    local_expert_offset: int,
    e_local: int,
):
    local_mask = (topk_idx >= local_expert_offset) & (topk_idx < local_expert_offset + e_local)
    tok, slot = torch.nonzero(local_mask, as_tuple=True)
    if tok.numel() == 0:
        device = topk_idx.device
        return (
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.float32),
            torch.zeros(e_local + 1, device=device, dtype=torch.long),
        )

    local_exp = (topk_idx[tok, slot] - local_expert_offset).to(torch.long)
    w = topk_w[tok, slot].to(torch.float32)

    order = torch.argsort(local_exp)
    tok = tok[order]
    local_exp = local_exp[order]
    w = w[order]

    counts = torch.bincount(local_exp, minlength=e_local)
    offsets = torch.empty(e_local + 1, device=topk_idx.device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return tok, local_exp, w, offsets


def _dequant_hidden_rows(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    a = hidden_states.index_select(0, token_ids).to(torch.float32)
    s = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous()
    s = s.repeat_interleave(BLOCK, dim=1)[:, :HIDDEN]
    return a * s


def _dequant_gemm1_chunk(
    expert_w: torch.Tensor,
    expert_scale: torch.Tensor,
    n0: int,
    n1: int,
) -> torch.Tensor:
    w = expert_w[n0:n1].to(torch.float32)
    sb0 = n0 // BLOCK
    sb1 = (n1 + BLOCK - 1) // BLOCK
    s = expert_scale[sb0:sb1]  # [nb, 56]
    s = s.repeat_interleave(BLOCK, dim=0)[: n1 - n0]
    s = s.repeat_interleave(BLOCK, dim=1)[:, :HIDDEN]
    return w * s


def _dequant_gemm2_chunk(
    expert_w: torch.Tensor,
    expert_scale: torch.Tensor,
    h0: int,
    h1: int,
) -> torch.Tensor:
    w = expert_w[h0:h1].to(torch.float32)
    sb0 = h0 // BLOCK
    sb1 = (h1 + BLOCK - 1) // BLOCK
    s = expert_scale[sb0:sb1]  # [hb, 16]
    s = s.repeat_interleave(BLOCK, dim=0)[: h1 - h0]
    s = s.repeat_interleave(BLOCK, dim=1)[:, :INTERMEDIATE]
    return w * s


def _gemm1_chunked(a_e: torch.Tensor, w1_e: torch.Tensor, s1_e: torch.Tensor, t_e: int) -> torch.Tensor:
    g1 = torch.empty((t_e, 2 * INTERMEDIATE), device=a_e.device, dtype=torch.float32)
    n_chunk = 1024 if t_e <= 32 else 512
    for n0 in range(0, 2 * INTERMEDIATE, n_chunk):
        n1 = min(n0 + n_chunk, 2 * INTERMEDIATE)
        w1_chunk = _dequant_gemm1_chunk(w1_e, s1_e, n0, n1)
        g1[:, n0:n1] = torch.mm(a_e, w1_chunk.t())
    return g1


def _gemm2_accum_chunked(
    out: torch.Tensor,
    tok_e: torch.Tensor,
    route_w: torch.Tensor,
    s_e: torch.Tensor,
    w2_e: torch.Tensor,
    s2_e: torch.Tensor,
):
    t_e = s_e.shape[0]
    h_chunk = 1024 if t_e <= 32 else 512
    route_w = route_w.view(-1, 1)
    for h0 in range(0, HIDDEN, h_chunk):
        h1 = min(h0 + h_chunk, HIDDEN)
        w2_chunk = _dequant_gemm2_chunk(w2_e, s2_e, h0, h1)
        y = torch.mm(s_e, w2_chunk.t())
        y.mul_(route_w)
        out[:, h0:h1].index_add_(0, tok_e, y)


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
    e_local = gemm1_weights.shape[0]

    topk_idx, topk_w = _route(routing_logits, routing_bias, routed_scaling_factor)
    sorted_tokens, _, sorted_weights, expert_offsets = _build_dispatch(
        topk_idx, topk_w, local_expert_offset, e_local
    )

    out = torch.zeros((t, HIDDEN), device=device, dtype=torch.float32)
    if sorted_tokens.numel() == 0:
        return out.to(torch.bfloat16)

    unique_tokens, inverse = torch.unique(sorted_tokens, sorted=True, return_inverse=True)
    hidden_deq = _dequant_hidden_rows(hidden_states, hidden_states_scale, unique_tokens)

    for e in range(e_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        tok_e = sorted_tokens[start:end]
        inv_e = inverse[start:end]
        a_e = hidden_deq.index_select(0, inv_e)

        g1 = _gemm1_chunked(a_e, gemm1_weights[e], gemm1_weights_scale[e], end - start)

        x1 = g1[:, :INTERMEDIATE]
        x2 = g1[:, INTERMEDIATE:]
        s_e = F.silu(x2) * x1

        _gemm2_accum_chunked(
            out,
            tok_e,
            sorted_weights[start:end],
            s_e,
            gemm2_weights[e],
            gemm2_weights_scale[e],
        )

    return out.to(torch.bfloat16)