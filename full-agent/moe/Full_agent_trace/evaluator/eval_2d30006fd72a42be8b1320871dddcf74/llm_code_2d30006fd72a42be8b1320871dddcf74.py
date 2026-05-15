import torch
import torch.nn.functional as F


HIDDEN = 7168
INTERMEDIATE = 2048
BLOCK = 128
N_GROUP = 8
TOPK_GROUP = 4
TOPK_EXPERT = 8


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
    top2_vals = torch.topk(sbg, k=2, dim=2, largest=True, sorted=False).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False).indices
    group_mask = torch.zeros((t, N_GROUP), device=routing_logits.device, dtype=torch.bool)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(t, N_GROUP, group_size).reshape(t, e_global)

    pruned = sb.masked_fill(~expert_mask, float("-inf"))
    topk_idx = torch.topk(pruned, k=TOPK_EXPERT, dim=1, largest=True, sorted=False).indices

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


def _gather_hidden_deq_unique(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    unique_tokens: torch.Tensor,
) -> torch.Tensor:
    a = hidden_states.index_select(0, unique_tokens).to(torch.float32)
    s = hidden_states_scale.index_select(1, unique_tokens).transpose(0, 1).contiguous()
    s = s.repeat_interleave(BLOCK, dim=1)
    return a * s


def _dequant_w1_chunk(
    expert_w: torch.Tensor,
    expert_scale: torch.Tensor,
    n0: int,
    n1: int,
) -> torch.Tensor:
    w = expert_w[n0:n1].to(torch.float32)
    sb0 = n0 // BLOCK
    sb1 = (n1 + BLOCK - 1) // BLOCK
    s = expert_scale[sb0:sb1]
    s = s.repeat_interleave(BLOCK, dim=0)
    s = s[(n0 - sb0 * BLOCK):(n0 - sb0 * BLOCK + (n1 - n0))]
    s = s.repeat_interleave(BLOCK, dim=1)
    return w * s


def _dequant_w2_chunk(
    expert_w: torch.Tensor,
    expert_scale: torch.Tensor,
    h0: int,
    h1: int,
) -> torch.Tensor:
    w = expert_w[h0:h1].to(torch.float32)
    sb0 = h0 // BLOCK
    sb1 = (h1 + BLOCK - 1) // BLOCK
    s = expert_scale[sb0:sb1]
    s = s.repeat_interleave(BLOCK, dim=0)
    s = s[(h0 - sb0 * BLOCK):(h0 - sb0 * BLOCK + (h1 - h0))]
    s = s.repeat_interleave(BLOCK, dim=1)
    return w * s


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,        # [T, 256] float32
    routing_bias: torch.Tensor,          # [256]    bfloat16
    hidden_states: torch.Tensor,         # [T, 7168] fp8
    hidden_states_scale: torch.Tensor,   # [56, T]  float32
    gemm1_weights: torch.Tensor,         # [32, 4096, 7168] fp8
    gemm1_weights_scale: torch.Tensor,   # [32, 32, 56]    float32
    gemm2_weights: torch.Tensor,         # [32, 7168, 2048] fp8
    gemm2_weights_scale: torch.Tensor,   # [32, 56, 16]    float32
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
    hidden_deq = _gather_hidden_deq_unique(hidden_states, hidden_states_scale, unique_tokens)

    n_chunk = 1024
    h_chunk = 1024

    for e in range(e_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        tok_e = sorted_tokens[start:end]
        inv_e = inverse[start:end]
        a_e = hidden_deq.index_select(0, inv_e)

        te = a_e.shape[0]
        g1 = torch.empty((te, 2 * INTERMEDIATE), device=device, dtype=torch.float32)
        w1e = gemm1_weights[e]
        s1e = gemm1_weights_scale[e]
        for n0 in range(0, 2 * INTERMEDIATE, n_chunk):
            n1 = min(n0 + n_chunk, 2 * INTERMEDIATE)
            w1_chunk = _dequant_w1_chunk(w1e, s1e, n0, n1)
            g1[:, n0:n1] = a_e @ w1_chunk.t()

        x1 = g1[:, :INTERMEDIATE]
        x2 = g1[:, INTERMEDIATE:]
        s = F.silu(x2) * x1

        w_e = sorted_weights[start:end].unsqueeze(1)

        w2e = gemm2_weights[e]
        s2e = gemm2_weights_scale[e]
        for h0 in range(0, HIDDEN, h_chunk):
            h1 = min(h0 + h_chunk, HIDDEN)
            w2_chunk = _dequant_w2_chunk(w2e, s2e, h0, h1)
            y_chunk = s @ w2_chunk.t()
            y_chunk.mul_(w_e)
            out[:, h0:h1].index_add_(0, tok_e, y_chunk)

    return out.to(torch.bfloat16)