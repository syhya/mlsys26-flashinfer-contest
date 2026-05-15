import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["MAX_TK"],
)
@triton.jit
def grouped_gemm1_kernel(
    A_ptr,
    A_scale_ptr,
    W_ptr,
    W_scale_ptr,
    Out_ptr,
    sorted_tokens_ptr,
    expert_offsets_ptr,
    H,
    N_dim,
    MAX_TK,
    stride_am,
    stride_ak,
    stride_ascale_k,
    stride_ascale_m,
    stride_we,
    stride_wn,
    stride_wk,
    stride_wscale_e,
    stride_wscale_n,
    stride_wscale_k,
    stride_outm,
    stride_outn,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    n_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    e_idx = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + e_idx)
    end_idx = tl.load(expert_offsets_ptr + e_idx + 1)
    tk = end_idx - start_idx

    row_start = m_idx * BLOCK_M
    if row_start >= tk:
        return

    m_offs = row_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < tk

    n_offs = (n_idx * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_offs < N_dim

    pair_rows = (start_idx + m_offs).to(tl.int64)
    token_ids = tl.load(sorted_tokens_ptr + pair_rows, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_base = tl.arange(0, BLOCK_K).to(tl.int64)

    for k in range(0, H, BLOCK_K):
        k_block = k // BLOCK_K
        k_offs = (k + k_base).to(tl.int64)

        a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
        w_ptrs = (
            W_ptr
            + e_idx.to(tl.int64) * stride_we
            + n_offs[:, None] * stride_wn
            + k_offs[None, :] * stride_wk
        )

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)

        dot = tl.dot(a, tl.trans(w), out_dtype=tl.float32)

        a_scale = tl.load(
            A_scale_ptr + k_block * stride_ascale_k + token_ids[:, None] * stride_ascale_m,
            mask=m_mask[:, None],
            other=0.0,
        )
        w_scale = tl.load(
            W_scale_ptr
            + e_idx.to(tl.int64) * stride_wscale_e
            + (n_offs // 128)[None, :] * stride_wscale_n
            + k_block * stride_wscale_k,
            mask=n_mask[None, :],
            other=0.0,
        )

        acc += dot * a_scale * w_scale

    out_ptrs = (
        Out_ptr
        + pair_rows[:, None] * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _route_tokens(routing_logits, routing_bias, local_expert_offset, e_local, routed_scaling_factor):
    device = routing_logits.device
    T, E_global = routing_logits.shape
    n_group = 8
    topk_group = 4
    top_k = 8
    group_size = E_global // n_group

    q = torch.sigmoid(routing_logits.float())
    rank_scores = q + routing_bias.float().view(1, E_global)

    grouped = rank_scores.view(T, n_group, group_size)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=topk_group, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, n_group), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)

    allowed = group_mask.unsqueeze(-1).expand(T, n_group, group_size).reshape(T, E_global)
    pruned = torch.where(
        allowed,
        rank_scores,
        torch.full_like(rank_scores, float("-inf")),
    )

    topk_idx = torch.topk(pruned, k=top_k, dim=1, largest=True, sorted=True).indices
    sel = torch.zeros((T, E_global), dtype=torch.bool, device=device)
    sel.scatter_(1, topk_idx, True)

    raw_w = q * sel
    weights = raw_w / (raw_w.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor

    local_start = int(local_expert_offset)
    expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)

    flat_token_idx, flat_k_idx = torch.nonzero(expert_mask, as_tuple=True)
    if flat_token_idx.numel() == 0:
        return None

    chosen_global = topk_idx[flat_token_idx, flat_k_idx]
    local_expert_idx = chosen_global - local_start
    pair_weights = weights[flat_token_idx, chosen_global]

    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order].to(torch.int32)
    sorted_weights = pair_weights[order].to(torch.float32)

    expert_counts = torch.bincount(sorted_experts, minlength=e_local)
    expert_offsets = torch.zeros(e_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    return sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    a = hidden_states[token_ids].float()
    scales = hidden_states_scale[:, token_ids.long()].transpose(0, 1).contiguous()
    scales = scales.repeat_interleave(128, dim=1)
    return a * scales


def _dequant_w2_chunk(gemm2_weights_e, gemm2_scales_e, h0, h1):
    hb0 = h0 // 128
    hb1 = (h1 + 127) // 128
    scale = gemm2_scales_e[hb0:hb1].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    scale = scale[: (h1 - hb0 * 128), :]
    row0 = h0 - hb0 * 128
    row1 = row0 + (h1 - h0)
    scale = scale[row0:row1]
    return gemm2_weights_e[h0:h1].float() * scale


def _fallback_exact_run(
    T,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    sorted_tokens,
    expert_offsets,
    sorted_weights,
):
    device = hidden_states.device
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    for e in range(E_local):
        s = int(expert_offsets[e].item())
        t = int(expert_offsets[e + 1].item())
        if s == t:
            continue

        tok = sorted_tokens[s:t].long()
        rw = sorted_weights[s:t]

        a = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok)
        w1_scale = gemm1_weights_scale[e].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        w1 = gemm1_weights[e].float() * w1_scale
        g1 = a @ w1.t()

        x1 = g1[:, :I]
        x2 = g1[:, I:]
        act = torch.nn.functional.silu(x2) * x1

        out_e = torch.empty((t - s, H), dtype=torch.float32, device=device)
        chunk_h = 1024
        for h0 in range(0, H, chunk_h):
            h1 = min(h0 + chunk_h, H)
            w2_chunk = _dequant_w2_chunk(gemm2_weights[e], gemm2_weights_scale[e], h0, h1)
            out_e[:, h0:h1] = act @ w2_chunk.t()

        out_e *= rw.unsqueeze(1)
        output.index_add_(0, tok, out_e)

    return output.to(torch.bfloat16)


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
    H = 7168
    I = 2048
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]
    device = hidden_states.device

    routed = _route_tokens(
        routing_logits,
        routing_bias,
        local_expert_offset,
        E_local,
        routed_scaling_factor,
    )
    if routed is None:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets = routed
    tk_total = sorted_tokens.numel()
    if tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    valid = True
    valid = valid and (sorted_experts.numel() == tk_total)
    valid = valid and (sorted_weights.numel() == tk_total)
    valid = valid and (expert_offsets.numel() == E_local + 1)
    valid = valid and (int(expert_offsets[0].item()) == 0)
    valid = valid and (int(expert_offsets[-1].item()) == tk_total)
    if tk_total > 1:
        valid = valid and bool(torch.all(sorted_experts[1:] >= sorted_experts[:-1]).item())
    counts_check = expert_offsets[1:].to(torch.int64) - expert_offsets[:-1].to(torch.int64)
    valid = valid and bool(torch.equal(counts_check.cpu(), expert_counts.to(torch.int64).cpu()))

    if not valid:
        return _fallback_exact_run(
            T,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            sorted_tokens,
            expert_offsets,
            sorted_weights,
        )

    max_count = int(expert_counts.max().item())
    if max_count == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    try:
        g1 = torch.empty((tk_total, 2 * I), dtype=torch.float32, device=device)
        grid = lambda META: (
            triton.cdiv(2 * I, META["BLOCK_N"]),
            triton.cdiv(max_count, META["BLOCK_M"]),
            E_local,
        )

        grouped_gemm1_kernel[grid](
            A_ptr=hidden_states,
            A_scale_ptr=hidden_states_scale,
            W_ptr=gemm1_weights,
            W_scale_ptr=gemm1_weights_scale,
            Out_ptr=g1,
            sorted_tokens_ptr=sorted_tokens,
            expert_offsets_ptr=expert_offsets,
            H=H,
            N_dim=2 * I,
            MAX_TK=max_count,
            stride_am=hidden_states.stride(0),
            stride_ak=hidden_states.stride(1),
            stride_ascale_k=hidden_states_scale.stride(0),
            stride_ascale_m=hidden_states_scale.stride(1),
            stride_we=gemm1_weights.stride(0),
            stride_wn=gemm1_weights.stride(1),
            stride_wk=gemm1_weights.stride(2),
            stride_wscale_e=gemm1_weights_scale.stride(0),
            stride_wscale_n=gemm1_weights_scale.stride(1),
            stride_wscale_k=gemm1_weights_scale.stride(2),
            stride_outm=g1.stride(0),
            stride_outn=g1.stride(1),
            BLOCK_K=128,
        )

        output = torch.zeros((T, H), dtype=torch.float32, device=device)

        for e in range(E_local):
            s = int(expert_offsets[e].item())
            t = int(expert_offsets[e + 1].item())
            if s == t:
                continue

            g1_e = g1[s:t]
            x1 = g1_e[:, :I]
            x2 = g1_e[:, I:]
            act = torch.nn.functional.silu(x2) * x1

            out_e = torch.empty((t - s, H), dtype=torch.float32, device=device)
            chunk_h = 1024
            for h0 in range(0, H, chunk_h):
                h1 = min(h0 + chunk_h, H)
                w2_chunk = _dequant_w2_chunk(gemm2_weights[e], gemm2_weights_scale[e], h0, h1)
                out_e[:, h0:h1] = act @ w2_chunk.t()

            out_e *= sorted_weights[s:t].unsqueeze(1)
            output.index_add_(0, sorted_tokens[s:t].long(), out_e)

        return output.to(torch.bfloat16)
    except Exception:
        return _fallback_exact_run(
            T,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            sorted_tokens,
            expert_offsets,
            sorted_weights,
        )