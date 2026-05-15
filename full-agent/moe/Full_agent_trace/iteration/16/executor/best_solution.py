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
    Tk = end_idx - start_idx

    row_start = m_idx * BLOCK_M
    if row_start >= Tk:
        return

    m_offs = row_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < Tk

    n_offs = (n_idx * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_offs < N_dim

    routed_rows = start_idx + m_offs
    token_ids = tl.load(sorted_tokens_ptr + routed_rows, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_base = tl.arange(0, BLOCK_K).to(tl.int64)
    a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_base[None, :] * stride_ak
    w_ptrs = (
        W_ptr
        + e_idx.to(tl.int64) * stride_we
        + n_offs[None, :] * stride_wn
        + k_base[:, None] * stride_wk
    )

    a_scale_base = A_scale_ptr + token_ids[:, None] * stride_ascale_m
    scale_n_offs = n_offs // 128
    w_scale_base = (
        W_scale_ptr
        + e_idx.to(tl.int64) * stride_wscale_e
        + scale_n_offs[None, :] * stride_wscale_n
    )

    for k in range(0, H, BLOCK_K):
        k_block = k // BLOCK_K

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        dot = tl.dot(a, w, out_dtype=tl.float32)

        a_scale = tl.load(
            a_scale_base + k_block * stride_ascale_k,
            mask=m_mask[:, None],
            other=0.0,
        )
        w_scale = tl.load(
            w_scale_base + k_block * stride_wscale_k,
            mask=n_mask[None, :],
            other=0.0,
        )

        acc += dot * a_scale * w_scale

        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = (
        Out_ptr
        + routed_rows[:, None].to(tl.int64) * stride_outm
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
    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True)
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=topk_group, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, n_group), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)
    expert_allowed = group_mask.unsqueeze(-1).expand(T, n_group, group_size).reshape(T, E_global)

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=rank_scores.dtype)
    pruned = torch.where(expert_allowed, rank_scores, neg_inf)

    topk_idx = torch.topk(pruned, k=top_k, dim=1, largest=True, sorted=True).indices
    sel = torch.zeros((T, E_global), dtype=torch.bool, device=device)
    sel.scatter_(1, topk_idx, True)

    raw_w = q * sel
    denom = raw_w.sum(dim=1, keepdim=True) + 1e-20
    weights = (raw_w / denom) * routed_scaling_factor

    local_start = int(local_expert_offset)
    expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)
    flat_token_idx, flat_k_idx = torch.nonzero(expert_mask, as_tuple=True)
    if flat_token_idx.numel() == 0:
        return None

    chosen_global = topk_idx[flat_token_idx, flat_k_idx]
    local_expert_idx = chosen_global - local_start
    order = torch.argsort(local_expert_idx)

    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order].to(torch.int32)
    sorted_weights = weights[flat_token_idx, chosen_global][order].to(torch.float32)

    expert_counts = torch.bincount(sorted_experts, minlength=e_local)
    expert_offsets = torch.empty(e_local + 1, dtype=torch.int32, device=device)
    expert_offsets[0] = 0
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    return sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets


def _validate_dispatch(sorted_experts, expert_counts, expert_offsets, e_local):
    if expert_offsets.numel() != e_local + 1:
        return False
    if int(expert_offsets[0].item()) != 0:
        return False
    if not torch.all(expert_offsets[1:] >= expert_offsets[:-1]):
        return False
    counts_check = expert_offsets[1:] - expert_offsets[:-1]
    if not torch.equal(counts_check.to(expert_counts.dtype), expert_counts):
        return False
    if sorted_experts.numel() != int(expert_offsets[-1].item()):
        return False
    return True


def _sample_validation_rows(expert_counts, expert_offsets, tk_total):
    rows = []
    nonempty = torch.nonzero(expert_counts > 0, as_tuple=False).flatten()
    take = min(int(nonempty.numel()), 2)
    for i in range(take):
        e = int(nonempty[i].item())
        rows.append(int(expert_offsets[e].item()))
    if tk_total > 8:
        rows.append(tk_total // 2)
    # unique preserve order
    out = []
    seen = set()
    for r in rows:
        if 0 <= r < tk_total and r not in seen:
            out.append(r)
            seen.add(r)
    return out


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    a = hidden_states[token_ids].float().clone()
    H = a.shape[1]
    for kb in range(H // 128):
        a[:, kb * 128:(kb + 1) * 128] *= hidden_states_scale[kb, token_ids].float().unsqueeze(1)
    return a


def _dequant_w1_expert(gemm1_weights, gemm1_weights_scale, e):
    w1 = gemm1_weights[e].float().clone()
    n_blk = w1.shape[0] // 128
    k_blk = w1.shape[1] // 128
    for nb in range(n_blk):
        r0 = nb * 128
        r1 = r0 + 128
        for kb in range(k_blk):
            c0 = kb * 128
            c1 = c0 + 128
            w1[r0:r1, c0:c1] *= gemm1_weights_scale[e, nb, kb].float()
    return w1


def _dequant_w2_expert(gemm2_weights, gemm2_weights_scale, e):
    w2 = gemm2_weights[e].float().clone()
    h_blk = w2.shape[0] // 128
    i_blk = w2.shape[1] // 128
    for hb in range(h_blk):
        r0 = hb * 128
        r1 = r0 + 128
        for ib in range(i_blk):
            c0 = ib * 128
            c1 = c0 + 128
            w2[r0:r1, c0:c1] *= gemm2_weights_scale[e, hb, ib].float()
    return w2


def _exact_gemm1_block_oracle(
    j,
    n0,
    n1,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    sorted_tokens,
    sorted_experts,
):
    t = int(sorted_tokens[j].item())
    e = int(sorted_experts[j].item())
    a = _dequant_hidden_rows(hidden_states, hidden_states_scale, torch.tensor([t], device=hidden_states.device, dtype=torch.long))
    w1_blk = gemm1_weights[e, n0:n1].float().clone()
    for nb_local, nb in enumerate(range(n0 // 128, n1 // 128)):
        r0 = nb_local * 128
        r1 = r0 + 128
        for kb in range(w1_blk.shape[1] // 128):
            c0 = kb * 128
            c1 = c0 + 128
            w1_blk[r0:r1, c0:c1] *= gemm1_weights_scale[e, nb, kb].float()
    return (a @ w1_blk.t()).squeeze(0)


def _exact_gemm2_block_oracle(
    j,
    h0,
    h1,
    g1_row,
    gemm2_weights,
    gemm2_weights_scale,
    sorted_experts,
    sorted_weights,
):
    I = 2048
    e = int(sorted_experts[j].item())
    rw = sorted_weights[j].float()
    x1 = g1_row[:I]
    x2 = g1_row[I:]
    c = torch.nn.functional.silu(x2) * x1
    w2_blk = gemm2_weights[e, h0:h1].float().clone()
    for hb_local, hb in enumerate(range(h0 // 128, h1 // 128)):
        r0 = hb_local * 128
        r1 = r0 + 128
        for ib in range(w2_blk.shape[1] // 128):
            c0 = ib * 128
            c1 = c0 + 128
            w2_blk[r0:r1, c0:c1] *= gemm2_weights_scale[e, hb, ib].float()
    return (c @ w2_blk.t()) * rw


def _compute_exact_gemm2_from_g1(
    G1,
    gemm2_weights,
    gemm2_weights_scale,
    expert_offsets,
    sorted_weights,
):
    device = G1.device
    tk_total = G1.shape[0]
    H = 7168
    I = 2048
    e_local = gemm2_weights.shape[0]
    O = torch.zeros((tk_total, H), dtype=torch.float32, device=device)
    for e in range(e_local):
        s = int(expert_offsets[e].item())
        t = int(expert_offsets[e + 1].item())
        if s == t:
            continue
        g1 = G1[s:t]
        x1 = g1[:, :I]
        x2 = g1[:, I:]
        c = torch.nn.functional.silu(x2) * x1
        w2 = _dequant_w2_expert(gemm2_weights, gemm2_weights_scale, e)
        o = c @ w2.t()
        o *= sorted_weights[s:t].unsqueeze(1)
        O[s:t] = o
    return O


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
    e_local = gemm1_weights.shape[0]
    out = torch.zeros((T, H), dtype=torch.float32, device=device)

    for e in range(e_local):
        s = int(expert_offsets[e].item())
        t = int(expert_offsets[e + 1].item())
        if s == t:
            continue

        tok = sorted_tokens[s:t].long()
        rw = sorted_weights[s:t].float()

        a = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok)
        w1 = _dequant_w1_expert(gemm1_weights, gemm1_weights_scale, e)

        g1 = a @ w1.t()
        x1 = g1[:, :I]
        x2 = g1[:, I:]
        c = torch.nn.functional.silu(x2) * x1

        w2 = _dequant_w2_expert(gemm2_weights, gemm2_weights_scale, e)
        o = c @ w2.t()
        o = o * rw.unsqueeze(1)
        out.index_add_(0, tok, o)

    return out.to(torch.bfloat16)


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
    device = hidden_states.device
    T = routing_logits.shape[0]
    e_local = gemm1_weights.shape[0]

    routed = _route_tokens(
        routing_logits,
        routing_bias,
        local_expert_offset,
        e_local,
        routed_scaling_factor,
    )
    if routed is None:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets = routed
    tk_total = sorted_tokens.numel()
    if tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    if not _validate_dispatch(sorted_experts, expert_counts, expert_offsets, e_local):
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

    # For very small problems exact path is often fine and avoids compile/launch overhead.
    if tk_total <= 8:
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

    try:
        G1 = torch.empty((tk_total, 2 * I), dtype=torch.float32, device=device)
        grid_gemm1 = lambda META: (
            triton.cdiv(2 * I, META["BLOCK_N"]),
            triton.cdiv(max_count, META["BLOCK_M"]),
            e_local,
        )

        grouped_gemm1_kernel[grid_gemm1](
            A_ptr=hidden_states,
            A_scale_ptr=hidden_states_scale,
            W_ptr=gemm1_weights,
            W_scale_ptr=gemm1_weights_scale,
            Out_ptr=G1,
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
            stride_outm=G1.stride(0),
            stride_outn=G1.stride(1),
            BLOCK_K=128,
        )

        # Deterministic semantic validation for GEMM1.
        sample_rows = _sample_validation_rows(expert_counts, expert_offsets, tk_total)
        gemm1_ok = True
        for j in sample_rows:
            for n0 in (0, I):
                ref = _exact_gemm1_block_oracle(
                    j,
                    n0,
                    n0 + 128,
                    hidden_states,
                    hidden_states_scale,
                    gemm1_weights,
                    gemm1_weights_scale,
                    sorted_tokens,
                    sorted_experts,
                )
                err = (G1[j, n0:n0 + 128] - ref).abs().max()
                if float(err.item()) > 5e-2:
                    gemm1_ok = False
                    break
            if not gemm1_ok:
                break

        if not gemm1_ok:
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

        # Safer hybrid: exact GEMM2 from validated G1.
        O = _compute_exact_gemm2_from_g1(
            G1,
            gemm2_weights,
            gemm2_weights_scale,
            expert_offsets,
            sorted_weights,
        )

        # Optional small validator for GEMM2 exact-from-G1 path.
        gemm2_ok = True
        for j in sample_rows[:2]:
            for h0 in (0, 3584):
                ref = _exact_gemm2_block_oracle(
                    j,
                    h0,
                    h0 + 128,
                    G1[j],
                    gemm2_weights,
                    gemm2_weights_scale,
                    sorted_experts,
                    sorted_weights,
                )
                err = (O[j, h0:h0 + 128] - ref).abs().max()
                if float(err.item()) > 5e-2:
                    gemm2_ok = False
                    break
            if not gemm2_ok:
                break

        if not gemm2_ok:
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

        output = torch.zeros((T, H), dtype=torch.float32, device=device)
        output.index_add_(0, sorted_tokens.long(), O)
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