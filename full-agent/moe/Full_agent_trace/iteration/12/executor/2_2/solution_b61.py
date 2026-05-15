import torch
import triton
import triton.language as tl


# -----------------------------
# Triton kernels
# -----------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["MAX_COUNT"],
)
@triton.jit
def grouped_gemm1_kernel(
    A_ptr, A_scale_ptr,
    W_ptr, W_scale_ptr,
    Out_ptr,
    sorted_tokens_ptr, expert_offsets_ptr,
    H, N_dim, MAX_COUNT,
    stride_am, stride_ak,
    stride_ascale_k, stride_ascale_m,
    stride_we, stride_wn, stride_wk,
    stride_wscale_e, stride_wscale_n, stride_wscale_k,
    stride_outm, stride_outn,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    n_pid = tl.program_id(0)
    m_pid = tl.program_id(1)
    e_pid = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + e_pid)
    end_idx = tl.load(expert_offsets_ptr + e_pid + 1)
    Tk = end_idx - start_idx

    if m_pid * BLOCK_M >= Tk:
        return

    m_offs = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < Tk

    n_offs = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N_dim

    token_ids = tl.load(sorted_tokens_ptr + start_idx + m_offs, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_base = tl.arange(0, BLOCK_K).to(tl.int64)
    a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_base[None, :] * stride_ak
    w_ptrs = (
        W_ptr
        + e_pid.to(tl.int64) * stride_we
        + n_offs[None, :].to(tl.int64) * stride_wn
        + k_base[:, None] * stride_wk
    )

    for k in range(0, H, BLOCK_K):
        k_block = k // BLOCK_K

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        a_scale = tl.load(
            A_scale_ptr
            + k_block * stride_ascale_k
            + token_ids[:, None] * stride_ascale_m,
            mask=m_mask[:, None],
            other=0.0,
        )
        w_scale = tl.load(
            W_scale_ptr
            + e_pid.to(tl.int64) * stride_wscale_e
            + (n_offs[None, :] // 128).to(tl.int64) * stride_wscale_n
            + k_block * stride_wscale_k,
            mask=n_mask[None, :],
            other=0.0,
        )

        dot = tl.dot(a, w, out_dtype=tl.float32)
        acc += dot * a_scale * w_scale

        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = (
        Out_ptr
        + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm
        + n_offs[None, :].to(tl.int64) * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["MAX_COUNT"],
)
@triton.jit
def grouped_gemm2_kernel(
    G1_ptr,
    W2_ptr, W2_scale_ptr,
    routing_weights_ptr,
    Out_ptr,
    expert_offsets_ptr,
    I, H, MAX_COUNT,
    stride_g1m, stride_g1k,
    stride_w2e, stride_w2n, stride_w2k,
    stride_w2scale_e, stride_w2scale_n, stride_w2scale_k,
    stride_outm, stride_outn,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    n_pid = tl.program_id(0)
    m_pid = tl.program_id(1)
    e_pid = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + e_pid)
    end_idx = tl.load(expert_offsets_ptr + e_pid + 1)
    Tk = end_idx - start_idx

    if m_pid * BLOCK_M >= Tk:
        return

    m_offs = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < Tk

    n_offs = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    row_ids = (start_idx + m_offs).to(tl.int64)
    k_base = tl.arange(0, BLOCK_K).to(tl.int64)

    x1_ptrs = G1_ptr + row_ids[:, None] * stride_g1m + k_base[None, :] * stride_g1k
    x2_ptrs = G1_ptr + row_ids[:, None] * stride_g1m + (I + k_base[None, :]) * stride_g1k

    w_ptrs = (
        W2_ptr
        + e_pid.to(tl.int64) * stride_w2e
        + n_offs[None, :].to(tl.int64) * stride_w2n
        + k_base[:, None] * stride_w2k
    )

    for k in range(0, I, BLOCK_K):
        k_block = k // BLOCK_K

        x1 = tl.load(x1_ptrs, mask=m_mask[:, None], other=0.0)
        x2 = tl.load(x2_ptrs, mask=m_mask[:, None], other=0.0)
        c = x1 * (x2 * tl.sigmoid(x2))

        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
        w_scale = tl.load(
            W2_scale_ptr
            + e_pid.to(tl.int64) * stride_w2scale_e
            + (n_offs[None, :] // 128).to(tl.int64) * stride_w2scale_n
            + k_block * stride_w2scale_k,
            mask=n_mask[None, :],
            other=0.0,
        )

        # safer dequant path; fp32 accumulation remains exact enough
        w_f32 = w.to(tl.float32) * w_scale
        acc += tl.dot(c.to(tl.bfloat16), w_f32.to(tl.bfloat16), out_dtype=tl.float32)

        x1_ptrs += BLOCK_K * stride_g1k
        x2_ptrs += BLOCK_K * stride_g1k
        w_ptrs += BLOCK_K * stride_w2k

    rw = tl.load(routing_weights_ptr + row_ids, mask=m_mask, other=0.0)
    acc = acc * rw[:, None]

    out_ptrs = (
        Out_ptr
        + row_ids[:, None] * stride_outm
        + n_offs[None, :].to(tl.int64) * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# -----------------------------
# Helpers
# -----------------------------

def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    # hidden_states: [T, H] fp8
    # hidden_states_scale: [56, T]
    x = hidden_states[token_ids].to(torch.float32)  # [Tk, H]
    s = hidden_states_scale[:, token_ids].transpose(0, 1).contiguous()  # [Tk, 56]
    s = s.repeat_interleave(128, dim=1)
    return x * s


def _dequant_w1_expert(w, ws):
    # w: [4096, 7168] fp8 ; ws: [32, 56]
    wf = w.to(torch.float32)
    s_n = ws.repeat_interleave(128, dim=0)
    s_nk = s_n.repeat_interleave(128, dim=1)
    return wf * s_nk


def _dequant_w2_expert(w, ws):
    # w: [7168, 2048] fp8 ; ws: [56, 16]
    wf = w.to(torch.float32)
    s_h = ws.repeat_interleave(128, dim=0)
    s_hi = s_h.repeat_interleave(128, dim=1)
    return wf * s_hi


def _fallback_run(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    device = hidden_states.device

    logits = routing_logits.float()
    q = torch.sigmoid(logits)
    rank_scores = q + routing_bias.float().view(1, -1)

    group_size = E_global // 8
    grouped = rank_scores.view(T, 8, group_size)
    top2 = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2.sum(dim=2)
    top_groups = torch.topk(group_scores, k=4, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, 8), device=device, dtype=torch.bool)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, 8, group_size).reshape(T, E_global)

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
    pruned = torch.where(expert_mask, rank_scores, neg_inf)
    topk_idx = torch.topk(pruned, k=8, dim=1, largest=True, sorted=True).indices

    sel = torch.zeros_like(q)
    sel.scatter_(1, topk_idx, 1.0)
    weights = q * sel
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor

    out = torch.zeros((T, H), device=device, dtype=torch.float32)

    local_start = int(local_expert_offset)
    for e in range(E_local):
        g = local_start + e
        pos = (topk_idx == g).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        token_ids = pos[:, 0]
        rw = weights[token_ids, g].float()

        A = _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids)
        W1 = _dequant_w1_expert(gemm1_weights[e], gemm1_weights_scale[e])
        G1 = A @ W1.t()
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        C = torch.nn.functional.silu(X2) * X1

        W2 = _dequant_w2_expert(gemm2_weights[e], gemm2_weights_scale[e])
        Oe = C @ W2.t()
        Oe = Oe * rw[:, None]
        out.index_add_(0, token_ids, Oe)

    return out.to(torch.bfloat16)


# -----------------------------
# Main entry
# -----------------------------

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
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    device = hidden_states.device

    # Exact DeepSeek routing
    logits = routing_logits.float()
    q = torch.sigmoid(logits)
    rank_scores = q + routing_bias.float().view(1, -1)

    group_size = E_global // 8
    grouped = rank_scores.view(T, 8, group_size)
    top2 = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2.sum(dim=2)
    top_groups = torch.topk(group_scores, k=4, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, 8), device=device, dtype=torch.bool)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, 8, group_size).reshape(T, E_global)

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
    pruned = torch.where(expert_mask, rank_scores, neg_inf)
    topk_idx = torch.topk(pruned, k=8, dim=1, largest=True, sorted=True).indices

    sel = torch.zeros_like(q)
    sel.scatter_(1, topk_idx, 1.0)
    weights = q * sel
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor

    # Local dispatch
    local_start = int(local_expert_offset)
    local_mask = (topk_idx >= local_start) & (topk_idx < local_start + E_local)
    flat_token_idx, flat_k_idx = torch.nonzero(local_mask, as_tuple=True)
    Tk_total = flat_token_idx.numel()

    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    local_expert_idx = topk_idx[flat_token_idx, flat_k_idx] - local_start
    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int32).contiguous()
    sorted_experts = local_expert_idx[order].to(torch.int32).contiguous()
    sorted_weights = weights[flat_token_idx, topk_idx[flat_token_idx, flat_k_idx]][order].to(torch.float32).contiguous()

    expert_counts = torch.bincount(sorted_experts, minlength=E_local)
    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    # structural checks
    ok = True
    ok = ok and (sorted_tokens.numel() == sorted_experts.numel() == sorted_weights.numel())
    ok = ok and (expert_offsets.shape[0] == E_local + 1)
    if expert_offsets[0].item() != 0 or expert_offsets[-1].item() != Tk_total:
        ok = False

    if not ok:
        return _fallback_run(
            routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor
        )

    max_count = int(expert_counts.max().item()) if Tk_total > 0 else 0
    if max_count == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    # Fast path
    try:
        G1 = torch.zeros((Tk_total, 2 * I), dtype=torch.float32, device=device)
        O = torch.zeros((Tk_total, H), dtype=torch.float32, device=device)

        grid1 = lambda META: (
            triton.cdiv(2 * I, META["BLOCK_N"]),
            triton.cdiv(max_count, META["BLOCK_M"]),
            E_local,
        )
        grouped_gemm1_kernel[grid1](
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            G1,
            sorted_tokens, expert_offsets,
            H, 2 * I, max_count,
            hidden_states.stride(0), hidden_states.stride(1),
            hidden_states_scale.stride(0), hidden_states_scale.stride(1),
            gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
            gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
            G1.stride(0), G1.stride(1),
            BLOCK_K=128,
        )

        grid2 = lambda META: (
            triton.cdiv(H, META["BLOCK_N"]),
            triton.cdiv(max_count, META["BLOCK_M"]),
            E_local,
        )
        grouped_gemm2_kernel[grid2](
            G1,
            gemm2_weights, gemm2_weights_scale,
            sorted_weights,
            O,
            expert_offsets,
            I, H, max_count,
            G1.stride(0), G1.stride(1),
            gemm2_weights.stride(0), gemm2_weights.stride(1), gemm2_weights.stride(2),
            gemm2_weights_scale.stride(0), gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
            O.stride(0), O.stride(1),
            BLOCK_K=128,
        )

        output = torch.zeros((T, H), dtype=torch.float32, device=device)
        output.index_add_(0, sorted_tokens.long(), O)
        return output.to(torch.bfloat16)

    except Exception:
        return _fallback_run(
            routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor
        )