import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# -----------------------------
# Routing
# -----------------------------
def _route_tokens(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    local_expert_offset: int,
    e_local: int,
    routed_scaling_factor: float,
):
    device = routing_logits.device
    T, E_global = routing_logits.shape
    n_group = 8
    group_size = E_global // n_group
    topk_group = 4
    top_k = 8

    q = torch.sigmoid(routing_logits.float())
    rank_scores = q + routing_bias.float().view(1, E_global)

    grouped = rank_scores.view(T, n_group, group_size)
    top2 = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2.sum(dim=2)

    top_groups = torch.topk(group_scores, k=topk_group, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, n_group), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)
    expert_allowed = group_mask.unsqueeze(-1).expand(T, n_group, group_size).reshape(T, E_global)

    pruned = torch.where(
        expert_allowed,
        rank_scores,
        torch.full((), float("-inf"), device=device, dtype=rank_scores.dtype),
    )

    topk_idx = torch.topk(pruned, k=top_k, dim=1, largest=True, sorted=True).indices
    sel = torch.zeros((T, E_global), dtype=torch.bool, device=device)
    sel.scatter_(1, topk_idx, True)

    raw_w = q * sel
    weights = raw_w / (raw_w.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * float(routed_scaling_factor)

    local_start = int(local_expert_offset)
    expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)

    flat_token_idx, flat_k_idx = torch.nonzero(expert_mask, as_tuple=True)
    if flat_token_idx.numel() == 0:
        return None

    local_expert_idx = topk_idx[flat_token_idx, flat_k_idx] - local_start
    pair_weights = weights[flat_token_idx, topk_idx[flat_token_idx, flat_k_idx]]

    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order].to(torch.int32)
    sorted_weights = pair_weights[order].to(torch.float32)

    expert_counts = torch.bincount(sorted_experts, minlength=e_local)
    expert_offsets = torch.zeros(e_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    return sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets


# -----------------------------
# Triton GEMM1 only
# Computes exact block-scaled fp8 GEMM1:
# A: [Tk, H] fp8 with scales [H/128, T] by global token
# W: [E, 2I, H] fp8 with scales [E, 2I/128, H/128]
# Out: [Tk_total, 2I] fp32
# -----------------------------
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
def _grouped_gemm1_kernel(
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
    stride_ascale_t,
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
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + pid_e).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + pid_e + 1).to(tl.int32)
    tk = end_idx - start_idx

    row_start = pid_m * BLOCK_M
    if row_start >= tk:
        return

    offs_m = row_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < tk

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    mask_n = offs_n < N_dim

    token_ids = tl.load(sorted_tokens_ptr + start_idx + offs_m, mask=mask_m, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_offsets = tl.arange(0, BLOCK_K).to(tl.int64)

    a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offsets[None, :] * stride_ak
    # W shape [E, N, K], so tile is [N, K]; tl.dot expects [M,K] x [K,N]
    w_ptrs = (
        W_ptr
        + pid_e.to(tl.int64) * stride_we
        + offs_n[None, :] * stride_wn
        + k_offsets[:, None] * stride_wk
    )

    scale_n_block = offs_n // 128
    a_scale_base = A_scale_ptr + token_ids[:, None] * stride_ascale_t
    w_scale_base = (
        W_scale_ptr
        + pid_e.to(tl.int64) * stride_wscale_e
        + scale_n_block[None, :] * stride_wscale_n
    )

    for k in range(0, H, BLOCK_K):
        k_block = k // BLOCK_K

        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)

        dot = tl.dot(a, w, out_dtype=tl.float32)

        a_scale = tl.load(
            a_scale_base + k_block * stride_ascale_k,
            mask=mask_m[:, None],
            other=0.0,
        )
        w_scale = tl.load(
            w_scale_base + k_block * stride_wscale_k,
            mask=mask_n[None, :],
            other=0.0,
        )
        acc += dot * a_scale * w_scale

        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = (
        Out_ptr
        + (start_idx + offs_m)[:, None].to(tl.int64) * stride_outm
        + offs_n[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# -----------------------------
# Exact helpers
# -----------------------------
def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    # hidden_states: [T,H] fp8
    # hidden_states_scale: [H//128, T]
    a = hidden_states[token_ids].float()
    scales = hidden_states_scale[:, token_ids].transpose(0, 1).contiguous()  # [Tk, 56]
    a = a.view(a.shape[0], 56, 128) * scales.unsqueeze(-1)
    return a.view(a.shape[0], 7168)


def _dequant_w1_expert(w, s):
    # w: [4096,7168], s: [32,56]
    scale = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return w.float() * scale


def _gemm1_exact(
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    sorted_tokens,
    expert_offsets,
):
    device = hidden_states.device
    Tk_total = sorted_tokens.numel()
    G1 = torch.empty((Tk_total, 4096), dtype=torch.float32, device=device)

    for e in range(gemm1_weights.shape[0]):
        s = int(expert_offsets[e].item())
        t = int(expert_offsets[e + 1].item())
        if s == t:
            continue
        tok = sorted_tokens[s:t].long()
        a = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok)
        w1 = _dequant_w1_expert(gemm1_weights[e], gemm1_weights_scale[e])
        G1[s:t] = a @ w1.t()
    return G1


def _gemm2_exact_scatter(
    T,
    G1,
    gemm2_weights,
    gemm2_weights_scale,
    sorted_tokens,
    sorted_weights,
    expert_offsets,
):
    device = G1.device
    H = 7168
    I = 2048
    out = torch.zeros((T, H), dtype=torch.float32, device=device)

    # chunk output rows by 128-blocks to limit temp memory
    hb_chunk_blocks = 4  # 512 output dims at a time

    for e in range(gemm2_weights.shape[0]):
        s = int(expert_offsets[e].item())
        t = int(expert_offsets[e + 1].item())
        if s == t:
            continue

        tok = sorted_tokens[s:t].long()
        rw = sorted_weights[s:t].float()

        g1 = G1[s:t]
        x1 = g1[:, :I]
        x2 = g1[:, I:]
        act = F.silu(x2) * x1

        out_e = torch.empty((t - s, H), dtype=torch.float32, device=device)

        for hb0 in range(0, 56, hb_chunk_blocks):
            hb1 = min(hb0 + hb_chunk_blocks, 56)
            h0 = hb0 * 128
            h1 = hb1 * 128

            w_chunk = gemm2_weights[e, h0:h1].float()
            s_chunk = gemm2_weights_scale[e, hb0:hb1]  # [bh, 16]
            scale_chunk = s_chunk.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
            w_chunk = w_chunk * scale_chunk
            out_e[:, h0:h1] = act @ w_chunk.t()

        out_e.mul_(rw.unsqueeze(1))
        out.index_add_(0, tok, out_e)

    return out


def _fallback_exact_run(
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
    T = routing_logits.shape[0]
    device = hidden_states.device
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

    sorted_tokens, _, sorted_weights, _, expert_offsets = routed
    G1 = _gemm1_exact(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        sorted_tokens,
        expert_offsets,
    )
    out = _gemm2_exact_scatter(
        T,
        G1,
        gemm2_weights,
        gemm2_weights_scale,
        sorted_tokens,
        sorted_weights,
        expert_offsets,
    )
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
    Tk_total = sorted_tokens.numel()
    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    valid = True
    valid = valid and (sorted_experts.numel() == Tk_total)
    valid = valid and (sorted_weights.numel() == Tk_total)
    valid = valid and (expert_offsets.numel() == e_local + 1)
    valid = valid and (int(expert_offsets[0].item()) == 0)
    valid = valid and (int(expert_offsets[-1].item()) == Tk_total)
    if Tk_total > 1:
        valid = valid and bool((sorted_experts[1:] >= sorted_experts[:-1]).all().item())
    counts_check = expert_offsets[1:] - expert_offsets[:-1]
    valid = valid and bool(torch.equal(counts_check, expert_counts))

    if not valid:
        return _fallback_exact_run(
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
        )

    # Fast path: Triton GEMM1 + exact PyTorch GEMM2/scatter
    # This avoids the lineage's likely downstream semantic bug.
    use_triton_gemm1 = True
    max_count = int(expert_counts.max().item()) if Tk_total > 0 else 0
    if max_count == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    try:
        G1 = torch.empty((Tk_total, 2 * I), dtype=torch.float32, device=device)
        grid = lambda META: (
            triton.cdiv(2 * I, META["BLOCK_N"]),
            triton.cdiv(max_count, META["BLOCK_M"]),
            e_local,
        )

        _grouped_gemm1_kernel[grid](
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
            stride_ascale_t=hidden_states_scale.stride(1),
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
    except Exception:
        use_triton_gemm1 = False

    if not use_triton_gemm1:
        G1 = _gemm1_exact(
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            sorted_tokens,
            expert_offsets,
        )

    out = _gemm2_exact_scatter(
        T,
        G1,
        gemm2_weights,
        gemm2_weights_scale,
        sorted_tokens,
        sorted_weights,
        expert_offsets,
    )
    return out.to(torch.bfloat16)