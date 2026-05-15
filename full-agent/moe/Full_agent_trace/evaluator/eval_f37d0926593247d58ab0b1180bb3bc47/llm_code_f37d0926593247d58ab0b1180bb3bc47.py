import torch
import triton
import triton.language as tl


# ----------------------------
# Routing / dispatch helpers
# ----------------------------

def _route_and_dispatch(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    local_expert_offset: int,
    e_local: int,
    routed_scaling_factor: float,
):
    device = routing_logits.device
    T, E_global = routing_logits.shape
    N_GROUP = 8
    TOPK_GROUP = 4
    TOP_K = 8
    group_size = E_global // N_GROUP

    logits = routing_logits.float()
    q = torch.sigmoid(logits)
    rank_scores = q + routing_bias.float().view(1, E_global)

    grouped = rank_scores.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True)
    group_scores = top2_vals.sum(dim=2)

    topg_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, topg_idx, True)
    expert_mask_allowed = group_mask.unsqueeze(-1).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.full((), float("-inf"), dtype=torch.float32, device=device)
    pruned = torch.where(expert_mask_allowed, rank_scores, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    picked_q = q.gather(1, topk_idx)
    weights8 = picked_q / (picked_q.sum(dim=1, keepdim=True) + 1e-20)
    weights8 = weights8 * routed_scaling_factor

    local_start = int(local_expert_offset)
    local_end = local_start + e_local
    local_mask = (topk_idx >= local_start) & (topk_idx < local_end)

    flat_token_idx, flat_slot_idx = torch.nonzero(local_mask, as_tuple=True)
    Tk_total = flat_token_idx.numel()

    if Tk_total == 0:
        return {
            "topk_idx": topk_idx,
            "weights8": weights8,
            "sorted_tokens": torch.empty((0,), device=device, dtype=torch.int32),
            "sorted_experts": torch.empty((0,), device=device, dtype=torch.int32),
            "sorted_weights": torch.empty((0,), device=device, dtype=torch.float32),
            "expert_counts": torch.zeros((e_local,), device=device, dtype=torch.int32),
            "expert_offsets": torch.zeros((e_local + 1,), device=device, dtype=torch.int32),
            "Tk_total": 0,
            "max_count": 0,
        }

    local_expert_idx = (topk_idx[flat_token_idx, flat_slot_idx] - local_start).to(torch.int32)
    flat_weights = weights8[flat_token_idx, flat_slot_idx].to(torch.float32)

    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order]
    sorted_weights = flat_weights[order]

    expert_counts64 = torch.bincount(sorted_experts.to(torch.int64), minlength=e_local)
    expert_counts = expert_counts64.to(torch.int32)
    expert_offsets = torch.zeros((e_local + 1,), device=device, dtype=torch.int32)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)
    max_count = int(expert_counts.max().item()) if Tk_total > 0 else 0

    return {
        "topk_idx": topk_idx,
        "weights8": weights8,
        "sorted_tokens": sorted_tokens,
        "sorted_experts": sorted_experts,
        "sorted_weights": sorted_weights,
        "expert_counts": expert_counts,
        "expert_offsets": expert_offsets,
        "Tk_total": Tk_total,
        "max_count": max_count,
    }


# ----------------------------
# Triton kernels
# ----------------------------

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
def _grouped_gemm1_kernel(
    A_ptr, A_scale_ptr,
    W_ptr, W_scale_ptr,
    Out_ptr,
    sorted_tokens_ptr, expert_offsets_ptr,
    MAX_COUNT, H, N_dim,
    stride_am, stride_ak,
    stride_ascale_k, stride_ascale_t,
    stride_we, stride_wn, stride_wk,
    stride_wscale_e, stride_wscale_n, stride_wscale_k,
    stride_outm, stride_outn,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + pid_e).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + pid_e + 1).to(tl.int32)
    Tk = end_idx - start_idx

    row_start = pid_m * BLOCK_M
    if row_start >= Tk:
        return

    m_offs = row_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < Tk

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N_dim

    token_ids = tl.load(sorted_tokens_ptr + start_idx + m_offs, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_block = 0
    for k in range(0, H, BLOCK_K):
        k_offs = tl.arange(0, BLOCK_K)

        a_ptrs = (
            A_ptr
            + token_ids[:, None] * stride_am
            + k_offs[None, :] * stride_ak
            + k * stride_ak
        )
        w_ptrs = (
            W_ptr
            + pid_e.to(tl.int64) * stride_we
            + n_offs[None, :].to(tl.int64) * stride_wn
            + k_offs[:, None].to(tl.int64) * stride_wk
            + k * stride_wk
        )

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        a_scale_ptrs = A_scale_ptr + k_block * stride_ascale_k + token_ids[:, None] * stride_ascale_t
        a_scale = tl.load(a_scale_ptrs, mask=m_mask[:, None], other=0.0)

        wn_blocks = (n_offs // 128).to(tl.int64)
        w_scale_ptrs = (
            W_scale_ptr
            + pid_e.to(tl.int64) * stride_wscale_e
            + wn_blocks[None, :] * stride_wscale_n
            + k_block * stride_wscale_k
        )
        w_scale = tl.load(w_scale_ptrs, mask=n_mask[None, :], other=0.0)

        dot = tl.dot(a, w, out_dtype=tl.float32)
        acc += dot * a_scale * w_scale
        k_block += 1

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
def _grouped_gemm2_kernel(
    G1_ptr,
    W2_ptr, W2_scale_ptr,
    Routing_weights_ptr,
    Out_ptr,
    expert_offsets_ptr,
    MAX_COUNT, I, H,
    stride_g1m, stride_g1k,
    stride_w2e, stride_w2h, stride_w2i,
    stride_w2scale_e, stride_w2scale_h, stride_w2scale_i,
    stride_outm, stride_outn,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + pid_e).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + pid_e + 1).to(tl.int32)
    Tk = end_idx - start_idx

    row_start = pid_m * BLOCK_M
    if row_start >= Tk:
        return

    m_offs = row_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < Tk

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < H

    rows = (start_idx + m_offs).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    i_block = 0
    for i0 in range(0, I, BLOCK_K):
        i_offs = tl.arange(0, BLOCK_K)

        x1_ptrs = G1_ptr + rows[:, None] * stride_g1m + (i0 + i_offs[None, :]) * stride_g1k
        x2_ptrs = G1_ptr + rows[:, None] * stride_g1m + (I + i0 + i_offs[None, :]) * stride_g1k

        x1 = tl.load(x1_ptrs, mask=m_mask[:, None], other=0.0)
        x2 = tl.load(x2_ptrs, mask=m_mask[:, None], other=0.0)
        c = x1 * (x2 * tl.sigmoid(x2))

        w_ptrs = (
            W2_ptr
            + pid_e.to(tl.int64) * stride_w2e
            + n_offs[None, :].to(tl.int64) * stride_w2h
            + (i0 + i_offs[:, None]).to(tl.int64) * stride_w2i
        )
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        hn_blocks = (n_offs // 128).to(tl.int64)
        w_scale_ptrs = (
            W2_scale_ptr
            + pid_e.to(tl.int64) * stride_w2scale_e
            + hn_blocks[None, :] * stride_w2scale_h
            + i_block * stride_w2scale_i
        )
        w_scale = tl.load(w_scale_ptrs, mask=n_mask[None, :], other=0.0)

        dot = tl.dot(c, w, out_dtype=tl.float32)
        acc += dot * w_scale
        i_block += 1

    rw = tl.load(Routing_weights_ptr + rows, mask=m_mask, other=0.0)
    acc *= rw[:, None]

    out_ptrs = (
        Out_ptr
        + rows[:, None] * stride_outm
        + n_offs[None, :].to(tl.int64) * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ----------------------------
# Exact fallback
# ----------------------------

def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    x = hidden_states[token_ids].float()  # [Tk, H]
    scales = hidden_states_scale[:, token_ids].transpose(0, 1).contiguous()  # [Tk, 56]
    return x.view(x.shape[0], 56, 128) * scales[:, :, None]


def _dequant_w1_expert(w, s):
    # w: [4096, 7168], s: [32, 56]
    wf = w.float().view(32, 128, 56, 128).permute(0, 2, 1, 3)
    out = wf * s[:, :, None, None]
    return out.permute(0, 2, 1, 3).reshape(4096, 7168)


def _dequant_w2_expert(w, s):
    # w: [7168, 2048], s: [56, 16]
    wf = w.float().view(56, 128, 16, 128).permute(0, 2, 1, 3)
    out = wf * s[:, :, None, None]
    return out.permute(0, 2, 1, 3).reshape(7168, 2048)


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
    device = hidden_states.device
    T = hidden_states.shape[0]
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]

    dispatch = _route_and_dispatch(
        routing_logits, routing_bias, local_expert_offset, E_local, routed_scaling_factor
    )
    Tk_total = dispatch["Tk_total"]
    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_tokens = dispatch["sorted_tokens"]
    sorted_weights = dispatch["sorted_weights"]
    expert_offsets = dispatch["expert_offsets"]

    out = torch.zeros((T, H), dtype=torch.float32, device=device)

    for e in range(E_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        tok = sorted_tokens[start:end].long()
        rw = sorted_weights[start:end]

        a = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok).reshape(-1, H)
        w1 = _dequant_w1_expert(gemm1_weights[e], gemm1_weights_scale[e])
        g1 = a @ w1.t()
        x1 = g1[:, :I]
        x2 = g1[:, I:]
        c = torch.nn.functional.silu(x2) * x1
        w2 = _dequant_w2_expert(gemm2_weights[e], gemm2_weights_scale[e])
        o = c @ w2.t()
        o = o * rw[:, None]
        out.index_add_(0, tok, o)

    return out.to(torch.bfloat16)


# ----------------------------
# Main entry
# ----------------------------

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
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]

    dispatch = _route_and_dispatch(
        routing_logits, routing_bias, local_expert_offset, E_local, routed_scaling_factor
    )

    Tk_total = dispatch["Tk_total"]
    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_tokens = dispatch["sorted_tokens"]
    sorted_experts = dispatch["sorted_experts"]
    sorted_weights = dispatch["sorted_weights"]
    expert_counts = dispatch["expert_counts"]
    expert_offsets = dispatch["expert_offsets"]
    max_count = dispatch["max_count"]

    if (
        expert_offsets.numel() != E_local + 1
        or int(expert_offsets[0].item()) != 0
        or int(expert_offsets[-1].item()) != Tk_total
        or not torch.equal((expert_offsets[1:] - expert_offsets[:-1]).to(torch.int32), expert_counts.to(torch.int32))
    ):
        return _fallback_run(
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

    try:
        G1 = torch.zeros((Tk_total, 2 * I), dtype=torch.float32, device=device)
        grid1 = lambda META: (
            triton.cdiv(2 * I, META["BLOCK_N"]),
            triton.cdiv(max_count, META["BLOCK_M"]),
            E_local,
        )

        _grouped_gemm1_kernel[grid1](
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            G1,
            sorted_tokens, expert_offsets,
            max_count, H, 2 * I,
            hidden_states.stride(0), hidden_states.stride(1),
            hidden_states_scale.stride(0), hidden_states_scale.stride(1),
            gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
            gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
            G1.stride(0), G1.stride(1),
            BLOCK_K=128,
        )

        O = torch.zeros((Tk_total, H), dtype=torch.float32, device=device)
        grid2 = lambda META: (
            triton.cdiv(H, META["BLOCK_N"]),
            triton.cdiv(max_count, META["BLOCK_M"]),
            E_local,
        )

        _grouped_gemm2_kernel[grid2](
            G1,
            gemm2_weights, gemm2_weights_scale,
            sorted_weights,
            O,
            expert_offsets,
            max_count, I, H,
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