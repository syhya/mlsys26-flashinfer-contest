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
    key=["max_count"],
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
    max_count,
    H,
    N_dim,
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

    for k_base in range(0, H, BLOCK_K):
        k_offs = k_base + tl.arange(0, BLOCK_K)
        k_blk = k_base // BLOCK_K

        a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
        w_ptrs = (
            W_ptr
            + pid_e.to(tl.int64) * stride_we
            + n_offs[None, :].to(tl.int64) * stride_wn
            + k_offs[:, None].to(tl.int64) * stride_wk
        )

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        dot_res = tl.dot(a, w, out_dtype=tl.float32)

        a_scale = tl.load(
            A_scale_ptr
            + k_blk * stride_ascale_k
            + token_ids[:, None] * stride_ascale_m,
            mask=m_mask[:, None],
            other=0.0,
        )
        w_scale = tl.load(
            W_scale_ptr
            + pid_e.to(tl.int64) * stride_wscale_e
            + (n_offs // 128)[None, :].to(tl.int64) * stride_wscale_n
            + k_blk * stride_wscale_k,
            mask=n_mask[None, :],
            other=0.0,
        )

        acc += dot_res * a_scale * w_scale

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
    key=["max_count"],
)
@triton.jit
def grouped_gemm2_kernel(
    G1_ptr,
    W2_ptr,
    W2_scale_ptr,
    Routing_weights_ptr,
    Out_ptr,
    expert_offsets_ptr,
    max_count,
    I,
    H,
    stride_g1m,
    stride_g1k,
    stride_w2e,
    stride_w2n,
    stride_w2k,
    stride_w2scale_e,
    stride_w2scale_n,
    stride_w2scale_k,
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
    Tk = end_idx - start_idx

    row_start = pid_m * BLOCK_M
    if row_start >= Tk:
        return

    m_offs = row_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < Tk

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    row_ids = (start_idx + m_offs).to(tl.int64)
    routing = tl.load(Routing_weights_ptr + row_ids, mask=m_mask, other=0.0)

    for k_base in range(0, I, BLOCK_K):
        k_offs = k_base + tl.arange(0, BLOCK_K)
        k_blk = k_base // BLOCK_K

        x1_ptrs = G1_ptr + row_ids[:, None] * stride_g1m + k_offs[None, :].to(tl.int64) * stride_g1k
        x2_ptrs = G1_ptr + row_ids[:, None] * stride_g1m + (I + k_offs)[None, :].to(tl.int64) * stride_g1k

        x1 = tl.load(x1_ptrs, mask=m_mask[:, None], other=0.0)
        x2 = tl.load(x2_ptrs, mask=m_mask[:, None], other=0.0)
        c = x1 * x2 * tl.sigmoid(x2)

        w_ptrs = (
            W2_ptr
            + pid_e.to(tl.int64) * stride_w2e
            + n_offs[None, :].to(tl.int64) * stride_w2n
            + k_offs[:, None].to(tl.int64) * stride_w2k
        )
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
        dot_res = tl.dot(c.to(w.dtype), w, out_dtype=tl.float32)

        w_scale = tl.load(
            W2_scale_ptr
            + pid_e.to(tl.int64) * stride_w2scale_e
            + (n_offs // 128)[None, :].to(tl.int64) * stride_w2scale_n
            + k_blk * stride_w2scale_k,
            mask=n_mask[None, :],
            other=0.0,
        )
        acc += dot_res * w_scale

    acc *= routing[:, None]

    out_ptrs = (
        Out_ptr
        + row_ids[:, None] * stride_outm
        + n_offs[None, :].to(tl.int64) * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _route_and_dispatch(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    e_local: int,
):
    device = routing_logits.device
    T, E_global = routing_logits.shape
    assert E_global == 256

    logits = routing_logits.float()
    q = torch.sigmoid(logits)
    rank_scores = q + routing_bias.float().view(1, -1)

    n_group = 8
    group_size = E_global // n_group
    topk_group = 4
    top_k = 8

    grouped = rank_scores.view(T, n_group, group_size)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=topk_group, dim=1, largest=True, sorted=True).indices
    group_keep = torch.zeros((T, n_group), dtype=torch.bool, device=device)
    group_keep.scatter_(1, top_groups, True)

    expert_keep = group_keep.unsqueeze(-1).expand(T, n_group, group_size).reshape(T, E_global)
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=rank_scores.dtype)
    pruned = torch.where(expert_keep, rank_scores, neg_inf)

    topk_idx = torch.topk(pruned, k=top_k, dim=1, largest=True, sorted=True).indices

    chosen_q = torch.gather(q, 1, topk_idx)
    chosen_q_sum = chosen_q.sum(dim=1, keepdim=True)
    chosen_w = chosen_q * (routed_scaling_factor / (chosen_q_sum + 1e-20))

    local_start = int(local_expert_offset)
    local_end = local_start + e_local
    local_mask = (topk_idx >= local_start) & (topk_idx < local_end)

    flat_token_idx, flat_slot_idx = torch.nonzero(local_mask, as_tuple=True)
    Tk_total = int(flat_token_idx.numel())
    if Tk_total == 0:
        return None

    local_expert_idx = (topk_idx[flat_token_idx, flat_slot_idx] - local_start).to(torch.int32)
    flat_weights = chosen_w[flat_token_idx, flat_slot_idx].to(torch.float32)

    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order]
    sorted_weights = flat_weights[order]

    expert_counts = torch.bincount(sorted_experts, minlength=e_local)
    expert_offsets = torch.empty(e_local + 1, dtype=torch.int32, device=device)
    expert_offsets[0] = 0
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0).to(torch.int32)

    max_count = int(expert_counts.max().item()) if Tk_total > 0 else 0

    return sorted_tokens, sorted_experts, sorted_weights, expert_offsets, Tk_total, max_count


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
    T = routing_logits.shape[0]
    device = hidden_states.device

    routed = _route_and_dispatch(
        routing_logits,
        routing_bias,
        local_expert_offset,
        routed_scaling_factor,
        E_local,
    )

    if routed is None:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_tokens, sorted_experts, sorted_weights, expert_offsets, Tk_total, max_count = routed

    if (
        sorted_tokens.numel() != Tk_total
        or sorted_experts.numel() != Tk_total
        or sorted_weights.numel() != Tk_total
        or expert_offsets[0].item() != 0
        or expert_offsets[-1].item() != Tk_total
    ):
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    G1 = torch.empty((Tk_total, 2 * I), dtype=torch.float32, device=device)
    O = torch.empty((Tk_total, H), dtype=torch.float32, device=device)

    grid1 = lambda META: (triton.cdiv(2 * I, META["BLOCK_N"]), triton.cdiv(max_count, META["BLOCK_M"]), E_local)
    grouped_gemm1_kernel[grid1](
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        G1,
        sorted_tokens,
        expert_offsets,
        max_count,
        H,
        2 * I,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states_scale.stride(0),
        hidden_states_scale.stride(1),
        gemm1_weights.stride(0),
        gemm1_weights.stride(1),
        gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0),
        gemm1_weights_scale.stride(1),
        gemm1_weights_scale.stride(2),
        G1.stride(0),
        G1.stride(1),
        BLOCK_K=128,
    )

    grid2 = lambda META: (triton.cdiv(H, META["BLOCK_N"]), triton.cdiv(max_count, META["BLOCK_M"]), E_local)
    grouped_gemm2_kernel[grid2](
        G1,
        gemm2_weights,
        gemm2_weights_scale,
        sorted_weights,
        O,
        expert_offsets,
        max_count,
        I,
        H,
        G1.stride(0),
        G1.stride(1),
        gemm2_weights.stride(0),
        gemm2_weights.stride(1),
        gemm2_weights.stride(2),
        gemm2_weights_scale.stride(0),
        gemm2_weights_scale.stride(1),
        gemm2_weights_scale.stride(2),
        O.stride(0),
        O.stride(1),
        BLOCK_K=128,
    )

    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    output.index_add_(0, sorted_tokens.long(), O)
    return output.to(torch.bfloat16)