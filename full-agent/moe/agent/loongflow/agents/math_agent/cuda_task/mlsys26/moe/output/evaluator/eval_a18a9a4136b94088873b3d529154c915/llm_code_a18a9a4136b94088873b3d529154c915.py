import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=3),
    ],
    key=["MAX_TK"],
)
@triton.jit
def grouped_gemm1_kernel(
    A_ptr, A_scale_ptr, W_ptr, W_scale_ptr, Out_ptr,
    sorted_tokens_ptr, expert_offsets_ptr,
    H: tl.constexpr, N_dim: tl.constexpr, MAX_TK,
    stride_am, stride_ak,
    stride_ascale_k, stride_ascale_m,
    stride_we, stride_wn, stride_wk,
    stride_wscale_e, stride_wscale_n, stride_wscale_k,
    stride_outm, stride_outn,
    BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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

    k0 = tl.arange(0, BLOCK_K).to(tl.int64)

    a_ptrs = A_ptr + token_ids[:, None] * stride_am + k0[None, :] * stride_ak
    w_ptrs = (
        W_ptr
        + e_idx.to(tl.int64) * stride_we
        + n_offs[:, None] * stride_wn
        + k0[None, :] * stride_wk
    )

    scale_n_offs = n_offs // 128
    a_scale_base = A_scale_ptr + token_ids[:, None] * stride_ascale_m
    w_scale_base = (
        W_scale_ptr
        + e_idx.to(tl.int64) * stride_wscale_e
        + scale_n_offs[None, :] * stride_wscale_n
    )

    for k in tl.range(0, H, BLOCK_K):
        k_block = k // BLOCK_K

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        dot = tl.dot(a, tl.trans(w), out_dtype=tl.float32)

        a_scale = tl.load(a_scale_base + k_block * stride_ascale_k, mask=m_mask[:, None], other=0.0)
        w_scale = tl.load(w_scale_base + k_block * stride_wscale_k, mask=n_mask[None, :], other=0.0)

        acc += dot * a_scale * w_scale

        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = (
        Out_ptr
        + routed_rows[:, None].to(tl.int64) * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=3),
    ],
    key=["MAX_TK"],
)
@triton.jit
def grouped_gemm2_kernel(
    G1_ptr, W2_ptr, W2_scale_ptr, Out_ptr,
    Routing_weights_ptr, expert_offsets_ptr,
    H: tl.constexpr, I: tl.constexpr, MAX_TK,
    stride_g1m, stride_g1k,
    stride_w2e, stride_w2h, stride_w2i,
    stride_w2scale_e, stride_w2scale_h, stride_w2scale_i,
    stride_outm, stride_outn,
    BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    start = tl.load(expert_offsets_ptr + pid_e)
    end = tl.load(expert_offsets_ptr + pid_e + 1)
    Tk = end - start

    if pid_m * BLOCK_M >= Tk:
        return

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < Tk

    n_offs = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_offs < H
    n_block_idx = pid_n  # BLOCK_N=128 always

    row_idx = (start + m_offs).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_range = tl.arange(0, BLOCK_K).to(tl.int64)

    scale_n_offs = n_offs // 128
    w2_scale_base = (
        W2_scale_ptr
        + pid_e.to(tl.int64) * stride_w2scale_e
        + n_block_idx * stride_w2scale_h
    )

    for k_start in tl.range(0, I, BLOCK_K):
        k_block = k_start // BLOCK_K
        k_offs = (k_start + k_range).to(tl.int64)

        x1 = tl.load(
            G1_ptr + row_idx[:, None] * stride_g1m + k_offs[None, :] * stride_g1k,
            mask=m_mask[:, None], other=0.0,
        )
        x2 = tl.load(
            G1_ptr + row_idx[:, None] * stride_g1m + (I + k_offs)[None, :] * stride_g1k,
            mask=m_mask[:, None], other=0.0,
        )
        c = x1 * x2 * tl.sigmoid(x2)  # float32

        w2 = tl.load(
            W2_ptr
            + pid_e.to(tl.int64) * stride_w2e
            + n_offs[None, :] * stride_w2h
            + k_offs[:, None] * stride_w2i,
            mask=n_mask[None, :], other=0.0,
        )
        w2_scale = tl.load(w2_scale_base + k_block * stride_w2scale_i)

        c_bf = c.to(tl.bfloat16)
        w2_bf = w2.to(tl.bfloat16)
        dot = tl.dot(c_bf, w2_bf, out_dtype=tl.float32)
        acc += dot * w2_scale

    rw = tl.load(Routing_weights_ptr + row_idx, mask=m_mask, other=0.0)
    acc = acc * rw[:, None]

    out_ptrs = (
        Out_ptr
        + row_idx[:, None] * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _route_tokens(routing_logits, routing_bias, local_expert_offset, e_local, routed_scaling_factor):
    device = routing_logits.device
    T, E_global = routing_logits.shape
    N_GROUP = 8
    TOPK_GROUP = 4
    TOP_K = 8
    GROUP_SIZE = E_global // N_GROUP

    q = torch.sigmoid(routing_logits.float())
    rank_scores = q + routing_bias.float().view(1, E_global)

    grouped = rank_scores.view(T, N_GROUP, GROUP_SIZE)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)
    expert_allowed = group_mask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_global)

    pruned = torch.where(expert_allowed, rank_scores, torch.full_like(rank_scores, float("-inf")))
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    sel = torch.zeros((T, E_global), dtype=torch.bool, device=device)
    sel.scatter_(1, topk_idx, True)

    raw_w = q * sel
    denom = raw_w.sum(dim=1, keepdim=True) + 1e-20
    weights = raw_w * (routed_scaling_factor / denom)

    local_start = int(local_expert_offset)
    expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)

    flat_token_idx, flat_k_idx = torch.nonzero(expert_mask, as_tuple=True)
    if flat_token_idx.numel() == 0:
        return None

    chosen_experts = topk_idx[flat_token_idx, flat_k_idx]
    local_expert_idx = chosen_experts - local_start
    chosen_weights = weights[flat_token_idx, chosen_experts]

    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order].to(torch.int32)
    sorted_weights = chosen_weights[order].to(torch.float32)

    expert_counts = torch.bincount(sorted_experts.to(torch.int64), minlength=e_local)
    expert_offsets = torch.zeros(e_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts.to(torch.int32), dim=0)

    return sorted_tokens, sorted_experts, sorted_weights, expert_counts.to(torch.int32), expert_offsets


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
    E_local = gemm1_weights.shape[0]

    routed = _route_tokens(
        routing_logits, routing_bias, local_expert_offset, E_local, routed_scaling_factor,
    )
    if routed is None:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets = routed
    Tk_total = sorted_tokens.numel()

    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    max_count = int(expert_counts.max().item())
    if max_count == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    G1 = torch.zeros((Tk_total, 2 * I), dtype=torch.float32, device=device)

    grid_gemm1 = lambda META: (
        triton.cdiv(2 * I, META["BLOCK_N"]),
        triton.cdiv(max_count, META["BLOCK_M"]),
        E_local,
    )

    grouped_gemm1_kernel[grid_gemm1](
        A_ptr=hidden_states,
        A_scale_ptr=hidden_states_scale,
        W_ptr=gemm1_weights,
        W_scale_ptr=gemm1_weights_scale,
        Out_ptr=G1,
        sorted_tokens_ptr=sorted_tokens,
        expert_offsets_ptr=expert_offsets,
        H=H, N_dim=2 * I, MAX_TK=max_count,
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

    O = torch.zeros((Tk_total, H), dtype=torch.float32, device=device)

    grid_gemm2 = lambda META: (
        triton.cdiv(H, META["BLOCK_N"]),
        triton.cdiv(max_count, META["BLOCK_M"]),
        E_local,
    )

    grouped_gemm2_kernel[grid_gemm2](
        G1_ptr=G1,
        W2_ptr=gemm2_weights,
        W2_scale_ptr=gemm2_weights_scale,
        Out_ptr=O,
        Routing_weights_ptr=sorted_weights,
        expert_offsets_ptr=expert_offsets,
        H=H, I=I, MAX_TK=max_count,
        stride_g1m=G1.stride(0),
        stride_g1k=G1.stride(1),
        stride_w2e=gemm2_weights.stride(0),
        stride_w2h=gemm2_weights.stride(1),
        stride_w2i=gemm2_weights.stride(2),
        stride_w2scale_e=gemm2_weights_scale.stride(0),
        stride_w2scale_h=gemm2_weights_scale.stride(1),
        stride_w2scale_i=gemm2_weights_scale.stride(2),
        stride_outm=O.stride(0),
        stride_outn=O.stride(1),
        BLOCK_K=128,
    )

    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    output.index_add_(0, sorted_tokens.long(), O)
    return output.to(torch.bfloat16)