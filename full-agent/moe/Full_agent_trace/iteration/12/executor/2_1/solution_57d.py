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
    key=["MAX_COUNT"],
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
    MAX_COUNT,
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
    n_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    e_idx = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + e_idx)
    end_idx = tl.load(expert_offsets_ptr + e_idx + 1)
    Tk = end_idx - start_idx

    if m_idx * BLOCK_M >= Tk:
        return

    m_offs = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < Tk
    n_mask = n_offs < N_dim

    token_ids = tl.load(sorted_tokens_ptr + start_idx + m_offs, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        k_offs = tl.arange(0, BLOCK_K).to(tl.int64)
        k_block = k // BLOCK_K

        a_ptrs = A_ptr + token_ids[:, None] * stride_am + (k + k_offs[None, :]) * stride_ak
        w_ptrs = (
            W_ptr
            + e_idx.to(tl.int64) * stride_we
            + n_offs[None, :].to(tl.int64) * stride_wn
            + (k + k_offs[:, None]) * stride_wk
        )

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        dot_res = tl.dot(a, w, out_dtype=tl.float32)

        a_scale_ptrs = A_scale_ptr + k_block * stride_ascale_k + token_ids * stride_ascale_m
        a_scale = tl.load(a_scale_ptrs, mask=m_mask, other=0.0)[:, None]

        w_scale_ptrs = (
            W_scale_ptr
            + e_idx.to(tl.int64) * stride_wscale_e
            + (n_offs // 128).to(tl.int64) * stride_wscale_n
            + k_block * stride_wscale_k
        )
        w_scale = tl.load(w_scale_ptrs, mask=n_mask, other=0.0)[None, :]

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
    key=["MAX_COUNT"],
)
@triton.jit
def grouped_gemm2_kernel(
    G1_ptr,
    W2_ptr,
    W2_scale_ptr,
    Routing_weights_ptr,
    Out_ptr,
    expert_offsets_ptr,
    MAX_COUNT,
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
    n_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    e_idx = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + e_idx)
    end_idx = tl.load(expert_offsets_ptr + e_idx + 1)
    Tk = end_idx - start_idx

    if m_idx * BLOCK_M >= Tk:
        return

    m_offs = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < Tk
    n_mask = n_offs < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    row_ids = (start_idx + m_offs).to(tl.int64)

    for k in range(0, I, BLOCK_K):
        k_offs = tl.arange(0, BLOCK_K).to(tl.int64)
        k_block = k // BLOCK_K

        x1_ptrs = G1_ptr + row_ids[:, None] * stride_g1m + (k + k_offs[None, :]) * stride_g1k
        x2_ptrs = G1_ptr + row_ids[:, None] * stride_g1m + (I + k + k_offs[None, :]) * stride_g1k

        x1 = tl.load(x1_ptrs, mask=m_mask[:, None], other=0.0)
        x2 = tl.load(x2_ptrs, mask=m_mask[:, None], other=0.0)
        c = x1 * x2 * tl.sigmoid(x2)

        w2_ptrs = (
            W2_ptr
            + e_idx.to(tl.int64) * stride_w2e
            + n_offs[None, :].to(tl.int64) * stride_w2n
            + (k + k_offs[:, None]) * stride_w2k
        )
        w = tl.load(w2_ptrs, mask=n_mask[None, :], other=0.0)

        dot_res = tl.dot(c.to(w.dtype), w, out_dtype=tl.float32)

        w_scale_ptrs = (
            W2_scale_ptr
            + e_idx.to(tl.int64) * stride_w2scale_e
            + (n_offs // 128).to(tl.int64) * stride_w2scale_n
            + k_block * stride_w2scale_k
        )
        w_scale = tl.load(w_scale_ptrs, mask=n_mask, other=0.0)[None, :]

        acc += dot_res * w_scale

    routing_w = tl.load(Routing_weights_ptr + row_ids, mask=m_mask, other=0.0)
    acc *= routing_w[:, None]

    out_ptrs = (
        Out_ptr
        + row_ids[:, None] * stride_outm
        + n_offs[None, :].to(tl.int64) * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _route(routing_logits, routing_bias, routed_scaling_factor):
    logits = routing_logits.float()
    bias = routing_bias.float().reshape(-1)

    q = torch.sigmoid(logits)
    score = q + bias

    T, E = q.shape
    n_group = 8
    topk_group = 4
    top_k = 8
    group_size = E // n_group

    score_grouped = score.view(T, n_group, group_size)
    top2_vals, _ = torch.topk(score_grouped, k=2, dim=2, largest=True, sorted=True)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=topk_group, dim=1, largest=True, sorted=True)
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_idx, True)

    expert_mask = group_mask.unsqueeze(-1).expand(T, n_group, group_size).reshape(T, E)
    neg_inf = torch.finfo(score.dtype).min
    pruned = score.masked_fill(~expert_mask, neg_inf)

    _, topk_idx = torch.topk(pruned, k=top_k, dim=1, largest=True, sorted=True)

    topk_q = torch.gather(q, 1, topk_idx)
    topk_w = topk_q / (topk_q.sum(dim=1, keepdim=True) + 1e-20)
    topk_w = topk_w * routed_scaling_factor
    return topk_idx, topk_w


def _build_local_dispatch(topk_idx, topk_w, local_expert_offset, e_local):
    local_start = int(local_expert_offset)
    mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)
    flat_token_idx, flat_k_idx = torch.nonzero(mask, as_tuple=True)
    tk_total = flat_token_idx.numel()
    if tk_total == 0:
        return None

    local_expert_idx = topk_idx[flat_token_idx, flat_k_idx] - local_start
    order = torch.argsort(local_expert_idx)

    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order].to(torch.int32)
    sorted_weights = topk_w[flat_token_idx, flat_k_idx][order].to(torch.float32)

    expert_counts = torch.bincount(sorted_experts, minlength=e_local)
    expert_offsets = torch.zeros(e_local + 1, dtype=torch.int32, device=topk_idx.device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    return sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets


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
    T = routing_logits.shape[0]
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]

    topk_idx, topk_w = _route(routing_logits, routing_bias, routed_scaling_factor)
    dispatch = _build_local_dispatch(topk_idx, topk_w, local_expert_offset, E_local)

    if dispatch is None:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets = dispatch
    tk_total = sorted_tokens.numel()

    if tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    if (
        expert_offsets.numel() != E_local + 1
        or int(expert_offsets[0].item()) != 0
        or int(expert_offsets[-1].item()) != tk_total
    ):
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    max_count = int(expert_counts.max().item()) if tk_total > 0 else 0
    if max_count == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    G1 = torch.empty((tk_total, 2 * I), dtype=torch.float32, device=device)
    O = torch.empty((tk_total, H), dtype=torch.float32, device=device)

    grid1 = lambda META: (triton.cdiv(2 * I, META["BLOCK_N"]), triton.cdiv(max_count, META["BLOCK_M"]), E_local)
    grouped_gemm1_kernel[grid1](
        A_ptr=hidden_states,
        A_scale_ptr=hidden_states_scale,
        W_ptr=gemm1_weights,
        W_scale_ptr=gemm1_weights_scale,
        Out_ptr=G1,
        sorted_tokens_ptr=sorted_tokens,
        expert_offsets_ptr=expert_offsets,
        MAX_COUNT=max_count,
        H=H,
        N_dim=2 * I,
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

    grid2 = lambda META: (triton.cdiv(H, META["BLOCK_N"]), triton.cdiv(max_count, META["BLOCK_M"]), E_local)
    grouped_gemm2_kernel[grid2](
        G1_ptr=G1,
        W2_ptr=gemm2_weights,
        W2_scale_ptr=gemm2_weights_scale,
        Routing_weights_ptr=sorted_weights,
        Out_ptr=O,
        expert_offsets_ptr=expert_offsets,
        MAX_COUNT=max_count,
        I=I,
        H=H,
        stride_g1m=G1.stride(0),
        stride_g1k=G1.stride(1),
        stride_w2e=gemm2_weights.stride(0),
        stride_w2n=gemm2_weights.stride(1),
        stride_w2k=gemm2_weights.stride(2),
        stride_w2scale_e=gemm2_weights_scale.stride(0),
        stride_w2scale_n=gemm2_weights_scale.stride(1),
        stride_w2scale_k=gemm2_weights_scale.stride(2),
        stride_outm=O.stride(0),
        stride_outn=O.stride(1),
        BLOCK_K=128,
    )

    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    output.index_add_(0, sorted_tokens.long(), O)
    return output.to(torch.bfloat16)