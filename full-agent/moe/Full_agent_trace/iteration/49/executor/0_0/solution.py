import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=4),
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
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + pid_e)
    end_idx = tl.load(expert_offsets_ptr + pid_e + 1)
    num_tokens = end_idx - start_idx

    row_start = pid_m * BLOCK_M
    if row_start >= num_tokens:
        return

    m_offs = row_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < num_tokens
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N_dim

    token_ids = tl.load(sorted_tokens_ptr + start_idx + m_offs, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_range = tl.arange(0, BLOCK_K)

    for k_start in tl.range(0, H, BLOCK_K):
        k_block_idx = k_start // BLOCK_K
        k_offs = k_start + k_range

        a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_fp8 = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)

        a_scale_ptrs = A_scale_ptr + k_block_idx * stride_ascale_k + token_ids[:, None] * stride_ascale_m
        a_scale = tl.load(a_scale_ptrs, mask=m_mask[:, None], other=1.0)

        w_ptrs = (
            W_ptr
            + pid_e.to(tl.int64) * stride_we
            + n_offs[None, :] * stride_wn
            + k_offs[:, None] * stride_wk
        )
        w_fp8 = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        n_block_idx = n_offs // 128
        w_scale_ptrs = (
            W_scale_ptr
            + pid_e.to(tl.int64) * stride_wscale_e
            + n_block_idx[None, :] * stride_wscale_n
            + k_block_idx * stride_wscale_k
        )
        w_scale = tl.load(w_scale_ptrs, mask=n_mask[None, :], other=1.0)

        dot_result = tl.dot(a_fp8, w_fp8, out_dtype=tl.float32)
        acc += dot_result * a_scale * w_scale

    out_ptrs = (
        Out_ptr
        + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# Parent GEMM2 restored because plan fallback underperformed badly in evaluation.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 128}, num_warps=4,  num_stages=4),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128}, num_warps=4,  num_stages=4),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128}, num_warps=4,  num_stages=5),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4,  num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4,  num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4,  num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8,  num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8,  num_stages=4),
    ],
    key=["MAX_TK"],
)
@triton.jit
def grouped_gemm2_kernel(
    G1_ptr, W2_ptr, W2_scale_ptr, Routing_weights_ptr, Out_ptr,
    expert_offsets_ptr,
    I: tl.constexpr, H: tl.constexpr, MAX_TK,
    stride_g1m, stride_g1k,
    stride_w2e, stride_w2n, stride_w2k,
    stride_w2scale_e, stride_w2scale_n, stride_w2scale_k,
    stride_outm, stride_outn,
    BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + pid_e)
    end_idx = tl.load(expert_offsets_ptr + pid_e + 1)
    num_tokens = end_idx - start_idx
    if pid_m * BLOCK_M >= num_tokens:
        return

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < num_tokens
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < H
    n_block_idx = pid_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_range = tl.arange(0, BLOCK_K)
    row_indices = (start_idx + m_offs).to(tl.int64)

    w2_base = W2_ptr + pid_e.to(tl.int64) * stride_w2e + n_offs[None, :] * stride_w2n
    w2scale_base = W2_scale_ptr + pid_e.to(tl.int64) * stride_w2scale_e + n_block_idx * stride_w2scale_n

    for k_start in tl.range(0, I, BLOCK_K):
        k_block_idx = k_start // BLOCK_K
        k_offs = k_start + k_range

        x1_ptrs = G1_ptr + row_indices[:, None] * stride_g1m + k_offs[None, :] * stride_g1k
        x2_ptrs = G1_ptr + row_indices[:, None] * stride_g1m + (I + k_offs)[None, :] * stride_g1k
        x1 = tl.load(x1_ptrs, mask=m_mask[:, None], other=0.0)
        x2 = tl.load(x2_ptrs, mask=m_mask[:, None], other=0.0)

        c = x1 * x2 * tl.sigmoid(x2)

        c_abs_max = tl.max(tl.abs(c))
        c_scale = tl.maximum(c_abs_max / 448.0, 1e-12)
        c_fp8 = (c / c_scale).to(tl.float8e4nv)

        w2_ptrs = w2_base + k_offs[:, None] * stride_w2k
        w2_fp8 = tl.load(w2_ptrs, mask=n_mask[None, :], other=0.0)

        w2_scale = tl.load(w2scale_base + k_block_idx * stride_w2scale_k)

        dot_result = tl.dot(c_fp8, w2_fp8, out_dtype=tl.float32)
        acc += dot_result * c_scale * w2_scale

    routing_w = tl.load(Routing_weights_ptr + row_indices, mask=m_mask, other=0.0)
    acc = acc * routing_w[:, None]

    out_ptrs = (
        Out_ptr
        + row_indices[:, None] * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _route_tokens(routing_logits, routing_bias, local_expert_offset, e_local, routed_scaling_factor):
    device = routing_logits.device
    T, E_global = routing_logits.shape
    N_GROUP = 8
    TOPK_GROUP = 4
    TOP_K = 8
    group_size = E_global // N_GROUP

    q = torch.sigmoid(routing_logits.float())
    rank_scores = q + routing_bias.float().view(1, E_global)

    grouped = rank_scores.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True)
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)
    expert_allowed = group_mask.unsqueeze(-1).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=rank_scores.dtype)
    pruned = torch.where(expert_allowed, rank_scores, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    sel = torch.zeros((T, E_global), dtype=torch.bool, device=device)
    sel.scatter_(1, topk_idx, True)
    raw_w = q * sel
    denom = raw_w.sum(dim=1, keepdim=True) + 1e-20
    weights = (raw_w / denom) * routed_scaling_factor

    local_start = int(local_expert_offset)

    flat_expert = topk_idx.reshape(-1)
    flat_token = torch.arange(T, device=device, dtype=torch.int64).repeat_interleave(TOP_K)
    flat_weight = weights.gather(1, topk_idx).reshape(-1)

    shifted = flat_expert - local_start
    in_range = (shifted >= 0) & (shifted < e_local)
    sentinel = torch.full_like(shifted, e_local)
    sort_key = torch.where(in_range, shifted, sentinel)

    order = torch.argsort(sort_key, stable=True)
    sort_key_sorted = sort_key[order]

    num_valid_t = (sort_key_sorted < e_local).sum()
    num_valid = int(num_valid_t.item())
    if num_valid == 0:
        return None

    valid_order = order[:num_valid]
    sorted_tokens = flat_token[valid_order].to(torch.int32)
    sorted_experts = shifted[valid_order].to(torch.int32)
    sorted_weights = flat_weight[valid_order].to(torch.float32)

    expert_counts = torch.bincount(sorted_experts, minlength=e_local)
    expert_offsets = torch.zeros(e_local + 1, dtype=torch.int32, device=device)
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
    H = 7168
    I = 2048
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]

    routed = _route_tokens(
        routing_logits, routing_bias,
        local_expert_offset, E_local, routed_scaling_factor,
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
        Routing_weights_ptr=sorted_weights,
        Out_ptr=O,
        expert_offsets_ptr=expert_offsets,
        I=I,
        H=H,
        MAX_TK=max_count,
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
