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

    start_idx = tl.load(expert_offsets_ptr + e_idx).to(tl.int64)
    end_idx = tl.load(expert_offsets_ptr + e_idx + 1).to(tl.int64)
    Tk = end_idx - start_idx

    if m_idx * BLOCK_M >= Tk:
        return

    m_offs = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = (n_idx * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    m_mask = m_offs < Tk
    n_mask = n_offs < N_dim

    token_ids = tl.load(sorted_tokens_ptr + start_idx + m_offs, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        k_offs = (k + tl.arange(0, BLOCK_K)).to(tl.int64)

        a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
        w_ptrs = (
            W_ptr
            + e_idx.to(tl.int64) * stride_we
            + n_offs[None, :] * stride_wn
            + k_offs[:, None] * stride_wk
        )

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        dot_res = tl.dot(a, w, out_dtype=tl.float32)

        k_idx = k // BLOCK_K
        a_scale = tl.load(
            A_scale_ptr + k_idx * stride_ascale_k + token_ids[:, None] * stride_ascale_m,
            mask=m_mask[:, None],
            other=0.0,
        )
        w_scale = tl.load(
            W_scale_ptr
            + e_idx.to(tl.int64) * stride_wscale_e
            + (n_offs[None, :] // 128) * stride_wscale_n
            + k_idx * stride_wscale_k,
            mask=n_mask[None, :],
            other=0.0,
        )

        acc += dot_res * a_scale * w_scale

    out_ptrs = Out_ptr + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm + n_offs[None, :] * stride_outn
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

    start_idx = tl.load(expert_offsets_ptr + e_idx).to(tl.int64)
    end_idx = tl.load(expert_offsets_ptr + e_idx + 1).to(tl.int64)
    Tk = end_idx - start_idx

    if m_idx * BLOCK_M >= Tk:
        return

    m_offs = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = (n_idx * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    m_mask = m_offs < Tk
    n_mask = n_offs < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    row_ids = (start_idx + m_offs).to(tl.int64)

    for k in range(0, I, BLOCK_K):
        k_offs = (k + tl.arange(0, BLOCK_K)).to(tl.int64)

        x1_ptrs = G1_ptr + row_ids[:, None] * stride_g1m + k_offs[None, :] * stride_g1k
        x2_ptrs = G1_ptr + row_ids[:, None] * stride_g1m + (I + k_offs)[None, :] * stride_g1k

        x1 = tl.load(x1_ptrs, mask=m_mask[:, None], other=0.0)
        x2 = tl.load(x2_ptrs, mask=m_mask[:, None], other=0.0)
        c = x1 * x2 * tl.sigmoid(x2)

        w_ptrs = (
            W2_ptr
            + e_idx.to(tl.int64) * stride_w2e
            + n_offs[None, :] * stride_w2n
            + k_offs[:, None] * stride_w2k
        )
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        w_scale = tl.load(
            W2_scale_ptr
            + e_idx.to(tl.int64) * stride_w2scale_e
            + (n_offs[None, :] // 128) * stride_w2scale_n
            + (k // BLOCK_K) * stride_w2scale_k,
            mask=n_mask[None, :],
            other=0.0,
        )

        w_bf16 = (w.to(tl.float32) * w_scale).to(tl.bfloat16)
        c_bf16 = c.to(tl.bfloat16)

        acc += tl.dot(c_bf16, w_bf16, out_dtype=tl.float32)

    routing = tl.load(Routing_weights_ptr + row_ids, mask=m_mask, other=0.0)
    acc = acc * routing[:, None]

    out_ptrs = Out_ptr + row_ids[:, None] * stride_outm + n_offs[None, :] * stride_outn
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _route_and_dispatch(routing_logits, routing_bias, local_expert_offset, routed_scaling_factor, e_local):
    device = routing_logits.device
    T, E_global = routing_logits.shape

    logits = routing_logits.to(torch.float32)
    q = torch.sigmoid(logits)
    rank_scores = q + routing_bias.to(torch.float32).view(1, E_global)

    N_GROUP = 8
    GROUP_SIZE = E_global // N_GROUP
    TOPK_GROUP = 4
    TOP_K = 8

    grouped = rank_scores.view(T, N_GROUP, GROUP_SIZE)
    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True)
    group_scores = top2_vals.sum(dim=2)

    _, top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True)
    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_global)

    neg_inf = torch.full((), float("-inf"), dtype=torch.float32, device=device)
    pruned = torch.where(expert_mask, rank_scores, neg_inf)

    _, topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True)

    selected_q = q.gather(1, topk_idx)
    selected_w = selected_q / (selected_q.sum(dim=1, keepdim=True) + 1e-20)
    selected_w = selected_w * routed_scaling_factor

    local_start = int(local_expert_offset)
    local_mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)
    flat_token_idx, flat_slot_idx = torch.nonzero(local_mask, as_tuple=True)

    if flat_token_idx.numel() == 0:
        return None

    local_expert_idx = (topk_idx[flat_token_idx, flat_slot_idx] - local_start).to(torch.int32)
    flat_weights = selected_w[flat_token_idx, flat_slot_idx].to(torch.float32)

    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order]
    sorted_weights = flat_weights[order]

    expert_counts = torch.bincount(sorted_experts, minlength=e_local)
    expert_offsets = torch.zeros(e_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    return sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets


def _fallback(
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

    routed = _route_and_dispatch(
        routing_logits, routing_bias, local_expert_offset, routed_scaling_factor, E_local
    )
    if routed is None:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_tokens, _, sorted_weights, expert_counts, expert_offsets = routed
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    hs_f32 = hidden_states.to(torch.float32)

    for e in range(E_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        token_ids = sorted_tokens[start:end].long()
        x_fp8 = hs_f32[token_ids]
        a_scale = hidden_states_scale[:, token_ids].transpose(0, 1).repeat_interleave(128, dim=1)
        x = x_fp8 * a_scale[:, :H]

        w1 = gemm1_weights[e].to(torch.float32)
        w1s = gemm1_weights_scale[e].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        w1 = w1 * w1s[: 2 * I, :H]

        g1 = x @ w1.t()
        x1 = g1[:, :I]
        x2 = g1[:, I:]
        c = torch.nn.functional.silu(x2) * x1

        w2 = gemm2_weights[e].to(torch.float32)
        w2s = gemm2_weights_scale[e].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        w2 = w2 * w2s[:H, :I]

        o = c @ w2.t()
        o = o * sorted_weights[start:end].unsqueeze(1)
        output.index_add_(0, token_ids, o)

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
    device = hidden_states.device
    T = routing_logits.shape[0]
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]

    routed = _route_and_dispatch(
        routing_logits, routing_bias, local_expert_offset, routed_scaling_factor, E_local
    )
    if routed is None:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets = routed
    Tk_total = sorted_tokens.numel()

    if (
        expert_offsets.shape[0] != E_local + 1
        or int(expert_offsets[0].item()) != 0
        or int(expert_offsets[-1].item()) != Tk_total
    ):
        return _fallback(
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

    if Tk_total <= 32:
        return _fallback(
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

    max_count = int(expert_counts.max().item())
    if max_count == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    try:
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
    except Exception:
        return _fallback(
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