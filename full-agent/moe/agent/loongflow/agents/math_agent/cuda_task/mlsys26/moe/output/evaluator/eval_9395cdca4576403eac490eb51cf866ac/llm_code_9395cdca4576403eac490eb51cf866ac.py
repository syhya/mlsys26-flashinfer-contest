import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["Tk"],
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
    expert_id,
    H: tl.constexpr,
    N_dim: tl.constexpr,
    Tk,
    stride_am,
    stride_ak,
    stride_ascale_m,
    stride_ascale_k,
    stride_wn,
    stride_wk,
    stride_wscale_n,
    stride_wscale_k,
    stride_outm,
    stride_outn,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_idx = tl.load(expert_offsets_ptr + expert_id)
    end_idx = tl.load(expert_offsets_ptr + expert_id + 1)
    num_tokens = end_idx - start_idx

    row_start = pid_m * BLOCK_M
    if row_start >= num_tokens:
        return

    m_offs = row_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < num_tokens

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N_dim

    token_ids = tl.load(
        sorted_tokens_ptr + start_idx + m_offs, mask=m_mask, other=0
    ).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    expert_offset = expert_id.to(tl.int64) * N_dim * H

    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_block_idx = k_start // BLOCK_K

        a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_fp8 = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)

        a_scale_ptrs = (
            A_scale_ptr + k_block_idx * stride_ascale_k + token_ids[:, None] * stride_ascale_m
        )
        a_scale = tl.load(a_scale_ptrs, mask=m_mask[:, None], other=1.0)

        a_f32 = a_fp8.to(tl.float32) * a_scale

        w_ptrs = (
            W_ptr
            + expert_offset
            + n_offs[None, :] * stride_wn
            + k_offs[:, None] * stride_wk
        )
        w_fp8 = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        n_block_idx = n_offs // 128
        w_scale_ptrs = (
            W_scale_ptr
            + expert_id.to(tl.int64) * (N_dim // 128) * (H // 128)
            + n_block_idx[None, :] * stride_wscale_n
            + k_block_idx * stride_wscale_k
        )
        w_scale = tl.load(w_scale_ptrs, mask=n_mask[None, :], other=1.0)

        w_f32 = w_fp8.to(tl.float32) * w_scale

        acc += tl.dot(a_f32, w_f32, out_dtype=tl.float32)

    out_ptrs = (
        Out_ptr
        + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["Tk"],
)
@triton.jit
def grouped_gemm2_swiglu_kernel(
    G1_ptr,
    W2_ptr,
    W2_scale_ptr,
    Routing_weights_ptr,
    Out_ptr,
    expert_offsets_ptr,
    expert_id,
    I: tl.constexpr,
    H: tl.constexpr,
    Tk,
    stride_g1m,
    stride_g1k,
    stride_w2n,
    stride_w2k,
    stride_w2scale_n,
    stride_w2scale_k,
    stride_outm,
    stride_outn,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_idx = tl.load(expert_offsets_ptr + expert_id)
    end_idx = tl.load(expert_offsets_ptr + expert_id + 1)
    num_tokens = end_idx - start_idx

    row_start = pid_m * BLOCK_M
    if row_start >= num_tokens:
        return

    m_offs = row_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < num_tokens

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    expert_offset = expert_id.to(tl.int64) * H * I

    for k_start in range(0, I, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_block_idx = k_start // BLOCK_K

        row_base = (start_idx + m_offs).to(tl.int64) * stride_g1m

        x1_ptrs = G1_ptr + row_base[:, None] + k_offs[None, :] * stride_g1k
        x1 = tl.load(x1_ptrs, mask=m_mask[:, None], other=0.0)

        x2_ptrs = G1_ptr + row_base[:, None] + (I + k_offs)[None, :] * stride_g1k
        x2 = tl.load(x2_ptrs, mask=m_mask[:, None], other=0.0)

        swiglu = x1 * x2 * tl.sigmoid(x2)

        w2_ptrs = (
            W2_ptr
            + expert_offset
            + n_offs[None, :] * stride_w2n
            + k_offs[:, None] * stride_w2k
        )
        w2_fp8 = tl.load(w2_ptrs, mask=n_mask[None, :], other=0.0)

        n_block_idx = n_offs // 128
        w2_scale_ptrs = (
            W2_scale_ptr
            + expert_id.to(tl.int64) * (H // 128) * (I // 128)
            + n_block_idx[None, :] * stride_w2scale_n
            + k_block_idx * stride_w2scale_k
        )
        w2_scale = tl.load(w2_scale_ptrs, mask=n_mask[None, :], other=1.0)

        w2_f32 = w2_fp8.to(tl.float32) * w2_scale

        acc += tl.dot(swiglu, w2_f32, out_dtype=tl.float32)

    routing_w = tl.load(
        Routing_weights_ptr + (start_idx + m_offs).to(tl.int64),
        mask=m_mask,
        other=0.0,
    )
    acc = acc * routing_w[:, None]

    out_ptrs = (
        Out_ptr
        + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


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
    T, H = hidden_states.shape
    E_local = gemm1_weights.shape[0]
    I = 2048
    E_global = 256
    N_GROUP = 8
    TOPK_GROUP = 4
    TOP_K = 8

    q = torch.sigmoid(routing_logits.float())
    rank_scores = q + routing_bias.float().view(1, E_global)

    group_size = E_global // N_GROUP
    grouped = rank_scores.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True)
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(
        group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True
    ).indices
    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)
    expert_allowed = (
        group_mask.unsqueeze(-1).expand(T, N_GROUP, group_size).reshape(T, E_global)
    )

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=rank_scores.dtype)
    pruned = torch.where(expert_allowed, rank_scores, neg_inf)

    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices
    sel = torch.zeros((T, E_global), dtype=torch.bool, device=device)
    sel.scatter_(1, topk_idx, True)

    raw_w = q * sel
    denom = raw_w.sum(dim=1, keepdim=True) + 1e-20
    weights = (raw_w / denom) * routed_scaling_factor

    local_start = int(local_expert_offset)
    expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + E_local)

    flat_token_idx, flat_k_idx = torch.nonzero(expert_mask, as_tuple=True)
    if flat_token_idx.numel() == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    local_expert_idx = topk_idx[flat_token_idx, flat_k_idx] - local_start
    order = torch.argsort(local_expert_idx)

    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order].to(torch.int32)
    sorted_weights = weights[flat_token_idx, topk_idx[flat_token_idx, flat_k_idx]][
        order
    ].to(torch.float32)

    expert_counts = torch.bincount(sorted_experts, minlength=E_local)
    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    Tk_total = sorted_tokens.numel()
    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    G1 = torch.zeros((Tk_total, 2 * I), dtype=torch.float32, device=device)
    O = torch.zeros((Tk_total, H), dtype=torch.float32, device=device)

    for e in range(E_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        Tk = end - start
        if Tk == 0:
            continue

        grid_gemm1 = (
            triton.cdiv(2 * I, 128),
            triton.cdiv(Tk, 128),
        )

        grouped_gemm1_kernel[grid_gemm1](
            A_ptr=hidden_states,
            A_scale_ptr=hidden_states_scale,
            W_ptr=gemm1_weights,
            W_scale_ptr=gemm1_weights_scale,
            Out_ptr=G1,
            sorted_tokens_ptr=sorted_tokens,
            expert_offsets_ptr=expert_offsets,
            expert_id=e,
            H=H,
            N_dim=2 * I,
            Tk=Tk,
            stride_am=hidden_states.stride(0),
            stride_ak=hidden_states.stride(1),
            stride_ascale_m=hidden_states_scale.stride(1),
            stride_ascale_k=hidden_states_scale.stride(0),
            stride_wn=gemm1_weights.stride(1),
            stride_wk=gemm1_weights.stride(2),
            stride_wscale_n=gemm1_weights_scale.stride(1),
            stride_wscale_k=gemm1_weights_scale.stride(2),
            stride_outm=G1.stride(0),
            stride_outn=G1.stride(1),
            BLOCK_K=128,
        )

        grid_gemm2 = (
            triton.cdiv(Tk, 128),
            triton.cdiv(H, 128),
        )

        grouped_gemm2_swiglu_kernel[grid_gemm2](
            G1_ptr=G1,
            W2_ptr=gemm2_weights,
            W2_scale_ptr=gemm2_weights_scale,
            Routing_weights_ptr=sorted_weights,
            Out_ptr=O,
            expert_offsets_ptr=expert_offsets,
            expert_id=e,
            I=I,
            H=H,
            Tk=Tk,
            stride_g1m=G1.stride(0),
            stride_g1k=G1.stride(1),
            stride_w2n=gemm2_weights.stride(1),
            stride_w2k=gemm2_weights.stride(2),
            stride_w2scale_n=gemm2_weights_scale.stride(1),
            stride_w2scale_k=gemm2_weights_scale.stride(2),
            stride_outm=O.stride(0),
            stride_outn=O.stride(1),
            BLOCK_K=128,
        )

    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    output.index_add_(0, sorted_tokens.long(), O)
    return output.to(torch.bfloat16)