import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=5),
    ],
    key=['MAX_TK'],
)
@triton.jit
def grouped_gemm1_kernel(
    A_ptr, A_scale_ptr,
    W_ptr, W_scale_ptr,
    Out_ptr,
    sorted_tokens_ptr, expert_offsets_ptr,
    H: tl.constexpr,
    N_dim: tl.constexpr,
    MAX_TK,
    stride_am, stride_ak,
    stride_ascale_k, stride_ascale_m,
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

    start_idx = tl.load(expert_offsets_ptr + pid_e)
    end_idx = tl.load(expert_offsets_ptr + pid_e + 1)
    num_tokens = end_idx - start_idx

    if pid_m * BLOCK_M >= num_tokens:
        return

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < num_tokens
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N_dim
    k_range = tl.arange(0, BLOCK_K)

    token_ids = tl.load(sorted_tokens_ptr + start_idx + m_offs, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, H, BLOCK_K, num_stages=4):
        k_block_idx = k_start // BLOCK_K
        k_offs = k_start + k_range

        # Load A (FP8 hidden states) - indirect token gather
        a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_fp8 = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)

        # Load A scales: hidden_states_scale[k_block_idx, token_id]
        a_scale_ptrs = A_scale_ptr + k_block_idx * stride_ascale_k + token_ids[:, None] * stride_ascale_m
        a_scale = tl.load(a_scale_ptrs, mask=m_mask[:, None], other=1.0)

        # Load W1 (FP8 weights)
        w_ptrs = W_ptr + pid_e.to(tl.int64) * stride_we + n_offs[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_fp8 = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        # Load W1 scales: w1_scale[e, n//128, k//128]
        n_block_idx = n_offs // 128
        w_scale_ptrs = W_scale_ptr + pid_e.to(tl.int64) * stride_wscale_e + n_block_idx[None, :] * stride_wscale_n + k_block_idx * stride_wscale_k
        w_scale = tl.load(w_scale_ptrs, mask=n_mask[None, :], other=1.0)

        # FP8 Tensor Core dot - both inputs FP8
        dot_result = tl.dot(a_fp8, w_fp8, out_dtype=tl.float32)
        acc += dot_result * a_scale * w_scale

    out_ptrs = Out_ptr + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm + n_offs[None, :] * stride_outn
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=5),
    ],
    key=['MAX_TK'],
)
@triton.jit
def grouped_gemm2_fp8_kernel(
    G1_ptr,
    W2_ptr, W2_scale_ptr,
    Routing_weights_ptr,
    Out_ptr,
    expert_offsets_ptr,
    I: tl.constexpr,
    H: tl.constexpr,
    MAX_TK,
    stride_g1m, stride_g1k,
    stride_w2e, stride_w2n, stride_w2k,
    stride_w2scale_e, stride_w2scale_n, stride_w2scale_k,
    stride_outm, stride_outn,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # Always 128
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
    k_range = tl.arange(0, BLOCK_K)

    # n_block_idx = pid_n — VALID ONLY when BLOCK_N=128
    n_block_idx = pid_n

    row_indices = (start_idx + m_offs).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute base pointers
    w2_base = W2_ptr + pid_e.to(tl.int64) * stride_w2e + n_offs[None, :] * stride_w2n
    w2scale_base = W2_scale_ptr + pid_e.to(tl.int64) * stride_w2scale_e + n_block_idx * stride_w2scale_n

    for k_start in tl.range(0, I, BLOCK_K, num_stages=4):
        k_block_idx = k_start // BLOCK_K
        k_offs = k_start + k_range

        # Load x1 (gate) and x2 (up) from G1 intermediate buffer
        x1_ptrs = G1_ptr + row_indices[:, None] * stride_g1m + k_offs[None, :] * stride_g1k
        x2_ptrs = G1_ptr + row_indices[:, None] * stride_g1m + (I + k_offs)[None, :] * stride_g1k
        x1 = tl.load(x1_ptrs, mask=m_mask[:, None], other=0.0)
        x2 = tl.load(x2_ptrs, mask=m_mask[:, None], other=0.0)

        # SwiGLU fused: silu(x2) * x1 in float32
        c = x1 * x2 * tl.sigmoid(x2)  # [BLOCK_M, BLOCK_K] float32

        # Per-tile SCALAR quantization — enables exact dequantization
        c_abs_max = tl.max(tl.abs(c))  # scalar reduction over entire tile
        c_scale = tl.maximum(c_abs_max / 448.0, 1e-12)  # scalar, prevents div-by-zero
        c_fp8 = (c / c_scale).to(tl.float8_e4m3fn)  # [BLOCK_M, BLOCK_K] FP8

        # Load W2 tile (FP8)
        w2_ptrs = w2_base + k_offs[:, None] * stride_w2k  # [BLOCK_K, BLOCK_N]
        w2_fp8 = tl.load(w2_ptrs, mask=n_mask[None, :], other=0.0)

        # Load scalar W2 scale for this (H-block, I-block) pair
        w2_scale = tl.load(w2scale_base + k_block_idx * stride_w2scale_k)  # scalar

        # FP8 Tensor Core dot — BOTH inputs FP8
        dot_result = tl.dot(c_fp8, w2_fp8, out_dtype=tl.float32)  # [BLOCK_M, BLOCK_N]

        # Exact dequantization: both scales are scalars for this tile
        acc += dot_result * c_scale * w2_scale

    # Apply routing weights (scalar per token)
    routing_w = tl.load(Routing_weights_ptr + row_indices, mask=m_mask, other=0.0)
    acc = acc * routing_w[:, None]

    # Store to output buffer
    out_ptrs = Out_ptr + row_indices[:, None] * stride_outm + n_offs[None, :] * stride_outn
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

    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    device = hidden_states.device

    # 1) DeepSeek-V3 No-Aux Routing (Vectorized)
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = torch.sigmoid(logits)  # [T, 256]
    s_with_bias = s + bias     # [T, 256]

    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    group_size = E_global // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)

    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=True)
    group_scores = top2_vals.sum(dim=2)  # [T, 8]

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=True)

    # Routing weights: normalize raw sigmoid scores
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # 2) Token Sorting and CSR Grouping (GPU-only vectorized)
    local_start = int(local_expert_offset)
    expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + E_local)

    flat_token_idx, flat_k_idx = torch.nonzero(expert_mask, as_tuple=True)
    Tk_total = flat_token_idx.numel()

    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    local_expert_idx = topk_idx[flat_token_idx, flat_k_idx] - local_start

    sorted_indices = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[sorted_indices].to(torch.int32)
    sorted_experts = local_expert_idx[sorted_indices].to(torch.int32)

    global_expert_idx = sorted_experts + local_start
    sorted_weights = weights[sorted_tokens.long(), global_expert_idx.long()].to(torch.float32)

    expert_counts = torch.bincount(sorted_experts, minlength=E_local)
    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    max_count = int(expert_counts.max().item())
    if max_count == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    # 3) Grouped GEMM1 (FP8 Tensor Cores + Block Scale Integration)
    G1 = torch.zeros((Tk_total, 2 * I), dtype=torch.float32, device=device)

    grid_gemm1 = lambda META: (
        triton.cdiv(2 * I, META['BLOCK_N']),
        triton.cdiv(max_count, META['BLOCK_M']),
        E_local
    )

    grouped_gemm1_kernel[grid_gemm1](
        A_ptr=hidden_states, A_scale_ptr=hidden_states_scale,
        W_ptr=gemm1_weights, W_scale_ptr=gemm1_weights_scale,
        Out_ptr=G1,
        sorted_tokens_ptr=sorted_tokens, expert_offsets_ptr=expert_offsets,
        H=H, N_dim=2 * I, MAX_TK=max_count,
        stride_am=hidden_states.stride(0), stride_ak=hidden_states.stride(1),
        stride_ascale_k=hidden_states_scale.stride(0), stride_ascale_m=hidden_states_scale.stride(1),
        stride_we=gemm1_weights.stride(0), stride_wn=gemm1_weights.stride(1), stride_wk=gemm1_weights.stride(2),
        stride_wscale_e=gemm1_weights_scale.stride(0), stride_wscale_n=gemm1_weights_scale.stride(1), stride_wscale_k=gemm1_weights_scale.stride(2),
        stride_outm=G1.stride(0), stride_outn=G1.stride(1),
        BLOCK_K=128,
    )

    # 4) Grouped GEMM2 (Fused SwiGLU + FP8 Tensor Cores)
    O = torch.zeros((Tk_total, H), dtype=torch.float32, device=device)

    grid_gemm2 = lambda META: (
        triton.cdiv(H, META['BLOCK_N']),
        triton.cdiv(max_count, META['BLOCK_M']),
        E_local
    )

    grouped_gemm2_fp8_kernel[grid_gemm2](
        G1_ptr=G1,
        W2_ptr=gemm2_weights, W2_scale_ptr=gemm2_weights_scale,
        Routing_weights_ptr=sorted_weights,
        Out_ptr=O,
        expert_offsets_ptr=expert_offsets,
        I=I, H=H, MAX_TK=max_count,
        stride_g1m=G1.stride(0), stride_g1k=G1.stride(1),
        stride_w2e=gemm2_weights.stride(0), stride_w2n=gemm2_weights.stride(1), stride_w2k=gemm2_weights.stride(2),
        stride_w2scale_e=gemm2_weights_scale.stride(0), stride_w2scale_n=gemm2_weights_scale.stride(1), stride_w2scale_k=gemm2_weights_scale.stride(2),
        stride_outm=O.stride(0), stride_outn=O.stride(1),
        BLOCK_K=128,
    )

    # 5) Weighted Scatter-Add Accumulation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    output.index_add_(0, sorted_tokens.long(), O)

    return output.to(torch.bfloat16)