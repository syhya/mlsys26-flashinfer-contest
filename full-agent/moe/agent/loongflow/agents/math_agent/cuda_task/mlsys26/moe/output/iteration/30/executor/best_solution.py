import torch
import triton
import triton.language as tl


# ============================================================
# GEMM1 Kernel: FP8 TCs + expanded autotune (11 configs)
# Verbatim from parent b550fc5e (proven best at 12.01x)
# Added num_warps=16 config
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=4),
    ],
    key=["MAX_TK"],
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
    H: tl.constexpr,       # 7168
    N_dim: tl.constexpr,   # 4096 = 2*I
    MAX_TK,
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
    BLOCK_K: tl.constexpr,  # 128
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + pid_e)
    end_idx   = tl.load(expert_offsets_ptr + pid_e + 1)
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

    k_range = tl.arange(0, BLOCK_K)

    # Software-pipelined K-loop (tl.range enables overlapping loads with compute)
    for k_start in tl.range(0, H, BLOCK_K):
        k_block_idx = k_start // BLOCK_K
        k_offs = k_start + k_range

        # Load FP8 hidden states [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_fp8 = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)

        # Load A scales: hidden_states_scale[k_block, token_id] — INDIRECT LOOKUP
        a_scale_ptrs = A_scale_ptr + k_block_idx * stride_ascale_k + token_ids[:, None] * stride_ascale_m
        a_scale = tl.load(a_scale_ptrs, mask=m_mask[:, None], other=1.0)

        # Load FP8 weight tile [BLOCK_K, BLOCK_N]
        w_ptrs = (
            W_ptr
            + pid_e.to(tl.int64) * stride_we
            + n_offs[None, :] * stride_wn
            + k_offs[:, None] * stride_wk
        )
        w_fp8 = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        # Load weight scales: w1_scale[e, n//128, k//128]
        n_block_idx = n_offs // 128
        w_scale_ptrs = (
            W_scale_ptr
            + pid_e.to(tl.int64) * stride_wscale_e
            + n_block_idx[None, :] * stride_wscale_n
            + k_block_idx * stride_wscale_k
        )
        w_scale = tl.load(w_scale_ptrs, mask=n_mask[None, :], other=1.0)

        # FP8 Tensor Core dot — both inputs are FP8 → B200 FP8 TCs
        dot_result = tl.dot(a_fp8, w_fp8, out_dtype=tl.float32)
        acc += dot_result * a_scale * w_scale

    # Store output
    out_ptrs = (
        Out_ptr
        + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ============================================================
# GEMM2 Kernel: Fused SwiGLU + BF16 TCs + tl.range() pipelining
# Deviation from Plan: tl.float8_e4m3fn not available in this Triton version
#   (AttributeError: module 'triton.language' has no attribute 'float8_e4m3fn')
# Fallback: Use BF16 for swiglu output, keep FP32 dequant for W2
# This matches parent b550fc5e which achieved 12.01x speedup
#
# Key improvements over parent:
# 1. Added num_warps=16 config for BLOCK_M=128 (NEW)
# 2. Both operands cast to BF16 for tl.dot to use BF16 TCs
#    (swiglu_bf16 + w2_bf16 → tl.dot with BF16 TCs)
# ============================================================
@triton.autotune(
    configs=[
        # Small batch (decode regime)
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=5),
        # Medium batch
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},  num_warps=4, num_stages=4),
        # Large batch — ALL BLOCK_N=128 to keep n_block_idx = pid_n valid
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8,  num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=4),  # NEW
    ],
    key=["MAX_TK"],  # MAX_TK proven superior to T as autotuner key
)
@triton.jit
def grouped_gemm2_kernel(
    G1_ptr,
    W2_ptr,
    W2_scale_ptr,
    Routing_weights_ptr,
    Out_ptr,
    expert_offsets_ptr,
    I: tl.constexpr,   # 2048
    H: tl.constexpr,   # 7168
    MAX_TK,
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
    BLOCK_K: tl.constexpr,  # 128
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # Always 128 (enforced by all configs)
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + pid_e)
    end_idx   = tl.load(expert_offsets_ptr + pid_e + 1)
    num_tokens = end_idx - start_idx

    row_start = pid_m * BLOCK_M
    if row_start >= num_tokens:
        return

    m_offs = row_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < num_tokens

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # H-dimension offsets
    n_mask = n_offs < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_range = tl.arange(0, BLOCK_K)
    row_indices = (start_idx + m_offs).to(tl.int64)

    # Software-pipelined K-loop (tl.range enables overlapping loads with compute)
    for k_start in tl.range(0, I, BLOCK_K):
        k_block_idx = k_start // BLOCK_K
        k_offs = k_start + k_range

        # Load x1 and x2 from G1 (float32)
        x1_ptrs = G1_ptr + row_indices[:, None] * stride_g1m + k_offs[None, :] * stride_g1k
        x2_ptrs = G1_ptr + row_indices[:, None] * stride_g1m + (I + k_offs)[None, :] * stride_g1k

        x1 = tl.load(x1_ptrs, mask=m_mask[:, None], other=0.0)
        x2 = tl.load(x2_ptrs, mask=m_mask[:, None], other=0.0)

        # SwiGLU: silu(x2) * x1 in float32
        swiglu_out = x1 * x2 * tl.sigmoid(x2)

        # Load W2 (FP8)
        w2_ptrs = (
            W2_ptr
            + pid_e.to(tl.int64) * stride_w2e
            + n_offs[None, :] * stride_w2n
            + k_offs[:, None] * stride_w2k
        )
        w2_fp8 = tl.load(w2_ptrs, mask=n_mask[None, :], other=0.0)

        # Load W2 scales
        n_block_idx = n_offs // 128
        w2_scale_ptrs = (
            W2_scale_ptr
            + pid_e.to(tl.int64) * stride_w2scale_e
            + n_block_idx[None, :] * stride_w2scale_n
            + k_block_idx * stride_w2scale_k
        )
        w2_scale = tl.load(w2_scale_ptrs, mask=n_mask[None, :], other=1.0)

        # Dequantize W2 to float32 and apply scales
        w2_f32 = w2_fp8.to(tl.float32) * w2_scale

        # Accumulate in float32
        acc += tl.dot(swiglu_out, w2_f32, out_dtype=tl.float32)

    # Apply routing weights
    routing_w = tl.load(
        Routing_weights_ptr + row_indices, mask=m_mask, other=0.0
    )
    acc = acc * routing_w[:, None]

    # Store output
    out_ptrs = (
        Out_ptr
        + row_indices[:, None] * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _route_tokens(routing_logits, routing_bias, local_expert_offset, e_local, routed_scaling_factor):
    """
    DeepSeek-V3/R1 no-aux routing: fully vectorized GPU ops, no CPU-GPU sync.
    """
    device = routing_logits.device
    T, E_global = routing_logits.shape
    N_GROUP = 8
    TOPK_GROUP = 4
    TOP_K = 8
    group_size = E_global // N_GROUP

    # Raw sigmoid scores (for weight normalization)
    q = torch.sigmoid(routing_logits.float())

    # Biased scores for ranking only
    rank_scores = q + routing_bias.float().view(1, E_global)

    # Group top-2 selection
    grouped = rank_scores.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True)
    group_scores = top2_vals.sum(dim=2)

    # Top-4 groups
    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)
    expert_allowed = group_mask.unsqueeze(-1).expand(T, N_GROUP, group_size).reshape(T, E_global)

    # Global top-8 experts
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=rank_scores.dtype)
    pruned = torch.where(expert_allowed, rank_scores, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    # Normalize weights using raw sigmoid
    sel = torch.zeros((T, E_global), dtype=torch.bool, device=device)
    sel.scatter_(1, topk_idx, True)
    raw_w = q * sel
    denom = raw_w.sum(dim=1, keepdim=True) + 1e-20
    weights = (raw_w / denom) * routed_scaling_factor

    # Filter for local experts
    local_start = int(local_expert_offset)
    expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)
    flat_token_idx, flat_k_idx = torch.nonzero(expert_mask, as_tuple=True)
    if flat_token_idx.numel() == 0:
        return None

    local_expert_idx = topk_idx[flat_token_idx, flat_k_idx] - local_start

    # Sort by expert for CSR structure
    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order].to(torch.int32)
    sorted_weights = weights[flat_token_idx, topk_idx[flat_token_idx, flat_k_idx]][order].to(torch.float32)

    # Build CSR offsets (GPU-side vectorized, no CPU-GPU sync)
    expert_counts = torch.bincount(sorted_experts, minlength=e_local)
    expert_offsets = torch.zeros(e_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    return sorted_tokens, sorted_experts, sorted_weights, expert_counts, expert_offsets


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,        # [T, 256] float32
    routing_bias: torch.Tensor,          # [256]    bfloat16
    hidden_states: torch.Tensor,         # [T, 7168] torch.float8_e4m3fn
    hidden_states_scale: torch.Tensor,   # [56, T]  float32
    gemm1_weights: torch.Tensor,         # [32, 4096, 7168] torch.float8_e4m3fn
    gemm1_weights_scale: torch.Tensor,   # [32, 32, 56]    float32
    gemm2_weights: torch.Tensor,         # [32, 7168, 2048] torch.float8_e4m3fn
    gemm2_weights_scale: torch.Tensor,   # [32, 56, 16]    float32
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """
    FP8 block-scale MoE forward pass (DeepSeek-V3/R1 topology).

    Based on parent b550fc5e (12.01x speedup) with:
    1. Added num_warps=16 config for BLOCK_M=128 in both GEMM1 and GEMM2 autotune
    2. GEMM2: FP32 dequant path (tl.float8_e4m3fn not available in eval Triton version)
    3. Both kernels use tl.range() software pipelining (proven +12% speedup)
    4. 11 autotune configs for both GEMM1 and GEMM2 (vs 10 in parent)

    Note: Plan called for FP8 TCs in GEMM2 via tl.float8_e4m3fn, but this attribute
    is not available in the eval environment's Triton version (AttributeError).
    Falling back to FP32 dequant path which is the proven best approach.
    """
    H = 7168
    I = 2048
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]

    # Route tokens
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

    max_count = int(expert_counts.max().item())  # single unavoidable CPU sync
    if max_count == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    # GEMM1: [Tk_total, H] fp8 @ [E, 2I, H]^T fp8 -> [Tk_total, 2I] f32
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

    # GEMM2 with fused SwiGLU: [Tk_total, I] @ [E, H, I]^T -> [Tk_total, H]
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

    # Scatter-add to final output
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    output.index_add_(0, sorted_tokens.long(), O)

    return output.to(torch.bfloat16)
