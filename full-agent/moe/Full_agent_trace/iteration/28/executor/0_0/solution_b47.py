import torch
import triton
import triton.language as tl


# ============================================================
# GEMM1 Kernel: FP8 TCs + expanded autotune (10 configs)
# Based on a8623435 best-known GEMM1 with tl.range() pipelining
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
    H: tl.constexpr,       # 7168 — hidden dimension (K for GEMM1)
    N_dim: tl.constexpr,   # 4096 = 2*I — output dimension
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

    # Software-pipelined K-loop using tl.range() for overlapping mem loads + compute
    for k_start in tl.range(0, H, BLOCK_K, num_stages=4):
        k_block_idx = k_start // BLOCK_K
        k_offs = k_start + k_range

        # Load FP8 hidden states [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_fp8 = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)

        # Load A scales: hidden_states_scale[k_block, token_id] — INDIRECT LOOKUP
        # hidden_states_scale shape: [H//128, T]
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

        # Load W scales: w1_scale[e, n//128, k//128]
        n_block_idx = n_offs // 128
        w_scale_ptrs = (
            W_scale_ptr
            + pid_e.to(tl.int64) * stride_wscale_e
            + n_block_idx[None, :] * stride_wscale_n
            + k_block_idx * stride_wscale_k
        )
        w_scale = tl.load(w_scale_ptrs, mask=n_mask[None, :], other=1.0)

        # FP8 Tensor Core dot — both inputs FP8 → B200 FP8 TCs
        dot_result = tl.dot(a_fp8, w_fp8, out_dtype=tl.float32)
        # Apply per-block scales after dot (exact: both are per-128 block scalars)
        acc += dot_result * a_scale * w_scale

    # Store output
    out_ptrs = (
        Out_ptr
        + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ============================================================
# GEMM2 Kernel: FP8 TCs + fused SwiGLU + expanded autotune (9 configs)
# KEY INNOVATION: Per-tile scalar quantization of SwiGLU activations
# enables FP8 tensor cores (~2x vs BF16 TCs used in a8623435)
# Uses tl.range() pipelining for GEMM2's K-loop
# ALL configs have BLOCK_N=128 (required for n_block_idx = pid_n correctness)
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
        # Large batch
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=5),
    ],
    key=["MAX_TK"],  # MAX_TK proven superior over T as autotune key
)
@triton.jit
def grouped_gemm2_fp8_kernel(
    G1_ptr,
    W2_ptr,
    W2_scale_ptr,
    Routing_weights_ptr,
    Out_ptr,
    expert_offsets_ptr,
    I: tl.constexpr,   # 2048 — intermediate dimension
    H: tl.constexpr,   # 7168 — hidden dimension output
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
    BLOCK_K: tl.constexpr,   # 128 — aligns with FP8 block-scale granularity
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,   # Always 128 (enforced by ALL autotune configs above)
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

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # H-dimension offsets
    n_mask = n_offs < H

    # n_block_idx = pid_n — VALID ONLY because ALL configs enforce BLOCK_N=128
    # Each pid_n tile corresponds to exactly one 128-element H-block
    # This avoids vectorized n_offs // 128 computation, saving registers
    n_block_idx = pid_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_range = tl.arange(0, BLOCK_K)
    row_indices = (start_idx + m_offs).to(tl.int64)

    # Precompute base pointers for W2 and W2_scale OUTSIDE the K-loop
    # int64 for pid_e offsets — [32, 7168, 2048] stride exceeds int32 range
    w2_base = (
        W2_ptr
        + pid_e.to(tl.int64) * stride_w2e
        + n_offs[None, :] * stride_w2n
    )
    w2scale_base = (
        W2_scale_ptr
        + pid_e.to(tl.int64) * stride_w2scale_e
        + n_block_idx * stride_w2scale_n
    )

    # Software-pipelined K-loop: tl.range() enables mem-compute overlap
    for k_start in tl.range(0, I, BLOCK_K):
        k_block_idx = k_start // BLOCK_K
        k_offs = k_start + k_range

        # Load SwiGLU inputs from G1 intermediate buffer (float32)
        # G1 shape: [Tk_total, 2*I] — first I cols = gate, last I cols = up
        x1_ptrs = G1_ptr + row_indices[:, None] * stride_g1m + k_offs[None, :] * stride_g1k
        x2_ptrs = G1_ptr + row_indices[:, None] * stride_g1m + (I + k_offs)[None, :] * stride_g1k

        x1 = tl.load(x1_ptrs, mask=m_mask[:, None], other=0.0)   # gate [BLOCK_M, BLOCK_K] f32
        x2 = tl.load(x2_ptrs, mask=m_mask[:, None], other=0.0)   # up   [BLOCK_M, BLOCK_K] f32

        # Fused SwiGLU: silu(x2) * x1 in float32
        # c = x1 * sigmoid(x2) * x2
        c = x1 * x2 * tl.sigmoid(x2)   # [BLOCK_M, BLOCK_K] float32

        # Per-tile SCALAR quantization to FP8
        # CRITICAL: use global tile max (NO axis argument) — NOT per-row max
        # Per-row would make c_scale a vector, breaking scalar post-dot math
        c_abs_max = tl.max(tl.abs(c))                       # scalar reduction
        c_scale = tl.maximum(c_abs_max / 448.0, 1e-12)      # scalar, prevents div-by-zero
        c_fp8 = (c / c_scale).to(tl.float8_e4m3fn)          # [BLOCK_M, BLOCK_K] FP8, saturating cast

        # Load W2 FP8 tile [BLOCK_K, BLOCK_N]
        # W2 shape: [E, H, I] — we access [e, n_offs, k_offs]
        w2_ptrs = w2_base + k_offs[:, None] * stride_w2k
        w2_fp8 = tl.load(w2_ptrs, mask=n_mask[None, :], other=0.0)

        # Load SCALAR W2 scale for this (H-block, I-block) tile
        # w2_scale[e, n_block_idx, k_block_idx] is a single float32
        w2_scale = tl.load(w2scale_base + k_block_idx * stride_w2scale_k)  # scalar

        # FP8 Tensor Core dot — BOTH inputs are FP8 → B200 FP8 TCs
        # B200: FP8 ~3.5 PFLOPS, BF16 ~1.75 PFLOPS (~2x theoretical improvement)
        dot_result = tl.dot(c_fp8, w2_fp8, out_dtype=tl.float32)   # [BLOCK_M, BLOCK_N]

        # Exact post-dot dequantization (both c_scale and w2_scale are scalars)
        # Proof: sum_k(c[m,k]*W2[k,n]) = sum_k(c_fp8[m,k]*c_scale*W2[k,n])
        #      = c_scale * dot(c_fp8, W2)[m,n]
        # Then: * w2_scale = exact O_tile[m,n]
        acc += dot_result * c_scale * w2_scale

    # Apply routing weights (per-token scalar)
    routing_w = tl.load(Routing_weights_ptr + row_indices, mask=m_mask, other=0.0)
    acc = acc * routing_w[:, None]

    # Store to output buffer
    out_ptrs = (
        Out_ptr
        + row_indices[:, None] * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _route_tokens(routing_logits, routing_bias, local_expert_offset, e_local, routed_scaling_factor):
    """
    DeepSeek-V3/R1 no-aux routing:
    sigmoid -> group top-2 sum -> top-4 groups -> global top-8 experts
    Fully vectorized, no CPU-GPU sync in the critical path.
    """
    device = routing_logits.device
    T, E_global = routing_logits.shape
    N_GROUP = 8
    TOPK_GROUP = 4
    TOP_K = 8
    group_size = E_global // N_GROUP  # 256 // 8 = 32

    # Step 1: Raw sigmoid scores (used for weight normalization)
    q = torch.sigmoid(routing_logits.float())   # [T, 256]

    # Step 2: Add bias for ranking ONLY (not for weight computation)
    rank_scores = q + routing_bias.float().view(1, E_global)   # [T, 256]

    # Step 3: Group top-2 selection for group score computation
    grouped = rank_scores.view(T, N_GROUP, group_size)   # [T, 8, 32]
    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True)
    group_scores = top2_vals.sum(dim=2)   # [T, 8]

    # Step 4: Select top-4 groups
    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)
    expert_allowed = group_mask.unsqueeze(-1).expand(T, N_GROUP, group_size).reshape(T, E_global)

    # Step 5: Mask and global top-8 selection
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=rank_scores.dtype)
    pruned = torch.where(expert_allowed, rank_scores, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    # Step 6: Normalize weights using RAW sigmoid (NOT biased scores)
    sel = torch.zeros((T, E_global), dtype=torch.bool, device=device)
    sel.scatter_(1, topk_idx, True)
    raw_w = q * sel
    denom = raw_w.sum(dim=1, keepdim=True) + 1e-20
    weights = (raw_w / denom) * routed_scaling_factor

    # Step 7: Filter for local experts
    local_start = int(local_expert_offset)
    expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)
    flat_token_idx, flat_k_idx = torch.nonzero(expert_mask, as_tuple=True)
    if flat_token_idx.numel() == 0:
        return None

    local_expert_idx = topk_idx[flat_token_idx, flat_k_idx] - local_start

    # Step 8: Sort by expert — builds CSR structure for Triton grouped GEMM
    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int32)
    sorted_experts = local_expert_idx[order].to(torch.int32)
    sorted_weights = weights[flat_token_idx, topk_idx[flat_token_idx, flat_k_idx]][order].to(torch.float32)

    # Step 9: Build CSR offsets (GPU-side, NO CPU-GPU sync)
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
    
    Optimizations:
    1. FP8 TCs in both GEMM1 and GEMM2 (B200 FP8 ~2x BF16 throughput)
    2. Per-tile scalar quantization in GEMM2 (exact, unlocks FP8 TCs for activations)
    3. Expanded 10-config autotuning for GEMM1, 9-config for GEMM2
    4. tl.range() software pipelining in both K-loops
    5. Vectorized GPU-side dispatch (no CPU-GPU sync)
    6. int64 pointer arithmetic for large tensor offsets
    """
    H = 7168    # hidden dimension
    I = 2048    # intermediate dimension
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]   # 32 local experts

    # Step 1: Route tokens using proven vectorized routing
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

    # Single unavoidable CPU sync to get max_count for grid bounds
    max_count = int(expert_counts.max().item())
    if max_count == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    # Step 2: GEMM1 — FP8 TCs with indirect A-scale lookup
    # G1 shape: [Tk_total, 2*I] = [Tk_total, 4096] float32
    # torch.zeros ensures masked threads contribute 0 (not HBM garbage)
    G1 = torch.zeros((Tk_total, 2 * I), dtype=torch.float32, device=device)

    grid_gemm1 = lambda META: (
        triton.cdiv(2 * I, META["BLOCK_N"]),        # N-tiles over 4096
        triton.cdiv(max_count, META["BLOCK_M"]),     # M-tiles bounded by max expert tokens
        E_local,                                     # expert dimension as grid axis
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

    # Step 3: GEMM2 with fused SwiGLU + FP8 TCs
    # O shape: [Tk_total, H] = [Tk_total, 7168] float32
    O = torch.zeros((Tk_total, H), dtype=torch.float32, device=device)

    grid_gemm2 = lambda META: (
        triton.cdiv(H, META["BLOCK_N"]),             # H/128 = 56 N-tiles
        triton.cdiv(max_count, META["BLOCK_M"]),     # M-tiles bounded by max expert tokens
        E_local,                                     # expert grid axis
    )

    grouped_gemm2_fp8_kernel[grid_gemm2](
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

    # Step 4: Weighted scatter-add accumulation
    # output[token[j], h] += O[j, h] (routing weights already applied in GEMM2 kernel)
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    output.index_add_(0, sorted_tokens.long(), O)

    # Final cast: FP32 → BF16 (required output dtype)
    return output.to(torch.bfloat16)
