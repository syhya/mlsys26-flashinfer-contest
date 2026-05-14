"""Retained Triton fastpath for the merged MoE submission.

This module keeps the best pieces of the former v24 Triton pipeline:
- fused routing/sort for small shapes
- async vs sync workspace specialization
- sync-path GEMM2 worklist compaction for long shapes

It is no longer responsible for global dispatch. The top-level entrypoint in
`kernel.py` decides when this path should run.
"""

import torch
import triton
import triton.language as tl

ASYNC_PATH_MAX_TOKENS = 256
SWIGLU_FP16_TARGET_MAX = 65000.0

_ws = {}
_ws_sync = {}

@triton.jit
def fused_moe_routing_kernel(
    routing_logits_ptr, routing_bias_ptr,
    topk_idx_ptr, topk_weights_ptr, int_buf_ptr, expert_start_ptr,
    T, routed_scaling_factor, local_start,
    E_GLOBAL: tl.constexpr, N_GROUP: tl.constexpr, GROUP_SIZE: tl.constexpr,
    TOP_K: tl.constexpr, TOPK_GROUP: tl.constexpr, E_LOCAL: tl.constexpr,
    GEMM2_BLOCK_M: tl.constexpr,
):
    pid_t = tl.program_id(0)
    if pid_t >= T:
        return
    token_counts_ptr = int_buf_ptr
    meta_ptr = int_buf_ptr + E_LOCAL
    completion_counter_ptr = int_buf_ptr + E_LOCAL + 3
    gemm2_tile_prefix_ptr = int_buf_ptr + 2 * E_LOCAL + 4
    NEG_INF: tl.constexpr = -3.4028235e+38
    offs = tl.arange(0, E_GLOBAL)
    logits = tl.load(routing_logits_ptr + pid_t * E_GLOBAL + offs).to(tl.float32)
    bias = tl.load(routing_bias_ptr + offs).to(tl.float32)
    s = tl.sigmoid(logits)
    s_with_bias = s + bias
    group_ids = offs // GROUP_SIZE
    group_offs = tl.arange(0, N_GROUP)
    group_scores = tl.zeros([N_GROUP], dtype=tl.float32)
    for g in tl.static_range(N_GROUP):
        g_vals = tl.where(group_ids == g, s_with_bias, NEG_INF)
        g_max1 = tl.max(g_vals, axis=0)
        g_argmax1 = tl.argmax(g_vals, axis=0)
        g_vals2 = tl.where(offs == g_argmax1, NEG_INF, g_vals)
        g_max2 = tl.max(g_vals2, axis=0)
        g_score = g_max1 + g_max2
        group_scores = tl.where(group_offs == g, g_score, group_scores)
    selected_group_mask = tl.zeros([E_GLOBAL], dtype=tl.float32)
    for _k in tl.static_range(TOPK_GROUP):
        best_g = tl.argmax(group_scores, axis=0)
        selected_group_mask = tl.where(group_ids == best_g, 1.0, selected_group_mask)
        group_scores = tl.where(group_offs == best_g, NEG_INF, group_scores)
    scores_pruned = tl.where(selected_group_mask > 0.5, s_with_bias, NEG_INF)
    topk_offs = tl.arange(0, TOP_K)
    topk_indices = tl.zeros([TOP_K], dtype=tl.int32)
    topk_s = tl.zeros([TOP_K], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        best_e = tl.argmax(scores_pruned, axis=0)
        topk_indices = tl.where(topk_offs == k, best_e.to(tl.int32), topk_indices)
        s_at_best = tl.sum(tl.where(offs == best_e, s, 0.0), axis=0)
        topk_s = tl.where(topk_offs == k, s_at_best, topk_s)
        scores_pruned = tl.where(offs == best_e, NEG_INF, scores_pruned)
        if best_e >= local_start and best_e < local_start + E_LOCAL:
            tl.atomic_add(token_counts_ptr + (best_e - local_start), 1)
    tl.store(topk_idx_ptr + pid_t * TOP_K + topk_offs, topk_indices)
    w_sum = tl.sum(topk_s, axis=0) + 1e-20
    topk_w = (topk_s / w_sum) * routed_scaling_factor
    tl.store(topk_weights_ptr + pid_t * TOP_K + topk_offs, topk_w)
    old = tl.atomic_add(completion_counter_ptr, 1)
    if old == T - 1:
        e_offs = tl.arange(0, E_LOCAL)
        counts = tl.load(token_counts_ptr + e_offs)
        inclusive = tl.cumsum(counts, axis=0)
        exclusive = inclusive - counts
        tl.store(expert_start_ptr + e_offs, exclusive)
        gemm2_tile_counts = (counts + GEMM2_BLOCK_M - 1) // GEMM2_BLOCK_M
        gemm2_tile_inclusive = tl.cumsum(gemm2_tile_counts, axis=0)
        gemm2_tile_exclusive = gemm2_tile_inclusive - gemm2_tile_counts
        tl.store(gemm2_tile_prefix_ptr + e_offs, gemm2_tile_exclusive)
        total = tl.sum(counts, axis=0)
        max_count = tl.max(counts, axis=0)
        total_gemm2_tiles = tl.sum(gemm2_tile_counts, axis=0)
        tl.store(meta_ptr, total)
        tl.store(meta_ptr + 1, max_count)
        tl.store(meta_ptr + 2, total_gemm2_tiles)
        tl.store(gemm2_tile_prefix_ptr + E_LOCAL, total_gemm2_tiles)


@triton.jit
def build_sorted_from_topk_kernel(
    topk_idx_ptr, topk_weights_ptr,
    sorted_token_ids_ptr, routing_weight_flat_ptr,
    expert_start_offsets_ptr, int_buf_ptr,
    T, local_start,
    TOP_K: tl.constexpr, E_LOCAL: tl.constexpr,
):
    pid_t = tl.program_id(0)
    if pid_t >= T:
        return
    counters_ptr = int_buf_ptr + E_LOCAL + 4
    for k in tl.static_range(TOP_K):
        eidx = tl.load(topk_idx_ptr + pid_t * TOP_K + k)
        if eidx >= local_start and eidx < local_start + E_LOCAL:
            local_idx = eidx - local_start
            expert_start = tl.load(expert_start_offsets_ptr + local_idx)
            pos = tl.atomic_add(counters_ptr + local_idx, 1)
            idx = expert_start + pos
            tl.store(sorted_token_ids_ptr + idx, pid_t)
            w = tl.load(topk_weights_ptr + pid_t * TOP_K + k)
            tl.store(routing_weight_flat_ptr + idx, w)


@triton.jit
def fused_routing_sort_kernel(
    routing_logits_ptr, routing_bias_ptr,
    topk_idx_ptr, topk_weights_ptr, int_buf_ptr, expert_start_ptr,
    sorted_token_ids_ptr, routing_weight_flat_ptr,
    T, routed_scaling_factor, local_start,
    E_GLOBAL: tl.constexpr, N_GROUP: tl.constexpr, GROUP_SIZE: tl.constexpr,
    TOP_K: tl.constexpr, TOPK_GROUP: tl.constexpr, E_LOCAL: tl.constexpr,
):
    """Fused routing + build_sorted: saves one kernel launch via two-phase barrier."""
    pid_t = tl.program_id(0)
    if pid_t >= T:
        return

    token_counts_ptr = int_buf_ptr
    meta_ptr = int_buf_ptr + E_LOCAL
    completion_counter_ptr = int_buf_ptr + E_LOCAL + 3
    sort_counters_ptr = int_buf_ptr + E_LOCAL + 4
    sort_ready_ptr = int_buf_ptr + 2 * E_LOCAL + 4

    NEG_INF: tl.constexpr = -3.4028235e+38

    # ---- Phase 1: Routing ----
    offs = tl.arange(0, E_GLOBAL)
    logits = tl.load(routing_logits_ptr + pid_t * E_GLOBAL + offs).to(tl.float32)
    bias = tl.load(routing_bias_ptr + offs).to(tl.float32)
    s = tl.sigmoid(logits)
    s_with_bias = s + bias
    group_ids = offs // GROUP_SIZE
    group_offs = tl.arange(0, N_GROUP)
    group_scores = tl.zeros([N_GROUP], dtype=tl.float32)
    for g in tl.static_range(N_GROUP):
        g_vals = tl.where(group_ids == g, s_with_bias, NEG_INF)
        g_max1 = tl.max(g_vals, axis=0)
        g_argmax1 = tl.argmax(g_vals, axis=0)
        g_vals2 = tl.where(offs == g_argmax1, NEG_INF, g_vals)
        g_max2 = tl.max(g_vals2, axis=0)
        g_score = g_max1 + g_max2
        group_scores = tl.where(group_offs == g, g_score, group_scores)
    selected_group_mask = tl.zeros([E_GLOBAL], dtype=tl.float32)
    for _k in tl.static_range(TOPK_GROUP):
        best_g = tl.argmax(group_scores, axis=0)
        selected_group_mask = tl.where(group_ids == best_g, 1.0, selected_group_mask)
        group_scores = tl.where(group_offs == best_g, NEG_INF, group_scores)
    scores_pruned = tl.where(selected_group_mask > 0.5, s_with_bias, NEG_INF)
    topk_offs = tl.arange(0, TOP_K)
    topk_indices = tl.zeros([TOP_K], dtype=tl.int32)
    topk_s = tl.zeros([TOP_K], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        best_e = tl.argmax(scores_pruned, axis=0)
        topk_indices = tl.where(topk_offs == k, best_e.to(tl.int32), topk_indices)
        s_at_best = tl.sum(tl.where(offs == best_e, s, 0.0), axis=0)
        topk_s = tl.where(topk_offs == k, s_at_best, topk_s)
        scores_pruned = tl.where(offs == best_e, NEG_INF, scores_pruned)
        if best_e >= local_start and best_e < local_start + E_LOCAL:
            tl.atomic_add(token_counts_ptr + (best_e - local_start), 1)
    tl.store(topk_idx_ptr + pid_t * TOP_K + topk_offs, topk_indices)
    w_sum = tl.sum(topk_s, axis=0) + 1e-20
    topk_w = (topk_s / w_sum) * routed_scaling_factor
    tl.store(topk_weights_ptr + pid_t * TOP_K + topk_offs, topk_w)

    # ---- Barrier 1: wait for all blocks to finish routing ----
    tl.atomic_add(completion_counter_ptr, 1)
    while tl.atomic_add(completion_counter_ptr, 0) < T:
        pass

    # ---- Cumsum (block 0 only, then signal) ----
    if pid_t == 0:
        e_offs = tl.arange(0, E_LOCAL)
        counts = tl.load(token_counts_ptr + e_offs)
        inclusive = tl.cumsum(counts, axis=0)
        exclusive = inclusive - counts
        tl.store(expert_start_ptr + e_offs, exclusive)
        total = tl.sum(counts, axis=0)
        max_count = tl.max(counts, axis=0)
        gemm2_tile_counts = (counts + 15) // 16
        tl.store(meta_ptr, total)
        tl.store(meta_ptr + 1, max_count)
        tl.store(meta_ptr + 2, tl.sum(gemm2_tile_counts, axis=0))
        tl.atomic_xchg(sort_ready_ptr, 1)

    # ---- Barrier 2: wait for cumsum ----
    while tl.atomic_add(sort_ready_ptr, 0) == 0:
        pass

    # ---- Phase 2: Build sorted ----
    for k in tl.static_range(TOP_K):
        eidx = tl.load(topk_idx_ptr + pid_t * TOP_K + k)
        if eidx >= local_start and eidx < local_start + E_LOCAL:
            local_idx = eidx - local_start
            expert_start_val = tl.load(expert_start_ptr + local_idx)
            pos = tl.atomic_add(sort_counters_ptr + local_idx, 1)
            idx = expert_start_val + pos
            tl.store(sorted_token_ids_ptr + idx, pid_t)
            w = tl.load(topk_weights_ptr + pid_t * TOP_K + k)
            tl.store(routing_weight_flat_ptr + idx, w)


@triton.jit
def fused_moe_gemm1_ksplit_kernel(
    A_ptr, A_scale_ptr,
    W_ptr, W_scale_ptr,
    out_ptr,
    sorted_token_ids_ptr, expert_start_ptr, token_counts_ptr,
    T, H: tl.constexpr, OUT_DIM: tl.constexpr,
    num_hidden_blocks: tl.constexpr,
    num_gemm1_out_blocks: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_SPLITS: tl.constexpr,
    USE_ATOMIC: tl.constexpr,
):
    """GEMM1 with K-splitting for increased parallelism."""
    pid_mnk = tl.program_id(0)
    pid_e = tl.program_id(1)

    num_tokens = tl.load(token_counts_ptr + pid_e)
    expert_start = tl.load(expert_start_ptr + pid_e)

    num_n_tiles: tl.constexpr = OUT_DIM // BLOCK_N
    num_k_iters_total: tl.constexpr = H // BLOCK_K
    k_iters_per_split: tl.constexpr = num_k_iters_total // K_SPLITS

    # Decode: M-outer, N-middle, K-inner
    pid_m = pid_mnk // (num_n_tiles * K_SPLITS)
    rem = pid_mnk % (num_n_tiles * K_SPLITS)
    pid_n = rem // K_SPLITS
    pid_k = rem % K_SPLITS

    token_start = pid_m * BLOCK_M
    if token_start >= num_tokens:
        return

    col_start = pid_n * BLOCK_N
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    token_mask = (token_start + m_offs) < num_tokens
    sorted_ids = tl.load(sorted_token_ids_ptr + expert_start + token_start + m_offs, mask=token_mask, other=0)

    w_row_ids = col_start + n_offs
    n_block_idx = w_row_ids // QUANT_BLOCK
    w_base = pid_e * OUT_DIM * H
    ws_base = pid_e * num_gemm1_out_blocks * num_hidden_blocks

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # K-range for this split (use constexpr loop count for Triton pipelining)
    k_start = pid_k * k_iters_per_split

    # Use block pointers for W1 so Blackwell can issue TMA-backed loads.
    w1_block_ptr = tl.make_block_ptr(
        base=W_ptr + w_base,
        shape=(OUT_DIM, H),
        strides=(H, 1),
        offsets=(col_start, k_start * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    for ki in range(k_iters_per_split):
        k_iter = k_start + ki
        k_offs = tl.arange(0, BLOCK_K)
        k_ids = k_iter * BLOCK_K + k_offs

        a_ptrs = A_ptr + sorted_ids[:, None] * H + k_ids[None, :]
        a_fp8 = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        w_fp8 = tl.load(w1_block_ptr)
        w1_block_ptr = tl.advance(w1_block_ptr, (0, BLOCK_K))

        partial = tl.dot(
            a_fp8.to(tl.float8e4nv),
            tl.trans(w_fp8.to(tl.float8e4nv)),
            out_dtype=tl.float32,
        )

        a_scale_ptrs = A_scale_ptr + k_iter * T + sorted_ids
        a_scales = tl.load(a_scale_ptrs, mask=token_mask, other=1.0)
        w_scale_ptrs = W_scale_ptr + ws_base + n_block_idx * num_hidden_blocks + k_iter
        w_scales = tl.load(w_scale_ptrs)
        acc += partial * (a_scales[:, None] * w_scales[None, :])

    out_rows = expert_start + token_start + m_offs
    out_offs = out_rows[:, None] * OUT_DIM + w_row_ids[None, :]

    if USE_ATOMIC:
        tl.atomic_add(out_ptr + out_offs, acc, mask=token_mask[:, None])
    else:
        tl.store(out_ptr + out_offs, acc, mask=token_mask[:, None])


@triton.jit
def fused_swiglu_simple_kernel(
    gemm1_out_ptr, swiglu_out_ptr, meta_ptr,
    I: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    num_total_tokens = tl.load(meta_ptr)
    if pid_t >= num_total_tokens:
        return
    offs = tl.arange(0, BLOCK_SIZE)
    base = pid_t * (2 * I)
    x1 = tl.load(gemm1_out_ptr + base + offs)
    x2 = tl.load(gemm1_out_ptr + base + I + offs)
    silu_x2 = x2 * tl.sigmoid(x2)
    result = silu_x2 * x1
    tl.store(swiglu_out_ptr + pid_t * I + offs, result)


@triton.jit
def fused_swiglu_fp16_kernel(
    gemm1_out_ptr, swiglu_f16_ptr, row_scale_ptr, meta_ptr,
    target_max,
    I: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU + per-row dynamic scaling to FP16 range.
    Stores FP16 SwiGLU output and FP32 scale factor (1/scale) for GEMM2 correction."""
    pid_t = tl.program_id(0)
    num_total_tokens = tl.load(meta_ptr)
    if pid_t >= num_total_tokens:
        return
    offs = tl.arange(0, BLOCK_SIZE)
    base = pid_t * (2 * I)
    x1 = tl.load(gemm1_out_ptr + base + offs)
    x2 = tl.load(gemm1_out_ptr + base + I + offs)
    silu_x2 = x2 * tl.sigmoid(x2)
    result = silu_x2 * x1  # float32

    # Per-row dynamic scaling: scale values into FP16 range
    max_val = tl.max(tl.abs(result), axis=0)
    scale = target_max / (max_val + 1e-20)  # scale to fit in FP16

    result_scaled = result * scale
    tl.store(swiglu_f16_ptr + pid_t * I + offs, result_scaled.to(tl.float16))
    tl.store(row_scale_ptr + pid_t, 1.0 / scale)  # store inverse for GEMM2 correction


@triton.jit
def fused_gemm2_ksplit_kernel(
    swiglu_out_ptr, W2_ptr, W2_scale_ptr,
    output_ptr, routing_weight_ptr,
    sorted_token_ids_ptr, expert_start_ptr, token_counts_ptr,
    T, H: tl.constexpr, I: tl.constexpr,
    num_hidden_blocks: tl.constexpr,
    num_intermediate_blocks: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_SPLITS: tl.constexpr,
):
    """GEMM2 with K-splitting. Always uses atomicAdd (multi-expert accumulation)."""
    pid_mnk = tl.program_id(0)
    pid_e = tl.program_id(1)

    num_tokens = tl.load(token_counts_ptr + pid_e)
    expert_start_val = tl.load(expert_start_ptr + pid_e)

    num_k_iters_total: tl.constexpr = I // BLOCK_K
    k_iters_per_split: tl.constexpr = num_k_iters_total // K_SPLITS

    # M-outer ordering: keeps float32 A-matrix rows in L2 across N-tiles
    num_nk: tl.constexpr = (H // BLOCK_N) * K_SPLITS
    pid_m = pid_mnk // num_nk
    rem = pid_mnk % num_nk
    pid_n = rem // K_SPLITS
    pid_k = rem % K_SPLITS

    token_start = pid_m * BLOCK_M
    if token_start >= num_tokens:
        return

    h_start = pid_n * BLOCK_N
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    token_mask = (token_start + m_offs) < num_tokens
    local_ids = expert_start_val + token_start + m_offs
    sorted_ids = tl.load(sorted_token_ids_ptr + local_ids, mask=token_mask, other=0)
    rw = tl.load(routing_weight_ptr + local_ids, mask=token_mask, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    h_ids = h_start + n_offs
    h_block_idx = h_ids // QUANT_BLOCK
    w2_base = pid_e * H * I
    ws2_base = pid_e * num_hidden_blocks * num_intermediate_blocks

    k_start = pid_k * k_iters_per_split

    # Use block pointers for W2 so Blackwell can issue TMA-backed loads.
    w2_block_ptr = tl.make_block_ptr(
        base=W2_ptr + w2_base,
        shape=(H, I),
        strides=(I, 1),
        offsets=(h_start, k_start * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    for ki in range(k_iters_per_split):
        k_iter = k_start + ki
        k_offs = tl.arange(0, BLOCK_K)
        k_ids = k_iter * BLOCK_K + k_offs
        s_ptrs = swiglu_out_ptr + local_ids[:, None] * I + k_ids[None, :]
        s_vals = tl.load(s_ptrs, mask=token_mask[:, None], other=0.0)
        w_raw = tl.load(w2_block_ptr)
        w_f32 = w_raw.to(tl.float32)
        w2_block_ptr = tl.advance(w2_block_ptr, (0, BLOCK_K))
        ws_ptrs = W2_scale_ptr + ws2_base + h_block_idx * num_intermediate_blocks + k_iter
        w_scales = tl.load(ws_ptrs)
        partial = tl.dot(s_vals, tl.trans(w_f32))
        acc += partial * w_scales[None, :]

    acc_weighted = acc * rw[:, None]
    out_ptrs = output_ptr + sorted_ids[:, None].to(tl.int64) * H + h_ids[None, :]
    tl.atomic_add(out_ptrs, acc_weighted, mask=token_mask[:, None])


@triton.jit
def fused_gemm2_fp16_kernel(
    swiglu_f16_ptr, row_scale_ptr, W2_ptr, W2_scale_ptr,
    output_ptr, routing_weight_ptr,
    sorted_token_ids_ptr, expert_start_ptr, token_counts_ptr,
    T, H: tl.constexpr, I: tl.constexpr,
    num_hidden_blocks: tl.constexpr,
    num_intermediate_blocks: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """GEMM2 using FP16 tensor cores (2x throughput vs TF32 on Blackwell).
    Inputs: FP16 SwiGLU (dynamically scaled) × FP8 weights (converted to FP16).
    Accumulator: FP32. Row scale correction applied in epilogue."""
    pid_mnk = tl.program_id(0)
    pid_e = tl.program_id(1)

    num_tokens = tl.load(token_counts_ptr + pid_e)
    expert_start_val = tl.load(expert_start_ptr + pid_e)

    num_k_iters_total: tl.constexpr = I // BLOCK_K
    num_nk: tl.constexpr = H // BLOCK_N
    pid_m = pid_mnk // num_nk
    rem = pid_mnk % num_nk
    pid_n = rem

    token_start = pid_m * BLOCK_M
    if token_start >= num_tokens:
        return

    h_start = pid_n * BLOCK_N
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    token_mask = (token_start + m_offs) < num_tokens
    local_ids = expert_start_val + token_start + m_offs
    sorted_ids = tl.load(sorted_token_ids_ptr + local_ids, mask=token_mask, other=0)
    rw = tl.load(routing_weight_ptr + local_ids, mask=token_mask, other=0.0)
    rs = tl.load(row_scale_ptr + local_ids, mask=token_mask, other=1.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    h_ids = h_start + n_offs
    h_block_idx = h_ids // QUANT_BLOCK
    w2_base = pid_e * H * I
    ws2_base = pid_e * num_hidden_blocks * num_intermediate_blocks

    w2_block_ptr = tl.make_block_ptr(
        base=W2_ptr + w2_base,
        shape=(H, I),
        strides=(I, 1),
        offsets=(h_start, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    for ki in range(num_k_iters_total):
        k_offs = tl.arange(0, BLOCK_K)
        k_ids = ki * BLOCK_K + k_offs
        # Load FP16 SwiGLU input
        s_ptrs = swiglu_f16_ptr + local_ids[:, None] * I + k_ids[None, :]
        s_f16 = tl.load(s_ptrs, mask=token_mask[:, None], other=0.0)
        # Load FP8 weights → convert to FP16
        w_raw = tl.load(w2_block_ptr)
        w_f16 = w_raw.to(tl.float16)
        w2_block_ptr = tl.advance(w2_block_ptr, (0, BLOCK_K))
        # FP16 tensor core dot product → FP32 accumulator
        partial = tl.dot(s_f16, tl.trans(w_f16), out_dtype=tl.float32)
        # Apply weight block scale (FP32)
        ws_ptrs = W2_scale_ptr + ws2_base + h_block_idx * num_intermediate_blocks + ki
        w_scales = tl.load(ws_ptrs)
        acc += partial * w_scales[None, :]

    # Epilogue: apply row scale correction (undo per-row SwiGLU scaling) and routing weight
    combined_scale = rw * rs  # routing_weight × (1/swiglu_scale)
    acc_weighted = acc * combined_scale[:, None]
    out_ptrs = output_ptr + sorted_ids[:, None].to(tl.int64) * H + h_ids[None, :]
    tl.atomic_add(out_ptrs, acc_weighted, mask=token_mask[:, None])


@triton.jit
def build_gemm2_worklist_kernel(
    gemm2_tile_prefix_ptr, worklist_expert_ptr, worklist_token_start_ptr,
    BLOCK_M: tl.constexpr,
    E_LOCAL: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tiles = tl.load(gemm2_tile_prefix_ptr + E_LOCAL)
    if pid >= total_tiles:
        return

    pid_e = 0
    local_tile_idx = 0
    for e in tl.static_range(E_LOCAL):
        tile_begin = tl.load(gemm2_tile_prefix_ptr + e)
        tile_end = tl.load(gemm2_tile_prefix_ptr + e + 1)
        if pid >= tile_begin and pid < tile_end:
            pid_e = e
            local_tile_idx = pid - tile_begin

    tl.store(worklist_expert_ptr + pid, pid_e)
    tl.store(worklist_token_start_ptr + pid, local_tile_idx * BLOCK_M)


@triton.jit
def fused_gemm2_fp16_worklist_kernel(
    swiglu_f16_ptr, row_scale_ptr, W2_ptr, W2_scale_ptr,
    output_ptr, routing_weight_ptr,
    sorted_token_ids_ptr, expert_start_ptr, token_counts_ptr,
    worklist_expert_ptr, worklist_token_start_ptr,
    T, H: tl.constexpr, I: tl.constexpr,
    num_hidden_blocks: tl.constexpr,
    num_intermediate_blocks: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)

    num_k_iters_total: tl.constexpr = I // BLOCK_K
    num_nk: tl.constexpr = H // BLOCK_N
    pid_tile = pid_mn // num_nk
    pid_n = pid_mn % num_nk

    pid_e = tl.load(worklist_expert_ptr + pid_tile)
    token_start = tl.load(worklist_token_start_ptr + pid_tile)
    num_tokens = tl.load(token_counts_ptr + pid_e)
    expert_start_val = tl.load(expert_start_ptr + pid_e)
    if token_start >= num_tokens:
        return

    h_start = pid_n * BLOCK_N
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    token_mask = (token_start + m_offs) < num_tokens
    local_ids = expert_start_val + token_start + m_offs
    sorted_ids = tl.load(sorted_token_ids_ptr + local_ids, mask=token_mask, other=0)
    rw = tl.load(routing_weight_ptr + local_ids, mask=token_mask, other=0.0)
    rs = tl.load(row_scale_ptr + local_ids, mask=token_mask, other=1.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    h_ids = h_start + n_offs
    h_block_idx = h_ids // QUANT_BLOCK
    w2_base = pid_e * H * I
    ws2_base = pid_e * num_hidden_blocks * num_intermediate_blocks

    w2_block_ptr = tl.make_block_ptr(
        base=W2_ptr + w2_base,
        shape=(H, I),
        strides=(I, 1),
        offsets=(h_start, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    for ki in range(num_k_iters_total):
        k_offs = tl.arange(0, BLOCK_K)
        k_ids = ki * BLOCK_K + k_offs
        s_ptrs = swiglu_f16_ptr + local_ids[:, None] * I + k_ids[None, :]
        s_f16 = tl.load(s_ptrs, mask=token_mask[:, None], other=0.0)
        w_raw = tl.load(w2_block_ptr)
        w_f16 = w_raw.to(tl.float16)
        w2_block_ptr = tl.advance(w2_block_ptr, (0, BLOCK_K))
        partial = tl.dot(s_f16, tl.trans(w_f16), out_dtype=tl.float32)
        ws_ptrs = W2_scale_ptr + ws2_base + h_block_idx * num_intermediate_blocks + ki
        w_scales = tl.load(ws_ptrs)
        acc += partial * w_scales[None, :]

    combined_scale = rw * rs
    acc_weighted = acc * combined_scale[:, None]
    out_ptrs = output_ptr + sorted_ids[:, None].to(tl.int64) * H + h_ids[None, :]
    tl.atomic_add(out_ptrs, acc_weighted, mask=token_mask[:, None])


def run_impl(
    routing_logits, routing_bias,
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
    fw_output=None,
):
    device = hidden_states.device
    T = routing_logits.shape[0]

    E_global = 256
    E_local = 32
    H = 7168
    I = 2048
    OUT_DIM = 4096
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    GROUP_SIZE = E_global // N_GROUP
    QUANT_BLOCK = 128
    num_hidden_blocks = H // QUANT_BLOCK     # 56
    num_intermediate_blocks = I // QUANT_BLOCK  # 16
    num_gemm1_out_blocks = OUT_DIM // QUANT_BLOCK  # 32
    local_start = int(local_expert_offset)

    # Adaptive GEMM1 config based on T:
    # - T<=2: BLOCK_M=32 + K_SPLITS=8  (GEMV: larger tile better, K-split for parallelism)
    # - T=3-4: BLOCK_M=16 + K_SPLITS=4 (small batch: lower regs + moderate K-split)
    # - T=5-32: BLOCK_M=16 + K_SPLITS=2 (small-medium T: modest parallelism boost)
    # - T=33-256: BLOCK_M=16, no K-split (medium T: lower regs -> higher occupancy)
    # - T>256: BLOCK_M=64, no K-split (best large-T compute-bound setting on current sweeps)
    # GEMM2: BLOCK_M=16 and never K-split (atomicAdd contention from 8 experts already dominates).
    # GEMM1: T=5-32 uses moderate K-split=2 for a small parallelism boost.
    if T <= 2:
        ks_g1, bm_g1 = 8, 32
    elif T <= 4:
        ks_g1, bm_g1 = 4, 16
    elif T <= 32:
        ks_g1, bm_g1 = 2, 16
    elif T <= ASYNC_PATH_MAX_TOKENS:
        ks_g1, bm_g1 = 1, 16  # Keep BM=16 through the async band to avoid over-tiling.
    else:
        ks_g1, bm_g1 = 1, 64

    if T <= ASYNC_PATH_MAX_TOKENS:
        max_assigned = T * TOP_K

        if T not in _ws:
            _pad = 128
            zb = torch.zeros(_pad, dtype=torch.float32, device=device)
            ib = zb.view(torch.int32)
            _ws[T] = (
                torch.empty((T, TOP_K), dtype=torch.int32, device=device),
                torch.empty((T, TOP_K), dtype=torch.float32, device=device),
                torch.empty(E_local, dtype=torch.int32, device=device),
                torch.empty(max_assigned, dtype=torch.int32, device=device),
                torch.empty(max_assigned, dtype=torch.float32, device=device),
                torch.empty((max_assigned, OUT_DIM), dtype=torch.float32, device=device),
                zb, ib, torch.empty((T, H), dtype=torch.float32, device=device), ib[E_local:E_local + 3],
                # Cache the FP16 SwiGLU output and row scale for the async GEMM2 path.
                torch.empty((max_assigned, I), dtype=torch.float16, device=device),
                torch.empty(max_assigned, dtype=torch.float32, device=device),
            )

        (topk_idx, topk_weights, expert_start_offsets, sorted_token_ids,
         routing_weight_flat, gemm1_out, zero_buf, int_buf, output_buf, meta_ptr_tensor,
         swiglu_f16, row_scale) = _ws[T]
        zero_buf.zero_()
        output = fw_output if fw_output is not None else output_buf
        output.zero_()

        fused_routing_sort_kernel[(T,)](
            routing_logits, routing_bias,
            topk_idx, topk_weights, int_buf, expert_start_offsets,
            sorted_token_ids, routing_weight_flat,
            T, routed_scaling_factor, local_start,
            E_GLOBAL=E_global, N_GROUP=N_GROUP, GROUP_SIZE=GROUP_SIZE,
            TOP_K=TOP_K, TOPK_GROUP=TOPK_GROUP, E_LOCAL=E_local,
        )

        # GEMM1 with K-split
        use_atomic_g1 = ks_g1 > 1
        if use_atomic_g1:
            gemm1_out.zero_()

        num_m1_tiles = (T + bm_g1 - 1) // bm_g1
        num_n1_tiles = OUT_DIM // 128
        fused_moe_gemm1_ksplit_kernel[(num_m1_tiles * num_n1_tiles * ks_g1, E_local)](
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm1_out,
            sorted_token_ids, expert_start_offsets, int_buf,
            T, H, OUT_DIM,
            num_hidden_blocks, num_gemm1_out_blocks,
            QUANT_BLOCK, bm_g1, 128, 128,
            K_SPLITS=ks_g1,
            USE_ATOMIC=use_atomic_g1,
            num_stages=4,
        )

        # Reuse the same fused SwiGLU plus FP16 quantization path as the sync branch.
        fused_swiglu_fp16_kernel[(max_assigned,)](
            gemm1_out, swiglu_f16, row_scale, meta_ptr_tensor,
            SWIGLU_FP16_TARGET_MAX, I, 2048,
        )

        # Run GEMM2 in FP16 so Blackwell tensor cores stay on the higher-throughput path.
        num_m2_tiles = (T + 15) // 16
        num_n2_tiles = H // 128
        fused_gemm2_fp16_kernel[(num_n2_tiles * num_m2_tiles, E_local)](
            swiglu_f16, row_scale,
            gemm2_weights, gemm2_weights_scale,
            output, routing_weight_flat,
            sorted_token_ids, expert_start_offsets, int_buf,
            T, H, I,
            num_hidden_blocks, num_intermediate_blocks,
            QUANT_BLOCK, 16, 128, 128,
            num_stages=3,
        )
    else:
        # Sync path (T > 256) - no K-splitting
        max_assigned = T * TOP_K
        bm_g2_sync = 32 if T >= 8192 else 16

        if T not in _ws_sync:
            _pad = 128
            zb = torch.zeros(_pad, dtype=torch.float32, device=device)
            ib = zb.view(torch.int32)
            meta_host = torch.empty(3, dtype=torch.int32, pin_memory=True)
            meta_stream = torch.cuda.Stream(device=device)
            max_gemm2_tiles = (max_assigned + bm_g2_sync - 1) // bm_g2_sync + E_local
            _ws_sync[T] = (
                torch.empty((T, TOP_K), dtype=torch.int32, device=device),
                torch.empty((T, TOP_K), dtype=torch.float32, device=device),
                torch.empty(E_local, dtype=torch.int32, device=device),
                torch.empty(max_assigned, dtype=torch.int32, device=device),
                torch.empty(max_assigned, dtype=torch.float32, device=device),
                torch.empty((max_assigned, OUT_DIM), dtype=torch.float32, device=device),
                zb, ib, torch.empty((T, H), dtype=torch.float32, device=device),
                ib[E_local:E_local + 3],
                ib[2 * E_local + 4:2 * E_local + 4 + E_local + 1],
                torch.empty(max_gemm2_tiles, dtype=torch.int32, device=device),
                torch.empty(max_gemm2_tiles, dtype=torch.int32, device=device),
                # Cache the FP16 SwiGLU output and row scale for the sync GEMM2 path.
                torch.empty((max_assigned, I), dtype=torch.float16, device=device),
                torch.empty(max_assigned, dtype=torch.float32, device=device),
                meta_host, meta_stream,
            )

        (topk_idx, topk_weights, expert_start_offsets,
         sorted_token_ids, routing_weight_flat, gemm1_out, zero_buf, int_buf, output_buf,
         meta_ptr_tensor,
         gemm2_tile_prefix,
         gemm2_worklist_expert, gemm2_worklist_token_start,
         swiglu_f16, row_scale, meta_host, meta_stream) = _ws_sync[T]
        zero_buf.zero_()
        output = fw_output if fw_output is not None else output_buf
        output.zero_()

        fused_moe_routing_kernel[(T,)](
            routing_logits, routing_bias,
            topk_idx, topk_weights, int_buf, expert_start_offsets,
            T, routed_scaling_factor, local_start,
            E_GLOBAL=E_global, N_GROUP=N_GROUP, GROUP_SIZE=GROUP_SIZE,
            TOP_K=TOP_K, TOPK_GROUP=TOPK_GROUP, E_LOCAL=E_local,
            GEMM2_BLOCK_M=bm_g2_sync,
        )

        current_stream = torch.cuda.current_stream(device=device)
        meta_stream.wait_stream(current_stream)
        with torch.cuda.stream(meta_stream):
            meta_host.copy_(meta_ptr_tensor, non_blocking=True)

        build_sorted_from_topk_kernel[(T,)](
            topk_idx, topk_weights,
            sorted_token_ids, routing_weight_flat,
            expert_start_offsets, int_buf,
            T, local_start, TOP_K=TOP_K, E_LOCAL=E_local,
        )

        meta_stream.synchronize()
        total_assigned, max_tokens, total_gemm2_tiles = meta_host.tolist()
        if total_assigned == 0 or max_tokens == 0 or total_gemm2_tiles == 0:
            return output

        # Use eight warps here because the larger sync tiles otherwise become register-bound.
        bm_sync = 64
        num_m1_tiles = (max_tokens + bm_sync - 1) // bm_sync
        num_n1_tiles = OUT_DIM // 128
        fused_moe_gemm1_ksplit_kernel[(num_m1_tiles * num_n1_tiles, E_local)](
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm1_out,
            sorted_token_ids, expert_start_offsets, int_buf,
            T, H, OUT_DIM,
            num_hidden_blocks, num_gemm1_out_blocks,
            QUANT_BLOCK, bm_sync, 128, 128,
            K_SPLITS=1, USE_ATOMIC=False,
            num_stages=4, num_warps=8,
        )

        fused_swiglu_fp16_kernel[(total_assigned,)](
            gemm1_out, swiglu_f16, row_scale, meta_ptr_tensor,
            SWIGLU_FP16_TARGET_MAX, I, 2048,
        )

        num_n2_tiles = H // 128
        build_gemm2_worklist_kernel[(total_gemm2_tiles,)](
            gemm2_tile_prefix, gemm2_worklist_expert, gemm2_worklist_token_start,
            bm_g2_sync, E_local,
        )
        fused_gemm2_fp16_worklist_kernel[(num_n2_tiles * total_gemm2_tiles,)](
            swiglu_f16, row_scale,
            gemm2_weights, gemm2_weights_scale,
            output, routing_weight_flat,
            sorted_token_ids, expert_start_offsets, int_buf,
            gemm2_worklist_expert, gemm2_worklist_token_start,
            T, H, I,
            num_hidden_blocks, num_intermediate_blocks,
            QUANT_BLOCK, bm_g2_sync, 128, 128,
            num_stages=3,
        )

    return output


@torch.no_grad()
def run(
    routing_logits, routing_bias,
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
    output,
) -> None:
    run_impl(
        routing_logits, routing_bias,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor,
        output,
    )
