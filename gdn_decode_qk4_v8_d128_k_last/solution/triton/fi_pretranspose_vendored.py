"""
Vendored subset of FlashInfer's GDN decode pretranspose kernel.

Source basis:
- reference/flashinfer/flashinfer/gdn_decode.py
- reference/flashinfer/flashinfer/gdn_kernels/gdn_decode_pretranspose.py

This copy intentionally keeps only the contest-relevant path:
- decode only (`T == 1`)
- float32 recurrent state
- pretranspose / K-last state layout `[B, HV, V, K]`
- no pool indexing and no BF16-state backend

The goal is to keep the CuTe source local and editable so future tuning can
modify the decode path directly without going through `flashinfer.gdn_decode`.
This is the active vendored CuTe implementation used by the current v35 decode
hybrid prepared for `submission-v21`.

Current contest-specific deltas versus upstream:
- `kernel.py` calls a contest-only fast path with the benchmark contract fixed
  at `T == 1`, FP32 contiguous state, preallocated output, and
  `use_qk_l2norm=False`
- the kernel body removes shared-memory `sOutput` staging and stores the final
  reduction directly to global output
- `num_blocks_per_state` is now parameterized so the dispatch can promote
  `B16_NUM_BLOCKS_PER_STATE=16` only for `batch_size == 16`
- a dedicated `TILE_V_LARGE=16` kernel is kept only for `batch_size == 48`
  with `num_blocks_per_state=4`, while `B32/B64` stay on
  `DEFAULT_NUM_BLOCKS_PER_STATE=8`

Original FlashInfer sources are Apache 2.0 licensed.
"""

import functools
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

TILE_V = 8
TILE_V_LARGE = 16
TILE_K = 128
NUM_STAGES = 2
NUM_THREADS = 128
DEFAULT_NUM_BLOCKS_PER_STATE = 8
B16_NUM_BLOCKS_PER_STATE = 16
PERSISTENT_CTA_PER_SM = 16


@cute.kernel
def gdn_decode_kernel_small_batch_pretranspose_vendored(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    num_blocks_per_state: cutlass.Constexpr[int],
    use_qk_output_shortcut: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    block_idx, _, _ = cute.arch.block_idx()
    batch_idx = block_idx // num_blocks_per_state
    batch_inner = block_idx % num_blocks_per_state
    num_v_tiles_per_block = num_v_tiles // num_blocks_per_state
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)
    i_t = 0

    smem = cutlass.utils.SmemAllocator()
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
    # Keep `v` in FP32 shared memory; BF16 staging regressed the B32/B64 band.
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)

    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_v_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    k_start = lane_id * vec_size

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_a = cutlass.Float32(a[i_n, i_t, i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    cute.arch.barrier()

    gSrc_batch = h0_source[(batch_idx, None, None)]
    gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0))
    gSrc = cute.local_tile(gSrc_batch, (TILE_V, TILE_K), (None, 0))

    thr_copy_load = tiled_copy_load.get_slice(tidx)

    start_v_tiles = batch_inner * num_v_tiles_per_block
    prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
    for v_tiles in range(start_v_tiles, start_v_tiles + prefetch_count):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES
        gSrc_tile = gSrc[(None, None, v_tiles)]
        sData_stage = sData[(None, None, stage)]
        thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
        thr_sData = thr_copy_load.partition_D(sData_stage)
        cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
        cute.arch.cp_async_commit_group()

    q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    cute.autovec_copy(q_tile, r_q_bf16)
    cute.autovec_copy(k_tile, r_k_bf16)

    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = cutlass.Float32(r_q_bf16[i])
        r_k[i] = cutlass.Float32(r_k_bf16[i])

    v_tile = cute.local_tile(v, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane_id))
    cute.autovec_copy(v_tile, r_v_bf16)
    for i in cutlass.range_constexpr(vec_size):
        sV[k_start + i] = cutlass.Float32(r_v_bf16[i])

    cute.arch.barrier()

    r_g = 0.0
    r_beta = 0.0
    if lane_id == 0:
        x = r_a + r_dt_bias
        beta_x = softplus_beta * x
        softplus_x = 0.0

        if beta_x <= softplus_threshold:
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            log_input = cutlass.Float32(1.0 + exp_beta_x)
            log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
            softplus_x = cutlass.Float32(
                (cutlass.Float32(1.0) / softplus_beta) * log_result
            )
        else:
            softplus_x = x

        r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
        r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))
        r_g = cute.exp(r_g_value, fastmath=True)

    r_g = cute.arch.shuffle_sync(r_g, 0)
    r_beta = cute.arch.shuffle_sync(r_beta, 0)

    if use_qk_l2norm:
        sum_q = 0.0
        sum_k = 0.0
        for i in cutlass.range_constexpr(vec_size):
            sum_q += r_q[i] * r_q[i]
            sum_k += r_k[i] * r_k[i]
        for offset in [16, 8, 4, 2, 1]:
            sum_q += cute.arch.shuffle_sync_bfly(
                sum_q, offset=offset, mask=-1, mask_and_clamp=31
            )
            sum_k += cute.arch.shuffle_sync_bfly(
                sum_k, offset=offset, mask=-1, mask_and_clamp=31
            )

        inv_norm_q = cute.rsqrt(sum_q + 1e-6, fastmath=True)
        inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
        for i in cutlass.range_constexpr(vec_size):
            r_q[i] = r_q[i] * inv_norm_q
            r_k[i] = r_k[i] * inv_norm_k

    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = r_q[i] * scale

    qk_dot = 0.0
    if cutlass.const_expr(use_qk_output_shortcut):
        for i in cutlass.range_constexpr(vec_size):
            qk_dot += r_q[i] * r_k[i]
        for offset in [16, 8, 4, 2, 1]:
            qk_dot += cute.arch.shuffle_sync_bfly(
                qk_dot, offset=offset, mask=-1, mask_and_clamp=31
            )

    end_v_tiles = start_v_tiles + num_v_tiles_per_block
    for v_tiles in range(start_v_tiles, end_v_tiles):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        next_v_tiles = v_tiles + prefetch_count
        if next_v_tiles < end_v_tiles:
            next_stage = (next_v_tiles - start_v_tiles) % NUM_STAGES
            gSrc_next = gSrc[(None, None, next_v_tiles)]
            sData_next = sData[(None, None, next_stage)]
            thr_gSrc = thr_copy_load.partition_S(gSrc_next)
            thr_sData = thr_copy_load.partition_D(sData_next)
            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        for row in cutlass.range_constexpr(0, TILE_V, 4):
            row_offset = tidx // 32
            sum_hk = 0.0
            sum_hq_base = 0.0

            sData_tile = cute.local_tile(
                sData, (1, vec_size, 1), (row + row_offset, lane_id, stage)
            )
            cute.autovec_copy(sData_tile, r_h)

            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = r_h[i] * r_g
                sum_hk += r_h[i] * r_k[i]
                if cutlass.const_expr(use_qk_output_shortcut):
                    sum_hq_base += r_h[i] * r_q[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                )
                if cutlass.const_expr(use_qk_output_shortcut):
                    sum_hq_base += cute.arch.shuffle_sync_bfly(
                        sum_hq_base, offset=offset, mask=-1, mask_and_clamp=31
                    )

            v_new = sV[v_tiles * TILE_V + row + row_offset] - sum_hk
            v_new = v_new * r_beta

            sum_hq = 0.0
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = r_h[i] + r_k[i] * v_new
                if not cutlass.const_expr(use_qk_output_shortcut):
                    sum_hq += r_h[i] * r_q[i]

            gDst_tile = cute.local_tile(
                gDst,
                (1, 1, vec_size, 1),
                (0, row + row_offset, lane_id, v_tiles),
            )
            cute.autovec_copy(r_h, gDst_tile)

            if not cutlass.const_expr(use_qk_output_shortcut):
                for offset in [16, 8, 4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                    )

            # Producer lanes already hold the final reduced value, so writing
            # directly to global output avoids the old `sOutput` shared-memory
            # round-trip and the trailing block-wide barrier. Reuse the
            # precomputed qk_dot to avoid a second h·q reduction after the
            # state update.
            o_idx = v_tiles * TILE_V + row + row_offset
            if lane_id == 0 and o_idx < V:
                if cutlass.const_expr(use_qk_output_shortcut):
                    o[(i_n, i_t, i_hv, o_idx)] = cutlass.BFloat16(
                        sum_hq_base + v_new * qk_dot
                    )
                else:
                    o[(i_n, i_t, i_hv, o_idx)] = cutlass.BFloat16(sum_hq)


@cute.kernel
def gdn_decode_kernel_large_batch_pretranspose_vendored_persistent(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    num_blocks_per_state: cutlass.Constexpr[int],
):
    """
    Persistent large-batch variant for the contest pretranspose decode hot path.

    This keeps the exact v19 math, but turns the B32/B48/B64 regime into a
    block-stealing launch so `B48/B64` stop paying for the large partial wave
    observed in NCU on B200.
    """
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    block_idx, _, _ = cute.arch.block_idx()
    grid_dim_x, _, _ = cute.arch.grid_dim()

    total_work = B * HV * num_blocks_per_state
    current_work_idx = block_idx
    num_v_tiles_per_block = num_v_tiles // num_blocks_per_state

    smem = cutlass.utils.SmemAllocator()
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)

    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_v_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    k_start = lane_id * vec_size

    for _ in cutlass.range_constexpr(2):
        if current_work_idx < total_work:
            batch_idx = current_work_idx // num_blocks_per_state
            batch_inner = current_work_idx % num_blocks_per_state
            i_n = batch_idx // HV
            i_hv = batch_idx % HV
            i_h = i_hv // (HV // H)
            i_t = 0

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_a = cutlass.Float32(a[i_n, i_t, i_hv])
            r_dt_bias = cutlass.Float32(dt_bias[i_hv])
            r_b = cutlass.Float32(b[i_n, i_t, i_hv])

            cute.arch.barrier()

            gSrc_batch = h0_source[(batch_idx, None, None)]
            gDst = cute.local_tile(
                h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0)
            )
            gSrc = cute.local_tile(gSrc_batch, (TILE_V, TILE_K), (None, 0))

            thr_copy_load = tiled_copy_load.get_slice(tidx)

            start_v_tiles = batch_inner * num_v_tiles_per_block
            prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
            for v_tiles in range(start_v_tiles, start_v_tiles + prefetch_count):
                stage = (v_tiles - start_v_tiles) % NUM_STAGES
                gSrc_tile = gSrc[(None, None, v_tiles)]
                sData_stage = sData[(None, None, stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
                thr_sData = thr_copy_load.partition_D(sData_stage)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
            k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
            cute.autovec_copy(q_tile, r_q_bf16)
            cute.autovec_copy(k_tile, r_k_bf16)

            for i in cutlass.range_constexpr(vec_size):
                r_q[i] = cutlass.Float32(r_q_bf16[i])
                r_k[i] = cutlass.Float32(r_k_bf16[i])

            v_tile = cute.local_tile(v, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane_id))
            cute.autovec_copy(v_tile, r_v_bf16)
            for i in cutlass.range_constexpr(vec_size):
                sV[k_start + i] = cutlass.Float32(r_v_bf16[i])

            cute.arch.barrier()

            r_g = 0.0
            r_beta = 0.0
            if lane_id == 0:
                x = r_a + r_dt_bias
                beta_x = softplus_beta * x
                softplus_x = 0.0

                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x, fastmath=True)
                    log_input = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta) * log_result
                    )
                else:
                    softplus_x = x

                r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
                r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))
                r_g = cute.exp(r_g_value, fastmath=True)

            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            if use_qk_l2norm:
                sum_q = 0.0
                sum_k = 0.0
                for i in cutlass.range_constexpr(vec_size):
                    sum_q += r_q[i] * r_q[i]
                    sum_k += r_k[i] * r_k[i]
                for offset in [16, 8, 4, 2, 1]:
                    sum_q += cute.arch.shuffle_sync_bfly(
                        sum_q, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k += cute.arch.shuffle_sync_bfly(
                        sum_k, offset=offset, mask=-1, mask_and_clamp=31
                    )

                inv_norm_q = cute.rsqrt(sum_q + 1e-6, fastmath=True)
                inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
                for i in cutlass.range_constexpr(vec_size):
                    r_q[i] = r_q[i] * inv_norm_q
                    r_k[i] = r_k[i] * inv_norm_k

            for i in cutlass.range_constexpr(vec_size):
                r_q[i] = r_q[i] * scale

            end_v_tiles = start_v_tiles + num_v_tiles_per_block
            for v_tiles in range(start_v_tiles, end_v_tiles):
                stage = (v_tiles - start_v_tiles) % NUM_STAGES

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                next_v_tiles = v_tiles + prefetch_count
                if next_v_tiles < end_v_tiles:
                    next_stage = (next_v_tiles - start_v_tiles) % NUM_STAGES
                    gSrc_next = gSrc[(None, None, next_v_tiles)]
                    sData_next = sData[(None, None, next_stage)]
                    thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                    thr_sData = thr_copy_load.partition_D(sData_next)
                    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                    cute.arch.cp_async_commit_group()

                for row in cutlass.range_constexpr(0, TILE_V, 4):
                    row_offset = tidx // 32
                    sum_hk = 0.0

                    sData_tile = cute.local_tile(
                        sData, (1, vec_size, 1), (row + row_offset, lane_id, stage)
                    )
                    cute.autovec_copy(sData_tile, r_h)

                    for i in cutlass.range_constexpr(vec_size):
                        r_h[i] = r_h[i] * r_g
                        sum_hk += r_h[i] * r_k[i]

                    for offset in [16, 8, 4, 2, 1]:
                        sum_hk += cute.arch.shuffle_sync_bfly(
                            sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                        )

                    v_new = sV[v_tiles * TILE_V + row + row_offset] - sum_hk
                    v_new = v_new * r_beta

                    sum_hq = 0.0
                    for i in cutlass.range_constexpr(vec_size):
                        r_h[i] = r_h[i] + r_k[i] * v_new
                        sum_hq += r_h[i] * r_q[i]

                    gDst_tile = cute.local_tile(
                        gDst,
                        (1, 1, vec_size, 1),
                        (0, row + row_offset, lane_id, v_tiles),
                    )
                    cute.autovec_copy(r_h, gDst_tile)

                    for offset in [16, 8, 4, 2, 1]:
                        sum_hq += cute.arch.shuffle_sync_bfly(
                            sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                        )

                    o_idx = v_tiles * TILE_V + row + row_offset
                    if lane_id == 0 and o_idx < V:
                        o[(i_n, i_t, i_hv, o_idx)] = cutlass.BFloat16(sum_hq)

            cute.arch.barrier()

        current_work_idx += grid_dim_x


@cute.kernel
def gdn_decode_kernel_large_v16_pretranspose_vendored(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    num_blocks_per_state: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    block_idx, _, _ = cute.arch.block_idx()
    batch_idx = block_idx // num_blocks_per_state
    batch_inner = block_idx % num_blocks_per_state
    num_v_tiles_per_block = num_v_tiles // num_blocks_per_state
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)
    i_t = 0

    smem = cutlass.utils.SmemAllocator()
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)

    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_v_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    k_start = lane_id * vec_size

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_a = cutlass.Float32(a[i_n, i_t, i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    cute.arch.barrier()

    gSrc_batch = h0_source[(batch_idx, None, None)]
    gDst = cute.local_tile(h0_source, (1, TILE_V_LARGE, TILE_K), (batch_idx, None, 0))
    gSrc = cute.local_tile(gSrc_batch, (TILE_V_LARGE, TILE_K), (None, 0))

    thr_copy_load = tiled_copy_load.get_slice(tidx)

    start_v_tiles = batch_inner * num_v_tiles_per_block
    prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
    for v_tiles in range(start_v_tiles, start_v_tiles + prefetch_count):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES
        gSrc_tile = gSrc[(None, None, v_tiles)]
        sData_stage = sData[(None, None, stage)]
        thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
        thr_sData = thr_copy_load.partition_D(sData_stage)
        cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
        cute.arch.cp_async_commit_group()

    q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    cute.autovec_copy(q_tile, r_q_bf16)
    cute.autovec_copy(k_tile, r_k_bf16)

    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = cutlass.Float32(r_q_bf16[i])
        r_k[i] = cutlass.Float32(r_k_bf16[i])

    v_tile = cute.local_tile(v, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane_id))
    cute.autovec_copy(v_tile, r_v_bf16)
    for i in cutlass.range_constexpr(vec_size):
        sV[k_start + i] = cutlass.Float32(r_v_bf16[i])

    cute.arch.barrier()

    r_g = 0.0
    r_beta = 0.0
    if lane_id == 0:
        x = r_a + r_dt_bias
        beta_x = softplus_beta * x
        softplus_x = 0.0

        if beta_x <= softplus_threshold:
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            log_input = cutlass.Float32(1.0 + exp_beta_x)
            log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
            softplus_x = cutlass.Float32(
                (cutlass.Float32(1.0) / softplus_beta) * log_result
            )
        else:
            softplus_x = x

        r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
        r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))
        r_g = cute.exp(r_g_value, fastmath=True)

    r_g = cute.arch.shuffle_sync(r_g, 0)
    r_beta = cute.arch.shuffle_sync(r_beta, 0)

    if use_qk_l2norm:
        sum_q = 0.0
        sum_k = 0.0
        for i in cutlass.range_constexpr(vec_size):
            sum_q += r_q[i] * r_q[i]
            sum_k += r_k[i] * r_k[i]
        for offset in [16, 8, 4, 2, 1]:
            sum_q += cute.arch.shuffle_sync_bfly(
                sum_q, offset=offset, mask=-1, mask_and_clamp=31
            )
            sum_k += cute.arch.shuffle_sync_bfly(
                sum_k, offset=offset, mask=-1, mask_and_clamp=31
            )

        inv_norm_q = cute.rsqrt(sum_q + 1e-6, fastmath=True)
        inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
        for i in cutlass.range_constexpr(vec_size):
            r_q[i] = r_q[i] * inv_norm_q
            r_k[i] = r_k[i] * inv_norm_k

    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = r_q[i] * scale

    end_v_tiles = start_v_tiles + num_v_tiles_per_block
    for v_tiles in range(start_v_tiles, end_v_tiles):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        next_v_tiles = v_tiles + prefetch_count
        if next_v_tiles < end_v_tiles:
            next_stage = (next_v_tiles - start_v_tiles) % NUM_STAGES
            gSrc_next = gSrc[(None, None, next_v_tiles)]
            sData_next = sData[(None, None, next_stage)]
            thr_gSrc = thr_copy_load.partition_S(gSrc_next)
            thr_sData = thr_copy_load.partition_D(sData_next)
            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        for row in cutlass.range_constexpr(0, TILE_V_LARGE, 4):
            row_offset = tidx // 32
            sum_hk = 0.0

            sData_tile = cute.local_tile(
                sData, (1, vec_size, 1), (row + row_offset, lane_id, stage)
            )
            cute.autovec_copy(sData_tile, r_h)

            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = r_h[i] * r_g
                sum_hk += r_h[i] * r_k[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                )

            v_new = sV[v_tiles * TILE_V_LARGE + row + row_offset] - sum_hk
            v_new = v_new * r_beta

            sum_hq = 0.0
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = r_h[i] + r_k[i] * v_new
                sum_hq += r_h[i] * r_q[i]

            gDst_tile = cute.local_tile(
                gDst,
                (1, 1, vec_size, 1),
                (0, row + row_offset, lane_id, v_tiles),
            )
            cute.autovec_copy(r_h, gDst_tile)

            for offset in [16, 8, 4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                )

            o_idx = v_tiles * TILE_V_LARGE + row + row_offset
            if lane_id == 0 and o_idx < V:
                o[(i_n, i_t, i_hv, o_idx)] = cutlass.BFloat16(sum_hq)


@cute.jit
def run_gdn_decode_kernel_small_batch_pretranspose_vendored(
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    num_blocks_per_state: cutlass.Constexpr[int],
    use_qk_output_shortcut: cutlass.Constexpr[bool],
    stream: cuda.CUstream = None,
):
    v_dim = h0_source.layout.shape[1]
    k_dim = h0_source.layout.shape[2]
    grid_batch = B * HV

    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    thread_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((1, 4))
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)
    vec_size = TILE_K // 32
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K)
    )
    smem_bytes = 4 * TILE_V * TILE_K * NUM_STAGES + 4 * k_dim + 32

    gdn_decode_kernel_small_batch_pretranspose_vendored(
        tiled_copy_load,
        h0_source,
        smem_layout_staged,
        vec_size,
        num_v_tiles,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_qk_l2norm,
        num_blocks_per_state,
        use_qk_output_shortcut,
    ).launch(
        grid=(grid_batch * num_blocks_per_state, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@cute.jit
def run_gdn_decode_kernel_large_v16_pretranspose_vendored(
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    num_blocks_per_state: cutlass.Constexpr[int],
    stream: cuda.CUstream = None,
):
    v_dim = h0_source.layout.shape[1]
    k_dim = h0_source.layout.shape[2]
    grid_batch = B * HV

    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    thread_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((1, 4))
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V_LARGE)
    vec_size = TILE_K // 32
    smem_layout_staged = cute.make_layout(
        (TILE_V_LARGE, TILE_K, NUM_STAGES),
        stride=(TILE_K, 1, TILE_V_LARGE * TILE_K),
    )
    smem_bytes = 4 * TILE_V_LARGE * TILE_K * NUM_STAGES + 4 * k_dim + 32

    gdn_decode_kernel_large_v16_pretranspose_vendored(
        tiled_copy_load,
        h0_source,
        smem_layout_staged,
        vec_size,
        num_v_tiles,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_qk_l2norm,
        num_blocks_per_state,
    ).launch(
        grid=(grid_batch * num_blocks_per_state, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@cute.jit
def run_gdn_decode_kernel_large_batch_pretranspose_vendored_persistent(
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    num_blocks_per_state: cutlass.Constexpr[int],
    stream: cuda.CUstream = None,
):
    v_dim = h0_source.layout.shape[1]
    k_dim = h0_source.layout.shape[2]
    grid_batch = B * HV

    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    thread_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((1, 4))
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)
    vec_size = TILE_K // 32
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K)
    )
    smem_bytes = 4 * TILE_V * TILE_K * NUM_STAGES + 4 * k_dim + 32

    hardware_info = cutlass.utils.HardwareInfo()
    sm_count = hardware_info.get_device_multiprocessor_count()
    full_wave_ctas = sm_count * PERSISTENT_CTA_PER_SM
    persistent_ctas = cutlass.min(grid_batch * num_blocks_per_state, full_wave_ctas)

    gdn_decode_kernel_large_batch_pretranspose_vendored_persistent(
        tiled_copy_load,
        h0_source,
        smem_layout_staged,
        vec_size,
        num_v_tiles,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_qk_l2norm,
        num_blocks_per_state,
    ).launch(
        grid=(persistent_ctas, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@functools.cache
def _get_compiled_decode_kernel_vendored(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    scale: float,
    use_qk_l2norm: bool,
    num_blocks_per_state: int,
    use_qk_output_shortcut: bool,
):
    return {}


@functools.cache
def _get_compiled_decode_kernel_contest_fastpath(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    scale: float,
    num_blocks_per_state: int,
    use_qk_output_shortcut: bool,
):
    return {}


@functools.cache
def _get_compiled_decode_kernel_contest_persistent(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    scale: float,
    num_blocks_per_state: int,
):
    return {}


@functools.cache
def _get_compiled_decode_kernel_contest_large_v16(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    scale: float,
    num_blocks_per_state: int,
):
    return {}


def run_pretranspose_decode_vendored(
    h0_source: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    scale: float,
    use_qk_l2norm: bool,
    num_blocks_per_state: int = DEFAULT_NUM_BLOCKS_PER_STATE,
    use_qk_output_shortcut: bool = False,
):
    cache_key = (
        B,
        T,
        H,
        HV,
        K,
        V,
        q.dtype,
        scale,
        use_qk_l2norm,
        num_blocks_per_state,
        use_qk_output_shortcut,
    )
    cache = _get_compiled_decode_kernel_vendored(*cache_key)

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(
            run_gdn_decode_kernel_small_batch_pretranspose_vendored,
            from_dlpack(h0_source, assumed_align=16),
            from_dlpack(A_log, assumed_align=16),
            from_dlpack(a, assumed_align=16),
            from_dlpack(dt_bias, assumed_align=16),
            from_dlpack(q, assumed_align=16),
            from_dlpack(k, assumed_align=16),
            from_dlpack(v, assumed_align=16),
            from_dlpack(b, assumed_align=16),
            from_dlpack(output, assumed_align=16),
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=scale,
            HV=HV,
            B=B,
            T=T,
            H=H,
            K=K,
            V=V,
            use_qk_l2norm=use_qk_l2norm,
            num_blocks_per_state=num_blocks_per_state,
            use_qk_output_shortcut=use_qk_output_shortcut,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cache["compiled"](h0_source, A_log, a, dt_bias, q, k, v, b, output, stream)


def run_pretranspose_decode_contest_fastpath_persistent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    scale: float | None = None,
    num_blocks_per_state: int = DEFAULT_NUM_BLOCKS_PER_STATE,
) -> None:
    """
    Contest-only persistent large-batch decode path.

    This uses the same math as the tagged v19 fast path but constrains the
    launch to one full wave and lets each CTA steal at most one extra work item
    from the dominant B48/B64 tail.
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape

    assert T == 1, f"Decode only supports T=1, got T={T}"
    assert state.shape == (B, HV, V, K), (
        f"Expected state shape [B={B}, HV={HV}, V={V}, K={K}], got {state.shape}"
    )
    assert state.dtype == torch.float32, f"state must be float32, got {state.dtype}"
    assert state.is_contiguous(), "contest fast path requires contiguous state"
    assert output is not None, "contest fast path requires preallocated output"
    assert output.shape == (B, T, HV, V), (
        f"Expected output shape [B={B}, T={T}, HV={HV}, V={V}], got {output.shape}"
    )
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    assert V % TILE_V == 0, (
        f"V must be divisible by {TILE_V} to prevent out-of-bounds access, got V={V}"
    )

    if scale is None:
        scale = K**-0.5

    h0_source = state.view(B * HV, V, K)
    cache_key = (
        B,
        T,
        H,
        HV,
        K,
        V,
        q.dtype,
        scale,
        num_blocks_per_state,
    )
    cache = _get_compiled_decode_kernel_contest_persistent(*cache_key)

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(
            run_gdn_decode_kernel_large_batch_pretranspose_vendored_persistent,
            from_dlpack(h0_source, assumed_align=16),
            from_dlpack(A_log, assumed_align=16),
            from_dlpack(a, assumed_align=16),
            from_dlpack(dt_bias, assumed_align=16),
            from_dlpack(q, assumed_align=16),
            from_dlpack(k, assumed_align=16),
            from_dlpack(v, assumed_align=16),
            from_dlpack(b, assumed_align=16),
            from_dlpack(output, assumed_align=16),
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=scale,
            HV=HV,
            B=B,
            T=T,
            H=H,
            K=K,
            V=V,
            use_qk_l2norm=False,
            num_blocks_per_state=num_blocks_per_state,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cache["compiled"](h0_source, A_log, a, dt_bias, q, k, v, b, output, stream)


def run_pretranspose_decode_contest_fastpath_large_v16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    scale: float | None = None,
    num_blocks_per_state: int = DEFAULT_NUM_BLOCKS_PER_STATE,
) -> None:
    """
    Contest-only large-batch path with a wider V tile (`16x128`).
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape

    assert T == 1, f"Decode only supports T=1, got T={T}"
    assert state.shape == (B, HV, V, K), (
        f"Expected state shape [B={B}, HV={HV}, V={V}, K={K}], got {state.shape}"
    )
    assert state.dtype == torch.float32, f"state must be float32, got {state.dtype}"
    assert state.is_contiguous(), "contest fast path requires contiguous state"
    assert output is not None, "contest fast path requires preallocated output"
    assert output.shape == (B, T, HV, V), (
        f"Expected output shape [B={B}, T={T}, HV={HV}, V={V}], got {output.shape}"
    )
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    assert V % TILE_V_LARGE == 0, (
        f"V must be divisible by {TILE_V_LARGE} for the large-v16 path, got V={V}"
    )

    if scale is None:
        scale = K**-0.5

    h0_source = state.view(B * HV, V, K)
    cache_key = (
        B,
        T,
        H,
        HV,
        K,
        V,
        q.dtype,
        scale,
        num_blocks_per_state,
    )
    cache = _get_compiled_decode_kernel_contest_large_v16(*cache_key)

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(
            run_gdn_decode_kernel_large_v16_pretranspose_vendored,
            from_dlpack(h0_source, assumed_align=16),
            from_dlpack(A_log, assumed_align=16),
            from_dlpack(a, assumed_align=16),
            from_dlpack(dt_bias, assumed_align=16),
            from_dlpack(q, assumed_align=16),
            from_dlpack(k, assumed_align=16),
            from_dlpack(v, assumed_align=16),
            from_dlpack(b, assumed_align=16),
            from_dlpack(output, assumed_align=16),
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=scale,
            HV=HV,
            B=B,
            T=T,
            H=H,
            K=K,
            V=V,
            use_qk_l2norm=False,
            num_blocks_per_state=num_blocks_per_state,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cache["compiled"](h0_source, A_log, a, dt_bias, q, k, v, b, output, stream)


def run_pretranspose_decode_contest_fastpath(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    scale: float | None = None,
    num_blocks_per_state: int = DEFAULT_NUM_BLOCKS_PER_STATE,
    use_qk_output_shortcut: bool = False,
) -> None:
    """
    Contest-only fast path for the active decode hybrid.

    Fixed contract:
    - `T == 1`
    - `state.dtype == torch.float32`
    - `state` is contiguous with layout `[B, HV, V, K]`
    - `output` is preallocated by the benchmark harness
    - `use_qk_l2norm=False`

    This avoids the generic wrapper branches that are not exercised by the
    contest harness and keeps the large-batch path focused on the hot contract
    that the official decode evaluation actually uses.
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape

    assert T == 1, f"Decode only supports T=1, got T={T}"
    assert state.shape == (B, HV, V, K), (
        f"Expected state shape [B={B}, HV={HV}, V={V}, K={K}], got {state.shape}"
    )
    assert state.dtype == torch.float32, f"state must be float32, got {state.dtype}"
    assert state.is_contiguous(), "contest fast path requires contiguous state"
    assert output is not None, "contest fast path requires preallocated output"
    assert output.shape == (B, T, HV, V), (
        f"Expected output shape [B={B}, T={T}, HV={HV}, V={V}], got {output.shape}"
    )
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    assert V % TILE_V == 0, (
        f"V must be divisible by {TILE_V} to prevent out-of-bounds access, got V={V}"
    )

    if scale is None:
        scale = K**-0.5

    h0_source = state.view(B * HV, V, K)
    cache_key = (
        B,
        T,
        H,
        HV,
        K,
        V,
        q.dtype,
        scale,
        num_blocks_per_state,
        use_qk_output_shortcut,
    )
    cache = _get_compiled_decode_kernel_contest_fastpath(*cache_key)

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(
            run_gdn_decode_kernel_small_batch_pretranspose_vendored,
            from_dlpack(h0_source, assumed_align=16),
            from_dlpack(A_log, assumed_align=16),
            from_dlpack(a, assumed_align=16),
            from_dlpack(dt_bias, assumed_align=16),
            from_dlpack(q, assumed_align=16),
            from_dlpack(k, assumed_align=16),
            from_dlpack(v, assumed_align=16),
            from_dlpack(b, assumed_align=16),
            from_dlpack(output, assumed_align=16),
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=scale,
            HV=HV,
            B=B,
            T=T,
            H=H,
            K=K,
            V=V,
            use_qk_l2norm=False,
            num_blocks_per_state=num_blocks_per_state,
            use_qk_output_shortcut=use_qk_output_shortcut,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cache["compiled"](h0_source, A_log, a, dt_bias, q, k, v, b, output, stream)


def gated_delta_rule_decode_pretranspose_vendored(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: float | None = None,
    output: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
    num_blocks_per_state: int = DEFAULT_NUM_BLOCKS_PER_STATE,
    use_qk_output_shortcut: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Contest-specialized vendored FlashInfer pretranspose decode entry point.

    Supported contract:
    - `T == 1`
    - `state.dtype == torch.float32`
    - `state.shape == [B, HV, V, K]`

    This compatibility wrapper is kept for completeness, but the active decode
    dispatch uses `run_pretranspose_decode_contest_fastpath()` directly.
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape

    assert T == 1, f"Decode only supports T=1, got T={T}"
    assert state.shape == (B, HV, V, K), (
        f"Expected state shape [B={B}, HV={HV}, V={V}, K={K}], got {state.shape}"
    )
    assert state.dtype == torch.float32, f"state must be float32, got {state.dtype}"
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    assert V % TILE_V == 0, (
        f"V must be divisible by {TILE_V} to prevent out-of-bounds access, got V={V}"
    )

    if scale is None:
        scale = K**-0.5

    output_provided = output is not None
    target_dtype = output.dtype if output_provided else q.dtype
    if output is None:
        output = torch.zeros((B, T, HV, V), dtype=torch.bfloat16, device=q.device)

    state_contiguous = state.contiguous()
    h0_source = state_contiguous.view(B * HV, V, K)

    run_pretranspose_decode_vendored(
        h0_source,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        B,
        T,
        H,
        HV,
        K,
        V,
        scale,
        use_qk_l2norm,
        num_blocks_per_state,
        use_qk_output_shortcut,
    )

    if state_contiguous.data_ptr() != state.data_ptr():
        state.copy_(state_contiguous)

    if output.dtype != target_dtype:
        output = output.to(target_dtype)

    return output, state
