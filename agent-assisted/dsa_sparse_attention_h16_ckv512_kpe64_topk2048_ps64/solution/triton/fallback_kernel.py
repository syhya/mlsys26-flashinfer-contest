# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications Copyright (c) 2026 yue-shui.
"""Fallback dispatcher for short-route and split-WMMA CUDA kernels."""

from functools import lru_cache

from .cuda_route_runtime import kernel_cuda as _run_cuda_route


_SHORT_ROUTE_MAX_T = 2


@lru_cache(maxsize=1)
def _get_short_route():
    from . import short_route

    return short_route


def _run_short_route(
    q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
):
    _get_short_route().run(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
        output,
        lse,
    )


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens = int(q_nope.shape[0])
    if num_tokens <= _SHORT_ROUTE_MAX_T:
        _run_short_route(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
        )
        return

    _run_cuda_route(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
    )
