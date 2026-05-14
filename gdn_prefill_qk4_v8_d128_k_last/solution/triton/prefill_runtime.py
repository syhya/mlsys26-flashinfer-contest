"""CUDA build and sidecar helpers for the active contest GDN wrapper.

Upstream PR #3001 expects ``gate`` and ``beta`` tensors as direct inputs to the
Blackwell chunk kernel. The contest definition instead passes the raw gate
parameters ``a``, ``b``, ``A_log`` and ``dt_bias``. This file keeps that
precompute step in one small CUDA sidecar and also exposes the recovered tiny
short-shape recurrent kernel entrypoint.

The helper returns ``log_gate`` in natural-log space:
``log_gate = -exp(A_log) * softplus(a + dt_bias)``.
"""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def build_recurrent_cuda():
    """Compile and load the CUDA sidecar once per process."""
    import tvm_ffi

    cu = str(Path(__file__).parent / "recurrent_kernel.cu")
    lib_path = tvm_ffi.cpp.build(
        name="gdn_prefill_runtime",
        cuda_files=[cu],
        extra_cuda_cflags=["--use_fast_math"],
    )
    return tvm_ffi.load_module(lib_path)


@torch.no_grad()
def compute_contest_gates(a, b, A_log, dt_bias):
    """Compute ``log_gate`` and ``beta`` with the scalar CUDA helper.

    Returns:
        ``(lib, log_gate, beta)`` where ``log_gate`` and ``beta`` both have
        shape ``(1, total_seq_len, num_heads)`` in float32.
    """
    lib = build_recurrent_cuda()
    total_seq_len, num_heads = a.shape
    log_gate = torch.empty(
        (1, total_seq_len, num_heads), device=a.device, dtype=torch.float32
    )
    beta = torch.empty((1, total_seq_len, num_heads), device=a.device, dtype=torch.float32)
    lib.compute_gates(a, b, A_log, dt_bias, log_gate.squeeze(0), beta.squeeze(0))
    return lib, log_gate, beta


@torch.no_grad()
def run_contest_short_prefill(
    q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state
):
    """Run the recovered submission-v25 tiny-shape recurrent CUDA kernel."""
    lib = build_recurrent_cuda()
    lib.kernel_cuda_prefill_trace_v1(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state
    )
