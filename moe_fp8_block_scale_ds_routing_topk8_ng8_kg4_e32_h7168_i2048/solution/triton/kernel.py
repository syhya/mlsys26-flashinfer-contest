"""Thin entrypoint for submission candidate `moe-submission-v30-compact-long-route`.

The submission surface is intentionally flat so the contest packer can include
every required source file from `solution/triton/` without relying on nested
packages. `kernel.py` only owns backend selection and forwards execution into
the matching helper module.

The default runtime carries the retained expert-major MoE pipeline plus current
CUDA routing updates inside `moe_reference_runtime.cu`: compact-row FP8
permute-copy, warp-row activation quantization, and dynamic row reduction for
the post-GEMM2 combine step. `moe_reference_backend.py` also carries narrow
SM100 fused-epilogue candidates gated only to shapes that passed the B200
benchmark gate.
The retained Triton fastpath is still available as an optional backend, but the
dispatch policy keeps it disabled until a same-round full sweep proves a real
global gain.
"""

from functools import lru_cache

import torch

from .moe_dispatch_policy import should_use_current_triton_path


@lru_cache(maxsize=1)
def _load_current_triton_path():
    # Delay backend imports so the entrypoint stays light and the reference
    # runtime is not compiled unless a workload actually selects that path.
    from . import moe_triton_backend

    return moe_triton_backend


@lru_cache(maxsize=1)
def _load_reference_path():
    # The reference backend owns the external CUDA helpers, so keep its import
    # lazy as well. This avoids paying the extension-build cost during module
    # import when a future dispatch policy routes everything to Triton.
    from . import moe_reference_backend

    return moe_reference_backend


@torch.no_grad()
def run(
    routing_logits, routing_bias,
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
    output,
) -> None:
    seq_len = hidden_states.shape[0]
    if should_use_current_triton_path(seq_len):
        # The retained Triton backend is kept behind a policy gate so we can
        # re-enable exact-shape fastpaths only after a same-round full sweep
        # proves that they improve the global average.
        current_triton = _load_current_triton_path()
        current_triton.run_impl(
            routing_logits, routing_bias,
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor,
            output,
        )
        return

    # Use the default backend for the compact expert-major workspace and the
    # retained CUDA routing helpers.
    reference_path = _load_reference_path()
    reference_path.fused_moe(
        routing_logits, routing_bias,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor,
        output=output,
    )
