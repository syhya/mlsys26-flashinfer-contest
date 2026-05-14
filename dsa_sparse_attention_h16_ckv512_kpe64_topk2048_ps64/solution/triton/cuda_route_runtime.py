"""Runtime loader for the split-WMMA CUDA route used by the hybrid dispatcher."""

import os
from functools import lru_cache
from pathlib import Path

import torch
import tvm_ffi


_CUDA_SRC = str(Path(__file__).with_name("cuda_route_kernel.cu"))
_CUDA_ARCH = os.environ.get("FLASHINFER_CUDA_ARCH", "sm_100a")


@lru_cache(maxsize=1)
def _get_mod():
    lib_path = tvm_ffi.cpp.build(
        name="dsa_sparse_attn_cuda_route",
        cuda_files=[_CUDA_SRC],
        # Keep the validation build aligned with the intended submission target.
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            f"-arch={_CUDA_ARCH}",
        ],
    )
    return tvm_ffi.load_module(lib_path)


def _to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.item()) if x.numel() > 0 else 0.0
    return float(x)


def _call_cuda(
    symbol: str, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
):
    fn = getattr(_get_mod(), symbol)
    fn(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        _to_float(sm_scale),
        output,
        lse,
    )


def kernel_cuda(
    q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
):
    _call_cuda(
        "kernel_cuda",
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse,
    )


def kernel_cuda_splits16(
    q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
):
    _call_cuda(
        "kernel_cuda_splits16",
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse,
    )
