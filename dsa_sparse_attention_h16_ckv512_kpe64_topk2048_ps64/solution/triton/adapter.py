"""Minimal adapter around vendored FlashInfer modular MLA decode.

This bypasses the upstream wrapper's H<128 runtime guard and fixes
``split_kv=1`` for the contest shape ``H=16, q_len=1``.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from .attention_config import AttentionFusion

from .attention_fusion_variant import StandardAttention

from .attention_mla_config import MLAConfig

from .attention_mla_decode import BlackwellMultiLatentAttentionForward


def _torch_to_cutlass_dtype(dtype: torch.dtype):
    dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    if dtype not in dtype_map:
        raise TypeError(f"Unsupported torch dtype for upstream MLA adapter: {dtype}")
    return dtype_map[dtype]


@functools.cache
def _get_num_sm(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _get_max_active_clusters(cluster_size: int, device_index: int) -> int:
    try:
        return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)
    except Exception:
        return _get_num_sm(device_index)


def _make_config(
    page_size: int,
    enable_pdl: bool,
    device_index: int,
) -> MLAConfig:
    cluster_shape_mnk = (2, 1, 1)
    return MLAConfig(
        latent_dim=512,
        rope_dim=64,
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        max_active_clusters=_get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1], device_index
        ),
        page_size=page_size,
        skip_correction_threshold=0.0,
        is_persistent=True,
        is_var_seq=False,
        is_var_split_kv=False,
        enable_pdl=enable_pdl,
        is_fp8=False,
        mma_o_stage=1,
    )


def _make_fake_tensors(cutlass_dtype, cutlass_out_dtype, page_size: int):
    sym_heads = cute.sym_int()
    sym_batch = cute.sym_int()
    sym_seq_q = cute.sym_int()
    sym_page_count = cute.sym_int()

    q_latent_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, 512),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    q_rope_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, 64),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    c_latent_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (cute.sym_int(), page_size, 512),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    c_rope_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (cute.sym_int(), page_size, 64),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    page_table_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch, sym_page_count),
        stride_order=(1, 0),
        assumed_align=16,
    )
    o_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_out_dtype,
        (sym_batch, sym_seq_q, sym_heads, 512),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    lse_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_batch, sym_seq_q, sym_heads),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return (
        q_latent_fake,
        q_rope_fake,
        c_latent_fake,
        c_rope_fake,
        page_table_fake,
        o_fake,
        lse_fake,
        cache_seqs_fake,
        stream_fake,
    )


@functools.cache
def _compile_kernel(
    torch_dtype: torch.dtype,
    torch_out_dtype: torch.dtype,
    page_size: int,
    enable_pdl: bool,
    device_index: int,
):
    cutlass_dtype = _torch_to_cutlass_dtype(torch_dtype)
    cutlass_out_dtype = _torch_to_cutlass_dtype(torch_out_dtype)

    if not BlackwellMultiLatentAttentionForward.can_implement(
        1,
        1,
        1,
        16,
        512,
        64,
        cutlass_dtype,
        cutlass_out_dtype,
        cutlass.Float32,
        cutlass.Float32,
        (128, 128),
        (128, 256),
        1,
        True,
        False,
        False,
        page_size,
    ):
        raise ValueError(
            "Vendored upstream MLA decode cannot implement this contest shape"
        )

    kernel_obj = BlackwellMultiLatentAttentionForward(
        _make_config(page_size, enable_pdl, device_index),
        fusion=AttentionFusion(variant=StandardAttention()),
    )
    (
        q_latent_fake,
        q_rope_fake,
        c_latent_fake,
        c_rope_fake,
        page_table_fake,
        o_fake,
        lse_fake,
        cache_seqs_fake,
        stream_fake,
    ) = _make_fake_tensors(cutlass_dtype, cutlass_out_dtype, page_size)

    return cute.compile(
        kernel_obj,
        q_latent_fake,
        q_rope_fake,
        c_latent_fake,
        c_rope_fake,
        page_table_fake,
        o_fake,
        lse_fake,
        None,  # workspace
        Int32(1),  # split_kv fixed for H=16
        cache_seqs_fake,
        None,  # block_split_kvs
        Float32(1.0),  # softmax_scale placeholder
        Float32(1.0),  # output_scale placeholder
        None,  # params_in
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


@torch.no_grad()
def run_upstream_mla_decode(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    c_latent: torch.Tensor,
    c_rope: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqs: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    if q_latent.dtype != torch.bfloat16 or q_rope.dtype != torch.bfloat16:
        raise TypeError("This adapter only supports bf16 contest inputs")
    if q_latent.shape[1] != 1 or q_rope.shape[1] != 1:
        raise ValueError("This adapter expects q_len=1 per request")

    device_index = q_latent.device.index
    compiled = _compile_kernel(
        q_latent.dtype,
        out.dtype,
        int(c_latent.shape[1]),
        True,
        device_index,
    )
    compiled(
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_table,
        out,
        lse,
        None,
        Int32(1),
        cache_seqs,
        None,
        Float32(float(softmax_scale)),
        Float32(1.0),
        None,
    )
