"""This module keeps the retained end-to-end pipeline structure:
- fused route + permute + copy into a compact expert-major workspace
- Triton AOT GEMM kernels for GEMM1/GEMM2
- CUDA helpers for routing, quantization, and final reduction
- narrow SM100 fused GEMM1/finalize epilogues with shape gates based on B200
  benchmark results

The original inline CUDA/C++ source has been moved into standalone runtime
files so the layout matches the mixed Python/CUDA submission style used by
other contest entries.
"""

from dataclasses import dataclass
from typing import List, Sequence

import torch
import triton
import triton.language as tl

from .moe_reference_runtime import get_moe_runtime_lib

@dataclass
class GemmConfig:
    block_size_m: int
    block_size_n: int
    block_size_k: int
    num_stages: int
    use_tma: bool
    swap_ab: bool = False


def get_blk_size_m(args):
    if args["seq_len"] <= 100:
        return 16
    elif args["seq_len"] <= 500:
        return 32
    elif args["seq_len"] <= 2000:
        return 64
    elif args["seq_len"] <= 20000:
        return 128
    else:
        return 256


@triton.jit
def gemm1_kernel(
    # input
    a_base_ptr, # [sum(s_i), 7168], fp8
    a_scale_base_ptr, # [7168//128, sum(s_i)], fp32
    a_offset_ptr, # [33], int32
    b_base_ptr, # [32, 4096, 7168], fp8
    b_scale_base_ptr, # [32, 4096//128, 7168//128], fp32
    seq_len,
    # output
    c_base_ptr, # [sum(s_i), 4096], fp32
    c_scale_base_ptr, # [2048//128, sum(s_i)], fp32
    # other
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TMA: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_gemm_end_tile_idx = 0

    # Iterate over the 32 expert-specific GEMM problems.
    for i in range(32):
        offset = tl.load(a_offset_ptr + i)
        gm = tl.load(a_offset_ptr + i + 1) - offset # s_i
        gn = 4096
        gk = 7168
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N) # Tile the output width for the current expert buffer.
        num_tiles = (num_m_tiles * num_n_tiles).to(tl.int32)

        # Check whether the current persistent tile id belongs to this expert.
        if tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles:
            lda = 7168
            ldb = 7168
            ldc = 4096

            a_ptr = a_base_ptr + offset * lda
            b_ptr = b_base_ptr + gn * gk * i
            c_ptr = c_base_ptr + offset * ldc

            a_scale_ptr = a_scale_base_ptr + offset
            b_scale_ptr = b_scale_base_ptr + gn // 128 * gk // 128 * i

            if USE_TMA:
                a_desc = tl.make_tensor_descriptor(
                    a_ptr,
                    shape=[gm, gk],
                    strides=[lda, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                )
            b1_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            if USE_TMA:
                c_desc = tl.make_tensor_descriptor(
                    c_ptr,
                    shape=[gm, gn],
                    strides=[ldc, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # Continue consuming tiles from the current expert in a persistent loop.
            while (tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles):
                k = gk
                tile_idx_in_gemm = tile_idx - last_gemm_end_tile_idx
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn1 = tile_n_idx * BLOCK_SIZE_N

                # Count FP8 scale groups along the K dimension.
                num_k_blocks = tl.cdiv(k, 128)

                if SWAP_AB:
                    acc1 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
                else:
                    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in tl.range(0, tl.cdiv(k, BLOCK_SIZE_K), num_stages=6, warp_specialize=True):
                    if USE_TMA:
                        a = a_desc.load(
                            [offs_am, kk * BLOCK_SIZE_K],
                        )
                    else:
                        a_idx = a_ptr + offs_am * lda + kk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_M)[:, None] * lda + tl.arange(0, BLOCK_SIZE_K)[None, :]
                        a = tl.load(a_idx, mask=((offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm)[:, None]) # [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    b1 = b1_desc.load(
                        [offs_bn1, kk * BLOCK_SIZE_K],
                    )

                    # Load the FP8 dequant scales for the current K block.
                    a_scale_idx = a_scale_ptr + offs_am + kk * seq_len * 8 + tl.arange(0, BLOCK_SIZE_M)
                    a_scale_vec = tl.load(a_scale_idx, mask=(offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm, other=1.0)
                    if SWAP_AB:
                        a_scale_col = tl.reshape(a_scale_vec, (1, BLOCK_SIZE_M))
                    else:
                        a_scale_col = tl.reshape(a_scale_vec, (BLOCK_SIZE_M, 1))
                    if BLOCK_SIZE_N == 128:
                        b1_scale = tl.load(b_scale_ptr + (offs_bn1 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128))
                        if SWAP_AB:
                            acc1 += tl.dot(b1, a.T) * a_scale_col * b1_scale
                        else:
                            acc1 += tl.dot(a, b1.T) * a_scale_col * b1_scale
                    else:
                        b1_scale = tl.load(b_scale_ptr + (offs_bn1 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128))
                        b1_scale2 = tl.load(b_scale_ptr + ((offs_bn1 + 128) // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128))
                        scales = tl.join(b1_scale, b1_scale2)
                        b_scale_vec = tl.reshape(scales, (2, 1))
                        scales_expanded = tl.broadcast_to(b_scale_vec, (2, 128))
                        if SWAP_AB:
                            final_scales = tl.reshape(scales_expanded, (256, 1))
                            acc1 += tl.dot(b1, a.T) * a_scale_col * final_scales
                        else:
                            final_scales = tl.reshape(scales_expanded, (1, 256))
                            acc1 += tl.dot(a, b1.T) * a_scale_col * final_scales

                # Store the final accumulator tile.
                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N
                if USE_TMA:
                    c_desc.store(
                        [offs_cm, offs_cn],
                        acc1.to(tl.float16).T if SWAP_AB else acc1.to(tl.float16),
                    )
                else:
                    c_idx = c_ptr + offs_cm * ldc + offs_cn + tl.arange(0, BLOCK_SIZE_M)[:, None] * ldc + tl.arange(0, BLOCK_SIZE_N)[None, :]
                    tl.store(c_idx, acc1.to(tl.float16).T if SWAP_AB else acc1.to(tl.float16), mask=((offs_cm + tl.arange(0, BLOCK_SIZE_M)) < gm)[:, None])

                # Advance to the next tile assigned to this persistent program.
                tile_idx += NUM_SM

        # Move to the next expert-specific GEMM problem.
        last_gemm_end_tile_idx = last_gemm_end_tile_idx + num_tiles

def launch_gemm1_kernel(
    permute_hidden_states: torch.tensor, # [sum(s), 7168], fp8
    permute_hidden_states_scale: torch.tensor, # [7168//128, sum(s)], fp32
    offset: torch.tensor, # [33], int32, exclusive row-prefix per expert; the last element is sum(s).
    gemm1_weights: torch.tensor, # [32, 4096, 7168], fp8
    gemm1_weights_scale: torch.tensor, # [32, 4096//128, 7168//128], fp32
    seq_len,
    output: torch.tensor, # [sum(s), 2048], fp32
    output_scale: torch.Tensor, # [2048//128, sum(s)], fp32
    *,
    block_size_m: int = 32,
    block_size_n: int = 128,
    block_size_k: int = 128,
    num_sm: int = 160,
) -> List[torch.Tensor]:
    grid = (num_sm,)

    gemm1_kernel[grid](
        # input
        a_base_ptr=permute_hidden_states,
        a_scale_base_ptr=permute_hidden_states_scale,
        a_offset_ptr=offset,
        b_base_ptr=gemm1_weights,
        b_scale_base_ptr=gemm1_weights_scale,
        seq_len=seq_len,
        # output
        c_base_ptr=output,
        c_scale_base_ptr=output_scale,
        NUM_SM=num_sm,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        num_stages=4,
        num_warps=8,
    )
    return

# The compact workspace stores 32 expert segments: each segment length and its
# routed token ids.
def gemm1(
    permute_hidden_states,
    permute_hidden_states_scale,
    offset,
    seq_len,
    gemm1_weights,
    gemm1_weights_scale,
    output,
    output_scale,
    num_sm=160,
):

    launch_gemm1_kernel(
        permute_hidden_states,
        permute_hidden_states_scale,
        offset,
        gemm1_weights,
        gemm1_weights_scale,
        seq_len,
        output,
        output_scale,
        block_size_m=32,
        block_size_n=128,
        block_size_k=128,
        num_sm=num_sm,
    )

    return



from triton.compiler import ASTSource, compile


@triton.jit
def gemm1_swiglu_quant_kernel(
    # input
    a_base_ptr,  # [sum(s_i), 7168], fp8
    a_scale_base_ptr,  # [7168//128, sum(s_i)], fp32
    a_offset_ptr,  # [33], int32
    b_base_ptr,  # [32, 4096, 7168], fp8
    b_scale_base_ptr,  # [32, 4096//128, 7168//128], fp32
    seq_len,
    # output
    c_base_ptr,  # [sum(s_i), 2048], fp8
    c_scale_base_ptr,  # [2048//128, sum(s_i)], fp32
    # other
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """GEMM1 epilogue fused with SwiGLU and FP8 block quantization.

    This keeps the existing expert-major packed A layout, borrowing the SM100
    fusion point from FlashInfer's CuTe DSL gather+SwiGLU kernel without
    switching the contest input contract to NVFP4.
    """
    tile_idx = tl.program_id(0)
    last_gemm_end_tile_idx = 0

    for i in range(32):
        offset = tl.load(a_offset_ptr + i)
        gm = tl.load(a_offset_ptr + i + 1) - offset
        gn: tl.constexpr = 2048
        gk: tl.constexpr = 7168
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles: tl.constexpr = gn // BLOCK_SIZE_N
        num_tiles = (num_m_tiles * num_n_tiles).to(tl.int32)

        if tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles:
            lda: tl.constexpr = 7168
            ldb: tl.constexpr = 7168
            ldc: tl.constexpr = 2048

            a_ptr = a_base_ptr + offset * lda
            b_ptr = b_base_ptr + (2 * gn) * gk * i
            c_ptr = c_base_ptr + offset * ldc

            a_scale_ptr = a_scale_base_ptr + offset
            b_scale_ptr = b_scale_base_ptr + ((2 * gn) // 128) * (gk // 128) * i

            if USE_TMA:
                a_desc = tl.make_tensor_descriptor(
                    a_ptr,
                    shape=[gm, gk],
                    strides=[lda, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                )

            b_linear_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[2 * gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            b_gate_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[2 * gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )

            while tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles:
                tile_idx_in_gemm = tile_idx - last_gemm_end_tile_idx
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N
                num_k_blocks: tl.constexpr = gk // 128

                acc_linear = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                for kk in tl.range(0, tl.cdiv(gk, BLOCK_SIZE_K), num_stages=6, warp_specialize=True):
                    if USE_TMA:
                        a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    else:
                        a_idx = (
                            a_ptr
                            + offs_am * lda
                            + kk * BLOCK_SIZE_K
                            + tl.arange(0, BLOCK_SIZE_M)[:, None] * lda
                            + tl.arange(0, BLOCK_SIZE_K)[None, :]
                        )
                        a = tl.load(
                            a_idx,
                            mask=((offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm)[:, None],
                            other=0.0,
                        )

                    b_linear = b_linear_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                    b_gate = b_gate_desc.load([gn + offs_bn, kk * BLOCK_SIZE_K])

                    a_scale_idx = a_scale_ptr + offs_am + kk * seq_len * 8 + tl.arange(0, BLOCK_SIZE_M)
                    a_scale_vec = tl.load(
                        a_scale_idx,
                        mask=(offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm,
                        other=1.0,
                    )
                    a_scale_col = tl.reshape(a_scale_vec, (BLOCK_SIZE_M, 1))

                    linear_scale = tl.load(
                        b_scale_ptr
                        + (offs_bn // 128) * num_k_blocks
                        + (kk * BLOCK_SIZE_K // 128)
                    )
                    gate_scale = tl.load(
                        b_scale_ptr
                        + ((gn + offs_bn) // 128) * num_k_blocks
                        + (kk * BLOCK_SIZE_K // 128)
                    )

                    acc_linear += tl.dot(a, b_linear.T) * a_scale_col * linear_scale
                    acc_gate += tl.dot(a, b_gate.T) * a_scale_col * gate_scale

                swiglu = acc_linear * (acc_gate * tl.sigmoid(acc_gate))
                max_abs = tl.max(tl.abs(swiglu), axis=1)
                group_scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)

                row_offsets = offs_am + tl.arange(0, BLOCK_SIZE_M)
                col_offsets = offs_bn + tl.arange(0, BLOCK_SIZE_N)
                row_mask = row_offsets < gm
                scale_stride = seq_len * 8
                tl.store(
                    c_scale_base_ptr + tile_n_idx * scale_stride + offset + row_offsets,
                    group_scale,
                    mask=row_mask,
                )

                q = swiglu / tl.reshape(group_scale, (BLOCK_SIZE_M, 1))
                q = tl.minimum(tl.maximum(q, -448.0), 448.0)
                c_idx = (
                    c_ptr
                    + row_offsets[:, None] * ldc
                    + col_offsets[None, :]
                )
                tl.store(c_idx, q.to(tl.float8e4nv), mask=row_mask[:, None])

                tile_idx += NUM_SM

        last_gemm_end_tile_idx = last_gemm_end_tile_idx + num_tiles


class gemm1Kernel:
    def __init__(self, num_sm):
        signature = {
            'a_base_ptr': '*fp8e4nv',
            'a_scale_base_ptr': '*fp32',
            'a_offset_ptr': '*i32',
            'b_base_ptr': '*fp8e4nv',
            'b_scale_base_ptr': '*fp32',
            'seq_len': 'i32',
            'c_base_ptr': '*fp16',
            'c_scale_base_ptr': '*fp32',
            'NUM_SM': 'constexpr',
            'BLOCK_SIZE_M': 'constexpr',
            'BLOCK_SIZE_N': 'constexpr',
            'BLOCK_SIZE_K': 'constexpr',
            'USE_TMA': 'constexpr',
            'SWAP_AB': 'constexpr',
        }

        # blk_sizes = [64, 128]
        # num_stages = [8, 6]
        self.configs = []
        self.configs.append(GemmConfig(block_size_m=64, block_size_n=128, block_size_k=128, num_stages=8, use_tma=False, swap_ab=False))
        self.configs.append(GemmConfig(block_size_m=64, block_size_n=256, block_size_k=128, num_stages=4, use_tma=False, swap_ab=False))
        self.configs.append(GemmConfig(block_size_m=128, block_size_n=256, block_size_k=128, num_stages=3, use_tma=True, swap_ab=False))
        constexprs_list = []
        options_list = []
        for i in range(len(self.configs)):
            constexprs_list.append({
                (8,): num_sm,
                (9,): self.configs[i].block_size_m,
                (10,): self.configs[i].block_size_n,
                (11,): self.configs[i].block_size_k,
                (12,): self.configs[i].use_tma,
                (13,): self.configs[i].swap_ab
            })
            options_list.append({
                "num_warps": 8,
                "num_stages": self.configs[i].num_stages,
            })

        attrs = {
            (0,): [['tt.divisibility', 16]],
            (1,): [['tt.divisibility', 16]],
            (2,): [['tt.divisibility', 16]],
            (3,): [['tt.divisibility', 16]],
            (4,): [['tt.divisibility', 16]],
            (5,): [['tt.divisibility', 16]],
            (6,): [['tt.divisibility', 16]],
            (7,): [['tt.divisibility', 16]]
        }

        self.num_sm = num_sm
        self.kernels = []
        for i in range(len(self.configs)):
            src = ASTSource(
                fn=gemm1_kernel,
                signature=signature,
                constexprs=constexprs_list[i],
                attrs=attrs
            )
            compiled_kernel = compile(src, options=options_list[i])
            self.kernels.append(compiled_kernel)

    def __call__(
            self,
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            seq_len,
            c_base,
            c_scale_base,
            stream=None):
        if stream is None:
            device = triton.runtime.driver.active.get_current_device()
            stream = triton.runtime.driver.active.get_current_stream(device)
        elif hasattr(stream, "cuda_stream"):
            # Accept either a raw CUDA stream pointer or a torch stream wrapper.
            stream = stream.cuda_stream

        kernel = None
        config = None
        if seq_len <= 4:
            kernel = self.kernels[0]
            config = self.configs[0]
        elif seq_len <= 1000:
            kernel = self.kernels[1]
            config = self.configs[1]
        else:
            kernel = self.kernels[2]
            config = self.configs[2]

        grid = (self.num_sm, 1, 1)
        launch_metadata = kernel.launch_metadata(grid, stream, a_base, a_scale_base, a_offset, b_base, b_scale_base, seq_len, c_base, c_scale_base)

        kernel.run(
            grid[0], grid[1], grid[2],
            stream,
            kernel.function,
            kernel.packed_metadata,
            launch_metadata,
            None,
            None,
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            seq_len,
            c_base,
            c_scale_base,
            self.num_sm,
            config.block_size_m,
            config.block_size_n,
            config.block_size_k,
            config.use_tma,
            config.swap_ab
        )
        return


class gemm1SwigluQuantKernel:
    def __init__(self, num_sm):
        signature = {
            'a_base_ptr': '*fp8e4nv',
            'a_scale_base_ptr': '*fp32',
            'a_offset_ptr': '*i32',
            'b_base_ptr': '*fp8e4nv',
            'b_scale_base_ptr': '*fp32',
            'seq_len': 'i32',
            'c_base_ptr': '*fp8e4nv',
            'c_scale_base_ptr': '*fp32',
            'NUM_SM': 'constexpr',
            'BLOCK_SIZE_M': 'constexpr',
            'BLOCK_SIZE_N': 'constexpr',
            'BLOCK_SIZE_K': 'constexpr',
            'USE_TMA': 'constexpr',
        }

        self.configs = [
            GemmConfig(block_size_m=64, block_size_n=128, block_size_k=128, num_stages=6, use_tma=False),
            GemmConfig(block_size_m=64, block_size_n=128, block_size_k=128, num_stages=5, use_tma=False),
            GemmConfig(block_size_m=128, block_size_n=128, block_size_k=128, num_stages=4, use_tma=True),
        ]

        attrs = {
            (0,): [['tt.divisibility', 16]],
            (1,): [['tt.divisibility', 16]],
            (2,): [['tt.divisibility', 16]],
            (3,): [['tt.divisibility', 16]],
            (4,): [['tt.divisibility', 16]],
            (5,): [['tt.divisibility', 16]],
            (6,): [['tt.divisibility', 16]],
            (7,): [['tt.divisibility', 16]],
        }

        self.num_sm = num_sm
        self.kernels = []
        for config in self.configs:
            src = ASTSource(
                fn=gemm1_swiglu_quant_kernel,
                signature=signature,
                constexprs={
                    (8,): num_sm,
                    (9,): config.block_size_m,
                    (10,): config.block_size_n,
                    (11,): config.block_size_k,
                    (12,): config.use_tma,
                },
                attrs=attrs,
            )
            self.kernels.append(
                compile(src, options={"num_warps": 8, "num_stages": config.num_stages})
            )

    def __call__(
            self,
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            seq_len,
            c_base,
            c_scale_base,
            stream=None):
        if stream is None:
            device = triton.runtime.driver.active.get_current_device()
            stream = triton.runtime.driver.active.get_current_stream(device)
        elif hasattr(stream, "cuda_stream"):
            stream = stream.cuda_stream

        if seq_len <= 4:
            kernel = self.kernels[0]
            config = self.configs[0]
        elif seq_len <= 1000:
            kernel = self.kernels[1]
            config = self.configs[1]
        else:
            kernel = self.kernels[2]
            config = self.configs[2]

        grid = (self.num_sm, 1, 1)
        launch_metadata = kernel.launch_metadata(
            grid, stream, a_base, a_scale_base, a_offset, b_base, b_scale_base,
            seq_len, c_base, c_scale_base
        )

        kernel.run(
            grid[0], grid[1], grid[2],
            stream,
            kernel.function,
            kernel.packed_metadata,
            launch_metadata,
            None,
            None,
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            seq_len,
            c_base,
            c_scale_base,
            self.num_sm,
            config.block_size_m,
            config.block_size_n,
            config.block_size_k,
            config.use_tma,
        )
        return

@triton.jit
def gemm2_kernel(
    # input
    a_base_ptr,
    a_scale_base_ptr, # [2048//128, sum(s_i)], fp32
    a_offset_ptr,
    b_base_ptr,
    b_scale_base_ptr,
    permute_weights_base_ptr, # [group_size], fp32 -> [s_i]
    permute_token_idx_base_ptr, # [group_size], int32 -> [s_i]
    seq_len,
    #output
    output_ptr, # [sum(s_i), 7168], fp32
    # other
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TMA: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_gemm_end_tile_idx = 0

    # Iterate over the 32 expert-specific GEMM problems.
    for i in range(32):
        # get the gemm size of the current problem
        offset = tl.load(a_offset_ptr + i)
        gm = tl.load(a_offset_ptr + i + 1) - offset
        gn = 7168
        gk = 2048
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        # Check whether the current persistent tile id belongs to this expert.
        if tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles:
            lda = 2048
            ldb = 2048
            ldc = 7168

            a_ptr = a_base_ptr + offset * lda
            b_ptr = b_base_ptr + gn * gk * i
            c_ptr = output_ptr + offset * ldc

            a_scale_ptr = a_scale_base_ptr + offset
            b_scale_ptr = b_scale_base_ptr + gn // 128 * gk // 128 * i

            # Routing weights for the tokens assigned to the current expert.
            p_weight_ptr = permute_weights_base_ptr + offset
            # Original token ids for the current expert; used when scattering results back.
            p_source_idx_ptr = permute_token_idx_base_ptr + offset

            # TMA
            if USE_TMA:
                a_desc = tl.make_tensor_descriptor(
                    a_ptr,
                    shape=[gm, gk],
                    strides=[lda, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                )
            b1_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            if USE_TMA:
                c_desc = tl.make_tensor_descriptor(
                    c_ptr,
                    shape=[gm, gn],
                    strides=[ldc, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # Continue consuming tiles from the current expert in a persistent loop.
            while (tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles):
                k = gk
                tile_idx_in_gemm = tile_idx - last_gemm_end_tile_idx
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn1 = tile_n_idx * BLOCK_SIZE_N

                # Count FP8 scale groups along the K dimension.
                num_k_blocks = tl.cdiv(k, 128)
                if SWAP_AB:
                    acc1 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
                else:
                    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in tl.range(0, tl.cdiv(k, BLOCK_SIZE_K), num_stages=8, warp_specialize=True):
                    if USE_TMA:
                        a = a_desc.load(
                            [offs_am, kk * BLOCK_SIZE_K],
                        )
                    else:
                        a_idx = a_ptr + offs_am * lda + kk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_M)[:, None] * lda + tl.arange(0, BLOCK_SIZE_K)[None, :]
                        a = tl.load(a_idx, mask=((offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm)[:, None]) # [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    b1 = b1_desc.load(
                        [offs_bn1, kk * BLOCK_SIZE_K],
                    )

                    a_scale_idx = a_scale_ptr + offs_am + kk * seq_len * 8 + tl.arange(0, BLOCK_SIZE_M)
                    a_scale_vec = tl.load(a_scale_idx, mask=(offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm, other=1.0)
                    if SWAP_AB:
                        a_scale_col = tl.reshape(a_scale_vec, (1, BLOCK_SIZE_M))
                    else:
                        a_scale_col = tl.reshape(a_scale_vec, (BLOCK_SIZE_M, 1))
                    if BLOCK_SIZE_N == 128:
                        b1_scale = tl.load(b_scale_ptr + (offs_bn1 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128)) # One scale value for a single 128-column tile.
                        if SWAP_AB:
                            acc1 += tl.dot(b1, a.T) * a_scale_col * b1_scale
                        else:
                            acc1 += tl.dot(a, b1.T) * a_scale_col * b1_scale
                    else:
                        b1_scale = tl.load(b_scale_ptr + (offs_bn1 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128))
                        b1_scale2 = tl.load(b_scale_ptr + ((offs_bn1 + 128) // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128))
                        scales = tl.join(b1_scale, b1_scale2)
                        b_scale_vec = tl.reshape(scales, (2, 1))
                        scales_expanded = tl.broadcast_to(b_scale_vec, (2, 128))
                        if SWAP_AB:
                            final_scales = tl.reshape(scales_expanded, (256, 1))
                            acc1 += tl.dot(b1, a.T) * a_scale_col * final_scales
                        else:
                            final_scales = tl.reshape(scales_expanded, (1, 256))
                            acc1 += tl.dot(a, b1.T) * a_scale_col * final_scales


                # Apply the routing weights before the final writeback stage.
                offs_m = offs_am + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offs_m < gm
                tile_weights = tl.load(p_weight_ptr + offs_m, mask=mask_m, other=0.0)
                if SWAP_AB:
                    acc1 = acc1 * tl.reshape(tile_weights, (1, BLOCK_SIZE_M))
                else:
                    acc1 = acc1 * tl.reshape(tile_weights, (BLOCK_SIZE_M, 1))

                # Store the final accumulator tile into the contiguous scratch buffer.
                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N
                if USE_TMA:
                    c_desc.store(
                        [offs_cm, offs_cn],
                        acc1.to(tl.bfloat16).T if SWAP_AB else acc1.to(tl.bfloat16),
                    )
                else:
                    c_idx = c_ptr + offs_cm * ldc + offs_cn + tl.arange(0, BLOCK_SIZE_M)[:, None] * ldc + tl.arange(0, BLOCK_SIZE_N)[None, :]
                    tl.store(c_idx, acc1.to(tl.bfloat16).T if SWAP_AB else acc1.to(tl.bfloat16), mask=((offs_cm + tl.arange(0, BLOCK_SIZE_M)) < gm)[:, None])

                # Advance to the next tile assigned to this persistent program.
                tile_idx += NUM_SM

        # Move to the next expert-specific GEMM problem.
        last_gemm_end_tile_idx = last_gemm_end_tile_idx + num_tiles


@triton.jit
def gemm2_finalize_kernel(
    # input
    a_base_ptr,
    a_scale_base_ptr,  # [2048//128, sum(s_i)], fp32
    a_offset_ptr,
    b_base_ptr,
    b_scale_base_ptr,
    permute_weights_base_ptr,  # [group_size], fp32 -> [s_i]
    permute_token_idx_base_ptr,  # [group_size], int32 -> [s_i]
    seq_len,
    # output
    output_ptr,  # [seq_len, 7168], bf16
    # other
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TMA: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """GEMM2 epilogue fused with router scaling and final scatter-add."""
    tile_idx = tl.program_id(0)
    last_gemm_end_tile_idx = 0

    for i in range(32):
        offset = tl.load(a_offset_ptr + i)
        gm = tl.load(a_offset_ptr + i + 1) - offset
        gn = 7168
        gk = 2048
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        if tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles:
            lda = 2048
            ldb = 2048
            ldc = 7168

            a_ptr = a_base_ptr + offset * lda
            b_ptr = b_base_ptr + gn * gk * i

            a_scale_ptr = a_scale_base_ptr + offset
            b_scale_ptr = b_scale_base_ptr + gn // 128 * gk // 128 * i
            p_weight_ptr = permute_weights_base_ptr + offset
            p_source_idx_ptr = permute_token_idx_base_ptr + offset

            if USE_TMA:
                a_desc = tl.make_tensor_descriptor(
                    a_ptr,
                    shape=[gm, gk],
                    strides=[lda, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                )
            b1_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )

            while tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles:
                tile_idx_in_gemm = tile_idx - last_gemm_end_tile_idx
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn1 = tile_n_idx * BLOCK_SIZE_N
                num_k_blocks = tl.cdiv(gk, 128)

                if SWAP_AB:
                    acc1 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
                else:
                    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                for kk in tl.range(0, tl.cdiv(gk, BLOCK_SIZE_K), num_stages=8, warp_specialize=True):
                    if USE_TMA:
                        a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    else:
                        a_idx = (
                            a_ptr
                            + offs_am * lda
                            + kk * BLOCK_SIZE_K
                            + tl.arange(0, BLOCK_SIZE_M)[:, None] * lda
                            + tl.arange(0, BLOCK_SIZE_K)[None, :]
                        )
                        a = tl.load(
                            a_idx,
                            mask=((offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm)[:, None],
                            other=0.0,
                        )
                    b1 = b1_desc.load([offs_bn1, kk * BLOCK_SIZE_K])

                    a_scale_idx = a_scale_ptr + offs_am + kk * seq_len * 8 + tl.arange(0, BLOCK_SIZE_M)
                    a_scale_vec = tl.load(
                        a_scale_idx,
                        mask=(offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm,
                        other=1.0,
                    )
                    if SWAP_AB:
                        a_scale_col = tl.reshape(a_scale_vec, (1, BLOCK_SIZE_M))
                    else:
                        a_scale_col = tl.reshape(a_scale_vec, (BLOCK_SIZE_M, 1))

                    if BLOCK_SIZE_N == 128:
                        b1_scale = tl.load(
                            b_scale_ptr
                            + (offs_bn1 // 128) * num_k_blocks
                            + (kk * BLOCK_SIZE_K // 128)
                        )
                        if SWAP_AB:
                            acc1 += tl.dot(b1, a.T) * a_scale_col * b1_scale
                        else:
                            acc1 += tl.dot(a, b1.T) * a_scale_col * b1_scale
                    else:
                        b1_scale = tl.load(
                            b_scale_ptr
                            + (offs_bn1 // 128) * num_k_blocks
                            + (kk * BLOCK_SIZE_K // 128)
                        )
                        b1_scale2 = tl.load(
                            b_scale_ptr
                            + ((offs_bn1 + 128) // 128) * num_k_blocks
                            + (kk * BLOCK_SIZE_K // 128)
                        )
                        scales = tl.join(b1_scale, b1_scale2)
                        b_scale_vec = tl.reshape(scales, (2, 1))
                        scales_expanded = tl.broadcast_to(b_scale_vec, (2, 128))
                        if SWAP_AB:
                            final_scales = tl.reshape(scales_expanded, (256, 1))
                            acc1 += tl.dot(b1, a.T) * a_scale_col * final_scales
                        else:
                            final_scales = tl.reshape(scales_expanded, (1, 256))
                            acc1 += tl.dot(a, b1.T) * a_scale_col * final_scales

                offs_m = offs_am + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offs_m < gm
                tile_weights = tl.load(p_weight_ptr + offs_m, mask=mask_m, other=0.0)
                source_rows = tl.load(p_source_idx_ptr + offs_m, mask=mask_m, other=0)

                if SWAP_AB:
                    acc1 = acc1 * tl.reshape(tile_weights, (1, BLOCK_SIZE_M))
                    acc_store = acc1.T
                else:
                    acc_store = acc1 * tl.reshape(tile_weights, (BLOCK_SIZE_M, 1))

                cols = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                out_idx = output_ptr + source_rows[:, None].to(tl.int64) * ldc + cols[None, :]
                tl.atomic_add(out_idx, acc_store, sem="relaxed", mask=mask_m[:, None])

                tile_idx += NUM_SM

        last_gemm_end_tile_idx = last_gemm_end_tile_idx + num_tiles


def launch_gemm2_kernel(
    gemm1_output,
    gemm1_output_scale, # [2048//128, sum(s_i)], fp32
    offset,
    gemm2_weights,
    gemm2_weights_scale,
    permute_weights,
    permute_token_idx,
    seq_len: int,
    output: torch.Tensor,
    *,
    block_size_m: int = 32,
    block_size_n: int = 128,
    block_size_k: int = 128,
    num_sm: int = 160,
) -> torch.Tensor:
    grid = (num_sm,)

    gemm2_kernel[grid](
        gemm1_output,
        gemm1_output_scale,
        offset,
        gemm2_weights,
        gemm2_weights_scale,
        permute_weights,
        permute_token_idx,
        seq_len,
        output,
        NUM_SM=num_sm,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        num_stages=4,
        num_warps=8,
    )
    return output


def gemm2(
    gemm1_output,
    gemm1_output_scale, # [2048//128, sum(s_i)], fp32
    offset,
    gemm2_weights,
    gemm2_weights_scale,
    permute_weights,
    permute_token_idx,
    seq_len: int,
    output,
    num_sm=160,
):
    launch_gemm2_kernel(
        gemm1_output,
        gemm1_output_scale,
        offset,
        gemm2_weights,
        gemm2_weights_scale,
        permute_weights,
        permute_token_idx,
        seq_len=seq_len,
        output=output,
        block_size_m=32,
        block_size_n=128,
        block_size_k=128,
        num_sm=num_sm,
    )

    return output



class gemm2Kernel:
    def __init__(self, num_sm):
        signature = {
            'a_base_ptr': '*fp8e4nv',
            'a_scale_base_ptr': '*fp32',
            'a_offset_ptr': '*i32',
            'b_base_ptr': '*fp8e4nv',
            'b_scale_base_ptr': '*fp32',
            'permute_weights_base_ptr': '*fp32',
            'permute_token_idx_base_ptr': '*i32',
            'seq_len': 'i32',
            'output_ptr': '*bf16',
            'NUM_SM': 'constexpr',
            'BLOCK_SIZE_M': 'constexpr',
            'BLOCK_SIZE_N': 'constexpr',
            'BLOCK_SIZE_K': 'constexpr',
            'USE_TMA': 'constexpr',
            'SWAP_AB': 'constexpr',
        }

        self.configs = []
        self.configs.append(GemmConfig(block_size_m=16, block_size_n=256, block_size_k=128, num_stages=4, use_tma=False, swap_ab=True))
        self.configs.append(GemmConfig(block_size_m=64, block_size_n=256, block_size_k=128, num_stages=4, use_tma=False, swap_ab=False))
        self.configs.append(GemmConfig(block_size_m=128, block_size_n=256, block_size_k=128, num_stages=3, use_tma=True, swap_ab=False))
        constexprs_list = []
        options_list = []
        for i in range(len(self.configs)):
            constexprs_list.append({
                (9,): num_sm,
                (10,): self.configs[i].block_size_m,
                (11,): self.configs[i].block_size_n,
                (12,): self.configs[i].block_size_k,
                (13,): self.configs[i].use_tma,
                (14,): self.configs[i].swap_ab,
            })
            options_list.append({
                "num_warps": 8,
                "num_stages": self.configs[i].num_stages,
            })

        attrs = {
            (0,): [['tt.divisibility', 16]],
            (1,): [['tt.divisibility', 16]],
            (2,): [['tt.divisibility', 16]],
            (3,): [['tt.divisibility', 16]],
            (4,): [['tt.divisibility', 16]],
            (5,): [['tt.divisibility', 16]],
            (6,): [['tt.divisibility', 16]],
            (7,): [['tt.divisibility', 16]],
            (8,): [['tt.divisibility', 16]]
        }

        self.num_sm = num_sm
        self.kernels = []
        for i in range(len(self.configs)):
            src = ASTSource(
                fn=gemm2_kernel,
                signature=signature,
                constexprs=constexprs_list[i],
                attrs=attrs
            )
            compiled_kernel = compile(src, options=options_list[i])
            self.kernels.append(compiled_kernel)


    def __call__(
            self,
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            permute_weights,
            permute_token_idx,
            seq_len,
            output,
            stream=None):
        if stream is None:
            device = triton.runtime.driver.active.get_current_device()
            stream = triton.runtime.driver.active.get_current_stream(device)
        elif hasattr(stream, "cuda_stream"):
            # Accept either a raw CUDA stream pointer or a torch stream wrapper.
            stream = stream.cuda_stream

        kernel = None
        config = None
        if seq_len <= 128:
            kernel = self.kernels[0]
            config = self.configs[0]
        elif seq_len <= 1024:
            kernel = self.kernels[1]
            config = self.configs[1]
        else:
            kernel = self.kernels[2]
            config = self.configs[2]

        grid = (self.num_sm, 1, 1)
        launch_metadata = kernel.launch_metadata(grid, stream, a_base, a_scale_base, a_offset, b_base, b_scale_base, permute_weights, permute_token_idx, seq_len, output)

        kernel.run(
            grid[0], grid[1], grid[2],
            stream,
            kernel.function,
            kernel.packed_metadata,
            launch_metadata,
            None,
            None,
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            permute_weights,
            permute_token_idx,
            seq_len,
            output,
            self.num_sm,
            config.block_size_m,
            config.block_size_n,
            config.block_size_k,
            config.use_tma,
            config.swap_ab,
        )
        return


class gemm2FinalizeKernel:
    def __init__(self, num_sm):
        signature = {
            'a_base_ptr': '*fp8e4nv',
            'a_scale_base_ptr': '*fp32',
            'a_offset_ptr': '*i32',
            'b_base_ptr': '*fp8e4nv',
            'b_scale_base_ptr': '*fp32',
            'permute_weights_base_ptr': '*fp32',
            'permute_token_idx_base_ptr': '*i32',
            'seq_len': 'i32',
            'output_ptr': '*bf16',
            'NUM_SM': 'constexpr',
            'BLOCK_SIZE_M': 'constexpr',
            'BLOCK_SIZE_N': 'constexpr',
            'BLOCK_SIZE_K': 'constexpr',
            'USE_TMA': 'constexpr',
            'SWAP_AB': 'constexpr',
        }

        self.configs = []
        self.configs.append(GemmConfig(block_size_m=16, block_size_n=256, block_size_k=128, num_stages=4, use_tma=False, swap_ab=True))
        self.configs.append(GemmConfig(block_size_m=64, block_size_n=256, block_size_k=128, num_stages=4, use_tma=False, swap_ab=False))
        self.configs.append(GemmConfig(block_size_m=128, block_size_n=256, block_size_k=128, num_stages=3, use_tma=True, swap_ab=False))

        attrs = {
            (0,): [['tt.divisibility', 16]],
            (1,): [['tt.divisibility', 16]],
            (2,): [['tt.divisibility', 16]],
            (3,): [['tt.divisibility', 16]],
            (4,): [['tt.divisibility', 16]],
            (5,): [['tt.divisibility', 16]],
            (6,): [['tt.divisibility', 16]],
            (7,): [['tt.divisibility', 16]],
            (8,): [['tt.divisibility', 16]],
        }

        self.num_sm = num_sm
        self.kernels = []
        for config in self.configs:
            src = ASTSource(
                fn=gemm2_finalize_kernel,
                signature=signature,
                constexprs={
                    (9,): num_sm,
                    (10,): config.block_size_m,
                    (11,): config.block_size_n,
                    (12,): config.block_size_k,
                    (13,): config.use_tma,
                    (14,): config.swap_ab,
                },
                attrs=attrs,
            )
            self.kernels.append(
                compile(src, options={"num_warps": 8, "num_stages": config.num_stages})
            )

    def __call__(
            self,
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            permute_weights,
            permute_token_idx,
            seq_len,
            output,
            stream=None):
        if stream is None:
            device = triton.runtime.driver.active.get_current_device()
            stream = triton.runtime.driver.active.get_current_stream(device)
        elif hasattr(stream, "cuda_stream"):
            stream = stream.cuda_stream

        if seq_len <= 128:
            kernel = self.kernels[0]
            config = self.configs[0]
        elif seq_len <= 1024:
            kernel = self.kernels[1]
            config = self.configs[1]
        else:
            kernel = self.kernels[2]
            config = self.configs[2]

        grid = (self.num_sm, 1, 1)
        launch_metadata = kernel.launch_metadata(
            grid, stream, a_base, a_scale_base, a_offset, b_base, b_scale_base,
            permute_weights, permute_token_idx, seq_len, output
        )

        kernel.run(
            grid[0], grid[1], grid[2],
            stream,
            kernel.function,
            kernel.packed_metadata,
            launch_metadata,
            None,
            None,
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            permute_weights,
            permute_token_idx,
            seq_len,
            output,
            self.num_sm,
            config.block_size_m,
            config.block_size_n,
            config.block_size_k,
            config.use_tma,
            config.swap_ab,
        )
        return

num_sm = torch.cuda.get_device_properties(0).multi_processor_count
gemm1_aot = gemm1Kernel(num_sm)
gemm2_aot = gemm2Kernel(num_sm)
gemm1_swiglu_quant_aot = None
gemm2_finalize_aot = None
_gemm1_swiglu_quant_disabled = False
_gemm2_finalize_disabled = False


def _get_gemm1_swiglu_quant_aot():
    global gemm1_swiglu_quant_aot
    if gemm1_swiglu_quant_aot is None:
        gemm1_swiglu_quant_aot = gemm1SwigluQuantKernel(num_sm)
    return gemm1_swiglu_quant_aot


def _get_gemm2_finalize_aot():
    global gemm2_finalize_aot
    if gemm2_finalize_aot is None:
        gemm2_finalize_aot = gemm2FinalizeKernel(num_sm)
    return gemm2_finalize_aot


def _should_use_gemm1_swiglu_quant(seq_len: int) -> bool:
    return (not _gemm1_swiglu_quant_disabled) and seq_len == 1


def _should_use_gemm2_finalize(seq_len: int) -> bool:
    return (not _gemm2_finalize_disabled) and seq_len <= 128

# Build the native helpers once per process, then share the same allocator
# policy with the Triton AOT kernels above.
my_lib = get_moe_runtime_lib()

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(alloc_fn)

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
):
    """Reference implementation used for local validation and debugging."""
    # Fixed DeepSeek-V3/R1 geometry
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]

    BLOCK = 128
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert H == 7168, "hidden_size must be 7168"
    assert I == 2048, "intermediate_size must be 2048"
    assert E_global == 256, "num_experts must be 256"
    assert E_local == 32, "num_local_experts must be 32"

    # Routing constants
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    # Block counts
    num_hidden_blocks = H // BLOCK          # 56
    num_intermediate_blocks = I // BLOCK    # 16
    num_gemm1_out_blocks = (2 * I) // BLOCK # 32

    # Shape checks
    assert hidden_states.shape == (T, H)
    assert hidden_states_scale.shape == (num_hidden_blocks, T)
    assert gemm1_weights.shape == (E_local, 2 * I, H)
    assert gemm1_weights_scale.shape == (E_local, num_gemm1_out_blocks, num_hidden_blocks)
    assert gemm2_weights.shape == (E_local, H, I)
    assert gemm2_weights_scale.shape == (E_local, num_hidden_blocks, num_intermediate_blocks)
    assert routing_bias.shape[-1] == E_global

    device = hidden_states.device

    # 1) FP8 block-scale dequantization
    # hidden_states: [T, H], scale: [H/128, T] (transposed layout)
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)                # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()            # [T, H/128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)                                   # [T, H/128, 128]
        .reshape(T, H)                                         # [T, H]
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded                              # [T, H] float32

    # W13: [E_local, 2I, H], scale: [E_local, (2I)/128, H/128]
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)  # [E, 2I, H/128]
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)  # [E, 2I, H]
    W13 = W13_fp32 * S13_expanded                              # [E, 2I, H] float32

    # W2: [E_local, H, I], scale: [E_local, H/128, I/128]
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)    # [E, H, I/128]
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)    # [E, H, I]
    W2 = W2_fp32 * S2_expanded                                 # [E, H, I] float32

    # 2) No-aux routing
    logits = routing_logits.to(torch.float32)                      # [T, E_global]
    bias = routing_bias.to(torch.float32).reshape(-1)              # [E_global]

    # Sigmoid
    s = 1.0 / (1.0 + torch.exp(-logits))                       # [T, E]
    s_with_bias = s + bias                                     # [T, E] (broadcast)

    # Grouping
    group_size = E_global // N_GROUP # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)    # [T, 8, 32]

    # Group scores = sum of top-2 values within each group
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)  # [T, 8, 2]
    group_scores = top2_vals.sum(dim=2)                        # [T, 8]

    # Select topk_group groups → group mask
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)  # [T, 4]
    group_mask = torch.zeros_like(group_scores)                # [T, 8]
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)  # [T, E]

    # Global top-k (within kept groups), based on s_with_bias
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)                  # [T, E]
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)  # [T, 8]

    # Combination weights: use s (without bias) for normalization
    M = torch.zeros_like(s)                                    # [T, E]
    M.scatter_(1, topk_idx, 1.0)                               # 0/1 mask
    weights = s * M                                            # [T, E]
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor  # [T, E]

    # 3) Local expert compute and accumulation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    local_start = int(local_expert_offset)

    # For each local expert: find selected tokens, run GEMM1→SwiGLU→GEMM2, accumulate by weights
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        # Tokens that selected this global expert ge in their top-k
        sel_mask_per_token = (topk_idx == ge).any(dim=1)       # [T] bool
        if not sel_mask_per_token.any():
            continue
        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)  # [Tk]
        Tk = token_idx.numel()

        # Gather inputs and weights for this expert
        A_e = A.index_select(0, token_idx)                     # [Tk, H]
        W13_e = W13[le]                                        # [2I, H]
        W2_e = W2[le]                                          # [H, I]

        # GEMM1: [Tk, H] @ [H, 2I] = [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())                             # [Tk, 2I]

        # SwiGLU: split and apply silu(x) = x / (1 + exp(-x))
        X1 = G1[:, :I]                                         # [Tk, I]
        X2 = G1[:, I:]                                         # [Tk, I]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))                  # [Tk, I]
        C = silu_X2 * X1                                       # [Tk, I]

        # GEMM2: [Tk, I] @ [I, H] = [Tk, H]
        O = C.matmul(W2_e.t())                                 # [Tk, H]

        # Accumulate with per-token routing weights for this expert
        w_tok = weights.index_select(0, token_idx)[:, ge]      # [Tk]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))  # [Tk,H] * [Tk,1]

    return output.to(torch.bfloat16)



class FusedMoeWorkspace:
    def __init__(self, seq_len: int, device: torch.device):
        total_tokens = seq_len * 8
        self.seq_len = seq_len
        self.total_tokens = total_tokens

        # fused_route_permute_copy_into outputs / scratch
        self.routing_idx = torch.empty((seq_len, 8), device=device, dtype=torch.int32)
        self.routing_weights = torch.empty((seq_len, 8), device=device, dtype=torch.float32)
        self.expert_counts = torch.empty((32,), device=device, dtype=torch.int32)
        self.expert_offsets = torch.empty((33,), device=device, dtype=torch.int32)
        self.total_tokens_device = torch.empty((1,), device=device, dtype=torch.int32)
        self.permute_token_idx = torch.empty((total_tokens,), device=device, dtype=torch.int32)
        self.permute_weight = torch.empty((total_tokens,), device=device, dtype=torch.float32)
        self.permute_hidden_states = torch.empty((total_tokens, 7168), device=device, dtype=torch.float8_e4m3fn)
        self.permute_hidden_states_scale = torch.empty((56, total_tokens), device=device, dtype=torch.float32)
        self.token2permuted_idx = torch.empty((seq_len * 8,), device=device, dtype=torch.int32)
        self.token_counts = torch.empty((seq_len,), device=device, dtype=torch.int32)

        # gemm / reduce outputs
        self.gemm1_output = torch.empty((total_tokens, 4096), device=device, dtype=torch.float16)
        self.gemm2_input = torch.empty((total_tokens, 2048), device=device, dtype=torch.float8_e4m3fn)
        self.gemm2_input_scale = torch.empty((2048 // 128, total_tokens), device=device, dtype=torch.float32)
        self.gemm2_output = torch.empty((total_tokens, 7168), device=device, dtype=torch.bfloat16)
        self.output = torch.empty((seq_len, 7168), device=device, dtype=torch.bfloat16)

        self.stream = torch.cuda.current_stream()


_MAX_SEQ_LEN = 15000
_FUSED_MOE_WORKSPACE = None


def _get_fused_moe_workspace(device: torch.device) -> FusedMoeWorkspace:
    global _FUSED_MOE_WORKSPACE
    if _FUSED_MOE_WORKSPACE is None:
        _FUSED_MOE_WORKSPACE = FusedMoeWorkspace(_MAX_SEQ_LEN, device)
    return _FUSED_MOE_WORKSPACE


@torch.no_grad()
def fused_moe(
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
    output: torch.Tensor | None = None,
):
    seq_len = hidden_states.size(0)
    ws = _get_fused_moe_workspace(hidden_states.device)
    stream = ws.stream

    # 1~3. Routing, permutation, and compact FP8 copy into the expert-major
    # workspace.
    my_lib.fusedRoutePermuteCopyIntoWrapper(
        routing_logits,
        routing_bias,
        routed_scaling_factor,
        hidden_states,
        hidden_states_scale,
        local_expert_offset,
        ws.routing_idx,
        ws.routing_weights,
        ws.expert_counts,
        ws.expert_offsets,
        ws.total_tokens_device,
        ws.permute_token_idx,
        ws.permute_weight,
        ws.permute_hidden_states,
        ws.permute_hidden_states_scale,
        ws.token2permuted_idx,
        ws.token_counts,
        seq_len
    )

    # 4. GEMM1 with an SM100 SwiGLU+quant epilogue. The fallback
    # preserves the previously retained GEMM1 + native actQuant sequence.
    global _gemm1_swiglu_quant_disabled, _gemm2_finalize_disabled
    fused_gemm1_ready = False
    if _should_use_gemm1_swiglu_quant(seq_len):
        try:
            _get_gemm1_swiglu_quant_aot()(
                ws.permute_hidden_states,
                ws.permute_hidden_states_scale,
                ws.expert_offsets,
                gemm1_weights,
                gemm1_weights_scale,
                seq_len,
                ws.gemm2_input,
                ws.gemm2_input_scale,
                stream,
            )
            fused_gemm1_ready = True
        except Exception:
            _gemm1_swiglu_quant_disabled = True

    if not fused_gemm1_ready:
        gemm1_aot(
            ws.permute_hidden_states,
            ws.permute_hidden_states_scale,
            ws.expert_offsets,
            gemm1_weights,
            gemm1_weights_scale,
            seq_len,
            ws.gemm1_output,
            ws.gemm2_input_scale,
            stream,
        )
        my_lib.actQuantWrapper(
            ws.gemm1_output,
            ws.gemm2_input,
            ws.gemm2_input_scale,
            ws.expert_offsets,
            seq_len,
        )

    # 5. GEMM2 and token-wise reduction back to the final output.
    output_buf = ws.output[:seq_len] if output is None else output

    fused_finalize_ready = False
    if _should_use_gemm2_finalize(seq_len):
        try:
            output_buf.zero_()
            _get_gemm2_finalize_aot()(
                ws.gemm2_input,
                ws.gemm2_input_scale,
                ws.expert_offsets,
                gemm2_weights,
                gemm2_weights_scale,
                ws.permute_weight,
                ws.permute_token_idx,
                seq_len,
                output_buf,
                stream,
            )
            fused_finalize_ready = True
        except Exception:
            _gemm2_finalize_disabled = True

    if not fused_finalize_ready:
        gemm2_aot(
            ws.gemm2_input,
            ws.gemm2_input_scale,
            ws.expert_offsets,
            gemm2_weights,
            gemm2_weights_scale,
            ws.permute_weight,
            ws.permute_token_idx,
            seq_len,
            ws.gemm2_output,
            stream,
        )

        my_lib.reduceAddWrapper(
            ws.gemm2_output,
            ws.token2permuted_idx,
            ws.token_counts,
            output_buf,
            seq_len,
        )
    return output_buf
