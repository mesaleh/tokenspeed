# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Probe exact TQ4 dequant into TokenSpeed's existing SM100 K stage.

The global readback exists only to make every shared-memory write observable and
to provide a bit-exact correctness check. Reported times are therefore upper
bounds for the conversion stage, not standalone attention timings.
"""

from __future__ import annotations

import math
import os

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor

from tokenspeed_mla.fmha_helpers import cvt_f32x4_to_f8x4_pack_i32
from tokenspeed_mla.tq4_cutedsl import dequantize_tq4_word_to_fp8


TQ4_UNIFORM_MIN = -2.5 / math.sqrt(512)
TQ4_UNIFORM_STEP = 5.0 / (15 * math.sqrt(512))


@cute.jit
def codebook_value(index, centroids: cute.Tensor, uniform: cutlass.Constexpr):
    if cutlass.const_expr(uniform):
        return cutlass.Float32(index) * TQ4_UNIFORM_STEP + TQ4_UNIFORM_MIN
    return cutlass.Float32(centroids[index])


def make_k_layouts():
    mma_qk_tiler = (128, 128, 128)
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        (128, 128),
    )
    staged = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_qk_tiler, cutlass.Float8E4M3FN, 12
    )
    staged = cute.logical_divide(staged, (None, None, None, 4))

    page_tile_size = 32
    for_tma = sm100_utils.make_smem_layout(
        OperandMajorMode.K,
        (mma_qk_tiler[0] // tiled_mma.thr_id.shape, mma_qk_tiler[2]),
        cutlass.Float8E4M3FN,
        12,
    )
    for_tma = cute.tiled_divide(for_tma, (page_tile_size, mma_qk_tiler[2]))
    for_tma = cute.logical_divide(for_tma, (None, None, None, 4))
    return staged, for_tma


@cute.kernel
def stage_kernel(
    packed: cute.Tensor,
    scales: cute.Tensor,
    centroids: cute.Tensor,
    output: cute.Tensor,
    staged_layout: cute.ComposedLayout,
    tma_layout: cute.ComposedLayout,
    uniform: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    stage = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        staged_layout.outer,
        byte_alignment=1024,
        swizzle=staged_layout.inner,
    )
    centroid_smem = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(16), byte_alignment=16
    )
    stage_tma_ptr = cute.recast_ptr(
        stage.iterator, tma_layout.inner, dtype=cutlass.Float8E4M3FN
    )
    stage_tma = cute.make_tensor(stage_tma_ptr, tma_layout.outer)
    packed_i32 = cute.recast_tensor(packed, cutlass.Int32)
    stage_tma_i32 = cute.recast_tensor(stage_tma, cutlass.Int32)
    output_i32 = cute.recast_tensor(output, cutlass.Int32)

    if tidx < 16:
        centroid_smem[tidx] = centroids[tidx]
    cute.arch.sync_threads()

    if tidx < 32:
        for iteration in cutlass.range_constexpr(128):
            word_linear = iteration * 32 + tidx
            row = word_linear // 64
            packed_word = word_linear % 64
            latent_stage = packed_word // 16
            word_col = packed_word % 16
            raw = packed_i32[bidx, row, packed_word]
            scale = cutlass.Float32(scales[bidx, row])

            if cutlass.const_expr(uniform):
                values0 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
                values1 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
                values0[0] = codebook_value((raw >> 0) & 0xF, centroid_smem, True) * scale
                values0[1] = codebook_value((raw >> 4) & 0xF, centroid_smem, True) * scale
                values0[2] = codebook_value((raw >> 8) & 0xF, centroid_smem, True) * scale
                values0[3] = codebook_value((raw >> 12) & 0xF, centroid_smem, True) * scale
                values1[0] = codebook_value((raw >> 16) & 0xF, centroid_smem, True) * scale
                values1[1] = codebook_value((raw >> 20) & 0xF, centroid_smem, True) * scale
                values1[2] = codebook_value((raw >> 24) & 0xF, centroid_smem, True) * scale
                values1[3] = codebook_value((raw >> 28) & 0xF, centroid_smem, True) * scale
                packed_fp8_0 = cvt_f32x4_to_f8x4_pack_i32(
                    values0, cutlass.Float8E4M3FN
                )
                packed_fp8_1 = cvt_f32x4_to_f8x4_pack_i32(
                    values1, cutlass.Float8E4M3FN
                )
            else:
                packed_fp8_0, packed_fp8_1 = dequantize_tq4_word_to_fp8(
                    raw, scale, centroid_smem
                )

            page_row = row % 32
            page = row // 32
            dim_word = word_col * 2
            stage_tma_i32[
                ((page_row, dim_word), page, 0, (latent_stage, 0))
            ] = packed_fp8_0
            stage_tma_i32[
                ((page_row, dim_word + 1), page, 0, (latent_stage, 0))
            ] = packed_fp8_1

    cute.arch.sync_threads()

    for iteration in cutlass.range_constexpr(64):
        word_linear = iteration * 128 + tidx
        row = word_linear // 128
        output_word = word_linear % 128
        latent_stage = output_word // 32
        dim_word = output_word % 32
        page_row = row % 32
        page = row // 32
        output_i32[bidx, row, output_word] = stage_tma_i32[
            ((page_row, dim_word), page, 0, (latent_stage, 0))
        ]


@cute.jit
def stage(
    packed: cute.Tensor,
    scales: cute.Tensor,
    centroids: cute.Tensor,
    output: cute.Tensor,
    uniform: cutlass.Constexpr,
):
    staged_layout, tma_layout = make_k_layouts()
    stage_kernel(
        packed, scales, centroids, output, staged_layout, tma_layout, uniform
    ).launch(
        grid=(packed.shape[0], 1, 1),
        block=(128, 1, 1),
        smem=cute.size_in_bytes(cutlass.Float8E4M3FN, staged_layout.outer) + 64,
        min_blocks_per_mp=1,
    )


def fake(dtype: type[cutlass.Numeric], shape: tuple, align: int):
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )


def main() -> None:
    blocks = cute.sym_int()
    uniform = os.environ.get("TQ4_PROBE_UNIFORM", "0") == "1"
    compiled = cute.compile(
        stage,
        fake(cutlass.Uint8, (blocks, 64, 256), 16),
        fake(cutlass.BFloat16, (blocks, 64), 16),
        fake(cutlass.Float32, (16,), 16),
        fake(cutlass.Float8E4M3FN, (blocks, 64, 512), 16),
        uniform,
        options="--enable-tvm-ffi --opt-level 3",
    )

    torch.manual_seed(11)
    num_blocks = int(os.environ.get("TQ4_PROBE_BLOCKS", "64"))
    benchmark_iterations = int(os.environ.get("TQ4_PROBE_ITERS", "1000"))
    if num_blocks <= 0 or benchmark_iterations <= 0:
        raise ValueError("TQ4_PROBE_BLOCKS and TQ4_PROBE_ITERS must be positive")
    packed = torch.randint(
        0, 256, (num_blocks, 64, 256), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(num_blocks, 64, device="cuda", dtype=torch.bfloat16) * 3
    centroids = torch.linspace(-0.12, 0.11, 16, device="cuda", dtype=torch.float32)
    output = torch.empty(
        (num_blocks, 64, 512), device="cuda", dtype=torch.float8_e4m3fn
    )

    compiled(packed, scales, centroids, output)
    torch.cuda.synchronize()

    low = packed & 0xF
    high = packed >> 4
    indices = torch.stack((low, high), dim=-1).reshape(num_blocks, 64, 512).long()
    codebook = (
        torch.arange(16, device="cuda", dtype=torch.float32) * TQ4_UNIFORM_STEP
        + TQ4_UNIFORM_MIN
        if uniform
        else centroids
    )
    expected = (codebook[indices] * scales[..., None].float()).to(
        torch.float8_e4m3fn
    )
    torch.testing.assert_close(output.float(), expected.float(), rtol=0, atol=0)

    for _ in range(20):
        compiled(packed, scales, centroids, output)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(benchmark_iterations):
        compiled(packed, scales, centroids, output)
    end.record()
    end.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000 / benchmark_iterations
    print(
        f"PASS blocks={num_blocks} uniform={uniform} stage_bytes=98368 "
        f"output={tuple(output.shape)} launch={elapsed_us:.3f} us"
    )


if __name__ == "__main__":
    main()
