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

"""Probe exact TQ4 dequant into TokenSpeed's existing SM100 V stage.

Each two-CTA cluster consumes one 128-token tile. CTA 0 and CTA 1 each own two
discontiguous 128-wide slices of the 512-wide latent vector. The global
readback exists only to validate every shared-memory write bit-for-bit, so the
reported time is an upper bound for the conversion stage.
"""

from __future__ import annotations

import os

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor

from tokenspeed_mla.tq4_cutedsl import dequantize_tq4_word_to_fp8


def make_v_layouts():
    mma_pv_tiler = (128, 256, 64)
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        mma_pv_tiler[:2],
    )
    staged = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_pv_tiler, cutlass.Float8E4M3FN, 8
    )
    staged = cute.logical_divide(
        cute.logical_divide(staged, (None, None, None, 4)),
        (None, None, None, (2, None)),
    )

    for_tma = sm100_utils.make_smem_layout(
        OperandMajorMode.MN,
        (mma_pv_tiler[1] // tiled_mma.thr_id.shape, mma_pv_tiler[2]),
        cutlass.Float8E4M3FN,
        8,
    )
    for_tma = cute.tiled_divide(
        for_tma,
        (
            tiled_mma.op.shape_mnk[1] // tiled_mma.thr_id.shape,
            32,
        ),
    )
    for_tma = cute.logical_divide(
        cute.logical_divide(for_tma, (None, None, None, 4)),
        (None, None, None, (2, None)),
    )
    return staged, for_tma


@cute.kernel
def stage_kernel(
    packed: cute.Tensor,
    scales: cute.Tensor,
    centroids: cute.Tensor,
    page_table: cute.Tensor,
    output: cute.Tensor,
    staged_layout: cute.ComposedLayout,
    tma_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tile = bidx // 2
    cta = bidx % 2

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
            token = word_linear // 32
            local_word = word_linear % 32
            latent_stage = local_word // 16
            word_col = local_word % 16
            global_word = latent_stage * 32 + cta * 16 + word_col
            token_stage = token // 64
            page = (token % 64) // 32
            page_row = token % 32
            physical_page = page_table[tile, token_stage * 2 + page]
            raw = packed_i32[physical_page, page_row, global_word]
            scale = cutlass.Float32(scales[physical_page, page_row])
            packed_fp8_0, packed_fp8_1 = dequantize_tq4_word_to_fp8(
                raw, scale, centroid_smem
            )

            dim_word = word_col * 2
            stage_tma_i32[
                ((dim_word, page_row), 0, page, ((latent_stage, token_stage), 0))
            ] = packed_fp8_0
            stage_tma_i32[
                (
                    (dim_word + 1, page_row),
                    0,
                    page,
                    ((latent_stage, token_stage), 0),
                )
            ] = packed_fp8_1

    cute.arch.sync_threads()

    for iteration in cutlass.range_constexpr(64):
        word_linear = iteration * 128 + tidx
        token = word_linear // 64
        output_word = word_linear % 64
        latent_stage = output_word // 32
        dim_word = output_word % 32
        token_stage = token // 64
        page = (token % 64) // 32
        page_row = token % 32
        output_i32[bidx, token, output_word] = stage_tma_i32[
            ((dim_word, page_row), 0, page, ((latent_stage, token_stage), 0))
        ]


@cute.jit
def stage(
    packed: cute.Tensor,
    scales: cute.Tensor,
    centroids: cute.Tensor,
    page_table: cute.Tensor,
    output: cute.Tensor,
):
    staged_layout, tma_layout = make_v_layouts()
    stage_kernel(
        packed, scales, centroids, page_table, output, staged_layout, tma_layout
    ).launch(
        grid=(output.shape[0], 1, 1),
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
    pages = cute.sym_int()
    tiles = cute.sym_int()
    ctas = cute.sym_int()
    compiled = cute.compile(
        stage,
        fake(cutlass.Uint8, (pages, 32, 256), 16),
        fake(cutlass.BFloat16, (pages, 32), 16),
        fake(cutlass.Float32, (16,), 16),
        fake(cutlass.Int32, (tiles, 4), 16),
        fake(cutlass.Float8E4M3FN, (ctas, 128, 256), 16),
        options="--enable-tvm-ffi --opt-level 3",
    )

    torch.manual_seed(17)
    num_tiles = int(os.environ.get("TQ4_PROBE_TILES", "32"))
    benchmark_iterations = int(os.environ.get("TQ4_PROBE_ITERS", "1000"))
    if num_tiles <= 0 or benchmark_iterations <= 0:
        raise ValueError("TQ4_PROBE_TILES and TQ4_PROBE_ITERS must be positive")
    num_pages = num_tiles * 4
    packed = torch.randint(
        0, 256, (num_pages, 32, 256), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(num_pages, 32, device="cuda", dtype=torch.bfloat16) * 3
    centroids = torch.linspace(-0.12, 0.11, 16, device="cuda", dtype=torch.float32)
    page_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        num_tiles, 4
    )
    output = torch.empty(
        (num_tiles * 2, 128, 256), device="cuda", dtype=torch.float8_e4m3fn
    )

    compiled(packed, scales, centroids, page_table, output)
    torch.cuda.synchronize()

    logical_packed = packed[page_table.long()].reshape(num_tiles, 128, 256)
    logical_scales = scales[page_table.long()].reshape(num_tiles, 128)
    indices = torch.stack((logical_packed & 0xF, logical_packed >> 4), dim=-1).reshape(
        num_tiles, 128, 512
    ).long()
    dense = (centroids[indices] * logical_scales[..., None].float()).to(
        torch.float8_e4m3fn
    )
    expected = torch.empty_like(output)
    expected[0::2, :, :128] = dense[:, :, :128]
    expected[0::2, :, 128:] = dense[:, :, 256:384]
    expected[1::2, :, :128] = dense[:, :, 128:256]
    expected[1::2, :, 128:] = dense[:, :, 384:512]
    torch.testing.assert_close(output.float(), expected.float(), rtol=0, atol=0)

    for _ in range(20):
        compiled(packed, scales, centroids, page_table, output)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(benchmark_iterations):
        compiled(packed, scales, centroids, page_table, output)
    end.record()
    end.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000 / benchmark_iterations
    print(
        f"PASS tiles={num_tiles} ctas={num_tiles * 2} stage_bytes=65600 "
        f"output={tuple(output.shape)} launch={elapsed_us:.3f} us"
    )


if __name__ == "__main__":
    main()
