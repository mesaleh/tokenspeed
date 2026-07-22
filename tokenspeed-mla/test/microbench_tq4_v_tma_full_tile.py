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

"""Validate a complete production-sized TQ4 V landing/conversion stage.

Each CTA issues eight natural-U4 TMA page-slice copies for its discontiguous
half of a 128x512 V tile. The packed view occupies the upper 16 KiB of the
existing 32 KiB FP8 V stage. Four warps capture one complete page-pair and
latent chunk per phase before overwriting the same backing through the exact
production V-stage view.
"""

from __future__ import annotations

import os

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor

from tokenspeed_mla.tq4_cutedsl import dequantize_tq4_word_to_fp8_shfl


TILE_TOKENS = 128
CTA_LATENT = 256
FULL_LATENT = 512
PAGE_TOKENS = 32
TMA_LATENT = 128
THREADS = 128
CONVERSION_PHASES = 4


def make_v_layouts():
    mma_tiler = (128, 256, 64)
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        mma_tiler[:2],
    )
    staged = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler, cutlass.Float8E4M3FN, 8
    )
    staged = cute.logical_divide(
        cute.logical_divide(staged, (None, None, None, 4)),
        (None, None, None, (2, None)),
    )

    for_tma = sm100_utils.make_smem_layout(
        OperandMajorMode.MN,
        (mma_tiler[1] // tiled_mma.thr_id.shape, mma_tiler[2]),
        cutlass.Float8E4M3FN,
        8,
    )
    for_tma = cute.tiled_divide(
        for_tma,
        (
            tiled_mma.op.shape_mnk[1] // tiled_mma.thr_id.shape,
            PAGE_TOKENS,
        ),
    )
    for_tma = cute.logical_divide(
        cute.logical_divide(for_tma, (None, None, None, 4)),
        (None, None, None, (2, None)),
    )
    return staged, for_tma


@cute.kernel
def full_tile_kernel(
    tma_atom: cute.CopyAtom,
    tma_tensor: cute.Tensor,
    scales: cute.Tensor,
    centroids: cute.Tensor,
    page_table: cute.Tensor,
    output: cute.Tensor,
    write_output: cutlass.Constexpr,
    staged_layout: cute.ComposedLayout,
    tma_layout: cute.ComposedLayout,
    SharedStorage: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    tile = block // 2
    cta = block % 2

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    barrier_ptr = storage.tma_barrier.data_ptr()
    fp8_stage = storage.stage.get_tensor(
        staged_layout.outer, swizzle=staged_layout.inner
    )
    packed_layout = cute.make_layout(
        (
            PAGE_TOKENS,
            TMA_LATENT,
            TILE_TOKENS // PAGE_TOKENS,
            CTA_LATENT // TMA_LATENT,
        ),
        stride=(
            TMA_LATENT,
            1,
            PAGE_TOKENS * TMA_LATENT,
            TILE_TOKENS * TMA_LATENT,
        ),
    )
    packed_stage = cute.make_tensor(
        cute.recast_ptr(
            storage.stage.data_ptr() + TILE_TOKENS * CTA_LATENT // 2,
            dtype=cutlass.Int4,
        ),
        packed_layout,
    )

    tiled_global = cute.flat_divide(tma_tensor, (PAGE_TOKENS, TMA_LATENT))
    tma_stage, tma_global = cpasync.tma_partition(
        tma_atom,
        0,
        cute.make_layout(1),
        cute.group_modes(packed_stage, 0, 2),
        cute.group_modes(tiled_global, 0, 2),
    )
    if tidx < 32:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(barrier_ptr, 1)
            cute.arch.mbarrier_expect_tx(
                barrier_ptr,
                TILE_TOKENS * CTA_LATENT * cutlass.Int4.width // 8,
            )
        cute.arch.mbarrier_init_fence()
        for token_stage in cutlass.range_constexpr(2):
            for page in cutlass.range_constexpr(2):
                physical_page = page_table[tile, token_stage * 2 + page]
                packed_page = token_stage * 2 + page
                for latent_stage in cutlass.range_constexpr(2):
                    global_chunk = latent_stage * 2 + cta
                    cute.copy(
                        tma_atom,
                        tma_global[None, 0, global_chunk, physical_page],
                        tma_stage[None, packed_page, latent_stage],
                        tma_bar_ptr=barrier_ptr,
                    )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(barrier_ptr)
        cute.arch.mbarrier_wait(barrier_ptr, 0)
    cute.arch.sync_threads()

    packed_i32 = cute.recast_tensor(packed_stage, cutlass.Int32)
    stage_tma_ptr = cute.recast_ptr(
        fp8_stage.iterator, tma_layout.inner, dtype=cutlass.Float8E4M3FN
    )
    stage_tma_i32 = cute.recast_tensor(
        cute.make_tensor(stage_tma_ptr, tma_layout.outer), cutlass.Int32
    )
    output_i32 = cute.recast_tensor(output, cutlass.Int32)
    words_per_thread = TILE_TOKENS * CTA_LATENT // (
        THREADS * 8 * CONVERSION_PHASES
    )
    raw_words = cute.make_rmem_tensor(
        cute.make_layout(words_per_thread), cutlass.Int32
    )
    centroid_lane = cutlass.Float32(centroids[tidx % 16])
    for phase in cutlass.range(CONVERSION_PHASES, unroll=1):
        token_stage = phase // 2
        latent_stage = phase % 2
        for iteration in cutlass.range_constexpr(words_per_thread):
            linear_word = iteration * THREADS + tidx
            token = linear_word // (TMA_LATENT // 8)
            chunk_word = linear_word % (TMA_LATENT // 8)
            page = token // PAGE_TOKENS
            page_row = token % PAGE_TOKENS
            packed_page = token_stage * 2 + page
            raw_words[iteration] = packed_i32[
                page_row, chunk_word, packed_page, latent_stage
            ]

        cute.arch.sync_threads()
        for iteration in cutlass.range_constexpr(words_per_thread):
            linear_word = iteration * THREADS + tidx
            token = linear_word // (TMA_LATENT // 8)
            chunk_word = linear_word % (TMA_LATENT // 8)
            page = token // PAGE_TOKENS
            page_row = token % PAGE_TOKENS
            physical_page = page_table[tile, token_stage * 2 + page]
            scale = cutlass.Float32(scales[physical_page, page_row])
            fp8_0, fp8_1 = dequantize_tq4_word_to_fp8_shfl(
                raw_words[iteration], scale, centroid_lane
            )
            dim_word = chunk_word * 2
            stage_tma_i32[
                ((dim_word, page_row), 0, page, ((latent_stage, token_stage), 0))
            ] = fp8_0
            stage_tma_i32[
                (
                    (dim_word + 1, page_row),
                    0,
                    page,
                    ((latent_stage, token_stage), 0),
                )
            ] = fp8_1

    if cutlass.const_expr(write_output):
        cute.arch.sync_threads()
        for iteration in cutlass.range_constexpr(
            TILE_TOKENS * CTA_LATENT // (THREADS * 4)
        ):
            linear_word = iteration * THREADS + tidx
            token = linear_word // (CTA_LATENT // 4)
            output_word = linear_word % (CTA_LATENT // 4)
            latent_stage = output_word // 32
            dim_word = output_word % 32
            token_stage = token // 64
            page = (token % 64) // PAGE_TOKENS
            page_row = token % PAGE_TOKENS
            output_i32[block, token, output_word] = stage_tma_i32[
                ((dim_word, page_row), 0, page, ((latent_stage, token_stage), 0))
            ]


@cute.jit
def full_tile(
    packed: cute.Tensor,
    scales: cute.Tensor,
    centroids: cute.Tensor,
    page_table: cute.Tensor,
    output: cute.Tensor,
    write_output: cutlass.Constexpr,
):
    packed_page_major = cute.make_tensor(
        packed.iterator,
        cute.make_layout(
            (packed.shape[1], packed.shape[2], packed.shape[0]),
            stride=(packed.stride[1], packed.stride[2], packed.stride[0]),
        ),
    )
    packed_i4 = cute.recast_tensor(packed_page_major, cutlass.Int4)
    tma_smem_layout = cute.make_layout(
        (PAGE_TOKENS, TMA_LATENT), stride=(TMA_LATENT, 1)
    )
    tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        packed_i4,
        tma_smem_layout,
        (PAGE_TOKENS, TMA_LATENT),
    )
    staged_layout, tma_layout = make_v_layouts()

    @cute.struct
    class SharedStorage:
        tma_barrier: cute.struct.MemRange[cutlass.Int64, 1]
        stage: cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Float8E4M3FN, cute.cosize(staged_layout.outer)
            ],
            1024,
        ]

    full_tile_kernel(
        tma_atom,
        tma_tensor,
        scales,
        centroids,
        page_table,
        output,
        write_output,
        staged_layout,
        tma_layout,
        SharedStorage,
    ).launch(
        grid=(output.shape[0], 1, 1),
        block=(THREADS, 1, 1),
        smem=SharedStorage.size_in_bytes(),
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
    write_output = os.environ.get("TQ4_PROBE_WRITE_OUTPUT", "1") == "1"
    compiled = cute.compile(
        full_tile,
        fake(cutlass.Uint8, (pages, PAGE_TOKENS, FULL_LATENT // 2), 16),
        fake(cutlass.BFloat16, (pages, PAGE_TOKENS), 16),
        fake(cutlass.Float32, (16,), 16),
        fake(cutlass.Int32, (tiles, TILE_TOKENS // PAGE_TOKENS), 16),
        fake(cutlass.Float8E4M3FN, (ctas, TILE_TOKENS, CTA_LATENT), 16),
        write_output,
        options="--enable-tvm-ffi --opt-level 3",
    )

    torch.manual_seed(37)
    num_tiles = int(os.environ.get("TQ4_PROBE_TILES", "32"))
    benchmark_iterations = int(os.environ.get("TQ4_PROBE_ITERS", "1000"))
    if num_tiles <= 0 or benchmark_iterations <= 0:
        raise ValueError("TQ4_PROBE_TILES and TQ4_PROBE_ITERS must be positive")
    num_pages = num_tiles * (TILE_TOKENS // PAGE_TOKENS)
    packed = torch.randint(
        0,
        256,
        (num_pages, PAGE_TOKENS, FULL_LATENT // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    scales = (
        torch.rand(num_pages, PAGE_TOKENS, device="cuda", dtype=torch.bfloat16) * 3
    )
    centroids = torch.linspace(
        -0.12, 0.11, 16, device="cuda", dtype=torch.float32
    )
    page_table = torch.randperm(
        num_pages, device="cuda", dtype=torch.int32
    ).reshape(num_tiles, TILE_TOKENS // PAGE_TOKENS)
    output = torch.empty(
        (num_tiles * 2, TILE_TOKENS, CTA_LATENT),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )

    compiled(packed, scales, centroids, page_table, output)
    torch.cuda.synchronize()
    if write_output:
        ordered_packed = packed[page_table.long()].reshape(
            num_tiles, TILE_TOKENS, FULL_LATENT // 2
        )
        ordered_scales = scales[page_table.long()].reshape(num_tiles, TILE_TOKENS)
        indices = torch.stack(
            (ordered_packed & 0xF, ordered_packed >> 4), dim=-1
        ).reshape(num_tiles, TILE_TOKENS, FULL_LATENT)
        dense = (centroids[indices.long()] * ordered_scales[..., None].float()).to(
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
        f"PASS tiles={num_tiles} ctas={num_tiles * 2} "
        f"write_output={write_output} output={tuple(output.shape)} "
        f"launch={elapsed_us:.3f} us"
    )


if __name__ == "__main__":
    main()
