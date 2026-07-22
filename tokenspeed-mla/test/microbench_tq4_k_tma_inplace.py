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

"""Prove natural SM100 TMA-U4 and register-buffered in-place conversion.

Each CTA loads one canonical 32-token by 128-latent page slice in TMA's natural
packed U4 format. A conversion warpgroup first captures the entire half-size
shared tile in registers, then overwrites the same backing storage with exact
FP8 values. This avoids both a dense shadow and a second shared payload buffer.

The global readback makes every byte observable. Timings include TMA load,
barrier synchronization, conversion, and readback, so they are an upper bound
for one page-slice conversion rather than an attention-kernel timing.
"""

from __future__ import annotations

import os

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync
from cutlass.cute.runtime import make_fake_compact_tensor

from tokenspeed_mla.tq4_cutedsl import dequantize_tq4_word_to_fp8_shfl


TILE_TOKENS = 32
TILE_LATENT = 128
THREADS = 128


@cute.kernel
def tma_inplace_kernel(
    tma_atom: cute.CopyAtom,
    tma_tensor: cute.Tensor,
    scales: cute.Tensor,
    centroids: cute.Tensor,
    output: cute.Tensor,
    packed_smem_layout: cute.Layout,
    fp8_smem_layout: cute.ComposedLayout,
    SharedStorage: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    chunk, page, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    barrier_ptr = storage.tma_barrier.data_ptr()
    fp8_stage = storage.stage.get_tensor(
        fp8_smem_layout.outer, swizzle=fp8_smem_layout.inner
    )
    packed_stage = cute.make_tensor(
        cute.recast_ptr(
            storage.stage.data_ptr(),
            dtype=cutlass.Int4,
        ),
        packed_smem_layout,
    )

    if tidx < 32:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(barrier_ptr, 1)
            cute.arch.mbarrier_expect_tx(
                barrier_ptr, TILE_TOKENS * TILE_LATENT * cutlass.Int4.width // 8
            )
        cute.arch.mbarrier_init_fence()

    tiled_global = cute.flat_divide(tma_tensor, (TILE_TOKENS, TILE_LATENT))
    tma_stage, tma_global = cpasync.tma_partition(
        tma_atom,
        0,
        cute.make_layout(1),
        cute.group_modes(packed_stage, 0, 2),
        cute.group_modes(tiled_global, 0, 2),
    )
    if tidx < 32:
        cute.copy(
            tma_atom,
            tma_global[None, 0, chunk, page],
            tma_stage,
            tma_bar_ptr=barrier_ptr,
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(barrier_ptr)
        cute.arch.mbarrier_wait(barrier_ptr, 0)
    cute.arch.sync_threads()

    packed_stage_i32 = cute.recast_tensor(packed_stage, cutlass.Int32)
    fp8_stage_i32 = cute.recast_tensor(fp8_stage, cutlass.Int32)
    output_i32 = cute.recast_tensor(output, cutlass.Int32)
    centroid_lane = cutlass.Float32(centroids[tidx % 16])
    words_per_thread = TILE_TOKENS * TILE_LATENT // (THREADS * 8)
    raw_words = cute.make_rmem_tensor(
        cute.make_layout(words_per_thread), cutlass.Int32
    )
    rows = cute.make_rmem_tensor(
        cute.make_layout(words_per_thread), cutlass.Int32
    )
    row_words = cute.make_rmem_tensor(
        cute.make_layout(words_per_thread), cutlass.Int32
    )
    for iteration in cutlass.range_constexpr(
        words_per_thread
    ):
        linear_word = iteration * THREADS + tidx
        row = linear_word // (TILE_LATENT // 8)
        row_word = linear_word % (TILE_LATENT // 8)
        raw_words[iteration] = packed_stage_i32[row, row_word]
        rows[iteration] = row
        row_words[iteration] = row_word

    cute.arch.sync_threads()
    for iteration in cutlass.range_constexpr(words_per_thread):
        row = rows[iteration]
        row_word = row_words[iteration]
        scale = cutlass.Float32(scales[page, row])
        fp8_0, fp8_1 = dequantize_tq4_word_to_fp8_shfl(
            raw_words[iteration], scale, centroid_lane
        )
        fp8_stage_i32[row, row_word * 2] = fp8_0
        fp8_stage_i32[row, row_word * 2 + 1] = fp8_1

    cute.arch.sync_threads()
    for iteration in cutlass.range_constexpr(
        TILE_TOKENS * TILE_LATENT // (THREADS * 4)
    ):
        linear_word = iteration * THREADS + tidx
        row = linear_word // (TILE_LATENT // 4)
        word = linear_word % (TILE_LATENT // 4)
        output_i32[page, row, chunk * (TILE_LATENT // 4) + word] = fp8_stage_i32[
            row, word
        ]


@cute.jit
def tma_inplace(
    packed: cute.Tensor,
    scales: cute.Tensor,
    centroids: cute.Tensor,
    output: cute.Tensor,
):
    packed_page_major = cute.make_tensor(
        packed.iterator,
        cute.make_layout(
            (packed.shape[1], packed.shape[2], packed.shape[0]),
            stride=(packed.stride[1], packed.stride[2], packed.stride[0]),
        ),
    )
    packed_i4 = cute.recast_tensor(packed_page_major, cutlass.Int4)
    packed_smem_layout = cute.make_layout(
        (TILE_TOKENS, TILE_LATENT),
        stride=(TILE_LATENT, 1),
    )
    fp8_smem_layout_staged = sm100_utils.make_smem_layout(
        OperandMajorMode.K,
        (TILE_TOKENS, TILE_LATENT),
        cutlass.Float8E4M3FN,
        1,
    )
    fp8_smem_layout = cute.select(fp8_smem_layout_staged, mode=[0, 1])
    tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        packed_i4,
        packed_smem_layout,
        (TILE_TOKENS, TILE_LATENT),
    )

    @cute.struct
    class SharedStorage:
        tma_barrier: cute.struct.MemRange[cutlass.Int64, 1]
        stage: cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Float8E4M3FN, cute.cosize(fp8_smem_layout)
            ],
            1024,
        ]

    tma_inplace_kernel(
        tma_atom,
        tma_tensor,
        scales,
        centroids,
        output,
        packed_smem_layout,
        fp8_smem_layout,
        SharedStorage,
    ).launch(
        grid=(packed_i4.shape[1] // TILE_LATENT, packed_i4.shape[2], 1),
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
    compiled = cute.compile(
        tma_inplace,
        fake(cutlass.Uint8, (pages, TILE_TOKENS, 256), 16),
        fake(cutlass.BFloat16, (pages, TILE_TOKENS), 16),
        fake(cutlass.Float32, (16,), 16),
        fake(cutlass.Float8E4M3FN, (pages, TILE_TOKENS, 512), 16),
        options="--enable-tvm-ffi --opt-level 3",
    )

    torch.manual_seed(23)
    num_pages = int(os.environ.get("TQ4_PROBE_PAGES", "128"))
    benchmark_iterations = int(os.environ.get("TQ4_PROBE_ITERS", "1000"))
    if num_pages <= 0 or benchmark_iterations <= 0:
        raise ValueError("TQ4_PROBE_PAGES and TQ4_PROBE_ITERS must be positive")
    packed = torch.randint(
        0, 256, (num_pages, TILE_TOKENS, 256), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(num_pages, TILE_TOKENS, device="cuda", dtype=torch.bfloat16) * 3
    centroids = torch.linspace(-0.12, 0.11, 16, device="cuda", dtype=torch.float32)
    output = torch.empty(
        (num_pages, TILE_TOKENS, 512), device="cuda", dtype=torch.float8_e4m3fn
    )

    compiled(packed, scales, centroids, output)
    torch.cuda.synchronize()
    indices = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(
        num_pages, TILE_TOKENS, 512
    )
    expected = (centroids[indices.long()] * scales[..., None].float()).to(
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
        f"PASS pages={num_pages} tile={TILE_TOKENS}x{TILE_LATENT} "
        f"output={tuple(output.shape)} launch={elapsed_us:.3f} us"
    )


if __name__ == "__main__":
    main()
