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

"""Falsify the C1 replicated-local M128 score-to-P owner mapping on SM100.

This is deliberately one step narrower than a complete attention reader.  It
uses the accepted 384-thread/two-CTA resource envelope, initializes one M128 x
N128 score stage in TMEM, and assigns exactly 64 row owners per CTA.  Rank 0
owns score rows 0..63 through CTA-local threads 0..63; rank 1 owns rows 64..127
through threads 64..127.  Each owner performs two N64 max passes followed by
two N64 exp/sum/E4M3 passes and writes its M64 x K128 result into the exact
two-stage transposed-P SMEM operand used by the planned V(TMEM) x P(SMEM) MMA.

Five tiles force the single score stage and both P stages to wrap.  The probe
exports the SMEM operand through its logical layout for an independent host
oracle.  It establishes ownership and register feasibility only: its score is
diagnostically initialized rather than produced by mixed QK, and it does not
issue PV MMA or make an endpoint-latency claim.
"""

import argparse

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

THREADS_PER_CTA = 384
TMEM_RETRIEVE_THREADS = 288
CLUSTER_SHAPE_MNK = (2, 1, 1)
TILES = 5
ROWS_PER_CTA = 64
SCORE_ROWS = 128
TOKENS = 128
N64 = 64
P_STAGES = 2

VP_TILER_MNK = (128, 64, 128)
SCORE_OFFSET = 128
TMEM_ALLOC_COLS = 512
SOFTMAX_SCALE_LOG2 = 0.125

Q_SMEM_BYTES = 128 * 512
Q_ROPE_SMEM_BYTES = 128 * 64
K_SMEM_BYTES = 128 * 256
K_ROPE_SMEM_BYTES = 128 * 64
V_SMEM_BYTES = 2 * 16 * 1024
P_SMEM_BYTES = P_STAGES * 8 * 1024
SMEM_PAYLOAD_BYTES = (
    Q_SMEM_BYTES
    + Q_ROPE_SMEM_BYTES
    + K_SMEM_BYTES
    + K_ROPE_SMEM_BYTES
    + V_SMEM_BYTES
    + P_SMEM_BYTES
)


@cute.struct
class SharedStorage:
    init_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


def make_vp_mma():
    return sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        VP_TILER_MNK[:2],
        tcgen05.OperandSource.TMEM,
    )


@cute.kernel
def ownership_kernel(
    p_output: cute.Tensor,
    max_output: cute.Tensor,
    sum_output: cute.Tensor,
    owner_output: cute.Tensor,
    p_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # Retain the complete accepted L0 payload so register feasibility is tested
    # inside the real shared-memory capacity envelope, not in a tiny kernel.
    q_smem = cutlass.Array(
        cutlass.Int8,
        Q_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    q_rope_smem = cutlass.Array(
        cutlass.Int8,
        Q_ROPE_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    k_smem = cutlass.Array(
        cutlass.Int8,
        K_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    k_rope_smem = cutlass.Array(
        cutlass.Int8,
        K_ROPE_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    v_smem = cutlass.Array(
        cutlass.Int8,
        V_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    p_smem = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        p_layout.outer,
        byte_alignment=128,
        swizzle=p_layout.inner,
    )

    retrieve_barrier = pipeline.NamedBarrier(
        barrier_id=1, num_threads=TMEM_RETRIEVE_THREADS
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=retrieve_barrier,
        allocator_warp_id=8,
        is_two_cta=False,
    )

    # Match production's initialized mbarrier environment before TMEM
    # allocation.  Omitting this makes Compute Sanitizer diagnose the probe
    # environment rather than the intended data path.
    if tidx == 0:
        cute.arch.mbarrier_init(storage.init_mbar.ptr, 1)
    pipeline.pipeline_init_arrive(
        cluster_shape_mn=CLUSTER_SHAPE_MNK[:2], is_relaxed=True
    )
    pipeline.pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE_MNK[:2])

    tmem.allocate(TMEM_ALLOC_COLS)
    if warp_idx <= 8:
        tmem.wait_for_alloc()
    # The named TMEM-retrieve barrier has exactly 288 participants, but every
    # later tile boundary is a full-CTA rendezvous.  Let support warps 9..11
    # join only after allocation so no warp can satisfy a future barrier phase
    # while the 288 TMEM users are still cycling an earlier one.
    cute.arch.sync_threads()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

    # Keep otherwise-unused capacity arrays live in generated host IR.
    if tidx == 0:
        q_smem[0] = cutlass.Int8(0)
        q_rope_smem[0] = cutlass.Int8(0)
        k_smem[0] = cutlass.Int8(0)
        k_rope_smem[0] = cutlass.Int8(0)
        v_smem[0] = cutlass.Int8(0)

    is_owner = (cta_rank == 0 and tidx < ROWS_PER_CTA) or (
        cta_rank == 1 and tidx >= ROWS_PER_CTA and tidx < SCORE_ROWS
    )

    for tile in cutlass.range_constexpr(TILES):
        # Diagnostic producer: every CTA receives the full M128 score, as
        # the replicated-local mixed-QK arm will.  Repetition-16 bounds
        # live score state to one N64 half per thread.
        for half in cutlass.range_constexpr(2):
            score_half = cute.make_tensor(
                tmem_ptr + SCORE_OFFSET + half * N64,
                cute.make_layout((SCORE_ROWS, N64), stride=(1 << 16, 1)),
            )
            score_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)),
                cutlass.Float32,
            )
            score_store = tcgen05.make_tmem_copy(score_store_atom, score_half)
            if tidx < SCORE_ROWS:
                store_thr = score_store.get_slice(tidx)
                score_coords = cute.make_identity_tensor(score_half.shape)
                store_reg_layout = store_thr.partition_S(score_coords)
                store_dst = store_thr.partition_D(score_half)
                store_regs = cute.make_fragment_like(store_reg_layout, cutlass.Float32)
                for element in cutlass.range_constexpr(cute.size(store_regs)):
                    row = store_reg_layout[element][0]
                    token = half * N64 + store_reg_layout[element][1]
                    signature = (
                        tile * 11 + row * 5 + token * 3 + (row * token) % 7
                    ) % 29
                    store_regs[element] = cutlass.Float32(signature - 14) * 0.125
                cute.copy(score_store, store_regs, store_dst)
        cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_threads()

        if is_owner:
            row_max = cutlass.Float32(-1.0e6)

            # First pass: load each half independently and keep only the
            # row maximum.  No warp exchange or peer-TMEM access exists.
            for half in cutlass.range_constexpr(2):
                score_half = cute.make_tensor(
                    tmem_ptr + SCORE_OFFSET + half * N64,
                    cute.make_layout((SCORE_ROWS, N64), stride=(1 << 16, 1)),
                )
                score_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
                    cutlass.Float32,
                )
                score_load = tcgen05.make_tmem_copy(score_load_atom, score_half)
                load_thr = score_load.get_slice(tidx)
                score_coords = cute.make_identity_tensor(score_half.shape)
                load_src = load_thr.partition_S(score_half)
                load_reg_layout = load_thr.partition_D(score_coords)
                load_regs = cute.make_fragment_like(load_reg_layout, cutlass.Float32)
                cute.copy(score_load, load_src, load_regs)
                cute.arch.fence_view_async_tmem_load()
                row_max = load_regs.load().reduce(cute.ReductionOp.MAX, row_max, 0)

            row_sum = cutlass.Float32(0.0)
            stage = tile % P_STAGES
            local_row = tidx - cta_rank * ROWS_PER_CTA

            # Second pass: reload, exponentiate, accumulate the unquantized
            # sum, and publish the exact transposed-P logical coordinate.
            for half in cutlass.range_constexpr(2):
                score_half = cute.make_tensor(
                    tmem_ptr + SCORE_OFFSET + half * N64,
                    cute.make_layout((SCORE_ROWS, N64), stride=(1 << 16, 1)),
                )
                score_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
                    cutlass.Float32,
                )
                score_load = tcgen05.make_tmem_copy(score_load_atom, score_half)
                load_thr = score_load.get_slice(tidx)
                score_coords = cute.make_identity_tensor(score_half.shape)
                load_src = load_thr.partition_S(score_half)
                load_reg_layout = load_thr.partition_D(score_coords)
                load_regs = cute.make_fragment_like(load_reg_layout, cutlass.Float32)
                cute.copy(score_load, load_src, load_regs)
                cute.arch.fence_view_async_tmem_load()
                for element in cutlass.range_constexpr(cute.size(load_regs)):
                    token = half * N64 + load_reg_layout[element][1]
                    probability = cute.math.exp2(
                        (load_regs[element] - row_max) * SOFTMAX_SCALE_LOG2,
                        fastmath=False,
                    )
                    row_sum += probability
                    p_smem[
                        (
                            (local_row, token % 32),
                            0,
                            token // 32,
                            stage,
                        )
                    ] = probability.to(cutlass.Float8E4M3FN)

            max_output[cta_rank, tile, local_row] = row_max
            sum_output[cta_rank, tile, local_row] = row_sum
            owner_output[cta_rank, tile, local_row] = tidx + 1

        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # Diagnostic consumer: read through the same logical operand that
        # PV MMA will consume.  Export before the stage is recycled.
        if is_owner:
            stage = tile % P_STAGES
            local_row = tidx - cta_rank * ROWS_PER_CTA
            for token in cutlass.range_constexpr(TOKENS):
                p_output[cta_rank, tile, local_row, token] = p_smem[
                    (
                        (local_row, token % 32),
                        0,
                        token // 32,
                        stage,
                    )
                ]
        cute.arch.sync_threads()

    cute.arch.sync_threads()
    if warp_idx == 8:
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def ownership_probe(
    p_output: cute.Tensor,
    max_output: cute.Tensor,
    sum_output: cute.Tensor,
    owner_output: cute.Tensor,
    stream,
):
    vp_mma = make_vp_mma()
    p_layout = sm100_utils.make_smem_layout_b(
        vp_mma,
        VP_TILER_MNK,
        cutlass.Float8E4M3FN,
        P_STAGES,
    )
    if cutlass.const_expr(cute.cosize(p_layout) != P_SMEM_BYTES):
        raise ValueError(f"transposed-P footprint changed: {cute.cosize(p_layout)}")
    kernel = ownership_kernel(
        p_output,
        max_output,
        sum_output,
        owner_output,
        p_layout,
    )
    kernel.launch(
        grid=CLUSTER_SHAPE_MNK,
        block=(THREADS_PER_CTA, 1, 1),
        cluster=CLUSTER_SHAPE_MNK,
        min_blocks_per_mp=1,
        stream=stream,
    )


def fake(dtype: type[cutlass.Numeric], shape: tuple[int, ...], align: int):
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )


def expected() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tile = torch.arange(TILES, dtype=torch.int64).view(TILES, 1, 1)
    row = torch.arange(SCORE_ROWS, dtype=torch.int64).view(1, SCORE_ROWS, 1)
    token = torch.arange(TOKENS, dtype=torch.int64).view(1, 1, TOKENS)
    signature = (tile * 11 + row * 5 + token * 3 + (row * token) % 7) % 29
    scores = (signature - 14).to(torch.float32) * 0.125
    row_max = scores.max(dim=-1).values
    probabilities = torch.exp2((scores - row_max.unsqueeze(-1)) * SOFTMAX_SCALE_LOG2)
    p_expected = probabilities.to(torch.float8_e4m3fn)
    sum_expected = probabilities.sum(dim=-1)

    p_expected = torch.stack(
        (p_expected[:, :ROWS_PER_CTA], p_expected[:, ROWS_PER_CTA:]), dim=0
    )
    max_expected = torch.stack(
        (row_max[:, :ROWS_PER_CTA], row_max[:, ROWS_PER_CTA:]), dim=0
    )
    sum_expected = torch.stack(
        (sum_expected[:, :ROWS_PER_CTA], sum_expected[:, ROWS_PER_CTA:]), dim=0
    )
    owners = torch.empty((2, TILES, ROWS_PER_CTA), dtype=torch.int32)
    owners[0] = torch.arange(1, ROWS_PER_CTA + 1, dtype=torch.int32).view(1, -1)
    owners[1] = torch.arange(ROWS_PER_CTA + 1, SCORE_ROWS + 1, dtype=torch.int32).view(
        1, -1
    )
    return p_expected, max_expected, sum_expected, owners


def verify(
    p_output: torch.Tensor,
    max_output: torch.Tensor,
    sum_output: torch.Tensor,
    owner_output: torch.Tensor,
) -> None:
    p_expected, max_expected, sum_expected, owners_expected = expected()
    torch.testing.assert_close(
        p_output.cpu().float(), p_expected.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(max_output.cpu(), max_expected, rtol=0, atol=0)
    torch.testing.assert_close(sum_output.cpu(), sum_expected, rtol=2.0e-6, atol=2.0e-5)
    torch.testing.assert_close(owner_output.cpu(), owners_expected, rtol=0, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-replays", type=int, default=100)
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    compiled = cute.compile(
        ownership_probe,
        fake(cutlass.Float8E4M3FN, (2, TILES, ROWS_PER_CTA, TOKENS), 16),
        fake(cutlass.Float32, (2, TILES, ROWS_PER_CTA), 16),
        fake(cutlass.Float32, (2, TILES, ROWS_PER_CTA), 16),
        fake(cutlass.Int32, (2, TILES, ROWS_PER_CTA), 16),
        make_fake_stream(),
        options="--enable-tvm-ffi --opt-level 3",
    )
    if args.compile_only:
        print(
            "PASS_C1_M0QP_S0_COMPILE_ONLY "
            f"tiles={TILES} p_stages={P_STAGES} smem_payload={SMEM_PAYLOAD_BYTES}"
        )
        return

    p_output = torch.empty(
        (2, TILES, ROWS_PER_CTA, TOKENS),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    max_output = torch.empty(
        (2, TILES, ROWS_PER_CTA), dtype=torch.float32, device="cuda"
    )
    sum_output = torch.empty_like(max_output)
    owner_output = torch.zeros(
        (2, TILES, ROWS_PER_CTA), dtype=torch.int32, device="cuda"
    )

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(p_output, max_output, sum_output, owner_output, stream)
    torch.cuda.synchronize()
    verify(p_output, max_output, sum_output, owner_output)

    if not args.skip_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            capture_stream = cuda_driver.CUstream(
                torch.cuda.current_stream().cuda_stream
            )
            compiled(
                p_output,
                max_output,
                sum_output,
                owner_output,
                capture_stream,
            )
        # A graph that captured no launch would leave these sentinels unchanged.
        # Poison after capture so replay cannot pass on stale eager data.
        p_output.fill_(0.5)
        max_output.fill_(-1234.0)
        sum_output.fill_(-1234.0)
        owner_output.zero_()
        for _ in range(args.graph_replays):
            graph.replay()
        torch.cuda.synchronize()
        verify(p_output, max_output, sum_output, owner_output)

    print(
        "PASS_C1_M0QP_S0_OWNERSHIP "
        f"tiles={TILES} score_wraps={TILES - 1} p_stages={P_STAGES} "
        f"p_wraps={TILES - P_STAGES} owner_threads=64_per_cta "
        "n64_passes=2max+2exp no_warp_exchange=True "
        f"graph_replays={0 if args.skip_graph else args.graph_replays} "
        f"smem_payload={SMEM_PAYLOAD_BYTES}"
    )


if __name__ == "__main__":
    main()
