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

"""Falsify the C1 replicated-local mixed-QK-to-P path on SM100.

This is deliberately one step narrower than a complete attention reader.  It
uses the accepted 384-thread/two-CTA resource envelope, produces one M128 x
N128 score stage in each CTA's local TMEM, and assigns exactly 64 row owners
per CTA.  Rank 0
owns score rows 0..63 through CTA-local threads 0..63; rank 1 owns rows 64..127
through threads 64..127.  Each owner performs two N64 max passes followed by
two N64 exp/sum/E4M3 passes and writes its M64 x K128 result into the exact
two-stage transposed-P SMEM operand used by the planned V(TMEM) x P(SMEM) MMA.

Five tiles force the single score stage and both P stages to wrap.  The probe
exports the SMEM operand through its logical layout for an independent host
oracle.  It establishes the first producer/owner composition and register
feasibility only: each CTA deliberately loads its own Q/K/RoPE copy, and the
probe does not issue PV MMA or make a traffic or endpoint-latency claim.
"""

import argparse

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
    make_ptr,
)

THREADS_PER_CTA = 384
TMEM_RETRIEVE_THREADS = 288
CLUSTER_SHAPE_MNK = (2, 1, 1)
TILES = 5
ROWS_PER_CTA = 64
SCORE_ROWS = 128
TOKENS = 128
N64 = 64
P_STAGES = 2

LATENT_K = 512
ROPE_K = 64
MIXED_TILER_MNK = (128, 128, 256)
LATENT_K_TILES = LATENT_K // MIXED_TILER_MNK[2]
ROPE_TILER_MNK = (128, 128, ROPE_K)
SF_VEC_SIZE = 32
SF_DTYPE = cutlass.Float8E8M0FNU
MIXED_B_SMEM_DTYPE = cutlass.Int8
VP_TILER_MNK = (128, 64, 128)
LIVE_SCALE_COLS = 16
SCALE_OFFSET = 64
SCORE_OFFSET = 128
TMEM_ALLOC_COLS = 512
SOFTMAX_SCALE_LOG2 = 0.015625

Q_SMEM_BYTES = 128 * LATENT_K
Q_ROPE_SMEM_BYTES = 128 * ROPE_K
# S1 deliberately uses the reviewed legal two-stage K composition variant.
K_SMEM_BYTES = 128 * MIXED_TILER_MNK[2] * LATENT_K_TILES
K_ROPE_SMEM_BYTES = 128 * ROPE_K
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
    tma_mbar: cute.struct.MemRange[cutlass.Int64, TILES * (LATENT_K_TILES + 1)]
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


def make_mmas():
    mixed = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        cutlass.Float4E2M1FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        SF_DTYPE,
        SF_VEC_SIZE,
        tcgen05.CtaGroup.ONE,
        MIXED_TILER_MNK[:2],
    )
    rope = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        ROPE_TILER_MNK[:2],
    )
    vp = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        VP_TILER_MNK[:2],
        tcgen05.OperandSource.TMEM,
    )
    return mixed, rope, vp


@cute.kernel
def ownership_kernel(
    p_output: cute.Tensor,
    max_output: cute.Tensor,
    sum_output: cute.Tensor,
    owner_output: cute.Tensor,
    token_scale: cute.Tensor,
    mixed_mma: cute.TiledMma,
    rope_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    tma_atom_rope_a: cute.CopyAtom,
    tma_tensor_rope_a: cute.Tensor,
    tma_atom_rope_b: cute.CopyAtom,
    tma_tensor_rope_b: cute.Tensor,
    mixed_a_layout: cute.ComposedLayout,
    mixed_b_layout: cute.ComposedLayout,
    rope_a_layout: cute.ComposedLayout,
    rope_b_layout: cute.ComposedLayout,
    sfa_layout: cute.Layout,
    sfb_layout: cute.Layout,
    sfa_cols: cutlass.Constexpr,
    p_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # The two-stage S1 variant spends the reviewed extra 32 KiB on K so the
    # first producer/owner composition is not confounded by K-stage overwrite
    # timing.  Q/K/RoPE use the exact accepted mixed-MMA SMEM layouts.
    q_smem = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        mixed_a_layout.outer,
        byte_alignment=128,
        swizzle=mixed_a_layout.inner,
    )
    q_rope_smem = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        rope_a_layout.outer,
        byte_alignment=128,
        swizzle=rope_a_layout.inner,
    )
    k_smem = smem.allocate_tensor(
        MIXED_B_SMEM_DTYPE,
        mixed_b_layout.outer,
        byte_alignment=128,
        swizzle=mixed_b_layout.inner,
    )
    k_rope_smem = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        rope_b_layout.outer,
        byte_alignment=128,
        swizzle=rope_b_layout.inner,
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

    # S1 intentionally gives each physical CTA a complete local one-CTA MMA
    # and TMA view.  Multicast is the next independently falsifiable gate.
    mma_tile_coord_v = cutlass.Int32(0)
    cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cutlass.Int32(0))
    g_a_mkl = cute.local_tile(
        tma_tensor_a,
        cute.slice_(MIXED_TILER_MNK, (None, 0, None)),
        (None, None, None),
    )
    g_b_nkl = cute.local_tile(
        tma_tensor_b,
        cute.slice_(MIXED_TILER_MNK, (0, None, None)),
        (None, None, None),
    )
    g_rope_a_mkl = cute.local_tile(
        tma_tensor_rope_a,
        cute.slice_(ROPE_TILER_MNK, (None, 0, None)),
        (None, None, None),
    )
    g_rope_b_nkl = cute.local_tile(
        tma_tensor_rope_b,
        cute.slice_(ROPE_TILER_MNK, (0, None, None)),
        (None, None, None),
    )
    mixed_thr_mma = mixed_mma.get_slice(mma_tile_coord_v)
    t_cg_a = mixed_thr_mma.partition_A(g_a_mkl)
    t_cg_b = mixed_thr_mma.partition_B(g_b_nkl)
    a_cta_layout = cute.make_layout(cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape)
    b_cta_layout = cute.make_layout(cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape)
    t_as_a, t_ag_a = cpasync.tma_partition(
        tma_atom_a,
        cta_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(q_smem, 0, 3),
        cute.group_modes(t_cg_a, 0, 3),
    )
    t_bs_b, t_bg_b = cpasync.tma_partition(
        tma_atom_b,
        cta_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(k_smem, 0, 3),
        cute.group_modes(t_cg_b, 0, 3),
    )
    rope_thr_mma = rope_mma.get_slice(mma_tile_coord_v)
    t_cg_rope_a = rope_thr_mma.partition_A(g_rope_a_mkl)
    t_cg_rope_b = rope_thr_mma.partition_B(g_rope_b_nkl)
    t_as_rope_a, t_ag_rope_a = cpasync.tma_partition(
        tma_atom_rope_a,
        cta_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(q_rope_smem, 0, 3),
        cute.group_modes(t_cg_rope_a, 0, 3),
    )
    t_bs_rope_b, t_bg_rope_b = cpasync.tma_partition(
        tma_atom_rope_b,
        cta_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(k_rope_smem, 0, 3),
        cute.group_modes(t_cg_rope_b, 0, 3),
    )
    t_ag_a = t_ag_a[(None, 0, None, 0)]
    t_bg_b = t_bg_b[(None, 0, None, 0)]
    t_ag_rope_a = t_ag_rope_a[(None, 0, None, 0)]
    t_bg_rope_b = t_bg_rope_b[(None, 0, None, 0)]

    ab_copy_bytes = (
        cute.size_in_bytes(
            cutlass.Float8E4M3FN,
            cute.slice_(q_smem, (None, None, None, 0)),
        )
        + cute.size_in_bytes(
            cutlass.Float4E2M1FN,
            cute.slice_(k_smem, (None, None, None, 0)),
        )
    ) * cute.size(mixed_mma.thr_id.shape)
    rope_copy_bytes = (
        cute.size_in_bytes(
            cutlass.Float8E4M3FN,
            cute.slice_(q_rope_smem, (None, None, None, 0)),
        )
        + cute.size_in_bytes(
            cutlass.Float8E4M3FN,
            cute.slice_(k_rope_smem, (None, None, None, 0)),
        )
    ) * cute.size(rope_mma.thr_id.shape)
    tma_barriers = pipeline.MbarrierArray(
        storage.tma_mbar.data_ptr(),
        TILES * (LATENT_K_TILES + 1),
        (
            pipeline.PipelineOp.TmaLoad,
            pipeline.CooperativeGroup(pipeline.Agent.Thread),
        ),
        tx_count=ab_copy_bytes,
    )
    mma_producer, mma_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, THREADS_PER_CTA
        ),
        barrier_storage=storage.mma_mbar.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
    ).make_participants()

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

    mixed_a = mixed_mma.make_fragment_A(q_smem)
    mixed_b = mixed_mma.make_fragment_B(k_smem)
    mixed_k_blocks = cute.size(mixed_a, mode=[2])
    if cutlass.const_expr(mixed_k_blocks != 8):
        raise ValueError(f"expected eight latent K blocks, got {mixed_k_blocks}")
    acc_fake = mixed_mma.make_fragment_C(
        mixed_mma.partition_shape_C(MIXED_TILER_MNK[:2])
    )
    score_tmem_ptr = tmem_ptr + SCORE_OFFSET
    acc = cute.make_tensor(score_tmem_ptr, acc_fake.layout)
    rope_acc_fake = rope_mma.make_fragment_C(
        rope_mma.partition_shape_C(ROPE_TILER_MNK[:2])
    )
    rope_acc = cute.make_tensor(score_tmem_ptr, rope_acc_fake.layout)
    acc_tile = rope_acc[(None, None), 0, 0]
    rope_a = rope_mma.make_fragment_A(q_rope_smem)
    rope_b = rope_mma.make_fragment_B(k_rope_smem)
    rope_k_blocks = cute.size(rope_a, mode=[2])
    if cutlass.const_expr(rope_k_blocks != 2):
        raise ValueError(f"expected two RoPE K blocks, got {rope_k_blocks}")

    sfa_tmem_ptr = cute.recast_ptr(tmem_ptr + SCALE_OFFSET, dtype=SF_DTYPE)
    t_sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
        mixed_mma,
        MIXED_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfa_layout, (None, None, None, 0)),
    )
    t_sfa = cute.make_tensor(sfa_tmem_ptr, t_sfa_layout)
    sfb_tmem_ptr = cute.recast_ptr(tmem_ptr + SCALE_OFFSET + sfa_cols, dtype=SF_DTYPE)
    t_sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
        mixed_mma,
        MIXED_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfb_layout, (None, None, None, 0)),
    )
    t_sfb = cute.make_tensor(sfb_tmem_ptr, t_sfb_layout)

    # SFA/SFB are exact UE8M0 unity.  Only the first 128 threads participate
    # because the TMEM copy atom owns one logical score row per thread.
    scale_init_tile = cute.make_tensor(
        tmem_ptr + SCALE_OFFSET,
        cute.make_layout((SCORE_ROWS, LIVE_SCALE_COLS), stride=(1 << 16, 1)),
    )
    if tidx < SCORE_ROWS:
        scale_init_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)), cutlass.Float32
        )
        scale_init = tcgen05.make_tmem_copy(scale_init_atom, scale_init_tile)
        scale_init_thr = scale_init.get_slice(tidx)
        scale_coords = cute.make_identity_tensor(scale_init_tile.shape)
        scale_regs_layout = scale_init_thr.partition_S(scale_coords)
        scale_dst = scale_init_thr.partition_D(scale_init_tile)
        scale_regs = cute.make_fragment_like(scale_regs_layout, cutlass.Float32)
        unity_word = cutlass.Uint32(0x7F7F7F7F).bitcast(cutlass.Float32)
        for element in cutlass.range_constexpr(cute.size(scale_regs)):
            scale_regs[element] = unity_word
        cute.copy(scale_init, scale_regs, scale_dst)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.sync_threads()

    # Retain the exact 32 KiB V capacity from L0 without adding a consumer yet.
    if tidx == 0:
        v_smem[0] = cutlass.Int8(0)

    is_owner = (cta_rank == 0 and tidx < ROWS_PER_CTA) or (
        cta_rank == 1 and tidx >= ROWS_PER_CTA and tidx < SCORE_ROWS
    )

    for tile in cutlass.range_constexpr(TILES):
        barrier_base = tile * (LATENT_K_TILES + 1)
        if warp_idx == 9:
            for latent_tile in cutlass.range_constexpr(LATENT_K_TILES):
                barrier_index = barrier_base + latent_tile
                tma_bar_ptr = tma_barriers.get_barrier(barrier_index)
                tma_barriers.arrive_and_expect_tx(barrier_index, ab_copy_bytes)
                cute.copy(
                    tma_atom_a,
                    t_ag_a[(None, latent_tile)],
                    t_as_a[(None, latent_tile)],
                    tma_bar_ptr=tma_bar_ptr,
                )
                cute.copy(
                    tma_atom_b,
                    t_bg_b[(None, latent_tile)],
                    t_bs_b[(None, latent_tile)],
                    tma_bar_ptr=tma_bar_ptr,
                )
            rope_barrier_index = barrier_base + LATENT_K_TILES
            rope_bar_ptr = tma_barriers.get_barrier(rope_barrier_index)
            tma_barriers.arrive_and_expect_tx(rope_barrier_index, rope_copy_bytes)
            cute.copy(
                tma_atom_rope_a,
                t_ag_rope_a[(None, 0)],
                t_as_rope_a[(None, 0)],
                tma_bar_ptr=rope_bar_ptr,
            )
            cute.copy(
                tma_atom_rope_b,
                t_bg_rope_b[(None, 0)],
                t_bs_rope_b[(None, 0)],
                tma_bar_ptr=rope_bar_ptr,
            )

        if warp_idx == 8:
            mma_producer.acquire_and_advance()
            mixed_mma.set(tcgen05.Field.ACCUMULATE, False)
            for latent_tile in cutlass.range_constexpr(LATENT_K_TILES):
                tma_barriers.wait(barrier_base + latent_tile, 0)
                for k_block in cutlass.range(mixed_k_blocks, unroll_full=True):
                    mixed_mma.set(
                        tcgen05.Field.SFA, t_sfa[None, None, k_block].iterator
                    )
                    mixed_mma.set(
                        tcgen05.Field.SFB, t_sfb[None, None, k_block].iterator
                    )
                    cute.gemm(
                        mixed_mma,
                        acc,
                        mixed_a[None, None, k_block, latent_tile],
                        mixed_b[None, None, k_block, latent_tile],
                        acc,
                    )
                    mixed_mma.set(tcgen05.Field.ACCUMULATE, True)
            mma_producer.commit()

        cute.arch.sync_threads()
        latent_full = mma_consumer.wait_and_advance()
        latent_full.release()
        cute.arch.sync_threads()

        # Apply the persistent per-token BF16 TurboQuant scale in place.  The
        # scale varies by tile, which makes any stale score reuse observable.
        mixed_score_tile = cute.make_tensor(
            score_tmem_ptr,
            cute.make_layout((SCORE_ROWS, TOKENS), stride=(1 << 16, 1)),
        )
        if tidx < SCORE_ROWS:
            score_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                cutlass.Float32,
            )
            score_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                cutlass.Float32,
            )
            score_load = tcgen05.make_tmem_copy(score_load_atom, mixed_score_tile)
            score_store = tcgen05.make_tmem_copy(score_store_atom, acc_tile)
            load_thr = score_load.get_slice(tidx)
            store_thr = score_store.get_slice(tidx)
            score_coords = cute.make_identity_tensor(mixed_score_tile.shape)
            load_src = load_thr.partition_S(mixed_score_tile)
            load_regs_layout = load_thr.partition_D(score_coords)
            store_regs_layout = store_thr.partition_S(score_coords)
            store_dst = store_thr.partition_D(acc_tile)
            load_regs = cute.make_fragment_like(load_regs_layout, cutlass.Float32)
            store_regs = cute.make_fragment_like(store_regs_layout, cutlass.Float32)
            cute.copy(score_load, load_src, load_regs)
            cute.arch.fence_view_async_tmem_load()
            for element in cutlass.range_constexpr(cute.size(store_regs)):
                token = store_regs_layout[element][1]
                store_regs[element] = load_regs[element] * token_scale[tile, token].to(
                    cutlass.Float32
                )
            cute.copy(score_store, store_regs, store_dst)
        cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_threads()

        if warp_idx == 8:
            tma_barriers.wait(barrier_base + LATENT_K_TILES, 0)
            mma_producer.acquire_and_advance()
            rope_mma.set(tcgen05.Field.ACCUMULATE, True)
            for k_block in cutlass.range(rope_k_blocks, unroll_full=True):
                cute.gemm(
                    rope_mma,
                    rope_acc,
                    rope_a[None, None, k_block, 0],
                    rope_b[None, None, k_block, 0],
                    rope_acc,
                )
            mma_producer.commit()

        rope_full = mma_consumer.wait_and_advance()
        rope_full.release()
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
        mma_producer.tail()
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def ownership_probe(
    mixed_a_ptr: cute.Pointer,
    mixed_b_ptr: cute.Pointer,
    rope_a_ptr: cute.Pointer,
    rope_b_ptr: cute.Pointer,
    p_output: cute.Tensor,
    max_output: cute.Tensor,
    sum_output: cute.Tensor,
    owner_output: cute.Tensor,
    token_scale: cute.Tensor,
    stream,
):
    mixed_mma, rope_mma, vp_mma = make_mmas()
    g_mixed_a = cute.make_tensor(
        mixed_a_ptr,
        cute.make_ordered_layout((SCORE_ROWS, LATENT_K, 1), order=(1, 0, 2)),
    )
    g_mixed_b = cute.make_tensor(
        mixed_b_ptr,
        cute.make_ordered_layout((TOKENS, LATENT_K, 1), order=(1, 0, 2)),
    )
    g_rope_a = cute.make_tensor(
        rope_a_ptr,
        cute.make_ordered_layout((SCORE_ROWS, ROPE_K, 1), order=(1, 0, 2)),
    )
    g_rope_b = cute.make_tensor(
        rope_b_ptr,
        cute.make_ordered_layout((TOKENS, ROPE_K, 1), order=(1, 0, 2)),
    )

    # This composition gate intentionally models two independent one-CTA
    # producers inside the physical two-CTA cluster.
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)), (mixed_mma.thr_id.shape,)
    )
    mixed_a_layout = sm100_utils.make_smem_layout_a(
        mixed_mma,
        MIXED_TILER_MNK,
        cutlass.Float8E4M3FN,
        LATENT_K_TILES,
    )
    mixed_b_layout = sm100_utils.make_smem_layout_b(
        mixed_mma,
        MIXED_TILER_MNK,
        MIXED_B_SMEM_DTYPE,
        LATENT_K_TILES,
    )
    a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), mixed_mma.thr_id)
    b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mixed_mma.thr_id)
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        g_mixed_a,
        cute.slice_(mixed_a_layout, (None, None, None, 0)),
        MIXED_TILER_MNK,
        mixed_mma,
        cta_layout_vmnk.shape,
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        g_mixed_b,
        cute.slice_(mixed_b_layout, (None, None, None, 0)),
        MIXED_TILER_MNK,
        mixed_mma,
        cta_layout_vmnk.shape,
        internal_type=MIXED_B_SMEM_DTYPE,
    )
    rope_a_layout = sm100_utils.make_smem_layout_a(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    rope_b_layout = sm100_utils.make_smem_layout_b(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    rope_a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), rope_mma.thr_id)
    rope_b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), rope_mma.thr_id)
    tma_atom_rope_a, tma_tensor_rope_a = cute.nvgpu.make_tiled_tma_atom_A(
        rope_a_op,
        g_rope_a,
        cute.slice_(rope_a_layout, (None, None, None, 0)),
        ROPE_TILER_MNK,
        rope_mma,
        cta_layout_vmnk.shape,
    )
    tma_atom_rope_b, tma_tensor_rope_b = cute.nvgpu.make_tiled_tma_atom_B(
        rope_b_op,
        g_rope_b,
        cute.slice_(rope_b_layout, (None, None, None, 0)),
        ROPE_TILER_MNK,
        rope_mma,
        cta_layout_vmnk.shape,
    )
    sfa_layout = blockscaled_utils.make_smem_layout_sfa(
        mixed_mma, MIXED_TILER_MNK, SF_VEC_SIZE, LATENT_K_TILES
    )
    sfb_layout = blockscaled_utils.make_smem_layout_sfb(
        mixed_mma, MIXED_TILER_MNK, SF_VEC_SIZE, LATENT_K_TILES
    )
    sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
        mixed_mma,
        MIXED_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfa_layout, (None, None, None, 0)),
    )
    sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
        mixed_mma,
        MIXED_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfb_layout, (None, None, None, 0)),
    )
    sfa_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(cute.make_ptr(SF_DTYPE, 0), sfa_tmem_layout)
    )
    sfb_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(cute.make_ptr(SF_DTYPE, 0), sfb_tmem_layout)
    )
    if cutlass.const_expr(sfa_cols + sfb_cols != LIVE_SCALE_COLS):
        raise ValueError(f"scale footprint changed: sfa={sfa_cols}, sfb={sfb_cols}")
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
        token_scale,
        mixed_mma,
        rope_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_rope_a,
        tma_tensor_rope_a,
        tma_atom_rope_b,
        tma_tensor_rope_b,
        mixed_a_layout,
        mixed_b_layout,
        rope_a_layout,
        rope_b_layout,
        sfa_layout,
        sfb_layout,
        sfa_cols,
        p_layout,
        cta_layout_vmnk,
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


def expected(
    query: torch.Tensor,
    key: torch.Tensor,
    rope_query: torch.Tensor,
    rope_key: torch.Tensor,
    token_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    latent = query @ key.T
    rope = rope_query @ rope_key.T
    scores = latent.unsqueeze(0) * token_scale.float().unsqueeze(1) + rope.unsqueeze(0)
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
    expected_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    p_expected, max_expected, sum_expected, owners_expected = expected_outputs
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
        make_ptr(
            cutlass.Float8E4M3FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float4E2M1FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        fake(cutlass.Float8E4M3FN, (2, TILES, ROWS_PER_CTA, TOKENS), 16),
        fake(cutlass.Float32, (2, TILES, ROWS_PER_CTA), 16),
        fake(cutlass.Float32, (2, TILES, ROWS_PER_CTA), 16),
        fake(cutlass.Int32, (2, TILES, ROWS_PER_CTA), 16),
        fake(cutlass.BFloat16, (TILES, TOKENS), 16),
        make_fake_stream(),
        options="--enable-tvm-ffi --opt-level 3",
    )
    if args.compile_only:
        print(
            "PASS_C1_M0QP_S1_COMPILE_ONLY "
            f"tiles={TILES} p_stages={P_STAGES} smem_payload={SMEM_PAYLOAD_BYTES}"
        )
        return

    row = torch.arange(SCORE_ROWS, dtype=torch.int64).view(SCORE_ROWS, 1)
    token = torch.arange(TOKENS, dtype=torch.int64).view(TOKENS, 1)
    latent_coordinate = torch.arange(LATENT_K, dtype=torch.int64).view(1, LATENT_K)
    rope_coordinate = torch.arange(ROPE_K, dtype=torch.int64).view(1, ROPE_K)
    query = (
        (
            (
                (row + 1) * (latent_coordinate + 3) * 17
                + row * 37
                + latent_coordinate * 19
            )
            % 257
        )
        % 3
    ).float()
    query = (query - 1.0) * 0.5
    key = (
        (
            (
                (token + 1) * (latent_coordinate + 5) * 23
                + token * 41
                + latent_coordinate * 29
            )
            % 263
        )
        % 3
    ).float()
    key = (key - 1.0) * 0.5
    rope_query = (
        (((row + 3) * (rope_coordinate + 1) * 11 + row * 13) % 127) % 3
    ).float()
    rope_query = (rope_query - 1.0) * 0.5
    rope_key = (
        (((token + 5) * (rope_coordinate + 7) * 19 + token * 17) % 131) % 3
    ).float()
    rope_key = (rope_key - 1.0) * 0.5
    tile = torch.arange(TILES, dtype=torch.int64).view(TILES, 1)
    token_row = torch.arange(TOKENS, dtype=torch.int64).view(1, TOKENS)
    token_scale = (0.75 + ((tile * 7 + token_row * 3) % 17).float() / 32.0).to(
        torch.bfloat16
    )
    expected_outputs = expected(query, key, rope_query, rope_key, token_scale)

    def to_cute_tensor(source: torch.Tensor, dtype):
        source_cuda = source.cuda().contiguous()
        result, _ = cutlass_torch.cute_tensor_like(
            source,
            dtype,
            is_dynamic_layout=True,
            assumed_align=16,
        )
        return cutlass_torch.convert_cute_tensor(
            source_cuda,
            result,
            dtype,
            is_dynamic_layout=True,
        )

    query_cute = to_cute_tensor(
        query.view(SCORE_ROWS, LATENT_K, 1), cutlass.Float8E4M3FN
    )
    key_cute = to_cute_tensor(key.view(TOKENS, LATENT_K, 1), cutlass.Float4E2M1FN)
    rope_query_cute = to_cute_tensor(
        rope_query.view(SCORE_ROWS, ROPE_K, 1), cutlass.Float8E4M3FN
    )
    rope_key_cute = to_cute_tensor(
        rope_key.view(TOKENS, ROPE_K, 1), cutlass.Float8E4M3FN
    )
    token_scale = token_scale.cuda().contiguous()

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
    compiled(
        query_cute.iterator,
        key_cute.iterator,
        rope_query_cute.iterator,
        rope_key_cute.iterator,
        p_output,
        max_output,
        sum_output,
        owner_output,
        token_scale,
        stream,
    )
    torch.cuda.synchronize()
    verify(p_output, max_output, sum_output, owner_output, expected_outputs)

    if not args.skip_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            capture_stream = cuda_driver.CUstream(
                torch.cuda.current_stream().cuda_stream
            )
            compiled(
                query_cute.iterator,
                key_cute.iterator,
                rope_query_cute.iterator,
                rope_key_cute.iterator,
                p_output,
                max_output,
                sum_output,
                owner_output,
                token_scale,
                capture_stream,
            )
        # A graph that captured no launch would leave these sentinels unchanged.
        # Poison after capture so replay cannot pass on stale eager data.
        # Probabilities are in [0, 1], so this sentinel cannot accidentally
        # equal a valid P element if a graph replay only partially writes it.
        p_output.fill_(-1.0)
        max_output.fill_(-1234.0)
        sum_output.fill_(-1234.0)
        owner_output.zero_()
        for _ in range(args.graph_replays):
            graph.replay()
        torch.cuda.synchronize()
        verify(p_output, max_output, sum_output, owner_output, expected_outputs)

    print(
        "PASS_C1_M0QP_S1_MIXED_QK_OWNERSHIP "
        f"tiles={TILES} score_wraps={TILES - 1} p_stages={P_STAGES} "
        f"p_wraps={TILES - P_STAGES} owner_threads=64_per_cta "
        "qk=fp8xfp4 token_scale=bf16 rope_k=64 replicated_local=True "
        "n64_passes=2max+2exp no_warp_exchange=True "
        f"graph_replays={0 if args.skip_graph else args.graph_replays} "
        f"smem_payload={SMEM_PAYLOAD_BYTES}"
    )


if __name__ == "__main__":
    main()
