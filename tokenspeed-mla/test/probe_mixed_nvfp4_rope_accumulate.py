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

"""Probe N8 mixed FP8-query x FP4-key score composition on SM100.

Kimi MLA scores combine a 512-wide latent product with a 64-wide RoPE product.
The N8 path keeps the rotated query in FP8, the persistent latent key in E2M1
FP4, and the hardware UE8M0 block scales at unity. It multiplies the completed
latent score by one nonuniform BF16 key-token scale, then accumulates ordinary
FP8 RoPE into the same TMEM tile. Both operations use the one-CTA M=128 shape
of TokenSpeed's production reader. This is a legality and ownership probe, not
an attention benchmark or an endpoint speed gate.
"""

import os
import statistics

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_ptr

THREADS = 128
CLUSTER_SHAPE_MNK = (1, 1, 1)
OBSERVED_M = 128
# A native SM100 mixed FP8 x FP4 instruction advances 32 latent coordinates.
# One K=256 shared tile therefore issues eight instructions, and the full
# 512-wide latent repeats the tile twice before token scaling.
LATENT_K = 512
MIXED_TILER_MNK = (128, 128, 256)
LATENT_K_TILES = LATENT_K // MIXED_TILER_MNK[2]
SCALE_STAGES = LATENT_K_TILES
ROPE_TILER_MNK = (128, 128, 128)
SF_VEC_SIZE = 32
SF_DTYPE = cutlass.Float8E8M0FNU
TMEM_ALLOC_COLS = 512
SCALE_SCRATCH_COLS = 128
LIVE_SCALE_COLS = 16
# SM100 mxf8f6f4 uses U4_UNPACK_U8 to expand each persistent packed E2M1 value
# into one byte in SMEM. The expanded byte is the instruction's E2M1 operand
# container (not two packed values); persistent K4 stays densely packed in
# global memory.
MIXED_B_SMEM_DTYPE = cutlass.Int8


@cute.struct
class SharedStorage:
    tma_mbar: cute.struct.MemRange[cutlass.Int64, LATENT_K_TILES]
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


def make_tiled_mmas():
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
    return mixed, rope


@cute.kernel
def mixed_accumulate_kernel(
    output: cute.Tensor,
    metadata: cute.Tensor,
    token_scale: cute.Tensor,
    elapsed_ns: cute.Tensor,
    mixed_mma: cute.TiledMma,
    rope_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    mixed_a_layout: cute.ComposedLayout,
    mixed_b_layout: cute.ComposedLayout,
    sfa_layout: cute.Layout,
    sfb_layout: cute.Layout,
    rope_a_layout: cute.ComposedLayout,
    rope_b_layout: cute.ComposedLayout,
    acc_cols: cutlass.Constexpr,
    sfa_cols: cutlass.Constexpr,
    sfb_cols: cutlass.Constexpr,
    total_cols: cutlass.Constexpr,
    cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    mma_tile_coord_v = cutlass.Int32(0)
    cta_rank = cutlass.Int32(0)
    is_leader_cta = True
    cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank)

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    s_mixed_a = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        mixed_a_layout.outer,
        byte_alignment=128,
        swizzle=mixed_a_layout.inner,
    )
    s_mixed_b = smem.allocate_tensor(
        MIXED_B_SMEM_DTYPE,
        mixed_b_layout.outer,
        byte_alignment=128,
        swizzle=mixed_b_layout.inner,
    )
    s_rope_a = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        rope_a_layout.outer,
        byte_alignment=128,
        swizzle=rope_a_layout.inner,
    )
    s_rope_b = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        rope_b_layout.outer,
        byte_alignment=128,
        swizzle=rope_b_layout.inner,
    )

    # Load the exact FP8 query and densely packed FP4 key through CUTLASS's
    # production mixed-precision TMA contract. In particular, B uses
    # U4_UNPACK_U8 rather than a hand-built SMEM approximation.
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
    # One CTA owns the complete M128 score tile.
    thr_mma = mixed_mma.get_slice(mma_tile_coord_v)
    t_cg_a = thr_mma.partition_A(g_a_mkl)
    t_cg_b = thr_mma.partition_B(g_b_nkl)
    a_cta_layout = cute.make_layout(
        cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape
    )
    b_cta_layout = cute.make_layout(
        cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape
    )
    t_as_a, t_ag_a = cpasync.tma_partition(
        tma_atom_a,
        cta_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(s_mixed_a, 0, 3),
        cute.group_modes(t_cg_a, 0, 3),
    )
    t_bs_b, t_bg_b = cpasync.tma_partition(
        tma_atom_b,
        cta_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(s_mixed_b, 0, 3),
        cute.group_modes(t_cg_b, 0, 3),
    )
    if cutlass.const_expr(os.environ.get("TQ_N8_S0_PRINT_LAYOUTS") == "1"):
        print(f"T_AS_A={t_as_a}")
        print(f"T_AG_A={t_ag_a}")
        print(f"T_BS_B={t_bs_b}")
        print(f"T_BG_B={t_bg_b}")
    t_ag_a = t_ag_a[(None, 0, None, 0)]
    t_bg_b = t_bg_b[(None, 0, None, 0)]
    ab_copy_bytes = (
        cute.size_in_bytes(
            cutlass.Float8E4M3FN,
            cute.slice_(s_mixed_a, (None, None, None, 0)),
        )
        + cute.size_in_bytes(
            cutlass.Float4E2M1FN,
            cute.slice_(s_mixed_b, (None, None, None, 0)),
        )
    ) * cute.size(mixed_mma.thr_id.shape)
    tma_barriers = pipeline.MbarrierArray(
        storage.tma_mbar.data_ptr(),
        LATENT_K_TILES,
        (
            pipeline.PipelineOp.TmaLoad,
            pipeline.CooperativeGroup(pipeline.Agent.Thread),
        ),
        tx_count=ab_copy_bytes,
    )

    rope_a_ptr = cute.recast_ptr(s_rope_a.iterator, dtype=cutlass.Float8E4M3FN)
    rope_b_ptr = cute.recast_ptr(s_rope_b.iterator, dtype=cutlass.Float8E4M3FN)
    for element in cutlass.range(tidx, cute.cosize(rope_a_layout.outer), THREADS):
        (rope_a_ptr + element).store(cutlass.Float8E4M3FN(1.0))
    for element in cutlass.range(tidx, cute.cosize(rope_b_layout.outer), THREADS):
        (rope_b_ptr + element).store(cutlass.Float8E4M3FN(1.0))
    cute.arch.sync_threads()

    mma_producer, mma_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.mma_mbar.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
    ).make_participants()
    pipeline.pipeline_init_arrive(
        cluster_shape_mn=CLUSTER_SHAPE_MNK[:2], is_relaxed=True
    )
    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_barrier,
        is_two_cta=False,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )
    tmem.allocate(TMEM_ALLOC_COLS)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    acc_tmem_ptr = tmem_ptr + SCALE_SCRATCH_COLS

    mixed_a = mixed_mma.make_fragment_A(s_mixed_a)
    mixed_b = mixed_mma.make_fragment_B(s_mixed_b)
    if cutlass.const_expr(os.environ.get("TQ_N8_S0_PRINT_LAYOUTS") == "1"):
        print(f"MIXED_A_FRAGMENT={mixed_a}")
        print(f"MIXED_B_FRAGMENT={mixed_b}")
    acc_shape = mixed_mma.partition_shape_C(MIXED_TILER_MNK[:2])
    acc_fake = mixed_mma.make_fragment_C(acc_shape)
    acc = cute.make_tensor(acc_tmem_ptr, acc_fake.layout)
    rope_acc_shape = rope_mma.partition_shape_C(ROPE_TILER_MNK[:2])
    rope_acc_fake = rope_mma.make_fragment_C(rope_acc_shape)
    rope_acc = cute.make_tensor(acc_tmem_ptr, rope_acc_fake.layout)
    if cutlass.const_expr(os.environ.get("TQ_N8_S0_PRINT_LAYOUTS") == "1"):
        print(f"MIXED_ACC={acc}")
        print(f"ROPE_ACC={rope_acc}")
    rope_a = rope_mma.make_fragment_A(s_rope_a)
    rope_b = rope_mma.make_fragment_B(s_rope_b)
    mixed_k_blocks = cute.size(mixed_a, mode=[2])
    if cutlass.const_expr(mixed_k_blocks != 8):
        raise ValueError(f"expected eight latent K blocks, got {mixed_k_blocks}")
    rope_k_blocks = cute.size(rope_a, mode=[2])
    if cutlass.const_expr(rope_k_blocks != 4):
        raise ValueError(f"expected four RoPE K blocks, got {rope_k_blocks}")

    sfa_tmem_ptr = cute.recast_ptr(tmem_ptr, dtype=SF_DTYPE)
    t_sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
        mixed_mma,
        MIXED_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfa_layout, (None, None, None, 0)),
    )
    t_sfa = cute.make_tensor(sfa_tmem_ptr, t_sfa_layout)
    sfb_tmem_ptr = cute.recast_ptr(
        tmem_ptr + sfa_cols, dtype=SF_DTYPE
    )
    t_sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
        mixed_mma,
        MIXED_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfb_layout, (None, None, None, 0)),
    )
    t_sfb = cute.make_tensor(sfb_tmem_ptr, t_sfb_layout)

    if cutlass.const_expr(
        os.environ.get("TQ_N8_S0_POISON_SCALE_PADDING") == "1"
    ):
        # Validation-only negative control: poison the entire aligned scale
        # scratch before overwriting the exact live footprint below. An exact
        # oracle pass then proves MMA does not consume alignment padding.
        scale_padding = cute.make_tensor(
            tmem_ptr,
            cute.make_layout((OBSERVED_M, SCALE_SCRATCH_COLS), stride=(65536, 1)),
        )
        padding_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
        )
        padding_store = tcgen05.make_tmem_copy(padding_store_atom, scale_padding)
        padding_thr = padding_store.get_slice(tidx)
        padding_coords = cute.make_identity_tensor(scale_padding.shape)
        padding_reg_layout = padding_thr.partition_S(padding_coords)
        padding_dst = padding_thr.partition_D(scale_padding)
        padding_regs = cute.make_fragment_like(
            padding_reg_layout, cutlass.Float32
        )
        poison_word = cutlass.Uint32(0x80808080).bitcast(cutlass.Float32)
        for element in cutlass.range_constexpr(cute.size(padding_regs)):
            padding_regs[element] = poison_word
        cute.copy(padding_store, padding_regs, padding_dst)
        cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_threads()

    # SFA and SFB are constants in this formulation. Deterministically fill
    # their exact 16-column TMEM footprint with UE8M0 unity instead of staging
    # redundant scale tensors through SMEM/S2T. Four packed unity bytes form
    # Float32 bit pattern 0x7f7f7f7f; the numerical Float32 value itself is
    # intentionally irrelevant. The score starts at the next required
    # 128-column TMEM alignment boundary.
    scale_init_tile = cute.make_tensor(
        tmem_ptr,
        cute.make_layout((OBSERVED_M, LIVE_SCALE_COLS), stride=(65536, 1)),
    )
    scale_init_store_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)), cutlass.Float32
    )
    scale_init_store = tcgen05.make_tmem_copy(
        scale_init_store_atom, scale_init_tile
    )
    scale_init_thr = scale_init_store.get_slice(tidx)
    scale_init_coords = cute.make_identity_tensor((OBSERVED_M, LIVE_SCALE_COLS))
    scale_init_reg_layout = scale_init_thr.partition_S(scale_init_coords)
    scale_init_dst = scale_init_thr.partition_D(scale_init_tile)
    scale_init_regs = cute.make_fragment_like(
        scale_init_reg_layout, cutlass.Float32
    )
    unity_word = cutlass.Uint32(0x7F7F7F7F).bitcast(cutlass.Float32)
    for element in cutlass.range_constexpr(cute.size(scale_init_regs)):
        scale_init_regs[element] = unity_word
    cute.copy(scale_init_store, scale_init_regs, scale_init_dst)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.sync_threads()
    pipeline.pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE_MNK[:2])
    cute.arch.sync_threads()
    if warp_idx == 0:
        for latent_tile in cutlass.range_constexpr(LATENT_K_TILES):
            tma_bar_ptr = tma_barriers.get_barrier(latent_tile)
            tma_barriers.arrive_and_expect_tx(latent_tile, ab_copy_bytes)
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
    if warp_idx == 0 and is_leader_cta:
        mma_producer.acquire_and_advance()
        mixed_mma.set(tcgen05.Field.ACCUMULATE, False)

        for latent_tile in cutlass.range_constexpr(LATENT_K_TILES):
            tma_barriers.wait(latent_tile, 0)
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

    composition_start_ns = cute.arch.globaltimer()
    latent_full = mma_consumer.wait_and_advance()
    latent_full.release()
    cute.arch.sync_threads()

    # The one-CTA M128 mixed and ordinary-FP8 instructions share the same score
    # layout. Load it to registers, apply the BF16 token scale, and write it back
    # in place before RoPE accumulation; no global score workspace exists.
    acc_tile = rope_acc[(None, None), 0, 0]
    mixed_score_tile = cute.make_tensor(
        acc_tmem_ptr,
        cute.make_layout((OBSERVED_M, 128), stride=(65536, 1)),
    )
    if cutlass.const_expr(os.environ.get("TQ_N8_S0_PRINT_LAYOUTS") == "1"):
        print(f"ACC_TILE={acc_tile}")
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
    )
    tmem_store_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
    )
    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, mixed_score_tile)
    tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, acc_tile)
    thr_load = tmem_load.get_slice(tidx)
    thr_store = tmem_store.get_slice(tidx)
    score_coords = cute.make_identity_tensor(mixed_score_tile.shape)
    t_coords = thr_load.partition_D(score_coords)
    t_tmem_load = thr_load.partition_S(mixed_score_tile)
    t_store_coords = thr_store.partition_S(score_coords)
    t_tmem_store = thr_store.partition_D(acc_tile)
    if cutlass.const_expr(os.environ.get("TQ_N8_S0_PRINT_LAYOUTS") == "1"):
        print(f"TMEM_LOAD_COORDS={t_coords}")
        print(f"TMEM_STORE_COORDS={t_store_coords}")
        print(f"TMEM_LOAD_TILE={t_tmem_load}")
        print(f"TMEM_STORE_TILE={t_tmem_store}")
    score_registers = cute.make_fragment_like(t_coords, cutlass.Float32)
    store_registers = cute.make_fragment_like(t_store_coords, cutlass.Float32)
    if cutlass.const_expr(cute.size(score_registers) != cute.size(store_registers)):
        raise ValueError(
            f"TMEM load/store register sizes do not match: "
            f"load={cute.size(score_registers)} store={cute.size(store_registers)}"
        )
    cute.copy(tmem_load, t_tmem_load, score_registers)
    cute.arch.fence_view_async_tmem_load()
    for element in cutlass.range_constexpr(cute.size(store_registers)):
        token = t_store_coords[element][1]
        store_registers[element] = (
            score_registers[element] * token_scale[token].to(cutlass.Float32)
        )
    cute.copy(tmem_store, store_registers, t_tmem_store)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.sync_threads()

    # Publish the in-place score update before ordinary FP8 RoPE accumulation.
    if warp_idx == 0 and is_leader_cta:
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
    composition_end_ns = cute.arch.globaltimer()
    cute.arch.sync_threads()

    output_matrix = cute.make_tensor(
        output[cta_rank, None, None].iterator,
        cute.make_layout((OBSERVED_M, 128), stride=(128, 1)),
    )
    output_tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
    )
    output_tmem_load = tcgen05.make_tmem_copy(output_tmem_load_atom, acc_tile)
    output_thr_load = output_tmem_load.get_slice(tidx)
    output_t_tmem = output_thr_load.partition_S(acc_tile)
    t_gmem = output_thr_load.partition_D(output_matrix)
    registers = cute.make_fragment_like(t_gmem, cutlass.Float32)
    cute.copy(output_tmem_load, output_t_tmem, registers)
    cute.arch.fence_view_async_tmem_load()
    cute.autovec_copy(registers, t_gmem)
    cute.arch.sync_threads()

    if tidx == 0:
        elapsed_ns[cta_rank] = composition_end_ns - composition_start_ns
    if tidx == 0 and is_leader_cta:
        metadata[0] = acc_cols
        metadata[1] = sfa_cols
        metadata[2] = sfb_cols
        metadata[3] = total_cols
        metadata[4] = TMEM_ALLOC_COLS

    if warp_idx == 0:
        if is_leader_cta:
            mma_producer.tail()
    tmem.relinquish_alloc_permit()
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def mixed_accumulate_probe(
    mixed_a_ptr: cute.Pointer,
    mixed_b_ptr: cute.Pointer,
    output: cute.Tensor,
    metadata: cute.Tensor,
    token_scale: cute.Tensor,
    elapsed_ns: cute.Tensor,
):
    mixed_mma, rope_mma = make_tiled_mmas()
    g_mixed_a = cute.make_tensor(
        mixed_a_ptr,
        cute.make_ordered_layout(
            (256, LATENT_K, 1), order=(1, 0, 2)
        ),
    )
    g_mixed_b = cute.make_tensor(
        mixed_b_ptr,
        cute.make_ordered_layout(
            (128, LATENT_K, 1), order=(1, 0, 2)
        ),
    )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (mixed_mma.thr_id.shape,)
    )
    mixed_a_layout = sm100_utils.make_smem_layout_a(
        mixed_mma, MIXED_TILER_MNK, cutlass.Float8E4M3FN, LATENT_K_TILES
    )
    mixed_b_layout = sm100_utils.make_smem_layout_b(
        mixed_mma, MIXED_TILER_MNK, MIXED_B_SMEM_DTYPE, LATENT_K_TILES
    )
    a_op = sm100_utils.cluster_shape_to_tma_atom_A(
        CLUSTER_SHAPE_MNK[:2], mixed_mma.thr_id
    )
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(
        CLUSTER_SHAPE_MNK[:2], mixed_mma.thr_id
    )
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
    sfa_layout = blockscaled_utils.make_smem_layout_sfa(
        mixed_mma, MIXED_TILER_MNK, SF_VEC_SIZE, SCALE_STAGES
    )
    sfb_layout = blockscaled_utils.make_smem_layout_sfb(
        mixed_mma, MIXED_TILER_MNK, SF_VEC_SIZE, SCALE_STAGES
    )
    if os.environ.get("TQ_N8_S0_PRINT_LAYOUTS") == "1":
        print(f"CTA_LAYOUT_VMNK={cta_layout_vmnk}")
        print(f"TMA_TENSOR_A={tma_tensor_a}")
        print(f"TMA_TENSOR_B={tma_tensor_b}")
        print(f"MIXED_A_LAYOUT={mixed_a_layout}")
        print(f"MIXED_B_LAYOUT={mixed_b_layout}")
        print(
            "AB_COPY_BYTES_PER_CTA="
            f"{cute.size_in_bytes(cutlass.Float8E4M3FN, cute.slice_(mixed_a_layout, (None, None, None, 0))) + cute.size_in_bytes(cutlass.Float4E2M1FN, cute.slice_(mixed_b_layout, (None, None, None, 0)))}"
        )
        print(f"SFA_SMEM_LAYOUT={sfa_layout}")
        print(f"SFB_SMEM_LAYOUT={sfb_layout}")
    rope_a_layout = sm100_utils.make_smem_layout_a(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    rope_b_layout = sm100_utils.make_smem_layout_b(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    acc_fake = mixed_mma.make_fragment_C(
        mixed_mma.partition_shape_C(MIXED_TILER_MNK[:2])
    )
    acc_cols = utils.get_num_tmem_alloc_cols(acc_fake)
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
    live_scale_cols = sfa_cols + sfb_cols
    if cutlass.const_expr(live_scale_cols != LIVE_SCALE_COLS):
        raise ValueError(
            f"scale footprint changed: sfa={sfa_cols}, sfb={sfb_cols}, "
            f"expected_total={LIVE_SCALE_COLS}"
        )
    if cutlass.const_expr(live_scale_cols > SCALE_SCRATCH_COLS):
        raise ValueError(
            f"scale footprint exceeds scratch: live={live_scale_cols}, "
            f"scratch={SCALE_SCRATCH_COLS}"
        )
    if os.environ.get("TQ_N8_S0_PRINT_LAYOUTS") == "1":
        print(f"SFA_TMEM_LAYOUT={sfa_tmem_layout}")
        print(f"SFB_TMEM_LAYOUT={sfb_tmem_layout}")
        print(f"ACC_COLS={acc_cols} SFA_COLS={sfa_cols} SFB_COLS={sfb_cols}")
    total_cols = SCALE_SCRATCH_COLS + acc_cols
    if cutlass.const_expr(total_cols > TMEM_ALLOC_COLS):
        raise ValueError(
            f"TMEM overflow: acc={acc_cols}, sfa={sfa_cols}, "
            f"sfb={sfb_cols}, total={total_cols}"
        )
    kernel = mixed_accumulate_kernel(
        output,
        metadata,
        token_scale,
        elapsed_ns,
        mixed_mma,
        rope_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        mixed_a_layout,
        mixed_b_layout,
        sfa_layout,
        sfb_layout,
        rope_a_layout,
        rope_b_layout,
        acc_cols,
        sfa_cols,
        sfb_cols,
        total_cols,
        cta_layout_vmnk,
    )
    kernel.launch(
        grid=CLUSTER_SHAPE_MNK,
        block=(THREADS, 1, 1),
        min_blocks_per_mp=1,
    )


def main() -> None:
    compiled = cute.compile(
        mixed_accumulate_probe,
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
        make_fake_compact_tensor(
            cutlass.Float32,
            (1, OBSERVED_M, 128),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.Int32,
            (5,),
            stride_order=(0,),
            assumed_align=4,
        ),
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (128,),
            stride_order=(0,),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.Int64,
            (1,),
            stride_order=(0,),
            assumed_align=8,
        ),
        options="--enable-tvm-ffi --opt-level 3",
    )
    if os.environ.get("TQ_N8_S0_COMPILE_ONLY") == "1":
        print(
            "PASS compile_only=True mixed_fp8_query_fp4_key=True "
            "sfa=unity sfb=unity token_scale=bf16_post_mma "
            f"tmem_alloc_cols={TMEM_ALLOC_COLS}"
        )
        return

    output = torch.empty((1, OBSERVED_M, 128), device="cuda", dtype=torch.float32)
    metadata = torch.empty(5, device="cuda", dtype=torch.int32)
    elapsed_ns = torch.empty(1, device="cuda", dtype=torch.int64)

    token_scale = (
        1.0 + torch.arange(128, device="cuda", dtype=torch.float32) / 128.0
    ).to(torch.bfloat16)
    all_rows = torch.arange(256, device="cuda", dtype=torch.int64)
    observed_rows = torch.arange(
        OBSERVED_M, device="cuda", dtype=torch.int64
    )
    token_ids = torch.arange(128, device="cuda", dtype=torch.int64)
    latent_ids = torch.arange(LATENT_K, device="cuda", dtype=torch.int64)
    query_value_full = (
        (
            (all_rows[:, None] + 1)
            * (latent_ids[None, :] + 3)
            * 17
            + all_rows[:, None] * 37
            + latent_ids[None, :] * 19
        )
        % 257
    ) % 3
    query_value = query_value_full[observed_rows]
    key_value = (
        (
            (token_ids[:, None] + 1)
            * (latent_ids[None, :] + 5)
            * 23
            + token_ids[:, None] * 41
            + latent_ids[None, :] * 29
        )
        % 263
    ) % 3
    query_source = query_value_full.float().view(256, LATENT_K, 1)
    key_source = key_value.float().view(128, LATENT_K, 1)
    query_cute, _ = cutlass_torch.cute_tensor_like(
        query_source.cpu(),
        cutlass.Float8E4M3FN,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    query_cute = cutlass_torch.convert_cute_tensor(
        query_source,
        query_cute,
        cutlass.Float8E4M3FN,
        is_dynamic_layout=True,
    )
    key_cute, _ = cutlass_torch.cute_tensor_like(
        key_source.cpu(),
        cutlass.Float4E2M1FN,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    key_cute = cutlass_torch.convert_cute_tensor(
        key_source,
        key_cute,
        cutlass.Float4E2M1FN,
        is_dynamic_layout=True,
    )
    latent_expected = query_value.float() @ key_value.float().T
    if torch.unique(latent_expected, dim=0).shape[0] != 128:
        raise AssertionError("latent oracle does not distinguish all output rows")
    if torch.unique(latent_expected.T, dim=0).shape[0] != 128:
        raise AssertionError("latent oracle does not distinguish all output tokens")
    token_expected = (
        latent_expected * token_scale.float().view(1, 128)
    ).view_as(output)
    for _ in range(4):
        token_expected = token_expected + 32.0

    compiled(
        query_cute.iterator,
        key_cute.iterator,
        output,
        metadata,
        token_scale,
        elapsed_ns,
    )
    torch.cuda.synchronize()
    if not torch.equal(output, token_expected):
        mismatch = output != token_expected
        print(
            "FAIL ownership_case=full_period_row_token_k_tma_unpack "
            f"mismatch_count={int(mismatch.sum().item())} "
            f"mismatch_rows={torch.nonzero(mismatch.any(dim=2), as_tuple=False).cpu().tolist()} "
            f"mismatch_tokens={torch.nonzero(mismatch.any(dim=(0, 1)), as_tuple=False).flatten().cpu().tolist()}"
        )
        for cta, row, token in (
            (0, 0, 0),
            (0, 0, 64),
            (0, 0, 124),
            (0, 0, 125),
            (0, 0, 126),
            (0, 0, 127),
            (0, 1, 127),
        ):
            print(
                f"sample[{cta},{row},{token}]={output[cta, row, token].item()} "
                f"expected={token_expected[cta, row, token].item()}"
            )
    torch.testing.assert_close(output, token_expected, rtol=0, atol=0)

    samples_ns = []
    for _ in range(20):
        compiled(
            query_cute.iterator,
            key_cute.iterator,
            output,
            metadata,
            token_scale,
            elapsed_ns,
        )
        torch.cuda.synchronize()
        samples_ns.append(max(elapsed_ns.cpu().tolist()))

    acc_cols, sfa_cols, sfb_cols, total_cols, allocated_cols = (
        metadata.cpu().tolist()
    )
    print(
        "PASS mixed_fp8_query_fp4_key=True same_score_tile_serialization=True "
        "cta_group=1 ownership_case=full_period_row_token_k_tma_unpack "
        "row_signatures=128 token_signatures=128 ab_stages=2 "
        "u4_unpack_u8=True "
        "sfa=unity sfb=unity "
        f"scale_padding_poisoned={os.environ.get('TQ_N8_S0_POISON_SCALE_PADDING') == '1'} "
        "token_scale=bf16_post_mma "
        "score_workspace_bytes=0 "
        f"acc_cols={acc_cols} sfa_cols={sfa_cols} sfb_cols={sfb_cols} "
        f"total_cols={total_cols} allocated_cols={allocated_cols} "
        f"serialization_median_ns={statistics.median(samples_ns):.1f} "
        f"serialization_max_ns={max(samples_ns)}"
    )


if __name__ == "__main__":
    main()
