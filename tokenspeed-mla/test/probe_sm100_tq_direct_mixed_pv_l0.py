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

"""Legality probe for direct group-two FP8-P x packed-FP4-V on SM100.

The persistent V input has the same token-major packed-E2M1 allocation used by
the accepted mixed-QK path.  A logical mode selection presents it as
latent-by-token to CUTLASS's generated TMA-B composition; no decoded global
shadow, second packed orientation, E2M1-to-E4M3 conversion, narrow LdMatrix,
or TMEM V operand is present.

L0 proves only generated layout/TMA/MMA legality and resource shape.  It does
not establish numerical attention, five-tile correction, endpoint speed,
memory saving, quality, DFlash compatibility, or production readiness.

The overlap and scale-init-tile parameters on this research branch are I1-only
instrumentation.  Accepted L0 evidence remains bound to commit 93cfae24; this
parameterized file has a distinct generated profile and is not an L0
requalification.
"""

import argparse

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr
from cutlass.experimental import primitives as prims

THREADS_PER_CTA = 384
TMEM_RETRIEVE_THREADS = 288
CLUSTER_SHAPE_MNK = (2, 1, 1)
ROWS = 128
TOKENS = 128
LATENT = 512
LATENT_SLICE = 256
LATENT_SLICES = LATENT // LATENT_SLICE

PV_TILER_MNK = (128, 256, 128)
SF_DTYPE = cutlass.Float8E8M0FNU
SF_VEC_SIZE = 32
MIXED_B_SMEM_DTYPE = cutlass.Int8

SCALE_OFFSET = 0
OUTPUT_OFFSET = 64
TMEM_ALLOC_COLS = 256
SMEM_CONTROL_BYTES = 44
SMEM_CONTROL_PADDING_BYTES = (-SMEM_CONTROL_BYTES) % 128
P_SMEM_BYTES = 8 * 1024
V_SMEM_STORAGE_BYTES = 16 * 1024
SMEM_PAYLOAD_BYTES = (
    SMEM_CONTROL_BYTES
    + SMEM_CONTROL_PADDING_BYTES
    + P_SMEM_BYTES
    + V_SMEM_STORAGE_BYTES
)


def make_mixed_pv_mma():
    return sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        cutlass.Float4E2M1FN,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        SF_DTYPE,
        SF_VEC_SIZE,
        tcgen05.CtaGroup.TWO,
        PV_TILER_MNK[:2],
    )


@cute.struct
class SharedStorage:
    init_mbar: cutlass.Int64
    tma_mbar: cutlass.Int64
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def direct_mixed_pv_kernel(
    layout_output: cute.Tensor,
    matrix_output: cute.Tensor,
    mixed_pv_mma: cute.TiledMma,
    tma_atom_p: cute.CopyAtom,
    tma_tensor_p: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    tma_tensor_v: cute.Tensor,
    p_layout: cute.ComposedLayout,
    v_layout: cute.ComposedLayout,
    sfa_layout: cute.Layout,
    sfb_layout: cute.Layout,
    sfa_cols: cutlass.Constexpr,
    sfb_cols: cutlass.Constexpr,
    output_cols: cutlass.Constexpr,
    cta_layout_vmnk: cute.Layout,
    ISSUE_P_TMA: cutlass.Constexpr[int],
    ISSUE_V_TMA: cutlass.Constexpr[int],
    ISSUE_MMA: cutlass.Constexpr[int],
    LATENT_TILE: cutlass.Constexpr[int],
    SFA_EXP: cutlass.Constexpr[int],
    SFB_EXP: cutlass.Constexpr[int],
    OVERLAP_TMA: cutlass.Constexpr[int],
    SCALE_INIT_TILES: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_global, _, _ = cute.arch.block_idx()
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    cluster_index = cute.arch.make_warp_uniform(
        cta_global // CLUSTER_SHAPE_MNK[0]
    )

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    p_smem = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        p_layout.outer,
        byte_alignment=128,
        swizzle=p_layout.inner,
    )
    v_storage = cutlass.Array(
        cutlass.Int8,
        cute.size_in_bytes(MIXED_B_SMEM_DTYPE, v_layout),
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    v_smem_ptr = cute.make_ptr(
        MIXED_B_SMEM_DTYPE,
        v_storage.data_ptr().ir_value(),
        cute.AddressSpace.smem,
        assumed_align=128,
    )
    v_smem = cute.make_tensor(
        cute.recast_ptr(
            v_smem_ptr,
            swizzle_=v_layout.inner,
            dtype=MIXED_B_SMEM_DTYPE,
        ),
        v_layout.outer,
    )

    mma_tile_coord = cta_rank
    cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank)
    g_p_mkl = cute.local_tile(
        tma_tensor_p,
        cute.slice_(PV_TILER_MNK, (None, 0, None)),
        (None, None, None),
    )
    g_v_nkl = cute.local_tile(
        tma_tensor_v,
        cute.slice_(PV_TILER_MNK, (0, None, None)),
        (None, None, None),
    )
    mixed_thr_mma = mixed_pv_mma.get_slice(mma_tile_coord)
    t_cg_p = mixed_thr_mma.partition_A(g_p_mkl)
    t_cg_v = mixed_thr_mma.partition_B(g_v_nkl)
    a_cta_layout = cute.make_layout(
        cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape
    )
    b_cta_layout = cute.make_layout(
        cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape
    )
    t_ps_p, t_pg_p = cpasync.tma_partition(
        tma_atom_p,
        cta_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(p_smem, 0, 3),
        cute.group_modes(t_cg_p, 0, 3),
    )
    t_vs_v, t_vg_v = cpasync.tma_partition(
        tma_atom_v,
        cta_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(v_smem, 0, 3),
        cute.group_modes(t_cg_v, 0, 3),
    )
    # Select the single K128 block and one of the two N256 latent slices.
    t_pg_p = t_pg_p[(None, 0, None, None)]
    t_vg_v = t_vg_v[(None, LATENT_TILE, None, None)]

    p_copy_bytes = cute.size_in_bytes(
        cutlass.Float8E4M3FN,
        cute.slice_(p_smem, (None, None, None, 0)),
    ) * cute.size(mixed_pv_mma.thr_id.shape)
    v_copy_bytes = cute.size_in_bytes(
        cutlass.Float4E2M1FN,
        cute.slice_(v_smem, (None, None, None, 0)),
    ) * cute.size(mixed_pv_mma.thr_id.shape)
    copy_bytes = ISSUE_P_TMA * p_copy_bytes + ISSUE_V_TMA * v_copy_bytes
    tma_barriers = pipeline.MbarrierArray(
        storage.tma_mbar.ptr,
        1,
        (
            pipeline.PipelineOp.TmaLoad,
            pipeline.CooperativeGroup(pipeline.Agent.Thread),
        ),
        tx_count=copy_bytes,
    )
    mma_producer, mma_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, THREADS_PER_CTA * CLUSTER_SHAPE_MNK[0]
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
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )
    if tidx == 0:
        cute.arch.mbarrier_init(storage.init_mbar.ptr, 1)
    pipeline.pipeline_init_arrive(
        cluster_shape_mn=CLUSTER_SHAPE_MNK[:2], is_relaxed=True
    )
    pipeline.pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE_MNK[:2])
    tmem.allocate(TMEM_ALLOC_COLS)
    if warp_idx <= 8:
        tmem.wait_for_alloc()
    cute.arch.sync_threads()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    prims.fence_mbarrier_init()
    cute.arch.sync_threads()

    # I1 overlap arms let the elected TMA producer start the disjoint P/V SMEM
    # transaction while the first four warps initialize scale TMEM.  Barrier
    # initialization and its CTA fence remain before issue, and the elected
    # MMA warp still waits for this transaction after scale publication.
    if warp_idx == 9 and cutlass.const_expr(
        OVERLAP_TMA == 1 and (ISSUE_P_TMA == 1 or ISSUE_V_TMA == 1)
    ):
        tma_bar_ptr = tma_barriers.get_barrier(0)
        tma_barriers.arrive_and_expect_tx(0, copy_bytes)
        if cutlass.const_expr(ISSUE_P_TMA == 1):
            cute.copy(
                tma_atom_p,
                t_pg_p[(None, 0, cluster_index)],
                t_ps_p[(None, 0)],
                tma_bar_ptr=tma_bar_ptr,
            )
        if cutlass.const_expr(ISSUE_V_TMA == 1):
            cute.copy(
                tma_atom_v,
                t_vg_v[(None, 0, cluster_index)],
                t_vs_v[(None, 0)],
                tma_bar_ptr=tma_bar_ptr,
            )

    p_operand = mixed_pv_mma.make_fragment_A(p_smem)
    v_operand = mixed_pv_mma.make_fragment_B(v_smem)
    k_blocks = cute.size(p_operand, mode=[2])
    if cutlass.const_expr(k_blocks != cute.size(v_operand, mode=[2])):
        raise ValueError(
            f"mixed PV K-block mismatch: A={k_blocks} "
            f"B={cute.size(v_operand, mode=[2])}"
        )
    acc_fake = mixed_pv_mma.make_fragment_C(
        mixed_pv_mma.partition_shape_C(PV_TILER_MNK[:2])
    )
    accumulator = cute.make_tensor(tmem_ptr + OUTPUT_OFFSET, acc_fake.layout)

    sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
        mixed_pv_mma,
        PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfa_layout, (None, None, None, 0)),
    )
    sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
        mixed_pv_mma,
        PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfb_layout, (None, None, None, 0)),
    )
    t_sfa = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + SCALE_OFFSET, dtype=SF_DTYPE),
        sfa_tmem_layout,
    )
    t_sfb = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + SCALE_OFFSET + sfa_cols, dtype=SF_DTYPE),
        sfb_tmem_layout,
    )

    scale_cols = sfa_cols + sfb_cols
    if tidx < ROWS:
        scale_init_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)),
            cutlass.Float32,
        )
        sfa_byte = 0x7F + SFA_EXP
        sfb_byte = 0x7F + SFB_EXP
        sfa_word = cutlass.Uint32(sfa_byte * 0x01010101).bitcast(
            cutlass.Float32
        )
        sfb_word = cutlass.Uint32(sfb_byte * 0x01010101).bitcast(
            cutlass.Float32
        )
        poison_word = cutlass.Uint32(0x81818181).bitcast(cutlass.Float32)
        # Full controls initialize the complete 0..63 defensive reserve.
        # Compact I1 arms initialize the one 16-column tile containing every
        # live SFA/SFB address and retain poison in its four unclaimed columns.
        for scale_block in cutlass.range_constexpr(SCALE_INIT_TILES):
            scale_init_tile = cute.make_tensor(
                tmem_ptr + SCALE_OFFSET + scale_block * 16,
                cute.make_layout((ROWS, 16), stride=(1 << 16, 1)),
            )
            scale_init = tcgen05.make_tmem_copy(
                scale_init_atom, scale_init_tile
            )
            scale_init_thr = scale_init.get_slice(tidx)
            scale_coords = cute.make_identity_tensor(scale_init_tile.shape)
            scale_regs_layout = scale_init_thr.partition_S(scale_coords)
            scale_dst = scale_init_thr.partition_D(scale_init_tile)
            scale_regs = cute.make_fragment_like(
                scale_regs_layout, cutlass.Float32
            )
            for element in cutlass.range_constexpr(cute.size(scale_regs)):
                scale_col = scale_block * 16 + element
                if cutlass.const_expr(scale_col < sfa_cols):
                    scale_regs[element] = sfa_word
                elif cutlass.const_expr(scale_col < scale_cols):
                    scale_regs[element] = sfb_word
                else:
                    scale_regs[element] = poison_word
            cute.copy(scale_init, scale_regs, scale_dst)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.sync_threads()
    # The elected rank-0 group-two MMA consumes scale TMEM from both SMs.
    # Publish each CTA's completed TMEM stores before the leader can issue;
    # a CTA-local barrier or the elected TMA completion does not order rank 1.
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    if tidx == 0 and cluster_index == 0:
        layout_output[cta_rank, 0] = cta_rank + 1
        layout_output[cta_rank, 1] = cute.size_in_bytes(
            cutlass.Float8E4M3FN, p_smem
        )
        layout_output[cta_rank, 2] = cute.size_in_bytes(
            cutlass.Float4E2M1FN, v_smem
        )
        layout_output[cta_rank, 3] = sfa_cols
        layout_output[cta_rank, 4] = sfb_cols
        layout_output[cta_rank, 5] = k_blocks
        layout_output[cta_rank, 6] = output_cols
        layout_output[cta_rank, 7] = OUTPUT_OFFSET
        layout_output[cta_rank, 8] = TMEM_ALLOC_COLS
        layout_output[cta_rank, 9] = copy_bytes
        layout_output[cta_rank, 10] = LATENT_TILE
        layout_output[cta_rank, 11] = SFA_EXP
        layout_output[cta_rank, 12] = SFB_EXP

    if warp_idx == 9 and cutlass.const_expr(
        OVERLAP_TMA == 0 and (ISSUE_P_TMA == 1 or ISSUE_V_TMA == 1)
    ):
        tma_bar_ptr = tma_barriers.get_barrier(0)
        tma_barriers.arrive_and_expect_tx(0, copy_bytes)
        if cutlass.const_expr(ISSUE_P_TMA == 1):
            cute.copy(
                tma_atom_p,
                t_pg_p[(None, 0, cluster_index)],
                t_ps_p[(None, 0)],
                tma_bar_ptr=tma_bar_ptr,
            )
        if cutlass.const_expr(ISSUE_V_TMA == 1):
            cute.copy(
                tma_atom_v,
                t_vg_v[(None, 0, cluster_index)],
                t_vs_v[(None, 0)],
                tma_bar_ptr=tma_bar_ptr,
            )

    if warp_idx == 8 and cta_rank == 0:
        # Generated UTMALDG.2CTA owns one elected-CTA completion barrier for
        # the complete two-SM transaction.  This differs from S4-C0's two
        # manually issued per-CTA TensorMaps, which required local waits plus
        # explicit cluster publication.
        if cutlass.const_expr(ISSUE_P_TMA == 1 or ISSUE_V_TMA == 1):
            tma_barriers.wait(0, 0)
        mma_producer.acquire_and_advance()
        if cutlass.const_expr(ISSUE_MMA == 1):
            mixed_pv_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(k_blocks, unroll_full=True):
                mixed_pv_mma.set(
                    tcgen05.Field.SFA, t_sfa[None, None, k_block].iterator
                )
                mixed_pv_mma.set(
                    tcgen05.Field.SFB, t_sfb[None, None, k_block].iterator
                )
                cute.gemm(
                    mixed_pv_mma,
                    accumulator,
                    p_operand[None, None, k_block, 0],
                    v_operand[None, None, k_block, 0],
                    accumulator,
                )
                mixed_pv_mma.set(tcgen05.Field.ACCUMULATE, True)
        mma_producer.commit()

    mma_full = mma_consumer.wait_and_advance()
    mma_full.release()
    cute.arch.sync_threads()

    if tidx < 128 and cutlass.const_expr(ISSUE_MMA == 1):
        t_acc = accumulator[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            cutlass.Float32,
        )
        tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, t_acc)
        thr_load = tmem_load.get_slice(tidx)
        g_output = cute.make_tensor(
            matrix_output.iterator
            + cta_global * LATENT_SLICE * (ROWS // CLUSTER_SHAPE_MNK[0]),
            cute.make_layout(
                (ROWS // CLUSTER_SHAPE_MNK[0], LATENT_SLICE),
                stride=(1, ROWS // CLUSTER_SHAPE_MNK[0]),
            ),
        )
        t_tmem = thr_load.partition_S(t_acc)
        t_gmem = thr_load.partition_D(g_output)
        r_acc = cute.make_fragment_like(t_gmem, cutlass.Float32)
        cute.copy(tmem_load, t_tmem, r_acc)
        cute.arch.fence_view_async_tmem_load()
        cute.autovec_copy(r_acc, t_gmem)
    cute.arch.sync_threads()

    if warp_idx == 8 and cta_rank == 0:
        mma_producer.tail()
    cute.arch.sync_threads()
    if warp_idx == 8:
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def direct_mixed_pv_probe(
    p_ptr: cute.Pointer,
    packed_v_ptr: cute.Pointer,
    layout_output: cute.Tensor,
    matrix_output: cute.Tensor,
    CLUSTERS: cutlass.Constexpr[int],
    ISSUE_P_TMA: cutlass.Constexpr[int],
    ISSUE_V_TMA: cutlass.Constexpr[int],
    ISSUE_MMA: cutlass.Constexpr[int],
    LATENT_TILE: cutlass.Constexpr[int],
    SFA_EXP: cutlass.Constexpr[int],
    SFB_EXP: cutlass.Constexpr[int],
    OVERLAP_TMA: cutlass.Constexpr[int],
    SCALE_INIT_TILES: cutlass.Constexpr[int],
    stream,
):
    mixed_pv_mma = make_mixed_pv_mma()
    g_p = cute.make_tensor(
        p_ptr,
        cute.make_ordered_layout((ROWS, TOKENS, CLUSTERS), order=(1, 0, 2)),
    )
    g_packed_v = cute.make_tensor(
        packed_v_ptr,
        cute.make_ordered_layout((TOKENS, LATENT, CLUSTERS), order=(1, 0, 2)),
    )
    g_packed_v_transpose = cute.make_tensor(
        g_packed_v.iterator,
        cute.select(g_packed_v.layout, mode=[1, 0, 2]),
    )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (mixed_pv_mma.thr_id.shape,)
    )
    p_layout = sm100_utils.make_smem_layout_a(
        mixed_pv_mma,
        PV_TILER_MNK,
        cutlass.Float8E4M3FN,
        1,
    )
    v_layout = sm100_utils.make_smem_layout_b(
        mixed_pv_mma,
        PV_TILER_MNK,
        MIXED_B_SMEM_DTYPE,
        1,
    )
    if cutlass.const_expr(
        cute.size_in_bytes(cutlass.Float8E4M3FN, p_layout) != P_SMEM_BYTES
    ):
        raise ValueError(
            "mixed PV P SMEM footprint changed: "
            f"{cute.size_in_bytes(cutlass.Float8E4M3FN, p_layout)} "
            f"!= {P_SMEM_BYTES}"
        )
    if cutlass.const_expr(
        cute.size_in_bytes(MIXED_B_SMEM_DTYPE, v_layout)
        != V_SMEM_STORAGE_BYTES
    ):
        raise ValueError(
            "mixed PV V SMEM storage changed: "
            f"{cute.size_in_bytes(MIXED_B_SMEM_DTYPE, v_layout)} "
            f"!= {V_SMEM_STORAGE_BYTES}"
        )
    a_op = sm100_utils.cluster_shape_to_tma_atom_A(
        CLUSTER_SHAPE_MNK[:2], mixed_pv_mma.thr_id
    )
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(
        CLUSTER_SHAPE_MNK[:2], mixed_pv_mma.thr_id
    )
    tma_atom_p, tma_tensor_p = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        g_p,
        cute.slice_(p_layout, (None, None, None, 0)),
        PV_TILER_MNK,
        mixed_pv_mma,
        cta_layout_vmnk.shape,
    )
    tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        g_packed_v_transpose,
        cute.slice_(v_layout, (None, None, None, 0)),
        PV_TILER_MNK,
        mixed_pv_mma,
        cta_layout_vmnk.shape,
        internal_type=MIXED_B_SMEM_DTYPE,
    )
    sfa_layout = blockscaled_utils.make_smem_layout_sfa(
        mixed_pv_mma, PV_TILER_MNK, SF_VEC_SIZE, 1
    )
    sfb_layout = blockscaled_utils.make_smem_layout_sfb(
        mixed_pv_mma, PV_TILER_MNK, SF_VEC_SIZE, 1
    )
    sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
        mixed_pv_mma,
        PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfa_layout, (None, None, None, 0)),
    )
    sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
        mixed_pv_mma,
        PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfb_layout, (None, None, None, 0)),
    )
    sfa_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(cute.make_ptr(SF_DTYPE, 0), sfa_tmem_layout)
    )
    sfb_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(cute.make_ptr(SF_DTYPE, 0), sfb_tmem_layout)
    )
    output_fragment = mixed_pv_mma.make_fragment_C(
        mixed_pv_mma.partition_shape_C(PV_TILER_MNK[:2])
    )
    output_cols = utils.get_num_tmem_alloc_cols(output_fragment)
    if cutlass.const_expr(SCALE_OFFSET + sfa_cols + sfb_cols > OUTPUT_OFFSET):
        raise ValueError(
            f"mixed PV scales overlap output: {sfa_cols}+{sfb_cols} "
            f"> {OUTPUT_OFFSET}"
        )
    if cutlass.const_expr(OUTPUT_OFFSET + output_cols > TMEM_ALLOC_COLS):
        raise ValueError(
            f"mixed PV output exceeds TMEM: {OUTPUT_OFFSET}+{output_cols} "
            f"> {TMEM_ALLOC_COLS}"
        )
    if cutlass.const_expr(
        SCALE_INIT_TILES not in (1, OUTPUT_OFFSET // 16)
    ):
        raise ValueError(
            f"I1 scale-init tiles must be 1 or {OUTPUT_OFFSET // 16}"
        )
    if cutlass.const_expr(sfa_cols + sfb_cols > SCALE_INIT_TILES * 16):
        raise ValueError(
            f"I1 live scales exceed initialized columns: "
            f"{sfa_cols}+{sfb_cols} > {SCALE_INIT_TILES * 16}"
        )
    print(f"S4_C1_MIXED_PV_THR_ID={mixed_pv_mma.thr_id}")
    print(f"S4_C1_CTA_LAYOUT_VMNK={cta_layout_vmnk}")
    print(f"S4_C1_P_LAYOUT={p_layout}")
    print(f"S4_C1_V_LAYOUT={v_layout}")
    print(
        "S4_C1_RESOURCE_PLAN="
        f"p_bytes={cute.size_in_bytes(cutlass.Float8E4M3FN, p_layout)} "
        f"v_smem_bytes={cute.size_in_bytes(MIXED_B_SMEM_DTYPE, v_layout)} "
        f"v_tma_bytes={cute.size_in_bytes(cutlass.Float4E2M1FN, v_layout)} "
        f"sfa_cols={sfa_cols} sfb_cols={sfb_cols} output_cols={output_cols} "
        f"overlap_tma={OVERLAP_TMA} scale_init_tiles={SCALE_INIT_TILES}"
    )
    kernel = direct_mixed_pv_kernel(
        layout_output,
        matrix_output,
        mixed_pv_mma,
        tma_atom_p,
        tma_tensor_p,
        tma_atom_v,
        tma_tensor_v,
        p_layout,
        v_layout,
        sfa_layout,
        sfb_layout,
        sfa_cols,
        sfb_cols,
        output_cols,
        cta_layout_vmnk,
        ISSUE_P_TMA,
        ISSUE_V_TMA,
        ISSUE_MMA,
        LATENT_TILE,
        SFA_EXP,
        SFB_EXP,
        OVERLAP_TMA,
        SCALE_INIT_TILES,
    )
    kernel.launch(
        grid=(CLUSTER_SHAPE_MNK[0] * CLUSTERS, 1, 1),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--tma-arm", choices=("both", "p", "v", "none"), default="both"
    )
    parser.add_argument("--no-mma", action="store_true")
    parser.add_argument(
        "--latent-tile", type=int, choices=range(LATENT_SLICES), default=0
    )
    parser.add_argument("--graph-replays", type=int, default=0)
    parser.add_argument("--sfa-exp", type=int, choices=(0, 1), default=0)
    parser.add_argument("--sfb-exp", type=int, choices=(0, 1), default=0)
    parser.add_argument("--overlap-tma", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--scale-init-tiles", type=int, choices=(1, 4), default=4
    )
    args = parser.parse_args()
    if args.clusters < 1:
        parser.error("--clusters must be positive")
    if args.tma_arm != "both" and not args.no_mma:
        parser.error("a partial/disabled TMA arm requires --no-mma")
    if args.graph_replays < 0:
        parser.error("--graph-replays must be non-negative")
    if args.graph_replays and args.no_mma:
        parser.error("--graph-replays requires the MMA arm")
    if args.sfa_exp and args.sfb_exp:
        parser.error("test SFA and SFB non-unity pointers independently")

    ctas = CLUSTER_SHAPE_MNK[0] * args.clusters
    compiled = cute.compile(
        direct_mixed_pv_probe,
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
        fake(cutlass.Int32, (CLUSTER_SHAPE_MNK[0], 13), 16),
        fake(
            cutlass.Float32,
            (ctas, LATENT_SLICE, ROWS // CLUSTER_SHAPE_MNK[0]),
            16,
        ),
        args.clusters,
        int(args.tma_arm in ("both", "p")),
        int(args.tma_arm in ("both", "v")),
        0 if args.no_mma else 1,
        args.latent_tile,
        args.sfa_exp,
        args.sfb_exp,
        args.overlap_tma,
        args.scale_init_tiles,
        make_fake_stream(),
        options="--enable-tvm-ffi --opt-level 3",
    )
    if args.compile_only:
        print(
            "PASS_S4_C1_L0_COMPILE_ONLY "
            f"clusters={args.clusters} threads={THREADS_PER_CTA} "
            f"cluster_shape={CLUSTER_SHAPE_MNK} "
            f"smem_payload={SMEM_PAYLOAD_BYTES} "
            f"tma_arm={args.tma_arm} issue_mma={int(not args.no_mma)} "
            f"overlap_tma={args.overlap_tma} "
            f"scale_init_tiles={args.scale_init_tiles}"
        )
        return

    row = torch.arange(ROWS, dtype=torch.int64).view(ROWS, 1)
    token = torch.arange(TOKENS, dtype=torch.int64).view(1, TOKENS)
    latent = torch.arange(LATENT, dtype=torch.int64).view(1, LATENT)
    p_base = (
        ((((row + 1) * (token + 3) * 17) % 7) - 3).float() * 0.25
    )
    v_base = (
        ((((token.T + 5) * (latent + 7) * 19 + latent * 3) % 5) - 2).float()
        * 0.5
    )

    # Give every tested cluster a unique, exactly representable P/V signature
    # without recomputing a full reference GEMM per cluster.  The low seven
    # cluster bits select a Walsh row sign for P; higher bits select a Walsh
    # latent sign for V.  A cluster-coordinate alias therefore changes at
    # least one complete output row or latent column through 65,536 clusters.
    cluster_ids = torch.arange(args.clusters, dtype=torch.int64)
    row_masks = torch.bitwise_and(
        cluster_ids.remainder(ROWS).view(-1, 1),
        torch.arange(ROWS, dtype=torch.int64).view(1, -1),
    )
    row_parity = torch.zeros_like(row_masks)
    for bit in range(7):
        row_parity.bitwise_xor_((row_masks >> bit) & 1)
    row_sign = (1 - 2 * row_parity).float()

    latent_masks = torch.bitwise_and(
        torch.div(cluster_ids, ROWS, rounding_mode="floor").view(-1, 1),
        torch.arange(LATENT, dtype=torch.int64).view(1, -1),
    )
    latent_parity = torch.zeros_like(latent_masks)
    for bit in range(9):
        latent_parity.bitwise_xor_((latent_masks >> bit) & 1)
    latent_sign = (1 - 2 * latent_parity).float()

    p_reference = p_base.unsqueeze(0) * row_sign.unsqueeze(2)
    v_reference = v_base.unsqueeze(0) * latent_sign.unsqueeze(1)
    p = p_reference * (0.5 ** (args.sfa_exp + args.sfb_exp))
    p_cute = to_cute_tensor(
        p,
        cutlass.Float8E4M3FN,
    )
    v_cute = to_cute_tensor(
        v_reference,
        cutlass.Float4E2M1FN,
    )
    layout_output = torch.zeros(
        (CLUSTER_SHAPE_MNK[0], 13), dtype=torch.int32, device="cuda"
    )
    matrix_output = torch.full(
        (ctas, LATENT_SLICE, ROWS // CLUSTER_SHAPE_MNK[0]),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        p_cute.iterator,
        v_cute.iterator,
        layout_output,
        matrix_output,
        stream,
    )
    torch.cuda.synchronize()

    latent_begin = args.latent_tile * LATENT_SLICE
    latent_end = latent_begin + LATENT_SLICE
    expected_base = p_base @ v_base[:, latent_begin:latent_end]
    expected_rows = (
        expected_base.unsqueeze(0)
        * row_sign.unsqueeze(2)
        * latent_sign[:, latent_begin:latent_end].unsqueeze(1)
    )
    expected_pair = torch.stack(
        (
            expected_rows[:, : ROWS // 2, :].permute(0, 2, 1),
            expected_rows[:, ROWS // 2 :, :].permute(0, 2, 1),
        ),
        dim=1,
    )
    expected = expected_pair.reshape(
        ctas, LATENT_SLICE, ROWS // CLUSTER_SHAPE_MNK[0]
    ).contiguous().cuda()

    def verify_output(label: str) -> None:
        if args.no_mma:
            return
        if torch.equal(matrix_output, expected):
            return
        actual_cpu = matrix_output.cpu()
        expected_cpu = expected.cpu()
        mismatch = (actual_cpu != expected_cpu).nonzero()[0].tolist()
        cta, latent_idx, row_idx = mismatch
        max_abs = torch.max(torch.abs(actual_cpu - expected_cpu)).item()
        raise AssertionError(
            f"{label} mixed PV oracle mismatch at "
            f"cta={cta} latent={latent_idx + latent_begin} row={row_idx}: "
            f"actual={actual_cpu[cta, latent_idx, row_idx].item()} "
            f"expected={expected_cpu[cta, latent_idx, row_idx].item()} "
            f"max_abs={max_abs}"
        )

    verify_output("eager")
    if not torch.all(layout_output[:, 0].cpu() == torch.tensor([1, 2])):
        raise AssertionError(f"CTA layout sentinel failed: {layout_output.cpu()}")
    if not torch.all(
        layout_output[:, 10].cpu()
        == torch.full((CLUSTER_SHAPE_MNK[0],), args.latent_tile)
    ):
        raise AssertionError(
            f"latent-tile sentinel failed: {layout_output.cpu()}"
        )
    if not torch.all(
        layout_output[:, 11:13].cpu()
        == torch.tensor([args.sfa_exp, args.sfb_exp]).view(1, 2)
    ):
        raise AssertionError(f"scale sentinel failed: {layout_output.cpu()}")

    if args.graph_replays:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            capture_stream = cuda_driver.CUstream(
                torch.cuda.current_stream().cuda_stream
            )
            compiled(
                p_cute.iterator,
                v_cute.iterator,
                layout_output,
                matrix_output,
                capture_stream,
            )
        for replay in range(args.graph_replays):
            matrix_output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            verify_output(f"graph replay {replay + 1}")
        print(
            "PASS_S4_C1_L0_GRAPH "
            f"clusters={args.clusters} latent_tile={args.latent_tile} "
            f"sfa_exp={args.sfa_exp} sfb_exp={args.sfb_exp} "
            f"overlap_tma={args.overlap_tma} "
            f"scale_init_tiles={args.scale_init_tiles} "
            f"replays={args.graph_replays}"
        )
    print(f"S4_C1_L0_LAYOUT={layout_output.cpu().tolist()}")
    print(
        "PASS_S4_C1_L0_EAGER "
        f"tma_arm={args.tma_arm} issue_mma={int(not args.no_mma)} "
        f"latent_tile={args.latent_tile} "
        f"sfa_exp={args.sfa_exp} sfb_exp={args.sfb_exp} "
        f"overlap_tma={args.overlap_tma} "
        f"scale_init_tiles={args.scale_init_tiles}"
    )


if __name__ == "__main__":
    main()
