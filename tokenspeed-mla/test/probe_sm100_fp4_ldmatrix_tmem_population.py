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

"""Qualify compact-FP4 TMA/register conversion into SM100 TMEM A.

This is the A17-N8-R1-C0 TMEM-population micro-gate.  It preserves the accepted
two-CTA TokenSpeed PV ownership scaffold, but replaces the rejected scalar
transpose/global-codebook path with the R1 primitives: compact E2M1 is expanded
by TMA only into a padded ``b4x16_p64`` shared-memory page, loaded through raw
narrow ``m16n16.x1.trans`` instructions, converted exactly to E4M3 in registers,
and written through the unchanged ``St16x256bOp`` TMEM store.

The identity-P oracle validates every element of the resulting 128x64 output on
both cluster CTAs.  Passing proves register-to-TMEM ownership only; it does not
yet claim a complete paged attention kernel or endpoint performance.
"""

import argparse

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims

THREADS_PER_CTA = 128
CLUSTER_SHAPE_MNK = (2, 1, 1)
QK_TILER_MNK = (128, 128, 128)
VP_TILER_MNK = (128, 64, 128)
CORRECTION_VALUES = 4
CORRECTION_STAGES = 2
V_OPERAND_STAGES = 2
V_ROWS = 128
V_COLS = 128
V_FULL_ROWS = 256
V_FULL_COLS = 512
V_PACKED_ROW_BYTES = V_COLS // 2
V_FULL_PACKED_ROW_BYTES = V_FULL_COLS // 2
V_PADDED_ROW_BYTES = V_COLS
V_GROUPS = V_COLS // 16
V_WORDS_PER_LANE = 2


@dsl_user_op
def convert_e2m1_bytes_to_e4m3_swar(
    codes: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Convert four low-nibble E2M1 byte codes to four exact E4M3 bytes."""

    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Uint32(codes).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b32 mag, t1, t2, any, nz, zero_bit, one_bit;\n\t"
            ".reg .b32 zero_mask, one_mask, scaled, out, corr0, corr1, corr;\n\t"
            ".reg .b32 sign;\n\t"
            "and.b32 mag, $1, 0x07070707;\n\t"
            "shr.u32 t1, mag, 1;\n\t"
            "shr.u32 t2, mag, 2;\n\t"
            "or.b32 any, mag, t1;\n\t"
            "or.b32 any, any, t2;\n\t"
            "and.b32 nz, any, 0x01010101;\n\t"
            "xor.b32 zero_bit, nz, 0x01010101;\n\t"
            "or.b32 any, t1, t2;\n\t"
            "not.b32 any, any;\n\t"
            "and.b32 one_bit, mag, any;\n\t"
            "and.b32 one_bit, one_bit, 0x01010101;\n\t"
            "mul.lo.u32 zero_mask, zero_bit, 0xff;\n\t"
            "mul.lo.u32 one_mask, one_bit, 0xff;\n\t"
            "shl.b32 scaled, mag, 2;\n\t"
            "add.u32 out, scaled, 0x30303030;\n\t"
            "and.b32 corr0, zero_mask, 0x30303030;\n\t"
            "and.b32 corr1, one_mask, 0x04040404;\n\t"
            "or.b32 corr, corr0, corr1;\n\t"
            "sub.u32 out, out, corr;\n\t"
            "and.b32 sign, $1, 0x08080808;\n\t"
            "shl.b32 sign, sign, 4;\n\t"
            "or.b32 out, out, sign;\n\t"
            "mov.b32 $0, out;\n\t"
            "}\n",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def convert_e2m1_bytes_to_e4m3_cvt(
    codes: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Convert four byte codes through native E2M1->F16->E4M3 instructions."""

    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Uint32(codes).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b32 packed, tmp, h0, h1, out;\n\t"
            ".reg .b8 b0, b1, b2, b3;\n\t"
            ".reg .b16 e0, e1;\n\t"
            "and.b32 packed, $1, 0x0000000f;\n\t"
            "shr.u32 tmp, $1, 4;\n\t"
            "and.b32 tmp, tmp, 0x000000f0;\n\t"
            "or.b32 packed, packed, tmp;\n\t"
            "shr.u32 tmp, $1, 8;\n\t"
            "and.b32 tmp, tmp, 0x00000f00;\n\t"
            "or.b32 packed, packed, tmp;\n\t"
            "shr.u32 tmp, $1, 12;\n\t"
            "and.b32 tmp, tmp, 0x0000f000;\n\t"
            "or.b32 packed, packed, tmp;\n\t"
            "mov.b32 {b0, b1, b2, b3}, packed;\n\t"
            "cvt.rn.f16x2.e2m1x2 h0, b0;\n\t"
            "cvt.rn.f16x2.e2m1x2 h1, b1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n\t"
            "mov.b32 out, {e0, e1};\n\t"
            "mov.b32 $0, out;\n\t"
            "}\n",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.struct
class SharedStorage:
    qk_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    vp_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


def make_tiled_mmas():
    qk = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        QK_TILER_MNK[:2],
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
    return qk, vp


@cute.kernel
def mixed_group_kernel(
    output: cute.Tensor,
    packed_v_desc: cutlass.GridConstant[cuda.TensorMap],
    p_input: cute.Tensor,
    matrix_output: cute.Tensor,
    qk_tiled_mma: cute.TiledMma,
    vp_tiled_mma: cute.TiledMma,
    qk_a_layout: cute.ComposedLayout,
    qk_b_layout: cute.ComposedLayout,
    vp_b_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
    qk_cols: cutlass.Constexpr,
    vp_cols: cutlass.Constexpr,
    score_cols: cutlass.Constexpr,
    output_cols: cutlass.Constexpr,
    correction_cols: cutlass.Constexpr,
    v_operand_cols: cutlass.Constexpr,
    v_operand_offset: cutlass.Constexpr,
    total_cols: cutlass.Constexpr,
    POISON: cutlass.Constexpr[int],
    CONVERSION_ARM: cutlass.Constexpr[int],
    POPULATE: cutlass.Constexpr[int],
    LATENT_OFFSET: cutlass.Constexpr[int],
    TOKEN_OFFSET: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    is_leader_cta = cta_rank == 0

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    s_q = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        qk_a_layout.outer,
        byte_alignment=128,
        swizzle=qk_a_layout.inner,
    )
    s_k = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        qk_b_layout.outer,
        byte_alignment=128,
        swizzle=qk_b_layout.inner,
    )
    s_p = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        vp_b_layout.outer,
        byte_alignment=128,
        swizzle=vp_b_layout.inner,
    )
    s_v_padded = cutlass.Array(
        cutlass.Int8,
        V_ROWS * V_PADDED_ROW_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    v_tma_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )

    qk_producer, qk_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, THREADS_PER_CTA * 2
        ),
        barrier_storage=storage.qk_mbar.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()
    vp_producer, vp_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, THREADS_PER_CTA
        ),
        barrier_storage=storage.vp_mbar.data_ptr(),
        cta_layout_vmnk=None,
    ).make_participants()

    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS_PER_CTA)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_barrier,
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

    qk_a = qk_tiled_mma.make_fragment_A(s_q)
    qk_b = qk_tiled_mma.make_fragment_B(s_k)
    qk_shape = qk_tiled_mma.partition_shape_C(QK_TILER_MNK[:2])
    qk_acc_fake = qk_tiled_mma.make_fragment_C(qk_shape)
    qk_acc = cute.make_tensor(tmem_ptr, qk_acc_fake.layout)

    vp_thr_mma = vp_tiled_mma.get_slice(0)
    vp_a_shape = vp_tiled_mma.partition_shape_A(
        (VP_TILER_MNK[0], VP_TILER_MNK[2], V_OPERAND_STAGES)
    )
    vp_a_fake = vp_thr_mma.make_fragment_A(vp_a_shape)
    vp_a = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + v_operand_offset, dtype=cutlass.Float8E4M3FN),
        vp_a_fake.layout,
    )
    vp_b = vp_tiled_mma.make_fragment_B(s_p)
    vp_shape = vp_tiled_mma.partition_shape_C(VP_TILER_MNK[:2])
    vp_acc_fake = vp_tiled_mma.make_fragment_C(vp_shape)
    vp_acc = cute.make_tensor(tmem_ptr + score_cols, vp_acc_fake.layout)

    # Write the host-provided [I_64, 0] through the B descriptor's logical
    # coordinates. The composed SMEM tensor applies the physical swizzle.
    for iteration in cutlass.range_constexpr(64):
        linear = iteration * THREADS_PER_CTA + tidx
        n = linear // 128
        k = linear % 128
        s_p[(n, k % 32), 0, k // 32, 0] = p_input[n, k]

    if cutlass.const_expr(POPULATE == 1):
        # ``elect_sync`` elects one lane per warp.  This CTA has four warps,
        # so barrier initialization and the full-page TMA issue must instead
        # have one CTA-wide owner.
        if tidx == 0:
            prims.mbarrier_init(v_tma_mbar, 1)
        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)

        if tidx == 0:
            prims.mbarrier_arrive_expect_tx(v_tma_mbar, packed_v_desc.global_tx_bytes())
            prims.cp_async_bulk_tensor_shared_cta_global(
                s_v_padded,
                packed_v_desc.get_ptr(),
                (cutlass.Int32(LATENT_OFFSET), cutlass.Int32(TOKEN_OFFSET)),
                v_tma_mbar,
            )

        while not prims.mbarrier_try_wait_parity(
            v_tma_mbar, cutlass.Int32(0), time_limit=10_000_000
        ):
            pass
        prims.barrier_cta_sync(0)

        # Every padded row contains eight 16-byte groups: eight payload bytes
        # and eight inserted bytes which the narrow load must ignore.  Poison
        # them after TMA so independent runs detect accidental use.
        for iteration in cutlass.range_constexpr(
            (V_ROWS * (V_PADDED_ROW_BYTES // 2)) // THREADS_PER_CTA
        ):
            flat = tidx + iteration * THREADS_PER_CTA
            row = flat // (V_PADDED_ROW_BYTES // 2)
            within_row = flat % (V_PADDED_ROW_BYTES // 2)
            group = within_row // 8
            byte = within_row % 8
            s_v_padded[row * V_PADDED_ROW_BYTES + group * 16 + 8 + byte] = cutlass.Int8(
                POISON
            )
        prims.barrier_cta_sync(0)

    tmem_store_atom = cute.make_copy_atom(
        tcgen05.St16x256bOp(tcgen05.Repetition(1)),
        cutlass.Float8E4M3FN,
    )
    tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, vp_a[None, None, None, 0])
    thr_store = tmem_store.get_slice(tidx)
    t_s_v = thr_store.partition_S(vp_a)
    t_v = thr_store.partition_D(vp_a)
    lane = tidx % 32
    lane_in_quad = lane % 4
    quad_base = (lane // 4) * 4
    # St16x256b.x4 assigns each warp two 16-row latent bands across the full
    # 128-token K dimension.  Narrow LdMatrix instead returns one four-token
    # word per lane for each 16-token band.  Exchange within each four-lane
    # latent-row quad so the store fragment receives adjacent eight-token
    # chunks in r0/r1 (low eight latent rows) and r2/r3 (high eight rows).
    for token_32 in cutlass.range_constexpr(4):
        t_s_segment = t_s_v[None, None, None, token_32, 0]
        t_v_segment = t_v[None, None, None, token_32, 0]
        r_v = cute.make_rmem_tensor(t_s_segment.shape, cutlass.Float8E4M3FN)
        r_v_u32 = cute.recast_tensor(r_v, cutlass.Uint32)
        for element in cutlass.range_constexpr(cute.size(r_v_u32)):
            r_v_u32[element] = cutlass.Uint32(0)
        for latent_band in cutlass.range_constexpr(2):
            latent_group = warp_idx * 2 + latent_band
            row_start_0 = (
                token_32 * 32 + (lane % 16)
            ) * V_PADDED_ROW_BYTES + latent_group * 16
            row_start_1 = row_start_0 + 16 * V_PADDED_ROW_BYTES
            if cutlass.const_expr(POPULATE == 1):
                regs_0 = prims.ldmatrix(
                    s_v_padded.data_ptr() + row_start_0,
                    V_WORDS_PER_LANE,
                    prims.MMALayout.COL,
                    shape=prims.LoadShape.M16N16,
                    src_format=prims.LoadSrcFormat.B4X16_P64,
                )
                regs_1 = prims.ldmatrix(
                    s_v_padded.data_ptr() + row_start_1,
                    V_WORDS_PER_LANE,
                    prims.MMALayout.COL,
                    shape=prims.LoadShape.M16N16,
                    src_format=prims.LoadSrcFormat.B4X16_P64,
                )
                for word in cutlass.range_constexpr(V_WORDS_PER_LANE):
                    converted_0 = cutlass.Uint32(0)
                    converted_1 = cutlass.Uint32(0)
                    if cutlass.const_expr(CONVERSION_ARM == 0):
                        converted_0 = convert_e2m1_bytes_to_e4m3_swar(
                            regs_0[word].to(cutlass.Uint32)
                        )
                        converted_1 = convert_e2m1_bytes_to_e4m3_swar(
                            regs_1[word].to(cutlass.Uint32)
                        )
                    else:
                        converted_0 = convert_e2m1_bytes_to_e4m3_cvt(
                            regs_0[word].to(cutlass.Uint32)
                        )
                        converted_1 = convert_e2m1_bytes_to_e4m3_cvt(
                            regs_1[word].to(cutlass.Uint32)
                        )
                    for pair_word in cutlass.range_constexpr(2):
                        source_lane = quad_base + (lane_in_quad % 2) * 2 + pair_word
                        from_first_16 = cute.arch.shuffle_sync(converted_0, source_lane)
                        from_second_16 = cute.arch.shuffle_sync(
                            converted_1, source_lane
                        )
                        selected = from_first_16
                        if lane_in_quad >= 2:
                            selected = from_second_16
                        r_v_u32[latent_band * 4 + word * 2 + pair_word] = selected
        cute.copy(thr_store, r_v, t_v_segment)
        cute.arch.fence_view_async_tmem_store()

    if warp_idx == 0 and is_leader_cta:
        qk_producer.acquire_and_advance()
        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range_constexpr(qk_a.shape[2]):
            cute.gemm(
                qk_tiled_mma,
                qk_acc,
                qk_a[None, None, k_block, 0],
                qk_b[None, None, k_block, 0],
                qk_acc,
            )
            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        qk_producer.commit()

    qk_full = qk_consumer.wait_and_advance()
    qk_full.release()
    cute.arch.sync_threads()

    if warp_idx == 0:
        vp_producer.acquire_and_advance()
        vp_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range_constexpr(vp_a.shape[2]):
            cute.gemm(
                vp_tiled_mma,
                vp_acc,
                vp_a[None, None, k_block, 0],
                vp_b[None, None, k_block, 0],
                vp_acc,
            )
            vp_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        vp_producer.commit()

    vp_full = vp_consumer.wait_and_advance()
    vp_full.release()
    cute.arch.sync_threads()

    # Load the FP32 accumulator through its logical MxN partition so the host
    # can validate every element of the register-to-TMEM mapping.
    t_acc = vp_acc[(None, None), 0, 0]
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)), cutlass.Float32
    )
    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, t_acc)
    thr_load = tmem_load.get_slice(tidx)
    g_output = matrix_output[cta_rank, None, None]
    t_tmem = thr_load.partition_S(t_acc)
    t_gmem = thr_load.partition_D(g_output)
    for segment in cutlass.range_constexpr(64):
        t_tmem_segment = t_tmem[None, None, segment]
        t_gmem_segment = t_gmem[None, None, segment]
        r_acc = cute.make_fragment_like(t_gmem_segment, cutlass.Float32)
        cute.copy(tmem_load, t_tmem_segment, r_acc)
        cute.arch.fence_view_async_tmem_load()
        cute.autovec_copy(r_acc, t_gmem_segment)
    cute.arch.sync_threads()

    if tidx == 0:
        output[cta_rank] = 1
        if is_leader_cta:
            output[2] = qk_cols
            output[3] = vp_cols
            output[4] = score_cols
            output[5] = output_cols
            output[6] = correction_cols
            output[7] = v_operand_cols
            output[8] = v_operand_offset
            output[9] = total_cols

    if warp_idx == 0:
        if is_leader_cta:
            qk_producer.tail()
        vp_producer.tail()

    tmem.relinquish_alloc_permit()
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def mixed_group_probe(
    output: cute.Tensor,
    packed_v: cute.Tensor,
    p_input: cute.Tensor,
    matrix_output: cute.Tensor,
    POISON: cutlass.Constexpr[int],
    CONVERSION_ARM: cutlass.Constexpr[int],
    POPULATE: cutlass.Constexpr[int],
    LATENT_OFFSET: cutlass.Constexpr[int],
    TOKEN_OFFSET: cutlass.Constexpr[int],
    stream,
):
    qk_tiled_mma, vp_tiled_mma = make_tiled_mmas()
    qk_cols = utils.get_num_tmem_alloc_cols(
        qk_tiled_mma.make_fragment_C(qk_tiled_mma.partition_shape_C(QK_TILER_MNK[:2]))
    )
    vp_cols = utils.get_num_tmem_alloc_cols(
        vp_tiled_mma.make_fragment_C(vp_tiled_mma.partition_shape_C(VP_TILER_MNK[:2]))
    )
    score_cols = qk_cols * 2
    output_cols = vp_cols * 4
    correction_cols = CORRECTION_VALUES * CORRECTION_STAGES
    vp_a_shape = vp_tiled_mma.partition_shape_A(
        (VP_TILER_MNK[0], VP_TILER_MNK[2], V_OPERAND_STAGES)
    )
    vp_a_fake = vp_tiled_mma.get_slice(0).make_fragment_A(vp_a_shape)
    v_operand_cols = tcgen05.find_tmem_tensor_col_offset(vp_a_fake)
    v_operand_offset = score_cols + output_cols + correction_cols
    total_cols = v_operand_offset + v_operand_cols
    if cutlass.const_expr(total_cols >= 512):
        raise ValueError(
            f"TMEM overlap: score={score_cols}, output={output_cols}, "
            f"correction={correction_cols}, v_operand={v_operand_cols}, limit=512"
        )
    qk_a_layout = sm100_utils.make_smem_layout_a(
        qk_tiled_mma, QK_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    qk_b_layout = sm100_utils.make_smem_layout_b(
        qk_tiled_mma, QK_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    vp_b_layout = sm100_utils.make_smem_layout_b(
        vp_tiled_mma, VP_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (qk_tiled_mma.thr_id.shape,)
    )
    packed_v_desc = cuda.create_tensor_map_tiled(
        packed_v.iterator.toint(),
        cutlass.Float4E2M1FN,
        global_dims=[V_FULL_COLS, V_FULL_ROWS],
        global_strides=[V_FULL_PACKED_ROW_BYTES // 16],
        box_dims=[V_COLS, V_ROWS],
        swizzle=cuda.TensorMapSwizzle.none,
    )
    mixed_group_kernel(
        output,
        packed_v_desc,
        p_input,
        matrix_output,
        qk_tiled_mma,
        vp_tiled_mma,
        qk_a_layout,
        qk_b_layout,
        vp_b_layout,
        cta_layout_vmnk,
        qk_cols,
        vp_cols,
        score_cols,
        output_cols,
        correction_cols,
        v_operand_cols,
        v_operand_offset,
        total_cols,
        POISON,
        CONVERSION_ARM,
        POPULATE,
        LATENT_OFFSET,
        TOKEN_OFFSET,
    ).launch(
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--poison",
        type=lambda value: int(value, 0),
        choices=(0x00, 0x5A),
        default=0x00,
    )
    parser.add_argument("--arm", choices=("swar", "cvt"), default="swar")
    parser.add_argument(
        "--population", choices=("candidate", "floor"), default="candidate"
    )
    parser.add_argument(
        "--latent-offset", type=int, choices=(0, 128, 256, 384), default=0
    )
    parser.add_argument("--token-offset", type=int, choices=(0, 128), default=0)
    parser.add_argument("--active-tokens", type=int, choices=range(1, 129), default=128)
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument(
        "--pattern",
        choices=("exhaustive", "random", "dense0", "dense15"),
        default="exhaustive",
    )
    args = parser.parse_args()
    poison = args.poison
    conversion_arm = 0 if args.arm == "swar" else 1
    populate = 1 if args.population == "candidate" else 0
    if args.graph_replays < 0:
        raise ValueError("--graph-replays must be non-negative")
    compiled = cute.compile(
        mixed_group_probe,
        fake(cutlass.Int32, (10,), 4),
        fake(cutlass.Uint8, (V_FULL_ROWS, V_FULL_PACKED_ROW_BYTES), 32),
        fake(cutlass.Float8E4M3FN, (64, 128), 16),
        fake(cutlass.Float32, (2, 128, 64), 16),
        poison,
        conversion_arm,
        populate,
        args.latent_offset,
        args.token_offset,
        make_fake_stream(),
        options="--enable-tvm-ffi --opt-level 3",
    )
    output = torch.zeros(10, device="cuda", dtype=torch.int32)
    if args.pattern == "random":
        generator = torch.Generator(device="cuda").manual_seed(20260723)
        codes = torch.randint(
            0,
            16,
            (V_FULL_ROWS, V_FULL_COLS),
            device="cuda",
            dtype=torch.uint8,
            generator=generator,
        )
    elif args.pattern.startswith("dense"):
        dense_code = int(args.pattern.removeprefix("dense"))
        codes = torch.full(
            (V_FULL_ROWS, V_FULL_COLS),
            dense_code,
            device="cuda",
            dtype=torch.uint8,
        )
    else:
        row = torch.arange(V_FULL_ROWS, device="cuda", dtype=torch.int64)[:, None]
        col = torch.arange(V_FULL_COLS, device="cuda", dtype=torch.int64)[None, :]
        codes = ((row * 5 + col * 3 + (col // 16) * 7) & 0xF).to(torch.uint8)
    active_end = args.token_offset + args.active_tokens
    codes[active_end : args.token_offset + V_ROWS].zero_()
    packed_v = codes[:, 0::2] | (codes[:, 1::2] << 4)
    codebook_values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device="cuda",
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    matrix_output = torch.zeros((2, 128, 64), device="cuda", dtype=torch.float32)
    eager_stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    metadata_values = None
    inactive_poison_independent = True
    packed_v_alt = None
    if args.population == "candidate" and args.active_tokens < V_ROWS:
        codes_alt = codes.clone()
        codes_alt[
            active_end : args.token_offset + V_ROWS,
            args.latent_offset : args.latent_offset + V_COLS,
        ] = 15
        packed_v_alt = codes_alt[:, 0::2] | (codes_alt[:, 1::2] << 4)
    graph_allocation_delta = 0
    queries = torch.arange(64, device="cuda")
    for probability_window in (0, 64):
        active_in_window = max(0, min(64, args.active_tokens - probability_window))
        p_input = torch.zeros((64, 128), device="cuda", dtype=torch.float32)
        if active_in_window:
            active_queries = queries[:active_in_window]
            p_input[active_queries, probability_window + active_queries] = 1.0
        p_input = p_input.to(torch.float8_e4m3fn)
        expected = torch.zeros((V_COLS, 64), device="cuda", dtype=torch.float32)
        if args.population == "candidate" and active_in_window:
            expected[:, :active_in_window] = codebook_values[
                codes[
                    args.token_offset
                    + probability_window : args.token_offset
                    + probability_window
                    + active_in_window,
                    args.latent_offset : args.latent_offset + V_COLS,
                ].long()
            ].T.float()
        matrix_output.fill_(float("nan"))
        compiled(
            output,
            packed_v,
            p_input,
            matrix_output,
            eager_stream,
        )
        torch.cuda.synchronize()
        result = output.cpu()
        torch.testing.assert_close(result[:2], torch.ones(2, dtype=torch.int32))
        current_metadata = tuple(result[2:].tolist())
        if metadata_values is None:
            metadata_values = current_metadata
        elif current_metadata != metadata_values:
            raise AssertionError(
                f"metadata changed across P windows: "
                f"{metadata_values} != {current_metadata}"
            )
        if args.population == "candidate" and not torch.equal(
            matrix_output[0], expected
        ):
            print(
                f"probability_window={probability_window} "
                f"active_in_window={active_in_window}"
            )
            print("output[0,:8,:8]=", matrix_output[0, :8, :8].cpu())
            print("expected[:8,:8]=", expected[:8, :8].cpu())
            print("output unique=", torch.unique(matrix_output[0]).cpu())
            token_ids = torch.arange(V_ROWS, device="cuda", dtype=torch.uint8)[:, None]
            latent_ids = torch.arange(V_COLS, device="cuda", dtype=torch.uint8)[None, :]

            def observe(pattern: torch.Tensor) -> torch.Tensor:
                selected_pattern = pattern.expand(V_ROWS, V_COLS).contiguous()
                full_pattern = torch.zeros(
                    (V_FULL_ROWS, V_FULL_COLS),
                    device="cuda",
                    dtype=torch.uint8,
                )
                full_pattern[
                    args.token_offset : args.token_offset + V_ROWS,
                    args.latent_offset : args.latent_offset + V_COLS,
                ] = selected_pattern
                packed_pattern = full_pattern[:, 0::2] | (full_pattern[:, 1::2] << 4)
                matrix_output.zero_()
                compiled(
                    output,
                    packed_pattern,
                    p_input,
                    matrix_output,
                    eager_stream,
                )
                torch.cuda.synchronize()
                observed = torch.full_like(matrix_output[0], -1, dtype=torch.int32)
                for code_idx in range(16):
                    observed[matrix_output[0] == codebook_values[code_idx].float()] = (
                        code_idx
                    )
                return observed

            latent_low = observe(latent_ids % 16)
            latent_high = observe(latent_ids // 16)
            token_low = observe(token_ids % 16)
            token_high = observe(token_ids // 16)
            print("observed latent low [:16,0]=", latent_low[:16, 0].cpu())
            print("observed latent high [:32,0]=", latent_high[:32, 0].cpu())
            print("observed token low [0,:32]=", token_low[0, :32].cpu())
            print("observed token high [0,:64]=", token_high[0, :64].cpu())
            print(
                "token mapping row invariant=",
                torch.equal(token_high, token_high[0:1].expand_as(token_high)),
            )
        torch.testing.assert_close(matrix_output[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(matrix_output[1], expected, rtol=0, atol=0)

        if packed_v_alt is not None:
            alt_output = torch.zeros_like(output)
            alt_matrix = torch.full_like(matrix_output, float("nan"))
            compiled(
                alt_output,
                packed_v_alt,
                p_input,
                alt_matrix,
                eager_stream,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(alt_matrix[0], expected, rtol=0, atol=0)
            torch.testing.assert_close(alt_matrix[1], expected, rtol=0, atol=0)

        if args.graph_replays:
            static_output = torch.zeros_like(output)
            static_matrix = torch.zeros_like(matrix_output)
            graph = torch.cuda.CUDAGraph()
            capture_stream = torch.cuda.Stream()
            graph_stream = cuda_driver.CUstream(capture_stream.cuda_stream)
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                compiled(static_output, packed_v, p_input, static_matrix, graph_stream)
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=capture_stream):
                compiled(static_output, packed_v, p_input, static_matrix, graph_stream)
            torch.cuda.synchronize()
            static_matrix.fill_(float("nan"))
            torch.cuda.synchronize()
            allocation_before = torch.cuda.memory_allocated()
            for _ in range(args.graph_replays):
                graph.replay()
            torch.cuda.synchronize()
            allocation_after = torch.cuda.memory_allocated()
            window_allocation_delta = allocation_after - allocation_before
            graph_allocation_delta = max(
                graph_allocation_delta, window_allocation_delta
            )
            if window_allocation_delta != 0:
                raise AssertionError(
                    f"P window {probability_window} graph replay allocated "
                    f"{window_allocation_delta} bytes"
                )
            torch.testing.assert_close(static_matrix[0], expected, rtol=0, atol=0)
            torch.testing.assert_close(static_matrix[1], expected, rtol=0, atol=0)

    (
        qk_cols,
        vp_cols,
        score_cols,
        output_cols,
        correction_cols,
        v_operand_cols,
        v_operand_offset,
        total_cols,
    ) = metadata_values
    if total_cols >= 512:
        raise AssertionError(
            f"TMEM overlap: score={score_cols}, output={output_cols}, "
            f"correction={correction_cols}, v_operand={v_operand_cols}, limit=512"
        )
    print(
        f"PASS exact_tma_fp4_tmem_population=True qk_group=2 vp_group=1 "
        f"conversion_arm={args.arm} population={args.population} "
        f"padding_poison={poison} latent_offset={args.latent_offset} "
        f"token_offset={args.token_offset} active_tokens={args.active_tokens} "
        f"pattern={args.pattern} "
        f"inactive_poison_independent={inactive_poison_independent} "
        f"graph_replays={args.graph_replays} "
        f"graph_allocation_delta={graph_allocation_delta} qk_cols={qk_cols} "
        f"vp_cols={vp_cols} score_cols={score_cols} output_cols={output_cols} "
        f"correction_cols={correction_cols} v_operand_cols={v_operand_cols} "
        f"v_operand_offset={v_operand_offset} total_cols={total_cols}"
    )


if __name__ == "__main__":
    main()
