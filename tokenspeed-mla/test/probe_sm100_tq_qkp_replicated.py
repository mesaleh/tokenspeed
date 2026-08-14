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

"""Falsify the C1 same-allocation mixed-QK-to-compact-PV path on SM100.

This probe advances the same-allocation path toward a complete attention reader.  It
uses the accepted 384-thread/two-CTA resource envelope and one cooperative
group-two M128 x N128 QK/RoPE instruction stream.  Each CTA receives 64 score
rows.  Its two N64 column halves live in paired lane groups, which exchange
their FP32 max and sum before publishing a local M64 x K128 result into the
exact two-stage transposed-P SMEM operand used by the planned
V(TMEM) x P(SMEM) MMA.

Five tiles force the single score stage, both P/correction stages, and two V
operand stages to wrap.  The same packed latent pointer backs mixed QK and the
four transposed V TensorMap boxes; narrow LdMatrix and native conversion
populate TMEM-A before four local PV MMAs.  S4-C0 carries row max/sum and the
power-of-two TurboQuant carrier across tiles, consumes P/correction one tile
late, rescales four persistent outputs, and drains the final tile.  This is
still a component probe, not an endpoint-latency or production claim.
"""

import argparse
import math
import statistics

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
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
    make_ptr,
)
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims

THREADS_PER_CTA = 384
TMEM_RETRIEVE_THREADS = 288
CLUSTER_SHAPE_MNK = (2, 1, 1)
TILES = 5
ROWS_PER_CTA = 64
SCORE_ROWS = 128
TOKENS = 128
N64 = 64
P_STAGES = 2
CORRECTION_VALUES = 4
CORRECTION_STAGES = 2
V_OPERAND_STAGES = 2
MASKED_TILE = 3

LATENT_K = 512
ROPE_K = 64
MIXED_TILER_MNK = (128, 128, 256)
LATENT_K_TILES = LATENT_K // MIXED_TILER_MNK[2]
NATIVE_QK_TILER_MNK = (128, 128, 128)
NATIVE_LATENT_K_TILES = LATENT_K // NATIVE_QK_TILER_MNK[2]
QK_BARRIER_SLOTS = max(LATENT_K_TILES, NATIVE_LATENT_K_TILES)
ROPE_TILER_MNK = (128, 128, ROPE_K)
SF_VEC_SIZE = 32
SF_DTYPE = cutlass.Float8E8M0FNU
MIXED_B_SMEM_DTYPE = cutlass.Int8
VP_TILER_MNK = (128, 64, 128)
NATIVE_PV_TILER_MNK = (128, 128, 128)
LIVE_SCALE_COLS = 20
V_OPERAND_OFFSET = 0
SCALE_OFFSET = 64
P_COR_OFFSET = 84
SCORE_OFFSET = 128
OUTPUT_OFFSET = 256
TMEM_ALLOC_COLS = 512
SOFTMAX_SCALE_LOG2 = 0.015625

LATENT_SLICES = LATENT_K // VP_TILER_MNK[0]
V_SLICE_COLS = VP_TILER_MNK[0]
V_FULL_PACKED_ROW_BYTES = LATENT_K // 2
V_PADDED_ROW_BYTES = V_SLICE_COLS
V_WORDS_PER_LANE = 2

Q_SMEM_BYTES = 128 * LATENT_K
Q_ROPE_SMEM_BYTES = 128 * ROPE_K
# Preserve S1's reviewed legal two-stage K composition capacity.
K_SMEM_BYTES = 128 * MIXED_TILER_MNK[2] * LATENT_K_TILES
K_ROPE_SMEM_BYTES = 128 * ROPE_K
V_SMEM_BYTES = 2 * 16 * 1024
P_SMEM_BYTES = P_STAGES * 8 * 1024
SOFTMAX_EXCHANGE_BYTES = 2 * SCORE_ROWS * 4
SMEM_CONTROL_BYTES = (
    8
    + TILES * (QK_BARRIER_SLOTS + 1) * 8
    + 2 * 8
    + 2 * 8
    + SOFTMAX_EXCHANGE_BYTES
    + 4
    + CORRECTION_STAGES * 4
    + CORRECTION_STAGES * 4
    + ROWS_PER_CTA * 4
    + 4
    + 8
    + 4
)
SMEM_CONTROL_PADDING_BYTES = (-SMEM_CONTROL_BYTES) % 128
V_TMA_BARRIER_BYTES = V_OPERAND_STAGES * 8
SMEM_PAYLOAD_BYTES = (
    SMEM_CONTROL_BYTES
    + SMEM_CONTROL_PADDING_BYTES
    + Q_SMEM_BYTES
    + Q_ROPE_SMEM_BYTES
    + K_SMEM_BYTES
    + K_ROPE_SMEM_BYTES
    + V_SMEM_BYTES
    + P_SMEM_BYTES
    + V_TMA_BARRIER_BYTES
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


@cute.jit
def consume_pv_tile(
    tidx,
    warp_idx,
    cta_global,
    cluster_index,
    consume_tile: cutlass.Constexpr[int],
    tmem_ptr,
    vp_output_cols,
    vp_acc_fake,
    p_cor,
    carrier_stage_smem,
    correction_scale_smem,
    consumer_carrier_smem,
    v_tma_mbar,
    v_smem,
    packed_v_desc,
    native_v_desc,
    native_cache_v_desc,
    vp_a,
    vp_b,
    vp_mma,
    native_v_a,
    native_v_b,
    native_vp_mma,
    native_v_acc_fake,
    native_p_smem,
    native_p_a,
    native_cache_v_b,
    native_pv_mma,
    native_pv_acc_fake,
    vp_producer,
    vp_consumer,
    matrix_output,
    POISON: cutlass.Constexpr[int],
    EXPORT_MATRIX: cutlass.Constexpr[int],
    POPULATE: cutlass.Constexpr[int],
    NATIVE_V: cutlass.Constexpr[int],
    NATIVE_PV: cutlass.Constexpr[int],
):
    """Consume one wrapped P/correction stage and accumulate compact V."""

    cta_rank = cta_global % CLUSTER_SHAPE_MNK[0]
    pcor_coords = cute.make_identity_tensor(p_cor.shape)
    pcor_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(CORRECTION_VALUES)),
        cutlass.Float32,
    )
    pcor_load = tcgen05.make_tmem_copy(pcor_load_atom, p_cor)
    stage = consume_tile % CORRECTION_STAGES

    # Bit 0 is interpreted per row before the persistent output is touched.
    if tidx < ROWS_PER_CTA:
        pcor_load_thr = pcor_load.get_slice(tidx)
        pcor_src = pcor_load_thr.partition_S(p_cor)
        pcor_regs_layout = pcor_load_thr.partition_D(pcor_coords)
        pcor_regs = cute.make_fragment_like(
            pcor_regs_layout[None, None, None, stage], cutlass.Float32
        )
        cute.copy(
            pcor_load,
            pcor_src[None, None, None, stage],
            pcor_regs,
        )
        cute.arch.fence_view_async_tmem_load()
        pcor_regs_i32 = cute.make_tensor(
            cute.recast_ptr(pcor_regs.iterator, dtype=cutlass.Int32),
            pcor_regs.layout,
        )
        correction_factor = cutlass.Float32(1.0)
        if cutlass.const_expr(consume_tile > 0):
            no_correction = pcor_regs_i32[3] & cutlass.Int32(1)
            correction_factor = (
                pcor_regs[2]
                * consumer_carrier_smem[0]
                / carrier_stage_smem[stage]
            )
            if no_correction != cutlass.Int32(0):
                correction_factor = cutlass.Float32(1.0)
        correction_scale_smem[tidx] = correction_factor
    # Every row must snapshot the incumbent carrier before thread 0 publishes
    # the carrier for this stage.  Without this CTA barrier, warp 1 can observe
    # the new value and lose the g_prev/g_current correction ratio.
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()
    if tidx == 0:
        consumer_carrier_smem[0] = carrier_stage_smem[stage]
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()

    if cutlass.const_expr(consume_tile > 0):
        for latent_slice in cutlass.range_constexpr(LATENT_SLICES):
            if cutlass.const_expr(NATIVE_PV == 1):
                native_pv_acc = cute.make_tensor(
                    tmem_ptr + OUTPUT_OFFSET + latent_slice * vp_output_cols,
                    native_pv_acc_fake.layout,
                )
                if tidx < 128:
                    t_acc = native_pv_acc[(None, None), 0, 0]
                    tmem_load_atom = cute.make_copy_atom(
                        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                        cutlass.Float32,
                    )
                    tmem_store_atom = cute.make_copy_atom(
                        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                        cutlass.Float32,
                    )
                    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, t_acc)
                    tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, t_acc)
                    thr_load = tmem_load.get_slice(tidx)
                    thr_store = tmem_store.get_slice(tidx)
                    output_coords = cute.make_identity_tensor(
                        (ROWS_PER_CTA, V_SLICE_COLS)
                    )
                    t_tmem = thr_load.partition_S(t_acc)
                    r_layout = thr_load.partition_D(output_coords)
                    t_tmem_store = thr_store.partition_D(t_acc)
                    r_acc = cute.make_fragment_like(r_layout, cutlass.Float32)
                    cute.copy(tmem_load, t_tmem, r_acc)
                    cute.arch.fence_view_async_tmem_load()
                    for element in cutlass.range_constexpr(cute.size(r_acc)):
                        row = r_layout[element][0]
                        r_acc[element] = (
                            r_acc[element] * correction_scale_smem[row]
                        )
                    cute.copy(tmem_store, r_acc, t_tmem_store)
                    cute.arch.fence_view_async_tmem_store()
            else:
                vp_acc = cute.make_tensor(
                    tmem_ptr + OUTPUT_OFFSET + latent_slice * vp_output_cols,
                    vp_acc_fake.layout,
                )
                if tidx < 128:
                    t_acc = vp_acc[(None, None), 0, 0]
                    tmem_load_atom = cute.make_copy_atom(
                        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)),
                        cutlass.Float32,
                    )
                    tmem_store_atom = cute.make_copy_atom(
                        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)),
                        cutlass.Float32,
                    )
                    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, t_acc)
                    tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, t_acc)
                    thr_load = tmem_load.get_slice(tidx)
                    thr_store = tmem_store.get_slice(tidx)
                    output_coords = cute.make_identity_tensor(
                        (V_SLICE_COLS, ROWS_PER_CTA)
                    )
                    t_tmem = thr_load.partition_S(t_acc)
                    r_layout = thr_load.partition_D(output_coords)
                    t_tmem_store = thr_store.partition_D(t_acc)
                    for segment in cutlass.range_constexpr(ROWS_PER_CTA):
                        r_acc = cute.make_fragment_like(
                            r_layout[None, None, segment], cutlass.Float32
                        )
                        cute.copy(
                            tmem_load,
                            t_tmem[None, None, segment],
                            r_acc,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        for element in cutlass.range_constexpr(cute.size(r_acc)):
                            r_acc[element] = (
                                r_acc[element] * correction_scale_smem[segment]
                            )
                        cute.copy(
                            tmem_store,
                            r_acc,
                            t_tmem_store[None, None, segment],
                        )
                    cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_threads()

    key_tile_index = cluster_index * TILES + consume_tile
    for latent_slice in cutlass.range_constexpr(LATENT_SLICES):
        v_sequence = consume_tile * LATENT_SLICES + latent_slice
        v_stage = v_sequence % V_OPERAND_STAGES
        v_phase = (v_sequence // V_OPERAND_STAGES) % 2
        v_bar_ptr = v_tma_mbar.get_barrier(v_stage)
        if cutlass.const_expr(NATIVE_PV == 1):
            v_stage_ptr = v_smem.data_ptr() + v_stage * (
                TOKENS * V_PADDED_ROW_BYTES // CLUSTER_SHAPE_MNK[0]
            )
        else:
            v_stage_ptr = v_smem.data_ptr() + v_stage * TOKENS * V_PADDED_ROW_BYTES
        if cutlass.const_expr(POPULATE == 1):
            if cutlass.const_expr(NATIVE_PV == 1):
                if tidx == 0:
                    prims.mbarrier_arrive_expect_tx(
                        v_bar_ptr, native_cache_v_desc.global_tx_bytes()
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        v_stage_ptr,
                        native_cache_v_desc.get_ptr(),
                        (
                            cutlass.Int32(
                                latent_slice * V_SLICE_COLS
                                + cta_rank
                                * (V_SLICE_COLS // CLUSTER_SHAPE_MNK[0])
                            ),
                            cutlass.Int32(0),
                            cutlass.Int32(key_tile_index),
                        ),
                        v_bar_ptr,
                    )
            elif tidx == 0:
                if cutlass.const_expr(NATIVE_V == 1):
                    prims.mbarrier_arrive_expect_tx(
                        v_bar_ptr, native_v_desc.global_tx_bytes()
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        v_stage_ptr,
                        native_v_desc.get_ptr(),
                        (
                            cutlass.Int32(0),
                            cutlass.Int32(latent_slice * V_SLICE_COLS),
                            cutlass.Int32(key_tile_index),
                        ),
                        v_bar_ptr,
                    )
                else:
                    prims.mbarrier_arrive_expect_tx(
                        v_bar_ptr, packed_v_desc.global_tx_bytes()
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        v_stage_ptr,
                        packed_v_desc.get_ptr(),
                        (
                            cutlass.Int32(latent_slice * V_SLICE_COLS),
                            cutlass.Int32(0),
                            cutlass.Int32(key_tile_index),
                        ),
                        v_bar_ptr,
                    )

            if cutlass.const_expr(NATIVE_PV == 1):
                while not prims.mbarrier_try_wait_parity(
                    v_bar_ptr, cutlass.Int32(v_phase), time_limit=10_000_000
                ):
                    pass
            else:
                while not prims.mbarrier_try_wait_parity(
                    v_bar_ptr, cutlass.Int32(v_phase), time_limit=10_000_000
                ):
                    pass
            cute.arch.sync_threads()
            if cutlass.const_expr(NATIVE_PV == 1):
                # The elected rank-0 MMA consumes B from both SMs.  A local
                # TMA mbarrier is insufficient: publish each CTA's completed
                # V slice before either CTA advances to the cooperative MMA.
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

            if cutlass.const_expr(
                POISON >= 0 and NATIVE_V == 0 and NATIVE_PV == 0
            ):
                if tidx < 128:
                    for iteration in cutlass.range_constexpr(
                        (TOKENS * (V_PADDED_ROW_BYTES // 2)) // 128
                    ):
                        flat = tidx + iteration * 128
                        row = flat // (V_PADDED_ROW_BYTES // 2)
                        within_row = flat % (V_PADDED_ROW_BYTES // 2)
                        group = within_row // 8
                        byte = within_row % 8
                        v_stage_ptr[
                            row * V_PADDED_ROW_BYTES + group * 16 + 8 + byte
                        ] = cutlass.Int8(POISON)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_threads()

        if tidx < 128 and cutlass.const_expr(
            NATIVE_V == 0 and NATIVE_PV == 0
        ):
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.St16x256bOp(tcgen05.Repetition(1)),
                cutlass.Float8E4M3FN,
            )
            tmem_store = tcgen05.make_tmem_copy(
                tmem_store_atom, vp_a[None, None, None, v_stage]
            )
            thr_store = tmem_store.get_slice(tidx)
            t_s_v = thr_store.partition_S(vp_a)
            t_v = thr_store.partition_D(vp_a)
            lane = tidx % 32
            lane_in_quad = lane % 4
            quad_base = (lane // 4) * 4
            local_warp = tidx // 32
            for token_32 in cutlass.range_constexpr(4):
                t_s_segment = t_s_v[None, None, None, token_32, v_stage]
                t_v_segment = t_v[None, None, None, token_32, v_stage]
                r_v = cute.make_rmem_tensor(t_s_segment.shape, cutlass.Float8E4M3FN)
                r_v_u32 = cute.recast_tensor(r_v, cutlass.Uint32)
                for element in cutlass.range_constexpr(cute.size(r_v_u32)):
                    r_v_u32[element] = cutlass.Uint32(0)
                if cutlass.const_expr(POPULATE == 1):
                    for latent_band in cutlass.range_constexpr(2):
                        latent_group = local_warp * 2 + latent_band
                        row_start_0 = (
                            (token_32 * 32 + (lane % 16)) * V_PADDED_ROW_BYTES
                            + latent_group * 16
                        )
                        row_start_1 = row_start_0 + 16 * V_PADDED_ROW_BYTES
                        regs_0 = prims.ldmatrix(
                            v_stage_ptr + row_start_0,
                            V_WORDS_PER_LANE,
                            prims.MMALayout.COL,
                            shape=prims.LoadShape.M16N16,
                            src_format=prims.LoadSrcFormat.B4X16_P64,
                        )
                        regs_1 = prims.ldmatrix(
                            v_stage_ptr + row_start_1,
                            V_WORDS_PER_LANE,
                            prims.MMALayout.COL,
                            shape=prims.LoadShape.M16N16,
                            src_format=prims.LoadSrcFormat.B4X16_P64,
                        )
                        for word in cutlass.range_constexpr(V_WORDS_PER_LANE):
                            converted_0 = convert_e2m1_bytes_to_e4m3_cvt(
                                regs_0[word].to(cutlass.Uint32)
                            )
                            converted_1 = convert_e2m1_bytes_to_e4m3_cvt(
                                regs_1[word].to(cutlass.Uint32)
                            )
                            for pair_word in cutlass.range_constexpr(2):
                                source_lane = (
                                    quad_base
                                    + (lane_in_quad % 2) * 2
                                    + pair_word
                                )
                                from_first_16 = cute.arch.shuffle_sync(
                                    converted_0, source_lane
                                )
                                from_second_16 = cute.arch.shuffle_sync(
                                    converted_1, source_lane
                                )
                                selected = from_first_16
                                if lane_in_quad >= 2:
                                    selected = from_second_16
                                r_v_u32[
                                    latent_band * 4 + word * 2 + pair_word
                                ] = selected
                cute.copy(thr_store, r_v, t_v_segment)
                cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_threads()

        if cutlass.const_expr(NATIVE_PV == 1):
            pv_acc = cute.make_tensor(
                tmem_ptr + OUTPUT_OFFSET + latent_slice * vp_output_cols,
                native_pv_acc_fake.layout,
            )
        else:
            pv_acc = cute.make_tensor(
                tmem_ptr + OUTPUT_OFFSET + latent_slice * vp_output_cols,
                vp_acc_fake.layout,
            )
        if warp_idx == 8:
            if cutlass.const_expr(NATIVE_PV == 1):
                if cta_global % CLUSTER_SHAPE_MNK[0] == 0:
                    vp_producer.acquire_and_advance()
                    native_pv_mma.set(
                        tcgen05.Field.ACCUMULATE,
                        cutlass.const_expr(consume_tile > 0),
                    )
                    for k_block in cutlass.range_constexpr(native_p_a.shape[2]):
                        cute.gemm(
                            native_pv_mma,
                            pv_acc,
                            native_p_a[None, None, k_block, stage],
                            native_cache_v_b[None, None, k_block, v_stage],
                            pv_acc,
                        )
                        native_pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                    vp_producer.commit()
            else:
                vp_producer.acquire_and_advance()
                if cutlass.const_expr(NATIVE_V == 1):
                    native_vp_mma.set(
                        tcgen05.Field.ACCUMULATE,
                        cutlass.const_expr(consume_tile > 0),
                    )
                    for k_block in cutlass.range_constexpr(native_v_a.shape[2]):
                        cute.gemm(
                            native_vp_mma,
                            pv_acc,
                            native_v_a[None, None, k_block, v_stage],
                            native_v_b[None, None, k_block, stage],
                            pv_acc,
                        )
                        native_vp_mma.set(tcgen05.Field.ACCUMULATE, True)
                else:
                    vp_mma.set(
                        tcgen05.Field.ACCUMULATE,
                        cutlass.const_expr(consume_tile > 0),
                    )
                    for k_block in cutlass.range_constexpr(vp_a.shape[2]):
                        cute.gemm(
                            vp_mma,
                            pv_acc,
                            vp_a[None, None, k_block, v_stage],
                            vp_b[None, None, k_block, stage],
                            pv_acc,
                        )
                        vp_mma.set(tcgen05.Field.ACCUMULATE, True)
                vp_producer.commit()

        vp_full = vp_consumer.wait_and_advance()
        vp_full.release()
        cute.arch.sync_threads()

        if cutlass.const_expr(EXPORT_MATRIX == 1 and NATIVE_PV == 1):
            if tidx < 128:
                t_acc = pv_acc[(None, None), 0, 0]
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                    cutlass.Float32,
                )
                tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, t_acc)
                thr_load = tmem_load.get_slice(tidx)
                g_output = cute.make_tensor(
                    matrix_output.iterator
                    + (
                        ((cta_global * TILES + consume_tile) * LATENT_K)
                        + latent_slice * V_SLICE_COLS
                    )
                    * ROWS_PER_CTA,
                    cute.make_layout(
                        (ROWS_PER_CTA, V_SLICE_COLS),
                        stride=(1, ROWS_PER_CTA),
                    ),
                )
                t_tmem = thr_load.partition_S(t_acc)
                t_gmem = thr_load.partition_D(g_output)
                r_acc = cute.make_fragment_like(t_gmem, cutlass.Float32)
                cute.copy(tmem_load, t_tmem, r_acc)
                cute.arch.fence_view_async_tmem_load()
                cute.autovec_copy(r_acc, t_gmem)
        if cutlass.const_expr(EXPORT_MATRIX == 1 and NATIVE_PV == 0):
            if tidx < 128:
                t_acc = pv_acc[(None, None), 0, 0]
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)),
                    cutlass.Float32,
                )
                tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, t_acc)
                thr_load = tmem_load.get_slice(tidx)
                g_output = cute.make_tensor(
                    matrix_output.iterator
                    + (
                        ((cta_global * TILES + consume_tile) * LATENT_K)
                        + latent_slice * V_SLICE_COLS
                    )
                    * ROWS_PER_CTA,
                    cute.make_layout(
                        (V_SLICE_COLS, ROWS_PER_CTA),
                        stride=(ROWS_PER_CTA, 1),
                    ),
                )
                t_tmem = thr_load.partition_S(t_acc)
                t_gmem = thr_load.partition_D(g_output)
                for segment in cutlass.range_constexpr(ROWS_PER_CTA):
                    t_tmem_segment = t_tmem[None, None, segment]
                    t_gmem_segment = t_gmem[None, None, segment]
                    r_acc = cute.make_fragment_like(t_gmem_segment, cutlass.Float32)
                    cute.copy(tmem_load, t_tmem_segment, r_acc)
                    cute.arch.fence_view_async_tmem_load()
                    cute.autovec_copy(r_acc, t_gmem_segment)
        cute.arch.sync_threads()


@cute.struct
class SharedStorage:
    init_mbar: cutlass.Int64
    tma_mbar: cute.struct.MemRange[
        cutlass.Int64, TILES * (QK_BARRIER_SLOTS + 1)
    ]
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    vp_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    softmax_max_exchange: cute.struct.MemRange[cutlass.Float32, SCORE_ROWS]
    softmax_sum_exchange: cute.struct.MemRange[cutlass.Float32, SCORE_ROWS]
    carrier_scale: cute.struct.MemRange[cutlass.Float32, 1]
    carrier_stage: cute.struct.MemRange[cutlass.Float32, CORRECTION_STAGES]
    carrier_exp_stage: cute.struct.MemRange[cutlass.Int32, CORRECTION_STAGES]
    correction_scale: cute.struct.MemRange[cutlass.Float32, ROWS_PER_CTA]
    consumer_carrier: cute.struct.MemRange[cutlass.Float32, 1]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


def make_mmas():
    mixed = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        cutlass.Float4E2M1FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        SF_DTYPE,
        SF_VEC_SIZE,
        tcgen05.CtaGroup.TWO,
        MIXED_TILER_MNK[:2],
    )
    rope = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        ROPE_TILER_MNK[:2],
    )
    native_qk = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        NATIVE_QK_TILER_MNK[:2],
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
    native_vp = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        VP_TILER_MNK[:2],
    )
    native_pv = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        NATIVE_PV_TILER_MNK[:2],
    )
    return mixed, native_qk, rope, vp, native_vp, native_pv


@cute.kernel
def ownership_kernel(
    layout_output: cute.Tensor,
    matrix_output: cute.Tensor,
    normalized_output: cute.Tensor,
    bf16_output: cute.Tensor,
    carrier_output: cute.Tensor,
    p_output: cute.Tensor,
    max_output: cute.Tensor,
    sum_output: cute.Tensor,
    owner_output: cute.Tensor,
    correction_output: cute.Tensor,
    flags_output: cute.Tensor,
    token_scale: cute.Tensor,
    packed_v_desc: cutlass.GridConstant[cuda.TensorMap],
    native_v_desc: cutlass.GridConstant[cuda.TensorMap],
    native_cache_v_desc: cutlass.GridConstant[cuda.TensorMap],
    mixed_mma: cute.TiledMma,
    native_qk_mma: cute.TiledMma,
    rope_mma: cute.TiledMma,
    vp_mma: cute.TiledMma,
    native_vp_mma: cute.TiledMma,
    native_pv_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    tma_atom_native_q: cute.CopyAtom,
    tma_tensor_native_q: cute.Tensor,
    tma_atom_native_k: cute.CopyAtom,
    tma_tensor_native_k: cute.Tensor,
    tma_atom_rope_a: cute.CopyAtom,
    tma_tensor_rope_a: cute.Tensor,
    tma_atom_rope_b: cute.CopyAtom,
    tma_tensor_rope_b: cute.Tensor,
    mixed_a_layout: cute.ComposedLayout,
    mixed_b_layout: cute.ComposedLayout,
    native_q_layout: cute.ComposedLayout,
    native_k_layout: cute.ComposedLayout,
    rope_a_layout: cute.ComposedLayout,
    rope_b_layout: cute.ComposedLayout,
    sfa_layout: cute.Layout,
    sfb_layout: cute.Layout,
    sfa_cols: cutlass.Constexpr,
    vp_output_cols: cutlass.Constexpr,
    v_operand_cols: cutlass.Constexpr,
    p_layout: cute.ComposedLayout,
    native_v_layout: cute.ComposedLayout,
    native_p_layout: cute.ComposedLayout,
    native_cache_v_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
    POISON: cutlass.Constexpr[int],
    EXPORT_MATRIX: cutlass.Constexpr[int],
    POPULATE: cutlass.Constexpr[int],
    NATIVE_V: cutlass.Constexpr[int],
    NATIVE_QK: cutlass.Constexpr[int],
    NATIVE_PV: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_global, _, _ = cute.arch.block_idx()
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    cluster_index = cute.arch.make_warp_uniform(cta_global // CLUSTER_SHAPE_MNK[0])

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    softmax_max_exchange = cute.make_tensor(
        storage.softmax_max_exchange.data_ptr(), cute.make_layout(SCORE_ROWS)
    )
    softmax_sum_exchange = cute.make_tensor(
        storage.softmax_sum_exchange.data_ptr(), cute.make_layout(SCORE_ROWS)
    )
    carrier_scale_smem = cute.make_tensor(
        storage.carrier_scale.data_ptr(), cute.make_layout(1)
    )
    carrier_stage_smem = cute.make_tensor(
        storage.carrier_stage.data_ptr(), cute.make_layout(CORRECTION_STAGES)
    )
    carrier_exp_stage_smem = cute.make_tensor(
        storage.carrier_exp_stage.data_ptr(), cute.make_layout(CORRECTION_STAGES)
    )
    correction_scale_smem = cute.make_tensor(
        storage.correction_scale.data_ptr(), cute.make_layout(ROWS_PER_CTA)
    )
    consumer_carrier_smem = cute.make_tensor(
        storage.consumer_carrier.data_ptr(), cute.make_layout(1)
    )

    # Preserve the reviewed extra 32 KiB K stage so the ownership comparison is
    # not confounded by K-stage overwrite timing.
    q_storage = cutlass.Array(
        cutlass.Int8,
        Q_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    q_smem_ptr = cute.make_ptr(
        cutlass.Float8E4M3FN,
        q_storage.data_ptr().ir_value(),
        cute.AddressSpace.smem,
        assumed_align=128,
    )
    q_smem = cute.make_tensor(
        cute.recast_ptr(
            q_smem_ptr,
            swizzle_=mixed_a_layout.inner,
            dtype=cutlass.Float8E4M3FN,
        ),
        mixed_a_layout.outer,
    )
    native_q_smem = cute.make_tensor(
        cute.recast_ptr(
            q_smem_ptr,
            swizzle_=native_q_layout.inner,
            dtype=cutlass.Float8E4M3FN,
        ),
        native_q_layout.outer,
    )
    q_rope_smem = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        rope_a_layout.outer,
        byte_alignment=128,
        swizzle=rope_a_layout.inner,
    )
    k_storage = cutlass.Array(
        cutlass.Int8,
        K_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    k_smem_ptr = cute.make_ptr(
        cutlass.Int8,
        k_storage.data_ptr().ir_value(),
        cute.AddressSpace.smem,
        assumed_align=128,
    )
    k_smem = cute.make_tensor(
        cute.recast_ptr(
            k_smem_ptr,
            swizzle_=mixed_b_layout.inner,
            dtype=MIXED_B_SMEM_DTYPE,
        ),
        mixed_b_layout.outer,
    )
    native_k_smem = cute.make_tensor(
        cute.recast_ptr(
            k_smem_ptr,
            swizzle_=native_k_layout.inner,
            dtype=cutlass.Float8E4M3FN,
        ),
        native_k_layout.outer,
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
    native_v_smem_ptr = cute.make_ptr(
            cutlass.Float8E4M3FN,
            v_smem.data_ptr().ir_value(),
            cute.AddressSpace.smem,
            assumed_align=128,
        )
    native_v_smem = cute.make_tensor(
        cute.recast_ptr(
            native_v_smem_ptr,
            swizzle_=native_v_layout.inner,
            dtype=cutlass.Float8E4M3FN,
        ),
        native_v_layout.outer,
    )
    native_cache_v_smem = cute.make_tensor(
        cute.recast_ptr(
            native_v_smem_ptr,
            swizzle_=native_cache_v_layout.inner,
            dtype=cutlass.Float8E4M3FN,
        ),
        native_cache_v_layout.outer,
    )
    p_storage = cutlass.Array(
        cutlass.Int8,
        P_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    p_smem_ptr = cute.make_ptr(
        cutlass.Float8E4M3FN,
        p_storage.data_ptr().ir_value(),
        cute.AddressSpace.smem,
        assumed_align=128,
    )
    p_smem = cute.make_tensor(
        cute.recast_ptr(
            p_smem_ptr,
            swizzle_=p_layout.inner,
            dtype=cutlass.Float8E4M3FN,
        ),
        p_layout.outer,
    )
    native_p_smem = cute.make_tensor(
        cute.recast_ptr(
            p_smem_ptr,
            swizzle_=native_p_layout.inner,
            dtype=cutlass.Float8E4M3FN,
        ),
        native_p_layout.outer,
    )
    v_tma_mbar = cutlass.Array(
        cutlass.Int64,
        V_OPERAND_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    v_tma_mbar_ptr = cute.make_ptr(
        cutlass.Int64,
        v_tma_mbar.data_ptr().ir_value(),
        cute.AddressSpace.smem,
        assumed_align=8,
    )

    # The group-two slice assigns each physical CTA its generated M64 operand
    # partition; the elected rank-0 warp later issues the cooperative MMA.
    mma_tile_coord_v = cta_rank
    cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank)
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
    g_native_q_mkl = cute.local_tile(
        tma_tensor_native_q,
        cute.slice_(NATIVE_QK_TILER_MNK, (None, 0, None)),
        (None, None, None),
    )
    g_native_k_nkl = cute.local_tile(
        tma_tensor_native_k,
        cute.slice_(NATIVE_QK_TILER_MNK, (0, None, None)),
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
    native_qk_thr_mma = native_qk_mma.get_slice(mma_tile_coord_v)
    t_cg_native_q = native_qk_thr_mma.partition_A(g_native_q_mkl)
    t_cg_native_k = native_qk_thr_mma.partition_B(g_native_k_nkl)
    t_ns_q, t_ng_q = cpasync.tma_partition(
        tma_atom_native_q,
        cta_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(native_q_smem, 0, 3),
        cute.group_modes(t_cg_native_q, 0, 3),
    )
    t_ns_k, t_ng_k = cpasync.tma_partition(
        tma_atom_native_k,
        cta_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(native_k_smem, 0, 3),
        cute.group_modes(t_cg_native_k, 0, 3),
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
    t_ag_a = t_ag_a[(None, 0, None, None)]
    t_bg_b = t_bg_b[(None, 0, None, None)]
    t_ng_q = t_ng_q[(None, 0, None, None)]
    t_ng_k = t_ng_k[(None, 0, None, None)]
    t_ag_rope_a = t_ag_rope_a[(None, 0, None, None)]
    t_bg_rope_b = t_bg_rope_b[(None, 0, None, None)]

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
    native_ab_copy_bytes = (
        cute.size_in_bytes(
            cutlass.Float8E4M3FN,
            cute.slice_(native_q_smem, (None, None, None, 0)),
        )
        + cute.size_in_bytes(
            cutlass.Float8E4M3FN,
            cute.slice_(native_k_smem, (None, None, None, 0)),
        )
    ) * cute.size(native_qk_mma.thr_id.shape)
    selected_ab_copy_bytes = ab_copy_bytes
    if cutlass.const_expr(NATIVE_QK == 1):
        selected_ab_copy_bytes = native_ab_copy_bytes
    native_v_copy_bytes = cute.size_in_bytes(
        cutlass.Float8E4M3FN,
        cute.slice_(native_cache_v_smem, (None, None, None, 0)),
    ) // cute.size(native_pv_mma.thr_id.shape)
    v_tma_barriers = pipeline.MbarrierArray(
        v_tma_mbar_ptr,
        V_OPERAND_STAGES,
        (
            pipeline.PipelineOp.TmaLoad,
            pipeline.CooperativeGroup(pipeline.Agent.Thread),
        ),
        tx_count=native_v_copy_bytes,
    )
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
        TILES * (QK_BARRIER_SLOTS + 1),
        (
            pipeline.PipelineOp.TmaLoad,
            pipeline.CooperativeGroup(pipeline.Agent.Thread),
        ),
        tx_count=selected_ab_copy_bytes,
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
    if cutlass.const_expr(NATIVE_PV == 1):
        vp_producer, vp_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, THREADS_PER_CTA * CLUSTER_SHAPE_MNK[0]
            ),
            barrier_storage=storage.vp_mbar.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        ).make_participants()
    else:
        vp_producer, vp_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, THREADS_PER_CTA
            ),
            barrier_storage=storage.vp_mbar.data_ptr(),
            cta_layout_vmnk=None,
            defer_sync=True,
        ).make_participants()

    retrieve_barrier = pipeline.NamedBarrier(
        barrier_id=1, num_threads=TMEM_RETRIEVE_THREADS
    )
    softmax_barrier_pair_02 = pipeline.NamedBarrier(barrier_id=2, num_threads=64)
    softmax_barrier_pair_13 = pipeline.NamedBarrier(barrier_id=3, num_threads=64)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=retrieve_barrier,
        allocator_warp_id=8,
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
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

    prims.fence_mbarrier_init()
    cute.arch.sync_threads()

    mixed_a = mixed_mma.make_fragment_A(q_smem)
    mixed_b = mixed_mma.make_fragment_B(k_smem)
    mixed_k_blocks = cute.size(mixed_a, mode=[2])
    if cutlass.const_expr(mixed_k_blocks != 8):
        raise ValueError(f"expected eight latent K blocks, got {mixed_k_blocks}")
    native_q = native_qk_mma.make_fragment_A(native_q_smem)
    native_k = native_qk_mma.make_fragment_B(native_k_smem)
    native_k_blocks = cute.size(native_q, mode=[2])
    if cutlass.const_expr(native_k_blocks != 4):
        raise ValueError(
            f"expected four native K blocks per K128 tile, got {native_k_blocks}"
        )
    acc_fake = mixed_mma.make_fragment_C(
        mixed_mma.partition_shape_C(MIXED_TILER_MNK[:2])
    )
    score_tmem_ptr = tmem_ptr + SCORE_OFFSET
    acc = cute.make_tensor(score_tmem_ptr, acc_fake.layout)
    native_acc_fake = native_qk_mma.make_fragment_C(
        native_qk_mma.partition_shape_C(NATIVE_QK_TILER_MNK[:2])
    )
    native_acc = cute.make_tensor(score_tmem_ptr, native_acc_fake.layout)
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

    # Legalize the complete candidate TMEM envelope before attaching data
    # movement.  Group-one PV is local to each CTA even though QK is group-two.
    vp_a_shape = vp_mma.partition_shape_A(
        (VP_TILER_MNK[0], VP_TILER_MNK[2], V_OPERAND_STAGES)
    )
    vp_a_fake = vp_mma.get_slice(0).make_fragment_A(vp_a_shape)
    vp_a = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + V_OPERAND_OFFSET, dtype=cutlass.Float8E4M3FN),
        vp_a_fake.layout,
    )
    vp_acc_fake = vp_mma.make_fragment_C(
        vp_mma.partition_shape_C(VP_TILER_MNK[:2])
    )
    vp_output = cute.make_tensor(tmem_ptr + OUTPUT_OFFSET, vp_acc_fake.layout)
    vp_b = vp_mma.make_fragment_B(p_smem)
    native_v_a = native_vp_mma.make_fragment_A(native_v_smem)
    native_v_b = native_vp_mma.make_fragment_B(p_smem)
    native_v_acc_fake = native_vp_mma.make_fragment_C(
        native_vp_mma.partition_shape_C(VP_TILER_MNK[:2])
    )
    native_p_a = native_pv_mma.make_fragment_A(native_p_smem)
    native_cache_v_b = native_pv_mma.make_fragment_B(native_cache_v_smem)
    native_pv_acc_fake = native_pv_mma.make_fragment_C(
        native_pv_mma.partition_shape_C(NATIVE_PV_TILER_MNK[:2])
    )
    p_cor = cute.make_tensor(
        tmem_ptr + P_COR_OFFSET,
        cute.make_layout(
            (SCORE_ROWS, CORRECTION_VALUES, CORRECTION_STAGES),
            stride=(1 << 16, 1, CORRECTION_VALUES),
        ),
    )

    # Only the first cluster publishes generated-coordinate evidence.  Reading
    # TensorMap transaction bytes also keeps the same-pointer descriptor in the
    # generated program before the delayed consumer issues its first TMA.
    if tidx == 0 and cluster_index == 0:
        layout_output[cta_rank, 0] = cta_rank + 1
        layout_output[cta_rank, 1] = V_OPERAND_OFFSET
        layout_output[cta_rank, 2] = v_operand_cols
        layout_output[cta_rank, 3] = SCALE_OFFSET
        layout_output[cta_rank, 4] = LIVE_SCALE_COLS
        layout_output[cta_rank, 5] = P_COR_OFFSET
        layout_output[cta_rank, 6] = CORRECTION_VALUES * CORRECTION_STAGES
        layout_output[cta_rank, 7] = SCORE_OFFSET
        layout_output[cta_rank, 8] = OUTPUT_OFFSET - SCORE_OFFSET - 64
        layout_output[cta_rank, 9] = OUTPUT_OFFSET
        layout_output[cta_rank, 10] = vp_output_cols * LATENT_SLICES
        layout_output[cta_rank, 11] = TMEM_ALLOC_COLS
        layout_output[cta_rank, 12] = packed_v_desc.global_tx_bytes()
        layout_output[cta_rank, 13] = native_v_desc.global_tx_bytes()
        layout_output[cta_rank, 14] = NATIVE_V
        layout_output[cta_rank, 15] = NATIVE_QK
        layout_output[cta_rank, 16] = NATIVE_PV

    pcor_coords = cute.make_identity_tensor(p_cor.shape)
    pcor_store_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(CORRECTION_VALUES)),
        cutlass.Float32,
    )
    pcor_store = tcgen05.make_tmem_copy(pcor_store_atom, p_cor)

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
        consumer_carrier_smem[0] = cutlass.Float32(0.0)
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()

    online_row_max = cutlass.Float32(-1.0e6)
    online_row_sum = cutlass.Float32(0.0)

    for tile in cutlass.range_constexpr(TILES):
        barrier_base = tile * (QK_BARRIER_SLOTS + 1)
        key_tile_index = cluster_index * TILES + tile
        if tidx == 0:
            if cutlass.const_expr(NATIVE_PV == 1):
                carrier_scale = cutlass.Float32(1.0)
                carrier_exp = cutlass.Int32(0)
            else:
                max_d = cutlass.Float32(0.0)
                for token in cutlass.range_constexpr(TOKENS):
                    max_d = cute.arch.fmax(
                        max_d, token_scale[tile, token].to(cutlass.Float32)
                    )
                # Exact power-of-two search for the smallest g satisfying
                # max(d_t) <= 224*g.  The bounded diagnostic range covers every
                # finite BF16 scale used by the reader contract without depending
                # on host libm rounding.
                carrier_scale = cutlass.Float32(2.0**-16)
                carrier_exp = cutlass.Int32(-16)
                for _ in cutlass.range_constexpr(32):
                    if max_d > cutlass.Float32(224.0) * carrier_scale:
                        carrier_scale = carrier_scale * cutlass.Float32(2.0)
                        carrier_exp = carrier_exp + cutlass.Int32(1)
            if cutlass.const_expr(tile == MASKED_TILE):
                carrier_scale = carrier_stage_smem[(tile - 1) % CORRECTION_STAGES]
                carrier_exp = carrier_exp_stage_smem[
                    (tile - 1) % CORRECTION_STAGES
                ]
            carrier_scale_smem[0] = carrier_scale
            carrier_stage_smem[tile % CORRECTION_STAGES] = carrier_scale
            carrier_exp_stage_smem[tile % CORRECTION_STAGES] = carrier_exp
            carrier_output[cta_global, tile] = carrier_scale
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        carrier_scale = carrier_scale_smem[0]
        if warp_idx == 9:
            if cutlass.const_expr(NATIVE_QK == 1):
                for latent_tile in cutlass.range_constexpr(NATIVE_LATENT_K_TILES):
                    barrier_index = barrier_base + latent_tile
                    tma_bar_ptr = tma_barriers.get_barrier(barrier_index)
                    tma_barriers.arrive_and_expect_tx(
                        barrier_index, native_ab_copy_bytes
                    )
                    cute.copy(
                        tma_atom_native_q,
                        t_ng_q[(None, latent_tile, cluster_index)],
                        t_ns_q[(None, latent_tile)],
                        tma_bar_ptr=tma_bar_ptr,
                    )
                    cute.copy(
                        tma_atom_native_k,
                        t_ng_k[(None, latent_tile, key_tile_index)],
                        t_ns_k[(None, latent_tile)],
                        tma_bar_ptr=tma_bar_ptr,
                    )
            else:
                for latent_tile in cutlass.range_constexpr(LATENT_K_TILES):
                    barrier_index = barrier_base + latent_tile
                    tma_bar_ptr = tma_barriers.get_barrier(barrier_index)
                    tma_barriers.arrive_and_expect_tx(barrier_index, ab_copy_bytes)
                    cute.copy(
                        tma_atom_a,
                        t_ag_a[(None, latent_tile, cluster_index)],
                        t_as_a[(None, latent_tile)],
                        tma_bar_ptr=tma_bar_ptr,
                    )
                    cute.copy(
                        tma_atom_b,
                        t_bg_b[(None, latent_tile, key_tile_index)],
                        t_bs_b[(None, latent_tile)],
                        tma_bar_ptr=tma_bar_ptr,
                    )
            rope_barrier_index = barrier_base + QK_BARRIER_SLOTS
            rope_bar_ptr = tma_barriers.get_barrier(rope_barrier_index)
            tma_barriers.arrive_and_expect_tx(rope_barrier_index, rope_copy_bytes)
            cute.copy(
                tma_atom_rope_a,
                t_ag_rope_a[(None, 0, cluster_index)],
                t_as_rope_a[(None, 0)],
                tma_bar_ptr=rope_bar_ptr,
            )
            cute.copy(
                tma_atom_rope_b,
                t_bg_rope_b[(None, 0, key_tile_index)],
                t_bs_rope_b[(None, 0)],
                tma_bar_ptr=rope_bar_ptr,
            )

        if warp_idx == 8 and cta_rank == 0:
            mma_producer.acquire_and_advance()
            if cutlass.const_expr(NATIVE_QK == 1):
                native_qk_mma.set(tcgen05.Field.ACCUMULATE, False)
                for latent_tile in cutlass.range_constexpr(NATIVE_LATENT_K_TILES):
                    tma_barriers.wait(barrier_base + latent_tile, 0)
                    for k_block in cutlass.range(
                        native_k_blocks, unroll_full=True
                    ):
                        cute.gemm(
                            native_qk_mma,
                            native_acc,
                            native_q[None, None, k_block, latent_tile],
                            native_k[None, None, k_block, latent_tile],
                            native_acc,
                        )
                        native_qk_mma.set(tcgen05.Field.ACCUMULATE, True)
            else:
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
        local_score_coords = cute.make_identity_tensor((ROWS_PER_CTA, TOKENS))
        if tidx < SCORE_ROWS and cutlass.const_expr(NATIVE_QK == 0):
            score_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                cutlass.Float32,
            )
            score_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                cutlass.Float32,
            )
            score_load = tcgen05.make_tmem_copy(score_load_atom, acc_tile)
            score_store = tcgen05.make_tmem_copy(score_store_atom, acc_tile)
            load_thr = score_load.get_slice(tidx)
            store_thr = score_store.get_slice(tidx)
            load_src = load_thr.partition_S(acc_tile)
            load_regs_layout = load_thr.partition_D(local_score_coords)
            store_regs_layout = store_thr.partition_S(local_score_coords)
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

        if warp_idx == 8 and cta_rank == 0:
            tma_barriers.wait(barrier_base + QK_BARRIER_SLOTS, 0)
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

        if tidx < SCORE_ROWS:
            # Group-two M128 folds each CTA-local M64 x N128 score tile across
            # two 64-column lane groups.  Load only the calling lane's N64
            # half, then exchange max/sum with its paired lane group exactly as
            # the stock TokenSpeed group-two softmax does.
            score_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                cutlass.Float32,
            )
            score_load = tcgen05.make_tmem_copy(score_load_atom, acc_tile)
            load_thr = score_load.get_slice(tidx)
            load_src = load_thr.partition_S(acc_tile)
            load_reg_layout = load_thr.partition_D(local_score_coords)
            load_regs = cute.make_fragment_like(load_reg_layout, cutlass.Float32)
            cute.copy(score_load, load_src, load_regs)
            cute.arch.fence_view_async_tmem_load()

            if cutlass.const_expr(tile == MASKED_TILE):
                for element in cutlass.range_constexpr(cute.size(load_regs)):
                    load_regs[element] = cutlass.Float32(-1.0e6)

            tile_row_max = load_regs.load().reduce(
                cute.ReductionOp.MAX, cutlass.Float32(-1.0e6), 0
            )
            softmax_max_exchange[tidx] = tile_row_max
            cute.arch.fence_view_async_shared()
            if warp_idx % 2 == 0:
                softmax_barrier_pair_02.wait()
            else:
                softmax_barrier_pair_13.wait()
            tile_row_max = cute.arch.fmax(
                tile_row_max, softmax_max_exchange[(tidx + N64) % SCORE_ROWS]
            )

            stage = tile % P_STAGES
            local_row = load_reg_layout[0][0]
            row_max_new = cute.arch.fmax(online_row_max, tile_row_max)
            prior_correction = cutlass.Float32(1.0)
            if cutlass.const_expr(tile > 0):
                prior_correction = cute.math.exp2(
                    (online_row_max - row_max_new) * SOFTMAX_SCALE_LOG2,
                    fastmath=False,
                )

            tile_row_sum = cutlass.Float32(0.0)
            for element in cutlass.range_constexpr(cute.size(load_regs)):
                token = load_reg_layout[element][1]
                probability = cutlass.Float32(0.0)
                if cutlass.const_expr(tile != MASKED_TILE):
                    probability = cute.math.exp2(
                        (load_regs[element] - row_max_new) * SOFTMAX_SCALE_LOG2,
                        fastmath=False,
                    )
                tile_row_sum += probability
                p_value = probability
                if cutlass.const_expr(NATIVE_PV == 0):
                    p_value = (
                        probability
                        * token_scale[tile, token].to(cutlass.Float32)
                        / carrier_scale
                    )
                p_coordinate = (
                    (local_row, token % 32),
                    0,
                    token // 32,
                    stage,
                )
                if cutlass.const_expr(NATIVE_PV == 1):
                    native_p_smem[p_coordinate] = p_value.to(
                        cutlass.Float8E4M3FN
                    )
                else:
                    p_smem[p_coordinate] = p_value.to(cutlass.Float8E4M3FN)

            softmax_sum_exchange[tidx] = tile_row_sum
            cute.arch.fence_view_async_shared()
            if warp_idx % 2 == 0:
                softmax_barrier_pair_02.wait()
            else:
                softmax_barrier_pair_13.wait()
            tile_row_sum += softmax_sum_exchange[(tidx + N64) % SCORE_ROWS]
            row_sum_new = prior_correction * online_row_sum + tile_row_sum

            carrier_exp = carrier_exp_stage_smem[stage]
            advance_g = cutlass.Int32(0)
            no_correction = cutlass.Int32(0)
            if cutlass.const_expr(tile == 0):
                no_correction = cutlass.Int32(1)
            else:
                prior_carrier = carrier_stage_smem[(tile - 1) % CORRECTION_STAGES]
                if carrier_scale != prior_carrier:
                    advance_g = cutlass.Int32(1)
                combined_correction = (
                    prior_correction * prior_carrier / carrier_scale
                )
                if combined_correction == cutlass.Float32(1.0):
                    no_correction = cutlass.Int32(1)
            if cutlass.const_expr(tile == MASKED_TILE):
                advance_g = cutlass.Int32(0)

            flags_scale = (
                no_correction
                | (advance_g << cutlass.Int32(1))
                | ((carrier_exp + cutlass.Int32(127)) << cutlass.Int32(8))
            )
            pcor_store_thr = pcor_store.get_slice(tidx)
            pcor_regs_layout = pcor_store_thr.partition_S(pcor_coords)
            pcor_dst = pcor_store_thr.partition_D(p_cor)
            pcor_regs = cute.make_fragment_like(
                pcor_regs_layout[None, None, None, stage], cutlass.Float32
            )
            pcor_regs_i32 = cute.make_tensor(
                cute.recast_ptr(pcor_regs.iterator, dtype=cutlass.Int32),
                pcor_regs.layout,
            )
            pcor_regs[0] = row_sum_new
            pcor_regs[1] = row_max_new
            pcor_regs[2] = prior_correction
            pcor_regs_i32[3] = flags_scale
            cute.copy(
                pcor_store,
                pcor_regs,
                pcor_dst[None, None, None, stage],
            )

            if tidx < ROWS_PER_CTA:
                max_output[cta_global, tile, local_row] = row_max_new
                sum_output[cta_global, tile, local_row] = row_sum_new
                owner_output[cta_global, tile, local_row] = (
                    cta_rank * ROWS_PER_CTA + local_row + 1
                )
                correction_output[cta_global, tile, local_row, 0] = row_sum_new
                correction_output[cta_global, tile, local_row, 1] = row_max_new
                correction_output[cta_global, tile, local_row, 2] = prior_correction
                flags_output[cta_global, tile, local_row] = flags_scale

            online_row_max = row_max_new
            online_row_sum = row_sum_new

        # Make the p-correction record's TMEM store locally explicit rather
        # than relying on a fence in the following score or rescale phase.
        cute.arch.fence_view_async_tmem_store()
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # Diagnostic consumer: read through the same logical operand that
        # PV MMA will consume.  Export before the stage is recycled.
        if tidx < ROWS_PER_CTA:
            stage = tile % P_STAGES
            local_row = tidx
            for token in cutlass.range_constexpr(TOKENS):
                p_coordinate = (
                    (local_row, token % 32),
                    0,
                    token // 32,
                    stage,
                )
                if cutlass.const_expr(NATIVE_PV == 1):
                    p_output[cta_global, tile, local_row, token] = native_p_smem[
                        p_coordinate
                    ]
                else:
                    p_output[cta_global, tile, local_row, token] = p_smem[
                        p_coordinate
                    ]
        cute.arch.sync_threads()

        if cutlass.const_expr(tile > 0):
            consume_pv_tile(
                tidx,
                warp_idx,
                cta_global,
                cluster_index,
                tile - 1,
                tmem_ptr,
                vp_output_cols,
                vp_acc_fake,
                p_cor,
                carrier_stage_smem,
                correction_scale_smem,
                consumer_carrier_smem,
                v_tma_barriers,
                v_smem,
                packed_v_desc,
                native_v_desc,
                native_cache_v_desc,
                vp_a,
                vp_b,
                vp_mma,
                native_v_a,
                native_v_b,
                native_vp_mma,
                native_v_acc_fake,
                native_p_smem,
                native_p_a,
                native_cache_v_b,
                native_pv_mma,
                native_pv_acc_fake,
                vp_producer,
                vp_consumer,
                matrix_output,
                POISON,
                EXPORT_MATRIX,
                POPULATE,
                NATIVE_V,
                NATIVE_PV,
            )
        continue

    consume_pv_tile(
        tidx,
        warp_idx,
        cta_global,
        cluster_index,
        TILES - 1,
        tmem_ptr,
        vp_output_cols,
        vp_acc_fake,
        p_cor,
        carrier_stage_smem,
        correction_scale_smem,
        consumer_carrier_smem,
        v_tma_barriers,
        v_smem,
        packed_v_desc,
        native_v_desc,
        native_cache_v_desc,
        vp_a,
        vp_b,
        vp_mma,
        native_v_a,
        native_v_b,
        native_vp_mma,
        native_v_acc_fake,
        native_p_smem,
        native_p_a,
        native_cache_v_b,
        native_pv_mma,
        native_pv_acc_fake,
        vp_producer,
        vp_consumer,
        matrix_output,
        POISON,
        EXPORT_MATRIX,
        POPULATE,
        NATIVE_V,
        NATIVE_PV,
    )

    if tidx < ROWS_PER_CTA:
        correction_scale_smem[tidx] = (
            consumer_carrier_smem[0] / online_row_sum
        )
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()

    for latent_slice in cutlass.range_constexpr(LATENT_SLICES):
        if cutlass.const_expr(NATIVE_PV == 1):
            native_pv_acc = cute.make_tensor(
                tmem_ptr + OUTPUT_OFFSET + latent_slice * vp_output_cols,
                native_pv_acc_fake.layout,
            )
            if tidx < 128:
                t_acc = native_pv_acc[(None, None), 0, 0]
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                    cutlass.Float32,
                )
                tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, t_acc)
                thr_load = tmem_load.get_slice(tidx)
                g_fp32 = cute.make_tensor(
                    normalized_output.iterator
                    + (cta_global * LATENT_K + latent_slice * V_SLICE_COLS)
                    * ROWS_PER_CTA,
                    cute.make_layout(
                        (ROWS_PER_CTA, V_SLICE_COLS),
                        stride=(1, ROWS_PER_CTA),
                    ),
                )
                g_bf16 = cute.make_tensor(
                    bf16_output.iterator
                    + (cta_global * LATENT_K + latent_slice * V_SLICE_COLS)
                    * ROWS_PER_CTA,
                    cute.make_layout(
                        (ROWS_PER_CTA, V_SLICE_COLS),
                        stride=(1, ROWS_PER_CTA),
                    ),
                )
                output_coords = cute.make_identity_tensor(
                    (ROWS_PER_CTA, V_SLICE_COLS)
                )
                t_tmem = thr_load.partition_S(t_acc)
                t_fp32 = thr_load.partition_D(g_fp32)
                t_bf16 = thr_load.partition_D(g_bf16)
                r_coords = thr_load.partition_D(output_coords)
                r_acc = cute.make_fragment_like(t_fp32, cutlass.Float32)
                cute.copy(tmem_load, t_tmem, r_acc)
                cute.arch.fence_view_async_tmem_load()
                for element in cutlass.range_constexpr(cute.size(r_acc)):
                    row = r_coords[element][0]
                    r_acc[element] = r_acc[element] * correction_scale_smem[row]
                cute.autovec_copy(r_acc, t_fp32)
                r_bf16 = cute.make_fragment_like(t_bf16, cutlass.BFloat16)
                r_bf16.store(r_acc.load().to(cutlass.BFloat16))
                cute.autovec_copy(r_bf16, t_bf16)
        else:
            vp_acc = cute.make_tensor(
                tmem_ptr + OUTPUT_OFFSET + latent_slice * vp_output_cols,
                vp_acc_fake.layout,
            )
            if tidx < 128:
                t_acc = vp_acc[(None, None), 0, 0]
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)),
                    cutlass.Float32,
                )
                tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, t_acc)
                thr_load = tmem_load.get_slice(tidx)
                g_fp32 = cute.make_tensor(
                    normalized_output.iterator
                    + (cta_global * LATENT_K + latent_slice * V_SLICE_COLS)
                    * ROWS_PER_CTA,
                    cute.make_layout(
                        (V_SLICE_COLS, ROWS_PER_CTA),
                        stride=(ROWS_PER_CTA, 1),
                    ),
                )
                g_bf16 = cute.make_tensor(
                    bf16_output.iterator
                    + (cta_global * LATENT_K + latent_slice * V_SLICE_COLS)
                    * ROWS_PER_CTA,
                    cute.make_layout(
                        (V_SLICE_COLS, ROWS_PER_CTA),
                        stride=(ROWS_PER_CTA, 1),
                    ),
                )
                t_tmem = thr_load.partition_S(t_acc)
                t_fp32 = thr_load.partition_D(g_fp32)
                t_bf16 = thr_load.partition_D(g_bf16)
                for segment in cutlass.range_constexpr(ROWS_PER_CTA):
                    r_acc = cute.make_fragment_like(
                        t_fp32[None, None, segment], cutlass.Float32
                    )
                    cute.copy(
                        tmem_load,
                        t_tmem[None, None, segment],
                        r_acc,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    for element in cutlass.range_constexpr(cute.size(r_acc)):
                        r_acc[element] = (
                            r_acc[element] * correction_scale_smem[segment]
                        )
                    cute.autovec_copy(r_acc, t_fp32[None, None, segment])
                    r_bf16 = cute.make_fragment_like(
                        t_bf16[None, None, segment], cutlass.BFloat16
                    )
                    r_bf16.store(r_acc.load().to(cutlass.BFloat16))
                    cute.autovec_copy(r_bf16, t_bf16[None, None, segment])
        cute.arch.sync_threads()

    cute.arch.sync_threads()
    if warp_idx == 8 and cta_rank == 0:
        mma_producer.tail()
    if warp_idx == 8:
        if cutlass.const_expr(NATIVE_PV == 1):
            if cta_rank == 0:
                vp_producer.tail()
        else:
            vp_producer.tail()
    cute.arch.sync_threads()
    if warp_idx == 8:
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def ownership_probe(
    mixed_a_ptr: cute.Pointer,
    mixed_b_ptr: cute.Pointer,
    native_latent_ptr: cute.Pointer,
    native_v_ptr: cute.Pointer,
    rope_a_ptr: cute.Pointer,
    rope_b_ptr: cute.Pointer,
    layout_output: cute.Tensor,
    matrix_output: cute.Tensor,
    normalized_output: cute.Tensor,
    bf16_output: cute.Tensor,
    carrier_output: cute.Tensor,
    p_output: cute.Tensor,
    max_output: cute.Tensor,
    sum_output: cute.Tensor,
    owner_output: cute.Tensor,
    correction_output: cute.Tensor,
    flags_output: cute.Tensor,
    token_scale: cute.Tensor,
    CLUSTERS: cutlass.Constexpr[int],
    POISON: cutlass.Constexpr[int],
    EXPORT_MATRIX: cutlass.Constexpr[int],
    POPULATE: cutlass.Constexpr[int],
    NATIVE_V: cutlass.Constexpr[int],
    NATIVE_QK: cutlass.Constexpr[int],
    NATIVE_PV: cutlass.Constexpr[int],
    stream,
):
    (
        mixed_mma,
        native_qk_mma,
        rope_mma,
        vp_mma,
        native_vp_mma,
        native_pv_mma,
    ) = make_mmas()
    g_mixed_a = cute.make_tensor(
        mixed_a_ptr,
        cute.make_ordered_layout((SCORE_ROWS, LATENT_K, CLUSTERS), order=(1, 0, 2)),
    )
    g_mixed_b = cute.make_tensor(
        mixed_b_ptr,
        cute.make_ordered_layout((TOKENS, LATENT_K, CLUSTERS * TILES), order=(1, 0, 2)),
    )
    g_native_latent = cute.make_tensor(
        native_latent_ptr,
        cute.make_ordered_layout(
            (TOKENS, LATENT_K, CLUSTERS * TILES), order=(1, 0, 2)
        ),
    )
    g_native_latent_transpose = cute.make_tensor(
        g_native_latent.iterator,
        cute.select(g_native_latent.layout, mode=[1, 0, 2]),
    )
    g_rope_a = cute.make_tensor(
        rope_a_ptr,
        cute.make_ordered_layout((SCORE_ROWS, ROPE_K, CLUSTERS), order=(1, 0, 2)),
    )
    g_rope_b = cute.make_tensor(
        rope_b_ptr,
        cute.make_ordered_layout((TOKENS, ROPE_K, CLUSTERS * TILES), order=(1, 0, 2)),
    )
    packed_v_desc = cuda.create_tensor_map_tiled(
        g_mixed_b.iterator.toint(),
        cutlass.Float4E2M1FN,
        global_dims=[LATENT_K, TOKENS, CLUSTERS * TILES],
        global_strides=[
            V_FULL_PACKED_ROW_BYTES // 16,
            (TOKENS * V_FULL_PACKED_ROW_BYTES) // 16,
        ],
        box_dims=[V_SLICE_COLS, TOKENS, 1],
        swizzle=cuda.TensorMapSwizzle.none,
    )
    g_native_v = cute.make_tensor(
        native_v_ptr,
        cute.make_ordered_layout(
            (TOKENS, LATENT_K, CLUSTERS * TILES), order=(0, 1, 2)
        ),
    )
    native_v_desc = cuda.create_tensor_map_tiled(
        g_native_v.iterator.toint(),
        cutlass.Float8E4M3FN,
        global_dims=[TOKENS, LATENT_K, CLUSTERS * TILES],
        global_strides=[
            TOKENS // 16,
            (TOKENS * LATENT_K) // 16,
        ],
        box_dims=[TOKENS, V_SLICE_COLS, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    native_cache_v_desc = cuda.create_tensor_map_tiled(
        g_native_latent_transpose.iterator.toint(),
        cutlass.Float8E4M3FN,
        global_dims=[LATENT_K, TOKENS, CLUSTERS * TILES],
        global_strides=[
            LATENT_K // 16,
            (TOKENS * LATENT_K) // 16,
        ],
        box_dims=[V_SLICE_COLS // CLUSTER_SHAPE_MNK[0], TOKENS, 1],
        swizzle=cuda.TensorMapSwizzle.s64b,
    )

    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (mixed_mma.thr_id.shape,)
    )
    if cutlass.const_expr(native_qk_mma.thr_id.shape != mixed_mma.thr_id.shape):
        raise ValueError("native and candidate QK CTA groups differ")
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
    native_q_layout = sm100_utils.make_smem_layout_a(
        native_qk_mma,
        NATIVE_QK_TILER_MNK,
        cutlass.Float8E4M3FN,
        NATIVE_LATENT_K_TILES,
    )
    native_k_layout = sm100_utils.make_smem_layout_b(
        native_qk_mma,
        NATIVE_QK_TILER_MNK,
        cutlass.Float8E4M3FN,
        NATIVE_LATENT_K_TILES,
    )
    if cutlass.const_expr(
        cute.size_in_bytes(cutlass.Float8E4M3FN, native_q_layout) > Q_SMEM_BYTES
    ):
        raise ValueError(
            "native Q footprint exceeds candidate capacity: "
            f"{cute.size_in_bytes(cutlass.Float8E4M3FN, native_q_layout)} "
            f"!= {Q_SMEM_BYTES}"
        )
    if cutlass.const_expr(
        cute.size_in_bytes(cutlass.Float8E4M3FN, native_k_layout) > K_SMEM_BYTES
    ):
        raise ValueError(
            "native K footprint exceeds candidate capacity: "
            f"{cute.size_in_bytes(cutlass.Float8E4M3FN, native_k_layout)} "
            f"!= {K_SMEM_BYTES}"
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
    native_a_op = sm100_utils.cluster_shape_to_tma_atom_A(
        CLUSTER_SHAPE_MNK[:2], native_qk_mma.thr_id
    )
    native_b_op = sm100_utils.cluster_shape_to_tma_atom_B(
        CLUSTER_SHAPE_MNK[:2], native_qk_mma.thr_id
    )
    tma_atom_native_q, tma_tensor_native_q = cute.nvgpu.make_tiled_tma_atom_A(
        native_a_op,
        g_mixed_a,
        cute.slice_(native_q_layout, (None, None, None, 0)),
        NATIVE_QK_TILER_MNK,
        native_qk_mma,
        cta_layout_vmnk.shape,
    )
    tma_atom_native_k, tma_tensor_native_k = cute.nvgpu.make_tiled_tma_atom_B(
        native_b_op,
        g_native_latent,
        cute.slice_(native_k_layout, (None, None, None, 0)),
        NATIVE_QK_TILER_MNK,
        native_qk_mma,
        cta_layout_vmnk.shape,
    )
    rope_a_layout = sm100_utils.make_smem_layout_a(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    rope_b_layout = sm100_utils.make_smem_layout_b(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    rope_a_op = sm100_utils.cluster_shape_to_tma_atom_A(
        CLUSTER_SHAPE_MNK[:2], rope_mma.thr_id
    )
    rope_b_op = sm100_utils.cluster_shape_to_tma_atom_B(
        CLUSTER_SHAPE_MNK[:2], rope_mma.thr_id
    )
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
    native_v_layout = sm100_utils.make_smem_layout_a(
        native_vp_mma,
        VP_TILER_MNK,
        cutlass.Float8E4M3FN,
        V_OPERAND_STAGES,
    )
    if cutlass.const_expr(
        cute.size_in_bytes(cutlass.Float8E4M3FN, native_v_layout) != V_SMEM_BYTES
    ):
        raise ValueError(
            f"native V footprint changed: "
            f"{cute.size_in_bytes(cutlass.Float8E4M3FN, native_v_layout)}"
        )
    native_p_layout = sm100_utils.make_smem_layout_a(
        native_pv_mma,
        NATIVE_PV_TILER_MNK,
        cutlass.Float8E4M3FN,
        P_STAGES,
    )
    native_cache_v_layout = sm100_utils.make_smem_layout_b(
        native_pv_mma,
        NATIVE_PV_TILER_MNK,
        cutlass.Float8E4M3FN,
        V_OPERAND_STAGES,
    )
    if cutlass.const_expr(
        cute.size_in_bytes(cutlass.Float8E4M3FN, native_p_layout) != P_SMEM_BYTES
    ):
        raise ValueError(
            "native P footprint does not match candidate capacity: "
            f"{cute.size_in_bytes(cutlass.Float8E4M3FN, native_p_layout)} "
            f"!= {P_SMEM_BYTES}"
        )
    if cutlass.const_expr(
        cute.size_in_bytes(cutlass.Float8E4M3FN, native_cache_v_layout)
        > V_SMEM_BYTES
    ):
        raise ValueError(
            "native V footprint exceeds candidate capacity: "
            f"{cute.size_in_bytes(cutlass.Float8E4M3FN, native_cache_v_layout)} "
            f"!= {V_SMEM_BYTES}"
        )
    mixed_acc_layout = mixed_mma.make_fragment_C(
        mixed_mma.partition_shape_C(MIXED_TILER_MNK[:2])
    ).layout
    mixed_acc_cols = utils.get_num_tmem_alloc_cols(
        mixed_mma.make_fragment_C(
            mixed_mma.partition_shape_C(MIXED_TILER_MNK[:2])
        )
    )
    vp_output_cols = utils.get_num_tmem_alloc_cols(
        vp_mma.make_fragment_C(vp_mma.partition_shape_C(VP_TILER_MNK[:2]))
    )
    native_vp_output_cols = utils.get_num_tmem_alloc_cols(
        native_vp_mma.make_fragment_C(
            native_vp_mma.partition_shape_C(VP_TILER_MNK[:2])
        )
    )
    if cutlass.const_expr(native_vp_output_cols != vp_output_cols):
        raise ValueError(
            f"native/candidate PV output mismatch: "
            f"{native_vp_output_cols} != {vp_output_cols}"
        )
    native_pv_output_cols = utils.get_num_tmem_alloc_cols(
        native_pv_mma.make_fragment_C(
            native_pv_mma.partition_shape_C(NATIVE_PV_TILER_MNK[:2])
        )
    )
    if cutlass.const_expr(native_pv_output_cols != vp_output_cols):
        raise ValueError(
            f"native cooperative/candidate PV output mismatch: "
            f"{native_pv_output_cols} != {vp_output_cols}"
        )
    vp_a_shape = vp_mma.partition_shape_A(
        (VP_TILER_MNK[0], VP_TILER_MNK[2], V_OPERAND_STAGES)
    )
    vp_a_fake = vp_mma.get_slice(0).make_fragment_A(vp_a_shape)
    v_operand_cols = tcgen05.find_tmem_tensor_col_offset(vp_a_fake)
    if cutlass.const_expr(v_operand_cols != SCALE_OFFSET - V_OPERAND_OFFSET):
        raise ValueError(f"V operand footprint changed: {v_operand_cols}")
    if cutlass.const_expr(SCALE_OFFSET + sfa_cols + sfb_cols > P_COR_OFFSET):
        raise ValueError("scale state overlaps p-correction")
    if cutlass.const_expr(
        P_COR_OFFSET + CORRECTION_VALUES * CORRECTION_STAGES > SCORE_OFFSET
    ):
        raise ValueError("p-correction overlaps score")
    if cutlass.const_expr(mixed_acc_cols != 64):
        raise ValueError(f"folded score footprint changed: {mixed_acc_cols}")
    if cutlass.const_expr(SCORE_OFFSET + mixed_acc_cols > OUTPUT_OFFSET):
        raise ValueError("score overlaps persistent output")
    if cutlass.const_expr(vp_output_cols != 64):
        raise ValueError(f"PV output footprint changed: {vp_output_cols}")
    if cutlass.const_expr(
        OUTPUT_OFFSET + vp_output_cols * LATENT_SLICES != TMEM_ALLOC_COLS
    ):
        raise ValueError("four persistent output slices do not fill TMEM tail")
    rope_acc_layout = rope_mma.make_fragment_C(
        rope_mma.partition_shape_C(ROPE_TILER_MNK[:2])
    ).layout
    print(f"S3_MIXED_THR_ID={mixed_mma.thr_id}")
    print(f"S3_CTA_LAYOUT_VMNK={cta_layout_vmnk}")
    print(f"S3_MIXED_ACC_LAYOUT={mixed_acc_layout}")
    print(f"S3_ROPE_ACC_LAYOUT={rope_acc_layout}")
    print(f"S3_SCALE_COLS={sfa_cols + sfb_cols}")
    print(
        "S4_NATIVE_LAYOUT="
        f"q={cute.size_in_bytes(cutlass.Float8E4M3FN, native_q_layout)} "
        f"k={cute.size_in_bytes(cutlass.Float8E4M3FN, native_k_layout)} "
        f"p={cute.size_in_bytes(cutlass.Float8E4M3FN, native_p_layout)} "
        f"v={cute.size_in_bytes(cutlass.Float8E4M3FN, native_cache_v_layout)} "
        f"output_cols={native_pv_output_cols}"
    )
    print(
        "S4_L0_TMEM="
        f"v={v_operand_cols} scale={sfa_cols + sfb_cols} "
        f"pcor={CORRECTION_VALUES * CORRECTION_STAGES} "
        f"score={mixed_acc_cols} free={OUTPUT_OFFSET - SCORE_OFFSET - mixed_acc_cols} "
        f"output={vp_output_cols * LATENT_SLICES} total={TMEM_ALLOC_COLS}"
    )
    kernel = ownership_kernel(
        layout_output,
        matrix_output,
        normalized_output,
        bf16_output,
        carrier_output,
        p_output,
        max_output,
        sum_output,
        owner_output,
        correction_output,
        flags_output,
        token_scale,
        packed_v_desc,
        native_v_desc,
        native_cache_v_desc,
        mixed_mma,
        native_qk_mma,
        rope_mma,
        vp_mma,
        native_vp_mma,
        native_pv_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_native_q,
        tma_tensor_native_q,
        tma_atom_native_k,
        tma_tensor_native_k,
        tma_atom_rope_a,
        tma_tensor_rope_a,
        tma_atom_rope_b,
        tma_tensor_rope_b,
        mixed_a_layout,
        mixed_b_layout,
        native_q_layout,
        native_k_layout,
        rope_a_layout,
        rope_b_layout,
        sfa_layout,
        sfb_layout,
        sfa_cols,
        vp_output_cols,
        v_operand_cols,
        p_layout,
        native_v_layout,
        native_p_layout,
        native_cache_v_layout,
        cta_layout_vmnk,
        POISON,
        EXPORT_MATRIX,
        POPULATE,
        NATIVE_V,
        NATIVE_QK,
        NATIVE_PV,
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


def expected(
    query: torch.Tensor,
    key: torch.Tensor,
    rope_query: torch.Tensor,
    rope_key: torch.Tensor,
    token_scale: torch.Tensor,
    populate: bool,
    score_key: torch.Tensor | None = None,
    postscale_scores: bool = True,
    value_key: torch.Tensor | None = None,
    native_pv: bool = False,
) -> tuple[torch.Tensor, ...]:
    latent_key = key if score_key is None else score_key
    latent = torch.einsum("rl,tnl->trn", query.float(), latent_key.float())
    rope = torch.einsum("rl,tnl->trn", rope_query.float(), rope_key.float())
    if postscale_scores:
        latent = latent * token_scale.float().unsqueeze(1)
    scores = latent + rope
    if native_pv:
        raw_carrier = torch.ones(TILES, dtype=torch.float32)
    else:
        max_d = token_scale.float().max(dim=-1).values
        raw_carrier = torch.pow(2.0, torch.ceil(torch.log2(max_d / 224.0)))
    pv_key = key if value_key is None else value_key

    p_tiles = []
    max_tiles = []
    sum_tiles = []
    correction_tiles = []
    flag_tiles = []
    carrier_tiles = []
    row_max = torch.full((SCORE_ROWS,), -1.0e6, dtype=torch.float32)
    row_sum = torch.zeros(SCORE_ROWS, dtype=torch.float32)
    prior_carrier = torch.tensor(0.0, dtype=torch.float32)
    probability_tiles = []
    for tile in range(TILES):
        masked = tile == MASKED_TILE
        tile_scores = scores[tile]
        tile_max = tile_scores.max(dim=-1).values
        if masked:
            tile_max = torch.full_like(tile_max, -1.0e6)
        row_max_new = torch.maximum(row_max, tile_max)
        prior_correction = torch.ones_like(row_max)
        if tile > 0:
            prior_correction = torch.exp2(
                (row_max - row_max_new) * SOFTMAX_SCALE_LOG2
            )
        probabilities = torch.zeros_like(tile_scores)
        if not masked:
            probabilities = torch.exp2(
                (tile_scores - row_max_new.unsqueeze(-1)) * SOFTMAX_SCALE_LOG2
            )
        row_sum_new = prior_correction * row_sum + probabilities.sum(dim=-1)
        carrier = prior_carrier if masked else raw_carrier[tile]
        p_values = probabilities
        if not native_pv:
            p_values = (
                probabilities * token_scale[tile].float().unsqueeze(0) / carrier
            )
        p_tile = p_values.to(torch.float8_e4m3fn)

        exponent = int(torch.log2(carrier).item())
        if tile == 0:
            no_correction = torch.ones(SCORE_ROWS, dtype=torch.int32)
            advance_g = 0
        else:
            combined = prior_correction * prior_carrier / carrier
            no_correction = (combined == 1.0).to(torch.int32)
            advance_g = int(not masked and carrier != prior_carrier)
        flags = (
            no_correction
            | (advance_g << 1)
            | ((exponent + 127) << 8)
        )

        p_tiles.append(p_tile)
        max_tiles.append(row_max_new)
        sum_tiles.append(row_sum_new)
        correction_tiles.append(
            torch.stack((row_sum_new, row_max_new, prior_correction), dim=-1)
        )
        flag_tiles.append(flags)
        carrier_tiles.append(carrier)
        probability_tiles.append(probabilities)
        row_max = row_max_new
        row_sum = row_sum_new
        prior_carrier = carrier

    p_expected_full = torch.stack(p_tiles)
    max_expected_full = torch.stack(max_tiles)
    sum_expected_full = torch.stack(sum_tiles)
    correction_expected_full = torch.stack(correction_tiles)
    flags_expected_full = torch.stack(flag_tiles)
    carrier_scale = torch.stack(carrier_tiles)
    advance_counts = [
        int((max_expected_full[tile] > max_expected_full[tile - 1]).sum())
        for tile in range(1, TILES)
        if tile != MASKED_TILE
    ]
    if not any(count > 0 for count in advance_counts):
        raise AssertionError("diagnostic lacks a new-row-max win")
    if not any(count < SCORE_ROWS for count in advance_counts):
        raise AssertionError("diagnostic lacks an incumbent-row-max win")
    if carrier_scale[MASKED_TILE] != carrier_scale[MASKED_TILE - 1]:
        raise AssertionError("fully masked tile advanced the carrier")
    if torch.count_nonzero(p_expected_full[MASKED_TILE].float()) != 0:
        raise AssertionError("fully masked tile produced nonzero P")
    print(
        "S4_C0_PATTERN "
        f"advance_rows={','.join(str(count) for count in advance_counts)} "
        f"carrier={','.join(f'{value.item():.8f}' for value in carrier_scale)} "
        f"masked_tile={MASKED_TILE}",
        flush=True,
    )

    matrix_by_cta = []
    bound_by_cta = []
    normalized_by_cta = []
    normalized_bound_by_cta = []
    reference_by_cta = []
    eps = 2.0**-24
    for row_begin in (0, ROWS_PER_CTA):
        running = torch.zeros((LATENT_K, ROWS_PER_CTA), dtype=torch.float64)
        sum_abs = torch.zeros_like(running)
        running_tiles = []
        running_bounds = []
        for tile in range(TILES):
            p_tile = p_expected_full[
                tile, row_begin : row_begin + ROWS_PER_CTA
            ].float()
            tile_output = pv_key[tile].T.double() @ p_tile.T.double()
            tile_abs = pv_key[tile].T.double().abs() @ p_tile.T.double().abs()
            if not populate:
                tile_output.zero_()
                tile_abs.zero_()
            if tile == 0:
                running = tile_output
                sum_abs = tile_abs
            else:
                factor = (
                    correction_expected_full[tile, row_begin : row_begin + ROWS_PER_CTA, 2]
                    * carrier_scale[tile - 1]
                    / carrier_scale[tile]
                ).double()
                running = running * factor.unsqueeze(0) + tile_output
                sum_abs = sum_abs * factor.unsqueeze(0).abs() + tile_abs
            n_ops = (tile + 1) * (2 * TOKENS - 1) + 2 * tile
            gamma_n = (n_ops * eps) / (1.0 - n_ops * eps)
            running_tiles.append(running.float())
            running_bounds.append(
                torch.maximum(
                    torch.full_like(sum_abs, 2.0e-5),
                    8.0 * gamma_n * sum_abs,
                ).float()
            )
        matrix_by_cta.append(torch.stack(running_tiles))
        bound_by_cta.append(torch.stack(running_bounds))

        final_scale = (
            carrier_scale[-1]
            / sum_expected_full[-1, row_begin : row_begin + ROWS_PER_CTA]
        ).double()
        normalized = running * final_scale.unsqueeze(0)
        normalized_by_cta.append(normalized.float())
        n_ops = TILES * (2 * TOKENS - 1) + 2 * (TILES - 1) + 2
        gamma_n = (n_ops * eps) / (1.0 - n_ops * eps)
        normalized_sum_abs = sum_abs * final_scale.unsqueeze(0).abs()
        normalized_bound_by_cta.append(
            (
                torch.maximum(
                    torch.full_like(normalized_sum_abs, 2.0e-5),
                    8.0 * gamma_n * normalized_sum_abs,
                )
                + 2.0e-6 * normalized.abs()
                + 2.0e-5
            ).float()
        )

        final_weights = torch.stack(probability_tiles)[
            :, row_begin : row_begin + ROWS_PER_CTA
        ]
        reference_numerator = torch.zeros_like(running)
        for tile in range(TILES):
            weighted = (
                final_weights[tile]
                * token_scale[tile].float().unsqueeze(0)
            )
            reference_numerator += key[tile].T.double() @ weighted.T.double()
        reference = reference_numerator / sum_expected_full[
            -1, row_begin : row_begin + ROWS_PER_CTA
        ].double().unsqueeze(0)
        if not populate:
            reference.zero_()
        reference_by_cta.append(reference.float())

    matrix_expected = torch.stack(matrix_by_cta)
    matrix_bound = torch.stack(bound_by_cta)
    normalized_expected = torch.stack(normalized_by_cta)
    normalized_bound = torch.stack(normalized_bound_by_cta)
    attention_reference = torch.stack(reference_by_cta)

    p_expected = torch.stack(
        (
            p_expected_full[:, :ROWS_PER_CTA],
            p_expected_full[:, ROWS_PER_CTA:],
        ),
        dim=0,
    )
    max_expected = torch.stack(
        (
            max_expected_full[:, :ROWS_PER_CTA],
            max_expected_full[:, ROWS_PER_CTA:],
        ),
        dim=0,
    )
    sum_expected = torch.stack(
        (
            sum_expected_full[:, :ROWS_PER_CTA],
            sum_expected_full[:, ROWS_PER_CTA:],
        ),
        dim=0,
    )
    correction_expected = torch.stack(
        (
            correction_expected_full[:, :ROWS_PER_CTA],
            correction_expected_full[:, ROWS_PER_CTA:],
        ),
        dim=0,
    )
    flags_expected = torch.stack(
        (
            flags_expected_full[:, :ROWS_PER_CTA],
            flags_expected_full[:, ROWS_PER_CTA:],
        ),
        dim=0,
    )
    owners = torch.empty((2, TILES, ROWS_PER_CTA), dtype=torch.int32)
    owners[0] = torch.arange(1, ROWS_PER_CTA + 1, dtype=torch.int32).view(1, -1)
    owners[1] = torch.arange(ROWS_PER_CTA + 1, SCORE_ROWS + 1, dtype=torch.int32).view(
        1, -1
    )
    carrier_expected = carrier_scale.view(1, TILES).repeat(2, 1)
    return (
        p_expected,
        max_expected,
        sum_expected,
        owners,
        matrix_expected,
        carrier_expected,
        correction_expected,
        flags_expected,
        normalized_expected,
        matrix_bound,
        normalized_bound,
        attention_reference,
    )


def verify(
    matrix_output: torch.Tensor,
    normalized_output: torch.Tensor,
    bf16_output: torch.Tensor,
    carrier_output: torch.Tensor,
    p_output: torch.Tensor,
    max_output: torch.Tensor,
    sum_output: torch.Tensor,
    owner_output: torch.Tensor,
    correction_output: torch.Tensor,
    flags_output: torch.Tensor,
    expected_outputs: tuple[torch.Tensor, ...],
    verify_matrix: bool,
) -> None:
    (
        p_expected,
        max_expected,
        sum_expected,
        owners_expected,
        matrix_expected,
        carrier_expected,
        correction_expected,
        flags_expected,
        normalized_expected,
        matrix_bound,
        normalized_bound,
        attention_reference,
    ) = expected_outputs
    torch.testing.assert_close(
        p_output.cpu().float(), p_expected.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(max_output.cpu(), max_expected, rtol=0, atol=0)
    torch.testing.assert_close(sum_output.cpu(), sum_expected, rtol=2.0e-6, atol=2.0e-5)
    torch.testing.assert_close(owner_output.cpu(), owners_expected, rtol=0, atol=0)
    torch.testing.assert_close(
        correction_output.cpu()[..., :2],
        correction_expected[..., :2],
        rtol=2.0e-6,
        atol=2.0e-5,
    )
    torch.testing.assert_close(
        correction_output.cpu()[..., 2],
        correction_expected[..., 2],
        rtol=2.0e-6,
        atol=2.0e-5,
    )
    torch.testing.assert_close(flags_output.cpu(), flags_expected, rtol=0, atol=0)
    if verify_matrix:
        matrix_error = (matrix_output.cpu() - matrix_expected).abs()
        if not torch.all(matrix_error <= matrix_bound):
            matrix_cpu = matrix_output.cpu()
            for cta in range(matrix_cpu.shape[0]):
                for latent_begin in range(0, LATENT_K, V_SLICE_COLS):
                    actual_slice = matrix_cpu[cta, -1, latent_begin : latent_begin + V_SLICE_COLS]
                    expected_slice = matrix_expected[cta, -1, latent_begin : latent_begin + V_SLICE_COLS]
                    denominator = expected_slice.abs().sum().item()
                    l1_ratio = actual_slice.abs().sum().item() / max(denominator, 1.0e-20)
                    signed_ratio = actual_slice.sum().item() / max(expected_slice.sum().item(), 1.0e-20)
                    print(
                        "S4_NATIVE_PV_DIAG "
                        f"cta={cta} latent_begin={latent_begin} "
                        f"actual_l1={actual_slice.abs().sum().item():.8f} "
                        f"expected_l1={denominator:.8f} l1_ratio={l1_ratio:.8f} "
                        f"signed_ratio={signed_ratio:.8f}",
                        flush=True,
                    )
            worst = torch.unravel_index(matrix_error.argmax(), matrix_error.shape)
            raise AssertionError(
                f"persistent output exceeds bound: max_error={matrix_error.max()} "
                f"max_bound={matrix_bound.max()} worst={tuple(int(x) for x in worst)} "
                f"actual={matrix_output.cpu()[worst].item()} "
                f"expected={matrix_expected[worst].item()}"
            )
    normalized_cpu = normalized_output.cpu()
    normalized_error = (normalized_cpu - normalized_expected).abs()
    if not torch.all(normalized_error <= normalized_bound):
        raise AssertionError(
            f"normalized output exceeds bound: max_error={normalized_error.max()} "
            f"max_bound={normalized_bound.max()}"
        )
    torch.testing.assert_close(
        bf16_output.cpu().view(torch.int16),
        normalized_cpu.to(torch.bfloat16).view(torch.int16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        carrier_output.cpu(), carrier_expected, rtol=0, atol=0
    )
    quantized_contract_error = (normalized_expected - attention_reference).abs()
    print(
        "S4_C0_QUALITY "
        f"max_abs={quantized_contract_error.max().item():.8f} "
        f"mean_abs={quantized_contract_error.mean().item():.8f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=1)
    parser.add_argument("--graph-replays", type=int, default=100)
    parser.add_argument("--benchmark-replays", type=int, default=0)
    parser.add_argument("--benchmark-samples", type=int, default=7)
    parser.add_argument("--compare-native-windows", type=int, default=0)
    parser.add_argument("--compare-native-warmups", type=int, default=20)
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--no-matrix-export", action="store_true")
    parser.add_argument("--native-qk", action="store_true")
    parser.add_argument("--native-pv", action="store_true")
    parser.add_argument(
        "--v-population",
        choices=("candidate", "native", "floor"),
        default="candidate",
    )
    parser.add_argument(
        "--poison",
        type=lambda value: int(value, 0),
        choices=(-1, 0, 0x5A),
        default=0x5A,
    )
    args = parser.parse_args()
    if args.clusters < 1:
        parser.error("--clusters must be positive")
    if args.benchmark_replays < 0:
        parser.error("--benchmark-replays must be non-negative")
    if args.benchmark_samples < 1:
        parser.error("--benchmark-samples must be positive")
    if args.compare_native_windows < 0:
        parser.error("--compare-native-windows must be non-negative")
    if args.compare_native_warmups < 0:
        parser.error("--compare-native-warmups must be non-negative")
    if args.compare_native_windows and args.benchmark_replays < 1:
        parser.error("--compare-native-windows requires --benchmark-replays")
    if args.compare_native_windows and (
        args.v_population != "candidate" or args.native_qk or args.native_pv
    ):
        parser.error(
            "--compare-native-windows owns both arms and requires "
            "--v-population=candidate without native-arm flags"
        )
    if args.compare_native_windows and not args.no_matrix_export:
        parser.error("--compare-native-windows requires --no-matrix-export")
    if args.native_qk and not args.native_pv:
        parser.error("standalone --native-qk requires --native-pv")
    if args.native_qk and args.v_population != "native":
        parser.error("--native-qk requires --v-population=native")
    if args.native_pv and not args.native_qk:
        parser.error("--native-pv requires --native-qk")
    if (args.benchmark_replays or args.compare_native_windows) and args.poison != -1:
        parser.error("benchmark timing requires --poison=-1")
    ctas = CLUSTER_SHAPE_MNK[0] * args.clusters

    def compile_reader(
        populate: int, native_v: int, native_qk: int, native_pv: int
    ):
        return cute.compile(
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
            fake(cutlass.Int32, (CLUSTER_SHAPE_MNK[0], 17), 16),
            fake(cutlass.Float32, (ctas, TILES, LATENT_K, ROWS_PER_CTA), 16),
            fake(cutlass.Float32, (ctas, LATENT_K, ROWS_PER_CTA), 16),
            fake(cutlass.BFloat16, (ctas, LATENT_K, ROWS_PER_CTA), 16),
            fake(cutlass.Float32, (ctas, TILES), 16),
            fake(cutlass.Float8E4M3FN, (ctas, TILES, ROWS_PER_CTA, TOKENS), 16),
            fake(cutlass.Float32, (ctas, TILES, ROWS_PER_CTA), 16),
            fake(cutlass.Float32, (ctas, TILES, ROWS_PER_CTA), 16),
            fake(cutlass.Int32, (ctas, TILES, ROWS_PER_CTA), 16),
            fake(cutlass.Float32, (ctas, TILES, ROWS_PER_CTA, 3), 16),
            fake(cutlass.Int32, (ctas, TILES, ROWS_PER_CTA), 16),
            fake(cutlass.BFloat16, (TILES, TOKENS), 16),
            args.clusters,
            args.poison,
            0 if args.no_matrix_export else 1,
            populate,
            native_v,
            native_qk,
            native_pv,
            make_fake_stream(),
            options="--enable-tvm-ffi --opt-level 3",
        )

    if args.compare_native_windows:
        comparison_compiled = {
            "candidate": compile_reader(1, 0, 0, 0),
            "native": compile_reader(1, 0, 1, 1),
        }
        compiled = comparison_compiled.get(
            args.v_population, comparison_compiled["candidate"]
        )
    else:
        comparison_compiled = None
        compiled = compile_reader(
            0 if args.v_population == "floor" else 1,
            int(args.v_population == "native" and not args.native_pv),
            int(args.native_qk),
            int(args.native_pv),
        )
    if args.compile_only:
        print(
            "PASS_C1_M0QP_S4_C0_COMPILE_ONLY "
            f"clusters={args.clusters} tiles={TILES} p_stages={P_STAGES} "
            f"smem_payload={SMEM_PAYLOAD_BYTES}"
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
    key_base = (
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
    key_base = (key_base - 1.0) * 0.5
    key_factors = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0]).view(TILES, 1, 1)
    key = key_base.unsqueeze(0) * key_factors
    rope_query = (
        (((row + 3) * (rope_coordinate + 1) * 11 + row * 13) % 127) % 3
    ).float()
    rope_query = (rope_query - 1.0) * 0.5
    rope_key_base = (
        (((token + 5) * (rope_coordinate + 7) * 19 + token * 17) % 131) % 3
    ).float()
    rope_key_base = (rope_key_base - 1.0) * 0.5
    rope_factors = torch.tensor([1.0, 0.5, 2.0, 1.0, 1.0]).view(TILES, 1, 1)
    rope_key = rope_key_base.unsqueeze(0) * rope_factors
    tile = torch.arange(TILES, dtype=torch.int64).view(TILES, 1)
    token_row = torch.arange(TOKENS, dtype=torch.int64).view(1, TOKENS)
    scale_factors = torch.tensor([1.0, 2.0, 0.5, 8.0, 4.0]).view(TILES, 1)
    token_scale = (
        (0.75 + ((tile * 7 + token_row * 3) % 17).float() / 32.0)
        * scale_factors
    ).to(torch.bfloat16)
    native_latent_values = None
    need_native_latent = args.native_qk or bool(args.compare_native_windows)
    if need_native_latent:
        native_latent_values = (
            key.float() * token_scale.float().unsqueeze(-1)
        ).to(torch.float8_e4m3fn)
    candidate_expected_outputs = expected(
        query,
        key,
        rope_query,
        rope_key,
        token_scale,
        args.v_population != "floor",
    )
    native_expected_outputs = None
    if need_native_latent:
        native_expected_outputs = expected(
            query,
            key,
            rope_query,
            rope_key,
            token_scale,
            args.v_population != "floor",
            native_latent_values.float(),
            False,
            native_latent_values.float(),
            True,
        )

    def repeat_clusters(values):
        return tuple(
            value.repeat((args.clusters,) + (1,) * (value.ndim - 1))
            for value in values
        )

    expected_outputs_by_arm = {
        "candidate": repeat_clusters(candidate_expected_outputs),
    }
    if native_expected_outputs is not None:
        expected_outputs_by_arm["native"] = repeat_clusters(native_expected_outputs)
    selected_expected_arm = "native" if args.native_pv else "candidate"

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
        query.unsqueeze(0).repeat(args.clusters, 1, 1), cutlass.Float8E4M3FN
    )
    key_cute = to_cute_tensor(
        key.repeat(args.clusters, 1, 1),
        cutlass.Float4E2M1FN,
    )
    if need_native_latent:
        native_latent_cute = to_cute_tensor(
            native_latent_values.float().repeat(args.clusters, 1, 1),
            cutlass.Float8E4M3FN,
        )
    else:
        native_latent_cute = to_cute_tensor(
            torch.zeros(16, dtype=torch.float32), cutlass.Float8E4M3FN
        )
    if (
        args.v_population == "native"
        and not args.native_pv
        and not args.compare_native_windows
    ):
        native_v_cute = to_cute_tensor(
            key.repeat(args.clusters, 1, 1).permute(0, 2, 1).contiguous(),
            cutlass.Float8E4M3FN,
        )
    else:
        native_v_cute = to_cute_tensor(
            torch.zeros(16, dtype=torch.float32), cutlass.Float8E4M3FN
        )
    rope_query_cute = to_cute_tensor(
        rope_query.unsqueeze(0).repeat(args.clusters, 1, 1), cutlass.Float8E4M3FN
    )
    rope_key_cute = to_cute_tensor(
        rope_key.repeat(args.clusters, 1, 1),
        cutlass.Float8E4M3FN,
    )
    token_scale = token_scale.cuda().contiguous()

    layout_output = torch.zeros(
        (CLUSTER_SHAPE_MNK[0], 17), dtype=torch.int32, device="cuda"
    )
    matrix_output = torch.empty(
        (ctas, TILES, LATENT_K, ROWS_PER_CTA),
        dtype=torch.float32,
        device="cuda",
    )
    normalized_output = torch.empty(
        (ctas, LATENT_K, ROWS_PER_CTA), dtype=torch.float32, device="cuda"
    )
    bf16_output = torch.empty(
        (ctas, LATENT_K, ROWS_PER_CTA), dtype=torch.bfloat16, device="cuda"
    )
    carrier_output = torch.empty(
        (ctas, TILES), dtype=torch.float32, device="cuda"
    )
    p_output = torch.empty(
        (ctas, TILES, ROWS_PER_CTA, TOKENS),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    max_output = torch.empty(
        (ctas, TILES, ROWS_PER_CTA), dtype=torch.float32, device="cuda"
    )
    sum_output = torch.empty_like(max_output)
    owner_output = torch.zeros(
        (ctas, TILES, ROWS_PER_CTA), dtype=torch.int32, device="cuda"
    )
    correction_output = torch.empty(
        (ctas, TILES, ROWS_PER_CTA, 3), dtype=torch.float32, device="cuda"
    )
    flags_output = torch.empty(
        (ctas, TILES, ROWS_PER_CTA), dtype=torch.int32, device="cuda"
    )

    def launch_reader(reader, launch_stream) -> None:
        reader(
            query_cute.iterator,
            key_cute.iterator,
            native_latent_cute.iterator,
            native_v_cute.iterator,
            rope_query_cute.iterator,
            rope_key_cute.iterator,
            layout_output,
            matrix_output,
            normalized_output,
            bf16_output,
            carrier_output,
            p_output,
            max_output,
            sum_output,
            owner_output,
            correction_output,
            flags_output,
            token_scale,
            launch_stream,
        )

    def expected_layout(
        native_v: bool, native_qk: bool, native_pv: bool
    ) -> torch.Tensor:
        return torch.tensor(
            [
                [
                    1, 0, 64, 64, 20, 84, 8, 128, 64, 256, 256, 512,
                    8192, 16384, int(native_v), int(native_qk),
                    int(native_pv),
                ],
                [
                    2, 0, 64, 64, 20, 84, 8, 128, 64, 256, 256, 512,
                    8192, 16384, int(native_v), int(native_qk),
                    int(native_pv),
                ],
            ],
            dtype=torch.int32,
        )

    def verify_current(arm: str) -> None:
        if comparison_compiled is not None:
            native_arm = arm == "native"
            native_v_flag = False
            native_qk_flag = native_arm
            native_pv_flag = native_arm
        else:
            native_v_flag = args.v_population == "native" and not args.native_pv
            native_qk_flag = args.native_qk
            native_pv_flag = args.native_pv
        layout_expected = expected_layout(
            native_v_flag, native_qk_flag, native_pv_flag
        )
        torch.testing.assert_close(
            layout_output.cpu()[:, : layout_expected.shape[1]],
            layout_expected,
            rtol=0,
            atol=0,
        )
        verify(
            matrix_output,
            normalized_output,
            bf16_output,
            carrier_output,
            p_output,
            max_output,
            sum_output,
            owner_output,
            correction_output,
            flags_output,
            expected_outputs_by_arm[arm],
            not args.no_matrix_export,
        )

    def reset_verification_outputs() -> None:
        matrix_output.fill_(float("nan"))
        normalized_output.fill_(float("nan"))
        bf16_output.fill_(float("nan"))
        carrier_output.fill_(float("nan"))
        p_output.zero_()
        max_output.fill_(float("nan"))
        sum_output.fill_(float("nan"))
        owner_output.zero_()
        correction_output.fill_(float("nan"))
        flags_output.zero_()

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    if comparison_compiled is not None:
        for arm in ("candidate", "native"):
            reset_verification_outputs()
            launch_reader(comparison_compiled[arm], stream)
            torch.cuda.synchronize()
            verify_current(arm)
    else:
        launch_reader(compiled, stream)
        torch.cuda.synchronize()
        verify_current(selected_expected_arm)

    if args.compare_native_windows:
        comparison_graphs = {}
        for arm in ("candidate", "native"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_stream = cuda_driver.CUstream(
                    torch.cuda.current_stream().cuda_stream
                )
                for _ in range(args.benchmark_replays):
                    launch_reader(comparison_compiled[arm], graph_stream)
            comparison_graphs[arm] = graph

        for warmup in range(args.compare_native_warmups):
            order = ("candidate", "native") if warmup % 2 == 0 else (
                "native",
                "candidate",
            )
            for arm in order:
                comparison_graphs[arm].replay()
        torch.cuda.synchronize()

        comparison_us = {"candidate": [], "native": []}
        comparison_orders = []
        for window in range(args.compare_native_windows):
            order = ("candidate", "native") if window % 2 == 0 else (
                "native",
                "candidate",
            )
            window_events = {}
            for arm in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                comparison_graphs[arm].replay()
                end.record()
                window_events[arm] = (start, end)
            window_events[order[-1]][1].synchronize()
            for arm in order:
                start, end = window_events[arm]
                value = start.elapsed_time(end) * 1000.0 / args.benchmark_replays
                comparison_us[arm].append(value)
            comparison_orders.append(order)

        for window, order in enumerate(comparison_orders):
            print(
                "COMPARE_WINDOW_C1_M0QP_S4_C0 "
                f"window={window} order={','.join(order)} "
                f"candidate_us={comparison_us['candidate'][window]:.6f} "
                f"native_us={comparison_us['native'][window]:.6f} "
                f"ratio={comparison_us['candidate'][window] / comparison_us['native'][window]:.8f}",
                flush=True,
            )

        for arm in ("candidate", "native"):
            reset_verification_outputs()
            comparison_graphs[arm].replay()
            torch.cuda.synchronize()
            verify_current(arm)

        paired_log_ratios = [
            math.log(candidate / native)
            for candidate, native in zip(
                comparison_us["candidate"], comparison_us["native"]
            )
        ]
        geometric_ratio = math.exp(statistics.mean(paired_log_ratios))
        # This component experiment reports a descriptive normal-approximation
        # interval.  The endpoint production gate uses its own preregistered
        # replicated benchmark and confidence procedure.
        if len(paired_log_ratios) > 1:
            log_se = statistics.stdev(paired_log_ratios) / math.sqrt(
                len(paired_log_ratios)
            )
            ratio_ci_low = math.exp(
                statistics.mean(paired_log_ratios) - 1.96 * log_se
            )
            ratio_ci_high = math.exp(
                statistics.mean(paired_log_ratios) + 1.96 * log_se
            )
        else:
            ratio_ci_low = float("nan")
            ratio_ci_high = float("nan")
        print(
            "COMPARE_C1_M0QP_S4_C0 "
            f"clusters={args.clusters} windows={args.compare_native_windows} "
            f"warmups={args.compare_native_warmups} "
            f"replays_per_graph={args.benchmark_replays} "
            f"candidate_mean_us={statistics.mean(comparison_us['candidate']):.6f} "
            f"native_mean_us={statistics.mean(comparison_us['native']):.6f} "
            f"candidate_median_us={statistics.median(comparison_us['candidate']):.6f} "
            f"native_median_us={statistics.median(comparison_us['native']):.6f} "
            f"matrix_export={not args.no_matrix_export} "
            "normalized_export=True "
            f"geometric_ratio={geometric_ratio:.8f} "
            f"ratio_ci95_low={ratio_ci_low:.8f} "
            f"ratio_ci95_high={ratio_ci_high:.8f}",
            flush=True,
        )

    if args.benchmark_replays and not args.compare_native_windows:
        # Time repeated kernel nodes inside one graph so Python dispatch cannot
        # dominate this bounded group-one/group-two schedule comparison.
        benchmark_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(benchmark_graph):
            benchmark_stream = cuda_driver.CUstream(
                torch.cuda.current_stream().cuda_stream
            )
            for _ in range(args.benchmark_replays):
                launch_reader(compiled, benchmark_stream)
        benchmark_graph.replay()
        torch.cuda.synchronize()
        benchmark_us = []
        for _ in range(args.benchmark_samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            benchmark_graph.replay()
            end.record()
            end.synchronize()
            benchmark_us.append(
                start.elapsed_time(end) * 1000.0 / args.benchmark_replays
            )
        verify_current(selected_expected_arm)
        print(
            "BENCH_C1_M0QP_S4_C0 "
            f"clusters={args.clusters} "
            f"padding_poison={args.poison} "
            f"replays_per_graph={args.benchmark_replays} "
            f"samples={args.benchmark_samples} "
            f"median_us={statistics.median(benchmark_us):.6f} "
            f"min_us={min(benchmark_us):.6f} max_us={max(benchmark_us):.6f} "
            f"all_us={','.join(f'{value:.6f}' for value in benchmark_us)}",
            flush=True,
        )

    if not args.skip_graph and args.graph_replays:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            capture_stream = cuda_driver.CUstream(
                torch.cuda.current_stream().cuda_stream
            )
            launch_reader(compiled, capture_stream)
        # A graph that captured no launch would leave these sentinels unchanged.
        # Poison after capture so replay cannot pass on stale eager data.
        # Full-range carrier values are nonnegative, so this sentinel cannot
        # equal a valid P element if a graph replay only partially writes it.
        p_output.fill_(-1.0)
        matrix_output.fill_(float("nan"))
        normalized_output.fill_(float("nan"))
        bf16_output.zero_()
        carrier_output.fill_(float("nan"))
        layout_output.fill_(-1)
        max_output.fill_(-1234.0)
        sum_output.fill_(-1234.0)
        owner_output.zero_()
        correction_output.fill_(float("nan"))
        flags_output.fill_(-1)
        for _ in range(args.graph_replays):
            graph.replay()
        torch.cuda.synchronize()
        verify_current(selected_expected_arm)

    print(
        "PASS_C1_M0QP_S4_C0_DELAYED_COMPLETE_READER "
        f"clusters={args.clusters} tiles={TILES} "
        f"score_wraps={TILES - 1} p_stages={P_STAGES} "
        f"p_wraps={TILES - P_STAGES} owner_lanes=128_per_cta "
        f"masked_tile={MASKED_TILE} consumer_delay=1 drain_tiles=1 "
        f"qk={'fp8xfp8' if args.native_qk else 'fp8xfp4'} "
        f"token_postscale={not args.native_qk} rope_k=64 cta_group=2 "
        "folded_n64_exchange=True peer_p_dsm=False "
        f"v_population={args.v_population} matrix_export={not args.no_matrix_export} "
        f"padding_poison={args.poison} "
        f"graph_replays={0 if args.skip_graph else args.graph_replays} "
        f"smem_payload={SMEM_PAYLOAD_BYTES}"
    )


if __name__ == "__main__":
    main()
