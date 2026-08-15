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

"""Rejected research prototype for native-E2M1 MLA decode on Blackwell.

The kernel consumes the no-shadow TurboQuant layout directly with native mixed
FP8-query/E2M1 QK and FP8-probability/E2M1 PV MMAs.  This module initially owns
one page-32 request split of at most five 128-token tiles; multi-split scheduling
and serving-backend integration remain separate stages.

The D0 saturated timing gate rejected this two-CTA ownership architecture even
though its RN path improved its matched C0 control.  Keep this module only as a
reproducible mechanism and correctness artifact on the rejected research
branch.  It must not be integrated or exported by a production build.
"""

import functools
import math
from typing import Optional

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch.nvvm_wrappers import FULL_MASK
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
    make_fake_tensor,
)
from cutlass.cute.typing import Float32, Int, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims


# Standalone-probe copy of TokenSpeed's centralized Blackwell helper from
# tokenspeed-kernel/thirdparty/cute_dsl/argmax.py. Production integration must
# import that utility rather than duplicate it.
@dsl_user_op
def ptx_redux_sync_max_f32(
    value: Float32, mask: Int = FULL_MASK, *, loc=None, ip=None
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(value).ir_value(loc=loc, ip=ip),
                Int32(mask).ir_value(loc=loc, ip=ip),
            ],
            """redux.sync.max.f32 $0, $1, $2;""",
            "=f,f,i",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


THREADS_PER_CTA = 384
TMEM_RETRIEVE_THREADS = 288
CLUSTER_SHAPE_MNK = (2, 1, 1)
TILES = 5
PAGE_SIZE = 32
ROWS_PER_CTA = 64
SCORE_ROWS = 128
TOKENS = 128
MAX_PAGES = TILES * TOKENS // PAGE_SIZE
N64 = 64
P_STAGES = 2
CORRECTION_VALUES = 4
CORRECTION_STAGES = 2
V_OPERAND_STAGES = 2
INITIAL_MIXED_V_PREFETCHES = 2
INITIAL_NATIVE_V_PREFETCHES = V_OPERAND_STAGES
NUM_HEADS = 8
QUERY_DIM = 576

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
NATIVE_PV_TILER_MNK = (128, 128, 128)
MIXED_PV_TILER_MNK = (128, 256, 128)
MIXED_PV_LATENT_SLICES = LATENT_K // MIXED_PV_TILER_MNK[1]
LIVE_SCALE_COLS = 20
SCALE_OFFSET = 64
MIXED_PV_SCALE_OFFSET = 0
MIXED_PV_SCALE_RESERVE_COLS = 64
P_COR_OFFSET = 84
SCORE_OFFSET = 128
OUTPUT_OFFSET = 256
TMEM_ALLOC_COLS = 512
PROBE_SOFTMAX_SCALE_LOG2 = 0.015625

NATIVE_PV_LATENT_SLICES = LATENT_K // NATIVE_PV_TILER_MNK[1]
NATIVE_PV_SLICE_COLS = NATIVE_PV_TILER_MNK[1]
MIXED_PV_SLICE_COLS = MIXED_PV_TILER_MNK[1]
# Compatibility aliases for the strict-native output code retained below.
LATENT_SLICES = NATIVE_PV_LATENT_SLICES
V_SLICE_COLS = NATIVE_PV_SLICE_COLS

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


@cute.jit
def make_paged_tiled_tma_atom(
    tma_load_op: cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp,
    gmem: cute.Tensor,
    smem_layout: cute.Layout,
    mma_tiler,
    tiled_mma: cute.TiledMma,
    is_k_load: cutlass.Constexpr[bool],
    internal_type=None,
):
    """Build the same non-executable page TMA atom as stock MLA decode."""

    ident = cute.make_identity_layout(gmem.shape)
    g_tile = cute.composition(ident, mma_tiler)
    cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape
    cta_v_map = cute.flat_divide(g_tile, (cta_mn,))
    cta_v_map = cute.select(cta_v_map, mode=[0, 2])
    page_tile_size = (
        min(PAGE_SIZE, cta_mn) if is_k_load else min(PAGE_SIZE, mma_tiler[1])
    )
    cta_v_map = cute.zipped_divide(
        cta_v_map,
        (page_tile_size, mma_tiler[1]) if is_k_load else (cta_mn, page_tile_size),
    )
    cta_v_map = cute.select(cta_v_map, mode=[0])
    from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir

    if cutlass.const_expr(internal_type is not None):
        use_unpack = internal_type.width == 8 and gmem.element_type.width < 8
        internal_mlir_type = (
            gmem.element_type.mlir_type if use_unpack else internal_type.mlir_type
        )
        tma_format = _cute_nvgpu_ir.TmaDataFormat(
            _cute_nvgpu_ir.get_default_tma_format(internal_mlir_type, use_unpack)
        )
        result = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
            gmem.value,
            smem_layout.value,
            cta_v_map,
            tma_load_op._to_ir(),
            num_multicast=1,
            tma_format=tma_format,
        )
    else:
        result = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
            gmem.value,
            smem_layout.value,
            cta_v_map,
            tma_load_op._to_ir(),
            num_multicast=1,
        )
    return (
        cute.CopyAtom(
            tma_load_op,
            cpasync.CopyBulkTensorTileG2SNonExecTrait(result[0]),
        ),
        result[1],
    )


@cute.jit
def physical_page_for(
    block_table,
    cluster_index,
    logical_page,
    MULTI_REQUEST: cutlass.Constexpr[int],
):
    if cutlass.const_expr(MULTI_REQUEST == 1):
        return block_table[cluster_index * MAX_PAGES + logical_page]
    return block_table[logical_page]


@cute.jit
def consume_pv_tile(
    tidx,
    warp_idx,
    cta_global,
    cluster_index,
    consume_tile: cutlass.Constexpr[int],
    tmem_ptr,
    mixed_pv_output_cols,
    native_pv_output_cols,
    mixed_pv_acc_fake,
    native_pv_acc_fake,
    p_cor,
    carrier_stage_smem,
    correction_scale_smem,
    consumer_carrier_smem,
    v_tma_mbar,
    mixed_v_copy_bytes,
    raw_v_smem,
    native_cache_v_desc,
    tma_atom_mixed_pv_v,
    t_mixed_vs_v,
    t_mixed_vg_v,
    block_table,
    mixed_p_a,
    mixed_v_b,
    mixed_pv_mma,
    t_mixed_pv_sfa,
    t_mixed_pv_sfb,
    native_p_smem,
    native_p_a,
    native_cache_v_b,
    native_pv_mma,
    vp_producer,
    vp_consumer,
    NATIVE_PV: cutlass.Constexpr[int],
    MULTI_REQUEST: cutlass.Constexpr[int],
):
    """Consume one delayed P stage with packed mixed or strict native PV."""

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
                pcor_regs[2] * consumer_carrier_smem[0] / carrier_stage_smem[stage]
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
        if cutlass.const_expr(NATIVE_PV == 1):
            for latent_slice in cutlass.range_constexpr(NATIVE_PV_LATENT_SLICES):
                pv_acc = cute.make_tensor(
                    tmem_ptr + OUTPUT_OFFSET + latent_slice * native_pv_output_cols,
                    native_pv_acc_fake.layout,
                )
                if tidx < 128:
                    t_acc = pv_acc[(None, None), 0, 0]
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
                        (ROWS_PER_CTA, NATIVE_PV_SLICE_COLS)
                    )
                    t_tmem = thr_load.partition_S(t_acc)
                    r_layout = thr_load.partition_D(output_coords)
                    t_tmem_store = thr_store.partition_D(t_acc)
                    r_acc = cute.make_fragment_like(r_layout, cutlass.Float32)
                    cute.copy(tmem_load, t_tmem, r_acc)
                    cute.arch.fence_view_async_tmem_load()
                    for element in cutlass.range_constexpr(cute.size(r_acc)):
                        row = r_layout[element][0]
                        r_acc[element] = r_acc[element] * correction_scale_smem[row]
                    cute.copy(tmem_store, r_acc, t_tmem_store)
                    cute.arch.fence_view_async_tmem_store()
        else:
            for latent_slice in cutlass.range_constexpr(MIXED_PV_LATENT_SLICES):
                pv_acc = cute.make_tensor(
                    tmem_ptr + OUTPUT_OFFSET + latent_slice * mixed_pv_output_cols,
                    mixed_pv_acc_fake.layout,
                )
                if tidx < 128:
                    t_acc = pv_acc[(None, None), 0, 0]
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
                        (ROWS_PER_CTA, MIXED_PV_SLICE_COLS)
                    )
                    t_tmem = thr_load.partition_S(t_acc)
                    r_layout = thr_load.partition_D(output_coords)
                    t_tmem_store = thr_store.partition_D(t_acc)
                    r_acc = cute.make_fragment_like(r_layout, cutlass.Float32)
                    cute.copy(tmem_load, t_tmem, r_acc)
                    cute.arch.fence_view_async_tmem_load()
                    for element in cutlass.range_constexpr(cute.size(r_acc)):
                        row = r_layout[element][0]
                        r_acc[element] = r_acc[element] * correction_scale_smem[row]
                    cute.copy(tmem_store, r_acc, t_tmem_store)
                    cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_threads()

    key_tile_index = cluster_index * TILES + consume_tile
    if cutlass.const_expr(NATIVE_PV == 1):
        for latent_slice in cutlass.range_constexpr(NATIVE_PV_LATENT_SLICES):
            v_sequence = consume_tile * NATIVE_PV_LATENT_SLICES + latent_slice
            v_stage = v_sequence % V_OPERAND_STAGES
            v_phase = (v_sequence // V_OPERAND_STAGES) % 2
            v_bar_ptr = v_tma_mbar.get_barrier(v_stage)
            # Tile zero slices 0 and 1 were prefetched at setup.  Every later
            # slice reuses a stage only after the prior cooperative MMA has
            # completed and all local consumers have released it.
            if cutlass.const_expr(
                consume_tile > 0 or latent_slice >= INITIAL_NATIVE_V_PREFETCHES
            ):
                if tidx == 0:
                    prims.mbarrier_arrive_expect_tx(
                        v_bar_ptr, native_cache_v_desc.global_tx_bytes()
                    )
                    v_stage_ptr = raw_v_smem.data_ptr() + v_stage * (
                        TOKENS * NATIVE_PV_SLICE_COLS // CLUSTER_SHAPE_MNK[0]
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        v_stage_ptr,
                        native_cache_v_desc.get_ptr(),
                        (
                            cutlass.Int32(
                                latent_slice * NATIVE_PV_SLICE_COLS
                                + cta_rank
                                * (NATIVE_PV_SLICE_COLS // CLUSTER_SHAPE_MNK[0])
                            ),
                            cutlass.Int32(0),
                            cutlass.Int32(key_tile_index),
                        ),
                        v_bar_ptr,
                    )
            while not prims.mbarrier_try_wait_parity(
                v_bar_ptr, cutlass.Int32(v_phase), time_limit=10_000_000
            ):
                pass
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            pv_acc = cute.make_tensor(
                tmem_ptr + OUTPUT_OFFSET + latent_slice * native_pv_output_cols,
                native_pv_acc_fake.layout,
            )
            if warp_idx == 8 and cta_rank == 0:
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
            vp_full = vp_consumer.wait_and_advance()
            cute.arch.sync_threads()

            cute.arch.sync_threads()
            # The stage remains owned until every thread has completed its
            # accumulator read/export; releasing before this point permits a
            # later group-two producer to reuse TMEM while rank 1 still reads.
            vp_full.release()
    else:
        for latent_slice in cutlass.range_constexpr(MIXED_PV_LATENT_SLICES):
            v_stage = latent_slice
            v_phase = consume_tile % 2
            v_bar_ptr = v_tma_mbar.get_barrier(v_stage)
            if cutlass.const_expr(
                consume_tile > 0 or latent_slice >= INITIAL_MIXED_V_PREFETCHES
            ):
                if warp_idx == 9:
                    if cta_rank == 0:
                        v_tma_mbar.arrive_and_expect_tx(
                            v_stage,
                            mixed_v_copy_bytes,
                        )
                    for page in cutlass.range_constexpr(TOKENS // PAGE_SIZE):
                        physical_page = physical_page_for(
                            block_table,
                            cluster_index,
                            consume_tile * (TOKENS // PAGE_SIZE) + page,
                            MULTI_REQUEST,
                        )
                        cute.copy(
                            tma_atom_mixed_pv_v,
                            t_mixed_vg_v[None, latent_slice, 0, physical_page],
                            t_mixed_vs_v[
                                None,
                                0,
                                page,
                                ((latent_slice, 0), 0),
                            ],
                            tma_bar_ptr=v_bar_ptr,
                        )
            # Generated UTMALDG.2CTA completes the elected rank-0 barrier for
            # the transaction spanning both SMs.  Waiting on rank 1's local
            # barrier deadlocks because it is deliberately not the owner.
            if warp_idx == 8 and cta_rank == 0:
                while not prims.mbarrier_try_wait_parity(
                    v_bar_ptr,
                    cutlass.Int32(v_phase),
                    time_limit=10_000_000,
                ):
                    pass
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            pv_acc = cute.make_tensor(
                tmem_ptr + OUTPUT_OFFSET + latent_slice * mixed_pv_output_cols,
                mixed_pv_acc_fake.layout,
            )
            if warp_idx == 8 and cta_rank == 0:
                vp_producer.acquire_and_advance()
                mixed_pv_mma.set(
                    tcgen05.Field.ACCUMULATE,
                    cutlass.const_expr(consume_tile > 0),
                )
                for k_block in cutlass.range_constexpr(mixed_p_a.shape[2]):
                    mixed_pv_mma.set(
                        tcgen05.Field.SFA,
                        t_mixed_pv_sfa[None, None, k_block].iterator,
                    )
                    mixed_pv_mma.set(
                        tcgen05.Field.SFB,
                        t_mixed_pv_sfb[None, None, k_block].iterator,
                    )
                    cute.gemm(
                        mixed_pv_mma,
                        pv_acc,
                        mixed_p_a[None, None, k_block, stage],
                        mixed_v_b[None, None, k_block, v_stage],
                        pv_acc,
                    )
                    mixed_pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                vp_producer.commit()
            vp_full = vp_consumer.wait_and_advance()
            cute.arch.sync_threads()

            cute.arch.sync_threads()
            # Match the native arm's accumulator lifetime: the empty signal
            # follows, rather than precedes, the final consumer TMEM load.
            vp_full.release()

    # Both CTAs own a disjoint half of the group-two accumulator.  The local
    # barriers above retire each CTA's TMEM loads, but the elected rank can
    # otherwise return and issue the next group-two QK MMA while its peer is
    # still exporting the preceding PV result.  Release/acquire the cluster at
    # the consumer boundary so TMEM pipeline reuse is symmetric across ranks.
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()


@cute.struct
class SharedStorage:
    init_mbar: cutlass.Int64
    tma_mbar: cute.struct.MemRange[cutlass.Int64, TILES * (QK_BARRIER_SLOTS + 1)]
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    vp_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    softmax_max_exchange: cute.struct.MemRange[cutlass.Float32, SCORE_ROWS]
    softmax_sum_exchange: cute.struct.MemRange[cutlass.Float32, SCORE_ROWS]
    carrier_scale: cute.struct.MemRange[cutlass.Float32, 1]
    carrier_inverse: cute.struct.MemRange[cutlass.Float32, 1]
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
    mixed_pv = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        cutlass.Float4E2M1FN,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        SF_DTYPE,
        SF_VEC_SIZE,
        tcgen05.CtaGroup.TWO,
        MIXED_PV_TILER_MNK[:2],
    )
    native_pv = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        NATIVE_PV_TILER_MNK[:2],
    )
    return mixed, native_qk, rope, mixed_pv, native_pv


@cute.kernel
def ownership_kernel(
    bf16_output: cute.Tensor,
    lse_output: cute.Tensor,
    token_scale: cute.Tensor,
    block_table: cute.Tensor,
    cache_len: cutlass.Int32,
    softmax_scale_log2: cutlass.Float32,
    native_cache_v_desc: cutlass.GridConstant[cuda.TensorMap],
    mixed_mma: cute.TiledMma,
    native_qk_mma: cute.TiledMma,
    rope_mma: cute.TiledMma,
    mixed_pv_mma: cute.TiledMma,
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
    tma_atom_mixed_pv_v: cute.CopyAtom,
    tma_tensor_mixed_pv_v: cute.Tensor,
    mixed_a_layout: cute.ComposedLayout,
    mixed_b_layout: cute.ComposedLayout,
    paged_mixed_b_layout: cute.ComposedLayout,
    native_q_layout: cute.ComposedLayout,
    native_k_layout: cute.ComposedLayout,
    rope_a_layout: cute.ComposedLayout,
    rope_b_layout: cute.ComposedLayout,
    paged_rope_b_layout: cute.ComposedLayout,
    sfa_layout: cute.Layout,
    sfb_layout: cute.Layout,
    sfa_cols: cutlass.Constexpr,
    mixed_pv_sfa_layout: cute.Layout,
    mixed_pv_sfb_layout: cute.Layout,
    mixed_pv_sfa_cols: cutlass.Constexpr,
    mixed_pv_sfb_cols: cutlass.Constexpr,
    mixed_pv_output_cols: cutlass.Constexpr,
    native_pv_output_cols: cutlass.Constexpr,
    mixed_p_layout: cute.ComposedLayout,
    mixed_v_layout: cute.ComposedLayout,
    paged_mixed_v_layout: cute.ComposedLayout,
    native_p_layout: cute.ComposedLayout,
    native_cache_v_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
    NATIVE_QK: cutlass.Constexpr[int],
    NATIVE_PV: cutlass.Constexpr[int],
    HOIST_CARRIER_INVERSE: cutlass.Constexpr[int],
    STAGE_ABSOLUTE_SCALES: cutlass.Constexpr[int],
    STAGE_NORMALIZED_SCALES: cutlass.Constexpr[int],
    PARALLEL_CARRIER_REDUCTION: cutlass.Constexpr[int],
    OVERLAP_SETUP: cutlass.Constexpr[int],
    PV_SFA_EXP: cutlass.Constexpr[int],
    PV_SFB_EXP: cutlass.Constexpr[int],
    MULTI_REQUEST: cutlass.Constexpr[int],
    ACTIVE_ROWS: cutlass.Constexpr[int],
    QUERY_LEN: cutlass.Constexpr[int],
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
    carrier_inverse_smem = cute.make_tensor(
        storage.carrier_inverse.data_ptr(), cute.make_layout(1)
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
    if cutlass.const_expr(STAGE_ABSOLUTE_SCALES == 1):
        if cutlass.const_expr(PARALLEL_CARRIER_REDUCTION == 1):
            scale_storage_bytes = TOKENS * 2 + 4 * 4
            if cutlass.const_expr(STAGE_NORMALIZED_SCALES == 1):
                scale_storage_bytes += TOKENS * 4
            scale_storage = cutlass.Array(
                cutlass.Int8,
                scale_storage_bytes,
                space=cutlass.AddressSpace.smem,
                alignment=16,
            )
            absolute_scale_ptr = cute.make_ptr(
                cutlass.BFloat16,
                scale_storage.data_ptr().ir_value(),
                cute.AddressSpace.smem,
                assumed_align=16,
            )
            absolute_scale_smem = cute.make_tensor(
                absolute_scale_ptr, cute.make_layout(TOKENS)
            )
            partial_bf16_offset = TOKENS
            if cutlass.const_expr(STAGE_NORMALIZED_SCALES == 1):
                normalized_scale_smem = cute.make_tensor(
                    cute.recast_ptr(
                        absolute_scale_ptr + TOKENS,
                        dtype=cutlass.Float32,
                    ),
                    cute.make_layout(TOKENS),
                )
                partial_bf16_offset += TOKENS * 2
            carrier_partial_smem = cute.make_tensor(
                cute.recast_ptr(
                    absolute_scale_ptr + partial_bf16_offset,
                    dtype=cutlass.Float32,
                ),
                cute.make_layout(4),
            )
        else:
            # Preserve exact A3 allocation structure for S-control.
            absolute_scale_storage = cutlass.Array(
                cutlass.Int8,
                TOKENS * 2,
                space=cutlass.AddressSpace.smem,
                alignment=16,
            )
            absolute_scale_smem = cute.make_tensor(
                cute.make_ptr(
                    cutlass.BFloat16,
                    absolute_scale_storage.data_ptr().ir_value(),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                cute.make_layout(TOKENS),
            )
            if cutlass.const_expr(STAGE_NORMALIZED_SCALES == 1):
                normalized_scale_storage = cutlass.Array(
                    cutlass.Int8,
                    TOKENS * 4,
                    space=cutlass.AddressSpace.smem,
                    alignment=16,
                )
                normalized_scale_smem = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        normalized_scale_storage.data_ptr().ir_value(),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout(TOKENS),
                )
    # Keep the generated two-CTA V completion barriers adjacent to the other
    # control state.  Their location is not the SM100 ownership rule: only
    # rank 0 may arm them, while both ranks must issue the cooperative copy.
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

    # Allocate the packed-V stages before the larger Q/K envelope.  This keeps
    # the storage layout easy to audit; experiments ruled out the absolute
    # shared-memory offset as the cause of the earlier UTMALDG.2CTA failure.
    v_smem = cutlass.Array(
        cutlass.Int8,
        V_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    v_smem_ptr = cute.make_ptr(
        MIXED_B_SMEM_DTYPE,
        v_smem.data_ptr().ir_value(),
        cute.AddressSpace.smem,
        assumed_align=128,
    )
    mixed_v_smem = cute.make_tensor(
        cute.recast_ptr(
            v_smem_ptr,
            swizzle_=mixed_v_layout.inner,
            dtype=MIXED_B_SMEM_DTYPE,
        ),
        mixed_v_layout.outer,
    )
    native_cache_v_smem = cute.make_tensor(
        cute.recast_ptr(
            v_smem_ptr,
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
    mixed_p_smem = cute.make_tensor(
        cute.recast_ptr(
            p_smem_ptr,
            swizzle_=mixed_p_layout.inner,
            dtype=cutlass.Float8E4M3FN,
        ),
        mixed_p_layout.outer,
    )
    native_p_smem = cute.make_tensor(
        cute.recast_ptr(
            p_smem_ptr,
            swizzle_=native_p_layout.inner,
            dtype=cutlass.Float8E4M3FN,
        ),
        native_p_layout.outer,
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
    k_rope_storage = cutlass.Array(
        cutlass.Int8,
        K_ROPE_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    k_rope_smem_ptr = cute.make_ptr(
        cutlass.Float8E4M3FN,
        k_rope_storage.data_ptr().ir_value(),
        cute.AddressSpace.smem,
        assumed_align=128,
    )
    k_rope_smem = cute.make_tensor(
        cute.recast_ptr(
            k_rope_smem_ptr,
            swizzle_=rope_b_layout.inner,
            dtype=cutlass.Float8E4M3FN,
        ),
        rope_b_layout.outer,
    )
    k_rope_smem_paged = cute.make_tensor(
        cute.recast_ptr(
            k_rope_smem_ptr,
            swizzle_=paged_rope_b_layout.inner,
            dtype=cutlass.Float8E4M3FN,
        ),
        paged_rope_b_layout.outer,
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
    mixed_thr_mma = mixed_mma.get_slice(mma_tile_coord_v)
    t_cg_a = mixed_thr_mma.partition_A(g_a_mkl)
    a_cta_layout = cute.make_layout(cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape)
    b_cta_layout = cute.make_layout(cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape)
    t_as_a, t_ag_a = cpasync.tma_partition(
        tma_atom_a,
        cta_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(q_smem, 0, 3),
        cute.group_modes(t_cg_a, 0, 3),
    )
    k_smem_paged = cute.make_tensor(
        cute.recast_ptr(
            k_smem_ptr,
            swizzle_=paged_mixed_b_layout.inner,
            dtype=MIXED_B_SMEM_DTYPE,
        ),
        paged_mixed_b_layout.outer,
    )
    g_page_b = cute.tiled_divide(tma_tensor_b, (PAGE_SIZE, MIXED_TILER_MNK[2]))
    t_page_b = g_page_b[None, 0, None, None]
    t_bs_b, t_bg_b = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        k_smem_paged,
        t_page_b,
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
    t_as_rope_a, t_ag_rope_a = cpasync.tma_partition(
        tma_atom_rope_a,
        cta_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(q_rope_smem, 0, 3),
        cute.group_modes(t_cg_rope_a, 0, 3),
    )
    g_page_rope_b = cute.tiled_divide(tma_tensor_rope_b, (PAGE_SIZE, ROPE_TILER_MNK[2]))
    t_page_rope_b = g_page_rope_b[None, 0, None, None]
    t_bs_rope_b, t_bg_rope_b = cpasync.tma_partition(
        tma_atom_rope_b,
        0,
        cute.make_layout(1),
        k_rope_smem_paged,
        t_page_rope_b,
    )
    mixed_v_smem_paged = cute.make_tensor(
        cute.recast_ptr(
            v_smem_ptr,
            swizzle_=paged_mixed_v_layout.inner,
            dtype=MIXED_B_SMEM_DTYPE,
        ),
        paged_mixed_v_layout.outer,
    )
    g_page_v = cute.flat_divide(
        tma_tensor_mixed_pv_v,
        (MIXED_PV_TILER_MNK[1], PAGE_SIZE),
    )
    cta_n = MIXED_PV_TILER_MNK[1] // mixed_pv_mma.thr_id.shape
    g_page_v = cute.logical_divide(g_page_v, (cta_n,))[
        (None, cta_rank), None, None, None, None
    ]
    t_page_v = cute.tiled_divide(g_page_v, (cta_n, PAGE_SIZE))[
        None, 0, 0, None, None, None
    ]
    t_mixed_vs_v, t_mixed_vg_v = cpasync.tma_partition(
        tma_atom_mixed_pv_v,
        0,
        cute.make_layout(1),
        mixed_v_smem_paged,
        t_page_v,
    )
    t_ag_a = t_ag_a[(None, 0, None, None)]
    t_ng_q = t_ng_q[(None, 0, None, None)]
    t_ng_k = t_ng_k[(None, 0, None, None)]
    t_ag_rope_a = t_ag_rope_a[(None, 0, None, None)]
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
    native_v_copy_bytes = native_cache_v_desc.global_tx_bytes()
    mixed_v_copy_bytes = cute.size_in_bytes(
        cutlass.Float4E2M1FN,
        cute.slice_(mixed_v_smem, (None, None, None, 0)),
    ) * cute.size(mixed_pv_mma.thr_id.shape)
    selected_v_copy_bytes = mixed_v_copy_bytes
    if cutlass.const_expr(NATIVE_PV == 1):
        selected_v_copy_bytes = native_v_copy_bytes
    v_tma_barriers = pipeline.MbarrierArray(
        v_tma_mbar_ptr,
        V_OPERAND_STAGES,
        (
            pipeline.PipelineOp.TmaLoad,
            pipeline.CooperativeGroup(pipeline.Agent.Thread),
        ),
        tx_count=selected_v_copy_bytes,
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

    # The candidate consumes packed V directly with two cooperative N256
    # block-scaled MMAs.  No decoded shadow, conversion, LdMatrix, or TMEM-V
    # operand exists in this complete reader.
    mixed_p_a = mixed_pv_mma.make_fragment_A(mixed_p_smem)
    mixed_v_b = mixed_pv_mma.make_fragment_B(mixed_v_smem)
    mixed_pv_acc_fake = mixed_pv_mma.make_fragment_C(
        mixed_pv_mma.partition_shape_C(MIXED_PV_TILER_MNK[:2])
    )
    native_p_a = native_pv_mma.make_fragment_A(native_p_smem)
    native_cache_v_b = native_pv_mma.make_fragment_B(native_cache_v_smem)
    native_pv_acc_fake = native_pv_mma.make_fragment_C(
        native_pv_mma.partition_shape_C(NATIVE_PV_TILER_MNK[:2])
    )
    mixed_pv_sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
        mixed_pv_mma,
        MIXED_PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(mixed_pv_sfa_layout, (None, None, None, 0)),
    )
    mixed_pv_sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
        mixed_pv_mma,
        MIXED_PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(mixed_pv_sfb_layout, (None, None, None, 0)),
    )
    t_mixed_pv_sfa = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + MIXED_PV_SCALE_OFFSET, dtype=SF_DTYPE),
        mixed_pv_sfa_tmem_layout,
    )
    t_mixed_pv_sfb = cute.make_tensor(
        cute.recast_ptr(
            tmem_ptr + MIXED_PV_SCALE_OFFSET + mixed_pv_sfa_cols,
            dtype=SF_DTYPE,
        ),
        mixed_pv_sfb_tmem_layout,
    )
    p_cor = cute.make_tensor(
        tmem_ptr + P_COR_OFFSET,
        cute.make_layout(
            (SCORE_ROWS, CORRECTION_VALUES, CORRECTION_STAGES),
            stride=(1 << 16, 1, CORRECTION_VALUES),
        ),
    )

    pcor_coords = cute.make_identity_tensor(p_cor.shape)
    pcor_store_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(CORRECTION_VALUES)),
        cutlass.Float32,
    )
    pcor_store = tcgen05.make_tmem_copy(pcor_store_atom, p_cor)

    # The overlap arm starts both first-tile N256 packed-V transactions as soon
    # as the mbarriers are initialized.  The strict native control likewise
    # performs its earliest legal two-stage prefetch.  The serial arm defers
    # the same candidate transactions until after scale publication.
    if warp_idx == 9 and cutlass.const_expr(NATIVE_PV == 0 and OVERLAP_SETUP == 1):
        for latent_slice in cutlass.range_constexpr(INITIAL_MIXED_V_PREFETCHES):
            v_bar_ptr = v_tma_barriers.get_barrier(latent_slice)
            if cta_rank == 0:
                v_tma_barriers.arrive_and_expect_tx(latent_slice, mixed_v_copy_bytes)
            for page in cutlass.range_constexpr(TOKENS // PAGE_SIZE):
                physical_page = physical_page_for(
                    block_table, cluster_index, page, MULTI_REQUEST
                )
                cute.copy(
                    tma_atom_mixed_pv_v,
                    t_mixed_vg_v[None, latent_slice, 0, physical_page],
                    t_mixed_vs_v[
                        None,
                        0,
                        page,
                        ((latent_slice, 0), 0),
                    ],
                    tma_bar_ptr=v_bar_ptr,
                )
    if tidx == 0 and cutlass.const_expr(NATIVE_PV == 1):
        for latent_slice in cutlass.range_constexpr(INITIAL_NATIVE_V_PREFETCHES):
            v_bar_ptr = v_tma_barriers.get_barrier(latent_slice)
            prims.mbarrier_arrive_expect_tx(
                v_bar_ptr, native_cache_v_desc.global_tx_bytes()
            )
            v_stage_ptr = v_smem.data_ptr() + latent_slice * (
                TOKENS * NATIVE_PV_SLICE_COLS // CLUSTER_SHAPE_MNK[0]
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                v_stage_ptr,
                native_cache_v_desc.get_ptr(),
                (
                    cutlass.Int32(
                        latent_slice * NATIVE_PV_SLICE_COLS
                        + cta_rank * (NATIVE_PV_SLICE_COLS // CLUSTER_SHAPE_MNK[0])
                    ),
                    cutlass.Int32(0),
                    cutlass.Int32(cluster_index * TILES),
                ),
                v_bar_ptr,
            )

    # Initialize the complete 0..63 mixed-PV defensive reserve, then the live
    # QK scale columns at 64..83.  The non-unity P0 controls alter exactly one
    # scale class and keep all unused PV-reserve bytes poisoned.
    mixed_pv_scale_cols = mixed_pv_sfa_cols + mixed_pv_sfb_cols
    if tidx < SCORE_ROWS and cutlass.const_expr(NATIVE_PV == 0):
        pv_scale_init_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)),
            cutlass.Float32,
        )
        pv_sfa_byte = 0x7F + PV_SFA_EXP
        pv_sfb_byte = 0x7F + PV_SFB_EXP
        pv_sfa_word = cutlass.Uint32(pv_sfa_byte * 0x01010101).bitcast(cutlass.Float32)
        pv_sfb_word = cutlass.Uint32(pv_sfb_byte * 0x01010101).bitcast(cutlass.Float32)
        poison_word = cutlass.Uint32(0x81818181).bitcast(cutlass.Float32)
        for scale_block in cutlass.range_constexpr(MIXED_PV_SCALE_RESERVE_COLS // 16):
            pv_scale_tile = cute.make_tensor(
                tmem_ptr + MIXED_PV_SCALE_OFFSET + scale_block * 16,
                cute.make_layout((SCORE_ROWS, 16), stride=(1 << 16, 1)),
            )
            pv_scale_init = tcgen05.make_tmem_copy(pv_scale_init_atom, pv_scale_tile)
            pv_scale_thr = pv_scale_init.get_slice(tidx)
            pv_scale_coords = cute.make_identity_tensor(pv_scale_tile.shape)
            pv_scale_regs_layout = pv_scale_thr.partition_S(pv_scale_coords)
            pv_scale_dst = pv_scale_thr.partition_D(pv_scale_tile)
            pv_scale_regs = cute.make_fragment_like(
                pv_scale_regs_layout, cutlass.Float32
            )
            for element in cutlass.range_constexpr(cute.size(pv_scale_regs)):
                scale_col = scale_block * 16 + element
                if cutlass.const_expr(scale_col < mixed_pv_sfa_cols):
                    pv_scale_regs[element] = pv_sfa_word
                elif cutlass.const_expr(scale_col < mixed_pv_scale_cols):
                    pv_scale_regs[element] = pv_sfb_word
                else:
                    pv_scale_regs[element] = poison_word
            cute.copy(pv_scale_init, pv_scale_regs, pv_scale_dst)

    # QK SFA/SFB are exact UE8M0 unity.  Only the first 128 threads
    # participate because the TMEM copy atom owns one logical score row.
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
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    if warp_idx == 9 and cutlass.const_expr(NATIVE_PV == 0 and OVERLAP_SETUP == 0):
        for latent_slice in cutlass.range_constexpr(INITIAL_MIXED_V_PREFETCHES):
            v_bar_ptr = v_tma_barriers.get_barrier(latent_slice)
            if cta_rank == 0:
                v_tma_barriers.arrive_and_expect_tx(latent_slice, mixed_v_copy_bytes)
            for page in cutlass.range_constexpr(TOKENS // PAGE_SIZE):
                physical_page = physical_page_for(
                    block_table, cluster_index, page, MULTI_REQUEST
                )
                cute.copy(
                    tma_atom_mixed_pv_v,
                    t_mixed_vg_v[None, latent_slice, 0, physical_page],
                    t_mixed_vs_v[
                        None,
                        0,
                        page,
                        ((latent_slice, 0), 0),
                    ],
                    tma_bar_ptr=v_bar_ptr,
                )

    if tidx == 0:
        consumer_carrier_smem[0] = cutlass.Float32(0.0)
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()

    online_row_max = cutlass.Float32(-1.0e6)
    online_row_sum = cutlass.Float32(0.0)

    for tile in cutlass.range_constexpr(TILES):
        barrier_base = tile * (QK_BARRIER_SLOTS + 1)
        key_tile_index = cluster_index * TILES + tile
        parallel_carrier_max = cutlass.Float32(0.0)
        if cutlass.const_expr(STAGE_ABSOLUTE_SCALES == 1):
            if tidx < TOKENS:
                staged_scale = cutlass.BFloat16(0.0)
                if tile * TOKENS + tidx < cache_len:
                    logical_token = tile * TOKENS + tidx
                    physical_page = physical_page_for(
                        block_table,
                        cluster_index,
                        logical_token // PAGE_SIZE,
                        MULTI_REQUEST,
                    )
                    staged_scale = token_scale[physical_page, logical_token % PAGE_SIZE]
                absolute_scale_smem[tidx] = staged_scale
                if cutlass.const_expr(PARALLEL_CARRIER_REDUCTION == 1):
                    warp_scale_max = ptx_redux_sync_max_f32(
                        staged_scale.to(cutlass.Float32)
                    )
                    lane = tidx % 32
                    if lane == 0:
                        carrier_partial_smem[warp_idx] = warp_scale_max
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()
            if cutlass.const_expr(PARALLEL_CARRIER_REDUCTION == 1):
                if tidx < 32:
                    partial_max = cutlass.Float32(0.0)
                    if tidx < 4:
                        partial_max = carrier_partial_smem[tidx]
                    parallel_carrier_max = ptx_redux_sync_max_f32(partial_max)
        if tidx == 0:
            if cutlass.const_expr(NATIVE_PV == 1):
                carrier_scale = cutlass.Float32(1.0)
                carrier_exp = cutlass.Int32(0)
            else:
                if cutlass.const_expr(PARALLEL_CARRIER_REDUCTION == 1):
                    max_d = parallel_carrier_max
                else:
                    max_d = cutlass.Float32(0.0)
                    for token in cutlass.range_constexpr(TOKENS):
                        if cutlass.const_expr(STAGE_ABSOLUTE_SCALES == 1):
                            absolute_scale = absolute_scale_smem[token].to(
                                cutlass.Float32
                            )
                        else:
                            absolute_scale = cutlass.Float32(0.0)
                            if tile * TOKENS + token < cache_len:
                                logical_token = tile * TOKENS + token
                                physical_page = physical_page_for(
                                    block_table,
                                    cluster_index,
                                    logical_token // PAGE_SIZE,
                                    MULTI_REQUEST,
                                )
                                absolute_scale = token_scale[
                                    physical_page, logical_token % PAGE_SIZE
                                ].to(cutlass.Float32)
                        max_d = cute.arch.fmax(max_d, absolute_scale)
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
            if tile * TOKENS >= cache_len:
                carrier_scale = cutlass.Float32(1.0)
                carrier_exp = cutlass.Int32(0)
                if cutlass.const_expr(tile > 0):
                    carrier_scale = carrier_stage_smem[(tile - 1) % CORRECTION_STAGES]
                    carrier_exp = carrier_exp_stage_smem[(tile - 1) % CORRECTION_STAGES]
            carrier_scale_smem[0] = carrier_scale
            if cutlass.const_expr(HOIST_CARRIER_INVERSE == 1):
                inverse_bits = cutlass.Uint32(
                    (cutlass.Int32(127) - carrier_exp) << cutlass.Int32(23)
                )
                carrier_inverse_smem[0] = inverse_bits.bitcast(cutlass.Float32)
            carrier_stage_smem[tile % CORRECTION_STAGES] = carrier_scale
            carrier_exp_stage_smem[tile % CORRECTION_STAGES] = carrier_exp
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        carrier_scale = cutlass.Float32(0.0)
        if cutlass.const_expr(HOIST_CARRIER_INVERSE == 0):
            carrier_scale = carrier_scale_smem[0]
        if cutlass.const_expr(STAGE_NORMALIZED_SCALES == 1):
            if tidx < TOKENS:
                normalized_scale_smem[tidx] = (
                    absolute_scale_smem[tidx].to(cutlass.Float32)
                    * carrier_inverse_smem[0]
                )
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()
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
                    for page in cutlass.range_constexpr(
                        TOKENS // PAGE_SIZE // CLUSTER_SHAPE_MNK[0]
                    ):
                        logical_page = (tile * CLUSTER_SHAPE_MNK[0] + cta_rank) * (
                            TOKENS // PAGE_SIZE // CLUSTER_SHAPE_MNK[0]
                        ) + page
                        physical_page = physical_page_for(
                            block_table,
                            cluster_index,
                            logical_page,
                            MULTI_REQUEST,
                        )
                        cute.copy(
                            tma_atom_b,
                            t_bg_b[None, latent_tile, physical_page],
                            t_bs_b[None, page, 0, (latent_tile, 0)],
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
            for page in cutlass.range_constexpr(
                TOKENS // PAGE_SIZE // CLUSTER_SHAPE_MNK[0]
            ):
                logical_page = (tile * CLUSTER_SHAPE_MNK[0] + cta_rank) * (
                    TOKENS // PAGE_SIZE // CLUSTER_SHAPE_MNK[0]
                ) + page
                physical_page = physical_page_for(
                    block_table,
                    cluster_index,
                    logical_page,
                    MULTI_REQUEST,
                )
                cute.copy(
                    tma_atom_rope_b,
                    t_bg_rope_b[None, 0, physical_page],
                    t_bs_rope_b[None, page, 0, 0],
                    tma_bar_ptr=rope_bar_ptr,
                )

        if warp_idx == 8 and cta_rank == 0:
            mma_producer.acquire_and_advance()
            if cutlass.const_expr(NATIVE_QK == 1):
                native_qk_mma.set(tcgen05.Field.ACCUMULATE, False)
                for latent_tile in cutlass.range_constexpr(NATIVE_LATENT_K_TILES):
                    tma_barriers.wait(barrier_base + latent_tile, 0)
                    for k_block in cutlass.range(native_k_blocks, unroll_full=True):
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
                if cutlass.const_expr(STAGE_ABSOLUTE_SCALES == 1):
                    absolute_scale = absolute_scale_smem[token].to(cutlass.Float32)
                else:
                    logical_token = tile * TOKENS + token
                    physical_page = physical_page_for(
                        block_table,
                        cluster_index,
                        logical_token // PAGE_SIZE,
                        MULTI_REQUEST,
                    )
                    absolute_scale = token_scale[
                        physical_page, logical_token % PAGE_SIZE
                    ].to(cutlass.Float32)
                store_regs[element] = load_regs[element] * absolute_scale
            cute.copy(score_store, store_regs, store_dst)
        cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_threads()
        # QK owns the score accumulator until both CTAs have completed their
        # in-place token-scale stores.  Releasing at wait completion lets the
        # elected RoPE producer overwrite rank 1's still-live TMEM rows.
        latent_full.release()

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
            local_row = load_reg_layout[0][0]
            query_row = cta_rank * ROWS_PER_CTA + local_row
            query_token = query_row // NUM_HEADS
            causal_bound = cache_len - (QUERY_LEN - 1) + query_token
            if causal_bound < cutlass.Int32(0):
                causal_bound = cutlass.Int32(0)
            for element in cutlass.range_constexpr(cute.size(load_regs)):
                token = load_reg_layout[element][1]
                token_valid = cutlass.Int32(0)
                if query_row < ACTIVE_ROWS:
                    if tile * TOKENS + token < causal_bound:
                        if tile * TOKENS + token < cache_len:
                            token_valid = cutlass.Int32(1)
                if token_valid == cutlass.Int32(0):
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
            row_max_new = cute.arch.fmax(online_row_max, tile_row_max)
            prior_correction = cutlass.Float32(1.0)
            if cutlass.const_expr(tile > 0):
                prior_correction = cute.math.exp2(
                    (online_row_max - row_max_new) * softmax_scale_log2,
                    fastmath=False,
                )

            tile_row_sum = cutlass.Float32(0.0)
            if cutlass.const_expr(HOIST_CARRIER_INVERSE == 1):
                # Load at first use instead of carrying this tile-wide scalar
                # through QK, RoPE, and the score reduction.
                carrier_inverse = carrier_inverse_smem[0]
            for element in cutlass.range_constexpr(cute.size(load_regs)):
                token = load_reg_layout[element][1]
                probability = cutlass.Float32(0.0)
                token_valid = cutlass.Int32(0)
                if query_row < ACTIVE_ROWS:
                    if tile * TOKENS + token < causal_bound:
                        if tile * TOKENS + token < cache_len:
                            token_valid = cutlass.Int32(1)
                if token_valid != cutlass.Int32(0):
                    probability = cute.math.exp2(
                        (load_regs[element] - row_max_new) * softmax_scale_log2,
                        fastmath=False,
                    )
                tile_row_sum += probability
                p_value = probability
                if cutlass.const_expr(NATIVE_PV == 0):
                    if cutlass.const_expr(STAGE_NORMALIZED_SCALES == 1):
                        p_value = probability * normalized_scale_smem[token]
                    elif cutlass.const_expr(HOIST_CARRIER_INVERSE == 1):
                        if cutlass.const_expr(STAGE_ABSOLUTE_SCALES == 1):
                            absolute_scale = absolute_scale_smem[token].to(
                                cutlass.Float32
                            )
                        else:
                            logical_token = tile * TOKENS + token
                            physical_page = physical_page_for(
                                block_table,
                                cluster_index,
                                logical_token // PAGE_SIZE,
                                MULTI_REQUEST,
                            )
                            absolute_scale = token_scale[
                                physical_page, logical_token % PAGE_SIZE
                            ].to(cutlass.Float32)
                        p_value = probability * absolute_scale * carrier_inverse
                    else:
                        logical_token = tile * TOKENS + token
                        physical_page = physical_page_for(
                            block_table,
                            cluster_index,
                            logical_token // PAGE_SIZE,
                            MULTI_REQUEST,
                        )
                        p_value = (
                            probability
                            * token_scale[physical_page, logical_token % PAGE_SIZE].to(
                                cutlass.Float32
                            )
                            / carrier_scale
                        )
                    if cutlass.const_expr(PV_SFA_EXP == 1 or PV_SFB_EXP == 1):
                        p_value = p_value * cutlass.Float32(0.5)
                p_coordinate = (
                    (local_row, token % 32),
                    0,
                    token // 32,
                    stage,
                )
                if cutlass.const_expr(NATIVE_PV == 1):
                    native_p_smem[p_coordinate] = p_value.to(cutlass.Float8E4M3FN)
                else:
                    mixed_p_smem[p_coordinate] = p_value.to(cutlass.Float8E4M3FN)

            softmax_sum_exchange[tidx] = tile_row_sum
            cute.arch.fence_view_async_shared()
            if warp_idx % 2 == 0:
                softmax_barrier_pair_02.wait()
            else:
                softmax_barrier_pair_13.wait()
            tile_row_sum += softmax_sum_exchange[(tidx + N64) % SCORE_ROWS]
            row_sum_new = prior_correction * online_row_sum + tile_row_sum

            if cutlass.const_expr(HOIST_CARRIER_INVERSE == 1):
                # Do not keep both tile-wide powers of two live across the
                # fully unrolled P population.  The inverse owns that region;
                # reload the carrier only when the correction path needs it.
                carrier_scale = carrier_scale_smem[0]
            carrier_exp = carrier_exp_stage_smem[stage]
            advance_g = cutlass.Int32(0)
            no_correction = cutlass.Int32(0)
            if cutlass.const_expr(tile == 0):
                no_correction = cutlass.Int32(1)
            else:
                prior_carrier = carrier_stage_smem[(tile - 1) % CORRECTION_STAGES]
                if carrier_scale != prior_carrier:
                    advance_g = cutlass.Int32(1)
                combined_correction = prior_correction * prior_carrier / carrier_scale
                if combined_correction == cutlass.Float32(1.0):
                    no_correction = cutlass.Int32(1)
            if tile * TOKENS >= cache_len:
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

            online_row_max = row_max_new
            online_row_sum = row_sum_new

        # Make the p-correction record's TMEM store locally explicit rather
        # than relying on a fence in the following score or rescale phase.
        cute.arch.fence_view_async_tmem_store()
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        # The RoPE stage remains full through softmax and P publication; only
        # now has every thread finished consuming the score accumulator.
        rope_full.release()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        if cutlass.const_expr(tile > 0):
            consume_pv_tile(
                tidx,
                warp_idx,
                cta_global,
                cluster_index,
                tile - 1,
                tmem_ptr,
                mixed_pv_output_cols,
                native_pv_output_cols,
                mixed_pv_acc_fake,
                native_pv_acc_fake,
                p_cor,
                carrier_stage_smem,
                correction_scale_smem,
                consumer_carrier_smem,
                v_tma_barriers,
                mixed_v_copy_bytes,
                v_smem,
                native_cache_v_desc,
                tma_atom_mixed_pv_v,
                t_mixed_vs_v,
                t_mixed_vg_v,
                block_table,
                mixed_p_a,
                mixed_v_b,
                mixed_pv_mma,
                t_mixed_pv_sfa,
                t_mixed_pv_sfb,
                native_p_smem,
                native_p_a,
                native_cache_v_b,
                native_pv_mma,
                vp_producer,
                vp_consumer,
                NATIVE_PV,
                MULTI_REQUEST,
            )
        continue

    consume_pv_tile(
        tidx,
        warp_idx,
        cta_global,
        cluster_index,
        TILES - 1,
        tmem_ptr,
        mixed_pv_output_cols,
        native_pv_output_cols,
        mixed_pv_acc_fake,
        native_pv_acc_fake,
        p_cor,
        carrier_stage_smem,
        correction_scale_smem,
        consumer_carrier_smem,
        v_tma_barriers,
        mixed_v_copy_bytes,
        v_smem,
        native_cache_v_desc,
        tma_atom_mixed_pv_v,
        t_mixed_vs_v,
        t_mixed_vg_v,
        block_table,
        mixed_p_a,
        mixed_v_b,
        mixed_pv_mma,
        t_mixed_pv_sfa,
        t_mixed_pv_sfb,
        native_p_smem,
        native_p_a,
        native_cache_v_b,
        native_pv_mma,
        vp_producer,
        vp_consumer,
        NATIVE_PV,
        MULTI_REQUEST,
    )

    if tidx < ROWS_PER_CTA:
        correction_scale = cutlass.Float32(0.0)
        lse = cutlass.Float32.inf * cutlass.Float32(-1.0)
        if online_row_sum > cutlass.Float32(0.0):
            correction_scale = consumer_carrier_smem[0] / online_row_sum
            lse = (
                cute.math.log2(online_row_sum, fastmath=True)
                + softmax_scale_log2 * online_row_max
            )
        correction_scale_smem[tidx] = correction_scale
        query_row = cta_rank * ROWS_PER_CTA + tidx
        if query_row < ACTIVE_ROWS:
            if cutlass.const_expr(MULTI_REQUEST == 1):
                lse_output[cluster_index * ACTIVE_ROWS + query_row] = lse
            else:
                lse_output[query_row] = lse
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()

    if cutlass.const_expr(NATIVE_PV == 0):
        for latent_slice in cutlass.range_constexpr(MIXED_PV_LATENT_SLICES):
            mixed_pv_acc = cute.make_tensor(
                tmem_ptr + OUTPUT_OFFSET + latent_slice * mixed_pv_output_cols,
                mixed_pv_acc_fake.layout,
            )
            if tidx < 128:
                t_acc = mixed_pv_acc[(None, None), 0, 0]
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                    cutlass.Float32,
                )
                tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, t_acc)
                thr_load = tmem_load.get_slice(tidx)
                output_coords = cute.make_identity_tensor(
                    (ROWS_PER_CTA, MIXED_PV_SLICE_COLS)
                )
                t_tmem = thr_load.partition_S(t_acc)
                r_coords = thr_load.partition_D(output_coords)
                r_acc = cute.make_fragment_like(r_coords, cutlass.Float32)
                cute.copy(tmem_load, t_tmem, r_acc)
                cute.arch.fence_view_async_tmem_load()
                for element in cutlass.range_constexpr(cute.size(r_acc)):
                    row = r_coords[element][0]
                    query_row = cta_rank * ROWS_PER_CTA + row
                    if query_row < ACTIVE_ROWS:
                        output_col = (
                            latent_slice * MIXED_PV_SLICE_COLS + r_coords[element][1]
                        )
                        output_value = (r_acc[element] * correction_scale_smem[row]).to(
                            cutlass.BFloat16
                        )
                        if cutlass.const_expr(MULTI_REQUEST == 1):
                            bf16_output[
                                cluster_index * ACTIVE_ROWS + query_row,
                                output_col,
                            ] = output_value
                        else:
                            bf16_output[query_row, output_col] = output_value
            cute.arch.sync_threads()

    for latent_slice in cutlass.range_constexpr(LATENT_SLICES):
        if cutlass.const_expr(NATIVE_PV == 1):
            native_pv_acc = cute.make_tensor(
                tmem_ptr + OUTPUT_OFFSET + latent_slice * native_pv_output_cols,
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
                output_coords = cute.make_identity_tensor((ROWS_PER_CTA, V_SLICE_COLS))
                t_tmem = thr_load.partition_S(t_acc)
                r_coords = thr_load.partition_D(output_coords)
                r_acc = cute.make_fragment_like(r_coords, cutlass.Float32)
                cute.copy(tmem_load, t_tmem, r_acc)
                cute.arch.fence_view_async_tmem_load()
                for element in cutlass.range_constexpr(cute.size(r_acc)):
                    row = r_coords[element][0]
                    query_row = cta_rank * ROWS_PER_CTA + row
                    if query_row < ACTIVE_ROWS:
                        output_col = latent_slice * V_SLICE_COLS + r_coords[element][1]
                        output_value = (r_acc[element] * correction_scale_smem[row]).to(
                            cutlass.BFloat16
                        )
                        if cutlass.const_expr(MULTI_REQUEST == 1):
                            bf16_output[
                                cluster_index * ACTIVE_ROWS + query_row,
                                output_col,
                            ] = output_value
                        else:
                            bf16_output[query_row, output_col] = output_value
        cute.arch.sync_threads()

    cute.arch.sync_threads()
    if warp_idx == 8 and cta_rank == 0:
        mma_producer.tail()
    if warp_idx == 8:
        if cta_rank == 0:
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
    rope_a_ptr: cute.Pointer,
    rope_b_ptr: cute.Pointer,
    bf16_output: cute.Tensor,
    lse_output: cute.Tensor,
    token_scale: cute.Tensor,
    block_table: cute.Tensor,
    cache_len: cutlass.Int32,
    softmax_scale_log2: cutlass.Float32,
    ACTIVE_ROWS: cutlass.Constexpr[int],
    QUERY_LEN: cutlass.Constexpr[int],
    PHYSICAL_PAGES: cutlass.Constexpr[int],
    CLUSTERS: cutlass.Constexpr[int],
    NATIVE_QK: cutlass.Constexpr[int],
    NATIVE_PV: cutlass.Constexpr[int],
    HOIST_CARRIER_INVERSE: cutlass.Constexpr[int],
    STAGE_ABSOLUTE_SCALES: cutlass.Constexpr[int],
    STAGE_NORMALIZED_SCALES: cutlass.Constexpr[int],
    PARALLEL_CARRIER_REDUCTION: cutlass.Constexpr[int],
    OVERLAP_SETUP: cutlass.Constexpr[int],
    PV_SFA_EXP: cutlass.Constexpr[int],
    PV_SFB_EXP: cutlass.Constexpr[int],
    stream,
):
    (
        mixed_mma,
        native_qk_mma,
        rope_mma,
        mixed_pv_mma,
        native_pv_mma,
    ) = make_mmas()
    g_mixed_a = cute.make_tensor(
        mixed_a_ptr,
        cute.make_layout(
            (ACTIVE_ROWS, LATENT_K, CLUSTERS),
            stride=(QUERY_DIM, 1, ACTIVE_ROWS * QUERY_DIM),
        ),
    )
    g_mixed_b = cute.make_tensor(
        mixed_b_ptr,
        cute.make_ordered_layout((TOKENS, LATENT_K, CLUSTERS * TILES), order=(1, 0, 2)),
    )
    g_paged_mixed_b = cute.make_tensor(
        mixed_b_ptr,
        cute.make_ordered_layout(
            (PAGE_SIZE, LATENT_K, PHYSICAL_PAGES), order=(1, 0, 2)
        ),
    )
    g_mixed_b_transpose = cute.make_tensor(
        g_mixed_b.iterator,
        cute.select(g_mixed_b.layout, mode=[1, 0, 2]),
    )
    g_native_latent = cute.make_tensor(
        native_latent_ptr,
        cute.make_ordered_layout((TOKENS, LATENT_K, CLUSTERS * TILES), order=(1, 0, 2)),
    )
    g_native_latent_transpose = cute.make_tensor(
        g_native_latent.iterator,
        cute.select(g_native_latent.layout, mode=[1, 0, 2]),
    )
    g_rope_a = cute.make_tensor(
        rope_a_ptr,
        cute.make_layout(
            (ACTIVE_ROWS, ROPE_K, CLUSTERS),
            stride=(QUERY_DIM, 1, ACTIVE_ROWS * QUERY_DIM),
        ),
    )
    g_rope_b = cute.make_tensor(
        rope_b_ptr,
        cute.make_ordered_layout((TOKENS, ROPE_K, CLUSTERS * TILES), order=(1, 0, 2)),
    )
    g_paged_rope_b = cute.make_tensor(
        rope_b_ptr,
        cute.make_ordered_layout((PAGE_SIZE, ROPE_K, PHYSICAL_PAGES), order=(1, 0, 2)),
    )
    native_cache_v_desc = cuda.create_tensor_map_tiled(
        g_native_latent_transpose.iterator.toint(),
        cutlass.Float8E4M3FN,
        global_dims=[LATENT_K, TOKENS, CLUSTERS * TILES],
        global_strides=[
            LATENT_K // 16,
            (TOKENS * LATENT_K) // 16,
        ],
        box_dims=[
            NATIVE_PV_SLICE_COLS // CLUSTER_SHAPE_MNK[0],
            TOKENS,
            1,
        ],
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
    paged_mixed_b_layout = sm100_utils.make_smem_layout(
        OperandMajorMode.K,
        (
            MIXED_TILER_MNK[0] // mixed_mma.thr_id.shape,
            MIXED_TILER_MNK[2],
        ),
        MIXED_B_SMEM_DTYPE,
        LATENT_K_TILES,
    )
    paged_mixed_b_layout = cute.tiled_divide(
        paged_mixed_b_layout, (PAGE_SIZE, MIXED_TILER_MNK[2])
    )
    paged_mixed_b_layout = cute.logical_divide(
        paged_mixed_b_layout,
        (None, None, None, LATENT_K_TILES),
    )
    paged_tma_atom_b, paged_tma_tensor_b = make_paged_tiled_tma_atom(
        b_op,
        g_paged_mixed_b,
        cute.select(paged_mixed_b_layout, mode=[0]),
        (MIXED_TILER_MNK[1], MIXED_TILER_MNK[2]),
        mixed_mma,
        True,
        MIXED_B_SMEM_DTYPE,
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
    paged_rope_b_layout = sm100_utils.make_smem_layout(
        OperandMajorMode.K,
        (
            ROPE_TILER_MNK[0] // rope_mma.thr_id.shape,
            ROPE_TILER_MNK[2],
        ),
        cutlass.Float8E4M3FN,
        1,
    )
    paged_rope_b_layout = cute.tiled_divide(
        paged_rope_b_layout, (PAGE_SIZE, ROPE_TILER_MNK[2])
    )
    paged_tma_atom_rope_b, paged_tma_tensor_rope_b = make_paged_tiled_tma_atom(
        rope_b_op,
        g_paged_rope_b,
        cute.select(paged_rope_b_layout, mode=[0]),
        (ROPE_TILER_MNK[1], ROPE_TILER_MNK[2]),
        rope_mma,
        True,
    )
    mixed_p_layout = sm100_utils.make_smem_layout_a(
        mixed_pv_mma,
        MIXED_PV_TILER_MNK,
        cutlass.Float8E4M3FN,
        P_STAGES,
    )
    mixed_v_layout = sm100_utils.make_smem_layout_b(
        mixed_pv_mma,
        MIXED_PV_TILER_MNK,
        MIXED_B_SMEM_DTYPE,
        V_OPERAND_STAGES,
    )
    mixed_pv_b_op = sm100_utils.cluster_shape_to_tma_atom_B(
        CLUSTER_SHAPE_MNK[:2], mixed_pv_mma.thr_id
    )
    tma_atom_mixed_pv_v, tma_tensor_mixed_pv_v = cute.nvgpu.make_tiled_tma_atom_B(
        mixed_pv_b_op,
        g_mixed_b_transpose,
        cute.slice_(mixed_v_layout, (None, None, None, 0)),
        MIXED_PV_TILER_MNK,
        mixed_pv_mma,
        cta_layout_vmnk.shape,
        internal_type=MIXED_B_SMEM_DTYPE,
    )
    paged_mixed_v_layout = sm100_utils.make_smem_layout(
        OperandMajorMode.MN,
        (
            MIXED_PV_TILER_MNK[1] // mixed_pv_mma.thr_id.shape,
            MIXED_PV_TILER_MNK[2],
        ),
        MIXED_B_SMEM_DTYPE,
        V_OPERAND_STAGES,
    )
    paged_mixed_v_layout = cute.tiled_divide(
        paged_mixed_v_layout,
        (
            MIXED_PV_TILER_MNK[1] // mixed_pv_mma.thr_id.shape,
            PAGE_SIZE,
        ),
    )
    paged_mixed_v_layout = cute.logical_divide(
        paged_mixed_v_layout,
        (None, None, None, MIXED_PV_LATENT_SLICES),
    )
    paged_mixed_v_layout = cute.logical_divide(
        paged_mixed_v_layout,
        (None, None, None, (MIXED_PV_LATENT_SLICES, None)),
    )
    paged_tma_atom_mixed_pv_v, paged_tma_tensor_mixed_pv_v = make_paged_tiled_tma_atom(
        mixed_pv_b_op,
        cute.make_tensor(
            g_paged_mixed_b.iterator,
            cute.select(g_paged_mixed_b.layout, mode=[1, 0, 2]),
        ),
        cute.select(paged_mixed_v_layout, mode=[0]),
        (MIXED_PV_TILER_MNK[1], MIXED_PV_TILER_MNK[2]),
        mixed_pv_mma,
        False,
        MIXED_B_SMEM_DTYPE,
    )
    tma_atom_b, tma_tensor_b = paged_tma_atom_b, paged_tma_tensor_b
    tma_atom_rope_b, tma_tensor_rope_b = (
        paged_tma_atom_rope_b,
        paged_tma_tensor_rope_b,
    )
    tma_atom_mixed_pv_v, tma_tensor_mixed_pv_v = (
        paged_tma_atom_mixed_pv_v,
        paged_tma_tensor_mixed_pv_v,
    )
    mixed_pv_sfa_layout = blockscaled_utils.make_smem_layout_sfa(
        mixed_pv_mma, MIXED_PV_TILER_MNK, SF_VEC_SIZE, 1
    )
    mixed_pv_sfb_layout = blockscaled_utils.make_smem_layout_sfb(
        mixed_pv_mma, MIXED_PV_TILER_MNK, SF_VEC_SIZE, 1
    )
    mixed_pv_sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
        mixed_pv_mma,
        MIXED_PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(mixed_pv_sfa_layout, (None, None, None, 0)),
    )
    mixed_pv_sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
        mixed_pv_mma,
        MIXED_PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(mixed_pv_sfb_layout, (None, None, None, 0)),
    )
    mixed_pv_sfa_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(cute.make_ptr(SF_DTYPE, 0), mixed_pv_sfa_tmem_layout)
    )
    mixed_pv_sfb_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(cute.make_ptr(SF_DTYPE, 0), mixed_pv_sfb_tmem_layout)
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
    if cutlass.const_expr(
        cute.size_in_bytes(cutlass.Float8E4M3FN, mixed_p_layout) != P_SMEM_BYTES
    ):
        raise ValueError(
            "mixed PV P footprint changed: "
            f"{cute.size_in_bytes(cutlass.Float8E4M3FN, mixed_p_layout)} "
            f"!= {P_SMEM_BYTES}"
        )
    if cutlass.const_expr(
        cute.size_in_bytes(MIXED_B_SMEM_DTYPE, mixed_v_layout) != V_SMEM_BYTES
    ):
        raise ValueError(
            "mixed PV V storage changed: "
            f"{cute.size_in_bytes(MIXED_B_SMEM_DTYPE, mixed_v_layout)} "
            f"!= {V_SMEM_BYTES}"
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
        cute.size_in_bytes(cutlass.Float8E4M3FN, native_cache_v_layout) > V_SMEM_BYTES
    ):
        raise ValueError(
            "native V footprint exceeds candidate capacity: "
            f"{cute.size_in_bytes(cutlass.Float8E4M3FN, native_cache_v_layout)} "
            f"!= {V_SMEM_BYTES}"
        )
    mixed_acc_cols = utils.get_num_tmem_alloc_cols(
        mixed_mma.make_fragment_C(mixed_mma.partition_shape_C(MIXED_TILER_MNK[:2]))
    )
    mixed_pv_output_cols = utils.get_num_tmem_alloc_cols(
        mixed_pv_mma.make_fragment_C(
            mixed_pv_mma.partition_shape_C(MIXED_PV_TILER_MNK[:2])
        )
    )
    native_pv_output_cols = utils.get_num_tmem_alloc_cols(
        native_pv_mma.make_fragment_C(
            native_pv_mma.partition_shape_C(NATIVE_PV_TILER_MNK[:2])
        )
    )
    if cutlass.const_expr(
        mixed_pv_sfa_cols + mixed_pv_sfb_cols > MIXED_PV_SCALE_RESERVE_COLS
    ):
        raise ValueError("mixed PV scales exceed defensive reserve")
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
    if cutlass.const_expr(mixed_pv_output_cols != 128):
        raise ValueError(f"mixed PV output footprint changed: {mixed_pv_output_cols}")
    if cutlass.const_expr(native_pv_output_cols != 64):
        raise ValueError(f"native PV output footprint changed: {native_pv_output_cols}")
    if cutlass.const_expr(
        OUTPUT_OFFSET + mixed_pv_output_cols * MIXED_PV_LATENT_SLICES != TMEM_ALLOC_COLS
    ):
        raise ValueError("two mixed PV outputs do not fill TMEM tail")
    if cutlass.const_expr(
        OUTPUT_OFFSET + native_pv_output_cols * NATIVE_PV_LATENT_SLICES
        != TMEM_ALLOC_COLS
    ):
        raise ValueError("four native PV outputs do not fill TMEM tail")
    kernel = ownership_kernel(
        bf16_output,
        lse_output,
        token_scale,
        block_table,
        cache_len,
        softmax_scale_log2,
        native_cache_v_desc,
        mixed_mma,
        native_qk_mma,
        rope_mma,
        mixed_pv_mma,
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
        tma_atom_mixed_pv_v,
        tma_tensor_mixed_pv_v,
        mixed_a_layout,
        mixed_b_layout,
        paged_mixed_b_layout,
        native_q_layout,
        native_k_layout,
        rope_a_layout,
        rope_b_layout,
        paged_rope_b_layout,
        sfa_layout,
        sfb_layout,
        sfa_cols,
        mixed_pv_sfa_layout,
        mixed_pv_sfb_layout,
        mixed_pv_sfa_cols,
        mixed_pv_sfb_cols,
        mixed_pv_output_cols,
        native_pv_output_cols,
        mixed_p_layout,
        mixed_v_layout,
        paged_mixed_v_layout,
        native_p_layout,
        native_cache_v_layout,
        cta_layout_vmnk,
        NATIVE_QK,
        NATIVE_PV,
        HOIST_CARRIER_INVERSE,
        STAGE_ABSOLUTE_SCALES,
        STAGE_NORMALIZED_SCALES,
        PARALLEL_CARRIER_REDUCTION,
        OVERLAP_SETUP,
        PV_SFA_EXP,
        PV_SFB_EXP,
        int(CLUSTERS > 1),
        ACTIVE_ROWS,
        QUERY_LEN,
    )
    kernel.launch(
        grid=(CLUSTER_SHAPE_MNK[0] * CLUSTERS, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
        cluster=CLUSTER_SHAPE_MNK,
        min_blocks_per_mp=1,
        stream=stream,
    )


@cute.jit
def _e2m1_decode_entry(
    query: cute.Tensor,
    packed_cache: cute.Tensor,
    token_scale: cute.Tensor,
    rope_cache: cute.Tensor,
    block_table: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
    cache_len: cutlass.Int32,
    softmax_scale_log2: cutlass.Float32,
    QUERY_LEN: cutlass.Constexpr[int],
    PHYSICAL_PAGES: cutlass.Constexpr[int],
    stream,
):
    active_rows = QUERY_LEN * NUM_HEADS
    query_latent = query.iterator
    query_rope = query.iterator + LATENT_K
    packed_latent = cute.recast_ptr(packed_cache.iterator, dtype=cutlass.Float4E2M1FN)
    flat_block_table = cute.make_tensor(
        block_table.iterator, cute.make_layout(MAX_PAGES)
    )
    flat_output = cute.make_tensor(
        output.iterator,
        cute.make_layout((active_rows, LATENT_K), stride=(LATENT_K, 1)),
    )
    flat_lse = cute.make_tensor(lse.iterator, cute.make_layout(active_rows))

    ownership_probe(
        query_latent,
        packed_latent,
        query_latent,
        query_rope,
        rope_cache.iterator,
        flat_output,
        flat_lse,
        token_scale,
        flat_block_table,
        cache_len,
        softmax_scale_log2,
        active_rows,
        QUERY_LEN,
        PHYSICAL_PAGES,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        stream,
    )


@functools.cache
def _get_compiled_e2m1_decode(query_len: int, physical_pages: int):
    query = make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (1, query_len, NUM_HEADS, QUERY_DIM),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    packed_cache = make_fake_compact_tensor(
        cutlass.Uint8,
        (physical_pages, PAGE_SIZE, LATENT_K // 2),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    token_scale = make_fake_compact_tensor(
        cutlass.BFloat16,
        (physical_pages, PAGE_SIZE),
        stride_order=(1, 0),
        assumed_align=16,
    )
    rope_cache = make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (physical_pages, PAGE_SIZE, ROPE_K),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    page_count = cute.sym_int()
    block_table = make_fake_tensor(
        cutlass.Int32,
        (1, page_count),
        stride=(page_count, 1),
        assumed_align=4,
    )
    output = make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, query_len, NUM_HEADS, LATENT_K),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    lse = make_fake_compact_tensor(
        cutlass.Float32,
        (1, query_len, NUM_HEADS),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _e2m1_decode_entry,
        query,
        packed_cache,
        token_scale,
        rope_cache,
        block_table,
        output,
        lse,
        cutlass.Int32(1),
        cutlass.Float32(1.0),
        query_len,
        physical_pages,
        stream,
        options="--enable-tvm-ffi --opt-level 3",
    )


def tokenspeed_mla_decode_e2m1(
    query: torch.Tensor,
    packed_cache: torch.Tensor,
    token_scale: torch.Tensor,
    rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_len: int,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    lse_out: Optional[torch.Tensor] = None,
    *,
    trusted_page_table: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one page-32 native-E2M1 MLA split on Blackwell.

    This first serving specialization owns one batch-one split of at most 640
    tokens. ``query`` is W5's combined FP8 tensor with shape ``[q,8,576]`` or
    ``[1,q,8,576]``; the cache tensors have shapes ``[P,32,256]`` uint8,
    ``[P,32]`` BF16, and ``[P,32,64]`` FP8.  The caller must provide reusable
    ``out`` and ``lse_out`` buffers to keep graph replay allocation-free.
    """

    if query.dim() == 3:
        query = query.unsqueeze(0)
    if query.dim() != 4:
        raise ValueError(f"query must have rank 3 or 4, got shape {query.shape}")
    batch, query_len, heads, query_dim = query.shape
    if batch != 1:
        raise ValueError(f"batch must be 1, got {batch}")
    if query_len not in (1, 5):
        raise ValueError(f"query length must be 1 or 5, got {query_len}")
    if heads != NUM_HEADS or query_dim != QUERY_DIM:
        raise ValueError(
            f"query must have shape [1,q,{NUM_HEADS},{QUERY_DIM}], got {query.shape}"
        )
    if query.dtype != torch.float8_e4m3fn:
        raise TypeError(f"query must be FP8 E4M3FN, got {query.dtype}")
    if packed_cache.dtype != torch.uint8:
        raise TypeError(f"packed_cache must be uint8, got {packed_cache.dtype}")
    if token_scale.dtype != torch.bfloat16:
        raise TypeError(f"token_scale must be BF16, got {token_scale.dtype}")
    if rope_cache.dtype != torch.float8_e4m3fn:
        raise TypeError(f"rope_cache must be FP8 E4M3FN, got {rope_cache.dtype}")
    if block_table.dtype != torch.int32:
        raise TypeError(f"block_table must be int32, got {block_table.dtype}")

    if packed_cache.dim() != 3 or tuple(packed_cache.shape[1:]) != (
        PAGE_SIZE,
        LATENT_K // 2,
    ):
        raise ValueError(
            f"packed_cache must have shape [P,{PAGE_SIZE},{LATENT_K // 2}], "
            f"got {packed_cache.shape}"
        )
    physical_pages = packed_cache.shape[0]
    if physical_pages <= 0:
        raise ValueError("packed_cache must contain at least one physical page")
    if token_scale.shape != (physical_pages, PAGE_SIZE):
        raise ValueError(
            f"token_scale must have shape [{physical_pages},{PAGE_SIZE}], "
            f"got {token_scale.shape}"
        )
    if rope_cache.shape != (physical_pages, PAGE_SIZE, ROPE_K):
        raise ValueError(
            f"rope_cache must have shape [{physical_pages},{PAGE_SIZE},{ROPE_K}], "
            f"got {rope_cache.shape}"
        )
    if block_table.dim() != 2 or block_table.shape[0] != 1:
        raise ValueError(f"block_table must have shape [1,N], got {block_table.shape}")
    if block_table.shape[1] < MAX_PAGES:
        raise ValueError(
            f"block_table needs at least {MAX_PAGES} entries, got {block_table.shape[1]}"
        )
    if not 0 <= cache_len <= TILES * TOKENS:
        raise ValueError(f"cache_len must be in [0,{TILES * TOKENS}], got {cache_len}")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
        raise ValueError(
            f"softmax_scale must be positive and finite, got {softmax_scale}"
        )

    tensors = (query, packed_cache, token_scale, rope_cache, block_table)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("all inputs must be CUDA tensors")
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the query device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs must be contiguous")
    capability = torch.cuda.get_device_capability(query.device)
    if capability not in ((10, 0), (10, 3)):
        raise ValueError(
            f"native E2M1 MLA decode requires SM100/SM103, got SM{capability[0]}{capability[1]}"
        )

    output_shape = (1, query_len, NUM_HEADS, LATENT_K)
    lse_shape = (1, query_len, NUM_HEADS)
    if out is None:
        out = torch.empty(output_shape, dtype=torch.bfloat16, device=query.device)
    if lse_out is None:
        lse_out = torch.empty(lse_shape, dtype=torch.float32, device=query.device)
    for name, tensor, shape, dtype in (
        ("out", out, output_shape, torch.bfloat16),
        ("lse_out", lse_out, lse_shape, torch.float32),
    ):
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} must have shape {shape} and dtype {dtype}, "
                f"got {tensor.shape} and {tensor.dtype}"
            )
        if tensor.device != query.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {query.device}")
        if any(torch._C._overlaps(tensor, source) for source in tensors):
            raise ValueError(f"{name} must not alias an input tensor")
    if torch._C._overlaps(out, lse_out):
        raise ValueError("out and lse_out must not alias")

    if not trusted_page_table:
        active_pages = block_table[0, :MAX_PAGES]
        torch._assert_async(
            ((active_pages >= 0) & (active_pages < physical_pages)).all(),
            "block_table contains a physical page outside packed_cache",
        )

    compiled = _get_compiled_e2m1_decode(query_len, physical_pages)
    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        compiled(
            query,
            packed_cache,
            token_scale,
            rope_cache,
            block_table,
            out,
            lse_out,
            cutlass.Int32(cache_len),
            cutlass.Float32(softmax_scale * math.log2(math.e)),
        )
    return out, lse_out
