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

"""Compare token-major K4 and word-vectorized page-local V2 PV on SM100.

Both arms multiply the same native E2M1 values by the same FP8 probabilities.
The control reads the historical token-major packed K4 carrier. The candidate
reads four independently addressable physical page-32 V2 pages in
``[page, latent, token_group_of_4]`` order. Each work item loads eight packed
bytes as two aligned words, maps four 16-bit halves to native E2M1 with
word-level Boolean operations, and writes four aligned words into the
already-legal padded K64 shared operand. One CTA fuses the four page-32 bands
and four 128-latent slices into a logical 128-token by 512-latent tile. The
default 80-tile grid models exactly 10,240 tokens.

This is the A17-N3 P0 word-vectorization falsifier. Its per-tile output is
measurement scaffolding, not an admissible serving workspace. A pass credits
only this page-local V2-to-native-E2M1 consumer primitive.
"""

import json
import math
import os

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

THREADS = 128
PAGE_TOKENS = 128
NUM_PAGES = int(os.environ.get("TQ_PV_NUM_PAGES", "80"))
PHYSICAL_PAGE_TOKENS = 32
PHYSICAL_PAGES_PER_TILE = PAGE_TOKENS // PHYSICAL_PAGE_TOKENS
NUM_PHYSICAL_PAGES = NUM_PAGES * PHYSICAL_PAGES_PER_TILE
LATENT_DIM = 512
QUERY_HEADS = 64
LOGICAL_K = 32
TOKEN_BANDS = PAGE_TOKENS // LOGICAL_K
LATENT_SLICE = 128
LATENT_SLICES = LATENT_DIM // LATENT_SLICE
A_STAGES = TOKEN_BANDS * LATENT_SLICES
P_STAGES = TOKEN_BANDS
SF_DTYPE = cutlass.Float8E8M0FNU
SF_VEC_SIZE = 32
CLUSTER_SHAPE_MNK = (1, 1, 1)
E2M1_PV_TILER_MNK = (LATENT_SLICE, QUERY_HEADS, 64)
OUTPUT_VIEW_TILER_MNK = (LATENT_SLICE, QUERY_HEADS, 32)


@cute.struct
class SharedStorage:
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


def make_tiled_mmas():
    e2m1_pv = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        SF_DTYPE,
        SF_VEC_SIZE,
        tcgen05.CtaGroup.ONE,
        E2M1_PV_TILER_MNK[:2],
    )
    output_view = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        OUTPUT_VIEW_TILER_MNK[:2],
    )
    return e2m1_pv, output_view


def e2m1_nibble_for_v2_code(code: int) -> int:
    """Map V2 order {-1.5, -0.5, 0.5, 1.5} to native E2M1 bits."""

    return (3 - 2 * ((code ^ (code >> 1)) & 0x1)) | (
        (((code >> 1) ^ 0x1) & 0x1) << 3
    )


def e2m1_word_for_v2_halfword(x: int) -> int:
    """Reference map: eight packed V2 codes to eight native E2M1 nibbles."""

    z = (x | (x << 8)) & 0x00FF00FF
    z = (z | (z << 4)) & 0x0F0F0F0F
    y = (z | (z << 2)) & 0x33333333
    sign_and_mantissa = (~((y << 1) ^ y)) & 0x22222222
    return (sign_and_mantissa | ((~(y << 2)) & 0x99999999)) & 0xFFFFFFFF


def lop3_truth_table_imm(fn) -> int:
    """PTX lop3 LUT using the standard index ``4*a + 2*b + c``."""

    return sum(
        int(bool(fn(a, b, c))) << (4 * a + 2 * b + c)
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    )


@dsl_user_op
def expand_v2_halfword_to_e2m1_word(
    packed: cutlass.Int32, *, loc=None, ip=None
) -> cutlass.Int32:
    """Map eight 2-bit codes to eight E2M1 nibbles in five SHL + five LOP3."""

    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(packed).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b32 shifted, z, y, sign_mantissa, out;\n\t"
            "shl.b32 shifted, $1, 8;\n\t"
            "lop3.b32 z, $1, shifted, 0x00ff00ff, 0xa8;\n\t"
            "shl.b32 shifted, z, 4;\n\t"
            "lop3.b32 z, z, shifted, 0x0f0f0f0f, 0xa8;\n\t"
            "shl.b32 shifted, z, 2;\n\t"
            "lop3.b32 y, z, shifted, 0x33333333, 0xa8;\n\t"
            "shl.b32 shifted, y, 1;\n\t"
            "lop3.b32 sign_mantissa, shifted, y, 0x22222222, 0x82;\n\t"
            "shl.b32 shifted, y, 2;\n\t"
            "lop3.b32 out, sign_mantissa, shifted, 0x99999999, 0xf2;\n\t"
            "mov.b32 $0, out;\n\t"
            "}\n",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.kernel
def mixed_pv_kernel(
    output: cute.Tensor,
    metadata: cute.Tensor,
    packed_v: cute.Tensor,
    p_input: cute.Tensor,
    page_table: cute.Tensor,
    use_v2: cutlass.Constexpr,
    populate_v: cutlass.Constexpr,
    e2m1_pv_mma: cute.TiledMma,
    output_view_mma: cute.TiledMma,
    e2m1_a_layout: cute.ComposedLayout,
    p_b_layout: cute.ComposedLayout,
    sfa_layout: cute.Layout,
    sfb_layout: cute.Layout,
    acc_cols: cutlass.Constexpr,
    sfa_cols: cutlass.Constexpr,
    sfb_cols: cutlass.Constexpr,
    total_cols: cutlass.Constexpr,
    cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    page, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    s_e2m1_v = smem.allocate_tensor(
        cutlass.Float4E2M1FN,
        e2m1_a_layout.outer,
        byte_alignment=1024,
        swizzle=e2m1_a_layout.inner,
    )
    s_p = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        p_b_layout.outer,
        byte_alignment=1024,
        swizzle=p_b_layout.inner,
    )
    s_sfa = smem.allocate_tensor(
        SF_DTYPE, sfa_layout, byte_alignment=128
    )
    s_sfb = smem.allocate_tensor(
        SF_DTYPE, sfb_layout, byte_alignment=128
    )

    # Populate the same padded K-major native-E2M1 MMA operand from either
    # storage contract. The descriptor advances 16 bytes at logical K16, so
    # the upper token group starts at storage K32 and each latent row spans 32
    # bytes. S<1,4,3> permutes element offsets but preserves nibble pairs.
    e2m1_bytes = cute.size_in_bytes(cutlass.Float4E2M1FN, s_e2m1_v)
    v_raw = cute.make_tensor(
        cute.recast_ptr(
            s_e2m1_v.iterator, swizzle_=None, dtype=cutlass.Uint8
        ),
        cute.make_layout(e2m1_bytes),
    )
    v_raw_i32 = cute.make_tensor(
        cute.recast_ptr(
            s_e2m1_v.iterator, swizzle_=None, dtype=cutlass.Int32
        ),
        cute.make_layout(e2m1_bytes // 4),
    )
    for destination_byte in cutlass.range(tidx, e2m1_bytes, THREADS):
        v_raw[destination_byte] = cutlass.Uint8(0)
    cute.arch.sync_threads()
    if cutlass.const_expr(populate_v):
        if cutlass.const_expr(use_v2):
            packed_v_i32 = cute.make_tensor(
                cute.recast_ptr(packed_v.iterator, dtype=cutlass.Int32),
                cute.make_layout(NUM_PHYSICAL_PAGES * LATENT_DIM * 2),
            )
            active_v2_items = A_STAGES * LATENT_SLICE
            for work_item in cutlass.range(tidx, active_v2_items, THREADS):
                tile = work_item // LATENT_SLICE
                latent = work_item % LATENT_SLICE
                latent_slice = tile // TOKEN_BANDS
                token_band = tile % TOKEN_BANDS
                physical_page = page_table[page, token_band]
                source_latent = latent_slice * LATENT_SLICE + latent
                source_word = (
                    cutlass.Int32(physical_page) * LATENT_DIM + source_latent
                ) * 2
                packed0 = cutlass.Int32(packed_v_i32[source_word])
                packed1 = cutlass.Int32(packed_v_i32[source_word + 1])

                output0 = expand_v2_halfword_to_e2m1_word(
                    packed0 & 0xFFFF
                )
                output1 = expand_v2_halfword_to_e2m1_word(
                    (packed0 >> 16) & 0xFFFF
                )
                output2 = expand_v2_halfword_to_e2m1_word(
                    packed1 & 0xFFFF
                )
                output3 = expand_v2_halfword_to_e2m1_word(
                    (packed1 >> 16) & 0xFFFF
                )

                stage_base = tile * LATENT_SLICE * 32
                logical_row = latent * 32
                swizzle = (logical_row & 0x80) >> 3
                physical_band0 = stage_base + (logical_row ^ swizzle)
                physical_band1 = stage_base + ((logical_row + 16) ^ swizzle)
                physical_word0 = physical_band0 // 4
                physical_word1 = physical_band1 // 4
                v_raw_i32[physical_word0] = output0
                v_raw_i32[physical_word0 + 1] = output1
                v_raw_i32[physical_word1] = output2
                v_raw_i32[physical_word1 + 1] = output3
        else:
            active_k4_bytes = A_STAGES * LATENT_SLICE * (LOGICAL_K // 2)
            for active_byte in cutlass.range(tidx, active_k4_bytes, THREADS):
                tile = active_byte // (LATENT_SLICE * (LOGICAL_K // 2))
                tile_byte = active_byte % (LATENT_SLICE * (LOGICAL_K // 2))
                latent = tile_byte // (LOGICAL_K // 2)
                token_pair = tile_byte % (LOGICAL_K // 2)
                latent_slice = tile // TOKEN_BANDS
                token_band = tile % TOKEN_BANDS
                token0 = token_band * LOGICAL_K + token_pair * 2
                token1 = token0 + 1
                source_latent = latent_slice * LATENT_SLICE + latent
                source_byte = source_latent // 2
                shift = (source_latent % 2) * 4
                code0 = (
                    cutlass.Int32(packed_v[page, token0, source_byte]) >> shift
                ) & 0xF
                code1 = (
                    cutlass.Int32(packed_v[page, token1, source_byte]) >> shift
                ) & 0xF
                storage_k = token_pair * 2
                if token_pair >= 8:
                    storage_k += 16
                logical_byte = latent * 32 + storage_k // 2
                physical_byte = (
                    tile * LATENT_SLICE * 32
                    + (logical_byte ^ ((logical_byte & 0x80) >> 3))
                )
                v_raw[physical_byte] = cutlass.Uint8(code0 | (code1 << 4))
    p_raw = cute.make_tensor(
        cute.recast_ptr(s_p.iterator, swizzle_=None, dtype=cutlass.Float8E4M3FN),
        cute.make_layout(cute.cosize(p_b_layout.outer)),
    )
    p_stage_elements = QUERY_HEADS * 64
    for logical_element in cutlass.range(
        tidx, P_STAGES * p_stage_elements, THREADS
    ):
        stage_element = logical_element % p_stage_elements
        stage = logical_element // p_stage_elements
        physical_element = stage * p_stage_elements + (
            stage_element ^ ((stage_element & 0x180) >> 3)
        )
        p_raw[physical_element] = cutlass.Float8E4M3FN(0.0)
    cute.arch.sync_threads()
    active_p_elements = P_STAGES * QUERY_HEADS * LOGICAL_K
    for active_element in cutlass.range(tidx, active_p_elements, THREADS):
        stage = active_element // (QUERY_HEADS * LOGICAL_K)
        stage_element = active_element % (QUERY_HEADS * LOGICAL_K)
        query = stage_element // LOGICAL_K
        token = stage_element % LOGICAL_K
        logical_element = query * 64 + token
        physical_element = stage * p_stage_elements + (
            logical_element ^ ((logical_element & 0x180) >> 3)
        )
        p_raw[physical_element] = p_input[
            page, query, stage * LOGICAL_K + token
        ]
    sfa_ptr = cute.recast_ptr(s_sfa.iterator, dtype=SF_DTYPE)
    sfb_ptr = cute.recast_ptr(s_sfb.iterator, dtype=SF_DTYPE)
    for element in cutlass.range(tidx, cute.cosize(sfa_layout), THREADS):
        (sfa_ptr + element).store(SF_DTYPE(1.0))
    for element in cutlass.range(tidx, cute.cosize(sfb_layout), THREADS):
        (sfb_ptr + element).store(SF_DTYPE(1.0))
    cute.arch.sync_threads()

    mma_producer, mma_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.mma_mbar.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()
    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_barrier,
        is_two_cta=False,
    )
    tmem.allocate(total_cols)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

    e2m1_v = e2m1_pv_mma.make_fragment_A(s_e2m1_v)
    p = e2m1_pv_mma.make_fragment_B(s_p)
    acc_shape = e2m1_pv_mma.partition_shape_C(E2M1_PV_TILER_MNK[:2])
    acc_fake = e2m1_pv_mma.make_fragment_C(acc_shape)
    acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
    view_shape = output_view_mma.partition_shape_C(OUTPUT_VIEW_TILER_MNK[:2])
    view_fake = output_view_mma.make_fragment_C(view_shape)
    output_view = cute.make_tensor(tmem_ptr, view_fake.layout)
    sfa_tmem_ptr = cute.recast_ptr(
        tmem_ptr + acc_cols, dtype=SF_DTYPE
    )
    t_sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
        e2m1_pv_mma,
        E2M1_PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfa_layout, (None, None, None, 0)),
    )
    t_sfa = cute.make_tensor(sfa_tmem_ptr, t_sfa_layout)
    sfb_tmem_ptr = cute.recast_ptr(
        tmem_ptr + acc_cols + sfa_cols, dtype=SF_DTYPE
    )
    t_sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
        e2m1_pv_mma,
        E2M1_PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfb_layout, (None, None, None, 0)),
    )
    t_sfb = cute.make_tensor(sfb_tmem_ptr, t_sfb_layout)

    if warp_idx == 0:
        scale_copy_atom = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
            SF_DTYPE,
        )
        sfa_copy = tcgen05.make_s2t_copy(
            scale_copy_atom, cute.filter_zeros(t_sfa)
        )
        sfa_thr = sfa_copy.get_slice(0)
        sfa_source = tcgen05.get_s2t_smem_desc_tensor(
            sfa_copy, sfa_thr.partition_S(cute.filter_zeros(s_sfa))
        )
        cute.copy(
            sfa_copy,
            sfa_source[None, None, None, None, 0],
            sfa_thr.partition_D(cute.filter_zeros(t_sfa)),
        )
        sfb_copy = tcgen05.make_s2t_copy(
            scale_copy_atom, cute.filter_zeros(t_sfb)
        )
        sfb_thr = sfb_copy.get_slice(0)
        sfb_source = tcgen05.get_s2t_smem_desc_tensor(
            sfb_copy, sfb_thr.partition_S(cute.filter_zeros(s_sfb))
        )
        cute.copy(
            sfb_copy,
            sfb_source[None, None, None, None, 0],
            sfb_thr.partition_D(cute.filter_zeros(t_sfb)),
        )
    output_tile = output_view[(None, None), 0, 0]
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), cutlass.Float32
    )
    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, output_tile)
    thr_load = tmem_load.get_slice(tidx)
    t_tmem = thr_load.partition_S(output_tile)

    for latent_slice in cutlass.range_constexpr(LATENT_SLICES):
        if warp_idx == 0:
            mma_producer.acquire_and_advance()
            for token_band in cutlass.range_constexpr(TOKEN_BANDS):
                e2m1_pv_mma.set(
                    tcgen05.Field.ACCUMULATE, token_band != 0
                )
                e2m1_pv_mma.set(
                    tcgen05.Field.SFA, t_sfa[None, None, 0].iterator
                )
                e2m1_pv_mma.set(
                    tcgen05.Field.SFB, t_sfb[None, None, 0].iterator
                )
                a_stage = latent_slice * TOKEN_BANDS + token_band
                cute.gemm(
                    e2m1_pv_mma,
                    acc,
                    e2m1_v[None, None, 0, a_stage],
                    p[None, None, 0, token_band],
                    acc,
                )
            mma_producer.commit()

        mma_full = mma_consumer.wait_and_advance()
        mma_full.release()
        cute.arch.sync_threads()

        output_matrix = cute.make_tensor(
            output.iterator
            + page * LATENT_DIM * QUERY_HEADS
            + latent_slice * LATENT_SLICE * QUERY_HEADS,
            cute.make_layout(
                (LATENT_SLICE, QUERY_HEADS), stride=(QUERY_HEADS, 1)
            ),
        )
        t_gmem = thr_load.partition_D(output_matrix)
        registers = cute.make_fragment_like(t_gmem, cutlass.Float32)
        cute.copy(tmem_load, t_tmem, registers)
        cute.arch.fence_view_async_tmem_load()
        cute.autovec_copy(registers, t_gmem)
        cute.arch.sync_threads()

    if tidx == 0 and page == 0:
        metadata[0] = acc_cols
        metadata[1] = sfa_cols
        metadata[2] = sfb_cols
        metadata[3] = total_cols
        metadata[4] = e2m1_bytes

    if warp_idx == 0:
        mma_producer.tail()
    tmem.relinquish_alloc_permit()
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def mixed_pv_probe(
    output: cute.Tensor,
    metadata: cute.Tensor,
    packed_v: cute.Tensor,
    p_input: cute.Tensor,
    page_table: cute.Tensor,
    use_v2: cutlass.Constexpr,
    populate_v: cutlass.Constexpr,
):
    e2m1_pv_mma, output_view_mma = make_tiled_mmas()
    e2m1_a_layout = sm100_utils.make_smem_layout_a(
        e2m1_pv_mma,
        E2M1_PV_TILER_MNK,
        cutlass.Float4E2M1FN,
        A_STAGES,
    )
    p_b_layout = sm100_utils.make_smem_layout_b(
        e2m1_pv_mma,
        E2M1_PV_TILER_MNK,
        cutlass.Float8E4M3FN,
        P_STAGES,
    )
    sfa_layout = blockscaled_utils.make_smem_layout_sfa(
        e2m1_pv_mma, E2M1_PV_TILER_MNK, SF_VEC_SIZE, 1
    )
    sfb_layout = blockscaled_utils.make_smem_layout_sfb(
        e2m1_pv_mma, E2M1_PV_TILER_MNK, SF_VEC_SIZE, 1
    )
    print(f"E2M1_PV_A_LAYOUT={e2m1_a_layout}")
    print(f"E2M1_PV_P_LAYOUT={p_b_layout}")
    acc_fake = e2m1_pv_mma.make_fragment_C(
        e2m1_pv_mma.partition_shape_C(E2M1_PV_TILER_MNK[:2])
    )
    acc_cols = utils.get_num_tmem_alloc_cols(acc_fake)
    sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
        e2m1_pv_mma,
        E2M1_PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfa_layout, (None, None, None, 0)),
    )
    sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
        e2m1_pv_mma,
        E2M1_PV_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfb_layout, (None, None, None, 0)),
    )
    sfa_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(
            cute.make_ptr(SF_DTYPE, 0), sfa_tmem_layout
        )
    )
    sfb_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(
            cute.make_ptr(SF_DTYPE, 0), sfb_tmem_layout
        )
    )
    raw_cols = acc_cols + sfa_cols + sfb_cols
    total_cols = max(32, 1 << math.ceil(math.log2(raw_cols)))
    print(
        f"TMEM_COLS acc={acc_cols} sfa={sfa_cols} sfb={sfb_cols} "
        f"raw={raw_cols} allocation={total_cols}"
    )
    if cutlass.const_expr(total_cols > 512):
        raise ValueError(
            f"TMEM overflow: acc={acc_cols}, sfa={sfa_cols}, "
            f"sfb={sfb_cols}, raw={raw_cols}, allocation={total_cols}"
        )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (e2m1_pv_mma.thr_id.shape,)
    )
    mixed_pv_kernel(
        output,
        metadata,
        packed_v,
        p_input,
        page_table,
        use_v2,
        populate_v,
        e2m1_pv_mma,
        output_view_mma,
        e2m1_a_layout,
        p_b_layout,
        sfa_layout,
        sfb_layout,
        acc_cols,
        sfa_cols,
        sfb_cols,
        total_cols,
        cta_layout_vmnk,
    ).launch(
        grid=(NUM_PAGES, 1, 1),
        block=(THREADS, 1, 1),
        cluster=CLUSTER_SHAPE_MNK,
        min_blocks_per_mp=1,
    )


def main() -> None:
    native_nibbles = [e2m1_nibble_for_v2_code(code) for code in range(4)]
    assert native_nibbles == [0xB, 0x9, 0x1, 0x3]
    for packed in range(1 << 16):
        reference = sum(
            native_nibbles[(packed >> (2 * index)) & 0x3] << (4 * index)
            for index in range(8)
        )
        assert e2m1_word_for_v2_halfword(packed) == reference
    assert lop3_truth_table_imm(lambda a, b, c: (a | b) & c) == 0xA8
    assert lop3_truth_table_imm(lambda a, b, c: (not (a ^ b)) and c) == 0x82
    assert (
        lop3_truth_table_imm(lambda a, b, c: a | ((not b) and c)) == 0xF2
    )

    scalar_destinations = []
    word_destinations = []
    for tile in range(A_STAGES):
        stage_base = tile * LATENT_SLICE * 32
        for latent in range(LATENT_SLICE):
            logical_row = latent * 32
            swizzle = (logical_row & 0x80) >> 3
            for token_group in range(LOGICAL_K // 4):
                token_pair0 = token_group * 2
                token_pair1 = token_pair0 + 1
                storage_k0 = token_pair0 * 2 + (16 if token_pair0 >= 8 else 0)
                storage_k1 = token_pair1 * 2 + (16 if token_pair1 >= 8 else 0)
                logical_byte0 = logical_row + storage_k0 // 2
                logical_byte1 = logical_row + storage_k1 // 2
                scalar_destinations.extend(
                    (
                        stage_base + (logical_byte0 ^ swizzle),
                        stage_base + (logical_byte1 ^ swizzle),
                    )
                )
            for logical_offset in (0, 4, 16, 20):
                physical_word = stage_base + (
                    (logical_row + logical_offset) ^ swizzle
                )
                assert physical_word % 4 == 0
                word_destinations.extend(range(physical_word, physical_word + 4))
    assert len(scalar_destinations) == 32768
    assert len(set(scalar_destinations)) == 32768
    assert set(word_destinations) == set(scalar_destinations)
    assert len(word_destinations) == len(set(word_destinations))
    print("HOST_MAP_LOP3_AND_DESTINATION_OWNERSHIP_PASS bytes=32768")

    output_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (NUM_PAGES, LATENT_DIM, QUERY_HEADS),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    metadata_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (5,),
        stride_order=(0,),
        assumed_align=4,
    )
    p_fake = make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (NUM_PAGES, QUERY_HEADS, PAGE_TOKENS),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    page_table_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (NUM_PAGES, PHYSICAL_PAGES_PER_TILE),
        stride_order=(1, 0),
        assumed_align=4,
    )
    control_compiled = cute.compile(
        mixed_pv_probe,
        output_fake,
        metadata_fake,
        make_fake_compact_tensor(
            cutlass.Uint8,
            (NUM_PAGES, PAGE_TOKENS, LATENT_DIM // 2),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        p_fake,
        page_table_fake,
        False,
        True,
        options="--enable-tvm-ffi --opt-level 3",
    )
    candidate_compiled = cute.compile(
        mixed_pv_probe,
        output_fake,
        metadata_fake,
        make_fake_compact_tensor(
            cutlass.Uint8,
            (NUM_PHYSICAL_PAGES, LATENT_DIM, PHYSICAL_PAGE_TOKENS // 4),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        p_fake,
        page_table_fake,
        True,
        True,
        options="--enable-tvm-ffi --opt-level 3",
    )
    floor_compiled = cute.compile(
        mixed_pv_probe,
        output_fake,
        metadata_fake,
        make_fake_compact_tensor(
            cutlass.Uint8,
            (NUM_PAGES, PAGE_TOKENS, LATENT_DIM // 2),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        p_fake,
        page_table_fake,
        False,
        False,
        options="--enable-tvm-ffi --opt-level 3",
    )
    if os.environ.get("TQ_PV_COMPILE_ONLY") == "1":
        print("COMPILE_ONLY_PASS all_three_specializations=True")
        return
    output = torch.full(
        (NUM_PAGES, LATENT_DIM, QUERY_HEADS),
        float("nan"),
        device="cuda",
        dtype=torch.float32,
    )
    metadata = torch.empty(5, device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(20260723)
    physical_codes = torch.randint(
        0,
        4,
        (NUM_PHYSICAL_PAGES, PHYSICAL_PAGE_TOKENS, LATENT_DIM),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    page_table = torch.randperm(
        NUM_PHYSICAL_PAGES, generator=generator, device="cuda", dtype=torch.int64
    ).reshape(NUM_PAGES, PHYSICAL_PAGES_PER_TILE)
    logical_codes = physical_codes[page_table].reshape(
        NUM_PAGES, PAGE_TOKENS, LATENT_DIM
    )
    v2_by_latent = physical_codes.permute(0, 2, 1).contiguous()
    packed_v2 = (
        v2_by_latent[:, :, 0::4]
        | (v2_by_latent[:, :, 1::4] << 2)
        | (v2_by_latent[:, :, 2::4] << 4)
        | (v2_by_latent[:, :, 3::4] << 6)
    ).contiguous()
    native_codebook = torch.tensor(
        [0xB, 0x9, 0x1, 0x3],
        device="cuda",
        dtype=torch.uint8,
    )
    assert [e2m1_nibble_for_v2_code(code) for code in range(4)] == [
        0xB,
        0x9,
        0x1,
        0x3,
    ]
    native_codes = native_codebook[logical_codes.long()]
    packed_k4 = (
        native_codes[:, :, 0::2] | (native_codes[:, :, 1::2] << 4)
    ).contiguous()
    v2_codebook = torch.tensor(
        [-1.5, -0.5, 0.5, 1.5], device="cuda", dtype=torch.float32
    )
    decoded_v = v2_codebook[logical_codes.long()]
    page_table_i32 = page_table.to(torch.int32)

    valid_tokens = torch.full(
        (NUM_PAGES,), PAGE_TOKENS, device="cuda", dtype=torch.int32
    )
    valid_tokens[-1] = PAGE_TOKENS - 11
    token_positions = torch.arange(PAGE_TOKENS, device="cuda")
    valid_mask = token_positions[None, :] < valid_tokens[:, None]

    def run_and_check(label: str, p_input: torch.Tensor) -> None:
        expected = torch.bmm(p_input.float(), decoded_v).transpose(1, 2)
        arm_outputs = {}
        for arm, compiled, packed in (
            ("control_k4", control_compiled, packed_k4),
            ("candidate_v2", candidate_compiled, packed_v2),
        ):
            output.fill_(float("nan"))
            compiled(output, metadata, packed, p_input, page_table_i32)
            torch.cuda.synchronize()
            mismatch = output != expected
            if mismatch.any():
                bad = mismatch.nonzero()
                print(
                    f"DIAGNOSTIC label={label} arm={arm} "
                    f"max_abs={(output - expected).abs().max().item()} "
                    f"bad_tiles={torch.unique(bad[:, 0]).cpu().tolist()} "
                    f"bad_latents={torch.unique(bad[:, 1]).cpu().tolist()} "
                    f"bad_heads={torch.unique(bad[:, 2]).cpu().tolist()} "
                    f"output_sample={output[0, :8, :8].cpu().tolist()} "
                    f"expected_sample={expected[0, :8, :8].cpu().tolist()} "
                    f"metadata={metadata.cpu().tolist()}"
                )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            arm_outputs[arm] = output.clone()
        torch.testing.assert_close(
            arm_outputs["candidate_v2"],
            arm_outputs["control_k4"],
            rtol=0,
            atol=0,
        )
        print(f"CORRECTNESS_PASS label={label}")

    def run_floor_check(p_input: torch.Tensor) -> None:
        output.fill_(float("nan"))
        floor_compiled(output, metadata, packed_k4, p_input, page_table_i32)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, torch.zeros_like(output), rtol=0, atol=0
        )
        print("CORRECTNESS_PASS label=shared-floor-zero-v")

    queries = torch.arange(QUERY_HEADS, device="cuda")
    for selector_base in range(0, 128, 16):
        p_selector = torch.zeros(
            (NUM_PAGES, QUERY_HEADS, PAGE_TOKENS),
            device="cuda",
            dtype=torch.bfloat16,
        )
        p_selector[
            :,
            queries,
            selector_base + queries.remainder(16),
        ] = 1.0
        p_selector.masked_fill_(~valid_mask[:, None, :], 0.0)
        p_selector = p_selector.to(torch.float8_e4m3fn)
        run_and_check(f"selector-{selector_base}", p_selector)

    p_values = torch.tensor(
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], device="cuda", dtype=torch.float32
    )
    p_indices = torch.randint(
        0,
        p_values.numel(),
        (NUM_PAGES, QUERY_HEADS, PAGE_TOKENS),
        generator=generator,
        device="cuda",
    )
    token_scales = torch.tensor(
        [0.5, 1.0, 2.0], device="cuda", dtype=torch.float32
    )[token_positions.remainder(3)]
    p_dense_fp32 = p_values[p_indices] * token_scales[None, None, :]
    p_dense_fp32.masked_fill_(~valid_mask[:, None, :], 0.0)
    p_dense = p_dense_fp32.to(torch.float8_e4m3fn)
    run_and_check("dense", p_dense)
    run_and_check("eager-replay", p_dense)
    run_floor_check(p_dense)
    if os.environ.get("TQ_PV_CORRECTNESS_ONLY") == "1":
        print("CORRECTNESS_ONLY_PASS all_surfaces=True")
        return

    warmup = max(20, int(os.environ.get("TQ_PV_BENCH_WARMUP", "20")))
    iterations = int(os.environ.get("TQ_PV_BENCH_ITERS", "200"))
    windows = max(12, int(os.environ.get("TQ_PV_BENCH_WINDOWS", "12")))
    arms = {
        "control_k4": (control_compiled, packed_k4),
        "candidate_v2": (candidate_compiled, packed_v2),
        "shared_floor": (floor_compiled, packed_k4),
    }
    arm_orders = (
        ("control_k4", "candidate_v2", "shared_floor"),
        ("candidate_v2", "shared_floor", "control_k4"),
        ("shared_floor", "control_k4", "candidate_v2"),
        ("shared_floor", "candidate_v2", "control_k4"),
        ("candidate_v2", "control_k4", "shared_floor"),
        ("control_k4", "shared_floor", "candidate_v2"),
    )
    for warmup_index in range(warmup):
        order = arm_orders[warmup_index % len(arm_orders)]
        for arm in order:
            compiled, packed = arms[arm]
            compiled(output, metadata, packed, p_dense, page_table_i32)
    torch.cuda.synchronize()

    def measure(compiled, packed) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            compiled(output, metadata, packed, p_dense, page_table_i32)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0 / iterations

    control_us = []
    candidate_us = []
    floor_us = []
    total_ratios = []
    excess_ratios = []
    for window in range(windows):
        order = arm_orders[window % len(arm_orders)]
        measured = {}
        for arm in order:
            compiled, packed = arms[arm]
            measured[arm] = measure(compiled, packed)
        control_us.append(measured["control_k4"])
        candidate_us.append(measured["candidate_v2"])
        floor_us.append(measured["shared_floor"])
        total_ratios.append(measured["candidate_v2"] / measured["control_k4"])
        control_excess = measured["control_k4"] - measured["shared_floor"]
        candidate_excess = measured["candidate_v2"] - measured["shared_floor"]
        if control_excess > 0:
            excess_ratios.append(candidate_excess / control_excess)
        else:
            excess_ratios.append(float("nan"))
        print(
            f"WINDOW index={window} order={order} "
            f"control_us={control_us[-1]:.6f} "
            f"candidate_us={candidate_us[-1]:.6f} "
            f"floor_us={floor_us[-1]:.6f} "
            f"total_ratio={total_ratios[-1]:.9f} "
            f"excess_ratio={excess_ratios[-1]:.9f}"
        )

    mean_excess_ratio = sum(excess_ratios) / len(excess_ratios)
    excess_ratio_variance = sum(
        (ratio - mean_excess_ratio) ** 2 for ratio in excess_ratios
    ) / (len(excess_ratios) - 1)
    excess_ratio_sem = math.sqrt(excess_ratio_variance / len(excess_ratios))
    one_sided_critical = {9: 1.859548, 12: 1.795885}.get(windows, 1.96)
    excess_ratio_lower = mean_excess_ratio - one_sided_critical * excess_ratio_sem
    excess_ratio_upper = mean_excess_ratio + one_sided_critical * excess_ratio_sem

    def range_over_mean(samples: list[float]) -> float:
        mean = sum(samples) / len(samples)
        return (max(samples) - min(samples)) / mean

    control_drift = range_over_mean(control_us)
    candidate_drift = range_over_mean(candidate_us)
    floor_drift = range_over_mean(floor_us)
    identifiable = all(
        control > floor for control, floor in zip(control_us, floor_us)
    ) and all(math.isfinite(ratio) for ratio in excess_ratios)
    if max(control_drift, candidate_drift, floor_drift) > 0.02:
        decision = "NO_DECISION_CONTROL_DRIFT"
    elif not identifiable:
        decision = "NO_DECISION_FLOOR_NOT_IDENTIFIABLE"
    elif excess_ratio_upper <= 0.50:
        decision = "ADVANCE_WORD_PRIMITIVE"
    elif excess_ratio_lower >= 0.70:
        decision = "REJECT_WORD_PRIMITIVE"
    else:
        decision = "GRAY_ONE_ABLATION"

    acc_cols, sfa_cols, sfb_cols, total_cols, e2m1_bytes = (
        metadata.cpu().tolist()
    )
    result = {
        "decision": decision,
        "logical_tiles": NUM_PAGES,
        "physical_page_tokens": PHYSICAL_PAGE_TOKENS,
        "physical_pages": NUM_PHYSICAL_PAGES,
        "shuffled_page_table": True,
        "partial_last_tile_tokens": int(valid_tokens[-1].item()),
        "control_storage_bytes": packed_k4.numel() * packed_k4.element_size(),
        "candidate_storage_bytes": packed_v2.numel() * packed_v2.element_size(),
        "control_us": control_us,
        "candidate_us": candidate_us,
        "shared_floor_us": floor_us,
        "paired_total_ratios": total_ratios,
        "paired_excess_ratios": excess_ratios,
        "mean_total_ratio": sum(total_ratios) / len(total_ratios),
        "mean_excess_ratio": mean_excess_ratio,
        "one_sided_lower_excess_ratio": excess_ratio_lower,
        "one_sided_upper_excess_ratio": excess_ratio_upper,
        "one_sided_critical": one_sided_critical,
        "control_range_over_mean": control_drift,
        "candidate_range_over_mean": candidate_drift,
        "floor_range_over_mean": floor_drift,
        "floor_identifiable": identifiable,
        "acc_cols": acc_cols,
        "sfa_cols": sfa_cols,
        "sfb_cols": sfb_cols,
        "total_cols": total_cols,
        "e2m1_bytes": e2m1_bytes,
    }
    print("PASS selector=True dense=True eager_replay=True latent_boundaries=True")
    print(f"RESULT {json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
