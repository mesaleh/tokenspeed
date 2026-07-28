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

"""Probe a fused block-scaled E2M1-V by FP8-P page tile on SM100.

The no-shadow TurboQuant PV path folds each token's BF16 cache scale into P,
casts the scaled probabilities to FP8, and multiplies a bounded transpose of
the canonical packed E2M1 V tile by a 64-query FP8 operand. A padded K64 shared
carrier gives the mixed descriptor distinct addresses for both logical K16
groups while one block-scaled K32 instruction consumes all 32 useful lanes.
Unit UE8M0 scale factors isolate the data path. A two-CTA cluster fuses four
32-token bands and two 256-latent slices into the full 128-token by 512-latent
page. The default 80-page grid models an exact 10,240-token context. Its
per-page output tensor is measurement scaffolding, not an admissible
integration workspace; production must reuse TokenSpeed's split-reduction
state.
"""

import math
import os

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor

THREADS = 128
PAGE_TOKENS = 128
NUM_PAGES = int(os.environ.get("TQ_PV_NUM_PAGES", "80"))
LATENT_DIM = 512
QUERY_HEADS = 64
CTA_QUERY_HEADS = QUERY_HEADS // 2
LOGICAL_K = 32
TOKEN_BANDS = PAGE_TOKENS // LOGICAL_K
LATENT_SLICE = 256
CTA_LATENT_SLICE = LATENT_SLICE // 2
LATENT_SLICES = LATENT_DIM // LATENT_SLICE
A_STAGES = TOKEN_BANDS * LATENT_SLICES
P_STAGES = TOKEN_BANDS
SF_DTYPE = cutlass.Float8E8M0FNU
SF_VEC_SIZE = 32
CLUSTER_SHAPE_MNK = (2, 1, 1)
E2M1_PV_TILER_MNK = (LATENT_SLICE, QUERY_HEADS, 64)
OUTPUT_VIEW_TILER_MNK = (LATENT_SLICE, QUERY_HEADS, 32)


@cute.struct
class SharedStorage:
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


def make_tiled_mmas():
    e2m1_pv = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        SF_DTYPE,
        SF_VEC_SIZE,
        tcgen05.CtaGroup.TWO,
        E2M1_PV_TILER_MNK[:2],
    )
    output_view = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        (128, QUERY_HEADS),
    )
    return e2m1_pv, output_view


@cute.kernel
def mixed_pv_kernel(
    output: cute.Tensor,
    metadata: cute.Tensor,
    packed_v: cute.Tensor,
    p_input: cute.Tensor,
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
    block, _, _ = cute.arch.block_idx()
    page = block // 2
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    is_leader_cta = cta_rank == 0

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

    # Transpose canonical [token, latent_pair] bytes into a padded K-major A
    # operand. The mixed descriptor advances 16 bytes at logical K16, so the
    # upper token group starts at storage K32 and each latent row spans 32
    # bytes. S<1,4,3> permutes element offsets but preserves nibble pairs.
    e2m1_bytes = cute.size_in_bytes(cutlass.Float4E2M1FN, s_e2m1_v)
    v_raw = cute.make_tensor(
        cute.recast_ptr(
            s_e2m1_v.iterator, swizzle_=None, dtype=cutlass.Uint8
        ),
        cute.make_layout(e2m1_bytes),
    )
    for destination_byte in cutlass.range(tidx, e2m1_bytes, THREADS):
        v_raw[destination_byte] = cutlass.Uint8(0)
    active_a_bytes = A_STAGES * CTA_LATENT_SLICE * (LOGICAL_K // 2)
    for active_byte in cutlass.range(tidx, active_a_bytes, THREADS):
        tile = active_byte // (CTA_LATENT_SLICE * (LOGICAL_K // 2))
        tile_byte = active_byte % (CTA_LATENT_SLICE * (LOGICAL_K // 2))
        latent = tile_byte // (LOGICAL_K // 2)
        token_pair = tile_byte % (LOGICAL_K // 2)
        latent_slice = tile // TOKEN_BANDS
        token_band = tile % TOKEN_BANDS
        token0 = token_band * LOGICAL_K + token_pair * 2
        token1 = token0 + 1
        source_latent = (
            latent_slice * LATENT_SLICE + cta_rank * CTA_LATENT_SLICE + latent
        )
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
            tile * CTA_LATENT_SLICE * 32
            + (logical_byte ^ ((logical_byte & 0x80) >> 3))
        )
        v_raw[physical_byte] = cutlass.Uint8(code0 | (code1 << 4))
    p_raw = cute.make_tensor(
        cute.recast_ptr(s_p.iterator, swizzle_=None, dtype=cutlass.Float8E4M3FN),
        cute.make_layout(cute.cosize(p_b_layout.outer)),
    )
    p_stage_elements = CTA_QUERY_HEADS * 64
    for logical_element in cutlass.range(
        tidx, P_STAGES * p_stage_elements, THREADS
    ):
        stage_element = logical_element % p_stage_elements
        stage = logical_element // p_stage_elements
        physical_element = stage * p_stage_elements + (
            stage_element ^ ((stage_element & 0x180) >> 3)
        )
        p_raw[physical_element] = cutlass.Float8E4M3FN(0.0)
    active_p_elements = P_STAGES * CTA_QUERY_HEADS * LOGICAL_K
    for active_element in cutlass.range(tidx, active_p_elements, THREADS):
        stage = active_element // (CTA_QUERY_HEADS * LOGICAL_K)
        stage_element = active_element % (CTA_QUERY_HEADS * LOGICAL_K)
        local_query = stage_element // LOGICAL_K
        token = stage_element % LOGICAL_K
        logical_element = local_query * 64 + token
        physical_element = stage * p_stage_elements + (
            logical_element ^ ((logical_element & 0x180) >> 3)
        )
        p_raw[physical_element] = p_input[
            page,
            cta_rank * CTA_QUERY_HEADS + local_query,
            stage * LOGICAL_K + token,
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
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, THREADS * 2
        ),
        barrier_storage=storage.mma_mbar.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()
    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_barrier,
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
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

    if warp_idx == 0 and is_leader_cta:
        scale_copy_atom = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO),
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
    output_tile = acc[(None, None), 0, 0]
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), cutlass.Float32
    )
    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, output_tile)
    thr_load = tmem_load.get_slice(tidx)
    t_tmem = thr_load.partition_S(output_tile)

    for latent_slice in cutlass.range_constexpr(LATENT_SLICES):
        if warp_idx == 0 and is_leader_cta:
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
            + latent_slice * LATENT_SLICE * QUERY_HEADS
            + cta_rank * CTA_LATENT_SLICE * QUERY_HEADS,
            cute.make_layout(
                (CTA_LATENT_SLICE, QUERY_HEADS), stride=(QUERY_HEADS, 1)
            ),
        )
        t_gmem = thr_load.partition_D(output_matrix)
        registers = cute.make_fragment_like(t_gmem, cutlass.Float32)
        cute.copy(tmem_load, t_tmem, registers)
        cute.arch.fence_view_async_tmem_load()
        cute.autovec_copy(registers, t_gmem)
        cute.arch.sync_threads()

    if tidx == 0 and page == 0 and is_leader_cta:
        metadata[0] = acc_cols
        metadata[1] = sfa_cols
        metadata[2] = sfb_cols
        metadata[3] = total_cols
        metadata[4] = e2m1_bytes

    if warp_idx == 0 and is_leader_cta:
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
        grid=(NUM_PAGES * 2, 1, 1),
        block=(THREADS, 1, 1),
        cluster=CLUSTER_SHAPE_MNK,
        min_blocks_per_mp=1,
    )


def main() -> None:
    compiled = cute.compile(
        mixed_pv_probe,
        make_fake_compact_tensor(
            cutlass.Float32,
            (NUM_PAGES, LATENT_DIM, QUERY_HEADS),
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
            cutlass.Uint8,
            (NUM_PAGES, PAGE_TOKENS, LATENT_DIM // 2),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (NUM_PAGES, QUERY_HEADS, PAGE_TOKENS),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        options="--enable-tvm-ffi --opt-level 3",
    )
    output = torch.full(
        (NUM_PAGES, LATENT_DIM, QUERY_HEADS),
        float("nan"),
        device="cuda",
        dtype=torch.float32,
    )
    metadata = torch.empty(5, device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(20260723)
    codes = torch.randint(
        0,
        16,
        (NUM_PAGES, PAGE_TOKENS, LATENT_DIM),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    packed_v = codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)
    codebook = torch.tensor(
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
        dtype=torch.bfloat16,
    )
    decoded_v = codebook.float()[codes.long()]

    def run_and_check(label: str, p_input: torch.Tensor) -> None:
        output.fill_(float("nan"))
        compiled(output, metadata, packed_v, p_input)
        torch.cuda.synchronize()
        expected = torch.bmm(p_input.float(), decoded_v).transpose(1, 2)
        mismatch = output != expected
        if mismatch.any() or label.startswith("selector"):
            bad = mismatch.nonzero()
            column_matches = {}
            if label.startswith("selector"):
                for query in range(QUERY_HEADS):
                    matches = (
                        decoded_v[0, :, : LATENT_DIM - 1]
                        == output[0, : LATENT_DIM - 1, query][None, :]
                    ).all(dim=1).nonzero()
                    column_matches[query] = matches.flatten().cpu().tolist()
            print(
                f"DIAGNOSTIC label={label} "
                f"max_abs={(output - expected).abs().max().item()} "
                f"bad_pages={torch.unique(bad[:, 0]).cpu().tolist()} "
                f"bad_rows={torch.unique(bad[:, 1]).cpu().tolist()} "
                f"bad_columns={torch.unique(bad[:, 2]).cpu().tolist()} "
                f"column_matches={column_matches} "
                f"output_sample={output[0, :8, :8].cpu().tolist()} "
                f"expected_sample={expected[0, :8, :8].cpu().tolist()} "
                f"metadata={metadata.cpu().tolist()}"
            )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

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
        p_selector = p_selector.to(torch.float8_e4m3fn)
        run_and_check(f"selector-{selector_base}", p_selector)

    p_values = torch.tensor(
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], device="cuda", dtype=torch.float32
    )
    query_ids = torch.arange(
        QUERY_HEADS, device="cuda", dtype=torch.int32
    )[:, None]
    token_ids = torch.arange(
        PAGE_TOKENS, device="cuda", dtype=torch.int32
    )[None, :]
    p_indices = (query_ids * 17 + token_ids * 5 + 1) % p_values.numel()
    token_scales = torch.tensor(
        [0.5, 1.0, 2.0], device="cuda", dtype=torch.float32
    )[token_ids.remainder(3)]
    p_dense_page = (p_values[p_indices] * token_scales).to(
        torch.float8_e4m3fn
    )
    p_dense = p_dense_page[None].expand(NUM_PAGES, -1, -1).contiguous()
    run_and_check("dense", p_dense)

    warmup = int(os.environ.get("TQ_PV_BENCH_WARMUP", "20"))
    iterations = int(os.environ.get("TQ_PV_BENCH_ITERS", "200"))
    for _ in range(warmup):
        compiled(output, metadata, packed_v, p_dense)
    torch.cuda.synchronize()
    timings = []
    for _ in range(2):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            compiled(output, metadata, packed_v, p_dense)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / iterations)

    acc_cols, sfa_cols, sfb_cols, total_cols, e2m1_bytes = (
        metadata.cpu().tolist()
    )
    print(
        f"PASS blockscaled_e2m1_v_fp8_p_n64_pages={NUM_PAGES} "
        f"latent_slices={LATENT_SLICES} "
        "token_bands=4 selector=True dense=True full_k32=True "
        "padded_k64=True cta_group=2 "
        f"acc_cols={acc_cols} sfa_cols={sfa_cols} sfb_cols={sfb_cols} "
        f"total_cols={total_cols} e2m1_bytes={e2m1_bytes} "
        f"microseconds={timings}"
    )


if __name__ == "__main__":
    main()
