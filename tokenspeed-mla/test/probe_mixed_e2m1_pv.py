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

"""Probe a narrow reversed E2M1-V by FP8-P MMA tile on SM100.

The no-shadow TurboQuant PV path folds each token's BF16 cache scale into P,
casts the scaled probabilities to FP8, and multiplies a bounded transpose of
the packed E2M1 V tile by a 16-query FP8 operand. The mixed instruction's lower
K=16 lane group is exact while its upper group aliases and shifts the lower
group, so this probe zeroes the upper group and issues eight logical K=16 tiles.
Four N=16 tiles cover Kimi's 64 heads. The final latent row is corrected from
canonical packed bytes. This is a legality probe, not an attention benchmark.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor

THREADS = 128
LOGICAL_K = 16
CLUSTER_SHAPE_MNK = (1, 1, 1)
E2M1_PV_TILER_MNK = (128, 16, 32)
OUTPUT_VIEW_TILER_MNK = (128, 16, 32)


@cute.struct
class SharedStorage:
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


def make_tiled_mmas():
    e2m1_pv = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
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


@cute.kernel
def mixed_pv_kernel(
    output: cute.Tensor,
    metadata: cute.Tensor,
    packed_v: cute.Tensor,
    codebook: cute.Tensor,
    p_input: cute.Tensor,
    p_reference: cute.Tensor,
    latent_base: cutlass.Int32,
    token_base: cutlass.Int32,
    e2m1_pv_mma: cute.TiledMma,
    output_view_mma: cute.TiledMma,
    e2m1_a_layout: cute.ComposedLayout,
    p_b_layout: cute.ComposedLayout,
    acc_cols: cutlass.Constexpr,
    cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
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

    # Transpose canonical [token, latent_pair] bytes into the E2M1 A operand's
    # [latent, token_pair] byte order without allocating a decoded tensor.
    e2m1_bytes = cute.size_in_bytes(cutlass.Float4E2M1FN, s_e2m1_v)
    v_raw = cute.make_tensor(
        cute.recast_ptr(
            s_e2m1_v.iterator, swizzle_=None, dtype=cutlass.Uint8
        ),
        cute.make_layout(e2m1_bytes),
    )
    for destination_byte in cutlass.range(tidx, 128 * 16, THREADS):
        v_raw[destination_byte] = cutlass.Uint8(0)
    for active_byte in cutlass.range(tidx, 128 * (LOGICAL_K // 2), THREADS):
        latent = active_byte // (LOGICAL_K // 2)
        token_pair = active_byte % (LOGICAL_K // 2)
        destination_byte = latent * 16 + token_pair
        token0 = token_base + token_pair * 2
        token1 = token0 + 1
        source_latent = latent_base + latent
        source_byte = source_latent // 2
        shift = (source_latent % 2) * 4
        code0 = (cutlass.Int32(packed_v[token0, source_byte]) >> shift) & 0xF
        code1 = (cutlass.Int32(packed_v[token1, source_byte]) >> shift) & 0xF
        physical_byte = destination_byte
        v_raw[physical_byte] = cutlass.Uint8(code0 | (code1 << 4))
    p_raw = cute.make_tensor(
        cute.recast_ptr(s_p.iterator, swizzle_=None, dtype=cutlass.Float8E4M3FN),
        cute.make_layout(cute.cosize(p_b_layout.outer)),
    )
    for logical_element in cutlass.range(tidx, 16 * 32, THREADS):
        physical_element = logical_element ^ ((logical_element & 0x80) >> 3)
        p_raw[physical_element] = cutlass.Float8E4M3FN(0.0)
    for active_element in cutlass.range(tidx, 16 * LOGICAL_K, THREADS):
        query = active_element // LOGICAL_K
        token = active_element % LOGICAL_K
        logical_element = query * 32 + token
        physical_element = logical_element ^ ((logical_element & 0x80) >> 3)
        p_raw[physical_element] = p_input[query, token]
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
    tmem.allocate(acc_cols)
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

    if warp_idx == 0:
        mma_producer.acquire_and_advance()
        e2m1_pv_mma.set(tcgen05.Field.ACCUMULATE, False)
        k_blocks = cute.size(e2m1_v, mode=[2])
        for k_block in cutlass.range(k_blocks, unroll_full=True):
            cute.gemm(
                e2m1_pv_mma,
                acc,
                e2m1_v[None, None, k_block, 0],
                p[None, None, k_block, 0],
                acc,
            )
            e2m1_pv_mma.set(tcgen05.Field.ACCUMULATE, True)
        mma_producer.commit()

    mma_full = mma_consumer.wait_and_advance()
    mma_full.release()
    cute.arch.sync_threads()

    output_tile = output_view[(None, None), 0, 0]
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(4)), cutlass.Float32
    )
    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, output_tile)
    thr_load = tmem_load.get_slice(tidx)
    output_matrix = cute.make_tensor(
        output[None, None].iterator,
        cute.make_layout((128, 16), stride=(16, 1)),
    )
    t_tmem = thr_load.partition_S(output_tile)
    t_gmem = thr_load.partition_D(output_matrix)
    coordinates = cute.make_identity_tensor((128, 16))
    t_coordinates = thr_load.partition_D(coordinates)
    registers = cute.make_fragment_like(t_gmem, cutlass.Float32)
    cute.copy(tmem_load, t_tmem, registers)
    cute.arch.fence_view_async_tmem_load()
    for element in cutlass.range_constexpr(cute.size(registers)):
        latent = t_coordinates[element][0]
        query = t_coordinates[element][1]
        if latent == 127:
            correction = cutlass.Float32(0.0)
            for token in cutlass.range(LOGICAL_K):
                global_token = token_base + token
                source_latent = latent_base + 127
                packed = cutlass.Int32(packed_v[global_token, source_latent // 2])
                code = (packed >> ((source_latent % 2) * 4)) & 0xF
                correction += cutlass.Float32(codebook[code]) * cutlass.Float32(
                    p_reference[query, token]
                )
            registers[element] = correction
    cute.autovec_copy(registers, t_gmem)
    cute.arch.sync_threads()

    if tidx == 0:
        metadata[0] = acc_cols
        metadata[1] = e2m1_bytes
        metadata[2] = cute.cosize(e2m1_a_layout.outer)

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
    codebook: cute.Tensor,
    p_input: cute.Tensor,
    p_reference: cute.Tensor,
    latent_base: cutlass.Int32,
    token_base: cutlass.Int32,
):
    e2m1_pv_mma, output_view_mma = make_tiled_mmas()
    e2m1_a_layout = sm100_utils.make_smem_layout_a(
        e2m1_pv_mma, E2M1_PV_TILER_MNK, cutlass.Float4E2M1FN, 1
    )
    p_b_layout = sm100_utils.make_smem_layout_b(
        e2m1_pv_mma, E2M1_PV_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    print(f"E2M1_PV_A_LAYOUT={e2m1_a_layout}")
    print(f"E2M1_PV_P_LAYOUT={p_b_layout}")
    acc_fake = e2m1_pv_mma.make_fragment_C(
        e2m1_pv_mma.partition_shape_C(E2M1_PV_TILER_MNK[:2])
    )
    acc_cols = utils.get_num_tmem_alloc_cols(acc_fake)
    if cutlass.const_expr(acc_cols > 512):
        raise ValueError(f"TMEM overflow: acc={acc_cols}")
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (e2m1_pv_mma.thr_id.shape,)
    )
    mixed_pv_kernel(
        output,
        metadata,
        packed_v,
        codebook,
        p_input,
        p_reference,
        latent_base,
        token_base,
        e2m1_pv_mma,
        output_view_mma,
        e2m1_a_layout,
        p_b_layout,
        acc_cols,
        cta_layout_vmnk,
    ).launch(
        grid=CLUSTER_SHAPE_MNK,
        block=(THREADS, 1, 1),
        cluster=CLUSTER_SHAPE_MNK,
        min_blocks_per_mp=1,
    )


def main() -> None:
    compiled = cute.compile(
        mixed_pv_probe,
        make_fake_compact_tensor(
            cutlass.Float32,
            (128, 16),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.Int32,
            (3,),
            stride_order=(0,),
            assumed_align=4,
        ),
        make_fake_compact_tensor(
            cutlass.Uint8,
            (128, 256),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (16,),
            stride_order=(0,),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (16, 32),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (16, 32),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        cutlass.Int32(0),
        cutlass.Int32(0),
        options="--enable-tvm-ffi --opt-level 3",
    )
    output = torch.full(
        (128, 16), float("nan"), device="cuda", dtype=torch.float32
    )
    metadata = torch.empty(3, device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(20260723)
    codes = torch.randint(
        0, 16, (128, 512), generator=generator, device="cuda", dtype=torch.uint8
    )
    packed_v = codes[:, 0::2] | (codes[:, 1::2] << 4)
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

    def run_and_check(
        label: str,
        p_input: torch.Tensor,
        expected: torch.Tensor,
        *,
        latent_bases: tuple[int, ...] = (0, 128, 256, 384),
    ) -> None:
        for latent_base in latent_bases:
            accumulated = torch.zeros_like(output)
            for token_base in range(0, 128, LOGICAL_K):
                p_block = torch.zeros(
                    (16, 32), device="cuda", dtype=torch.float8_e4m3fn
                )
                p_block[:, :LOGICAL_K] = p_input[
                    :, token_base : token_base + LOGICAL_K
                ]
                p_reference = p_block.float().to(torch.bfloat16)
                output.fill_(float("nan"))
                compiled(
                    output,
                    metadata,
                    packed_v,
                    codebook,
                    p_block,
                    p_reference,
                    cutlass.Int32(latent_base),
                    cutlass.Int32(token_base),
                )
                torch.cuda.synchronize()
                accumulated += output
            expected_slice = expected[:, latent_base : latent_base + 128].T
            mismatch = accumulated != expected_slice
            if mismatch.any() or label.startswith("selector"):
                bad = mismatch.nonzero()
                column_matches = {}
                if label.startswith("selector"):
                    for query in range(16):
                        matches = (
                            decoded_v[:, latent_base : latent_base + 127]
                            == accumulated[:127, query][None, :]
                        ).all(dim=1).nonzero()
                        column_matches[query] = matches.flatten().cpu().tolist()
                print(
                    f"DIAGNOSTIC label={label} latent_base={latent_base} "
                    f"max_abs={(accumulated - expected_slice).abs().max().item()} "
                    f"bad_rows={torch.unique(bad[:, 0]).cpu().tolist()} "
                    f"bad_columns={torch.unique(bad[:, 1]).cpu().tolist()} "
                    f"column_matches={column_matches} "
                    f"output_sample={accumulated[:8, :8].cpu().tolist()} "
                    f"expected_sample={expected_slice[:8, :8].cpu().tolist()} "
                    f"metadata={metadata.cpu().tolist()}"
                )
            torch.testing.assert_close(accumulated, expected_slice, rtol=0, atol=0)

    selector = torch.arange(16, device="cuda")
    for selector_base in range(0, 128, LOGICAL_K):
        p_selector = torch.zeros(
            (16, 128), device="cuda", dtype=torch.bfloat16
        )
        p_selector[selector, selector_base + selector] = 1.0
        p_selector = p_selector.to(torch.float8_e4m3fn)
        run_and_check(
            f"selector-{selector_base}",
            p_selector,
            decoded_v[selector_base : selector_base + LOGICAL_K],
            latent_bases=(0,),
        )

    p_values = torch.tensor(
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], device="cuda", dtype=torch.float32
    )
    query_ids = torch.arange(16, device="cuda", dtype=torch.int32)[:, None]
    token_ids = torch.arange(128, device="cuda", dtype=torch.int32)[None, :]
    p_indices = (query_ids * 17 + token_ids * 5 + 1) % p_values.numel()
    token_scales = torch.tensor(
        [0.5, 1.0, 2.0], device="cuda", dtype=torch.float32
    )[token_ids.remainder(3)]
    p_dense = (p_values[p_indices] * token_scales).to(torch.float8_e4m3fn)
    run_and_check("dense", p_dense, p_dense.float() @ decoded_v)
    acc_cols, a_bytes, a_cosize = metadata.cpu().tolist()
    print(
        "PASS mixed_e2m1_v_fp8_p_n16=True latent_slices=4 "
        "selector=True dense=True boundary_correction=True cta_group=1 "
        f"acc_cols={acc_cols} a_bytes={a_bytes} a_cosize={a_cosize}"
    )


if __name__ == "__main__":
    main()
