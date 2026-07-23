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

"""Probe reversed E2M1-V by FP8-P MMA with a production output view on SM100.

The no-shadow TurboQuant PV path folds each token's BF16 cache scale into P,
casts the scaled probabilities to FP8, and multiplies a bounded transpose of
the packed E2M1 V tile by that FP8 operand. The native mixed instruction has a
deterministic final-row boundary defect, so the consumer recomputes only that
row from canonical packed bytes and P. This is a legality probe, not an
attention benchmark.
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
CLUSTER_SHAPE_MNK = (1, 1, 1)
E2M1_PV_TILER_MNK = (128, 64, 128)
OUTPUT_VIEW_TILER_MNK = (128, 64, 128)


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
    packed_boundary: cute.Tensor,
    codebook: cute.Tensor,
    p_input: cute.Tensor,
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
        byte_alignment=128,
        swizzle=e2m1_a_layout.inner,
    )
    s_p = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        p_b_layout.outer,
        byte_alignment=128,
        swizzle=p_b_layout.inner,
    )

    # E2M1 code 0x2 and FP8 value 1 both represent +1. The K=128 product must
    # therefore be exactly 128 in every useful output element.
    e2m1_bytes = cute.size_in_bytes(cutlass.Float4E2M1FN, s_e2m1_v)
    e2m1_ptr = cute.recast_ptr(s_e2m1_v.iterator, dtype=cutlass.Uint8)
    for byte_idx in cutlass.range(tidx, e2m1_bytes, THREADS):
        (e2m1_ptr + byte_idx).store(cutlass.Uint8(0x22))
    p_ptr = cute.recast_ptr(s_p.iterator, dtype=cutlass.Float8E4M3FN)
    for element in cutlass.range(tidx, cute.cosize(p_b_layout.outer), THREADS):
        (p_ptr + element).store(cutlass.Float8E4M3FN(1.0))
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
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), cutlass.Float32
    )
    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, output_tile)
    thr_load = tmem_load.get_slice(tidx)
    output_matrix = cute.make_tensor(
        output[None, None].iterator,
        cute.make_layout((128, 64), stride=(64, 1)),
    )
    t_tmem = thr_load.partition_S(output_tile)
    t_gmem = thr_load.partition_D(output_matrix)
    coordinates = cute.make_identity_tensor((128, 64))
    t_coordinates = thr_load.partition_D(coordinates)
    registers = cute.make_fragment_like(t_gmem, cutlass.Float32)
    cute.copy(tmem_load, t_tmem, registers)
    cute.arch.fence_view_async_tmem_load()
    packed_ptr = cute.recast_ptr(packed_boundary.iterator, dtype=cutlass.Uint8)
    codebook_bf16 = cute.make_tensor(
        cute.recast_ptr(codebook.iterator, dtype=cutlass.BFloat16),
        codebook.layout,
    )
    for element in cutlass.range_constexpr(cute.size(registers)):
        latent = t_coordinates[element][0]
        query = t_coordinates[element][1]
        if latent == 127:
            correction = cutlass.Float32(0.0)
            for token in cutlass.range(128):
                packed = cutlass.Int32((packed_ptr + token // 2).load())
                code = (packed >> ((token % 2) * 4)) & 0xF
                correction += cutlass.Float32(codebook_bf16[code]) * cutlass.Float32(
                    p_input[query, token]
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
    packed_boundary: cute.Tensor,
    codebook: cute.Tensor,
    p_input: cute.Tensor,
):
    e2m1_pv_mma, output_view_mma = make_tiled_mmas()
    e2m1_a_layout = sm100_utils.make_smem_layout_a(
        e2m1_pv_mma, E2M1_PV_TILER_MNK, cutlass.Float4E2M1FN, 1
    )
    p_b_layout = sm100_utils.make_smem_layout_b(
        e2m1_pv_mma, E2M1_PV_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
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
        packed_boundary,
        codebook,
        p_input,
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
            (128, 64),
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
            (64,),
            stride_order=(0,),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (16,),
            stride_order=(0,),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (64, 128),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        options="--enable-tvm-ffi --opt-level 3",
    )
    output = torch.empty((128, 64), device="cuda", dtype=torch.float32)
    metadata = torch.empty(3, device="cuda", dtype=torch.int32)
    packed_boundary = torch.full((64,), 0x22, device="cuda", dtype=torch.uint8)
    codebook = torch.zeros(16, device="cuda", dtype=torch.bfloat16)
    codebook[2] = 1.0
    p_input = (
        torch.where(
            torch.arange(128, device="cuda") % 2 == 0,
            torch.tensor(1.0, device="cuda"),
            torch.tensor(2.0, device="cuda"),
        )[None, :]
        .expand(64, 128)
        .contiguous()
        .to(torch.bfloat16)
    )
    compiled(output, metadata, packed_boundary, codebook, p_input)
    torch.cuda.synchronize()
    expected = torch.full_like(output, 128.0)
    expected[127, :] = 192.0
    mismatch = output != expected
    if mismatch.any():
        bad = mismatch.nonzero()
        print(
            "DIAGNOSTIC "
            f"unique={torch.unique(output).cpu().tolist()} "
            f"bad_rows={torch.unique(bad[:, 0]).cpu().tolist()} "
            f"bad_columns={torch.unique(bad[:, 1]).cpu().tolist()} "
            f"metadata={metadata.cpu().tolist()}"
        )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    acc_cols, a_bytes, a_cosize = metadata.cpu().tolist()
    print(
        "PASS mixed_e2m1_v_fp8_p_boundary_correction=True cta_group=1 "
        f"acc_cols={acc_cols} a_bytes={a_bytes} a_cosize={a_cosize}"
    )


if __name__ == "__main__":
    main()
