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

"""Compile and launch a two-CTA QK MMA followed by a local one-CTA VP MMA.

This is a hardware/software legality probe, not an attention benchmark. It
isolates the mixed CTA-group transition needed by the packed-V transpose path:
the leading CTA issues a two-CTA ``128x128x128`` QK operation, then each cluster
member issues its own one-CTA ``128x64x128`` ``V * P^T`` operation under the
same two-CTA TMEM allocation. Separate UMMA barriers prove both operations have
completed before tensor memory is released.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor

THREADS_PER_CTA = 128
CLUSTER_SHAPE_MNK = (2, 1, 1)
QK_TILER_MNK = (128, 128, 128)
VP_TILER_MNK = (128, 64, 128)
CORRECTION_VALUES = 4
CORRECTION_STAGES = 2


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
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        VP_TILER_MNK[:2],
    )
    return qk, vp


@cute.kernel
def mixed_group_kernel(
    output: cute.Tensor,
    qk_tiled_mma: cute.TiledMma,
    vp_tiled_mma: cute.TiledMma,
    qk_a_layout: cute.ComposedLayout,
    qk_b_layout: cute.ComposedLayout,
    vp_a_layout: cute.ComposedLayout,
    vp_b_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
    qk_cols: cutlass.Constexpr,
    vp_cols: cutlass.Constexpr,
    score_cols: cutlass.Constexpr,
    output_cols: cutlass.Constexpr,
    correction_cols: cutlass.Constexpr,
    total_cols: cutlass.Constexpr,
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
    s_v = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        vp_a_layout.outer,
        byte_alignment=128,
        swizzle=vp_a_layout.inner,
    )
    s_p = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        vp_b_layout.outer,
        byte_alignment=128,
        swizzle=vp_b_layout.inner,
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

    vp_a = vp_tiled_mma.make_fragment_A(s_v)
    vp_b = vp_tiled_mma.make_fragment_B(s_p)
    vp_shape = vp_tiled_mma.partition_shape_C(VP_TILER_MNK[:2])
    vp_acc_fake = vp_tiled_mma.make_fragment_C(vp_shape)
    vp_acc = cute.make_tensor(tmem_ptr, vp_acc_fake.layout)

    cute.arch.sync_threads()
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

    if tidx == 0:
        output[cta_rank] = 1
        if is_leader_cta:
            output[2] = qk_cols
            output[3] = vp_cols
            output[4] = score_cols
            output[5] = output_cols
            output[6] = correction_cols
            output[7] = total_cols

    if warp_idx == 0:
        if is_leader_cta:
            qk_producer.tail()
        vp_producer.tail()

    tmem.relinquish_alloc_permit()
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def mixed_group_probe(output: cute.Tensor):
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
    total_cols = score_cols + output_cols + correction_cols
    if cutlass.const_expr(total_cols >= 512):
        raise ValueError(
            f"TMEM overlap: score={score_cols}, output={output_cols}, "
            f"correction={correction_cols}, limit=512"
        )
    qk_a_layout = sm100_utils.make_smem_layout_a(
        qk_tiled_mma, QK_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    qk_b_layout = sm100_utils.make_smem_layout_b(
        qk_tiled_mma, QK_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    vp_a_layout = sm100_utils.make_smem_layout_a(
        vp_tiled_mma, VP_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    vp_b_layout = sm100_utils.make_smem_layout_b(
        vp_tiled_mma, VP_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (qk_tiled_mma.thr_id.shape,)
    )
    mixed_group_kernel(
        output,
        qk_tiled_mma,
        vp_tiled_mma,
        qk_a_layout,
        qk_b_layout,
        vp_a_layout,
        vp_b_layout,
        cta_layout_vmnk,
        qk_cols,
        vp_cols,
        score_cols,
        output_cols,
        correction_cols,
        total_cols,
    ).launch(
        grid=CLUSTER_SHAPE_MNK,
        block=(THREADS_PER_CTA, 1, 1),
        cluster=CLUSTER_SHAPE_MNK,
        min_blocks_per_mp=1,
    )


def fake(dtype: type[cutlass.Numeric], shape: tuple[int, ...], align: int):
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )


def main() -> None:
    compiled = cute.compile(
        mixed_group_probe,
        fake(cutlass.Int32, (8,), 4),
        options="--enable-tvm-ffi --opt-level 3",
    )
    output = torch.zeros(8, device="cuda", dtype=torch.int32)
    compiled(output)
    torch.cuda.synchronize()
    result = output.cpu()
    torch.testing.assert_close(result[:2], torch.ones(2, dtype=torch.int32))
    qk_cols, vp_cols, score_cols, output_cols, correction_cols, total_cols = result[
        2:
    ].tolist()
    if total_cols >= 512:
        raise AssertionError(
            f"TMEM overlap: score={score_cols}, output={output_cols}, "
            f"correction={correction_cols}, limit=512"
        )
    print(
        f"PASS qk_group=2 vp_group=1 qk_cols={qk_cols} "
        f"vp_cols={vp_cols} score_cols={score_cols} output_cols={output_cols} "
        f"correction_cols={correction_cols} total_cols={total_cols}"
    )


if __name__ == "__main__":
    main()
