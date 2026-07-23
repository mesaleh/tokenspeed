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

"""Probe mixed native NVFP4 latent and FP8 RoPE accumulation on SM100.

Kimi MLA scores combine a 512-wide latent product with a 64-wide RoPE product.
The intended compressed path uses native block-scaled FP4 MMA for the rotated
latent cache while retaining RoPE at higher precision. This probe establishes
whether block-scaled FP4 and ordinary FP8 UMMA operations can target the same
TMEM accumulator. The FP4 instruction uses its required M=256 geometry, while
the FP8 operation and observable score view retain TokenSpeed's M=128 geometry
over the compatible first half of that accumulator. It is a legality probe,
not an attention benchmark.
"""

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
CLUSTER_SHAPE_MNK = (2, 1, 1)
# A native FP4 mainloop tile spans 256 latent coordinates. Kimi's 512-wide
# latent therefore uses two pipeline iterations with this same scale layout.
NVFP4_TILER_MNK = (256, 128, 256)
ROPE_TILER_MNK = (128, 128, 128)
SF_VEC_SIZE = 16


@cute.struct
class SharedStorage:
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


def make_tiled_mmas():
    nvfp4 = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float8E4M3FN,
        SF_VEC_SIZE,
        tcgen05.CtaGroup.TWO,
        NVFP4_TILER_MNK[:2],
    )
    rope = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        ROPE_TILER_MNK[:2],
    )
    return nvfp4, rope


@cute.kernel
def mixed_accumulate_kernel(
    output: cute.Tensor,
    metadata: cute.Tensor,
    nvfp4_mma: cute.TiledMma,
    rope_mma: cute.TiledMma,
    nvfp4_a_layout: cute.ComposedLayout,
    nvfp4_b_layout: cute.ComposedLayout,
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
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    is_leader_cta = cta_rank == 0

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    s_nvfp4_a = smem.allocate_tensor(
        cutlass.Float4E2M1FN,
        nvfp4_a_layout.outer,
        byte_alignment=128,
        swizzle=nvfp4_a_layout.inner,
    )
    s_nvfp4_b = smem.allocate_tensor(
        cutlass.Float4E2M1FN,
        nvfp4_b_layout.outer,
        byte_alignment=128,
        swizzle=nvfp4_b_layout.inner,
    )
    s_sfa = smem.allocate_tensor(cutlass.Float8E4M3FN, sfa_layout, byte_alignment=128)
    s_sfb = smem.allocate_tensor(cutlass.Float8E4M3FN, sfb_layout, byte_alignment=128)
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

    # E2M1 code 0x2 represents +1. Pack two +1 values per byte and use unit
    # scale factors. The native latent product is 256 and the FP8 RoPE product
    # is 128, so a successful mixed accumulation produces 384 in every score.
    nvfp4_a_bytes = cute.size_in_bytes(cutlass.Float4E2M1FN, s_nvfp4_a)
    nvfp4_b_bytes = cute.size_in_bytes(cutlass.Float4E2M1FN, s_nvfp4_b)
    nvfp4_a_ptr = cute.recast_ptr(s_nvfp4_a.iterator, dtype=cutlass.Uint8)
    nvfp4_b_ptr = cute.recast_ptr(s_nvfp4_b.iterator, dtype=cutlass.Uint8)
    for byte_idx in cutlass.range(tidx, nvfp4_a_bytes, THREADS):
        (nvfp4_a_ptr + byte_idx).store(cutlass.Uint8(0x22))
    for byte_idx in cutlass.range(tidx, nvfp4_b_bytes, THREADS):
        (nvfp4_b_ptr + byte_idx).store(cutlass.Uint8(0x22))

    sfa_ptr = cute.recast_ptr(s_sfa.iterator, dtype=cutlass.Float8E4M3FN)
    sfb_ptr = cute.recast_ptr(s_sfb.iterator, dtype=cutlass.Float8E4M3FN)
    for element in cutlass.range(tidx, cute.cosize(sfa_layout), THREADS):
        (sfa_ptr + element).store(cutlass.Float8E4M3FN(1.0))
    for element in cutlass.range(tidx, cute.cosize(sfb_layout), THREADS):
        (sfb_ptr + element).store(cutlass.Float8E4M3FN(1.0))

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
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS * 2),
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
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

    nvfp4_a = nvfp4_mma.make_fragment_A(s_nvfp4_a)
    nvfp4_b = nvfp4_mma.make_fragment_B(s_nvfp4_b)
    acc_shape = nvfp4_mma.partition_shape_C(NVFP4_TILER_MNK[:2])
    acc_fake = nvfp4_mma.make_fragment_C(acc_shape)
    acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
    rope_acc_shape = rope_mma.partition_shape_C(ROPE_TILER_MNK[:2])
    rope_acc_fake = rope_mma.make_fragment_C(rope_acc_shape)
    rope_acc = cute.make_tensor(tmem_ptr, rope_acc_fake.layout)
    rope_a = rope_mma.make_fragment_A(s_rope_a)
    rope_b = rope_mma.make_fragment_B(s_rope_b)

    sfa_tmem_ptr = cute.recast_ptr(tmem_ptr + acc_cols, dtype=cutlass.Float8E4M3FN)
    t_sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
        nvfp4_mma,
        NVFP4_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfa_layout, (None, None, None, 0)),
    )
    t_sfa = cute.make_tensor(sfa_tmem_ptr, t_sfa_layout)
    sfb_tmem_ptr = cute.recast_ptr(
        tmem_ptr + acc_cols + sfa_cols, dtype=cutlass.Float8E4M3FN
    )
    t_sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
        nvfp4_mma,
        NVFP4_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfb_layout, (None, None, None, 0)),
    )
    t_sfb = cute.make_tensor(sfb_tmem_ptr, t_sfb_layout)

    if warp_idx == 0 and is_leader_cta:
        scale_copy_atom = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO),
            cutlass.Float8E4M3FN,
        )
        sfa_copy = tcgen05.make_s2t_copy(scale_copy_atom, cute.filter_zeros(t_sfa))
        sfa_thr = sfa_copy.get_slice(0)
        sfa_source = tcgen05.get_s2t_smem_desc_tensor(
            sfa_copy, sfa_thr.partition_S(cute.filter_zeros(s_sfa))
        )
        cute.copy(
            sfa_copy,
            sfa_source[None, None, None, None, 0],
            sfa_thr.partition_D(cute.filter_zeros(t_sfa)),
        )
        sfb_copy = tcgen05.make_s2t_copy(scale_copy_atom, cute.filter_zeros(t_sfb))
        sfb_thr = sfb_copy.get_slice(0)
        sfb_source = tcgen05.get_s2t_smem_desc_tensor(
            sfb_copy, sfb_thr.partition_S(cute.filter_zeros(s_sfb))
        )
        cute.copy(
            sfb_copy,
            sfb_source[None, None, None, None, 0],
            sfb_thr.partition_D(cute.filter_zeros(t_sfb)),
        )

        mma_producer.acquire_and_advance()
        nvfp4_mma.set(tcgen05.Field.ACCUMULATE, False)
        nvfp4_k_blocks = cute.size(nvfp4_a, mode=[2])
        for k_block in cutlass.range(nvfp4_k_blocks, unroll_full=True):
            nvfp4_mma.set(tcgen05.Field.SFA, t_sfa[None, None, k_block].iterator)
            nvfp4_mma.set(tcgen05.Field.SFB, t_sfb[None, None, k_block].iterator)
            cute.gemm(
                nvfp4_mma,
                acc,
                nvfp4_a[None, None, k_block, 0],
                nvfp4_b[None, None, k_block, 0],
                acc,
            )
            nvfp4_mma.set(tcgen05.Field.ACCUMULATE, True)

        rope_mma.set(tcgen05.Field.ACCUMULATE, True)
        rope_k_blocks = cute.size(rope_a, mode=[2])
        for k_block in cutlass.range(rope_k_blocks, unroll_full=True):
            cute.gemm(
                rope_mma,
                rope_acc,
                rope_a[None, None, k_block, 0],
                rope_b[None, None, k_block, 0],
                rope_acc,
            )
        mma_producer.commit()

    mma_full = mma_consumer.wait_and_advance()
    mma_full.release()
    cute.arch.sync_threads()

    acc_tile = rope_acc[(None, None), 0, 0]
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
    )
    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, acc_tile)
    thr_load = tmem_load.get_slice(tidx)
    output_matrix = cute.make_tensor(
        output[cta_rank, None, None].iterator,
        cute.make_layout((64, 128), stride=(128, 1)),
    )
    t_tmem = thr_load.partition_S(acc_tile)
    t_gmem = thr_load.partition_D(output_matrix)
    registers = cute.make_fragment_like(t_gmem, cutlass.Float32)
    cute.copy(tmem_load, t_tmem, registers)
    cute.arch.fence_view_async_tmem_load()
    cute.autovec_copy(registers, t_gmem)
    cute.arch.sync_threads()

    if tidx == 0 and is_leader_cta:
        metadata[0] = acc_cols
        metadata[1] = sfa_cols
        metadata[2] = sfb_cols
        metadata[3] = total_cols

    if warp_idx == 0 and is_leader_cta:
        mma_producer.tail()
    tmem.relinquish_alloc_permit()
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def mixed_accumulate_probe(output: cute.Tensor, metadata: cute.Tensor):
    nvfp4_mma, rope_mma = make_tiled_mmas()
    nvfp4_a_layout = sm100_utils.make_smem_layout_a(
        nvfp4_mma, NVFP4_TILER_MNK, cutlass.Float4E2M1FN, 1
    )
    nvfp4_b_layout = sm100_utils.make_smem_layout_b(
        nvfp4_mma, NVFP4_TILER_MNK, cutlass.Float4E2M1FN, 1
    )
    sfa_layout = blockscaled_utils.make_smem_layout_sfa(
        nvfp4_mma, NVFP4_TILER_MNK, SF_VEC_SIZE, 1
    )
    sfb_layout = blockscaled_utils.make_smem_layout_sfb(
        nvfp4_mma, NVFP4_TILER_MNK, SF_VEC_SIZE, 1
    )
    rope_a_layout = sm100_utils.make_smem_layout_a(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    rope_b_layout = sm100_utils.make_smem_layout_b(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    acc_fake = nvfp4_mma.make_fragment_C(
        nvfp4_mma.partition_shape_C(NVFP4_TILER_MNK[:2])
    )
    acc_cols = utils.get_num_tmem_alloc_cols(acc_fake)
    sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
        nvfp4_mma,
        NVFP4_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfa_layout, (None, None, None, 0)),
    )
    sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
        nvfp4_mma,
        NVFP4_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfb_layout, (None, None, None, 0)),
    )
    sfa_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(cute.make_ptr(cutlass.Float8E4M3FN, 0), sfa_tmem_layout)
    )
    sfb_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(cute.make_ptr(cutlass.Float8E4M3FN, 0), sfb_tmem_layout)
    )
    total_cols = acc_cols + sfa_cols + sfb_cols
    if cutlass.const_expr(total_cols >= 512):
        raise ValueError(
            f"TMEM overflow: acc={acc_cols}, sfa={sfa_cols}, "
            f"sfb={sfb_cols}, total={total_cols}"
        )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (nvfp4_mma.thr_id.shape,)
    )
    mixed_accumulate_kernel(
        output,
        metadata,
        nvfp4_mma,
        rope_mma,
        nvfp4_a_layout,
        nvfp4_b_layout,
        sfa_layout,
        sfb_layout,
        rope_a_layout,
        rope_b_layout,
        acc_cols,
        sfa_cols,
        sfb_cols,
        total_cols,
        cta_layout_vmnk,
    ).launch(
        grid=CLUSTER_SHAPE_MNK,
        block=(THREADS, 1, 1),
        cluster=CLUSTER_SHAPE_MNK,
        min_blocks_per_mp=1,
    )


def main() -> None:
    compiled = cute.compile(
        mixed_accumulate_probe,
        make_fake_compact_tensor(
            cutlass.Float32,
            (2, 64, 128),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.Int32,
            (4,),
            stride_order=(0,),
            assumed_align=4,
        ),
        options="--enable-tvm-ffi --opt-level 3",
    )
    output = torch.empty((2, 64, 128), device="cuda", dtype=torch.float32)
    metadata = torch.empty(4, device="cuda", dtype=torch.int32)
    compiled(output, metadata)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output,
        torch.full_like(output, 384.0),
        rtol=0,
        atol=0,
    )
    acc_cols, sfa_cols, sfb_cols, total_cols = metadata.cpu().tolist()
    print(
        "PASS mixed_nvfp4_latent_fp8_rope_accumulate=True cta_group=2 "
        f"acc_cols={acc_cols} sfa_cols={sfa_cols} sfb_cols={sfb_cols} "
        f"total_cols={total_cols}"
    )


if __name__ == "__main__":
    main()
