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

"""Probe mixed native E2M1 latent and FP8 RoPE accumulation on SM100.

Kimi MLA scores combine a 512-wide latent product with a 64-wide RoPE product.
The intended compressed path uses native mixed FP8-by-E2M1 MMA for the rotated
latent cache while retaining RoPE at higher precision. The mixed operation uses
eight padded N columns to absorb its final-column boundary quirk, then exposes
the first 128 columns through TokenSpeed's ordinary FP8 score view. The latent
and RoPE products remain separate until the consumer applies TurboQuant's BF16
per-token scale to the latent score and adds RoPE in registers. It is a legality
probe, not an attention benchmark.
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
# The native mixed instruction advances K by 32; the probe covers Kimi's full
# 512-wide latent dimension. Eight padded N columns absorb the mixed-F4
# boundary quirk while the production-compatible score view remains M128xN128.
E2M1_TILER_MNK = (128, 136, 512)
ROPE_TILER_MNK = (128, 128, 128)


@cute.struct
class SharedStorage:
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


def make_tiled_mmas():
    e2m1 = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        cutlass.Float4E2M1FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        E2M1_TILER_MNK[:2],
    )
    rope = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        ROPE_TILER_MNK[:2],
    )
    return e2m1, rope


@cute.kernel
def mixed_accumulate_kernel(
    output: cute.Tensor,
    metadata: cute.Tensor,
    token_scale: cute.Tensor,
    e2m1_mma: cute.TiledMma,
    rope_mma: cute.TiledMma,
    e2m1_a_layout: cute.ComposedLayout,
    e2m1_b_layout: cute.ComposedLayout,
    rope_a_layout: cute.ComposedLayout,
    rope_b_layout: cute.ComposedLayout,
    acc_cols: cutlass.Constexpr,
    rope_cols: cutlass.Constexpr,
    total_cols: cutlass.Constexpr,
    cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    is_leader_cta = cta_rank == 0

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    s_e2m1_a = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        e2m1_a_layout.outer,
        byte_alignment=128,
        swizzle=e2m1_a_layout.inner,
    )
    s_e2m1_b = smem.allocate_tensor(
        cutlass.Float4E2M1FN,
        e2m1_b_layout.outer,
        byte_alignment=128,
        swizzle=e2m1_b_layout.inner,
    )
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

    # E2M1 code 0x2 represents +1. Pack two B values per byte, fill FP8 A with
    # +1. The mixed latent product is 512 and the FP8 RoPE product is 128, so
    # every observable score must equal 640. MmaF8F6F4Op consumes E2M1 values
    # directly; the production kernel applies TurboQuant's BF16 token scale to
    # the raw score before softmax rather than using hardware MX scale factors.
    e2m1_a_ptr = cute.recast_ptr(s_e2m1_a.iterator, dtype=cutlass.Float8E4M3FN)
    e2m1_b_bytes = cute.size_in_bytes(cutlass.Float4E2M1FN, s_e2m1_b)
    e2m1_b_ptr = cute.recast_ptr(s_e2m1_b.iterator, dtype=cutlass.Uint8)
    for element in cutlass.range(tidx, cute.cosize(e2m1_a_layout.outer), THREADS):
        (e2m1_a_ptr + element).store(cutlass.Float8E4M3FN(1.0))
    for byte_idx in cutlass.range(tidx, e2m1_b_bytes, THREADS):
        (e2m1_b_ptr + byte_idx).store(cutlass.Uint8(0x22))

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
    # The allocator requires a power-of-two column count, so the 384-column
    # latent-plus-RoPE working set rounds to the full 512-column allocation.
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

    e2m1_a = e2m1_mma.make_fragment_A(s_e2m1_a)
    e2m1_b = e2m1_mma.make_fragment_B(s_e2m1_b)
    acc_shape = e2m1_mma.partition_shape_C(E2M1_TILER_MNK[:2])
    acc_fake = e2m1_mma.make_fragment_C(acc_shape)
    acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
    rope_acc_shape = rope_mma.partition_shape_C(ROPE_TILER_MNK[:2])
    rope_acc_fake = rope_mma.make_fragment_C(rope_acc_shape)
    latent_view = cute.make_tensor(tmem_ptr, rope_acc_fake.layout)
    rope_acc = cute.make_tensor(tmem_ptr + acc_cols, rope_acc_fake.layout)
    rope_a = rope_mma.make_fragment_A(s_rope_a)
    rope_b = rope_mma.make_fragment_B(s_rope_b)

    if warp_idx == 0 and is_leader_cta:
        mma_producer.acquire_and_advance()
        e2m1_mma.set(tcgen05.Field.ACCUMULATE, False)
        e2m1_k_blocks = cute.size(e2m1_a, mode=[2])
        for k_block in cutlass.range(e2m1_k_blocks, unroll_full=True):
            cute.gemm(
                e2m1_mma,
                acc,
                e2m1_a[None, None, k_block, 0],
                e2m1_b[None, None, k_block, 0],
                acc,
            )
            e2m1_mma.set(tcgen05.Field.ACCUMULATE, True)

        rope_mma.set(tcgen05.Field.ACCUMULATE, False)
        rope_k_blocks = cute.size(rope_a, mode=[2])
        for k_block in cutlass.range(rope_k_blocks, unroll_full=True):
            cute.gemm(
                rope_mma,
                rope_acc,
                rope_a[None, None, k_block, 0],
                rope_b[None, None, k_block, 0],
                rope_acc,
            )
            rope_mma.set(tcgen05.Field.ACCUMULATE, True)
        mma_producer.commit()

    mma_full = mma_consumer.wait_and_advance()
    mma_full.release()
    cute.arch.sync_threads()

    latent_tile = latent_view[(None, None), 0, 0]
    rope_tile = rope_acc[(None, None), 0, 0]
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
    )
    latent_load = tcgen05.make_tmem_copy(tmem_load_atom, latent_tile)
    rope_load = tcgen05.make_tmem_copy(tmem_load_atom, rope_tile)
    latent_thr = latent_load.get_slice(tidx)
    rope_thr = rope_load.get_slice(tidx)
    output_matrix = cute.make_tensor(
        output[cta_rank, None, None].iterator,
        cute.make_layout((128, 128), stride=(128, 1)),
    )
    coordinates = cute.make_identity_tensor((128, 128))
    t_latent = latent_thr.partition_S(latent_tile)
    t_rope = rope_thr.partition_S(rope_tile)
    t_gmem = latent_thr.partition_D(output_matrix)
    t_coordinates = latent_thr.partition_D(coordinates)
    latent_registers = cute.make_fragment_like(t_gmem, cutlass.Float32)
    rope_registers = cute.make_fragment_like(t_gmem, cutlass.Float32)
    cute.copy(latent_load, t_latent, latent_registers)
    cute.copy(rope_load, t_rope, rope_registers)
    cute.arch.fence_view_async_tmem_load()
    for element in cutlass.range_constexpr(cute.size(latent_registers)):
        token = t_coordinates[element][1]
        latent_registers[element] = (
            latent_registers[element] * cutlass.Float32(token_scale[token])
            + rope_registers[element]
        )
    cute.autovec_copy(latent_registers, t_gmem)
    cute.arch.sync_threads()

    if tidx == 0 and is_leader_cta:
        metadata[0] = acc_cols
        metadata[1] = rope_cols
        metadata[2] = total_cols
        metadata[3] = e2m1_b_bytes

    if warp_idx == 0 and is_leader_cta:
        mma_producer.tail()
    tmem.relinquish_alloc_permit()
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def mixed_accumulate_probe(
    output: cute.Tensor, metadata: cute.Tensor, token_scale: cute.Tensor
):
    e2m1_mma, rope_mma = make_tiled_mmas()
    e2m1_a_layout = sm100_utils.make_smem_layout_a(
        e2m1_mma, E2M1_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    e2m1_b_layout = sm100_utils.make_smem_layout_b(
        e2m1_mma, E2M1_TILER_MNK, cutlass.Float4E2M1FN, 1
    )
    rope_a_layout = sm100_utils.make_smem_layout_a(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    rope_b_layout = sm100_utils.make_smem_layout_b(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    acc_fake = e2m1_mma.make_fragment_C(e2m1_mma.partition_shape_C(E2M1_TILER_MNK[:2]))
    acc_cols = utils.get_num_tmem_alloc_cols(acc_fake)
    rope_fake = rope_mma.make_fragment_C(rope_mma.partition_shape_C(ROPE_TILER_MNK[:2]))
    rope_cols = utils.get_num_tmem_alloc_cols(rope_fake)
    total_cols = acc_cols + rope_cols
    if cutlass.const_expr(total_cols > 512):
        raise ValueError(
            f"TMEM overflow: latent={acc_cols}, rope={rope_cols}, total={total_cols}"
        )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (e2m1_mma.thr_id.shape,)
    )
    mixed_accumulate_kernel(
        output,
        metadata,
        token_scale,
        e2m1_mma,
        rope_mma,
        e2m1_a_layout,
        e2m1_b_layout,
        rope_a_layout,
        rope_b_layout,
        acc_cols,
        rope_cols,
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
            (1, 128, 128),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.Int32,
            (4,),
            stride_order=(0,),
            assumed_align=4,
        ),
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (128,),
            stride_order=(0,),
            assumed_align=16,
        ),
        options="--enable-tvm-ffi --opt-level 3",
    )
    output = torch.empty((1, 128, 128), device="cuda", dtype=torch.float32)
    metadata = torch.empty(4, device="cuda", dtype=torch.int32)
    token_scale = torch.where(
        torch.arange(128, device="cuda") % 2 == 0,
        torch.tensor(1.0, device="cuda"),
        torch.tensor(2.0, device="cuda"),
    ).to(torch.bfloat16)
    compiled(output, metadata, token_scale)
    torch.cuda.synchronize()
    expected = (512.0 * token_scale.float()[None, None, :] + 128.0).expand_as(output)
    mismatch = output != expected
    if mismatch.any():
        bad = mismatch.nonzero()
        print(
            "DIAGNOSTIC "
            f"unique={torch.unique(output).cpu().tolist()} "
            f"bad_columns={torch.unique(bad[:, 2]).cpu().tolist()} "
            f"bad_count_by_cta={mismatch.sum(dim=(1, 2)).cpu().tolist()} "
            f"metadata={metadata.cpu().tolist()} "
            f"samples={output[0, [0, 63, 64, 127]][:, [0, 62, 63, 64, 126, 127]].cpu().tolist()}"
        )
    torch.testing.assert_close(
        output,
        expected,
        rtol=0,
        atol=0,
    )
    acc_cols, rope_cols, total_cols, b_bytes = metadata.cpu().tolist()
    print(
        "PASS mixed_e2m1_scaled_latent_fp8_rope_accumulate=True cta_group=1 "
        f"latent_cols={acc_cols} rope_cols={rope_cols} "
        f"total_cols={total_cols} b_bytes={b_bytes}"
    )


if __name__ == "__main__":
    main()
