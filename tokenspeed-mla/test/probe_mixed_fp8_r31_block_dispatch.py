# Copyright (c) 2026 LightSeek Foundation

"""F0: prove one SM100 kernel can dispatch FP8 or R31 MMA per CTA.

This is a type-system and generated-control-flow probe, not an attention
benchmark.  Block 0 executes the production M64 FP8-by-FP8 score operator over
unit operands.  Block 1 executes the production M64 FP8-by-E2M1 score operator
over a unit query and a zero packed operand.  Both paths target the same TMEM
accumulator layout, but only one path is selected by each block.
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
GRID = (2, 1, 1)
TILER_MNK = (64, 128, 128)


@cute.struct
class SharedStorage:
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


def make_tiled_mmas() -> tuple[cute.TiledMma, cute.TiledMma]:
    fp8_mma = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        TILER_MNK[:2],
    )
    r31_mma = cute.make_tiled_mma(
        tcgen05.MmaF8F6F4Op(
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
            cutlass.Float32,
            (64, 128, 32),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
    )
    return fp8_mma, r31_mma


@cute.kernel
def mixed_dispatch_kernel(
    output: cute.Tensor,
    metadata: cute.Tensor,
    fp8_mma: cute.TiledMma,
    r31_mma: cute.TiledMma,
    fp8_a_layout: cute.ComposedLayout,
    fp8_b_layout: cute.ComposedLayout,
    r31_a_layout: cute.ComposedLayout,
    r31_b_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    owner = cute.arch.make_warp_uniform(cute.arch.block_idx()[0])

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    s_fp8_a = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        fp8_a_layout.outer,
        byte_alignment=128,
        swizzle=fp8_a_layout.inner,
    )
    s_fp8_b = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        fp8_b_layout.outer,
        byte_alignment=128,
        swizzle=fp8_b_layout.inner,
    )
    s_r31_a = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        r31_a_layout.outer,
        byte_alignment=128,
        swizzle=r31_a_layout.inner,
    )
    # Match the production R31 mainloop: TMA expands packed E2M1 into an Int8
    # physical shared-memory carrier consumed by MmaF8F6F4Op.
    s_r31_b = smem.allocate_tensor(
        cutlass.Int8,
        r31_b_layout.outer,
        byte_alignment=128,
        swizzle=r31_b_layout.inner,
    )

    fp8_a_ptr = cute.recast_ptr(s_fp8_a.iterator, dtype=cutlass.Float8E4M3FN)
    fp8_b_ptr = cute.recast_ptr(s_fp8_b.iterator, dtype=cutlass.Float8E4M3FN)
    r31_a_ptr = cute.recast_ptr(s_r31_a.iterator, dtype=cutlass.Float8E4M3FN)
    r31_b_ptr = cute.recast_ptr(s_r31_b.iterator, dtype=cutlass.Uint8)
    for element in cutlass.range(tidx, cute.cosize(fp8_a_layout.outer), THREADS):
        (fp8_a_ptr + element).store(cutlass.Float8E4M3FN(1.0))
    for element in cutlass.range(tidx, cute.cosize(fp8_b_layout.outer), THREADS):
        (fp8_b_ptr + element).store(cutlass.Float8E4M3FN(1.0))
    for element in cutlass.range(tidx, cute.cosize(r31_a_layout.outer), THREADS):
        (r31_a_ptr + element).store(cutlass.Float8E4M3FN(1.0))
    r31_b_bytes = cute.size_in_bytes(cutlass.Int8, s_r31_b)
    for byte_index in cutlass.range(tidx, r31_b_bytes, THREADS):
        (r31_b_ptr + byte_index).store(cutlass.Uint8(0))
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
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

    fp8_a = fp8_mma.make_fragment_A(s_fp8_a)
    fp8_b = fp8_mma.make_fragment_B(s_fp8_b)
    r31_a = r31_mma.make_fragment_A(s_r31_a)
    r31_b = r31_mma.make_fragment_B(s_r31_b)
    acc_shape = fp8_mma.partition_shape_C(TILER_MNK[:2])
    acc_fake = fp8_mma.make_fragment_C(acc_shape)
    acc = cute.make_tensor(tmem_ptr, acc_fake.layout)

    if warp_idx == 0:
        mma_producer.acquire_and_advance()
        if owner == 0:
            fp8_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(
                cute.size(fp8_a, mode=[2]), unroll_full=True
            ):
                cute.gemm(
                    fp8_mma,
                    acc,
                    fp8_a[None, None, k_block, 0],
                    fp8_b[None, None, k_block, 0],
                    acc,
                )
                fp8_mma.set(tcgen05.Field.ACCUMULATE, True)
        else:
            r31_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(
                cute.size(r31_a, mode=[2]), unroll_full=True
            ):
                cute.gemm(
                    r31_mma,
                    acc,
                    r31_a[None, None, k_block, 0],
                    r31_b[None, None, k_block, 0],
                    acc,
                )
                r31_mma.set(tcgen05.Field.ACCUMULATE, True)
        mma_producer.commit()

    mma_full = mma_consumer.wait_and_advance()
    mma_full.release()
    cute.arch.sync_threads()

    acc_tile = acc[(None, None), 0, 0]
    tmem_load_atom = cute.make_copy_atom(
        # The raw M64 score accumulator exposes a (...,16,2) TMEM V layout.
        # Load its paired columns directly; the Ld32 form expects a flattened
        # 32-column V layout that only the production WS epilogue synthesizes.
        tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(16)), cutlass.Float32
    )
    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, acc_tile)
    thread_load = tmem_load.get_slice(tidx)
    output_matrix = cute.make_tensor(
        output[owner, None, None].iterator,
        cute.make_layout((64, 128), stride=(128, 1)),
    )
    t_tmem = thread_load.partition_S(acc_tile)
    t_gmem = thread_load.partition_D(output_matrix)
    registers = cute.make_fragment_like(t_gmem, cutlass.Float32)
    cute.copy(tmem_load, t_tmem, registers)
    cute.arch.fence_view_async_tmem_load()
    cute.autovec_copy(registers, t_gmem)
    cute.arch.sync_threads()

    if tidx == 0:
        metadata[owner] = owner
    if warp_idx == 0:
        mma_producer.tail()
    tmem.relinquish_alloc_permit()
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def mixed_dispatch_probe(output: cute.Tensor, metadata: cute.Tensor):
    fp8_mma, r31_mma = make_tiled_mmas()
    fp8_a_layout = sm100_utils.make_smem_layout_a(
        fp8_mma, TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    fp8_b_layout = sm100_utils.make_smem_layout_b(
        fp8_mma, TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    r31_a_layout = sm100_utils.make_smem_layout_a(
        r31_mma, TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    r31_b_layout = sm100_utils.make_smem_layout_b(
        r31_mma, TILER_MNK, cutlass.Int8, 1
    )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)), (fp8_mma.thr_id.shape,)
    )
    mixed_dispatch_kernel(
        output,
        metadata,
        fp8_mma,
        r31_mma,
        fp8_a_layout,
        fp8_b_layout,
        r31_a_layout,
        r31_b_layout,
        cta_layout_vmnk,
    ).launch(
        grid=GRID,
        block=(THREADS, 1, 1),
        cluster=(1, 1, 1),
        min_blocks_per_mp=1,
    )


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("mixed FP8/R31 dispatch probe requires SM100")
    compiled = cute.compile(
        mixed_dispatch_probe,
        make_fake_compact_tensor(
            cutlass.Float32,
            (2, 64, 128),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.Int32,
            (2,),
            stride_order=(0,),
            assumed_align=4,
        ),
        options="--enable-tvm-ffi --opt-level 3",
    )
    output = torch.empty((2, 64, 128), device="cuda", dtype=torch.float32)
    metadata = torch.full((2,), -1, device="cuda", dtype=torch.int32)
    compiled(output, metadata)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output[0], torch.full_like(output[0], 128.0), rtol=0, atol=0
    )
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]), rtol=0, atol=0)
    if metadata.cpu().tolist() != [0, 1]:
        raise AssertionError(f"owner dispatch metadata mismatch: {metadata.cpu().tolist()}")
    print(
        "PASS mixed_fp8_r31_block_dispatch=True blocks=2 "
        "fp8_expected=128 r31_expected=0"
    )


if __name__ == "__main__":
    main()
