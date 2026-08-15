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

"""Compile-gate the no-shadow SM100 compact-QK/transposed-PV resource map.

This is the first A17-N8-R1-C1-M0qP falsifier.  It does not implement or time
attention.  It combines the exact one-CTA mixed-FP8xFP4 QK and one-CTA
transposed ordinary-FP8 PV fragment types in a two-CTA, 384-thread launch,
constructs legal TensorMaps for the aligned latent and RoPE page planes, and
materializes the complete candidate SMEM/TMEM envelope.  A failure here stops
the more expensive score-to-P ownership implementation.
"""

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

THREADS_PER_CTA = 384
TMEM_RETRIEVE_THREADS = 288
CLUSTER_SHAPE_MNK = (2, 1, 1)
PAGE_SIZE = 64
NUM_PAGES = 4

LATENT_K = 512
ROPE_K = 64
MIXED_TILER_MNK = (128, 128, 256)
VP_TILER_MNK = (128, 64, 128)
SF_VEC_SIZE = 32
SF_DTYPE = cutlass.Float8E8M0FNU

V_OFFSET = 0
SCALE_OFFSET = 64
P_COR_OFFSET = 80
SCORE_OFFSET = 128
OUTPUT_OFFSET = 256
TMEM_ALLOC_COLS = 512

Q_SMEM_BYTES = 128 * LATENT_K
Q_ROPE_SMEM_BYTES = 128 * ROPE_K
K_SMEM_BYTES = 128 * MIXED_TILER_MNK[2]
K_ROPE_SMEM_BYTES = 128 * ROPE_K
V_SMEM_BYTES = 2 * 16 * 1024
P_SMEM_BYTES = 2 * 8 * 1024
SMEM_PAYLOAD_BYTES = (
    Q_SMEM_BYTES
    + Q_ROPE_SMEM_BYTES
    + K_SMEM_BYTES
    + K_ROPE_SMEM_BYTES
    + V_SMEM_BYTES
    + P_SMEM_BYTES
)


@cute.struct
class SharedStorage:
    init_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


def make_tiled_mmas():
    mixed = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        cutlass.Float4E2M1FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        SF_DTYPE,
        SF_VEC_SIZE,
        tcgen05.CtaGroup.ONE,
        MIXED_TILER_MNK[:2],
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
    return mixed, vp


@cute.kernel
def layout_kernel(
    output: cute.Tensor,
    pv_latent_desc: cutlass.GridConstant[cuda.TensorMap],
    rope_desc: cutlass.GridConstant[cuda.TensorMap],
    qk_tma_atom: cute.CopyAtom,
    qk_tma_tensor: cute.Tensor,
    scale_pages: cute.Tensor,
    mixed_mma: cute.TiledMma,
    vp_mma: cute.TiledMma,
    mixed_acc_layout: cute.Layout,
    sfa_layout: cute.Layout,
    sfb_layout: cute.Layout,
    vp_a_layout: cute.Layout,
    vp_o_layout: cute.Layout,
    mixed_acc_cols: cutlass.Constexpr,
    sfa_cols: cutlass.Constexpr,
    sfb_cols: cutlass.Constexpr,
    vp_a_cols: cutlass.Constexpr,
    vp_o_cols: cutlass.Constexpr,
):
    del (
        pv_latent_desc,
        rope_desc,
        qk_tma_atom,
        qk_tma_tensor,
        scale_pages,
        mixed_mma,
        vp_mma,
    )

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # Materialize the reviewed one-stage compact-QK and two-stage compact-V/P
    # envelope.  These arrays are intentionally distinct so generated launch
    # SMEM, rather than arithmetic alone, is the decision evidence.
    q_smem = cutlass.Array(
        cutlass.Int8,
        Q_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    q_rope_smem = cutlass.Array(
        cutlass.Int8,
        Q_ROPE_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    k_smem = cutlass.Array(
        cutlass.Int8,
        K_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    k_rope_smem = cutlass.Array(
        cutlass.Int8,
        K_ROPE_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    v_smem = cutlass.Array(
        cutlass.Int8,
        V_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    p_smem = cutlass.Array(
        cutlass.Int8,
        P_SMEM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    retrieve_barrier = pipeline.NamedBarrier(
        barrier_id=1, num_threads=TMEM_RETRIEVE_THREADS
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=retrieve_barrier,
        allocator_warp_id=8,
        is_two_cta=False,
    )

    # Production attention initializes its pipeline mbarriers before TMEM
    # allocation.  Keep one real barrier in this otherwise compile-only probe
    # so sanitizer observes the same initialized shared-barrier environment.
    if tidx == 0:
        cute.arch.mbarrier_init(storage.init_mbar.ptr, 1)
    pipeline.pipeline_init_arrive(
        cluster_shape_mn=CLUSTER_SHAPE_MNK[:2], is_relaxed=True
    )
    pipeline.pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE_MNK[:2])

    # TmemAllocator elects the configured MMA warp to issue the hardware
    # request while maintaining the allocation size in the DSL object used by
    # the later deallocation.
    tmem.allocate(TMEM_ALLOC_COLS)

    if warp_idx <= 8:
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

        # Construct every exact candidate TMEM view at its frozen base.  The
        # scale layouts are byte-typed views; V is the TMEM-A E4M3 operand.
        v_operand = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + V_OFFSET, dtype=cutlass.Float8E4M3FN),
            vp_a_layout,
        )
        sfa = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + SCALE_OFFSET, dtype=SF_DTYPE), sfa_layout
        )
        sfb = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + SCALE_OFFSET + sfa_cols, dtype=SF_DTYPE),
            sfb_layout,
        )
        p_cor = cute.make_tensor(
            tmem_ptr + P_COR_OFFSET,
            cute.make_layout((128, 4, 2), stride=(1 << 16, 1, 4)),
        )
        score = cute.make_tensor(tmem_ptr + SCORE_OFFSET, mixed_acc_layout)
        output_staged = cute.make_tensor(tmem_ptr + OUTPUT_OFFSET, vp_o_layout)

        # Force all views and every SMEM allocation to remain visible in host
        # IR while avoiding a numerical claim in this compile-only gate.
        if tidx == 0:
            q_smem[0] = cutlass.Int8(0)
            q_rope_smem[0] = cutlass.Int8(0)
            k_smem[0] = cutlass.Int8(0)
            k_rope_smem[0] = cutlass.Int8(0)
            v_smem[0] = cutlass.Int8(0)
            p_smem[0] = cutlass.Int8(0)
            output[cta_rank, 0] = mixed_acc_cols
            output[cta_rank, 1] = sfa_cols
            output[cta_rank, 2] = sfb_cols
            output[cta_rank, 3] = vp_a_cols
            output[cta_rank, 4] = vp_o_cols
            output[cta_rank, 5] = V_OFFSET
            output[cta_rank, 6] = SCALE_OFFSET
            output[cta_rank, 7] = P_COR_OFFSET
            output[cta_rank, 8] = SCORE_OFFSET
            output[cta_rank, 9] = OUTPUT_OFFSET
            output[cta_rank, 10] = TMEM_ALLOC_COLS
            output[cta_rank, 11] = SMEM_PAYLOAD_BYTES

        # Referencing the views keeps their base/layout legalization in the
        # generated program even though this first gate issues no MMA.
        del v_operand, sfa, sfb, p_cor, score, output_staged

    # All support warps remain live until the 288 TMEM users have completed
    # their layout work.  A full-CTA rendezvous then makes the one-CTA TMEM
    # deallocation lifetime explicit without adding a second partial barrier.
    cute.arch.sync_threads()
    if warp_idx == 8:
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def layout_probe(
    output: cute.Tensor,
    latent_pages: cute.Tensor,
    scale_pages: cute.Tensor,
    rope_pages: cute.Tensor,
    stream,
):
    mixed_mma, vp_mma = make_tiled_mmas()

    # PV uses the raw narrow padded-source contract and consumes one 128-value
    # (64-byte packed) latent slice per TensorMap box.
    pv_latent_desc = cuda.create_tensor_map_tiled(
        latent_pages.iterator.toint(),
        cutlass.Float4E2M1FN,
        global_dims=[LATENT_K, PAGE_SIZE, NUM_PAGES],
        global_strides=[(LATENT_K // 2) // 16, PAGE_SIZE * (LATENT_K // 2) // 16],
        box_dims=[128, 32, 1],
        swizzle=cuda.TensorMapSwizzle.none,
    )
    rope_desc = cuda.create_tensor_map_tiled(
        rope_pages.iterator.toint(),
        cutlass.Float8E4M3FN,
        global_dims=[ROPE_K, PAGE_SIZE, NUM_PAGES],
        global_strides=[ROPE_K // 16, PAGE_SIZE * ROPE_K // 16],
        box_dims=[ROPE_K, 32, 1],
        swizzle=cuda.TensorMapSwizzle.none,
    )

    # QK consumes the exact same physical plane through the accepted
    # U4_UNPACK_U8 high-level B descriptor.  The page stride remains 256
    # packed bytes; the first page supplies this bounded legality gate.
    latent_fp4 = cute.make_tensor(
        cute.recast_ptr(latent_pages.iterator, dtype=cutlass.Float4E2M1FN),
        cute.make_ordered_layout((PAGE_SIZE, LATENT_K, NUM_PAGES), order=(1, 0, 2)),
    )
    mixed_b_layout = sm100_utils.make_smem_layout_b(
        mixed_mma, MIXED_TILER_MNK, cutlass.Int8, 1
    )
    qk_cta_layout = cute.tiled_divide(
        cute.make_layout((1, 1, 1)), (mixed_mma.thr_id.shape,)
    )
    qk_b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mixed_mma.thr_id)
    qk_tma_atom, qk_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        qk_b_op,
        latent_fp4,
        cute.slice_(mixed_b_layout, (None, None, None, 0)),
        MIXED_TILER_MNK,
        mixed_mma,
        qk_cta_layout.shape,
        internal_type=cutlass.Int8,
    )

    mixed_acc_fake = mixed_mma.make_fragment_C(
        mixed_mma.partition_shape_C(MIXED_TILER_MNK[:2])
    )
    mixed_acc_cols = utils.get_num_tmem_alloc_cols(mixed_acc_fake)

    sfa_smem_layout = blockscaled_utils.make_smem_layout_sfa(
        mixed_mma, MIXED_TILER_MNK, SF_VEC_SIZE, 1
    )
    sfb_smem_layout = blockscaled_utils.make_smem_layout_sfb(
        mixed_mma, MIXED_TILER_MNK, SF_VEC_SIZE, 1
    )
    sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
        mixed_mma,
        MIXED_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfa_smem_layout, (None, None, None, 0)),
    )
    sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
        mixed_mma,
        MIXED_TILER_MNK,
        SF_VEC_SIZE,
        cute.slice_(sfb_smem_layout, (None, None, None, 0)),
    )
    sfa_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(cute.make_ptr(SF_DTYPE, 0), sfa_layout)
    )
    sfb_cols = tcgen05.find_tmem_tensor_col_offset(
        cute.make_tensor(cute.make_ptr(SF_DTYPE, 0), sfb_layout)
    )

    vp_a_shape = vp_mma.partition_shape_A((VP_TILER_MNK[0], VP_TILER_MNK[2], 2))
    vp_a_fake = vp_mma.get_slice(0).make_fragment_A(vp_a_shape)
    vp_a_cols = tcgen05.find_tmem_tensor_col_offset(vp_a_fake)
    vp_o_shape = vp_mma.partition_shape_C(VP_TILER_MNK[:2])
    vp_o_fake = vp_mma.make_fragment_C(cute.append(vp_o_shape, 4))
    vp_o_cols = utils.get_num_tmem_alloc_cols(vp_o_fake)

    if cutlass.const_expr(mixed_acc_cols != 128):
        raise ValueError(f"mixed score footprint changed: {mixed_acc_cols}")
    if cutlass.const_expr(sfa_cols + sfb_cols != 16):
        raise ValueError(f"mixed scale footprint changed: {sfa_cols}+{sfb_cols}")
    if cutlass.const_expr(vp_a_cols != 64):
        raise ValueError(f"V operand footprint changed: {vp_a_cols}")
    if cutlass.const_expr(vp_o_cols != 256):
        raise ValueError(f"O footprint changed: {vp_o_cols}")
    if cutlass.const_expr(SCALE_OFFSET + sfa_cols + sfb_cols > P_COR_OFFSET):
        raise ValueError("scale state overlaps p_cor")
    if cutlass.const_expr(P_COR_OFFSET + 8 > SCORE_OFFSET):
        raise ValueError("p_cor overlaps aligned score")
    if cutlass.const_expr(SCORE_OFFSET + mixed_acc_cols > OUTPUT_OFFSET):
        raise ValueError("score overlaps output")
    if cutlass.const_expr(OUTPUT_OFFSET + vp_o_cols > TMEM_ALLOC_COLS):
        raise ValueError("output exceeds TMEM allocation")

    kernel = layout_kernel(
        output,
        pv_latent_desc,
        rope_desc,
        qk_tma_atom,
        qk_tma_tensor,
        scale_pages,
        mixed_mma,
        vp_mma,
        mixed_acc_fake.layout,
        sfa_layout,
        sfb_layout,
        vp_a_fake.layout,
        vp_o_fake.layout,
        mixed_acc_cols,
        sfa_cols,
        sfb_cols,
        vp_a_cols,
        vp_o_cols,
    )
    kernel.launch(
        grid=CLUSTER_SHAPE_MNK,
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


def main() -> None:
    compiled = cute.compile(
        layout_probe,
        fake(cutlass.Int32, (2, 12), 16),
        fake(cutlass.Uint8, (NUM_PAGES, PAGE_SIZE, LATENT_K // 2), 128),
        fake(cutlass.BFloat16, (NUM_PAGES, PAGE_SIZE), 128),
        fake(cutlass.Float8E4M3FN, (NUM_PAGES, PAGE_SIZE, ROPE_K), 128),
        make_fake_stream(),
        options="--enable-tvm-ffi --opt-level 3",
    )

    output = torch.full((2, 12), -1, dtype=torch.int32, device="cuda")
    latent_pages = torch.zeros(
        (NUM_PAGES, PAGE_SIZE, LATENT_K // 2), dtype=torch.uint8, device="cuda"
    )
    scale_pages = torch.ones(
        (NUM_PAGES, PAGE_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    rope_pages = torch.zeros(
        (NUM_PAGES, PAGE_SIZE, ROPE_K), dtype=torch.float8_e4m3fn, device="cuda"
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(output, latent_pages, scale_pages, rope_pages, stream)
    torch.cuda.synchronize()

    expected = torch.tensor(
        [
            128,
            8,
            8,
            64,
            256,
            V_OFFSET,
            SCALE_OFFSET,
            P_COR_OFFSET,
            SCORE_OFFSET,
            OUTPUT_OFFSET,
            TMEM_ALLOC_COLS,
            SMEM_PAYLOAD_BYTES,
        ],
        dtype=torch.int32,
    )
    result = output.cpu()
    for cta in range(CLUSTER_SHAPE_MNK[0]):
        torch.testing.assert_close(result[cta], expected, rtol=0, atol=0)
    print(
        "PASS_C1_M0QP_LAYOUT "
        f"metadata={result[0].tolist()} "
        "planes=latent256+scale2+rope64 "
        f"smem_payload={SMEM_PAYLOAD_BYTES} "
        f"retrieve_threads={TMEM_RETRIEVE_THREADS} "
        f"cluster_ctas={CLUSTER_SHAPE_MNK[0]}"
    )


if __name__ == "__main__":
    main()
