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

"""Validate packed codebook V through mixed-group SM100 MMA operations.

This is a correctness and hardware/software legality probe, not an attention
benchmark. It unpacks canonical token-major 4-bit V, transposes integer
codebook indices in bounded shared memory, decodes only into registers/TMEM,
and proves the resulting ``V * [I, 0]^T`` output exactly on both cluster CTAs.
The leading CTA also issues a two-CTA ``128x128x128`` QK operation before each
cluster member issues its local one-CTA ``128x64x128`` VP operation under the
same two-CTA TMEM allocation. Separate UMMA barriers guard deallocation, and
the footprint check reserves two 32-column V operand stages.
"""

import json
import os

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
V_OPERAND_STAGES = 2
LATENT_DIM = 512
LATENT_SLICES = LATENT_DIM // VP_TILER_MNK[0]
CTA_LATENT_STAGES = LATENT_SLICES // CLUSTER_SHAPE_MNK[0]
NUM_CLUSTERS = int(os.environ.get("TQ_TMEM_CLUSTERS", "1"))
OBSERVE_FULL_OUTPUT = NUM_CLUSTERS == 1


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
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        VP_TILER_MNK[:2],
        tcgen05.OperandSource.TMEM,
    )
    return qk, vp


@cute.kernel
def mixed_group_kernel(
    output: cute.Tensor,
    packed_v: cute.Tensor,
    codebook: cute.Tensor,
    p_input: cute.Tensor,
    matrix_output: cute.Tensor,
    qk_tiled_mma: cute.TiledMma,
    vp_tiled_mma: cute.TiledMma,
    qk_a_layout: cute.ComposedLayout,
    qk_b_layout: cute.ComposedLayout,
    v_index_layout: cute.ComposedLayout,
    vp_b_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
    qk_cols: cutlass.Constexpr,
    vp_cols: cutlass.Constexpr,
    score_cols: cutlass.Constexpr,
    output_cols: cutlass.Constexpr,
    correction_cols: cutlass.Constexpr,
    v_operand_cols: cutlass.Constexpr,
    v_operand_offset: cutlass.Constexpr,
    total_cols: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    cluster_idx = cute.arch.make_warp_uniform(cute.arch.block_idx()[0] // 2)
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
    s_p = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        vp_b_layout.outer,
        byte_alignment=128,
        swizzle=vp_b_layout.inner,
    )
    s_v_index = smem.allocate_tensor(
        cutlass.Int8,
        v_index_layout.outer,
        byte_alignment=128,
        swizzle=v_index_layout.inner,
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

    vp_thr_mma = vp_tiled_mma.get_slice(0)
    vp_a_shape = vp_tiled_mma.partition_shape_A(
        (VP_TILER_MNK[0], VP_TILER_MNK[2], V_OPERAND_STAGES)
    )
    vp_a_fake = vp_thr_mma.make_fragment_A(vp_a_shape)
    vp_a = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + v_operand_offset, dtype=cutlass.Float8E4M3FN),
        vp_a_fake.layout,
    )
    vp_b = vp_tiled_mma.make_fragment_B(s_p)
    vp_shape = vp_tiled_mma.partition_shape_C(VP_TILER_MNK[:2])
    vp_acc_fake = vp_tiled_mma.make_fragment_C(vp_shape)
    vp_acc = cute.make_tensor(tmem_ptr + score_cols, vp_acc_fake.layout)

    # Map a host-provided [I_64, 0] from the local QK C-thread partition into
    # the transposed VP B descriptor. This mirrors softmax's C-to-B ownership;
    # the composed SMEM tensor applies the physical swizzle.
    qk_local_shape = (
        QK_TILER_MNK[0] // CLUSTER_SHAPE_MNK[0],
        QK_TILER_MNK[1],
    )
    c_p = cute.make_identity_tensor(qk_local_shape)
    score_acc = qk_acc[(None, None), 0, 0]
    score_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
    )
    score_load = tcgen05.make_tmem_copy(score_load_atom, score_acc)
    score_thr = score_load.get_slice(tidx)
    t_c_p = score_thr.partition_D(c_p)
    t_g_p = score_thr.partition_D(p_input)
    r_p = cute.make_fragment_like(t_g_p, cutlass.Float8E4M3FN)
    cute.autovec_copy(t_g_p, r_p)
    for element in cutlass.range_constexpr(cute.size(r_p)):
        query = t_c_p[element][0]
        token = t_c_p[element][1]
        s_p[(query, token % 32), 0, token // 32, 0] = r_p[element]

    # Integer indices occupy bounded SMEM. Decoded FP8 values move from
    # registers directly to TMEM; the two cluster CTAs own disjoint pairs of
    # 128-wide latent slices and never create a decoded SMEM/DSMEM tile.
    packed_i32_ptr = cute.recast_ptr(packed_v.iterator, dtype=cutlass.Int32)
    mma_k_bits = VP_TILER_MNK[2] * cutlass.Float8E4M3FN.width
    tmem_store_atom = cute.make_copy_atom(
        tcgen05.St16x256bOp(tcgen05.Repetition(mma_k_bits // 256)),
        cutlass.Float8E4M3FN,
    )
    tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, vp_a[None, None, None, 0])
    thr_store = tmem_store.get_slice(tidx)
    r_v_shape = thr_store.partition_S(vp_a).shape[:-1]
    r_v = cute.make_rmem_tensor(r_v_shape, cutlass.Float8E4M3FN)
    t_v = thr_store.partition_D(vp_a)
    index_load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Int8, num_bits_per_copy=32
    )
    index_load = cute.make_tiled_copy_S(index_load_atom, tmem_store)
    thr_index_load = index_load.get_slice(tidx)
    t_s_v_index = thr_index_load.partition_S(s_v_index)
    codebook_fp8 = cute.make_tensor(
        cute.recast_ptr(codebook.iterator, dtype=cutlass.Float8E4M3FN),
        codebook.layout,
    )

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

    for cta_stage in cutlass.range_constexpr(CTA_LATENT_STAGES):
        latent_slice = cta_rank + cta_stage * CLUSTER_SHAPE_MNK[0]
        for iteration in cutlass.range_constexpr(16):
            linear_word = iteration * THREADS_PER_CTA + tidx
            token = linear_word // 16
            latent_word = linear_word % 16
            raw_word = (
                packed_i32_ptr
                + token * (LATENT_DIM // 8)
                + latent_slice * 16
                + latent_word
            ).load()
            for nibble in cutlass.range_constexpr(8):
                latent = latent_word * 8 + nibble
                code = cutlass.Int8((raw_word >> (nibble * 4)) & 0xF)
                s_v_index[
                    (latent, token % 32), 0, token // 32, cta_stage
                ] = code
        cute.arch.sync_threads()

        r_codes = cute.make_rmem_tensor(
            t_s_v_index[None, None, None, None, cta_stage].shape,
            cutlass.Int8,
        )
        cute.copy(
            thr_index_load,
            t_s_v_index[None, None, None, None, cta_stage],
            r_codes,
        )
        for element in cutlass.range_constexpr(cute.size(r_codes)):
            r_v[element] = codebook_fp8[cutlass.Int32(r_codes[element])]
        cute.copy(thr_store, r_v, t_v[None, None, None, None, cta_stage])
        cute.arch.fence_view_async_tmem_store()

        vp_acc_slice = cute.make_tensor(
            tmem_ptr + score_cols + latent_slice * vp_cols,
            vp_acc_fake.layout,
        )
        if warp_idx == 0:
            vp_producer.acquire_and_advance()
            vp_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range_constexpr(vp_a.shape[2]):
                cute.gemm(
                    vp_tiled_mma,
                    vp_acc_slice,
                    vp_a[None, None, k_block, cta_stage],
                    vp_b[None, None, k_block, 0],
                    vp_acc_slice,
                )
                vp_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            vp_producer.commit()

        vp_full = vp_consumer.wait_and_advance()
        vp_full.release()
        cute.arch.sync_threads()

        # Load the FP32 accumulator through its logical MxN partition so the
        # host validates every slice and the real epilogue mapping.
        t_acc = vp_acc_slice[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            cutlass.Float32,
        )
        tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, t_acc)
        thr_load = tmem_load.get_slice(tidx)
        t_tmem = thr_load.partition_S(t_acc)
        if cutlass.const_expr(OBSERVE_FULL_OUTPUT):
            g_output = cute.make_tensor(
                matrix_output[0, cta_rank, None, None].iterator
                + latent_slice * VP_TILER_MNK[0],
                cute.make_layout((128, 64), stride=(1, LATENT_DIM)),
            )
        else:
            # The dummy destination supplies the same logical thread
            # partition for the TMEM load; only one register is published.
            g_output = cute.make_tensor(
                matrix_output.iterator,
                cute.make_layout((128, 64), stride=(1, 128)),
            )
        t_gmem = thr_load.partition_D(g_output)
        r_acc = cute.make_fragment_like(t_gmem, cutlass.Float32)
        cute.copy(tmem_load, t_tmem, r_acc)
        cute.arch.fence_view_async_tmem_load()
        if cutlass.const_expr(OBSERVE_FULL_OUTPUT):
            cute.autovec_copy(r_acc, t_gmem)
        elif tidx == 0:
            matrix_output[cluster_idx, cta_rank, latent_slice] = r_acc[0]
        cute.arch.sync_threads()

    if tidx == 0:
        output[cta_rank] = 1
        if is_leader_cta:
            output[2] = qk_cols
            output[3] = vp_cols
            output[4] = score_cols
            output[5] = output_cols
            output[6] = correction_cols
            output[7] = v_operand_cols
            output[8] = v_operand_offset
            output[9] = total_cols

    if warp_idx == 0:
        if is_leader_cta:
            qk_producer.tail()
        vp_producer.tail()

    tmem.relinquish_alloc_permit()
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def mixed_group_probe(
    output: cute.Tensor,
    packed_v: cute.Tensor,
    codebook: cute.Tensor,
    p_input: cute.Tensor,
    matrix_output: cute.Tensor,
):
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
    vp_a_shape = vp_tiled_mma.partition_shape_A(
        (VP_TILER_MNK[0], VP_TILER_MNK[2], V_OPERAND_STAGES)
    )
    vp_a_fake = vp_tiled_mma.get_slice(0).make_fragment_A(vp_a_shape)
    v_operand_cols = tcgen05.find_tmem_tensor_col_offset(vp_a_fake)
    v_operand_offset = score_cols + output_cols + correction_cols
    total_cols = v_operand_offset + v_operand_cols
    if cutlass.const_expr(total_cols >= 512):
        raise ValueError(
            f"TMEM overlap: score={score_cols}, output={output_cols}, "
            f"correction={correction_cols}, v_operand={v_operand_cols}, limit=512"
        )
    qk_a_layout = sm100_utils.make_smem_layout_a(
        qk_tiled_mma, QK_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    qk_b_layout = sm100_utils.make_smem_layout_b(
        qk_tiled_mma, QK_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    v_index_layout = sm100_utils.make_smem_layout_a(
        vp_tiled_mma, VP_TILER_MNK, cutlass.Int8, CTA_LATENT_STAGES
    )
    vp_b_layout = sm100_utils.make_smem_layout_b(
        vp_tiled_mma, VP_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (qk_tiled_mma.thr_id.shape,)
    )
    mixed_group_kernel(
        output,
        packed_v,
        codebook,
        p_input,
        matrix_output,
        qk_tiled_mma,
        vp_tiled_mma,
        qk_a_layout,
        qk_b_layout,
        v_index_layout,
        vp_b_layout,
        cta_layout_vmnk,
        qk_cols,
        vp_cols,
        score_cols,
        output_cols,
        correction_cols,
        v_operand_cols,
        v_operand_offset,
        total_cols,
    ).launch(
        grid=(NUM_CLUSTERS * CLUSTER_SHAPE_MNK[0], 1, 1),
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
        fake(cutlass.Int32, (10,), 4),
        fake(cutlass.Uint8, (128, LATENT_DIM // 2), 16),
        fake(cutlass.Uint8, (16,), 16),
        fake(cutlass.Float8E4M3FN, (64, 128), 16),
        fake(
            cutlass.Float32,
            (
                (1, 2, 64, LATENT_DIM)
                if OBSERVE_FULL_OUTPUT
                else (NUM_CLUSTERS, 2, LATENT_SLICES)
            ),
            16,
        ),
        options="--enable-tvm-ffi --opt-level 3",
    )
    output = torch.zeros(10, device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(20260723)
    codes = torch.randint(
        0,
        16,
        (128, LATENT_DIM),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    packed_v = codes[:, 0::2] | (codes[:, 1::2] << 4)
    codebook_values = torch.tensor(
        [
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            -0.25,
            -0.125,
            0.0,
            0.125,
            0.25,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
        ],
        device="cuda",
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    matrix_output = torch.zeros(
        (1, 2, 64, LATENT_DIM)
        if OBSERVE_FULL_OUTPUT
        else (NUM_CLUSTERS, 2, LATENT_SLICES),
        device="cuda",
        dtype=torch.float32,
    )
    p_input = torch.eye(64, 128, device="cuda", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    compiled(
        output,
        packed_v,
        codebook_values.view(torch.uint8),
        p_input,
        matrix_output,
    )
    torch.cuda.synchronize()
    result = output.cpu()
    torch.testing.assert_close(result[:2], torch.ones(2, dtype=torch.int32))
    (
        qk_cols,
        vp_cols,
        score_cols,
        output_cols,
        correction_cols,
        v_operand_cols,
        v_operand_offset,
        total_cols,
    ) = result[2:].tolist()
    if total_cols >= 512:
        raise AssertionError(
            f"TMEM overlap: score={score_cols}, output={output_cols}, "
            f"correction={correction_cols}, v_operand={v_operand_cols}, limit=512"
        )
    if OBSERVE_FULL_OUTPUT:
        matrix = matrix_output[0]
        expected = codebook_values[codes[:64].long()].float()
        zero = torch.zeros_like(expected[:, :128])
        torch.testing.assert_close(matrix[0, :, 0:128], expected[:, 0:128])
        torch.testing.assert_close(matrix[0, :, 128:256], zero)
        torch.testing.assert_close(matrix[0, :, 256:384], expected[:, 256:384])
        torch.testing.assert_close(matrix[0, :, 384:512], zero)
        torch.testing.assert_close(matrix[1, :, 0:128], zero)
        torch.testing.assert_close(matrix[1, :, 128:256], expected[:, 128:256])
        torch.testing.assert_close(matrix[1, :, 256:384], zero)
        torch.testing.assert_close(matrix[1, :, 384:512], expected[:, 384:512])
        torch.testing.assert_close(matrix.sum(dim=0), expected, rtol=0, atol=0)
    for _ in range(20):
        compiled(
            output,
            packed_v,
            codebook_values.view(torch.uint8),
            p_input,
            matrix_output,
        )
    torch.cuda.synchronize()
    timings = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):
            compiled(
                output,
                packed_v,
                codebook_values.view(torch.uint8),
                p_input,
                matrix_output,
            )
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / 200)
    timings.sort()
    print(
        f"PASS random_codebook_matrix={OBSERVE_FULL_OUTPUT} clusters={NUM_CLUSTERS} "
        f"qk_group=2 vp_group=1 qk_cols={qk_cols} "
        f"vp_cols={vp_cols} score_cols={score_cols} output_cols={output_cols} "
        f"correction_cols={correction_cols} v_operand_cols={v_operand_cols} "
        f"v_operand_offset={v_operand_offset} total_cols={total_cols}"
    )
    print(
        json.dumps(
            {
                "status": "PASS" if OBSERVE_FULL_OUTPUT else "DIAGNOSTIC",
                "scope": "128token_512latent_fullwidth_cluster_scaling",
                "clusters": NUM_CLUSTERS,
                "full_output_checked": OBSERVE_FULL_OUTPUT,
                "windows_us": timings,
                "median_us": timings[len(timings) // 2],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
