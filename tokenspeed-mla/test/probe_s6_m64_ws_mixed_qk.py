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

"""Prove raw one-CTA M64 WS E4M3 x packed-E2M1 QK ownership on SM100.

This is the S6-L0a legality probe. Persistent B remains two E2M1 values per
byte. TMA performs U4_UNPACK_U8 into the padded one-byte SMEM containers
required by ``tcgen05.mma.ws.kind::f8f6f4``. The raw instruction descriptor is
derived from named fields and has no hardware scale operands; the per-token
BF16 scale is applied algebraically to the completed latent score before the
ordinary FP8 RoPE accumulation.

The probe intentionally makes no serving or speed claim.
"""

import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Boolean, Int32
from cutlass._mlir.dialects import builtin, llvm, nvvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr

THREADS = 128
CLUSTER_SHAPE_MNK = (1, 1, 1)
M = 64
N = 128
LATENT_K = 512
LATENT_TILER_MNK = (M, N, 256)
LATENT_STAGES = LATENT_K // LATENT_TILER_MNK[2]
ROPE_K = 64
ROPE_TILER_MNK = (M, N, ROPE_K)
TMEM_ALLOC_COLS = 128
PACKED_B_SMEM_DTYPE = cutlass.Int8
MAX_DYNAMIC_SMEM_BYTES = 232448


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


# These sizes follow the concrete lowered SMEM allocations: 56 bytes of
# barriers/allocator state, two latent stages, and one RoPE stage.  Keep the
# arithmetic explicit so a tile/layout edit cannot silently grow past SM100's
# opt-in shared-memory limit.
_qk_smem_offset = _align_up(56, 128)
_qk_smem_offset = _align_up(_qk_smem_offset + 32768, 128)  # latent A
_qk_smem_offset = _align_up(_qk_smem_offset + 65536, 128)  # padded latent B
_qk_smem_offset = _align_up(_qk_smem_offset + 4096, 128)  # RoPE A
DYNAMIC_SMEM_BYTES = _qk_smem_offset + 8192  # RoPE B
if DYNAMIC_SMEM_BYTES != 110720:
    raise AssertionError(f"unexpected QK dynamic SMEM: {DYNAMIC_SMEM_BYTES}")
if DYNAMIC_SMEM_BYTES > MAX_DYNAMIC_SMEM_BYTES:
    raise AssertionError("QK dynamic SMEM exceeds the SM100 opt-in limit")


def make_f8f6f4_idesc(
    *,
    m: int,
    n: int,
    a_format: int,
    b_format: int,
    c_format: int,
    a_major: int = 0,
    b_major: int = 0,
) -> int:
    """Build NVIDIA's SM100 unscaled F8F6F4 instruction descriptor."""

    if m not in (64, 128, 256) or n % 8 or not 8 <= n <= 256:
        raise ValueError(f"invalid SM100 MMA shape: M={m}, N={n}")
    for name, value, bits in (
        ("a_format", a_format, 3),
        ("b_format", b_format, 3),
        ("c_format", c_format, 2),
        ("a_major", a_major, 1),
        ("b_major", b_major, 1),
    ):
        if not 0 <= value < (1 << bits):
            raise ValueError(f"{name} does not fit {bits} bits: {value}")
    return (
        (c_format << 4)
        | (a_format << 7)
        | (b_format << 10)
        | (a_major << 15)
        | (b_major << 16)
        | ((n // 8) << 17)
        | ((m // 16) << 24)
    )


IDESC_E4M3_E2M1_F32_M64_N128_KMAJ = make_f8f6f4_idesc(
    m=M,
    n=N,
    a_format=0,  # E4M3
    b_format=5,  # E2M1
    c_format=1,  # F32
)
if IDESC_E4M3_E2M1_F32_M64_N128_KMAJ != 0x04201410:
    raise AssertionError("the named-field M64/N128 descriptor changed")
IDESC_E4M3_E4M3_F32_M64_N128_KMAJ = make_f8f6f4_idesc(
    m=M,
    n=N,
    a_format=0,  # E4M3
    b_format=0,  # E4M3
    c_format=1,  # F32
)
if IDESC_E4M3_E4M3_F32_M64_N128_KMAJ != 0x04200010:
    raise AssertionError("the named-field dense M64/N128 descriptor changed")


@cute.struct
class SharedStorage:
    tma_mbar: cute.struct.MemRange[cutlass.Int64, LATENT_STAGES + 1]
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


@cute.jit
def tcgen05_mma_ws_f8f6f4_one(
    t_cr_a: cute.Tensor,
    t_cr_b: cute.Tensor,
    t_ct_c: cute.Tensor,
    idesc_value: Int32,
    accumulate: Boolean,
):
    """Issue one K32 raw weight-stationary F8F6F4 instruction."""

    d = builtin.unrealized_conversion_cast([Int32.mlir_type], [t_ct_c.iterator.value])
    d = llvm.inttoptr(llvm.PointerType.get(6), d)
    a = tcgen05.smem_descriptor_to_int(t_cr_a.iterator).ir_value()
    b = tcgen05.smem_descriptor_to_int(t_cr_b.iterator).ir_value()
    with cute.arch.elect_one():
        nvvm.tcgen05_mma_ws(
            nvvm.Tcgen05MMAKind.F8F6F4,
            d,
            a,
            b,
            idesc_value.ir_value(),
            accumulate.ir_value(),
        )


def make_tiled_mmas():
    mixed_op = tcgen05.MmaF8F6F4Op(
        cutlass.Float8E4M3FN,
        cutlass.Float4E2M1FN,
        cutlass.Float32,
        (M, N, 32),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
    )
    mixed = cute.make_tiled_mma(mixed_op)
    rope = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        ROPE_TILER_MNK[:2],
    )
    # Production's M64 WS path views the physical score accumulator through a
    # two-CTA M128 split-N epilogue mapping.  The raw M64 instruction still owns
    # one CTA and writes only its M64 slice; this helper supplies the matching
    # TMEM load/store lane layout without changing instruction ownership.
    splitn_epi = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        (M * 2, N),
    )
    return mixed, rope, splitn_epi


@cute.kernel
def mixed_qk_kernel(
    output: cute.Tensor,
    metadata: cute.Tensor,
    token_scale: cute.Tensor,
    elapsed_ns: cute.Tensor,
    mixed_mma: cute.TiledMma,
    rope_mma: cute.TiledMma,
    splitn_epi_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    tma_atom_rope_a: cute.CopyAtom,
    tma_tensor_rope_a: cute.Tensor,
    tma_atom_rope_b: cute.CopyAtom,
    tma_tensor_rope_b: cute.Tensor,
    mixed_a_layout: cute.ComposedLayout,
    mixed_b_layout: cute.ComposedLayout,
    rope_a_layout: cute.ComposedLayout,
    rope_b_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cutlass.Int32(0))

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    s_mixed_a = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        mixed_a_layout.outer,
        byte_alignment=128,
        swizzle=mixed_a_layout.inner,
    )
    s_mixed_b = smem.allocate_tensor(
        PACKED_B_SMEM_DTYPE,
        mixed_b_layout.outer,
        byte_alignment=128,
        swizzle=mixed_b_layout.inner,
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

    g_a = cute.local_tile(
        tma_tensor_a,
        cute.slice_(LATENT_TILER_MNK, (None, 0, None)),
        (None, None, None),
    )
    g_b = cute.local_tile(
        tma_tensor_b,
        cute.slice_(LATENT_TILER_MNK, (0, None, None)),
        (None, None, None),
    )
    g_rope_a = cute.local_tile(
        tma_tensor_rope_a,
        cute.slice_(ROPE_TILER_MNK, (None, 0, None)),
        (None, None, None),
    )
    g_rope_b = cute.local_tile(
        tma_tensor_rope_b,
        cute.slice_(ROPE_TILER_MNK, (0, None, None)),
        (None, None, None),
    )
    thr_mma = mixed_mma.get_slice(0)
    t_cg_a = thr_mma.partition_A(g_a)
    t_cg_b = thr_mma.partition_B(g_b)
    a_cta_layout = cute.make_layout(cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape)
    b_cta_layout = cute.make_layout(cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape)
    t_as_a, t_ag_a = cpasync.tma_partition(
        tma_atom_a,
        cta_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(s_mixed_a, 0, 3),
        cute.group_modes(t_cg_a, 0, 3),
    )
    t_bs_b, t_bg_b = cpasync.tma_partition(
        tma_atom_b,
        cta_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(s_mixed_b, 0, 3),
        cute.group_modes(t_cg_b, 0, 3),
    )
    rope_thr_mma = rope_mma.get_slice(0)
    t_cg_rope_a = rope_thr_mma.partition_A(g_rope_a)
    t_cg_rope_b = rope_thr_mma.partition_B(g_rope_b)
    t_as_rope_a, t_ag_rope_a = cpasync.tma_partition(
        tma_atom_rope_a,
        cta_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(s_rope_a, 0, 3),
        cute.group_modes(t_cg_rope_a, 0, 3),
    )
    t_bs_rope_b, t_bg_rope_b = cpasync.tma_partition(
        tma_atom_rope_b,
        cta_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(s_rope_b, 0, 3),
        cute.group_modes(t_cg_rope_b, 0, 3),
    )
    t_ag_a = t_ag_a[(None, 0, None, 0)]
    t_bg_b = t_bg_b[(None, 0, None, 0)]
    t_ag_rope_a = t_ag_rope_a[(None, 0, None, 0)]
    t_bg_rope_b = t_bg_rope_b[(None, 0, None, 0)]

    latent_copy_bytes = cute.size_in_bytes(
        cutlass.Float8E4M3FN,
        cute.slice_(s_mixed_a, (None, None, None, 0)),
    ) + cute.size_in_bytes(
        cutlass.Float4E2M1FN,
        cute.slice_(s_mixed_b, (None, None, None, 0)),
    )
    rope_copy_bytes = cute.size_in_bytes(
        cutlass.Float8E4M3FN,
        cute.slice_(s_rope_a, (None, None, None, 0)),
    ) + cute.size_in_bytes(
        cutlass.Float8E4M3FN,
        cute.slice_(s_rope_b, (None, None, None, 0)),
    )
    tma_barriers = pipeline.MbarrierArray(
        storage.tma_mbar.data_ptr(),
        LATENT_STAGES + 1,
        (
            pipeline.PipelineOp.TmaLoad,
            pipeline.CooperativeGroup(pipeline.Agent.Thread),
        ),
        tx_count=latent_copy_bytes,
    )
    mma_producer, mma_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.mma_mbar.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
    ).make_participants()
    pipeline.pipeline_init_arrive(
        cluster_shape_mn=CLUSTER_SHAPE_MNK[:2], is_relaxed=True
    )
    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_barrier,
        is_two_cta=False,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )
    tmem.allocate(TMEM_ALLOC_COLS)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

    mixed_a = mixed_mma.make_fragment_A(s_mixed_a)
    mixed_b = mixed_mma.make_fragment_B(s_mixed_b)
    acc_shape = mixed_mma.partition_shape_C(LATENT_TILER_MNK[:2])
    acc_fake = mixed_mma.make_fragment_C(acc_shape)
    acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
    splitn_acc_shape = splitn_epi_mma.partition_shape_C((M * 2, N))
    splitn_acc_fake = splitn_epi_mma.make_fragment_C(splitn_acc_shape)
    splitn_acc = cute.make_tensor(tmem_ptr, splitn_acc_fake.layout)
    rope_a = rope_mma.make_fragment_A(s_rope_a)
    rope_b = rope_mma.make_fragment_B(s_rope_b)
    mixed_k_blocks = cute.size(mixed_a, mode=[2])
    rope_k_blocks = cute.size(rope_a, mode=[2])

    pipeline.pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE_MNK[:2])
    cute.arch.sync_threads()
    if warp_idx == 0:
        for latent_stage in cutlass.range_constexpr(LATENT_STAGES):
            tma_bar = tma_barriers.get_barrier(latent_stage)
            tma_barriers.arrive_and_expect_tx(latent_stage, latent_copy_bytes)
            cute.copy(
                tma_atom_a,
                t_ag_a[(None, latent_stage)],
                t_as_a[(None, latent_stage)],
                tma_bar_ptr=tma_bar,
            )
            cute.copy(
                tma_atom_b,
                t_bg_b[(None, latent_stage)],
                t_bs_b[(None, latent_stage)],
                tma_bar_ptr=tma_bar,
            )
        rope_bar = tma_barriers.get_barrier(LATENT_STAGES)
        tma_barriers.arrive_and_expect_tx(LATENT_STAGES, rope_copy_bytes)
        cute.copy(
            tma_atom_rope_a,
            t_ag_rope_a[(None, 0)],
            t_as_rope_a[(None, 0)],
            tma_bar_ptr=rope_bar,
        )
        cute.copy(
            tma_atom_rope_b,
            t_bg_rope_b[(None, 0)],
            t_bs_rope_b[(None, 0)],
            tma_bar_ptr=rope_bar,
        )

    if warp_idx == 0:
        mma_producer.acquire_and_advance()
        for latent_stage in cutlass.range_constexpr(LATENT_STAGES):
            tma_barriers.wait(latent_stage, 0)
            for k_block in cutlass.range(mixed_k_blocks, unroll_full=True):
                tcgen05_mma_ws_f8f6f4_one(
                    mixed_a[None, None, k_block, latent_stage],
                    mixed_b[None, None, k_block, latent_stage],
                    acc,
                    Int32(IDESC_E4M3_E2M1_F32_M64_N128_KMAJ),
                    Boolean(latent_stage != 0 or k_block != 0),
                )
        mma_producer.commit()

    latent_ready = mma_consumer.wait_and_advance()
    latent_ready.release()
    cute.arch.sync_threads()
    start_ns = cute.arch.globaltimer()

    score_tile = splitn_acc[(None, None), 0, 0]
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
    )
    tmem_store_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
    )
    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, score_tile)
    tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, score_tile)
    thr_load = tmem_load.get_slice(tidx)
    thr_store = tmem_store.get_slice(tidx)
    # Match production softmax: the split-N TMEM view is an implementation
    # layout, while register coordinates describe this CTA's logical M64xN128
    # score tile.
    coords = cute.make_identity_tensor((M, N))
    load_coords = thr_load.partition_D(coords)
    t_tmem_load = thr_load.partition_S(score_tile)
    t_tmem_store = thr_store.partition_D(score_tile)
    load_regs = cute.make_fragment_like(load_coords, cutlass.Float32)
    cute.copy(tmem_load, t_tmem_load, load_regs)
    cute.arch.fence_view_async_tmem_load()
    for element in cutlass.range_constexpr(cute.size(load_regs)):
        token = load_coords[element][1]
        load_regs[element] = load_regs[element] * token_scale[token].to(cutlass.Float32)
    cute.copy(tmem_store, load_regs, t_tmem_store)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.sync_threads()

    if warp_idx == 0:
        tma_barriers.wait(LATENT_STAGES, 0)
        mma_producer.acquire_and_advance()
        for k_block in cutlass.range(rope_k_blocks, unroll_full=True):
            # Keep latent and RoPE on the same production WS accumulator
            # mapping.  A regular CuTe M64 GEMM uses a different logical C
            # view and corrupts deterministic M16/N64 quadrants even though the
            # underlying TMEM pointer is shared.
            tcgen05_mma_ws_f8f6f4_one(
                rope_a[None, None, k_block, 0],
                rope_b[None, None, k_block, 0],
                acc,
                Int32(IDESC_E4M3_E4M3_F32_M64_N128_KMAJ),
                Boolean(True),
            )
        mma_producer.commit()
    rope_ready = mma_consumer.wait_and_advance()
    rope_ready.release()
    end_ns = cute.arch.globaltimer()
    cute.arch.sync_threads()

    output_view = cute.make_tensor(
        output.iterator,
        cute.make_layout((M, N), stride=(N, 1)),
    )
    output_load = tcgen05.make_tmem_copy(tmem_load_atom, score_tile)
    output_thr = output_load.get_slice(tidx)
    output_src = output_thr.partition_S(score_tile)
    output_dst = output_thr.partition_D(output_view)
    output_regs = cute.make_fragment_like(output_dst, cutlass.Float32)
    cute.copy(output_load, output_src, output_regs)
    cute.arch.fence_view_async_tmem_load()
    cute.autovec_copy(output_regs, output_dst)
    cute.arch.sync_threads()

    if tidx == 0:
        metadata[0] = IDESC_E4M3_E2M1_F32_M64_N128_KMAJ
        metadata[1] = utils.get_num_tmem_alloc_cols(acc_fake)
        metadata[2] = mixed_k_blocks
        metadata[3] = rope_k_blocks
        elapsed_ns[0] = end_ns - start_ns

    if warp_idx == 0:
        mma_producer.tail()
    tmem.relinquish_alloc_permit()
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def mixed_qk_probe(
    mixed_a_ptr: cute.Pointer,
    mixed_b_ptr: cute.Pointer,
    rope_a_ptr: cute.Pointer,
    rope_b_ptr: cute.Pointer,
    output: cute.Tensor,
    metadata: cute.Tensor,
    token_scale: cute.Tensor,
    elapsed_ns: cute.Tensor,
    stream,
):
    mixed_mma, rope_mma, splitn_epi_mma = make_tiled_mmas()
    g_mixed_a = cute.make_tensor(
        mixed_a_ptr,
        cute.make_ordered_layout((M, LATENT_K, 1), order=(1, 0, 2)),
    )
    g_mixed_b = cute.make_tensor(
        mixed_b_ptr,
        cute.make_ordered_layout((N, LATENT_K, 1), order=(1, 0, 2)),
    )
    g_rope_a = cute.make_tensor(
        rope_a_ptr,
        cute.make_ordered_layout((M, ROPE_K, 1), order=(1, 0, 2)),
    )
    g_rope_b = cute.make_tensor(
        rope_b_ptr,
        cute.make_ordered_layout((N, ROPE_K, 1), order=(1, 0, 2)),
    )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (mixed_mma.thr_id.shape,)
    )
    mixed_a_layout = sm100_utils.make_smem_layout_a(
        mixed_mma,
        LATENT_TILER_MNK,
        cutlass.Float8E4M3FN,
        LATENT_STAGES,
    )
    mixed_b_layout = sm100_utils.make_smem_layout_b(
        mixed_mma,
        LATENT_TILER_MNK,
        PACKED_B_SMEM_DTYPE,
        LATENT_STAGES,
    )
    tma_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load,
        g_mixed_a,
        cute.slice_(mixed_a_layout, (None, None, None, 0)),
        LATENT_TILER_MNK,
        mixed_mma,
        cta_layout_vmnk.shape,
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load,
        g_mixed_b,
        cute.slice_(mixed_b_layout, (None, None, None, 0)),
        LATENT_TILER_MNK,
        mixed_mma,
        cta_layout_vmnk.shape,
        internal_type=PACKED_B_SMEM_DTYPE,
    )
    rope_a_layout = sm100_utils.make_smem_layout_a(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    rope_b_layout = sm100_utils.make_smem_layout_b(
        rope_mma, ROPE_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    tma_atom_rope_a, tma_tensor_rope_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load,
        g_rope_a,
        cute.slice_(rope_a_layout, (None, None, None, 0)),
        ROPE_TILER_MNK,
        rope_mma,
        cta_layout_vmnk.shape,
    )
    tma_atom_rope_b, tma_tensor_rope_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load,
        g_rope_b,
        cute.slice_(rope_b_layout, (None, None, None, 0)),
        ROPE_TILER_MNK,
        rope_mma,
        cta_layout_vmnk.shape,
    )
    kernel = mixed_qk_kernel(
        output,
        metadata,
        token_scale,
        elapsed_ns,
        mixed_mma,
        rope_mma,
        splitn_epi_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_rope_a,
        tma_tensor_rope_a,
        tma_atom_rope_b,
        tma_tensor_rope_b,
        mixed_a_layout,
        mixed_b_layout,
        rope_a_layout,
        rope_b_layout,
        cta_layout_vmnk,
    )
    kernel.launch(
        grid=CLUSTER_SHAPE_MNK,
        block=(THREADS, 1, 1),
        min_blocks_per_mp=1,
        stream=stream,
    )


def _to_cute_tensor(source: torch.Tensor, dtype):
    source = source.to(device="cuda", dtype=torch.float32).contiguous()
    result, _ = cutlass_torch.cute_tensor_like(
        source.cpu(),
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    return cutlass_torch.convert_cute_tensor(
        source,
        result,
        dtype,
        is_dynamic_layout=True,
    )


def compile_probe():
    return cute.compile(
        mixed_qk_probe,
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_fake_compact_tensor(
            cutlass.Float32, (M, N), stride_order=(1, 0), assumed_align=16
        ),
        make_fake_compact_tensor(
            cutlass.Int32, (4,), stride_order=(0,), assumed_align=4
        ),
        make_fake_compact_tensor(
            cutlass.BFloat16, (N,), stride_order=(0,), assumed_align=16
        ),
        make_fake_compact_tensor(
            cutlass.Int64, (1,), stride_order=(0,), assumed_align=8
        ),
        make_fake_stream(),
        options="--enable-tvm-ffi --opt-level 3",
    )


def _audit_generated(compiled) -> dict[str, int]:
    ptx = getattr(compiled, "__ptx__", None)
    sass = getattr(compiled, "__sass__", None)
    cubin = getattr(compiled, "__cubin__", None)
    mlir = getattr(compiled, "__mlir__", None)
    if not isinstance(ptx, str) or not isinstance(sass, str):
        raise AssertionError("set CUTE_DSL_KEEP=all before generated-code audit")
    if not isinstance(cubin, bytes) or not isinstance(mlir, str):
        raise AssertionError("generated CUBIN/MLIR artifacts were not retained")

    ptx_mma = ptx.count("tcgen05.mma.ws.cta_group::1.kind::f8f6f4")
    sass_mma = sass.count(" UTCQMMA.WS")
    if ptx_mma != 18 or sass_mma != 18:
        raise AssertionError(f"QK MMA count differs: PTX={ptx_mma} SASS={sass_mma}")
    for allocation in ('"32768:1"', '"65536:1"', '"4096:1"', '"8192:1"'):
        if allocation not in mlir:
            raise AssertionError(f"QK lowered SMEM allocation missing: {allocation}")

    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
        cubin_file.write(cubin)
        cubin_file.flush()
        resource_output = subprocess.run(
            ["cuobjdump", "--dump-resource-usage", cubin_file.name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    matches = re.findall(
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resource_output
    )
    if len(matches) != 1:
        raise AssertionError(f"expected one QK resource record, got {matches}")
    registers, stack, static_shared, local = (int(value) for value in matches[0])
    if registers > 160 or stack != 0 or local != 0:
        raise AssertionError(
            "QK resource gate failed: "
            f"registers={registers} stack={stack} local={local}"
        )

    dump_dir = os.environ.get("TQ_S6_DUMP_GENERATED_DIR")
    if dump_dir:
        output_dir = Path(dump_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix, payload, binary in (
            ("ptx", ptx, False),
            ("sass", sass, False),
            ("mlir", mlir, False),
            ("cubin", cubin, True),
            ("resources.txt", resource_output, False),
        ):
            output_path = output_dir / f"qk.{suffix}"
            (
                output_path.write_bytes(payload)
                if binary
                else output_path.write_text(payload)
            )
    return {
        "ptx_mma": ptx_mma,
        "sass_mma": sass_mma,
        "registers": registers,
        "stack": stack,
        "static_shared": static_shared,
        "local": local,
    }


def _build_case(
    *,
    active_rows: int,
    inactive_poison: float,
    query_poison_coord: tuple[int, int] | None = None,
    key_poison_coord: tuple[int, int] | None = None,
    rope_query_poison_coord: tuple[int, int] | None = None,
    rope_key_poison_coord: tuple[int, int] | None = None,
    scale_poison_token: int | None = None,
    zero_latent: bool = False,
    zero_rope: bool = False,
    unit_scale: bool = False,
):
    if active_rows not in (8, 40):
        raise ValueError(f"unsupported active row count: {active_rows}")
    rows = torch.arange(M, device="cuda", dtype=torch.int64)
    tokens = torch.arange(N, device="cuda", dtype=torch.int64)
    latent = torch.arange(LATENT_K, device="cuda", dtype=torch.int64)
    rope = torch.arange(ROPE_K, device="cuda", dtype=torch.int64)
    q_source = (
        (
            (
                (rows[:, None] + 1) * (latent[None, :] + 3) * 17
                + rows[:, None] * 37
                + latent[None, :] * 19
            )
            % 257
            + (
                (rows[:, None] + 5) * (latent[None, :] + 11) * 31
                + rows[:, None] * 43
                + latent[None, :] * 47
            )
            % 263
        )
        % 3
    ).float()
    k_source = (
        (
            (
                (tokens[:, None] + 1) * (latent[None, :] + 5) * 23
                + tokens[:, None] * 41
                + latent[None, :] * 29
            )
            % 263
            + (
                (tokens[:, None] + 7) * (latent[None, :] + 13) * 37
                + tokens[:, None] * 53
                + latent[None, :] * 59
            )
            % 269
        )
        % 3
    ).float()
    qr_source = (
        (
            (
                (rows[:, None] + 1) * (rope[None, :] + 7) * 31
                + rows[:, None] * 43
                + rope[None, :] * 47
            )
            % 251
            + (
                (rows[:, None] + 5) * (rope[None, :] + 13) * 41
                + rows[:, None] * 61
                + rope[None, :] * 67
            )
            % 263
        )
        % 3
    ).float()
    kr_source = (
        (
            (
                (tokens[:, None] + 1) * (rope[None, :] + 11) * 37
                + tokens[:, None] * 53
                + rope[None, :] * 59
            )
            % 269
            + (
                (tokens[:, None] + 7) * (rope[None, :] + 17) * 43
                + tokens[:, None] * 71
                + rope[None, :] * 73
            )
            % 271
        )
        % 3
    ).float()
    q_source[active_rows:, :] = inactive_poison
    qr_source[active_rows:, :] = inactive_poison
    if zero_latent:
        q_source.zero_()
        k_source.zero_()
    if zero_rope or os.environ.get("TQ_S6_ZERO_ROPE") == "1":
        qr_source.zero_()
        kr_source.zero_()
    if query_poison_coord is not None:
        row, coordinate = query_poison_coord
        old_value = float(q_source[row, coordinate].item())
        q_source[row, coordinate] = 2.0 if old_value != 2.0 else 0.5
    if key_poison_coord is not None:
        token, coordinate = key_poison_coord
        old_value = float(k_source[token, coordinate].item())
        k_source[token, coordinate] = 2.0 if old_value != 2.0 else 0.5
    if rope_query_poison_coord is not None:
        row, coordinate = rope_query_poison_coord
        old_value = float(qr_source[row, coordinate].item())
        qr_source[row, coordinate] = 2.0 if old_value != 2.0 else 0.5
    if rope_key_poison_coord is not None:
        token, coordinate = rope_key_poison_coord
        old_value = float(kr_source[token, coordinate].item())
        kr_source[token, coordinate] = 2.0 if old_value != 2.0 else 0.5
    token_scale = (
        1.0 + torch.arange(N, device="cuda", dtype=torch.float32) / 256.0
    ).to(torch.bfloat16)
    if unit_scale or os.environ.get("TQ_S6_UNIT_SCALE") == "1":
        token_scale.fill_(1.0)
    if scale_poison_token is not None:
        token_scale[scale_poison_token] = torch.tensor(
            1.75, device="cuda", dtype=torch.bfloat16
        )
    latent_expected = (q_source @ k_source.T) * token_scale.float()[None, :]
    rope_expected = qr_source @ kr_source.T
    expected = latent_expected + rope_expected
    # q5 supplies enough independent rows for a strong 128-token signature
    # gate. q1 still runs the complete exact and distributed-poison matrix,
    # but is not used to demand global token-signature uniqueness from 8 rows.
    if active_rows == 40 and not zero_latent and not zero_rope:
        if torch.unique(latent_expected[:active_rows], dim=0).shape[0] != active_rows:
            raise AssertionError("latent QK oracle does not distinguish active rows")
        if torch.unique(latent_expected[:active_rows].T, dim=0).shape[0] != N:
            raise AssertionError("latent QK oracle does not distinguish tokens")
        if torch.unique(rope_expected[:active_rows], dim=0).shape[0] != active_rows:
            raise AssertionError("RoPE oracle does not distinguish active rows")
        if torch.unique(rope_expected[:active_rows].T, dim=0).shape[0] != N:
            raise AssertionError("RoPE oracle does not distinguish tokens")
    return (
        _to_cute_tensor(q_source.view(M, LATENT_K, 1), cutlass.Float8E4M3FN),
        _to_cute_tensor(k_source.view(N, LATENT_K, 1), cutlass.Float4E2M1FN),
        _to_cute_tensor(qr_source.view(M, ROPE_K, 1), cutlass.Float8E4M3FN),
        _to_cute_tensor(kr_source.view(N, ROPE_K, 1), cutlass.Float8E4M3FN),
        token_scale,
        expected,
    )


def _run_case(compiled, case, output, metadata, elapsed_ns, stream) -> torch.Tensor:
    q, k, qr, kr, token_scale, expected = case
    output.fill_(float("nan"))
    compiled(
        q.iterator,
        k.iterator,
        qr.iterator,
        kr.iterator,
        output,
        metadata,
        token_scale,
        elapsed_ns,
        stream,
    )
    torch.cuda.synchronize()
    if os.environ.get("TQ_S6_DIAG") == "1":
        mismatch = output != expected
        row_counts = mismatch.sum(dim=1)
        col_counts = mismatch.sum(dim=0)
        print(
            "DIAG "
            f"mismatches={int(mismatch.sum().item())} "
            f"matching_rows={torch.nonzero(row_counts == 0).flatten().tolist()} "
            f"row_counts={row_counts.tolist()} "
            f"matching_cols={torch.nonzero(col_counts == 0).flatten().tolist()} "
            f"col_counts={col_counts.tolist()} "
            f"output_r0={output[0].tolist()} "
            f"expected_r0={expected[0].tolist()}"
        )
        return output.clone()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    return output.clone()


def main() -> None:
    compiled = compile_probe()
    generated = None
    if os.environ.get("TQ_S6_AUDIT_GENERATED") == "1":
        generated = _audit_generated(compiled)
    if os.environ.get("TQ_S6_COMPILE_ONLY") == "1":
        print(
            "PASS compile_only=True m64_ws=True mixed_e4m3_e2m1=True "
            f"idesc=0x{IDESC_E4M3_E2M1_F32_M64_N128_KMAJ:08x}"
        )
        return

    output = torch.empty((M, N), device="cuda", dtype=torch.float32)
    metadata = torch.empty(4, device="cuda", dtype=torch.int32)
    elapsed_ns = torch.empty(1, device="cuda", dtype=torch.int64)
    eager_stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    normal_cases = {}
    for active_rows in (8, 40):
        low_poison = _build_case(active_rows=active_rows, inactive_poison=0.5)
        high_poison = _build_case(active_rows=active_rows, inactive_poison=2.0)
        low_output = _run_case(
            compiled, low_poison, output, metadata, elapsed_ns, eager_stream
        )
        if os.environ.get("TQ_S6_DIAG") == "1":
            return
        high_output = _run_case(
            compiled, high_poison, output, metadata, elapsed_ns, eager_stream
        )
        if not torch.equal(low_output[:active_rows], high_output[:active_rows]):
            raise AssertionError(
                f"inactive row poison changed q{active_rows // 8} rows"
            )
        if torch.equal(low_output[active_rows:], high_output[active_rows:]):
            raise AssertionError(
                f"inactive row poison did not exercise q{active_rows // 8}"
            )
        normal_cases[active_rows] = low_poison

    q5_normal = normal_cases[40]
    q5_output = _run_case(
        compiled, q5_normal, output, metadata, elapsed_ns, eager_stream
    )
    for scale_token in (17, 79, 113):
        scale_poison = _build_case(
            active_rows=40,
            inactive_poison=0.5,
            scale_poison_token=scale_token,
        )
        scale_output = _run_case(
            compiled, scale_poison, output, metadata, elapsed_ns, eager_stream
        )
        scale_changed = (scale_output != q5_output).any(dim=0)
        if torch.nonzero(scale_changed, as_tuple=False).flatten().tolist() != [
            scale_token
        ]:
            raise AssertionError(
                f"BF16 token-scale poison escaped column {scale_token}"
            )

    for key_token, key_coordinate in ((7, 37), (71, 173), (119, 421)):
        key_poison = _build_case(
            active_rows=40,
            inactive_poison=0.5,
            key_poison_coord=(key_token, key_coordinate),
        )
        key_output = _run_case(
            compiled, key_poison, output, metadata, elapsed_ns, eager_stream
        )
        key_changed = (key_output != q5_output).any(dim=0)
        if torch.nonzero(key_changed, as_tuple=False).flatten().tolist() != [key_token]:
            raise AssertionError(f"packed-E2M1 poison escaped column {key_token}")

    for query_row, query_coordinate in ((3, 29), (19, 211), (37, 467)):
        query_poison = _build_case(
            active_rows=40,
            inactive_poison=0.5,
            query_poison_coord=(query_row, query_coordinate),
        )
        query_output = _run_case(
            compiled, query_poison, output, metadata, elapsed_ns, eager_stream
        )
        query_changed = (query_output != q5_output).any(dim=1)
        if torch.nonzero(query_changed, as_tuple=False).flatten().tolist() != [
            query_row
        ]:
            raise AssertionError(f"latent query poison escaped row {query_row}")

    for rope_row, rope_coordinate in ((5, 11), (23, 37)):
        rope_query_poison = _build_case(
            active_rows=40,
            inactive_poison=0.5,
            rope_query_poison_coord=(rope_row, rope_coordinate),
        )
        rope_query_output = _run_case(
            compiled, rope_query_poison, output, metadata, elapsed_ns, eager_stream
        )
        rope_query_changed = (rope_query_output != q5_output).any(dim=1)
        if torch.nonzero(rope_query_changed, as_tuple=False).flatten().tolist() != [
            rope_row
        ]:
            raise AssertionError(f"RoPE query poison escaped row {rope_row}")

    for rope_token, rope_coordinate in ((13, 7), (83, 29), (121, 53)):
        rope_key_poison = _build_case(
            active_rows=40,
            inactive_poison=0.5,
            rope_key_poison_coord=(rope_token, rope_coordinate),
        )
        rope_key_output = _run_case(
            compiled, rope_key_poison, output, metadata, elapsed_ns, eager_stream
        )
        rope_key_changed = (rope_key_output != q5_output).any(dim=0)
        if torch.nonzero(rope_key_changed, as_tuple=False).flatten().tolist() != [
            rope_token
        ]:
            raise AssertionError(f"RoPE key poison escaped column {rope_token}")

    latent_only = _build_case(
        active_rows=40,
        inactive_poison=0.5,
        zero_rope=True,
    )
    _run_case(compiled, latent_only, output, metadata, elapsed_ns, eager_stream)
    rope_only = _build_case(
        active_rows=40,
        inactive_poison=0.5,
        zero_latent=True,
        unit_scale=True,
    )
    _run_case(compiled, rope_only, output, metadata, elapsed_ns, eager_stream)
    unit_scale = _build_case(
        active_rows=40,
        inactive_poison=0.5,
        unit_scale=True,
    )
    _run_case(compiled, unit_scale, output, metadata, elapsed_ns, eager_stream)

    q, k, qr, kr, token_scale, expected = q5_normal

    graph_output = torch.full_like(output, float("nan"))
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    graph_stream = cuda_driver.CUstream(capture_stream.cuda_stream)
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        compiled(
            q.iterator,
            k.iterator,
            qr.iterator,
            kr.iterator,
            graph_output,
            metadata,
            token_scale,
            elapsed_ns,
            graph_stream,
        )
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=capture_stream):
        compiled(
            q.iterator,
            k.iterator,
            qr.iterator,
            kr.iterator,
            graph_output,
            metadata,
            token_scale,
            elapsed_ns,
            graph_stream,
        )
    torch.cuda.synchronize()
    allocation_before = torch.cuda.memory_allocated()
    for _ in range(100):
        graph_output.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    allocation_after = torch.cuda.memory_allocated()
    if allocation_after != allocation_before:
        raise AssertionError(
            f"graph replay allocation grew: {allocation_before} -> {allocation_after}"
        )
    torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)

    samples = []
    for _ in range(20):
        compiled(
            q.iterator,
            k.iterator,
            qr.iterator,
            kr.iterator,
            output,
            metadata,
            token_scale,
            elapsed_ns,
            eager_stream,
        )
        torch.cuda.synchronize()
        samples.append(int(elapsed_ns.item()))
    observed = metadata.cpu().tolist()
    if observed != [IDESC_E4M3_E2M1_F32_M64_N128_KMAJ, TMEM_ALLOC_COLS, 8, 2]:
        raise AssertionError(f"kernel metadata differs: {observed}")
    generated_fields = ""
    if generated is not None:
        generated_fields = (
            f" generated_ptx_mma={generated['ptx_mma']}"
            f" generated_sass_mma={generated['sass_mma']}"
            f" registers={generated['registers']} stack={generated['stack']}"
            f" static_shared={generated['static_shared']} local={generated['local']}"
            f" dynamic_shared={DYNAMIC_SMEM_BYTES}"
        )
    print(
        "PASS m64_ws=True qk=True q1_rows=8 q5_rows=40 "
        "packed_hbm=True u4_unpack_u8=True padded_smem=True "
        "inactive_row_poison=True packed_nibble_poison=True "
        "latent_query_poison=True rope_query_poison=True rope_key_poison=True "
        "distinct_row_token_signatures=True "
        "bf16_token_scale_poison=True bf16_token_postscale=True "
        "separate_qk_oracle=True separate_rope_oracle=True "
        "unit_scale_oracle=True fp8_rope_same_score_tile=True graph_replays=100 "
        f"graph_allocation_growth={allocation_after - allocation_before} "
        f"idesc=0x{observed[0]:08x} acc_cols={observed[1]} "
        f"mixed_k_blocks={observed[2]} rope_k_blocks={observed[3]} "
        f"interval_median_ns={statistics.median(samples):.1f} "
        f"interval_max_ns={max(samples)}{generated_fields}"
    )


if __name__ == "__main__":
    main()
