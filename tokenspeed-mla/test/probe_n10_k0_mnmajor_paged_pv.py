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

"""Prove paged direct B-MN TMA from canonical packed E2M1 on SM100.

The persistent allocation remains token-major ``[128, 512]`` packed E2M1,
identical to the accepted QK key allocation.  PV presents its first 256 latent
coordinates as a zero-copy logical ``[N=latent,page-offset,physical-page]`` B
operand.  Four page-table-selected logical copies load one K128 tile directly
into the B-MN shared-memory layout consumed by
``tcgen05.mma.ws.kind::f8f6f4``.  There is no expanded input, second cache
orientation, or scalar CTA-local V population.  Each page copy lowers to two
3-D TMA instructions because the M64 layout separates its N128 halves.

The probe also charges an exact scalar power-of-two P carrier correction in
the output epilogue.  Per-row production correction remains a C0 gate.  The
probe intentionally makes no serving or speed claim.
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
TOKENS = 128
PAGE_SIZE = 32
PHYSICAL_PAGES = 24
LATENT_K = 512
PV_N = 256
PV_TILER_MNK = (M, PV_N, TOKENS)
TMEM_ALLOC_COLS = 256
PACKED_B_SMEM_DTYPE = cutlass.Int8
MAX_DYNAMIC_SMEM_BYTES = 232448


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


# The lowered kernel allocates 40 bytes of barriers/allocator state followed
# by the FP8 P tile and E2M1 B-MN address span, each 128-byte aligned.  TMA
# transfers 16 KiB of packed values, while the two N128 halves occupy a sparse
# 32-KiB SMEM span in the one-CTA M64 layout.
_pv_smem_offset = _align_up(40, 128)
_pv_smem_offset = _align_up(_pv_smem_offset + 8192, 128)
DYNAMIC_SMEM_BYTES = _pv_smem_offset + 32768
if DYNAMIC_SMEM_BYTES != 41088:
    raise AssertionError(f"unexpected PV dynamic SMEM: {DYNAMIC_SMEM_BYTES}")
if DYNAMIC_SMEM_BYTES > MAX_DYNAMIC_SMEM_BYTES:
    raise AssertionError("PV dynamic SMEM exceeds the SM100 opt-in limit")


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


IDESC_E4M3_E2M1_F32_M64_N256_B_MN = make_f8f6f4_idesc(
    m=M,
    n=PV_N,
    a_format=0,  # E4M3
    b_format=5,  # E2M1
    c_format=1,  # F32
    b_major=1,  # MN-major
)
if IDESC_E4M3_E2M1_F32_M64_N256_B_MN != 0x04411410:
    raise AssertionError("the named-field M64/N256 descriptor changed")


@cute.struct
class SharedStorage:
    tma_mbar: cutlass.Int64
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
        (M, PV_N, 32),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        OperandMajorMode.K,
        OperandMajorMode.MN,
    )
    mixed = cute.make_tiled_mma(mixed_op)
    # Production maps M64/N256 output through an M128/N128 epilogue view and a
    # physical ((64,2),(64,2)) destination.  This preserves the raw WS TMEM
    # topology while exposing the complete logical M64xN256 output.
    epi = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        (M * 2, PV_N // 2),
    )
    return mixed, epi


@cute.jit
def make_paged_tma_atom_b(
    tma_load_op: cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp,
    gmem: cute.Tensor,
    smem_layout: cute.Layout,
    mixed_mma: cute.TiledMma,
):
    """Build a non-executable page-local packed-E2M1 TMA-B atom."""

    ident = cute.make_identity_layout(gmem.shape)
    g_tile = cute.composition(ident, (PV_N, TOKENS))
    cta_n = PV_N // mixed_mma.thr_id.shape
    cta_v_map = cute.flat_divide(g_tile, (cta_n,))
    cta_v_map = cute.select(cta_v_map, mode=[0, 2])
    cta_v_map = cute.zipped_divide(cta_v_map, (cta_n, PAGE_SIZE))
    cta_v_map = cute.select(cta_v_map, mode=[0])

    from cutlass._mlir.dialects import cute_nvgpu as cute_nvgpu_ir

    tma_format = cute_nvgpu_ir.TmaDataFormat(
        cute_nvgpu_ir.get_default_tma_format(
            cutlass.Float4E2M1FN.mlir_type, True
        )
    )
    result = cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
        gmem.value,
        smem_layout.value,
        cta_v_map,
        tma_load_op._to_ir(),
        num_multicast=1,
        tma_format=tma_format,
    )
    return (
        cute.CopyAtom(
            tma_load_op,
            cpasync.CopyBulkTensorTileG2SNonExecTrait(result[0]),
        ),
        result[1],
    )


@cute.kernel
def mixed_pv_kernel(
    output: cute.Tensor,
    metadata: cute.Tensor,
    carrier: cute.Tensor,
    elapsed_ns: cute.Tensor,
    mixed_mma: cute.TiledMma,
    epi_mma: cute.TiledMma,
    page_ids: cute.Tensor,
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    mixed_a_layout: cute.ComposedLayout,
    mixed_b_layout: cute.ComposedLayout,
    mixed_b_tma_layout: cute.ComposedLayout,
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
    s_mixed_b_tma = cute.make_tensor(
        s_mixed_b.iterator,
        mixed_b_tma_layout.outer,
    )

    g_a = cute.local_tile(
        tma_tensor_a,
        cute.slice_(PV_TILER_MNK, (None, 0, None)),
        (None, None, None),
    )
    thr_mma = mixed_mma.get_slice(0)
    t_cg_a = thr_mma.partition_A(g_a)
    a_cta_layout = cute.make_layout(cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape)
    t_as_a, t_ag_a = cpasync.tma_partition(
        tma_atom_a,
        cta_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(s_mixed_a, 0, 3),
        cute.group_modes(t_cg_a, 0, 3),
    )
    t_ag_a = t_ag_a[(None, 0, None, 0)]

    g_b = cute.flat_divide(tma_tensor_b, (PV_N, PAGE_SIZE))
    g_b = cute.logical_divide(g_b, (PV_N,))[(None, 0), None, None, None, None]
    tiled_g_b = cute.tiled_divide(g_b, (PV_N, PAGE_SIZE))
    tiled_g_b = tiled_g_b[None, 0, 0, None, None, None]
    t_bs_b, t_bg_b = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        s_mixed_b_tma,
        tiled_g_b,
    )

    copy_bytes = cute.size_in_bytes(
        cutlass.Float8E4M3FN,
        cute.slice_(s_mixed_a, (None, None, None, 0)),
    )
    b_copy_bytes = cute.size_in_bytes(
        cutlass.Float4E2M1FN,
        cute.slice_(s_mixed_b, (None, None, None, 0)),
    )
    charged_bytes = copy_bytes + b_copy_bytes
    tma_barrier = pipeline.MbarrierArray(
        storage.tma_mbar.ptr,
        1,
        (
            pipeline.PipelineOp.TmaLoad,
            pipeline.CooperativeGroup(pipeline.Agent.Thread),
        ),
        tx_count=charged_bytes,
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
    acc_shape = mixed_mma.partition_shape_C(PV_TILER_MNK[:2])
    acc_fake = mixed_mma.make_fragment_C(acc_shape)
    acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
    epi_shape = epi_mma.partition_shape_C((M * 2, PV_N // 2))
    epi_fake = epi_mma.make_fragment_C(epi_shape)
    epi_acc = cute.make_tensor(tmem_ptr, epi_fake.layout)
    k_blocks = cute.size(mixed_a, mode=[2])

    pipeline.pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE_MNK[:2])
    cute.arch.sync_threads()
    start_ns = cutlass.Int64(0)
    if tidx == 0:
        start_ns = cute.arch.globaltimer()
    if warp_idx == 0:
        bar = tma_barrier.get_barrier(0)
        tma_barrier.arrive_and_expect_tx(0, charged_bytes)
        cute.copy(tma_atom_a, t_ag_a[(None, 0)], t_as_a[(None, 0)], tma_bar_ptr=bar)
        for logical_page in cutlass.range_constexpr(TOKENS // PAGE_SIZE):
            physical_page = page_ids[logical_page]
            cute.copy(
                tma_atom_b,
                t_bg_b[None, 0, 0, physical_page],
                t_bs_b[None, 0, logical_page, 0],
                tma_bar_ptr=bar,
            )

    if warp_idx == 0:
        tma_barrier.wait(0, 0)
        mma_producer.acquire_and_advance()
        for k_block in cutlass.range(k_blocks, unroll_full=True):
            tcgen05_mma_ws_f8f6f4_one(
                mixed_a[None, None, k_block, 0],
                mixed_b[None, None, k_block, 0],
                acc,
                Int32(IDESC_E4M3_E2M1_F32_M64_N256_B_MN),
                Boolean(k_block != 0),
            )
        mma_producer.commit()
    ready = mma_consumer.wait_and_advance()
    ready.release()
    cute.arch.sync_threads()

    t_acc = epi_acc[(None, None), 0, 0]
    load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
        cutlass.Float32,
    )
    tmem_load = tcgen05.make_tmem_copy(load_atom, t_acc)
    load_thr = tmem_load.get_slice(tidx)
    t_src = load_thr.partition_S(t_acc)
    output_view = cute.make_tensor(
        output.iterator,
        cute.make_layout((M, PV_N), stride=(PV_N, 1)),
    )
    output_physical = cute.make_tensor(
        output_view.iterator,
        cute.make_layout(
            ((64, 2), (64, 2)),
            stride=((output_view.stride[0], 128), (output_view.stride[1], 64)),
        ),
    )
    output_coords = cute.make_identity_tensor((128, 128))
    t_dst = load_thr.partition_D(output_physical)
    r_layout = load_thr.partition_D(output_coords)
    regs = cute.make_fragment_like(r_layout, cutlass.Float32)
    cute.copy(tmem_load, t_src, regs)
    cute.arch.fence_view_async_tmem_load()
    correction = carrier[0]
    for element in cutlass.range(cute.size(regs), vectorize=True, unroll_full=True):
        regs[element] = regs[element] * correction
    cute.autovec_copy(regs, t_dst)
    cute.arch.sync_threads()

    if tidx == 0:
        metadata[0] = IDESC_E4M3_E2M1_F32_M64_N256_B_MN
        metadata[1] = utils.get_num_tmem_alloc_cols(acc_fake)
        metadata[2] = k_blocks
        metadata[3] = charged_bytes
        elapsed_ns[0] = cute.arch.globaltimer() - start_ns

    if warp_idx == 0:
        mma_producer.tail()
    tmem.relinquish_alloc_permit()
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def mixed_pv_probe(
    p_ptr: cute.Pointer,
    packed_cache_ptr: cute.Pointer,
    page_ids: cute.Tensor,
    output: cute.Tensor,
    metadata: cute.Tensor,
    carrier: cute.Tensor,
    elapsed_ns: cute.Tensor,
    stream,
):
    mixed_mma, epi_mma = make_tiled_mmas()
    g_p = cute.make_tensor(
        p_ptr,
        cute.make_ordered_layout((M, TOKENS, 1), order=(1, 0, 2)),
    )
    # The allocation is physically [physical-page, page-offset, latent].  This
    # logical B view swaps only CuTe modes: adjacent N coordinates remain
    # adjacent in storage, page-local K advances by one latent row, and the
    # runtime page table selects the physical-page coordinate.
    g_b = cute.make_tensor(
        packed_cache_ptr,
        cute.make_layout(
            (PV_N, PAGE_SIZE, PHYSICAL_PAGES),
            stride=(1, LATENT_K, LATENT_K * PAGE_SIZE),
        ),
    )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (mixed_mma.thr_id.shape,)
    )
    mixed_a_layout = sm100_utils.make_smem_layout_a(
        mixed_mma,
        PV_TILER_MNK,
        cutlass.Float8E4M3FN,
        1,
    )
    mixed_b_layout = sm100_utils.make_smem_layout_b(
        mixed_mma,
        PV_TILER_MNK,
        PACKED_B_SMEM_DTYPE,
        1,
    )
    mixed_b_tma_layout = sm100_utils.make_smem_layout(
        OperandMajorMode.MN,
        (PV_N, TOKENS),
        PACKED_B_SMEM_DTYPE,
        1,
    )
    mixed_b_tma_layout = cute.tiled_divide(
        mixed_b_tma_layout,
        (PV_N, PAGE_SIZE),
    )
    if cutlass.const_expr(os.environ.get("TQ_S6_PRINT_LAYOUTS") == "1"):
        print(f"S6_MIXED_B_LAYOUT={mixed_b_layout}")
        print(f"S6_MIXED_B_OUTER={mixed_b_layout.outer}")
        print(f"S6_MIXED_B_INNER={mixed_b_layout.inner}")
        print(f"S6_MIXED_B_TMA_LAYOUT={mixed_b_tma_layout}")
    tma_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load,
        g_p,
        cute.slice_(mixed_a_layout, (None, None, None, 0)),
        PV_TILER_MNK,
        mixed_mma,
        cta_layout_vmnk.shape,
    )
    tma_atom_b, tma_tensor_b = make_paged_tma_atom_b(
        tma_load,
        g_b,
        cute.select(mixed_b_tma_layout, mode=[0]),
        mixed_mma,
    )
    mixed_pv_kernel(
        output,
        metadata,
        carrier,
        elapsed_ns,
        mixed_mma,
        epi_mma,
        page_ids,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        mixed_a_layout,
        mixed_b_layout,
        mixed_b_tma_layout,
        cta_layout_vmnk,
    ).launch(
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
        mixed_pv_probe,
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_fake_compact_tensor(
            cutlass.Int32,
            (TOKENS // PAGE_SIZE,),
            stride_order=(0,),
            assumed_align=4,
        ),
        make_fake_compact_tensor(
            cutlass.Float32, (M, PV_N), stride_order=(1, 0), assumed_align=16
        ),
        make_fake_compact_tensor(
            cutlass.Int32, (4,), stride_order=(0,), assumed_align=4
        ),
        make_fake_compact_tensor(
            cutlass.Float32, (1,), stride_order=(0,), assumed_align=4
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
    if not isinstance(ptx, str):
        raise AssertionError("set CUTE_DSL_KEEP=ptx before generated-code audit")
    if not isinstance(cubin, bytes) or not isinstance(mlir, str):
        raise AssertionError("generated CUBIN/MLIR artifacts were not retained")

    ptx_mma = ptx.count("tcgen05.mma.ws.cta_group::1.kind::f8f6f4")
    ptx_tma_2d = ptx.count("cp.async.bulk.tensor.2d.shared::cta.global.tile")
    ptx_tma_3d = ptx.count("cp.async.bulk.tensor.3d.shared::cta.global.tile")
    ptx_tma = ptx_tma_2d + ptx_tma_3d
    if ptx_mma != 4 or ptx_tma_2d != 1 or ptx_tma_3d != 8:
        raise AssertionError(
            "PV instruction count differs: "
            f"MMA={ptx_mma} TMA2D={ptx_tma_2d} TMA3D={ptx_tma_3d}"
        )
    if any(
        opcode in ptx
        for opcode in ("ld.global.u8", "ld.global.s8", "ld.global.b8")
    ):
        raise AssertionError("paged packed-V path contains a scalar global byte load")
    for allocation in ('"8192:1"', '"32768:1"'):
        if allocation not in mlir:
            raise AssertionError(f"PV lowered SMEM allocation missing: {allocation}")

    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
        cubin_file.write(cubin)
        cubin_file.flush()
        if not isinstance(sass, str):
            sass = subprocess.run(
                ["cuobjdump", "--dump-sass", cubin_file.name],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        resource_output = subprocess.run(
            ["cuobjdump", "--dump-resource-usage", cubin_file.name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    sass_mma = sass.count(" UTCQMMA.WS")
    sass_tma_2d = sass.count(" UTMALDG.2D")
    sass_tma_3d = sass.count(" UTMALDG.3D")
    sass_tma = sass_tma_2d + sass_tma_3d
    if sass_mma != 4 or sass_tma_2d != 1 or sass_tma_3d != 8:
        raise AssertionError(
            "PV SASS instruction count differs: "
            f"MMA={sass_mma} TMA2D={sass_tma_2d} TMA3D={sass_tma_3d}"
        )
    matches = re.findall(
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resource_output
    )
    if len(matches) != 1:
        raise AssertionError(f"expected one PV resource record, got {matches}")
    registers, stack, static_shared, local = (int(value) for value in matches[0])
    if registers > 160 or stack != 0 or local != 0:
        raise AssertionError(
            "PV resource gate failed: "
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
            output_path = output_dir / f"pv.{suffix}"
            (
                output_path.write_bytes(payload)
                if binary
                else output_path.write_text(payload)
            )
    return {
        "ptx_mma": ptx_mma,
        "ptx_tma": ptx_tma,
        "ptx_tma_2d": ptx_tma_2d,
        "ptx_tma_3d": ptx_tma_3d,
        "sass_mma": sass_mma,
        "sass_tma": sass_tma,
        "sass_tma_2d": sass_tma_2d,
        "sass_tma_3d": sass_tma_3d,
        "registers": registers,
        "stack": stack,
        "static_shared": static_shared,
        "local": local,
    }


def _build_case(
    *,
    active_rows: int,
    inactive_poison: float,
    carrier_value: float,
    p_poison_coord: tuple[int, int] | None = None,
    cache_poison_coord: tuple[int, int] | None = None,
    unused_poison: float = 0.5,
    seq_len: int = TOKENS,
    page_ids_values: tuple[int, ...] = (23, 5, 17, 2),
    page_padding_poison: float = 6.0,
):
    if active_rows not in (8, 40):
        raise ValueError(f"unsupported active row count: {active_rows}")
    if not 1 <= seq_len <= TOKENS:
        raise ValueError(f"sequence length is out of range: {seq_len}")
    if len(page_ids_values) != TOKENS // PAGE_SIZE:
        raise ValueError(f"expected four page IDs, got {page_ids_values}")
    if len(set(page_ids_values)) != len(page_ids_values):
        raise ValueError(f"page IDs must be distinct: {page_ids_values}")
    if min(page_ids_values) < 0 or max(page_ids_values) >= PHYSICAL_PAGES:
        raise ValueError(f"physical page ID is out of range: {page_ids_values}")
    rows = torch.arange(M, device="cuda", dtype=torch.int64)
    tokens = torch.arange(TOKENS, device="cuda", dtype=torch.int64)
    latent = torch.arange(LATENT_K, device="cuda", dtype=torch.int64)
    p_codebook = torch.tensor([0.0, 56.0, 112.0, 224.0], device="cuda")
    e2m1_codebook = torch.tensor(
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
    )
    p_indices = (
        (
            (rows[:, None] + 1) * (tokens[None, :] + 3) * 17
            + rows[:, None] * 37
            + tokens[None, :] * 19
        )
        % 257
        + (
            (rows[:, None] + 5) * (tokens[None, :] + 11) * 31
            + rows[:, None] * 43
            + tokens[None, :] * 47
        )
        % 263
    ) % 4
    cache_indices = (
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
    ) % 16
    if torch.unique(cache_indices[:, :PV_N]).numel() != 16:
        raise AssertionError("the oracle does not exercise all 16 E2M1 codes")
    if not bool(torch.signbit(e2m1_codebook[8]).item()):
        raise AssertionError("the E2M1 codebook lost its negative-zero code")
    p_source = p_codebook[p_indices]
    cache_source = e2m1_codebook[cache_indices]
    # Four base-4 digits make every one of the 128 token columns explicit in
    # the active P rows.  The remaining rows retain the aperiodic cross-term
    # pattern, so this ownership guard does not reduce the full-matrix oracle
    # to a basis-only case.
    for digit in range(4):
        p_source[digit, :] = p_codebook[(tokens // (4**digit)) % 4]
    p_source[active_rows:, :] = inactive_poison
    if os.environ.get("TQ_S6_P_BASIS") == "1":
        basis_token = int(os.environ.get("TQ_S6_P_BASIS_TOKEN", "23"))
        if not 0 <= basis_token < TOKENS:
            raise ValueError(f"basis token is out of range: {basis_token}")
        p_source.zero_()
        p_source[0, basis_token] = 1.0
    else:
        p_source[:, seq_len:] = 0.0
    cache_source[:, PV_N:] = unused_poison
    if p_poison_coord is not None:
        row, token = p_poison_coord
        old_value = float(p_source[row, token].item())
        p_source[row, token] = 112.0 if old_value != 112.0 else 56.0
    if cache_poison_coord is not None:
        token, coordinate = cache_poison_coord
        old_value = float(cache_source[token, coordinate].item())
        cache_source[token, coordinate] = 6.0 if old_value != 6.0 else 0.5
    p = _to_cute_tensor(p_source.view(M, TOKENS, 1), cutlass.Float8E4M3FN)
    physical_cache_source = torch.full(
        (PHYSICAL_PAGES, PAGE_SIZE, LATENT_K),
        page_padding_poison,
        device="cuda",
        dtype=torch.float32,
    )
    for logical_page, physical_page in enumerate(page_ids_values):
        logical_begin = logical_page * PAGE_SIZE
        logical_end = min(logical_begin + PAGE_SIZE, seq_len)
        valid_tokens = max(0, logical_end - logical_begin)
        if valid_tokens:
            physical_cache_source[physical_page, :valid_tokens] = cache_source[
                logical_begin:logical_end
            ]
    cache = _to_cute_tensor(
        physical_cache_source.view(
            PHYSICAL_PAGES * PAGE_SIZE, LATENT_K, 1
        ),
        cutlass.Float4E2M1FN,
    )
    page_ids = torch.tensor(page_ids_values, device="cuda", dtype=torch.int32)
    expected = (p_source[:, :seq_len] @ cache_source[:seq_len, :PV_N]) * carrier_value
    if seq_len == TOKENS and os.environ.get("TQ_S6_P_BASIS") != "1":
        if torch.unique(p_source[:active_rows], dim=0).shape[0] != active_rows:
            raise AssertionError("P oracle does not distinguish active rows")
        if torch.unique(p_source[:active_rows].T, dim=0).shape[0] != TOKENS:
            raise AssertionError("P oracle does not distinguish tokens")
        if torch.unique(cache_source[:, :PV_N], dim=0).shape[0] != TOKENS:
            raise AssertionError("V oracle does not distinguish tokens")
        if torch.unique(cache_source[:, :PV_N].T, dim=0).shape[0] != PV_N:
            raise AssertionError("V oracle does not distinguish output columns")
        if torch.unique(expected[:active_rows], dim=0).shape[0] != active_rows:
            raise AssertionError("PV oracle does not distinguish active rows")
        if torch.unique(expected[:active_rows].T, dim=0).shape[0] != PV_N:
            raise AssertionError("PV oracle does not distinguish output columns")
    carrier = torch.tensor([carrier_value], device="cuda", dtype=torch.float32)
    return p, cache, page_ids, carrier, expected


def _run_case(compiled, case, output, metadata, elapsed_ns, stream) -> torch.Tensor:
    p, cache, page_ids, carrier, expected = case
    output.fill_(float("nan"))
    compiled(
        p.iterator,
        cache.iterator,
        page_ids,
        output,
        metadata,
        carrier,
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
            f"output_r0_head={output[0, :16].tolist()} "
            f"expected_r0_head={expected[0, :16].tolist()}"
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
            f"idesc=0x{IDESC_E4M3_E2M1_F32_M64_N256_B_MN:08x}"
        )
        return

    output = torch.empty((M, PV_N), device="cuda", dtype=torch.float32)
    metadata = torch.empty(4, device="cuda", dtype=torch.int32)
    elapsed_ns = torch.empty(1, device="cuda", dtype=torch.int64)
    eager_stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    normal_cases = {}
    for active_rows in (8, 40):
        low_poison = _build_case(
            active_rows=active_rows,
            inactive_poison=0.5,
            carrier_value=1.0 / 256.0,
        )
        high_poison = _build_case(
            active_rows=active_rows,
            inactive_poison=2.0,
            carrier_value=1.0 / 256.0,
        )
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

    q5_case = normal_cases[40]
    q5_output = _run_case(compiled, q5_case, output, metadata, elapsed_ns, eager_stream)
    for p_row, p_token in ((3, 7), (19, 39), (37, 103)):
        p_poison = _build_case(
            active_rows=40,
            inactive_poison=0.5,
            carrier_value=1.0 / 256.0,
            p_poison_coord=(p_row, p_token),
        )
        p_output = _run_case(
            compiled, p_poison, output, metadata, elapsed_ns, eager_stream
        )
        changed_rows = (
            torch.nonzero((p_output != q5_output).any(dim=1)).flatten().tolist()
        )
        if changed_rows != [p_row]:
            raise AssertionError(
                f"one-coordinate P poison escaped row {p_row}: {changed_rows}"
            )

    for cache_token, cache_coordinate in (
        (7, 13),
        (39, 77),
        (71, 141),
        (103, 205),
    ):
        cache_poison = _build_case(
            active_rows=40,
            inactive_poison=0.5,
            carrier_value=1.0 / 256.0,
            cache_poison_coord=(cache_token, cache_coordinate),
        )
        cache_output = _run_case(
            compiled, cache_poison, output, metadata, elapsed_ns, eager_stream
        )
        changed_cols = (
            torch.nonzero((cache_output != q5_output).any(dim=0)).flatten().tolist()
        )
        if changed_cols != [cache_coordinate]:
            raise AssertionError(
                "one-coordinate packed-E2M1 poison escaped column "
                f"{cache_coordinate}: {changed_cols}"
            )

    unused_poison = _build_case(
        active_rows=40,
        inactive_poison=0.5,
        carrier_value=1.0 / 256.0,
        unused_poison=6.0,
    )
    unused_output = _run_case(
        compiled, unused_poison, output, metadata, elapsed_ns, eager_stream
    )
    if not torch.equal(unused_output, q5_output):
        raise AssertionError("unused packed latent half changed the N256 PV slice")

    permuted_pages = _build_case(
        active_rows=40,
        inactive_poison=0.5,
        carrier_value=1.0 / 256.0,
        page_ids_values=(4, 21, 0, 12),
    )
    permuted_output = _run_case(
        compiled, permuted_pages, output, metadata, elapsed_ns, eager_stream
    )
    if not torch.equal(permuted_output, q5_output):
        raise AssertionError("physical-page permutation changed the PV result")

    partial_outputs = {}
    for seq_len in (31, 32, 33):
        partial_case = _build_case(
            active_rows=40,
            inactive_poison=0.5,
            carrier_value=1.0 / 256.0,
            seq_len=seq_len,
        )
        partial_outputs[seq_len] = _run_case(
            compiled, partial_case, output, metadata, elapsed_ns, eager_stream
        )
    partial_padding_poison = _build_case(
        active_rows=40,
        inactive_poison=0.5,
        carrier_value=1.0 / 256.0,
        seq_len=33,
        page_padding_poison=0.5,
    )
    partial_padding_output = _run_case(
        compiled,
        partial_padding_poison,
        output,
        metadata,
        elapsed_ns,
        eager_stream,
    )
    if not torch.equal(partial_padding_output, partial_outputs[33]):
        raise AssertionError("partial-page padding poison changed the PV result")
    for carrier_value in (2.0**-16, 1.0 / 256.0, 1.0, 2.0**16):
        boundary_case = _build_case(
            active_rows=40,
            inactive_poison=0.5,
            carrier_value=carrier_value,
        )
        _run_case(compiled, boundary_case, output, metadata, elapsed_ns, eager_stream)

    p, cache, page_ids, carrier, expected = q5_case
    graph_output = torch.full_like(output, float("nan"))
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    graph_stream = cuda_driver.CUstream(capture_stream.cuda_stream)
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        compiled(
            p.iterator,
            cache.iterator,
            page_ids,
            graph_output,
            metadata,
            carrier,
            elapsed_ns,
            graph_stream,
        )
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=capture_stream):
        compiled(
            p.iterator,
            cache.iterator,
            page_ids,
            graph_output,
            metadata,
            carrier,
            elapsed_ns,
            graph_stream,
        )
    torch.cuda.synchronize()
    graph_output.fill_(float("nan"))
    allocation_before = torch.cuda.memory_allocated()
    for _ in range(100):
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
            p.iterator,
            cache.iterator,
            page_ids,
            output,
            metadata,
            carrier,
            elapsed_ns,
            eager_stream,
        )
        torch.cuda.synchronize()
        samples.append(int(elapsed_ns.item()))
    observed = metadata.cpu().tolist()
    if observed[:3] != [
        IDESC_E4M3_E2M1_F32_M64_N256_B_MN,
        TMEM_ALLOC_COLS,
        4,
    ]:
        raise AssertionError(f"kernel metadata differs: {observed}")
    generated_fields = ""
    if generated is not None:
        generated_fields = (
            f" generated_ptx_mma={generated['ptx_mma']}"
            f" generated_ptx_tma={generated['ptx_tma']}"
            f" generated_ptx_tma_2d={generated['ptx_tma_2d']}"
            f" generated_ptx_tma_3d={generated['ptx_tma_3d']}"
            f" generated_sass_mma={generated['sass_mma']}"
            f" generated_sass_tma={generated['sass_tma']}"
            f" generated_sass_tma_2d={generated['sass_tma_2d']}"
            f" generated_sass_tma_3d={generated['sass_tma_3d']}"
            f" registers={generated['registers']} stack={generated['stack']}"
            f" static_shared={generated['static_shared']} local={generated['local']}"
            f" dynamic_shared={DYNAMIC_SMEM_BYTES}"
        )
    print(
        "PASS m64_ws=True pv=True q1_rows=8 q5_rows=40 "
        "same_token_major_allocation=True logical_paged_b_mn_view=True "
        "packed_hbm=True packed_mn_tma_payload=True direct_v_tma=True "
        "inactive_row_poison=True p_coordinate_poison=True "
        "packed_nibble_poison=True distributed_coordinate_poisons=True "
        "distinct_row_token_column_signatures=True unused_half_poison=True "
        "all_e2m1_codes=True physical_page_permutation=True "
        "partial_pages=31,32,33 partial_page_padding_poison=True "
        "scalar_carrier_boundaries=True scalar_carrier_correction=True "
        "graph_replays=100 "
        f"graph_allocation_growth={allocation_after - allocation_before} "
        f"idesc=0x{observed[0]:08x} acc_cols={observed[1]} "
        f"mixed_k_blocks={observed[2]} copy_bytes={observed[3]} "
        f"full_interval_median_ns={statistics.median(samples):.1f} "
        f"full_interval_max_ns={max(samples)}{generated_fields}"
    )


if __name__ == "__main__":
    main()
