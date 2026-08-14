#!/usr/bin/env python3
"""SM100 mixed-PV overlap x compact-scale setup factorial.

Four production-shaped mixed arms isolate TMA/scale-setup overlap and live
scale initialization width.  A launch-matched ordinary N256 arm anchors the
remaining component gap.  This is a component attribution and optimization
probe, not an endpoint or production benchmark.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream, make_ptr
from cutlass.experimental import primitives as prims

import probe_sm100_tq_direct_mixed_pv_l0 as mixed_l0


THREADS_PER_CTA = mixed_l0.THREADS_PER_CTA
TMEM_RETRIEVE_THREADS = mixed_l0.TMEM_RETRIEVE_THREADS
CLUSTER_SHAPE_MNK = mixed_l0.CLUSTER_SHAPE_MNK
ROWS = mixed_l0.ROWS
TOKENS = mixed_l0.TOKENS
LATENT = mixed_l0.LATENT
OUTPUT_CHUNK = 256
ROWS_PER_CTA = ROWS // CLUSTER_SHAPE_MNK[0]
K_TILE = 128
ORDINARY_TMEM_COLS = 128
ORDINARY_P_SMEM_BYTES = 8 * 1024
ORDINARY_V_SMEM_BYTES = 16 * 1024

MIXED_CONFIGS = {
    "mixed_serial64": (0, 4),
    "mixed_serial16": (0, 1),
    "mixed_overlap64": (1, 4),
    "mixed_overlap16": (1, 1),
}
ARMS = tuple(MIXED_CONFIGS) + ("native_n256",)
CONDITIONS = ARMS


@cute.struct
class OrdinarySharedStorage:
    init_mbar: cutlass.Int64
    tma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


def make_ordinary_pv_mma(n_tile: int):
    return sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        (ROWS, n_tile),
    )


@cute.kernel
def ordinary_pv_kernel(
    layout_output: cute.Tensor,
    matrix_output: cute.Tensor,
    ordinary_mma: cute.TiledMma,
    tma_atom_p: cute.CopyAtom,
    tma_tensor_p: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    tma_tensor_v: cute.Tensor,
    p_layout: cute.ComposedLayout,
    v_layout: cute.ComposedLayout,
    output_cols: cutlass.Constexpr[int],
    cta_layout_vmnk: cute.Layout,
    N_TILE: cutlass.Constexpr[int],
    STAGES: cutlass.Constexpr[int],
    LATENT_PAIR: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_global, _, _ = cute.arch.block_idx()
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    cluster_index = cute.arch.make_warp_uniform(
        cta_global // CLUSTER_SHAPE_MNK[0]
    )

    smem = utils.SmemAllocator()
    storage = smem.allocate(OrdinarySharedStorage)
    p_smem = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        p_layout.outer,
        byte_alignment=128,
        swizzle=p_layout.inner,
    )
    v_smem = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        v_layout.outer,
        byte_alignment=128,
        swizzle=v_layout.inner,
    )

    mma_tile_coord = cta_rank
    cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank)
    tiler_mnk = (ROWS, N_TILE, K_TILE)
    g_p_mkl = cute.local_tile(
        tma_tensor_p,
        cute.slice_(tiler_mnk, (None, 0, None)),
        (None, None, None),
    )
    g_v_nkl = cute.local_tile(
        tma_tensor_v,
        cute.slice_(tiler_mnk, (0, None, None)),
        (None, None, None),
    )
    ordinary_thr_mma = ordinary_mma.get_slice(mma_tile_coord)
    t_cg_p = ordinary_thr_mma.partition_A(g_p_mkl)
    t_cg_v = ordinary_thr_mma.partition_B(g_v_nkl)
    a_cta_layout = cute.make_layout(
        cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape
    )
    b_cta_layout = cute.make_layout(
        cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape
    )
    t_ps_p, t_pg_p = cpasync.tma_partition(
        tma_atom_p,
        cta_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(p_smem, 0, 3),
        cute.group_modes(t_cg_p, 0, 3),
    )
    t_vs_v, t_vg_v = cpasync.tma_partition(
        tma_atom_v,
        cta_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(v_smem, 0, 3),
        cute.group_modes(t_cg_v, 0, 3),
    )
    t_pg_p = t_pg_p[(None, 0, None, None)]

    p_copy_bytes = cute.size_in_bytes(
        cutlass.Float8E4M3FN,
        cute.slice_(p_smem, (None, None, None, 0)),
    ) * cute.size(ordinary_mma.thr_id.shape)
    v_copy_bytes = cute.size_in_bytes(
        cutlass.Float8E4M3FN,
        cute.slice_(v_smem, (None, None, None, 0)),
    ) * cute.size(ordinary_mma.thr_id.shape)
    tma_barriers = pipeline.MbarrierArray(
        storage.tma_mbar.data_ptr(),
        STAGES,
        (
            pipeline.PipelineOp.TmaLoad,
            pipeline.CooperativeGroup(pipeline.Agent.Thread),
        ),
        tx_count=0,
    )
    mma_producer, mma_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, THREADS_PER_CTA * CLUSTER_SHAPE_MNK[0]
        ),
        barrier_storage=storage.mma_mbar.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
    ).make_participants()

    retrieve_barrier = pipeline.NamedBarrier(
        barrier_id=1, num_threads=TMEM_RETRIEVE_THREADS
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=retrieve_barrier,
        allocator_warp_id=8,
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )
    if tidx == 0:
        cute.arch.mbarrier_init(storage.init_mbar.ptr, 1)
    pipeline.pipeline_init_arrive(
        cluster_shape_mn=CLUSTER_SHAPE_MNK[:2], is_relaxed=True
    )
    pipeline.pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE_MNK[:2])
    tmem.allocate(ORDINARY_TMEM_COLS)
    if warp_idx <= 8:
        tmem.wait_for_alloc()
    cute.arch.sync_threads()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    prims.fence_mbarrier_init()
    cute.arch.sync_threads()

    p_operand = ordinary_mma.make_fragment_A(p_smem)
    v_operand = ordinary_mma.make_fragment_B(v_smem)
    k_blocks = cute.size(p_operand, mode=[2])
    if cutlass.const_expr(k_blocks != cute.size(v_operand, mode=[2])):
        raise ValueError(
            f"ordinary PV K-block mismatch: A={k_blocks} "
            f"B={cute.size(v_operand, mode=[2])}"
        )
    acc_fake = ordinary_mma.make_fragment_C(
        ordinary_mma.partition_shape_C(tiler_mnk[:2])
    )

    if tidx == 0 and cluster_index == 0:
        layout_output[cta_rank, 0] = cta_rank + 1
        layout_output[cta_rank, 1] = cute.size_in_bytes(
            cutlass.Float8E4M3FN, p_smem
        )
        layout_output[cta_rank, 2] = cute.size_in_bytes(
            cutlass.Float8E4M3FN, v_smem
        )
        layout_output[cta_rank, 3] = p_copy_bytes
        layout_output[cta_rank, 4] = v_copy_bytes
        layout_output[cta_rank, 5] = N_TILE
        layout_output[cta_rank, 6] = STAGES
        layout_output[cta_rank, 7] = k_blocks
        layout_output[cta_rank, 8] = output_cols
        layout_output[cta_rank, 9] = ORDINARY_TMEM_COLS
        layout_output[cta_rank, 10] = LATENT_PAIR
        layout_output[cta_rank, 11] = p_copy_bytes + STAGES * v_copy_bytes

    # Issue both N128 V stages before MMA 0 so stage 1 can overlap it.  The
    # distinct barriers are waited only at their matching consumer boundary.
    if warp_idx == 9:
        for stage in cutlass.range_constexpr(STAGES):
            tma_bar_ptr = tma_barriers.get_barrier(stage)
            stage_bytes = v_copy_bytes
            if cutlass.const_expr(stage == 0):
                stage_bytes += p_copy_bytes
            tma_barriers.arrive_and_expect_tx(stage, stage_bytes)
            if cutlass.const_expr(stage == 0):
                cute.copy(
                    tma_atom_p,
                    t_pg_p[(None, 0, cluster_index)],
                    t_ps_p[(None, 0)],
                    tma_bar_ptr=tma_bar_ptr,
                )
            v_tile = t_vg_v[
                (
                    None,
                    LATENT_PAIR * STAGES + stage,
                    None,
                    None,
                )
            ]
            cute.copy(
                tma_atom_v,
                v_tile[(None, 0, cluster_index)],
                t_vs_v[(None, stage)],
                tma_bar_ptr=tma_bar_ptr,
            )

    if warp_idx == 8 and cta_rank == 0:
        mma_producer.acquire_and_advance()
        for stage in cutlass.range_constexpr(STAGES):
            tma_barriers.wait(stage, 0)
            accumulator = cute.make_tensor(
                tmem_ptr + stage * output_cols, acc_fake.layout
            )
            ordinary_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(k_blocks, unroll_full=True):
                cute.gemm(
                    ordinary_mma,
                    accumulator,
                    p_operand[None, None, k_block, 0],
                    v_operand[None, None, k_block, stage],
                    accumulator,
                )
                ordinary_mma.set(tcgen05.Field.ACCUMULATE, True)
        mma_producer.commit()

    mma_full = mma_consumer.wait_and_advance()
    mma_full.release()
    cute.arch.sync_threads()

    if tidx < 128:
        for stage in cutlass.range_constexpr(STAGES):
            accumulator = cute.make_tensor(
                tmem_ptr + stage * output_cols, acc_fake.layout
            )
            t_acc = accumulator[(None, None), 0, 0]
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(
                    tcgen05.copy.Repetition(output_cols // 4)
                ),
                cutlass.Float32,
            )
            tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, t_acc)
            thr_load = tmem_load.get_slice(tidx)
            g_output = cute.make_tensor(
                matrix_output.iterator
                + cta_global * OUTPUT_CHUNK * ROWS_PER_CTA
                + stage * N_TILE * ROWS_PER_CTA,
                cute.make_layout(
                    (ROWS_PER_CTA, N_TILE),
                    stride=(1, ROWS_PER_CTA),
                ),
            )
            t_tmem = thr_load.partition_S(t_acc)
            t_gmem = thr_load.partition_D(g_output)
            r_acc = cute.make_fragment_like(t_gmem, cutlass.Float32)
            cute.copy(tmem_load, t_tmem, r_acc)
            cute.arch.fence_view_async_tmem_load()
            cute.autovec_copy(r_acc, t_gmem)
    cute.arch.sync_threads()

    if warp_idx == 8 and cta_rank == 0:
        mma_producer.tail()
    cute.arch.sync_threads()
    if warp_idx == 8:
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def ordinary_pv_probe(
    p_ptr: cute.Pointer,
    native_v_ptr: cute.Pointer,
    layout_output: cute.Tensor,
    matrix_output: cute.Tensor,
    CLUSTERS: cutlass.Constexpr[int],
    N_TILE: cutlass.Constexpr[int],
    LATENT_PAIR: cutlass.Constexpr[int],
    stream,
):
    stages = OUTPUT_CHUNK // N_TILE
    tiler_mnk = (ROWS, N_TILE, K_TILE)
    ordinary_mma = make_ordinary_pv_mma(N_TILE)
    g_p = cute.make_tensor(
        p_ptr,
        cute.make_ordered_layout((ROWS, TOKENS, CLUSTERS), order=(1, 0, 2)),
    )
    g_native_v = cute.make_tensor(
        native_v_ptr,
        cute.make_ordered_layout((TOKENS, LATENT, CLUSTERS), order=(1, 0, 2)),
    )
    g_native_v_transpose = cute.make_tensor(
        g_native_v.iterator,
        cute.select(g_native_v.layout, mode=[1, 0, 2]),
    )
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (ordinary_mma.thr_id.shape,)
    )
    p_layout = sm100_utils.make_smem_layout_a(
        ordinary_mma,
        tiler_mnk,
        cutlass.Float8E4M3FN,
        1,
    )
    v_layout = sm100_utils.make_smem_layout_b(
        ordinary_mma,
        tiler_mnk,
        cutlass.Float8E4M3FN,
        stages,
    )
    if cutlass.const_expr(
        cute.size_in_bytes(cutlass.Float8E4M3FN, p_layout)
        != ORDINARY_P_SMEM_BYTES
    ):
        raise ValueError(
            "ordinary P SMEM footprint changed: "
            f"{cute.size_in_bytes(cutlass.Float8E4M3FN, p_layout)}"
        )
    if cutlass.const_expr(
        cute.size_in_bytes(cutlass.Float8E4M3FN, v_layout)
        != ORDINARY_V_SMEM_BYTES
    ):
        raise ValueError(
            "ordinary V SMEM footprint changed: "
            f"{cute.size_in_bytes(cutlass.Float8E4M3FN, v_layout)}"
        )
    a_op = sm100_utils.cluster_shape_to_tma_atom_A(
        CLUSTER_SHAPE_MNK[:2], ordinary_mma.thr_id
    )
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(
        CLUSTER_SHAPE_MNK[:2], ordinary_mma.thr_id
    )
    tma_atom_p, tma_tensor_p = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        g_p,
        cute.slice_(p_layout, (None, None, None, 0)),
        tiler_mnk,
        ordinary_mma,
        cta_layout_vmnk.shape,
    )
    tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        g_native_v_transpose,
        cute.slice_(v_layout, (None, None, None, 0)),
        tiler_mnk,
        ordinary_mma,
        cta_layout_vmnk.shape,
    )
    output_fragment = ordinary_mma.make_fragment_C(
        ordinary_mma.partition_shape_C(tiler_mnk[:2])
    )
    output_cols = utils.get_num_tmem_alloc_cols(output_fragment)
    if cutlass.const_expr(output_cols * stages != ORDINARY_TMEM_COLS):
        raise ValueError(
            f"ordinary output TMEM changed: {output_cols}*{stages} "
            f"!= {ORDINARY_TMEM_COLS}"
        )
    print(f"S4_C1A_I1_N{N_TILE}_THR_ID={ordinary_mma.thr_id}")
    print(f"S4_C1A_I1_N{N_TILE}_P_LAYOUT={p_layout}")
    print(f"S4_C1A_I1_N{N_TILE}_V_LAYOUT={v_layout}")
    print(
        f"S4_C1A_I1_N{N_TILE}_RESOURCE_PLAN="
        f"p_bytes={cute.size_in_bytes(cutlass.Float8E4M3FN, p_layout)} "
        f"v_bytes={cute.size_in_bytes(cutlass.Float8E4M3FN, v_layout)} "
        f"stages={stages} output_cols={output_cols}"
    )
    kernel = ordinary_pv_kernel(
        layout_output,
        matrix_output,
        ordinary_mma,
        tma_atom_p,
        tma_tensor_p,
        tma_atom_v,
        tma_tensor_v,
        p_layout,
        v_layout,
        output_cols,
        cta_layout_vmnk,
        N_TILE,
        stages,
        LATENT_PAIR,
    )
    kernel.launch(
        grid=(CLUSTER_SHAPE_MNK[0] * CLUSTERS, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
        cluster=CLUSTER_SHAPE_MNK,
        min_blocks_per_mp=1,
        stream=stream,
    )


def compile_mixed(arm: str, clusters: int, latent_tile: int):
    overlap_tma, scale_init_tiles = MIXED_CONFIGS[arm]
    ctas = CLUSTER_SHAPE_MNK[0] * clusters
    return cute.compile(
        mixed_l0.direct_mixed_pv_probe,
        make_ptr(
            cutlass.Float8E4M3FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float4E2M1FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        mixed_l0.fake(cutlass.Int32, (CLUSTER_SHAPE_MNK[0], 13), 16),
        mixed_l0.fake(
            cutlass.Float32,
            (ctas, OUTPUT_CHUNK, ROWS_PER_CTA),
            16,
        ),
        clusters,
        1,
        1,
        1,
        latent_tile,
        0,
        0,
        overlap_tma,
        scale_init_tiles,
        make_fake_stream(),
        options="--enable-tvm-ffi --opt-level 3",
    )


def compile_ordinary(clusters: int, n_tile: int, latent_pair: int):
    ctas = CLUSTER_SHAPE_MNK[0] * clusters
    return cute.compile(
        ordinary_pv_probe,
        make_ptr(
            cutlass.Float8E4M3FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        mixed_l0.fake(cutlass.Int32, (CLUSTER_SHAPE_MNK[0], 12), 16),
        mixed_l0.fake(
            cutlass.Float32,
            (ctas, OUTPUT_CHUNK, ROWS_PER_CTA),
            16,
        ),
        clusters,
        n_tile,
        latent_pair,
        make_fake_stream(),
        options="--enable-tvm-ffi --opt-level 3",
    )


def walsh_sign(keys: torch.Tensor, coordinates: torch.Tensor, bits: int):
    masks = torch.bitwise_and(keys.view(-1, 1), coordinates.view(1, -1))
    parity = torch.zeros_like(masks)
    for bit in range(bits):
        parity.bitwise_xor_((masks >> bit) & 1)
    return (1 - 2 * parity).float()


def condition_name(condition: str) -> str:
    return condition


def williams_orders(windows: int) -> list[tuple[str, ...]]:
    if windows not in (10, 30):
        raise ValueError("Williams timing supports 10 or 30 windows")
    # Odd-order Williams design: five cyclic shifts plus their reverses.  Each
    # ten-row block balances every position and every directed adjacency twice.
    base = (0, 1, 4, 2, 3)
    forward_rows = [
        tuple(CONDITIONS[(index + shift) % len(CONDITIONS)] for index in base)
        for shift in range(len(CONDITIONS))
    ]
    rows = forward_rows + [tuple(reversed(row)) for row in forward_rows]
    orders = rows * (windows // len(rows))

    expected = windows // len(CONDITIONS)
    for condition in CONDITIONS:
        for position in range(len(CONDITIONS)):
            observed = sum(order[position] == condition for order in orders)
            if observed != expected:
                raise AssertionError(
                    f"Williams position imbalance condition={condition} "
                    f"position={position} observed={observed} expected={expected}"
                )
    for first in CONDITIONS:
        for second in CONDITIONS:
            if first == second:
                continue
            observed = sum(
                sum(
                    order[position] == first
                    and order[position + 1] == second
                    for position in range(len(CONDITIONS) - 1)
                )
                for order in orders
            )
            if observed != expected:
                raise AssertionError(
                    f"Williams adjacency imbalance first={first} second={second} "
                    f"observed={observed} expected={expected}"
                )
    return orders


def paired_log_ratio_summary(
    numerators: list[float], denominators: list[float]
) -> dict[str, object]:
    log_ratios = [
        math.log(numerator / denominator)
        for numerator, denominator in zip(numerators, denominators)
    ]
    count = len(log_ratios)
    if count == 30:
        t_critical_95 = 2.045229642
    elif count == 10:
        t_critical_95 = 2.262157163
    else:
        raise ValueError(f"unsupported paired sample count: {count}")
    mean_log = statistics.fmean(log_ratios)
    standard_error = statistics.stdev(log_ratios) / math.sqrt(count)
    ci_low = math.exp(mean_log - t_critical_95 * standard_error)
    ci_high = math.exp(mean_log + t_critical_95 * standard_error)
    return {
        "paired_geomean_ratio": math.exp(mean_log),
        "paired_log_ci95": [ci_low, ci_high],
        "samples": count,
    }


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_sha256(path: str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def cubin_resource_usage(cubin: bytes) -> dict[str, int]:
    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
        cubin_file.write(cubin)
        cubin_file.flush()
        result = subprocess.run(
            ["cuobjdump", "-res-usage", cubin_file.name],
            check=True,
            capture_output=True,
            text=True,
        )
    matches = re.findall(
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", result.stdout
    )
    if len(matches) != 1:
        raise AssertionError(
            f"expected one cuobjdump resource record, observed {len(matches)}"
        )
    registers, stack, shared, local = (int(value) for value in matches[0])
    if registers > 160 or stack != 0 or local != 0:
        raise AssertionError(
            "generated resource gate failed: "
            f"registers={registers} stack={stack} local={local}"
        )
    return {
        "registers": registers,
        "stack_bytes": stack,
        "static_shared_bytes": shared,
        "local_bytes": local,
    }


def audit_compiled_wrapper(arm: str, latent_tile: int, wrapper) -> dict[str, object]:
    ptx = getattr(wrapper, "__ptx__", None)
    sass = getattr(wrapper, "__sass__", None)
    cubin = getattr(wrapper, "__cubin__", None)
    mlir = getattr(wrapper, "__mlir__", None)
    if not isinstance(ptx, str) or not isinstance(sass, str):
        raise AssertionError(
            "generated audit requires CUTE_DSL_KEEP=all before process start"
        )
    if not isinstance(cubin, bytes) or not isinstance(mlir, str):
        raise AssertionError(
            "generated audit requires in-memory cubin and MLIR artifacts"
        )

    mma_lines = re.findall(r"^\s*tcgen05\.mma[^\n]+;\s*$", ptx, re.M)
    tma_count = ptx.count("cp.async.bulk.tensor")
    if len(mma_lines) != 4 or tma_count != 2:
        raise AssertionError(
            f"{arm}/tile{latent_tile} opcode count failed: "
            f"mma={len(mma_lines)} tma={tma_count}"
        )
    predicate_states = []
    for mma_line in mma_lines:
        predicate_match = re.search(r",\s*(%p\d+)\s*;\s*$", mma_line)
        if predicate_match is None:
            raise AssertionError(f"cannot parse MMA accumulate predicate: {mma_line}")
        predicate = predicate_match.group(1)
        assignments = re.findall(
            rf"mov\.pred\s+{re.escape(predicate)},\s*(-?1|0)\s*;", ptx
        )
        if len(assignments) != 1:
            raise AssertionError(
                f"cannot uniquely resolve {predicate}: assignments={assignments}"
            )
        predicate_states.append(int(assignments[0]))
    if predicate_states != [0] + [-1] * 3:
        raise AssertionError(
            f"{arm}/tile{latent_tile} accumulate fields changed: {predicate_states}"
        )

    mixed_opcode_count = ptx.count(
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32"
    )
    ordinary_opcode_count = ptx.count(
        "tcgen05.mma.cta_group::2.kind::f8f6f4 ["
    )
    scale_store_pattern = "tcgen05.st.sync.aligned.32x32b.x4.b32"
    scale_store_offsets = [
        match.start() for match in re.finditer(re.escape(scale_store_pattern), ptx)
    ]
    scale_wait_offsets = [
        match.start()
        for match in re.finditer(
            re.escape("tcgen05.wait::st.sync.aligned;"), ptx
        )
    ]
    tma_offsets = [
        match.start() for match in re.finditer("cp.async.bulk.tensor", ptx)
    ]
    mbarrier_init_offsets = [
        match.start() for match in re.finditer("mbarrier.init.shared.b64", ptx)
    ]
    regular_cluster_arrive_offsets = [
        match.start()
        for match in re.finditer(r"barrier\.cluster\.arrive;", ptx)
    ]
    cluster_wait_offsets = [
        match.start()
        for match in re.finditer(r"barrier\.cluster\.wait;", ptx)
    ]

    schedule = "native"
    if arm in MIXED_CONFIGS:
        overlap_tma, scale_init_tiles = MIXED_CONFIGS[arm]
        expected_scale_stores = scale_init_tiles * 4
        if mixed_opcode_count != 4 or ordinary_opcode_count != 0:
            raise AssertionError(
                f"mixed opcode audit failed: mixed={mixed_opcode_count} "
                f"ordinary={ordinary_opcode_count}"
            )
        if len(scale_store_offsets) != expected_scale_stores:
            raise AssertionError(
                f"{arm}/tile{latent_tile} scale stores changed: "
                f"{len(scale_store_offsets)} != {expected_scale_stores}"
            )
        if len(scale_wait_offsets) != 1:
            raise AssertionError(
                f"{arm}/tile{latent_tile} scale waits={len(scale_wait_offsets)}"
            )
        if len(regular_cluster_arrive_offsets) != 1 or len(cluster_wait_offsets) != 2:
            raise AssertionError(
                f"{arm}/tile{latent_tile} cluster publication changed: "
                f"arrive={len(regular_cluster_arrive_offsets)} "
                f"wait={len(cluster_wait_offsets)}"
            )
        if not mbarrier_init_offsets or max(mbarrier_init_offsets) >= min(tma_offsets):
            raise AssertionError(
                f"{arm}/tile{latent_tile} TMA precedes mbarrier initialization"
            )
        if overlap_tma:
            if max(tma_offsets) >= min(scale_store_offsets):
                raise AssertionError(
                    f"{arm}/tile{latent_tile} does not place TMA before scale stores"
                )
            schedule = "tma_before_scale_store"
        else:
            if min(tma_offsets) <= max(
                scale_wait_offsets + regular_cluster_arrive_offsets + cluster_wait_offsets
            ):
                raise AssertionError(
                    f"{arm}/tile{latent_tile} does not serialize TMA after publication"
                )
            schedule = "tma_after_scale_publication"
    elif mixed_opcode_count != 0 or ordinary_opcode_count != 4:
        raise AssertionError(
            f"ordinary opcode audit failed: mixed={mixed_opcode_count} "
            f"ordinary={ordinary_opcode_count}"
        )
    elif scale_store_offsets or scale_wait_offsets or regular_cluster_arrive_offsets:
        raise AssertionError(f"native arm unexpectedly contains mixed scale setup")

    return {
        "arm": arm,
        "latent_tile": latent_tile,
        "mma_count": len(mma_lines),
        "tma_count": tma_count,
        "accumulate_predicate_states": predicate_states,
        "scale_store_count": len(scale_store_offsets),
        "scale_wait_count": len(scale_wait_offsets),
        "regular_cluster_arrive_count": len(regular_cluster_arrive_offsets),
        "cluster_wait_count": len(cluster_wait_offsets),
        "schedule": schedule,
        "mixed_blockscaled_opcode_count": mixed_opcode_count,
        "ordinary_unscaled_opcode_count": ordinary_opcode_count,
        "resources": cubin_resource_usage(cubin),
        "sha256": {
            "cubin": sha256_bytes(cubin),
            "ptx": sha256_bytes(ptx.encode()),
            "sass": sha256_bytes(sass.encode()),
            "mlir": sha256_bytes(mlir.encode()),
        },
    }


def audit_generated_code(compiled: dict[str, tuple[object, object]]):
    wrappers = []
    for arm in ARMS:
        for latent_tile, wrapper in enumerate(compiled[arm]):
            wrappers.append(audit_compiled_wrapper(arm, latent_tile, wrapper))
    return {
        "status": "PASS_S4_C1A_I1_GENERATED_AUDIT",
        "source_sha256": {
            "i1": source_sha256(__file__),
            "mixed_l0_parameterized_for_i1": source_sha256(mixed_l0.__file__),
        },
        "wrappers": wrappers,
    }


def summarize_timings(
    samples: dict[str, list[float]], raw_windows: list[dict[str, object]]
):
    arm_summary = {
        name: {
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "samples_ms": values,
        }
        for name, values in samples.items()
    }
    ratio_pairs = {
        "candidate_vs_native": ("mixed_overlap16", "native_n256"),
        "overlap_full": ("mixed_overlap64", "mixed_serial64"),
        "overlap_compact": ("mixed_overlap16", "mixed_serial16"),
        "compact_serial": ("mixed_serial16", "mixed_serial64"),
        "compact_overlap": ("mixed_overlap16", "mixed_overlap64"),
        "cumulative": ("mixed_overlap16", "mixed_serial64"),
    }
    ratios = {}
    for name, (numerator, denominator) in ratio_pairs.items():
        ratio = paired_log_ratio_summary(samples[numerator], samples[denominator])
        ratio["ratio_of_means"] = (
            arm_summary[numerator]["mean_ms"] / arm_summary[denominator]["mean_ms"]
        )
        ratio["formal_faster"] = ratio["paired_log_ci95"][1] < 1.0
        ratio["formal_parity_1p05"] = (
            ratio["ratio_of_means"] <= 1.05
            and ratio["paired_log_ci95"][1] <= 1.05
        )
        ratios[name] = ratio

    return {
        "arm_summary": arm_summary,
        "ratios": ratios,
        "population_component_parity": ratios["candidate_vs_native"][
            "formal_parity_1p05"
        ],
        "population_partial_cumulative": (
            ratios["cumulative"]["formal_faster"]
            and ratios["cumulative"]["ratio_of_means"] <= 0.99
        ),
    }


def make_inputs(nodes: int, clusters: int):
    samples = nodes * clusters
    sample_ids = torch.arange(samples, dtype=torch.int64)
    row = torch.arange(ROWS, dtype=torch.int64).view(ROWS, 1)
    token = torch.arange(TOKENS, dtype=torch.int64).view(1, TOKENS)
    latent = torch.arange(LATENT, dtype=torch.int64).view(1, LATENT)
    p_base = (
        ((((row + 1) * (token + 3) * 17) % 7) - 3).float() * 0.25
    )
    v_base = (
        ((((token.T + 5) * (latent + 7) * 19 + latent * 3) % 5) - 2).float()
        * 0.5
    )
    row_sign = walsh_sign(
        sample_ids.remainder(ROWS),
        torch.arange(ROWS, dtype=torch.int64),
        7,
    )
    latent_sign = walsh_sign(
        torch.div(sample_ids, ROWS, rounding_mode="floor"),
        torch.arange(LATENT, dtype=torch.int64),
        9,
    )
    p_reference = p_base.unsqueeze(0) * row_sign.unsqueeze(2)
    v_reference = v_base.unsqueeze(0) * latent_sign.unsqueeze(1)

    p_cute = mixed_l0.to_cute_tensor(p_reference, cutlass.Float8E4M3FN)
    mixed_v_cute = mixed_l0.to_cute_tensor(
        v_reference, cutlass.Float4E2M1FN
    )
    native_v_cute = mixed_l0.to_cute_tensor(
        v_reference, cutlass.Float8E4M3FN
    )

    expected_base = p_base @ v_base
    expected_rows = (
        expected_base.unsqueeze(0)
        * row_sign.unsqueeze(2)
        * latent_sign.unsqueeze(1)
    )
    expected_pair = torch.stack(
        (
            expected_rows[:, :ROWS_PER_CTA, :].permute(0, 2, 1),
            expected_rows[:, ROWS_PER_CTA:, :].permute(0, 2, 1),
        ),
        dim=1,
    ).reshape(samples * CLUSTER_SHAPE_MNK[0], LATENT, ROWS_PER_CTA)
    ctas = clusters * CLUSTER_SHAPE_MNK[0]
    expected_full = expected_pair.reshape(nodes, ctas, LATENT, ROWS_PER_CTA)
    expected = torch.stack(
        (
            expected_full[:, :, :OUTPUT_CHUNK, :],
            expected_full[:, :, OUTPUT_CHUNK:, :],
        ),
        dim=1,
    ).contiguous().cuda()
    return p_cute, mixed_v_cute, native_v_cute, expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--graph-replays", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup-cycles", type=int, default=10)
    parser.add_argument("--windows", type=int, default=30)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--arm", choices=("all",) + ARMS, default="all")
    parser.add_argument("--audit-generated", action="store_true")
    args = parser.parse_args()
    if args.clusters < 1 or args.nodes < 1:
        parser.error("--clusters and --nodes must be positive")
    if args.graph_replays < 0:
        parser.error("--graph-replays must be non-negative")
    if args.warmup_cycles < 0:
        parser.error("--warmup-cycles must be non-negative")
    if args.benchmark and args.windows not in (10, 30):
        parser.error("--benchmark supports ten smoke or 30 scored windows")
    if args.benchmark and args.arm != "all":
        parser.error("--benchmark requires --arm all")
    if args.audit_generated and args.arm != "all":
        parser.error("--audit-generated requires --arm all")
    scored_benchmark = args.benchmark and args.windows == 30
    if scored_benchmark and (
        args.nodes != 100
        or args.warmup_cycles != 10
        or args.clusters not in (128, 512)
    ):
        parser.error(
            "a 30-window scored benchmark requires --nodes 100, "
            "--warmup-cycles 10, and --clusters 128 or 512"
        )

    selected = ARMS if args.arm == "all" else (args.arm,)
    selected_conditions = selected
    compiled: dict[str, tuple[object, object]] = {}
    for arm in selected:
        if arm not in MIXED_CONFIGS:
            continue
        compiled[arm] = (
            compile_mixed(arm, args.clusters, 0),
            compile_mixed(arm, args.clusters, 1),
        )
    if "native_n256" in selected:
        compiled["native_n256"] = (
            compile_ordinary(args.clusters, 256, 0),
            compile_ordinary(args.clusters, 256, 1),
        )
    generated_audit = None
    if args.benchmark or args.audit_generated:
        generated_audit = audit_generated_code(compiled)
        print(
            "S4_C1A_I1_GENERATED_AUDIT_JSON="
            + json.dumps(generated_audit, sort_keys=True)
        )
    if args.compile_only:
        print(
            "PASS_S4_C1A_I1_COMPILE_ONLY "
            f"clusters={args.clusters} arms={','.join(selected)}"
        )
        return

    p_cute, mixed_v_cute, native_v_cute, expected = make_inputs(
        args.nodes, args.clusters
    )
    ctas = args.clusters * CLUSTER_SHAPE_MNK[0]
    outputs = torch.empty(
        (args.nodes, 2, ctas, OUTPUT_CHUNK, ROWS_PER_CTA),
        dtype=torch.float32,
        device="cuda",
    )
    mixed_layout = torch.zeros(
        (CLUSTER_SHAPE_MNK[0], 13), dtype=torch.int32, device="cuda"
    )
    ordinary_layout = torch.zeros(
        (CLUSTER_SHAPE_MNK[0], 12), dtype=torch.int32, device="cuda"
    )
    p_node_stride = args.clusters * ROWS * TOKENS
    v_node_stride = args.clusters * TOKENS * LATENT

    def issue_condition(arm: str, stream) -> None:
        wrappers = compiled[arm]
        for node in range(args.nodes):
            p_ptr = p_cute.iterator + node * p_node_stride
            if arm in MIXED_CONFIGS:
                v_ptr = mixed_v_cute.iterator + node * v_node_stride
                layout = mixed_layout
            else:
                v_ptr = native_v_cute.iterator + node * v_node_stride
                layout = ordinary_layout
            for tile in range(2):
                wrappers[tile](
                    p_ptr,
                    v_ptr,
                    layout,
                    outputs[node, tile],
                    stream,
                )

    def verify(arm: str, label: str) -> None:
        if torch.equal(outputs, expected):
            return
        actual_cpu = outputs.cpu()
        expected_cpu = expected.cpu()
        mismatch = (actual_cpu != expected_cpu).nonzero()[0].tolist()
        node, tile, cta, latent_idx, row_idx = mismatch
        max_abs = torch.max(torch.abs(actual_cpu - expected_cpu)).item()
        raise AssertionError(
            f"{label} {arm} mismatch node={node} "
            f"tile={tile} cta={cta} "
            f"latent={tile * OUTPUT_CHUNK + latent_idx} row={row_idx}: "
            f"actual={actual_cpu[node, tile, cta, latent_idx, row_idx].item()} "
            f"expected={expected_cpu[node, tile, cta, latent_idx, row_idx].item()} "
            f"max_abs={max_abs}"
        )

    default_stream = cuda_driver.CUstream(
        torch.cuda.current_stream().cuda_stream
    )
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for arm in selected_conditions:
        outputs.fill_(float("nan"))
        issue_condition(arm, default_stream)
        torch.cuda.synchronize()
        verify(arm, "eager")
        print(
            "PASS_S4_C1A_I1_EAGER "
            f"arm={arm} "
            f"clusters={args.clusters} nodes={args.nodes}"
        )
        if args.graph_replays or args.benchmark:
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                capture_stream = cuda_driver.CUstream(
                    torch.cuda.current_stream().cuda_stream
                )
                issue_condition(arm, capture_stream)
            graphs[arm] = graph
            for replay in range(args.graph_replays):
                outputs.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                verify(arm, f"graph replay {replay + 1}")
            if args.graph_replays:
                print(
                    "PASS_S4_C1A_I1_GRAPH "
                    f"arm={arm} "
                    f"clusters={args.clusters} nodes={args.nodes} "
                    f"replays={args.graph_replays}"
                )

    if not args.benchmark:
        return

    warmup_orders = williams_orders(10)
    for warmup_cycle in range(args.warmup_cycles):
        for arm in warmup_orders[warmup_cycle % len(warmup_orders)]:
            outputs.fill_(float("nan"))
            torch.cuda.synchronize()
            graphs[arm].replay()
            torch.cuda.synchronize()
            verify(arm, f"warmup cycle {warmup_cycle + 1}")

    orders = williams_orders(args.windows)
    samples: dict[str, list[float]] = {
        condition_name(condition): [] for condition in CONDITIONS
    }
    raw_windows = []
    for window, order in enumerate(orders, start=1):
        row = {
            "window": window,
            "order": [condition_name(condition) for condition in order],
            "timings_ms": {},
        }
        for arm in order:
            name = condition_name(arm)
            outputs.fill_(float("nan"))
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[arm].replay()
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)
            verify(arm, f"timed window {window}")
            samples[name].append(elapsed_ms)
            row["timings_ms"][name] = elapsed_ms
        raw_windows.append(row)

    timing_summary = summarize_timings(samples, raw_windows)
    result = {
        "status": (
            "PASS_S4_C1A_I1_TIMING_SCORED"
            if scored_benchmark
            else "S4_C1A_I1_TIMING_SMOKE_UNSCORED"
        ),
        "scored": scored_benchmark,
        "clusters": args.clusters,
        "nodes_per_graph": args.nodes,
        "warmup_cycles": args.warmup_cycles,
        "windows": args.windows,
        "arms": ARMS,
        "generated_audit": generated_audit,
        "timing_summary": timing_summary,
        "raw_windows": raw_windows,
    }
    print("S4_C1A_I1_TIMING_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
