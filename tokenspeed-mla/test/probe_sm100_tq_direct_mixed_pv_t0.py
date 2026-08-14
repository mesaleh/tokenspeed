#!/usr/bin/env python3
"""Launch-matched three-arm SM100 direct mixed-PV component gate.

Each invocation covers M128xN256.  The production-shape ordinary control
issues two N128 tile groups inside the same launch, so mixed N256, ordinary
N256, and ordinary N128 have identical full-512 launch and P-load counts.
"""

import argparse
import itertools
import json
import math
import statistics

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

ARMS = ("mixed_n256", "native_n256", "native_n128_pair")


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
    print(f"S4_C1A_T0_N{N_TILE}_THR_ID={ordinary_mma.thr_id}")
    print(f"S4_C1A_T0_N{N_TILE}_P_LAYOUT={p_layout}")
    print(f"S4_C1A_T0_N{N_TILE}_V_LAYOUT={v_layout}")
    print(
        f"S4_C1A_T0_N{N_TILE}_RESOURCE_PLAN="
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


def compile_mixed(clusters: int, latent_tile: int):
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


def summarize_timings(samples: dict[str, list[float]]):
    arm_summary = {
        arm: {
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "samples_ms": values,
        }
        for arm, values in samples.items()
    }
    ratio_pairs = (
        ("mixed_n256", "native_n256"),
        ("native_n256", "native_n128_pair"),
        ("mixed_n256", "native_n128_pair"),
    )
    ratios = {}
    for numerator, denominator in ratio_pairs:
        log_ratios = [
            math.log(num / den)
            for num, den in zip(samples[numerator], samples[denominator])
        ]
        mean_log = statistics.fmean(log_ratios)
        # The scored gate has 24 paired windows (df=23); six is reserved for
        # an unscored harness smoke test (df=5).
        t_critical_95 = 2.068657610 if len(log_ratios) == 24 else 2.570581836
        standard_error = statistics.stdev(log_ratios) / math.sqrt(
            len(log_ratios)
        )
        ratio_name = f"{numerator}/{denominator}"
        ratio_of_means = (
            arm_summary[numerator]["mean_ms"]
            / arm_summary[denominator]["mean_ms"]
        )
        ci_low = math.exp(mean_log - t_critical_95 * standard_error)
        ci_high = math.exp(mean_log + t_critical_95 * standard_error)
        ratios[ratio_name] = {
            "ratio_of_means": ratio_of_means,
            "paired_geomean_ratio": math.exp(mean_log),
            "paired_log_ci95": [ci_low, ci_high],
            "point_pass_1p10": ratio_of_means <= 1.10,
            "formal_pass_1p10": ratio_of_means <= 1.10 and ci_high <= 1.10,
            "preferred_point_pass_1p05": ratio_of_means <= 1.05,
        }
    return arm_summary, ratios


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
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--arm", choices=("all",) + ARMS, default="all")
    args = parser.parse_args()
    if args.clusters < 1 or args.nodes < 1:
        parser.error("--clusters and --nodes must be positive")
    if args.graph_replays < 0:
        parser.error("--graph-replays must be non-negative")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.windows < 6 or args.windows % 6:
        parser.error("--windows must be a positive multiple of six")
    if args.benchmark and args.windows not in (6, 24):
        parser.error("--benchmark supports six smoke or 24 scored windows")
    if args.benchmark and args.arm != "all":
        parser.error("--benchmark requires --arm all")
    scored_benchmark = args.benchmark and args.windows == 24
    if scored_benchmark and (
        args.nodes != 100
        or args.warmups != 20
        or args.clusters not in (128, 512)
    ):
        parser.error(
            "a 24-window scored benchmark requires --nodes 100, "
            "--warmups 20, and --clusters 128 or 512"
        )

    selected = ARMS if args.arm == "all" else (args.arm,)
    compiled: dict[str, tuple[object, object]] = {}
    if "mixed_n256" in selected:
        compiled["mixed_n256"] = (
            compile_mixed(args.clusters, 0),
            compile_mixed(args.clusters, 1),
        )
    if "native_n256" in selected:
        compiled["native_n256"] = (
            compile_ordinary(args.clusters, 256, 0),
            compile_ordinary(args.clusters, 256, 1),
        )
    if "native_n128_pair" in selected:
        compiled["native_n128_pair"] = (
            compile_ordinary(args.clusters, 128, 0),
            compile_ordinary(args.clusters, 128, 1),
        )
    if args.compile_only:
        print(
            "PASS_S4_C1A_T0_COMPILE_ONLY "
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

    def issue_arm(arm: str, stream) -> None:
        wrappers = compiled[arm]
        for node in range(args.nodes):
            p_ptr = p_cute.iterator + node * p_node_stride
            if arm == "mixed_n256":
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
            f"{label} {arm} mismatch node={node} tile={tile} cta={cta} "
            f"latent={tile * OUTPUT_CHUNK + latent_idx} row={row_idx}: "
            f"actual={actual_cpu[node, tile, cta, latent_idx, row_idx].item()} "
            f"expected={expected_cpu[node, tile, cta, latent_idx, row_idx].item()} "
            f"max_abs={max_abs}"
        )

    default_stream = cuda_driver.CUstream(
        torch.cuda.current_stream().cuda_stream
    )
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for arm in selected:
        outputs.fill_(float("nan"))
        issue_arm(arm, default_stream)
        torch.cuda.synchronize()
        verify(arm, "eager")
        print(
            "PASS_S4_C1A_T0_EAGER "
            f"arm={arm} clusters={args.clusters} nodes={args.nodes}"
        )
        if args.graph_replays or args.benchmark:
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                capture_stream = cuda_driver.CUstream(
                    torch.cuda.current_stream().cuda_stream
                )
                issue_arm(arm, capture_stream)
            graphs[arm] = graph
            for replay in range(args.graph_replays):
                outputs.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                verify(arm, f"graph replay {replay + 1}")
            if args.graph_replays:
                print(
                    "PASS_S4_C1A_T0_GRAPH "
                    f"arm={arm} clusters={args.clusters} nodes={args.nodes} "
                    f"replays={args.graph_replays}"
                )

    if not args.benchmark:
        return

    for warmup in range(args.warmups):
        for offset in range(len(ARMS)):
            arm = ARMS[(warmup + offset) % len(ARMS)]
            outputs.fill_(float("nan"))
            torch.cuda.synchronize()
            graphs[arm].replay()
            torch.cuda.synchronize()
            verify(arm, f"warmup {warmup + 1}")

    permutations = list(itertools.permutations(ARMS))
    orders = [
        permutations[window % len(permutations)]
        for window in range(args.windows)
    ]
    samples: dict[str, list[float]] = {arm: [] for arm in ARMS}
    raw_windows = []
    for window, order in enumerate(orders, start=1):
        row = {"window": window, "order": list(order), "timings_ms": {}}
        for arm in order:
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
            samples[arm].append(elapsed_ms)
            row["timings_ms"][arm] = elapsed_ms
        raw_windows.append(row)

    arm_summary, ratios = summarize_timings(samples)
    result = {
        "status": (
            "PASS_S4_C1A_T0_TIMING_SCORED"
            if scored_benchmark
            else "S4_C1A_T0_TIMING_SMOKE_UNSCORED"
        ),
        "scored": scored_benchmark,
        "clusters": args.clusters,
        "nodes_per_graph": args.nodes,
        "warmups_per_arm": args.warmups,
        "windows": args.windows,
        "arm_summary": arm_summary,
        "ratios": ratios,
        "raw_windows": raw_windows,
    }
    print("S4_C1A_T0_TIMING_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
