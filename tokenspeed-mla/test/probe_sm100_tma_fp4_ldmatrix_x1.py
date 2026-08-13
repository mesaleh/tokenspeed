"""Qualify compact-FP4 TMA -> padded narrow LdMatrix on SM100.

This began as the A17-N8-R1-D0 composition micro-gate.  C1-M0 extends it with
a two-CTA cluster load arm: one elected CTA issues a compact TensorMap load
with mask ``0b11``, and both CTA-local barriers and destinations must complete
with the same exact payload.  The independent arm is the matched duplicate-
request ablation.  The gate deliberately stops at a
bounded raw-register diagnostic: compact E2M1 is TMA-expanded into padded
``b4x16_p64`` shared memory, each 16x16 group is loaded through the raw
``m16n16.trans.b8x16.b4x16_p64`` wrapper path, and both 32-bit carrier words
per lane are published for an independent CPU lane-map oracle.

The source uses the CUTLASS 4.7 experimental API and must be run only through
the isolated 4.7 environment recorded by the experiment.  It does not modify
the cache, populate TMEM, issue PV MMA, or claim endpoint performance.
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims

_ROWS = 16
_K = 128
_PACKED_ROW_BYTES = _K // 2
_PADDED_ROW_BYTES = _K
_GROUPS = _K // 16
_WARP_SIZE = 32
_WORDS_PER_LANE = 2


@cute.kernel
def kernel(
    tma_src_desc: cutlass.GridConstant[cuda.TensorMap],
    dst: cute.Tensor,
    POISON: cutlass.Constexpr[int],
    MULTICAST: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    cta_global, _, _ = cute.arch.block_idx()
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    is_leader_cta = cta_rank == 0
    smem = cutlass.Array(
        cutlass.Int8,
        _ROWS * _PADDED_ROW_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )

    if tidx == 0:
        prims.mbarrier_init(mbar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    if tidx == 0:
        prims.mbarrier_arrive_expect_tx(mbar, tma_src_desc.global_tx_bytes())

    # A multicast signals the barrier at this same SMEM offset in every
    # destination CTA.  Do not let the leader issue until both CTAs have
    # initialized and armed their local barriers.  Keep the same cluster
    # synchronization in the independent arm so its counter path is matched.
    prims.barrier_cta_sync(0)
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()

    if tidx == 0:
        if cutlass.const_expr(MULTICAST == 1):
            if is_leader_cta:
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    smem,
                    tma_src_desc.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(0)],
                    mbar,
                    [],
                    multicast_mask=cutlass.Int16(0b11),
                )
        else:
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem,
                tma_src_desc.get_ptr(),
                (cutlass.Int32(0), cutlass.Int32(0)),
                mbar,
            )

    while not prims.mbarrier_try_wait_parity(
        mbar, cutlass.Int32(0), time_limit=10_000_000
    ):
        pass
    prims.barrier_cta_sync(0)

    # Each padded row contains eight 16-byte groups: eight bytes of packed FP4
    # followed by eight bytes that the narrow load must ignore.  Overwrite all
    # inserted padding after TMA so two poison runs prove that independence.
    for i in cutlass.range_constexpr((_ROWS * (_PADDED_ROW_BYTES // 2)) // _WARP_SIZE):
        flat = tidx + i * _WARP_SIZE
        row = flat // (_PADDED_ROW_BYTES // 2)
        within_row = flat % (_PADDED_ROW_BYTES // 2)
        group = within_row // 8
        byte = within_row % 8
        smem[row * _PADDED_ROW_BYTES + group * 16 + 8 + byte] = cutlass.Int8(POISON)
    prims.barrier_cta_sync(0)

    lane = tidx % _WARP_SIZE
    for group in cutlass.range_constexpr(_GROUPS):
        row_start = (lane % _ROWS) * _PADDED_ROW_BYTES + group * 16
        regs = prims.ldmatrix(
            smem.data_ptr() + row_start,
            _WORDS_PER_LANE,
            prims.MMALayout.COL,
            shape=prims.LoadShape.M16N16,
            src_format=prims.LoadSrcFormat.B4X16_P64,
        )
        dst[cta_global, group, lane, 0] = regs[0].to(cutlass.Uint32)
        dst[cta_global, group, lane, 1] = regs[1].to(cutlass.Uint32)


@cute.jit
def host(
    src: cute.Tensor,
    dst: cute.Tensor,
    POISON: cutlass.Constexpr[int],
    MULTICAST: cutlass.Constexpr[int],
    CLUSTERS: cutlass.Constexpr[int],
) -> None:
    tma_src = cuda.create_tensor_map_tiled(
        src.iterator.toint(),
        cutlass.Float4E2M1FN,
        global_dims=[_K, _ROWS],
        global_strides=[_PACKED_ROW_BYTES // 16],
        box_dims=[_K, _ROWS],
        swizzle=cuda.TensorMapSwizzle.none,
    )
    kernel(tma_src, dst, POISON, MULTICAST).launch(
        grid=(2 * CLUSTERS, 1, 1),
        block=(_WARP_SIZE, 1, 1),
        cluster=(2, 1, 1),
    )


@lru_cache(maxsize=None)
def compile_gate(poison: int, multicast: int, clusters: int):
    fake_src = make_fake_compact_tensor(
        cutlass.Uint8,
        (_ROWS, _PACKED_ROW_BYTES),
        stride_order=(1, 0),
        assumed_align=32,
    )
    fake_dst = make_fake_compact_tensor(
        cutlass.Uint32,
        (2 * clusters, _GROUPS, _WARP_SIZE, _WORDS_PER_LANE),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    return cute.compile(
        host,
        fake_src,
        fake_dst,
        poison,
        multicast,
        clusters,
        options="--enable-tvm-ffi",
    )


def _codes() -> torch.Tensor:
    row = torch.arange(_ROWS, dtype=torch.int64).view(_ROWS, 1)
    col = torch.arange(_K, dtype=torch.int64).view(1, _K)
    # The row and high column bits both affect the code.  Every 16x16 group
    # contains every E2M1 code in every column position across its rows.
    return ((row * 5 + col * 3 + (col // 16) * 7) & 0xF).to(torch.uint8)


def _packed_source(codes: torch.Tensor) -> torch.Tensor:
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous().cuda()


def _expected(codes: torch.Tensor) -> torch.Tensor:
    expected = torch.empty((_GROUPS, _WARP_SIZE, _WORDS_PER_LANE), dtype=torch.uint32)
    for group in range(_GROUPS):
        for lane in range(_WARP_SIZE):
            output_row = lane // 4
            output_col = (lane % 4) * 4
            for word in range(_WORDS_PER_LANE):
                latent = group * 16 + output_row + word * 8
                value = 0
                for byte in range(4):
                    token = output_col + byte
                    value |= int(codes[token, latent]) << (8 * byte)
                expected[group, lane, word] = value
    return expected


def run(
    poison: int, multicast: int, clusters: int
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = _codes()
    src = _packed_source(codes)
    dst = torch.full(
        (2 * clusters, _GROUPS, _WARP_SIZE, _WORDS_PER_LANE),
        0xDEADBEEF,
        dtype=torch.uint32,
        device="cuda",
    )
    compiled = compile_gate(poison, multicast, clusters)
    compiled(src, dst)
    torch.cuda.synchronize()
    return dst.cpu(), _expected(codes)


def verify(multicast: int, clusters: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("an SM100 GPU is required")

    first, expected = run(0x00, multicast, clusters)
    second, expected_second = run(0x5A, multicast, clusters)
    for cta in range(2 * clusters):
        torch.testing.assert_close(first[cta], expected, rtol=0, atol=0)
        torch.testing.assert_close(second[cta], expected_second, rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    observed_bytes = first.view(torch.uint8)
    assert not torch.any(first == 0xDEADBEEF), "diagnostic output retained poison"
    assert set(int(x) for x in torch.unique(observed_bytes)) == set(range(16))
    print(
        "PASS exact_tma_fp4_ldmatrix=True groups=8 lanes=32 words_per_lane=2 "
        f"codes=16 load={'multicast' if multicast else 'independent'} "
        f"clusters={clusters} all_ctas=True padding_poison_independent=True",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--load", choices=("multicast", "independent"), default="multicast"
    )
    parser.add_argument("--clusters", type=int, default=1)
    args = parser.parse_args()
    if args.clusters < 1:
        parser.error("--clusters must be positive")
    verify(1 if args.load == "multicast" else 0, args.clusters)
