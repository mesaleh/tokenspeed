# Copyright (c) 2026 LightSeek Foundation

"""Isolated Compute Sanitizer launcher for the R31-R5 mixed producer."""

import argparse
import json
from types import SimpleNamespace

import cutlass
import torch
from cutlass import Float32, Int32

import benchmark_mla_decode_tq_r31_mixed_native as benchmark


def _effective_splits(sequence_length: int, declared_splits: int) -> int:
    tiles = (sequence_length + 127) // 128
    if tiles == 0:
        return 0
    tiles_per_cta = (tiles + declared_splits - 1) // declared_splits
    return (tiles + tiles_per_cta - 1) // tiles_per_cta


def _cuda_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.cuda().contiguous()


def run(fixture_path: str) -> dict[str, object]:
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    batch = int(fixture["batch"])
    query_len = int(fixture["query_len"])
    hot_splits = int(fixture["hot_splits"])
    cold_splits = int(fixture["cold_splits"])
    benchmark.BATCH = batch
    benchmark.QUERY_LEN = query_len
    benchmark.HOT_SPLITS = hot_splits
    benchmark.COLD_SPLITS = cold_splits

    hot_query = _cuda_tensor(fixture["hot_query"])
    hot_cache = _cuda_tensor(fixture["hot_cache"])
    hot_table = _cuda_tensor(fixture["hot_table"])
    hot_seq = _cuda_tensor(fixture["hot_seq"])
    hot_causal_seq = _cuda_tensor(fixture["hot_causal_seq"])
    cold_query_latent = _cuda_tensor(fixture["cold_query_latent"])
    cold_query_rope = _cuda_tensor(fixture["cold_query_rope"])
    cold_cache = _cuda_tensor(fixture["cold_cache"])
    cold_high_rope = _cuda_tensor(fixture["cold_high_rope"])
    cold_table = _cuda_tensor(fixture["cold_table"])
    cold_seq = _cuda_tensor(fixture["cold_seq"])
    cold_causal_seq = _cuda_tensor(fixture["cold_causal_seq"])
    cold_scale = _cuda_tensor(fixture["cold_scale"])
    cold_residual_rope = _cuda_tensor(fixture["cold_residual_rope"])
    reference_workspace = _cuda_tensor(fixture["reference_workspace"])
    arena_layout = str(fixture.get("arena_layout", "contiguous"))
    if arena_layout not in ("contiguous", "layer-sharded"):
        raise ValueError(f"unsupported arena layout {arena_layout!r}")
    arena_backing = []
    if arena_layout == "layer-sharded":
        cold_state = SimpleNamespace(
            packed_cold=cold_cache,
            cold_scale=cold_scale,
            cold_high=cold_high_rope,
            cold_residual=cold_residual_rope,
        )
        arena_backing.append(
            benchmark._move_cold_to_layer_sharded_arena(cold_state)
        )
        cold_cache = cold_state.packed_cold
        cold_scale = cold_state.cold_scale
        cold_high_rope = cold_state.cold_high
        cold_residual_rope = cold_state.cold_residual
        hot_cache, hot_arena = benchmark._move_hot_to_layer_sharded_arena(
            hot_cache
        )
        arena_backing.append(hot_arena)
        arena_metadata = benchmark._arena_metadata(
            cold_state, hot_cache, arena_layout
        )
    else:
        arena_metadata = {
            "arena_layout": arena_layout,
            "hot_cache_stride": tuple(hot_cache.stride()),
            "cold_cache_strides": {
                "packed": tuple(cold_cache.stride()),
                "scale": tuple(cold_scale.stride()),
                "high": tuple(cold_high_rope.stride()),
                "residual": tuple(cold_residual_rope.stride()),
            },
        }
    arena_metadata["arena_bytes_allocated"] = sum(
        backing.numel() * backing.element_size() for backing in arena_backing
    )

    rows = batch * query_len * benchmark.HEADS
    hot_workspace_bytes = rows * hot_splits * (benchmark.LATENT + 1) * 4
    cold_workspace_bytes = rows * cold_splits * (benchmark.LATENT + 1) * 4
    workspace = torch.empty(
        hot_workspace_bytes + cold_workspace_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    hot_workspace = workspace[:hot_workspace_bytes]
    cold_workspace = workspace[hot_workspace_bytes:]
    output_sentinel = torch.empty(
        batch,
        query_len,
        benchmark.HEADS,
        benchmark.LATENT,
        dtype=torch.bfloat16,
        device="cuda",
    )
    compiled = benchmark._compile_mixed_producer(
        hot_query,
        hot_cache,
        hot_table,
        hot_workspace,
        hot_seq,
        cold_query_latent,
        cold_query_rope,
        cold_cache,
        cold_high_rope,
        cold_table,
        cold_workspace,
        cold_seq,
        output_sentinel,
        cold_scale,
        cold_residual_rope,
    )

    workspace.view(torch.float32).fill_(float("nan"))
    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        compiled(
            hot_query[..., : benchmark.LATENT],
            hot_query[..., benchmark.LATENT :],
            hot_cache[..., : benchmark.LATENT],
            hot_cache[..., benchmark.LATENT :],
            hot_table,
            hot_workspace,
            Int32(hot_splits),
            hot_seq,
            hot_causal_seq,
            cold_query_latent,
            cold_query_rope,
            cold_cache,
            cold_high_rope,
            cold_table,
            cold_workspace,
            Int32(cold_splits),
            cold_seq,
            cold_causal_seq,
            output_sentinel,
            Float32(benchmark.SOFTMAX_SCALE),
            Float32(1.0),
            cold_scale,
            cold_residual_rope,
        )
    torch.cuda.synchronize()

    candidate = workspace.view(torch.float32)
    reference = reference_workspace.view(torch.float32)
    rows_per_request = query_len * benchmark.HEADS
    hot_acc = candidate[: rows * hot_splits * benchmark.LATENT].view(
        rows, hot_splits, benchmark.LATENT
    )
    hot_lse_start = rows * hot_splits * benchmark.LATENT
    hot_region_end = rows * hot_splits * (benchmark.LATENT + 1)
    hot_lse = candidate[hot_lse_start:hot_region_end].view(rows, hot_splits)
    reference_hot_acc = reference[: rows * hot_splits * benchmark.LATENT].view(
        rows, hot_splits, benchmark.LATENT
    )
    reference_hot_lse = reference[hot_lse_start:hot_region_end].view(
        rows, hot_splits
    )

    cold_offset = rows * hot_splits * (benchmark.LATENT + 1)
    cold_acc = candidate[
        cold_offset : cold_offset + rows * cold_splits * benchmark.LATENT
    ].view(rows, cold_splits, benchmark.LATENT)
    cold_lse = candidate[
        cold_offset
        + rows * cold_splits * benchmark.LATENT : cold_offset
        + rows * cold_splits * (benchmark.LATENT + 1)
    ].view(rows, cold_splits)
    reference_cold_acc = reference[
        cold_offset : cold_offset + rows * cold_splits * benchmark.LATENT
    ].view(rows, cold_splits, benchmark.LATENT)
    reference_cold_lse = reference[
        cold_offset
        + rows * cold_splits * benchmark.LATENT : cold_offset
        + rows * cold_splits * (benchmark.LATENT + 1)
    ].view(rows, cold_splits)

    for request in range(batch):
        row_slice = slice(
            request * rows_per_request,
            (request + 1) * rows_per_request,
        )
        hot_active = _effective_splits(int(hot_seq[request].item()), hot_splits)
        cold_active = _effective_splits(int(cold_seq[request].item()), cold_splits)
        torch.testing.assert_close(
            hot_acc[row_slice, :hot_active],
            reference_hot_acc[row_slice, :hot_active],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            hot_lse[row_slice, :hot_active],
            reference_hot_lse[row_slice, :hot_active],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cold_acc[row_slice, :cold_active],
            reference_cold_acc[row_slice, :cold_active],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cold_lse[row_slice, :cold_active],
            reference_cold_lse[row_slice, :cold_active],
            rtol=0,
            atol=0,
        )

    schema = (
        "r31-r5-f2-exact-arena-sanitizer-v1"
        if arena_layout == "layer-sharded"
        else "r31-r5-f1-mixed-native-sanitizer-v1"
    )
    return {
        "schema": schema,
        "fixture": fixture_path,
        "shape": {
            "batch": batch,
            "query_len": query_len,
            "hot_splits": hot_splits,
            "cold_splits": cold_splits,
            **arena_metadata,
        },
        "active_workspace_bit_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture")
    args = parser.parse_args()
    print(json.dumps(run(args.fixture), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
