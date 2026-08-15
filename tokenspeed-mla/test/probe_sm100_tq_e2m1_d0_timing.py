#!/usr/bin/env python3

"""Decision-grade saturated timing gate for D0's page-32 E2M1 reader.

Each arm owns disjoint tensors containing ``nodes`` independent requests. The
D0 arms launch those requests as independent two-CTA clusters in one kernel;
the dense arm launches the installed production TokenSpeed FP8 MLA backend at
the same batch. This measures the component under saturation without turning a
serialized sequence of tiny kernels into an accidental launch-latency test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import statistics
from pathlib import Path

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
    make_ptr,
)
from tokenspeed_mla.mla_decode import tokenspeed_mla_decode

MODULE_PATH = Path(__file__).parent / "tokenspeed_mla" / "mla_decode_e2m1.py"
MODULE_SPEC = importlib.util.spec_from_file_location("mla_decode_e2m1", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load D0 module from {MODULE_PATH}")
d0 = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(d0)

CONTROL_ARMS = ("c0", "rn", "native")
ALL_ARMS = CONTROL_ARMS + ("dense",)
ARM_CONFIGS = {
    "c0": (0, 0, 0, 0, 0, 0),
    "rn": (0, 0, 1, 1, 1, 1),
    "native": (1, 1, 0, 0, 0, 0),
}
PHYSICAL_PAGES_PER_REQUEST = 37
SCORED_NODES = 100
SCORED_WINDOWS = 30
SCORED_WARMUP_WINDOWS = 24
SUPPORTED_WINDOWS = (6, SCORED_WINDOWS)
T_CRITICAL_95 = {6: 2.570581836, 30: 2.045229642}
BASE_PERMUTATION = torch.tensor(
    [31, 1, 36, 4, 22, 0, 26, 9, 3, 24, 6, 35, 20, 2, 25, 8, 23, 5, 27, 21],
    dtype=torch.int64,
)
WORKSPACE_BYTES = 16 * 1024 * 1024


def fake(dtype: type[cutlass.Numeric], shape: tuple[int, ...], align: int):
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )


def compile_reader(query_len: int, clusters: int, arm: str):
    active_rows = query_len * d0.NUM_HEADS
    physical_pages = clusters * PHYSICAL_PAGES_PER_REQUEST
    return cute.compile(
        d0.ownership_probe,
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        fake(cutlass.BFloat16, (clusters * active_rows, d0.LATENT_K), 16),
        fake(cutlass.Float32, (clusters * active_rows,), 16),
        fake(cutlass.BFloat16, (physical_pages, d0.PAGE_SIZE), 16),
        fake(cutlass.Int32, (clusters * d0.MAX_PAGES,), 16),
        cutlass.Int32(640),
        cutlass.Float32(d0.PROBE_SOFTMAX_SCALE_LOG2),
        active_rows,
        query_len,
        physical_pages,
        clusters,
        *ARM_CONFIGS[arm],
        1,
        0,
        0,
        make_fake_stream(),
        options="--enable-tvm-ffi --opt-level 3",
    )


def to_cute_tensor_and_storage(source: torch.Tensor, dtype):
    source_cuda = source.cuda().contiguous()
    result, storage = cutlass_torch.cute_tensor_like(
        source,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    result = cutlass_torch.convert_cute_tensor(
        source_cuda,
        result,
        dtype,
        is_dynamic_layout=True,
    )
    return result, storage


def make_population(query_len: int, population: int) -> tuple[torch.Tensor, ...]:
    active_rows = query_len * d0.NUM_HEADS
    phase = population * 31 + 7
    row = torch.arange(active_rows, dtype=torch.int64).view(active_rows, 1)
    token = torch.arange(d0.TOKENS, dtype=torch.int64).view(d0.TOKENS, 1)
    latent = torch.arange(d0.LATENT_K, dtype=torch.int64).view(1, d0.LATENT_K)
    rope = torch.arange(d0.ROPE_K, dtype=torch.int64).view(1, d0.ROPE_K)
    query = (
        (((row + 1) * (latent + 3) * 17 + row * 37 + latent * 19 + phase) % 257) % 3
    ).float()
    query = (query - 1.0) * 0.5
    key_base = (
        (((token + 1) * (latent + 5) * 23 + token * 41 + latent * 29 + phase * 3) % 263)
        % 3
    ).float()
    key_base = (key_base - 1.0) * 0.5
    key = key_base.unsqueeze(0) * torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0]).view(
        d0.TILES, 1, 1
    )
    rope_query = (
        (((row + 3) * (rope + 1) * 11 + row * 13 + phase * 5) % 127) % 3
    ).float()
    rope_query = (rope_query - 1.0) * 0.5
    rope_key_base = (
        (((token + 5) * (rope + 7) * 19 + token * 17 + phase * 11) % 131) % 3
    ).float()
    rope_key_base = (rope_key_base - 1.0) * 0.5
    rope_key = rope_key_base.unsqueeze(0) * torch.tensor(
        [1.0, 0.5, 2.0, 1.0, 1.0]
    ).view(d0.TILES, 1, 1)
    tile = torch.arange(d0.TILES, dtype=torch.int64).view(d0.TILES, 1)
    token_row = torch.arange(d0.TOKENS, dtype=torch.int64).view(1, d0.TOKENS)
    scale = (
        (0.75 + ((tile * 7 + token_row * 3 + population) % 17).float() / 32.0)
        * torch.tensor([1.0, 2.0, 0.5, 8.0, 4.0]).view(d0.TILES, 1)
    ).to(torch.bfloat16)
    return query, key, rope_query, rope_key, scale


def physical_pages(
    logical: torch.Tensor,
    permutation: torch.Tensor,
    fill_value: float,
) -> torch.Tensor:
    logical_pages = logical.reshape(d0.MAX_PAGES, d0.PAGE_SIZE, logical.shape[-1])
    result = torch.full(
        (
            PHYSICAL_PAGES_PER_REQUEST,
            d0.PAGE_SIZE,
            logical.shape[-1],
        ),
        fill_value,
        dtype=logical.dtype,
    )
    result[permutation] = logical_pages
    return result


def build_host_batch(query_len: int, nodes: int) -> dict[str, torch.Tensor]:
    queries = []
    key_pages = []
    native_logical = []
    rope_pages = []
    scale_pages = []
    block_tables = []
    fingerprints = set()
    for node in range(nodes):
        values = make_population(query_len, node)
        digest = hashlib.sha256()
        for value in values:
            digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        fingerprint = digest.hexdigest()
        if fingerprint in fingerprints:
            raise AssertionError(f"duplicate problem population at node {node}")
        fingerprints.add(fingerprint)
        query, key, rope_query, rope_key, scale = values
        permutation = (BASE_PERMUTATION + node * 7) % PHYSICAL_PAGES_PER_REQUEST
        key_page = physical_pages(key, permutation, 6.0)
        rope_page = physical_pages(rope_key, permutation, 7.0)
        scale_page = physical_pages(scale.reshape(-1, 1), permutation, 32.0).squeeze(-1)
        queries.append(torch.cat((query, rope_query), dim=-1))
        key_pages.append(key_page)
        native_logical.append(key.float() * scale.float().unsqueeze(-1))
        rope_pages.append(rope_page)
        scale_pages.append(scale_page)
        block_tables.append(permutation + node * PHYSICAL_PAGES_PER_REQUEST)
    return {
        "query": torch.stack(queries),
        "key_pages": torch.cat(key_pages),
        "native_logical": torch.stack(native_logical),
        "rope_pages": torch.cat(rope_pages),
        "scale_pages": torch.cat(scale_pages),
        "block_tables": torch.stack(block_tables),
        "fingerprints": torch.tensor([len(fingerprints)]),
    }


def build_runtime(query_len: int, nodes: int):
    host = build_host_batch(query_len, nodes)
    active_rows = query_len * d0.NUM_HEADS
    records: dict[str, dict[str, object]] = {}
    ranges: list[tuple[int, int, str]] = []

    def retain(arm: str, name: str, tensor: torch.Tensor) -> torch.Tensor:
        begin = tensor.data_ptr()
        ranges.append(
            (begin, begin + tensor.numel() * tensor.element_size(), f"{arm}:{name}")
        )
        return tensor

    for arm in CONTROL_ARMS:
        query, query_storage = to_cute_tensor_and_storage(
            host["query"], cutlass.Float8E4M3FN
        )
        packed, packed_storage = to_cute_tensor_and_storage(
            host["key_pages"], cutlass.Float4E2M1FN
        )
        native, native_storage = to_cute_tensor_and_storage(
            host["native_logical"], cutlass.Float8E4M3FN
        )
        rope, rope_storage = to_cute_tensor_and_storage(
            host["rope_pages"], cutlass.Float8E4M3FN
        )
        records[arm] = {
            "query": query,
            "query_storage": retain(arm, "query", query_storage),
            "packed": packed,
            "packed_storage": retain(arm, "packed", packed_storage),
            "native": native,
            "native_storage": retain(arm, "native", native_storage),
            "rope": rope,
            "rope_storage": retain(arm, "rope", rope_storage),
            "scale": retain(arm, "scale", host["scale_pages"].cuda().contiguous()),
            "block_table": retain(
                arm,
                "block_table",
                host["block_tables"]
                .to(device="cuda", dtype=torch.int32)
                .contiguous()
                .flatten(),
            ),
            "out": retain(
                arm,
                "out",
                torch.empty(
                    (nodes * active_rows, d0.LATENT_K),
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
            ),
            "lse": retain(
                arm,
                "lse",
                torch.empty(
                    (nodes * active_rows,),
                    dtype=torch.float32,
                    device="cuda",
                ),
            ),
        }

    dense_query = retain(
        "dense",
        "query",
        host["query"]
        .reshape(nodes, query_len, d0.NUM_HEADS, d0.QUERY_DIM)
        .to(device="cuda", dtype=torch.float8_e4m3fn)
        .contiguous(),
    )
    decoded_pages = torch.empty_like(host["key_pages"], dtype=torch.float32)
    for node in range(nodes):
        begin = node * PHYSICAL_PAGES_PER_REQUEST
        end = begin + PHYSICAL_PAGES_PER_REQUEST
        decoded_pages[begin:end] = host["key_pages"][begin:end].float() * host[
            "scale_pages"
        ][begin:end].float().unsqueeze(-1)
    records["dense"] = {
        "query": dense_query,
        "cache": retain(
            "dense",
            "cache",
            torch.cat((decoded_pages, host["rope_pages"]), dim=-1)
            .to(device="cuda", dtype=torch.float8_e4m3fn)
            .contiguous(),
        ),
        "workspace": retain(
            "dense",
            "workspace",
            torch.empty(WORKSPACE_BYTES, dtype=torch.int8, device="cuda"),
        ),
        "block_table": retain(
            "dense",
            "block_table",
            host["block_tables"].to(device="cuda", dtype=torch.int32).contiguous(),
        ),
        "seq_lens": retain(
            "dense",
            "seq_lens",
            torch.full((nodes,), 640, dtype=torch.int32, device="cuda"),
        ),
        "out": retain(
            "dense",
            "out",
            torch.empty(
                (nodes, query_len, d0.NUM_HEADS, d0.LATENT_K),
                dtype=torch.bfloat16,
                device="cuda",
            ),
        ),
    }
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise AssertionError(f"allocation overlap: {previous[2]} and {current[2]}")
    return records, ranges, int(host["fingerprints"].item())


def launch_control(reader, record: dict[str, object], stream) -> None:
    reader(
        record["query"].iterator,
        record["packed"].iterator,
        record["native"].iterator,
        record["query"].iterator + d0.LATENT_K,
        record["rope"].iterator,
        record["out"],
        record["lse"],
        record["scale"],
        record["block_table"],
        cutlass.Int32(640),
        cutlass.Float32(d0.PROBE_SOFTMAX_SCALE_LOG2),
        stream,
    )


def launch_dense(record: dict[str, object]) -> None:
    tokenspeed_mla_decode(
        record["query"],
        record["cache"],
        record["workspace"],
        d0.LATENT_K,
        d0.ROPE_K,
        record["block_table"],
        record["seq_lens"],
        640,
        d0.PROBE_SOFTMAX_SCALE_LOG2 / math.log2(math.e),
        out=record["out"],
        is_var_seq=True,
        causal_mask=True,
    )


def poison(record: dict[str, object], arm: str) -> None:
    record["out"].fill_(float("nan"))
    if arm != "dense":
        record["lse"].fill_(float("nan"))


def snapshot(record: dict[str, object], arm: str):
    out = record["out"]
    if not torch.isfinite(out.float()).all():
        raise AssertionError(f"{arm} left a non-finite output")
    if arm == "dense":
        return out.clone(), None
    lse = record["lse"]
    if not torch.isfinite(lse).all():
        raise AssertionError(f"{arm} left a non-finite LSE")
    return out.clone(), lse.clone()


def verify(record: dict[str, object], arm: str, expected) -> None:
    expected_out, expected_lse = expected
    if not torch.equal(record["out"].view(torch.int16), expected_out.view(torch.int16)):
        raise AssertionError(f"{arm} output drifted")
    if arm != "dense" and not torch.equal(record["lse"], expected_lse):
        raise AssertionError(f"{arm} LSE drifted")


def three_arm_orders(windows: int) -> list[tuple[str, ...]]:
    if windows == 0:
        return []
    if windows % 6:
        raise ValueError("three-arm windows must be a multiple of six")
    return list(itertools.permutations(CONTROL_ARMS)) * (windows // 6)


def pair_orders(windows: int) -> list[tuple[str, str]]:
    if windows == 0:
        return []
    if windows % 2:
        raise ValueError("paired windows must be even")
    return [("rn", "dense"), ("dense", "rn")] * (windows // 2)


def paired_log_summary(
    numerators: list[float], denominators: list[float]
) -> dict[str, float | int]:
    logs = [
        math.log(numerator / denominator)
        for numerator, denominator in zip(numerators, denominators)
    ]
    count = len(logs)
    mean_log = statistics.fmean(logs)
    standard_error = statistics.stdev(logs) / math.sqrt(count)
    t_critical = T_CRITICAL_95[count]
    return {
        "count": count,
        "geometric_ratio": math.exp(mean_log),
        "ci95_low": math.exp(mean_log - t_critical * standard_error),
        "ci95_high": math.exp(mean_log + t_critical * standard_error),
    }


def measure_graph(
    graph: torch.cuda.CUDAGraph,
    record: dict[str, object],
    arm: str,
    expected,
    nodes: int,
) -> float:
    poison(record, arm)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0 / nodes
    verify(record, arm, expected)
    return elapsed_us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-len", type=int, choices=(1, 5), required=True)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--windows", type=int, choices=SUPPORTED_WINDOWS, default=6)
    parser.add_argument("--warmup-windows", type=int, default=6)
    args = parser.parse_args()
    if args.nodes < 2:
        parser.error("--nodes must be at least two")
    if args.warmup_windows < 0 or args.warmup_windows % 6:
        parser.error("--warmup-windows must be a non-negative multiple of six")
    scored = (
        args.nodes == SCORED_NODES
        and args.windows == SCORED_WINDOWS
        and args.warmup_windows == SCORED_WARMUP_WINDOWS
    )
    if args.windows == SCORED_WINDOWS and not scored:
        parser.error("scored timing requires nodes100, windows30, warmup24")

    compiled = {
        arm: compile_reader(args.query_len, args.nodes, arm) for arm in CONTROL_ARMS
    }
    records, ranges, populations = build_runtime(args.query_len, args.nodes)
    print(
        "PASS_D0_TIMING_ALLOCATION_SELF_CHECK "
        f"nodes={args.nodes} ranges={len(ranges)} populations={populations} "
        f"allocated_bytes={torch.cuda.memory_allocated()}",
        flush=True,
    )

    expected = {}
    for arm in CONTROL_ARMS:
        poison(records[arm], arm)
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        launch_control(compiled[arm], records[arm], stream)
        torch.cuda.synchronize()
        expected[arm] = snapshot(records[arm], arm)
    poison(records["dense"], "dense")
    launch_dense(records["dense"])
    torch.cuda.synchronize()
    expected["dense"] = snapshot(records["dense"], "dense")

    if not torch.equal(
        expected["c0"][0].view(torch.int16),
        expected["rn"][0].view(torch.int16),
    ):
        raise AssertionError("C0/RN output mismatch")
    if not torch.equal(expected["c0"][1], expected["rn"][1]):
        raise AssertionError("C0/RN LSE mismatch")
    records["rn"]["out"].view(torch.int16).flatten()[0] ^= 1
    detected = False
    try:
        verify(records["rn"], "rn", expected["rn"])
    except AssertionError:
        detected = True
    records["rn"]["out"].copy_(expected["rn"][0])
    if not detected:
        raise AssertionError("timing verifier negative control was not detected")
    verify(records["rn"], "rn", expected["rn"])
    print("PASS_D0_TIMING_EAGER_AND_NEGATIVE_CONTROL", flush=True)

    graphs = {}
    for arm in CONTROL_ARMS:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
            launch_control(compiled[arm], records[arm], stream)
        graphs[arm] = graph
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        launch_dense(records["dense"])
    graphs["dense"] = graph
    for arm in ALL_ARMS:
        measure_graph(graphs[arm], records[arm], arm, expected[arm], args.nodes)
    print("PASS_D0_TIMING_GRAPH_CAPTURE", flush=True)

    for order in three_arm_orders(args.warmup_windows):
        for arm in order:
            measure_graph(graphs[arm], records[arm], arm, expected[arm], args.nodes)
    control_samples = {arm: [] for arm in CONTROL_ARMS}
    control_windows = []
    for window, order in enumerate(three_arm_orders(args.windows), start=1):
        row = {"window": window, "order": list(order), "timings_us": {}}
        for arm in order:
            elapsed = measure_graph(
                graphs[arm], records[arm], arm, expected[arm], args.nodes
            )
            control_samples[arm].append(elapsed)
            row["timings_us"][arm] = elapsed
        control_windows.append(row)

    for order in pair_orders(args.warmup_windows):
        for arm in order:
            measure_graph(graphs[arm], records[arm], arm, expected[arm], args.nodes)
    dense_samples = {"rn": [], "dense": []}
    dense_windows = []
    for window, order in enumerate(pair_orders(args.windows), start=1):
        row = {"window": window, "order": list(order), "timings_us": {}}
        for arm in order:
            elapsed = measure_graph(
                graphs[arm], records[arm], arm, expected[arm], args.nodes
            )
            dense_samples[arm].append(elapsed)
            row["timings_us"][arm] = elapsed
        dense_windows.append(row)

    ratios = {
        "rn_vs_c0": paired_log_summary(control_samples["rn"], control_samples["c0"]),
        "rn_vs_native": paired_log_summary(
            control_samples["rn"], control_samples["native"]
        ),
        "rn_vs_dense": paired_log_summary(dense_samples["rn"], dense_samples["dense"]),
    }
    means = {
        **{arm: statistics.fmean(values) for arm, values in control_samples.items()},
        "dense": statistics.fmean(dense_samples["dense"]),
    }
    gates = {
        "rn_vs_c0_ci95_high_le_0_99": ratios["rn_vs_c0"]["ci95_high"] <= 0.99,
        "rn_vs_dense_ci95_high_le_1_10": ratios["rn_vs_dense"]["ci95_high"] <= 1.10,
    }
    passed = all(gates.values())
    result = {
        "status": (
            "PASS_D0_TIMING_SCORED"
            if scored and passed
            else "REJECT_D0_TIMING_SCORED" if scored else "PASS_D0_TIMING_REHEARSAL"
        ),
        "scored": scored,
        "query_len": args.query_len,
        "nodes": args.nodes,
        "windows": args.windows,
        "warmup_windows": args.warmup_windows,
        "means_us": means,
        "ratios": ratios,
        "gates": gates,
        "control_windows": control_windows,
        "dense_windows": dense_windows,
    }
    print("D0_TIMING_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    if scored and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
