#!/usr/bin/env python3
"""L2-cold exact-10K latency comparison for private TQ4 controls.

Each six-sample block covers every dense/M128/M64 execution order. Every timed
sample is one CUDA-graph replay after read-only eviction over at least four
times the device L2.
Intervals are within-run paired percentile-bootstrap precision estimates; a
gate requires separate process invocations with different seeds.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import statistics
import time
from collections.abc import Callable
from itertools import permutations
from pathlib import Path

import torch

from tokenspeed_mla.mla_decode import tokenspeed_mla_decode
from tokenspeed_mla.mla_decode_tq4 import (
    _tokenspeed_mla_decode_tq4_m128_control,
    _tokenspeed_mla_decode_tq4_m64_control,
)
from tokenspeed_mla.tq4_contract import dequantize_tq4_reference


def _stable_seed(base: int, q_len: int, seed_offset: int) -> int:
    payload = f"{base}:{q_len}:{seed_offset}".encode()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    )


def _case(
    q_len: int,
    seed_offset: int,
    page_layout: str,
    pool_fragmentation_factor: int,
) -> dict:
    torch.manual_seed(_stable_seed(20260803, q_len, seed_offset))
    device = torch.device("cuda")
    batch, heads, page_size = 1, 8, 32
    seq_len = 10_000 + q_len
    tile_count = (seq_len + 127) // 128
    logical_pages = tile_count * (128 // page_size)
    physical_pages = logical_pages * pool_fragmentation_factor
    low = torch.randint(
        0,
        16,
        (physical_pages, page_size, 256),
        device=device,
        dtype=torch.uint8,
    )
    high = torch.randint(0, 16, low.shape, device=device, dtype=torch.uint8)
    packed = (low | (high << 4)).contiguous()
    scales = (0.5 + torch.rand((physical_pages, page_size), device=device)).to(
        torch.bfloat16
    )
    centroids = torch.linspace(-1.5, 1.5, 16, device=device, dtype=torch.float32)
    rope = torch.randn(
        (physical_pages, page_size, 64), device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    latent = dequantize_tq4_reference(
        packed, scales, centroids, dtype=torch.float8_e4m3fn
    )
    codebook = (
        scales.float().unsqueeze(-1) * centroids.view(1, 1, -1)
    ).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    dense = torch.cat((latent, rope), dim=-1).contiguous()
    query = torch.randn((batch, q_len, heads, 576), device=device).to(
        torch.float8_e4m3fn
    )
    if page_layout == "shuffled":
        block_tables = torch.randperm(
            physical_pages, device=device, dtype=torch.int32
        )[:logical_pages]
    elif page_layout == "sequential":
        block_tables = torch.arange(logical_pages, device=device, dtype=torch.int32)
    else:
        raise ValueError(f"unsupported page layout: {page_layout}")
    block_tables = block_tables.view(1, -1)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    custom_mask = None
    if q_len > 1:
        mask = torch.ones((q_len, seq_len), device=device, dtype=torch.bool)
        mask[:, seq_len - q_len :] = torch.tril(
            torch.ones((q_len, q_len), device=device, dtype=torch.bool)
        )
        custom_mask = mask.flatten().contiguous()
    result = {
        "query": query,
        "packed": packed,
        "scales": scales,
        "centroids": centroids,
        "codebook": codebook,
        "rope": rope,
        "dense": dense,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "custom_mask": custom_mask,
        "workspace": torch.empty(256 * 1024 * 1024, device=device, dtype=torch.int8),
        "outputs": {
            name: torch.empty(
                (batch, q_len, heads, 512),
                device=device,
                dtype=torch.bfloat16,
            )
            for name in ("dense", "m128", "m64")
        },
        "max_seq_len": seq_len,
        "logical_pages": logical_pages,
        "physical_pages": physical_pages,
    }
    # Release construction-only low/high nibbles and the dequantized latent so
    # telemetry reflects the actual benchmark pools rather than allocator cache.
    del low, high, latent
    torch.cuda.empty_cache()
    return result


def _warm(functions: dict[str, Callable[[], object]], repeat: int) -> None:
    for _ in range(repeat):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()


def _capture(functions: dict[str, Callable[[], object]]) -> dict[str, torch.cuda.CUDAGraph]:
    graphs = {}
    for name, fn in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name] = graph
    torch.cuda.synchronize()
    return graphs


def _time_graph_ms(
    graph: torch.cuda.CUDAGraph,
    eviction: torch.Tensor,
    eviction_sink: torch.Tensor,
    eviction_stream: torch.cuda.Stream,
    *,
    include_hot_probe: bool = False,
    enforce_host_queue: bool = True,
) -> dict[str, float | bool]:
    # Queue a read-only eviction on a separate stream, then make the benchmark
    # stream wait on its completion event. All benchmark work is submitted
    # before the eviction completes; the explicit event dependency prevents
    # PDL from overlapping a graph prologue with eviction while CUDA events
    # exclude both eviction and Python dispatch from target-kernel latency.
    current_stream = torch.cuda.current_stream()
    eviction_start = torch.cuda.Event(enable_timing=True)
    eviction_done = torch.cuda.Event(enable_timing=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    hot_start = torch.cuda.Event(enable_timing=True) if include_hot_probe else None
    hot_end = torch.cuda.Event(enable_timing=True) if include_hot_probe else None
    host_enqueue_start_ns = time.perf_counter_ns()
    with torch.cuda.stream(eviction_stream):
        eviction_start.record(eviction_stream)
        torch.sum(eviction, dim=0, dtype=torch.int64, out=eviction_sink)
        eviction_done.record(eviction_stream)
    current_stream.wait_event(eviction_done)
    start.record(current_stream)
    graph.replay()
    end.record(current_stream)
    if include_hot_probe:
        hot_start.record(current_stream)
        graph.replay()
        hot_end.record(current_stream)
    host_enqueue_end_ns = time.perf_counter_ns()
    end.synchronize()
    if hot_end is not None:
        hot_end.synchronize()
    eviction_ms = float(eviction_start.elapsed_time(eviction_done))
    host_enqueue_ms = (host_enqueue_end_ns - host_enqueue_start_ns) / 1_000_000.0
    queued_before_eviction_completed = host_enqueue_ms < eviction_ms
    if enforce_host_queue and not queued_before_eviction_completed:
        raise RuntimeError(
            "host did not queue the complete timed interval before L2 eviction "
            "finished; increase --eviction-multiplier"
        )
    result: dict[str, float | bool] = {
        "graph_ms": float(start.elapsed_time(end)),
        "eviction_ms": eviction_ms,
        "host_enqueue_ms": host_enqueue_ms,
        "queued_before_eviction_completed": queued_before_eviction_completed,
    }
    if hot_start is not None and hot_end is not None:
        result["immediate_hot_graph_ms"] = float(hot_start.elapsed_time(hot_end))
    return result


def _time_valid_graph_ms(
    graph: torch.cuda.CUDAGraph,
    eviction: torch.Tensor,
    eviction_sink: torch.Tensor,
    eviction_stream: torch.cuda.Stream,
    *,
    include_hot_probe: bool,
    max_host_race_retries: int,
) -> dict:
    discarded = []
    for attempt in range(max_host_race_retries + 1):
        measurement = _time_graph_ms(
            graph,
            eviction,
            eviction_sink,
            eviction_stream,
            include_hot_probe=include_hot_probe,
            enforce_host_queue=False,
        )
        if measurement["queued_before_eviction_completed"]:
            return {
                **measurement,
                "host_race_retries": attempt,
                "discarded_host_race_attempts": discarded,
            }
        discarded.append(measurement)
    raise RuntimeError(
        "host failed to queue the complete timed interval before L2 eviction "
        f"finished after {max_host_race_retries + 1} attempts"
    )


def _correctness(outputs: dict[str, torch.Tensor]) -> dict[str, float | bool]:
    m128_error = (outputs["dense"].float() - outputs["m128"].float()).abs()
    m64_error = (outputs["dense"].float() - outputs["m64"].float()).abs()
    packed_cross_error = (outputs["m128"].float() - outputs["m64"].float()).abs()
    result = {
        "m128_output_max_abs": float(m128_error.max()),
        "m128_output_mean_abs": float(m128_error.mean()),
        "m64_output_max_abs": float(m64_error.max()),
        "m64_output_mean_abs": float(m64_error.mean()),
        "packed_cross_max_abs": float(packed_cross_error.max()),
        "finite": bool(all(torch.isfinite(out.float()).all() for out in outputs.values())),
    }
    if not result["finite"] or max(
        result["m128_output_max_abs"],
        result["m64_output_max_abs"],
        result["packed_cross_max_abs"],
    ) > 0.125:
        raise AssertionError(f"exact-10K timed specialization is incorrect: {result}")
    return result


def _ratio_interval(
    numerator: list[float],
    denominator: list[float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, float]:
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired non-empty samples are required")
    point = statistics.fmean(numerator) / statistics.fmean(denominator)
    rng = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        indices = [rng.randrange(len(numerator)) for _ in numerator]
        ratios.append(
            statistics.fmean(numerator[i] for i in indices)
            / statistics.fmean(denominator[i] for i in indices)
        )
    ratios.sort()
    lower = ratios[int(0.025 * resamples)]
    upper = ratios[min(resamples - 1, int(0.975 * resamples))]
    return {
        "ratio": point,
        "within_run_ratio_ci95_low": lower,
        "within_run_ratio_ci95_high": upper,
        "delta_pct": (point - 1.0) * 100.0,
        "within_run_delta_pct_ci95_low": (lower - 1.0) * 100.0,
        "within_run_delta_pct_ci95_high": (upper - 1.0) * 100.0,
    }


def _difference_interval_us(
    treatment: list[float],
    control: list[float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, float]:
    if len(treatment) != len(control) or not treatment:
        raise ValueError("paired non-empty samples are required")
    point = (statistics.fmean(treatment) - statistics.fmean(control)) * 1_000.0
    rng = random.Random(seed)
    differences = []
    for _ in range(resamples):
        indices = [rng.randrange(len(treatment)) for _ in treatment]
        differences.append(
            (
                statistics.fmean(treatment[i] for i in indices)
                - statistics.fmean(control[i] for i in indices)
            )
            * 1_000.0
        )
    differences.sort()
    return {
        "delta_us": point,
        "within_run_delta_us_ci95_low": differences[int(0.025 * resamples)],
        "within_run_delta_us_ci95_high": differences[
            min(resamples - 1, int(0.975 * resamples))
        ],
    }


def _run(
    q_len: int,
    *,
    samples: int,
    warmup: int,
    bootstrap_resamples: int,
    eviction_multiplier: float,
    seed_offset: int,
    enable_pdl: bool,
    page_layout: str,
    pool_fragmentation_factor: int,
    cache_probe_samples: int,
    minimum_dense_cold_hot_ratio: float,
    minimum_cold_not_faster_fraction: float,
    max_host_race_retries: int,
) -> dict:
    case_started_unix_ns = time.time_ns()
    case = _case(q_len, seed_offset, page_layout, pool_fragmentation_factor)
    common = dict(
        query=case["query"],
        workspace_buffer=case["workspace"],
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        max_seq_len=case["max_seq_len"],
        softmax_scale=576**-0.5,
        custom_mask=case["custom_mask"],
        enable_pdl=enable_pdl,
        return_lse=False,
    )

    def dense() -> object:
        return tokenspeed_mla_decode(
            kv_cache=case["dense"],
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            out=case["outputs"]["dense"],
            **common,
        )

    packed_common = dict(
        kv_nope_packed=case["packed"],
        kv_nope_scale=case["scales"],
        kv_rope=case["rope"],
        centroids=case["centroids"],
        kv_nope_codebook=case["codebook"],
        fp8_rope=True,
        **common,
    )

    def m128() -> object:
        return _tokenspeed_mla_decode_tq4_m128_control(
            out=case["outputs"]["m128"], **packed_common
        )

    def m64() -> object:
        return _tokenspeed_mla_decode_tq4_m64_control(
            out=case["outputs"]["m64"], **packed_common
        )

    functions = {"dense": dense, "m128": m128, "m64": m64}
    # These direct calls compile and validate the exact auto-split, workspace,
    # return_lse=False, codebook, FP8-RoPE specializations that will be timed.
    for output in case["outputs"].values():
        output.fill_(float("nan"))
    for fn in functions.values():
        fn()
    torch.cuda.synchronize()
    direct_correctness = _correctness(case["outputs"])
    _warm(functions, warmup)
    graphs = _capture(functions)
    for output in case["outputs"].values():
        output.fill_(float("nan"))
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    graph_correctness = _correctness(case["outputs"])

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = int(
        getattr(properties, "L2_cache_size", getattr(properties, "l2_cache_size", 0))
    )
    if l2_bytes <= 0:
        raise RuntimeError("device L2 size is unavailable; cannot prove an L2-cold gate")
    eviction_bytes = int(l2_bytes * eviction_multiplier)
    eviction = torch.zeros(eviction_bytes, device="cuda", dtype=torch.uint8)
    eviction_sink = torch.zeros((), device="cuda", dtype=torch.int64)
    eviction_stream = torch.cuda.Stream(device=torch.cuda.current_device())
    # Absorb lazy event/stream/reduction initialization before the race gate is
    # active and before collecting either cache-control or target samples.
    for graph in graphs.values():
        _time_graph_ms(
            graph,
            eviction,
            eviction_sink,
            eviction_stream,
            include_hot_probe=True,
            enforce_host_queue=False,
        )
    # This is a one-shot process. Freeze existing tracked objects and disable
    # generational collection across all gated enqueue windows; reference
    # counting remains active. On a gate exception the process exits, while the
    # successful path restores GC before serializing artifacts.
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.freeze()
    gc.disable()
    cache_probe_records = {name: [] for name in functions}
    orders = list(permutations(functions))
    for probe_index in range(cache_probe_samples):
        for name in orders[probe_index % len(orders)]:
            cache_probe_records[name].append(
                _time_valid_graph_ms(
                    graphs[name],
                    eviction,
                    eviction_sink,
                    eviction_stream,
                    include_hot_probe=True,
                    max_host_race_retries=max_host_race_retries,
                )
            )
    cache_probe = {}
    for name, records in cache_probe_records.items():
        cold_ms = statistics.fmean(float(record["graph_ms"]) for record in records)
        hot_ms = statistics.fmean(
            float(record["immediate_hot_graph_ms"]) for record in records
        )
        cache_probe[name] = {
            "cold_ms": cold_ms,
            "immediate_hot_ms": hot_ms,
            "cold_hot_ratio": cold_ms / hot_ms,
            "cold_not_faster_fraction": statistics.fmean(
                float(record["graph_ms"])
                >= float(record["immediate_hot_graph_ms"])
                for record in records
            ),
            "records": records,
        }
    if cache_probe["dense"]["cold_hot_ratio"] < minimum_dense_cold_hot_ratio:
        raise RuntimeError(
            "L2 eviction did not meet the predeclared dense cold/hot ratio: "
            f"{cache_probe['dense']['cold_hot_ratio']:.6f} < "
            f"{minimum_dense_cold_hot_ratio:.6f}"
        )
    unreliable_arms = {
        name: {
            "cold_hot_ratio": probe["cold_hot_ratio"],
            "cold_not_faster_fraction": probe["cold_not_faster_fraction"],
        }
        for name, probe in cache_probe.items()
        if probe["cold_hot_ratio"] <= 1.0
        or probe["cold_not_faster_fraction"]
        < minimum_cold_not_faster_fraction
    }
    if unreliable_arms:
        raise RuntimeError(
            "L2 eviction was not reliably colder for every arm: "
            f"{unreliable_arms}"
        )
    timings = {name: [] for name in functions}
    timing_records = {name: [] for name in functions}
    for sample_index in range(samples):
        # Cycle through all six orders so slow thermal/clock drift is balanced
        # across the dense control and both packed candidates.
        for name in orders[sample_index % len(orders)]:
            sample_started_unix_ns = time.time_ns()
            sample_started_monotonic_ns = time.monotonic_ns()
            measurement = _time_valid_graph_ms(
                graphs[name],
                eviction,
                eviction_sink,
                eviction_stream,
                include_hot_probe=False,
                max_host_race_retries=max_host_race_retries,
            )
            timings[name].append(float(measurement["graph_ms"]))
            timing_records[name].append(
                {
                    "sample_index": sample_index,
                    "sample_started_unix_ns": sample_started_unix_ns,
                    "sample_started_monotonic_ns": sample_started_monotonic_ns,
                    **measurement,
                    "sample_finished_unix_ns": time.time_ns(),
                    "sample_finished_monotonic_ns": time.monotonic_ns(),
                }
            )
    gc.unfreeze()
    if gc_was_enabled:
        gc.enable()

    post_timing_correctness = _correctness(case["outputs"])

    dense_ms = statistics.fmean(timings["dense"])
    m128_ms = statistics.fmean(timings["m128"])
    m64_ms = statistics.fmean(timings["m64"])
    m64_vs_dense = _ratio_interval(
        timings["m64"],
        timings["dense"],
        seed=_stable_seed(20260803, q_len, seed_offset),
        resamples=bootstrap_resamples,
    )
    m64_vs_m128 = _ratio_interval(
        timings["m64"],
        timings["m128"],
        seed=_stable_seed(20260813, q_len, seed_offset),
        resamples=bootstrap_resamples,
    )
    m64_minus_dense = _difference_interval_us(
        timings["m64"],
        timings["dense"],
        seed=_stable_seed(20260823, q_len, seed_offset),
        resamples=bootstrap_resamples,
    )
    return {
        "q_len": q_len,
        "seq_len": case["max_seq_len"],
        "page_layout": page_layout,
        "pool_fragmentation_factor": pool_fragmentation_factor,
        "case_started_unix_ns": case_started_unix_ns,
        "case_finished_unix_ns": time.time_ns(),
        "enable_pdl": enable_pdl,
        "dense_m64_ms": dense_ms,
        "packed_m128_ms": m128_ms,
        "packed_m64_ms": m64_ms,
        "m64_vs_dense_pct": m64_vs_dense["delta_pct"],
        "m64_vs_m128_pct": m64_vs_m128["delta_pct"],
        "m64_vs_dense": m64_vs_dense,
        "m64_vs_m128": m64_vs_m128,
        "m64_minus_dense": m64_minus_dense,
        "direct_correctness": direct_correctness,
        "graph_correctness": graph_correctness,
        "post_timing_correctness": post_timing_correctness,
        "representation": {
            "dense_row_bytes": 576,
            "packed_row_bytes": 338,
            "packed_codebook": True,
            "packed_fp8_rope": True,
            "logical_request_pages": case["logical_pages"],
            "physical_pool_pages": case["physical_pages"],
            "dense_logical_request_bytes": case["logical_pages"] * 32 * 576,
            "packed_logical_request_bytes": case["logical_pages"] * 32 * 338,
            "dense_allocated_pool_bytes": case["dense"].nbytes,
            "packed_allocated_pool_bytes": (
                case["packed"].nbytes
                + case["scales"].nbytes
                + case["rope"].nbytes
                + case["codebook"].nbytes
            ),
            "packed_shared_centroid_bytes": case["centroids"].nbytes,
            "gross_row_saving_pct": (1.0 - 338.0 / 576.0) * 100.0,
        },
        "cache_control": {
            "mode": "graph_replay_after_cross_stream_read_only_l2_eviction",
            "device_l2_bytes": l2_bytes,
            "eviction_bytes": eviction_bytes,
            "eviction_multiplier": eviction_multiplier,
            "cache_probe_samples": cache_probe_samples,
            "minimum_dense_cold_hot_ratio": minimum_dense_cold_hot_ratio,
            "minimum_cold_not_faster_fraction": minimum_cold_not_faster_fraction,
            "max_host_race_retries": max_host_race_retries,
            "cache_probe": cache_probe,
        },
        "samples_ms": timings,
        "timing_records": timing_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--eviction-multiplier", type=float, default=4.0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--run-id", default="independent-run-1")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--disable-pdl", action="store_true")
    parser.add_argument(
        "--page-layout", choices=("shuffled", "sequential"), default="shuffled"
    )
    parser.add_argument("--pool-fragmentation-factor", type=int, default=256)
    parser.add_argument("--cache-probe-samples", type=int, default=12)
    parser.add_argument("--minimum-dense-cold-hot-ratio", type=float, default=1.01)
    parser.add_argument("--minimum-cold-not-faster-fraction", type=float, default=0.8)
    parser.add_argument("--max-host-race-retries", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.samples < 6 or args.samples % 6:
        parser.error("--samples must be a positive multiple of 6")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    if args.eviction_multiplier < 4.0:
        parser.error("--eviction-multiplier must be at least 4.0")
    if args.cache_probe_samples < 6 or args.cache_probe_samples % 6:
        parser.error("--cache-probe-samples must be a positive multiple of 6")
    if args.pool_fragmentation_factor < 1:
        parser.error("--pool-fragmentation-factor must be positive")
    if args.page_layout == "shuffled" and args.pool_fragmentation_factor < 2:
        parser.error("shuffled layout requires --pool-fragmentation-factor >= 2")
    if args.minimum_dense_cold_hot_ratio <= 1.0:
        parser.error("--minimum-dense-cold-hot-ratio must exceed 1.0")
    if not 0.5 < args.minimum_cold_not_faster_fraction <= 1.0:
        parser.error("--minimum-cold-not-faster-fraction must be in (0.5, 1.0]")
    if not 0 <= args.max_host_race_retries <= 10:
        parser.error("--max-host-race-retries must be in [0, 10]")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("--run-id must be filesystem-safe")
    if not 0 <= args.device_index < torch.cuda.device_count():
        parser.error("--device-index must name a visible CUDA device")
    torch.cuda.set_device(args.device_index)
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("SM100 required")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_uuid = getattr(properties, "uuid", None)
    if not device_uuid:
        raise RuntimeError("CUDA device UUID is unavailable; telemetry cannot be joined")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    run_started_unix_ns = time.time_ns()
    cases = []
    for q_len in (1, 5):
        case = _run(
            q_len,
            samples=args.samples,
            warmup=args.warmup,
            bootstrap_resamples=args.bootstrap_resamples,
            eviction_multiplier=args.eviction_multiplier,
            seed_offset=args.seed_offset,
            enable_pdl=not args.disable_pdl,
            page_layout=args.page_layout,
            pool_fragmentation_factor=args.pool_fragmentation_factor,
            cache_probe_samples=args.cache_probe_samples,
            minimum_dense_cold_hot_ratio=args.minimum_dense_cold_hot_ratio,
            minimum_cold_not_faster_fraction=(
                args.minimum_cold_not_faster_fraction
            ),
            max_host_race_retries=args.max_host_race_retries,
        )
        cases.append(case)
        case_record = {
            "record_type": "case",
            "run_id": args.run_id,
            "seed_offset": args.seed_offset,
            "device_index": args.device_index,
            "device_uuid": str(device_uuid),
            "enable_pdl": not args.disable_pdl,
            "page_layout": args.page_layout,
            "case": case,
        }
        encoded_case = json.dumps(case_record, sort_keys=True)
        print(encoded_case, flush=True)
        if args.output_dir is not None:
            pdl_label = "pdl-off" if args.disable_pdl else "pdl-on"
            case_name = (
                f"case-{args.run_id}-seed{args.seed_offset}-{pdl_label}-"
                f"{args.page_layout}-q{q_len}.json"
            )
            with (args.output_dir / case_name).open("x", encoding="utf-8") as artifact:
                artifact.write(encoded_case + "\n")
    summary = {
        "record_type": "summary",
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "device_index": args.device_index,
        "device_uuid": str(device_uuid),
        "run_started_unix_ns": run_started_unix_ns,
        "run_finished_unix_ns": time.time_ns(),
        "enable_pdl": not args.disable_pdl,
        "page_layout": args.page_layout,
        "pool_fragmentation_factor": args.pool_fragmentation_factor,
        "run_id": args.run_id,
        "seed_offset": args.seed_offset,
        "samples": args.samples,
        "warmup": args.warmup,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_method": "within_run_paired_percentile",
        "interval_scope": "single_process_single_gpu_precision_only",
        "gate_requires": "two_independent_process_runs_with_distinct_seed_offsets",
        "order_cycle": list(permutations(("dense", "m128", "m64"))),
        "cases": cases,
    }
    encoded_summary = json.dumps(summary, sort_keys=True)
    print(encoded_summary, flush=True)
    if args.output_dir is not None:
        pdl_label = "pdl-off" if args.disable_pdl else "pdl-on"
        summary_name = (
            f"summary-{args.run_id}-seed{args.seed_offset}-{pdl_label}-"
            f"{args.page_layout}.json"
        )
        with (args.output_dir / summary_name).open("x", encoding="utf-8") as artifact:
            artifact.write(encoded_summary + "\n")


if __name__ == "__main__":
    main()
