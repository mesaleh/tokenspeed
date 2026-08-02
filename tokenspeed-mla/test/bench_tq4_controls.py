#!/usr/bin/env python3
"""Balanced exact-10K latency comparison for private TQ4 controls.

Each six-sample block covers every dense/M128/M64 execution order. Confidence
intervals use paired, non-parametric percentile bootstrap resampling of those
interleaved sample blocks.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Callable
from itertools import permutations

import torch

from tokenspeed_mla.mla_decode import tokenspeed_mla_decode
from tokenspeed_mla.mla_decode_tq4 import (
    _tokenspeed_mla_decode_tq4_m128_control,
    _tokenspeed_mla_decode_tq4_m64_control,
)
from tokenspeed_mla.tq4_contract import dequantize_tq4_reference


def _case(q_len: int) -> dict:
    torch.manual_seed(20260803 + q_len)
    device = torch.device("cuda")
    batch, heads, page_size = 1, 8, 32
    seq_len = 10_000 + q_len
    tile_count = (seq_len + 127) // 128
    pages = tile_count * (128 // page_size)
    low = torch.randint(0, 16, (pages, page_size, 256), device=device, dtype=torch.uint8)
    high = torch.randint(0, 16, low.shape, device=device, dtype=torch.uint8)
    packed = (low | (high << 4)).contiguous()
    scales = (0.5 + torch.rand((pages, page_size), device=device)).to(torch.bfloat16)
    centroids = torch.linspace(-1.5, 1.5, 16, device=device, dtype=torch.float32)
    rope = torch.randn((pages, page_size, 64), device=device, dtype=torch.bfloat16)
    latent = dequantize_tq4_reference(
        packed, scales, centroids, dtype=torch.float8_e4m3fn
    )
    dense = torch.cat((latent, rope.to(torch.float8_e4m3fn)), dim=-1).contiguous()
    query = torch.randn((batch, q_len, heads, 576), device=device).to(
        torch.float8_e4m3fn
    )
    block_tables = torch.arange(pages, device=device, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    custom_mask = None
    if q_len > 1:
        mask = torch.ones((q_len, seq_len), device=device, dtype=torch.bool)
        mask[:, seq_len - q_len :] = torch.tril(
            torch.ones((q_len, q_len), device=device, dtype=torch.bool)
        )
        custom_mask = mask.flatten().contiguous()
    return {
        "query": query,
        "packed": packed,
        "scales": scales,
        "centroids": centroids,
        "rope": rope,
        "dense": dense,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "custom_mask": custom_mask,
        "workspace": torch.empty(256 * 1024 * 1024, device=device, dtype=torch.int8),
        "out": torch.empty((batch, q_len, heads, 512), device=device, dtype=torch.bfloat16),
        "max_seq_len": seq_len,
    }


def _warm(functions: dict[str, Callable[[], object]], repeat: int) -> None:
    for _ in range(repeat):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()


def _time_ms(fn: Callable[[], object], repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / repeat)


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
        "ratio_ci95_low": lower,
        "ratio_ci95_high": upper,
        "delta_pct": (point - 1.0) * 100.0,
        "delta_pct_ci95_low": (lower - 1.0) * 100.0,
        "delta_pct_ci95_high": (upper - 1.0) * 100.0,
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
        "delta_us_ci95_low": differences[int(0.025 * resamples)],
        "delta_us_ci95_high": differences[
            min(resamples - 1, int(0.975 * resamples))
        ],
    }


def _run(
    q_len: int,
    *,
    samples: int,
    inner_repeat: int,
    warmup: int,
    bootstrap_resamples: int,
) -> dict:
    case = _case(q_len)
    common = dict(
        query=case["query"],
        workspace_buffer=case["workspace"],
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        max_seq_len=case["max_seq_len"],
        softmax_scale=576**-0.5,
        custom_mask=case["custom_mask"],
        enable_pdl=True,
        return_lse=False,
        out=case["out"],
    )

    def dense() -> object:
        return tokenspeed_mla_decode(
            kv_cache=case["dense"],
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            **common,
        )

    packed_common = dict(
        kv_nope_packed=case["packed"],
        kv_nope_scale=case["scales"],
        kv_rope=case["rope"],
        centroids=case["centroids"],
        **common,
    )

    def m128() -> object:
        return _tokenspeed_mla_decode_tq4_m128_control(**packed_common)

    def m64() -> object:
        return _tokenspeed_mla_decode_tq4_m64_control(**packed_common)

    functions = {"dense": dense, "m128": m128, "m64": m64}
    _warm(functions, warmup)
    timings = {name: [] for name in functions}
    orders = list(permutations(functions))
    for sample_index in range(samples):
        # Cycle through all six orders so slow thermal/clock drift is balanced
        # across the dense control and both packed candidates.
        for name in orders[sample_index % len(orders)]:
            timings[name].append(_time_ms(functions[name], inner_repeat))

    dense_ms = statistics.fmean(timings["dense"])
    m128_ms = statistics.fmean(timings["m128"])
    m64_ms = statistics.fmean(timings["m64"])
    m64_vs_dense = _ratio_interval(
        timings["m64"],
        timings["dense"],
        seed=20260803 + q_len,
        resamples=bootstrap_resamples,
    )
    m64_vs_m128 = _ratio_interval(
        timings["m64"],
        timings["m128"],
        seed=20260813 + q_len,
        resamples=bootstrap_resamples,
    )
    m64_minus_dense = _difference_interval_us(
        timings["m64"],
        timings["dense"],
        seed=20260823 + q_len,
        resamples=bootstrap_resamples,
    )
    return {
        "q_len": q_len,
        "seq_len": case["max_seq_len"],
        "dense_m64_ms": dense_ms,
        "packed_m128_ms": m128_ms,
        "packed_m64_ms": m64_ms,
        "m64_vs_dense_pct": m64_vs_dense["delta_pct"],
        "m64_vs_m128_pct": m64_vs_m128["delta_pct"],
        "m64_vs_dense": m64_vs_dense,
        "m64_vs_m128": m64_vs_m128,
        "m64_minus_dense": m64_minus_dense,
        "samples_ms": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--inner-repeat", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    if args.samples < 6 or args.samples % 6:
        parser.error("--samples must be a positive multiple of 6")
    if args.inner_repeat <= 0 or args.warmup < 0:
        parser.error("--inner-repeat must be positive and --warmup non-negative")
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("SM100 required")
    cases = [
        _run(
            q_len,
            samples=args.samples,
            inner_repeat=args.inner_repeat,
            warmup=args.warmup,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        for q_len in (1, 5)
    ]
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "samples": args.samples,
                "inner_repeat": args.inner_repeat,
                "warmup": args.warmup,
                "bootstrap_resamples": args.bootstrap_resamples,
                "bootstrap_method": "paired_percentile",
                "order_cycle": list(permutations(("dense", "m128", "m64"))),
                "cases": cases,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
