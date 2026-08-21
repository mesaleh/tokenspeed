"""Q3E matched-graph performance gate for the physical mixed producer."""

from __future__ import annotations

import argparse
import json
import math

import torch

from tokenspeed_mla import (
    reduce_mla_mixed_workspace,
    tokenspeed_mla_decode,
    tokenspeed_mla_decode_tq_r31,
    tokenspeed_mla_decode_tq_r31_mixed_split_query,
    tokenspeed_mla_decode_tq_r31_physical_mixed_split_query,
)

PAGE = 32
LATENT = 512
ROPE = 64
HEADS = 8
SEQUENCE_LENGTH = 7_440
HOT_LENGTH = 1_860
COLD_LENGTH = SEQUENCE_LENGTH - HOT_LENGTH
HOT_SPLITS = 3
COLD_SPLITS = 11
SOFTMAX_SCALE = 0.125


def _padded_pages(length: int) -> int:
    return math.ceil(math.ceil(length / PAGE) / 4) * 4


def _measure(graph: torch.cuda.CUDAGraph, repeats: int = 100) -> float:
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / repeats


def _case(
    batch: int,
    query_len: int,
    *,
    windows: int = 9,
    replays: int = 100,
    hot_splits: int = HOT_SPLITS,
    cold_splits: int = COLD_SPLITS,
) -> dict[str, object]:
    device = torch.device("cuda:0")
    hot_page_count = _padded_pages(HOT_LENGTH)
    cold_page_count = _padded_pages(COLD_LENGTH)
    query_latent = torch.zeros(
        batch, query_len, HEADS, LATENT, device=device, dtype=torch.float8_e4m3fn
    )
    hot_query_rope = torch.zeros(
        batch, query_len, HEADS, ROPE, device=device, dtype=torch.float8_e4m3fn
    )
    cold_query_rope = torch.zeros(
        batch, query_len, HEADS, 4 * ROPE, device=device, dtype=torch.float8_e4m3fn
    )
    hot_query = torch.cat((query_latent, hot_query_rope), dim=-1).contiguous()
    hot_cache = torch.zeros(
        batch * hot_page_count,
        PAGE,
        LATENT + ROPE,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    packed = torch.zeros(
        batch * cold_page_count,
        PAGE,
        LATENT // 2,
        device=device,
        dtype=torch.uint8,
    )
    scale = torch.ones(
        batch * cold_page_count, PAGE, device=device, dtype=torch.bfloat16
    )
    high = torch.zeros(
        batch * cold_page_count,
        PAGE,
        ROPE,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    residual = torch.zeros(
        batch * cold_page_count,
        PAGE,
        ROPE // 2,
        device=device,
        dtype=torch.uint8,
    )
    hot_table = torch.arange(
        batch * hot_page_count, device=device, dtype=torch.int32
    ).view(batch, hot_page_count)
    cold_table = torch.arange(
        batch * cold_page_count, device=device, dtype=torch.int32
    ).view(batch, cold_page_count)
    hot_lengths = torch.full((batch,), HOT_LENGTH, device=device, dtype=torch.int32)
    cold_lengths = torch.full(
        (batch,), COLD_LENGTH, device=device, dtype=torch.int32
    )
    hot_causal = hot_lengths + (query_len - 1)
    cold_causal = cold_lengths

    rows = batch * query_len * HEADS
    hot_workspace_bytes = rows * hot_splits * (LATENT + 1) * 4
    cold_workspace_bytes = rows * cold_splits * (LATENT + 1) * 4
    workspace_bytes = hot_workspace_bytes + cold_workspace_bytes
    normalized_workspace = torch.empty(workspace_bytes, device=device, dtype=torch.int8)
    physical_workspace = torch.empty_like(normalized_workspace)
    sequential_workspace = torch.empty_like(normalized_workspace)
    normalized_out = torch.empty(
        batch, query_len, HEADS, LATENT, device=device, dtype=torch.bfloat16
    )
    physical_out = torch.empty_like(normalized_out)
    sequential_out = torch.empty_like(normalized_out)
    output_sentinel = torch.empty_like(normalized_out)
    normalized_lse = torch.empty(
        batch, query_len, HEADS, device=device, dtype=torch.float32
    )
    physical_lse = torch.empty_like(normalized_lse)
    sequential_lse = torch.empty_like(normalized_lse)
    physical_fault = torch.zeros(1, device=device, dtype=torch.int32)
    sequential_fault = torch.zeros_like(physical_fault)

    common = dict(
        query_latent=query_latent,
        hot_query_rope=hot_query_rope,
        hot_cache=hot_cache,
        hot_block_tables=hot_table,
        hot_seq_lens=hot_lengths,
        hot_causal_seqs=hot_causal,
        hot_max_seq_len=HOT_LENGTH,
        cold_query_rope=cold_query_rope,
        cold_packed_latent=packed,
        cold_reconstruction_scale=scale,
        cold_high_rope=high,
        cold_residual_rope=residual,
        cold_block_tables=cold_table,
        cold_seq_lens=cold_lengths,
        cold_causal_seqs=cold_causal,
        cold_max_seq_len=COLD_LENGTH,
        hot_splits=hot_splits,
        cold_splits=cold_splits,
        softmax_scale=SOFTMAX_SCALE,
        enable_pdl=False,
    )

    def normalized_launch() -> None:
        tokenspeed_mla_decode_tq_r31_mixed_split_query(
            **common,
            workspace_buffer=normalized_workspace,
            out=normalized_out,
            lse_out=normalized_lse,
        )

    def physical_launch() -> None:
        tokenspeed_mla_decode_tq_r31_physical_mixed_split_query(
            **common,
            workspace_buffer=physical_workspace,
            out=physical_out,
            lse_out=physical_lse,
            fault_status=physical_fault,
        )

    def sequential_launch() -> None:
        tokenspeed_mla_decode(
            query=hot_query,
            kv_cache=hot_cache,
            workspace_buffer=sequential_workspace[:hot_workspace_bytes],
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=hot_table,
            seq_lens=hot_lengths,
            max_seq_len=HOT_LENGTH,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=False,
            enable_pdl=False,
            split_kv_override=hot_splits,
            producer_only=True,
        )
        tokenspeed_mla_decode_tq_r31(
            query_latent=query_latent,
            query_rope=cold_query_rope,
            packed_latent=packed,
            reconstruction_scale=scale,
            high_rope=high,
            residual_rope=residual,
            workspace_buffer=sequential_workspace[hot_workspace_bytes:],
            block_tables=cold_table,
            seq_lens=cold_lengths,
            max_seq_len=COLD_LENGTH,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=True,
            enable_pdl=False,
            split_kv_override=cold_splits,
            producer_only=True,
            _physical_split_score=True,
            _physical_split_score_lookahead=False,
            _physical_split_score_dual_tmem=True,
            _physical_split_score_fault_status=sequential_fault,
        )
        reduce_mla_mixed_workspace(
            sequential_workspace,
            hot_lengths,
            cold_lengths,
            hot_splits,
            cold_splits,
            sequential_out,
            sequential_lse,
        )

    normalized_launch()
    physical_launch()
    sequential_launch()
    torch.cuda.synchronize()
    physical_eager = physical_out.clone()
    lse_eager = physical_lse.clone()

    normalized_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(normalized_graph):
        normalized_launch()
    physical_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(physical_graph):
        physical_launch()
    sequential_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(sequential_graph):
        sequential_launch()

    normalized_samples: list[float] = []
    physical_samples: list[float] = []
    sequential_samples: list[float] = []
    window_deltas: list[float] = []
    for _ in range(windows):
        normalized_a = _measure(normalized_graph, replays)
        physical = _measure(physical_graph, replays)
        sequential = _measure(sequential_graph, replays)
        normalized_b = _measure(normalized_graph, replays)
        normalized = (normalized_a + normalized_b) / 2.0
        normalized_samples.extend((normalized_a, normalized_b))
        physical_samples.append(physical)
        sequential_samples.append(sequential)
        window_deltas.append((physical / normalized - 1.0) * 100.0)

    physical_graph.replay()
    torch.cuda.synchronize()
    normalized_us = sum(normalized_samples) / len(normalized_samples)
    physical_us = sum(physical_samples) / len(physical_samples)
    sequential_us = sum(sequential_samples) / len(sequential_samples)
    output_delta = (physical_out.float() - normalized_out.float()).abs()
    lse_delta = (physical_lse - normalized_lse).abs()
    graph_output_delta = (physical_out.float() - physical_eager.float()).abs()
    graph_lse_delta = (physical_lse - lse_eager).abs()
    result = {
        "batch": batch,
        "query_len": query_len,
        "sequence_length": SEQUENCE_LENGTH,
        "hot_length": HOT_LENGTH,
        "cold_length": COLD_LENGTH,
        "hot_splits": hot_splits,
        "cold_splits": cold_splits,
        "windows": windows,
        "replays": replays,
        "normalized_samples_us": normalized_samples,
        "physical_samples_us": physical_samples,
        "sequential_samples_us": sequential_samples,
        "window_delta_pct": window_deltas,
        "normalized_us": normalized_us,
        "physical_us": physical_us,
        "sequential_physical_us": sequential_us,
        "physical_vs_normalized_pct": (physical_us / normalized_us - 1.0) * 100.0,
        "physical_vs_sequential_pct": (physical_us / sequential_us - 1.0) * 100.0,
        "output_max_abs": float(output_delta.max()),
        "lse_max_abs": float(lse_delta.max()),
        "graph_output_max_abs": float(graph_output_delta.max()),
        "graph_lse_max_abs": float(graph_lse_delta.max()),
        "physical_fault_status": int(physical_fault.item()),
        "sequential_fault_status": int(sequential_fault.item()),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if (
        result["output_max_abs"] > 0.02
        or result["lse_max_abs"] > 2.0e-5
        or result["graph_output_max_abs"] != 0.0
        or result["graph_lse_max_abs"] != 0.0
        or result["physical_fault_status"] != 0
        or result["sequential_fault_status"] != 0
    ):
        raise AssertionError(result)
    return result


def run(
    windows: int = 9,
    replays: int = 100,
    hot_splits: int = HOT_SPLITS,
    cold_splits: int = COLD_SPLITS,
) -> list[dict[str, object]]:
    cells = [
        _case(
            batch,
            query_len,
            windows=windows,
            replays=replays,
            hot_splits=hot_splits,
            cold_splits=cold_splits,
        )
        for query_len in (1, 5)
        for batch in (8, 5, 1)
    ]
    mean_pct = sum(float(cell["physical_vs_normalized_pct"]) for cell in cells) / len(
        cells
    )
    summary = {"cells": cells, "arithmetic_mean_pct": mean_pct}
    print(json.dumps(summary, sort_keys=True), flush=True)
    if any(float(cell["physical_vs_normalized_pct"]) > 10.0 for cell in cells):
        raise AssertionError("a Q3E cell exceeded the 10% regression gate")
    if mean_pct > 5.0:
        raise AssertionError("the Q3E arithmetic mean exceeded the 5% gate")
    return cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int)
    parser.add_argument("--query-len", type=int)
    parser.add_argument("--windows", type=int, default=9)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--hot-splits", type=int, default=HOT_SPLITS)
    parser.add_argument("--cold-splits", type=int, default=COLD_SPLITS)
    args = parser.parse_args()
    if (args.batch is None) != (args.query_len is None):
        parser.error("--batch and --query-len must be provided together")
    if args.batch is None:
        run(args.windows, args.replays, args.hot_splits, args.cold_splits)
    else:
        _case(
            args.batch,
            args.query_len,
            windows=args.windows,
            replays=args.replays,
            hot_splits=args.hot_splits,
            cold_splits=args.cold_splits,
        )
