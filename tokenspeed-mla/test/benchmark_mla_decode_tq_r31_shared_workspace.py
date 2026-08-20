# Copyright (c) 2026 LightSeek Foundation

"""R31-R2 shared-workspace rotated-domain mixed-decode gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

import benchmark_mla_decode_tq_r31_segmented as fixture
from tokenspeed_mla import (
    reduce_mla_mixed_workspace,
    tokenspeed_mla_decode,
    tokenspeed_mla_decode_tq_r31,
)


PAGE = fixture.PAGE
LATENT = fixture.LATENT
ROPE = fixture.ROPE
HEADS = fixture.HEADS
QUERY_LEN = fixture.QUERY_LEN
BATCH = fixture.BATCH
SEQ_LEN = fixture.SEQ_LEN
SOFTMAX_SCALE = fixture.SOFTMAX_SCALE
HOT_DECLARED = 8
COLD_DECLARED = 9
TOTAL_DECLARED = HOT_DECLARED + COLD_DECLARED
ATTENTION_BASELINE_US = 18.566080
COMPLETE_ADDED_CEILING_US = 13.0
ROTATED_WRITER_DELTA_US = 1.7013013164202375


def _capture(function) -> torch.cuda.CUDAGraph:
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    torch.cuda.synchronize()
    return graph


def _measure(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "stdev_us": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_us": min(samples),
        "max_us": max(samples),
    }


def _merge_reference(
    hot_output: torch.Tensor,
    hot_lse: torch.Tensor,
    cold_output: torch.Tensor,
    cold_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = torch.maximum(hot_lse, cold_lse)
    merged_lse = maximum + torch.log2(
        torch.exp2(hot_lse - maximum) + torch.exp2(cold_lse - maximum)
    )
    merged = hot_output.float() * torch.exp2(hot_lse - merged_lse).unsqueeze(
        -1
    ) + cold_output.float() * torch.exp2(cold_lse - merged_lse).unsqueeze(-1)
    return merged.to(torch.bfloat16), merged_lse


def run(windows: int, replays: int, writer_delta_us: float) -> dict[str, object]:
    from sglang.kernels.ops.quantization.hadamard import (
        hadamard_transform_with_signs,
    )

    state, config, dense_query, dense_cache, dense_table, dense_seq = (
        fixture._build_state("balanced")
    )
    if state.hot_requests != list(range(BATCH)) or state.prefix_requests != list(
        range(BATCH)
    ):
        raise AssertionError("R31-R2 requires one hot and one cold segment per request")
    if state.all_cold_requests:
        raise AssertionError("balanced R31-R2 must not contain all-cold requests")

    # Recreate the frozen BF16 source so hot cache rotation has exactly one
    # BF16->rotated-BF16->FP8 rounding path, matching the native owner oracle.
    torch.manual_seed(0xA173100)
    source = (
        torch.randn(BATCH, SEQ_LEN, LATENT + ROPE, device="cuda") * 0.1
    ).to(torch.bfloat16)
    query_source = (
        torch.randn(BATCH, QUERY_LEN, HEADS, LATENT + ROPE, device="cuda") * 0.1
    ).to(torch.bfloat16)
    hot_lengths, cold_lengths = fixture._layout_lengths("balanced")
    hot_source = torch.cat(
        [
            source[request, cold_lengths[request] :]
            for request in range(BATCH)
            if hot_lengths[request]
        ],
        dim=0,
    ).contiguous()
    rotated_hot_latent = torch.empty_like(hot_source[..., :LATENT])
    hadamard_transform_with_signs(
        hot_source[..., :LATENT],
        config.signs1,
        config.signs2,
        scale=1.0 / math.sqrt(LATENT),
        out=rotated_hot_latent,
    )
    rotated_hot_cache = torch.cat(
        (rotated_hot_latent, hot_source[..., LATENT:]), dim=-1
    ).to(torch.float8_e4m3fn)
    rotated_hot_cache = rotated_hot_cache.view(-1, PAGE, LATENT + ROPE)
    rotated_hot_query = torch.cat(
        (
            state.r31_query_latent,
            query_source[..., LATENT:].to(torch.float8_e4m3fn),
        ),
        dim=-1,
    ).contiguous()

    rows = BATCH * QUERY_LEN * HEADS
    shared_workspace_bytes = rows * TOTAL_DECLARED * (LATENT + 1) * 4
    shared_workspace = torch.empty(
        shared_workspace_bytes, dtype=torch.int8, device="cuda"
    )
    hot_workspace_bytes = rows * HOT_DECLARED * (LATENT + 1) * 4
    hot_shared_workspace = shared_workspace[:hot_workspace_bytes]
    cold_shared_workspace = shared_workspace[hot_workspace_bytes:]
    baseline_workspace = torch.empty(
        rows * 64 * (LATENT + 1) * 4, dtype=torch.int8, device="cuda"
    )
    hot_reference_workspace = torch.empty(
        rows * HOT_DECLARED * (LATENT + 1) * 4,
        dtype=torch.int8,
        device="cuda",
    )
    cold_reference_workspace = torch.empty(
        rows * COLD_DECLARED * (LATENT + 1) * 4,
        dtype=torch.int8,
        device="cuda",
    )
    baseline_output = torch.empty(
        BATCH, QUERY_LEN, HEADS, LATENT, dtype=torch.bfloat16, device="cuda"
    )
    baseline_lse = torch.empty(
        BATCH, QUERY_LEN, HEADS, dtype=torch.float32, device="cuda"
    )
    producer_sentinel = torch.empty_like(baseline_output)
    hot_reference_output = torch.empty_like(baseline_output)
    hot_reference_lse = torch.empty_like(baseline_lse)
    cold_reference_output = torch.empty_like(baseline_output)
    cold_reference_lse = torch.empty_like(baseline_lse)
    mixed_output = torch.empty_like(baseline_output)
    mixed_lse = torch.empty_like(baseline_lse)
    inverse_mixed_output = torch.empty_like(mixed_output)
    hot_max_seq_len = int(state.hot_seq.max().item())
    cold_max_seq_len = int(state.prefix_seq.max().item())

    def baseline_launch() -> None:
        tokenspeed_mla_decode(
            query=dense_query,
            kv_cache=dense_cache,
            workspace_buffer=baseline_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=dense_table,
            seq_lens=dense_seq,
            max_seq_len=SEQ_LEN,
            softmax_scale=SOFTMAX_SCALE,
            out=baseline_output,
            causal_mask=True,
            enable_pdl=True,
            return_lse=True,
            lse_out=baseline_lse,
        )

    def hot_reference_launch() -> None:
        tokenspeed_mla_decode(
            query=rotated_hot_query,
            kv_cache=rotated_hot_cache,
            workspace_buffer=hot_reference_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=state.hot_table,
            seq_lens=state.hot_seq,
            max_seq_len=hot_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=hot_reference_output,
            causal_mask=True,
            enable_pdl=True,
            return_lse=True,
            lse_out=hot_reference_lse,
            split_kv_override=HOT_DECLARED,
        )

    def cold_reference_launch() -> None:
        tokenspeed_mla_decode_tq_r31(
            query_latent=state.r31_query_latent,
            query_rope=state.r31_query_rope,
            packed_latent=state.packed_cold,
            reconstruction_scale=state.cold_scale,
            high_rope=state.cold_high,
            residual_rope=state.cold_residual,
            workspace_buffer=cold_reference_workspace,
            block_tables=state.prefix_table,
            seq_lens=state.prefix_seq,
            max_seq_len=cold_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=cold_reference_output,
            causal_mask=False,
            enable_pdl=True,
            return_lse=True,
            lse_out=cold_reference_lse,
            split_kv_override=COLD_DECLARED,
        )

    def hot_producer_launch() -> None:
        tokenspeed_mla_decode(
            query=rotated_hot_query,
            kv_cache=rotated_hot_cache,
            workspace_buffer=hot_shared_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=state.hot_table,
            seq_lens=state.hot_seq,
            max_seq_len=hot_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=producer_sentinel,
            causal_mask=True,
            enable_pdl=True,
            split_kv_override=HOT_DECLARED,
            producer_only=True,
        )

    def cold_producer_launch() -> None:
        tokenspeed_mla_decode_tq_r31(
            query_latent=state.r31_query_latent,
            query_rope=state.r31_query_rope,
            packed_latent=state.packed_cold,
            reconstruction_scale=state.cold_scale,
            high_rope=state.cold_high,
            residual_rope=state.cold_residual,
            workspace_buffer=cold_shared_workspace,
            block_tables=state.prefix_table,
            seq_lens=state.prefix_seq,
            max_seq_len=cold_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=producer_sentinel,
            causal_mask=False,
            enable_pdl=True,
            split_kv_override=COLD_DECLARED,
            producer_only=True,
        )

    def reduction_launch() -> None:
        reduce_mla_mixed_workspace(
            shared_workspace,
            state.hot_seq,
            state.prefix_seq,
            HOT_DECLARED,
            COLD_DECLARED,
            mixed_output,
            mixed_lse,
        )

    def sequential_candidate_launch() -> None:
        hot_producer_launch()
        cold_producer_launch()
        reduction_launch()

    hot_stream = torch.cuda.Stream()
    cold_stream = torch.cuda.Stream()
    fork_event = torch.cuda.Event()
    hot_done_event = torch.cuda.Event()
    cold_done_event = torch.cuda.Event()

    def concurrent_producers_launch() -> None:
        launch_origin = torch.cuda.current_stream()
        fork_event.record(launch_origin)
        hot_stream.wait_event(fork_event)
        cold_stream.wait_event(fork_event)
        with torch.cuda.stream(hot_stream):
            hot_producer_launch()
            hot_done_event.record(hot_stream)
        with torch.cuda.stream(cold_stream):
            cold_producer_launch()
            cold_done_event.record(cold_stream)
        launch_origin.wait_event(hot_done_event)
        launch_origin.wait_event(cold_done_event)

    def concurrent_candidate_launch() -> None:
        concurrent_producers_launch()
        reduction_launch()

    baseline_launch()
    hot_reference_launch()
    cold_reference_launch()
    shared_workspace.view(torch.float32).fill_(float("nan"))
    sequential_candidate_launch()
    torch.cuda.synchronize()
    reference_output, reference_lse = _merge_reference(
        hot_reference_output,
        hot_reference_lse,
        cold_reference_output,
        cold_reference_lse,
    )
    torch.testing.assert_close(mixed_output, reference_output, rtol=0, atol=0.01)
    torch.testing.assert_close(mixed_lse, reference_lse, rtol=0, atol=2.0e-5)
    if not torch.isfinite(mixed_output).all() or not torch.isfinite(mixed_lse).all():
        raise AssertionError("mixed reducer consumed a poisoned unwritten split")
    sequential_output = mixed_output.clone()
    sequential_lse = mixed_lse.clone()
    concurrent_candidate_launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(mixed_output, sequential_output, rtol=0, atol=0)
    torch.testing.assert_close(mixed_lse, sequential_lse, rtol=0, atol=0)

    hadamard_transform_with_signs(
        mixed_output,
        config.signs2,
        config.signs1,
        scale=1.0 / math.sqrt(LATENT),
        out=inverse_mixed_output,
    )
    torch.cuda.synchronize()
    dense_output_max_abs = float(
        (inverse_mixed_output.float() - baseline_output.float()).abs().max()
    )
    dense_lse_max_abs = float((mixed_lse - baseline_lse).abs().max())

    baseline_graph = _capture(baseline_launch)
    sequential_graph = _capture(sequential_candidate_launch)
    concurrent_graph = _capture(concurrent_candidate_launch)
    phase_graphs = {
        "hot_producer": _capture(hot_producer_launch),
        "cold_producer": _capture(cold_producer_launch),
        "combined_reduction": _capture(reduction_launch),
        "concurrent_producers": _capture(concurrent_producers_launch),
    }
    for _ in range(10):
        _measure(baseline_graph, replays)
        _measure(sequential_graph, replays)
        _measure(concurrent_graph, replays)
    samples = {"baseline": [], "sequential": [], "concurrent": []}
    graph_by_name = {
        "baseline": baseline_graph,
        "sequential": sequential_graph,
        "concurrent": concurrent_graph,
    }
    orders = (
        ("baseline", "sequential", "concurrent"),
        ("concurrent", "sequential", "baseline"),
    )
    for window in range(windows):
        for name in orders[window % len(orders)]:
            samples[name].append(_measure(graph_by_name[name], replays))
    timing = {name: _summary(values) for name, values in samples.items()}
    phase_us = {}
    for name, graph in phase_graphs.items():
        for _ in range(5):
            _measure(graph, replays)
        phase_us[name] = statistics.fmean(
            _measure(graph, replays) for _ in range(windows)
        )

    attention_added_us = (
        timing["concurrent"]["mean_us"] - timing["baseline"]["mean_us"]
    )
    charged_complete_added_us = attention_added_us + writer_delta_us
    result = {
        "schema": "r31-r2-shared-workspace-v1",
        "shape": {
            "batch": BATCH,
            "query_len": QUERY_LEN,
            "sequence_len": SEQ_LEN,
            "hot_lengths": state.hot_seq.tolist(),
            "cold_lengths": state.prefix_seq.tolist(),
            "hot_declared_splits": HOT_DECLARED,
            "cold_declared_splits": COLD_DECLARED,
        },
        "timing": timing,
        "phase_mean_us": phase_us,
        "attention_added_us": attention_added_us,
        "charged_rotated_writer_delta_us": writer_delta_us,
        "charged_complete_added_us": charged_complete_added_us,
        "ceiling": {
            "frozen_attention_baseline_us": ATTENTION_BASELINE_US,
            "complete_added_ceiling_us": COMPLETE_ADDED_CEILING_US,
        },
        "correctness": {
            "shared_matches_two_reductions": True,
            "poisoned_unwritten_slots_ignored": True,
            "same_stream_matches_concurrent": True,
            "dense_output_max_abs": dense_output_max_abs,
            "dense_lse_max_abs": dense_lse_max_abs,
        },
        "workspace_bytes": shared_workspace_bytes,
        "workspace_mib": shared_workspace_bytes / (1024**2),
        "complete_memory_saving_percent": 27.5007925,
        "gates": {
            "charged_complete_added_le_13us": (
                charged_complete_added_us <= COMPLETE_ADDED_CEILING_US
            ),
            "dense_output_max_abs_le_1e_3": dense_output_max_abs <= 1.0e-3,
            "dense_lse_max_abs_le_1e_3": dense_lse_max_abs <= 1.0e-3,
        },
    }
    result["passed"] = all(result["gates"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument(
        "--writer-delta-us", type=float, default=ROTATED_WRITER_DELTA_US
    )
    args = parser.parse_args()
    result = run(args.windows, args.replays, args.writer_delta_us)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
