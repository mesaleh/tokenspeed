# Copyright (c) 2026 LightSeek Foundation

"""R31-R5 F1 gate for one native FP8/R31 producer grid."""

import argparse
import json
import math
import statistics

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

import benchmark_mla_decode_tq_r31_segmented as fixture
from tokenspeed_mla import (
    reduce_mla_mixed_workspace,
    tokenspeed_mla_decode,
    tokenspeed_mla_decode_tq_r31,
)
from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
)
from tokenspeed_mla.mla_decode_tq_e2m1 import _as_cute_tensor
from tokenspeed_mla.mla_decode_tq_r31_mixed_native import (
    BlackwellMixedFP8R31Producer,
)
from tokenspeed_mla.mla_helpers import get_mla_decode_fold_sq_factor
from tokenspeed_mla.utils import get_max_active_clusters


PAGE = fixture.PAGE
LATENT = fixture.LATENT
ROPE = fixture.ROPE
HEADS = fixture.HEADS
QUERY_LEN = fixture.QUERY_LEN
BATCH = fixture.BATCH
SEQ_LEN = fixture.SEQ_LEN
SOFTMAX_SCALE = fixture.SOFTMAX_SCALE
HOT_SPLITS = 8
COLD_SPLITS = 9
QK_TILER = (64, 128)
PV_TILER = (64, 256)
WRITER_DELTA_US = 1.7013013164202375
Q5_COMPONENT_CEILINGS_US = {1: 13.5067, 5: 29.2079, 8: 37.5659}


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


def _compile_mixed_producer(
    hot_query: torch.Tensor,
    hot_cache: torch.Tensor,
    hot_table: torch.Tensor,
    hot_workspace: torch.Tensor,
    hot_seq: torch.Tensor,
    cold_query_latent: torch.Tensor,
    cold_query_rope: torch.Tensor,
    cold_cache: torch.Tensor,
    cold_high_rope: torch.Tensor,
    cold_table: torch.Tensor,
    cold_workspace: torch.Tensor,
    cold_seq: torch.Tensor,
    output_sentinel: torch.Tensor,
    cold_scale: torch.Tensor,
    cold_residual_rope: torch.Tensor,
):
    fold_sq_factor = get_mla_decode_fold_sq_factor(HEADS, QUERY_LEN, QK_TILER[0])
    common = dict(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=QK_TILER,
        mma_pv_tiler_mn=PV_TILER,
        max_active_clusters=get_max_active_clusters(1),
        page_size=PAGE,
        skip_correction_threshold=0.0,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=False,
        fold_sq_factor=fold_sq_factor,
        num_heads=HEADS,
        seq_len_q=QUERY_LEN,
        cp_world=1,
        use_runtime_causal_bound=True,
        producer_only=True,
    )
    hot_kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
        **common,
        is_causal=True,
    )
    use_packed_p_scale_math = BATCH == 1 and QUERY_LEN == 5
    cold_kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
        **common,
        is_causal=True,
        use_tq_e2m1=True,
        use_tq_r31_rope=True,
        tq_s1_scale_tma=True,
        tq_s1_scale_stages=3,
        tq_s1_k_rope_stages=1,
        tq_s1_packed_p_scale_math=use_packed_p_scale_math,
        tq_s1_early_final_pcor=use_packed_p_scale_math,
        tq_r31_async_expand=True,
    )
    kernel = BlackwellMixedFP8R31Producer(hot_kernel, cold_kernel)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        _as_cute_tensor(hot_query[..., :LATENT], cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(hot_query[..., LATENT:], cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(hot_cache[..., :LATENT], cutlass.Float8E4M3FN, 2, 16),
        _as_cute_tensor(hot_cache[..., LATENT:], cutlass.Float8E4M3FN, 2, 16),
        _as_cute_tensor(hot_table, cutlass.Int32, 1, 4),
        _as_cute_tensor(hot_workspace, cutlass.Int8, 0, 32),
        Int32(1),
        _as_cute_tensor(hot_seq, cutlass.Int32, 0, 4),
        _as_cute_tensor(hot_seq, cutlass.Int32, 0, 4),
        _as_cute_tensor(cold_query_latent, cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(cold_query_rope, cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(cold_cache, cutlass.Uint8, 2, 16),
        _as_cute_tensor(cold_high_rope, cutlass.Float8E4M3FN, 2, 16),
        _as_cute_tensor(cold_table, cutlass.Int32, 1, 4),
        _as_cute_tensor(cold_workspace, cutlass.Int8, 0, 32),
        Int32(1),
        _as_cute_tensor(cold_seq, cutlass.Int32, 0, 4),
        _as_cute_tensor(cold_seq, cutlass.Int32, 0, 4),
        _as_cute_tensor(output_sentinel, cutlass.BFloat16, 3, 16),
        Float32(1.0),
        Float32(1.0),
        _as_cute_tensor(cold_scale, cutlass.BFloat16, 1, 16),
        _as_cute_tensor(cold_residual_rope, cutlass.Uint8, 2, 16),
        stream,
        options="--enable-tvm-ffi --opt-level 3",
    )


def run(
    windows: int,
    replays: int,
    batch: int,
    query_len: int,
    hot_splits: int,
    cold_splits: int,
    correctness_only: bool = False,
    sanitizer_fixture: str | None = None,
    causal_owner: str = "hot",
) -> dict[str, object]:
    global BATCH, QUERY_LEN, HOT_SPLITS, COLD_SPLITS
    BATCH = batch
    QUERY_LEN = query_len
    HOT_SPLITS = hot_splits
    COLD_SPLITS = cold_splits
    if causal_owner not in ("hot", "cold"):
        raise ValueError(f"unknown causal owner {causal_owner!r}")
    fixture.BATCH = batch
    fixture.QUERY_LEN = query_len
    fixture.HOT_CAPACITY = 2_560 * batch
    from sglang.kernels.ops.quantization.hadamard import (
        hadamard_transform_with_signs,
    )

    state, config, dense_query, dense_cache, dense_table, dense_seq = (
        fixture._build_state("balanced")
    )
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
    hot_cache = torch.cat(
        (rotated_hot_latent, hot_source[..., LATENT:]), dim=-1
    ).to(torch.float8_e4m3fn)
    hot_cache = hot_cache.view(-1, PAGE, LATENT + ROPE)
    hot_query = torch.cat(
        (
            state.r31_query_latent,
            query_source[..., LATENT:].to(torch.float8_e4m3fn),
        ),
        dim=-1,
    ).contiguous()

    rows = BATCH * QUERY_LEN * HEADS
    hot_workspace_bytes = rows * HOT_SPLITS * (LATENT + 1) * 4
    cold_workspace_bytes = rows * COLD_SPLITS * (LATENT + 1) * 4
    shared_workspace = torch.empty(
        hot_workspace_bytes + cold_workspace_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    hot_workspace = shared_workspace[:hot_workspace_bytes]
    cold_workspace = shared_workspace[hot_workspace_bytes:]
    reference_workspace = torch.empty_like(shared_workspace)
    reference_hot_workspace = reference_workspace[:hot_workspace_bytes]
    reference_cold_workspace = reference_workspace[hot_workspace_bytes:]
    output_sentinel = torch.empty(
        BATCH, QUERY_LEN, HEADS, LATENT, dtype=torch.bfloat16, device="cuda"
    )
    reference_output = torch.empty_like(output_sentinel)
    reference_lse = torch.empty(
        BATCH, QUERY_LEN, HEADS, dtype=torch.float32, device="cuda"
    )
    mixed_output = torch.empty_like(output_sentinel)
    mixed_lse = torch.empty_like(reference_lse)
    baseline_workspace = torch.empty(
        rows * 64 * (LATENT + 1) * 4, dtype=torch.int8, device="cuda"
    )
    baseline_output = torch.empty_like(output_sentinel)
    baseline_lse = torch.empty_like(reference_lse)
    hot_max_seq_len = int(state.hot_seq.max().item())
    cold_max_seq_len = int(state.prefix_seq.max().item())
    hot_causal_seq = state.hot_seq + (
        (QUERY_LEN - 1) if causal_owner == "cold" else 0
    )
    cold_causal_seq = state.prefix_seq + (
        (QUERY_LEN - 1) if causal_owner == "hot" else 0
    )

    compiled = _compile_mixed_producer(
        hot_query,
        hot_cache,
        state.hot_table,
        hot_workspace,
        state.hot_seq,
        state.r31_query_latent,
        state.r31_query_rope,
        state.packed_cold,
        state.cold_high,
        state.prefix_table,
        cold_workspace,
        state.prefix_seq,
        output_sentinel,
        state.cold_scale,
        state.cold_residual,
    )

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
            query=hot_query,
            kv_cache=hot_cache,
            workspace_buffer=reference_hot_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=state.hot_table,
            seq_lens=state.hot_seq,
            max_seq_len=hot_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=causal_owner == "hot",
            enable_pdl=True,
            split_kv_override=HOT_SPLITS,
            producer_only=True,
        )

    def cold_reference_launch() -> None:
        tokenspeed_mla_decode_tq_r31(
            query_latent=state.r31_query_latent,
            query_rope=state.r31_query_rope,
            packed_latent=state.packed_cold,
            reconstruction_scale=state.cold_scale,
            high_rope=state.cold_high,
            residual_rope=state.cold_residual,
            workspace_buffer=reference_cold_workspace,
            block_tables=state.prefix_table,
            seq_lens=state.prefix_seq,
            max_seq_len=cold_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=causal_owner == "cold",
            enable_pdl=True,
            split_kv_override=COLD_SPLITS,
            producer_only=True,
        )

    def reference_launch() -> None:
        hot_reference_launch()
        cold_reference_launch()
        reduce_mla_mixed_workspace(
            reference_workspace,
            state.hot_seq,
            state.prefix_seq,
            HOT_SPLITS,
            COLD_SPLITS,
            reference_output,
            reference_lse,
        )

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
            hot_reference_launch()
            hot_done_event.record(hot_stream)
        with torch.cuda.stream(cold_stream):
            cold_reference_launch()
            cold_done_event.record(cold_stream)
        launch_origin.wait_event(hot_done_event)
        launch_origin.wait_event(cold_done_event)

    def concurrent_reference_launch() -> None:
        concurrent_producers_launch()
        reduce_mla_mixed_workspace(
            reference_workspace,
            state.hot_seq,
            state.prefix_seq,
            HOT_SPLITS,
            COLD_SPLITS,
            reference_output,
            reference_lse,
        )

    def mixed_producer_launch() -> None:
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            compiled(
                hot_query[..., :LATENT],
                hot_query[..., LATENT:],
                hot_cache[..., :LATENT],
                hot_cache[..., LATENT:],
                state.hot_table,
                hot_workspace,
                Int32(HOT_SPLITS),
                state.hot_seq,
                hot_causal_seq,
                state.r31_query_latent,
                state.r31_query_rope,
                state.packed_cold,
                state.cold_high,
                state.prefix_table,
                cold_workspace,
                Int32(COLD_SPLITS),
                state.prefix_seq,
                cold_causal_seq,
                output_sentinel,
                Float32(SOFTMAX_SCALE),
                Float32(1.0),
                state.cold_scale,
                state.cold_residual,
            )

    def mixed_launch() -> None:
        mixed_producer_launch()
        reduce_mla_mixed_workspace(
            shared_workspace,
            state.hot_seq,
            state.prefix_seq,
            HOT_SPLITS,
            COLD_SPLITS,
            mixed_output,
            mixed_lse,
        )

    reference_workspace.view(torch.float32).fill_(float("nan"))
    shared_workspace.view(torch.float32).fill_(float("nan"))
    reference_launch()
    mixed_launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(mixed_output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(mixed_lse, reference_lse, rtol=0, atol=0)
    if not torch.isfinite(mixed_output).all() or not torch.isfinite(mixed_lse).all():
        raise AssertionError("mixed producer consumed a poisoned workspace slot")

    if sanitizer_fixture is not None:
        torch.save(
            {
                "batch": BATCH,
                "query_len": QUERY_LEN,
                "hot_splits": HOT_SPLITS,
                "cold_splits": COLD_SPLITS,
                "causal_owner": causal_owner,
                "hot_query": hot_query.cpu(),
                "hot_cache": hot_cache.cpu(),
                "hot_table": state.hot_table.cpu(),
                "hot_seq": state.hot_seq.cpu(),
                "hot_causal_seq": hot_causal_seq.cpu(),
                "cold_query_latent": state.r31_query_latent.cpu(),
                "cold_query_rope": state.r31_query_rope.cpu(),
                "cold_cache": state.packed_cold.cpu(),
                "cold_high_rope": state.cold_high.cpu(),
                "cold_table": state.prefix_table.cpu(),
                "cold_seq": state.prefix_seq.cpu(),
                "cold_causal_seq": cold_causal_seq.cpu(),
                "cold_scale": state.cold_scale.cpu(),
                "cold_residual_rope": state.cold_residual.cpu(),
                "reference_workspace": reference_workspace.cpu(),
            },
            sanitizer_fixture,
        )

    if correctness_only:
        return {
            "schema": "r31-r5-f1-mixed-native-correctness-v1",
            "shape": {
                "batch": BATCH,
                "query_len": QUERY_LEN,
                "sequence_len": SEQ_LEN,
                "hot_splits": HOT_SPLITS,
                "cold_splits": COLD_SPLITS,
                "causal_owner": causal_owner,
            },
            "correctness": {
                "bit_exact_vs_two_producers": True,
                "poisoned_unwritten_slots_ignored": True,
            },
        }

    baseline_graph = _capture(baseline_launch)
    reference_graph = _capture(reference_launch)
    concurrent_graph = _capture(concurrent_reference_launch)
    mixed_graph = _capture(mixed_launch)
    producer_graph = _capture(mixed_producer_launch)
    for _ in range(10):
        _measure(baseline_graph, replays)
        _measure(reference_graph, replays)
        _measure(concurrent_graph, replays)
        _measure(mixed_graph, replays)
    samples = {"baseline": [], "reference": [], "concurrent": [], "mixed": []}
    graph_by_name = {
        "baseline": baseline_graph,
        "reference": reference_graph,
        "concurrent": concurrent_graph,
        "mixed": mixed_graph,
    }
    orders = (
        ("baseline", "reference", "concurrent", "mixed"),
        ("mixed", "concurrent", "reference", "baseline"),
    )
    for window in range(windows):
        for name in orders[window % len(orders)]:
            samples[name].append(_measure(graph_by_name[name], replays))
    producer_samples = [_measure(producer_graph, replays) for _ in range(windows)]
    timing = {name: _summary(values) for name, values in samples.items()}
    timing["mixed_producer"] = _summary(producer_samples)
    delta_us = timing["mixed"]["mean_us"] - timing["reference"]["mean_us"]
    attention_added_us = timing["mixed"]["mean_us"] - timing["baseline"]["mean_us"]
    charged_complete_added_us = attention_added_us + WRITER_DELTA_US
    # Only q5 has a fresh production-control TPOT budget.  q1 remains useful
    # as a descriptive shape guard, but must not emit a formal pass/fail until
    # its control is rerun under the same request contract.
    component_ceiling_us = (
        Q5_COMPONENT_CEILINGS_US[BATCH] if QUERY_LEN == 5 else None
    )
    return {
        "schema": "r31-r5-f1-mixed-native-v1",
        "shape": {
            "batch": BATCH,
            "query_len": QUERY_LEN,
            "sequence_len": SEQ_LEN,
            "hot_splits": HOT_SPLITS,
            "cold_splits": COLD_SPLITS,
            "causal_owner": causal_owner,
        },
        "correctness": {
            "bit_exact_vs_two_producers": True,
            "poisoned_unwritten_slots_ignored": True,
        },
        "timing": timing,
        "mixed_minus_reference_us": delta_us,
        "mixed_minus_concurrent_us": (
            timing["mixed"]["mean_us"] - timing["concurrent"]["mean_us"]
        ),
        "attention_added_us": attention_added_us,
        "writer_delta_us": WRITER_DELTA_US,
        "charged_complete_added_us": charged_complete_added_us,
        "component_ceiling_us": component_ceiling_us,
        "charged_complete_within_ceiling": (
            charged_complete_added_us <= component_ceiling_us
            if component_ceiling_us is not None
            else None
        ),
        "mixed_faster": delta_us < 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--batch", type=int, choices=(1, 5, 8), default=8)
    parser.add_argument("--query-len", type=int, choices=(1, 5), default=5)
    parser.add_argument("--hot-splits", type=int, default=3)
    parser.add_argument("--cold-splits", type=int, default=14)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--sanitizer-fixture")
    parser.add_argument("--causal-owner", choices=("hot", "cold"), default="hot")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.windows,
                args.replays,
                args.batch,
                args.query_len,
                args.hot_splits,
                args.cold_splits,
                args.correctness_only,
                args.sanitizer_fixture,
                args.causal_owner,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
