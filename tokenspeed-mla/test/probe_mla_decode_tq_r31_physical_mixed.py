"""Q3E correctness, dynamic-batch, graph, and fault probe."""

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
from tokenspeed_mla.mla_decode_tq_r31_mixed_api import _COMPILED_MIXED_KERNELS

PAGE = 32
LATENT = 512
ROPE = 64
HEADS = 8
SPLITS = 2
SOFTMAX_SCALE = 0.125


def _lengths(batch: int, sequence_length: int) -> tuple[list[int], list[int]]:
    patterns = [
        (129, sequence_length - 129),
        (0, sequence_length),
        (sequence_length, 0),
        (257, sequence_length - 257),
        (1, sequence_length - 1),
        (sequence_length - 1, 1),
        (128, sequence_length - 128),
        (sequence_length - 128, 128),
    ]
    selected = patterns[:batch]
    return [pair[0] for pair in selected], [pair[1] for pair in selected]


def _case(
    batch: int,
    query_len: int,
    *,
    sequence_length: int = 384,
    graph_replays: int = 100,
    exercise_faults: bool = False,
) -> dict[str, object]:
    torch.manual_seed(0xA173E00 + batch * 10 + query_len)
    device = torch.device("cuda:0")
    pages_per_request = math.ceil(sequence_length / PAGE)
    pages = batch * pages_per_request
    query_latent = (torch.randn(batch, query_len, HEADS, LATENT, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    hot_query_rope = (torch.randn(batch, query_len, HEADS, ROPE, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    cold_query_rope = (
        torch.randn(batch, query_len, HEADS, 4 * ROPE, device=device) * 0.1
    ).to(torch.float8_e4m3fn)
    hot_query = torch.cat((query_latent, hot_query_rope), dim=-1).contiguous()
    hot_cache = (torch.randn(pages, PAGE, LATENT + ROPE, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    packed = torch.randint(
        0, 256, (pages, PAGE, LATENT // 2), device=device, dtype=torch.uint8
    )
    scale_pattern = torch.tensor(
        [0.5, 1.0, 2.0, 4.0], device=device, dtype=torch.bfloat16
    )
    scale = scale_pattern.repeat(math.ceil(pages * PAGE / 4))[: pages * PAGE]
    scale = scale.view(pages, PAGE).contiguous()
    physical_high = (torch.randn(pages, PAGE, ROPE, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    physical_residual = torch.randint(
        0, 256, (pages, PAGE, ROPE // 2), device=device, dtype=torch.uint8
    )
    table = torch.arange(pages, device=device, dtype=torch.int32).view(
        batch, pages_per_request
    )
    hot_lengths_list, cold_lengths_list = _lengths(batch, sequence_length)
    hot_lengths = torch.tensor(hot_lengths_list, device=device, dtype=torch.int32)
    cold_lengths = torch.tensor(cold_lengths_list, device=device, dtype=torch.int32)

    rows = batch * query_len * HEADS
    owner_bytes = rows * SPLITS * (LATENT + 1) * 4
    reference_workspace = torch.empty(2 * owner_bytes, device=device, dtype=torch.int8)
    mixed_workspace = torch.empty_like(reference_workspace)
    reference_out = torch.empty(
        batch, query_len, HEADS, LATENT, device=device, dtype=torch.bfloat16
    )
    mixed_out = torch.empty_like(reference_out)
    reference_lse = torch.empty(
        batch, query_len, HEADS, device=device, dtype=torch.float32
    )
    mixed_lse = torch.empty_like(reference_lse)
    output_sentinel = torch.empty_like(reference_out)
    reference_fault = torch.zeros(1, device=device, dtype=torch.int32)
    mixed_fault = torch.zeros_like(reference_fault)

    def sequential_launch() -> None:
        tokenspeed_mla_decode(
            query=hot_query,
            kv_cache=hot_cache,
            workspace_buffer=reference_workspace[:owner_bytes],
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=table,
            seq_lens=hot_lengths,
            max_seq_len=sequence_length,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=True,
            enable_pdl=False,
            split_kv_override=SPLITS,
            producer_only=True,
        )
        tokenspeed_mla_decode_tq_r31(
            query_latent=query_latent,
            query_rope=cold_query_rope,
            packed_latent=packed,
            reconstruction_scale=scale,
            high_rope=physical_high,
            residual_rope=physical_residual,
            workspace_buffer=reference_workspace[owner_bytes:],
            block_tables=table,
            seq_lens=cold_lengths,
            max_seq_len=sequence_length,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=True,
            enable_pdl=False,
            split_kv_override=SPLITS,
            producer_only=True,
            _physical_split_score=True,
            _physical_split_score_lookahead=False,
            _physical_split_score_dual_tmem=True,
            _physical_split_score_fault_status=reference_fault,
        )
        reduce_mla_mixed_workspace(
            reference_workspace,
            hot_lengths,
            cold_lengths,
            SPLITS,
            SPLITS,
            reference_out,
            reference_lse,
        )

    def mixed_launch() -> None:
        tokenspeed_mla_decode_tq_r31_physical_mixed_split_query(
            query_latent=query_latent,
            hot_query_rope=hot_query_rope,
            hot_cache=hot_cache,
            hot_block_tables=table,
            hot_seq_lens=hot_lengths,
            hot_causal_seqs=hot_lengths,
            hot_max_seq_len=sequence_length,
            cold_query_rope=cold_query_rope,
            cold_packed_latent=packed,
            cold_reconstruction_scale=scale,
            cold_high_rope=physical_high,
            cold_residual_rope=physical_residual,
            cold_block_tables=table,
            cold_seq_lens=cold_lengths,
            cold_causal_seqs=cold_lengths,
            cold_max_seq_len=sequence_length,
            workspace_buffer=mixed_workspace,
            hot_splits=SPLITS,
            cold_splits=SPLITS,
            softmax_scale=SOFTMAX_SCALE,
            out=mixed_out,
            lse_out=mixed_lse,
            fault_status=mixed_fault,
            enable_pdl=False,
        )

    sequential_launch()
    mixed_launch()
    torch.cuda.synchronize()
    eager_out = mixed_out.clone()
    eager_lse = mixed_lse.clone()
    output_pointer = mixed_out.data_ptr()
    workspace_pointer = mixed_workspace.data_ptr()
    status_pointer = mixed_fault.data_ptr()

    output_delta = (mixed_out.float() - reference_out.float()).abs()
    lse_delta = (mixed_lse - reference_lse).abs()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mixed_launch()
    for _ in range(graph_replays):
        graph.replay()
    torch.cuda.synchronize()
    graph_output_delta = (mixed_out.float() - eager_out.float()).abs()
    graph_lse_delta = (mixed_lse - eager_lse).abs()

    result: dict[str, object] = {
        "batch": batch,
        "query_len": query_len,
        "sequence_length": sequence_length,
        "output_max_abs": float(output_delta.max()),
        "output_mean_abs": float(output_delta.mean()),
        "lse_max_abs": float(lse_delta.max()),
        "graph_output_max_abs": float(graph_output_delta.max()),
        "graph_lse_max_abs": float(graph_lse_delta.max()),
        "reference_fault_status": int(reference_fault.item()),
        "mixed_fault_status": int(mixed_fault.item()),
        "pointers_stable": (
            mixed_out.data_ptr() == output_pointer
            and mixed_workspace.data_ptr() == workspace_pointer
            and mixed_fault.data_ptr() == status_pointer
        ),
        "compiled_mixed_variants": len(_COMPILED_MIXED_KERNELS),
    }
    if exercise_faults:
        negative: dict[str, bool] = {}
        kwargs = dict(
            query_latent=query_latent,
            hot_query_rope=hot_query_rope,
            hot_cache=hot_cache,
            hot_block_tables=table,
            hot_seq_lens=hot_lengths,
            hot_causal_seqs=hot_lengths,
            hot_max_seq_len=sequence_length,
            cold_query_rope=cold_query_rope,
            cold_packed_latent=packed,
            cold_reconstruction_scale=scale,
            cold_high_rope=physical_high,
            cold_residual_rope=physical_residual,
            cold_block_tables=table,
            cold_seq_lens=cold_lengths,
            cold_causal_seqs=cold_lengths,
            cold_max_seq_len=sequence_length,
            workspace_buffer=mixed_workspace,
            hot_splits=SPLITS,
            cold_splits=SPLITS,
            softmax_scale=SOFTMAX_SCALE,
            out=mixed_out,
            lse_out=mixed_lse,
            enable_pdl=False,
        )
        for name, bad_status in (
            ("missing", None),
            ("wrong_dtype", torch.zeros(1, device=device, dtype=torch.uint8)),
            ("wrong_shape", torch.zeros(2, device=device, dtype=torch.int32)),
            ("wrong_device", torch.zeros(1, dtype=torch.int32)),
            ("alias", cold_lengths[:1]),
        ):
            try:
                tokenspeed_mla_decode_tq_r31_physical_mixed_split_query(
                    **kwargs, fault_status=bad_status
                )
            except (TypeError, ValueError):
                negative[name] = True
            else:
                negative[name] = False

        # The core kernel's fault-status argument was inserted immediately
        # before prepare_only. Exercise the old normalized public API so this
        # probe also guards the repaired positional child calls.
        legacy_reference_workspace = torch.empty_like(reference_workspace)
        legacy_mixed_workspace = torch.empty_like(reference_workspace)
        legacy_reference_out = torch.empty_like(reference_out)
        legacy_mixed_out = torch.empty_like(reference_out)
        legacy_reference_lse = torch.empty_like(reference_lse)
        legacy_mixed_lse = torch.empty_like(reference_lse)
        tokenspeed_mla_decode(
            query=hot_query,
            kv_cache=hot_cache,
            workspace_buffer=legacy_reference_workspace[:owner_bytes],
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=table,
            seq_lens=hot_lengths,
            max_seq_len=sequence_length,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=True,
            enable_pdl=False,
            split_kv_override=SPLITS,
            producer_only=True,
        )
        tokenspeed_mla_decode_tq_r31(
            query_latent=query_latent,
            query_rope=cold_query_rope,
            packed_latent=packed,
            reconstruction_scale=scale,
            high_rope=physical_high,
            residual_rope=physical_residual,
            workspace_buffer=legacy_reference_workspace[owner_bytes:],
            block_tables=table,
            seq_lens=cold_lengths,
            max_seq_len=sequence_length,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=True,
            enable_pdl=False,
            split_kv_override=SPLITS,
            producer_only=True,
        )
        reduce_mla_mixed_workspace(
            legacy_reference_workspace,
            hot_lengths,
            cold_lengths,
            SPLITS,
            SPLITS,
            legacy_reference_out,
            legacy_reference_lse,
        )
        tokenspeed_mla_decode_tq_r31_mixed_split_query(
            query_latent=query_latent,
            hot_query_rope=hot_query_rope,
            hot_cache=hot_cache,
            hot_block_tables=table,
            hot_seq_lens=hot_lengths,
            hot_causal_seqs=hot_lengths,
            hot_max_seq_len=sequence_length,
            cold_query_rope=cold_query_rope,
            cold_packed_latent=packed,
            cold_reconstruction_scale=scale,
            cold_high_rope=physical_high,
            cold_residual_rope=physical_residual,
            cold_block_tables=table,
            cold_seq_lens=cold_lengths,
            cold_causal_seqs=cold_lengths,
            cold_max_seq_len=sequence_length,
            workspace_buffer=legacy_mixed_workspace,
            hot_splits=SPLITS,
            cold_splits=SPLITS,
            softmax_scale=SOFTMAX_SCALE,
            out=legacy_mixed_out,
            lse_out=legacy_mixed_lse,
            enable_pdl=False,
        )
        torch.cuda.synchronize()
        legacy_output_delta = (
            legacy_mixed_out.float() - legacy_reference_out.float()
        ).abs()
        legacy_lse_delta = (legacy_mixed_lse - legacy_reference_lse).abs()
        result["legacy_output_max_abs"] = float(legacy_output_delta.max())
        result["legacy_lse_max_abs"] = float(legacy_lse_delta.max())

        scale[0, 0] = float("nan")
        mixed_fault.zero_()
        mixed_launch()
        torch.cuda.synchronize()
        result["invalid_scale_fault_status"] = int(mixed_fault.item())
        result["invalid_scale_finite"] = bool(torch.isfinite(mixed_out).all())
        result["negative_validation"] = negative

    print(json.dumps(result, sort_keys=True), flush=True)
    if (
        result["output_max_abs"] > 0.02
        or result["lse_max_abs"] > 2.0e-5
        or result["graph_output_max_abs"] != 0.0
        or result["graph_lse_max_abs"] != 0.0
        or result["reference_fault_status"] != 0
        or result["mixed_fault_status"] != 0
        or not result["pointers_stable"]
    ):
        raise AssertionError(result)
    if exercise_faults and (
        not all(result["negative_validation"].values())
        or result["legacy_output_max_abs"] > 0.02
        or result["legacy_lse_max_abs"] > 2.0e-5
        or result["invalid_scale_fault_status"] == 0
        or not result["invalid_scale_finite"]
    ):
        raise AssertionError(result)
    return result


def run(graph_replays: int = 100) -> list[dict[str, object]]:
    results = []
    for query_len in (1, 5):
        for batch in (8, 5, 1):
            results.append(
                _case(
                    batch,
                    query_len,
                    graph_replays=graph_replays,
                    exercise_faults=batch == 1 and query_len == 5,
                )
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int)
    parser.add_argument("--query-len", type=int)
    parser.add_argument("--graph-replays", type=int, default=100)
    parser.add_argument("--exercise-faults", action="store_true")
    args = parser.parse_args()
    if (args.batch is None) != (args.query_len is None):
        parser.error("--batch and --query-len must be provided together")
    if args.batch is None:
        run(args.graph_replays)
    else:
        _case(
            args.batch,
            args.query_len,
            graph_replays=args.graph_replays,
            exercise_faults=args.exercise_faults,
        )
