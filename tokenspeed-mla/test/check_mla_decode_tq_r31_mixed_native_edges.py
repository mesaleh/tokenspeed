# Copyright (c) 2026 LightSeek Foundation

"""Correctness gate for mixed ownership, empty owners, and partial pages."""

import json
import math

import torch
from cutlass import Float32, Int32

import benchmark_mla_decode_tq_r31_mixed_native as benchmark
import benchmark_mla_decode_tq_r31_segmented as fixture
from tokenspeed_mla import (
    reduce_mla_mixed_workspace,
    tokenspeed_mla_decode,
    tokenspeed_mla_decode_tq_r31,
    tokenspeed_mla_decode_tq_r31_mixed,
)


def _page_table(lengths: list[int]) -> torch.Tensor:
    pages_per_request = [
        (length + fixture.PAGE - 1) // fixture.PAGE for length in lengths
    ]
    width = max(1, max(pages_per_request))
    rows = []
    offset = 0
    for page_count in pages_per_request:
        if page_count == 0:
            rows.append(torch.zeros(width, dtype=torch.int32, device="cuda"))
            continue
        pages = torch.arange(
            offset,
            offset + page_count,
            dtype=torch.int32,
            device="cuda",
        )
        offset += page_count
        if page_count < width:
            pages = torch.cat((pages, pages[-1:].expand(width - page_count)))
        rows.append(pages)
    return torch.stack(rows).contiguous()


def _merge(
    hot_output: torch.Tensor | None,
    hot_lse: torch.Tensor | None,
    cold_output: torch.Tensor | None,
    cold_lse: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hot_output is None:
        assert cold_output is not None and cold_lse is not None
        return cold_output, cold_lse
    if cold_output is None:
        assert hot_lse is not None
        return hot_output, hot_lse
    assert hot_lse is not None and cold_lse is not None
    maximum = torch.maximum(hot_lse, cold_lse)
    merged_lse = maximum + torch.log2(
        torch.exp2(hot_lse - maximum) + torch.exp2(cold_lse - maximum)
    )
    merged_output = hot_output.float() * torch.exp2(hot_lse - merged_lse).unsqueeze(
        -1
    ) + cold_output.float() * torch.exp2(cold_lse - merged_lse).unsqueeze(-1)
    return merged_output.to(torch.bfloat16), merged_lse


def _reference_owner(
    *,
    query: torch.Tensor,
    cache: torch.Tensor,
    table: torch.Tensor,
    sequence_length: int,
    causal: bool,
    splits: int,
    cold_query_rope: torch.Tensor | None = None,
    cold_scale: torch.Tensor | None = None,
    cold_high_rope: torch.Tensor | None = None,
    cold_residual_rope: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(
        1,
        benchmark.QUERY_LEN,
        benchmark.HEADS,
        benchmark.LATENT,
        dtype=torch.bfloat16,
        device="cuda",
    )
    lse = torch.empty(
        1,
        benchmark.QUERY_LEN,
        benchmark.HEADS,
        dtype=torch.float32,
        device="cuda",
    )
    rows = benchmark.QUERY_LEN * benchmark.HEADS
    workspace = torch.empty(
        rows * 64 * (benchmark.LATENT + 1) * 4,
        dtype=torch.int8,
        device="cuda",
    )
    sequence = torch.tensor([sequence_length], dtype=torch.int32, device="cuda")
    if cold_query_rope is None:
        tokenspeed_mla_decode(
            query=query,
            kv_cache=cache,
            workspace_buffer=workspace,
            kv_lora_rank=benchmark.LATENT,
            qk_rope_head_dim=benchmark.ROPE,
            block_tables=table,
            seq_lens=sequence,
            max_seq_len=sequence_length,
            softmax_scale=benchmark.SOFTMAX_SCALE,
            out=output,
            causal_mask=causal,
            enable_pdl=True,
            split_kv_override=splits,
            return_lse=True,
            lse_out=lse,
        )
    else:
        assert cold_scale is not None
        assert cold_high_rope is not None
        assert cold_residual_rope is not None
        tokenspeed_mla_decode_tq_r31(
            query_latent=query,
            query_rope=cold_query_rope,
            packed_latent=cache,
            reconstruction_scale=cold_scale,
            high_rope=cold_high_rope,
            residual_rope=cold_residual_rope,
            workspace_buffer=workspace,
            block_tables=table,
            seq_lens=sequence,
            max_seq_len=sequence_length,
            softmax_scale=benchmark.SOFTMAX_SCALE,
            out=output,
            causal_mask=causal,
            enable_pdl=True,
            split_kv_override=splits,
            return_lse=True,
            lse_out=lse,
        )
    return output, lse


def run() -> dict[str, object]:
    from sglang.kernels.ops.quantization.hadamard import (
        hadamard_transform_with_signs,
    )

    benchmark.BATCH = 8
    benchmark.QUERY_LEN = 5
    benchmark.HOT_SPLITS = 3
    benchmark.COLD_SPLITS = 14
    fixture.BATCH = benchmark.BATCH
    fixture.QUERY_LEN = benchmark.QUERY_LEN
    fixture.HOT_CAPACITY = 2_560 * benchmark.BATCH

    hot_lengths = [10_752, 0, 4_096, 0, 2_048, 1_024, 1_536, 1_024]
    cold_lengths = [fixture.SEQ_LEN - value for value in hot_lengths]
    if sum(hot_lengths) != fixture.HOT_CAPACITY:
        raise AssertionError("edge layout changed the frozen hot capacity")
    original_layout = fixture._layout_lengths
    fixture._layout_lengths = lambda _: (hot_lengths, cold_lengths)
    try:
        state, config, dense_query, _, _, _ = fixture._build_state("edge")
    finally:
        fixture._layout_lengths = original_layout

    hot_latent = state.hot_cache[..., : benchmark.LATENT].reshape(-1, benchmark.LATENT)
    rotated_hot_latent = torch.empty_like(hot_latent, dtype=torch.bfloat16)
    hadamard_transform_with_signs(
        hot_latent.to(torch.bfloat16),
        config.signs1,
        config.signs2,
        scale=1.0 / math.sqrt(benchmark.LATENT),
        out=rotated_hot_latent,
    )
    hot_cache = torch.cat(
        (
            rotated_hot_latent.view(
                state.hot_cache.shape[0], fixture.PAGE, benchmark.LATENT
            ),
            state.hot_cache[..., benchmark.LATENT :].to(torch.bfloat16),
        ),
        dim=-1,
    ).to(torch.float8_e4m3fn)
    hot_query = torch.cat(
        (
            state.r31_query_latent,
            dense_query[..., benchmark.LATENT :],
        ),
        dim=-1,
    ).contiguous()
    hot_table = _page_table(hot_lengths)
    cold_table = _page_table(cold_lengths)

    # Preserve the physical page allocation while making every non-empty final
    # page partial. This covers zero-length owners and non-page-aligned bounds
    # in the same graph-stable batch.
    hot_runtime = [length - 3 if length else 0 for length in hot_lengths]
    cold_runtime = [length - 5 if length else 0 for length in cold_lengths]
    hot_seq = torch.tensor(hot_runtime, dtype=torch.int32, device="cuda")
    cold_seq = torch.tensor(cold_runtime, dtype=torch.int32, device="cuda")
    causal_owner = [
        "hot",
        "cold",
        "hot",
        "cold",
        "hot",
        "cold",
        "hot",
        "cold",
    ]
    causal_delta = benchmark.QUERY_LEN - 1
    hot_causal_seq = torch.tensor(
        [
            length + (0 if owner == "hot" else causal_delta)
            for length, owner in zip(hot_runtime, causal_owner, strict=True)
        ],
        dtype=torch.int32,
        device="cuda",
    )
    cold_causal_seq = torch.tensor(
        [
            length + (0 if owner == "cold" else causal_delta)
            for length, owner in zip(cold_runtime, causal_owner, strict=True)
        ],
        dtype=torch.int32,
        device="cuda",
    )

    rows = benchmark.BATCH * benchmark.QUERY_LEN * benchmark.HEADS
    hot_workspace_bytes = rows * benchmark.HOT_SPLITS * (benchmark.LATENT + 1) * 4
    cold_workspace_bytes = rows * benchmark.COLD_SPLITS * (benchmark.LATENT + 1) * 4
    workspace = torch.empty(
        hot_workspace_bytes + cold_workspace_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    direct_workspace = torch.empty_like(workspace)
    hot_workspace = direct_workspace[:hot_workspace_bytes]
    cold_workspace = direct_workspace[hot_workspace_bytes:]
    output = torch.empty(
        benchmark.BATCH,
        benchmark.QUERY_LEN,
        benchmark.HEADS,
        benchmark.LATENT,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output_lse = torch.empty(
        benchmark.BATCH,
        benchmark.QUERY_LEN,
        benchmark.HEADS,
        dtype=torch.float32,
        device="cuda",
    )
    direct_output = torch.empty_like(output)
    direct_output_lse = torch.empty_like(output_lse)
    sentinel = torch.empty_like(output)
    compiled = benchmark._compile_mixed_producer(
        hot_query,
        hot_cache,
        hot_table,
        hot_workspace,
        hot_seq,
        state.r31_query_latent,
        state.r31_query_rope,
        state.packed_cold,
        state.cold_high,
        cold_table,
        cold_workspace,
        cold_seq,
        sentinel,
        state.cold_scale,
        state.cold_residual,
    )

    def direct_launch() -> None:
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            compiled(
                hot_query[..., : benchmark.LATENT],
                hot_query[..., benchmark.LATENT :],
                hot_cache[..., : benchmark.LATENT],
                hot_cache[..., benchmark.LATENT :],
                hot_table,
                hot_workspace,
                Int32(benchmark.HOT_SPLITS),
                hot_seq,
                hot_causal_seq,
                state.r31_query_latent,
                state.r31_query_rope,
                state.packed_cold,
                state.cold_high,
                cold_table,
                cold_workspace,
                Int32(benchmark.COLD_SPLITS),
                cold_seq,
                cold_causal_seq,
                sentinel,
                Float32(benchmark.SOFTMAX_SCALE),
                Float32(1.0),
                state.cold_scale,
                state.cold_residual,
            )
        reduce_mla_mixed_workspace(
            direct_workspace,
            hot_seq,
            cold_seq,
            benchmark.HOT_SPLITS,
            benchmark.COLD_SPLITS,
            direct_output,
            direct_output_lse,
        )

    def candidate_launch() -> None:
        tokenspeed_mla_decode_tq_r31_mixed(
            hot_query=hot_query,
            hot_cache=hot_cache,
            hot_block_tables=hot_table,
            hot_seq_lens=hot_seq,
            hot_causal_seqs=hot_causal_seq,
            hot_max_seq_len=max(hot_runtime),
            cold_query_latent=state.r31_query_latent,
            cold_query_rope=state.r31_query_rope,
            cold_packed_latent=state.packed_cold,
            cold_reconstruction_scale=state.cold_scale,
            cold_high_rope=state.cold_high,
            cold_residual_rope=state.cold_residual,
            cold_block_tables=cold_table,
            cold_seq_lens=cold_seq,
            cold_causal_seqs=cold_causal_seq,
            cold_max_seq_len=max(cold_runtime),
            workspace_buffer=workspace,
            hot_splits=benchmark.HOT_SPLITS,
            cold_splits=benchmark.COLD_SPLITS,
            softmax_scale=benchmark.SOFTMAX_SCALE,
            out=output,
            lse_out=output_lse,
            enable_pdl=True,
        )

    direct_workspace.view(torch.float32).fill_(float("nan"))
    workspace.view(torch.float32).fill_(float("nan"))
    direct_launch()
    candidate_launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, direct_output, rtol=0, atol=0)
    torch.testing.assert_close(output_lse, direct_output_lse, rtol=0, atol=0)
    first_output = output.clone()
    first_lse = output_lse.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate_launch()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, first_output, rtol=0, atol=0)
    torch.testing.assert_close(output_lse, first_lse, rtol=0, atol=0)

    reference_outputs = []
    reference_lses = []
    for request in range(benchmark.BATCH):
        hot_output = hot_lse = None
        cold_output = cold_lse = None
        if hot_runtime[request]:
            hot_output, hot_lse = _reference_owner(
                query=hot_query[request : request + 1],
                cache=hot_cache,
                table=hot_table[request : request + 1],
                sequence_length=hot_runtime[request],
                causal=causal_owner[request] == "hot",
                splits=benchmark.HOT_SPLITS,
            )
        if cold_runtime[request]:
            cold_output, cold_lse = _reference_owner(
                query=state.r31_query_latent[request : request + 1],
                cache=state.packed_cold,
                table=cold_table[request : request + 1],
                sequence_length=cold_runtime[request],
                causal=causal_owner[request] == "cold",
                splits=benchmark.COLD_SPLITS,
                cold_query_rope=state.r31_query_rope[request : request + 1],
                cold_scale=state.cold_scale,
                cold_high_rope=state.cold_high,
                cold_residual_rope=state.cold_residual,
            )
        merged_output, merged_lse = _merge(
            hot_output,
            hot_lse,
            cold_output,
            cold_lse,
        )
        reference_outputs.append(merged_output[0])
        reference_lses.append(merged_lse[0])
    reference_output = torch.stack(reference_outputs)
    reference_lse = torch.stack(reference_lses)
    output_max_abs = float((output.float() - reference_output.float()).abs().max())
    lse_max_abs = float((output_lse - reference_lse).abs().max())
    torch.testing.assert_close(output, reference_output, rtol=0, atol=1.0e-3)
    torch.testing.assert_close(output_lse, reference_lse, rtol=0, atol=1.0e-3)
    if not torch.isfinite(output).all() or not torch.isfinite(output_lse).all():
        raise AssertionError("mixed edge result contains NaN or Inf")

    return {
        "schema": "r31-r5-f1-mixed-native-edges-v1",
        "batch": benchmark.BATCH,
        "query_len": benchmark.QUERY_LEN,
        "hot_lengths": hot_runtime,
        "cold_lengths": cold_runtime,
        "causal_owner": causal_owner,
        "empty_hot_requests": [1, 3],
        "empty_cold_requests": [0],
        "all_nonempty_final_pages_partial": True,
        "bit_exact_public_api_vs_direct_mixed": True,
        "graph_replay_bit_exact": True,
        "output_max_abs": output_max_abs,
        "lse_max_abs": lse_max_abs,
        "within_tolerance": True,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
