# Copyright (c) 2026 LightSeek Foundation

"""R31-R0 correctness and timing gate for segmented FP8/R31 MLA decode.

The primary shape is Kimi K2.6 DFlash verification at c8/q5 with 10,752 live
rows per request.  It compares one dense FP8 call with a graph-stable segmented
path and includes compact dispatch, R31 inverse rotation, scatter, and LSE
merge in the candidate graph.  JIT compilation, allocation, and synchronization
are outside scored windows.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from tokenspeed_mla import tokenspeed_mla_decode, tokenspeed_mla_decode_tq_r31
from tokenspeed_mla.utils import get_num_sm

PAGE = 32
LATENT = 512
ROPE = 64
HEADS = 8
QUERY_LEN = 5
BATCH = 8
SEQ_LEN = 10_752
HOT_CAPACITY = 51_200
SOFTMAX_SCALE = 1.0 / math.sqrt(LATENT + ROPE)
DENSE_OUTPUT_ATOL = 1.0e-3
DENSE_LSE_ATOL = 1.0e-3


@triton.jit
def _finalize_segments_kernel(
    hot_output,
    hot_lse,
    cold_output,
    cold_lse,
    hot_map,
    cold_map,
    output,
    output_lse,
    Q: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch_index = row // (Q * H)
    query_index = (row // H) % Q
    head_index = row % H
    hot_index = tl.load(hot_map + batch_index)
    cold_index = tl.load(cold_map + batch_index)
    has_hot = hot_index >= 0
    has_cold = cold_index >= 0

    offsets = tl.arange(0, BLOCK_D)
    feature_mask = offsets < D
    hot_base = ((hot_index * Q + query_index) * H + head_index) * D
    cold_base = ((cold_index * Q + query_index) * H + head_index) * D
    hot_value = tl.load(
        hot_output + hot_base + offsets,
        mask=feature_mask & has_hot,
        other=0.0,
    ).to(tl.float32)
    cold_value = tl.load(
        cold_output + cold_base + offsets,
        mask=feature_mask & has_cold,
        other=0.0,
    ).to(tl.float32)
    hot_log = tl.load(
        hot_lse + (hot_index * Q + query_index) * H + head_index,
        mask=has_hot,
        other=-float("inf"),
    ).to(tl.float32)
    cold_log = tl.load(
        cold_lse + (cold_index * Q + query_index) * H + head_index,
        mask=has_cold,
        other=-float("inf"),
    ).to(tl.float32)

    both = has_hot & has_cold
    max_log = tl.maximum(hot_log, cold_log)
    has_mass = max_log != -float("inf")
    finite_combined_log = max_log + tl.log2(
        tl.exp2(hot_log - max_log) + tl.exp2(cold_log - max_log)
    )
    combined_log = tl.where(has_mass, finite_combined_log, -float("inf"))
    merged_log = tl.where(
        both,
        combined_log,
        tl.where(has_hot, hot_log, cold_log),
    )
    hot_weight = tl.where(has_mass, tl.exp2(hot_log - combined_log), 0.0)
    cold_weight = tl.where(has_mass, tl.exp2(cold_log - combined_log), 0.0)
    combined = hot_value * hot_weight + cold_value * cold_weight
    selected = tl.where(
        both,
        combined,
        tl.where(has_hot, hot_value, tl.where(has_cold, cold_value, 0.0)),
    )
    merged_log = tl.where(has_hot | has_cold, merged_log, -float("inf"))

    output_base = row * D
    tl.store(output + output_base + offsets, selected, mask=feature_mask)
    tl.store(output_lse + row, merged_log)


def _finalize_segments(
    hot_output: torch.Tensor,
    hot_lse: torch.Tensor,
    cold_output: torch.Tensor,
    cold_lse: torch.Tensor,
    hot_map: torch.Tensor,
    cold_map: torch.Tensor,
    output: torch.Tensor,
    output_lse: torch.Tensor,
) -> None:
    rows = output.shape[0] * output.shape[1] * output.shape[2]
    _finalize_segments_kernel[(rows,)](
        hot_output,
        hot_lse,
        cold_output,
        cold_lse,
        hot_map,
        cold_map,
        output,
        output_lse,
        Q=output.shape[1],
        H=output.shape[2],
        D=output.shape[3],
        BLOCK_D=triton.next_power_of_2(output.shape[3]),
        num_warps=4,
    )


def _page_table(
    requests: list[int], offsets: list[int], lengths: list[int]
) -> torch.Tensor:
    widths = [lengths[index] // PAGE for index in requests]
    max_pages = max(widths)
    padded_pages = math.ceil(max_pages / 4) * 4
    rows = []
    for request, width in zip(requests, widths, strict=True):
        pages = torch.arange(
            offsets[request],
            offsets[request] + width,
            dtype=torch.int32,
            device="cuda",
        )
        if width < padded_pages:
            pages = torch.cat((pages, pages[-1:].expand(padded_pages - width)))
        rows.append(pages)
    return torch.stack(rows).contiguous()


def _offsets(lengths: list[int]) -> list[int]:
    result = []
    pages = 0
    for length in lengths:
        result.append(pages)
        pages += length // PAGE
    return result


def _layout_lengths(name: str) -> tuple[list[int], list[int]]:
    if name == "balanced":
        hot = [HOT_CAPACITY // BATCH] * BATCH
        cold = [SEQ_LEN - value for value in hot]
    elif name == "skewed":
        hot = [SEQ_LEN] * 4 + [HOT_CAPACITY - 4 * SEQ_LEN] + [0] * 3
        cold = [SEQ_LEN - value for value in hot]
    else:
        raise ValueError(f"unknown layout {name!r}")
    if sum(hot) != HOT_CAPACITY or any(
        h + c != SEQ_LEN or h % PAGE or c % PAGE for h, c in zip(hot, cold, strict=True)
    ):
        raise AssertionError("layout does not preserve exact page ownership")
    return hot, cold


def _compact_segments(
    source: torch.Tensor, hot_lengths: list[int], cold_lengths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    cold_rows = []
    hot_rows = []
    for request in range(BATCH):
        cold_length = cold_lengths[request]
        if cold_length:
            cold_rows.append(source[request, :cold_length])
        if hot_lengths[request]:
            hot_rows.append(source[request, cold_length:])
    cold = torch.cat(cold_rows, dim=0).contiguous()
    hot = torch.cat(hot_rows, dim=0).to(torch.float8_e4m3fn).contiguous()
    return cold, hot


def _merge_reference(
    hot_output: torch.Tensor,
    hot_lse: torch.Tensor,
    cold_output: torch.Tensor,
    cold_lse: torch.Tensor,
    hot_map: list[int],
    cold_map: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    lses = []
    for hot_index, cold_index in zip(hot_map, cold_map, strict=True):
        if hot_index < 0:
            outputs.append(cold_output[cold_index])
            lses.append(cold_lse[cold_index])
            continue
        if cold_index < 0:
            outputs.append(hot_output[hot_index])
            lses.append(hot_lse[hot_index])
            continue
        hot_log = hot_lse[hot_index]
        cold_log = cold_lse[cold_index]
        maximum = torch.maximum(hot_log, cold_log)
        merged_log = maximum + torch.log2(
            torch.exp2(hot_log - maximum) + torch.exp2(cold_log - maximum)
        )
        merged = hot_output[hot_index].float() * torch.exp2(
            hot_log - merged_log
        ).unsqueeze(-1) + cold_output[cold_index].float() * torch.exp2(
            cold_log - merged_log
        ).unsqueeze(
            -1
        )
        outputs.append(merged.to(torch.bfloat16))
        lses.append(merged_log)
    return torch.stack(outputs), torch.stack(lses)


def _assert_empty_segment_merge() -> None:
    hot = torch.arange(8, dtype=torch.bfloat16, device="cuda").view(1, 1, 1, 8)
    cold = (hot + 17).contiguous()
    hot_lse = torch.empty(1, 1, 1, dtype=torch.float32, device="cuda")
    cold_lse = torch.empty_like(hot_lse)
    output = torch.empty_like(hot)
    output_lse = torch.empty_like(hot_lse)
    present = torch.tensor([0], dtype=torch.int32, device="cuda")

    hot_lse.fill_(float("-inf"))
    cold_lse.fill_(3.0)
    _finalize_segments(
        hot, hot_lse, cold, cold_lse, present, present, output, output_lse
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, cold, rtol=0, atol=0)
    torch.testing.assert_close(output_lse, cold_lse, rtol=0, atol=0)

    hot_lse.fill_(2.0)
    cold_lse.fill_(float("-inf"))
    _finalize_segments(
        hot, hot_lse, cold, cold_lse, present, present, output, output_lse
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, hot, rtol=0, atol=0)
    torch.testing.assert_close(output_lse, hot_lse, rtol=0, atol=0)

    hot_lse.fill_(float("-inf"))
    _finalize_segments(
        hot, hot_lse, cold, cold_lse, present, present, output, output_lse
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    assert torch.isneginf(output_lse).all()


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


@dataclass
class SegmentState:
    hot_requests: list[int]
    prefix_requests: list[int]
    all_cold_requests: list[int]
    hot_map: list[int]
    cold_map: list[int]
    hot_query: torch.Tensor
    r31_query_latent: torch.Tensor
    r31_query_rope: torch.Tensor
    hot_cache: torch.Tensor
    packed_cold: torch.Tensor
    cold_scale: torch.Tensor
    cold_high: torch.Tensor
    cold_residual: torch.Tensor
    hot_table: torch.Tensor
    prefix_table: torch.Tensor | None
    all_cold_table: torch.Tensor | None
    hot_seq: torch.Tensor
    prefix_seq: torch.Tensor | None
    all_cold_seq: torch.Tensor | None


def _build_state(layout: str):
    from sglang.kernels.jit.tq_mla_frontend_n10_native import (
        tq_mla_r31_native_cache_writer_out,
        tq_mla_r31_native_frontend_out,
    )
    from sglang.srt.layers.quantization.kv_turboquant import NativeE2M1MLAConfig

    torch.manual_seed(0xA173100 + (0 if layout == "balanced" else 1))
    config = NativeE2M1MLAConfig(device="cuda")
    hot_lengths, cold_lengths = _layout_lengths(layout)
    source = (torch.randn(BATCH, SEQ_LEN, LATENT + ROPE, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    query_source = (
        torch.randn(BATCH, QUERY_LEN, HEADS, LATENT + ROPE, device="cuda") * 0.1
    ).to(torch.bfloat16)
    dense_query = query_source.to(torch.float8_e4m3fn).contiguous()
    dense_cache = source.to(torch.float8_e4m3fn).view(-1, PAGE, LATENT + ROPE)
    dense_table = torch.arange(
        BATCH * (SEQ_LEN // PAGE), dtype=torch.int32, device="cuda"
    ).view(BATCH, SEQ_LEN // PAGE)
    dense_seq = torch.full((BATCH,), SEQ_LEN, dtype=torch.int32, device="cuda")

    cold_source, hot_cache = _compact_segments(source, hot_lengths, cold_lengths)
    cold_tokens = cold_source.shape[0]
    packed_flat = torch.empty(
        cold_tokens, 1, LATENT // 2, dtype=torch.uint8, device="cuda"
    )
    scale_flat = torch.empty(cold_tokens, 1, dtype=torch.bfloat16, device="cuda")
    high_flat = torch.empty(
        cold_tokens, 1, ROPE, dtype=torch.float8_e4m3fn, device="cuda"
    )
    residual_flat = torch.empty(
        cold_tokens, 1, ROPE // 2, dtype=torch.uint8, device="cuda"
    )
    fault = torch.zeros(1, dtype=torch.int32, device="cuda")
    zero_count = torch.zeros(1, dtype=torch.int64, device="cuda")
    tq_mla_r31_native_cache_writer_out(
        cold_source[:, None, :LATENT],
        cold_source[:, None, LATENT:],
        torch.arange(cold_tokens, dtype=torch.int32, device="cuda"),
        config.signs1,
        config.signs2,
        config.rope_signs1,
        config.rope_signs2,
        packed_flat,
        scale_flat,
        high_flat,
        residual_flat,
        fault,
        zero_count,
        grid=config.grid,
        residual_scale=config.r31_residual_scale,
        strict=True,
    )

    query_tokens = BATCH * QUERY_LEN
    r31_query_latent = torch.empty(
        query_tokens, HEADS, LATENT, dtype=torch.float8_e4m3fn, device="cuda"
    )
    r31_query_rope_planes = torch.empty(
        query_tokens,
        HEADS,
        4,
        ROPE,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    dummy_packed = torch.empty(
        query_tokens, 1, LATENT // 2, dtype=torch.uint8, device="cuda"
    )
    dummy_scale = torch.empty(query_tokens, 1, dtype=torch.bfloat16, device="cuda")
    dummy_high = torch.empty(
        query_tokens, 1, ROPE, dtype=torch.float8_e4m3fn, device="cuda"
    )
    dummy_residual = torch.empty(
        query_tokens, 1, ROPE // 2, dtype=torch.uint8, device="cuda"
    )
    tq_mla_r31_native_frontend_out(
        query_source[..., :LATENT].reshape(query_tokens, HEADS, LATENT),
        query_source[..., LATENT:].reshape(query_tokens, HEADS, ROPE),
        source[:, -QUERY_LEN:, :LATENT].reshape(query_tokens, 1, LATENT),
        source[:, -QUERY_LEN:, LATENT:].reshape(query_tokens, 1, ROPE),
        torch.arange(query_tokens, dtype=torch.int32, device="cuda"),
        config.signs1,
        config.signs2,
        config.rope_signs1,
        config.rope_signs2,
        r31_query_latent,
        r31_query_rope_planes,
        dummy_packed,
        dummy_scale,
        dummy_high,
        dummy_residual,
        fault,
        zero_count,
        grid=config.grid,
        residual_scale=config.r31_residual_scale,
        strict=True,
    )
    torch.cuda.synchronize()
    if int(fault.item()) != 0:
        raise AssertionError(f"R31 frontend fault status {int(fault.item())}")
    r31_query_latent = r31_query_latent.view(
        BATCH, QUERY_LEN, HEADS, LATENT
    ).contiguous()
    r31_query_rope = r31_query_rope_planes.view(
        BATCH, QUERY_LEN, HEADS, 4 * ROPE
    ).contiguous()

    hot_requests = [index for index, length in enumerate(hot_lengths) if length]
    prefix_requests = [
        index
        for index, (hot_length, cold_length) in enumerate(
            zip(hot_lengths, cold_lengths, strict=True)
        )
        if hot_length and cold_length
    ]
    all_cold_requests = [
        index
        for index, (hot_length, cold_length) in enumerate(
            zip(hot_lengths, cold_lengths, strict=True)
        )
        if not hot_length and cold_length
    ]
    cold_requests = prefix_requests + all_cold_requests
    hot_map = [
        hot_requests.index(index) if index in hot_requests else -1
        for index in range(BATCH)
    ]
    cold_map = [
        cold_requests.index(index) if index in cold_requests else -1
        for index in range(BATCH)
    ]
    hot_offsets = _offsets(hot_lengths)
    cold_offsets = _offsets(cold_lengths)
    index = torch.tensor(hot_requests, dtype=torch.int64, device="cuda")
    hot_query = dense_query.index_select(0, index).contiguous()
    hot_table = _page_table(hot_requests, hot_offsets, hot_lengths)
    hot_seq = torch.tensor(
        [hot_lengths[index] for index in hot_requests],
        dtype=torch.int32,
        device="cuda",
    )
    prefix_table = (
        _page_table(prefix_requests, cold_offsets, cold_lengths)
        if prefix_requests
        else None
    )
    prefix_seq = (
        torch.tensor(
            [cold_lengths[index] for index in prefix_requests],
            dtype=torch.int32,
            device="cuda",
        )
        if prefix_requests
        else None
    )
    all_cold_table = (
        _page_table(all_cold_requests, cold_offsets, cold_lengths)
        if all_cold_requests
        else None
    )
    all_cold_seq = (
        torch.tensor(
            [cold_lengths[index] for index in all_cold_requests],
            dtype=torch.int32,
            device="cuda",
        )
        if all_cold_requests
        else None
    )
    state = SegmentState(
        hot_requests=hot_requests,
        prefix_requests=prefix_requests,
        all_cold_requests=all_cold_requests,
        hot_map=hot_map,
        cold_map=cold_map,
        hot_query=hot_query,
        r31_query_latent=r31_query_latent,
        r31_query_rope=r31_query_rope,
        hot_cache=hot_cache.view(-1, PAGE, LATENT + ROPE),
        packed_cold=packed_flat.view(-1, PAGE, LATENT // 2),
        cold_scale=scale_flat.view(-1, PAGE),
        cold_high=high_flat.view(-1, PAGE, ROPE),
        cold_residual=residual_flat.view(-1, PAGE, ROPE // 2),
        hot_table=hot_table,
        prefix_table=prefix_table,
        all_cold_table=all_cold_table,
        hot_seq=hot_seq,
        prefix_seq=prefix_seq,
        all_cold_seq=all_cold_seq,
    )
    return state, config, dense_query, dense_cache, dense_table, dense_seq


def run(layout: str, windows: int, replays: int) -> dict[str, object]:
    from sglang.kernels.ops.quantization.hadamard import (
        hadamard_transform_with_signs,
    )

    _assert_empty_segment_merge()
    state, config, dense_query, dense_cache, dense_table, dense_seq = _build_state(
        layout
    )
    workspace_bytes = (
        get_num_sm(torch.device("cuda")) * HEADS * QUERY_LEN * (LATENT + 1) * 4
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.int8, device="cuda")

    full_output = torch.empty(
        BATCH, QUERY_LEN, HEADS, LATENT, dtype=torch.bfloat16, device="cuda"
    )
    full_lse = torch.empty(BATCH, QUERY_LEN, HEADS, dtype=torch.float32, device="cuda")
    all_hot_output = torch.empty_like(full_output)
    all_hot_lse = torch.empty_like(full_lse)
    hot_batch = len(state.hot_requests)
    cold_requests = state.prefix_requests + state.all_cold_requests
    cold_batch = len(cold_requests)
    hot_output = torch.empty(
        hot_batch, QUERY_LEN, HEADS, LATENT, dtype=torch.bfloat16, device="cuda"
    )
    hot_lse = torch.empty(
        hot_batch, QUERY_LEN, HEADS, dtype=torch.float32, device="cuda"
    )
    cold_rotated = torch.empty(
        cold_batch, QUERY_LEN, HEADS, LATENT, dtype=torch.bfloat16, device="cuda"
    )
    cold_output = torch.empty_like(cold_rotated)
    cold_lse = torch.empty(
        cold_batch, QUERY_LEN, HEADS, dtype=torch.float32, device="cuda"
    )
    final_output = torch.empty_like(full_output)
    final_lse = torch.empty_like(full_lse)
    hot_map_device = torch.tensor(state.hot_map, dtype=torch.int32, device="cuda")
    cold_map_device = torch.tensor(state.cold_map, dtype=torch.int32, device="cuda")
    hot_max_seq_len = int(state.hot_seq.max().item())

    def baseline_launch() -> None:
        tokenspeed_mla_decode(
            query=dense_query,
            kv_cache=dense_cache,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=dense_table,
            seq_lens=dense_seq,
            max_seq_len=SEQ_LEN,
            softmax_scale=SOFTMAX_SCALE,
            out=full_output,
            causal_mask=True,
            enable_pdl=True,
            return_lse=True,
            lse_out=full_lse,
        )

    def all_hot_dispatch_launch() -> None:
        # The all-hot branch is deliberately the same single FP8 API call: no
        # R31 launch, inverse rotation, finalizer, or alternate causal metadata.
        tokenspeed_mla_decode(
            query=dense_query,
            kv_cache=dense_cache,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=dense_table,
            seq_lens=dense_seq,
            max_seq_len=SEQ_LEN,
            softmax_scale=SOFTMAX_SCALE,
            out=all_hot_output,
            causal_mask=True,
            enable_pdl=True,
            return_lse=True,
            lse_out=all_hot_lse,
        )

    prefix_count = len(state.prefix_requests)
    all_cold_count = len(state.all_cold_requests)
    prefix_max_seq_len = int(state.prefix_seq.max().item()) if prefix_count else None
    all_cold_max_seq_len = (
        int(state.all_cold_seq.max().item()) if all_cold_count else None
    )

    def r31_launch(
        requests: list[int],
        table: torch.Tensor,
        seq: torch.Tensor,
        output_slice: torch.Tensor,
        lse_slice: torch.Tensor,
        *,
        causal: bool,
    ) -> None:
        request_index = torch.tensor(requests, dtype=torch.int64, device="cuda")
        tokenspeed_mla_decode_tq_r31(
            query_latent=state.r31_query_latent.index_select(0, request_index),
            query_rope=state.r31_query_rope.index_select(0, request_index),
            packed_latent=state.packed_cold,
            reconstruction_scale=state.cold_scale,
            high_rope=state.cold_high,
            residual_rope=state.cold_residual,
            workspace_buffer=workspace,
            block_tables=table,
            seq_lens=seq,
            max_seq_len=int(seq.max().item()),
            softmax_scale=SOFTMAX_SCALE,
            out=output_slice,
            causal_mask=causal,
            enable_pdl=True,
            return_lse=True,
            lse_out=lse_slice,
        )

    # Pre-materialize compact query tensors; index_select must not occur in the
    # scored graph.
    prefix_index = torch.tensor(state.prefix_requests, dtype=torch.int64, device="cuda")
    all_cold_index = torch.tensor(
        state.all_cold_requests, dtype=torch.int64, device="cuda"
    )
    prefix_q_latent = state.r31_query_latent.index_select(0, prefix_index)
    prefix_q_rope = state.r31_query_rope.index_select(0, prefix_index)
    all_cold_q_latent = state.r31_query_latent.index_select(0, all_cold_index)
    all_cold_q_rope = state.r31_query_rope.index_select(0, all_cold_index)

    def hot_launch() -> None:
        tokenspeed_mla_decode(
            query=state.hot_query,
            kv_cache=state.hot_cache,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=state.hot_table,
            seq_lens=state.hot_seq,
            max_seq_len=hot_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=hot_output,
            causal_mask=True,
            enable_pdl=True,
            return_lse=True,
            lse_out=hot_lse,
        )

    def prefix_launch() -> None:
        tokenspeed_mla_decode_tq_r31(
            query_latent=prefix_q_latent,
            query_rope=prefix_q_rope,
            packed_latent=state.packed_cold,
            reconstruction_scale=state.cold_scale,
            high_rope=state.cold_high,
            residual_rope=state.cold_residual,
            workspace_buffer=workspace,
            block_tables=state.prefix_table,
            seq_lens=state.prefix_seq,
            max_seq_len=prefix_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=cold_rotated[:prefix_count],
            causal_mask=False,
            enable_pdl=True,
            return_lse=True,
            lse_out=cold_lse[:prefix_count],
        )

    def all_cold_launch() -> None:
        tokenspeed_mla_decode_tq_r31(
            query_latent=all_cold_q_latent,
            query_rope=all_cold_q_rope,
            packed_latent=state.packed_cold,
            reconstruction_scale=state.cold_scale,
            high_rope=state.cold_high,
            residual_rope=state.cold_residual,
            workspace_buffer=workspace,
            block_tables=state.all_cold_table,
            seq_lens=state.all_cold_seq,
            max_seq_len=all_cold_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=cold_rotated[prefix_count:],
            causal_mask=True,
            enable_pdl=True,
            return_lse=True,
            lse_out=cold_lse[prefix_count:],
        )

    def inverse_launch() -> None:
        hadamard_transform_with_signs(
            cold_rotated,
            config.signs2,
            config.signs1,
            scale=1.0 / math.sqrt(LATENT),
            out=cold_output,
        )

    def finalize_launch() -> None:
        _finalize_segments(
            hot_output,
            hot_lse,
            cold_output,
            cold_lse,
            hot_map_device,
            cold_map_device,
            final_output,
            final_lse,
        )

    def candidate_launch() -> None:
        attention_only_launch()
        postprocess_launch()

    def attention_only_launch() -> None:
        hot_launch()
        if prefix_count:
            prefix_launch()
        if all_cold_count:
            all_cold_launch()

    def postprocess_launch() -> None:
        inverse_launch()
        finalize_launch()

    baseline_launch()
    all_hot_dispatch_launch()
    candidate_launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(all_hot_output, full_output, rtol=0, atol=0)
    torch.testing.assert_close(all_hot_lse, full_lse, rtol=0, atol=0)

    reference_output, reference_lse = _merge_reference(
        hot_output,
        hot_lse,
        cold_output,
        cold_lse,
        state.hot_map,
        state.cold_map,
    )
    torch.testing.assert_close(final_output, reference_output, rtol=0, atol=0.01)
    torch.testing.assert_close(final_lse, reference_lse, rtol=0, atol=2e-5)

    causal_checks: dict[str, object] = {}
    if prefix_count:
        causal_output = torch.empty_like(cold_rotated[:prefix_count])
        causal_lse = torch.empty_like(cold_lse[:prefix_count])
        r31_launch(
            state.prefix_requests,
            state.prefix_table,
            state.prefix_seq,
            causal_output,
            causal_lse,
            causal=True,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            causal_output[:, -1], cold_rotated[:prefix_count, -1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            causal_lse[:, -1], cold_lse[:prefix_count, -1], rtol=0, atol=0
        )
        earlier_delta = float(
            (causal_lse[:, :-1] - cold_lse[:prefix_count, :-1]).abs().max()
        )
        if earlier_delta == 0.0:
            raise AssertionError("q5 causal override did not alter prefix visibility")
        causal_checks["prefix_earlier_lse_max_abs_delta"] = earlier_delta
    if all_cold_count:
        default_output = torch.empty_like(cold_rotated[prefix_count:])
        default_lse = torch.empty_like(cold_lse[prefix_count:])
        request_index = torch.tensor(
            state.all_cold_requests, dtype=torch.int64, device="cuda"
        )
        tokenspeed_mla_decode_tq_r31(
            query_latent=state.r31_query_latent.index_select(0, request_index),
            query_rope=state.r31_query_rope.index_select(0, request_index),
            packed_latent=state.packed_cold,
            reconstruction_scale=state.cold_scale,
            high_rope=state.cold_high,
            residual_rope=state.cold_residual,
            workspace_buffer=workspace,
            block_tables=state.all_cold_table,
            seq_lens=state.all_cold_seq,
            max_seq_len=int(state.all_cold_seq.max().item()),
            softmax_scale=SOFTMAX_SCALE,
            out=default_output,
            enable_pdl=True,
            return_lse=True,
            lse_out=default_lse,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            default_output, cold_rotated[prefix_count:], rtol=0, atol=0
        )
        torch.testing.assert_close(default_lse, cold_lse[prefix_count:], rtol=0, atol=0)
        causal_checks["default_q5_matches_explicit_causal"] = True

    # Quantization is not the segmented-algebra oracle, but report its observed
    # distance to the original dense FP8 path for later endpoint correlation.
    dense_output_max_abs = float(
        (final_output.float() - full_output.float()).abs().max()
    )
    dense_lse_max_abs = float((final_lse - full_lse).abs().max())
    if dense_output_max_abs > DENSE_OUTPUT_ATOL:
        raise AssertionError(
            "segmented output exceeds the dense reconstruction tolerance: "
            f"{dense_output_max_abs} > {DENSE_OUTPUT_ATOL}"
        )
    if dense_lse_max_abs > DENSE_LSE_ATOL:
        raise AssertionError(
            "segmented LSE exceeds the dense reconstruction tolerance: "
            f"{dense_lse_max_abs} > {DENSE_LSE_ATOL}"
        )

    baseline_graph = _capture(baseline_launch)
    candidate_graph = _capture(candidate_launch)
    phase_graphs = {
        "attention_only": _capture(attention_only_launch),
        "hot_fp8": _capture(hot_launch),
        "inverse_rotation": _capture(inverse_launch),
        "finalize_merge": _capture(finalize_launch),
        "postprocess": _capture(postprocess_launch),
    }
    if prefix_count:
        phase_graphs["r31_noncausal_prefix"] = _capture(prefix_launch)
    if all_cold_count:
        phase_graphs["r31_causal_all_cold"] = _capture(all_cold_launch)
    for _ in range(12):
        _measure(baseline_graph, replays)
        _measure(candidate_graph, replays)
    baseline_samples = []
    candidate_samples = []
    for window in range(windows):
        if window % 2:
            candidate_samples.append(_measure(candidate_graph, replays))
            baseline_samples.append(_measure(baseline_graph, replays))
        else:
            baseline_samples.append(_measure(baseline_graph, replays))
            candidate_samples.append(_measure(candidate_graph, replays))

    baseline_mean = statistics.fmean(baseline_samples)
    candidate_mean = statistics.fmean(candidate_samples)
    added_us = candidate_mean - baseline_mean
    phase_mean_us = {}
    for name, graph in phase_graphs.items():
        for _ in range(5):
            _measure(graph, replays)
        phase_mean_us[name] = statistics.fmean(
            _measure(graph, replays) for _ in range(windows)
        )
    candidate_scratch_bytes = (
        workspace.numel()
        + hot_output.numel() * hot_output.element_size()
        + hot_lse.numel() * hot_lse.element_size()
        + cold_rotated.numel() * cold_rotated.element_size()
        + cold_output.numel() * cold_output.element_size()
        + cold_lse.numel() * cold_lse.element_size()
        + final_output.numel() * final_output.element_size()
        + final_lse.numel() * final_lse.element_size()
        + state.hot_query.numel() * state.hot_query.element_size()
        + prefix_q_latent.numel() * prefix_q_latent.element_size()
        + prefix_q_rope.numel() * prefix_q_rope.element_size()
        + all_cold_q_latent.numel() * all_cold_q_latent.element_size()
        + all_cold_q_rope.numel() * all_cold_q_rope.element_size()
        + state.hot_table.numel() * state.hot_table.element_size()
        + (
            0
            if state.prefix_table is None
            else state.prefix_table.numel() * state.prefix_table.element_size()
        )
        + (
            0
            if state.all_cold_table is None
            else state.all_cold_table.numel() * state.all_cold_table.element_size()
        )
        + state.hot_seq.numel() * state.hot_seq.element_size()
        + (
            0
            if state.prefix_seq is None
            else state.prefix_seq.numel() * state.prefix_seq.element_size()
        )
        + (
            0
            if state.all_cold_seq is None
            else state.all_cold_seq.numel() * state.all_cold_seq.element_size()
        )
        + hot_map_device.numel() * hot_map_device.element_size()
        + cold_map_device.numel() * cold_map_device.element_size()
    )
    incremental_segment_scratch_bytes = candidate_scratch_bytes - workspace.numel()
    return {
        "status": "PASS" if added_us <= 13.0 else "FAIL_PERF",
        "layout": layout,
        "batch": BATCH,
        "query_len": QUERY_LEN,
        "seq_len": SEQ_LEN,
        "hot_lengths": _layout_lengths(layout)[0],
        "cold_lengths": _layout_lengths(layout)[1],
        "hot_requests": state.hot_requests,
        "prefix_requests": state.prefix_requests,
        "all_cold_requests": state.all_cold_requests,
        "baseline_mean_us": baseline_mean,
        "baseline_median_us": statistics.median(baseline_samples),
        "candidate_mean_us": candidate_mean,
        "candidate_median_us": statistics.median(candidate_samples),
        "added_mean_us": added_us,
        "delta_pct": (candidate_mean / baseline_mean - 1.0) * 100.0,
        "preferred_6_5us_gate": added_us <= 6.5,
        "relaxed_13us_gate": added_us <= 13.0,
        "merge_output_max_abs": float(
            (final_output.float() - reference_output.float()).abs().max()
        ),
        "merge_lse_max_abs": float((final_lse - reference_lse).abs().max()),
        "all_hot_output_bit_identical": True,
        "all_hot_lse_bit_identical": True,
        "empty_segment_merge_passed": True,
        "dense_output_max_abs": dense_output_max_abs,
        "dense_lse_max_abs": dense_lse_max_abs,
        "dense_output_atol": DENSE_OUTPUT_ATOL,
        "dense_lse_atol": DENSE_LSE_ATOL,
        "dense_reconstruction_gate": True,
        "causal_checks": causal_checks,
        "workspace_bytes": workspace.numel(),
        "candidate_scratch_bytes": candidate_scratch_bytes,
        "candidate_scratch_mib": candidate_scratch_bytes / (1 << 20),
        "incremental_segment_scratch_bytes": incremental_segment_scratch_bytes,
        "incremental_segment_scratch_mib": incremental_segment_scratch_bytes
        / (1 << 20),
        "phase_mean_us": phase_mean_us,
        "windows": windows,
        "replays_per_window": replays,
        "baseline_samples_us": baseline_samples,
        "candidate_samples_us": candidate_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout", choices=("balanced", "skewed", "both"), default="both"
    )
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--replays", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("R31-R0 requires one SM100 GPU")
    if args.windows <= 0 or args.replays <= 0:
        raise ValueError("windows and replays must be positive")
    layouts = ("balanced", "skewed") if args.layout == "both" else (args.layout,)
    results = [run(layout, args.windows, args.replays) for layout in layouts]
    print(json.dumps({"schema_version": 1, "results": results}, sort_keys=True))
    if any(result["status"] != "PASS" for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
