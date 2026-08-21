"""Matched graph benchmark for normalized, serial, and lookahead R31."""

from __future__ import annotations

import json
import math

import torch

from tokenspeed_mla import tokenspeed_mla_decode_tq_r31


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


def _case(batch: int, query_len: int, sequence_length: int = 7_440) -> dict[str, object]:
    device = torch.device("cuda:0")
    page_count = math.ceil(math.ceil(sequence_length / 32) / 4) * 4
    pages = batch * page_count
    query_latent = torch.zeros(
        (batch, query_len, 8, 512), device=device, dtype=torch.float8_e4m3fn
    )
    query_rope = torch.zeros(
        (batch, query_len, 8, 256), device=device, dtype=torch.float8_e4m3fn
    )
    packed = torch.zeros((pages, 32, 256), device=device, dtype=torch.uint8)
    scale = torch.ones((pages, 32), device=device, dtype=torch.bfloat16)
    high = torch.zeros(
        (pages, 32, 64), device=device, dtype=torch.float8_e4m3fn
    )
    residual = torch.zeros((pages, 32, 32), device=device, dtype=torch.uint8)
    table = torch.arange(pages, device=device, dtype=torch.int32).view(
        batch, page_count
    )
    lengths = torch.full(
        (batch,), sequence_length, device=device, dtype=torch.int32
    )
    control_workspace = torch.empty((256 << 20,), device=device, dtype=torch.int8)
    serial_workspace = torch.empty_like(control_workspace)
    candidate_workspace = torch.empty_like(control_workspace)
    control_out = torch.empty(
        (batch, query_len, 8, 512), device=device, dtype=torch.bfloat16
    )
    candidate_out = torch.empty_like(control_out)
    serial_out = torch.empty_like(control_out)

    common = dict(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed,
        reconstruction_scale=scale,
        high_rope=high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=lengths,
        max_seq_len=sequence_length,
        softmax_scale=0.125,
        enable_pdl=False,
    )
    tokenspeed_mla_decode_tq_r31(
        **common, workspace_buffer=control_workspace, out=control_out
    )
    tokenspeed_mla_decode_tq_r31(
        **common,
        workspace_buffer=serial_workspace,
        out=serial_out,
        _physical_split_score=True,
        _physical_split_score_lookahead=False,
    )
    tokenspeed_mla_decode_tq_r31(
        **common,
        workspace_buffer=candidate_workspace,
        out=candidate_out,
        _physical_split_score=True,
    )
    torch.cuda.synchronize()
    control_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(control_graph):
        tokenspeed_mla_decode_tq_r31(
            **common, workspace_buffer=control_workspace, out=control_out
        )
    serial_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(serial_graph):
        tokenspeed_mla_decode_tq_r31(
            **common,
            workspace_buffer=serial_workspace,
            out=serial_out,
            _physical_split_score=True,
            _physical_split_score_lookahead=False,
        )
    candidate_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(candidate_graph):
        tokenspeed_mla_decode_tq_r31(
            **common,
            workspace_buffer=candidate_workspace,
            out=candidate_out,
            _physical_split_score=True,
        )
    control_a_us = _measure(control_graph)
    serial_us = _measure(serial_graph)
    candidate_us = _measure(candidate_graph)
    control_b_us = _measure(control_graph)
    control_us = (control_a_us + control_b_us) / 2.0
    result = {
        "batch": batch,
        "query_len": query_len,
        "sequence_length": sequence_length,
        "control_a_us": control_a_us,
        "serial_us": serial_us,
        "candidate_us": candidate_us,
        "control_b_us": control_b_us,
        "candidate_delta_pct": (candidate_us / control_us - 1.0) * 100.0,
        "serial_delta_pct": (serial_us / control_us - 1.0) * 100.0,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def run() -> list[dict[str, object]]:
    cells = [_case(batch, query_len) for query_len in (1, 5) for batch in (1, 5, 8)]
    print(
        json.dumps(
            {
                "cells": cells,
                "arithmetic_mean_pct": sum(
                    float(cell["candidate_delta_pct"]) for cell in cells
                )
                / len(cells),
            },
            sort_keys=True,
        )
    )
    return cells


if __name__ == "__main__":
    run()
