# Copyright (c) 2026 LightSeek Foundation

"""Bracket the public packed E2M1 reader against dense FP8 MLA decode."""

import math
import os
import statistics

import torch
from tokenspeed_mla import (
    tokenspeed_mla_decode,
    tokenspeed_mla_decode_tq_e2m1,
)
from tokenspeed_mla.utils import get_num_sm

PAGE_SIZE = 32
LATENT = 512
ROPE = 64
HEADS = 8
E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _build_cache(batch: int, seq_len: int):
    if seq_len % PAGE_SIZE:
        raise ValueError("benchmark sequence length must be page aligned")
    device = torch.device("cuda")
    pages_per_request = seq_len // PAGE_SIZE
    num_pages = batch * pages_per_request
    num_tokens = num_pages * PAGE_SIZE

    packed = torch.empty(
        (num_pages, PAGE_SIZE, LATENT // 2), dtype=torch.uint8, device=device
    )
    scale = torch.empty((num_pages, PAGE_SIZE), dtype=torch.bfloat16, device=device)
    reciprocal_rope = torch.empty(
        (num_pages, PAGE_SIZE, ROPE), dtype=torch.bfloat16, device=device
    )
    dense_cache = torch.empty(
        (num_pages, PAGE_SIZE, LATENT + ROPE),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    packed_flat = packed.view(num_tokens, LATENT // 2)
    scale_flat = scale.view(num_tokens)
    reciprocal_rope_flat = reciprocal_rope.view(num_tokens, ROPE)
    dense_flat = dense_cache.view(num_tokens, LATENT + ROPE)
    pair_dim = torch.arange(LATENT // 2, device=device)[None, :]
    rope_dim = torch.arange(ROPE, device=device)[None, :]
    lut = E2M1.to(device)
    chunk = 4096
    for begin in range(0, num_tokens, chunk):
        end = min(begin + chunk, num_tokens)
        token = torch.arange(begin, end, device=device)[:, None]
        low = ((token * 5 + (pair_dim * 2) * 3 + 1) % 15 + 1).to(torch.uint8)
        high = ((token * 5 + (pair_dim * 2 + 1) * 3 + 1) % 15 + 1).to(torch.uint8)
        packed_flat[begin:end] = low | (high << 4)
        token_scale = torch.pow(2.0, (-8 + token[:, 0] % 7).float()).to(torch.bfloat16)
        scale_flat[begin:end] = token_scale
        dense_flat[begin:end, :LATENT:2] = (
            lut[low.long()] * token_scale[:, None].float()
        ).to(torch.float8_e4m3fn)
        dense_flat[begin:end, 1:LATENT:2] = (
            lut[high.long()] * token_scale[:, None].float()
        ).to(torch.float8_e4m3fn)
        raw_rope = (((token * 7 + rope_dim * 11 + 3) % 9) - 4) / 2.0
        reciprocal_rope_flat[begin:end] = (raw_rope / token_scale[:, None].float()).to(
            torch.bfloat16
        )
        dense_flat[begin:end, LATENT:] = raw_rope.to(torch.float8_e4m3fn)

    rows = []
    for request in range(batch):
        pages = torch.arange(
            request * pages_per_request,
            (request + 1) * pages_per_request,
            dtype=torch.int32,
            device=device,
        )
        if request % 2:
            pages = torch.flip(pages, dims=(0,))
        rows.append(pages)
    block_tables = torch.stack(rows).contiguous()
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    return packed, scale, reciprocal_rope, dense_cache, block_tables, seq_lens


def _capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
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


def _run_shape(
    query_len,
    batch,
    seq_len,
    cache,
    replays,
    windows,
):
    packed, scale, reciprocal_rope, dense_cache, block_tables, seq_lens = cache
    torch.manual_seed(20260816 + query_len)
    query_latent = (
        torch.randint(
            -4,
            5,
            (batch, query_len, HEADS, LATENT),
            device="cuda",
        )
        / 2.0
    ).to(torch.float8_e4m3fn)
    query_rope = (
        torch.randint(
            -4,
            5,
            (batch, query_len, HEADS, ROPE),
            device="cuda",
        )
        / 2.0
    ).to(torch.bfloat16)
    dense_query = torch.cat(
        (query_latent, query_rope.to(torch.float8_e4m3fn)), dim=-1
    ).contiguous()
    candidate_workspace = torch.empty(
        get_num_sm(torch.device("cuda")) * HEADS * query_len * (LATENT + 1) * 4,
        dtype=torch.int8,
        device="cuda",
    )
    dense_workspace = torch.empty_like(candidate_workspace)
    candidate_out = torch.empty(
        (batch, query_len, HEADS, LATENT), dtype=torch.bfloat16, device="cuda"
    )
    dense_out = torch.empty_like(candidate_out)
    dense_block_tables = block_tables.clone()
    dense_seq_lens = seq_lens.clone()
    softmax_scale = 1.0 / math.sqrt(LATENT + ROPE)

    def candidate_call():
        tokenspeed_mla_decode_tq_e2m1(
            query_latent=query_latent,
            query_rope=query_rope,
            packed_latent=packed,
            reconstruction_scale=scale,
            reciprocal_rope=reciprocal_rope,
            workspace_buffer=candidate_workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=seq_len,
            softmax_scale=softmax_scale,
            out=candidate_out,
        )

    def dense_call():
        tokenspeed_mla_decode(
            query=dense_query,
            kv_cache=dense_cache,
            workspace_buffer=dense_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=dense_block_tables,
            seq_lens=dense_seq_lens,
            max_seq_len=seq_len,
            softmax_scale=softmax_scale,
            out=dense_out,
        )

    candidate_graph = _capture(candidate_call)
    dense_graph = _capture(dense_call)
    for _ in range(12):
        _measure(candidate_graph, replays)
        _measure(dense_graph, replays)

    candidate_samples = []
    dense_samples = []
    for window in range(windows):
        order = (
            ((candidate_graph, candidate_samples), (dense_graph, dense_samples))
            if window % 2 == 0
            else ((dense_graph, dense_samples), (candidate_graph, candidate_samples))
        )
        for graph, samples in order:
            samples.append(_measure(graph, replays))

    log_ratios = [
        math.log(candidate / dense)
        for candidate, dense in zip(candidate_samples, dense_samples)
    ]
    mean_log = statistics.fmean(log_ratios)
    standard_error = statistics.stdev(log_ratios) / math.sqrt(windows)
    critical = 2.045229642 if windows == 30 else 2.262157163
    ratio = math.exp(mean_log)
    lower = math.exp(mean_log - critical * standard_error)
    upper = math.exp(mean_log + critical * standard_error)
    print(
        "TQ_E2M1_PUBLIC_TIMING "
        f"q_len={query_len} batch={batch} seq_len={seq_len} "
        f"dense_us={statistics.fmean(dense_samples):.6f} "
        f"candidate_us={statistics.fmean(candidate_samples):.6f} "
        f"candidate_over_dense={ratio:.6f} "
        f"ci95=[{lower:.6f},{upper:.6f}] "
        f"windows={windows} replays={replays}",
        flush=True,
    )


def main():
    batch = int(os.environ.get("TQ_PUBLIC_BATCH", "100"))
    seq_len = int(os.environ.get("TQ_PUBLIC_SEQ_LEN", "10240"))
    replays = int(os.environ.get("TQ_PUBLIC_REPLAYS", "100"))
    windows = int(os.environ.get("TQ_PUBLIC_WINDOWS", "10"))
    if windows not in (10, 30):
        raise ValueError("TQ_PUBLIC_WINDOWS must be 10 or 30")
    cache = _build_cache(batch, seq_len)
    for query_len in (1, 5):
        _run_shape(query_len, batch, seq_len, cache, replays, windows)


if __name__ == "__main__":
    main()
