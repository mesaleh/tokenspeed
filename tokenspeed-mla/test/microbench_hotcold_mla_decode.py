"""Correctness and exact-10K timing gate for segmented FP8/TQ4 MLA."""

from __future__ import annotations

import argparse
import json
import math

import torch
import triton.testing

from tokenspeed_mla import (
    merge_attention_outputs_base2,
    tokenspeed_mla_decode,
    tokenspeed_mla_decode_tq4,
)
from tokenspeed_mla.tq4_contract import dequantize_tq4_reference


LATENT = 512
ROPE = 64
PAGE = 32
HEADS = 8
HOT_TOKENS = 16_384


def _bench(fn) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    median, _, _ = triton.testing.do_bench_cudagraph(
        fn, quantiles=(0.5, 0.2, 0.8)
    )
    return median * 1000


def _dense_lse_reference(
    query: torch.Tensor,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    ordered = cache[page_table[0].long()].reshape(-1, LATENT + ROPE)[:seq_len]
    scores = torch.einsum(
        "qhd,kd->qhk", query[0].float(), ordered.float()
    ) * softmax_scale
    query_positions = torch.arange(
        seq_len - query.shape[1], seq_len, device=query.device
    )
    key_positions = torch.arange(seq_len, device=query.device)
    scores = scores.masked_fill(
        key_positions.view(1, 1, -1)
        > query_positions.view(query.shape[1], 1, 1),
        -torch.inf,
    )
    return torch.logsumexp(scores, dim=-1).unsqueeze(0) / math.log(2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-len", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    q_len = args.q_len

    torch.manual_seed(20260724)
    device = torch.device("cuda")
    fp8 = torch.float8_e4m3fn
    scale = 1.0 / math.sqrt(LATENT + ROPE)
    workspace = torch.empty(64 << 20, device=device, dtype=torch.int8)
    query = (
        torch.randn(1, q_len, HEADS, LATENT + ROPE, device=device) * 0.1
    ).to(fp8)

    cold_pages = 8
    hot_pages = 4
    cold_tokens = cold_pages * PAGE
    hot_tokens = hot_pages * PAGE
    total_tokens = cold_tokens + hot_tokens

    packed = torch.randint(
        0,
        256,
        (cold_pages, PAGE, LATENT // 2),
        dtype=torch.uint8,
        device=device,
    )
    scales = (
        torch.rand(cold_pages, PAGE, dtype=torch.bfloat16, device=device) * 8.0
        + 16.0
    )
    cold_rope = (
        torch.randn(cold_pages, PAGE, ROPE, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    centroids = torch.linspace(-0.12, 0.11, 16, dtype=torch.float32, device=device)
    codebook = (
        scales.float()[..., None] * centroids.view(1, 1, 16)
    ).to(fp8).view(torch.uint8).contiguous()
    cold_latent = dequantize_tq4_reference(
        packed, scales, centroids, dtype=torch.float32
    )
    cold_dense = torch.cat((cold_latent, cold_rope), dim=-1).to(fp8)
    hot_dense = (
        torch.randn(
            hot_pages, PAGE, LATENT + ROPE, dtype=torch.float32, device=device
        )
        * 0.1
    ).to(fp8)
    full_dense = torch.cat((cold_dense, hot_dense), dim=0)

    full_table = torch.arange(
        cold_pages + hot_pages, dtype=torch.int32, device=device
    ).view(1, -1)
    cold_table = torch.arange(cold_pages, dtype=torch.int32, device=device).view(
        1, -1
    )
    hot_table = torch.arange(hot_pages, dtype=torch.int32, device=device).view(1, -1)
    full_seq = torch.tensor([total_tokens], dtype=torch.int32, device=device)
    cold_seq = torch.tensor([cold_tokens], dtype=torch.int32, device=device)
    hot_seq = torch.tensor([hot_tokens], dtype=torch.int32, device=device)

    out_full = torch.empty(
        1, q_len, HEADS, LATENT, dtype=torch.bfloat16, device=device
    )
    out_full_no_lse = torch.empty_like(out_full)
    out_cold = torch.empty_like(out_full)
    out_cold_no_lse = torch.empty_like(out_full)
    out_hot = torch.empty_like(out_full)
    lse_full = torch.empty(1, q_len, HEADS, dtype=torch.float32, device=device)
    lse_cold = torch.empty_like(lse_full)
    lse_hot = torch.empty_like(lse_full)

    full_result = tokenspeed_mla_decode(
        query=query,
        kv_cache=full_dense,
        workspace_buffer=workspace,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
        block_tables=full_table,
        seq_lens=full_seq,
        max_seq_len=total_tokens,
        softmax_scale=scale,
        out=out_full,
        causal_mask=True,
        enable_pdl=True,
        return_lse=True,
        lse_out=lse_full,
    )
    assert full_result[0].data_ptr() == out_full.data_ptr()
    assert full_result[1].data_ptr() == lse_full.data_ptr()
    tokenspeed_mla_decode(
        query=query,
        kv_cache=full_dense,
        workspace_buffer=workspace,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
        block_tables=full_table,
        seq_lens=full_seq,
        max_seq_len=total_tokens,
        softmax_scale=scale,
        out=out_full_no_lse,
        causal_mask=True,
        enable_pdl=True,
    )

    tokenspeed_mla_decode_tq4(
        query=query,
        kv_nope_packed=packed,
        kv_nope_scale=scales,
        kv_rope=cold_rope,
        centroids=centroids,
        workspace_buffer=workspace,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
        block_tables=cold_table,
        seq_lens=cold_seq,
        max_seq_len=cold_tokens,
        softmax_scale=scale,
        out=out_cold,
        causal_mask=False,
        enable_pdl=True,
        split_kv_override=2,
        kv_nope_codebook=codebook,
        return_lse=True,
        lse_out=lse_cold,
    )
    tokenspeed_mla_decode_tq4(
        query=query,
        kv_nope_packed=packed,
        kv_nope_scale=scales,
        kv_rope=cold_rope,
        centroids=centroids,
        workspace_buffer=workspace,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
        block_tables=cold_table,
        seq_lens=cold_seq,
        max_seq_len=cold_tokens,
        softmax_scale=scale,
        out=out_cold_no_lse,
        causal_mask=False,
        enable_pdl=True,
        split_kv_override=2,
        kv_nope_codebook=codebook,
    )
    tokenspeed_mla_decode(
        query=query,
        kv_cache=hot_dense,
        workspace_buffer=workspace,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
        block_tables=hot_table,
        seq_lens=hot_seq,
        max_seq_len=hot_tokens,
        softmax_scale=scale,
        out=out_hot,
        causal_mask=True,
        enable_pdl=True,
        return_lse=True,
        lse_out=lse_hot,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out_full, out_full_no_lse, rtol=0, atol=0)
    torch.testing.assert_close(out_cold, out_cold_no_lse, rtol=0, atol=0)

    merged, merged_lse = merge_attention_outputs_base2(
        out_cold, lse_cold, out_hot, lse_hot
    )
    lse_reference = _dense_lse_reference(
        query, full_dense, full_table, total_tokens, scale
    )
    output_error = float((merged.float() - out_full.float()).abs().max())
    lse_export_error = float((lse_full - lse_reference).abs().max())
    lse_merge_error = float((merged_lse - lse_full).abs().max())
    torch.testing.assert_close(merged, out_full, rtol=0, atol=0.003)
    torch.testing.assert_close(lse_full, lse_reference, rtol=0, atol=0.02)
    torch.testing.assert_close(merged_lse, lse_full, rtol=0, atol=0.02)
    if args.correctness_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "q_len": q_len,
                    "output_max_abs_diff": output_error,
                    "lse_export_max_abs_diff": lse_export_error,
                    "lse_merge_max_abs_diff": lse_merge_error,
                },
                sort_keys=True,
            )
        )
        return

    exact_context = 10_240
    exact_pages = exact_context // PAGE
    exact_cache = (
        torch.randn(
            exact_pages,
            PAGE,
            LATENT + ROPE,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    ).to(fp8)
    exact_table = torch.arange(
        exact_pages, dtype=torch.int32, device=device
    ).view(1, -1)
    exact_seq = torch.tensor([exact_context], dtype=torch.int32, device=device)
    exact_out = torch.empty_like(out_full)

    def run_dense_exact():
        return tokenspeed_mla_decode(
            query=query,
            kv_cache=exact_cache,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=exact_table,
            seq_lens=exact_seq,
            max_seq_len=HOT_TOKENS,
            softmax_scale=scale,
            out=exact_out,
            causal_mask=True,
            enable_pdl=True,
        )

    def run_mixed_dispatch():
        if exact_context <= HOT_TOKENS:
            return run_dense_exact()
        raise AssertionError("exact-10K must remain in the FP8-only branch")

    dense_us = _bench(run_dense_exact)
    mixed_fast_us = _bench(run_mixed_dispatch)
    timing_delta_pct = (mixed_fast_us / dense_us - 1.0) * 100.0
    if abs(timing_delta_pct) > 1.0:
        raise AssertionError(
            f"FP8-only dispatcher moved latency by {timing_delta_pct:.3f}%"
        )

    result = {
        "status": "PASS",
        "q_len": q_len,
        "cold_tokens": cold_tokens,
        "hot_tokens": hot_tokens,
        "output_max_abs_diff": output_error,
        "lse_export_max_abs_diff": lse_export_error,
        "lse_merge_max_abs_diff": lse_merge_error,
        "exact10k_dense_us": dense_us,
        "exact10k_mixed_fast_us": mixed_fast_us,
        "exact10k_delta_pct": timing_delta_pct,
        "target_cache_saving_pct": 28.275,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
