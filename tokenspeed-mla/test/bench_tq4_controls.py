#!/usr/bin/env python3
"""Directional exact-10K latency comparison for private TQ4 controls."""

from __future__ import annotations

import json
from collections.abc import Callable

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


def _time_ms(fn: Callable[[], object], warmup: int = 10, repeat: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / repeat)


def _run(q_len: int) -> dict:
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

    dense_ms = _time_ms(dense)
    m128_ms = _time_ms(m128)
    m64_ms = _time_ms(m64)
    return {
        "q_len": q_len,
        "seq_len": case["max_seq_len"],
        "dense_m64_ms": dense_ms,
        "packed_m128_ms": m128_ms,
        "packed_m64_ms": m64_ms,
        "m64_vs_dense_pct": (m64_ms / dense_ms - 1.0) * 100.0,
        "m64_vs_m128_pct": (m64_ms / m128_ms - 1.0) * 100.0,
    }


def main() -> None:
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("SM100 required")
    print(json.dumps({"cases": [_run(1), _run(5)]}, sort_keys=True))


if __name__ == "__main__":
    main()
