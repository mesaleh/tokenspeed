#!/usr/bin/env python3
"""Compile and compare the private TQ4 M=128 control on one SM100 GPU."""

from __future__ import annotations

import json

import torch

from tokenspeed_mla.mla_decode import tokenspeed_mla_decode
from tokenspeed_mla.mla_decode_tq4 import _tokenspeed_mla_decode_tq4_m128_control
from tokenspeed_mla.tq4_contract import dequantize_tq4_reference


def _make_case(
    q_len: int,
    tree_mask: bool,
    *,
    heads: int,
    use_codebook: bool,
    fp8_rope: bool,
) -> dict[str, torch.Tensor | int | bool | None]:
    torch.manual_seed(20260802 + q_len)
    device = torch.device("cuda")
    batch, page_size = 1, 32
    # Nine K tiles with split_kv=1 cycle every multi-stage raw/K/V pipeline,
    # including empty-barrier phase reuse and raw/K shared-memory aliasing.
    seq_len = 1024 + q_len
    tile_count = (seq_len + 127) // 128
    pages = tile_count * (128 // page_size)

    low = torch.randint(0, 16, (pages, page_size, 256), device=device, dtype=torch.uint8)
    high = torch.randint(0, 16, low.shape, device=device, dtype=torch.uint8)
    packed = (low | (high << 4)).contiguous()
    scales = (
        0.5
        + torch.rand((pages, page_size), device=device, dtype=torch.float32)
    ).to(torch.bfloat16)
    centroids = torch.linspace(-1.5, 1.5, 16, device=device, dtype=torch.float32)
    rope_bf16 = torch.randn(
        (pages, page_size, 64), device=device, dtype=torch.bfloat16
    )
    latent_fp8 = dequantize_tq4_reference(
        packed, scales, centroids, dtype=torch.float8_e4m3fn
    )
    codebook = None
    if use_codebook:
        codebook = (
            scales.float().unsqueeze(-1) * centroids.view(1, 1, -1)
        ).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    rope_fp8 = rope_bf16.to(torch.float8_e4m3fn)
    rope_storage = rope_fp8 if fp8_rope else rope_bf16
    dense_cache = torch.cat((latent_fp8, rope_fp8), dim=-1).contiguous()
    query = torch.randn(
        (batch, q_len, heads, 576), device=device, dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    block_tables = torch.arange(pages, device=device, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    workspace = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.int8)

    custom_mask = None
    if tree_mask:
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
        "rope_bf16": rope_bf16,
        "rope_storage": rope_storage,
        "codebook": codebook,
        "dense_cache": dense_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "workspace": workspace,
        "custom_mask": custom_mask,
        "max_seq_len": seq_len,
        "fp8_rope": fp8_rope,
    }


def _run_case(
    q_len: int,
    tree_mask: bool,
    *,
    heads: int,
    use_codebook: bool,
    fp8_rope: bool,
    enable_pdl: bool,
) -> dict[str, float | int | bool]:
    case = _make_case(
        q_len,
        tree_mask,
        heads=heads,
        use_codebook=use_codebook,
        fp8_rope=fp8_rope,
    )
    query = case["query"]
    common = dict(
        query=query,
        workspace_buffer=case["workspace"],
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        max_seq_len=case["max_seq_len"],
        softmax_scale=576**-0.5,
        custom_mask=case["custom_mask"],
        enable_pdl=enable_pdl,
        return_lse=True,
    )
    dense_output, dense_lse = tokenspeed_mla_decode(
        kv_cache=case["dense_cache"],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        **common,
    )
    packed_output, packed_lse = _tokenspeed_mla_decode_tq4_m128_control(
        kv_nope_packed=case["packed"],
        kv_nope_scale=case["scales"],
        kv_rope=case["rope_storage"],
        centroids=case["centroids"],
        kv_nope_codebook=case["codebook"],
        fp8_rope=case["fp8_rope"],
        split_kv_override=1,
        **common,
    )
    torch.cuda.synchronize()
    output_error = (dense_output.float() - packed_output.float()).abs()
    lse_error = (dense_lse - packed_lse).abs()
    result = {
        "q_len": q_len,
        "tree_mask": tree_mask,
        "heads": heads,
        "use_codebook": use_codebook,
        "fp8_rope": fp8_rope,
        "enable_pdl": enable_pdl,
        "output_max_abs": float(output_error.max()),
        "output_mean_abs": float(output_error.mean()),
        "lse_max_abs": float(lse_error.max()),
        "finite": bool(
            torch.isfinite(packed_output.float()).all()
            and torch.isfinite(packed_lse).all()
        ),
    }
    if not result["finite"]:
        raise AssertionError(f"non-finite packed control output: {result}")
    if result["output_max_abs"] > 0.125 or result["lse_max_abs"] > 0.125:
        raise AssertionError(f"packed control drift exceeds component gate: {result}")
    return result


def main() -> None:
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError(
            f"probe requires SM100, got {torch.cuda.get_device_capability()}"
        )
    results = [
        _run_case(
            1,
            False,
            heads=8,
            use_codebook=False,
            fp8_rope=False,
            enable_pdl=True,
        ),
        _run_case(
            5,
            True,
            heads=16,
            use_codebook=True,
            fp8_rope=True,
            enable_pdl=True,
        ),
    ]
    print(json.dumps({"status": "pass", "cases": results}, sort_keys=True))


if __name__ == "__main__":
    main()
