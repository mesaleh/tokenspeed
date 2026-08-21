"""Small exact-specialization fixture for Compute Sanitizer."""

from __future__ import annotations

import json
import math

import torch

from probe_r31_physical_split_score import _launch


def run() -> dict[str, object]:
    torch.manual_seed(20260827)
    device = torch.device("cuda:0")
    batch, query_len, sequence_length = 1, 5, 384
    pages = math.ceil(sequence_length / 32)
    query_latent = (torch.randn((batch, query_len, 8, 512), device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    query_rope = (torch.randn((batch, query_len, 8, 256), device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    packed = torch.randint(
        0, 256, (pages, 32, 256), device=device, dtype=torch.uint8
    )
    scale = torch.ones((pages, 32), device=device, dtype=torch.bfloat16)
    high = (torch.randn((pages, 32, 64), device=device) * 0.125).to(
        torch.float8_e4m3fn
    )
    residual = torch.randint(
        0, 256, (pages, 32, 32), device=device, dtype=torch.uint8
    )
    table = torch.arange(pages, device=device, dtype=torch.int32).view(1, pages)
    lengths = torch.tensor([sequence_length], device=device, dtype=torch.int32)
    output, fault_status = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed,
        scale=scale,
        high_rope=high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=lengths,
        physical=True,
        lookahead=False,
        dual_tmem=True,
    )
    torch.cuda.synchronize()
    result = {
        "schema": "r31-r6-q3c-dual-tmem-sanitizer-v1",
        "batch": batch,
        "query_len": query_len,
        "sequence_length": sequence_length,
        "finite": bool(torch.isfinite(output).all()),
        "fault_status": int(fault_status.item()),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["finite"] or result["fault_status"] != 0:
        raise AssertionError(result)
    return result


if __name__ == "__main__":
    run()
