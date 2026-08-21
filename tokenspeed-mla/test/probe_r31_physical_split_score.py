"""Initial algebra/compile probe for the physical R31 split-score mainloop."""

from __future__ import annotations

import json

import torch
from cutlass import Float32, Int32
from tokenspeed_mla.mla_decode_tq_r31 import _get_compiled_tq_r31_kernel


def _launch(
    *,
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    packed_latent: torch.Tensor,
    scale: torch.Tensor,
    high_rope: torch.Tensor,
    residual_rope: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    physical: bool,
    lookahead: bool = True,
) -> torch.Tensor:
    output = torch.empty_like(query_latent, dtype=torch.bfloat16)
    compiled = _get_compiled_tq_r31_kernel(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        reconstruction_scale=scale,
        high_rope=high_rope,
        residual_rope=residual_rope,
        block_tables=block_tables,
        output=output,
        lse=None,
        workspace=None,
        seq_lens=seq_lens,
        fold_sq_factor=1,
        causal_mask=False,
        enable_pdl=False,
        producer_only=False,
        physical_split_score=physical,
        physical_split_score_lookahead=lookahead,
    )
    import tvm_ffi

    with torch.cuda.device(query_latent.device), tvm_ffi.use_torch_stream():
        compiled(
            query_latent,
            query_rope,
            packed_latent,
            high_rope,
            block_tables,
            output,
            None,
            None,
            Int32(1),
            seq_lens,
            seq_lens,
            None,
            Float32(0.125),
            Float32(1.0),
            scale,
            None,
            residual_rope,
        )
    return output


def run() -> dict[str, object]:
    torch.manual_seed(20260824)
    device = torch.device("cuda:0")
    query_latent = (
        torch.randn((1, 1, 8, 512), device=device) * 0.25
    ).to(torch.float8_e4m3fn)
    query_rope = (torch.randn((1, 1, 8, 256), device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    packed_latent = torch.randint(
        0, 256, (12, 32, 256), device=device, dtype=torch.uint8
    )
    row_scales = torch.tensor(
        [0.5, 1.0, 2.0, 4.0], device=device, dtype=torch.bfloat16
    ).repeat(3).view(12, 1)
    scale = row_scales.expand(12, 32).contiguous()
    normalized_high = (torch.randn((12, 32, 64), device=device) * 0.125).to(
        torch.float8_e4m3fn
    )
    physical_high = (normalized_high.float() * scale.float().unsqueeze(-1)).to(
        torch.float8_e4m3fn
    )
    residual = torch.zeros((12, 32, 32), device=device, dtype=torch.uint8)
    table = torch.arange(12, device=device, dtype=torch.int32).view(1, 12)
    seq_lens = torch.tensor([384], device=device, dtype=torch.int32)
    normalized = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        scale=scale,
        high_rope=normalized_high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=seq_lens,
        physical=False,
    )
    serial = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        scale=scale,
        high_rope=physical_high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=seq_lens,
        physical=True,
        lookahead=False,
    )
    lookahead = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        scale=scale,
        high_rope=physical_high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=seq_lens,
        physical=True,
    )
    torch.cuda.synchronize()
    serial_delta = (serial.float() - normalized.float()).abs()
    lookahead_delta = (lookahead.float() - normalized.float()).abs()
    result = {
        "serial_finite": bool(torch.isfinite(serial).all()),
        "serial_max_abs": float(serial_delta.max()),
        "serial_mean_abs": float(serial_delta.mean()),
        "lookahead_finite": bool(torch.isfinite(lookahead).all()),
        "lookahead_max_abs": float(lookahead_delta.max()),
        "lookahead_mean_abs": float(lookahead_delta.mean()),
    }
    print(json.dumps(result, sort_keys=True))
    if (
        not result["serial_finite"]
        or not result["lookahead_finite"]
        or result["serial_max_abs"] > 0.02
        or result["lookahead_max_abs"] > 0.02
    ):
        raise AssertionError(result)
    return result


if __name__ == "__main__":
    run()
