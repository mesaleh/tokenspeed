# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Correctness and CUDA-graph timing gate for native packed TQ4 MLA decode."""

from __future__ import annotations

import argparse
import json
import math

import torch
import triton.testing

from tokenspeed_mla import tokenspeed_mla_decode, tokenspeed_mla_decode_tq4
from tokenspeed_mla.tq4_contract import dequantize_tq4_reference


LATENT = 512
ROPE = 64
PAGE = 32


def bench(fn) -> dict[str, float]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    median, p20, p80 = triton.testing.do_bench_cudagraph(
        fn, quantiles=(0.5, 0.2, 0.8)
    )
    return {
        "median_us": median * 1000,
        "p20_us": p20 * 1000,
        "p80_us": p80 * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=10221)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--profile-split", type=int, default=None)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--heads", type=int, choices=(8, 16), default=16)
    parser.add_argument("--q-len", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--codebook", action="store_true")
    args = parser.parse_args()
    if args.context <= args.q_len:
        raise ValueError("--context must exceed --q-len")

    torch.manual_seed(20260722)
    device = torch.device("cuda")
    fp8 = torch.float8_e4m3fn
    # The native kernel loads complete 128-token tiles, so keep the synthetic
    # block table padded exactly as a serving-time max-context table is.
    pages = math.ceil(args.context / 128) * (128 // PAGE)

    query = (
        torch.randn(1, args.q_len, args.heads, LATENT + ROPE, device=device) * 0.1
    ).to(fp8)
    packed = torch.randint(
        0, 256, (pages, PAGE, LATENT // 2), device=device, dtype=torch.uint8
    )
    # Production TQ norms put dequant scales around the low twenties.  Tiny
    # scales can hide operand-layout corruption behind an absolute tolerance.
    scales = (
        torch.rand(pages, PAGE, device=device, dtype=torch.bfloat16) * 8.0
        + 16.0
    )
    rope = (
        torch.randn(pages, PAGE, ROPE, device=device, dtype=torch.bfloat16) * 0.1
    )
    centroids = torch.linspace(
        -0.12, 0.11, 16, device=device, dtype=torch.float32
    )
    codebook = (
        scales.float()[..., None] * centroids[None, None, :]
    ).to(fp8).view(torch.uint8).contiguous()

    dense_latent = dequantize_tq4_reference(
        packed, scales, centroids, dtype=torch.float32
    )
    dense_cache = torch.cat((dense_latent, rope), dim=-1).to(fp8)
    page_table = torch.randperm(pages, device=device, dtype=torch.int32)[None]
    seq_lens = torch.tensor([args.context], device=device, dtype=torch.int32)
    workspace = torch.empty(64 << 20, device=device, dtype=torch.int8)
    out_dense = torch.empty(
        1, args.q_len, args.heads, LATENT, device=device, dtype=torch.bfloat16
    )
    out_tq4 = torch.empty_like(out_dense)
    out_tq4_reference = torch.empty_like(out_dense)
    scale = 1.0 / math.sqrt(LATENT + ROPE)

    def run_dense():
        return tokenspeed_mla_decode(
            query=query,
            kv_cache=dense_cache,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=args.context,
            softmax_scale=scale,
            out=out_dense,
            causal_mask=True,
            enable_pdl=True,
        )

    splits = (
        [32]
        if args.splits is None
        else [int(value) for value in args.splits.split(",") if value]
    )
    if not splits:
        raise ValueError("--splits must contain at least one value")

    def run_tq4(split_kv: int):
        return tokenspeed_mla_decode_tq4(
            query=query,
            kv_nope_packed=packed,
            kv_nope_scale=scales,
            kv_rope=rope,
            centroids=centroids,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=args.context,
            softmax_scale=scale,
            out=out_tq4,
            causal_mask=True,
            enable_pdl=True,
            split_kv_override=split_kv,
            kv_nope_codebook=codebook if args.codebook else None,
        )

    def run_tq4_reference(split_kv: int):
        return tokenspeed_mla_decode_tq4(
            query=query,
            kv_nope_packed=packed,
            kv_nope_scale=scales,
            kv_rope=rope,
            centroids=centroids,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=args.context,
            softmax_scale=scale,
            out=out_tq4_reference,
            causal_mask=True,
            enable_pdl=True,
            split_kv_override=split_kv,
        )

    def check_codebook_identity(split_kv: int):
        if args.codebook:
            run_tq4_reference(split_kv)
            run_tq4(split_kv)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                out_tq4, out_tq4_reference, rtol=0, atol=0
            )

    if args.profile_split is not None:
        check_codebook_identity(args.profile_split)
        run_dense()
        run_tq4(args.profile_split)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("dense_profile")
        run_dense()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("tq4_profile")
        run_tq4(args.profile_split)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        difference = float((out_tq4.float() - out_dense.float()).abs().max())
        torch.testing.assert_close(out_tq4, out_dense, rtol=0, atol=args.atol)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "context": args.context,
                    "split": args.profile_split,
                    "max_abs_diff": difference,
                },
                sort_keys=True,
            )
        )
        return

    check_codebook_identity(splits[0])
    run_dense()
    run_tq4(splits[0])
    torch.cuda.synchronize()
    dense_timing = bench(run_dense)
    tq4_timings = {}
    max_abs_diff = 0.0
    for split_kv in splits:
        run_tq4(split_kv)
        torch.cuda.synchronize()
        difference = float((out_tq4.float() - out_dense.float()).abs().max())
        torch.testing.assert_close(out_tq4, out_dense, rtol=0, atol=args.atol)
        max_abs_diff = max(max_abs_diff, difference)
        tq4_timings[str(split_kv)] = bench(
            lambda split_kv=split_kv: run_tq4(split_kv)
        )
    result = {
        "status": "PASS",
        "context": args.context,
        "q_len": args.q_len,
        "heads": args.heads,
        "dense": dense_timing,
        "tq4": tq4_timings,
        "max_abs_diff": max_abs_diff,
        "atol": args.atol,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
