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

from tokenspeed_mla import get_num_sm, tokenspeed_mla_decode, tokenspeed_mla_decode_tq4
from tokenspeed_mla.tq4_contract import dequantize_tq4_reference

LATENT = 512
ROPE = 64
PAGE = 32


def bench(fn) -> dict[str, float]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    median, p20, p80 = triton.testing.do_bench_cudagraph(fn, quantiles=(0.5, 0.2, 0.8))
    return {
        "median_us": median * 1000,
        "p20_us": p20 * 1000,
        "p80_us": p80 * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=10221)
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Configured kernel context; defaults to the active sequence length.",
    )
    parser.add_argument("--splits", default=None)
    parser.add_argument("--profile-split", type=int, default=None)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Decode batch size; use 2 to match the Kimi DFlash target graph.",
    )
    parser.add_argument(
        "--cache-layers",
        type=int,
        default=1,
        help=(
            "Rotate timing through this many independent compressed-layer KV "
            "caches; use 19 for the H36 material-value target or 61 for a "
            "fully compressed Kimi target."
        ),
    )
    parser.add_argument(
        "--dense-cache-layers",
        type=int,
        default=1,
        help="Rotate through this many independent dense FP8 caches.",
    )
    parser.add_argument("--mixed-dense-before", type=int, default=0)
    parser.add_argument("--mixed-dense-after", type=int, default=0)
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="Skip the dense reconstruction oracle for large independent-cache rings.",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Time only the independent dense ring; requires --timing-only.",
    )
    parser.add_argument("--heads", type=int, choices=(8, 16), default=16)
    parser.add_argument("--q-len", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--codebook", action="store_true")
    parser.add_argument("--no-codebook", action="store_true")
    parser.add_argument("--native-e2m1", action="store_true")
    parser.add_argument("--e2m1-data", action="store_true")
    parser.add_argument("--fp8-rope", action="store_true")
    parser.add_argument(
        "--scale-mode",
        choices=("legacy", "unit", "realistic"),
        default="legacy",
    )
    args = parser.parse_args()
    if args.codebook and args.no_codebook:
        raise ValueError("--codebook and --no-codebook are mutually exclusive")
    if args.context <= args.q_len:
        raise ValueError("--context must exceed --q-len")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.cache_layers <= 0:
        raise ValueError("--cache-layers must be positive")
    if args.dense_cache_layers <= 0:
        raise ValueError("--dense-cache-layers must be positive")
    if args.mixed_dense_before < 0 or args.mixed_dense_after < 0:
        raise ValueError("mixed dense layer counts must be non-negative")
    mixed_dense_layers = args.mixed_dense_before + args.mixed_dense_after
    mixed_mode = mixed_dense_layers > 0
    if mixed_mode:
        if not args.timing_only:
            raise ValueError("mixed rings require --timing-only")
        if args.dense_cache_layers != mixed_dense_layers:
            raise ValueError("--dense-cache-layers must equal mixed dense before+after")
    if args.dense_only and (not args.timing_only or mixed_mode):
        raise ValueError("--dense-only requires non-mixed --timing-only")
    max_context = args.context if args.max_context is None else args.max_context
    if max_context < args.context:
        raise ValueError("--max-context must be >= --context")

    torch.manual_seed(20260722)
    device = torch.device("cuda")
    fp8 = torch.float8_e4m3fn
    # The native kernel loads complete 128-token tiles, so keep the synthetic
    # block table padded exactly as a serving-time max-context table is.
    pages_per_request = math.ceil(max_context / 128) * (128 // PAGE)
    pages = args.batch * pages_per_request
    pre_free_bytes, total_device_bytes = torch.cuda.mem_get_info(device)
    large_ring = mixed_mode or args.dense_cache_layers >= 61
    required_free_bytes = (25 if large_ring else 12) << 30
    if pre_free_bytes < required_free_bytes:
        raise RuntimeError(
            f"insufficient free GPU memory: need {required_free_bytes} bytes, "
            f"found {pre_free_bytes}"
        )
    rope_element_bytes = 1 if args.fp8_rope else 2
    calculated_tq_cache_bytes = (
        args.cache_layers * pages * PAGE * (LATENT // 2 + 2 + ROPE * rope_element_bytes)
    )
    calculated_dense_cache_bytes = (
        args.dense_cache_layers * pages * PAGE * (LATENT + ROPE)
    )

    query = (
        torch.randn(
            args.batch,
            args.q_len,
            args.heads,
            LATENT + ROPE,
            device=device,
        )
        * 0.1
    ).to(fp8)
    packed = torch.randint(
        0, 256, (pages, PAGE, LATENT // 2), device=device, dtype=torch.uint8
    )

    # Unit scale catches operand-layout corruption exactly. Realistic native
    # E2M1 norm corrections are small; the legacy range remains useful for
    # exercising arbitrary Lloyd centroids.
    def make_scales(*shape: int) -> torch.Tensor:
        if args.scale_mode == "unit":
            return torch.ones(*shape, device=device, dtype=torch.bfloat16)
        scales = torch.rand(*shape, device=device, dtype=torch.bfloat16)
        if args.scale_mode == "realistic":
            return scales * 0.15 + 0.05
        return scales * 8.0 + 16.0

    scales = make_scales(pages, PAGE)
    rope_dtype = fp8 if args.fp8_rope else torch.bfloat16
    rope = (torch.randn(pages, PAGE, ROPE, device=device) * 0.1).to(rope_dtype)
    e2m1_data = args.native_e2m1 or args.e2m1_data
    if e2m1_data:
        centroids = torch.tensor(
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
            device=device,
            dtype=torch.float32,
        )
    else:
        centroids = torch.linspace(-0.12, 0.11, 16, device=device, dtype=torch.float32)
    codebook = (
        (scales.float()[..., None] * centroids[None, None, :])
        .to(fp8)
        .view(torch.uint8)
        .contiguous()
    )
    use_codebook = not args.no_codebook and (args.codebook or e2m1_data)

    packed_layers = [packed]
    scale_layers = [scales]
    rope_layers = [rope]
    codebook_layers = [codebook if use_codebook else None]
    if args.cache_layers > 1:
        extra_packed = torch.randint(
            0,
            256,
            (
                args.cache_layers - 1,
                pages,
                PAGE,
                LATENT // 2,
            ),
            device=device,
            dtype=torch.uint8,
        )
        extra_scales = make_scales(args.cache_layers - 1, pages, PAGE)
        extra_rope = (
            torch.randn(
                args.cache_layers - 1,
                pages,
                PAGE,
                ROPE,
                device=device,
            )
            * 0.1
        ).to(rope_dtype)
        packed_layers.extend(extra_packed.unbind())
        scale_layers.extend(extra_scales.unbind())
        rope_layers.extend(extra_rope.unbind())
        if use_codebook:
            extra_codebook = (
                (extra_scales.float()[..., None] * centroids[None, None, None, :])
                .to(fp8)
                .view(torch.uint8)
                .contiguous()
            )
            codebook_layers.extend(extra_codebook.unbind())
        else:
            codebook_layers.extend([None] * (args.cache_layers - 1))

    # Keep timing-only controls on the same nonzero distribution as the
    # correctness oracle. Zero-filled K/V makes the SM100 softmax path skip
    # accumulator corrections and produces an artificially fast control.
    dense_latent = dequantize_tq4_reference(
        packed, scales, centroids, dtype=torch.float32
    )
    dense_cache = torch.cat((dense_latent, rope.float()), dim=-1).to(fp8)
    del dense_latent
    dense_layers = [dense_cache]
    if args.dense_cache_layers > 1:
        extra_dense = torch.empty(
            args.dense_cache_layers - 1,
            pages,
            PAGE,
            LATENT + ROPE,
            device=device,
            dtype=fp8,
        )
        extra_dense.copy_(dense_cache)
        dense_layers.extend(extra_dense.unbind())
    page_table = torch.stack(
        [
            torch.randperm(pages_per_request, device=device, dtype=torch.int32)
            + batch_index * pages_per_request
            for batch_index in range(args.batch)
        ]
    )
    seq_lens = torch.full((args.batch,), args.context, device=device, dtype=torch.int32)
    workspace = torch.empty(64 << 20, device=device, dtype=torch.int8)
    out_dense = torch.empty(
        args.batch,
        args.q_len,
        args.heads,
        LATENT,
        device=device,
        dtype=torch.bfloat16,
    )
    out_tq4 = torch.empty_like(out_dense)
    out_tq4_reference = torch.empty_like(out_dense)
    scale = 1.0 / math.sqrt(LATENT + ROPE)

    def run_dense_layer(layer_index: int):
        return tokenspeed_mla_decode(
            query=query,
            kv_cache=dense_layers[layer_index],
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=max_context,
            softmax_scale=scale,
            out=out_dense,
            causal_mask=True,
            enable_pdl=True,
        )

    def run_dense():
        return run_dense_layer(0)

    def run_dense_cache_ring():
        result = None
        for layer_index in range(args.dense_cache_layers):
            result = run_dense_layer(layer_index)
        return result

    splits = (
        [32]
        if args.splits is None
        else [int(value) for value in args.splits.split(",") if value]
    )
    if not splits:
        raise ValueError("--splits must contain at least one value")

    def run_tq4_layer(layer_index: int, split_kv: int):
        return tokenspeed_mla_decode_tq4(
            query=query,
            kv_nope_packed=packed_layers[layer_index],
            kv_nope_scale=scale_layers[layer_index],
            kv_rope=rope_layers[layer_index],
            centroids=centroids,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=max_context,
            softmax_scale=scale,
            out=out_tq4,
            causal_mask=True,
            enable_pdl=True,
            split_kv_override=split_kv,
            kv_nope_codebook=codebook_layers[layer_index],
            native_e2m1=args.native_e2m1,
            fp8_rope=args.fp8_rope,
        )

    def run_tq4(split_kv: int):
        return run_tq4_layer(0, split_kv)

    def run_tq4_cache_ring(split_kv: int):
        result = None
        for layer_index in range(args.cache_layers):
            result = run_tq4_layer(layer_index, split_kv)
        return result

    def run_mixed_cache_ring(split_kv: int):
        result = None
        dense_index = 0
        for _ in range(args.mixed_dense_before):
            result = run_dense_layer(dense_index)
            dense_index += 1
        for layer_index in range(args.cache_layers):
            result = run_tq4_layer(layer_index, split_kv)
        for _ in range(args.mixed_dense_after):
            result = run_dense_layer(dense_index)
            dense_index += 1
        return result

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
            max_seq_len=max_context,
            softmax_scale=scale,
            out=out_tq4_reference,
            causal_mask=True,
            enable_pdl=True,
            split_kv_override=split_kv,
            fp8_rope=args.fp8_rope,
        )

    def check_codebook_identity(split_kv: int):
        if args.codebook and not args.native_e2m1:
            run_tq4_reference(split_kv)
            run_tq4(split_kv)
            torch.cuda.synchronize()
            torch.testing.assert_close(out_tq4, out_tq4_reference, rtol=0, atol=0)

    if args.profile_split is not None:
        check_codebook_identity(args.profile_split)
        run_dense()
        run_tq4(args.profile_split)
        torch.cuda.synchronize()
        difference = float((out_tq4.float() - out_dense.float()).abs().max())
        torch.testing.assert_close(out_tq4, out_dense, rtol=0, atol=args.atol)
        torch.cuda.nvtx.range_push("dense_profile")
        run_dense()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("tq4_profile")
        run_tq4_cache_ring(args.profile_split)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "context": args.context,
                    "max_context": max_context,
                    "batch": args.batch,
                    "cache_layers": args.cache_layers,
                    "codebook": use_codebook,
                    "split": args.profile_split,
                    "max_abs_diff": difference,
                },
                sort_keys=True,
            )
        )
        return

    if not args.timing_only:
        check_codebook_identity(splits[0])
        run_dense()
        run_tq4(splits[0])
        torch.cuda.synchronize()
    dense_timing = bench(
        run_dense_cache_ring if args.dense_cache_layers > 1 else run_dense
    )
    tq4_timings = {}
    mixed_timings = {}
    max_abs_diff = 0.0
    for split_kv in ([] if args.dense_only else splits):
        if not args.timing_only:
            run_tq4(split_kv)
            torch.cuda.synchronize()
            difference = float((out_tq4.float() - out_dense.float()).abs().max())
            torch.testing.assert_close(out_tq4, out_dense, rtol=0, atol=args.atol)
            max_abs_diff = max(max_abs_diff, difference)
        timed_fn = (
            (lambda split_kv=split_kv: run_mixed_cache_ring(split_kv))
            if mixed_mode
            else (lambda split_kv=split_kv: run_tq4_cache_ring(split_kv))
        )
        timing = bench(timed_fn)
        if mixed_mode:
            mixed_timings[str(split_kv)] = timing
            continue
        tq4_timings[str(split_kv)] = {
            name: value / args.cache_layers for name, value in timing.items()
        }
    result = {
        "status": "PASS",
        "context": args.context,
        "max_context": max_context,
        "batch": args.batch,
        "cache_layers": args.cache_layers,
        "codebook": use_codebook,
        "q_len": args.q_len,
        "heads": args.heads,
        "dense": dense_timing,
        "dense_per_layer": {
            name: value / args.dense_cache_layers
            for name, value in dense_timing.items()
        },
        "dense_cache_layers": args.dense_cache_layers,
        # All supported benchmark shapes fold q_len into heads, so the dense
        # kernel's simplified policy sees S=1. Record the resolved policy next
        # to the explicit TQ split keys so the comparison is auditable.
        "dense_split_kv": min(max(1, get_num_sm(device) // args.batch // 2), 32),
        "tq4": tq4_timings,
        "mixed": mixed_timings,
        "mixed_dense_before": args.mixed_dense_before,
        "mixed_dense_after": args.mixed_dense_after,
        "tq4_cache_layers": args.cache_layers,
        "correctness_cache_layers": 0 if args.timing_only else 1,
        "timing_only": args.timing_only,
        "dense_only": args.dense_only,
        "pre_free_bytes": pre_free_bytes,
        "total_device_bytes": total_device_bytes,
        "required_free_bytes": required_free_bytes,
        "calculated_tq_cache_bytes": calculated_tq_cache_bytes,
        "calculated_dense_cache_bytes": calculated_dense_cache_bytes,
        "max_abs_diff": max_abs_diff,
        "atol": args.atol,
        "native_e2m1": args.native_e2m1,
        "e2m1_data": e2m1_data,
        "scale_mode": args.scale_mode,
        "fp8_rope": args.fp8_rope,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
