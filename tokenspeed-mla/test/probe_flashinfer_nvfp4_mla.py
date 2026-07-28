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

"""Probe FlashInfer's no-shadow SM100 NVFP4 path at Kimi MLA dimensions."""

import argparse
import json
import math

import torch
from flashinfer import mla
from flashinfer.fp4_quantization import nvfp4_quantize_paged_kv_cache


PAGE = 32
HEADS = 64
LATENT = 512
ROPE = 64
HEAD_DIM = LATENT + ROPE
WORKSPACE_BYTES = 256 << 20


def bench(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--skip-reference", action="store_true")
    args = parser.parse_args()

    if args.context < args.q_len:
        raise ValueError("--context must be at least --q-len")

    device = torch.device("cuda")
    fp8 = torch.float8_e4m3fn
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pages = math.ceil(args.context / PAGE)

    query_bf16 = (
        torch.randn(
            (1, args.q_len, HEADS, HEAD_DIM),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.bfloat16)
    query = query_bf16.to(fp8)
    cache = (
        torch.randn(
            (pages, 1, PAGE, HEAD_DIM),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.bfloat16)

    (packed_k, _), (block_scale_k, _), global_scale_k, _ = (
        nvfp4_quantize_paged_kv_cache(cache, cache, kv_layout="HND")
    )
    page_table = torch.arange(pages, device=device, dtype=torch.int32)[None]
    seq_lens = torch.tensor([args.context], device=device, dtype=torch.int32)
    workspace = torch.zeros(WORKSPACE_BYTES, device=device, dtype=torch.int8)
    out_dtype = torch.bfloat16 if args.output_dtype == "bf16" else fp8
    output = torch.empty(
        (1, args.q_len, HEADS, LATENT), device=device, dtype=out_dtype
    )
    raw_decode = mla.get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
    bmm1_scale = global_scale_k / math.sqrt(HEAD_DIM)
    bmm2_scale = global_scale_k

    def run() -> None:
        raw_decode(
            output,
            None,
            query.flatten(0, 1),
            packed_k,
            packed_k,
            workspace,
            page_table,
            seq_lens,
            args.q_len,
            args.context,
            bmm1_scale,
            bmm2_scale,
            -1,
            -1,
            0,
            1,
            -1,
            0,
            torch.cuda.get_device_properties(device).multi_processor_count,
            True,
            workspace.numel() * workspace.element_size(),
            None,
            None,
            block_scale_k,
            block_scale_k,
            None,
            True,
        )

    run()
    torch.cuda.synchronize()
    result = {
        "status": "PASS",
        "context": args.context,
        "q_len": args.q_len,
        "output_dtype": args.output_dtype,
        "packed_bytes": packed_k.numel() * packed_k.element_size(),
        "scale_bytes": block_scale_k.numel() * block_scale_k.element_size(),
        "bf16_cache_bytes": cache.numel() * cache.element_size(),
        "global_scale": global_scale_k,
    }

    if not args.skip_reference:
        logical_cache = cache.view(-1, HEAD_DIM)[: args.context].float()
        ref_query = query.float().view(args.q_len, HEADS, HEAD_DIM)
        scores = torch.einsum("qhd,td->qht", ref_query, logical_cache)
        probs = torch.softmax(scores / math.sqrt(HEAD_DIM), dim=-1)
        reference = torch.einsum(
            "qht,td->qhd", probs, logical_cache[:, :LATENT]
        )
        actual = output.float().view(args.q_len, HEADS, LATENT)
        error = actual - reference
        result.update(
            max_abs_diff=float(error.abs().max()),
            mean_abs_diff=float(error.abs().mean()),
            rms_diff=float(error.square().mean().sqrt()),
            reference_rms=float(reference.square().mean().sqrt()),
            cosine=float(
                torch.nn.functional.cosine_similarity(
                    actual.flatten(), reference.flatten(), dim=0
                )
            ),
        )

    result["microseconds"] = bench(run, args.warmup, args.iterations)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
