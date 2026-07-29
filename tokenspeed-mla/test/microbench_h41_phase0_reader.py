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

"""H41 W1 CUDA-graph timing for material TQ4 MLA reader rings.

This benchmark keeps independent control and candidate caches resident at the
same time.  Every dense control layer starts from the same finite, nonzero
values as its corresponding candidate layer.  Selected candidate layers use
packed TQ4; their latent control copies are reconstructed from the exact FP8
codebooks consumed by the reader.  N19 intentionally retains BF16 RoPE while
the normal dense control stores RoPE in FP8.  The decisive sequence is
control/candidate/control in one process.

The output is timing-only evidence.  It is not an end-to-end serving result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from typing import Any

import torch

from tokenspeed_mla import tokenspeed_mla_decode, tokenspeed_mla_decode_tq4

LATENT = 512
ROPE = 64
PAGE = 32
TOTAL_LAYERS = 61
E2M1_CENTROIDS = (
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
)


def percentile(values: list[float], q: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p20": percentile(values, 0.20),
        "median": percentile(values, 0.50),
        "p80": percentile(values, 0.80),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def gpu_covariates() -> dict[str, Any]:
    fields = (
        "timestamp,index,pstate,clocks.sm,clocks.mem,temperature.gpu,"
        "power.draw,utilization.gpu,memory.used"
    )
    command = [
        "nvidia-smi",
        "--id=0",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return {"error": repr(error), "command": command}
    names = fields.split(",")
    values = [part.strip() for part in completed.stdout.strip().split(",")]
    return dict(zip(names, values, strict=False))


def capture_graph(fn: Callable[[], torch.Tensor], warmups: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    samples: int,
    replays_per_sample: int,
    output: torch.Tensor,
) -> dict[str, Any]:
    before = gpu_covariates()
    sample_ring_us: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        for _ in range(replays_per_sample):
            graph.replay()
        end.record()
        end.synchronize()
        sample_ring_us.append(start.elapsed_time(end) * 1000.0 / replays_per_sample)
    after = gpu_covariates()
    return {
        "sample_ring_us": sample_ring_us,
        "ring_us": summarize(sample_ring_us),
        "per_layer_us": summarize(
            [value / TOTAL_LAYERS for value in sample_ring_us]
        ),
        "replays": samples * replays_per_sample,
        "gpu_before": before,
        "gpu_after": after,
        "output_checksum": float(output.float().sum()),
    }


def fill_finite_nonzero(
    destination: torch.Tensor, scratch: torch.Tensor, generator: torch.Generator
) -> None:
    scratch.uniform_(-0.125, 0.125, generator=generator)
    destination.copy_(scratch)


def reconstruct_codebook_layer(
    destination: torch.Tensor,
    packed: torch.Tensor,
    codebook_bytes: torch.Tensor,
    rope: torch.Tensor,
    *,
    chunk_pages: int = 256,
) -> None:
    """Reconstruct exactly the finite FP8 values loaded by the codebook reader."""

    codebook = codebook_bytes.view(torch.float8_e4m3fn)
    for first in range(0, packed.shape[0], chunk_pages):
        last = min(first + chunk_pages, packed.shape[0])
        packed_chunk = packed[first:last]
        indices = torch.empty(
            (*packed_chunk.shape[:-1], LATENT),
            device=packed.device,
            dtype=torch.uint8,
        )
        indices[..., 0::2] = packed_chunk & 0x0F
        indices[..., 1::2] = packed_chunk >> 4
        values = torch.gather(
            codebook[first:last].float(), -1, indices.to(torch.long)
        )
        destination[first:last, ..., :LATENT].copy_(values)
        destination[first:last, ..., LATENT:].copy_(rope[first:last])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(10219, 37932), required=True)
    parser.add_argument("--max-context", type=int, default=256000)
    parser.add_argument("--selected-layers", type=int, choices=(14, 19), required=True)
    parser.add_argument("--dense-before", type=int, default=None)
    parser.add_argument("--split-kv", type=int, choices=(32, 40, 64), required=True)
    parser.add_argument("--fp8-rope", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=5)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--replays-per-sample", type=int, default=100)
    parser.add_argument(
        "--allocation-order",
        choices=("candidate-first", "control-first"),
        required=True,
        help="Balance this across fresh processes to expose HBM-placement bias.",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--correctness-atol", type=float, default=0.002)
    args = parser.parse_args()

    if args.batch != 1 or args.q_len != 5 or args.heads != 8:
        raise ValueError("H41 W1 is pinned to batch=1, q_len=5, heads=8")
    if args.context > args.max_context:
        raise ValueError("context must not exceed max-context")
    if args.max_context != 256000:
        raise ValueError("H41 W1 is pinned to max-context=256000")
    if args.warmups < 100:
        raise ValueError("H41 W1 requires at least 100 graph warmups per arm")
    if args.samples * args.replays_per_sample < 2000:
        raise ValueError("H41 W1 requires at least 2000 timed replays per arm")
    if (args.selected_layers, args.fp8_rope) not in ((14, True), (19, False)):
        raise ValueError("H41 W1 permits only N14/FP8-RoPE or N19/BF16-RoPE")
    expected_before = {14: 24, 19: 21}[args.selected_layers]
    dense_before = expected_before if args.dense_before is None else args.dense_before
    if dense_before != expected_before:
        raise ValueError(
            f"H41 W1 N{args.selected_layers} requires dense-before={expected_before}"
        )
    dense_layers = TOTAL_LAYERS - args.selected_layers
    dense_after = dense_layers - dense_before
    if dense_before < 0 or dense_after < 0:
        raise ValueError("selected and boundary layers must total 61")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    fp8 = torch.float8_e4m3fn
    pages_per_request = math.ceil(args.max_context / 128) * (128 // PAGE)
    pages = args.batch * pages_per_request
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    if free_before < (30 << 30):
        raise RuntimeError(
            f"H41 W1 needs at least 30 GiB free; found {free_before / 2**30:.2f} GiB"
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    query = torch.empty(
        args.batch, args.q_len, args.heads, LATENT + ROPE,
        device=device, dtype=torch.bfloat16,
    )
    query.uniform_(-0.125, 0.125, generator=generator)
    query = query.to(fp8)
    centroids = torch.tensor(E2M1_CENTROIDS, device=device, dtype=torch.float32)

    dense_shapes = {
        "candidate": (dense_layers, pages, PAGE, LATENT + ROPE),
        "control": (TOTAL_LAYERS, pages, PAGE, LATENT + ROPE),
    }
    control_dense: torch.Tensor | None = None
    if args.allocation_order == "control-first":
        control_dense = torch.empty(
            dense_shapes["control"], device=device, dtype=fp8
        )

    packed = torch.empty(
        args.selected_layers, pages, PAGE, LATENT // 2,
        device=device, dtype=torch.uint8,
    )
    packed.random_(0, 256, generator=generator)
    scales = torch.empty(
        args.selected_layers, pages, PAGE, device=device, dtype=torch.bfloat16
    )
    scales.uniform_(0.05, 0.20, generator=generator)
    rope = torch.empty(
        args.selected_layers, pages, PAGE, ROPE, device=device, dtype=torch.bfloat16
    )
    rope.normal_(0.0, 0.1, generator=generator)
    if args.fp8_rope:
        rope = rope.to(fp8)
    codebooks = (
        scales.float()[..., None] * centroids[None, None, None, :]
    ).to(fp8).view(torch.uint8).contiguous()

    candidate_dense = torch.empty(
        dense_shapes["candidate"], device=device, dtype=fp8
    )
    if control_dense is None:
        control_dense = torch.empty(
            dense_shapes["control"], device=device, dtype=fp8
        )
    dense_scratch = torch.empty(
        pages, PAGE, LATENT + ROPE, device=device, dtype=torch.bfloat16
    )
    for candidate_index in range(dense_layers):
        fill_finite_nonzero(
            candidate_dense[candidate_index], dense_scratch, generator
        )
        control_index = (
            candidate_index
            if candidate_index < dense_before
            else candidate_index + args.selected_layers
        )
        control_dense[control_index].copy_(candidate_dense[candidate_index])
    del dense_scratch
    for selected_index in range(args.selected_layers):
        reconstruct_codebook_layer(
            control_dense[dense_before + selected_index],
            packed[selected_index],
            codebooks[selected_index],
            rope[selected_index],
        )
    torch.cuda.synchronize()

    page_generator = torch.Generator(device=device)
    page_generator.manual_seed(args.seed + 1)
    page_table = torch.stack(
        [
            torch.randperm(
                pages_per_request,
                device=device,
                dtype=torch.int32,
                generator=page_generator,
            )
            + batch_index * pages_per_request
            for batch_index in range(args.batch)
        ]
    )
    seq_lens = torch.full(
        (args.batch,), args.context, device=device, dtype=torch.int32
    )
    workspace = torch.empty(64 << 20, device=device, dtype=torch.int8)
    control_out = torch.empty(
        args.batch, args.q_len, args.heads, LATENT,
        device=device, dtype=torch.bfloat16,
    )
    candidate_out = torch.empty_like(control_out)
    check_dense_out = torch.empty_like(control_out)
    check_tq_out = torch.empty_like(control_out)
    softmax_scale = 1.0 / math.sqrt(LATENT + ROPE)

    def dense_call(cache: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return tokenspeed_mla_decode(
            query=query,
            kv_cache=cache,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=args.max_context,
            softmax_scale=softmax_scale,
            out=out,
            causal_mask=True,
            enable_pdl=True,
        )

    def tq_call(index: int, out: torch.Tensor) -> torch.Tensor:
        return tokenspeed_mla_decode_tq4(
            query=query,
            kv_nope_packed=packed[index],
            kv_nope_scale=scales[index],
            kv_rope=rope[index],
            centroids=centroids,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=args.max_context,
            softmax_scale=softmax_scale,
            out=out,
            causal_mask=True,
            enable_pdl=True,
            split_kv_override=args.split_kv,
            kv_nope_codebook=codebooks[index],
            fp8_rope=args.fp8_rope,
        )

    def run_control() -> torch.Tensor:
        result = control_out
        for layer in range(TOTAL_LAYERS):
            result = dense_call(control_dense[layer], control_out)
        return result

    def run_candidate() -> torch.Tensor:
        result = candidate_out
        for layer in range(dense_before):
            result = dense_call(candidate_dense[layer], candidate_out)
        for layer in range(args.selected_layers):
            result = tq_call(layer, candidate_out)
        for layer in range(dense_before, dense_layers):
            result = dense_call(candidate_dense[layer], candidate_out)
        return result

    dense_call(control_dense[dense_before], check_dense_out)
    tq_call(0, check_tq_out)
    torch.cuda.synchronize()
    max_abs_diff = float((check_dense_out.float() - check_tq_out.float()).abs().max())
    torch.testing.assert_close(
        check_tq_out,
        check_dense_out,
        rtol=0,
        atol=args.correctness_atol,
    )

    control_graph = capture_graph(run_control, args.warmups)
    candidate_graph = capture_graph(run_candidate, args.warmups)
    sequence: list[dict[str, Any]] = []
    for name, graph, output in (
        ("control_1", control_graph, control_out),
        ("candidate", candidate_graph, candidate_out),
        ("control_2", control_graph, control_out),
    ):
        sequence.append(
            {
                "arm": name,
                **time_graph(
                    graph,
                    samples=args.samples,
                    replays_per_sample=args.replays_per_sample,
                    output=output,
                ),
            }
        )

    control_1 = sequence[0]["ring_us"]["mean"]
    candidate = sequence[1]["ring_us"]["mean"]
    control_2 = sequence[2]["ring_us"]["mean"]
    control_mean = (control_1 + control_2) / 2.0
    result = {
        "status": "TIMING_ONLY",
        "experiment": "H41_W1_MATERIAL_READER",
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "context": args.context,
        "max_context": args.max_context,
        "batch": args.batch,
        "q_len": args.q_len,
        "heads": args.heads,
        "selected_layers": args.selected_layers,
        "dense_before": dense_before,
        "dense_after": dense_after,
        "fp8_rope": args.fp8_rope,
        "split_kv": args.split_kv,
        "allocation_order": args.allocation_order,
        "warmups_per_graph": args.warmups,
        "samples_per_arm": args.samples,
        "replays_per_sample": args.replays_per_sample,
        "sequence": sequence,
        "control_flank_drift_fraction": (control_2 - control_1) / control_mean,
        "candidate_minus_control_ring_us": candidate - control_mean,
        "candidate_minus_control_per_layer_us": (
            (candidate - control_mean) / args.selected_layers
        ),
        "max_abs_diff_one_selected_layer": max_abs_diff,
        "correctness_atol": args.correctness_atol,
        "free_bytes_before_allocations": free_before,
        "free_bytes_after_allocations": torch.cuda.mem_get_info(device)[0],
        "total_device_bytes": total_bytes,
        "calculated_control_dense_bytes": control_dense.numel(),
        "calculated_candidate_dense_bytes": candidate_dense.numel(),
        "calculated_tq_packed_bytes": packed.numel(),
        "calculated_tq_scale_bytes": scales.numel() * scales.element_size(),
        "calculated_tq_rope_bytes": rope.numel() * rope.element_size(),
        "calculated_tq_codebook_bytes": codebooks.numel(),
        "seed": args.seed,
        "interpretation": (
            "Timing-only reader evidence; not an end-to-end serving result or "
            "a quality evaluation."
        ),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
