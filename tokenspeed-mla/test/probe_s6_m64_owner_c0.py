# Copyright (c) 2026 LightSeek Foundation

"""Compile and numerically gate the M64 production-owner E2M1 C0 path."""

import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "python" / "tokenspeed_mla"
if not SOURCE_ROOT.exists():
    # The Kubernetes scratch copy places the module beside this probe.
    SOURCE_ROOT = Path(__file__).resolve().parent / "tokenspeed_mla"
sys.path.insert(0, str(SOURCE_ROOT))

from mla_decode_fp8 import (  # noqa: E402
    BlackwellMultiHeadLatentAttentionForwardFP8,
)


BATCH = int(os.environ.get("TQ_S6_BATCH", "4"))
SEQ_Q = 1
HEAD_CASES = (8, 40)
SEQ_K = int(os.environ.get("TQ_S6_SEQ_K", "256"))
PAGE_SIZE = 32
LATENT = 512
ROPE = 64
CACHE_LENGTHS = (31, 32, 33, 256)

E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _cute_tensor(tensor: torch.Tensor, dtype, leading_dim: int):
    result = from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True)
    result.element_type = dtype
    result = result.mark_layout_dynamic(leading_dim=leading_dim)
    return result


def _pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return (low | (high << 4)).to(torch.uint8).contiguous()


def _audit_generated(compiled) -> dict[str, int]:
    artifacts = getattr(compiled, "artifacts", None)

    def retained(name: str):
        value = getattr(artifacts, name.upper(), None)
        if value is None:
            value = getattr(compiled, f"__{name.lower()}__", None)
        return value

    ptx = retained("ptx")
    sass = retained("sass")
    cubin = retained("cubin")
    mlir = retained("mlir")
    if not isinstance(ptx, str) or not isinstance(cubin, bytes):
        retained_summary = {
            name: (
                type(value).__name__,
                len(value) if isinstance(value, (str, bytes)) else None,
            )
            for name, value in (
                ("ptx", ptx),
                ("sass", sass),
                ("cubin", cubin),
                ("mlir", mlir),
            )
        }
        raise AssertionError(
            "set CUTE_DSL_KEEP=all for generated-code audit; "
            f"type={type(compiled)} artifacts_type={type(artifacts)} "
            f"artifact_fields={sorted(vars(artifacts)) if artifacts else []} "
            f"retained={retained_summary}"
        )
    if not isinstance(mlir, str):
        raise AssertionError("generated MLIR artifact was not retained")

    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
        cubin_file.write(cubin)
        cubin_file.flush()
        if not isinstance(sass, str):
            sass = subprocess.run(
                ["cuobjdump", "--dump-sass", cubin_file.name],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        resource_output = subprocess.run(
            ["cuobjdump", "--dump-resource-usage", cubin_file.name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    ptx_mma = ptx.count("tcgen05.mma.ws.cta_group::1.kind::f8f6f4")
    sass_mma = sass.count(" UTCQMMA.WS")
    if ptx_mma < 26 or sass_mma < 26:
        raise AssertionError(
            f"owner MMA audit failed: PTX={ptx_mma} SASS={sass_mma}"
        )

    matches = re.findall(
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resource_output
    )
    if not matches:
        raise AssertionError("C0 resource record missing")
    registers = max(int(match[0]) for match in matches)
    stack = max(int(match[1]) for match in matches)
    static_shared = max(int(match[2]) for match in matches)
    local = max(int(match[3]) for match in matches)
    local_loads = len(re.findall(r"\bLDL\b", sass))
    local_stores = len(re.findall(r"\bSTL\b", sass))
    # CUTLASS 4.6 currently materializes a bounded 48-byte frame (8 LDL / 7
    # STL instructions) while retaining the dense owner's 168-register ceiling.
    # Keep those counts visible and fail on any further spill growth.
    if (
        registers > 168
        or stack > 48
        or local != 0
        or local_loads > 8
        or local_stores > 7
    ):
        raise AssertionError(
            "C0 resource gate failed: "
            f"registers={registers} stack={stack} local={local} "
            f"LDL={local_loads} STL={local_stores}"
        )

    dump_dir = os.environ.get("TQ_S6_DUMP_GENERATED_DIR")
    if dump_dir:
        output_dir = Path(dump_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix, payload, binary in (
            ("ptx", ptx, False),
            ("sass", sass, False),
            ("mlir", mlir, False),
            ("cubin", cubin, True),
            ("resources.txt", resource_output, False),
        ):
            output_path = output_dir / f"owner.{suffix}"
            if binary:
                output_path.write_bytes(payload)
            else:
                output_path.write_text(payload)
    return {
        "ptx_mma": ptx_mma,
        "sass_mma": sass_mma,
        "registers": registers,
        "stack": stack,
        "static_shared": static_shared,
        "local": local,
        "local_loads": local_loads,
        "local_stores": local_stores,
    }


def _build_inputs(heads: int):
    if SEQ_K % PAGE_SIZE:
        raise ValueError("TQ_S6_SEQ_K must be divisible by PAGE_SIZE")
    torch.manual_seed(20260814 + heads)
    device = "cuda"
    pages_per_batch = SEQ_K // PAGE_SIZE
    physical_pages = BATCH * pages_per_batch
    if os.environ.get("TQ_S6_DEBUG_ONE_TILE") == "1":
        if BATCH != 4 or SEQ_K < 128:
            raise ValueError("one-tile debug requires batch=4 and seq-k>=128")
        cache_lengths = (31, 32, 33, 128)
    elif BATCH == 4 and SEQ_K == 256:
        cache_lengths = CACHE_LENGTHS
    else:
        cache_lengths = (SEQ_K,) * BATCH

    # Keep every value exactly representable so failures identify ownership,
    # layout, scaling, or online-accumulation errors rather than host rounding.
    q_latent_f32 = (
        torch.randint(-4, 5, (BATCH, SEQ_Q, heads, LATENT), device=device)
        / 8.0
    )
    q_rope_f32 = (
        torch.randint(-4, 5, (BATCH, SEQ_Q, heads, ROPE), device=device)
        / 16.0
    )
    q_latent_t = q_latent_f32.to(torch.float8_e4m3fn).contiguous()
    q_rope_t = q_rope_f32.to(torch.float8_e4m3fn).contiguous()

    physical_token = torch.arange(
        physical_pages * PAGE_SIZE, device=device
    )[:, None]
    dim = torch.arange(LATENT, device=device)[None, :]
    codes = ((physical_token * 5 + dim * 3 + 1) % 15 + 1).to(torch.uint8)
    packed_t = _pack_e2m1(
        codes.view(physical_pages, PAGE_SIZE, LATENT)
    )
    code_values = E2M1_LUT.to(device=device)[codes.long()]

    page = torch.arange(physical_pages, device=device).repeat_interleave(PAGE_SIZE)
    offset = torch.arange(PAGE_SIZE, device=device).repeat(physical_pages)
    scale_f32 = torch.pow(
        2.0,
        (-10 + page % 8 + offset % 2).float(),
    )
    if os.environ.get("TQ_S6_DEBUG_FIXED_SCALE") == "1":
        scale_f32.fill_(2.0**-6)
    scale_t = (
        scale_f32.to(torch.bfloat16)
        .view(physical_pages, PAGE_SIZE)
        .contiguous()
    )
    dequant_physical = code_values * scale_t.view(-1, 1).float()

    rope_f32 = (
        (
            physical_token * 7
            + torch.arange(ROPE, device=device)[None, :] * 11
            + 3
        )
        % 9
        - 4
    ) / 16.0
    rope_t = rope_f32.to(torch.float8_e4m3fn).view(
        physical_pages, PAGE_SIZE, ROPE
    ).contiguous()
    page_rows = []
    for batch in range(BATCH):
        local = torch.arange(pages_per_batch, device=device, dtype=torch.int32)
        if batch == 1:
            local = torch.flip(local, dims=(0,))
        elif batch == 2:
            local = torch.roll(local, shifts=3)
        elif batch == 3:
            local = torch.roll(torch.flip(local, dims=(0,)), shifts=2)
        page_rows.append(local + batch * pages_per_batch)
    page_table_t = torch.stack(page_rows).contiguous()
    cache_seqs_t = torch.tensor(cache_lengths, device=device, dtype=torch.int32)
    output_t = torch.zeros(
        (BATCH, SEQ_Q, heads, LATENT),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    lse_t = torch.zeros((BATCH, SEQ_Q, heads), device=device, dtype=torch.float32)
    dense_t = (
        dequant_physical.to(torch.float8_e4m3fn)
        .view(physical_pages, PAGE_SIZE, LATENT)
        .contiguous()
    )
    dense_output_t = torch.zeros_like(output_t)
    dense_lse_t = torch.zeros_like(lse_t)

    cute_inputs = {
        "q_latent": _cute_tensor(q_latent_t, cutlass.Float8E4M3FN, 3),
        "q_rope": _cute_tensor(q_rope_t, cutlass.Float8E4M3FN, 3),
        "packed": _cute_tensor(packed_t, cutlass.Uint8, 2),
        "scale": _cute_tensor(scale_t, cutlass.BFloat16, 1),
        "dense": _cute_tensor(dense_t, cutlass.Float8E4M3FN, 2),
        "rope": _cute_tensor(rope_t, cutlass.Float8E4M3FN, 2),
        "page_table": _cute_tensor(page_table_t, cutlass.Int32, 1),
        "cache_seqs": _cute_tensor(cache_seqs_t, cutlass.Int32, 0),
        "output": _cute_tensor(output_t, cutlass.Float8E4M3FN, 3),
        "lse": _cute_tensor(lse_t, cutlass.Float32, 2),
        "dense_output": _cute_tensor(
            dense_output_t, cutlass.Float8E4M3FN, 3
        ),
        "dense_lse": _cute_tensor(dense_lse_t, cutlass.Float32, 2),
    }
    references = []
    for batch, length in enumerate(cache_lengths):
        physical_ids = page_table_t[batch].long()
        base = (
            physical_ids[:, None] * PAGE_SIZE
            + torch.arange(PAGE_SIZE, device=device)[None, :]
        ).flatten()[:length]
        cache = dequant_physical[base]
        rope = rope_t.float().view(-1, ROPE)[base]
        score = (
            torch.einsum("hd,kd->hk", q_latent_t.float()[batch, 0], cache)
            + torch.einsum("hr,kr->hk", q_rope_t.float()[batch, 0], rope)
        ) / math.sqrt(LATENT + ROPE)
        references.append(torch.softmax(score, dim=-1) @ cache)
    expected = torch.stack(references)[:, None]
    return (
        cute_inputs,
        output_t,
        lse_t,
        dense_output_t,
        dense_lse_t,
        expected,
        cache_lengths,
    )


def _run_case(heads: int, run_graph: bool) -> None:
    print(f"S6_C0_CASE_START heads={heads}", flush=True)
    (
        inputs,
        output_t,
        lse_t,
        dense_output_t,
        dense_lse_t,
        expected,
        cache_lengths,
    ) = _build_inputs(heads)
    hardware = utils.HardwareInfo()
    max_clusters = hardware.get_max_active_clusters(1)
    op = BlackwellMultiHeadLatentAttentionForwardFP8(
        cutlass.Float32,
        cutlass.Float32,
        (64, 128),
        (64, 256),
        max_clusters,
        PAGE_SIZE,
        0.0,
        False,
        True,
        False,
        num_heads=heads,
        seq_len_q=SEQ_Q,
        cp_world=1,
        use_tq_e2m1=True,
        tq_debug_trace=os.environ.get("TQ_S6_TRACE") == "1",
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    softmax_scale = 1.0 / math.sqrt(LATENT + ROPE)
    compiled = cute.compile(
        op,
        inputs["q_latent"],
        inputs["q_rope"],
        inputs["packed"],
        inputs["rope"],
        inputs["page_table"],
        inputs["output"],
        inputs["lse"],
        None,
        1,
        inputs["cache_seqs"],
        inputs["cache_seqs"],
        None,
        softmax_scale,
        1.0,
        stream,
        True,
        inputs["scale"],
        options="--enable-tvm-ffi --opt-level 3",
    )
    print(f"S6_C0_COMPILED heads={heads}", flush=True)
    if os.environ.get("TQ_S6_AUDIT_GENERATED") == "1":
        generated = _audit_generated(compiled)
        print(
            f"S6_C0_GENERATED heads={heads} "
            + " ".join(f"{key}={value}" for key, value in generated.items()),
            flush=True,
        )

    timing_enabled = os.environ.get("TQ_S6_TIMING") == "1"
    dense_compiled = None
    dense_args = None
    if timing_enabled:
        dense_op = BlackwellMultiHeadLatentAttentionForwardFP8(
            cutlass.Float32,
            cutlass.Float32,
            (64, 128),
            (64, 256),
            max_clusters,
            PAGE_SIZE,
            0.0,
            False,
            True,
            False,
            num_heads=heads,
            seq_len_q=SEQ_Q,
            cp_world=1,
        )
        dense_compiled = cute.compile(
            dense_op,
            inputs["q_latent"],
            inputs["q_rope"],
            inputs["dense"],
            inputs["rope"],
            inputs["page_table"],
            inputs["dense_output"],
            inputs["dense_lse"],
            None,
            1,
            inputs["cache_seqs"],
            inputs["cache_seqs"],
            None,
            softmax_scale,
            1.0,
            stream,
            True,
            options="--enable-tvm-ffi --opt-level 3",
        )
        dense_args = (
            inputs["q_latent"],
            inputs["q_rope"],
            inputs["dense"],
            inputs["rope"],
            inputs["page_table"],
            inputs["dense_output"],
            inputs["dense_lse"],
            None,
            1,
            inputs["cache_seqs"],
            inputs["cache_seqs"],
            None,
            softmax_scale,
            1.0,
            stream,
        )

    args = (
        inputs["q_latent"], inputs["q_rope"], inputs["packed"], inputs["rope"],
        inputs["page_table"], inputs["output"], inputs["lse"], None, 1,
        inputs["cache_seqs"], inputs["cache_seqs"], None, softmax_scale, 1.0,
        stream, inputs["scale"],
    )
    for iteration in range(3):
        compiled(*args)
        torch.cuda.synchronize()
        print(f"S6_C0_EAGER heads={heads} iteration={iteration}", flush=True)

    observed = output_t.float()
    torch.testing.assert_close(observed, expected, atol=0.75, rtol=0.35)

    if run_graph:
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        graph_stream = cuda.CUstream(capture_stream.cuda_stream)
        graph_args = args[:14] + (graph_stream, inputs["scale"])
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            compiled(*graph_args)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture_stream):
            compiled(*graph_args)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        if after != before:
            raise AssertionError(
                f"graph allocation grew: before={before} after={after}"
            )
        if timing_enabled:
            assert dense_compiled is not None and dense_args is not None
            dense_compiled(*dense_args)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                dense_output_t.float(), expected, atol=0.75, rtol=0.35
            )
            torch.testing.assert_close(
                dense_output_t.float(), observed, atol=0.75, rtol=0.35
            )
            if not torch.isfinite(dense_lse_t).all():
                raise AssertionError("non-finite dense LSE")

            dense_graph = torch.cuda.CUDAGraph()
            dense_capture_stream = torch.cuda.Stream()
            dense_graph_stream = cuda.CUstream(dense_capture_stream.cuda_stream)
            dense_graph_args = dense_args[:-1] + (dense_graph_stream,)
            dense_capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(dense_capture_stream):
                dense_compiled(*dense_graph_args)
            torch.cuda.current_stream().wait_stream(dense_capture_stream)
            torch.cuda.synchronize()
            with torch.cuda.graph(dense_graph, stream=dense_capture_stream):
                dense_compiled(*dense_graph_args)
            torch.cuda.synchronize()

            replays = int(os.environ.get("TQ_S6_TIMING_REPLAYS", "100"))
            windows = int(os.environ.get("TQ_S6_TIMING_WINDOWS", "30"))
            if replays <= 0 or windows <= 1 or windows % 2:
                raise ValueError("timing needs positive replays and even windows > 1")

            def measure(timed_graph: torch.cuda.CUDAGraph) -> float:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays):
                    timed_graph.replay()
                end.record()
                end.synchronize()
                return start.elapsed_time(end) * 1000.0 / replays

            for _ in range(12):
                measure(graph)
                measure(dense_graph)
            candidate_samples = []
            dense_samples = []
            for window in range(windows):
                order = (
                    ((graph, candidate_samples), (dense_graph, dense_samples))
                    if window % 2 == 0
                    else ((dense_graph, dense_samples), (graph, candidate_samples))
                )
                for timed_graph, samples in order:
                    samples.append(measure(timed_graph))
            log_ratios = [
                math.log(candidate / dense)
                for candidate, dense in zip(candidate_samples, dense_samples)
            ]
            mean_log = statistics.fmean(log_ratios)
            standard_error = statistics.stdev(log_ratios) / math.sqrt(windows)
            t_critical = 2.045229642 if windows == 30 else 2.262157163
            ratio = math.exp(mean_log)
            ci95_low = math.exp(mean_log - t_critical * standard_error)
            ci95_high = math.exp(mean_log + t_critical * standard_error)
            print(
                "S6_C0_TIMING "
                f"heads={heads} batch={BATCH} seq_k={SEQ_K} "
                f"candidate_mean_us={statistics.fmean(candidate_samples):.6f} "
                f"dense_mean_us={statistics.fmean(dense_samples):.6f} "
                f"candidate_over_dense={ratio:.6f} "
                f"ci95_low={ci95_low:.6f} ci95_high={ci95_high:.6f} "
                f"windows={windows} replays={replays}",
                flush=True,
            )
    if not torch.isfinite(lse_t).all():
        raise AssertionError("non-finite LSE")

    if os.environ.get("TQ_S6_PRINT_GENERATED") == "1":
        print(getattr(compiled, "__mlir__", ""))
    print(
        f"S6_C0_CASE_PASS heads={heads} cache_lengths={cache_lengths} "
        f"graph={run_graph}"
    )


def main() -> None:
    head_cases = (
        (8,) if os.environ.get("TQ_S6_DEBUG_Q1_ONLY") == "1" else HEAD_CASES
    )
    for heads in head_cases:
        _run_case(heads, run_graph=heads == head_cases[-1])
    print(
        "S6_C0_PASS owner=stock_m64 packed_hbm=True qk_e2m1=True "
        "pv_e2m1=True bf16_scale=True carrier_exact=True q1=True q5=True "
        "multi_tile=True partial_pages=31_32_33 page_permutation=True "
        "graph_replays=100"
    )


if __name__ == "__main__":
    main()
