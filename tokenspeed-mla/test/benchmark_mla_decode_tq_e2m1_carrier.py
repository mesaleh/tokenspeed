# Copyright (c) 2026 LightSeek Foundation

"""Crossed same-process timing for two packed E2M1 carrier selectors."""

import hashlib
import importlib.util
import math
import os
import statistics
import struct
import subprocess
import sys
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from benchmark_mla_decode_tq_e2m1_public import (
    HEADS,
    LATENT,
    ROPE,
    _build_cache,
)
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack
from tokenspeed_mla.mla_helpers import get_mla_decode_fold_sq_factor
from tokenspeed_mla.utils import get_max_active_clusters, get_num_sm


def _load_owner(module_name: str, source: str):
    qualified_name = f"tokenspeed_mla.{module_name}"
    spec = importlib.util.spec_from_file_location(qualified_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {qualified_name} from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module.BlackwellMultiHeadLatentAttentionForwardFP8


def _as_cute_tensor(tensor: torch.Tensor, dtype, leading_dim: int, align: int):
    result = from_dlpack(tensor, assumed_align=align, enable_tvm_ffi=True)
    result.element_type = dtype
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _compile(
    owner,
    tensors: dict[str, torch.Tensor],
    query_len: int,
    return_lse: bool,
):
    fold = get_mla_decode_fold_sq_factor(HEADS, query_len, 64)
    split = owner.get_split_kv(
        1,
        query_len // fold,
        tensors["seq_len"],
        (64, 128),
        get_num_sm(torch.device("cuda")),
        1,
    )
    workspace_size = owner.get_workspace_size(
        HEADS * fold,
        query_len // fold,
        LATENT,
        1,
        split,
        cutlass.Float32,
    )
    workspace = (
        None
        if workspace_size == 0
        else torch.empty(workspace_size, dtype=torch.int8, device="cuda")
    )
    output = torch.empty(
        (1, query_len, HEADS, LATENT), dtype=torch.bfloat16, device="cuda"
    )
    lse = (
        torch.empty((1, query_len, HEADS), dtype=torch.float32, device="cuda")
        if return_lse
        else None
    )
    kernel = owner(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(64, 128),
        mma_pv_tiler_mn=(64, 256),
        max_active_clusters=get_max_active_clusters(1),
        page_size=32,
        skip_correction_threshold=0.0,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=False,
        fold_sq_factor=fold,
        is_causal=query_len > 1,
        num_heads=HEADS,
        seq_len_q=query_len,
        cp_world=1,
        use_tq_e2m1=True,
        tq_s1_scale_tma=True,
        tq_s1_scale_stages=3,
        tq_s1_k_rope_stages=2,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel,
        _as_cute_tensor(tensors["query_latent"], cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(tensors["query_rope"], cutlass.BFloat16, 3, 16),
        _as_cute_tensor(tensors["packed"], cutlass.Uint8, 2, 16),
        _as_cute_tensor(tensors["reciprocal_rope"], cutlass.BFloat16, 2, 16),
        _as_cute_tensor(tensors["block_tables"], cutlass.Int32, 1, 4),
        _as_cute_tensor(output, cutlass.BFloat16, 3, 16),
        (_as_cute_tensor(lse, cutlass.Float32, 2, 4) if lse is not None else None),
        (
            _as_cute_tensor(workspace, cutlass.Int8, 0, 32)
            if workspace is not None
            else None
        ),
        Int32(split),
        _as_cute_tensor(tensors["seq_lens"], cutlass.Int32, 0, 4),
        _as_cute_tensor(tensors["seq_lens"], cutlass.Int32, 0, 4),
        None,
        Float32(1.0),
        Float32(1.0),
        stream,
        False,
        _as_cute_tensor(tensors["scale"], cutlass.BFloat16, 1, 16),
        options="--enable-tvm-ffi --opt-level 3",
    )
    softmax_scale = Float32(1.0 / math.sqrt(LATENT + ROPE))

    def call():
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            compiled(
                tensors["query_latent"],
                tensors["query_rope"],
                tensors["packed"],
                tensors["reciprocal_rope"],
                tensors["block_tables"],
                output,
                lse,
                workspace,
                Int32(split),
                tensors["seq_lens"],
                tensors["seq_lens"],
                None,
                softmax_scale,
                Float32(1.0),
                tensors["scale"],
            )

    return call, output, lse, compiled


def _report_artifact(name: str, query_len: int, compiled) -> tuple[str, str]:
    artifacts = getattr(compiled, "artifacts", None)
    cubin = getattr(artifacts, "CUBIN", None)
    ptx = getattr(artifacts, "PTX", None)
    if not isinstance(cubin, bytes) or not isinstance(ptx, str):
        raise RuntimeError(
            "carrier benchmark requires retained CUBIN/PTX artifacts; "
            "set CUTE_DSL_KEEP=all"
        )
    cubin_hash = hashlib.sha256(cubin).hexdigest()
    ptx_hash = hashlib.sha256(ptx.encode()).hexdigest()
    print(
        "TQ_CARRIER_ARTIFACT "
        f"name={name} q_len={query_len} "
        f"cubin_sha256={cubin_hash} "
        f"ptx_sha256={ptx_hash}",
        flush=True,
    )
    dump_root = os.environ.get("TQ_CARRIER_DUMP_DIR")
    if dump_root:
        output = Path(dump_root)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{name}-q{query_len}.cubin").write_bytes(cubin)
        (output / f"{name}-q{query_len}.ptx").write_text(ptx)
    return cubin_hash, ptx_hash


def _f32_from_bf16(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits << 16))[0]


def _inject_boundary_scales(scale: torch.Tensor) -> None:
    values = []
    writer_limit = 224.0 * 2.0**16
    for exponent in range(-16, 17):
        boundary = 224.0 * 2.0**exponent
        boundary_bits = struct.unpack(">I", struct.pack(">f", boundary))[0] >> 16
        for bits in range(max(1, boundary_bits - 1), boundary_bits + 2):
            value = _f32_from_bf16(bits)
            if value <= writer_limit:
                values.append(value)
    device_values = torch.tensor(values, dtype=torch.bfloat16, device=scale.device)
    flat = scale.view(-1)
    flat.copy_(
        device_values.repeat(
            (flat.numel() + device_values.numel() - 1) // device_values.numel()
        )[: flat.numel()]
    )


def _gpu_state(stage: str) -> None:
    fields = (
        "index,uuid,driver_version,pstate,clocks.sm,clocks.mem,"
        "ecc.errors.uncorrected.volatile.total"
    )
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rows = [row.split(", ") for row in output.splitlines()]
    expected_sm = os.environ.get("TQ_CARRIER_EXPECT_SM_CLOCK")
    expected_mem = os.environ.get("TQ_CARRIER_EXPECT_MEM_CLOCK")
    for row in rows:
        if row[3] != "P0" or int(row[6]) != 0:
            raise RuntimeError(f"GPU state is not scoreable: {row}")
        if expected_sm is not None and row[4] != expected_sm:
            raise RuntimeError(f"unexpected SM clock: {row[4]} != {expected_sm}")
        if expected_mem is not None and row[5] != expected_mem:
            raise RuntimeError(f"unexpected memory clock: {row[5]} != {expected_mem}")
    print(f"TQ_CARRIER_GPU_STATE stage={stage} rows={rows}", flush=True)


def _capture(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return graph


def _measure(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def _run(
    query_len: int,
    seq_len: int,
    control_owner,
    candidate_owner,
    replays: int,
    windows: int,
    *,
    return_lse: bool,
    scale_mode: str,
    timing: bool,
):
    physical_seq_len = math.ceil(seq_len / 32) * 32
    packed, scale, reciprocal_rope, _, block_tables, seq_lens = _build_cache(
        1, physical_seq_len
    )
    seq_lens.fill_(seq_len)
    padded_pages = math.ceil(block_tables.shape[1] / 4) * 4
    if block_tables.shape[1] < padded_pages:
        block_tables = torch.cat(
            (
                block_tables,
                block_tables[:, -1:].expand(-1, padded_pages - block_tables.shape[1]),
            ),
            dim=1,
        ).contiguous()
    if scale_mode == "boundary":
        _inject_boundary_scales(scale)
    elif scale_mode != "fixture":
        raise ValueError("TQ_CARRIER_SCALE_MODE must be fixture or boundary")
    torch.manual_seed(20260816 + query_len)
    tensors = {
        "query_latent": (
            torch.randint(-4, 5, (1, query_len, HEADS, LATENT), device="cuda") / 2.0
        ).to(torch.float8_e4m3fn),
        "query_rope": (
            torch.randint(-4, 5, (1, query_len, HEADS, ROPE), device="cuda") / 2.0
        ).to(torch.bfloat16),
        "packed": packed,
        "scale": scale,
        "reciprocal_rope": reciprocal_rope,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "seq_len": seq_len,
    }
    owners = {"control": control_owner, "candidate": candidate_owner}
    compile_order = os.environ.get("TQ_CARRIER_COMPILE_ORDER", "control,candidate")
    order = tuple(compile_order.split(","))
    if set(order) != {"control", "candidate"} or len(order) != 2:
        raise ValueError(
            "TQ_CARRIER_COMPILE_ORDER must be control,candidate or candidate,control"
        )
    compiled_roles = {}
    for name in order:
        compiled_roles[name] = _compile(owners[name], tensors, query_len, return_lse)
    control_call, control_output, control_lse, control_compiled = compiled_roles[
        "control"
    ]
    candidate_call, candidate_output, candidate_lse, candidate_compiled = (
        compiled_roles["candidate"]
    )
    control_identity = _report_artifact("control", query_len, control_compiled)
    candidate_identity = _report_artifact("candidate", query_len, candidate_compiled)
    if control_identity[0] == candidate_identity[0]:
        raise AssertionError("control and candidate CUBIN hashes must differ")
    if control_identity[1] == candidate_identity[1]:
        raise AssertionError("control and candidate PTX hashes must differ")
    control_call()
    candidate_call()
    torch.cuda.synchronize()
    if not torch.equal(candidate_output, control_output):
        difference = (candidate_output.float() - control_output.float()).abs()
        raise AssertionError(
            f"eager output mismatch max={difference.max().item()} "
            f"count={torch.count_nonzero(difference).item()}"
        )
    if return_lse and not torch.equal(candidate_lse, control_lse):
        raise AssertionError("eager control/candidate LSE differs")
    control_graph = _capture(control_call)
    candidate_graph = _capture(candidate_call)
    control_graph.replay()
    candidate_graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(candidate_output, control_output):
        raise AssertionError("captured control/candidate outputs differ")
    if return_lse and not torch.equal(candidate_lse, control_lse):
        raise AssertionError("captured control/candidate LSE differs")

    if not timing:
        print(
            "TQ_CARRIER_PARITY "
            f"q_len={query_len} seq_len={seq_len} scale_mode={scale_mode} "
            f"return_lse={return_lse} compile_order={compile_order}",
            flush=True,
        )
        return

    for _ in range(12):
        _measure(control_graph, replays)
        _measure(candidate_graph, replays)
    samples = {"control": [], "candidate": []}
    graphs = {"control": control_graph, "candidate": candidate_graph}
    for window in range(windows):
        order = (
            ("control", "candidate")
            if window % 2 == 0
            else (
                "candidate",
                "control",
            )
        )
        for name in order:
            samples[name].append(_measure(graphs[name], replays))

    log_ratios = [
        math.log(candidate / control)
        for candidate, control in zip(samples["candidate"], samples["control"])
    ]
    mean_log = statistics.fmean(log_ratios)
    standard_error = statistics.stdev(log_ratios) / math.sqrt(windows)
    critical = 2.045229642 if windows == 30 else 2.262157163
    ratio = math.exp(mean_log)
    lower = math.exp(mean_log - critical * standard_error)
    upper = math.exp(mean_log + critical * standard_error)
    print(
        "TQ_CARRIER_CROSSED "
        f"q_len={query_len} seq_len={seq_len} "
        f"scale_mode={scale_mode} return_lse={return_lse} "
        f"compile_order={compile_order} "
        f"control_us={statistics.fmean(samples['control']):.6f} "
        f"candidate_us={statistics.fmean(samples['candidate']):.6f} "
        f"candidate_over_control={ratio:.6f} "
        f"ci95=[{lower:.6f},{upper:.6f}] "
        f"windows={windows} replays={replays}",
        flush=True,
    )


def main():
    control_source = os.environ["TQ_CARRIER_CONTROL_SOURCE"]
    candidate_source = os.environ["TQ_CARRIER_CANDIDATE_SOURCE"]
    if Path(control_source).resolve() == Path(candidate_source).resolve():
        raise ValueError("control and candidate source paths must differ")
    control_owner = _load_owner("mla_decode_fp8_carrier_control", control_source)
    candidate_owner = _load_owner("mla_decode_fp8_carrier_candidate", candidate_source)
    replays = int(os.environ.get("TQ_CARRIER_REPLAYS", "500"))
    windows = int(os.environ.get("TQ_CARRIER_WINDOWS", "30"))
    if windows not in (10, 30):
        raise ValueError("TQ_CARRIER_WINDOWS must be 10 or 30")
    sequence_lengths = tuple(
        int(value)
        for value in os.environ.get("TQ_CARRIER_SEQ_LENS", "10240").split(",")
    )
    if not sequence_lengths or any(value <= 0 for value in sequence_lengths):
        raise ValueError("TQ_CARRIER_SEQ_LENS must contain positive integers")
    query_lengths = tuple(
        int(value)
        for value in os.environ.get("TQ_CARRIER_QUERY_LENS", "1,5").split(",")
    )
    if not query_lengths or any(value not in (1, 5) for value in query_lengths):
        raise ValueError("TQ_CARRIER_QUERY_LENS must contain only 1 and/or 5")
    return_lse = os.environ.get("TQ_CARRIER_RETURN_LSE", "0") == "1"
    scale_mode = os.environ.get("TQ_CARRIER_SCALE_MODE", "fixture")
    timing = os.environ.get("TQ_CARRIER_TIMING", "1") == "1"
    _gpu_state("before")
    for query_len in query_lengths:
        for seq_len in sequence_lengths:
            _run(
                query_len,
                seq_len,
                control_owner,
                candidate_owner,
                replays,
                windows,
                return_lse=return_lse,
                scale_mode=scale_mode,
                timing=timing,
            )
    _gpu_state("after")


if __name__ == "__main__":
    main()
