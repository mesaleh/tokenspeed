# Copyright (c) 2026 LightSeek Foundation

"""Isolate device-address placement in the packed E2M1 MLA reader.

The scored path compiles the accepted reader once, keeps all inputs and the
non-scanned tensor family fixed, and captures one graph per disjoint workspace
or output view.  It is a research harness, not a runtime allocator.
"""

import hashlib
import inspect
import json
import math
import os
import random
import statistics
import subprocess
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from benchmark_mla_decode_tq_e2m1_public import (
    HEADS,
    LATENT,
    ROPE,
    _build_cache,
)
from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
)
from tokenspeed_mla.mla_helpers import get_mla_decode_fold_sq_factor
from tokenspeed_mla.utils import get_max_active_clusters, get_num_sm


PERIOD = 8 * 1024 * 1024
CLASS_STEP = 128 * 1024
NUM_CLASSES = PERIOD // CLASS_STEP
GUARD_BYTES = 256
GUARD_VALUE = 0x5A


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _as_cute_tensor(tensor: torch.Tensor, dtype, leading_dim: int, align: int):
    result = from_dlpack(tensor, assumed_align=align, enable_tvm_ffi=True)
    result.element_type = dtype
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _artifact_identity(compiled) -> dict[str, str]:
    artifacts = getattr(compiled, "artifacts", None)
    cubin = getattr(artifacts, "CUBIN", None)
    ptx = getattr(artifacts, "PTX", None)
    if not isinstance(cubin, bytes) or not isinstance(ptx, str):
        raise RuntimeError(
            "placement benchmark requires retained CUBIN/PTX artifacts; "
            "set CUTE_DSL_KEEP=all"
        )
    return {
        "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
    }


def _gpu_state(stage: str) -> list[list[str]]:
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
    expected_sm = os.environ.get("TQ_PLACEMENT_EXPECT_SM_CLOCK")
    expected_mem = os.environ.get("TQ_PLACEMENT_EXPECT_MEM_CLOCK")
    for row in rows:
        if row[3] != "P0" or int(row[6]) != 0:
            raise RuntimeError(f"GPU state is not scoreable: {row}")
        if expected_sm is not None and row[4] != expected_sm:
            raise RuntimeError(f"unexpected SM clock: {row[4]} != {expected_sm}")
        if expected_mem is not None and row[5] != expected_mem:
            raise RuntimeError(f"unexpected memory clock: {row[5]} != {expected_mem}")
    print(
        "TQ_PLACEMENT_GPU_STATE "
        + json.dumps({"stage": stage, "rows": rows}, sort_keys=True),
        flush=True,
    )
    return rows


def _build_fixture(query_len: int, seq_len: int) -> dict[str, torch.Tensor | int]:
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
                block_tables[:, -1:].expand(
                    -1, padded_pages - block_tables.shape[1]
                ),
            ),
            dim=1,
        ).contiguous()
    torch.manual_seed(20260816 + query_len)
    query_latent = (
        torch.randint(-4, 5, (1, query_len, HEADS, LATENT), device="cuda")
        / 2.0
    ).to(torch.float8_e4m3fn)
    query_rope = (
        torch.randint(-4, 5, (1, query_len, HEADS, ROPE), device="cuda")
        / 2.0
    ).to(torch.bfloat16)
    return {
        "query_latent": query_latent,
        "query_rope": query_rope,
        "packed": packed,
        "scale": scale,
        "reciprocal_rope": reciprocal_rope,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "seq_len": seq_len,
    }


def _workspace_sizes(query_len: int, seq_len: int) -> tuple[int, int, int]:
    fold = get_mla_decode_fold_sq_factor(HEADS, query_len, 64)
    split = BlackwellMultiHeadLatentAttentionForwardFP8.get_split_kv(
        1,
        query_len // fold,
        seq_len,
        (64, 128),
        get_num_sm(torch.device("cuda")),
        1,
    )
    required = BlackwellMultiHeadLatentAttentionForwardFP8.get_workspace_size(
        HEADS * fold,
        query_len // fold,
        LATENT,
        1,
        split,
        cutlass.Float32,
    )
    production_capacity = (
        get_num_sm(torch.device("cuda"))
        * HEADS
        * max(query_len, 8)
        * (LATENT + 1)
        * 4
    )
    if required <= 0 or production_capacity < required:
        raise RuntimeError(
            f"invalid workspace sizing: required={required}, "
            f"production_capacity={production_capacity}"
        )
    return split, required, production_capacity


def _make_class_views(
    *,
    payload_bytes: int,
    capacity_bytes: int,
    permutation_seed: int,
    target_classes: list[int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[dict[str, int]]]:
    if payload_bytes <= 0 or capacity_bytes < payload_bytes:
        raise ValueError("class-view payload/capacity is invalid")
    slot_span = _round_up(
        GUARD_BYTES + capacity_bytes + PERIOD + GUARD_BYTES,
        CLASS_STEP,
    )
    parent = torch.empty(
        slot_span * len(target_classes),
        dtype=dtype,
        device="cuda",
    )
    parent.fill_(GUARD_VALUE)
    permutation = target_classes.copy()
    random.Random(permutation_seed).shuffle(permutation)
    views: dict[int, torch.Tensor] = {}
    records: list[dict[str, int]] = []
    parent_ptr = parent.data_ptr()
    for physical_slot, target_class in enumerate(permutation):
        raw_start = physical_slot * slot_span + GUARD_BYTES
        raw_ptr = parent_ptr + raw_start
        prefix = (target_class - (raw_ptr % PERIOD)) % PERIOD
        if prefix % 256:
            raise RuntimeError(
                f"target class {target_class} is unreachable from raw pointer "
                f"{raw_ptr:#x} with a 256-byte-aligned prefix"
            )
        start = raw_start + prefix
        end = start + payload_bytes
        slot_end = (physical_slot + 1) * slot_span
        if start - GUARD_BYTES < physical_slot * slot_span:
            raise RuntimeError("prefix guard escaped its physical slot")
        if end + GUARD_BYTES > slot_end:
            raise RuntimeError("suffix guard escaped its physical slot")
        view = parent.narrow(0, start, payload_bytes)
        if not view.is_contiguous() or view.data_ptr() % 32:
            raise RuntimeError("class view violates contiguous/32-byte contract")
        if view.data_ptr() % PERIOD != target_class:
            raise RuntimeError("class view does not realize its target residue")
        views[target_class] = view
        records.append(
            {
                "target_class": target_class,
                "physical_slot": physical_slot,
                "slot_span": slot_span,
                "raw_start": raw_start,
                "prefix": prefix,
                "view_start": start,
                "view_ptr": view.data_ptr(),
                "parent_ptr": parent_ptr,
                "payload_bytes": payload_bytes,
                "capacity_bytes": capacity_bytes,
                "residue_128k": view.data_ptr() % CLASS_STEP,
                "residue_256k": view.data_ptr() % (256 * 1024),
                "residue_512k": view.data_ptr() % (512 * 1024),
                "residue_1m": view.data_ptr() % (1024 * 1024),
                "residue_2m": view.data_ptr() % (2 * 1024 * 1024),
                "residue_4m": view.data_ptr() % (4 * 1024 * 1024),
                "residue_8m": view.data_ptr() % PERIOD,
                "residue_16m": view.data_ptr() % (16 * 1024 * 1024),
                "residue_32m": view.data_ptr() % (32 * 1024 * 1024),
            }
        )
    if len(views) != len(target_classes):
        raise RuntimeError("class permutation did not produce unique views")
    return parent, views, records


def _compile(
    tensors: dict[str, torch.Tensor | int],
    query_len: int,
    output: torch.Tensor,
    lse: torch.Tensor | None,
    workspace: torch.Tensor,
    split: int,
):
    fold = get_mla_decode_fold_sq_factor(HEADS, query_len, 64)
    kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
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
    return cute.compile(
        kernel,
        _as_cute_tensor(tensors["query_latent"], cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(tensors["query_rope"], cutlass.BFloat16, 3, 16),
        _as_cute_tensor(tensors["packed"], cutlass.Uint8, 2, 16),
        _as_cute_tensor(tensors["reciprocal_rope"], cutlass.BFloat16, 2, 16),
        _as_cute_tensor(tensors["block_tables"], cutlass.Int32, 1, 4),
        _as_cute_tensor(output, cutlass.BFloat16, 3, 16),
        _as_cute_tensor(lse, cutlass.Float32, 2, 4) if lse is not None else None,
        _as_cute_tensor(workspace, cutlass.Int8, 0, 32),
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


def _invoke(
    compiled,
    tensors: dict[str, torch.Tensor | int],
    output: torch.Tensor,
    lse: torch.Tensor | None,
    workspace: torch.Tensor,
    split: int,
):
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
            Float32(1.0 / math.sqrt(LATENT + ROPE)),
            Float32(1.0),
            tensors["scale"],
        )


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


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    begin = 0
    while begin < len(ordered):
        end = begin + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[begin]]:
            end += 1
        rank = (begin + end - 1) / 2.0
        for index in ordered[begin:end]:
            ranks[index] = rank
        begin = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (lvalue - left_mean) * (rvalue - right_mean)
        for lvalue, rvalue in zip(left, right)
    )
    left_sum = sum((value - left_mean) ** 2 for value in left)
    right_sum = sum((value - right_mean) ** 2 for value in right)
    if left_sum == 0.0 or right_sum == 0.0:
        return 0.0
    return numerator / math.sqrt(left_sum * right_sum)


def _check_guards(parent: torch.Tensor, records: list[dict[str, int]]) -> None:
    for record in records:
        start = record["view_start"]
        end = start + record["payload_bytes"]
        before = parent.narrow(0, start - GUARD_BYTES, GUARD_BYTES)
        after = parent.narrow(0, end, GUARD_BYTES)
        if not torch.all(before == GUARD_VALUE).item():
            raise AssertionError(
                f"prefix redzone changed for class {record['target_class']}"
            )
        if not torch.all(after == GUARD_VALUE).item():
            raise AssertionError(
                f"suffix redzone changed for class {record['target_class']}"
            )


def _poison_outputs(output: torch.Tensor, lse: torch.Tensor | None) -> None:
    output.fill_(float("nan"))
    if lse is not None:
        lse.fill_(float("nan"))


def _check_outputs_written(output: torch.Tensor, lse: torch.Tensor | None) -> None:
    if torch.isnan(output).any().item():
        raise AssertionError("reader left NaN poison in the output")
    if lse is not None and torch.isnan(lse).any().item():
        raise AssertionError("reader left NaN poison in the LSE output")


def main():
    if NUM_CLASSES != 64:
        raise RuntimeError("placement class geometry changed unexpectedly")
    query_len = int(os.environ.get("TQ_PLACEMENT_QUERY_LEN", "5"))
    seq_len = int(os.environ.get("TQ_PLACEMENT_SEQ_LEN", "10240"))
    if query_len not in (1, 5) or seq_len <= 0:
        raise ValueError("placement harness supports positive K and q1/q5")
    owner = os.environ.get("TQ_PLACEMENT_OWNER", "workspace")
    if owner not in ("workspace", "output"):
        raise ValueError("TQ_PLACEMENT_OWNER must be workspace or output")
    allocation_model = os.environ.get("TQ_PLACEMENT_ALLOCATION_MODEL", "production")
    if allocation_model not in ("production", "exact"):
        raise ValueError("allocation model must be production or exact")
    permutation_seed = int(
        os.environ.get("TQ_PLACEMENT_PERMUTATION_SEED", "2026081601")
    )
    timing_seed = int(os.environ.get("TQ_PLACEMENT_TIMING_SEED", "2026081699"))
    poison_bytes = int(os.environ.get("TQ_PLACEMENT_POISON_BYTES", "0"))
    windows = int(os.environ.get("TQ_PLACEMENT_WINDOWS", "30"))
    replays = int(os.environ.get("TQ_PLACEMENT_REPLAYS", "500"))
    warm_traversals = int(os.environ.get("TQ_PLACEMENT_WARM_TRAVERSALS", "12"))
    timing = os.environ.get("TQ_PLACEMENT_TIMING", "1") == "1"
    return_lse = os.environ.get("TQ_PLACEMENT_RETURN_LSE", "0") == "1"
    target_classes_value = os.environ.get("TQ_PLACEMENT_TARGET_CLASSES")
    target_classes = (
        list(range(0, PERIOD, CLASS_STEP))
        if target_classes_value is None
        else [int(value) for value in target_classes_value.split(",")]
    )
    if (
        not target_classes
        or len(set(target_classes)) != len(target_classes)
        or any(
            value < 0 or value >= PERIOD or value % CLASS_STEP
            for value in target_classes
        )
    ):
        raise ValueError(
            "target classes must be unique 128-KiB multiples in [0, 8 MiB)"
        )
    if windows != 30 or replays < 500 or warm_traversals < 12:
        raise ValueError(
            "scored geometry requires 30 windows, >=500 replays, "
            ">=12 warm traversals"
        )
    _gpu_state("before")
    tensors = _build_fixture(query_len, seq_len)
    input_pointers = {
        name: value.data_ptr()
        for name, value in tensors.items()
        if isinstance(value, torch.Tensor)
    }
    poison = None
    if poison_bytes:
        poison = torch.empty(poison_bytes, dtype=torch.uint8, device="cuda")
        poison.fill_(0xA5)
    split, workspace_required, production_capacity = _workspace_sizes(
        query_len, seq_len
    )
    workspace_capacity = (
        production_capacity if allocation_model == "production" else workspace_required
    )
    output_shape = (1, query_len, HEADS, LATENT)
    output_bytes = math.prod(output_shape) * 2
    fixed_workspace = torch.empty(
        workspace_capacity, dtype=torch.int8, device="cuda"
    )[:workspace_required]
    fixed_output = torch.empty(output_shape, dtype=torch.bfloat16, device="cuda")
    lse = (
        torch.empty((1, query_len, HEADS), dtype=torch.float32, device="cuda")
        if return_lse
        else None
    )
    if owner == "workspace":
        parent, byte_views, records = _make_class_views(
            payload_bytes=workspace_required,
            capacity_bytes=workspace_capacity,
            permutation_seed=permutation_seed,
            target_classes=target_classes,
            dtype=torch.int8,
        )
        workspaces = byte_views
        outputs = {target: fixed_output for target in byte_views}
    else:
        parent, byte_views, records = _make_class_views(
            payload_bytes=output_bytes,
            capacity_bytes=output_bytes,
            permutation_seed=permutation_seed,
            target_classes=target_classes,
            dtype=torch.uint8,
        )
        workspaces = {target: fixed_workspace for target in byte_views}
        outputs = {
            target: byte_view.view(torch.bfloat16).view(output_shape)
            for target, byte_view in byte_views.items()
        }
        for target, output in outputs.items():
            if output.data_ptr() % PERIOD != target:
                raise RuntimeError("typed output view changed target residue")
    first_class = min(workspaces)
    compiled = _compile(
        tensors,
        query_len,
        outputs[first_class],
        lse,
        workspaces[first_class],
        split,
    )
    artifacts = _artifact_identity(compiled)
    source_path = Path(__file__).resolve()
    reader_source = inspect.getsourcefile(BlackwellMultiHeadLatentAttentionForwardFP8)
    if reader_source is None:
        raise RuntimeError("cannot resolve the accepted reader source path")
    reader_path = Path(reader_source).resolve()
    reader_sha256 = hashlib.sha256(reader_path.read_bytes()).hexdigest()
    expected_reader_sha256 = os.environ.get("TQ_PLACEMENT_EXPECT_READER_SHA256")
    expected_cubin_sha256 = os.environ.get("TQ_PLACEMENT_EXPECT_CUBIN_SHA256")
    if timing and (expected_reader_sha256 is None or expected_cubin_sha256 is None):
        raise RuntimeError(
            "scored timing requires expected reader and CUBIN SHA-256 identities"
        )
    if expected_reader_sha256 is not None and reader_sha256 != expected_reader_sha256:
        raise RuntimeError(
            f"reader SHA-256 drift: {reader_sha256} != {expected_reader_sha256}"
        )
    if (
        expected_cubin_sha256 is not None
        and artifacts["cubin_sha256"] != expected_cubin_sha256
    ):
        raise RuntimeError(
            "CUBIN SHA-256 drift: "
            f"{artifacts['cubin_sha256']} != {expected_cubin_sha256}"
        )
    config = {
        "schema_version": 1,
        "owner": owner,
        "allocation_model": allocation_model,
        "query_len": query_len,
        "seq_len": seq_len,
        "split": split,
        "workspace_required": workspace_required,
        "workspace_capacity": workspace_capacity,
        "production_capacity": production_capacity,
        "output_bytes": output_bytes,
        "period": PERIOD,
        "class_step": CLASS_STEP,
        "num_classes": len(target_classes),
        "target_classes": target_classes,
        "permutation_seed": permutation_seed,
        "timing_seed": timing_seed,
        "poison_bytes": poison_bytes,
        "poison_ptr": poison.data_ptr() if poison is not None else None,
        "windows": windows,
        "replays": replays,
        "warm_traversals": warm_traversals,
        "return_lse": return_lse,
        "timing": timing,
        "input_pointers": input_pointers,
        "fixed_workspace_ptr": fixed_workspace.data_ptr(),
        "fixed_output_ptr": fixed_output.data_ptr(),
        "harness_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "reader_source": str(reader_path),
        "reader_sha256": reader_sha256,
        **artifacts,
    }
    print("TQ_PLACEMENT_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    print("TQ_PLACEMENT_LAYOUT " + json.dumps(records, sort_keys=True), flush=True)
    reference_output = None
    reference_lse = None
    graphs: dict[int, torch.cuda.CUDAGraph] = {}
    for target_class in sorted(workspaces):
        output = outputs[target_class]
        workspace = workspaces[target_class]
        _poison_outputs(output, lse)
        _invoke(compiled, tensors, output, lse, workspace, split)
        torch.cuda.synchronize()
        _check_outputs_written(output, lse)
        if reference_output is None:
            reference_output = output.clone()
            reference_lse = lse.clone() if lse is not None else None
        elif not torch.equal(output, reference_output):
            difference = (output.float() - reference_output.float()).abs()
            raise AssertionError(
                f"eager output mismatch class={target_class} "
                f"max={difference.max().item()} "
                f"count={torch.count_nonzero(difference).item()}"
            )
        if lse is not None and not torch.equal(lse, reference_lse):
            raise AssertionError(f"eager LSE mismatch class={target_class}")
        graphs[target_class] = _capture(
            lambda output=output, workspace=workspace: _invoke(
                compiled, tensors, output, lse, workspace, split
            )
        )
    for target_class, graph in graphs.items():
        _poison_outputs(outputs[target_class], lse)
        graph.replay()
        torch.cuda.synchronize()
        _check_outputs_written(outputs[target_class], lse)
        if not torch.equal(outputs[target_class], reference_output):
            raise AssertionError(f"captured output mismatch class={target_class}")
        if lse is not None and not torch.equal(lse, reference_lse):
            raise AssertionError(f"captured LSE mismatch class={target_class}")
    _check_guards(parent, records)
    if not timing:
        print(
            "TQ_PLACEMENT_PARITY "
            + json.dumps(
                {
                    "classes": len(target_classes),
                    "output_sha256": hashlib.sha256(
                        reference_output.cpu().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    "lse_sha256": (
                        hashlib.sha256(
                            reference_lse.cpu().view(torch.uint8).numpy().tobytes()
                        ).hexdigest()
                        if reference_lse is not None
                        else None
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _gpu_state("after")
        return
    classes = sorted(graphs)
    for traversal in range(warm_traversals):
        order = classes.copy()
        random.Random(timing_seed + 100_000 + traversal).shuffle(order)
        for target_class in order:
            _measure(graphs[target_class], replays)
    samples = {target_class: [] for target_class in classes}
    for window in range(windows):
        order = classes.copy()
        random.Random(timing_seed + window).shuffle(order)
        for target_class in order:
            samples[target_class].append(_measure(graphs[target_class], replays))
    record_by_class = {record["target_class"]: record for record in records}
    results = []
    for target_class in classes:
        values = samples[target_class]
        record = dict(record_by_class[target_class])
        record.update(
            {
                "mean_us": statistics.fmean(values),
                "median_us": statistics.median(values),
                "stdev_us": statistics.stdev(values),
                "min_us": min(values),
                "max_us": max(values),
                "samples_us": values,
            }
        )
        results.append(record)
        print("TQ_PLACEMENT_CLASS " + json.dumps(record, sort_keys=True), flush=True)
    means = [record["mean_us"] for record in results]
    slots = [float(record["physical_slot"]) for record in results]
    ranked_means = _rank(means)
    ranked_slots = _rank(slots)
    sorted_by_mean = sorted(results, key=lambda record: record["mean_us"])
    median_record = sorted_by_mean[len(sorted_by_mean) // 2]
    best = sorted_by_mean[0]
    worst = sorted_by_mean[-1]
    summary = {
        "best_class": best["target_class"],
        "best_mean_us": best["mean_us"],
        "median_class": median_record["target_class"],
        "median_mean_us": median_record["mean_us"],
        "worst_class": worst["target_class"],
        "worst_mean_us": worst["mean_us"],
        "best_over_median": best["mean_us"] / median_record["mean_us"],
        "worst_over_best": worst["mean_us"] / best["mean_us"],
        "spearman_mean_vs_physical_slot": _correlation(
            ranked_means, ranked_slots
        ),
    }
    print("TQ_PLACEMENT_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    for target_class, graph in graphs.items():
        _poison_outputs(outputs[target_class], lse)
        graph.replay()
        torch.cuda.synchronize()
        _check_outputs_written(outputs[target_class], lse)
        if not torch.equal(outputs[target_class], reference_output):
            raise AssertionError(f"post-timing output mismatch class={target_class}")
        if lse is not None and not torch.equal(lse, reference_lse):
            raise AssertionError(f"post-timing LSE mismatch class={target_class}")
    _check_guards(parent, records)
    _gpu_state("after")


if __name__ == "__main__":
    main()
