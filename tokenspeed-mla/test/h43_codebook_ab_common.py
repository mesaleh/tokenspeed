#!/usr/bin/env python3
"""Shared, CPU-testable helpers for the H43 reader experiment."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable


def load_contract(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    if value.get("schema_version") != 1:
        raise ValueError(
            f"unsupported H43 contract schema: {value.get('schema_version')}"
        )
    if value.get("experiment") != "H43_D1_CODEBOOK_READER_AB":
        raise ValueError("unexpected H43 experiment identifier")
    accepted = value["accepted_service"]
    for rank_index, rank_name in enumerate(("rank0", "rank1")):
        rank = accepted[rank_name]
        if (
            rank["process_marker"] != f"--node-rank {rank_index}"
            or rank["node_rank_marker"] != f"node_rank={rank_index}"
        ):
            raise ValueError(f"{rank_name} restore predicates are inconsistent")
    geometry = value["geometry"]
    if geometry["cache_rows"] % geometry["page_size"]:
        raise ValueError("cache rows must be divisible by page size")
    if (
        geometry["dense_before"] + geometry["selected_layers"] + geometry["dense_after"]
        != geometry["total_layers"]
    ):
        raise ValueError("layer-ring geometry is inconsistent")
    if geometry["selected_layer_start"] != geometry["dense_before"]:
        raise ValueError("selected-layer start is inconsistent")
    if geometry["query_length"] not in geometry["correctness_query_lengths"]:
        raise ValueError("timed query length is absent from correctness lengths")
    pilot = value["pilot"]
    timing = value["timing"]
    if pilot["aa_pairs_per_process"] != timing["pairs_per_process"]:
        raise ValueError("pilot and decision pair counts must match")
    if pilot["preflight_processes_per_context"] != 2:
        raise ValueError("H43 pilot requires exactly two preflight processes")
    if pilot["expanded_processes_per_context"] > value["seeds"]["max_sequence"]:
        raise ValueError("expanded process count exceeds predeclared sequences")
    memory = value["memory_bytes_per_rank"]
    if (
        memory["codebook_cache"] - memory["no_codebook_cache"]
        != memory["codebook_allocation"]
        or memory["dense_control_cache"] - memory["codebook_cache"]
        != memory["persistent_saving"]
        or memory["persistent_saving"] - memory["writer_workspace"]
        != memory["projected_net_saving"]
    ):
        raise ValueError("memory arithmetic is inconsistent")
    required_gates = value["qualification"]["required_gates"]
    if not required_gates or len(required_gates) != len(set(required_gates)):
        raise ValueError("qualification gates must be nonempty and unique")
    ncu = value["ncu"]
    if (
        ncu["expected_ring_calls"] != geometry["total_layers"]
        or ncu["selected_call_start"] != geometry["dense_before"]
        or ncu["selected_call_count"] != geometry["selected_layers"]
        or ncu["launch_kernel_order"] != ["split_kv_kernel", "reduction_kernel"]
    ):
        raise ValueError("NCU call selection differs from the 61-layer ring")
    if not ncu["required_metrics"] or len(ncu["required_metrics"]) != len(
        set(ncu["required_metrics"])
    ):
        raise ValueError("NCU metrics must be nonempty and unique")
    timeouts = value["timeouts_seconds"]
    if any(
        not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0
        for timeout in timeouts.values()
    ):
        raise ValueError("H43 timeouts must be positive integer seconds")
    maintenance = value["maintenance_seconds"]
    expected_experiments = {
        "qualification": timeouts["qualification_experiment"],
        f"decision_n{pilot['base_processes_per_context']}": timeouts[
            f"decision_n{pilot['base_processes_per_context']}_experiment"
        ],
        f"decision_n{pilot['expanded_processes_per_context']}": timeouts[
            f"decision_n{pilot['expanded_processes_per_context']}_experiment"
        ],
    }
    for label, experiment_seconds in expected_experiments.items():
        window = maintenance[label]
        terminal_minimum = (
            experiment_seconds
            + timeouts["rank1_ready"]
            + timeouts["marker_ready"]
            + 30
            + timeouts["gpu_idle"]
            + timeouts["rank1_ready"]
            + timeouts["http_health"]
            + 60
            + timeouts["final_validation"]
        )
        if window["experiment"] != experiment_seconds or not (
            window["experiment"]
            < window["failsafe"]
            < window["alert"]
            < window["terminal"]
        ):
            raise ValueError(f"maintenance timeout ordering is invalid for {label}")
        if window["terminal"] < terminal_minimum:
            raise ValueError(f"maintenance terminal budget is too small for {label}")
    return value


def canonical_json_digest(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ephemeral(path: Path, cache_contract: dict[str, Any]) -> bool:
    exact = set(cache_contract["ephemeral_exact_basenames"])
    prefixes = tuple(cache_contract["ephemeral_basename_prefixes"])
    return any(part in exact or part.startswith(prefixes) for part in path.parts)


def compiled_artifact_manifest(
    root: Path, cache_contract: dict[str, Any]
) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"compiled cache root is missing: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if _ephemeral(relative, cache_contract):
            continue
        if path.is_symlink():
            raise ValueError(f"compiled cache contains a symlink: {relative}")
        if path.is_file():
            files.append(
                {
                    "path": relative.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    value = {"root": str(root), "files": files}
    value["digest"] = canonical_json_digest(files)
    return value


def run_text(command: list[str], *, timeout: int) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def gpu_covariates(contract: dict[str, Any]) -> dict[str, str]:
    fields = contract["telemetry"]["query_fields"]
    machine = contract["machine"]
    stdout = run_text(
        [
            "nvidia-smi",
            f"--id={machine['gpu_index']}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        timeout=contract["timeouts_seconds"]["command"],
    )
    rows = [row for row in stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected one GPU telemetry row, got {len(rows)}")
    values = [part.strip() for part in rows[0].split(",")]
    if len(values) != len(fields):
        raise ValueError(
            f"GPU telemetry field count {len(values)} != {len(fields)}: {stdout}"
        )
    return dict(zip(fields, values, strict=True))


def compute_apps(contract: dict[str, Any]) -> list[dict[str, str]]:
    machine = contract["machine"]
    stdout = run_text(
        [
            "nvidia-smi",
            f"--id={machine['gpu_index']}",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=contract["timeouts_seconds"]["command"],
    )
    if not stdout:
        return []
    keys = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(keys):
            raise ValueError(f"invalid compute-app row: {line}")
        rows.append(dict(zip(keys, values, strict=True)))
    return rows


def parse_float(value: str, *, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is not finite: {value!r}")
    return parsed


def telemetry_reasons(sample: dict[str, str], contract: dict[str, Any]) -> list[str]:
    machine = contract["machine"]
    inactive = contract["telemetry"]["inactive_event_value"]
    reasons: list[str] = []
    expected_strings = {
        "index": str(machine["gpu_index"]),
        "uuid": machine["gpu_uuid"],
        "name": machine["gpu_name"],
        "pstate": machine["pstate"],
        "clocks.sm": str(machine["sm_clock_mhz"]),
        "clocks.max.sm": str(machine["sm_clock_mhz"]),
        "clocks_event_reasons.hw_slowdown": inactive,
        "clocks_event_reasons.sw_thermal_slowdown": inactive,
    }
    for key, expected in expected_strings.items():
        if sample.get(key) != expected:
            reasons.append(f"{key}={sample.get(key)!r}, expected {expected!r}")
    try:
        power = parse_float(sample["power.draw.instant"], label="power.draw.instant")
        reported_limit = parse_float(sample["power.limit"], label="power.limit")
        if not 0.0 < power <= machine["power_limit_w"]:
            reasons.append(f"power.draw.instant={power} outside (0, 1200]")
        if reported_limit != machine["power_limit_w"]:
            reasons.append(
                f"power.limit={reported_limit}, expected {machine['power_limit_w']}"
            )
        parse_float(sample["temperature.gpu"], label="temperature.gpu")
        parse_float(sample["temperature.gpu.tlimit"], label="temperature.gpu.tlimit")
    except (KeyError, ValueError) as error:
        reasons.append(str(error))
    return reasons


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("summary requires at least one value")
    return {
        "mean": statistics.fmean(values),
        "p20": percentile(values, 0.20),
        "median": percentile(values, 0.50),
        "p80": percentile(values, 0.80),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def bootstrap_lower_bound(
    values: list[float], *, draws: int, quantile: float, seed: int
) -> float:
    if len(values) < 2:
        raise ValueError("bootstrap requires at least two process values")
    rng = random.Random(seed)
    means = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    ]
    return percentile(means, quantile)


def pilot_sigma(
    process_pairs: Iterable[tuple[int, list[float]]], *, pairs_per_process: int
) -> float:
    if pairs_per_process < 2:
        raise ValueError("pilot needs at least two pairs per process")
    by_context: dict[int, list[list[float]]] = {}
    for context, pairs in process_pairs:
        if len(pairs) != pairs_per_process:
            raise ValueError(
                f"each pilot process needs exactly {pairs_per_process} A/A pairs"
            )
        by_context.setdefault(context, []).append(pairs)
    context_sigmas: list[float] = []
    for context, processes in by_context.items():
        if len(processes) != 2:
            raise ValueError(f"context {context} needs exactly two pilot processes")
        flattened = [value for process in processes for value in process]
        process_means = [statistics.fmean(process) for process in processes]
        within_process_mean_sigma = statistics.stdev(flattened) / math.sqrt(
            pairs_per_process
        )
        context_sigmas.append(
            max(within_process_mean_sigma, statistics.stdev(process_means))
        )
    if len(context_sigmas) != 2:
        raise ValueError("pilot requires exactly two contexts")
    return max(context_sigmas)


def select_process_count(sigma: float, contract: dict[str, Any]) -> int | None:
    pilot = contract["pilot"]
    if sigma <= pilot["base_sigma_max_us_per_layer"]:
        return pilot["base_processes_per_context"]
    if sigma <= pilot["expanded_sigma_max_us_per_layer"]:
        return pilot["expanded_processes_per_context"]
    return None


def projected_no_decision_probability(
    contaminated_pairs: int, *, process_count: int
) -> float:
    if not 0 <= contaminated_pairs <= 80:
        raise ValueError("pilot contamination count must be in [0, 80]")
    p = (contaminated_pairs + 0.5) / 81.0
    retained_order_probability = sum(
        math.comb(10, excluded) * p**excluded * (1.0 - p) ** (10 - excluded)
        for excluded in range(3)
    )
    return 1.0 - retained_order_probability ** (4 * process_count)
