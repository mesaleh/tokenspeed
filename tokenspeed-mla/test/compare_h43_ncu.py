#!/usr/bin/env python3
"""Fail-closed comparison of reference and D1 Nsight Compute resource rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from h43_codebook_ab_common import canonical_json_digest, load_contract, sha256_file


def parse_number(value: str) -> float:
    parsed = float(value.replace(",", "").strip())
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite NCU value: {value!r}")
    return parsed


def load_ncu(path: Path, contract: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    required = contract["ncu"]["required_metrics"]
    required_columns = {"ID", "Kernel Name", *required}
    header_index = next(
        (index for index, row in enumerate(rows) if required_columns.issubset(row)),
        None,
    )
    if header_index is None:
        raise ValueError("NCU CSV has no raw metric header or profiled kernels")
    header = rows[header_index]
    if len(header) != len(set(header)):
        raise ValueError("NCU raw metric header contains duplicate columns")
    indices = {name: header.index(name) for name in required_columns}
    by_launch: dict[int, dict[str, Any]] = {}
    for row in rows[header_index + 1 :]:
        if len(row) != len(header):
            continue
        raw_id = row[indices["ID"]].strip()
        if not raw_id:
            continue
        try:
            launch_id = int(raw_id)
        except ValueError as error:
            raise ValueError(f"NCU raw launch ID is invalid: {raw_id!r}") from error
        if launch_id in by_launch:
            raise ValueError(f"NCU raw CSV repeats launch {launch_id}")
        by_launch[launch_id] = {
            "id": launch_id,
            "kernel_name": row[indices["Kernel Name"]],
            "metrics": {
                metric: parse_number(row[indices[metric]]) for metric in required
            },
        }
    launches = [by_launch[key] for key in sorted(by_launch)]
    ncu = contract["ncu"]
    kernel_order = ncu["launch_kernel_order"]
    expected_calls = ncu["expected_ring_calls"]
    expected_launches = expected_calls * len(kernel_order)
    if len(launches) != expected_launches:
        raise ValueError(
            f"NCU captured {len(launches)} launches, expected {expected_launches} "
            f"({expected_calls} calls x {len(kernel_order)} kernels)"
        )
    missing = [
        (launch["id"], sorted(set(required) - set(launch["metrics"])))
        for launch in launches
        if set(launch["metrics"]) != set(required)
    ]
    if missing:
        raise ValueError(f"NCU launch metrics are incomplete: {missing}")
    by_class: dict[str, list[dict[str, Any]]] = {marker: [] for marker in kernel_order}
    for call_index in range(expected_calls):
        call_launches = launches[
            call_index * len(kernel_order) : (call_index + 1) * len(kernel_order)
        ]
        observed_order: list[str] = []
        for launch in call_launches:
            kernel_name = launch["kernel_name"] or ""
            matches = [marker for marker in kernel_order if marker in kernel_name]
            if len(matches) != 1:
                raise ValueError(
                    f"launch {launch['id']} kernel cannot be classified exactly once: "
                    f"{kernel_name!r}"
                )
            observed_order.append(matches[0])
        if observed_order != kernel_order:
            raise ValueError(
                f"call {call_index} kernel order differs: {observed_order}, "
                f"expected {kernel_order}"
            )
        start = ncu["selected_call_start"]
        count = ncu["selected_call_count"]
        if start <= call_index < start + count:
            for marker, launch in zip(kernel_order, call_launches, strict=True):
                by_class[marker].append(launch)
    if any(len(values) != ncu["selected_call_count"] for values in by_class.values()):
        raise ValueError("NCU selected call extraction is incomplete")
    return by_class


def summarize(launches: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows = [launch["metrics"] for launch in launches]
    block_limit_names = (
        "launch__occupancy_limit_blocks",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
    )
    return {
        "launch_ids": [launch["id"] for launch in launches],
        "kernel_names": sorted({launch["kernel_name"] for launch in launches}),
        "registers_per_thread_max": max(
            row["launch__registers_per_thread"] for row in metric_rows
        ),
        "static_smem_per_block_max": max(
            row["launch__shared_mem_per_block_static"] for row in metric_rows
        ),
        "dynamic_smem_per_block_max": max(
            row["launch__shared_mem_per_block_dynamic"] for row in metric_rows
        ),
        "achieved_occupancy_min_pct": min(
            row["sm__warps_active.avg.pct_of_peak_sustained_active"]
            for row in metric_rows
        ),
        "achieved_occupancy_mean_pct": math.fsum(
            row["sm__warps_active.avg.pct_of_peak_sustained_active"]
            for row in metric_rows
        )
        / len(metric_rows),
        "theoretical_occupancy_min_pct": min(
            row["sm__maximum_warps_per_active_cycle_pct"] for row in metric_rows
        ),
        "resident_blocks_per_sm_min": min(
            min(row[name] for name in block_limit_names) for row in metric_rows
        ),
        "local_load_instructions_sum": sum(
            row["smsp__inst_executed_op_local_ld.sum"] for row in metric_rows
        ),
        "local_store_instructions_sum": sum(
            row["smsp__inst_executed_op_local_st.sum"] for row in metric_rows
        ),
    }


def compare_resource_classes(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, dict[str, bool]]:
    tolerance = contract["ncu"]["achieved_occupancy_mean_tolerance_pct_points"]
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("NCU achieved-occupancy tolerance is invalid")
    gates: dict[str, dict[str, bool]] = {}
    for name in contract["ncu"]["launch_kernel_order"]:
        reference_class = reference[name]
        candidate_class = candidate[name]
        achieved_regression = (
            reference_class["achieved_occupancy_mean_pct"]
            - candidate_class["achieved_occupancy_mean_pct"]
        )
        gates[name] = {
            "registers": candidate_class["registers_per_thread_max"]
            <= reference_class["registers_per_thread_max"],
            "static_smem": candidate_class["static_smem_per_block_max"]
            <= reference_class["static_smem_per_block_max"],
            "dynamic_smem": candidate_class["dynamic_smem_per_block_max"]
            <= reference_class["dynamic_smem_per_block_max"],
            "achieved_occupancy_mean": achieved_regression <= tolerance + 1e-12,
            "theoretical_occupancy": candidate_class["theoretical_occupancy_min_pct"]
            >= reference_class["theoretical_occupancy_min_pct"],
            "resident_blocks": candidate_class["resident_blocks_per_sm_min"]
            >= reference_class["resident_blocks_per_sm_min"]
            and candidate_class["resident_blocks_per_sm_min"] >= 1,
            "reference_no_local_spills": reference_class["local_load_instructions_sum"]
            == 0
            and reference_class["local_store_instructions_sum"] == 0,
            "candidate_no_local_spills": candidate_class["local_load_instructions_sum"]
            == 0
            and candidate_class["local_store_instructions_sum"] == 0,
        }
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract.resolve())
    reference_path = args.reference.resolve()
    candidate_path = args.candidate.resolve()
    reference = {
        name: summarize(launches)
        for name, launches in load_ncu(reference_path, contract).items()
    }
    candidate = {
        name: summarize(launches)
        for name, launches in load_ncu(candidate_path, contract).items()
    }
    gates = compare_resource_classes(reference, candidate, contract)
    value = {
        "schema_version": 1,
        "status": (
            "PASS"
            if all(all(class_gates.values()) for class_gates in gates.values())
            else "FAIL"
        ),
        "experiment": contract["experiment"],
        "contract_digest": canonical_json_digest(contract),
        "reference_csv_sha256": sha256_file(reference_path),
        "candidate_csv_sha256": sha256_file(candidate_path),
        "achieved_occupancy_mean_tolerance_pct_points": contract["ncu"][
            "achieved_occupancy_mean_tolerance_pct_points"
        ],
        "reference": reference,
        "candidate": candidate,
        "gates": gates,
    }
    value["result_digest"] = canonical_json_digest(value)
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))
    if value["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
