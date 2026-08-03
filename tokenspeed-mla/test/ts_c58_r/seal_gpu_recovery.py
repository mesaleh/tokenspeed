#!/usr/bin/env python3
"""Seal exact four-GPU recovery between two TS-C58-R health snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_common import load_json, require, sha256_bytes, sha256_file, write_json_exclusive


STRICT_EQUAL = (
    "name", "compute_capability", "driver_version",
    "application_graphics_mhz", "application_memory_mhz",
    "recovery_action", "fabric_state", "fabric_status",
)
MONOTONIC = (
    "ecc_corrected_volatile", "ecc_corrected_aggregate",
    "ecc_uncorrected_aggregate", "retired_pages_single_bit",
    "retired_pages_double_bit",
)


def by_uuid(rows: list[dict]) -> dict[str, dict]:
    require(isinstance(rows, list) and len(rows) == 4, "GPU snapshot width differs")
    result = {row.get("uuid"): row for row in rows if isinstance(row, dict)}
    require(len(result) == 4 and None not in result, "GPU snapshot UUIDs differ")
    return result


def healthy(row: dict, label: str) -> None:
    require(row.get("ecc_uncorrected_volatile") == 0, f"{label} volatile uncorrected ECC differs")
    require(row.get("pending_remapped_rows") == 0, f"{label} pending remapped rows differ")
    require(row.get("recovery_action") == "None", f"{label} recovery action differs")
    require(str(row.get("fabric_state", "")).strip().endswith("Completed"),
            f"{label} fabric state differs")
    require(str(row.get("fabric_status", "")).strip().endswith("Success"),
            f"{label} fabric status differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--execution-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")
    before, before_raw = load_json(args.before)
    after, after_raw = load_json(args.after)
    execution, execution_raw = load_json(args.execution_seal)
    require(execution.get("record_type") == "ts-c58-r-execution-seal"
            and execution.get("status") == "pass", "execution seal differs")
    for value, position in ((before, "before"), (after, "after")):
        require(value.get("schema_version") == 1, f"{position} snapshot schema differs")
        require(value.get("record_type") == "ts-c58-r-gpu-health-snapshot",
                f"{position} snapshot type differs")
        require(value.get("position") == position, f"{position} snapshot position differs")
        require(value.get("capture_sha256") == sha256_file(
            Path(__file__).resolve().with_name("capture_gpu_health.py")
        ), f"{position} capture tool differs")
        require(value.get("compute_clients") == [], f"{position} compute clients are not empty")
        require(value.get("nvlink_status", {}).get("return_code") == 0,
                f"{position} NVLink query failed")
        require(value.get("topology", {}).get("return_code") == 0,
                f"{position} topology query failed")
    require(before.get("phase_id") == after.get("phase_id"), "health phase differs")
    require(before.get("phase_id") == execution.get("cell_id"), "health/execution phase differs")
    require(isinstance(before.get("captured_unix_ns"), int)
            and isinstance(after.get("captured_unix_ns"), int)
            and 0 < before["captured_unix_ns"] < after["captured_unix_ns"],
            "health snapshot timestamps differ")
    require(before["captured_unix_ns"] < execution.get("started_unix_ns", 0)
            < execution.get("finished_unix_ns", 0) < after["captured_unix_ns"],
            "execution is not enclosed by health snapshots")
    require(before["nvlink_status"]["stdout"] == after["nvlink_status"]["stdout"],
            "NVLink status changed")
    require(before["topology"]["stdout"] == after["topology"]["stdout"],
            "GPU topology changed")

    prior = by_uuid(before["gpus"])
    current = by_uuid(after["gpus"])
    require(set(prior) == set(current), "allocated GPU set changed")
    require(execution.get("target_uuid") in prior, "execution target is outside allocated GPUs")
    deltas: dict[str, dict[str, int]] = {}
    for uuid in sorted(prior):
        old = prior[uuid]
        new = current[uuid]
        healthy(old, f"before {uuid}")
        healthy(new, f"after {uuid}")
        for field in STRICT_EQUAL:
            require(old.get(field) == new.get(field), f"{uuid} strict field changed: {field}")
        gpu_deltas = {}
        for field in MONOTONIC:
            old_value = old.get(field)
            new_value = new.get(field)
            require(isinstance(old_value, int) and not isinstance(old_value, bool)
                    and isinstance(new_value, int) and not isinstance(new_value, bool),
                    f"{uuid} monotonic field is not integral: {field}")
            require(new_value >= old_value, f"{uuid} monotonic field decreased: {field}")
            gpu_deltas[field] = new_value - old_value
        deltas[uuid] = gpu_deltas

    before_xid = before.get("xid_events", {})
    after_xid = after.get("xid_events", {})
    xid_assessed = before_xid.get("available") is True and after_xid.get("available") is True
    new_xid_lines: list[str] = []
    if xid_assessed:
        old_lines = set(before_xid.get("matching_lines", []))
        new_xid_lines = [line for line in after_xid.get("matching_lines", []) if line not in old_lines]
        require(new_xid_lines == [], "new readable Xid/recovery event appeared")

    seal = {
        "schema_version": 1,
        "record_type": "ts-c58-r-gpu-recovery-seal",
        "status": "pass",
        "phase_id": before["phase_id"],
        "gpu_uuids": sorted(prior),
        "before_sha256": sha256_bytes(before_raw),
        "after_sha256": sha256_bytes(after_raw),
        "execution_seal_sha256": sha256_bytes(execution_raw),
        "monotonic_deltas": deltas,
        "xid_assessed": xid_assessed,
        "before_xid_available": before_xid.get("available") is True,
        "after_xid_available": after_xid.get("available") is True,
        "new_xid_lines": new_xid_lines,
        "capture_sha256": before["capture_sha256"],
        "sealer_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, seal)
    print(json.dumps(seal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
