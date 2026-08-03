#!/usr/bin/env python3
"""Capture one machine-derived four-GPU TS-C58-R health snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from evidence_common import canonical_uuid, require, sha256_file, write_json_exclusive


GPU_QUERY = (
    "index,uuid,name,compute_cap,driver_version,"
    "clocks.applications.graphics,clocks.applications.memory,"
    "ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,"
    "ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total,"
    "retired_pages.single_bit_ecc.count,retired_pages.double_bit.count,"
    "remapped_rows.pending,gpu_recovery_action,fabric.state,fabric.status"
)


def run(argv: list[str], *, check: bool = True) -> dict:
    started = time.time_ns()
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=45, check=False)
    finished = time.time_ns()
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"health command failed rc={completed.returncode}: {argv}: {completed.stderr.strip()}"
        )
    return {
        "argv": argv,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "started_unix_ns": started,
        "finished_unix_ns": finished,
    }


def integer(value: str, field: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise RuntimeError(f"non-integral GPU field {field}: {value!r}") from exc
    require(result >= 0, f"negative GPU field {field}")
    return result


def optional_integer(value: str, field: str) -> int | None:
    if value == "[N/A]":
        return None
    return integer(value, field)


def yes_no(value: str, field: str) -> bool:
    require(value in {"Yes", "No"}, f"non-boolean GPU field {field}: {value!r}")
    return value == "Yes"


def parse_gpus(raw: str) -> list[dict]:
    rows = []
    for line in raw.strip().splitlines():
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 17, "GPU health row width differs")
        row = {
            "index": integer(fields[0], "index"),
            "uuid": canonical_uuid(fields[1]),
            "name": fields[2],
            "compute_capability": fields[3],
            "driver_version": fields[4],
            "application_graphics_mhz": integer(fields[5], "application graphics"),
            "application_memory_mhz": integer(fields[6], "application memory"),
            "ecc_corrected_volatile": integer(fields[7], "corrected volatile ECC"),
            "ecc_uncorrected_volatile": integer(fields[8], "uncorrected volatile ECC"),
            "ecc_corrected_aggregate": integer(fields[9], "corrected aggregate ECC"),
            "ecc_uncorrected_aggregate": integer(fields[10], "uncorrected aggregate ECC"),
            "retired_pages_single_bit": optional_integer(fields[11], "single-bit retired pages"),
            "retired_pages_double_bit": optional_integer(fields[12], "double-bit retired pages"),
            "pending_remapped_rows": yes_no(fields[13], "pending remapped rows"),
            "recovery_action": fields[14],
            "fabric_state": fields[15],
            "fabric_status": fields[16],
        }
        require(row["name"] and row["compute_capability"] and row["driver_version"],
                "GPU identity field is empty")
        rows.append(row)
    require(len(rows) == 4, "health snapshot does not contain exactly four GPUs")
    require(sorted(row["index"] for row in rows) == [0, 1, 2, 3], "GPU indices differ")
    require(len({row["uuid"] for row in rows}) == 4, "GPU UUIDs are not unique")
    return sorted(rows, key=lambda row: row["index"])


def parse_clients(raw: str) -> list[dict]:
    result = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 4, "compute-client row width differs")
        result.append(
            {
                "gpu_uuid": canonical_uuid(fields[0]),
                "pid": integer(fields[1], "compute-client pid"),
                "process_name": fields[2],
                "used_memory_mib": integer(fields[3], "compute-client memory"),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--position", choices=("before", "after"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")

    gpu_command = run([
        "nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"
    ])
    clients_command = run([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    nvlink_command = run(["nvidia-smi", "nvlink", "--status"])
    topology_command = run(["nvidia-smi", "topo", "-m"])
    xid_command = run(["dmesg", "--color=never"], check=False)
    snapshot = {
        "schema_version": 1,
        "record_type": "ts-c58-r-gpu-health-snapshot",
        "phase_id": args.phase_id,
        "position": args.position,
        "captured_unix_ns": time.time_ns(),
        "gpus": parse_gpus(gpu_command["stdout"]),
        "compute_clients": parse_clients(clients_command["stdout"]),
        "gpu_query": gpu_command,
        "compute_client_query": clients_command,
        "nvlink_status": nvlink_command,
        "topology": topology_command,
        "xid_events": {
            "available": xid_command["return_code"] == 0,
            "command": xid_command,
            "matching_lines": [
                line for line in xid_command["stdout"].splitlines()
                if "xid" in line.lower() or "recovery" in line.lower()
            ],
        },
        "capture_sha256": sha256_file(Path(__file__).resolve()),
        "pid": os.getpid(),
    }
    write_json_exclusive(args.output, snapshot)
    print(json.dumps(snapshot, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
