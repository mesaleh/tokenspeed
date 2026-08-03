#!/usr/bin/env python3
"""Validate sanitizer admission for a later, separately reviewed B1965 run.

This TS-C58-R version intentionally performs admission only. It never scores
performance; the B1965 phase must extend/review its analyzer after correctness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_common import canonical_uuid, load_json, require, sha256_bytes, sha256_file, write_json_exclusive


def validate_execution_seal(path: Path, provenance: dict, tool: str) -> tuple[dict, bytes]:
    seal, raw = load_json(path)
    require(seal.get("record_type") == "ts-c58-r-execution-seal", f"{tool} seal type differs")
    require(seal.get("status") == "pass" and seal.get("actual_outcome") == "clean",
            f"{tool} seal did not cleanly pass")
    require(seal.get("sanitizer_tool") == tool, f"{tool} identity differs")
    require(seal.get("source_commit") == provenance["source_commit"], f"{tool} source differs")
    require(seal.get("source_identity_sha256") == provenance["source_identity_sha256"],
            f"{tool} source identity differs")
    require(seal.get("target_uuid") == canonical_uuid(provenance["target_uuid"]),
            f"{tool} target UUID differs")
    require(seal.get("device_index") == provenance["device_index"], f"{tool} CUDA ordinal differs")
    require(isinstance(seal.get("sanitizer_binary_sha256"), str)
            and isinstance(seal.get("sanitizer_version"), str),
            f"{tool} sanitizer identity differs")
    return seal, raw


def validate_exception(path: Path, provenance: dict) -> tuple[dict, bytes]:
    value, raw = load_json(path)
    require(value.get("record_type") == "ts-c58-r-synccheck-exception", "exception type differs")
    require(value.get("status") == "reviewed_tool_limitation", "exception status differs")
    for field in ("source_commit", "source_identity_sha256", "image_digest", "driver_version",
                  "gpu_name", "compute_capability", "tool_hashes"):
        require(value.get(field) == provenance.get(field), f"exception scope differs: {field}")
    expected_sealer = provenance["tool_hashes"].get("seal_synccheck_exception.py")
    require(value.get("exception_sealer_sha256") == expected_sealer, "exception sealer differs")
    canonical_uuid(value.get("origin_target_uuid"))
    require(isinstance(value.get("origin_device_index"), int), "exception origin ordinal differs")
    return value, raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--racecheck-seal", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synccheck-seal", type=Path)
    group.add_argument("--synccheck-exception", type=Path)
    parser.add_argument("--admission-only", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")
    provenance, provenance_raw = load_json(args.provenance)
    require(provenance.get("record_type") == "ts-c58-r-provenance", "provenance type differs")
    require(provenance.get("status") == "pass", "provenance did not pass")
    race, race_raw = validate_execution_seal(args.racecheck_seal, provenance, "racecheck")
    if args.synccheck_seal is not None:
        sync, sync_raw = validate_execution_seal(args.synccheck_seal, provenance, "synccheck")
        sync_mode = "normal_zero_error_seal"
        sync_ref = sha256_bytes(sync_raw)
        sync_sanitizer = sync
    else:
        sync, sync_raw = validate_exception(args.synccheck_exception, provenance)
        sync_mode = "reviewed_tool_limitation"
        sync_ref = sha256_bytes(sync_raw)
        sync_sanitizer = sync
    require(
        (race["sanitizer_binary_sha256"], race["sanitizer_version"])
        == (sync_sanitizer.get("sanitizer_binary_sha256"), sync_sanitizer.get("sanitizer_version")),
        "racecheck/synccheck sanitizer identity differs",
    )
    admission = {
        "schema_version": 1,
        "record_type": "ts-c58-r-b1965-admission",
        "status": "pass",
        "decision": "ADMISSION_ONLY_NO_PERFORMANCE_DECISION",
        "source_commit": provenance["source_commit"],
        "source_identity_sha256": provenance["source_identity_sha256"],
        "image_digest": provenance["image_digest"],
        "target_uuid": provenance["target_uuid"],
        "device_index": provenance["device_index"],
        "provenance_sha256": sha256_bytes(provenance_raw),
        "racecheck_seal_sha256": sha256_bytes(race_raw),
        "synccheck_admission": sync_mode,
        "synccheck_reference_sha256": sync_ref,
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, admission)
    print(json.dumps(admission, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
