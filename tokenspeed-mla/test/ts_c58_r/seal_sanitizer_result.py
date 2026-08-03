#!/usr/bin/env python3
"""Seal a TS-C58-R run only when it matches its predeclared outcome contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from evidence_common import (
    EvidenceError,
    canonical_uuid,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    validate_execution_spec,
    validate_source_identity,
    write_json_exclusive,
)


ERROR_SUMMARY = re.compile(r"^========= ERROR SUMMARY: ([0-9]+) errors?$", re.MULTILINE)
RACECHECK_SUMMARY = re.compile(
    r"^========= RACECHECK SUMMARY: ([0-9]+) hazards? displayed \(([0-9]+) errors?, ([0-9]+) warnings?\)$",
    re.MULTILINE,
)


def classify(spec: dict, record: dict, report: str, return_code: int) -> tuple[str, dict]:
    errors = [int(value) for value in ERROR_SUMMARY.findall(report)]
    races = [tuple(int(item) for item in values) for values in RACECHECK_SUMMARY.findall(report)]
    summary = {"error_summaries": errors, "racecheck_summaries": races}
    if record.get("timed_out") is True:
        require(return_code == 124 and record.get("killed_process_group") is True,
                "timeout process evidence differs")
        return "diagnosed_timeout", summary
    require(record.get("timed_out") is False, "timed_out is not boolean false")
    if return_code == 0:
        require(not any(errors), "clean run has a nonzero error summary")
        require(not any(any(item) for item in races), "clean run has a nonzero race summary")
        if spec["sanitizer_tool"] is not None:
            require("========= COMPUTE-SANITIZER" in report, "sanitizer banner is absent")
            require(errors or races, "sanitizer summary is absent")
        return "clean", summary
    if return_code == 99 and spec["sanitizer_tool"] == "synccheck":
        require(errors and any(value > 0 for value in errors), "sync error summary is absent")
        for pattern in spec["required_report_regex"]:
            require(re.search(pattern, report, re.MULTILINE) is not None,
                    f"required sync-error pattern is absent: {pattern}")
        return "diagnosed_sync_error", summary
    raise EvidenceError(f"unclassified process outcome: rc={return_code}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--return-code", type=Path, required=True)
    parser.add_argument("--command-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")

    source_root = args.source_root.resolve()
    identity, identity_raw = load_json(args.identity)
    validate_source_identity(identity, source_root)
    spec, spec_raw = load_json(args.spec)
    validate_execution_spec(spec, sha256_bytes(identity_raw))
    record, record_raw = load_json(args.command_record)
    require(record.get("schema_version") == 1, "command record schema differs")
    require(record.get("record_type") == "ts-c58-r-command-record", "command record type differs")
    runner = Path(__file__).resolve().with_name("run_compute_sanitizer.py")
    require(record.get("runner_sha256") == sha256_file(runner), "runner hash differs")
    require(record.get("cell_id") == spec["cell_id"], "command cell differs")
    require(record.get("source_commit") == identity["source_commit"], "command source differs")
    require(record.get("source_identity_sha256") == sha256_bytes(identity_raw),
            "command identity hash differs")
    require(record.get("execution_spec_sha256") == sha256_bytes(spec_raw), "command spec hash differs")
    require(record.get("command") == spec["command"], "executed command differs")
    require(record.get("sanitizer_tool") == spec["sanitizer_tool"], "executed tool differs")
    if spec["sanitizer_tool"] is None:
        require(record.get("argv") == spec["command"], "unsanitized argv differs")
        require(record.get("sanitizer_binary_path") is None
                and record.get("sanitizer_binary_sha256") is None
                and record.get("sanitizer_version") is None,
                "unsanitized binary identity differs")
    else:
        sanitizer_path = record.get("sanitizer_binary_path")
        require(isinstance(sanitizer_path, str) and Path(sanitizer_path).is_absolute(),
                "sanitizer binary path differs")
        require(record.get("sanitizer_binary_sha256") == sha256_file(Path(sanitizer_path)),
                "sanitizer binary hash differs")
        require(isinstance(record.get("sanitizer_version"), str)
                and bool(record["sanitizer_version"].strip()),
                "sanitizer version differs")
        require(record.get("argv") == [
            sanitizer_path,
            "--tool", spec["sanitizer_tool"],
            "--error-exitcode", "99",
            "--print-limit", "10000",
            "--target-processes", "all",
            *spec["command"],
        ], "sanitizer argv differs")
    require(record.get("target_uuid") == canonical_uuid(spec["target_uuid"]), "target UUID differs")
    require(record.get("device_index") == spec["device_index"], "CUDA ordinal differs")
    require(record.get("timeout_seconds") == spec["timeout_seconds"], "timeout differs")
    require(record.get("target_quiescent") is True, "target GPU is not quiescent")
    require(record.get("remaining_target_compute_clients") == [], "target clients remain")
    require(record.get("quiescence_error") is None, "target quiescence check failed")
    require(isinstance(record.get("pid"), int) and not isinstance(record.get("pid"), bool)
            and record["pid"] > 0, "runner pid differs")
    for start, finish, label in (
        (record.get("started_unix_ns"), record.get("finished_unix_ns"), "unix"),
        (record.get("started_monotonic_ns"), record.get("finished_monotonic_ns"), "monotonic"),
    ):
        require(isinstance(start, int) and not isinstance(start, bool)
                and isinstance(finish, int) and not isinstance(finish, bool)
                and 0 < start < finish, f"{label} timestamps differ")

    try:
        report_raw = args.report.read_bytes()
        report = report_raw.decode("utf-8")
        return_code_raw = args.return_code.read_bytes()
        return_code = int(return_code_raw.decode("ascii").strip())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise EvidenceError(f"invalid process evidence: {exc}") from exc
    require(record.get("report_path") == str(args.report.resolve()), "report path differs")
    require(record.get("report_sha256") == sha256_bytes(report_raw), "report hash differs")
    require(record.get("return_code_path") == str(args.return_code.resolve()), "return-code path differs")
    require(record.get("return_code") == return_code, "return code differs")

    outcome, summary = classify(spec, record, report, return_code)
    require(outcome in spec["acceptable_outcomes"],
            f"observed outcome is not predeclared: {outcome}")
    result_path = spec["result_path"]
    result_sha256 = None
    if outcome in spec["result_outcomes"]:
        result, result_raw = load_json(Path(result_path))
        for field, expected in spec["result_requirements"].items():
            require(result.get(field) == expected, f"result requirement differs: {field}")
        rendered = json.dumps(result, sort_keys=True)
        require(report.count(rendered) == 1, "report does not bind exact result stdout")
        result_sha256 = sha256_bytes(result_raw)
    else:
        if result_path is not None:
            require(not Path(result_path).exists(), "non-clean outcome unexpectedly wrote a result")
    seal = {
        "schema_version": 1,
        "record_type": "ts-c58-r-execution-seal",
        "status": "pass",
        "cell_id": spec["cell_id"],
        "actual_outcome": outcome,
        "acceptable_outcomes": spec["acceptable_outcomes"],
        "source_commit": identity["source_commit"],
        "source_identity_sha256": sha256_bytes(identity_raw),
        "execution_spec_sha256": sha256_bytes(spec_raw),
        "target_uuid": canonical_uuid(spec["target_uuid"]),
        "device_index": spec["device_index"],
        "sanitizer_tool": spec["sanitizer_tool"],
        "sanitizer_binary_path": record["sanitizer_binary_path"],
        "sanitizer_binary_sha256": record["sanitizer_binary_sha256"],
        "sanitizer_version": record["sanitizer_version"],
        "return_code": return_code,
        "timed_out": record["timed_out"],
        "tool_summary": summary,
        "report_sha256": sha256_bytes(report_raw),
        "return_code_sha256": sha256_bytes(return_code_raw),
        "command_record_sha256": sha256_bytes(record_raw),
        "result_path": result_path,
        "result_outcomes": spec["result_outcomes"],
        "result_sha256": result_sha256,
        "runner_sha256": record["runner_sha256"],
        "sealer_sha256": sha256_file(Path(__file__).resolve()),
        "started_unix_ns": record["started_unix_ns"],
        "finished_unix_ns": record["finished_unix_ns"],
    }
    write_json_exclusive(args.output, seal)
    print(json.dumps(seal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
