#!/usr/bin/env python3
"""Validate strict zero-hazard reference and candidate H43 racecheck logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from h43_codebook_ab_common import (
    canonical_json_digest,
    load_contract,
    sha256_file,
)

SUMMARY = re.compile(
    r"^========= RACECHECK SUMMARY: (\d+) hazards displayed "
    r"\((\d+) errors, (\d+) warnings\)$"
)


def load_racecheck_log(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line]
    if lines.count("========= COMPUTE-SANITIZER") != 1:
        raise ValueError(f"{path} lacks one Compute Sanitizer header")
    summaries = [
        match
        for line in text.splitlines()
        if (match := SUMMARY.fullmatch(line)) is not None
    ]
    if len(summaries) != 1:
        raise ValueError(f"{path} lacks one exact racecheck summary")
    observed = tuple(int(value) for value in summaries[0].groups())
    expected = (
        contract["racecheck"]["required_hazards"],
        contract["racecheck"]["required_errors"],
        contract["racecheck"]["required_warnings"],
    )
    if observed != expected:
        raise ValueError(f"{path} racecheck summary {observed} != {expected}")
    if lines != ["========= COMPUTE-SANITIZER", summaries[0].group(0)]:
        raise ValueError(f"{path} contains an unexpected racecheck diagnostic")
    return {
        "sha256": sha256_file(path),
        "hazards": observed[0],
        "errors": observed[1],
        "warnings": observed[2],
    }


def load_target_log(
    path: Path, contract: dict[str, Any], expected_context: int
) -> dict[str, Any]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            records.append(json.loads(line))
    if len(records) != 1:
        raise ValueError(f"{path} lacks one exact target result")
    value = records[0]
    observed_digest = value.pop("result_digest", None)
    if observed_digest != canonical_json_digest(value):
        raise ValueError(f"{path} target result digest mismatch")
    expected = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": contract["experiment"],
        "contract_digest": canonical_json_digest(contract),
        "context": expected_context,
        "sequence": 1,
        "query_length": contract["racecheck"]["target_query_length"],
        "target_tq_calls": contract["racecheck"]["target_tq_calls"],
        "output_finite": True,
    }
    expected_fields = {*expected, "codebook_sha256", "output_sha256"}
    if set(value) != expected_fields:
        raise ValueError(
            f"{path} target fields {sorted(value)} != {sorted(expected_fields)}"
        )
    mismatches = {
        key: (value.get(key), expected_value)
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"{path} target result mismatch: {mismatches}")
    for field in ("codebook_sha256", "output_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", value.get(field, "")):
            raise ValueError(f"{path} target {field} is invalid")
    return {**value, "result_digest": observed_digest, "log_sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--reference-target", type=Path, required=True)
    parser.add_argument("--candidate-target", type=Path, required=True)
    parser.add_argument("--reference-exit-code", type=int, required=True)
    parser.add_argument("--candidate-exit-code", type=int, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract.resolve())
    if str(args.context) not in contract["contexts"]:
        raise ValueError("context is absent from the H43 contract")
    if args.reference_exit_code != 0 or args.candidate_exit_code != 0:
        raise ValueError("racecheck target exit codes must both be zero")

    reference_log = load_racecheck_log(args.reference_log, contract)
    candidate_log = load_racecheck_log(args.candidate_log, contract)
    reference_target = load_target_log(args.reference_target, contract, args.context)
    candidate_target = load_target_log(args.candidate_target, contract, args.context)
    for field in ("codebook_sha256", "output_sha256"):
        if reference_target[field] != candidate_target[field]:
            raise ValueError(f"reference/candidate target {field} mismatch")

    value = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": contract["experiment"],
        "contract_digest": canonical_json_digest(contract),
        "context": args.context,
        "reference_exit_code": args.reference_exit_code,
        "candidate_exit_code": args.candidate_exit_code,
        "reference_log": reference_log,
        "candidate_log": candidate_log,
        "reference_target": reference_target,
        "candidate_target": candidate_target,
    }
    value["result_digest"] = canonical_json_digest(value)
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
