#!/usr/bin/env python3
"""Select the TS-C58-R repair class from fully sealed split-site evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_common import (
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    validate_source_identity,
    write_json_exclusive,
)


ALL_CELLS = {
    "split-site-prepare",
    "split-site-disassembly",
    "aligned-single-unsanitized",
    "unaligned-single-unsanitized",
    "aligned-split-unsanitized",
    "unaligned-split-unsanitized",
    "aligned-single-synccheck",
    "unaligned-single-synccheck",
    "aligned-split-synccheck",
    "unaligned-split-synccheck",
}
RECOVERY_CELLS = {
    "aligned-single-synccheck",
    "unaligned-single-synccheck",
    "aligned-split-synccheck",
    "unaligned-split-synccheck",
}
SPLIT_CELLS = {
    "aligned-split-synccheck",
    "unaligned-split-synccheck",
}


def select_decision(aligned: str, unaligned: str) -> tuple[str, bool]:
    outcomes = {"clean", "diagnosed_sync_error"}
    require(aligned in outcomes and unaligned in outcomes, "split-site outcome differs")
    if aligned == "diagnosed_sync_error" and unaligned == "clean":
        return "select_verified_unaligned_handoff", True
    if aligned == "diagnosed_sync_error" and unaligned == "diagnosed_sync_error":
        return "select_single_pc_or_replacement_protocol", False
    if aligned == "clean" and unaligned == "clean":
        return "static_site_split_not_sufficient", False
    return "counter_hypothesis_unaligned_only_diagnoses", False


def load_by_field(paths: list[Path], field: str, record_type: str) -> tuple[dict, dict]:
    values = {}
    raw_values = {}
    for path in paths:
        value, raw = load_json(path)
        require(value.get("record_type") == record_type and value.get("status") == "pass",
                f"{record_type} differs")
        key = value.get(field)
        require(isinstance(key, str) and key not in values, f"duplicate or invalid {field}")
        values[key] = value
        raw_values[key] = raw
    return values, raw_values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--spec-suite", type=Path, required=True)
    parser.add_argument("--prepared-build", type=Path, required=True)
    parser.add_argument("--disassembly", type=Path, required=True)
    parser.add_argument("--execution-seal", type=Path, action="append", required=True)
    parser.add_argument("--recovery-seal", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")

    source_root = args.source_root.resolve()
    identity, identity_raw = load_json(args.identity)
    validate_source_identity(identity, source_root)
    identity_hash = sha256_bytes(identity_raw)
    provenance, provenance_raw = load_json(args.provenance)
    require(
        provenance.get("record_type") == "ts-c58-r-provenance"
        and provenance.get("status") == "pass"
        and provenance.get("source_commit") == identity["source_commit"]
        and provenance.get("source_identity_sha256") == identity_hash,
        "split-site provenance differs",
    )
    tools = provenance.get("tool_hashes", {})
    require(tools.get(Path(__file__).name) == sha256_file(Path(__file__).resolve()),
            "provenance does not bind split-site analyzer")
    suite, suite_raw = load_json(args.spec_suite)
    require(
        suite.get("record_type") == "ts-c58-r-execution-spec-suite"
        and suite.get("status") == "pass"
        and suite.get("campaign") == "split-site-discriminator"
        and suite.get("source_commit") == identity["source_commit"]
        and suite.get("source_identity_sha256") == identity_hash
        and suite.get("target_uuid") == provenance.get("target_uuid")
        and suite.get("device_index") == provenance.get("device_index")
        and sha256_bytes(suite_raw) == provenance.get("execution_spec_suite_sha256"),
        "split-site spec suite differs",
    )
    specs = suite.get("spec_sha256s")
    require(isinstance(specs, dict) and set(specs) == ALL_CELLS,
            "split-site spec cell set differs")

    executions, execution_raw = load_by_field(
        args.execution_seal, "cell_id", "ts-c58-r-execution-seal"
    )
    require(set(executions) == ALL_CELLS, "execution seal cell set differs")
    for cell, execution in executions.items():
        require(
            execution.get("source_commit") == identity["source_commit"]
            and execution.get("source_identity_sha256") == identity_hash
            and execution.get("target_uuid") == provenance.get("target_uuid")
            and execution.get("device_index") == provenance.get("device_index")
            and execution.get("execution_spec_sha256") == specs[cell]
            and execution.get("runner_sha256") == tools.get("run_compute_sanitizer.py")
            and execution.get("sealer_sha256") == tools.get("seal_sanitizer_result.py"),
            f"execution seal scope differs: {cell}",
        )
        if cell in SPLIT_CELLS:
            require(execution.get("actual_outcome") in {"clean", "diagnosed_sync_error"},
                    f"split execution outcome differs: {cell}")
        else:
            require(execution.get("actual_outcome") == "clean",
                    f"control execution outcome differs: {cell}")

    prepared, prepared_raw = load_json(args.prepared_build)
    require(
        prepared.get("record_type") == "ts-c58-r-split-site-litmus"
        and prepared.get("status") == "pass"
        and prepared.get("mode") == "prepare"
        and executions["split-site-prepare"].get("result_sha256")
        == sha256_bytes(prepared_raw),
        "prepared build execution binding differs",
    )
    disassembly, disassembly_raw = load_json(args.disassembly)
    require(
        disassembly.get("record_type") == "ts-c58-r-split-site-disassembly"
        and disassembly.get("status") == "pass"
        and disassembly.get("verified_layout") is True
        and disassembly.get("prepared_build_sha256") == sha256_bytes(prepared_raw)
        and executions["split-site-disassembly"].get("result_sha256")
        == sha256_bytes(disassembly_raw),
        "split-site disassembly execution binding differs",
    )

    recoveries, recovery_raw = load_by_field(
        args.recovery_seal, "phase_id", "ts-c58-r-gpu-recovery-seal"
    )
    require(set(recoveries) == RECOVERY_CELLS, "recovery seal cell set differs")
    for cell, recovery in recoveries.items():
        require(
            recovery.get("execution_seal_sha256") == sha256_bytes(execution_raw[cell])
            and recovery.get("sealer_sha256") == tools.get("seal_gpu_recovery.py"),
            f"recovery execution binding differs: {cell}",
        )

    aligned = executions["aligned-split-synccheck"]["actual_outcome"]
    unaligned = executions["unaligned-split-synccheck"]["actual_outcome"]
    decision, prediction_confirmed = select_decision(aligned, unaligned)
    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-split-site-analysis",
        "status": "pass",
        "decision": decision,
        "prediction_confirmed": prediction_confirmed,
        "aligned_split_outcome": aligned,
        "unaligned_split_outcome": unaligned,
        "source_commit": identity["source_commit"],
        "source_identity_sha256": identity_hash,
        "target_uuid": provenance["target_uuid"],
        "device_index": provenance["device_index"],
        "provenance_sha256": sha256_bytes(provenance_raw),
        "spec_suite_sha256": sha256_bytes(suite_raw),
        "prepared_build_sha256": sha256_bytes(prepared_raw),
        "disassembly_sha256": sha256_bytes(disassembly_raw),
        "execution_seal_sha256s": {
            cell: sha256_bytes(raw) for cell, raw in sorted(execution_raw.items())
        },
        "recovery_seal_sha256s": {
            cell: sha256_bytes(raw) for cell, raw in sorted(recovery_raw.items())
        },
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
