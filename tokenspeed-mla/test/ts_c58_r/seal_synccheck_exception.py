#!/usr/bin/env python3
"""Seal the narrowly reviewed TS-C58-R synccheck-tool-limitation exception."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_common import canonical_uuid, load_json, require, sha256_bytes, sha256_file, write_json_exclusive


EXPECTED_LITMUS = {
    "aligned-full-unsanitized": "clean",
    "aligned-full-synccheck": "clean",
    "unaligned-partial-unsanitized": "clean",
    "unaligned-partial-synccheck": "clean",
    "aligned-partial-synccheck": "diagnosed_sync_error",
}
COUNT_CELL = "unaligned-wrong-count-synccheck"
REQUIRED_RECOVERY_PHASES = {
    "accepted-target-synccheck",
    "dense-control-synccheck",
    "aligned-full-synccheck",
    "unaligned-partial-synccheck",
    "aligned-partial-synccheck",
    "unaligned-wrong-count-synccheck",
}


def named_paths(values: list[str], option: str) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        require(separator == "=" and name and raw_path and name not in result,
                f"invalid or duplicate {option}: {value}")
        result[name] = Path(raw_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--target-synccheck-seal", type=Path, required=True)
    parser.add_argument("--dense-control-seal", type=Path, required=True)
    parser.add_argument("--pc-mapping", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--control-proof", type=Path, required=True)
    parser.add_argument("--disassembly", type=Path, required=True)
    parser.add_argument("--litmus-seal", action="append", default=[])
    parser.add_argument("--recovery-seal", action="append", default=[])
    parser.add_argument("--reviewer-log", type=Path, required=True)
    parser.add_argument("--reviewer-session", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")
    provenance, provenance_raw = load_json(args.provenance)
    require(provenance.get("record_type") == "ts-c58-r-provenance"
            and provenance.get("status") == "pass", "provenance differs")
    target, target_raw = load_json(args.target_synccheck_seal)
    require(target.get("record_type") == "ts-c58-r-execution-seal", "target seal type differs")
    require(target.get("status") == "pass" and target.get("sanitizer_tool") == "synccheck"
            and target.get("cell_id") == "accepted-target-synccheck"
            and target.get("actual_outcome") == "diagnosed_sync_error", "target sync finding differs")
    require(target.get("source_commit") == provenance["source_commit"], "target source differs")
    require(target.get("source_identity_sha256") == provenance["source_identity_sha256"]
            and target.get("target_uuid") == canonical_uuid(provenance["target_uuid"])
            and target.get("device_index") == provenance["device_index"],
            "target execution scope differs")
    sanitizer_identity = (
        target.get("sanitizer_binary_sha256"), target.get("sanitizer_version")
    )
    require(all(isinstance(item, str) and item.strip() for item in sanitizer_identity),
            "target sanitizer identity differs")
    dense, dense_raw = load_json(args.dense_control_seal)
    require(dense.get("record_type") == "ts-c58-r-execution-seal"
            and dense.get("status") == "pass"
            and dense.get("cell_id") == "dense-control-synccheck"
            and dense.get("sanitizer_tool") == "synccheck"
            and dense.get("actual_outcome") in {"clean", "diagnosed_sync_error"},
            "dense control seal differs")
    require(dense.get("source_commit") == provenance["source_commit"], "dense source differs")
    require(dense.get("source_identity_sha256") == provenance["source_identity_sha256"]
            and dense.get("target_uuid") == canonical_uuid(provenance["target_uuid"])
            and dense.get("device_index") == provenance["device_index"],
            "dense execution scope differs")
    require((dense.get("sanitizer_binary_sha256"), dense.get("sanitizer_version"))
            == sanitizer_identity, "dense sanitizer identity differs")

    mapping, mapping_raw = load_json(args.pc_mapping)
    inventory, inventory_raw = load_json(args.source_inventory)
    proof, proof_raw = load_json(args.control_proof)
    require(mapping.get("record_type") == "ts-c58-r-pc-mapping" and mapping.get("status") == "pass",
            "PC mapping differs")
    require(inventory.get("record_type") == "ts-c58-r-barrier-source-inventory"
            and inventory.get("status") == "pass", "source inventory differs")
    require(proof.get("record_type") == "ts-c58-r-control-proof"
            and proof.get("status") == "pass" and proof.get("contract_conformant") is True,
            "contract-conformance proof differs")
    require(proof.get("barrier_semantics") in {"aligned_full_cta", "unaligned_exact_count"},
            "proved barrier semantics are not exception-eligible")
    for value, label in ((mapping, "mapping"), (inventory, "inventory"), (proof, "proof")):
        require(value.get("source_commit") == provenance["source_commit"], f"{label} source differs")
        require(value.get("source_identity_sha256") == provenance["source_identity_sha256"],
                f"{label} identity differs")
    for value, label in ((mapping, "mapping"), (proof, "proof")):
        require(value.get("image_digest") == provenance["image_digest"]
                and value.get("driver_version") == provenance["driver_version"]
                and value.get("gpu_name") == provenance["gpu_name"]
                and value.get("compute_capability") == provenance["compute_capability"]
                and value.get("target_uuid") == canonical_uuid(provenance["target_uuid"])
                and value.get("device_index") == provenance["device_index"],
                f"{label} runtime scope differs")

    litmus_paths = named_paths(args.litmus_seal, "litmus seal")
    require(set(litmus_paths) == set(EXPECTED_LITMUS) | {COUNT_CELL}, "litmus seal set differs")
    litmus_hashes = {}
    for cell, path in sorted(litmus_paths.items()):
        seal, raw = load_json(path)
        require(seal.get("record_type") == "ts-c58-r-execution-seal" and seal.get("status") == "pass",
                f"litmus seal differs: {cell}")
        require(seal.get("cell_id") == cell, f"litmus cell identity differs: {cell}")
        require(seal.get("source_commit") == provenance["source_commit"]
                and seal.get("source_identity_sha256") == provenance["source_identity_sha256"]
                and seal.get("target_uuid") == canonical_uuid(provenance["target_uuid"])
                and seal.get("device_index") == provenance["device_index"],
                f"litmus execution scope differs: {cell}")
        if seal.get("sanitizer_tool") == "synccheck":
            require((seal.get("sanitizer_binary_sha256"), seal.get("sanitizer_version"))
                    == sanitizer_identity, f"litmus sanitizer identity differs: {cell}")
        else:
            require(seal.get("sanitizer_tool") is None
                    and seal.get("sanitizer_binary_sha256") is None
                    and seal.get("sanitizer_version") is None,
                    f"unsanitized litmus identity differs: {cell}")
        if cell == COUNT_CELL:
            require(seal.get("actual_outcome") in {"diagnosed_sync_error", "diagnosed_timeout"},
                    "count litmus outcome differs")
        else:
            require(seal.get("actual_outcome") == EXPECTED_LITMUS[cell],
                    f"litmus outcome differs: {cell}")
        litmus_hashes[cell] = sha256_bytes(raw)

    recovery_paths = named_paths(args.recovery_seal, "recovery seal")
    require(set(recovery_paths) == REQUIRED_RECOVERY_PHASES, "recovery phase set differs")
    recovery_hashes = {}
    execution_hash_by_phase = {
        "accepted-target-synccheck": sha256_bytes(target_raw),
        "dense-control-synccheck": sha256_bytes(dense_raw),
    }
    execution_hash_by_phase.update(
        {
            cell: litmus_hashes[cell]
            for cell in (
                "aligned-full-synccheck",
                "unaligned-partial-synccheck",
                "aligned-partial-synccheck",
                "unaligned-wrong-count-synccheck",
            )
        }
    )
    for phase, path in sorted(recovery_paths.items()):
        seal, raw = load_json(path)
        require(seal.get("record_type") == "ts-c58-r-gpu-recovery-seal"
                and seal.get("status") == "pass" and seal.get("phase_id") == phase,
                f"recovery seal differs: {phase}")
        require(seal.get("execution_seal_sha256") == execution_hash_by_phase[phase],
                f"recovery seal does not bind execution: {phase}")
        recovery_hashes[phase] = sha256_bytes(raw)

    require(args.disassembly.is_file() and args.disassembly.stat().st_size > 0,
            "disassembly evidence is absent")
    require(args.reviewer_log.is_file() and args.reviewer_log.stat().st_size > 0,
            "reviewer log is absent")
    require(bool(args.reviewer_session.strip()), "reviewer session is empty")
    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-synccheck-exception",
        "status": "reviewed_tool_limitation",
        "source_commit": provenance["source_commit"],
        "source_identity_sha256": provenance["source_identity_sha256"],
        "image_digest": provenance["image_digest"],
        "driver_version": provenance["driver_version"],
        "gpu_name": provenance["gpu_name"],
        "compute_capability": provenance["compute_capability"],
        "tool_hashes": provenance["tool_hashes"],
        "sanitizer_binary_sha256": sanitizer_identity[0],
        "sanitizer_version": sanitizer_identity[1],
        "origin_target_uuid": canonical_uuid(provenance["target_uuid"]),
        "origin_device_index": provenance["device_index"],
        "target_synccheck_seal_sha256": sha256_bytes(target_raw),
        "dense_control_seal_sha256": sha256_bytes(dense_raw),
        "pc_mapping_sha256": sha256_bytes(mapping_raw),
        "source_inventory_sha256": sha256_bytes(inventory_raw),
        "control_proof_sha256": sha256_bytes(proof_raw),
        "disassembly_sha256": sha256_file(args.disassembly),
        "litmus_seal_sha256s": litmus_hashes,
        "recovery_seal_sha256s": recovery_hashes,
        "reviewer_log_sha256": sha256_file(args.reviewer_log),
        "reviewer_session": args.reviewer_session,
        "exception_sealer_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
