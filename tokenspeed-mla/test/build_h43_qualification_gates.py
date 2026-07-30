#!/usr/bin/env python3
"""Seal independently hashed H43 qualification-gate records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from h43_codebook_ab_common import (
    canonical_json_digest,
    load_contract,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--source-manifest-digest", required=True)
    parser.add_argument("--installed-mla-sha256", required=True)
    parser.add_argument("--cache-artifact-digest", required=True)
    parser.add_argument("--codebook-sha256", required=True)
    parser.add_argument("--aggregate-ecc-baseline", required=True)
    return parser.parse_args()


def load_record(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    if not isinstance(value, dict):
        raise ValueError(f"gate record is not an object: {path}")
    return value


def valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract.resolve())
    evidence_root = args.evidence_root.resolve()
    contract_digest = canonical_json_digest(contract)
    digests = (
        args.source_manifest_digest,
        args.installed_mla_sha256,
        args.cache_artifact_digest,
        args.codebook_sha256,
    )
    if any(not valid_sha256(value) for value in digests):
        raise ValueError("qualification identity contains an invalid SHA-256")
    if not args.aggregate_ecc_baseline.isdigit():
        raise ValueError("aggregate ECC baseline must be a nonnegative integer")

    gates: list[dict[str, str]] = []
    for name in contract["qualification"]["required_gates"]:
        path = (evidence_root / "gates" / f"{name}.json").resolve()
        if not path.is_relative_to(evidence_root) or path.is_symlink():
            raise ValueError(f"unsafe gate-record path: {path}")
        record = load_record(path)
        expected = {
            "schema_version": 1,
            "status": "PASS",
            "name": name,
            "contract_digest": contract_digest,
            "source_manifest_digest": args.source_manifest_digest,
            "command_exit_code": 0,
        }
        for key, expected_value in expected.items():
            if record.get(key) != expected_value:
                raise ValueError(
                    f"{name}.{key}={record.get(key)!r}, expected {expected_value!r}"
                )
        record_copy = dict(record)
        record_digest = record_copy.pop("gate_record_digest", None)
        if record_digest != canonical_json_digest(record_copy):
            raise ValueError(f"{name} gate-record digest mismatch")
        evidence = record.get("evidence", [])
        if not evidence:
            raise ValueError(f"{name} has no evidence files")
        for item in evidence:
            evidence_path = (evidence_root / item["path"]).resolve()
            if (
                not evidence_path.is_relative_to(evidence_root)
                or evidence_path.is_symlink()
            ):
                raise ValueError(f"{name} has unsafe evidence path: {evidence_path}")
            if sha256_file(evidence_path) != item.get("sha256"):
                raise ValueError(f"{name} evidence digest mismatch: {evidence_path}")
        gates.append(
            {"name": name, "status": "PASS", "evidence_sha256": sha256_file(path)}
        )

    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": contract["experiment"],
        "contract_digest": contract_digest,
        "physical_host": contract["machine"]["physical_host"],
        "device_uuid": contract["machine"]["gpu_uuid"],
        "source_manifest_digest": args.source_manifest_digest,
        "installed_mla_sha256": args.installed_mla_sha256,
        "cache_artifact_digest": args.cache_artifact_digest,
        "codebook_sha256": args.codebook_sha256,
        "aggregate_ecc_baseline": args.aggregate_ecc_baseline,
        "gates": gates,
    }
    value["qualification_gates_digest"] = canonical_json_digest(value)
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
