#!/usr/bin/env python3
"""Create one H43 gate record from already successful command evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h43_codebook_ab_common import canonical_json_digest, load_contract, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-manifest-digest", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--command", action="append", required=True)
    args = parser.parse_args()

    contract = load_contract(args.contract.resolve())
    if args.name not in contract["qualification"]["required_gates"]:
        raise ValueError(f"gate is absent from the contract: {args.name}")
    evidence_root = args.evidence_root.resolve()
    evidence = []
    for supplied in args.evidence:
        path = (evidence_root / supplied).resolve()
        if (
            not path.is_relative_to(evidence_root)
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(f"unsafe or missing gate evidence: {path}")
        evidence.append(
            {
                "path": path.relative_to(evidence_root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    value = {
        "schema_version": 1,
        "status": "PASS",
        "name": args.name,
        "contract_digest": canonical_json_digest(contract),
        "source_manifest_digest": args.source_manifest_digest,
        "command": args.command,
        "command_exit_code": 0,
        "evidence": evidence,
    }
    value["gate_record_digest"] = canonical_json_digest(value)
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
