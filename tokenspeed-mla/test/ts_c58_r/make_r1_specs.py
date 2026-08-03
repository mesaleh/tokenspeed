#!/usr/bin/env python3
"""Generate the complete pre-execution TS-C58-R1 repair suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from evidence_common import (
    canonical_uuid,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    validate_execution_spec,
    validate_source_identity,
    write_json_exclusive,
)


CELLS = (
    "accepted-target-unsanitized",
    "dense-control-unsanitized",
    "m128-disassembly",
    "dense-disassembly",
    "accepted-target-racecheck",
    "accepted-target-synccheck",
    "dense-control-synccheck",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--target-uuid", required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--spec-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.spec_root.exists():
        parser.error("suite output and spec root must be absent")

    source_root = args.source_root.resolve()
    identity_path = args.identity.resolve()
    identity, identity_raw = load_json(identity_path)
    validate_source_identity(identity, source_root)
    identity_hash = sha256_bytes(identity_raw)
    target_uuid = canonical_uuid(args.target_uuid)
    require(0 <= args.device_index < 4, "CUDA ordinal differs")
    evidence_root = args.evidence_root.resolve()
    tool_root = Path(__file__).resolve().parent
    probe = tool_root / "probe_tq4_m128_sanitizer.py"
    capture = tool_root / "capture_disassembly.py"

    def exact_environment(cell: str) -> list[str]:
        dump_dir = evidence_root / "compiler-artifacts" / cell
        cache_dir = evidence_root / "compiler-cache" / cell
        return [
            "/usr/bin/env",
            f"CUTE_DSL_CACHE_DIR={cache_dir}",
            f"CUTE_DSL_DUMP_DIR={dump_dir}",
            "CUTE_DSL_KEEP=ir,ptx,cubin",
            f"CUTE_EXPERIMENTAL_DSL_CACHE_DIR={cache_dir}",
            f"CUTE_EXPERIMENTAL_DSL_DUMP_DIR={dump_dir}",
            "CUTE_EXPERIMENTAL_DSL_KEEP=ir,ptx,cubin",
            "PYTHONDONTWRITEBYTECODE=1",
            "TORCH_EXTENSIONS_DIR=/workspace/torch-extensions",
        ]

    def probe_command(cell: str, arm: str, mode: str, expected: str | None = None) -> list[str]:
        command = exact_environment(cell) + [
            sys.executable,
            str(probe),
            "--source-root", str(source_root),
            "--identity", str(identity_path),
            "--device-index", str(args.device_index),
            "--target-uuid", target_uuid,
            "--arm", arm,
            "--mode", mode,
        ]
        if expected is not None:
            command.extend([
                "--expected", str(evidence_root / expected / "result.json"),
            ])
        command.extend([
            "--output", str(evidence_root / cell / "result.json"),
        ])
        return command

    def disassembly_command(cell: str, arm: str, oracle_cell: str) -> list[str]:
        return [
            sys.executable,
            str(capture),
            "--source-root", str(source_root),
            "--identity", str(identity_path),
            "--provenance", str(evidence_root / "provenance.json"),
            "--target-uuid", target_uuid,
            "--device-index", str(args.device_index),
            "--arm", arm,
            "--oracle", str(evidence_root / oracle_cell / "result.json"),
            "--oracle-seal", str(evidence_root / "cells" / oracle_cell / "execution-seal.json"),
            "--artifact-root", str(evidence_root / "compiler-artifacts" / oracle_cell),
            "--nvdisasm", "/usr/local/cuda/bin/nvdisasm",
            "--disassembly", str(evidence_root / cell / "result.sass"),
            "--output", str(evidence_root / cell / "result.json"),
        ]

    def requirements(record_type: str, **extra: object) -> dict[str, object]:
        return {
            "status": "pass",
            "record_type": record_type,
            "source_commit": identity["source_commit"],
            "source_identity_sha256": identity_hash,
            "target_uuid": target_uuid,
            "device_index": args.device_index,
            **extra,
        }

    suite: dict[str, dict] = {}

    def add(
        cell: str,
        *,
        command: list[str],
        sanitizer_tool: str | None,
        timeout_seconds: int,
        result_requirements: dict[str, object],
    ) -> None:
        suite[cell] = {
            "schema_version": 1,
            "record_type": "ts-c58-r-execution-spec",
            "cell_id": cell,
            "source_identity_sha256": identity_hash,
            "target_uuid": target_uuid,
            "device_index": args.device_index,
            "command": command,
            "sanitizer_tool": sanitizer_tool,
            "timeout_seconds": timeout_seconds,
            "acceptable_outcomes": ["clean"],
            "required_report_regex": [],
            "result_path": str(evidence_root / cell / "result.json"),
            "result_outcomes": ["clean"],
            "result_requirements": result_requirements,
        }

    for cell, arm in (
        ("accepted-target-unsanitized", "m128"),
        ("dense-control-unsanitized", "dense"),
    ):
        add(
            cell,
            command=probe_command(cell, arm, "unsanitized"),
            sanitizer_tool=None,
            timeout_seconds=300,
            result_requirements=requirements(
                "ts-c58-r-decode-oracle",
                arm=arm,
                mode="unsanitized",
                compiler_artifacts_present=True,
                compiler_keep="ir,ptx,cubin",
                wrapper_sha256=sha256_file(probe),
            ),
        )

    for cell, arm, oracle_cell in (
        ("m128-disassembly", "m128", "accepted-target-unsanitized"),
        ("dense-disassembly", "dense", "dense-control-unsanitized"),
    ):
        add(
            cell,
            command=disassembly_command(cell, arm, oracle_cell),
            sanitizer_tool=None,
            timeout_seconds=120,
            result_requirements=requirements(
                "ts-c58-r-disassembly-manifest",
                arm=arm,
                image_digest=identity["image_digest"],
                tool_sha256=sha256_file(capture),
            ),
        )

    for cell, arm, mode, oracle_cell, tool in (
        (
            "accepted-target-racecheck",
            "m128",
            "racecheck",
            "accepted-target-unsanitized",
            "racecheck",
        ),
        (
            "accepted-target-synccheck",
            "m128",
            "synccheck",
            "accepted-target-unsanitized",
            "synccheck",
        ),
        (
            "dense-control-synccheck",
            "dense",
            "synccheck",
            "dense-control-unsanitized",
            "synccheck",
        ),
    ):
        add(
            cell,
            command=probe_command(cell, arm, mode, expected=oracle_cell),
            sanitizer_tool=tool,
            timeout_seconds=300,
            result_requirements=requirements(
                "ts-c58-r-decode-oracle",
                arm=arm,
                mode=mode,
                hashes_match_unsanitized=True,
                compiler_semantic_contract_matches_unsanitized=True,
                compiler_artifacts_present=True,
                compiler_keep="ir,ptx,cubin",
                wrapper_sha256=sha256_file(probe),
            ),
        )

    require(tuple(suite) == CELLS, "R1 cell order differs")
    spec_hashes = {}
    for cell, spec in suite.items():
        validate_execution_spec(spec, identity_hash)
        path = args.spec_root / f"{cell}.json"
        write_json_exclusive(path, spec)
        spec_hashes[cell] = sha256_file(path)

    manifest = {
        "schema_version": 1,
        "record_type": "ts-c58-r-execution-spec-suite",
        "status": "pass",
        "campaign": "r1-repair",
        "source_commit": identity["source_commit"],
        "source_identity_sha256": identity_hash,
        "target_uuid": target_uuid,
        "device_index": args.device_index,
        "evidence_root": str(evidence_root),
        "spec_root": str(args.spec_root.resolve()),
        "cell_order": list(CELLS),
        "spec_sha256s": spec_hashes,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
