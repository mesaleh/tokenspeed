#!/usr/bin/env python3
"""Generate the pre-execution TS-C58-R split-site command-spec suite."""

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
    litmus = tool_root / "barrier_split_site_litmus.py"
    disassembler = tool_root / "capture_split_site_disassembly.py"

    def exact_environment() -> list[str]:
        return [
            "/usr/bin/env",
            "PYTHONDONTWRITEBYTECODE=1",
            "TORCH_EXTENSIONS_DIR=/workspace/torch-extensions",
        ]

    def litmus_command(cell: str, litmus_cell: str | None) -> list[str]:
        command = exact_environment() + [
            sys.executable,
            str(litmus),
            "--device-index",
            str(args.device_index),
            "--target-uuid",
            target_uuid,
        ]
        if litmus_cell is None:
            command.append("--prepare-only")
        else:
            command.extend(
                [
                    "--cell",
                    litmus_cell,
                    "--expected-build",
                    str(evidence_root / "split-site-prepare" / "result.json"),
                    "--expected-disassembly",
                    str(evidence_root / "split-site-disassembly" / "result.json"),
                ]
            )
        command.extend(["--output", str(evidence_root / cell / "result.json")])
        return command

    def disassembly_command() -> list[str]:
        cell = "split-site-disassembly"
        return exact_environment() + [
            sys.executable,
            str(disassembler),
            "--device-index",
            str(args.device_index),
            "--target-uuid",
            target_uuid,
            "--prepared-build",
            str(evidence_root / "split-site-prepare" / "result.json"),
            "--litmus-tool",
            str(litmus),
            "--cuobjdump",
            "/usr/local/cuda/bin/cuobjdump",
            "--sass-output",
            str(evidence_root / cell / "sass.txt"),
            "--output",
            str(evidence_root / cell / "result.json"),
        ]

    def result_requirements(record_type: str, **extra) -> dict:
        return {
            "status": "pass",
            "record_type": record_type,
            "target_uuid": target_uuid,
            "device_index": args.device_index,
            **extra,
        }

    suite: dict[str, dict] = {}

    def add(
        cell: str,
        *,
        command: list[str],
        tool: str | None,
        timeout: int,
        outcomes: list[str],
        patterns: list[str],
        requirements: dict,
        result_outcomes: list[str] | None = None,
    ) -> None:
        suite[cell] = {
            "schema_version": 1,
            "record_type": "ts-c58-r-execution-spec",
            "cell_id": cell,
            "source_identity_sha256": identity_hash,
            "target_uuid": target_uuid,
            "device_index": args.device_index,
            "command": command,
            "sanitizer_tool": tool,
            "timeout_seconds": timeout,
            "acceptable_outcomes": outcomes,
            "required_report_regex": patterns,
            "result_path": str(evidence_root / cell / "result.json"),
            "result_outcomes": result_outcomes if result_outcomes is not None else outcomes,
            "result_requirements": requirements,
        }

    add(
        "split-site-prepare",
        command=litmus_command("split-site-prepare", None),
        tool=None,
        timeout=600,
        outcomes=["clean"],
        patterns=[],
        requirements=result_requirements(
            "ts-c58-r-split-site-litmus",
            mode="prepare",
            cell=None,
            tool_sha256=sha256_file(litmus),
        ),
    )
    add(
        "split-site-disassembly",
        command=disassembly_command(),
        tool=None,
        timeout=120,
        outcomes=["clean"],
        patterns=[],
        requirements=result_requirements(
            "ts-c58-r-split-site-disassembly",
            verified_layout=True,
            litmus_tool_sha256=sha256_file(litmus),
            tool_sha256=sha256_file(disassembler),
        ),
    )

    cells = (
        ("aligned-single-unsanitized", "aligned-single-288", None, ["clean"]),
        ("unaligned-single-unsanitized", "unaligned-single-288", None, ["clean"]),
        ("aligned-split-unsanitized", "aligned-split-128-128-32", None, ["clean"]),
        ("unaligned-split-unsanitized", "unaligned-split-128-128-32", None, ["clean"]),
        ("aligned-single-synccheck", "aligned-single-288", "synccheck", ["clean"]),
        ("unaligned-single-synccheck", "unaligned-single-288", "synccheck", ["clean"]),
        (
            "aligned-split-synccheck",
            "aligned-split-128-128-32",
            "synccheck",
            ["clean", "diagnosed_sync_error"],
        ),
        (
            "unaligned-split-synccheck",
            "unaligned-split-128-128-32",
            "synccheck",
            ["clean", "diagnosed_sync_error"],
        ),
    )
    for cell, litmus_cell, tool, outcomes in cells:
        patterns = (
            [r"Divergent thread\(s\) in block", r"ERROR SUMMARY: [1-9][0-9]* errors"]
            if "diagnosed_sync_error" in outcomes
            else []
        )
        add(
            cell,
            command=litmus_command(cell, litmus_cell),
            tool=tool,
            timeout=120,
            outcomes=outcomes,
            patterns=patterns,
            requirements=result_requirements(
                "ts-c58-r-split-site-litmus",
                mode="run",
                cell=litmus_cell,
                tool_sha256=sha256_file(litmus),
            ),
            result_outcomes=["clean"],
        )

    spec_hashes = {}
    for cell, spec in sorted(suite.items()):
        validate_execution_spec(spec, identity_hash)
        path = args.spec_root / f"{cell}.json"
        write_json_exclusive(path, spec)
        spec_hashes[cell] = sha256_file(path)
    manifest = {
        "schema_version": 1,
        "record_type": "ts-c58-r-execution-spec-suite",
        "status": "pass",
        "campaign": "split-site-discriminator",
        "source_commit": identity["source_commit"],
        "source_identity_sha256": identity_hash,
        "target_uuid": target_uuid,
        "device_index": args.device_index,
        "evidence_root": str(evidence_root),
        "spec_root": str(args.spec_root.resolve()),
        "spec_sha256s": spec_hashes,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
