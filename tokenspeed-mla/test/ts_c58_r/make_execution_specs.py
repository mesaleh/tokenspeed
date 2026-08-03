#!/usr/bin/env python3
"""Generate the complete pre-execution TS-C58-R command-spec suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from evidence_common import canonical_uuid, load_json, require, sha256_bytes, sha256_file, validate_source_identity, write_json_exclusive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--target-uuid", required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--accepted-oracle", type=Path, required=True)
    parser.add_argument("--accepted-oracle-seal", type=Path, required=True)
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
    accepted_oracle = args.accepted_oracle.resolve()
    accepted_oracle_seal = args.accepted_oracle_seal.resolve()
    oracle, oracle_raw = load_json(accepted_oracle)
    oracle_seal, oracle_seal_raw = load_json(accepted_oracle_seal)
    require(
        oracle.get("record_type") == "ts-c58-r-decode-oracle"
        and oracle.get("status") == "pass"
        and oracle.get("arm") == "m128"
        and oracle.get("mode") == "unsanitized"
        and oracle.get("source_commit") == identity["source_commit"]
        and oracle.get("source_identity_sha256") == identity_hash
        and oracle.get("target_uuid") == target_uuid
        and oracle.get("device_index") == args.device_index,
        "accepted oracle differs",
    )
    require(
        oracle_seal.get("record_type") == "ts-c58-r-execution-seal"
        and oracle_seal.get("status") == "pass"
        and oracle_seal.get("cell_id") == "accepted-target-unsanitized"
        and oracle_seal.get("actual_outcome") == "clean"
        and oracle_seal.get("sanitizer_tool") is None
        and oracle_seal.get("source_commit") == identity["source_commit"]
        and oracle_seal.get("source_identity_sha256") == identity_hash
        and oracle_seal.get("target_uuid") == target_uuid
        and oracle_seal.get("device_index") == args.device_index
        and oracle_seal.get("result_sha256") == sha256_bytes(oracle_raw),
        "accepted oracle execution seal differs",
    )
    evidence_root = args.evidence_root.resolve()
    tool_root = Path(__file__).resolve().parent
    probe = tool_root / "probe_tq4_m128_sanitizer.py"
    map_probe = tool_root / "probe_m128_synccheck_map.py"
    litmus = tool_root / "barrier_litmus.py"
    require(
        oracle_seal.get("runner_sha256") == sha256_file(tool_root / "run_compute_sanitizer.py")
        and oracle_seal.get("sealer_sha256")
        == sha256_file(tool_root / "seal_sanitizer_result.py"),
        "accepted oracle execution seal tool identity differs",
    )

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

    def probe_command(cell: str, arm: str, mode: str, *, expected: str | None = None) -> list[str]:
        output = evidence_root / cell / "result.json"
        command = exact_environment(cell) + [
            sys.executable, str(probe),
            "--source-root", str(source_root),
            "--identity", str(identity_path),
            "--device-index", str(args.device_index),
            "--target-uuid", target_uuid,
            "--arm", arm,
            "--mode", mode,
        ]
        if expected is not None:
            command.extend(["--expected", str(evidence_root / expected / "result.json")])
        command.extend(["--output", str(output)])
        return command

    def litmus_command(cell: str, litmus_cell: str | None) -> list[str]:
        command = exact_environment(cell) + [
            sys.executable, str(litmus),
            "--device-index", str(args.device_index),
            "--target-uuid", target_uuid,
        ]
        if litmus_cell is None:
            command.append("--prepare-only")
        else:
            command.extend([
                "--cell", litmus_cell,
                "--expected-build", str(evidence_root / "litmus-prepare" / "result.json"),
            ])
        command.extend(["--output", str(evidence_root / cell / "result.json")])
        return command

    def map_probe_command(cell: str) -> list[str]:
        return exact_environment(cell) + [
            sys.executable, str(map_probe),
            "--source-root", str(source_root),
            "--identity", str(identity_path),
            "--device-index", str(args.device_index),
            "--target-uuid", target_uuid,
            "--expected", str(accepted_oracle),
            "--expected-seal", str(accepted_oracle_seal),
            "--output", str(evidence_root / cell / "result.json"),
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

    def add(cell: str, *, command: list[str], tool: str | None, timeout: int,
            outcomes: list[str], patterns: list[str], result: bool,
            requirements: dict | None = None,
            artifact_outcomes: list[str] | None = None) -> None:
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
            "result_path": str(evidence_root / cell / "result.json") if result else None,
            "result_outcomes": (
                artifact_outcomes if artifact_outcomes is not None
                else (outcomes if result else [])
            ),
            "result_requirements": requirements or {},
        }

    add(
        "accepted-target-unsanitized",
        command=probe_command("accepted-target-unsanitized", "m128", "unsanitized"),
        tool=None, timeout=300, outcomes=["clean"], patterns=[], result=True,
        requirements=result_requirements(
            "ts-c58-r-decode-oracle", arm="m128", mode="unsanitized",
            source_commit=identity["source_commit"], source_identity_sha256=identity_hash,
            compiler_artifacts_present=True, compiler_keep="ir,ptx,cubin",
            wrapper_sha256=sha256_file(probe),
        ),
    )
    add(
        "accepted-target-racecheck",
        command=probe_command("accepted-target-racecheck", "m128", "racecheck",
                              expected="accepted-target-unsanitized"),
        tool="racecheck", timeout=300, outcomes=["clean"], patterns=[], result=True,
        requirements=result_requirements(
            "ts-c58-r-decode-oracle", arm="m128", mode="racecheck",
            source_commit=identity["source_commit"], source_identity_sha256=identity_hash,
            hashes_match_unsanitized=True,
            compiler_artifacts_present=True, compiler_keep="ir,ptx,cubin",
            wrapper_sha256=sha256_file(probe),
        ),
    )
    add(
        "accepted-target-synccheck-map",
        command=map_probe_command("accepted-target-synccheck-map"),
        tool="synccheck", timeout=120, outcomes=["diagnosed_sync_error"],
        patterns=[
            r"Divergent thread\(s\) in block",
            r"\+0x15520",
            r"ERROR SUMMARY: [1-9][0-9]* errors",
        ],
        result=True,
        requirements=result_requirements(
            "ts-c58-r-synccheck-map-probe",
            source_commit=identity["source_commit"],
            source_identity_sha256=identity_hash,
            caught_cuda_error=True,
            caught_error_type="AcceleratorError",
            artifacts_match_unsanitized=True,
            compiler_artifacts_present=True,
            compiler_keep="ir,ptx,cubin",
            expected_sha256=sha256_bytes(oracle_raw),
            expected_seal_sha256=sha256_bytes(oracle_seal_raw),
            wrapper_sha256=sha256_file(map_probe),
        ),
    )
    add(
        "dense-control-unsanitized",
        command=probe_command("dense-control-unsanitized", "dense", "unsanitized"),
        tool=None, timeout=300, outcomes=["clean"], patterns=[], result=True,
        requirements=result_requirements(
            "ts-c58-r-decode-oracle", arm="dense", mode="unsanitized",
            source_commit=identity["source_commit"], source_identity_sha256=identity_hash,
            compiler_artifacts_present=True, compiler_keep="ir,ptx,cubin",
            wrapper_sha256=sha256_file(probe),
        ),
    )
    add(
        "dense-control-synccheck",
        command=probe_command("dense-control-synccheck", "dense", "synccheck",
                              expected="dense-control-unsanitized"),
        tool="synccheck", timeout=120,
        outcomes=["clean", "diagnosed_sync_error"],
        patterns=[r"Divergent thread\(s\) in block"], result=True,
        requirements=result_requirements(
            "ts-c58-r-decode-oracle", arm="dense", mode="synccheck",
            source_commit=identity["source_commit"], source_identity_sha256=identity_hash,
            hashes_match_unsanitized=True,
            compiler_artifacts_present=True, compiler_keep="ir,ptx,cubin",
            wrapper_sha256=sha256_file(probe),
        ),
    )
    litmus_cells = {
        "litmus-prepare": (None, None, ["clean"], 600, True, "prepare"),
        "aligned-full-unsanitized": ("aligned-full-512-512", None, ["clean"], 120, True, "run"),
        "aligned-full-synccheck": ("aligned-full-512-512", "synccheck", ["clean"], 120, True, "run"),
        "unaligned-partial-unsanitized": ("unaligned-partial-128-128", None, ["clean"], 120, True, "run"),
        "unaligned-partial-synccheck": ("unaligned-partial-128-128", "synccheck", ["clean"], 120, True, "run"),
        "aligned-partial-synccheck": (
            "aligned-partial-128-128", "synccheck", ["diagnosed_sync_error"], 120, True, "run"
        ),
        "unaligned-wrong-count-synccheck": (
            "unaligned-wrong-count-128-160", "synccheck",
            ["diagnosed_sync_error", "diagnosed_timeout"], 60, True, "run"
        ),
    }
    for cell, (litmus_cell, tool, outcomes, timeout, result, mode) in litmus_cells.items():
        patterns = [r"Divergent thread\(s\) in block"] if "diagnosed_sync_error" in outcomes else []
        requirements = None
        if result:
            requirements = result_requirements(
                "ts-c58-r-barrier-litmus", mode=mode, cell=litmus_cell,
                tool_sha256=sha256_file(litmus),
            )
        add(
            cell,
            command=litmus_command(cell, litmus_cell),
            tool=tool,
            timeout=timeout,
            outcomes=outcomes,
            patterns=patterns,
            result=result,
            requirements=requirements,
            artifact_outcomes=(
                ["diagnosed_sync_error"]
                if cell == "unaligned-wrong-count-synccheck"
                else None
            ),
        )

    spec_hashes = {}
    for cell, spec in sorted(suite.items()):
        path = args.spec_root / f"{cell}.json"
        write_json_exclusive(path, spec)
        spec_hashes[cell] = sha256_file(path)
    manifest = {
        "schema_version": 1,
        "record_type": "ts-c58-r-execution-spec-suite",
        "status": "pass",
        "source_commit": identity["source_commit"],
        "source_identity_sha256": identity_hash,
        "target_uuid": target_uuid,
        "device_index": args.device_index,
        "evidence_root": str(evidence_root),
        "spec_root": str(args.spec_root.resolve()),
        "spec_sha256s": spec_hashes,
        "accepted_oracle_sha256": sha256_bytes(oracle_raw),
        "accepted_oracle_seal_sha256": sha256_bytes(oracle_seal_raw),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
