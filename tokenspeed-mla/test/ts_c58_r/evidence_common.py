#!/usr/bin/env python3
"""Shared fail-closed evidence helpers for the TS-C58-R experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, NoReturn


COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PHYSICAL_UUID = re.compile(
    r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
IMAGE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
PTX_NAMED_BARRIER = re.compile(
    r"^\s*(bar(?:rier)?\.sync)(\.aligned)?\s+"
    r"(0x[0-9a-f]+|[0-9]+),\s*"
    r"(0x[0-9a-f]+|[0-9]+)\s*;\s*$",
    re.IGNORECASE,
)
SASS_NAMED_BARRIER = re.compile(
    r"^\s*/\*([0-9a-f]+)\*/\s+"
    r"(?:@[!A-Z0-9.]+\s+)?(BAR\.SYNC(?:\.[A-Z_]+)*)\s+"
    r"0x([0-9a-f]+),\s*0x([0-9a-f]+)\s*;",
    re.IGNORECASE,
)
SASS_FUNCTION = re.compile(r"^\s*\.global\s+(\S+)\s*$")


class EvidenceError(RuntimeError):
    """Evidence is incomplete, ambiguous, or inconsistent."""


def fail(message: str) -> NoReturn:
    raise EvidenceError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise EvidenceError(f"cannot hash {path}: {exc}") from exc


def compiler_semantic_contract(
    artifact_root: Path, artifacts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bind stable compiler IR and named-barrier semantics, not register allocation."""
    require(artifact_root.is_absolute(), "compiler artifact root is not absolute")
    artifact_root = artifact_root.resolve()
    require(isinstance(artifacts, list) and artifacts, "compiler artifact inventory is empty")
    layout: list[dict[str, str]] = []
    stable_ir: list[dict[str, Any]] = []
    ptx_named_barriers: list[dict[str, Any]] = []
    suffix_counts: dict[str, int] = {}
    for entry in artifacts:
        require(
            isinstance(entry, dict)
            and set(entry) == {"path", "size_bytes", "sha256", "suffix"},
            "compiler artifact entry differs",
        )
        relative = entry["path"]
        suffix = entry["suffix"]
        require(
            isinstance(relative, str)
            and relative
            and not relative.startswith("/")
            and ".." not in Path(relative).parts,
            "compiler artifact path is unsafe",
        )
        require(isinstance(suffix, str) and suffix == Path(relative).suffix,
                "compiler artifact suffix differs")
        require(
            isinstance(entry["size_bytes"], int)
            and not isinstance(entry["size_bytes"], bool)
            and entry["size_bytes"] > 0
            and isinstance(entry["sha256"], str)
            and SHA256.fullmatch(entry["sha256"]) is not None,
            "compiler artifact size or hash differs",
        )
        path = (artifact_root / relative).resolve()
        try:
            path.relative_to(artifact_root)
        except ValueError as exc:
            raise EvidenceError("compiler artifact escapes its root") from exc
        require(path.is_file(), "compiler artifact is absent")
        require(path.stat().st_size == entry["size_bytes"],
                "compiler artifact size does not match inventory")
        require(sha256_file(path) == entry["sha256"],
                "compiler artifact hash does not match inventory")
        layout.append({"path": relative, "suffix": suffix})
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        if suffix == ".mlir":
            stable_ir.append(dict(entry))
        elif suffix == ".ptx":
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise EvidenceError(f"cannot read compiler PTX: {exc}") from exc
            for line in text.splitlines():
                match = PTX_NAMED_BARRIER.fullmatch(line)
                if match is None:
                    continue
                instruction = match.group(1).lower()
                aligned_suffix = match.group(2) is not None
                ptx_named_barriers.append(
                    {
                        "path": relative,
                        "ordinal": len(ptx_named_barriers),
                        "instruction": instruction + (
                            ".aligned" if aligned_suffix else ""
                        ),
                        "aligned": instruction == "bar.sync" or aligned_suffix,
                        "barrier_id": int(match.group(3), 0),
                        "count": int(match.group(4), 0),
                    }
                )
    require(suffix_counts.get(".ptx") == 1, "compiler PTX count differs")
    require(suffix_counts.get(".cubin") == 1, "compiler CUBIN count differs")
    require(bool(stable_ir), "stable compiler IR is absent")
    require(bool(ptx_named_barriers), "compiler PTX named barriers are absent")
    return {
        "artifact_layout": layout,
        "stable_ir": stable_ir,
        "ptx_named_barriers": ptx_named_barriers,
    }


def compiler_sass_contract(
    artifact_root: Path,
    artifacts: list[dict[str, Any]],
    nvdisasm: Path,
) -> dict[str, Any]:
    """Bind normalized named-barrier SASS while tolerating address/register drift."""
    compiler_semantic_contract(artifact_root, artifacts)
    artifact_root = artifact_root.resolve()
    cubins = [entry for entry in artifacts if entry.get("suffix") == ".cubin"]
    require(len(cubins) == 1, "compiler CUBIN count differs")
    cubin = (artifact_root / cubins[0]["path"]).resolve()
    try:
        cubin.relative_to(artifact_root)
    except ValueError as exc:
        raise EvidenceError("compiler CUBIN escapes its root") from exc

    nvdisasm = nvdisasm.resolve()
    require(nvdisasm.is_absolute() and nvdisasm.is_file(), "nvdisasm path differs")
    version = subprocess.run(
        [str(nvdisasm), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    require(bool(version), "nvdisasm version is empty")
    completed = subprocess.run(
        [str(nvdisasm), "--print-line-info-ptx", "--print-code", str(cubin)],
        check=False,
        capture_output=True,
        timeout=60,
    )
    require(completed.returncode == 0, "nvdisasm failed")
    require(completed.stderr == b"", "nvdisasm stderr is not empty")
    require(bool(completed.stdout), "nvdisasm output is empty")
    try:
        text = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError("nvdisasm output is not UTF-8") from exc

    current_function = None
    barriers: list[dict[str, Any]] = []
    for line in text.splitlines():
        function_match = SASS_FUNCTION.fullmatch(line)
        if function_match is not None:
            current_function = function_match.group(1)
        barrier_match = SASS_NAMED_BARRIER.fullmatch(line)
        if barrier_match is None:
            continue
        require(current_function is not None, "SASS named barrier is not function scoped")
        _, instruction, barrier_id, count = barrier_match.groups()
        barriers.append(
            {
                "ordinal": len(barriers),
                "function": current_function,
                "instruction": instruction.upper(),
                "barrier_id": int(barrier_id, 16),
                "count": int(count, 16),
            }
        )
    require(bool(barriers), "SASS named barriers are absent")
    return {
        "nvdisasm_path": str(nvdisasm),
        "nvdisasm_sha256": sha256_file(nvdisasm),
        "nvdisasm_version": version,
        "named_barriers": barriers,
    }


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> NoReturn:
    fail(f"non-finite JSON number: {token}")


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value, raw


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                fail(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    write_exclusive(path, canonical_json(value))


def canonical_uuid(value: Any) -> str:
    require(isinstance(value, str) and PHYSICAL_UUID.fullmatch(value) is not None,
            "physical GPU UUID differs")
    return value.upper().replace("GPU-", "GPU-", 1)


def positive_int(value: Any, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"{field} must be a positive integer")
    return value


def exact_keys(value: dict[str, Any], keys: set[str], field: str) -> None:
    require(set(value) == keys, f"{field} keys differ: {sorted(set(value) ^ keys)}")


def validate_source_identity(identity: dict[str, Any], source_root: Path) -> dict[str, Any]:
    exact_keys(
        identity,
        {"schema_version", "record_type", "source_commit", "source_hashes", "image_digest"},
        "source identity",
    )
    require(identity["schema_version"] == 1, "source identity schema differs")
    require(identity["record_type"] == "ts-c58-r-source-identity", "source identity type differs")
    require(isinstance(identity["source_commit"], str) and COMMIT.fullmatch(identity["source_commit"]),
            "source commit differs")
    require(isinstance(identity["image_digest"], str) and IMAGE.fullmatch(identity["image_digest"]),
            "image digest differs")
    hashes = identity["source_hashes"]
    require(isinstance(hashes, dict) and hashes, "source hash set is empty")
    for relative, expected in sorted(hashes.items()):
        require(
            isinstance(relative, str)
            and relative
            and not relative.startswith("/")
            and ".." not in Path(relative).parts,
            f"unsafe source path: {relative!r}",
        )
        require(isinstance(expected, str) and SHA256.fullmatch(expected),
                f"source hash syntax differs: {relative}")
        source = (source_root / relative).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as exc:
            raise EvidenceError(f"source escapes root: {relative}") from exc
        require(source.is_file(), f"source is absent: {relative}")
        require(sha256_file(source) == expected, f"source hash differs: {relative}")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", identity["source_commit"], "HEAD"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    require(completed.returncode == 0, "declared source commit is not an ancestor of checkout")
    completed = subprocess.run(
        [
            "git", "diff", "--quiet", identity["source_commit"], "--",
            "tokenspeed-mla/python", "tokenspeed-mla/test",
            ":(exclude)tokenspeed-mla/test/ts_c58_r",
        ],
        cwd=source_root,
        check=False,
        timeout=30,
    )
    require(completed.returncode == 0, "kernel/test tree drifts from declared source commit")
    completed = subprocess.run(
        [
            "git", "status", "--porcelain", "--untracked-files=all", "--",
            "tokenspeed-mla/python", "tokenspeed-mla/test",
        ],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    outside_tools = [
        line for line in completed.stdout.splitlines()
        if "tokenspeed-mla/test/ts_c58_r/" not in line
    ]
    require(not outside_tools, f"uncommitted kernel/test paths exist: {outside_tools}")
    return identity


def validate_execution_spec(spec: dict[str, Any], identity_sha256: str) -> dict[str, Any]:
    exact_keys(
        spec,
        {
            "schema_version", "record_type", "cell_id", "source_identity_sha256",
            "target_uuid", "device_index", "command", "sanitizer_tool",
            "timeout_seconds", "acceptable_outcomes", "required_report_regex",
            "result_path", "result_outcomes", "result_requirements",
        },
        "execution spec",
    )
    require(spec["schema_version"] == 1, "execution spec schema differs")
    require(spec["record_type"] == "ts-c58-r-execution-spec", "execution spec type differs")
    require(isinstance(spec["cell_id"], str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", spec["cell_id"]),
            "cell id differs")
    require(spec["source_identity_sha256"] == identity_sha256, "execution identity hash differs")
    canonical_uuid(spec["target_uuid"])
    require(isinstance(spec["device_index"], int) and not isinstance(spec["device_index"], bool)
            and 0 <= spec["device_index"] < 4, "CUDA ordinal differs")
    command = spec["command"]
    require(isinstance(command, list) and command and all(isinstance(x, str) and x for x in command),
            "execution command differs")
    for option, expected in (("--target-uuid", canonical_uuid(spec["target_uuid"])),
                             ("--device-index", str(spec["device_index"]))):
        require(command.count(option) == 1, f"execution command {option} count differs")
        index = command.index(option)
        require(index + 1 < len(command), f"execution command {option} value is absent")
        actual = command[index + 1]
        if option == "--target-uuid":
            actual = canonical_uuid(actual)
        require(actual == expected, f"execution command {option} value differs")
    require(command.count("--output") <= 1, "execution command output option count differs")
    if "--output" in command:
        output_index = command.index("--output")
        require(output_index + 1 < len(command) and Path(command[output_index + 1]).is_absolute(),
                "execution command output path differs")
    require(spec["sanitizer_tool"] in {None, "racecheck", "synccheck"}, "sanitizer tool differs")
    positive_int(spec["timeout_seconds"], "timeout_seconds")
    require(spec["timeout_seconds"] <= 1800, "execution timeout exceeds 30 minutes")
    outcomes = spec["acceptable_outcomes"]
    require(isinstance(outcomes, list) and outcomes and len(outcomes) == len(set(outcomes)),
            "acceptable outcomes differ")
    allowed = {"clean", "diagnosed_sync_error", "diagnosed_timeout"}
    require(set(outcomes) <= allowed, "unknown acceptable outcome")
    if len(outcomes) > 1:
        require(
            (
                spec["cell_id"] == "unaligned-wrong-count-synccheck"
                and set(outcomes) == {"diagnosed_sync_error", "diagnosed_timeout"}
            )
            or (
                spec["cell_id"] == "dense-control-synccheck"
                and set(outcomes) == {"clean", "diagnosed_sync_error"}
            )
            or (
                spec["cell_id"] in {
                    "aligned-split-synccheck",
                    "unaligned-split-synccheck",
                }
                and set(outcomes) == {"clean", "diagnosed_sync_error"}
            ),
            "multi-outcome set is not a reviewed diagnostic contract",
        )
    patterns = spec["required_report_regex"]
    require(isinstance(patterns, list) and all(isinstance(x, str) and x for x in patterns),
            "required report regex differs")
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise EvidenceError(f"invalid report regex: {pattern}: {exc}") from exc
    if "diagnosed_sync_error" in outcomes:
        require(spec["sanitizer_tool"] == "synccheck" and patterns,
                "sync-error outcome requires synccheck and report patterns")
    if "diagnosed_timeout" in outcomes:
        require(spec["timeout_seconds"] <= 120, "timeout diagnostic exceeds two minutes")
    result_path = spec["result_path"]
    result_outcomes = spec["result_outcomes"]
    requirements = spec["result_requirements"]
    require(isinstance(result_outcomes, list)
            and len(result_outcomes) == len(set(result_outcomes))
            and set(result_outcomes) <= set(outcomes),
            "result outcomes differ")
    require("diagnosed_timeout" not in result_outcomes,
            "a killed timeout cannot produce a trusted result artifact")
    if "clean" in outcomes:
        require("clean" in result_outcomes, "clean outcome must bind a result artifact")
    if result_outcomes:
        require(isinstance(result_path, str) and Path(result_path).is_absolute(),
                "result-producing execution requires an absolute result path")
        require(isinstance(requirements, dict) and requirements,
                "result-producing execution requirements are empty")
        require(command.count("--output") == 1, "result-producing output option count differs")
        output_index = command.index("--output")
        require(output_index + 1 < len(command) and command[output_index + 1] == result_path,
                "result-producing output path differs")
    else:
        require(result_path is None and requirements == {},
                "non-result execution must not predeclare a result artifact")
        require(command.count("--output") == 0, "non-result execution declares an output option")
    return spec


def finite_number(value: Any, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{field} is not finite")
    return result
