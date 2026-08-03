#!/usr/bin/env python3
"""Capture and seal PTX plus SM100 SASS for one accepted TS-C58-R oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

from evidence_common import (
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    validate_source_identity,
    write_exclusive,
    write_json_exclusive,
)


SASS_BARRIER = re.compile(
    r"^\s*/\*([0-9a-f]+)\*/\s+"
    r"(?:@[!A-Z0-9.]+\s+)?(BAR\.SYNC(?:\.[A-Z_]+)*)\s+"
    r"0x([0-9a-f]+),\s*0x([0-9a-f]+)\s*;",
    re.IGNORECASE | re.MULTILINE,
)
PTX_BARRIER = re.compile(
    r"^\s*bar\.sync\s+(0x[0-9a-f]+|[0-9]+),\s*"
    r"(0x[0-9a-f]+|[0-9]+)\s*;\s*$",
    re.IGNORECASE | re.MULTILINE,
)
FUNCTION = re.compile(r"^\s*\.global\s+(\S+)\s*$", re.MULTILINE)


def parse_number(value: str) -> int:
    return int(value, 0)


def parse_sass_barriers(text: str) -> list[dict]:
    return [
        {
            "address_hex": f"0x{address.lower()}",
            "instruction": instruction.upper(),
            "barrier_id": int(barrier_id, 16),
            "count": int(count, 16),
        }
        for address, instruction, barrier_id, count in SASS_BARRIER.findall(text)
    ]


def parse_ptx_barriers(text: str) -> list[dict]:
    lines = text.splitlines()
    result = []
    for line_number, line in enumerate(lines, start=1):
        match = PTX_BARRIER.fullmatch(line)
        if match:
            result.append(
                {
                    "line": line_number,
                    "text": line.strip(),
                    "barrier_id": parse_number(match.group(1)),
                    "count": parse_number(match.group(2)),
                }
            )
    return result


def require_artifact(oracle: dict, path: Path, suffix: str) -> dict:
    matches = [
        item for item in oracle.get("compiler_artifacts", [])
        if isinstance(item, dict) and item.get("suffix") == suffix
    ]
    require(len(matches) == 1, f"oracle {suffix} artifact is absent or ambiguous")
    expected = matches[0]
    require(path.name == expected.get("path"), f"oracle {suffix} path differs")
    require(path.stat().st_size == expected.get("size_bytes"), f"oracle {suffix} size differs")
    require(sha256_file(path) == expected.get("sha256"), f"oracle {suffix} hash differs")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--arm", choices=("m128", "dense"), required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--oracle-seal", type=Path, required=True)
    parser.add_argument("--ptx", type=Path, required=True)
    parser.add_argument("--cubin", type=Path, required=True)
    parser.add_argument("--nvdisasm", type=Path, required=True)
    parser.add_argument("--disassembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.disassembly.exists() or args.output.exists():
        parser.error("disassembly and output must be absent")

    source_root = args.source_root.resolve()
    identity, identity_raw = load_json(args.identity)
    validate_source_identity(identity, source_root)
    identity_sha256 = sha256_bytes(identity_raw)
    provenance, provenance_raw = load_json(args.provenance)
    require(
        provenance.get("record_type") == "ts-c58-r-provenance"
        and provenance.get("status") == "pass",
        "provenance differs",
    )
    require(
        provenance.get("source_commit") == identity["source_commit"]
        and provenance.get("source_identity_sha256") == identity_sha256,
        "provenance source differs",
    )
    require(
        provenance.get("tool_hashes", {}).get(Path(__file__).name) == sha256_file(Path(__file__)),
        "provenance does not bind the disassembly tool",
    )

    oracle, oracle_raw = load_json(args.oracle)
    oracle_seal, oracle_seal_raw = load_json(args.oracle_seal)
    oracle_cell = {
        "m128": "accepted-target-unsanitized",
        "dense": "dense-control-unsanitized",
    }[args.arm]
    require(
        oracle.get("record_type") == "ts-c58-r-decode-oracle"
        and oracle.get("status") == "pass"
        and oracle.get("arm") == args.arm
        and oracle.get("mode") == "unsanitized",
        "accepted unsanitized oracle differs",
    )
    require(
        oracle.get("compiler_artifacts_present") is True
        and oracle.get("compiler_keep") == "ir,ptx,cubin",
        "accepted oracle artifact contract differs",
    )
    require(
        oracle.get("source_commit") == identity["source_commit"]
        and oracle.get("source_identity_sha256") == identity_sha256
        and oracle.get("target_uuid") == provenance.get("target_uuid")
        and oracle.get("device_index") == provenance.get("device_index"),
        "oracle execution scope differs",
    )
    require(
        oracle_seal.get("record_type") == "ts-c58-r-execution-seal"
        and oracle_seal.get("status") == "pass"
        and oracle_seal.get("cell_id") == oracle_cell
        and oracle_seal.get("actual_outcome") == "clean"
        and oracle_seal.get("sanitizer_tool") is None,
        "accepted oracle execution seal differs",
    )
    require(
        oracle_seal.get("source_commit") == identity["source_commit"]
        and oracle_seal.get("source_identity_sha256") == identity_sha256
        and oracle_seal.get("target_uuid") == provenance.get("target_uuid")
        and oracle_seal.get("device_index") == provenance.get("device_index")
        and oracle_seal.get("result_sha256") == sha256_bytes(oracle_raw),
        "oracle execution seal does not bind the accepted result",
    )
    require(
        oracle_seal.get("runner_sha256")
        == provenance.get("tool_hashes", {}).get("run_compute_sanitizer.py")
        and oracle_seal.get("sealer_sha256")
        == provenance.get("tool_hashes", {}).get("seal_sanitizer_result.py"),
        "oracle execution seal tool identity differs",
    )

    ptx = args.ptx.resolve()
    cubin = args.cubin.resolve()
    require(ptx.is_file() and cubin.is_file(), "compiler artifacts are absent")
    require_artifact(oracle, ptx, ".ptx")
    require_artifact(oracle, cubin, ".cubin")
    ptx_raw = ptx.read_bytes()
    ptx_text = ptx_raw.decode("utf-8")
    ptx_barriers = parse_ptx_barriers(ptx_text)
    require(any(row["barrier_id"] == 1 and row["count"] == 288 for row in ptx_barriers),
            "PTX ID-1/count-288 barrier is absent")
    ptx_has_tq4_barrier = any(
        row["barrier_id"] == 6 and row["count"] == 128 for row in ptx_barriers
    )
    require(
        ptx_has_tq4_barrier == (args.arm == "m128"),
        "PTX ID-6/count-128 presence differs from the selected arm",
    )

    nvdisasm = args.nvdisasm.resolve()
    require(nvdisasm.is_file() and nvdisasm.is_absolute(), "nvdisasm path differs")
    version = subprocess.run(
        [str(nvdisasm), "--version"], check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    require(bool(version), "nvdisasm version is empty")
    argv = [str(nvdisasm), "--print-line-info-ptx", "--print-code", str(cubin)]
    completed = subprocess.run(argv, check=False, capture_output=True, timeout=60)
    require(completed.returncode == 0, "nvdisasm failed")
    require(completed.stderr == b"", "nvdisasm stderr is not empty")
    require(bool(completed.stdout), "nvdisasm output is empty")
    disassembly_text = completed.stdout.decode("utf-8")
    sass_barriers = parse_sass_barriers(disassembly_text)
    require(any(row["barrier_id"] == 1 and row["count"] == 288 for row in sass_barriers),
            "SASS ID-1/count-288 barrier is absent")
    sass_has_tq4_barrier = any(
        row["barrier_id"] == 6 and row["count"] == 128 for row in sass_barriers
    )
    require(
        sass_has_tq4_barrier == (args.arm == "m128"),
        "SASS ID-6/count-128 presence differs from the selected arm",
    )
    functions = sorted(set(FUNCTION.findall(disassembly_text)))
    require(len(functions) == 1, "disassembly function set is not singular")
    write_exclusive(args.disassembly, completed.stdout)

    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-disassembly-manifest",
        "status": "pass",
        "arm": args.arm,
        "source_commit": identity["source_commit"],
        "source_identity_sha256": identity_sha256,
        "image_digest": provenance["image_digest"],
        "driver_version": provenance["driver_version"],
        "gpu_name": provenance["gpu_name"],
        "compute_capability": provenance["compute_capability"],
        "target_uuid": provenance["target_uuid"],
        "device_index": provenance["device_index"],
        "provenance_sha256": sha256_bytes(provenance_raw),
        "oracle_sha256": sha256_bytes(oracle_raw),
        "oracle_seal_sha256": sha256_bytes(oracle_seal_raw),
        "ptx_path": str(ptx),
        "ptx_sha256": sha256_bytes(ptx_raw),
        "ptx_size_bytes": len(ptx_raw),
        "ptx_named_barriers": ptx_barriers,
        "cubin_path": str(cubin),
        "cubin_sha256": sha256_file(cubin),
        "cubin_size_bytes": cubin.stat().st_size,
        "cuda_functions": functions,
        "nvdisasm_path": str(nvdisasm),
        "nvdisasm_sha256": sha256_file(nvdisasm),
        "nvdisasm_version": version,
        "nvdisasm_argv": argv,
        "disassembly_path": str(args.disassembly.resolve()),
        "disassembly_sha256": sha256_bytes(completed.stdout),
        "sass_named_barriers": sass_barriers,
        "tool_sha256": sha256_file(Path(__file__)),
    }
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
