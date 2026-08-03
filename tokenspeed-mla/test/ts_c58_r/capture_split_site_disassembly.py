#!/usr/bin/env python3
"""Prove the prepared split-site extension has the intended SM100 barrier PCs."""

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
    write_exclusive,
    write_json_exclusive,
)


FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$")
BARRIER = re.compile(
    r"^\s*/\*([0-9a-f]+)\*/\s+"
    r"(?:@[!A-Z0-9.]+\s+)?(BAR\.SYNC(?:\.[A-Z_]+)*)\s+"
    r"0x([0-9a-f]+),\s*0x([0-9a-f]+)\s*;",
    re.IGNORECASE,
)
EXPECTED_SITES = {
    "aligned_single_288": 1,
    "unaligned_single_288": 1,
    "aligned_split_288": 3,
    "unaligned_split_288": 3,
}


def parse_function_barriers(text: str) -> dict[str, list[dict]]:
    current_function = None
    result: dict[str, list[dict]] = {}
    for line in text.splitlines():
        match = FUNCTION.fullmatch(line)
        if match is not None:
            current_function = match.group(1)
            continue
        match = BARRIER.match(line)
        if match is None or current_function is None:
            continue
        address, instruction, barrier_id, count = match.groups()
        result.setdefault(current_function, []).append(
            {
                "address_hex": f"0x{address.lower()}",
                "instruction": instruction.upper(),
                "barrier_id": int(barrier_id, 16),
                "count": int(count, 16),
            }
        )
    return result


def select_kernels(functions: dict[str, list[dict]]) -> dict[str, dict]:
    selected = {}
    for semantic_name, expected_count in EXPECTED_SITES.items():
        symbol_pattern = re.compile(
            rf"(?:^|[0-9]){re.escape(semantic_name)}(?:P|$)"
        )
        matches = [
            (name, barriers)
            for name, barriers in functions.items()
            if symbol_pattern.search(name) is not None
        ]
        require(len(matches) == 1, f"kernel function is absent or ambiguous: {semantic_name}")
        symbol, barriers = matches[0]
        require(len(barriers) == expected_count,
                f"barrier-site count differs for {semantic_name}")
        require(
            all(row["barrier_id"] == 8 and row["count"] == 288 for row in barriers),
            f"barrier operands differ for {semantic_name}",
        )
        addresses = [row["address_hex"] for row in barriers]
        require(len(addresses) == len(set(addresses)),
                f"barrier PCs are not distinct for {semantic_name}")
        selected[semantic_name] = {
            "symbol": symbol,
            "expected_sites": expected_count,
            "barriers": barriers,
        }
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--target-uuid", required=True)
    parser.add_argument("--prepared-build", type=Path, required=True)
    parser.add_argument("--litmus-tool", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--sass-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sass_output.exists() or args.output.exists():
        parser.error("SASS and result outputs must be absent")

    prepared, prepared_raw = load_json(args.prepared_build)
    litmus_tool = args.litmus_tool.resolve()
    require(litmus_tool.is_file(), "split-site litmus tool is absent")
    require(
        prepared.get("record_type") == "ts-c58-r-split-site-litmus"
        and prepared.get("status") == "pass"
        and prepared.get("mode") == "prepare"
        and prepared.get("cell") is None
        and prepared.get("device_index") == args.device_index
        and prepared.get("target_uuid") == args.target_uuid
        and prepared.get("tool_sha256") == sha256_file(litmus_tool),
        "prepared split-site build differs",
    )
    extension = Path(prepared["extension_path"]).resolve()
    require(extension.is_file(), "prepared split-site extension is absent")
    require(sha256_file(extension) == prepared["extension_sha256"],
            "prepared split-site extension hash differs")
    cuobjdump = args.cuobjdump.resolve()
    require(cuobjdump.is_file() and cuobjdump.is_absolute(), "cuobjdump path differs")
    version = subprocess.run(
        [str(cuobjdump), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    version_text = (version.stdout + version.stderr).strip()
    require(bool(version_text), "cuobjdump version is empty")
    argv = [str(cuobjdump), "--dump-sass", str(extension)]
    completed = subprocess.run(argv, check=False, capture_output=True, timeout=60)
    require(completed.returncode == 0, "cuobjdump failed")
    require(completed.stderr == b"", "cuobjdump stderr is not empty")
    require(bool(completed.stdout), "cuobjdump SASS is empty")
    sass_text = completed.stdout.decode("utf-8")
    functions = parse_function_barriers(sass_text)
    selected = select_kernels(functions)
    write_exclusive(args.sass_output, completed.stdout)
    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-split-site-disassembly",
        "status": "pass",
        "verified_layout": True,
        "target_uuid": args.target_uuid,
        "device_index": args.device_index,
        "prepared_build_sha256": sha256_bytes(prepared_raw),
        "extension_path": str(extension),
        "extension_sha256": prepared["extension_sha256"],
        "litmus_tool_sha256": sha256_file(litmus_tool),
        "cuobjdump_path": str(cuobjdump),
        "cuobjdump_sha256": sha256_file(cuobjdump),
        "cuobjdump_version": version_text,
        "cuobjdump_argv": argv,
        "sass_path": str(args.sass_output.resolve()),
        "sass_sha256": sha256_bytes(completed.stdout),
        "kernels": selected,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
