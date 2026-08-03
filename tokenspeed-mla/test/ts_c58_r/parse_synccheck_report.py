#!/usr/bin/env python3
"""Parse exact barrier PCs and thread geometry from one synccheck report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from evidence_common import load_json, require, sha256_bytes, sha256_file, write_json_exclusive


LOCATION = re.compile(r"^=========     at (.+)\+0x([0-9a-f]+)$", re.MULTILINE)
THREAD = re.compile(
    r"^=========     by thread \(([0-9]+),([0-9]+),([0-9]+)\) in block \(([0-9]+),([0-9]+),([0-9]+)\)$",
    re.MULTILINE,
)
ERRORS = re.compile(r"^========= ERROR SUMMARY: ([0-9]+) errors?$", re.MULTILINE)
SUPPRESSED = re.compile(r"^========= ERROR SUMMARY: ([0-9]+) errors were not printed\.", re.MULTILINE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--expected-errors", type=int)
    group.add_argument("--execution-seal", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")
    raw = args.report.read_bytes()
    text = raw.decode("utf-8")
    locations = LOCATION.findall(text)
    threads = [tuple(int(item) for item in values) for values in THREAD.findall(text)]
    errors = [int(value) for value in ERRORS.findall(text)]
    suppressed = [int(value) for value in SUPPRESSED.findall(text)]
    execution_seal_sha256 = None
    expected_errors = args.expected_errors
    if args.execution_seal is not None:
        execution, execution_raw = load_json(args.execution_seal)
        require(
            execution.get("record_type") == "ts-c58-r-execution-seal"
            and execution.get("status") == "pass"
            and execution.get("cell_id")
            in {"accepted-target-synccheck-map", "dense-control-synccheck"}
            and execution.get("sanitizer_tool") == "synccheck"
            and execution.get("actual_outcome") == "diagnosed_sync_error"
            and execution.get("report_sha256") == sha256_bytes(raw),
            "mapping execution seal differs",
        )
        sealed_errors = execution.get("tool_summary", {}).get("error_summaries")
        require(isinstance(sealed_errors, list) and len(sealed_errors) == 1,
                "sealed synccheck error summary differs")
        expected_errors = sealed_errors[0]
        execution_seal_sha256 = sha256_bytes(execution_raw)
    require(isinstance(expected_errors, int) and not isinstance(expected_errors, bool)
            and expected_errors > 0, "expected error total differs")
    require(errors == [expected_errors], "synccheck error total differs")
    require(len(locations) == len(threads), "synccheck location/thread cardinality differs")
    require(len(set(locations)) == 1, "synccheck reports multiple barrier PCs")
    require(len(set(threads)) == len(threads), "synccheck repeats thread records")
    suppressed_count = suppressed[0] if suppressed else 0
    require(not suppressed or len(suppressed) == 1, "suppressed-error summary differs")
    require(len(threads) + suppressed_count == expected_errors,
            "printed plus suppressed error count differs")
    if args.require_complete:
        require(suppressed_count == 0 and len(threads) == expected_errors,
                "synccheck thread list is incomplete")
    symbol, offset = locations[0]
    thread_x = sorted(value[0] for value in threads)
    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-synccheck-thread-map",
        "status": "pass",
        "report_sha256": sha256_bytes(raw),
        "kernel_symbol": symbol,
        "pc_offset_hex": f"0x{offset}",
        "error_count": expected_errors,
        "printed_count": len(threads),
        "suppressed_count": suppressed_count,
        "complete": suppressed_count == 0,
        "threads": [list(value) for value in sorted(threads)],
        "thread_x_min": min(thread_x),
        "thread_x_max": max(thread_x),
        "thread_x_contiguous": thread_x == list(range(min(thread_x), max(thread_x) + 1)),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "execution_seal_sha256": execution_seal_sha256,
    }
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
