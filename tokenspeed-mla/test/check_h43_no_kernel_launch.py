#!/usr/bin/env python3
"""Require an NCU-wrapped source-only prebuild to launch no CUDA kernel."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from h43_codebook_ab_common import sha256_file


def validate_ncu_no_kernel_launch(text: str) -> dict[str, int]:
    if not text.strip():
        raise RuntimeError("NCU launch proof is empty")
    if re.search(r"\b(?:ERROR|ERR_[A-Z0-9_]+)\b", text):
        raise RuntimeError("NCU launch proof contains a profiler error")
    lines = text.splitlines()
    connected = [
        line for line in lines if re.search(r"==PROF==\s+Connected to process\b", line)
    ]
    disconnected = [
        line
        for line in lines
        if re.search(r"==PROF==\s+Disconnected from process\b", line)
    ]
    no_kernel = [line for line in lines if "No kernels were profiled" in line]
    if not connected or not disconnected or not no_kernel:
        raise RuntimeError(
            "NCU launch proof lacks connected/disconnected profiler banners or the "
            "explicit no-kernels result"
        )
    unsafe_warnings = [
        line
        for line in lines
        if "==WARNING==" in line and "No kernels were profiled" not in line
    ]
    if unsafe_warnings:
        raise RuntimeError(
            f"NCU launch proof contains an unrelated profiler warning: {unsafe_warnings[:3]}"
        )
    kernel_rows = [
        line
        for line in lines
        if re.match(r'^"?\d+"?,', line) and "Metric Name" not in line
    ]
    if kernel_rows:
        raise RuntimeError(
            f"source-only prebuild unexpectedly launched a kernel: {kernel_rows[:3]}"
        )
    return {
        "connected_banners": len(connected),
        "disconnected_banners": len(disconnected),
        "explicit_no_kernel_results": len(no_kernel),
        "kernel_launch_rows": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu-log", type=Path, required=True)
    args = parser.parse_args()
    path = args.ncu_log.resolve()
    text = path.read_text(encoding="utf-8", errors="replace")
    proof = validate_ncu_no_kernel_launch(text)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "ncu_log_sha256": sha256_file(path),
                **proof,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
