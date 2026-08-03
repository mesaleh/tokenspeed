#!/usr/bin/env python3
"""Run one predeclared TS-C58-R command without a shell and record raw evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from evidence_common import (
    canonical_uuid,
    load_json,
    sha256_bytes,
    sha256_file,
    validate_execution_spec,
    validate_source_identity,
    write_exclusive,
    write_json_exclusive,
)


def compute_clients(target_uuid: str) -> list[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    prefix = canonical_uuid(target_uuid).lower() + ","
    return [line.strip() for line in completed.stdout.splitlines() if line.strip().lower().startswith(prefix)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--return-code", type=Path, required=True)
    parser.add_argument("--command-record", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.report, args.return_code, args.command_record)
    if any(path.exists() for path in outputs):
        parser.error("all output paths must be absent")

    source_root = args.source_root.resolve()
    identity, identity_raw = load_json(args.identity)
    validate_source_identity(identity, source_root)
    spec, spec_raw = load_json(args.spec)
    validate_execution_spec(spec, sha256_bytes(identity_raw))
    if spec["result_path"] is not None and Path(spec["result_path"]).exists():
        parser.error("predeclared result path must be absent")
    command = list(spec["command"])
    if command.count("--output") == 1:
        command_output = Path(command[command.index("--output") + 1])
        if command_output.exists():
            parser.error("command output path must be absent")
    tool = spec["sanitizer_tool"]
    sanitizer_path = None
    sanitizer_sha256 = None
    sanitizer_version = None
    if tool is None:
        argv = command
    else:
        sanitizer = shutil.which("compute-sanitizer")
        if sanitizer is None:
            raise RuntimeError("compute-sanitizer is unavailable")
        sanitizer_path = str(Path(sanitizer).resolve())
        sanitizer_sha256 = sha256_file(Path(sanitizer_path))
        version = subprocess.run(
            [sanitizer_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        sanitizer_version = (version.stdout + version.stderr).strip()
        if not sanitizer_version:
            raise RuntimeError("compute-sanitizer version output is empty")
        argv = [
            sanitizer_path,
            "--tool", tool,
            "--error-exitcode", "99",
            "--print-limit", "10000",
            "--target-processes", "all",
            *command,
        ]

    started_unix_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    timed_out = False
    killed_process_group = False
    process = subprocess.Popen(
        argv,
        cwd=source_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    try:
        report_raw, _ = process.communicate(timeout=spec["timeout_seconds"])
        return_code = int(process.returncode)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
            killed_process_group = True
        except ProcessLookupError:
            pass
        report_raw, _ = process.communicate(timeout=30)
        return_code = 124
    finished_monotonic_ns = time.monotonic_ns()
    finished_unix_ns = time.time_ns()

    quiescence_error = None
    clients: list[str] = []
    try:
        deadline = time.monotonic() + 30
        clients = compute_clients(spec["target_uuid"])
        while clients and time.monotonic() < deadline:
            time.sleep(1)
            clients = compute_clients(spec["target_uuid"])
        target_quiescent = not clients
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        target_quiescent = False
        quiescence_error = f"{type(exc).__name__}: {exc}"

    write_exclusive(args.report, report_raw)
    write_exclusive(args.return_code, f"{return_code}\n".encode())
    record = {
        "schema_version": 1,
        "record_type": "ts-c58-r-command-record",
        "cell_id": spec["cell_id"],
        "source_commit": identity["source_commit"],
        "source_identity_path": str(args.identity.resolve()),
        "source_identity_sha256": sha256_bytes(identity_raw),
        "execution_spec_path": str(args.spec.resolve()),
        "execution_spec_sha256": sha256_bytes(spec_raw),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "argv": argv,
        "command": command,
        "sanitizer_tool": tool,
        "sanitizer_binary_path": sanitizer_path,
        "sanitizer_binary_sha256": sanitizer_sha256,
        "sanitizer_version": sanitizer_version,
        "target_uuid": canonical_uuid(spec["target_uuid"]),
        "device_index": spec["device_index"],
        "timeout_seconds": spec["timeout_seconds"],
        "timed_out": timed_out,
        "killed_process_group": killed_process_group,
        "process_group_id": process.pid,
        "return_code": return_code,
        "target_quiescent": target_quiescent,
        "remaining_target_compute_clients": clients,
        "quiescence_error": quiescence_error,
        "report_path": str(args.report.resolve()),
        "report_sha256": sha256_bytes(report_raw),
        "return_code_path": str(args.return_code.resolve()),
        "started_unix_ns": started_unix_ns,
        "finished_unix_ns": finished_unix_ns,
        "started_monotonic_ns": started_monotonic_ns,
        "finished_monotonic_ns": finished_monotonic_ns,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json_exclusive(args.command_record, record)
    print(json.dumps(record, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
