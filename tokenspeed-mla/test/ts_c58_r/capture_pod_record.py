#!/usr/bin/env python3
"""Capture the exact TUK pod object with explicit kubeconfig/context/namespace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from evidence_common import require, write_json_exclusive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", type=Path, required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", default="workload")
    parser.add_argument("--pod", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")
    require(args.namespace == "workload", "namespace differs")
    completed = subprocess.run(
        [
            "kubectl", "--kubeconfig", str(args.kubeconfig.resolve()),
            "--context", args.context, "--namespace", args.namespace,
            "get", "pod", args.pod, "--output", "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    value = json.loads(completed.stdout)
    require(value.get("metadata", {}).get("name") == args.pod, "pod name differs")
    require(value.get("metadata", {}).get("namespace") == args.namespace, "pod namespace differs")
    require(value.get("status", {}).get("phase") == "Running", "pod is not Running")
    require(bool(value.get("metadata", {}).get("uid")), "pod UID is absent")
    require(bool(value.get("spec", {}).get("nodeName")), "pod node is absent")
    write_json_exclusive(args.output, value)
    print(json.dumps({
        "status": "pass",
        "pod": args.pod,
        "uid": value["metadata"]["uid"],
        "node": value["spec"]["nodeName"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
