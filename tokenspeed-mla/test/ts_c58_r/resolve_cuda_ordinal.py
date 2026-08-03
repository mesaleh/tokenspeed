#!/usr/bin/env python3
"""Resolve a physical GB200 UUID to CUDA ordinal without assuming index order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from evidence_common import canonical_uuid, load_json, require, sha256_bytes, sha256_file, validate_source_identity, write_json_exclusive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--target-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")
    root = args.source_root.resolve()
    identity, identity_raw = load_json(args.identity)
    validate_source_identity(identity, root)
    target = canonical_uuid(args.target_uuid)
    physical_raw = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    physical = []
    for line in physical_raw.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 2, "physical GPU row differs")
        physical.append({"physical_index": int(fields[0]), "uuid": canonical_uuid(fields[1])})
    require(len(physical) == 4 and len({row["uuid"] for row in physical}) == 4,
            "physical four-GPU set differs")
    visible = []
    for ordinal in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(ordinal)
        raw_uuid = str(getattr(properties, "uuid", ""))
        uuid = canonical_uuid(raw_uuid if raw_uuid.lower().startswith("gpu-") else f"GPU-{raw_uuid}")
        visible.append(
            {
                "device_index": ordinal,
                "uuid": uuid,
                "name": torch.cuda.get_device_name(ordinal),
                "compute_capability": list(torch.cuda.get_device_capability(ordinal)),
            }
        )
    require(len(visible) == 4 and len({row["uuid"] for row in visible}) == 4,
            "visible CUDA four-GPU set differs")
    require({row["uuid"] for row in visible} == {row["uuid"] for row in physical},
            "CUDA and nvidia-smi UUID sets differ")
    cuda_match = [row for row in visible if row["uuid"] == target]
    physical_match = [row for row in physical if row["uuid"] == target]
    require(len(cuda_match) == len(physical_match) == 1, "target UUID is absent or ambiguous")
    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-cuda-ordinal-resolution",
        "status": "pass",
        "source_commit": identity["source_commit"],
        "source_identity_sha256": sha256_bytes(identity_raw),
        "target_uuid": target,
        "device_index": cuda_match[0]["device_index"],
        "physical_index": physical_match[0]["physical_index"],
        "visible_devices": visible,
        "physical_devices": physical,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
