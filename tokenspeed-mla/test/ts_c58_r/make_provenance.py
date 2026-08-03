#!/usr/bin/env python3
"""Seal exact source, image, pod, device, and TS-C58-R tool provenance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from evidence_common import canonical_uuid, load_json, require, sha256_bytes, sha256_file, validate_source_identity, write_json_exclusive


TOOL_FILES = (
    "evidence_common.py",
    "probe_tq4_m128_sanitizer.py",
    "probe_m128_synccheck_map.py",
    "probe_dense_synccheck_control.py",
    "run_compute_sanitizer.py",
    "seal_sanitizer_result.py",
    "capture_gpu_health.py",
    "capture_pod_record.py",
    "seal_gpu_recovery.py",
    "seal_synccheck_exception.py",
    "inspect_barrier_source.py",
    "analyze_r1_results.py",
    "analyze_split_site_results.py",
    "barrier_litmus.py",
    "barrier_split_site_litmus.py",
    "capture_split_site_disassembly.py",
    "capture_disassembly.py",
    "map_barrier_pc.py",
    "test_mapping_tools.py",
    "test_split_site_tools.py",
    "make_provenance.py",
    "parse_synccheck_report.py",
    "make_execution_specs.py",
    "make_r1_specs.py",
    "make_split_site_specs.py",
    "resolve_cuda_ordinal.py",
    "analyze_b1965.py",
)


def required_text(path: Path, field: str) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"cannot read {field}: {exc}") from exc
    require(bool(value), f"{field} is empty")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--health-snapshot", type=Path, required=True)
    parser.add_argument("--ordinal-resolution", type=Path, required=True)
    parser.add_argument("--spec-suite", type=Path, required=True)
    parser.add_argument("--pod-record", type=Path, required=True)
    parser.add_argument("--container-name", default="experiment")
    parser.add_argument("--pod-name-file", type=Path, required=True)
    parser.add_argument("--pod-uid-file", type=Path, required=True)
    parser.add_argument("--namespace", default="workload")
    parser.add_argument("--target-uuid", required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")
    require(args.namespace == "workload", "namespace differs")
    source_root = args.source_root.resolve()
    identity, identity_raw = load_json(args.identity)
    validate_source_identity(identity, source_root)
    actual_image = os.environ.get("TS_C58_R_IMAGE_DIGEST")
    require(actual_image == identity["image_digest"], "runtime image digest assertion differs")
    health, health_raw = load_json(args.health_snapshot)
    require(health.get("record_type") == "ts-c58-r-gpu-health-snapshot", "health snapshot differs")
    require(health.get("position") == "before", "provenance health snapshot is not pre-phase")
    target_uuid = canonical_uuid(args.target_uuid)
    resolution, resolution_raw = load_json(args.ordinal_resolution)
    require(resolution.get("record_type") == "ts-c58-r-cuda-ordinal-resolution"
            and resolution.get("status") == "pass", "CUDA ordinal resolution differs")
    require(resolution.get("source_commit") == identity["source_commit"]
            and resolution.get("source_identity_sha256") == sha256_bytes(identity_raw),
            "CUDA ordinal source identity differs")
    require(resolution.get("target_uuid") == target_uuid
            and resolution.get("device_index") == args.device_index,
            "CUDA target resolution differs")
    matches = [row for row in health.get("gpus", []) if row.get("uuid") == target_uuid]
    require(len(matches) == 1, "target GPU is absent or ambiguous in health snapshot")
    target = matches[0]
    require(target.get("index") == resolution.get("physical_index"),
            "target UUID/physical-index mapping differs")
    suite, suite_raw = load_json(args.spec_suite)
    require(suite.get("record_type") == "ts-c58-r-execution-spec-suite"
            and suite.get("status") == "pass", "execution spec suite differs")
    require(suite.get("source_commit") == identity["source_commit"]
            and suite.get("source_identity_sha256") == sha256_bytes(identity_raw)
            and suite.get("target_uuid") == target_uuid
            and suite.get("device_index") == args.device_index,
            "execution spec suite identity differs")
    expected_generator = {
        None: "make_execution_specs.py",
        "r1-repair": "make_r1_specs.py",
        "split-site-discriminator": "make_split_site_specs.py",
    }.get(suite.get("campaign"))
    require(expected_generator is not None, "execution spec campaign differs")
    require(
        suite.get("generator_sha256")
        == sha256_file(Path(__file__).resolve().with_name(expected_generator)),
        "execution spec generator differs",
    )
    pod, pod_raw = load_json(args.pod_record)
    pod_name = required_text(args.pod_name_file, "pod name")
    pod_uid = required_text(args.pod_uid_file, "pod UID")
    node_name = os.environ.get("NODE_NAME", "").strip()
    require(bool(node_name), "Downward API node name is empty")
    metadata = pod.get("metadata", {})
    require(metadata.get("name") == pod_name and metadata.get("uid") == pod_uid
            and metadata.get("namespace") == args.namespace, "pod record identity differs")
    require(pod.get("spec", {}).get("nodeName") == node_name, "pod record node differs")
    containers = pod.get("spec", {}).get("containers", [])
    container = [item for item in containers if item.get("name") == args.container_name]
    require(len(container) == 1 and container[0].get("image") == identity["image_digest"],
            "pod container image differs")
    statuses = pod.get("status", {}).get("containerStatuses", [])
    status = [item for item in statuses if item.get("name") == args.container_name]
    digest = identity["image_digest"].rsplit("@", 1)[1]
    require(len(status) == 1 and digest in str(status[0].get("imageID", "")),
            "runtime container image ID differs")
    tool_root = Path(__file__).resolve().parent
    tool_hashes = {}
    for name in TOOL_FILES:
        path = tool_root / name
        require(path.is_file(), f"required tool is absent: {name}")
        tool_hashes[name] = sha256_file(path)
    checkout_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, check=True,
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-provenance",
        "status": "pass",
        "source_commit": identity["source_commit"],
        "checkout_head": checkout_head,
        "source_identity_sha256": sha256_bytes(identity_raw),
        "source_hashes": identity["source_hashes"],
        "image_digest": identity["image_digest"],
        "namespace": args.namespace,
        "pod_name": pod_name,
        "pod_uid": pod_uid,
        "node_name": node_name,
        "target_uuid": target_uuid,
        "device_index": args.device_index,
        "driver_version": target["driver_version"],
        "gpu_name": target["name"],
        "compute_capability": target["compute_capability"],
        "health_snapshot_sha256": sha256_bytes(health_raw),
        "ordinal_resolution_sha256": sha256_bytes(resolution_raw),
        "execution_spec_suite_sha256": sha256_bytes(suite_raw),
        "pod_record_sha256": sha256_bytes(pod_raw),
        "tool_hashes": tool_hashes,
        "provenance_tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
