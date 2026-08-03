#!/usr/bin/env python3
"""Map one complete synccheck PC to sealed SASS, PTX, and source semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from capture_disassembly import parse_sass_function_barriers
from evidence_common import load_json, require, sha256_bytes, sha256_file, write_json_exclusive


def semantic_role(inventory: dict, barrier_id: int, count: int) -> tuple[str, list[int]]:
    contract = inventory.get("inferred_contract", {})
    candidates = (
        ("tmem_pointer_handoff", contract.get("tmem_barrier")),
        ("tq4_conversion_rendezvous", contract.get("tq4_conversion_barrier")),
    )
    matches = [
        (role, value.get("intended_threads"))
        for role, value in candidates
        if isinstance(value, dict)
        and value.get("id") == barrier_id
        and value.get("count") == count
    ]
    require(len(matches) == 1, "SASS barrier does not map uniquely to the source contract")
    role, intended_threads = matches[0]
    require(
        isinstance(intended_threads, list)
        and len(intended_threads) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in intended_threads)
        and intended_threads[0] <= intended_threads[1],
        "source intended-thread interval differs",
    )
    return role, intended_threads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("m128", "dense"), required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--thread-map", type=Path, required=True)
    parser.add_argument("--synccheck-seal", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--disassembly-manifest", type=Path, required=True)
    parser.add_argument("--disassembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")

    provenance, provenance_raw = load_json(args.provenance)
    thread_map, thread_map_raw = load_json(args.thread_map)
    synccheck_seal, synccheck_seal_raw = load_json(args.synccheck_seal)
    inventory, inventory_raw = load_json(args.source_inventory)
    manifest, manifest_raw = load_json(args.disassembly_manifest)
    require(
        provenance.get("record_type") == "ts-c58-r-provenance"
        and provenance.get("status") == "pass",
        "provenance differs",
    )
    require(
        provenance.get("tool_hashes", {}).get(Path(__file__).name) == sha256_file(Path(__file__)),
        "provenance does not bind the PC mapper",
    )
    require(
        thread_map.get("record_type") == "ts-c58-r-synccheck-thread-map"
        and thread_map.get("status") == "pass"
        and thread_map.get("complete") is True,
        "synccheck thread map is incomplete",
    )
    require(
        synccheck_seal.get("record_type") == "ts-c58-r-execution-seal"
        and synccheck_seal.get("status") == "pass"
        and synccheck_seal.get("cell_id")
        == {
            "m128": "accepted-target-synccheck-map",
            "dense": "dense-control-synccheck",
        }[args.arm]
        and synccheck_seal.get("sanitizer_tool") == "synccheck"
        and synccheck_seal.get("actual_outcome") == "diagnosed_sync_error",
        "selected-arm synccheck execution seal differs",
    )
    require(
        synccheck_seal.get("source_commit") == provenance.get("source_commit")
        and synccheck_seal.get("source_identity_sha256")
        == provenance.get("source_identity_sha256")
        and synccheck_seal.get("target_uuid") == provenance.get("target_uuid")
        and synccheck_seal.get("device_index") == provenance.get("device_index")
        and synccheck_seal.get("report_sha256") == thread_map.get("report_sha256"),
        "thread map is not bound to the accepted synccheck report",
    )
    require(
        thread_map.get("execution_seal_sha256") == sha256_bytes(synccheck_seal_raw),
        "thread map does not bind the accepted synccheck execution seal",
    )
    require(
        synccheck_seal.get("runner_sha256")
        == provenance.get("tool_hashes", {}).get("run_compute_sanitizer.py")
        and synccheck_seal.get("sealer_sha256")
        == provenance.get("tool_hashes", {}).get("seal_sanitizer_result.py"),
        "synccheck execution seal tool identity differs",
    )
    require(
        inventory.get("record_type") == "ts-c58-r-barrier-source-inventory"
        and inventory.get("status") == "pass",
        "source inventory differs",
    )
    require(
        manifest.get("record_type") == "ts-c58-r-disassembly-manifest"
        and manifest.get("status") == "pass"
        and manifest.get("arm") == args.arm,
        "disassembly manifest differs",
    )
    require(
        thread_map.get("tool_sha256")
        == provenance.get("tool_hashes", {}).get("parse_synccheck_report.py")
        and inventory.get("tool_sha256")
        == provenance.get("tool_hashes", {}).get("inspect_barrier_source.py")
        and manifest.get("tool_sha256")
        == provenance.get("tool_hashes", {}).get("capture_disassembly.py"),
        "mapping input tool identity differs",
    )
    for value, label in ((inventory, "inventory"), (manifest, "manifest")):
        require(
            value.get("source_commit") == provenance.get("source_commit")
            and value.get("source_identity_sha256") == provenance.get("source_identity_sha256"),
            f"{label} source differs",
        )
    for field in (
        "image_digest", "driver_version", "gpu_name", "compute_capability",
        "target_uuid", "device_index",
    ):
        require(manifest.get(field) == provenance.get(field), f"manifest runtime differs: {field}")
    require(
        manifest.get("provenance_sha256") == sha256_bytes(provenance_raw),
        "disassembly manifest does not bind this provenance",
    )

    disassembly_raw = args.disassembly.read_bytes()
    require(
        sha256_bytes(disassembly_raw) == manifest.get("disassembly_sha256"),
        "disassembly hash differs",
    )
    disassembly_text = disassembly_raw.decode("utf-8")
    symbol = thread_map.get("kernel_symbol")
    require(
        isinstance(symbol, str) and symbol in manifest.get("cuda_functions", []),
        "synccheck kernel symbol differs from disassembly",
    )
    offset = thread_map.get("pc_offset_hex")
    require(isinstance(offset, str) and offset.startswith("0x"), "synccheck PC syntax differs")
    address = int(offset, 16)
    barriers = parse_sass_function_barriers(disassembly_text)
    at_pc = [
        row for row in barriers
        if row["function"] == symbol and int(row["address_hex"], 16) == address
    ]
    require(len(at_pc) == 1, "synccheck PC is not one exact named SASS barrier")
    mapped = at_pc[0]
    matching_sass = [
        row["address_hex"] for row in barriers
        if row["function"] == symbol
        and row["barrier_id"] == mapped["barrier_id"]
        and row["count"] == mapped["count"]
    ]
    matching_ptx = [
        row for row in manifest.get("ptx_named_barriers", [])
        if row.get("barrier_id") == mapped["barrier_id"] and row.get("count") == mapped["count"]
    ]
    require(bool(matching_ptx), "SASS barrier operands have no exact PTX match")
    role, intended_threads = semantic_role(
        inventory, mapped["barrier_id"], mapped["count"]
    )

    threads = thread_map.get("threads")
    require(
        isinstance(threads, list)
        and all(
            isinstance(row, list)
            and len(row) == 6
            and all(isinstance(item, int) and not isinstance(item, bool) for item in row)
            for row in threads
        ),
        "synccheck thread rows differ",
    )
    thread_x = [row[0] for row in threads]
    require(
        len(thread_x) == thread_map.get("error_count")
        and thread_x,
        "complete synccheck thread geometry differs",
    )
    intended_set = set(range(intended_threads[0], intended_threads[1] + 1))
    require(set(thread_x).issubset(intended_set), "reported threads escape intended participants")

    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-pc-mapping",
        "status": "pass",
        "arm": args.arm,
        "source_commit": provenance["source_commit"],
        "source_identity_sha256": provenance["source_identity_sha256"],
        "image_digest": provenance["image_digest"],
        "driver_version": provenance["driver_version"],
        "gpu_name": provenance["gpu_name"],
        "compute_capability": provenance["compute_capability"],
        "target_uuid": provenance["target_uuid"],
        "device_index": provenance["device_index"],
        "kernel_symbol": symbol,
        "pc_offset_hex": f"0x{address:x}",
        "sass_instruction": mapped["instruction"],
        "barrier_id": mapped["barrier_id"],
        "barrier_count": mapped["count"],
        "matching_sass_addresses": matching_sass,
        "matching_ptx": matching_ptx,
        "semantic_role": role,
        "intended_thread_x_interval": intended_threads,
        "reported_thread_x_min": min(thread_x),
        "reported_thread_x_max": max(thread_x),
        "reported_thread_count": len(thread_x),
        "reported_block_count": len({tuple(row[3:]) for row in threads}),
        "reported_threads_are_intended_subset": True,
        "provenance_sha256": sha256_bytes(provenance_raw),
        "thread_map_sha256": sha256_bytes(thread_map_raw),
        "target_synccheck_seal_sha256": sha256_bytes(synccheck_seal_raw),
        "synccheck_report_sha256": thread_map["report_sha256"],
        "source_inventory_sha256": sha256_bytes(inventory_raw),
        "disassembly_manifest_sha256": sha256_bytes(manifest_raw),
        "disassembly_sha256": sha256_bytes(disassembly_raw),
        "ptx_sha256": manifest["ptx_sha256"],
        "cubin_sha256": manifest["cubin_sha256"],
        "nvdisasm_sha256": manifest["nvdisasm_sha256"],
        "tool_sha256": sha256_file(Path(__file__)),
    }
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
