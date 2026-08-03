#!/usr/bin/env python3
"""Emit the accepted M128 named-barrier source inventory before GPU mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from evidence_common import load_json, require, sha256_bytes, sha256_file, validate_source_identity, write_json_exclusive


def matching_lines(lines: list[str], pattern: str) -> list[dict]:
    regex = re.compile(pattern)
    return [
        {"line": index, "text": text.rstrip()}
        for index, text in enumerate(lines, 1)
        if regex.search(text)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")
    root = args.source_root.resolve()
    identity, identity_raw = load_json(args.identity)
    validate_source_identity(identity, root)
    source = root / "tokenspeed-mla/python/tokenspeed_mla/mla_decode_fp8.py"
    lines = source.read_text(encoding="utf-8").splitlines()
    inventory = {
        "warp_declarations": matching_lines(
            lines,
            r"self\.(compute_warp_ids|correction_warp_ids|mma_warp_id|load_tma_[kv]_warp_id|empty_warp_ids|tq4_conversion_warp_ids)\s*=",
        ),
        "named_barrier_declarations": matching_lines(
            lines, r"self\.[a-z0-9_]*bar[a-z0-9_]*\s*=\s*pipeline\.NamedBarrier"
        ),
        "tmem_wait_or_retrieve": matching_lines(lines, r"tmem\.(wait_for_alloc|retrieve_ptr)\("),
        "tq4_conversion_rendezvous": matching_lines(lines, r"self\.tq4_conversion_sync_bar\.arrive_and_wait\(\)"),
        "conversion_control_flow": matching_lines(lines, r"warp_idx\s*[<>]=?\s*self\.tq4_conversion_warp_ids"),
    }
    require(len(inventory["named_barrier_declarations"]) == 6, "named barrier declaration count differs")
    require(len(inventory["tmem_wait_or_retrieve"]) == 6, "TMEM retrieve call count differs")
    require(len(inventory["tq4_conversion_rendezvous"]) == 5,
            "static TQ4 conversion rendezvous site count differs")
    source_text = "\n".join(lines)
    require("barrier_id=1" in source_text and "barrier_id=6" in source_text,
            "expected barrier IDs are absent")
    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-barrier-source-inventory",
        "status": "pass",
        "source_commit": identity["source_commit"],
        "source_identity_sha256": sha256_bytes(identity_raw),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "inferred_contract": {
            "m128_cta_threads": 512,
            "tmem_barrier": {"id": 1, "count": 288, "intended_threads": [0, 287]},
            "tq4_conversion_barrier": {"id": 6, "count": 128, "intended_threads": [384, 511]},
            "dynamic_conversion_rendezvous_per_tile": 6,
        },
        "inventory": inventory,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
