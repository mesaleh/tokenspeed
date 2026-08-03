#!/usr/bin/env python3
"""Run one isolated dense or native-TQ4-M128 arm and bind output/LSE hashes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import torch

from evidence_common import (
    canonical_uuid,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    validate_source_identity,
    write_json_exclusive,
)


CASES = (
    ("q1-h8-basic", 1, False, 8, False, False, True),
    ("q5-h8-codebook-fp8-tree", 5, True, 8, True, True, True),
    ("q1-h16-basic", 1, False, 16, False, False, True),
)


def load_case_builder(source_root: Path):
    path = source_root / "tokenspeed-mla/test/probe_tq4_m128_control.py"
    spec = importlib.util.spec_from_file_location("ts_c58_r_case_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load case builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._make_case, sha256_file(path)


def tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    raw = value.view(torch.uint8).cpu().numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "nbytes": len(raw),
        "finite": bool(torch.isfinite(value.float()).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--target-uuid", required=True)
    parser.add_argument("--arm", choices=("dense", "m128"), required=True)
    parser.add_argument("--mode", choices=("unsanitized", "racecheck", "synccheck"), required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")
    if args.mode == "unsanitized" and args.expected is not None:
        parser.error("unsanitized mode cannot consume an oracle")
    if args.mode != "unsanitized" and args.expected is None:
        parser.error("sanitizer mode requires an oracle")

    source_root = args.source_root.resolve()
    identity, identity_raw = load_json(args.identity)
    validate_source_identity(identity, source_root)
    target_uuid = canonical_uuid(args.target_uuid)
    require(0 <= args.device_index < torch.cuda.device_count(), "CUDA ordinal is not visible")
    sys.path.insert(0, str(source_root / "tokenspeed-mla/python"))
    make_case, builder_sha256 = load_case_builder(source_root)
    expected_builder = identity["source_hashes"].get(
        "tokenspeed-mla/test/probe_tq4_m128_control.py"
    )
    require(builder_sha256 == expected_builder, "case builder identity differs")
    from tokenspeed_mla.mla_decode import tokenspeed_mla_decode
    from tokenspeed_mla.mla_decode_tq4 import _tokenspeed_mla_decode_tq4_m128_control

    torch.cuda.set_device(args.device_index)
    require(torch.cuda.get_device_capability() == (10, 0), "SM100 is required")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    raw_uuid = str(getattr(properties, "uuid", ""))
    device_uuid = canonical_uuid(raw_uuid if raw_uuid.lower().startswith("gpu-") else f"GPU-{raw_uuid}")
    require(device_uuid == target_uuid, "CUDA ordinal does not map to target UUID")

    results = []
    for case_id, q_len, tree_mask, heads, use_codebook, fp8_rope, enable_pdl in CASES:
        case = make_case(
            q_len,
            tree_mask,
            heads=heads,
            use_codebook=use_codebook,
            fp8_rope=fp8_rope,
        )
        common = {
            "query": case["query"],
            "workspace_buffer": case["workspace"],
            "block_tables": case["block_tables"],
            "seq_lens": case["seq_lens"],
            "max_seq_len": case["max_seq_len"],
            "softmax_scale": 576**-0.5,
            "custom_mask": case["custom_mask"],
            "enable_pdl": enable_pdl,
            "return_lse": True,
        }
        if args.arm == "dense":
            output, lse = tokenspeed_mla_decode(
                kv_cache=case["dense_cache"],
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                **common,
            )
        else:
            output, lse = _tokenspeed_mla_decode_tq4_m128_control(
                kv_nope_packed=case["packed"],
                kv_nope_scale=case["scales"],
                kv_rope=case["rope_storage"],
                centroids=case["centroids"],
                kv_nope_codebook=case["codebook"],
                fp8_rope=case["fp8_rope"],
                split_kv_override=1,
                **common,
            )
        torch.cuda.synchronize()
        output_digest = tensor_digest(output)
        lse_digest = tensor_digest(lse)
        require(output_digest["finite"] and lse_digest["finite"], f"non-finite case: {case_id}")
        results.append(
            {
                "case_id": case_id,
                "q_len": q_len,
                "tree_mask": tree_mask,
                "heads": heads,
                "use_codebook": use_codebook,
                "fp8_rope": fp8_rope,
                "enable_pdl": enable_pdl,
                "output": output_digest,
                "lse": lse_digest,
            }
        )

    value: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "ts-c58-r-decode-oracle",
        "status": "pass",
        "arm": args.arm,
        "mode": args.mode,
        "source_commit": identity["source_commit"],
        "source_identity_sha256": sha256_bytes(identity_raw),
        "source_root": str(source_root),
        "builder_sha256": builder_sha256,
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "device_index": args.device_index,
        "device_uuid": device_uuid,
        "target_uuid": target_uuid,
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cases": results,
    }
    if args.expected is not None:
        expected, expected_raw = load_json(args.expected)
        expected_comparable = {
            key: expected.get(key)
            for key in (
                "status", "record_type", "arm", "source_commit", "source_identity_sha256",
                "builder_sha256", "wrapper_sha256", "device_index", "device_uuid",
                "target_uuid", "compute_capability", "cases",
            )
        }
        actual_comparable = {
            key: value.get(key)
            for key in expected_comparable
        }
        require(expected.get("mode") == "unsanitized", "expected oracle mode differs")
        require(expected_comparable == actual_comparable, "sanitized output differs from oracle")
        value["expected_sha256"] = sha256_bytes(expected_raw)
        value["hashes_match_unsanitized"] = True
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
