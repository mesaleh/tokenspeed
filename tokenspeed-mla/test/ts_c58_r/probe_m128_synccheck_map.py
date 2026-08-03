#!/usr/bin/env python3
"""Run one M128 mapping launch and preserve artifacts after synccheck poisons CUDA."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

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
from probe_tq4_m128_sanitizer import compiler_artifacts


def load_case_builder(source_root: Path):
    path = source_root / "tokenspeed-mla/test/probe_tq4_m128_control.py"
    spec = importlib.util.spec_from_file_location("ts_c58_r_mapping_case_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load case builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._make_case, sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--target-uuid", required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--expected-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")

    source_root = args.source_root.resolve()
    identity, identity_raw = load_json(args.identity)
    validate_source_identity(identity, source_root)
    identity_sha256 = sha256_bytes(identity_raw)
    target_uuid = canonical_uuid(args.target_uuid)
    expected, expected_raw = load_json(args.expected)
    expected_seal, expected_seal_raw = load_json(args.expected_seal)
    require(
        expected.get("record_type") == "ts-c58-r-decode-oracle"
        and expected.get("status") == "pass"
        and expected.get("arm") == "m128"
        and expected.get("mode") == "unsanitized"
        and expected.get("compiler_artifacts_present") is True
        and expected.get("compiler_keep") == "ir,ptx,cubin",
        "accepted M128 oracle differs",
    )
    require(
        expected.get("source_commit") == identity["source_commit"]
        and expected.get("source_identity_sha256") == identity_sha256
        and expected.get("target_uuid") == target_uuid
        and expected.get("device_index") == args.device_index,
        "accepted M128 oracle scope differs",
    )
    require(
        expected_seal.get("record_type") == "ts-c58-r-execution-seal"
        and expected_seal.get("status") == "pass"
        and expected_seal.get("cell_id") == "accepted-target-unsanitized"
        and expected_seal.get("actual_outcome") == "clean"
        and expected_seal.get("sanitizer_tool") is None
        and expected_seal.get("result_sha256") == sha256_bytes(expected_raw),
        "accepted M128 oracle seal differs",
    )
    require(
        expected_seal.get("source_commit") == identity["source_commit"]
        and expected_seal.get("source_identity_sha256") == identity_sha256
        and expected_seal.get("target_uuid") == target_uuid
        and expected_seal.get("device_index") == args.device_index,
        "accepted M128 oracle seal scope differs",
    )
    tool_root = Path(__file__).resolve().parent
    require(
        expected_seal.get("runner_sha256") == sha256_file(tool_root / "run_compute_sanitizer.py")
        and expected_seal.get("sealer_sha256")
        == sha256_file(tool_root / "seal_sanitizer_result.py"),
        "accepted M128 oracle seal tool identity differs",
    )

    require(0 <= args.device_index < torch.cuda.device_count(), "CUDA ordinal is not visible")
    sys.path.insert(0, str(source_root / "tokenspeed-mla/python"))
    make_case, builder_sha256 = load_case_builder(source_root)
    require(
        builder_sha256
        == identity["source_hashes"].get("tokenspeed-mla/test/probe_tq4_m128_control.py"),
        "case builder identity differs",
    )
    from tokenspeed_mla.mla_decode_tq4 import _tokenspeed_mla_decode_tq4_m128_control

    torch.cuda.set_device(args.device_index)
    capability = torch.cuda.get_device_capability()
    require(capability == (10, 0), "SM100 is required")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    raw_uuid = str(getattr(properties, "uuid", ""))
    device_uuid = canonical_uuid(
        raw_uuid if raw_uuid.lower().startswith("gpu-") else f"GPU-{raw_uuid}"
    )
    require(device_uuid == target_uuid, "CUDA ordinal does not map to target UUID")
    device_name = torch.cuda.get_device_name()

    case = make_case(1, False, heads=8, use_codebook=False, fp8_rope=False)
    caught: torch.AcceleratorError | None = None
    try:
        _tokenspeed_mla_decode_tq4_m128_control(
            query=case["query"],
            kv_nope_packed=case["packed"],
            kv_nope_scale=case["scales"],
            kv_rope=case["rope_storage"],
            centroids=case["centroids"],
            kv_nope_codebook=case["codebook"],
            fp8_rope=case["fp8_rope"],
            workspace_buffer=case["workspace"],
            block_tables=case["block_tables"],
            seq_lens=case["seq_lens"],
            max_seq_len=case["max_seq_len"],
            softmax_scale=576**-0.5,
            custom_mask=case["custom_mask"],
            enable_pdl=True,
            return_lse=True,
            split_kv_override=1,
        )
        torch.cuda.synchronize()
    except torch.AcceleratorError as exc:
        caught = exc
    require(caught is not None, "synccheck mapping launch did not raise AcceleratorError")
    message = str(caught)
    require("CUDA error: unspecified launch failure" in message, "CUDA failure class differs")

    dump_dir, cache_dir, keep, artifacts = compiler_artifacts()
    require(artifacts == expected.get("compiler_artifacts"),
            "mapping compiler artifacts differ from accepted oracle")
    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-synccheck-map-probe",
        "status": "pass",
        "source_commit": identity["source_commit"],
        "source_identity_sha256": identity_sha256,
        "source_root": str(source_root),
        "builder_sha256": builder_sha256,
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "device_index": args.device_index,
        "device_uuid": device_uuid,
        "target_uuid": target_uuid,
        "device_name": device_name,
        "compute_capability": list(capability),
        "compiler_dump_dir": str(dump_dir),
        "compiler_cache_dir": str(cache_dir),
        "compiler_keep": keep,
        "compiler_artifacts_present": True,
        "compiler_artifacts": artifacts,
        "artifacts_match_unsanitized": True,
        "expected_sha256": sha256_bytes(expected_raw),
        "expected_seal_sha256": sha256_bytes(expected_seal_raw),
        "caught_cuda_error": True,
        "caught_error_type": type(caught).__name__,
        "caught_error_message_sha256": sha256_bytes(message.encode("utf-8")),
        "case": {
            "case_id": "q1-h8-basic",
            "q_len": 1,
            "heads": 8,
            "tree_mask": False,
            "use_codebook": False,
            "fp8_rope": False,
            "enable_pdl": True,
            "split_kv_override": 1,
        },
    }
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
