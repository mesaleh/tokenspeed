#!/usr/bin/env python3
"""Compile the exact H43 CuTe keys without allocating or launching a reader."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from h43_codebook_ab_common import (
    canonical_json_digest,
    compiled_artifact_manifest,
    load_contract,
    sha256_file,
)
from tokenspeed_mla.mla_decode import (
    _get_compiled_mla_kernel,
    tokenspeed_mla_decode,
    tokenspeed_mla_decode_tq4,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest-digest", required=True)
    parser.add_argument("--installed-mla-sha256", required=True)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    return parser.parse_args()


def key_arguments(
    *, q_len: int, tq4: bool, tiles_per_split: int = 1, codebook: bool = False
) -> dict[str, Any]:
    dense_default = (
        inspect.signature(tokenspeed_mla_decode).parameters["is_var_seq"].default
    )
    tq_default = (
        inspect.signature(tokenspeed_mla_decode_tq4).parameters["is_var_seq"].default
    )
    if dense_default is not True or tq_default is not dense_default:
        raise RuntimeError(
            "H43 requires the shared production variable-sequence default"
        )
    is_var_seq = dense_default
    return {
        "torch_dtype": torch.float8_e4m3fn,
        "page_size": 32,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "is_persistent": not is_var_seq,
        "is_var_seq": is_var_seq,
        "is_var_split_kv": False,
        "skip_correction_threshold": 0.0,
        "is_workspace_size_zero": False,
        "fold_sq": True,
        "causal_mask": True,
        "num_heads": 8,
        "seq_len_q": q_len,
        "tree_mask_mode": False,
        "fold_q_chunk_size": 0,
        "use_pdl": True,
        "tq4_cache": tq4,
        "tq4_fp8_rope": tq4,
        "tq4_tiles_per_split": tiles_per_split,
        "tq4_codebook": codebook,
        "return_lse": False,
    }


def serializable(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, torch.dtype) else value
        for key, value in arguments.items()
    }


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract.resolve())
    cache_root = args.cache_root.resolve()
    if cache_root.name != args.source_manifest_digest:
        raise RuntimeError("cache namespace is not keyed by source-manifest digest")
    configured_cache = Path(os.environ.get("CUTE_DSL_CACHE_DIR", "")).resolve()
    if configured_cache != cache_root:
        raise RuntimeError(
            f"CUTE_DSL_CACHE_DIR {configured_cache} does not match {cache_root}"
        )
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("PYTHONDONTWRITEBYTECODE=1 is required")
    imported = Path(
        __import__("tokenspeed_mla.mla_decode_fp8", fromlist=["x"]).__file__
    ).resolve()
    if sha256_file(imported) != args.installed_mla_sha256:
        raise RuntimeError("installed mla_decode_fp8.py digest mismatch")
    before = compiled_artifact_manifest(cache_root, contract["cache"])
    if args.phase == "cold" and before["files"]:
        raise RuntimeError("cold prebuild requires an empty cache namespace")
    if args.phase == "warm" and not before["files"]:
        raise RuntimeError("warm prebuild requires existing compiled artifacts")

    keys: list[tuple[str, dict[str, Any]]] = []
    for q_len in contract["geometry"]["correctness_query_lengths"]:
        keys.append((f"dense-q{q_len}", key_arguments(q_len=q_len, tq4=False)))
    for tiles in sorted(
        {value["tq4_tiles_per_split"] for value in contract["contexts"].values()}
    ):
        for q_len in contract["geometry"]["correctness_query_lengths"]:
            for codebook in (False, True):
                keys.append(
                    (
                        f"tq-tiles{tiles}-q{q_len}-codebook{int(codebook)}",
                        key_arguments(
                            q_len=q_len,
                            tq4=True,
                            tiles_per_split=tiles,
                            codebook=codebook,
                        ),
                    )
                )
    expected_count = (
        contract["cache"]["dense_dispatch_keys"] + contract["cache"]["tq_dispatch_keys"]
    )
    if len(keys) != expected_count:
        raise RuntimeError(f"prebuild key count {len(keys)} != {expected_count}")

    evidence: list[dict[str, Any]] = []
    for label, arguments in keys:
        cache_before = _get_compiled_mla_kernel.cache_info()
        started = time.monotonic()
        _get_compiled_mla_kernel(**arguments)
        elapsed = time.monotonic() - started
        cache_after = _get_compiled_mla_kernel.cache_info()
        if cache_after.misses - cache_before.misses != 1:
            raise RuntimeError(f"{label} did not create one fresh dispatch entry")
        evidence.append(
            {
                "label": label,
                "arguments": serializable(arguments),
                "wall_time_seconds": elapsed,
                "artifact_manifest": compiled_artifact_manifest(
                    cache_root, contract["cache"]
                ),
            }
        )
    after = compiled_artifact_manifest(cache_root, contract["cache"])
    if args.phase == "warm" and before["digest"] != after["digest"]:
        raise RuntimeError("warm prebuild created or changed a compiled artifact")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": contract["experiment"],
        "phase": args.phase,
        "contract_digest": canonical_json_digest(contract),
        "source_manifest_digest": args.source_manifest_digest,
        "installed_mla_path": str(imported),
        "installed_mla_sha256": args.installed_mla_sha256,
        "cache_before": before,
        "cache_after": after,
        "keys": evidence,
        "reader_launches": 0,
        "ring_allocations": 0,
    }
    result["result_digest"] = canonical_json_digest(result)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
