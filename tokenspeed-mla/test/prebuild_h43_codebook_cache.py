#!/usr/bin/env python3
"""Compile the exact H43 CuTe keys without allocating or launching a reader."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import tokenspeed_mla.mla_decode as mla_decode
import torch
from h43_aot_loader import (
    AOT_MANIFEST_NAME,
    dispatch_key,
    install_h43_aot_from_environment,
    load_aot_manifest,
    write_aot_manifest,
)
from h43_codebook_ab_common import (
    canonical_json_digest,
    compiled_artifact_manifest,
    load_contract,
    sha256_file,
)
from tokenspeed_mla.mla_decode import tokenspeed_mla_decode, tokenspeed_mla_decode_tq4


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


def export_aot_kernel(
    *,
    compiled: Any,
    cache_root: Path,
    label: str,
    key: str,
) -> dict[str, Any]:
    from cutlass import cute

    function_name = f"h43_{label.replace('-', '_')}"
    object_dir = cache_root / "objects"
    library_dir = cache_root / "libraries"
    object_dir.mkdir(exist_ok=True)
    library_dir.mkdir(exist_ok=True)
    object_path = object_dir / f"{function_name}.o"
    library_path = library_dir / f"{function_name}.so"
    temporary_object = object_dir / f".tmp-{function_name}.o"
    temporary_library = library_dir / f".tmp-{function_name}.so"
    if object_path.exists() or library_path.exists():
        raise RuntimeError(f"H43 AOT artifact already exists for {label}")
    compiled.export_to_c(str(temporary_object), function_name=function_name)
    runtime_libraries = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-o",
            str(temporary_library),
            str(temporary_object),
            *runtime_libraries,
        ],
        check=True,
        timeout=120,
    )
    os.replace(temporary_object, object_path)
    os.replace(temporary_library, library_path)
    return {
        "dispatch_key": key,
        "label": label,
        "function_name": function_name,
        "object": object_path.relative_to(cache_root).as_posix(),
        "object_sha256": sha256_file(object_path),
        "library": library_path.relative_to(cache_root).as_posix(),
        "library_sha256": sha256_file(library_path),
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
    if torch.cuda.device_count() != 1:
        raise RuntimeError("H43 prebuild requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    torch.cuda.init()
    if torch.cuda.current_device() != 0:
        raise RuntimeError("H43 prebuild failed to bind visible CUDA device 0")
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

    getter = mla_decode._get_compiled_mla_kernel
    if args.phase == "warm":
        getter = install_h43_aot_from_environment(expected_count)

    evidence: list[dict[str, Any]] = []
    aot_entries: dict[str, dict[str, Any]] = {}
    for label, arguments in keys:
        cache_before = getter.cache_info()
        started = time.monotonic()
        compiled = getter(**arguments)
        elapsed = time.monotonic() - started
        cache_after = getter.cache_info()
        if cache_after.misses - cache_before.misses != 1:
            raise RuntimeError(f"{label} did not create one fresh dispatch entry")
        key = dispatch_key(mla_decode._get_compiled_mla_kernel, **arguments)
        if args.phase == "cold":
            entry = export_aot_kernel(
                compiled=compiled,
                cache_root=cache_root,
                label=label,
                key=key,
            )
            if key in aot_entries:
                raise RuntimeError(f"duplicate H43 AOT dispatch key for {label}")
            aot_entries[key] = entry
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
    if args.phase == "cold":
        write_aot_manifest(
            cache_root / AOT_MANIFEST_NAME,
            experiment=contract["experiment"],
            source_manifest_digest=args.source_manifest_digest,
            installed_mla_sha256=args.installed_mla_sha256,
            entries=aot_entries,
        )
    load_aot_manifest(
        cache_root / AOT_MANIFEST_NAME,
        expected_source_manifest_digest=args.source_manifest_digest,
        expected_installed_mla_sha256=args.installed_mla_sha256,
        expected_entries=expected_count,
    )
    after = compiled_artifact_manifest(cache_root, contract["cache"])
    expected_artifacts = 2 * expected_count + 1
    if len(after["files"]) != expected_artifacts:
        raise RuntimeError(
            f"H43 AOT artifact count {len(after['files'])} != {expected_artifacts}"
        )
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
