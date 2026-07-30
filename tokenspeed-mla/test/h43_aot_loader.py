"""Source-bound TVM-FFI AOT artifacts for the H43 MLA experiment."""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

AOT_MANIFEST_NAME = "h43-aot-manifest.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FUNCTION_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_digest(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("H43 AOT dispatch key refuses non-finite floats")
        return {"float_hex": value.hex()}
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ == "dtype":
        return {"torch_dtype": str(value)}
    raise TypeError(f"unsupported H43 AOT dispatch-key value: {value!r}")


def dispatch_key(function: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    payload = [
        {"name": name, "value": _serializable(value)}
        for name, value in bound.arguments.items()
    ]
    return canonical_json_digest(payload)


def _artifact_path(root: Path, relative: str, suffix: str) -> Path:
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or relative_path.suffix != suffix
        or any(part in ("", ".", "..") for part in relative_path.parts)
    ):
        raise ValueError(f"invalid H43 AOT artifact path: {relative}")
    unresolved = root / relative_path
    current = unresolved
    while current != root:
        if current.is_symlink():
            raise ValueError(f"symlinked H43 AOT artifact path: {relative}")
        current = current.parent
    path = unresolved.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"unsafe or missing H43 AOT artifact: {relative}")
    return path


def write_aot_manifest(
    path: Path,
    *,
    experiment: str,
    source_manifest_digest: str,
    installed_mla_sha256: str,
    entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if (
        path.name != AOT_MANIFEST_NAME
        or path.exists()
        or not experiment
        or not SHA256_RE.fullmatch(source_manifest_digest)
        or not SHA256_RE.fullmatch(installed_mla_sha256)
        or not entries
    ):
        raise ValueError("H43 AOT manifest path is invalid or already exists")
    value: dict[str, Any] = {
        "schema_version": 1,
        "experiment": experiment,
        "source_manifest_digest": source_manifest_digest,
        "installed_mla_sha256": installed_mla_sha256,
        "expected_entries": len(entries),
        "entries": entries,
    }
    value["manifest_digest"] = canonical_json_digest(value)
    temporary = path.with_name(f".tmp-{path.name}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return value


def load_aot_manifest(
    path: Path,
    *,
    expected_source_manifest_digest: str,
    expected_installed_mla_sha256: str,
    expected_entries: int,
) -> dict[str, Any]:
    if path.is_symlink() or path.parent.is_symlink() or path.name != AOT_MANIFEST_NAME:
        raise ValueError("unsafe H43 AOT manifest path")
    path = path.resolve()
    root = path.parent
    if root.is_symlink() or not path.is_file():
        raise ValueError("H43 AOT manifest is missing")
    with path.open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    if not isinstance(value, dict):
        raise ValueError("H43 AOT manifest is not an object")
    if set(value) != {
        "schema_version",
        "experiment",
        "source_manifest_digest",
        "installed_mla_sha256",
        "expected_entries",
        "entries",
        "manifest_digest",
    }:
        raise ValueError("H43 AOT manifest fields differ from schema")
    sealed = dict(value)
    observed_digest = sealed.pop("manifest_digest", None)
    if observed_digest != canonical_json_digest(sealed):
        raise ValueError("H43 AOT manifest digest mismatch")
    expected = {
        "schema_version": 1,
        "source_manifest_digest": expected_source_manifest_digest,
        "installed_mla_sha256": expected_installed_mla_sha256,
        "expected_entries": expected_entries,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"H43 AOT manifest {key} mismatch")
    entries = value.get("entries")
    if not isinstance(entries, dict) or len(entries) != expected_entries:
        raise ValueError("H43 AOT manifest entry count mismatch")
    labels: set[str] = set()
    functions: set[str] = set()
    artifacts: set[str] = set()
    for key, entry in entries.items():
        if not SHA256_RE.fullmatch(key) or not isinstance(entry, dict):
            raise ValueError("invalid H43 AOT manifest entry")
        if set(entry) != {
            "dispatch_key",
            "label",
            "function_name",
            "object",
            "object_sha256",
            "library",
            "library_sha256",
        }:
            raise ValueError("H43 AOT entry fields differ from schema")
        if entry.get("dispatch_key") != key:
            raise ValueError("H43 AOT entry dispatch key mismatch")
        label = entry.get("label")
        function_name = entry.get("function_name")
        if (
            not isinstance(label, str)
            or not FUNCTION_RE.fullmatch(str(function_name))
            or label in labels
            or function_name in functions
        ):
            raise ValueError("duplicate or invalid H43 AOT label/function")
        labels.add(label)
        functions.add(function_name)
        for path_field, digest_field, suffix in (
            ("object", "object_sha256", ".o"),
            ("library", "library_sha256", ".so"),
        ):
            relative = entry.get(path_field)
            expected_digest = entry.get(digest_field)
            if (
                not isinstance(relative, str)
                or relative in artifacts
                or not SHA256_RE.fullmatch(str(expected_digest))
            ):
                raise ValueError("invalid H43 AOT artifact identity")
            artifacts.add(relative)
            artifact = _artifact_path(root, relative, suffix)
            if sha256_file(artifact) != expected_digest:
                raise ValueError(f"H43 AOT artifact digest mismatch: {relative}")
    return value


def install_h43_aot_from_environment(
    expected_entries: int | None = None,
) -> Callable[..., Any]:
    manifest_value = os.environ.get("H43_AOT_MANIFEST", "")
    source_digest = os.environ.get("H43_SOURCE_MANIFEST_DIGEST", "")
    installed_digest = os.environ.get("H43_INSTALLED_MLA_SHA256", "")
    if not manifest_value or not SHA256_RE.fullmatch(source_digest):
        raise RuntimeError("H43 AOT/source environment is incomplete")
    if not SHA256_RE.fullmatch(installed_digest):
        raise RuntimeError("H43 installed MLA digest environment is invalid")
    if expected_entries is None:
        expected_text = os.environ.get("H43_AOT_EXPECTED_ENTRIES", "")
        if not expected_text.isdigit() or int(expected_text) < 1:
            raise RuntimeError("H43 AOT expected-entry environment is invalid")
        expected_entries = int(expected_text)
    manifest_path = Path(manifest_value)
    cache_root = Path(os.environ.get("CUTE_DSL_CACHE_DIR", "")).resolve()
    if manifest_path.parent.resolve() != cache_root:
        raise RuntimeError("H43 AOT manifest is outside CUTE_DSL_CACHE_DIR")
    manifest = load_aot_manifest(
        manifest_path,
        expected_source_manifest_digest=source_digest,
        expected_installed_mla_sha256=installed_digest,
        expected_entries=expected_entries,
    )
    mla = importlib.import_module("tokenspeed_mla.mla_decode")
    installed = Path(
        importlib.import_module("tokenspeed_mla.mla_decode_fp8").__file__
    ).resolve()
    if sha256_file(installed) != installed_digest:
        raise RuntimeError("H43 installed MLA source changed before AOT load")
    original = mla._get_compiled_mla_kernel
    prior_manifest = getattr(original, "_h43_aot_manifest", None)
    if prior_manifest:
        if prior_manifest != str(manifest_path.resolve()):
            raise RuntimeError("H43 AOT loader cannot change manifests in-process")
        return original

    modules: dict[Path, Any] = {}
    functions: dict[str, Any] = {}
    lock = threading.Lock()

    def load_kernel(*args: Any, **kwargs: Any) -> Any:
        key = dispatch_key(original, *args, **kwargs)
        entry = manifest["entries"].get(key)
        if entry is None:
            raise RuntimeError(f"H43 AOT manifest has no dispatch key {key}")
        with lock:
            if key not in functions:
                from cutlass import cute

                library = _artifact_path(
                    manifest_path.resolve().parent, entry["library"], ".so"
                )
                if sha256_file(library) != entry["library_sha256"]:
                    raise RuntimeError(
                        "H43 AOT library changed immediately before load"
                    )
                module = cute.runtime.load_module(str(library), enable_tvm_ffi=True)
                function = getattr(module, entry["function_name"])
                if not callable(function):
                    raise RuntimeError("H43 AOT export is not callable")
                modules[library] = module
                functions[key] = function
            return functions[key]

    patched = functools.lru_cache(maxsize=None)(load_kernel)
    patched = functools.wraps(original)(patched)
    setattr(patched, "_h43_aot_manifest", str(manifest_path.resolve()))
    mla._get_compiled_mla_kernel = patched
    return patched
