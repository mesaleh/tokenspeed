#!/usr/bin/env python3
"""Fail-closed negative tests for the frozen A17-N9-Q0 evidence chain."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch


class NegativeTestError(RuntimeError):
    pass


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise NegativeTestError(f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expect_failure(
    name: str, function: Callable[[], Any], expected_text: str | None = None
) -> dict[str, Any]:
    try:
        function()
    except Exception as error:
        if expected_text is not None and expected_text not in str(error):
            raise NegativeTestError(
                f"{name}: wrong failure: {type(error).__name__}: {error}"
            ) from error
        return {
            "name": name,
            "pass": True,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    raise NegativeTestError(f"{name}: tamper was accepted")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    evaluator = _load(args.evaluator_source, "a17_n9_negative_evaluator")
    packer = _load(args.packer_source, "a17_n9_negative_packer")
    n8 = evaluator._load_module(
        args.n8_evaluator,
        name="a17_n9_negative_n8",
        expected_sha256=evaluator.EXPECTED_N8_EVALUATOR_SHA256,
    )
    payload = torch.load(args.n9_operands, map_location="cpu", weights_only=True)
    n8_payload = torch.load(args.n8_operands, map_location="cpu", weights_only=True)
    records, capture_hashes, capture_bytes = n8.N7._load_records(args.capture_root)
    if capture_bytes != evaluator.EXPECTED_CAPTURE_BYTES:
        raise NegativeTestError("canonical capture byte count differs")

    tests: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="a17-n9-negative-") as temporary:
        temporary_path = Path(temporary)

        for name, canonical, expected in (
            (
                "n8_operand_hash_tamper",
                args.n8_operands,
                evaluator.EXPECTED_N8_OPERANDS_SHA256,
            ),
            (
                "n8_result_hash_tamper",
                args.n8_result,
                evaluator.EXPECTED_N8_RESULT_SHA256,
            ),
            (
                "n8_evaluator_source_hash_tamper",
                args.n8_evaluator,
                evaluator.EXPECTED_N8_EVALUATOR_SHA256,
            ),
        ):
            bad = temporary_path / f"{name}.bin"
            bad.write_bytes(canonical.read_bytes() + b"tamper")
            tests.append(
                _expect_failure(
                    name,
                    lambda bad=bad, expected=expected, name=name: evaluator._require_hash(
                        bad, expected, name
                    ),
                    "hash differs",
                )
            )

        bad_capture_hashes = dict(capture_hashes)
        first_capture = sorted(bad_capture_hashes)[0]
        bad_capture_hashes[first_capture] = "0" * 64
        tests.append(
            _expect_failure(
                "stale_capture_inventory",
                lambda: evaluator._validate_n9_operands(
                    records=records,
                    capture_hashes=bad_capture_hashes,
                    payload=payload,
                    packer=packer,
                    n8=n8,
                    n8_payload=n8_payload,
                ),
                "dependency/self-test binding differs",
            )
        )

        bad_signs = copy.deepcopy(payload)
        bad_signs["signs1_sha256"] = "0" * 64
        tests.append(
            _expect_failure(
                "sign_contract_tamper",
                lambda: evaluator._validate_n9_operands(
                    records=records,
                    capture_hashes=capture_hashes,
                    payload=bad_signs,
                    packer=packer,
                    n8=n8,
                    n8_payload=n8_payload,
                ),
                "sign contract differs",
            )
        )

        original_quantizer = packer.quantize_e2m1_codes

        def wrong_tie_quantizer(values: torch.Tensor) -> torch.Tensor:
            if not torch.isfinite(values).all():
                raise packer.OperandPreparationError("E2M1 input is nonfinite")
            magnitude = torch.abs(values)
            midpoints = packer.E2M1_MIDPOINTS.to(
                device=values.device, dtype=values.dtype
            )
            magnitude_code = torch.bucketize(
                magnitude.contiguous(), midpoints, right=False
            )
            sign_code = torch.signbit(values).to(torch.int64) * 8
            return (magnitude_code + sign_code).to(torch.uint8)

        packer.quantize_e2m1_codes = wrong_tie_quantizer
        try:
            tests.append(
                _expect_failure(
                    "e2m1_tie_rule_tamper",
                    packer._self_test,
                    "midpoint ties-to-even",
                )
            )
        finally:
            packer.quantize_e2m1_codes = original_quantizer

        bad_source = temporary_path / "prepare_bad_e2m1_table.py"
        bad_source.write_text(
            args.packer_source.read_text().replace(
                "        6.0,\n        -0.0,", "        5.5,\n        -0.0,", 1
            )
        )
        manifest = json.loads(args.n9_manifest.read_text())
        tests.append(
            _expect_failure(
                "e2m1_table_source_tamper",
                lambda: evaluator._require_hash(
                    bad_source,
                    str(manifest["source_sha256"]),
                    "N9 operand packer",
                ),
                "hash differs",
            )
        )

        bad_range_layer = copy.deepcopy(payload["layers"][0])
        bad_range_layer["row_ue8m0_codes"][0] = 255
        tests.append(
            _expect_failure(
                "ue8m0_exponent_range_tamper",
                lambda: evaluator._decode_layer(bad_range_layer, packer),
                "nonfinite",
            )
        )

        incomplete = copy.deepcopy(payload)
        incomplete["layers"][0]["surfaces"].pop()
        tests.append(
            _expect_failure(
                "incomplete_surface_set",
                lambda: evaluator._validate_n9_operands(
                    records=records,
                    capture_hashes=capture_hashes,
                    payload=incomplete,
                    packer=packer,
                    n8=n8,
                    n8_payload=n8_payload,
                ),
                "surface coverage differs",
            )
        )

        malformed = temporary_path / "malformed-operands.pt"
        malformed.write_bytes(b"not a torch artifact")
        tests.append(
            _expect_failure(
                "malformed_operand_artifact",
                lambda: evaluator._load_n9_payload(
                    operands=malformed,
                    manifest_path=args.n9_manifest,
                    packer_source=args.packer_source,
                ),
                "hash differs",
            )
        )

    if len(tests) != 10 or not all(test["pass"] for test in tests):
        raise NegativeTestError("negative test coverage/pass count differs")
    return {
        "schema_version": 1,
        "experiment_id": evaluator.EXPERIMENT_ID,
        "threads": torch.get_num_threads(),
        "canonical_identity": {
            "evaluator_sha256": _sha256(args.evaluator_source),
            "packer_sha256": _sha256(args.packer_source),
            "n9_operands_sha256": _sha256(args.n9_operands),
            "n9_manifest_sha256": _sha256(args.n9_manifest),
        },
        "tests": tests,
        "summary": {"tests": len(tests), "all_pass": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "evaluator_source",
        "packer_source",
        "capture_root",
        "n8_evaluator",
        "n8_operands",
        "n8_result",
        "n9_operands",
        "n9_manifest",
        "output",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", dest=name, type=Path, required=True
        )
    args = parser.parse_args()
    result = run(args)
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
