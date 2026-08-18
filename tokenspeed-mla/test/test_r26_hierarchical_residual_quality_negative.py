#!/usr/bin/env python3
"""Fail-closed and algebra tests for A17-N10-E0-R26-Q0 evidence."""

from __future__ import annotations

import argparse
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


def _expect_failure(
    name: str, function: Callable[[], Any], expected_text: str
) -> dict[str, Any]:
    try:
        function()
    except Exception as error:
        if expected_text not in str(error):
            raise NegativeTestError(
                f"{name}: wrong failure: {type(error).__name__}: {error}"
            ) from error
        return {
            "name": name,
            "pass": True,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    raise NegativeTestError(f"{name}: invalid input was accepted")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    evaluator = _load(args.evaluator_source, "a17_r26_negative_evaluator")
    n8 = evaluator._load_module(
        args.n8_evaluator,
        name="a17_r26_negative_n8",
        expected_sha256=evaluator.EXPECTED_N8_EVALUATOR_SHA256,
    )
    packer = evaluator._load_module(
        args.packer_source,
        name="a17_r26_negative_packer",
        expected_sha256=evaluator.EXPECTED_PACKER_SHA256,
    )
    tests: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="a17-r26-negative-") as temporary:
        for name, canonical, expected in (
            (
                "accepted_n8_result_hash_tamper",
                args.n8_result,
                evaluator.EXPECTED_N8_RESULT_SHA256,
            ),
            (
                "accepted_n8_evaluator_hash_tamper",
                args.n8_evaluator,
                evaluator.EXPECTED_N8_EVALUATOR_SHA256,
            ),
            (
                "accepted_n8_operands_hash_tamper",
                args.n8_operands,
                evaluator.EXPECTED_N8_OPERANDS_SHA256,
            ),
            (
                "accepted_packer_hash_tamper",
                args.packer_source,
                evaluator.EXPECTED_PACKER_SHA256,
            ),
        ):
            bad = Path(temporary) / f"{name}.bin"
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

    n8_payload = torch.load(args.n8_operands, map_location="cpu", weights_only=True)
    _records, capture_hashes, capture_bytes = n8.N7._load_records(args.capture_root)
    evaluator._validate_capture_identity(
        capture_hashes=capture_hashes,
        capture_bytes=capture_bytes,
        n8_payload=n8_payload,
    )
    tests.append({"name": "canonical_capture_identity", "pass": True})
    stale_hashes = dict(capture_hashes)
    stale_hashes[sorted(stale_hashes)[0]] = "0" * 64
    tests.append(
        _expect_failure(
            "stale_capture_inventory",
            lambda: evaluator._validate_capture_identity(
                capture_hashes=stale_hashes,
                capture_bytes=capture_bytes,
                n8_payload=n8_payload,
            ),
            "capture inventory differs",
        )
    )

    original_e2m1_max = evaluator.E2M1_MAX_MAGNITUDE

    def reject_mutated_scale_contract() -> None:
        try:
            evaluator.E2M1_MAX_MAGNITUDE = 4.0
            evaluator._require_residual_geometry()
        finally:
            evaluator.E2M1_MAX_MAGNITUDE = original_e2m1_max

    tests.append(
        _expect_failure(
            "residual_scale_contract_mutation",
            reject_mutated_scale_contract,
            "range contract differs from sealed values",
        )
    )

    original_group_size = evaluator.RESIDUAL_GROUP_SIZE

    def reject_mutated_geometry() -> None:
        try:
            evaluator.RESIDUAL_GROUP_SIZE = 16
            evaluator._require_residual_geometry()
        finally:
            evaluator.RESIDUAL_GROUP_SIZE = original_group_size

    tests.append(
        _expect_failure(
            "residual_geometry_mutation",
            reject_mutated_geometry,
            "geometry differs from two sealed groups of 32",
        )
    )

    exponent, group_scale, scale_raw = evaluator._smallest_no_clip_ue8m0_scale(
        torch.tensor([[0.0, 6.0], [6.0001, 12.0]], dtype=torch.float32)
    )
    if not torch.equal(
        exponent, torch.tensor([[0, 0], [1, 1]], dtype=torch.int32)
    ) or not torch.equal(group_scale, torch.tensor([[1.0, 1.0], [2.0, 2.0]])):
        raise NegativeTestError("smallest no-clipping UE8M0 scale differs")
    if not torch.equal(scale_raw, (exponent + 127).to(torch.uint8)):
        raise NegativeTestError("UE8M0 raw exponent encoding differs")
    tests.append({"name": "smallest_no_clip_ue8m0_scale", "pass": True})

    original_factor = evaluator.NORMALIZATION_FACTOR

    def reject_mutated_factor() -> None:
        try:
            evaluator.NORMALIZATION_FACTOR = 8.0
            evaluator._require_normalization_factor()
        finally:
            evaluator.NORMALIZATION_FACTOR = original_factor

    tests.append(
        _expect_failure(
            "normalization_factor_mutation",
            reject_mutated_factor,
            "differs from sealed power-of-two 16",
        )
    )

    key = torch.tensor([[0.3, -0.7] * 32, [0.35, 1.7] * 32], dtype=torch.bfloat16)
    scale = torch.tensor([0.5, 2.0], dtype=torch.bfloat16)
    candidate_scale, exact, normalized, stored, residual, zero = (
        evaluator._prepare_reciprocal_rope(key, scale, packer)
    )
    if zero.any() or not torch.equal(candidate_scale, scale.to(torch.float32)):
        raise NegativeTestError("positive-scale reciprocal preparation differs")
    if not torch.equal(
        stored.view(torch.uint8), normalized.to(torch.float8_e4m3fn).view(torch.uint8)
    ):
        raise NegativeTestError("normalized E4M3 reciprocal rounding differs")
    tests.append({"name": "positive_scale_rounding", "pass": True})
    diagnostics = evaluator._float8_diagnostics(
        normalized,
        stored,
        expected_dtype=torch.float8_e4m3fn,
        label="R26 negative-suite normalized E4M3 high",
    )
    if (
        residual["diagnostics"]["clipped_coordinates"] != 0
        or not residual["diagnostics"]["packed_round_trip_bit_identical"]
        or not torch.equal(
            packer.unpack_e2m1_codes(residual["packed"]), residual["codes"]
        )
    ):
        raise NegativeTestError("hierarchical E2M1 residual encoding differs")
    tests.append({"name": "residual_packed_round_trip", "pass": True})
    if not diagnostics["raw_bit_reconstruction_bit_identical"]:
        raise NegativeTestError("E4M3 raw-byte reconstruction differs")
    tests.append({"name": "e4m3_raw_byte_reconstruction", "pass": True})
    tests.append(
        _expect_failure(
            "reciprocal_float_emulation_rejected",
            lambda: evaluator._float8_diagnostics(
                normalized,
                normalized.to(torch.bfloat16),
                expected_dtype=torch.float8_e4m3fn,
                label="R26 wrong high format",
            ),
            "dtype differs",
        )
    )

    zero_key = torch.tensor([[0.5, -1.0] * 32], dtype=torch.bfloat16)
    zero_scale = torch.zeros(1, dtype=torch.bfloat16)
    unity, zero_exact, zero_normalized, zero_stored, zero_residual, zero_mask = (
        evaluator._prepare_reciprocal_rope(zero_key, zero_scale, packer)
    )
    if (
        not zero_mask.item()
        or unity.item() != 1.0
        or not torch.equal(zero_exact, zero_key.to(torch.float32))
        or not torch.equal(
            zero_normalized, zero_key.to(torch.float32) / evaluator.NORMALIZATION_FACTOR
        )
        or not torch.equal(
            zero_stored.view(torch.uint8),
            (zero_key.to(torch.float32) / evaluator.NORMALIZATION_FACTOR)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8),
        )
        or zero_residual["diagnostics"]["clipped_coordinates"] != 0
        or not zero_residual["diagnostics"]["packed_round_trip_bit_identical"]
    ):
        raise NegativeTestError("zero-scale hierarchical-RoPE policy differs")
    tests.append({"name": "zero_scale_unity_raw_rope", "pass": True})

    tests.extend(
        [
            _expect_failure(
                "key_not_bf16",
                lambda: evaluator._prepare_reciprocal_rope(
                    key.to(torch.float32), scale, packer
                ),
                "must originate as BF16",
            ),
            _expect_failure(
                "scale_not_bf16",
                lambda: evaluator._prepare_reciprocal_rope(
                    key, scale.to(torch.float32), packer
                ),
                "must originate as BF16",
            ),
            _expect_failure(
                "wrong_rope_width",
                lambda: evaluator._prepare_reciprocal_rope(key[:, :63], scale, packer),
                "geometry differs",
            ),
            _expect_failure(
                "negative_scale",
                lambda: evaluator._prepare_reciprocal_rope(
                    key,
                    torch.tensor([-0.5, 2.0], dtype=torch.bfloat16),
                    packer,
                ),
                "negative",
            ),
            _expect_failure(
                "nonfinite_key",
                lambda: evaluator._prepare_reciprocal_rope(
                    torch.full((2, 64), float("inf"), dtype=torch.bfloat16),
                    scale,
                    packer,
                ),
                "nonfinite",
            ),
            _expect_failure(
                "nonfinite_scale",
                lambda: evaluator._prepare_reciprocal_rope(
                    key,
                    torch.tensor([float("nan"), 2.0], dtype=torch.bfloat16),
                    packer,
                ),
                "nonfinite",
            ),
            _expect_failure(
                "reciprocal_overflow",
                lambda: evaluator._prepare_reciprocal_rope(
                    torch.full(
                        (1, 64),
                        torch.finfo(torch.bfloat16).max,
                        dtype=torch.bfloat16,
                    ),
                    torch.tensor(
                        [torch.finfo(torch.bfloat16).tiny], dtype=torch.bfloat16
                    ),
                    packer,
                ),
                "overflows or is nonfinite",
            ),
            _expect_failure(
                "query_rope_not_bf16",
                lambda: evaluator._query_rope_e4m3(
                    torch.zeros((1, 64, 64), dtype=torch.float32),
                    layer_id=0,
                    label="negative",
                ),
                "must originate as BF16",
            ),
            _expect_failure(
                "query_rope_wrong_geometry",
                lambda: evaluator._query_rope_e4m3(
                    torch.zeros((64, 64), dtype=torch.bfloat16),
                    layer_id=0,
                    label="negative",
                ),
                "geometry differs",
            ),
        ]
    )

    query = torch.tensor(
        [[[1.0, -0.5] * 32]],
        dtype=torch.bfloat16,
    )
    query_high, query_residual, query_diagnostics = evaluator._query_rope_e4m3(
        query, layer_id=0, label="positive"
    )
    query_high_expected = (
        query[0].to(torch.float32) * evaluator.NORMALIZATION_FACTOR
    ).to(torch.float8_e4m3fn)
    query_residual_expected = query[0].to(torch.float8_e4m3fn)
    if (
        not torch.equal(
            query_high,
            query_high_expected.to(torch.float32),
        )
        or not torch.equal(query_residual, query_residual_expected.to(torch.float32))
        or not query_diagnostics["high"]["raw_bit_reconstruction_bit_identical"]
        or not query_diagnostics["residual"]["raw_bit_reconstruction_bit_identical"]
    ):
        raise NegativeTestError("hierarchical E4M3 query rounding differs")
    tests.append({"name": "hierarchical_query_rounding", "pass": True})

    q_rot = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    raw = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    q_rope = torch.tensor([[7.0, 8.0]], dtype=torch.float32)
    reciprocal = torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    score_scale = torch.tensor([0.5, 2.0], dtype=torch.float32)
    observed_score = evaluator._reciprocal_score(
        q_rot=q_rot,
        raw=raw,
        q_rope=q_rope,
        reciprocal_rope=reciprocal,
        scale=score_scale,
    )
    expected_score = torch.tensor([[77.0, 380.0]], dtype=torch.float32)
    if not torch.equal(observed_score, expected_score):
        raise NegativeTestError(
            f"single-accumulator scale algebra differs: {observed_score}"
        )
    tests.append({"name": "single_accumulator_scale_algebra", "pass": True})
    factorization = evaluator._rope_normalization_factorization_self_test()
    if not factorization["pass"] or factorization["normalization_factor"] != 16.0:
        raise NegativeTestError("RoPE normalization factorization self-test differs")
    tests.append({"name": "rope_normalization_factorization", "pass": True})
    tests.append(
        _expect_failure(
            "wrong_rope_normalization_factorization",
            lambda: evaluator._rope_normalization_factorization_self_test(
                query_factor=8.0
            ),
            "normalization factors differ",
        )
    )
    hierarchical_factorization = evaluator._hierarchical_factorization_self_test()
    if not hierarchical_factorization["pass"]:
        raise NegativeTestError("hierarchical factorization self-test differs")
    tests.append({"name": "hierarchical_scale_factorization", "pass": True})
    tests.append(
        _expect_failure(
            "wrong_hierarchical_scale_factorization",
            lambda: evaluator._hierarchical_factorization_self_test(
                residual_scale_override=torch.ones((2, 2), dtype=torch.float32)
            ),
            "hierarchical residual scale factorization differs",
        )
    )

    labels = ["prefill_final"] + [f"target_verify_q{row}" for row in range(5)]
    expected_keys = {(layer, label) for layer in (0, 30, 60) for label in labels}
    expected_scores = {key: torch.empty(0) for key in expected_keys}
    expected_cells = {key: {} for key in expected_keys}
    complete_layers = [
        {
            "layer": layer,
            "surfaces": [{"label": label} for label in labels],
        }
        for layer in (0, 30, 60)
    ]
    if (
        evaluator._validate_evaluated_cell_set(
            expected_scores=expected_scores,
            expected_cells=expected_cells,
            layers=complete_layers,
        )
        != 18
    ):
        raise NegativeTestError("complete-cell validation differs")
    tests.append({"name": "complete_cell_coverage", "pass": True})
    tests.extend(
        [
            _expect_failure(
                "incomplete_cell_coverage",
                lambda: evaluator._validate_evaluated_cell_set(
                    expected_scores=expected_scores,
                    expected_cells=expected_cells,
                    layers=complete_layers[:-1],
                ),
                "cell set differs",
            ),
            _expect_failure(
                "malformed_cell_structure",
                lambda: evaluator._validate_evaluated_cell_set(
                    expected_scores=expected_scores,
                    expected_cells=expected_cells,
                    layers=[{"layer": 0}],
                ),
                "structure is malformed",
            ),
            _expect_failure(
                "wrong_pv_factorization",
                lambda: evaluator._pv_factorization_self_test(
                    n8.N7, scale_override=torch.ones(2, dtype=torch.float32)
                ),
                "P/V token-scale factorization differs",
            ),
        ]
    )

    result = {
        "schema_version": 1,
        "experiment_id": evaluator.EXPERIMENT_ID,
        "evaluator_sha256": hashlib.sha256(
            args.evaluator_source.read_bytes()
        ).hexdigest(),
        "tests": tests,
        "tests_passed": sum(test["pass"] for test in tests),
        "all_pass": all(test["pass"] for test in tests),
    }
    if result["tests_passed"] != 34:
        raise NegativeTestError(
            f"negative test coverage differs: {result['tests_passed']} != 34"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-source", type=Path, required=True)
    parser.add_argument("--n8-evaluator", type=Path, required=True)
    parser.add_argument("--n8-result", type=Path, required=True)
    parser.add_argument("--n8-operands", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--packer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
