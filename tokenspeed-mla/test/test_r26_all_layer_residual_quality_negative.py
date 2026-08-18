#!/usr/bin/env python3
"""Fail-closed tests for the R26 three-role/all-layer quality gate."""

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


def _complete_roles(evaluator: ModuleType) -> list[dict[str, Any]]:
    return [
        {
            "role": role,
            "layers": [
                {
                    "layer": layer,
                    "surfaces": [
                        {"label": label} for label in evaluator.SURFACE_LABELS
                    ],
                }
                for layer in evaluator.EXPECTED_LAYERS
            ],
        }
        for role in evaluator.EXPECTED_ROLES
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    evaluator = _load(args.evaluator_source, "a17_r26_q1_negative_evaluator")
    r19 = evaluator._load_r19(args.r19_source)
    tests: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="a17-r26-q1-negative-") as temporary:
        for name, canonical, expected in (
            (
                "r26_source_hash_tamper",
                args.r26_source,
                evaluator.EXPECTED_R26_SOURCE_SHA256,
            ),
            (
                "r19_source_hash_tamper",
                args.r19_source,
                evaluator.EXPECTED_R19_SOURCE_SHA256,
            ),
            (
                "r19_result_hash_tamper",
                args.r19_result_1,
                evaluator.EXPECTED_R19_RESULT_SHA256,
            ),
            (
                "r18_source_hash_tamper",
                args.r18_source,
                r19.R18_SOURCE_SHA256,
            ),
            (
                "packer_source_hash_tamper",
                args.packer_source,
                r19.PACKER_SHA256,
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

    r19_result, r19_hash = evaluator._load_r19_result(
        args.r19_result_1, args.r19_result_2
    )
    if r19_hash != evaluator.EXPECTED_R19_RESULT_SHA256:
        raise NegativeTestError("R19 duplicate-result identity differs")
    tests.append({"name": "r19_duplicate_result_identity", "pass": True})
    bridge = evaluator._bridge_map(r19_result)
    if len(bridge) != evaluator.EXPECTED_BRIDGE_CELLS:
        raise NegativeTestError("R19 bridge coverage differs")
    tests.append({"name": "r19_bridge_coverage", "pass": True})

    broken_result = dict(r19_result)
    broken_result["test"] = list(r19_result["test"][:-1])
    tests.append(
        _expect_failure(
            "incomplete_bridge_rejected",
            lambda: evaluator._bridge_map(broken_result),
            "bridge cell count differs",
        )
    )

    if evaluator._positive_ratio(1.0, 2.0, "positive") != 0.5:
        raise NegativeTestError("positive ratio differs")
    tests.append({"name": "positive_ratio", "pass": True})
    tests.extend(
        [
            _expect_failure(
                "zero_ratio_denominator",
                lambda: evaluator._positive_ratio(1.0, 0.0, "negative"),
                "ratio input is invalid",
            ),
            _expect_failure(
                "bridge_tolerance_miss",
                lambda: evaluator._relative_difference(1.1, 1.0, "negative"),
                "bridge differs",
            ),
        ]
    )

    power, residual = evaluator._n8_scale_factorization(
        torch.tensor([0.5, 2.0], dtype=torch.float32)
    )
    if not torch.equal(power * residual, torch.tensor([0.5, 2.0])):
        raise NegativeTestError("accepted-N8 scale factorization differs")
    tests.append({"name": "accepted_n8_scale_factorization", "pass": True})

    complete = _complete_roles(evaluator)
    if evaluator._validate_complete_roles(complete) != evaluator.EXPECTED_CELLS:
        raise NegativeTestError("complete role/cell validation differs")
    tests.append({"name": "complete_role_cell_coverage", "pass": True})
    tests.extend(
        [
            _expect_failure(
                "incomplete_role_cell_coverage",
                lambda: evaluator._validate_complete_roles(complete[:-1]),
                "evaluated cell set differs",
            ),
            _expect_failure(
                "malformed_role_cell_structure",
                lambda: evaluator._validate_complete_roles([{"role": "train"}]),
                "structure is malformed",
            ),
        ]
    )

    decision_cases = {
        "advance": (
            (
                evaluator.EXPECTED_CELLS,
                True,
                evaluator.EXPECTED_BRIDGE_CELLS,
            ),
            evaluator.DECISION_ADVANCE,
        ),
        "quality": (
            (
                evaluator.EXPECTED_CELLS,
                False,
                evaluator.EXPECTED_BRIDGE_CELLS,
            ),
            evaluator.DECISION_REJECT,
        ),
        "cells": (
            (
                evaluator.EXPECTED_CELLS - 1,
                True,
                evaluator.EXPECTED_BRIDGE_CELLS,
            ),
            evaluator.DECISION_REJECT,
        ),
        "bridge": (
            (
                evaluator.EXPECTED_CELLS,
                True,
                evaluator.EXPECTED_BRIDGE_CELLS - 1,
            ),
            evaluator.DECISION_REJECT,
        ),
    }
    for name, (arguments, expected) in decision_cases.items():
        observed = evaluator._decision(*arguments)
        if observed != expected:
            raise NegativeTestError(f"decision self-test differs: {name}")
        tests.append({"name": f"decision_{name}", "pass": True})

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
    if result["tests_passed"] != 19:
        raise NegativeTestError(
            f"negative test coverage differs: {result['tests_passed']} != 19"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-source", type=Path, required=True)
    parser.add_argument("--r26-source", type=Path, required=True)
    parser.add_argument("--r19-source", type=Path, required=True)
    parser.add_argument("--r18-source", type=Path, required=True)
    parser.add_argument("--packer-source", type=Path, required=True)
    parser.add_argument("--r19-result-1", type=Path, required=True)
    parser.add_argument("--r19-result-2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
