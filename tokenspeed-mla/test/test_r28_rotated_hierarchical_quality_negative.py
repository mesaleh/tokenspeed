#!/usr/bin/env python3
"""Fail-closed tests for the R28 rotated hierarchical evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).with_name("evaluate_r28_rotated_hierarchical_rope_quality.py")
SPEC = importlib.util.spec_from_file_location("r28_rotated_quality", SOURCE)
assert SPEC is not None and SPEC.loader is not None
R28 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R28
SPEC.loader.exec_module(R28)


class FakeMath:
    @staticmethod
    def make_sign_contract(device: torch.device | str, dim: int) -> SimpleNamespace:
        assert dim == 64
        return SimpleNamespace(
            signs1=torch.ones(dim, dtype=torch.float32, device=device),
            signs2=torch.ones(dim, dtype=torch.float32, device=device),
            signs1_sha256=R28.EXPECTED_SIGNS1_SHA256,
            signs2_sha256=R28.EXPECTED_SIGNS2_SHA256,
        )

    @staticmethod
    def rotate(x: torch.Tensor, signs: SimpleNamespace) -> torch.Tensor:
        del signs
        return x


def test_identity_transform_dot_self_test_passes() -> None:
    result = R28._transform_dot_self_test(FakeMath)
    assert result["pass"]
    assert result["scaled_absolute_error"] == 0.0


@pytest.mark.parametrize("dimension", (0, 1, 63, 65, 128))
def test_transform_rejects_wrong_dimension(dimension: int) -> None:
    with pytest.raises(R28.EvidenceError, match="geometry"):
        R28._rotate(torch.zeros((2, dimension)), FakeMath)


@pytest.mark.parametrize("which", ("signs1", "signs2"))
def test_sign_identity_is_sealed(which: str) -> None:
    class MutatedMath(FakeMath):
        @staticmethod
        def make_sign_contract(
            device: torch.device | str, dim: int
        ) -> SimpleNamespace:
            result = FakeMath.make_sign_contract(device, dim)
            setattr(result, f"{which}_sha256", "0" * 64)
            return result

    with pytest.raises(R28.EvidenceError, match="sign identity"):
        R28._rotate(torch.zeros((2, 64)), MutatedMath)


def test_transform_rejects_nonfinite_output() -> None:
    class NonfiniteMath(FakeMath):
        @staticmethod
        def rotate(x: torch.Tensor, signs: SimpleNamespace) -> torch.Tensor:
            del signs
            result = x.clone()
            result[0, 0] = torch.nan
            return result

    with pytest.raises(R28.EvidenceError, match="nonfinite"):
        R28._rotate(torch.zeros((2, 64)), NonfiniteMath)


def test_transform_dot_tolerance_is_sealed() -> None:
    class InexactMath(FakeMath):
        calls = 0

        @classmethod
        def rotate(cls, x: torch.Tensor, signs: SimpleNamespace) -> torch.Tensor:
            del signs
            cls.calls += 1
            return x + (1.0e-2 if cls.calls == 1 else 0.0)

    with pytest.raises(R28.EvidenceError, match="exceeds tolerance"):
        R28._transform_dot_self_test(InexactMath)


def test_finalize_renames_candidate_identity_and_metrics() -> None:
    result = {
        "experiment_id": "old",
        "contract": {},
        "layers": [
            {
                "surfaces": [
                    {
                        "identity": {
                            "r26_score_sha256": "score",
                            "r26_output_sha256": "output",
                        },
                        "r26": {"mse": 1.0},
                        "r26_to_n8_mse_ratio": 0.9,
                        "r26_to_n8_attention_kl_ratio": 0.8,
                    }
                ]
            }
        ],
        "summary": {
            "max_r26_to_n8_mse_ratio": 0.9,
            "max_r26_to_n8_attention_kl_ratio": 0.8,
            "max_r26_to_dense_score_rmse": 1.0,
            "max_absolute_r26_p_lost_energy": 0.0,
        },
    }
    final = R28._finalize_result(result, {"pass": True})
    surface = final["layers"][0]["surfaces"][0]
    assert final["experiment_id"] == R28.EXPERIMENT_ID
    assert "r26" not in surface and surface["r28"] == {"mse": 1.0}
    assert surface["identity"]["r28_score_sha256"] == "score"
    assert surface["identity"]["r28_output_sha256"] == "output"
    assert final["summary"]["max_r28_to_n8_mse_ratio"] == 0.9


def test_contract_constants_are_sealed() -> None:
    assert R28.TRANSFORM_DIMENSION == 64
    assert R28.MAX_TRANSFORM_DOT_SCALED_ABSOLUTE_ERROR == 2.0e-6
    assert len(R28.EXPECTED_R26_SOURCE_SHA256) == 64
    assert len(R28.EXPECTED_Q0_MATH_SHA256) == 64
    assert len(R28.EXPECTED_Q0_CONTRACT_SHA256) == 64
