#!/usr/bin/env python3
"""Fail-closed tests for the R29 two-summand query evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).with_name("evaluate_r29_two_summand_rotated_query_quality.py")
SPEC = importlib.util.spec_from_file_location("r29_two_summand_quality", SOURCE)
assert SPEC is not None and SPEC.loader is not None
R29 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R29
SPEC.loader.exec_module(R29)


class CandidateNumericalMiss(RuntimeError):
    pass


class FakeR26:
    CandidateNumericalMiss = CandidateNumericalMiss

    @staticmethod
    def _require_normalization_factor() -> float:
        return 16.0

    @staticmethod
    def _float8_diagnostics(
        source: torch.Tensor,
        stored: torch.Tensor,
        *,
        expected_dtype: torch.dtype,
        label: str,
    ) -> dict[str, object]:
        assert stored.dtype == expected_dtype
        reconstructed = stored.to(torch.float32)
        nonzero = source != 0
        lost = nonzero & (reconstructed == 0)
        source_energy = torch.sum(source.to(torch.float64) ** 2).item()
        lost_energy = torch.sum(source[lost].to(torch.float64) ** 2).item()
        return {
            "label": label,
            "endpoint_count": 0,
            "nonzero_to_zero_energy_ratio": (
                lost_energy / source_energy if source_energy else 0.0
            ),
        }


class FakeR28:
    @staticmethod
    def _rotate(value: torch.Tensor, q0_math: object) -> torch.Tensor:
        return value


def _query() -> torch.Tensor:
    values = torch.linspace(-7.3, 6.7, 64 * 4, dtype=torch.float32)
    return values.reshape(1, 4, 64).to(torch.bfloat16)


def test_query_uses_four_real_e4m3_operands() -> None:
    high, high_correction, residual, residual_correction, metrics = R29._query_operands(
        FakeR26,
        FakeR28,
        object(),
        _query(),
        layer_id=0,
        label="test",
    )
    assert high.shape == high_correction.shape == residual.shape
    assert residual.shape == residual_correction.shape == (4, 64)
    assert set(metrics) == {
        "high",
        "high_correction",
        "residual",
        "residual_correction",
    }


def test_second_summand_reduces_query_reconstruction_error() -> None:
    _, _, primary, correction, _ = R29._query_operands(
        FakeR26,
        FakeR28,
        object(),
        _query(),
        layer_id=0,
        label="test",
    )
    exact = _query()[0].to(torch.float32)
    primary_error = torch.mean((primary - exact) ** 2)
    corrected_error = torch.mean((primary + correction - exact) ** 2)
    assert corrected_error < primary_error


@pytest.mark.parametrize(
    "bad",
    [torch.zeros((1, 4, 64), dtype=torch.float32), torch.zeros((4, 64))],
)
def test_query_rejects_wrong_source_dtype_or_geometry(bad: torch.Tensor) -> None:
    with pytest.raises(R29.EvidenceError):
        R29._query_operands(
            FakeR26,
            FakeR28,
            object(),
            bad,
            layer_id=0,
            label="bad",
        )


def test_query_rejects_correction_endpoint() -> None:
    class EndpointR26(FakeR26):
        @staticmethod
        def _float8_diagnostics(*args: object, label: str, **kwargs: object) -> dict:
            return {
                "label": label,
                "endpoint_count": int("correction" in label),
                "nonzero_to_zero_energy_ratio": 0.0,
            }

    with pytest.raises(CandidateNumericalMiss, match="endpoint"):
        R29._query_operands(
            EndpointR26,
            FakeR28,
            object(),
            _query(),
            layer_id=0,
            label="endpoint",
        )


def test_frozen_r28_identity_is_sealed() -> None:
    assert len(R29.EXPECTED_R28_SOURCE_SHA256) == 64
    assert R29.EXPERIMENT_ID == "a17-n10-e0-r29-q0-20260818"
