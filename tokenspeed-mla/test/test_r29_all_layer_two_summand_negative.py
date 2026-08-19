#!/usr/bin/env python3
"""Fail-closed tests for the R29 all-layer evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SOURCE = Path(__file__).with_name("evaluate_r29_all_layer_two_summand_quality.py")
SPEC = importlib.util.spec_from_file_location("r29_all_layer_quality", SOURCE)
assert SPEC is not None and SPEC.loader is not None
R29 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R29
SPEC.loader.exec_module(R29)


def test_frozen_source_identities_are_sealed() -> None:
    assert len(R29.EXPECTED_R26_SOURCE_SHA256) == 64
    assert len(R29.EXPECTED_R28_Q0_SOURCE_SHA256) == 64
    assert len(R29.EXPECTED_R29_Q0_SOURCE_SHA256) == 64
    assert R29.EXPERIMENT_ID == "a17-n10-e0-r29-q1-20260818"


@pytest.mark.parametrize(
    ("cells", "passed", "bridges", "expected"),
    [
        (R29.EXPECTED_CELLS, True, R29.EXPECTED_BRIDGE_CELLS, R29.DECISION_ADVANCE),
        (R29.EXPECTED_CELLS, False, R29.EXPECTED_BRIDGE_CELLS, R29.DECISION_REJECT),
        (
            R29.EXPECTED_CELLS - 1,
            True,
            R29.EXPECTED_BRIDGE_CELLS,
            R29.DECISION_REJECT,
        ),
        (
            R29.EXPECTED_CELLS,
            True,
            R29.EXPECTED_BRIDGE_CELLS - 1,
            R29.DECISION_REJECT,
        ),
    ],
)
def test_decision_requires_complete_cells_and_bridges(
    cells: int, passed: bool, bridges: int, expected: str
) -> None:
    assert R29._decision(cells, passed, bridges) == expected


def _complete_roles() -> list[dict]:
    return [
        {
            "role": role,
            "layers": [
                {
                    "layer": layer,
                    "surfaces": [{"label": label} for label in R29.SURFACE_LABELS],
                }
                for layer in R29.EXPECTED_LAYERS
            ],
        }
        for role in R29.EXPECTED_ROLES
    ]


def test_complete_role_matrix_is_exact() -> None:
    assert R29._validate_complete_roles(_complete_roles()) == R29.EXPECTED_CELLS


def test_missing_surface_is_no_evidence() -> None:
    roles = _complete_roles()
    roles[-1]["layers"][-1]["surfaces"].pop()
    with pytest.raises(R29.EvidenceError, match="cell set differs"):
        R29._validate_complete_roles(roles)


def test_release_q0_contract_is_path_and_hash_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / "q0_contract.py"
    contract.write_text("SEALED = True\n", encoding="utf-8")
    module = ModuleType("q0_contract")
    module.__file__ = str(contract)
    monkeypatch.setitem(sys.modules, "q0_contract", module)
    expected = hashlib.sha256(contract.read_bytes()).hexdigest()
    R29._release_q0_contract_module(tmp_path / "q0_math.py", expected)
    assert "q0_contract" not in sys.modules


def test_release_q0_contract_rejects_foreign_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / "q0_contract.py"
    foreign = tmp_path / "foreign.py"
    contract.write_text("SEALED = True\n", encoding="utf-8")
    foreign.write_text("SEALED = False\n", encoding="utf-8")
    module = ModuleType("q0_contract")
    module.__file__ = str(foreign)
    monkeypatch.setitem(sys.modules, "q0_contract", module)
    expected = hashlib.sha256(contract.read_bytes()).hexdigest()
    with pytest.raises(R29.EvidenceError, match="path differs"):
        R29._release_q0_contract_module(tmp_path / "q0_math.py", expected)


def test_thread_count_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R29.torch, "get_num_threads", lambda: 2)
    with pytest.raises(R29.EvidenceError, match="exactly one"):
        R29.evaluate(SimpleNamespace())
