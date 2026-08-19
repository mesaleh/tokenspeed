#!/usr/bin/env python3
"""Evaluate R29 on the sealed R19 three-role, all-layer holdout."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

EXPERIMENT_ID = "a17-n10-e0-r29-q1-20260818"
DECISION_ADVANCE = "ADVANCE_R29_ALL_LAYER_TO_NATIVE_COMPONENT"
DECISION_REJECT = "REJECT_R29_Q1"
DECISION_NO_EVIDENCE = "NO_DECISION_EVIDENCE"

EXPECTED_R26_SOURCE_SHA256 = (
    "ae7457736ae7162bdc216b7ee7d8de4117fb1f46a8ea6b6e90a58c8dec4815cf"
)
EXPECTED_R29_Q0_SOURCE_SHA256 = (
    "c1b9453ae4ba52475a7184f1321ecb0fc669d390aede90e71e68d5130a6f63fd"
)
EXPECTED_R28_Q0_SOURCE_SHA256 = (
    "7ca4f2de7aa52b607ebb8d3bedc085453e3b8f7c41826f1b65af8e8a30e961cc"
)
EXPECTED_R19_SOURCE_SHA256 = (
    "ffbf548e9fea7231de409235fbaa312f731c050c73ad72e65956b9979e541723"
)
EXPECTED_R19_RESULT_SHA256 = (
    "e2a6f2a3d47f8c8d56f6d7387ed3769eafdf496de75bf5492bcc759ee6949412"
)
EXPECTED_ROLES = ("train", "validation", "test")
EXPECTED_LAYERS = tuple(range(61))
SURFACE_LABELS = (
    "prefill_final",
    "target_verify_q0",
    "target_verify_q1",
    "target_verify_q2",
    "target_verify_q3",
    "target_verify_q4",
)
EXPECTED_CELLS = len(EXPECTED_ROLES) * len(EXPECTED_LAYERS) * len(SURFACE_LABELS)
EXPECTED_BRIDGE_CELLS = 645
MAX_N8_RELATIVE_RATIO = 1.10
MAX_ABSOLUTE_P_LOST_ENERGY = 1.0e-6
BRIDGE_RELATIVE_TOLERANCE = 5.0e-6


class EvidenceError(RuntimeError):
    """Frozen inputs cannot support an R29-Q1 decision."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor_f32(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().to(torch.float32)
    return hashlib.sha256(value.numpy().astype("<f4", copy=False).tobytes()).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(_canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise EvidenceError(f"{label} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise EvidenceError(f"{label} hash differs: {actual} != {expected}")
    return actual


def _load_module(path: Path, name: str, expected: str) -> ModuleType:
    _require_hash(path, expected, name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvidenceError(f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _release_q0_contract_module(q0_math_source: Path, expected_hash: str) -> None:
    """Release q0_math's generic dependency before R19 imports its own."""
    contract = sys.modules.get("q0_contract")
    expected_path = q0_math_source.resolve().parent / "q0_contract.py"
    if contract is None or not getattr(contract, "__file__", None):
        raise EvidenceError("frozen q0_math contract module is not loaded")
    actual_path = Path(contract.__file__).resolve()
    if actual_path != expected_path.resolve():
        raise EvidenceError(
            f"frozen q0_math contract path differs: {actual_path} != {expected_path}"
        )
    _require_hash(actual_path, expected_hash, "frozen q0_math contract module")
    del sys.modules["q0_contract"]


def _load_r19(path: Path) -> ModuleType:
    _require_hash(path, EXPECTED_R19_SOURCE_SHA256, "frozen R19 evaluator")
    source_dir = path.resolve().parent
    sys.path.insert(0, str(source_dir))
    try:
        module = _load_module(path, "a17_r29_q1_frozen_r19", EXPECTED_R19_SOURCE_SHA256)
    finally:
        sys.path.remove(str(source_dir))
    for name, expected in (
        ("analyze_q0.py", module.ANALYZE_SHA256),
        ("q0_math.py", module.Q0_MATH_SHA256),
    ):
        _require_hash(source_dir / name, expected, f"frozen R19 {name}")
    return module


def _load_r19_result(run1: Path, run2: Path) -> tuple[dict[str, Any], str]:
    hash1 = _require_hash(run1, EXPECTED_R19_RESULT_SHA256, "R19 final run 1")
    hash2 = _require_hash(run2, EXPECTED_R19_RESULT_SHA256, "R19 final run 2")
    if run1.read_bytes() != run2.read_bytes():
        raise EvidenceError("R19 duplicate results are not byte-identical")
    result = json.loads(run1.read_text(encoding="utf-8"))
    if (
        result.get("experiment_id") != "a17-n10-e0-r19-20260818"
        or result.get("decision") != "REJECT_R19_HELDOUT_POLICY"
        or result.get("identity", {}).get("evaluator_sha256")
        != EXPECTED_R19_SOURCE_SHA256
    ):
        raise EvidenceError("R19 final result identity differs")
    return result, hash1 if hash1 == hash2 else ""


def _bridge_map(result: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, float]]:
    bridge: dict[tuple[str, int, str], dict[str, float]] = {}
    for role in EXPECTED_ROLES:
        rows = result.get(role)
        if not isinstance(rows, list):
            raise EvidenceError(f"R19 {role} result is not a list")
        for row in rows:
            layer = int(row["layer"])
            metrics = row.get("metrics", {})
            if not isinstance(metrics, dict):
                raise EvidenceError(
                    f"R19 {role} metrics are malformed at layer={layer}"
                )
            for label, cell in metrics.items():
                mse_ratio = float(cell["mse_to_n8_ratio"])
                kl_ratio = float(cell["attention_kl_to_n8_ratio"])
                if mse_ratio <= 0.0 or kl_ratio <= 0.0:
                    raise EvidenceError("R19 bridge ratio is nonpositive")
                key = (role, layer, str(label))
                if key in bridge:
                    raise EvidenceError(f"R19 bridge cell is duplicated: {key}")
                bridge[key] = {
                    "mse": float(cell["mse"]) / mse_ratio,
                    "attention_kl": float(cell["attention_kl"]) / kl_ratio,
                }
    if len(bridge) != EXPECTED_BRIDGE_CELLS:
        raise EvidenceError(
            f"R19 bridge cell count differs: {len(bridge)} != {EXPECTED_BRIDGE_CELLS}"
        )
    return bridge


def _positive_ratio(numerator: float, denominator: float, label: str) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator < 0.0
        or denominator <= 0.0
    ):
        raise EvidenceError(f"{label} ratio input is invalid")
    return numerator / denominator


def _relative_difference(observed: float, expected: float, label: str) -> float:
    if not math.isfinite(observed) or not math.isfinite(expected) or expected <= 0.0:
        raise EvidenceError(f"{label} bridge input is invalid")
    value = abs(observed - expected) / expected
    if value > BRIDGE_RELATIVE_TOLERANCE:
        raise EvidenceError(f"{label} bridge differs: {value}")
    return value


def _n8_scale_factorization(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    exponent = torch.floor(torch.log2(scale) + 0.5).to(torch.int32)
    if (exponent < -127).any() or (exponent > 127).any():
        raise EvidenceError("accepted-N8 split-scale exponent is outside UE8M0")
    power = torch.ldexp(torch.ones_like(scale), exponent)
    residual = (scale / power).to(torch.bfloat16).to(torch.float32)
    return power, residual


def _quality_metrics(
    *,
    score: torch.Tensor,
    dense_score: torch.Tensor,
    dense_output: torch.Tensor,
    dense_probability: torch.Tensor,
    raw: torch.Tensor,
    scale: torch.Tensor,
    attention_scale: float,
    r19: ModuleType,
    r18: ModuleType,
) -> dict[str, Any]:
    if not torch.isfinite(score).all():
        raise EvidenceError("R29-Q1 candidate score is nonfinite")
    output, p = r19._accepted_n8_output(score, raw, scale, attention_scale, r18)
    numerator = r18._score_numerator(score, attention_scale)
    probability = r18._probability(numerator)
    mse = float(
        torch.mean(
            (output.to(torch.float64) - dense_output.to(torch.float64)) ** 2
        ).item()
    )
    kl = float(r18._kl(dense_probability, probability))
    score_error = score.to(torch.float64) - dense_score.to(torch.float64)
    return {
        "mse": mse,
        "attention_kl": kl,
        "to_dense_score_rmse": float(torch.sqrt(torch.mean(score_error**2)).item()),
        "output_sha256": _sha256_tensor_f32(output),
        "score_sha256": _sha256_tensor_f32(score),
        "p": p,
    }


def _evaluate_layer(
    *,
    role: str,
    layer: int,
    records: Any,
    bridge: dict[tuple[str, int, str], dict[str, float]],
    r26: ModuleType,
    r19: ModuleType,
    r18: ModuleType,
    packer: ModuleType,
) -> dict[str, Any]:
    capture = r19.analyze._build_layer_capture(
        records, layer=layer, expected_length=r19.INPUT_LENGTHS[role]
    )
    signs = r19.q0_math.make_sign_contract("cpu")
    source = capture.k_nope[:, 0].to(torch.float32)
    carrier = r19.q0_math.quantize_rows(source, signs, family="k4")
    dense_values_all = r19.q0_math.rotate(source, signs)
    raw_all = carrier.raw.to(torch.float32)
    stored_scale = carrier.scale.to(torch.bfloat16)
    (
        candidate_scale,
        exact_reciprocal,
        normalized_high,
        stored_high,
        residual,
        zero,
    ) = r26._prepare_reciprocal_rope(capture.k_pe[:, 0], stored_scale, packer)
    if zero.any() and not torch.equal(raw_all[zero], torch.zeros_like(raw_all[zero])):
        raise EvidenceError(f"zero-scale latent row is nonzero: {role}:{layer}")
    high_metrics = r26._float8_diagnostics(
        normalized_high,
        stored_high,
        expected_dtype=torch.float8_e4m3fn,
        label=f"R29-Q1 high RoPE {role}:{layer}",
    )
    surfaces = []
    for label, q_nope, q_pe, q_inc, query_position in r19._surface_specs(capture):
        eligible = capture.positions <= query_position
        raw = raw_all[eligible]
        scale = candidate_scale[eligible]
        power, n8_residual_scale = _n8_scale_factorization(scale)
        qn = q_nope[0].to(torch.float32)
        qp = q_pe[0].to(torch.float32)
        qi = q_inc[0].to(torch.float32)
        qrot = r19.q0_math.e4m3fn(
            r19.q0_math.rotate(qn, signs),
            label=f"R29-Q1 rotated query {role}:{layer}:{label}",
        )
        key_rope = capture.k_pe[eligible, 0].to(torch.float32)
        key_rope_inc = capture.k_pe_inc[eligible, 0].to(torch.float32)
        dense_score = qn @ source[eligible].transpose(0, 1)
        dense_score += qp @ key_rope.transpose(0, 1)
        dense_numerator = r18._score_numerator(dense_score, capture.attention_scale)
        dense_probability = r18._probability(dense_numerator)
        dense_values = dense_values_all[eligible]
        dense_output = (dense_numerator @ dense_values) / dense_numerator.sum(
            dim=1, keepdim=True
        )

        n8_score = (qrot @ raw.transpose(0, 1)) * power.unsqueeze(0)
        n8_score *= n8_residual_scale.unsqueeze(0)
        n8_score += qi[:, 512:] @ key_rope_inc.transpose(0, 1)
        n8_metrics = _quality_metrics(
            score=n8_score,
            dense_score=dense_score,
            dense_output=dense_output,
            dense_probability=dense_probability,
            raw=raw,
            scale=power * n8_residual_scale,
            attention_scale=float(capture.attention_scale),
            r19=r19,
            r18=r18,
        )

        (
            q_high,
            q_high_correction,
            q_residual,
            q_residual_correction,
            query_metrics,
        ) = r26._query_rope_r29(q_pe, layer_id=layer, label=f"{role}:{label}")
        selected_high = stored_high[eligible].to(torch.float32)
        selected_residual = residual["raw"][eligible]
        selected_residual_scale = residual["scale"][eligible]
        candidate_score = r26._hierarchical_score(
            q_rot=qrot,
            raw=raw,
            q_high=q_high,
            high=selected_high,
            q_residual=q_residual,
            residual_raw=selected_residual,
            residual_scale=selected_residual_scale,
            scale=scale,
        )
        groups_count, group_size = r26._require_residual_geometry()
        high_correction_score = q_high_correction @ selected_high.transpose(0, 1)
        residual_correction_groups = torch.einsum(
            "hgd,tgd->htg",
            q_residual_correction.reshape(-1, groups_count, group_size),
            selected_residual.reshape(raw.shape[0], groups_count, group_size),
        )
        residual_correction_score = torch.sum(
            residual_correction_groups * selected_residual_scale.unsqueeze(0),
            dim=-1,
        )
        candidate_score += (
            high_correction_score + residual_correction_score
        ) * scale.unsqueeze(0)
        candidate = _quality_metrics(
            score=candidate_score,
            dense_score=dense_score,
            dense_output=dense_output,
            dense_probability=dense_probability,
            raw=raw,
            scale=scale,
            attention_scale=float(capture.attention_scale),
            r19=r19,
            r18=r18,
        )
        mse_ratio = _positive_ratio(candidate["mse"], n8_metrics["mse"], "MSE")
        kl_ratio = _positive_ratio(
            candidate["attention_kl"], n8_metrics["attention_kl"], "attention KL"
        )
        bridge_key = (role, layer, label)
        bridge_relative = None
        if bridge_key in bridge:
            expected = bridge[bridge_key]
            bridge_relative = {
                "mse": _relative_difference(
                    n8_metrics["mse"], expected["mse"], f"{bridge_key}:MSE"
                ),
                "attention_kl": _relative_difference(
                    n8_metrics["attention_kl"],
                    expected["attention_kl"],
                    f"{bridge_key}:KL",
                ),
            }
        lost_energy = float(candidate["p"]["nonzero_to_zero_energy_ratio"])
        surface_pass = (
            mse_ratio <= MAX_N8_RELATIVE_RATIO
            and kl_ratio <= MAX_N8_RELATIVE_RATIO
            and lost_energy <= MAX_ABSOLUTE_P_LOST_ENERGY
            and candidate["p"]["lost_energy_gate_pass"]
            and candidate["p"]["saturation_gate_pass"]
            and all(
                diagnostic["endpoint_count"] == 0
                for diagnostic in query_metrics.values()
            )
        )
        surfaces.append(
            {
                "label": label,
                "query_position": int(query_position),
                "causal_key_count": int(eligible.sum().item()),
                "accepted_n8": n8_metrics,
                "r29": candidate,
                "r29_to_n8_mse_ratio": mse_ratio,
                "r29_to_n8_attention_kl_ratio": kl_ratio,
                "bridge_relative_difference": bridge_relative,
                "query_rope": query_metrics,
                "surface_pass": surface_pass,
            }
        )

    residual_metrics = residual["diagnostics"]
    layer_pass = (
        high_metrics["finite"]
        and high_metrics["endpoint_count"] == 0
        and high_metrics["raw_bit_reconstruction_bit_identical"]
        and residual_metrics["clipped_coordinates"] == 0
        and residual_metrics["packed_round_trip_bit_identical"]
        and all(surface["surface_pass"] for surface in surfaces)
    )
    return {
        "layer": layer,
        "zero_scale_rows": int(zero.sum().item()),
        "high_rope": high_metrics,
        "residual_rope": residual_metrics,
        "surfaces": surfaces,
        "layer_pass": layer_pass,
    }


def _validate_complete_roles(roles: list[dict[str, Any]]) -> int:
    try:
        cells = {
            (str(role["role"]), int(layer["layer"]), str(surface["label"]))
            for role in roles
            for layer in role["layers"]
            for surface in layer["surfaces"]
        }
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError("R29-Q1 role/cell structure is malformed") from error
    expected = {
        (role, layer, surface)
        for role in EXPECTED_ROLES
        for layer in EXPECTED_LAYERS
        for surface in SURFACE_LABELS
    }
    if cells != expected:
        raise EvidenceError("R29-Q1 evaluated cell set differs")
    return len(cells)


def _decision(cell_count: int, all_pass: bool, bridge_count: int) -> str:
    if cell_count != EXPECTED_CELLS or bridge_count != EXPECTED_BRIDGE_CELLS:
        return DECISION_REJECT
    return DECISION_ADVANCE if all_pass else DECISION_REJECT


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if torch.get_num_threads() != 1:
        raise EvidenceError("R29-Q1 requires exactly one Torch thread")
    r26 = _load_module(
        args.r26_source, "a17_r29_q1_frozen_r26", EXPECTED_R26_SOURCE_SHA256
    )
    r29 = _load_module(
        args.r29_q0_source,
        "a17_r29_q1_frozen_r29_q0",
        EXPECTED_R29_Q0_SOURCE_SHA256,
    )
    r28 = _load_module(
        args.r28_source,
        "a17_r29_q1_frozen_r28_q0",
        EXPECTED_R28_Q0_SOURCE_SHA256,
    )
    q0_math = r28._load_q0_math(args.q0_math_source)
    transform_test = r28._transform_dot_self_test(q0_math)
    r28._install_r28(r26, q0_math)
    r29._install_r29(r28, r26, q0_math)
    r26._query_rope_r29 = lambda q_pe, *, layer_id, label: r29._query_operands(
        r26, r28, q0_math, q_pe, layer_id=layer_id, label=label
    )
    r26.EXPERIMENT_ID = EXPERIMENT_ID
    r26.DECISION_ADVANCE = DECISION_ADVANCE
    r26.DECISION_REJECT = DECISION_REJECT
    r26.DECISION_NO_EVIDENCE = DECISION_NO_EVIDENCE
    _release_q0_contract_module(args.q0_math_source, r28.EXPECTED_Q0_CONTRACT_SHA256)
    r19 = _load_r19(args.r19_source)
    r18 = r19._load_module(
        args.r18_source, "a17_r29_q1_frozen_r18", r19.R18_SOURCE_SHA256
    )
    packer = r19._load_module(
        args.packer_source, "a17_r29_q1_frozen_packer", r19.PACKER_SHA256
    )
    if not packer._self_test().get("pass"):
        raise EvidenceError("frozen E2M1 packer self-test failed")
    r19_result, r19_result_hash = _load_r19_result(args.r19_result_1, args.r19_result_2)
    bridge = _bridge_map(r19_result)

    role_paths = {
        "train": (args.train_root, args.train_transport_receipt),
        "validation": (args.validation_root, args.validation_transport_receipt),
        "test": (args.test_root, args.test_transport_receipt),
    }
    roles = []
    bridge_seen: set[tuple[str, int, str]] = set()
    for role in EXPECTED_ROLES:
        root, transport_receipt = role_paths[role]
        records, identity = r19._load_role(root, role, transport_receipt)
        layers = []
        for layer in EXPECTED_LAYERS:
            result = _evaluate_layer(
                role=role,
                layer=layer,
                records=records,
                bridge=bridge,
                r26=r26,
                r19=r19,
                r18=r18,
                packer=packer,
            )
            for surface in result["surfaces"]:
                if surface["bridge_relative_difference"] is not None:
                    bridge_seen.add((role, layer, surface["label"]))
            layers.append(result)
            gc.collect()
        roles.append(
            {
                "role": role,
                "identity": identity,
                "layers": layers,
                "role_pass": all(layer["layer_pass"] for layer in layers),
            }
        )
        del records
        gc.collect()

    cell_count = _validate_complete_roles(roles)
    if bridge_seen != set(bridge):
        raise EvidenceError("R29-Q1 did not reproduce every available R19 bridge cell")
    surfaces = [
        surface
        for role in roles
        for layer in role["layers"]
        for surface in layer["surfaces"]
    ]
    all_pass = all(surface["surface_pass"] for surface in surfaces) and all(
        role["role_pass"] for role in roles
    )
    decision = _decision(cell_count, all_pass, len(bridge_seen))
    bridge_values = [
        value
        for surface in surfaces
        if surface["bridge_relative_difference"] is not None
        for value in surface["bridge_relative_difference"].values()
    ]
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "offline_all_layer_falsifier": True,
        "cannot_confirm_native_speed_or_live_memory": True,
        "post_rope_transform_self_test": transform_test,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
        "contract": {
            "roles": list(EXPECTED_ROLES),
            "layers": list(EXPECTED_LAYERS),
            "surfaces": list(SURFACE_LABELS),
            "expected_cells": EXPECTED_CELLS,
            "expected_bridge_cells": EXPECTED_BRIDGE_CELLS,
            "max_n8_relative_ratio": MAX_N8_RELATIVE_RATIO,
            "max_absolute_p_lost_energy": MAX_ABSOLUTE_P_LOST_ENERGY,
            "bridge_relative_tolerance": BRIDGE_RELATIVE_TOLERANCE,
            "logical_persistent_row_bytes": 356,
            "dense_row_bytes": 576,
            "projected_target_row_saving_percent": (1.0 - 356.0 / 576.0) * 100.0,
            "post_rope_transform": (
                "seed-42 randomized normalized-Hadamard applied identically to "
                "post-RoPE query and reciprocal key"
            ),
            "query_rope_format": (
                "two E4M3 summands for both factor-16 and unity query operands"
            ),
            "transient_query_correction_bytes_per_head": 128,
            "additional_rope_k32_instructions_per_score_tile": 4,
            "transform_dimension": r28.TRANSFORM_DIMENSION,
            "transform_signs1_sha256": r28.EXPECTED_SIGNS1_SHA256,
            "transform_signs2_sha256": r28.EXPECTED_SIGNS2_SHA256,
        },
        "identity": {
            "evaluator_sha256": _sha256_file(Path(__file__)),
            "r26_q0_source_sha256": EXPECTED_R26_SOURCE_SHA256,
            "r29_q0_source_sha256": EXPECTED_R29_Q0_SOURCE_SHA256,
            "r28_q0_source_sha256": EXPECTED_R28_Q0_SOURCE_SHA256,
            "q0_math_source_sha256": r28.EXPECTED_Q0_MATH_SHA256,
            "q0_contract_source_sha256": r28.EXPECTED_Q0_CONTRACT_SHA256,
            "r19_source_sha256": EXPECTED_R19_SOURCE_SHA256,
            "r19_result_sha256": r19_result_hash,
            "r18_source_sha256": r19.R18_SOURCE_SHA256,
            "packer_sha256": r19.PACKER_SHA256,
        },
        "roles": roles,
        "summary": {
            "cells": cell_count,
            "bridge_cells": len(bridge_seen),
            "all_pass": all_pass,
            "passing_cells": sum(surface["surface_pass"] for surface in surfaces),
            "failing_cells": sum(not surface["surface_pass"] for surface in surfaces),
            "max_r29_to_n8_mse_ratio": max(
                surface["r29_to_n8_mse_ratio"] for surface in surfaces
            ),
            "max_r29_to_n8_attention_kl_ratio": max(
                surface["r29_to_n8_attention_kl_ratio"] for surface in surfaces
            ),
            "max_absolute_r29_p_lost_energy": max(
                surface["r29"]["p"]["nonzero_to_zero_energy_ratio"]
                for surface in surfaces
            ),
            "max_bridge_relative_difference": max(bridge_values),
            "high_endpoint_count": sum(
                layer["high_rope"]["endpoint_count"]
                for role in roles
                for layer in role["layers"]
            ),
            "query_endpoint_count": sum(
                sum(
                    diagnostic["endpoint_count"]
                    for diagnostic in surface["query_rope"].values()
                )
                for surface in surfaces
            ),
            "residual_clipped_coordinates": sum(
                layer["residual_rope"]["clipped_coordinates"]
                for role in roles
                for layer in role["layers"]
            ),
        },
        "decision": decision,
    }


def _add_path(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(
        f"--{name.replace('_', '-')}", dest=name, type=Path, required=True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "r29_q0_source",
        "r28_source",
        "q0_math_source",
        "r26_source",
        "r19_source",
        "r18_source",
        "packer_source",
        "r19_result_1",
        "r19_result_2",
        "train_root",
        "train_transport_receipt",
        "validation_root",
        "validation_transport_receipt",
        "test_root",
        "test_transport_receipt",
        "output",
    ):
        _add_path(parser, name)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        result = evaluate(args)
    except (
        EvidenceError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        result = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "error_type": type(error).__name__,
            "error": str(error),
            "decision": DECISION_NO_EVIDENCE,
            "evaluator_sha256": _sha256_file(Path(__file__)),
        }
    _atomic_json(args.output, result)
    print(_canonical_json_bytes(result).decode("utf-8"))
    return 0 if result["decision"] == DECISION_ADVANCE else 2


if __name__ == "__main__":
    raise SystemExit(main())
