#!/usr/bin/env python3
"""Evaluate two-summand E4M3 queries on the frozen R28 Q0 gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

EXPERIMENT_ID = "a17-n10-e0-r29-q0-20260818"
DECISION_ADVANCE = "ADVANCE_R29_TWO_SUMMAND_QUERY_TO_Q1"
DECISION_REJECT = "REJECT_R29_Q0"
DECISION_NO_EVIDENCE = "NO_DECISION_EVIDENCE"

EXPECTED_R28_SOURCE_SHA256 = (
    "7ca4f2de7aa52b607ebb8d3bedc085453e3b8f7c41826f1b65af8e8a30e961cc"
)


class EvidenceError(RuntimeError):
    """Frozen inputs cannot support an R29-Q0 decision."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _load_module(path: Path, name: str, expected: str) -> ModuleType:
    if not path.is_file():
        raise EvidenceError(f"{name} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise EvidenceError(f"{name} hash differs: {actual} != {expected}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvidenceError(f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _query_operands(
    r26: ModuleType,
    r28: ModuleType,
    q0_math: ModuleType,
    q_pe: torch.Tensor,
    *,
    layer_id: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if q_pe.dtype != torch.bfloat16:
        raise EvidenceError(
            f"R29 query RoPE must originate as BF16: {layer_id}:{label}"
        )
    if q_pe.ndim != 3 or q_pe.shape[0] != 1 or q_pe.shape[2] != 64:
        raise EvidenceError(
            f"R29 query RoPE geometry differs: {layer_id}:{label}:{tuple(q_pe.shape)}"
        )
    exact = r28._rotate(q_pe[0].to(torch.float32), q0_math)
    shifted = exact * r26._require_normalization_factor()
    high_primary = shifted.to(torch.float8_e4m3fn)
    high_correction_exact = shifted - high_primary.to(torch.float32)
    high_correction = high_correction_exact.to(torch.float8_e4m3fn)
    residual_primary = exact.to(torch.float8_e4m3fn)
    residual_correction_exact = exact - residual_primary.to(torch.float32)
    residual_correction = residual_correction_exact.to(torch.float8_e4m3fn)

    operands = (
        ("high", shifted, high_primary),
        ("high_correction", high_correction_exact, high_correction),
        ("residual", exact, residual_primary),
        ("residual_correction", residual_correction_exact, residual_correction),
    )
    diagnostics: dict[str, Any] = {}
    for name, source, stored in operands:
        if not torch.isfinite(stored.to(torch.float32)).all():
            raise r26.CandidateNumericalMiss(
                f"R29 {name} query operand is nonfinite: {layer_id}:{label}"
            )
        diagnostic = r26._float8_diagnostics(
            source,
            stored,
            expected_dtype=torch.float8_e4m3fn,
            label=f"R29 {name} E4M3 query RoPE {layer_id}:{label}",
        )
        if diagnostic["endpoint_count"] != 0:
            raise r26.CandidateNumericalMiss(
                f"R29 {name} query reaches an endpoint: {layer_id}:{label}"
            )
        diagnostics[name] = diagnostic

    return (
        high_primary.to(torch.float32),
        high_correction.to(torch.float32),
        residual_primary.to(torch.float32),
        residual_correction.to(torch.float32),
        diagnostics,
    )


def _install_r29(r28: ModuleType, r26: ModuleType, q0_math: ModuleType) -> None:
    original_surface = r26._surface_quality

    def surface_quality(**kwargs: Any) -> dict[str, Any]:
        base = original_surface(**kwargs)
        layer_id = int(kwargs["layer_id"])
        label = str(kwargs["label"])
        capture = kwargs["capture"]
        query_position = int(kwargs["query_position"])
        eligible = capture.positions <= query_position
        raw = kwargs["raw"][eligible]
        scale = kwargs["scale"][eligible]
        exact_reciprocal = kwargs["exact_reciprocal"][eligible]
        high = kwargs["stored_high"][eligible].to(torch.float32)
        residual_raw = kwargs["residual"]["raw"][eligible]
        residual_scale = kwargs["residual"]["scale"][eligible]
        dense_values = kwargs["dense_values"][eligible]
        n7 = kwargs["n7"]

        (
            high_primary,
            high_correction,
            residual_primary,
            residual_correction,
            metrics,
        ) = _query_operands(
            r26,
            r28,
            q0_math,
            kwargs["q_pe"],
            layer_id=layer_id,
            label=label,
        )
        qn = kwargs["q_nope"][0].to(torch.float32)
        q_rot = n7.e4m3fn(
            n7.rotate(qn, kwargs["signs"]), label="R29 rotated latent query"
        )
        candidate_score = r26._hierarchical_score(
            q_rot=q_rot,
            raw=raw,
            q_high=high_primary,
            high=high,
            q_residual=residual_primary,
            residual_raw=residual_raw,
            residual_scale=residual_scale,
            scale=scale,
        )
        high_correction_score = high_correction @ high.transpose(0, 1)
        groups_count, group_size = r26._require_residual_geometry()
        residual_correction_groups = torch.einsum(
            "hgd,tgd->htg",
            residual_correction.reshape(-1, groups_count, group_size),
            residual_raw.reshape(raw.shape[0], groups_count, group_size),
        )
        residual_correction_score = torch.sum(
            residual_correction_groups * residual_scale.unsqueeze(0), dim=-1
        )
        candidate_score += (
            high_correction_score + residual_correction_score
        ) * scale.unsqueeze(0)

        q_original = kwargs["q_pe"][0].to(torch.float32)
        key_original = capture.k_pe[eligible, 0].to(torch.float32)
        key_nope = capture.k_nope[eligible, 0].to(torch.float32)
        dense_score = qn @ key_nope.transpose(0, 1)
        dense_score += q_original @ key_original.transpose(0, 1)
        dense_numerator = n7._score_numerator(dense_score, capture.attention_scale)
        dense_probability = n7._probability(dense_numerator)
        dense_output = (dense_numerator @ dense_values) / dense_numerator.sum(
            dim=1, keepdim=True
        )
        arm_args = {
            "dense_score": dense_score,
            "dense_output": dense_output,
            "dense_probability": dense_probability,
            "raw_values": raw,
            "reconstructed_scale": scale,
            "attention_scale": capture.attention_scale,
            "n7": n7,
        }
        candidate = r26._quality_arm(
            score=candidate_score,
            label="R29 two-summand rotated query candidate",
            **arm_args,
        )
        query_only_score = r26._reciprocal_score(
            q_rot=q_rot,
            raw=raw,
            q_rope=residual_primary + residual_correction,
            reciprocal_rope=exact_reciprocal,
            scale=scale,
        )
        query_only = r26._quality_arm(
            score=query_only_score,
            label="R29 exact-key two-summand-query diagnostic",
            **arm_args,
        )
        n8 = base["accepted_n8"]
        mse_ratio = n7._positive_ratio(
            candidate["mse"], n8["mse"], label="R29/N8 complete output MSE"
        )
        kl_ratio = n7._positive_ratio(
            candidate["attention_kl"],
            n8["attention_kl"],
            label="R29/N8 attention KL",
        )
        lost_energy = float(candidate["p"]["nonzero_to_zero_energy_ratio"])
        endpoints = sum(value["endpoint_count"] for value in metrics.values())
        surface_pass = (
            mse_ratio <= r26.MAX_N8_RELATIVE_RATIO
            and kl_ratio <= r26.MAX_N8_RELATIVE_RATIO
            and lost_energy <= r26.MAX_ABSOLUTE_P_LOST_ENERGY
            and candidate["p"]["lost_energy_gate_pass"]
            and candidate["p"]["saturation_gate_pass"]
            and endpoints == 0
        )

        def public(arm: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in arm.items()
                if key not in {"score", "output", "probability"}
            }

        base["identity"]["r26_score_sha256"] = r26._sha256_tensor_f32(candidate_score)
        base["identity"]["r26_output_sha256"] = r26._sha256_tensor_f32(
            candidate["output"]
        )
        base["identity"]["query_only_score_sha256"] = r26._sha256_tensor_f32(
            query_only_score
        )
        base["query_rope"] = metrics
        base["exact_key_quantized_query"] = public(query_only)
        base["r26"] = public(candidate)
        base["r26_to_n8_mse_ratio"] = mse_ratio
        base["r26_to_n8_attention_kl_ratio"] = kl_ratio
        base["output_boundary_pass"] = mse_ratio <= r26.MAX_N8_RELATIVE_RATIO
        base["attention_kl_boundary_pass"] = kl_ratio <= r26.MAX_N8_RELATIVE_RATIO
        base["absolute_lost_energy_boundary_pass"] = (
            lost_energy <= r26.MAX_ABSOLUTE_P_LOST_ENERGY
        )
        base["surface_pass"] = surface_pass
        return base

    r26.EXPERIMENT_ID = EXPERIMENT_ID
    r26.DECISION_ADVANCE = DECISION_ADVANCE
    r26.DECISION_REJECT = DECISION_REJECT
    r26.DECISION_NO_EVIDENCE = DECISION_NO_EVIDENCE
    r26._surface_quality = surface_quality


def _finalize_result(result: dict[str, Any], r28: ModuleType) -> dict[str, Any]:
    result["experiment_id"] = EXPERIMENT_ID
    result["evaluator_sha256"] = _sha256_file(Path(__file__))
    result["r28_frozen_source_sha256"] = EXPECTED_R28_SOURCE_SHA256
    contract = result.get("contract")
    if isinstance(contract, dict):
        contract["query_rope_format"] = (
            "seed-42 randomized normalized-Hadamard post-RoPE, then two "
            "E4M3 summands for both factor-16 and unity query operands"
        )
        contract["logical_persistent_row_bytes"] = 356
        contract["transient_query_correction_bytes_per_head"] = 128
        contract["additional_rope_k32_instructions_per_score_tile"] = 4

    surfaces = [
        surface
        for layer in result.get("layers", [])
        for surface in layer.get("surfaces", [])
    ]
    for surface in surfaces:
        identity = surface.get("identity", {})
        if "r28_score_sha256" in identity:
            identity["r29_score_sha256"] = identity.pop("r28_score_sha256")
        if "r28_output_sha256" in identity:
            identity["r29_output_sha256"] = identity.pop("r28_output_sha256")
        if "r28" in surface:
            surface["r29"] = surface.pop("r28")
        if "r28_to_n8_mse_ratio" in surface:
            surface["r29_to_n8_mse_ratio"] = surface.pop("r28_to_n8_mse_ratio")
        if "r28_to_n8_attention_kl_ratio" in surface:
            surface["r29_to_n8_attention_kl_ratio"] = surface.pop(
                "r28_to_n8_attention_kl_ratio"
            )

    summary = result.get("summary")
    if isinstance(summary, dict):
        for old, new in (
            ("max_r28_to_n8_mse_ratio", "max_r29_to_n8_mse_ratio"),
            (
                "max_r28_to_n8_attention_kl_ratio",
                "max_r29_to_n8_attention_kl_ratio",
            ),
            ("max_r28_to_dense_score_rmse", "max_r29_to_dense_score_rmse"),
            (
                "max_absolute_r28_p_lost_energy",
                "max_absolute_r29_p_lost_energy",
            ),
        ):
            if old in summary:
                summary[new] = summary.pop(old)
        query_diagnostics = [
            diagnostic
            for surface in surfaces
            for diagnostic in surface["query_rope"].values()
        ]
        summary["query_endpoint_count"] = sum(
            value["endpoint_count"] for value in query_diagnostics
        )
        summary["max_query_nonzero_to_zero_energy_ratio"] = max(
            value["nonzero_to_zero_energy_ratio"] for value in query_diagnostics
        )
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if torch.get_num_threads() != 1:
        raise EvidenceError("R29-Q0 requires exactly one Torch thread")
    r28 = _load_module(
        args.r28_source, "a17_r29_frozen_r28", EXPECTED_R28_SOURCE_SHA256
    )
    original_install = r28._install_r28

    def install(r26: ModuleType, q0_math: ModuleType) -> None:
        original_install(r26, q0_math)
        _install_r29(r28, r26, q0_math)

    r28._install_r28 = install
    return _finalize_result(r28.evaluate(args), r28)


def _add_path(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(
        f"--{name.replace('_', '-')}", dest=name, type=Path, required=True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "r28_source",
        "r26_source",
        "q0_math_source",
        "capture_root",
        "n8_evaluator",
        "accepted_n8_result",
        "n8_operands",
        "n8_scores_1",
        "n8_metadata_1",
        "n8_run_identity_1",
        "n8_scores_2",
        "n8_metadata_2",
        "n8_run_identity_2",
        "n8_runner_source",
        "packer_source",
        "output",
    ):
        _add_path(parser, name)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        result = evaluate(args)
    except RuntimeError as error:
        valid_numerical_miss = type(error).__name__ == "CandidateNumericalMiss"
        result = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "valid_numerical_miss": valid_numerical_miss,
            "error_type": type(error).__name__,
            "error": str(error),
            "decision": (
                DECISION_REJECT if valid_numerical_miss else DECISION_NO_EVIDENCE
            ),
            "evaluator_sha256": _sha256_file(Path(__file__)),
        }
    except (
        EvidenceError,
        OSError,
        AttributeError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        result = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "valid_numerical_miss": False,
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
