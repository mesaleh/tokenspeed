#!/usr/bin/env python3
"""Evaluate randomized-Hadamard R26 RoPE on the frozen 18-cell gate."""

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

EXPERIMENT_ID = "a17-n10-e0-r28-q0-20260818"
DECISION_ADVANCE = "ADVANCE_R28_ROTATED_HIERARCHICAL_TO_Q1"
DECISION_REJECT = "REJECT_R28_Q0"
DECISION_NO_EVIDENCE = "NO_DECISION_EVIDENCE"

EXPECTED_R26_SOURCE_SHA256 = (
    "ae7457736ae7162bdc216b7ee7d8de4117fb1f46a8ea6b6e90a58c8dec4815cf"
)
EXPECTED_Q0_MATH_SHA256 = (
    "4d0d7bc3545d7486c0760b75335c4bfbe855d1fd400e7fdb69fdcd401dc3e9c8"
)
EXPECTED_Q0_CONTRACT_SHA256 = (
    "b4596ef6de2f9cf68848a16e21202689e59e209490224b8d900a2a32655e3a94"
)
EXPECTED_SIGNS1_SHA256 = (
    "33719ab2b6cda4377e80d48774ff29ea6a0b1ac5b2c303ae9cdee164a2a0096d"
)
EXPECTED_SIGNS2_SHA256 = (
    "03d91dbd6c7a82475724210621127a699e6b9a28c2b863c26958cafa61cb0e65"
)
TRANSFORM_DIMENSION = 64
MAX_TRANSFORM_DOT_SCALED_ABSOLUTE_ERROR = 2.0e-6


class EvidenceError(RuntimeError):
    """Frozen inputs cannot support an R28-Q0 decision."""


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


def _load_q0_math(path: Path) -> ModuleType:
    source_dir = path.resolve().parent
    _require_hash(
        source_dir / "q0_contract.py",
        EXPECTED_Q0_CONTRACT_SHA256,
        "frozen q0_math contract",
    )
    sys.path.insert(0, str(source_dir))
    try:
        return _load_module(path, "a17_r28_frozen_q0_math", EXPECTED_Q0_MATH_SHA256)
    finally:
        sys.path.remove(str(source_dir))


def _signs(q0_math: ModuleType, device: torch.device | str) -> Any:
    signs = q0_math.make_sign_contract(device, dim=TRANSFORM_DIMENSION)
    if (
        signs.signs1_sha256 != EXPECTED_SIGNS1_SHA256
        or signs.signs2_sha256 != EXPECTED_SIGNS2_SHA256
    ):
        raise EvidenceError("R28 dimension-64 sign identity differs")
    return signs


def _rotate(x: torch.Tensor, q0_math: ModuleType) -> torch.Tensor:
    if x.shape[-1] != TRANSFORM_DIMENSION:
        raise EvidenceError(f"R28 transform geometry differs: {tuple(x.shape)}")
    transformed = q0_math.rotate(x.to(torch.float32), _signs(q0_math, x.device))
    if not torch.isfinite(transformed).all():
        raise EvidenceError("R28 transformed tensor is nonfinite")
    return transformed


def _transform_dot_self_test(q0_math: ModuleType) -> dict[str, Any]:
    q = torch.arange(1, 1 + 3 * TRANSFORM_DIMENSION, dtype=torch.float32).reshape(
        3, TRANSFORM_DIMENSION
    )
    q = torch.sin(q * 0.071) + torch.cos(q * 0.013)
    k = torch.arange(1, 1 + 5 * TRANSFORM_DIMENSION, dtype=torch.float32).reshape(
        5, TRANSFORM_DIMENSION
    )
    k = torch.sin(k * 0.037) - torch.cos(k * 0.019)
    expected = q @ k.transpose(0, 1)
    observed = _rotate(q, q0_math) @ _rotate(k, q0_math).transpose(0, 1)
    maximum_absolute_error = float(torch.max(torch.abs(observed - expected)).item())
    scale = max(1.0, float(torch.max(torch.abs(expected)).item()))
    scaled_absolute_error = maximum_absolute_error / scale
    if scaled_absolute_error > MAX_TRANSFORM_DOT_SCALED_ABSOLUTE_ERROR:
        raise EvidenceError(
            "R28 transform dot-product error exceeds tolerance: "
            f"{scaled_absolute_error}"
        )
    return {
        "pass": True,
        "dimension": TRANSFORM_DIMENSION,
        "signs1_sha256": EXPECTED_SIGNS1_SHA256,
        "signs2_sha256": EXPECTED_SIGNS2_SHA256,
        "maximum_absolute_error": maximum_absolute_error,
        "scaled_absolute_error": scaled_absolute_error,
        "maximum_scaled_absolute_error": (MAX_TRANSFORM_DOT_SCALED_ABSOLUTE_ERROR),
    }


def _install_r28(r26: ModuleType, q0_math: ModuleType) -> None:
    def prepare(
        key_rope_bf16: torch.Tensor,
        token_scale_bf16: torch.Tensor,
        packer: ModuleType,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
        torch.Tensor,
    ]:
        if key_rope_bf16.dtype != torch.bfloat16:
            raise EvidenceError(
                f"R28 key RoPE must originate as BF16, got {key_rope_bf16.dtype}"
            )
        if token_scale_bf16.dtype != torch.bfloat16:
            raise EvidenceError(
                "R28 scale must originate as BF16, got " f"{token_scale_bf16.dtype}"
            )
        if key_rope_bf16.ndim != 2 or key_rope_bf16.shape[1] != TRANSFORM_DIMENSION:
            raise EvidenceError(
                f"R28 key RoPE geometry differs: {tuple(key_rope_bf16.shape)}"
            )
        if token_scale_bf16.shape != (key_rope_bf16.shape[0],):
            raise EvidenceError("R28 scale geometry differs")

        key = key_rope_bf16.to(torch.float32)
        scale = token_scale_bf16.to(torch.float32)
        if not torch.isfinite(key).all() or not torch.isfinite(scale).all():
            raise EvidenceError("R28 source key/scale is nonfinite")
        if (scale < 0).any():
            raise EvidenceError("R28 token scale is negative")
        zero = scale == 0
        candidate_scale = scale.clone()
        candidate_scale[zero] = 1.0
        exact = _rotate(key / candidate_scale.unsqueeze(1), q0_math)
        normalized = exact / r26._require_normalization_factor()
        high = normalized.to(torch.float8_e4m3fn)
        if (
            not torch.isfinite(normalized).all()
            or not torch.isfinite(high.to(torch.float32)).all()
        ):
            raise r26.CandidateNumericalMiss(
                "R28 normalized rotated E4M3 high RoPE overflows or is nonfinite"
            )
        residual_exact = (
            exact - high.to(torch.float32) * r26._require_normalization_factor()
        )
        residual = r26._quantize_residual_group32(residual_exact, packer)
        return candidate_scale, exact, normalized, high, residual, zero

    def query(
        q_pe: torch.Tensor, *, layer_id: int, label: str
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if q_pe.dtype != torch.bfloat16:
            raise EvidenceError(
                f"R28 query RoPE must originate as BF16: {layer_id}:{label}"
            )
        if q_pe.ndim != 3 or q_pe.shape[0] != 1 or q_pe.shape[2] != 64:
            raise EvidenceError(
                f"R28 query RoPE geometry differs: {layer_id}:{label}:"
                f"{tuple(q_pe.shape)}"
            )
        exact = _rotate(q_pe[0].to(torch.float32), q0_math)
        shifted = exact * r26._require_normalization_factor()
        high = shifted.to(torch.float8_e4m3fn)
        residual = exact.to(torch.float8_e4m3fn)
        if (
            not torch.isfinite(high.to(torch.float32)).all()
            or not torch.isfinite(residual.to(torch.float32)).all()
        ):
            raise r26.CandidateNumericalMiss(
                f"R28 rotated E4M3 query RoPE is nonfinite: {layer_id}:{label}"
            )
        high_diagnostics = r26._float8_diagnostics(
            shifted,
            high,
            expected_dtype=torch.float8_e4m3fn,
            label=f"R28 high rotated E4M3 query RoPE {layer_id}:{label}",
        )
        residual_diagnostics = r26._float8_diagnostics(
            exact,
            residual,
            expected_dtype=torch.float8_e4m3fn,
            label=f"R28 residual rotated E4M3 query RoPE {layer_id}:{label}",
        )
        if (
            high_diagnostics["endpoint_count"] != 0
            or residual_diagnostics["endpoint_count"] != 0
        ):
            raise r26.CandidateNumericalMiss(
                "R28 rotated E4M3 query reaches a saturated endpoint: "
                f"{layer_id}:{label}"
            )
        return (
            high.to(torch.float32),
            residual.to(torch.float32),
            {"high": high_diagnostics, "residual": residual_diagnostics},
        )

    def surface_quality(
        *,
        layer_id: int,
        label: str,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        query_position: int,
        capture: Any,
        dense_values: torch.Tensor,
        raw: torch.Tensor,
        scale: torch.Tensor,
        exact_reciprocal: torch.Tensor,
        stored_high: torch.Tensor,
        residual: dict[str, Any],
        n8_native_score: torch.Tensor,
        n8_surface: dict[str, Any],
        n7: ModuleType,
        signs: Any,
    ) -> dict[str, Any]:
        eligible = capture.positions <= query_position
        key_count = int(eligible.sum().item())
        if n8_native_score.shape != (64, key_count):
            raise EvidenceError(f"N8 native score geometry differs: {layer_id}:{label}")
        k_orig = capture.k_nope[eligible, 0].to(torch.float32)
        key_rope = capture.k_pe[eligible, 0].to(torch.float32)
        qn = q_nope[0].to(torch.float32)
        qr_high, qr_residual, query_rope_metrics = query(
            q_pe, layer_id=layer_id, label=label
        )
        qr_original = q_pe[0].to(torch.float32)
        qr_exact = _rotate(qr_original, q0_math)
        q_rot = n7.e4m3fn(n7.rotate(qn, signs), label="R28 rotated latent query")
        raw = raw[eligible]
        scale = scale[eligible]
        exact_reciprocal = exact_reciprocal[eligible]
        stored_high = stored_high[eligible].to(torch.float32)
        residual_raw = residual["raw"][eligible]
        residual_scale = residual["scale"][eligible]
        n8_native_score = n8_native_score.to(torch.float32)
        dense_values = dense_values[eligible]

        dense_score = qn @ k_orig.transpose(0, 1)
        dense_score += qr_original @ key_rope.transpose(0, 1)
        exact_score = r26._reciprocal_score(
            q_rot=q_rot,
            raw=raw,
            q_rope=qr_exact,
            reciprocal_rope=exact_reciprocal,
            scale=scale,
        )
        key_only_score = r26._hierarchical_score(
            q_rot=q_rot,
            raw=raw,
            q_high=qr_exact * r26._require_normalization_factor(),
            high=stored_high,
            q_residual=qr_exact,
            residual_raw=residual_raw,
            residual_scale=residual_scale,
            scale=scale,
        )
        query_only_score = r26._reciprocal_score(
            q_rot=q_rot,
            raw=raw,
            q_rope=qr_residual,
            reciprocal_rope=exact_reciprocal,
            scale=scale,
        )
        candidate_score = r26._hierarchical_score(
            q_rot=q_rot,
            raw=raw,
            q_high=qr_high,
            high=stored_high,
            q_residual=qr_residual,
            residual_raw=residual_raw,
            residual_scale=residual_scale,
            scale=scale,
        )
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
        n8 = r26._quality_arm(
            score=n8_native_score,
            label="accepted N8 identity reconstruction",
            **arm_args,
        )
        if n8["mse"] != float(n8_surface["candidate_mse"]) or n8[
            "attention_kl"
        ] != float(n8_surface["candidate_attention_kl"]):
            raise EvidenceError(
                f"accepted N8 cell did not reproduce: {layer_id}:{label}"
            )
        exact = r26._quality_arm(
            score=exact_score, label="R28 exact transformed reciprocal", **arm_args
        )
        key_only = r26._quality_arm(
            score=key_only_score,
            label="R28 quantized-key exact-query diagnostic",
            **arm_args,
        )
        query_only = r26._quality_arm(
            score=query_only_score,
            label="R28 exact-key quantized-query diagnostic",
            **arm_args,
        )
        candidate = r26._quality_arm(
            score=candidate_score,
            label="R28 rotated hierarchical RoPE candidate",
            **arm_args,
        )
        mse_ratio = n7._positive_ratio(
            candidate["mse"], n8["mse"], label="R28/N8 complete output MSE"
        )
        kl_ratio = n7._positive_ratio(
            candidate["attention_kl"],
            n8["attention_kl"],
            label="R28/N8 attention KL",
        )
        lost_energy = float(candidate["p"]["nonzero_to_zero_energy_ratio"])
        surface_pass = (
            mse_ratio <= r26.MAX_N8_RELATIVE_RATIO
            and kl_ratio <= r26.MAX_N8_RELATIVE_RATIO
            and lost_energy <= r26.MAX_ABSOLUTE_P_LOST_ENERGY
            and candidate["p"]["lost_energy_gate_pass"]
            and candidate["p"]["saturation_gate_pass"]
        )

        def public(arm: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in arm.items()
                if key not in {"score", "output", "probability"}
            }

        return {
            "label": label,
            "query_position": query_position,
            "causal_key_count": key_count,
            "identity": {
                "dense_score_sha256": r26._sha256_tensor_f32(dense_score),
                "dense_output_sha256": r26._sha256_tensor_f32(dense_output),
                "accepted_n8_score_sha256": r26._sha256_tensor_f32(n8_native_score),
                "accepted_n8_output_sha256": r26._sha256_tensor_f32(n8["output"]),
                "exact_reciprocal_score_sha256": r26._sha256_tensor_f32(exact_score),
                "key_only_score_sha256": r26._sha256_tensor_f32(key_only_score),
                "query_only_score_sha256": r26._sha256_tensor_f32(query_only_score),
                "r26_score_sha256": r26._sha256_tensor_f32(candidate_score),
                "r26_output_sha256": r26._sha256_tensor_f32(candidate["output"]),
            },
            "query_rope": query_rope_metrics,
            "accepted_n8": public(n8),
            "exact_reciprocal": public(exact),
            "quantized_key_exact_query": public(key_only),
            "exact_key_quantized_query": public(query_only),
            "r26": public(candidate),
            "r26_to_n8_mse_ratio": mse_ratio,
            "r26_to_n8_attention_kl_ratio": kl_ratio,
            "output_boundary_pass": mse_ratio <= r26.MAX_N8_RELATIVE_RATIO,
            "attention_kl_boundary_pass": kl_ratio <= r26.MAX_N8_RELATIVE_RATIO,
            "absolute_lost_energy_boundary_pass": (
                lost_energy <= r26.MAX_ABSOLUTE_P_LOST_ENERGY
            ),
            "surface_pass": surface_pass,
        }

    r26.EXPERIMENT_ID = EXPERIMENT_ID
    r26.DECISION_ADVANCE = DECISION_ADVANCE
    r26.DECISION_REJECT = DECISION_REJECT
    r26.DECISION_NO_EVIDENCE = DECISION_NO_EVIDENCE
    r26._prepare_reciprocal_rope = prepare
    r26._query_rope_e4m3 = query
    r26._surface_quality = surface_quality


def _finalize_result(
    result: dict[str, Any], transform_test: dict[str, Any]
) -> dict[str, Any]:
    result["experiment_id"] = EXPERIMENT_ID
    result["evaluator_sha256"] = _sha256_file(Path(__file__))
    result["r26_frozen_source_sha256"] = EXPECTED_R26_SOURCE_SHA256
    result["q0_math_source_sha256"] = EXPECTED_Q0_MATH_SHA256
    result["q0_contract_source_sha256"] = EXPECTED_Q0_CONTRACT_SHA256
    result["post_rope_transform_self_test"] = transform_test
    contract = result.get("contract")
    if isinstance(contract, dict):
        contract["query_rope_format"] = (
            "seed-42 randomized normalized-Hadamard post-RoPE, then E4M3 "
            "rounded independently at factor 16 and unity"
        )
        contract["key_rope_transform"] = (
            "same seed-42 randomized normalized-Hadamard on reciprocal RoPE"
        )
        contract["transform_dimension"] = TRANSFORM_DIMENSION
        contract["transform_signs1_sha256"] = EXPECTED_SIGNS1_SHA256
        contract["transform_signs2_sha256"] = EXPECTED_SIGNS2_SHA256
    for layer in result.get("layers", []):
        for surface in layer.get("surfaces", []):
            identity = surface.get("identity", {})
            if "r26_score_sha256" in identity:
                identity["r28_score_sha256"] = identity.pop("r26_score_sha256")
            if "r26_output_sha256" in identity:
                identity["r28_output_sha256"] = identity.pop("r26_output_sha256")
            if "r26" in surface:
                surface["r28"] = surface.pop("r26")
            if "r26_to_n8_mse_ratio" in surface:
                surface["r28_to_n8_mse_ratio"] = surface.pop("r26_to_n8_mse_ratio")
            if "r26_to_n8_attention_kl_ratio" in surface:
                surface["r28_to_n8_attention_kl_ratio"] = surface.pop(
                    "r26_to_n8_attention_kl_ratio"
                )
    summary = result.get("summary")
    if isinstance(summary, dict):
        for old, new in (
            ("max_r26_to_n8_mse_ratio", "max_r28_to_n8_mse_ratio"),
            (
                "max_r26_to_n8_attention_kl_ratio",
                "max_r28_to_n8_attention_kl_ratio",
            ),
            ("max_r26_to_dense_score_rmse", "max_r28_to_dense_score_rmse"),
            (
                "max_absolute_r26_p_lost_energy",
                "max_absolute_r28_p_lost_energy",
            ),
        ):
            if old in summary:
                summary[new] = summary.pop(old)
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if torch.get_num_threads() != 1:
        raise EvidenceError("R28-Q0 requires exactly one Torch thread")
    r26 = _load_module(
        args.r26_source, "a17_r28_frozen_r26", EXPECTED_R26_SOURCE_SHA256
    )
    q0_math = _load_q0_math(args.q0_math_source)
    transform_test = _transform_dot_self_test(q0_math)
    _install_r28(r26, q0_math)
    return _finalize_result(r26.evaluate(args), transform_test)


def _add_path(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(
        f"--{name.replace('_', '-')}", dest=name, type=Path, required=True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
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
