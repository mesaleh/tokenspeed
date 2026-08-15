#!/usr/bin/env python3
"""Evaluate A17-N10 BF16 reciprocal-RoPE quality against accepted A17-N8.

N10 keeps the accepted N8 E2M1 latent and arbitrary BF16 token scale ``d``.
It stores each RoPE key as ``BF16(k_pe / d)`` so one score accumulator can
produce ``d * (q_latent @ c + q_rope @ reciprocal_rope)``.  This evaluator is
an offline falsifier: it proves the storage algebra and quality boundary, but
does not claim native-kernel legality, speed, or live memory savings.
"""

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

import numpy as np
import torch


EXPERIMENT_ID = "a17-n10-q0-20260815"
DECISION_ADVANCE = "ADVANCE_N10_BF16_RECIPROCAL_ROPE_TO_S0"
DECISION_REJECT = "REJECT_N10_BF16_RECIPROCAL_ROPE_QUALITY"
DECISION_NO_EVIDENCE = "NO_DECISION_EVIDENCE"
EXPECTED_CAPTURE_FILES = 363
EXPECTED_CAPTURE_BYTES = 6_157_142
EXPECTED_N8_EVALUATOR_SHA256 = (
    "3ee6c858ee06301417848bb21758f265d206dfa59e53e4715cdcb9aa3ec3b1f8"
)
EXPECTED_N8_RESULT_SHA256 = (
    "0347f24104eae60c39b6806c546dbae8dd4906f07ea93bad1fd2f86c4a5672a4"
)
EXPECTED_N8_OPERANDS_SHA256 = (
    "522c4ec36298bb35a929f6bce9074740fe03165667d4f49beec26b773a54382f"
)
MAX_N8_RELATIVE_RATIO = 1.10
MAX_ABSOLUTE_P_LOST_ENERGY = 1.0e-6


class EvidenceError(RuntimeError):
    """Raised when frozen inputs cannot support an N10 decision."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor_f32(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _sha256_tensor_raw(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes(order="C")).hexdigest()


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


def _load_module(path: Path, *, name: str, expected_sha256: str) -> ModuleType:
    _require_hash(path, expected_sha256, name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvidenceError(f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reproduce_n8(
    args: argparse.Namespace, n8: ModuleType
) -> tuple[dict[str, Any], str]:
    _require_hash(args.n8_operands, EXPECTED_N8_OPERANDS_SHA256, "accepted N8 operands")
    _require_hash(
        args.accepted_n8_result, EXPECTED_N8_RESULT_SHA256, "accepted N8 result"
    )
    result = n8.evaluate(
        capture_root=args.capture_root,
        operands_path=args.n8_operands,
        scores1_path=args.n8_scores_1,
        metadata1_path=args.n8_metadata_1,
        scores2_path=args.n8_scores_2,
        metadata2_path=args.n8_metadata_2,
        run_identity1_path=args.n8_run_identity_1,
        run_identity2_path=args.n8_run_identity_2,
        runner_source=args.n8_runner_source,
    )
    result_bytes = n8._canonical_json_bytes(result) + b"\n"
    result_hash = hashlib.sha256(result_bytes).hexdigest()
    if result_hash != EXPECTED_N8_RESULT_SHA256:
        raise EvidenceError(
            "accepted N8 evaluator did not reproduce its canonical result: "
            f"{result_hash} != {EXPECTED_N8_RESULT_SHA256}"
        )
    if result_bytes != args.accepted_n8_result.read_bytes():
        raise EvidenceError("reproduced N8 result is not byte-identical")
    if result.get("decision") != n8.DECISION_ADVANCE:
        raise EvidenceError("reproduced N8 control no longer advances")
    return result, result_hash


def _surface_specs(
    capture: Any, n7: ModuleType
) -> list[tuple[str, torch.Tensor, torch.Tensor, int]]:
    specs = [
        (
            "prefill_final",
            capture.prefill_q_nope,
            capture.prefill_q_pe,
            int(capture.prefill_position),
        )
    ]
    for row in range(n7.VERIFY_ROWS):
        specs.append(
            (
                f"target_verify_q{row}",
                capture.verify_q_nope[row : row + 1],
                capture.verify_q_pe[row : row + 1],
                int(capture.verify_positions[row]),
            )
        )
    return specs


def _prepare_reciprocal_rope(
    key_rope_bf16: torch.Tensor, token_scale_bf16: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return candidate scale, exact quotient, stored BF16 quotient, zero mask."""
    if key_rope_bf16.dtype != torch.bfloat16:
        raise EvidenceError(f"N10 key RoPE must originate as BF16, got {key_rope_bf16.dtype}")
    if token_scale_bf16.dtype != torch.bfloat16:
        raise EvidenceError(f"N10 scale must originate as BF16, got {token_scale_bf16.dtype}")
    if key_rope_bf16.ndim != 2 or key_rope_bf16.shape[1] != 64:
        raise EvidenceError(f"N10 key RoPE geometry differs: {tuple(key_rope_bf16.shape)}")
    if token_scale_bf16.shape != (key_rope_bf16.shape[0],):
        raise EvidenceError("N10 scale geometry differs")

    key = key_rope_bf16.to(torch.float32)
    scale = token_scale_bf16.to(torch.float32)
    if not torch.isfinite(key).all() or not torch.isfinite(scale).all():
        raise EvidenceError("N10 source key/scale is nonfinite")
    if (scale < 0).any():
        raise EvidenceError("N10 token scale is negative")

    zero = scale == 0
    candidate_scale = scale.clone()
    candidate_scale[zero] = 1.0
    exact = key / candidate_scale.unsqueeze(1)
    stored = exact.to(torch.bfloat16)
    if not torch.isfinite(exact).all() or not torch.isfinite(stored).all():
        raise EvidenceError("N10 reciprocal RoPE overflows or is nonfinite")
    return candidate_scale, exact, stored, zero


def _bf16_diagnostics(exact: torch.Tensor, stored: torch.Tensor) -> dict[str, Any]:
    stored_f32 = stored.to(torch.float32)
    error = stored_f32.to(torch.float64) - exact.to(torch.float64)
    absolute = torch.abs(stored_f32)
    bits = stored.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    reconstructed = (bits << 16).contiguous().view(torch.float32)
    if not torch.equal(reconstructed, stored_f32):
        raise EvidenceError("independent BF16 raw-bit reconstruction differs")
    exponent = bits & 0x7F80
    mantissa = bits & 0x007F
    subnormal = (exponent == 0) & (mantissa != 0)
    endpoint = absolute == torch.finfo(torch.bfloat16).max
    finite_values = absolute.reshape(-1)
    quantiles = torch.quantile(
        finite_values.to(torch.float64),
        torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], dtype=torch.float64),
    )
    return {
        "finite": bool(torch.isfinite(stored_f32).all()),
        "elements": stored.numel(),
        "zero_count": int((stored_f32 == 0).sum().item()),
        "subnormal_count": int(subnormal.sum().item()),
        "endpoint_count": int(endpoint.sum().item()),
        "absolute_quantiles": {
            label: float(value)
            for label, value in zip(
                ("min", "p50", "p90", "p99", "max"), quantiles.tolist()
            )
        },
        "rounding_rmse": float(torch.sqrt(torch.mean(error * error)).item()),
        "rounding_max_abs": float(torch.max(torch.abs(error)).item()),
        "stored_raw_sha256": _sha256_tensor_raw(stored),
        "raw_bit_reconstruction_bit_identical": True,
    }


def _scale_diagnostics(scale: torch.Tensor) -> dict[str, Any]:
    if scale.ndim != 1 or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise EvidenceError("N10 candidate scale diagnostics input is invalid")
    quantiles = torch.quantile(
        scale.to(torch.float64),
        torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], dtype=torch.float64),
    )
    return {
        "elements": scale.numel(),
        "positive": True,
        "quantiles": {
            label: float(value)
            for label, value in zip(
                ("min", "p50", "p90", "p99", "max"), quantiles.tolist()
            )
        },
        "sha256": _sha256_tensor_f32(scale),
    }


def _validate_capture_identity(
    *,
    capture_hashes: dict[str, str],
    capture_bytes: int,
    n8_payload: dict[str, Any],
) -> None:
    if (
        len(capture_hashes) != EXPECTED_CAPTURE_FILES
        or capture_bytes != EXPECTED_CAPTURE_BYTES
    ):
        raise EvidenceError(
            f"capture identity differs: files={len(capture_hashes)}, "
            f"bytes={capture_bytes}"
        )
    if dict(sorted(capture_hashes.items())) != n8_payload.get("capture_sha256"):
        raise EvidenceError("capture inventory differs from accepted N8 operands")


def _validate_evaluated_cell_set(
    *,
    expected_scores: dict[tuple[int, str], torch.Tensor],
    expected_cells: dict[tuple[int, str], dict[str, Any]],
    layers: list[dict[str, Any]],
) -> int:
    try:
        evaluated = {
            (int(layer["layer"]), str(surface["label"]))
            for layer in layers
            for surface in layer["surfaces"]
        }
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError("N10 evaluated cell structure is malformed") from error
    if set(expected_scores) != evaluated or set(expected_cells) != evaluated:
        raise EvidenceError("N8/N10 evaluated cell set differs")
    if len(evaluated) != 18:
        raise EvidenceError(f"N10 evaluator covered {len(evaluated)} cells")
    return len(evaluated)


def _pv_factorization_self_test(
    n7: ModuleType, *, scale_override: torch.Tensor | None = None
) -> dict[str, Any]:
    score = torch.zeros((1, 2), dtype=torch.float32)
    raw = torch.zeros((2, 512), dtype=torch.float32)
    raw[0, :2] = torch.tensor([1.0, 2.0])
    raw[1, :2] = torch.tensor([3.0, 4.0])
    scale = torch.tensor([0.5, 2.0], dtype=torch.float32)
    effective_scale = scale if scale_override is None else scale_override
    output, metrics = n7._block_normalized_output(
        score=score,
        scale=1.0,
        reconstructed_scale=effective_scale,
        raw_values=raw,
        label="N10 P/V factorization self-test",
        safe_peak=224.0,
        candidate_owned=True,
    )
    expected = torch.mean(raw * scale.unsqueeze(1), dim=0, keepdim=True)
    if not torch.equal(output, expected):
        raise EvidenceError("N10 P/V token-scale factorization differs")
    return {
        "pass": True,
        "expected_output_sha256": _sha256_tensor_f32(expected),
        "observed_output_sha256": _sha256_tensor_f32(output),
        "lost_energy_gate_pass": metrics["lost_energy_gate_pass"],
        "saturation_gate_pass": metrics["saturation_gate_pass"],
    }


def _query_rope_bf16(
    q_pe: torch.Tensor, *, layer_id: int, label: str
) -> torch.Tensor:
    if q_pe.dtype != torch.bfloat16:
        raise EvidenceError(
            f"N10 query RoPE must originate as BF16: {layer_id}:{label}"
        )
    if q_pe.ndim != 3 or q_pe.shape[0] != 1 or q_pe.shape[2] != 64:
        raise EvidenceError(
            f"N10 query RoPE geometry differs: {layer_id}:{label}:"
            f"{tuple(q_pe.shape)}"
        )
    return q_pe[0].to(torch.float32)


def _reciprocal_score(
    *,
    q_rot: torch.Tensor,
    raw: torch.Tensor,
    q_rope: torch.Tensor,
    reciprocal_rope: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    latent = q_rot @ raw.transpose(0, 1)
    rope = q_rope @ reciprocal_rope.transpose(0, 1)
    return (latent + rope) * scale.unsqueeze(0)


def _quality_arm(
    *,
    score: torch.Tensor,
    dense_score: torch.Tensor,
    dense_output: torch.Tensor,
    dense_probability: torch.Tensor,
    raw_values: torch.Tensor,
    reconstructed_scale: torch.Tensor,
    attention_scale: float,
    n7: ModuleType,
    label: str,
) -> dict[str, Any]:
    if not torch.isfinite(score).all():
        raise EvidenceError(f"{label} score is nonfinite")
    output, p_metrics = n7._block_normalized_output(
        score=score,
        scale=attention_scale,
        reconstructed_scale=reconstructed_scale,
        raw_values=raw_values,
        label=label,
        safe_peak=224.0,
        candidate_owned=True,
    )
    numerator = n7._score_numerator(score, attention_scale, candidate_owned=True)
    probability = n7._probability(numerator)
    error = score.to(torch.float64) - dense_score.to(torch.float64)
    per_head_rmse = torch.sqrt(torch.mean(error * error, dim=1))
    return {
        "score": score,
        "output": output,
        "probability": probability,
        "mse": n7.mse(output, dense_output),
        "attention_kl": n7._kl(dense_probability, probability, candidate_owned=True),
        "relative_l2": n7.relative_l2(output, dense_output),
        "cosine": n7.cosine(output, dense_output),
        "to_dense_score_rmse": n7._rmse(score, dense_score),
        "to_dense_per_head_score_rmse_min": float(per_head_rmse.min().item()),
        "to_dense_per_head_score_rmse_max": float(per_head_rmse.max().item()),
        "p": p_metrics,
    }


def _surface_quality(
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
    stored_reciprocal: torch.Tensor,
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
    qr = _query_rope_bf16(q_pe, layer_id=layer_id, label=label)
    q_rot = n7.e4m3fn(n7.rotate(qn, signs), label="N10 rotated query")
    raw = raw[eligible]
    scale = scale[eligible]
    exact_reciprocal = exact_reciprocal[eligible]
    stored_reciprocal = stored_reciprocal[eligible].to(torch.float32)
    n8_native_score = n8_native_score.to(torch.float32)
    dense_values = dense_values[eligible]

    dense_score = qn @ k_orig.transpose(0, 1) + qr @ key_rope.transpose(0, 1)
    exact_score = _reciprocal_score(
        q_rot=q_rot,
        raw=raw,
        q_rope=qr,
        reciprocal_rope=exact_reciprocal,
        scale=scale,
    )
    candidate_score = _reciprocal_score(
        q_rot=q_rot,
        raw=raw,
        q_rope=qr,
        reciprocal_rope=stored_reciprocal,
        scale=scale,
    )
    dense_numerator = n7._score_numerator(dense_score, capture.attention_scale)
    dense_probability = n7._probability(dense_numerator)
    dense_output = (dense_numerator @ dense_values) / dense_numerator.sum(
        dim=1, keepdim=True
    )

    n8 = _quality_arm(
        score=n8_native_score,
        dense_score=dense_score,
        dense_output=dense_output,
        dense_probability=dense_probability,
        raw_values=raw,
        reconstructed_scale=scale,
        attention_scale=capture.attention_scale,
        n7=n7,
        label="accepted N8 identity reconstruction",
    )
    if (
        n8["mse"] != float(n8_surface["candidate_mse"])
        or n8["attention_kl"] != float(n8_surface["candidate_attention_kl"])
    ):
        raise EvidenceError(f"accepted N8 cell did not reproduce: {layer_id}:{label}")
    exact = _quality_arm(
        score=exact_score,
        dense_score=dense_score,
        dense_output=dense_output,
        dense_probability=dense_probability,
        raw_values=raw,
        reconstructed_scale=scale,
        attention_scale=capture.attention_scale,
        n7=n7,
        label="N10 exact reciprocal-RoPE diagnostic",
    )
    candidate = _quality_arm(
        score=candidate_score,
        dense_score=dense_score,
        dense_output=dense_output,
        dense_probability=dense_probability,
        raw_values=raw,
        reconstructed_scale=scale,
        attention_scale=capture.attention_scale,
        n7=n7,
        label="N10 BF16 reciprocal-RoPE candidate",
    )
    mse_ratio = n7._positive_ratio(
        candidate["mse"], n8["mse"], label="N10/N8 complete output MSE"
    )
    kl_ratio = n7._positive_ratio(
        candidate["attention_kl"],
        n8["attention_kl"],
        label="N10/N8 attention KL",
    )
    lost_energy = float(candidate["p"]["nonzero_to_zero_energy_ratio"])
    surface_pass = (
        mse_ratio <= MAX_N8_RELATIVE_RATIO
        and kl_ratio <= MAX_N8_RELATIVE_RATIO
        and lost_energy <= MAX_ABSOLUTE_P_LOST_ENERGY
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
            "dense_score_sha256": _sha256_tensor_f32(dense_score),
            "dense_output_sha256": _sha256_tensor_f32(dense_output),
            "accepted_n8_score_sha256": _sha256_tensor_f32(n8_native_score),
            "accepted_n8_output_sha256": _sha256_tensor_f32(n8["output"]),
            "exact_reciprocal_score_sha256": _sha256_tensor_f32(exact_score),
            "n10_score_sha256": _sha256_tensor_f32(candidate_score),
            "n10_output_sha256": _sha256_tensor_f32(candidate["output"]),
        },
        "accepted_n8": public(n8),
        "exact_reciprocal": public(exact),
        "n10": public(candidate),
        "n10_to_n8_mse_ratio": mse_ratio,
        "n10_to_n8_attention_kl_ratio": kl_ratio,
        "output_boundary_pass": mse_ratio <= MAX_N8_RELATIVE_RATIO,
        "attention_kl_boundary_pass": kl_ratio <= MAX_N8_RELATIVE_RATIO,
        "absolute_lost_energy_boundary_pass": lost_energy <= MAX_ABSOLUTE_P_LOST_ENERGY,
        "surface_pass": surface_pass,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    n8 = _load_module(
        args.n8_evaluator,
        name="a17_n10_frozen_n8",
        expected_sha256=EXPECTED_N8_EVALUATOR_SHA256,
    )
    n8_result, reproduced_n8_hash = _reproduce_n8(args, n8)
    n8_payload, n8_native_scores, native_identity = n8._load_native_inputs(
        operands_path=args.n8_operands,
        scores1_path=args.n8_scores_1,
        metadata1_path=args.n8_metadata_1,
        scores2_path=args.n8_scores_2,
        metadata2_path=args.n8_metadata_2,
        run_identity1_path=args.n8_run_identity_1,
        run_identity2_path=args.n8_run_identity_2,
        runner_source=args.n8_runner_source,
    )
    records, capture_hashes, capture_bytes = n8.N7._load_records(args.capture_root)
    _validate_capture_identity(
        capture_hashes=capture_hashes,
        capture_bytes=capture_bytes,
        n8_payload=n8_payload,
    )
    pv_factorization_self_test = _pv_factorization_self_test(n8.N7)

    n8_cells = {
        (int(layer["layer"]), str(surface["label"])): surface
        for layer in n8_result["layers"]
        for surface in layer["surfaces"]
    }
    layers: list[dict[str, Any]] = []
    for payload_layer in n8_payload["layers"]:
        layer_id = int(payload_layer["layer"])
        capture = n8.N7._build_layer_capture(
            records,
            layer=layer_id,
            expected_length=n8.N7.INPUT_LENGTHS[n8.N7.STRATUM],
        )
        signs = n8.N7.make_sign_contract("cpu")
        carrier = n8.N7.quantize_rows(
            capture.k_nope[:, 0].to(torch.float32), signs, family="k4"
        )
        raw = payload_layer["raw_key_e2m1"].to(torch.float32)
        stored_scale = payload_layer["token_scale_bf16"]
        if not torch.equal(raw, carrier.raw) or not torch.equal(
            stored_scale, carrier.scale.to(torch.bfloat16)
        ):
            raise EvidenceError(f"accepted N8 carrier differs at layer={layer_id}")
        candidate_scale, exact_reciprocal, stored_reciprocal, zero = (
            _prepare_reciprocal_rope(capture.k_pe[:, 0], stored_scale)
        )
        if zero.any() and not torch.equal(
            raw[zero], torch.zeros_like(raw[zero])
        ):
            raise EvidenceError(f"zero-scale N8 row has nonzero latent codes: layer={layer_id}")

        surfaces = []
        for label, q_nope, q_pe, query_position in _surface_specs(capture, n8.N7):
            key = (layer_id, label)
            if key not in n8_cells or key not in n8_native_scores:
                raise EvidenceError(f"quality cell is missing: {key}")
            surfaces.append(
                _surface_quality(
                    layer_id=layer_id,
                    label=label,
                    q_nope=q_nope,
                    q_pe=q_pe,
                    query_position=query_position,
                    capture=capture,
                    dense_values=carrier.rotated_reference,
                    raw=raw,
                    scale=candidate_scale,
                    exact_reciprocal=exact_reciprocal,
                    stored_reciprocal=stored_reciprocal,
                    n8_native_score=n8_native_scores[key],
                    n8_surface=n8_cells[key],
                    n7=n8.N7,
                    signs=signs,
                )
            )
        reciprocal_metrics = _bf16_diagnostics(exact_reciprocal, stored_reciprocal)
        scale_metrics = _scale_diagnostics(candidate_scale)
        layer_pass = (
            reciprocal_metrics["finite"]
            and reciprocal_metrics["endpoint_count"] == 0
            and reciprocal_metrics["raw_bit_reconstruction_bit_identical"]
            and all(surface["surface_pass"] for surface in surfaces)
        )
        layers.append(
            {
                "layer": layer_id,
                "zero_scale_rows": int(zero.sum().item()),
                "latent_raw_sha256": _sha256_tensor_f32(raw),
                "candidate_scale": scale_metrics,
                "reciprocal_rope": reciprocal_metrics,
                "surfaces": surfaces,
                "layer_pass": layer_pass,
            }
        )
        del capture, signs, carrier, raw, stored_scale, candidate_scale
        del exact_reciprocal, stored_reciprocal, zero
        gc.collect()

    cell_count = _validate_evaluated_cell_set(
        expected_scores=n8_native_scores,
        expected_cells=n8_cells,
        layers=layers,
    )
    surfaces = [surface for layer in layers for surface in layer["surfaces"]]
    if len(surfaces) != cell_count:
        raise EvidenceError("N10 cell count differs after validation")
    all_surfaces_pass = all(surface["surface_pass"] for surface in surfaces)
    all_layers_pass = all(layer["layer_pass"] for layer in layers)
    decision = (
        DECISION_ADVANCE if all_surfaces_pass and all_layers_pass else DECISION_REJECT
    )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stratum": n8.N7.STRATUM,
        "offline_falsifier": True,
        "cannot_confirm_speed_or_live_memory": True,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": torch.get_num_threads(),
        },
        "contract": {
            "max_n8_relative_output_mse_ratio": MAX_N8_RELATIVE_RATIO,
            "max_n8_relative_attention_kl_ratio": MAX_N8_RELATIVE_RATIO,
            "max_absolute_normalized_p_lost_energy": MAX_ABSOLUTE_P_LOST_ENERGY,
            "lost_energy_threshold_is_absolute": True,
            "latent_storage": "N8 E2M1 256B plus arbitrary BF16 scale 2B",
            "rope_storage": "64 BF16 reciprocal-scaled coordinates 128B",
            "persistent_row_bytes": 386,
            "zero_scale_policy": "all-zero E2M1 codes, unity BF16 scale, raw BF16 RoPE",
            "selected_layers": list(n8.N7.SELECTED_LAYERS),
            "verify_rows": n8.N7.VERIFY_ROWS,
        },
        "capture_files": len(capture_hashes),
        "capture_bytes": capture_bytes,
        "capture_sha256": capture_hashes,
        "evaluator_sha256": _sha256_file(Path(__file__)),
        "n8_evaluator_sha256": EXPECTED_N8_EVALUATOR_SHA256,
        "n8_reproduction_sha256": reproduced_n8_hash,
        "n8_reproduction_bit_identical": True,
        "n8_native_identity": native_identity,
        "pv_factorization_self_test": pv_factorization_self_test,
        "layers": layers,
        "summary": {
            "cells": len(surfaces),
            "all_surfaces_pass": all_surfaces_pass,
            "all_layers_pass": all_layers_pass,
            "max_n10_to_n8_mse_ratio": max(
                surface["n10_to_n8_mse_ratio"] for surface in surfaces
            ),
            "max_n10_to_n8_attention_kl_ratio": max(
                surface["n10_to_n8_attention_kl_ratio"] for surface in surfaces
            ),
            "max_n10_to_dense_score_rmse": max(
                surface["n10"]["to_dense_score_rmse"] for surface in surfaces
            ),
            "max_exact_to_dense_score_rmse": max(
                surface["exact_reciprocal"]["to_dense_score_rmse"]
                for surface in surfaces
            ),
            "max_absolute_n10_p_lost_energy": max(
                surface["n10"]["p"]["nonzero_to_zero_energy_ratio"]
                for surface in surfaces
            ),
            "zero_scale_rows": sum(layer["zero_scale_rows"] for layer in layers),
            "reciprocal_subnormal_count": sum(
                layer["reciprocal_rope"]["subnormal_count"] for layer in layers
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
    except (EvidenceError, OSError, AttributeError, ValueError, KeyError, TypeError) as error:
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
