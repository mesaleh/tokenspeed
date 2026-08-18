#!/usr/bin/env python3
"""Evaluate R26 hierarchical residual reciprocal-RoPE quality against A17-N8.

R26 keeps N10's accepted E2M1 latent and arbitrary BF16 token scale ``d``.
It stores an E4M3 high term ``h = E4M3((k_pe / d) / 16)`` plus two fixed
group-32 E2M1 residuals for ``(k_pe / d) - 16 * h``.  One score accumulator
still receives ``d`` exactly once.  This offline falsifier proves format
semantics, storage algebra, and frozen quality; it does not claim native-kernel
legality, speed, or live memory.
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

EXPERIMENT_ID = "a17-n10-e0-r26-q0-20260818"
DECISION_ADVANCE = "ADVANCE_R26_HIERARCHICAL_RESIDUAL_TO_Q1"
DECISION_REJECT = "REJECT_R26_Q0"
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
EXPECTED_PACKER_SHA256 = (
    "a62e74a8315c7a18847b7b7bdf8217e7602fa815bf9a739be2c2bb8296e697bf"
)
MAX_N8_RELATIVE_RATIO = 1.10
MAX_ABSOLUTE_P_LOST_ENERGY = 1.0e-6
NORMALIZATION_FACTOR = 16.0
RESIDUAL_GROUP_SIZE = 32
RESIDUAL_GROUPS = 2
E2M1_MAX_MAGNITUDE = 6.0
UE8M0_EXPONENT_MIN = -127
UE8M0_EXPONENT_MAX = 127


class EvidenceError(RuntimeError):
    """Raised when frozen inputs cannot support an R26 decision."""


class CandidateNumericalMiss(RuntimeError):
    """Raised when valid frozen inputs violate an R26 numerical gate."""


def _require_normalization_factor() -> float:
    """Return R26's sealed power-of-two factor, failing closed on mutation."""
    factor = float(NORMALIZATION_FACTOR)
    if (
        not math.isfinite(factor)
        or factor <= 0.0
        or factor != 16.0
        or not math.log2(factor).is_integer()
    ):
        raise EvidenceError(
            f"R26 normalization factor differs from sealed power-of-two 16: {factor}"
        )
    return factor


def _require_residual_geometry() -> tuple[int, int]:
    if RESIDUAL_GROUP_SIZE != 32 or RESIDUAL_GROUPS != 2:
        raise EvidenceError(
            "R26 residual geometry differs from two sealed groups of 32: "
            f"groups={RESIDUAL_GROUPS}, size={RESIDUAL_GROUP_SIZE}"
        )
    if (
        E2M1_MAX_MAGNITUDE != 6.0
        or UE8M0_EXPONENT_MIN != -127
        or UE8M0_EXPONENT_MAX != 127
    ):
        raise EvidenceError(
            "R26 residual E2M1/UE8M0 range contract differs from sealed values"
        )
    return RESIDUAL_GROUPS, RESIDUAL_GROUP_SIZE


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
    return hashlib.sha256(
        value.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


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
    """Return scale, exact/high terms, quantized residual, and zero mask."""
    if key_rope_bf16.dtype != torch.bfloat16:
        raise EvidenceError(
            f"R26 key RoPE must originate as BF16, got {key_rope_bf16.dtype}"
        )
    if token_scale_bf16.dtype != torch.bfloat16:
        raise EvidenceError(
            f"R26 scale must originate as BF16, got {token_scale_bf16.dtype}"
        )
    if key_rope_bf16.ndim != 2 or key_rope_bf16.shape[1] != 64:
        raise EvidenceError(
            f"R26 key RoPE geometry differs: {tuple(key_rope_bf16.shape)}"
        )
    if token_scale_bf16.shape != (key_rope_bf16.shape[0],):
        raise EvidenceError("R26 scale geometry differs")

    key = key_rope_bf16.to(torch.float32)
    scale = token_scale_bf16.to(torch.float32)
    if not torch.isfinite(key).all() or not torch.isfinite(scale).all():
        raise EvidenceError("R26 source key/scale is nonfinite")
    if (scale < 0).any():
        raise EvidenceError("R26 token scale is negative")

    zero = scale == 0
    candidate_scale = scale.clone()
    candidate_scale[zero] = 1.0
    exact = key / candidate_scale.unsqueeze(1)
    normalized = exact / _require_normalization_factor()
    high = normalized.to(torch.float8_e4m3fn)
    if (
        not torch.isfinite(exact).all()
        or not torch.isfinite(normalized).all()
        or not torch.isfinite(high.to(torch.float32)).all()
    ):
        raise EvidenceError("R26 normalized E4M3 high RoPE overflows or is nonfinite")
    residual_exact = exact - high.to(torch.float32) * _require_normalization_factor()
    residual = _quantize_residual_group32(residual_exact, packer)
    return candidate_scale, exact, normalized, high, residual, zero


def _smallest_no_clip_ue8m0_scale(
    maximum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose the unique smallest UE8M0 scale whose E2M1 input is <= 6."""
    if maximum.ndim != 2 or maximum.shape[1] != RESIDUAL_GROUPS:
        raise EvidenceError("R26 residual maximum geometry differs")
    if not torch.isfinite(maximum).all() or (maximum < 0).any():
        raise EvidenceError("R26 residual maximum is invalid")
    zero = maximum == 0
    maximum64 = maximum.to(torch.float64)
    safe = torch.where(zero, torch.ones_like(maximum64), maximum64)
    exponent = torch.ceil(torch.log2(safe / E2M1_MAX_MAGNITUDE)).to(torch.int32)
    exponent = torch.where(zero, torch.zeros_like(exponent), exponent)
    scale64 = torch.ldexp(torch.ones_like(maximum64), exponent)
    underscaled = (~zero) & (maximum64 > scale64 * E2M1_MAX_MAGNITUDE)
    exponent = exponent + underscaled.to(torch.int32)
    if (exponent < UE8M0_EXPONENT_MIN).any() or (exponent > UE8M0_EXPONENT_MAX).any():
        raise CandidateNumericalMiss("R26 residual UE8M0 exponent is out of range")
    scale = torch.ldexp(torch.ones_like(maximum), exponent)
    if (maximum > scale * E2M1_MAX_MAGNITUDE).any():
        raise EvidenceError("R26 residual no-clipping scale postcondition failed")
    encoded = (exponent + 127).to(torch.uint8)
    return exponent, scale, encoded


def _quantize_residual_group32(
    residual: torch.Tensor, packer: ModuleType
) -> dict[str, Any]:
    groups_count, group_size = _require_residual_geometry()
    if residual.ndim != 2 or residual.shape[1] != groups_count * group_size:
        raise EvidenceError(f"R26 residual geometry differs: {tuple(residual.shape)}")
    if not torch.isfinite(residual).all():
        raise CandidateNumericalMiss("R26 residual input is nonfinite")
    groups = residual.to(torch.float32).reshape(-1, groups_count, group_size)
    maximum = torch.amax(torch.abs(groups), dim=-1)
    exponent, scale, scale_raw = _smallest_no_clip_ue8m0_scale(maximum)
    normalized = groups / scale.unsqueeze(-1)
    clipped = torch.abs(normalized) > E2M1_MAX_MAGNITUDE
    if clipped.any():
        raise EvidenceError("R26 residual normalization clips E2M1")
    codes = packer.quantize_e2m1_codes(normalized)
    zero_group = maximum == 0
    codes[zero_group.unsqueeze(-1).expand_as(codes)] = 0
    packed = packer.pack_e2m1_codes(codes)
    unpacked = packer.unpack_e2m1_codes(packed)
    if not torch.equal(unpacked, codes):
        raise EvidenceError("R26 residual packed-code round trip differs")
    raw = packer.decode_e2m1_codes(unpacked).to(torch.float32)
    reconstruction = (raw * scale.unsqueeze(-1)).reshape_as(residual)
    error = reconstruction.to(torch.float64) - residual.to(torch.float64)
    return {
        "raw": raw.reshape_as(residual),
        "scale": scale,
        "scale_exponent": exponent,
        "scale_raw": scale_raw,
        "codes": codes.reshape_as(residual),
        "packed": packed.reshape(residual.shape[0], -1),
        "reconstruction": reconstruction,
        "diagnostics": {
            "groups": int(maximum.numel()),
            "group_size": group_size,
            "zero_groups": int(zero_group.sum().item()),
            "ue8m0_exponent_min": int(exponent.min().item()),
            "ue8m0_exponent_max": int(exponent.max().item()),
            "normalized_max_abs": float(torch.abs(normalized).max().item()),
            "clipped_coordinates": int(clipped.sum().item()),
            "endpoint_code_count": int((torch.abs(raw) == 6.0).sum().item()),
            "rounding_rmse": float(torch.sqrt(torch.mean(error * error)).item()),
            "rounding_max_abs": float(torch.abs(error).max().item()),
            "packed_round_trip_bit_identical": True,
            "codes_sha256": _sha256_tensor_raw(codes),
            "packed_sha256": _sha256_tensor_raw(packed),
            "ue8m0_raw_sha256": _sha256_tensor_raw(scale_raw),
            "reconstruction_sha256": _sha256_tensor_f32(reconstruction),
        },
    }


def _float8_diagnostics(
    exact: torch.Tensor,
    stored: torch.Tensor,
    *,
    expected_dtype: torch.dtype,
    label: str,
) -> dict[str, Any]:
    if stored.dtype != expected_dtype:
        raise EvidenceError(
            f"{label} dtype differs: {stored.dtype} != {expected_dtype}"
        )
    stored_f32 = stored.to(torch.float32)
    error = stored_f32.to(torch.float64) - exact.to(torch.float64)
    absolute = torch.abs(stored_f32)
    raw = stored.contiguous().view(torch.uint8).clone()
    reconstructed = raw.view(expected_dtype)
    if not torch.equal(reconstructed.view(torch.uint8), raw) or not torch.equal(
        reconstructed.to(torch.float32), stored_f32
    ):
        raise EvidenceError(f"{label} raw-byte reconstruction differs")
    bits = raw.to(torch.int16)
    if expected_dtype == torch.float8_e5m2:
        exponent = bits & 0x7C
        mantissa = bits & 0x03
    elif expected_dtype == torch.float8_e4m3fn:
        exponent = bits & 0x78
        mantissa = bits & 0x07
    else:
        raise EvidenceError(f"{label} unsupported diagnostic dtype: {expected_dtype}")
    subnormal = (exponent == 0) & (mantissa != 0)
    endpoint = absolute == torch.finfo(expected_dtype).max
    underflow = (exact != 0) & (stored_f32 == 0)
    exact_energy = torch.sum(exact.to(torch.float64).square())
    lost_energy = torch.sum(exact[underflow].to(torch.float64).square())
    underflow_energy_ratio = (
        float((lost_energy / exact_energy).item()) if exact_energy.item() != 0 else 0.0
    )
    finite_values = absolute.reshape(-1)
    quantiles = torch.quantile(
        finite_values.to(torch.float64),
        torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], dtype=torch.float64),
    )
    return {
        "finite": bool(torch.isfinite(stored_f32).all()),
        "elements": stored.numel(),
        "zero_count": int((stored_f32 == 0).sum().item()),
        "nonzero_to_zero_count": int(underflow.sum().item()),
        "nonzero_to_zero_energy_ratio": underflow_energy_ratio,
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
        "stored_dtype": str(stored.dtype),
        "stored_raw_sha256": _sha256_tensor_raw(stored),
        "raw_bit_reconstruction_bit_identical": True,
    }


def _scale_diagnostics(scale: torch.Tensor) -> dict[str, Any]:
    if scale.ndim != 1 or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise EvidenceError("R26 candidate scale diagnostics input is invalid")
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
        raise EvidenceError("R26 evaluated cell structure is malformed") from error
    if set(expected_scores) != evaluated or set(expected_cells) != evaluated:
        raise EvidenceError("N8/R26 evaluated cell set differs")
    if len(evaluated) != 18:
        raise EvidenceError(f"R26 evaluator covered {len(evaluated)} cells")
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
        label="R26 P/V factorization self-test",
        safe_peak=224.0,
        candidate_owned=True,
    )
    expected = torch.mean(raw * scale.unsqueeze(1), dim=0, keepdim=True)
    if not torch.equal(output, expected):
        raise EvidenceError("R26 P/V token-scale factorization differs")
    return {
        "pass": True,
        "expected_output_sha256": _sha256_tensor_f32(expected),
        "observed_output_sha256": _sha256_tensor_f32(output),
        "lost_energy_gate_pass": metrics["lost_energy_gate_pass"],
        "saturation_gate_pass": metrics["saturation_gate_pass"],
    }


def _rope_normalization_factorization_self_test(
    *, query_factor: float = 16.0, key_divisor: float = 16.0
) -> dict[str, Any]:
    """Prove the sealed query/key normalization cancels before token scaling."""
    factor = _require_normalization_factor()
    if query_factor != factor or key_divisor != factor:
        raise EvidenceError(
            "R26 RoPE normalization factors differ from the sealed factor: "
            f"query={query_factor}, key={key_divisor}, sealed={factor}"
        )
    query = torch.tensor([[0.5, -2.0]], dtype=torch.float32)
    reciprocal = torch.tensor([[4.0, 0.25], [-1.0, 8.0]], dtype=torch.float32)
    scale = torch.tensor([0.5, 2.0], dtype=torch.float32)
    exact = (query @ reciprocal.transpose(0, 1)) * scale.unsqueeze(0)
    normalized = (
        (query * query_factor) @ (reciprocal / key_divisor).transpose(0, 1)
    ) * scale.unsqueeze(0)
    if not torch.equal(normalized, exact):
        raise EvidenceError("R26 RoPE normalization factorization differs")
    return {
        "pass": True,
        "normalization_factor": factor,
        "expected_score_sha256": _sha256_tensor_f32(exact),
        "observed_score_sha256": _sha256_tensor_f32(normalized),
    }


def _hierarchical_factorization_self_test(
    *, residual_scale_override: torch.Tensor | None = None
) -> dict[str, Any]:
    """Prove high, residual-group, and token scales compose exactly once."""
    q = torch.zeros((1, 64), dtype=torch.float32)
    q[0, :4] = torch.tensor([0.5, -2.0, 1.0, 4.0])
    high = torch.zeros((2, 64), dtype=torch.float32)
    high[0, :4] = torch.tensor([0.25, 0.5, -1.0, 2.0])
    high[1, :4] = torch.tensor([-0.5, 0.25, 1.0, -1.0])
    residual_raw = torch.zeros((2, 64), dtype=torch.float32)
    residual_raw[0, :4] = torch.tensor([1.0, -0.5, 2.0, -1.0])
    residual_raw[1, :4] = torch.tensor([-2.0, 1.0, 0.5, 3.0])
    residual_scale = torch.tensor([[0.25, 2.0], [0.5, 4.0]], dtype=torch.float32)
    effective_residual_scale = (
        residual_scale if residual_scale_override is None else residual_scale_override
    )
    token_scale = torch.tensor([0.5, 2.0], dtype=torch.float32)
    observed = _hierarchical_score(
        q_rot=torch.zeros((1, 1), dtype=torch.float32),
        raw=torch.zeros((2, 1), dtype=torch.float32),
        q_high=q * _require_normalization_factor(),
        high=high,
        q_residual=q,
        residual_raw=residual_raw,
        residual_scale=effective_residual_scale,
        scale=token_scale,
    )
    reconstructed = high * _require_normalization_factor()
    reconstructed += (
        residual_raw.reshape(2, 2, 32) * residual_scale.unsqueeze(-1)
    ).reshape(2, 64)
    expected = (q @ reconstructed.transpose(0, 1)) * token_scale.unsqueeze(0)
    if not torch.equal(observed, expected):
        raise EvidenceError("R26 hierarchical residual scale factorization differs")
    return {
        "pass": True,
        "expected_score_sha256": _sha256_tensor_f32(expected),
        "observed_score_sha256": _sha256_tensor_f32(observed),
    }


def _query_rope_e4m3(
    q_pe: torch.Tensor, *, layer_id: int, label: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if q_pe.dtype != torch.bfloat16:
        raise EvidenceError(
            f"R26 query RoPE must originate as BF16: {layer_id}:{label}"
        )
    if q_pe.ndim != 3 or q_pe.shape[0] != 1 or q_pe.shape[2] != 64:
        raise EvidenceError(
            f"R26 query RoPE geometry differs: {layer_id}:{label}:"
            f"{tuple(q_pe.shape)}"
        )
    exact = q_pe[0].to(torch.float32)
    shifted = exact * _require_normalization_factor()
    high = shifted.to(torch.float8_e4m3fn)
    residual = exact.to(torch.float8_e4m3fn)
    if (
        not torch.isfinite(high.to(torch.float32)).all()
        or not torch.isfinite(residual.to(torch.float32)).all()
    ):
        raise EvidenceError(f"R26 E4M3 query RoPE is nonfinite: {layer_id}:{label}")
    high_diagnostics = _float8_diagnostics(
        shifted,
        high,
        expected_dtype=torch.float8_e4m3fn,
        label=f"R26 high E4M3 query RoPE {layer_id}:{label}",
    )
    residual_diagnostics = _float8_diagnostics(
        exact,
        residual,
        expected_dtype=torch.float8_e4m3fn,
        label=f"R26 residual E4M3 query RoPE {layer_id}:{label}",
    )
    if (
        high_diagnostics["endpoint_count"] != 0
        or residual_diagnostics["endpoint_count"] != 0
    ):
        raise EvidenceError(
            "R26 E4M3 query RoPE reaches a saturated endpoint: " f"{layer_id}:{label}"
        )
    return (
        high.to(torch.float32),
        residual.to(torch.float32),
        {
            "high": high_diagnostics,
            "residual": residual_diagnostics,
        },
    )


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


def _hierarchical_score(
    *,
    q_rot: torch.Tensor,
    raw: torch.Tensor,
    q_high: torch.Tensor,
    high: torch.Tensor,
    q_residual: torch.Tensor,
    residual_raw: torch.Tensor,
    residual_scale: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    groups_count, group_size = _require_residual_geometry()
    tokens = raw.shape[0]
    if (
        q_high.shape[-1] != groups_count * group_size
        or q_residual.shape != q_high.shape
        or high.shape != (tokens, groups_count * group_size)
        or residual_raw.shape != high.shape
        or residual_scale.shape != (tokens, groups_count)
    ):
        raise EvidenceError("R26 hierarchical score geometry differs")
    latent = q_rot @ raw.transpose(0, 1)
    high_score = q_high @ high.transpose(0, 1)
    residual_group_score = torch.einsum(
        "hgd,tgd->htg",
        q_residual.reshape(-1, groups_count, group_size),
        residual_raw.reshape(tokens, groups_count, group_size),
    )
    residual_score = torch.sum(
        residual_group_score * residual_scale.unsqueeze(0), dim=-1
    )
    return (latent + high_score + residual_score) * scale.unsqueeze(0)


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
    qr_high, qr_residual, query_rope_metrics = _query_rope_e4m3(
        q_pe, layer_id=layer_id, label=label
    )
    qr_exact = q_pe[0].to(torch.float32)
    q_rot = n7.e4m3fn(n7.rotate(qn, signs), label="R26 rotated query")
    raw = raw[eligible]
    scale = scale[eligible]
    exact_reciprocal = exact_reciprocal[eligible]
    stored_high = stored_high[eligible].to(torch.float32)
    residual_raw = residual["raw"][eligible]
    residual_scale = residual["scale"][eligible]
    n8_native_score = n8_native_score.to(torch.float32)
    dense_values = dense_values[eligible]

    dense_score = qn @ k_orig.transpose(0, 1) + qr_exact @ key_rope.transpose(0, 1)
    exact_score = _reciprocal_score(
        q_rot=q_rot,
        raw=raw,
        q_rope=qr_exact,
        reciprocal_rope=exact_reciprocal,
        scale=scale,
    )
    key_only_score = _hierarchical_score(
        q_rot=q_rot,
        raw=raw,
        q_high=qr_exact * _require_normalization_factor(),
        high=stored_high,
        q_residual=qr_exact,
        residual_raw=residual_raw,
        residual_scale=residual_scale,
        scale=scale,
    )
    query_only_score = _reciprocal_score(
        q_rot=q_rot,
        raw=raw,
        q_rope=qr_residual,
        reciprocal_rope=exact_reciprocal,
        scale=scale,
    )
    candidate_score = _hierarchical_score(
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
    if n8["mse"] != float(n8_surface["candidate_mse"]) or n8["attention_kl"] != float(
        n8_surface["candidate_attention_kl"]
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
        label="R26 exact reciprocal-RoPE diagnostic",
    )
    key_only = _quality_arm(
        score=key_only_score,
        dense_score=dense_score,
        dense_output=dense_output,
        dense_probability=dense_probability,
        raw_values=raw,
        reconstructed_scale=scale,
        attention_scale=capture.attention_scale,
        n7=n7,
        label="R26 quantized-key exact-query diagnostic",
    )
    query_only = _quality_arm(
        score=query_only_score,
        dense_score=dense_score,
        dense_output=dense_output,
        dense_probability=dense_probability,
        raw_values=raw,
        reconstructed_scale=scale,
        attention_scale=capture.attention_scale,
        n7=n7,
        label="R26 exact-key quantized-query diagnostic",
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
        label="R26 hierarchical E4M3-by-E2M1 residual-RoPE candidate",
    )
    mse_ratio = n7._positive_ratio(
        candidate["mse"], n8["mse"], label="R26/N8 complete output MSE"
    )
    kl_ratio = n7._positive_ratio(
        candidate["attention_kl"],
        n8["attention_kl"],
        label="R26/N8 attention KL",
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
            "key_only_score_sha256": _sha256_tensor_f32(key_only_score),
            "query_only_score_sha256": _sha256_tensor_f32(query_only_score),
            "r26_score_sha256": _sha256_tensor_f32(candidate_score),
            "r26_output_sha256": _sha256_tensor_f32(candidate["output"]),
        },
        "query_rope": query_rope_metrics,
        "accepted_n8": public(n8),
        "exact_reciprocal": public(exact),
        "quantized_key_exact_query": public(key_only),
        "exact_key_quantized_query": public(query_only),
        "r26": public(candidate),
        "r26_to_n8_mse_ratio": mse_ratio,
        "r26_to_n8_attention_kl_ratio": kl_ratio,
        "output_boundary_pass": mse_ratio <= MAX_N8_RELATIVE_RATIO,
        "attention_kl_boundary_pass": kl_ratio <= MAX_N8_RELATIVE_RATIO,
        "absolute_lost_energy_boundary_pass": lost_energy <= MAX_ABSOLUTE_P_LOST_ENERGY,
        "surface_pass": surface_pass,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    n8 = _load_module(
        args.n8_evaluator,
        name="a17_r26_frozen_n8",
        expected_sha256=EXPECTED_N8_EVALUATOR_SHA256,
    )
    packer = _load_module(
        args.packer_source,
        name="a17_r26_frozen_e2m1_packer",
        expected_sha256=EXPECTED_PACKER_SHA256,
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
    rope_normalization_factorization_self_test = (
        _rope_normalization_factorization_self_test()
    )
    hierarchical_factorization_self_test = _hierarchical_factorization_self_test()

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
        (
            candidate_scale,
            exact_reciprocal,
            normalized_reciprocal,
            stored_high,
            residual,
            zero,
        ) = _prepare_reciprocal_rope(capture.k_pe[:, 0], stored_scale, packer)
        if zero.any() and not torch.equal(raw[zero], torch.zeros_like(raw[zero])):
            raise EvidenceError(
                f"zero-scale N8 row has nonzero latent codes: layer={layer_id}"
            )

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
                    stored_high=stored_high,
                    residual=residual,
                    n8_native_score=n8_native_scores[key],
                    n8_surface=n8_cells[key],
                    n7=n8.N7,
                    signs=signs,
                )
            )
        reciprocal_metrics = _float8_diagnostics(
            normalized_reciprocal,
            stored_high,
            expected_dtype=torch.float8_e4m3fn,
            label=f"R26 normalized E4M3 high RoPE layer={layer_id}",
        )
        residual_metrics = residual["diagnostics"]
        scale_metrics = _scale_diagnostics(candidate_scale)
        layer_pass = (
            reciprocal_metrics["finite"]
            and reciprocal_metrics["endpoint_count"] == 0
            and reciprocal_metrics["raw_bit_reconstruction_bit_identical"]
            and residual_metrics["clipped_coordinates"] == 0
            and residual_metrics["packed_round_trip_bit_identical"]
            and all(surface["surface_pass"] for surface in surfaces)
        )
        layers.append(
            {
                "layer": layer_id,
                "zero_scale_rows": int(zero.sum().item()),
                "latent_raw_sha256": _sha256_tensor_f32(raw),
                "candidate_scale": scale_metrics,
                "high_rope": reciprocal_metrics,
                "residual_rope": residual_metrics,
                "surfaces": surfaces,
                "layer_pass": layer_pass,
            }
        )
        del capture, signs, carrier, raw, stored_scale, candidate_scale
        del exact_reciprocal, normalized_reciprocal, stored_high, residual, zero
        gc.collect()

    cell_count = _validate_evaluated_cell_set(
        expected_scores=n8_native_scores,
        expected_cells=n8_cells,
        layers=layers,
    )
    surfaces = [surface for layer in layers for surface in layer["surfaces"]]
    if len(surfaces) != cell_count:
        raise EvidenceError("R26 cell count differs after validation")
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
            "normalization_factor": _require_normalization_factor(),
            "normalization_factor_selection": (
                "smallest power of two satisfying the sealed R24 reciprocal range"
            ),
            "residual_groups": RESIDUAL_GROUPS,
            "residual_group_size": RESIDUAL_GROUP_SIZE,
            "residual_scale_rule": (
                "smallest UE8M0 power of two satisfying abs(E2M1 input) <= 6"
            ),
            "query_rope_format": (
                "E4M3 rounded independently from BF16 times 16 and BF16"
            ),
            "rope_storage": (
                "64B E4M3 high plus 32B packed E2M1 residual plus 2B UE8M0"
            ),
            "persistent_row_bytes": 356,
            "dense_row_bytes": 576,
            "projected_target_row_saving_percent": (1.0 - 356.0 / 576.0) * 100.0,
            "zero_scale_policy": (
                "zero latent-scale rows use unity BF16 token scale; "
                "high and residual terms still encode raw RoPE"
            ),
            "selected_layers": list(n8.N7.SELECTED_LAYERS),
            "verify_rows": n8.N7.VERIFY_ROWS,
        },
        "capture_files": len(capture_hashes),
        "capture_bytes": capture_bytes,
        "capture_sha256": capture_hashes,
        "evaluator_sha256": _sha256_file(Path(__file__)),
        "n8_evaluator_sha256": EXPECTED_N8_EVALUATOR_SHA256,
        "packer_sha256": EXPECTED_PACKER_SHA256,
        "n8_reproduction_sha256": reproduced_n8_hash,
        "n8_reproduction_bit_identical": True,
        "n8_native_identity": native_identity,
        "pv_factorization_self_test": pv_factorization_self_test,
        "rope_normalization_factorization_self_test": (
            rope_normalization_factorization_self_test
        ),
        "hierarchical_factorization_self_test": hierarchical_factorization_self_test,
        "layers": layers,
        "summary": {
            "cells": len(surfaces),
            "all_surfaces_pass": all_surfaces_pass,
            "all_layers_pass": all_layers_pass,
            "max_r26_to_n8_mse_ratio": max(
                surface["r26_to_n8_mse_ratio"] for surface in surfaces
            ),
            "max_r26_to_n8_attention_kl_ratio": max(
                surface["r26_to_n8_attention_kl_ratio"] for surface in surfaces
            ),
            "max_r26_to_dense_score_rmse": max(
                surface["r26"]["to_dense_score_rmse"] for surface in surfaces
            ),
            "max_exact_to_dense_score_rmse": max(
                surface["exact_reciprocal"]["to_dense_score_rmse"]
                for surface in surfaces
            ),
            "max_absolute_r26_p_lost_energy": max(
                surface["r26"]["p"]["nonzero_to_zero_energy_ratio"]
                for surface in surfaces
            ),
            "zero_scale_rows": sum(layer["zero_scale_rows"] for layer in layers),
            "reciprocal_subnormal_count": sum(
                layer["high_rope"]["subnormal_count"] for layer in layers
            ),
            "reciprocal_nonzero_to_zero_count": sum(
                layer["high_rope"]["nonzero_to_zero_count"] for layer in layers
            ),
            "max_query_nonzero_to_zero_energy_ratio": max(
                max(
                    surface["query_rope"]["high"]["nonzero_to_zero_energy_ratio"],
                    surface["query_rope"]["residual"]["nonzero_to_zero_energy_ratio"],
                )
                for surface in surfaces
            ),
            "query_endpoint_count": sum(
                surface["query_rope"]["high"]["endpoint_count"]
                + surface["query_rope"]["residual"]["endpoint_count"]
                for surface in surfaces
            ),
            "residual_clipped_coordinates": sum(
                layer["residual_rope"]["clipped_coordinates"] for layer in layers
            ),
            "residual_endpoint_code_count": sum(
                layer["residual_rope"]["endpoint_code_count"] for layer in layers
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
