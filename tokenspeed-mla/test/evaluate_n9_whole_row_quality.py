#!/usr/bin/env python3
"""Evaluate A17-N9 whole-row UE8M0 quality against the accepted N8 oracle."""

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

EXPERIMENT_ID = "a17-n9-q0-20260815"
DECISION_ADVANCE = "ADVANCE_N9_Q0_TO_NATIVE_SCORE"
DECISION_REJECT = "REJECT_N9_WHOLE_ROW_UE8M0_QUALITY"
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
EXPECTED_N9_OPERANDS_SHA256 = (
    "81fe473792993d2ad670d5cff1591df2f036135f4e3c1dff71701a9ed8295e6a"
)
MAX_N8_RELATIVE_RATIO = 1.10
MAX_ABSOLUTE_P_LOST_ENERGY = 1.0e-6


class EvidenceError(RuntimeError):
    """Raised when inputs cannot support an N9 quality decision."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


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


def _load_module(path: Path, *, name: str, expected_sha256: str | None) -> ModuleType:
    if expected_sha256 is not None:
        _require_hash(path, expected_sha256, name)
    elif not path.is_file():
        raise EvidenceError(f"{name} is missing: {path}")
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
    reproduced_sha256 = hashlib.sha256(result_bytes).hexdigest()
    if reproduced_sha256 != EXPECTED_N8_RESULT_SHA256:
        raise EvidenceError(
            "accepted N8 evaluator did not reproduce its canonical result: "
            f"{reproduced_sha256} != {EXPECTED_N8_RESULT_SHA256}"
        )
    if result_bytes != args.accepted_n8_result.read_bytes():
        raise EvidenceError("reproduced N8 result is not byte-identical")
    if result.get("decision") != n8.DECISION_ADVANCE:
        raise EvidenceError("reproduced N8 control no longer advances")
    return result, reproduced_sha256


def _surface_specs(
    capture: Any, n7: ModuleType
) -> list[tuple[str, Any, Any, Any, int]]:
    specs = [
        (
            "prefill_final",
            capture.prefill_q_nope,
            capture.prefill_q_pe,
            capture.prefill_q_inc,
            int(capture.prefill_position),
        )
    ]
    for row in range(n7.VERIFY_ROWS):
        specs.append(
            (
                f"target_verify_q{row}",
                capture.verify_q_nope[row : row + 1],
                capture.verify_q_pe[row : row + 1],
                capture.verify_q_inc[row : row + 1],
                int(capture.verify_positions[row]),
            )
        )
    return specs


def _diagnostics_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1.0e-14, abs_tol=1.0e-18)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _diagnostics_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _diagnostics_equal(left[key], right[key]) for key in left
        )
    return left == right


def _load_n9_payload(
    *, operands: Path, manifest_path: Path, packer_source: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(EXPECTED_N9_OPERANDS_SHA256) != 64:
        raise EvidenceError("N9 operand hash is not frozen in evaluator source")
    operand_hash = _require_hash(
        operands, EXPECTED_N9_OPERANDS_SHA256, "frozen N9 operands"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("artifact_sha256") != operand_hash
        or int(manifest.get("artifact_bytes", -1)) != operands.stat().st_size
    ):
        raise EvidenceError("N9 operand manifest identity differs")
    packer_hash = _require_hash(
        packer_source, str(manifest.get("source_sha256")), "N9 operand packer"
    )
    payload = torch.load(operands, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise EvidenceError("N9 operand payload is not an object")
    return payload, {
        "operands_sha256": operand_hash,
        "manifest_sha256": _sha256_file(manifest_path),
        "packer_sha256": packer_hash,
    }


def _validate_n9_operands(
    *,
    records: Any,
    capture_hashes: dict[str, str],
    payload: dict[str, Any],
    packer: ModuleType,
    n8: ModuleType,
    n8_payload: dict[str, Any],
) -> dict[str, Any]:
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("capture_sha256") != dict(sorted(capture_hashes.items()))
        or payload.get("n8_operands_sha256") != EXPECTED_N8_OPERANDS_SHA256
        or payload.get("n8_result_sha256") != EXPECTED_N8_RESULT_SHA256
        or payload.get("q0_source_sha256") != packer.EXPECTED_Q0_SOURCE_SHA256
        or payload.get("self_test") != packer._self_test()
    ):
        raise EvidenceError("N9 frozen dependency/self-test binding differs")
    expected_representation = {
        "rotation": "signed_normalized_wht_seed42_no_l2_normalization",
        "scale_scope": "whole_512_coordinate_row",
        "scale_selection": "minimum_float64_reconstruction_sse",
        "candidate_exponents": [-127, 127],
        "sse_tie_rule": "smaller_exponent",
        "float64_reduction": "torch_sum_dim2_one_thread_pinned_runtime",
        "e2m1_rounding": "nearest_even_saturating_finite",
        "zero_row_scale": "unity",
        "e2m1_storage": "hardware_nibbles_low_coordinate_first",
        "ue8m0_bias": 127,
    }
    if payload.get("representation") != expected_representation:
        raise EvidenceError("N9 representation contract differs")
    n7 = n8.N7
    payload_layers = payload.get("layers")
    n8_layers = {int(layer["layer"]): layer for layer in n8_payload.get("layers", [])}
    if (
        not isinstance(payload_layers, list)
        or [int(layer.get("layer", -1)) for layer in payload_layers]
        != list(n7.SELECTED_LAYERS)
        or set(n8_layers) != set(n7.SELECTED_LAYERS)
    ):
        raise EvidenceError("N9/N8 operand layer coverage differs")
    checked_surfaces = 0
    reconstructed_rows = 0
    for payload_layer, layer_id in zip(payload_layers, n7.SELECTED_LAYERS):
        capture = n7._build_layer_capture(
            records, layer=layer_id, expected_length=n7.INPUT_LENGTHS[n7.STRATUM]
        )
        signs = n7.make_sign_contract("cpu")
        if (
            payload.get("signs1_sha256") != signs.signs1_sha256
            or payload.get("signs2_sha256") != signs.signs2_sha256
        ):
            raise EvidenceError("N9 sign contract differs")
        rotated = n7.rotate(capture.k_nope[:, 0].to(torch.float32), signs)
        n8_scale = n8_layers[int(layer_id)].get("token_scale_bf16")
        if not isinstance(n8_scale, torch.Tensor):
            raise EvidenceError(f"N8 scale is missing at layer={layer_id}")
        quantized = packer._quantize_whole_row(rotated, n8_scale=n8_scale)
        expected_tensors = {
            "packed_key_e2m1": quantized["packed"].cpu().contiguous(),
            "row_ue8m0_codes": quantized["ue8m0_codes"].cpu().contiguous(),
            "key_rope_fp8": capture.k_pe_inc[:, 0]
            .to(torch.float8_e4m3fn)
            .cpu()
            .contiguous(),
        }
        for name, expected in expected_tensors.items():
            observed = payload_layer.get(name)
            if not isinstance(observed, torch.Tensor) or not torch.equal(
                observed, expected
            ):
                raise EvidenceError(f"N9 operand {name} differs at layer={layer_id}")
        if not _diagnostics_equal(
            payload_layer.get("quantization_diagnostics"),
            quantized["diagnostics"],
        ):
            raise EvidenceError(f"N9 diagnostics differ at layer={layer_id}")
        observed_surfaces = payload_layer.get("surfaces")
        specs = _surface_specs(capture, n7)
        if not isinstance(observed_surfaces, list) or len(observed_surfaces) != 6:
            raise EvidenceError(f"N9 surface coverage differs at layer={layer_id}")
        for observed, (label, q_nope, _q_pe, q_inc, query_position) in zip(
            observed_surfaces, specs
        ):
            key_count = int((capture.positions <= query_position).sum().item())
            q_rot = n7.e4m3fn(
                n7.rotate(q_nope[0].to(torch.float32), signs),
                label="N9 operand crosscheck rotated query",
            ).to(torch.float8_e4m3fn)
            q_rope = q_inc[0, :, 512:].to(torch.float8_e4m3fn)
            if (
                observed.get("label") != label
                or int(observed.get("query_position", -1)) != query_position
                or int(observed.get("key_count", -1)) != key_count
                or not isinstance(observed.get("q_rot_fp8"), torch.Tensor)
                or not torch.equal(observed["q_rot_fp8"], q_rot.cpu().contiguous())
                or not isinstance(observed.get("q_rope_fp8"), torch.Tensor)
                or not torch.equal(observed["q_rope_fp8"], q_rope.cpu().contiguous())
            ):
                raise EvidenceError(
                    f"N9 query operand differs at layer={layer_id} surface={label}"
                )
            checked_surfaces += 1
        reconstructed_rows += int(rotated.shape[0])
        del capture, signs, rotated, quantized
        gc.collect()
    if checked_surfaces != 18 or reconstructed_rows != 2_319:
        raise EvidenceError(
            f"N9 crosscheck covered {checked_surfaces} cells/{reconstructed_rows} rows"
        )
    return {
        "layers": len(payload_layers),
        "surfaces": checked_surfaces,
        "rows": reconstructed_rows,
        "tensors_bit_identical": True,
        "geometry_bit_identical": True,
        "exhaustive_requantization_bit_identical": True,
        "packer_self_test": True,
        "pass": True,
    }


def _decode_layer(
    layer: dict[str, Any], packer: ModuleType
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codes = packer.unpack_e2m1_codes(layer["packed_key_e2m1"])
    raw = packer.decode_e2m1_codes(codes).to(torch.float32)
    exponent = layer["row_ue8m0_codes"].to(torch.int32) - 127
    if exponent.shape != (raw.shape[0],):
        raise EvidenceError("N9 row scale geometry differs")
    scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    reconstruction = raw * scale.unsqueeze(-1)
    if not torch.isfinite(reconstruction).all():
        raise EvidenceError("N9 reconstruction is nonfinite")
    return raw, scale, reconstruction


def _whole_row_algebra_score(
    q_rot: torch.Tensor,
    raw: torch.Tensor,
    scale: torch.Tensor,
    q_rope: torch.Tensor,
    key_rope: torch.Tensor,
) -> torch.Tensor:
    return (q_rot @ raw.transpose(0, 1)) * scale.unsqueeze(0) + (
        q_rope @ key_rope.transpose(0, 1)
    )


def _surface_quality(
    *,
    layer_id: int,
    label: str,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    q_inc: torch.Tensor,
    query_position: int,
    capture: Any,
    dense_values: torch.Tensor,
    raw: torch.Tensor,
    row_scale: torch.Tensor,
    reconstruction: torch.Tensor,
    n8_raw: torch.Tensor,
    n8_scale: torch.Tensor,
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
    rope = capture.k_pe[eligible, 0].to(torch.float32)
    key_rope = capture.k_pe_inc[eligible, 0].to(torch.float32)
    qn = q_nope[0].to(torch.float32)
    qp = q_pe[0].to(torch.float32)
    qi = q_inc[0].to(torch.float32)
    q_rot = n7.e4m3fn(n7.rotate(qn, signs), label="N9 rotated query")
    q_rope = qi[:, 512:]
    raw = raw[eligible]
    row_scale = row_scale[eligible]
    reconstruction = reconstruction[eligible]
    n8_raw = n8_raw[eligible]
    n8_scale = n8_scale[eligible]
    n8_native_score = n8_native_score.to(torch.float32)
    rotated_dense = dense_values[eligible]
    dense_score = qn @ k_orig.transpose(0, 1) + qp @ rope.transpose(0, 1)
    n9_score = _whole_row_algebra_score(
        q_rot, raw, row_scale, q_rope, key_rope
    )
    if not torch.isfinite(n9_score).all():
        raise EvidenceError(f"N9 score is nonfinite: {layer_id}:{label}")
    dense_numerator = n7._score_numerator(dense_score, capture.attention_scale)
    dense_probability = n7._probability(dense_numerator)
    dense_output = (dense_numerator @ rotated_dense) / dense_numerator.sum(
        dim=1, keepdim=True
    )
    n8_output, _n8_p = n7._block_normalized_output(
        score=n8_native_score,
        scale=capture.attention_scale,
        reconstructed_scale=n8_scale,
        raw_values=n8_raw,
        label="accepted N8 identity reconstruction",
        safe_peak=224.0,
        candidate_owned=True,
    )
    n8_numerator = n7._score_numerator(
        n8_native_score, capture.attention_scale, candidate_owned=True
    )
    n8_probability = n7._probability(n8_numerator)
    ones = torch.ones(key_count, dtype=torch.float32)
    n9_output, n9_p = n7._block_normalized_output(
        score=n9_score,
        scale=capture.attention_scale,
        reconstructed_scale=ones,
        raw_values=reconstruction,
        label="N9 whole-row algebraic candidate",
        safe_peak=224.0,
        candidate_owned=True,
    )
    n9_numerator = n7._score_numerator(
        n9_score, capture.attention_scale, candidate_owned=True
    )
    n9_probability = n7._probability(n9_numerator)
    n8_mse = n7.mse(n8_output, dense_output)
    n9_mse = n7.mse(n9_output, dense_output)
    n8_kl = n7._kl(dense_probability, n8_probability, candidate_owned=True)
    n9_kl = n7._kl(dense_probability, n9_probability, candidate_owned=True)
    if (
        n8_mse != float(n8_surface["candidate_mse"])
        or n8_kl != float(n8_surface["candidate_attention_kl"])
    ):
        raise EvidenceError(f"accepted N8 cell did not reproduce: {layer_id}:{label}")
    mse_ratio = n7._positive_ratio(
        n9_mse, n8_mse, label="N9/N8 complete output MSE"
    )
    kl_ratio = n7._positive_ratio(n9_kl, n8_kl, label="N9/N8 attention KL")
    score_error = n9_score.to(torch.float64) - dense_score.to(torch.float64)
    per_head_score_rmse = torch.sqrt(torch.mean(score_error * score_error, dim=1))
    lost_energy = float(n9_p["nonzero_to_zero_energy_ratio"])
    surface_pass = (
        mse_ratio <= MAX_N8_RELATIVE_RATIO
        and kl_ratio <= MAX_N8_RELATIVE_RATIO
        and lost_energy <= MAX_ABSOLUTE_P_LOST_ENERGY
        and n9_p["lost_energy_gate_pass"]
        and n9_p["saturation_gate_pass"]
    )
    return {
        "label": label,
        "query_position": query_position,
        "causal_key_count": key_count,
        "identity": {
            "dense_score_sha256": _sha256_tensor(dense_score),
            "dense_output_sha256": _sha256_tensor(dense_output),
            "dense_probability_sha256": _sha256_tensor(dense_probability),
            "accepted_n8_score_sha256": _sha256_tensor(n8_native_score),
            "accepted_n8_output_sha256": _sha256_tensor(n8_output),
            "accepted_n8_probability_sha256": _sha256_tensor(n8_probability),
            "n9_score_sha256": _sha256_tensor(n9_score),
            "n9_output_sha256": _sha256_tensor(n9_output),
            "n9_probability_sha256": _sha256_tensor(n9_probability),
        },
        "accepted_n8_mse": n8_mse,
        "n9_mse": n9_mse,
        "n9_to_n8_mse_ratio": mse_ratio,
        "accepted_n8_attention_kl": n8_kl,
        "n9_attention_kl": n9_kl,
        "n9_to_n8_attention_kl_ratio": kl_ratio,
        "n9_relative_l2": n7.relative_l2(n9_output, dense_output),
        "n9_cosine": n7.cosine(n9_output, dense_output),
        "n9_to_dense_score_rmse": n7._rmse(n9_score, dense_score),
        "n9_to_dense_per_head_score_rmse_min": float(
            per_head_score_rmse.min().item()
        ),
        "n9_to_dense_per_head_score_rmse_max": float(
            per_head_score_rmse.max().item()
        ),
        "n9_p": n9_p,
        "output_boundary_pass": mse_ratio <= MAX_N8_RELATIVE_RATIO,
        "attention_kl_boundary_pass": kl_ratio <= MAX_N8_RELATIVE_RATIO,
        "absolute_lost_energy_boundary_pass": (
            lost_energy <= MAX_ABSOLUTE_P_LOST_ENERGY
        ),
        "surface_pass": surface_pass,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    n8 = _load_module(
        args.n8_evaluator,
        name="a17_n9_frozen_n8",
        expected_sha256=EXPECTED_N8_EVALUATOR_SHA256,
    )
    n8_result, reproduced_n8_hash = _reproduce_n8(args, n8)
    packer = _load_module(
        args.n9_packer_source,
        name="a17_n9_frozen_packer",
        expected_sha256=None,
    )
    payload, operand_identity = _load_n9_payload(
        operands=args.n9_operands,
        manifest_path=args.n9_manifest,
        packer_source=args.n9_packer_source,
    )
    n8_payload, n8_native_scores, _n8_native_identity = n8._load_native_inputs(
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
    if (
        len(capture_hashes) != EXPECTED_CAPTURE_FILES
        or capture_bytes != EXPECTED_CAPTURE_BYTES
    ):
        raise EvidenceError(
            f"capture identity differs: files={len(capture_hashes)}, bytes={capture_bytes}"
        )
    operand_crosscheck = _validate_n9_operands(
        records=records,
        capture_hashes=capture_hashes,
        payload=payload,
        packer=packer,
        n8=n8,
        n8_payload=n8_payload,
    )
    n8_cells = {
        (int(layer["layer"]), str(surface["label"])): surface
        for layer in n8_result["layers"]
        for surface in layer["surfaces"]
    }
    n8_layers = {int(layer["layer"]): layer for layer in n8_payload["layers"]}
    layers = []
    for payload_layer in payload["layers"]:
        layer_id = int(payload_layer["layer"])
        capture = n8.N7._build_layer_capture(
            records,
            layer=layer_id,
            expected_length=n8.N7.INPUT_LENGTHS[n8.N7.STRATUM],
        )
        signs = n8.N7.make_sign_contract("cpu")
        raw, row_scale, reconstruction = _decode_layer(payload_layer, packer)
        n8_layer = n8_layers[layer_id]
        n8_raw = n8_layer["raw_key_e2m1"].to(torch.float32)
        n8_scale = n8_layer["token_scale_bf16"].to(torch.float32)
        carrier = n8.N7.quantize_rows(
            capture.k_nope[:, 0].to(torch.float32), signs, family="k4"
        )
        if not torch.equal(n8_raw, carrier.raw) or not torch.equal(
            n8_scale, carrier.scale
        ):
            raise EvidenceError(f"accepted N8 carrier differs at layer={layer_id}")
        rotated_dense = carrier.rotated_reference
        n9_error = reconstruction.to(torch.float64) - rotated_dense.to(torch.float64)
        n8_reconstruction = n8_raw * n8_scale.unsqueeze(-1)
        n8_error = n8_reconstruction.to(torch.float64) - rotated_dense.to(torch.float64)
        n9_reconstruction_sse = float(torch.sum(n9_error * n9_error).item())
        n8_reconstruction_sse = float(torch.sum(n8_error * n8_error).item())
        n9_reconstruction_mse = float(torch.mean(n9_error * n9_error).item())
        n8_reconstruction_mse = float(torch.mean(n8_error * n8_error).item())
        surfaces = []
        for label, q_nope, q_pe, q_inc, query_position in _surface_specs(
            capture, n8.N7
        ):
            key = (layer_id, label)
            if key not in n8_cells or key not in n8_native_scores:
                raise EvidenceError(f"quality cell is missing: {key}")
            surfaces.append(
                _surface_quality(
                    layer_id=layer_id,
                    label=label,
                    q_nope=q_nope,
                    q_pe=q_pe,
                    q_inc=q_inc,
                    query_position=query_position,
                    capture=capture,
                    dense_values=rotated_dense,
                    raw=raw,
                    row_scale=row_scale,
                    reconstruction=reconstruction,
                    n8_raw=n8_raw,
                    n8_scale=n8_scale,
                    n8_native_score=n8_native_scores[key],
                    n8_surface=n8_cells[key],
                    n7=n8.N7,
                    signs=signs,
                )
            )
        exponent = payload_layer["row_ue8m0_codes"].to(torch.int64) - 127
        values, counts = torch.unique(exponent, return_counts=True)
        exponent_histogram = {
            str(int(value)): int(count)
            for value, count in zip(values.tolist(), counts.tolist())
        }
        rope_metrics = n8.N7._raw_rope_metrics(
            capture.k_pe_inc[:, 0].to(torch.float32)
        )
        layer_pass = (
            rope_metrics["finite"]
            and rope_metrics["endpoint_count"] == 0
            and all(surface["surface_pass"] for surface in surfaces)
        )
        layers.append(
            {
                "layer": layer_id,
                "quantization": payload_layer["quantization_diagnostics"],
                "exponent_histogram": exponent_histogram,
                "latent_reconstruction": {
                    "n9_sse": n9_reconstruction_sse,
                    "n8_sse": n8_reconstruction_sse,
                    "n9_to_n8_sse_ratio": n8.N7._positive_ratio(
                        n9_reconstruction_sse,
                        n8_reconstruction_sse,
                        label="N9/N8 latent reconstruction SSE",
                    ),
                    "n9_mse": n9_reconstruction_mse,
                    "n8_mse": n8_reconstruction_mse,
                    "n9_to_n8_mse_ratio": n8.N7._positive_ratio(
                        n9_reconstruction_mse,
                        n8_reconstruction_mse,
                        label="N9/N8 latent reconstruction MSE",
                    ),
                },
                "raw_rope": rope_metrics,
                "surfaces": surfaces,
                "layer_pass": layer_pass,
            }
        )
        del (
            capture,
            signs,
            raw,
            row_scale,
            reconstruction,
            rotated_dense,
            n8_reconstruction,
            n9_error,
            n8_error,
            carrier,
        )
        gc.collect()
    evaluated_keys = {
        (int(layer["layer"]), str(surface["label"]))
        for layer in layers
        for surface in layer["surfaces"]
    }
    if set(n8_native_scores) != evaluated_keys or set(n8_cells) != evaluated_keys:
        raise EvidenceError("N8/N9 evaluated cell set differs")
    surfaces = [surface for layer in layers for surface in layer["surfaces"]]
    if len(surfaces) != 18:
        raise EvidenceError(f"N9 evaluator covered {len(surfaces)} cells")
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
            "block_tokens": n8.N7.BLOCK_TOKENS,
            "full_range_safe_peak": 224.0,
            "scale_scope": "whole_512_coordinate_row",
            "candidate_exponents": [-127, 127],
            "selected_layers": list(n8.N7.SELECTED_LAYERS),
            "verify_rows": n8.N7.VERIFY_ROWS,
        },
        "capture_files": len(capture_hashes),
        "capture_bytes": capture_bytes,
        "capture_sha256": capture_hashes,
        "evaluator_sha256": _sha256_file(Path(__file__)),
        "packer_sha256": _sha256_file(args.n9_packer_source),
        "n8_evaluator_sha256": EXPECTED_N8_EVALUATOR_SHA256,
        "n8_reproduction_sha256": reproduced_n8_hash,
        "n8_reproduction_bit_identical": True,
        "operand_identity": operand_identity,
        "operand_crosscheck": operand_crosscheck,
        "layers": layers,
        "summary": {
            "cells": len(surfaces),
            "all_surfaces_pass": all_surfaces_pass,
            "all_layers_pass": all_layers_pass,
            "max_n9_to_n8_mse_ratio": max(
                surface["n9_to_n8_mse_ratio"] for surface in surfaces
            ),
            "max_n9_to_n8_attention_kl_ratio": max(
                surface["n9_to_n8_attention_kl_ratio"] for surface in surfaces
            ),
            "max_n9_to_dense_score_rmse": max(
                surface["n9_to_dense_score_rmse"] for surface in surfaces
            ),
            "max_n9_to_dense_per_head_score_rmse": max(
                surface["n9_to_dense_per_head_score_rmse_max"]
                for surface in surfaces
            ),
            "max_absolute_n9_p_lost_energy": max(
                surface["n9_p"]["nonzero_to_zero_energy_ratio"]
                for surface in surfaces
            ),
            "max_n9_to_n8_latent_reconstruction_mse_ratio": max(
                layer["latent_reconstruction"]["n9_to_n8_mse_ratio"]
                for layer in layers
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
        "n9_packer_source",
        "n9_operands",
        "n9_manifest",
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
