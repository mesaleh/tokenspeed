#!/usr/bin/env python3
"""Build immutable A17-N9 whole-row operands from the frozen Kimi capture.

The artifact stores the actual E2M1 hardware nibbles (two coordinates per
byte), one UE8M0 exponent code per complete latent row, raw FP8 RoPE, and the
same FP8 rotated queries used by the accepted N8 quality oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

EXPERIMENT_ID = "a17-n9-q0-20260815"
STRATUM = "short768"
EXPECTED_CAPTURE_FILES = 363
EXPECTED_CAPTURE_BYTES = 6_157_142
LATENT_SIZE = 512
UE8M0_BIAS = 127
UE8M0_EXPONENT_MIN = -127
UE8M0_EXPONENT_MAX = 127
ZERO_ROW_EXPONENT = 0
ROW_CHUNK = 32

EXPECTED_Q0_SOURCE_SHA256 = {
    "q0_contract.py": "b4596ef6de2f9cf68848a16e21202689e59e209490224b8d900a2a32655e3a94",
    "analyze_q0.py": "ce4bc9a02e2e3d7f4cd4bea08ea361c933c8dedef058ce9dba4694f2372996a9",
    "q0_math.py": "4d0d7bc3545d7486c0760b75335c4bfbe855d1fd400e7fdb69fdcd401dc3e9c8",
}
EXPECTED_N8_OPERANDS_SHA256 = (
    "522c4ec36298bb35a929f6bce9074740fe03165667d4f49beec26b773a54382f"
)
EXPECTED_N8_RESULT_SHA256 = (
    "0347f24104eae60c39b6806c546dbae8dd4906f07ea93bad1fd2f86c4a5672a4"
)

# Hardware nibble order: sign:e2:m1. Code 8 is negative zero.
E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
E2M1_MAGNITUDES = E2M1_VALUES[:8]
E2M1_MIDPOINTS = (E2M1_MAGNITUDES[:-1] + E2M1_MAGNITUDES[1:]) * 0.5


class OperandPreparationError(RuntimeError):
    """Raised when frozen evidence cannot produce a decision-grade artifact."""


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


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise OperandPreparationError(f"{label} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise OperandPreparationError(f"{label} hash differs: {actual} != {expected}")
    return actual


def _load_q0_sources(q0_dir: Path) -> tuple[ModuleType, ModuleType, ModuleType]:
    for name, expected in EXPECTED_Q0_SOURCE_SHA256.items():
        _require_hash(q0_dir / name, expected, f"frozen {name}")

    # analyze_q0 imports q0_contract/q0_math by their original names. Reject a
    # preloaded module from another path rather than silently mixing contracts.
    q0_resolved = q0_dir.resolve()
    for name in ("q0_contract", "q0_math"):
        existing = sys.modules.get(name)
        if (
            existing is not None
            and Path(existing.__file__).resolve().parent != q0_resolved
        ):
            raise OperandPreparationError(
                f"preloaded {name} came from another directory"
            )
    sys.path.insert(0, str(q0_resolved))
    try:
        import q0_contract  # type: ignore[import-not-found]
        import q0_math  # type: ignore[import-not-found]

        spec = importlib.util.spec_from_file_location(
            "a17_n9_frozen_analyze_q0", q0_dir / "analyze_q0.py"
        )
        if spec is None or spec.loader is None:
            raise OperandPreparationError("cannot load frozen analyze_q0.py")
        analyze_q0 = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = analyze_q0
        spec.loader.exec_module(analyze_q0)
    finally:
        sys.path.remove(str(q0_resolved))
    return q0_contract, q0_math, analyze_q0


def decode_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.uint8 or bool(torch.any(codes > 15).item()):
        raise OperandPreparationError("E2M1 code tensor contains an invalid nibble")
    return E2M1_VALUES.to(codes.device)[codes.to(torch.int64)]


def quantize_e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    """Round finite values to E2M1, ties-to-even, preserving signed zero."""
    if not torch.isfinite(values).all():
        raise OperandPreparationError("E2M1 input is nonfinite")
    magnitude = torch.abs(values)
    midpoints = E2M1_MIDPOINTS.to(device=values.device, dtype=values.dtype)
    # bucketize(right=False) selects the lower code at an exact midpoint.
    # E2M1 ties-to-even selects the upper code only at boundaries 1, 3, 5.
    magnitude_code = torch.bucketize(magnitude.contiguous(), midpoints, right=False)
    upper_even_tie = (
        (magnitude == midpoints[1])
        | (magnitude == midpoints[3])
        | (magnitude == midpoints[5])
    )
    magnitude_code = magnitude_code + upper_even_tie.to(torch.int64)
    sign_code = torch.signbit(values).to(torch.int64) * 8
    return (magnitude_code + sign_code).to(torch.uint8)


def pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.uint8 or codes.shape[-1] % 2:
        raise OperandPreparationError("E2M1 codes must be uint8 with even width")
    if bool(torch.any(codes > 15).item()):
        raise OperandPreparationError("E2M1 code exceeds one nibble")
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def unpack_e2m1_codes(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise OperandPreparationError("packed E2M1 tensor must be uint8")
    result = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.uint8)
    result[..., 0::2] = packed & 0xF
    result[..., 1::2] = packed >> 4
    return result


def _quantize_whole_row(
    rotated: torch.Tensor, *, n8_scale: torch.Tensor | None = None
) -> dict[str, Any]:
    """Choose the minimum-SSE power-of-two scale for every complete row."""
    if rotated.shape[-1] != LATENT_SIZE or not torch.isfinite(rotated).all():
        raise OperandPreparationError("rotated latent geometry/range differs")
    rows = rotated.to(torch.float64).reshape(-1, LATENT_SIZE)
    if n8_scale is not None:
        n8_scale64 = n8_scale.to(torch.float64).reshape(-1)
        if n8_scale64.shape != (rows.shape[0],) or not torch.all(n8_scale64 > 0):
            raise OperandPreparationError("accepted N8 scale geometry/range differs")
    else:
        n8_scale64 = None
    candidates = torch.arange(
        UE8M0_EXPONENT_MIN, UE8M0_EXPONENT_MAX + 1, dtype=torch.int32
    )
    scales = torch.ldexp(torch.ones(candidates.numel(), dtype=torch.float64), candidates)
    selected_exponents = []
    selected_codes = []
    selected_sse = []
    runner_up_margin = []
    exact_sse_ties = 0
    for start in range(0, rows.shape[0], ROW_CHUNK):
        chunk = rows[start : start + ROW_CHUNK]
        zero_rows = torch.all(chunk == 0, dim=1)
        normalized = chunk[:, None, :] / scales[None, :, None]
        codes = quantize_e2m1_codes(normalized)
        decoded = decode_e2m1_codes(codes).to(torch.float64)
        error = decoded * scales[None, :, None] - chunk[:, None, :]
        # One CPU thread and a pinned torch version make this float64 reduction
        # order part of the artifact identity. Duplicate runs must be byte exact.
        sse = torch.sum(error * error, dim=2, dtype=torch.float64)
        best_index = torch.argmin(sse, dim=1)
        best_exponent = candidates[best_index].clone()
        best_sse = sse.gather(1, best_index[:, None]).squeeze(1)
        tie_count = torch.sum(sse == best_sse[:, None], dim=1)
        exact_sse_ties += int(torch.count_nonzero(tie_count > 1).item())
        second = torch.topk(sse, k=2, dim=1, largest=False, sorted=True).values[:, 1]
        margin = second - best_sse
        if bool(zero_rows.any()):
            best_exponent[zero_rows] = ZERO_ROW_EXPONENT
            best_sse[zero_rows] = 0
            margin[zero_rows] = 0
        best_scale = torch.ldexp(
            torch.ones_like(best_exponent, dtype=torch.float64), best_exponent
        )
        best_codes = quantize_e2m1_codes(chunk / best_scale[:, None])
        best_codes[zero_rows] = 0
        selected_exponents.append(best_exponent)
        selected_codes.append(best_codes)
        selected_sse.append(best_sse)
        runner_up_margin.append(margin)

    exponent = torch.cat(selected_exponents).to(torch.int32)
    codes = torch.cat(selected_codes)
    row_sse = torch.cat(selected_sse)
    sse_margin = torch.cat(runner_up_margin)
    scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float64), exponent)
    decoded = decode_e2m1_codes(codes).to(torch.float64)
    reconstruction = decoded * scale.unsqueeze(-1)
    zero_row = torch.all(rows == 0, dim=1)
    ue8m0_codes = (exponent + UE8M0_BIAS).to(torch.uint8)
    ue8m0_codes[zero_row] = UE8M0_BIAS + ZERO_ROW_EXPONENT
    selected_normalized = rows / scale.unsqueeze(-1)
    code_counts = torch.bincount(codes.flatten().to(torch.int64), minlength=16)
    maximum = torch.amax(torch.abs(rows), dim=1)
    absmax_anchor = torch.zeros_like(exponent)
    nonzero = ~zero_row
    if bool(nonzero.any()):
        absmax_anchor[nonzero] = torch.ceil(
            torch.log2(maximum[nonzero] / 6.0)
        ).to(torch.int32)
    n8_nearest = None
    if n8_scale64 is not None:
        n8_nearest = torch.round(torch.log2(n8_scale64)).to(torch.int32)
    diagnostics = {
        "ue8m0_exponent_min": (
            None if not bool(nonzero.any()) else int(exponent[nonzero].min().item())
        ),
        "ue8m0_exponent_max": (
            None if not bool(nonzero.any()) else int(exponent[nonzero].max().item())
        ),
        "exact_zero_rows": int(torch.count_nonzero(zero_row).item()),
        "exact_sse_tie_rows": exact_sse_ties,
        "e2m1_code_counts": [int(value) for value in code_counts.tolist()],
        "maximum_normalized_magnitude": float(
            torch.abs(selected_normalized).max().item()
        ),
        "endpoint_selections": int(torch.count_nonzero(torch.abs(decoded) == 6).item()),
        "clipped_normalized_coordinates": int(
            torch.count_nonzero(torch.abs(selected_normalized) > 6).item()
        ),
        "row_sse_min": float(row_sse.min().item()),
        "row_sse_max": float(row_sse.max().item()),
        "runner_up_sse_margin_min_non_tie": (
            None
            if not bool((sse_margin > 0).any())
            else float(sse_margin[sse_margin > 0].min().item())
        ),
        "exponent_minus_absmax_anchor_min": int(
            (exponent - absmax_anchor).min().item()
        ),
        "exponent_minus_absmax_anchor_max": int(
            (exponent - absmax_anchor).max().item()
        ),
        "latent_reconstruction_mse": float(
            torch.mean((reconstruction - rows) ** 2).item()
        ),
    }
    if n8_nearest is not None:
        diagnostics.update(
            {
                "exponent_minus_n8_nearest_min": int(
                    (exponent - n8_nearest).min().item()
                ),
                "exponent_minus_n8_nearest_max": int(
                    (exponent - n8_nearest).max().item()
                ),
                "exponent_equal_n8_nearest_rows": int(
                    torch.count_nonzero(exponent == n8_nearest).item()
                ),
            }
        )
    return {
        "codes": codes.reshape_as(rotated),
        "packed": pack_e2m1_codes(codes.reshape_as(rotated)),
        "exponent": exponent.reshape(*rotated.shape[:-1]),
        "ue8m0_codes": ue8m0_codes.reshape(*rotated.shape[:-1]),
        "reconstruction": reconstruction.to(torch.float32).reshape_as(rotated),
        "diagnostics": diagnostics,
    }


def _self_test() -> dict[str, Any]:
    exact_codes = torch.arange(16, dtype=torch.uint8)
    if not torch.equal(quantize_e2m1_codes(E2M1_VALUES), exact_codes):
        raise OperandPreparationError("exact E2M1 codebook self-test failed")
    positive_even = torch.tensor([0, 2, 2, 4, 4, 6, 6], dtype=torch.uint8)
    negative_even = positive_even + 8
    if not torch.equal(quantize_e2m1_codes(E2M1_MIDPOINTS), positive_even):
        raise OperandPreparationError("positive midpoint ties-to-even self-test failed")
    if not torch.equal(quantize_e2m1_codes(-E2M1_MIDPOINTS), negative_even):
        raise OperandPreparationError("negative midpoint ties-to-even self-test failed")
    signed_zero = torch.tensor([0.0, -0.0], dtype=torch.float32)
    if quantize_e2m1_codes(signed_zero).tolist() != [0, 8]:
        raise OperandPreparationError("signed-zero E2M1 self-test failed")
    endpoints = torch.tensor([6.0, -6.0, 7.0, -7.0], dtype=torch.float32)
    if quantize_e2m1_codes(endpoints).tolist() != [7, 15, 7, 15]:
        raise OperandPreparationError("E2M1 endpoint/saturation self-test failed")
    packed = pack_e2m1_codes(exact_codes)
    if packed.tolist() != [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]:
        raise OperandPreparationError("hardware nibble packing self-test failed")
    if not torch.equal(unpack_e2m1_codes(packed), exact_codes):
        raise OperandPreparationError("hardware nibble round-trip self-test failed")
    exponent_bytes = (
        torch.tensor(
            [UE8M0_EXPONENT_MIN, ZERO_ROW_EXPONENT, UE8M0_EXPONENT_MAX],
            dtype=torch.int32,
        )
        + UE8M0_BIAS
    ).to(torch.uint8)
    if exponent_bytes.tolist() != [0, 127, 254]:
        raise OperandPreparationError("UE8M0 exponent byte boundary test failed")
    zero = _quantize_whole_row(torch.zeros((1, LATENT_SIZE), dtype=torch.float32))
    if (
        bool(zero["codes"].any())
        or zero["ue8m0_codes"].unique().tolist() != [UE8M0_BIAS]
        or zero["diagnostics"]["exact_zero_rows"] != 1
    ):
        raise OperandPreparationError("zero-row canonicalization self-test failed")

    # A one-hot value of 1.0 has exact reconstructions at exponents -2..1.
    # Ascending enumeration must select the smaller exponent, -2.
    tied_row = torch.zeros((1, LATENT_SIZE), dtype=torch.float64)
    tied_row[0, 0] = 1.0
    tied = _quantize_whole_row(tied_row)
    if (
        tied["exponent"].item() != -2
        or tied["diagnostics"]["exact_sse_tie_rows"] != 1
        or tied["diagnostics"]["row_sse_max"] != 0.0
    ):
        raise OperandPreparationError("whole-row SSE tie rule self-test failed")

    # Independently enumerate a scalar row and compare its selected exponent,
    # codes, and SSE to the vectorized oracle.
    scalar_row = torch.linspace(-3.25, 4.75, LATENT_SIZE, dtype=torch.float64)[None]
    vector = _quantize_whole_row(scalar_row)
    scalar_best: tuple[float, int, torch.Tensor] | None = None
    for candidate in range(UE8M0_EXPONENT_MIN, UE8M0_EXPONENT_MAX + 1):
        scale = math.ldexp(1.0, candidate)
        codes = quantize_e2m1_codes(scalar_row[0] / scale)
        decoded = decode_e2m1_codes(codes).to(torch.float64) * scale
        sse = 0.0
        for coordinate in range(LATENT_SIZE):
            error = float(decoded[coordinate].item() - scalar_row[0, coordinate].item())
            sse += error * error
        if scalar_best is None or sse < scalar_best[0]:
            scalar_best = (sse, candidate, codes)
    assert scalar_best is not None
    if (
        vector["exponent"].item() != scalar_best[1]
        or not torch.equal(vector["codes"][0], scalar_best[2])
        or not math.isclose(
            vector["diagnostics"]["row_sse_max"],
            scalar_best[0],
            rel_tol=1.0e-14,
            abs_tol=1.0e-14,
        )
    ):
        raise OperandPreparationError("scalar exhaustive reference self-test failed")
    return {
        "exact_codes": 16,
        "midpoint_ties": 14,
        "signed_zero": True,
        "endpoints_and_out_of_range": True,
        "nibble_round_trip": True,
        "ue8m0_exponent_byte_boundaries": True,
        "whole_row_sse_tie_selects_smaller_exponent": True,
        "scalar_exhaustive_reference": True,
        "zero_row_unity": True,
        "pass": True,
    }


def _surface_operands(
    *,
    label: str,
    q_nope: torch.Tensor,
    q_inc: torch.Tensor,
    query_position: int,
    positions: torch.Tensor,
    q0_math: ModuleType,
    signs: Any,
) -> dict[str, Any]:
    eligible = positions <= query_position
    key_count = int(eligible.sum().item())
    expected = torch.arange(positions.numel()) < key_count
    if not torch.equal(eligible.cpu(), expected):
        raise OperandPreparationError(
            f"{label}: causal keys are not one ordered prefix"
        )
    q_rot = q0_math.e4m3fn(
        q0_math.rotate(q_nope[0].to(torch.float32), signs),
        label=f"{label} rotated query",
    ).to(torch.float8_e4m3fn)
    q_rope = q_inc[0, :, LATENT_SIZE:].to(torch.float8_e4m3fn)
    if q_rot.shape != (64, LATENT_SIZE) or q_rope.shape != (64, 64):
        raise OperandPreparationError(f"{label}: query geometry differs")
    return {
        "label": label,
        "query_position": int(query_position),
        "key_count": key_count,
        "q_rot_fp8": q_rot.cpu().contiguous(),
        "q_rope_fp8": q_rope.cpu().contiguous(),
    }


def _build_payload(
    *,
    records: Any,
    capture_hashes: dict[str, str],
    capture_bytes: int,
    q0_contract: ModuleType,
    q0_math: ModuleType,
    analyze_q0: ModuleType,
    q0_dir: Path,
    n8_operands: Path,
    n8_payload: dict[str, Any],
    n8_result: Path,
    self_test: dict[str, Any],
) -> dict[str, Any]:
    layers = []
    signs = q0_math.make_sign_contract("cpu")
    n8_layers = {
        int(layer["layer"]): layer for layer in n8_payload.get("layers", [])
    }
    if set(n8_layers) != set(q0_contract.SELECTED_LAYERS):
        raise OperandPreparationError("accepted N8 layer coverage differs")
    for layer_id in q0_contract.SELECTED_LAYERS:
        capture = analyze_q0._build_layer_capture(
            records,
            layer=layer_id,
            expected_length=q0_contract.INPUT_LENGTHS[STRATUM],
        )
        rotated = q0_math.rotate(capture.k_nope[:, 0].to(torch.float32), signs)
        n8_scale = n8_layers[int(layer_id)].get("token_scale_bf16")
        if not isinstance(n8_scale, torch.Tensor):
            raise OperandPreparationError(
                f"layer {layer_id}: accepted N8 token scale is missing"
            )
        quantized = _quantize_whole_row(rotated, n8_scale=n8_scale)
        if quantized["packed"].shape != (773, LATENT_SIZE // 2):
            raise OperandPreparationError(
                f"layer {layer_id}: packed latent shape differs"
            )
        key_rope = capture.k_pe_inc[:, 0].to(torch.float8_e4m3fn).cpu().contiguous()
        specs = [
            (
                "prefill_final",
                capture.prefill_q_nope,
                capture.prefill_q_inc,
                int(capture.prefill_position),
            )
        ]
        for row in range(q0_contract.VERIFY_ROWS):
            specs.append(
                (
                    f"target_verify_q{row}",
                    capture.verify_q_nope[row : row + 1],
                    capture.verify_q_inc[row : row + 1],
                    int(capture.verify_positions[row]),
                )
            )
        surfaces = [
            _surface_operands(
                label=label,
                q_nope=q_nope,
                q_inc=q_inc,
                query_position=query_position,
                positions=capture.positions,
                q0_math=q0_math,
                signs=signs,
            )
            for label, q_nope, q_inc, query_position in specs
        ]
        layers.append(
            {
                "layer": int(layer_id),
                "packed_key_e2m1": quantized["packed"].cpu().contiguous(),
                "row_ue8m0_codes": quantized["ue8m0_codes"].cpu().contiguous(),
                "key_rope_fp8": key_rope,
                "surfaces": surfaces,
                "quantization_diagnostics": quantized["diagnostics"],
            }
        )
    if len(layers) != 3 or sum(len(layer["surfaces"]) for layer in layers) != 18:
        raise OperandPreparationError("N9 artifact does not cover all 18 cells")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stratum": STRATUM,
        "representation": {
            "rotation": "signed_normalized_wht_seed42_no_l2_normalization",
            "scale_scope": "whole_512_coordinate_row",
            "scale_selection": "minimum_float64_reconstruction_sse",
            "candidate_exponents": [UE8M0_EXPONENT_MIN, UE8M0_EXPONENT_MAX],
            "sse_tie_rule": "smaller_exponent",
            "float64_reduction": "torch_sum_dim2_one_thread_pinned_runtime",
            "e2m1_rounding": "nearest_even_saturating_finite",
            "zero_row_scale": "unity",
            "e2m1_storage": "hardware_nibbles_low_coordinate_first",
            "ue8m0_bias": UE8M0_BIAS,
        },
        "capture_bytes": capture_bytes,
        "capture_sha256": dict(sorted(capture_hashes.items())),
        "q0_source_sha256": {
            name: _sha256_file(q0_dir / name) for name in EXPECTED_Q0_SOURCE_SHA256
        },
        "n8_operands_sha256": _sha256_file(n8_operands),
        "n8_result_sha256": _sha256_file(n8_result),
        "signs1_sha256": signs.signs1_sha256,
        "signs2_sha256": signs.signs2_sha256,
        "self_test": self_test,
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--q0-dir", type=Path)
    parser.add_argument("--n8-operands", type=Path)
    parser.add_argument("--n8-result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    self_test = _self_test()
    if args.self_test:
        print(_canonical_json_bytes(self_test).decode("utf-8"))
        return 0
    required = {
        "capture-root": args.capture_root,
        "q0-dir": args.q0_dir,
        "n8-operands": args.n8_operands,
        "n8-result": args.n8_result,
        "output": args.output,
        "manifest": args.manifest,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")

    _require_hash(args.n8_operands, EXPECTED_N8_OPERANDS_SHA256, "accepted N8 operands")
    _require_hash(args.n8_result, EXPECTED_N8_RESULT_SHA256, "accepted N8 result")
    n8_payload = torch.load(args.n8_operands, map_location="cpu", weights_only=True)
    if not isinstance(n8_payload, dict):
        raise OperandPreparationError("accepted N8 operand payload is not an object")
    q0_contract, q0_math, analyze_q0 = _load_q0_sources(args.q0_dir)
    records, capture_hashes, capture_bytes = analyze_q0._load_records(args.capture_root)
    if (
        len(capture_hashes) != EXPECTED_CAPTURE_FILES
        or capture_bytes != EXPECTED_CAPTURE_BYTES
    ):
        raise OperandPreparationError(
            f"capture identity differs: files={len(capture_hashes)}, bytes={capture_bytes}"
        )
    payload = _build_payload(
        records=records,
        capture_hashes=capture_hashes,
        capture_bytes=capture_bytes,
        q0_contract=q0_contract,
        q0_math=q0_math,
        analyze_q0=analyze_q0,
        q0_dir=args.q0_dir,
        n8_operands=args.n8_operands,
        n8_payload=n8_payload,
        n8_result=args.n8_result,
        self_test=self_test,
    )
    _atomic_torch_save(args.output, payload)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "artifact": str(args.output),
        "artifact_sha256": _sha256_file(args.output),
        "artifact_bytes": args.output.stat().st_size,
        "source_sha256": _sha256_file(Path(__file__)),
        "capture_files": len(capture_hashes),
        "capture_bytes": capture_bytes,
        "n8_operands_sha256": EXPECTED_N8_OPERANDS_SHA256,
        "n8_result_sha256": EXPECTED_N8_RESULT_SHA256,
        "layers": [
            {
                "layer": layer["layer"],
                "surface_key_counts": {
                    surface["label"]: surface["key_count"]
                    for surface in layer["surfaces"]
                },
                "quantization_diagnostics": layer["quantization_diagnostics"],
            }
            for layer in payload["layers"]
        ],
        "self_test": self_test,
    }
    _atomic_json(args.manifest, manifest)
    print(_canonical_json_bytes(manifest).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
