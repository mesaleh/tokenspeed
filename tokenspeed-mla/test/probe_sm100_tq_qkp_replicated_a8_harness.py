#!/usr/bin/env python3

"""Decision-grade address-disjoint harness for the frozen A4 C0/RN reader.

This file owns host allocations, poisoning, verification, graph populations,
ordering, and timing only.  The measured kernels are imported from the exact
A4 probe and guarded by a frozen source-region hash before compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import torch

import probe_sm100_tq_qkp_replicated as base


ARMS = ("c0", "rn", "native")
ARM_CONFIGS = {
    "c0": (0, 0, 0, 0, 0, 0),
    "rn": (0, 0, 1, 1, 1, 1),
    "native": (1, 1, 0, 0, 0, 0),
}
FROZEN_KERNEL_REGION_SHA256 = (
    "426748da54be0e3c9df405005f911273311dd039dc5755d4b43447a64811f3a1"
)
FROZEN_BASE_FILE_SHA256 = (
    "43e76da94fd6bafbd9e98225f97c85723cba15ab42eb10a67b312b7f73d7d8ce"
)
SCORED_NODES = 100
SCORED_WINDOWS = 30
SCORED_WARMUP_WINDOWS = 24
VERIFY_NODE_CHUNK = 4
SUPPORTED_TIMING_WINDOWS = (6, SCORED_WINDOWS)
T_CRITICAL_95 = {
    6: 2.570581836,
    30: 2.045229642,
}
EXPECTED_C512_CUBIN_SHA256 = {
    "c0": "7016c9c789f28de8dbc16eace265fcde5c8cf4db3f93af16e3eda9c9bda7041a",
    "rn": "3d2e562aaee004f5190e22bac131833113e60eaf8baba982cc7b4f55e8578569",
    "native": "59c2256118b0146f915e52f0034bbeffbdd7d7dbdc20da81adcaca5ff3443021",
}
EXPECTED_C512_SASS_COUNTS = {
    "c0": {
        "mma": 130,
        "tma": 60,
        "ldl": 102,
        "stl": 7,
        "mufu_rcp": 330,
        "fchk": 329,
        "calls": 332,
        "ffma": 1661,
    },
    "rn": {
        "mma": 130,
        "tma": 60,
        "ldl": 69,
        "stl": 1,
        "mufu_rcp": 10,
        "fchk": 9,
        "calls": 12,
        "ffma": 61,
    },
    "native": {
        "mma": 170,
        "tma": 70,
        "ldl": 0,
        "stl": 0,
        "mufu_rcp": 10,
        "fchk": 9,
        "calls": 12,
        "ffma": 61,
    },
}
COMPLETE_OUTPUT_FIELDS = {
    "layout",
    "normalized",
    "bf16",
    "carrier",
    "p",
    "max",
    "sum",
    "owner",
    "correction",
    "flags",
}


def assert_frozen_base_source() -> tuple[str, str]:
    source_path = Path(base.__file__)
    raw = source_path.read_bytes()
    file_digest = hashlib.sha256(raw).hexdigest()
    if file_digest != FROZEN_BASE_FILE_SHA256:
        raise AssertionError(
            f"A4 base/oracle file drifted: {file_digest} != "
            f"{FROZEN_BASE_FILE_SHA256}"
        )
    marker = b"\ndef fake("
    if marker not in raw:
        raise AssertionError("cannot locate frozen kernel/host boundary")
    region = raw.split(marker, 1)[0] + b"\n"
    digest = hashlib.sha256(region).hexdigest()
    if digest != FROZEN_KERNEL_REGION_SHA256:
        raise AssertionError(
            f"A4 kernel region drifted: {digest} != "
            f"{FROZEN_KERNEL_REGION_SHA256}"
        )
    return file_digest, digest


def williams_orders(windows: int) -> list[tuple[str, ...]]:
    if windows < 1 or windows % 6:
        raise ValueError("three-arm Williams windows must be a multiple of six")
    rows = list(itertools.permutations(ARMS))
    orders = rows * (windows // len(rows))
    expected_position = windows // len(ARMS)
    for arm in ARMS:
        for position in range(len(ARMS)):
            observed = sum(order[position] == arm for order in orders)
            if observed != expected_position:
                raise AssertionError(
                    f"position imbalance arm={arm} position={position} "
                    f"observed={observed} expected={expected_position}"
                )
    expected_adjacency = windows // len(ARMS)
    for first in ARMS:
        for second in ARMS:
            if first == second:
                continue
            observed = sum(
                sum(
                    order[position] == first
                    and order[position + 1] == second
                    for position in range(len(ARMS) - 1)
                )
                for order in orders
            )
            if observed != expected_adjacency:
                raise AssertionError(
                    f"adjacency imbalance first={first} second={second} "
                    f"observed={observed} expected={expected_adjacency}"
                )
    return orders


def paired_log_summary(
    numerators: list[float], denominators: list[float]
) -> dict[str, float | int]:
    if len(numerators) != len(denominators) or len(numerators) < 2:
        raise ValueError("paired summary requires equal nontrivial samples")
    logs = [
        math.log(numerator / denominator)
        for numerator, denominator in zip(numerators, denominators)
    ]
    count = len(logs)
    if count not in T_CRITICAL_95:
        raise ValueError(f"unsupported paired timing sample count: {count}")
    t_critical = T_CRITICAL_95[count]
    mean_log = statistics.fmean(logs)
    se = statistics.stdev(logs) / math.sqrt(count)
    return {
        "count": count,
        "geometric_ratio": math.exp(mean_log),
        "ci95_low": math.exp(mean_log - t_critical * se),
        "ci95_high": math.exp(mean_log + t_critical * se),
        "mean_log_ratio": mean_log,
    }


def scale_case_exponents(scale_case: int) -> tuple[int, ...]:
    active_tiles = (0, 1, 2, 4)
    values = list(range(-16, 17))
    start = scale_case * len(active_tiles)
    selected = [values[min(start + index, len(values) - 1)] for index in range(4)]
    result = [selected[0]] * base.TILES
    for tile, exponent in zip(active_tiles, selected):
        result[tile] = exponent
    result[base.MASKED_TILE] = result[base.MASKED_TILE - 1]
    return tuple(result)


def assert_scale_case_coverage() -> None:
    observed = {
        exponent
        for scale_case in range(9)
        for tile, exponent in enumerate(scale_case_exponents(scale_case))
        if tile != base.MASKED_TILE
    }
    expected = set(range(-16, 17))
    if observed != expected:
        raise AssertionError(
            f"scale cases do not cover carrier contract: {sorted(expected - observed)}"
        )


def make_host_pattern(
    scale_case: int, pv_scale_falsifier: int
) -> tuple[torch.Tensor, ...]:
    row = torch.arange(base.SCORE_ROWS, dtype=torch.int64).view(
        base.SCORE_ROWS, 1
    )
    token = torch.arange(base.TOKENS, dtype=torch.int64).view(base.TOKENS, 1)
    latent_coordinate = torch.arange(base.LATENT_K, dtype=torch.int64).view(
        1, base.LATENT_K
    )
    rope_coordinate = torch.arange(base.ROPE_K, dtype=torch.int64).view(
        1, base.ROPE_K
    )
    query = (
        (
            (
                (row + 1) * (latent_coordinate + 3) * 17
                + row * 37
                + latent_coordinate * 19
            )
            % 257
        )
        % 3
    ).float()
    query = (query - 1.0) * 0.5
    key_base = (
        (
            (
                (token + 1) * (latent_coordinate + 5) * 23
                + token * 41
                + latent_coordinate * 29
            )
            % 263
        )
        % 3
    ).float()
    key_base = (key_base - 1.0) * 0.5
    key_factors = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0]).view(
        base.TILES, 1, 1
    )
    key = key_base.unsqueeze(0) * key_factors
    rope_query = (
        (((row + 3) * (rope_coordinate + 1) * 11 + row * 13) % 127) % 3
    ).float()
    rope_query = (rope_query - 1.0) * 0.5
    rope_key_base = (
        (((token + 5) * (rope_coordinate + 7) * 19 + token * 17) % 131) % 3
    ).float()
    rope_key_base = (rope_key_base - 1.0) * 0.5
    rope_factors = torch.tensor([1.0, 0.5, 2.0, 1.0, 1.0]).view(
        base.TILES, 1, 1
    )
    rope_key = rope_key_base.unsqueeze(0) * rope_factors
    tile = torch.arange(base.TILES, dtype=torch.int64).view(base.TILES, 1)
    token_row = torch.arange(base.TOKENS, dtype=torch.int64).view(1, base.TOKENS)
    scale_factors = torch.tensor([1.0, 2.0, 0.5, 8.0, 4.0]).view(
        base.TILES, 1
    )
    token_scale = (
        (0.75 + ((tile * 7 + token_row * 3) % 17).float() / 32.0)
        * scale_factors
    ).to(torch.bfloat16)

    if scale_case >= 0:
        # Extreme carrier tests isolate scale/P state from the value path so
        # native E4M3 conversion cannot overflow and self-certify NaNs.
        key.zero_()
        exponents = scale_case_exponents(scale_case)
        for tile_index, exponent in enumerate(exponents):
            boundary = torch.tensor(
                224.0 * (2.0**exponent), dtype=torch.bfloat16
            )
            below = torch.nextafter(
                boundary,
                torch.tensor(float("-inf"), dtype=torch.bfloat16),
            )
            prior_boundary = torch.tensor(
                224.0 * (2.0 ** (exponent - 1)), dtype=torch.bfloat16
            )
            above_prior = torch.nextafter(
                prior_boundary,
                torch.tensor(float("inf"), dtype=torch.bfloat16),
            )
            if not (below < boundary and above_prior > prior_boundary):
                raise AssertionError("BF16 boundary neighbors collapsed")
            token_scale[tile_index].fill_(boundary * 0.5)
            token_scale[tile_index, 0] = below
            token_scale[tile_index, 1] = boundary
            token_scale[tile_index, 2] = above_prior

    if pv_scale_falsifier:
        query.zero_()
        rope_query.zero_()
        if torch.count_nonzero(torch.diff(key[0, :, 0], n=2)) == 0:
            raise AssertionError("PV scale diagnostic is accidentally affine")
    return query, key, rope_query, rope_key, token_scale


def repeat_clusters(
    values: tuple[torch.Tensor, ...], clusters: int
) -> tuple[torch.Tensor, ...]:
    return tuple(
        value.repeat((clusters,) + (1,) * (value.ndim - 1)) for value in values
    )


def compile_reader(
    arm: str,
    clusters: int,
    matrix_export: bool,
    overlap_setup: int,
    pv_sfa_exp: int,
    pv_sfb_exp: int,
):
    native_qk, native_pv, hoist, absolute, normalized, parallel = ARM_CONFIGS[arm]
    ctas = base.CLUSTER_SHAPE_MNK[0] * clusters
    return base.cute.compile(
        base.ownership_probe,
        base.make_ptr(
            base.cutlass.Float8E4M3FN,
            0,
            base.cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        base.make_ptr(
            base.cutlass.Float4E2M1FN,
            0,
            base.cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        base.make_ptr(
            base.cutlass.Float8E4M3FN,
            0,
            base.cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        base.make_ptr(
            base.cutlass.Float8E4M3FN,
            0,
            base.cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        base.make_ptr(
            base.cutlass.Float8E4M3FN,
            0,
            base.cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        base.fake(base.cutlass.Int32, (base.CLUSTER_SHAPE_MNK[0], 17), 16),
        base.fake(
            base.cutlass.Float32,
            (ctas, base.TILES, base.LATENT_K, base.ROWS_PER_CTA),
            16,
        ),
        base.fake(
            base.cutlass.Float32,
            (ctas, base.LATENT_K, base.ROWS_PER_CTA),
            16,
        ),
        base.fake(
            base.cutlass.BFloat16,
            (ctas, base.LATENT_K, base.ROWS_PER_CTA),
            16,
        ),
        base.fake(base.cutlass.Float32, (ctas, base.TILES), 16),
        base.fake(
            base.cutlass.Float8E4M3FN,
            (ctas, base.TILES, base.ROWS_PER_CTA, base.TOKENS),
            16,
        ),
        base.fake(
            base.cutlass.Float32,
            (ctas, base.TILES, base.ROWS_PER_CTA),
            16,
        ),
        base.fake(
            base.cutlass.Float32,
            (ctas, base.TILES, base.ROWS_PER_CTA),
            16,
        ),
        base.fake(
            base.cutlass.Int32,
            (ctas, base.TILES, base.ROWS_PER_CTA),
            16,
        ),
        base.fake(
            base.cutlass.Float32,
            (ctas, base.TILES, base.ROWS_PER_CTA, 3),
            16,
        ),
        base.fake(
            base.cutlass.Int32,
            (ctas, base.TILES, base.ROWS_PER_CTA),
            16,
        ),
        base.fake(base.cutlass.BFloat16, (base.TILES, base.TOKENS), 16),
        clusters,
        int(matrix_export),
        native_qk,
        native_pv,
        hoist,
        absolute,
        normalized,
        parallel,
        overlap_setup,
        pv_sfa_exp,
        pv_sfb_exp,
        base.make_fake_stream(),
        options="--enable-tvm-ffi --opt-level 3",
    )


def dump_generated(compiled: dict[str, object], dump_dir: Path) -> None:
    for arm, reader in compiled.items():
        arm_dir = dump_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for suffix, attribute, binary in (
            ("ptx", "__ptx__", False),
            ("sass", "__sass__", False),
            ("mlir", "__mlir__", False),
            ("cubin", "__cubin__", True),
        ):
            payload = getattr(reader, attribute, None)
            if payload is None:
                raise RuntimeError(f"CUTE_DSL_KEEP=all did not retain {attribute}")
            path = arm_dir / f"c0r.{suffix}"
            path.write_bytes(payload) if binary else path.write_text(payload)


def build_generated_audit(
    compiled: dict[str, object],
) -> dict[str, dict[str, str | int]]:
    result = {}
    for arm, reader in compiled.items():
        sass = reader.__sass__
        result[arm] = {
            "cubin_sha256": hashlib.sha256(reader.__cubin__).hexdigest(),
            "mma": sass.count(" UTCQMMA"),
            "tma": sass.count(" UTMALDG"),
            "ldl": sass.count(" LDL"),
            "stl": sass.count(" STL"),
            "mufu_rcp": sass.count(" MUFU.RCP"),
            "fchk": sass.count(" FCHK"),
            "calls": sass.count(" CALL.REL.NOINC"),
            "ffma": sass.count(" FFMA"),
        }
    return result


def require_c512_generated_identity(
    args: argparse.Namespace,
    audit: dict[str, dict[str, str | int]],
) -> None:
    exact_c512 = (
        args.clusters == 512
        and not args.matrix_export
        and args.overlap_setup == 1
        and not args.pv_sfa_exp
        and not args.pv_sfb_exp
    )
    if not exact_c512:
        return
    for arm, values in audit.items():
        cubin_sha256 = values["cubin_sha256"]
        if cubin_sha256 != EXPECTED_C512_CUBIN_SHA256[arm]:
            raise AssertionError(
                f"c512 {arm} cubin identity drifted: {cubin_sha256}"
            )
        counts = {key: values[key] for key in EXPECTED_C512_SASS_COUNTS[arm]}
        if counts != EXPECTED_C512_SASS_COUNTS[arm]:
            raise AssertionError(
                f"c512 {arm} SASS profile drifted: {counts} != "
                f"{EXPECTED_C512_SASS_COUNTS[arm]}"
            )


def expected_layout(arm: str, overlap_setup: int) -> torch.Tensor:
    native = int(arm == "native")
    return torch.tensor(
        [
            [
                1, 0, 64, 64, 20, 84, 8, 128, 64, 256, 256, 512,
                16384, 8192, overlap_setup, native, native,
            ],
            [
                2, 0, 64, 64, 20, 84, 8, 128, 64, 256, 256, 512,
                16384, 8192, overlap_setup, native, native,
            ],
        ],
        dtype=torch.int32,
        device="cuda",
    )


def expand_expected(expected: torch.Tensor, nodes: int) -> torch.Tensor:
    return expected.unsqueeze(0).expand((nodes,) + expected.shape)


def materialize_output(outputs: dict[str, object], name: str) -> torch.Tensor:
    value = outputs[name]
    if isinstance(value, list):
        return torch.stack(value)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"unexpected output holder: {name}")
    return value


def require_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    expanded = expand_expected(expected, actual.shape[0])
    if actual.dtype == torch.float8_e4m3fn:
        passed = torch.equal(actual.view(torch.uint8), expanded.view(torch.uint8))
    else:
        passed = torch.equal(actual, expanded)
    if not passed:
        raise AssertionError(f"exact verification failed: {name}")


def require_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> None:
    expanded = expand_expected(expected, actual.shape[0])
    if not torch.isclose(actual, expanded, rtol=rtol, atol=atol).all():
        error = (actual.float() - expanded.float()).abs().max().item()
        raise AssertionError(f"bounded verification failed: {name} max={error}")


def require_bounded_by_node(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    bound: torch.Tensor,
) -> None:
    for begin in range(0, actual.shape[0], VERIFY_NODE_CHUNK):
        end = min(begin + VERIFY_NODE_CHUNK, actual.shape[0])
        count = end - begin
        expected_chunk = expand_expected(expected, count)
        bound_chunk = expand_expected(bound, count)
        delta = (actual[begin:end] - expected_chunk).abs()
        if not (delta <= bound_chunk).all():
            excess = (delta - bound_chunk).max().item()
            raise AssertionError(
                f"{name} output exceeds error bound: nodes={begin}:{end} "
                f"max_excess={excess}"
            )


def verify_pool(
    arm: str,
    outputs: dict[str, object],
    expected: tuple[torch.Tensor | None, ...],
    matrix_export: bool,
    overlap_setup: int,
) -> None:
    checked: set[str] = set()
    expected_fields = set(COMPLETE_OUTPUT_FIELDS)
    if matrix_export:
        expected_fields.add("matrix")

    def exact(name: str, actual: torch.Tensor, oracle: torch.Tensor) -> None:
        require_exact(name, actual, oracle)
        checked.add(name)

    def close(
        name: str,
        actual: torch.Tensor,
        oracle: torch.Tensor,
        rtol: float,
        atol: float,
    ) -> None:
        require_close(name, actual, oracle, rtol, atol)
        checked.add(name)

    layout = expected_layout(arm, overlap_setup)
    layout_output = materialize_output(outputs, "layout")
    p_output = materialize_output(outputs, "p")
    max_output = materialize_output(outputs, "max")
    sum_output = materialize_output(outputs, "sum")
    owner_output = materialize_output(outputs, "owner")
    correction_output = materialize_output(outputs, "correction")
    flags_output = materialize_output(outputs, "flags")
    matrix_output = materialize_output(outputs, "matrix")
    normalized_output = materialize_output(outputs, "normalized")
    bf16_output = materialize_output(outputs, "bf16")
    carrier_output = materialize_output(outputs, "carrier")
    exact("layout", layout_output, layout)
    exact("p", p_output, expected[0])  # type: ignore[arg-type]
    exact("max", max_output, expected[1])  # type: ignore[arg-type]
    close("sum", sum_output, expected[2], 2.0e-6, 2.0e-5)  # type: ignore[arg-type]
    exact("owner", owner_output, expected[3])  # type: ignore[arg-type]
    require_close(
        "correction-state",
        correction_output[..., :2],
        expected[6][..., :2],  # type: ignore[index]
        2.0e-6,
        2.0e-5,
    )
    require_close(
        "correction-factor",
        correction_output[..., 2],
        expected[6][..., 2],  # type: ignore[index]
        2.0e-6,
        2.0e-5,
    )
    checked.add("correction")
    exact("flags", flags_output, expected[7])  # type: ignore[arg-type]
    if matrix_export:
        require_bounded_by_node(
            "matrix",
            matrix_output,
            expected[4],  # type: ignore[arg-type]
            expected[9],  # type: ignore[arg-type]
        )
        checked.add("matrix")
    require_bounded_by_node(
        "normalized",
        normalized_output,
        expected[8],  # type: ignore[arg-type]
        expected[10],  # type: ignore[arg-type]
    )
    checked.add("normalized")
    for begin in range(0, bf16_output.shape[0], VERIFY_NODE_CHUNK):
        end = min(begin + VERIFY_NODE_CHUNK, bf16_output.shape[0])
        if not torch.equal(
            bf16_output[begin:end].view(torch.int16),
            normalized_output[begin:end].to(torch.bfloat16).view(torch.int16),
        ):
            raise AssertionError(
                f"BF16 epilogue differs from normalized output: "
                f"nodes={begin}:{end}"
            )
    checked.add("bf16")
    exact("carrier", carrier_output, expected[5])  # type: ignore[arg-type]
    if checked != expected_fields:
        raise AssertionError(
            f"complete-state verifier coverage drifted: "
            f"checked={sorted(checked)} expected={sorted(expected_fields)}"
        )


def poison_outputs(outputs: dict[str, object], matrix_export: bool) -> None:
    for tensor in outputs["layout"]:  # type: ignore[union-attr]
        tensor.fill_(-1)
    if matrix_export:
        materialize_output(outputs, "matrix").fill_(float("nan"))
    materialize_output(outputs, "normalized").fill_(float("nan"))
    materialize_output(outputs, "bf16").fill_(float("nan"))
    for tensor in outputs["carrier"]:  # type: ignore[union-attr]
        tensor.fill_(float("nan"))
    materialize_output(outputs, "p").view(torch.uint8).fill_(0x7F)
    materialize_output(outputs, "max").fill_(float("nan"))
    materialize_output(outputs, "sum").fill_(float("nan"))
    materialize_output(outputs, "owner").fill_(-1)
    materialize_output(outputs, "correction").fill_(float("nan"))
    materialize_output(outputs, "flags").fill_(-1)


def allocation_ranges(
    node_pools: dict[str, torch.Tensor],
    independent_nodes: dict[str, list[torch.Tensor]],
    nodes: int,
) -> list[tuple[int, int, str, int]]:
    ranges = []
    for name, tensor in node_pools.items():
        if not tensor.is_contiguous() or tensor.shape[0] != nodes:
            raise AssertionError(f"node pool is not leading-contiguous: {name}")
        node_bytes = tensor[0].numel() * tensor.element_size()
        if tensor.stride(0) * tensor.element_size() != node_bytes:
            raise AssertionError(f"node stride contains padding: {name}")
        for node in range(nodes):
            begin = tensor.data_ptr() + node * node_bytes
            ranges.append((begin, begin + node_bytes, name, node))
    for name, tensors in independent_nodes.items():
        if len(tensors) != nodes:
            raise AssertionError(f"independent node count drifted: {name}")
        for node, tensor in enumerate(tensors):
            if not tensor.is_contiguous():
                raise AssertionError(f"independent input is not contiguous: {name}")
            begin = tensor.data_ptr()
            ranges.append(
                (
                    begin,
                    begin + tensor.numel() * tensor.element_size(),
                    name,
                    node,
                )
            )
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise AssertionError(
                f"allocation overlap: {previous[2]}[{previous[3]}] and "
                f"{current[2]}[{current[3]}]"
            )
    return ranges


def build_runtime(
    args: argparse.Namespace,
    compiled: dict[str, object],
):
    query, key, rope_query, rope_key, token_scale = make_host_pattern(
        args.scale_case, args.pv_sfa_exp + args.pv_sfb_exp
    )
    native_latent = (
        key.float() * token_scale.float().unsqueeze(-1)
    ).to(torch.float8_e4m3fn)
    require_mixed_max_wins = not (
        args.scale_case >= 0 or args.pv_sfa_exp or args.pv_sfb_exp
    )
    candidate_expected = base.expected(
        query,
        key,
        rope_query,
        rope_key,
        token_scale,
        True,
        pv_scale_exp=args.pv_sfa_exp + args.pv_sfb_exp,
        require_mixed_max_wins=require_mixed_max_wins,
    )
    native_expected = base.expected(
        query,
        key,
        rope_query,
        rope_key,
        token_scale,
        True,
        native_latent.float(),
        False,
        native_latent.float(),
        True,
        require_mixed_max_wins=False,
    )

    active_expected = set(range(12))
    if not args.matrix_export:
        active_expected -= {4, 9, 11}

    def expected_to_gpu(values: tuple[torch.Tensor, ...]):
        repeated = repeat_clusters(values, args.clusters)
        return tuple(
            value.cuda().contiguous() if index in active_expected else None
            for index, value in enumerate(repeated)
        )

    candidate_expected_gpu = expected_to_gpu(candidate_expected)
    expected = {
        "c0": candidate_expected_gpu,
        "rn": candidate_expected_gpu,
        "native": expected_to_gpu(native_expected),
    }

    def make_cute_nodes(source_node: torch.Tensor, dtype):
        source_cuda = source_node.cuda().contiguous()
        template_cute, template_backing = base.cutlass_torch.cute_tensor_like(
            source_node,
            dtype,
            is_dynamic_layout=True,
            assumed_align=16,
        )
        template_cute = base.cutlass_torch.convert_cute_tensor(
            source_cuda,
            template_cute,
            dtype,
            is_dynamic_layout=True,
        )
        cute_nodes = [template_cute]
        backings = [template_backing]
        for _ in range(1, args.nodes):
            backing = template_backing.clone()
            cute_tensor = base.cutlass_torch.from_dlpack(
                backing,
                assumed_align=16,
            )
            cute_tensor.element_type = dtype
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=base.cutlass_torch.get_leading_dim(backing)
            )
            cute_nodes.append(cute_tensor)
            backings.append(backing)
        return cute_nodes, backings

    query_node = query.unsqueeze(0).repeat(args.clusters, 1, 1)
    key_node = key.repeat(args.clusters, 1, 1)
    native_node = native_latent.float().repeat(args.clusters, 1, 1)
    rope_query_node = rope_query.unsqueeze(0).repeat(args.clusters, 1, 1)
    rope_key_node = rope_key.repeat(args.clusters, 1, 1)
    query_cute, query_backing = make_cute_nodes(
        query_node, base.cutlass.Float8E4M3FN
    )
    key_cute, key_backing = make_cute_nodes(key_node, base.cutlass.Float4E2M1FN)
    native_cute, native_backing = make_cute_nodes(
        native_node, base.cutlass.Float8E4M3FN
    )
    rope_query_cute, rope_query_backing = make_cute_nodes(
        rope_query_node, base.cutlass.Float8E4M3FN
    )
    rope_key_cute, rope_key_backing = make_cute_nodes(
        rope_key_node, base.cutlass.Float8E4M3FN
    )
    token_scale_pool = token_scale.cuda().contiguous().unsqueeze(0).repeat(
        args.nodes, 1, 1
    )

    ctas = base.CLUSTER_SHAPE_MNK[0] * args.clusters
    outputs = {
        # These two logical records have non-16-byte node strides at small
        # populations, so give every node an independently aligned allocation.
        "layout": [
            torch.empty(
                (base.CLUSTER_SHAPE_MNK[0], 17),
                dtype=torch.int32,
                device="cuda",
            )
            for _ in range(args.nodes)
        ],
        "matrix": torch.empty(
            (
                args.nodes,
                ctas,
                base.TILES,
                base.LATENT_K,
                base.ROWS_PER_CTA,
            ),
            dtype=torch.float32,
            device="cuda",
        ),
        "normalized": torch.empty(
            (args.nodes, ctas, base.LATENT_K, base.ROWS_PER_CTA),
            dtype=torch.float32,
            device="cuda",
        ),
        "bf16": torch.empty(
            (args.nodes, ctas, base.LATENT_K, base.ROWS_PER_CTA),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        "carrier": [
            torch.empty((ctas, base.TILES), dtype=torch.float32, device="cuda")
            for _ in range(args.nodes)
        ],
        "p": torch.empty(
            (
                args.nodes,
                ctas,
                base.TILES,
                base.ROWS_PER_CTA,
                base.TOKENS,
            ),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        "max": torch.empty(
            (args.nodes, ctas, base.TILES, base.ROWS_PER_CTA),
            dtype=torch.float32,
            device="cuda",
        ),
        "sum": torch.empty(
            (args.nodes, ctas, base.TILES, base.ROWS_PER_CTA),
            dtype=torch.float32,
            device="cuda",
        ),
        "owner": torch.empty(
            (args.nodes, ctas, base.TILES, base.ROWS_PER_CTA),
            dtype=torch.int32,
            device="cuda",
        ),
        "correction": torch.empty(
            (args.nodes, ctas, base.TILES, base.ROWS_PER_CTA, 3),
            dtype=torch.float32,
            device="cuda",
        ),
        "flags": torch.empty(
            (args.nodes, ctas, base.TILES, base.ROWS_PER_CTA),
            dtype=torch.int32,
            device="cuda",
        ),
    }

    independent_nodes = {
        "query": query_backing,
        "key": key_backing,
        "native": native_backing,
        "rope_query": rope_query_backing,
        "rope_key": rope_key_backing,
        "layout": outputs["layout"],
        "carrier": outputs["carrier"],
    }
    node_pools = {
        "token_scale": token_scale_pool,
        **{
            name: tensor
            for name, tensor in outputs.items()
            if name not in ("layout", "carrier")
        },
    }
    ranges = allocation_ranges(node_pools, independent_nodes, args.nodes)
    if len(ranges) != args.nodes * (
        len(independent_nodes) + len(node_pools)
    ):
        raise AssertionError("allocation cardinality self-check failed")

    def launch_node(arm: str, node: int, stream) -> None:
        compiled[arm](
            query_cute[node].iterator,
            key_cute[node].iterator,
            native_cute[node].iterator,
            rope_query_cute[node].iterator,
            rope_key_cute[node].iterator,
            outputs["layout"][node],
            outputs["matrix"][node],
            outputs["normalized"][node],
            outputs["bf16"][node],
            outputs["carrier"][node],
            outputs["p"][node],
            outputs["max"][node],
            outputs["sum"][node],
            outputs["owner"][node],
            outputs["correction"][node],
            outputs["flags"][node],
            token_scale_pool[node],
            stream,
        )

    return outputs, expected, launch_node, ranges


def run_verifier_negative_control(
    outputs: dict[str, object],
    expected: tuple[torch.Tensor | None, ...],
    matrix_export: bool,
    overlap_setup: int,
) -> None:
    flags_output = materialize_output(outputs, "flags")
    original = int(flags_output[0, 0, 0, 0].item())
    flags_output[0, 0, 0, 0] = original ^ 1
    detected = False
    try:
        verify_pool("c0", outputs, expected, matrix_export, overlap_setup)
    except AssertionError:
        detected = True
    flags_output[0, 0, 0, 0] = original
    if not detected:
        raise AssertionError("verification negative control was not detected")
    verify_pool("c0", outputs, expected, matrix_export, overlap_setup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--graph-replays", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--warmup-windows", type=int, default=6)
    parser.add_argument("--matrix-export", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--arm", choices=("all",) + ARMS, default="all")
    parser.add_argument("--scale-case", type=int, choices=tuple(range(-1, 9)), default=-1)
    parser.add_argument("--overlap-setup", type=int, choices=(0, 1), default=1)
    parser.add_argument("--pv-sfa-exp", type=int, choices=(0, 1), default=0)
    parser.add_argument("--pv-sfb-exp", type=int, choices=(0, 1), default=0)
    parser.add_argument("--dump-generated-dir")
    args = parser.parse_args()
    if args.clusters < 1 or args.nodes < 1:
        parser.error("--clusters and --nodes must be positive")
    if args.graph_replays < 0:
        parser.error("--graph-replays must be non-negative")
    if args.windows not in SUPPORTED_TIMING_WINDOWS:
        parser.error(
            "--windows must be 6 (rehearsal) or 30 (scored candidate)"
        )
    if args.warmup_windows < 0 or args.warmup_windows % 6:
        parser.error("--warmup-windows must be a non-negative multiple of six")
    if args.pv_sfa_exp and args.pv_sfb_exp:
        parser.error("falsify SFA and SFB independently")
    if args.scale_case >= 0 and (args.pv_sfa_exp or args.pv_sfb_exp):
        parser.error("scale cases and PV scale falsifiers are separate runs")
    if args.benchmark and args.arm != "all":
        parser.error("benchmark requires all three arms")
    scored = args.benchmark and args.windows == SCORED_WINDOWS
    if scored and (
        args.nodes != SCORED_NODES
        or args.warmup_windows != SCORED_WARMUP_WINDOWS
        or args.clusters not in (128, 512)
        or args.matrix_export
        or args.graph_replays
        or args.overlap_setup != 1
        or args.scale_case != -1
        or args.pv_sfa_exp
        or args.pv_sfb_exp
    ):
        parser.error(
            "scored timing requires nodes100, warmup24, windows30, "
            "clusters128/512, overlap-setup1, the default scale case, "
            "unity PV scales, no matrix export, and no graph-replay loop"
        )

    base_digest, kernel_digest = assert_frozen_base_source()
    assert_scale_case_coverage()
    selected = ARMS if args.arm == "all" else (args.arm,)
    compiled = {
        arm: compile_reader(
            arm,
            args.clusters,
            args.matrix_export,
            args.overlap_setup,
            args.pv_sfa_exp,
            args.pv_sfb_exp,
        )
        for arm in selected
    }
    if args.dump_generated_dir:
        dump_generated(compiled, Path(args.dump_generated_dir))
    generated_audit = build_generated_audit(compiled)
    require_c512_generated_identity(args, generated_audit)
    print(
        "A8_GENERATED_AUDIT_JSON="
        + json.dumps(generated_audit, sort_keys=True),
        flush=True,
    )
    if args.compile_only:
        print(
            "PASS_A8_COMPILE_ONLY "
            f"base_sha256={base_digest} kernel_sha256={kernel_digest} "
            f"clusters={args.clusters} "
            f"arms={','.join(selected)}",
            flush=True,
        )
        return

    outputs, expected, launch_node, ranges = build_runtime(args, compiled)
    print(
        "PASS_A8_ALLOCATION_SELF_CHECK "
        f"nodes={args.nodes} ranges={len(ranges)} "
        f"allocated_bytes={torch.cuda.memory_allocated()}",
        flush=True,
    )
    stream = base.cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for arm in selected:
        poison_outputs(outputs, args.matrix_export)
        for node in range(args.nodes):
            launch_node(arm, node, stream)
        torch.cuda.synchronize()
        verify_pool(
            arm, outputs, expected[arm], args.matrix_export, args.overlap_setup
        )
        print(
            f"PASS_A8_EAGER arm={arm} clusters={args.clusters} nodes={args.nodes}",
            flush=True,
        )
        if args.graph_replays or args.benchmark:
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                capture_stream = base.cuda_driver.CUstream(
                    torch.cuda.current_stream().cuda_stream
                )
                for node in range(args.nodes):
                    launch_node(arm, node, capture_stream)
            graphs[arm] = graph
            for replay in range(args.graph_replays):
                poison_outputs(outputs, args.matrix_export)
                torch.cuda.synchronize()
                graph.replay()
                torch.cuda.synchronize()
                verify_pool(
                    arm,
                    outputs,
                    expected[arm],
                    args.matrix_export,
                    args.overlap_setup,
                )
            if args.graph_replays:
                print(
                    f"PASS_A8_GRAPH arm={arm} clusters={args.clusters} "
                    f"nodes={args.nodes} replays={args.graph_replays}",
                    flush=True,
                )

    if selected == ARMS:
        # Run against a valid C0 state, then restore and reverify internally.
        poison_outputs(outputs, args.matrix_export)
        for node in range(args.nodes):
            launch_node("c0", node, stream)
        torch.cuda.synchronize()
        verify_pool(
            "c0", outputs, expected["c0"], args.matrix_export, args.overlap_setup
        )
        run_verifier_negative_control(
            outputs, expected["c0"], args.matrix_export, args.overlap_setup
        )
        print("PASS_A8_VERIFIER_NEGATIVE_CONTROL", flush=True)

    if not args.benchmark:
        return

    audit_events: list[tuple[str, int, str]] = []
    warmup_orders = (
        williams_orders(args.warmup_windows) if args.warmup_windows else []
    )
    for warmup, order in enumerate(warmup_orders, start=1):
        for arm in order:
            audit_events.append(("poison", -warmup, arm))
            poison_outputs(outputs, args.matrix_export)
            torch.cuda.synchronize()
            graphs[arm].replay()
            torch.cuda.synchronize()
            verify_pool(
                arm, outputs, expected[arm], args.matrix_export, args.overlap_setup
            )
            audit_events.append(("verify", -warmup, arm))

    samples = {arm: [] for arm in ARMS}
    raw_windows = []
    for window, order in enumerate(williams_orders(args.windows), start=1):
        row = {"window": window, "order": list(order), "timings_us": {}}
        for arm in order:
            audit_events.append(("poison", window, arm))
            poison_outputs(outputs, args.matrix_export)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            audit_events.append(("event_start", window, arm))
            start.record()
            graphs[arm].replay()
            end.record()
            end.synchronize()
            audit_events.append(("event_end", window, arm))
            elapsed_us = start.elapsed_time(end) * 1000.0 / args.nodes
            verify_pool(
                arm, outputs, expected[arm], args.matrix_export, args.overlap_setup
            )
            audit_events.append(("verify", window, arm))
            samples[arm].append(elapsed_us)
            row["timings_us"][arm] = elapsed_us
        raw_windows.append(row)

    for window in range(1, args.windows + 1):
        for arm in ARMS:
            stages = [
                stage
                for stage, event_window, event_arm in audit_events
                if event_window == window and event_arm == arm
            ]
            if stages != ["poison", "event_start", "event_end", "verify"]:
                raise AssertionError(
                    f"event ordering drift window={window} arm={arm}: {stages}"
                )
    for warmup in range(1, args.warmup_windows + 1):
        for arm in ARMS:
            stages = [
                stage
                for stage, event_window, event_arm in audit_events
                if event_window == -warmup and event_arm == arm
            ]
            if stages != ["poison", "verify"]:
                raise AssertionError(
                    f"warmup ordering drift window={warmup} arm={arm}: {stages}"
                )

    ratios = {
        "rn_vs_native": paired_log_summary(samples["rn"], samples["native"]),
        "c0_vs_native": paired_log_summary(samples["c0"], samples["native"]),
        "rn_vs_c0": paired_log_summary(samples["rn"], samples["c0"]),
    }
    summary = {
        arm: {
            "mean_us": statistics.fmean(values),
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
        }
        for arm, values in samples.items()
    }
    result = {
        "status": "PASS_A8_TIMING_SCORED" if scored else "PASS_A8_TIMING_REHEARSAL",
        "scored": scored,
        "base_source_sha256": base_digest,
        "kernel_region_sha256": kernel_digest,
        "clusters": args.clusters,
        "nodes": args.nodes,
        "windows": args.windows,
        "warmup_windows": args.warmup_windows,
        "scale_case": args.scale_case,
        "overlap_setup": args.overlap_setup,
        "pv_sfa_exp": args.pv_sfa_exp,
        "pv_sfb_exp": args.pv_sfb_exp,
        "matrix_export": args.matrix_export,
        "graph_replays": args.graph_replays,
        "generated_audit": generated_audit,
        "summary": summary,
        "ratios": ratios,
        "raw_windows": raw_windows,
    }
    print("A8_TIMING_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
