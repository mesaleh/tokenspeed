#!/usr/bin/env python3
"""Two-stage analysis and internal-pilot selection for H43."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from h43_codebook_ab_common import (
    bootstrap_lower_bound,
    canonical_json_digest,
    load_contract,
    pilot_sigma,
    projected_no_decision_probability,
    select_process_count,
    telemetry_reasons,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("pilot", "validity", "performance"), required=True
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--decision-contract", type=Path)
    parser.add_argument("--qualification-gates", type=Path)
    parser.add_argument("--validity-result", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def result_path(
    root: Path, section: str, context: int, sequence: int | None = None
) -> Path:
    base = root / section / f"context{context}"
    if sequence is not None:
        base /= f"seq{sequence:02}"
    return base / "result.json"


def validate_result_inventory(
    root: Path, expected: list[Path], section: str, failures: list[str]
) -> None:
    section_root = root / section
    observed = (
        {path.relative_to(root) for path in section_root.rglob("result.json")}
        if section_root.is_dir()
        else set()
    )
    expected_relative = {path.relative_to(root) for path in expected}
    if observed != expected_relative:
        failures.append(
            f"{section} result inventory differs: observed={sorted(map(str, observed))}, "
            f"expected={sorted(map(str, expected_relative))}"
        )


def expect_equal(observed: Any, expected: Any, label: str, failures: list[str]) -> None:
    if observed != expected:
        failures.append(f"{label}: observed {observed!r}, expected {expected!r}")


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def balanced_process_recovery(
    pairs: list[dict[str, Any]], selected_layers: int
) -> float:
    recoveries_by_order: dict[str, list[float]] = {"AB": [], "BA": []}
    for pair in pairs:
        if not pair["telemetry_valid"]:
            continue
        order = "AB" if pair["order"] == ["no_codebook", "codebook"] else "BA"
        recoveries_by_order[order].append(
            (pair["durations_us"]["no_codebook"] - pair["durations_us"]["codebook"])
            / selected_layers
        )
    if not all(recoveries_by_order.values()):
        raise ValueError("balanced process recovery requires both retained orders")
    return statistics.fmean(
        statistics.fmean(recoveries_by_order[order]) for order in ("AB", "BA")
    )


def validate_qualification_gates(
    value: dict[str, Any], contract: dict[str, Any], failures: list[str]
) -> None:
    expected_names = contract["qualification"]["required_gates"]
    expected = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": contract["experiment"],
        "contract_digest": canonical_json_digest(contract),
        "physical_host": contract["machine"]["physical_host"],
        "device_uuid": contract["machine"]["gpu_uuid"],
    }
    for key, expected_value in expected.items():
        expect_equal(
            value.get(key), expected_value, f"qualification_gates.{key}", failures
        )
    gates = value.get("gates", [])
    expect_equal(
        [gate.get("name") for gate in gates],
        expected_names,
        "qualification_gates.names",
        failures,
    )
    for gate in gates:
        name = gate.get("name", "unknown")
        expect_equal(
            gate.get("status"), "PASS", f"qualification_gates.{name}.status", failures
        )
        if not valid_sha256(gate.get("evidence_sha256")):
            failures.append(f"qualification_gates.{name}.evidence_sha256 is invalid")
    for label in (
        "source_manifest_digest",
        "installed_mla_sha256",
        "cache_artifact_digest",
        "codebook_sha256",
    ):
        if not valid_sha256(value.get(label)):
            failures.append(f"qualification_gates.{label} is invalid")
    aggregate = value.get("aggregate_ecc_baseline")
    if not isinstance(aggregate, str) or not aggregate.isdigit():
        failures.append("qualification_gates.aggregate_ecc_baseline is invalid")
    observed_digest = value.get("qualification_gates_digest")
    without_digest = dict(value)
    without_digest.pop("qualification_gates_digest", None)
    try:
        expected_digest = canonical_json_digest(without_digest)
    except (TypeError, ValueError) as error:
        failures.append(f"qualification_gates is not canonical JSON: {error}")
        expected_digest = None
    expect_equal(
        observed_digest,
        expected_digest,
        "qualification_gates.digest",
        failures,
    )


def validate_decision_contract(
    value: dict[str, Any], contract: dict[str, Any], failures: list[str]
) -> None:
    value_copy = dict(value)
    observed_digest = value_copy.pop("decision_contract_digest", None)
    try:
        expected_digest = canonical_json_digest(value_copy)
    except (TypeError, ValueError) as error:
        failures.append(f"decision_contract is not canonical JSON: {error}")
        expected_digest = None
    expect_equal(
        observed_digest,
        expected_digest,
        "decision_contract.digest",
        failures,
    )
    expected = {
        "schema_version": 1,
        "status": "READY",
        "experiment": contract["experiment"],
        "contract_digest": canonical_json_digest(contract),
        "failures": [],
    }
    for key, expected_value in expected.items():
        expect_equal(
            value.get(key), expected_value, f"decision_contract.{key}", failures
        )
    process_count = value.get("processes_per_context")
    pilot = contract["pilot"]
    allowed_counts = {
        pilot["base_processes_per_context"],
        pilot["expanded_processes_per_context"],
    }
    if process_count not in allowed_counts:
        failures.append(f"decision_contract.processes_per_context={process_count!r}")
    else:
        expected_timeout = contract["timeouts_seconds"][
            f"decision_n{process_count}_experiment"
        ]
        expect_equal(
            value.get("decision_timeout_seconds"),
            expected_timeout,
            "decision_contract.decision_timeout_seconds",
            failures,
        )
    for label in (
        "pilot_raw_inputs_digest",
        "qualification_gates_digest",
        "cache_artifact_digest",
        "source_manifest_digest",
        "installed_mla_sha256",
        "codebook_sha256",
    ):
        if not valid_sha256(value.get(label)):
            failures.append(f"decision_contract.{label} is invalid")
    aggregate = value.get("aggregate_ecc_baseline")
    if not isinstance(aggregate, str) or not aggregate.isdigit():
        failures.append("decision_contract.aggregate_ecc_baseline is invalid")
    sigma = value.get("sigma_pilot_us_per_layer")
    if not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma < 0:
        failures.append("decision_contract.sigma_pilot_us_per_layer is invalid")
    elif process_count == pilot["base_processes_per_context"]:
        if sigma > pilot["base_sigma_max_us_per_layer"]:
            failures.append("decision_contract base n is inconsistent with pilot sigma")
    elif process_count == pilot["expanded_processes_per_context"] and not (
        pilot["base_sigma_max_us_per_layer"]
        < sigma
        <= pilot["expanded_sigma_max_us_per_layer"]
    ):
        failures.append("decision_contract expanded n is inconsistent with pilot sigma")
    projected = value.get("projected_campaign_no_decision_probability")
    if (
        not isinstance(projected, (int, float))
        or not math.isfinite(projected)
        or projected < 0
        or projected > pilot["max_projected_campaign_no_decision_probability"]
    ):
        failures.append("decision_contract projected invalidation is invalid")
    runtime = value.get("projected_decision_runtime_seconds")
    timeout = value.get("decision_timeout_seconds")
    if (
        not finite_positive(runtime)
        or not finite_positive(timeout)
        or runtime > timeout
    ):
        failures.append("decision_contract projected runtime is invalid")


def validate_gpu_sample(
    sample: dict[str, str],
    contract: dict[str, Any],
    aggregate_ecc_baseline: str,
    label: str,
    failures: list[str],
) -> None:
    for reason in telemetry_reasons(sample, contract):
        failures.append(f"{label}: {reason}")
    machine = contract["machine"]
    expected = {
        "ecc.errors.uncorrected.volatile.total": "0",
        "ecc.errors.uncorrected.aggregate.total": aggregate_ecc_baseline,
        "gpu_recovery_action": machine["recovery_action"],
        "fabric.state": machine["fabric_state"],
        "fabric.status": machine["fabric_status"],
    }
    for key, value in expected.items():
        expect_equal(sample.get(key), value, f"{label}.{key}", failures)


def validate_gpu_binding_and_health(
    sample: dict[str, str],
    contract: dict[str, Any],
    aggregate_ecc_baseline: str,
    label: str,
    failures: list[str],
) -> None:
    machine = contract["machine"]
    expected = {
        "index": str(machine["gpu_index"]),
        "uuid": machine["gpu_uuid"],
        "name": machine["gpu_name"],
        "ecc.errors.uncorrected.volatile.total": "0",
        "ecc.errors.uncorrected.aggregate.total": aggregate_ecc_baseline,
        "gpu_recovery_action": machine["recovery_action"],
        "fabric.state": machine["fabric_state"],
        "fabric.status": machine["fabric_status"],
    }
    for key, value in expected.items():
        expect_equal(sample.get(key), value, f"{label}.{key}", failures)


def validate_gpu_boundaries(
    result: dict[str, Any],
    contract: dict[str, Any],
    aggregate_ecc_baseline: str,
    mode: str,
    label: str,
    failures: list[str],
) -> None:
    validate_gpu_binding_and_health(
        result.get("gpu_before_cuda", {}),
        contract,
        aggregate_ecc_baseline,
        f"{label}.gpu_before_cuda",
        failures,
    )
    final_validator = (
        validate_gpu_sample if mode == "smoke" else validate_gpu_binding_and_health
    )
    final_validator(
        result.get("gpu_final", {}),
        contract,
        aggregate_ecc_baseline,
        f"{label}.gpu_final",
        failures,
    )


def validate_correctness(
    result: dict[str, Any], contract: dict[str, Any], label: str, failures: list[str]
) -> None:
    expected_q = contract["geometry"]["correctness_query_lengths"]
    cases = result.get("correctness", [])
    expect_equal(
        [case.get("q_len") for case in cases],
        expected_q,
        f"{label}.q_lengths",
        failures,
    )
    selected = contract["geometry"]["selected_layers"]
    atol = contract["correctness"]["dense_oracle_atol"]
    for case in cases:
        q_label = f"{label}.q{case.get('q_len')}"
        expect_equal(
            case.get("eager_all_14_bit_identical"), True, f"{q_label}.eager", failures
        )
        expect_equal(
            case.get("graph_all_14_bit_identical"), True, f"{q_label}.graph", failures
        )
        expect_equal(
            case.get("dense_oracle_layer"), 0, f"{q_label}.oracle_layer", failures
        )
        differences = case.get("dense_max_abs_differences", [])
        if len(differences) != 2 or any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or value > atol
            for value in differences
        ):
            failures.append(
                f"{q_label}: invalid dense-oracle differences {differences}"
            )
        validate_allocation_pair(
            case.get("graph_replay_allocation_before"),
            case.get("graph_replay_allocation_after"),
            f"{q_label}.allocation",
            failures,
        )
    compile_evidence = result.get("compile_evidence", [])
    expected_labels = [
        f"{family}-q{q_len}"
        for q_len in expected_q
        for family in ("dense", "tq-no-codebook", "tq-codebook")
    ]
    expect_equal(
        [item.get("label") for item in compile_evidence],
        expected_labels,
        f"{label}.compile_labels",
        failures,
    )
    for item in compile_evidence:
        expect_equal(
            item.get("misses"), 1, f"{label}.{item.get('label')}.misses", failures
        )
        expect_equal(
            item.get("currsize"), 1, f"{label}.{item.get('label')}.currsize", failures
        )
    checksums = result.get("selected_output_checksums")
    if checksums is not None:
        no_codebook = checksums.get("no_codebook", [])
        codebook = checksums.get("codebook", [])
        expect_equal(
            len(no_codebook), selected, f"{label}.no_codebook_checksums", failures
        )
        expect_equal(len(codebook), selected, f"{label}.codebook_checksums", failures)
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in [*no_codebook, *codebook]
        ):
            failures.append(f"{label}.selected_output_checksums are invalid")
    if selected != 14:
        failures.append("contract selected-layer count drifted from 14")


def validate_allocation_pair(
    before: Any, after: Any, label: str, failures: list[str]
) -> None:
    if (
        not isinstance(before, int)
        or isinstance(before, bool)
        or before < 0
        or not isinstance(after, int)
        or isinstance(after, bool)
        or after < 0
    ):
        failures.append(f"{label}: allocation counters are invalid")
    else:
        expect_equal(after, before, label, failures)


def validate_common_result(
    result: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_digest: str,
    context: int,
    sequence: int,
    mode: str,
    expected_cache_digest: str | None,
    expected_source_digest: str | None,
    expected_installed_mla_digest: str | None,
    expected_aggregate_ecc: str | None,
    expected_codebook_digest: str | None,
    failures: list[str],
) -> None:
    label = f"{mode}.context{context}.seq{sequence:02}"
    machine = contract["machine"]
    context_contract = contract["contexts"][str(context)]
    expected = {
        "schema_version": 1,
        "status": mode.upper(),
        "experiment": contract["experiment"],
        "mode": mode,
        "physical_host": machine["physical_host"],
        "container_hostname": machine["container_hostname"],
        "device_name": machine["gpu_name"],
        "device_uuid": machine["gpu_uuid"],
        "contract_digest": contract_digest,
        "context": context,
        "requested_split": context_contract["requested_split"],
        "effective_nonempty_split": context_contract["effective_nonempty_split"],
        "tq4_tiles_per_split": context_contract["tq4_tiles_per_split"],
        "sequence": sequence,
        "page_table_seed": contract["seeds"]["page_table_base"] + sequence,
        "memory": contract["memory_bytes_per_rank"],
        "applications_before_cuda": [],
    }
    for key, value in expected.items():
        expect_equal(result.get(key), value, f"{label}.{key}", failures)
    if expected_source_digest is not None:
        expect_equal(
            result.get("source_manifest_digest"),
            expected_source_digest,
            f"{label}.source_manifest_digest",
            failures,
        )
    if expected_installed_mla_digest is not None:
        expect_equal(
            result.get("installed_mla_sha256"),
            expected_installed_mla_digest,
            f"{label}.installed_mla_sha256",
            failures,
        )
    if expected_aggregate_ecc is not None:
        expect_equal(
            str(result.get("aggregate_ecc_baseline")),
            expected_aggregate_ecc,
            f"{label}.aggregate_ecc_baseline",
            failures,
        )
    if expected_codebook_digest is not None:
        expect_equal(
            result.get("codebook_sha256"),
            expected_codebook_digest,
            f"{label}.codebook_sha256",
            failures,
        )
    before_digest = result.get("cache_artifacts_before", {}).get("digest")
    after_digest = result.get("cache_artifacts_after", {}).get("digest")
    expect_equal(after_digest, before_digest, f"{label}.cache_unchanged", failures)
    if expected_cache_digest is not None:
        expect_equal(
            before_digest, expected_cache_digest, f"{label}.cache_digest", failures
        )
    aggregate = str(result.get("aggregate_ecc_baseline"))
    validate_gpu_boundaries(result, contract, aggregate, mode, label, failures)
    validate_correctness(result, contract, label, failures)
    if not valid_sha256(result.get("codebook_sha256")):
        failures.append(f"{label}.codebook_sha256 is invalid")
    if not valid_sha256(result.get("page_table_sha256")):
        failures.append(f"{label}.page_table_sha256 is invalid")
    query_digests = result.get("query_sha256", {})
    expect_equal(
        sorted(query_digests),
        [str(value) for value in contract["geometry"]["correctness_query_lengths"]],
        f"{label}.query_sha256.keys",
        failures,
    )
    if any(not valid_sha256(value) for value in query_digests.values()):
        failures.append(f"{label}.query_sha256 is invalid")
    free_bytes = result.get("free_bytes_before_allocations")
    total_bytes = result.get("total_device_bytes")
    if (
        not isinstance(free_bytes, int)
        or free_bytes < 20 << 30
        or not isinstance(total_bytes, int)
        or total_bytes < free_bytes
    ):
        failures.append(f"{label}.device_memory is invalid")
    if not finite_positive(result.get("wall_time_seconds")):
        failures.append(f"{label}.wall_time_seconds is invalid")
    raw_digest = result.get("raw_result_digest")
    without_digest = dict(result)
    without_digest.pop("raw_result_digest", None)
    expect_equal(
        raw_digest,
        canonical_json_digest(without_digest),
        f"{label}.raw_digest",
        failures,
    )


def pilot_stage(
    root: Path, contract: dict[str, Any], qualification_gates: dict[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    contract_digest = canonical_json_digest(contract)
    contexts = [int(value) for value in contract["contexts"]]
    process_pairs: list[tuple[int, list[float]]] = []
    results: list[dict[str, Any]] = []
    contamination = 0
    source_digests: set[str] = set()
    cache_digests: set[str] = set()
    codebook_digests: set[str] = set()
    installed_mla_digests: set[str] = set()
    aggregate_ecc_baselines: set[str] = set()
    page_table_digests: dict[int, set[str]] = {}
    query_digest_sets: set[str] = set()
    second_wall_times: list[float] = []
    validate_qualification_gates(qualification_gates, contract, failures)
    expected_qualification = [
        result_path(root, "qualification/smoke", context) for context in contexts
    ] + [
        result_path(root, "qualification/preflight", context, sequence)
        for sequence in (1, 2)
        for context in contexts
    ]
    validate_result_inventory(root, expected_qualification, "qualification", failures)
    for context in contexts:
        smoke = load_json(result_path(root, "qualification/smoke", context))
        validate_common_result(
            smoke,
            contract=contract,
            contract_digest=contract_digest,
            context=context,
            sequence=1,
            mode="smoke",
            expected_cache_digest=None,
            expected_source_digest=None,
            expected_installed_mla_digest=None,
            expected_aggregate_ecc=None,
            expected_codebook_digest=None,
            failures=failures,
        )
        results.append(smoke)
        source_digests.add(smoke.get("source_manifest_digest", ""))
        cache_digests.add(smoke.get("cache_artifacts_after", {}).get("digest", ""))
        codebook_digests.add(smoke.get("codebook_sha256", ""))
        installed_mla_digests.add(smoke.get("installed_mla_sha256", ""))
        aggregate_ecc_baselines.add(str(smoke.get("aggregate_ecc_baseline")))
        page_table_digests.setdefault(1, set()).add(smoke.get("page_table_sha256", ""))
        query_digest_sets.add(canonical_json_digest(smoke.get("query_sha256", {})))
        for sequence in (1, 2):
            result = load_json(
                result_path(root, "qualification/preflight", context, sequence)
            )
            validate_common_result(
                result,
                contract=contract,
                contract_digest=contract_digest,
                context=context,
                sequence=sequence,
                mode="preflight",
                expected_cache_digest=None,
                expected_source_digest=None,
                expected_installed_mla_digest=None,
                expected_aggregate_ecc=None,
                expected_codebook_digest=None,
                failures=failures,
            )
            if "pairs" in result or "sentinel_before" in result:
                failures.append(
                    f"preflight context{context} seq{sequence} exposed A/B timing"
                )
            pairs = result.get("aa_pairs", [])
            expect_equal(
                len(pairs),
                contract["pilot"]["aa_pairs_per_process"],
                f"preflight.context{context}.seq{sequence}.pair_count",
                failures,
            )
            recoveries: list[float] = []
            aggregate = str(result.get("aggregate_ecc_baseline"))
            for pair_index, pair in enumerate(pairs):
                expect_equal(
                    pair.get("pair_index"),
                    pair_index,
                    "A/A pair index",
                    failures,
                )
                no_first = (pair_index % 2 == 0) == (sequence % 2 == 1)
                expected_order = (
                    ["no_codebook_a", "no_codebook_b"]
                    if no_first
                    else ["no_codebook_b", "no_codebook_a"]
                )
                expect_equal(pair.get("order"), expected_order, "A/A order", failures)
                durations = pair.get("durations_us", {})
                if set(durations) != {"no_codebook_a", "no_codebook_b"} or any(
                    not finite_positive(value) for value in durations.values()
                ):
                    failures.append(f"invalid A/A durations: {durations}")
                    continue
                expected_recovery = (
                    durations["no_codebook_a"] - durations["no_codebook_b"]
                ) / contract["geometry"]["selected_layers"]
                observed_recovery = pair.get("aa_recovery_us_per_selected_layer")
                if not isinstance(observed_recovery, (int, float)) or not math.isfinite(
                    observed_recovery
                ):
                    failures.append("A/A recovery is not finite numeric")
                    continue
                if not math.isclose(
                    observed_recovery, expected_recovery, rel_tol=0.0, abs_tol=1e-12
                ):
                    failures.append("A/A recovery arithmetic mismatch")
                recoveries.append(observed_recovery)
                reasons = telemetry_reasons(pair.get("telemetry_before", {}), contract)
                reasons += telemetry_reasons(pair.get("telemetry_after", {}), contract)
                expect_equal(
                    pair.get("telemetry_reasons"),
                    reasons,
                    "A/A telemetry reasons",
                    failures,
                )
                expect_equal(
                    pair.get("telemetry_valid"),
                    not reasons,
                    "A/A telemetry validity",
                    failures,
                )
                contamination += bool(reasons)
                for sample_name in ("telemetry_before", "telemetry_after"):
                    validate_gpu_binding_and_health(
                        pair.get(sample_name, {}),
                        contract,
                        aggregate,
                        f"A/A.{sample_name}",
                        failures,
                    )
            process_pairs.append((context, recoveries))
            results.append(result)
            source_digests.add(result.get("source_manifest_digest", ""))
            cache_digests.add(result.get("cache_artifacts_after", {}).get("digest", ""))
            codebook_digests.add(result.get("codebook_sha256", ""))
            installed_mla_digests.add(result.get("installed_mla_sha256", ""))
            aggregate_ecc_baselines.add(str(result.get("aggregate_ecc_baseline")))
            page_table_digests.setdefault(sequence, set()).add(
                result.get("page_table_sha256", "")
            )
            query_digest_sets.add(canonical_json_digest(result.get("query_sha256", {})))
            if sequence == 2:
                wall_time = result.get("wall_time_seconds")
                if finite_positive(wall_time):
                    second_wall_times.append(float(wall_time))
                else:
                    failures.append(
                        f"preflight context{context} seq{sequence} wall time is invalid"
                    )
    if len(source_digests) != 1:
        failures.append(f"qualification source digests differ: {source_digests}")
    if len(cache_digests) != 1:
        failures.append(
            f"qualification compiled-artifact digests differ: {cache_digests}"
        )
    if len(codebook_digests) != 1:
        failures.append(f"qualification codebook digests differ: {codebook_digests}")
    if len(installed_mla_digests) != 1:
        failures.append(
            f"qualification installed MLA digests differ: {installed_mla_digests}"
        )
    if len(aggregate_ecc_baselines) != 1:
        failures.append(
            f"qualification aggregate ECC baselines differ: {aggregate_ecc_baselines}"
        )
    if any(len(values) != 1 for values in page_table_digests.values()) or len(
        {next(iter(values)) for values in page_table_digests.values() if values}
    ) != len(page_table_digests):
        failures.append(
            f"qualification page-table permutation digests are invalid: {page_table_digests}"
        )
    if len(query_digest_sets) != 1:
        failures.append(f"qualification query digests differ: {query_digest_sets}")
    for label, values in (
        ("source manifest", source_digests),
        ("compiled artifact", cache_digests),
        ("codebook", codebook_digests),
        ("installed MLA", installed_mla_digests),
    ):
        if len(values) == 1 and not valid_sha256(next(iter(values))):
            failures.append(f"qualification {label} digest is invalid: {values}")
    if len(aggregate_ecc_baselines) == 1:
        aggregate = next(iter(aggregate_ecc_baselines))
        if not aggregate.isdigit():
            failures.append(
                f"qualification aggregate ECC baseline is invalid: {aggregate}"
            )
    if len(source_digests) == 1:
        expect_equal(
            qualification_gates.get("source_manifest_digest"),
            next(iter(source_digests)),
            "qualification_gates.source_manifest_digest",
            failures,
        )
    if len(cache_digests) == 1:
        expect_equal(
            qualification_gates.get("cache_artifact_digest"),
            next(iter(cache_digests)),
            "qualification_gates.cache_artifact_digest",
            failures,
        )
    if len(codebook_digests) == 1:
        expect_equal(
            qualification_gates.get("codebook_sha256"),
            next(iter(codebook_digests)),
            "qualification_gates.codebook_sha256",
            failures,
        )
    if len(installed_mla_digests) == 1:
        expect_equal(
            qualification_gates.get("installed_mla_sha256"),
            next(iter(installed_mla_digests)),
            "qualification_gates.installed_mla_sha256",
            failures,
        )
    if len(aggregate_ecc_baselines) == 1:
        expect_equal(
            str(qualification_gates.get("aggregate_ecc_baseline")),
            next(iter(aggregate_ecc_baselines)),
            "qualification_gates.aggregate_ecc_baseline",
            failures,
        )
    sigma = (
        pilot_sigma(
            process_pairs,
            pairs_per_process=contract["pilot"]["aa_pairs_per_process"],
        )
        if not failures
        else None
    )
    process_count = select_process_count(sigma, contract) if sigma is not None else None
    projected = (
        projected_no_decision_probability(contamination, process_count=process_count)
        if process_count is not None
        else None
    )
    if sigma is not None and process_count is None:
        failures.append(f"pilot sigma {sigma} exceeds the frozen maximum")
    if (
        projected is not None
        and projected
        > contract["pilot"]["max_projected_campaign_no_decision_probability"]
    ):
        failures.append(
            f"projected telemetry NO_DECISION probability {projected} exceeds 0.05"
        )
    timeout_key = f"decision_n{process_count}_experiment" if process_count else None
    timeout_seconds = (
        contract["timeouts_seconds"].get(timeout_key, 0) if timeout_key else 0
    )
    projected_runtime = (
        1.2
        * 2
        * process_count
        * (max(second_wall_times) + contract["process_launch_overhead_seconds"])
        if process_count is not None and second_wall_times
        else None
    )
    if projected_runtime is not None and projected_runtime > timeout_seconds:
        failures.append(
            f"projected decision runtime {projected_runtime} exceeds {timeout_seconds}"
        )
    raw_inputs_digest = canonical_json_digest(results)
    status = "READY" if not failures else "NO_DECISION"
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "experiment": contract["experiment"],
        "contract_digest": contract_digest,
        "pilot_raw_inputs_digest": raw_inputs_digest,
        "qualification_gates_digest": qualification_gates.get(
            "qualification_gates_digest"
        ),
        "sigma_pilot_us_per_layer": sigma,
        "contaminated_pilot_pairs": contamination,
        "projected_campaign_no_decision_probability": projected,
        "processes_per_context": process_count,
        "decision_timeout_seconds": timeout_seconds,
        "projected_decision_runtime_seconds": projected_runtime,
        "cache_artifact_digest": next(iter(cache_digests), None),
        "source_manifest_digest": next(iter(source_digests), None),
        "installed_mla_sha256": next(iter(installed_mla_digests), None),
        "aggregate_ecc_baseline": next(iter(aggregate_ecc_baselines), None),
        "codebook_sha256": next(iter(codebook_digests), None),
        "failures": failures,
    }
    value["decision_contract_digest"] = canonical_json_digest(value)
    return value


def decision_results(
    root: Path, contract: dict[str, Any], decision_contract: dict[str, Any]
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for context_text in contract["contexts"]:
        context = int(context_text)
        for sequence in range(1, decision_contract["processes_per_context"] + 1):
            values.append(load_json(result_path(root, "decision", context, sequence)))
    return values


def validity_stage(
    root: Path, contract: dict[str, Any], decision_contract: dict[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    sentinel_drift_abstentions = 0
    contract_digest = canonical_json_digest(contract)
    validate_decision_contract(decision_contract, contract, failures)
    results = (
        decision_results(root, contract, decision_contract) if not failures else []
    )
    expected_positions = (
        [
            (int(context_text), sequence)
            for context_text in contract["contexts"]
            for sequence in range(1, decision_contract["processes_per_context"] + 1)
        ]
        if not failures
        else []
    )
    validate_result_inventory(
        root,
        [
            result_path(root, "decision", context, sequence)
            for context, sequence in expected_positions
        ],
        "decision",
        failures,
    )
    page_table_digests: dict[int, set[str]] = {}
    query_digest_sets: set[str] = set()
    for result, (context, sequence) in zip(results, expected_positions, strict=True):
        validate_common_result(
            result,
            contract=contract,
            contract_digest=contract_digest,
            context=context,
            sequence=sequence,
            mode="decision",
            expected_cache_digest=decision_contract["cache_artifact_digest"],
            expected_source_digest=decision_contract["source_manifest_digest"],
            expected_installed_mla_digest=decision_contract["installed_mla_sha256"],
            expected_aggregate_ecc=decision_contract["aggregate_ecc_baseline"],
            expected_codebook_digest=decision_contract["codebook_sha256"],
            failures=failures,
        )
        page_table_digests.setdefault(sequence, set()).add(
            result.get("page_table_sha256", "")
        )
        query_digest_sets.add(canonical_json_digest(result.get("query_sha256", {})))
        expected_capture = (
            ["no_codebook", "codebook"] if sequence % 2 else ["codebook", "no_codebook"]
        )
        expect_equal(
            result.get("capture_order"), expected_capture, "capture_order", failures
        )
        validate_allocation_pair(
            result.get("graph_replay_allocation_before"),
            result.get("graph_replay_allocation_after"),
            "decision graph allocation",
            failures,
        )
        checksums = result.get("selected_output_checksums", {})
        expect_equal(
            checksums.get("no_codebook"),
            checksums.get("codebook"),
            "selected output checksums",
            failures,
        )
        aggregate = str(result.get("aggregate_ecc_baseline"))
        pairs = result.get("pairs", [])
        expect_equal(
            len(pairs), contract["timing"]["pairs_per_process"], "pair count", failures
        )
        retained = {"AB": 0, "BA": 0}
        for pair_index, pair in enumerate(pairs):
            expect_equal(pair.get("pair_index"), pair_index, "pair index", failures)
            no_first = (pair_index % 2 == 0) == (sequence % 2 == 1)
            expected_order = (
                ["no_codebook", "codebook"] if no_first else ["codebook", "no_codebook"]
            )
            expect_equal(pair.get("order"), expected_order, "pair order", failures)
            durations = pair.get("durations_us", {})
            if set(durations) != {"no_codebook", "codebook"} or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in durations.values()
            ):
                failures.append(f"invalid pair durations: {durations}")
            reasons = telemetry_reasons(pair.get("telemetry_before", {}), contract)
            reasons += telemetry_reasons(pair.get("telemetry_after", {}), contract)
            expect_equal(
                pair.get("telemetry_reasons"),
                reasons,
                "pair telemetry reasons",
                failures,
            )
            expect_equal(
                pair.get("telemetry_valid"),
                not reasons,
                "pair telemetry validity",
                failures,
            )
            for sample_name in ("telemetry_before", "telemetry_after"):
                validate_gpu_binding_and_health(
                    pair.get(sample_name, {}),
                    contract,
                    aggregate,
                    sample_name,
                    failures,
                )
            if not reasons:
                retained["AB" if no_first else "BA"] += 1
        if retained["AB"] < contract["timing"]["minimum_retained_ab_pairs"]:
            failures.append(f"retained AB pairs too low: {retained}")
        if retained["BA"] < contract["timing"]["minimum_retained_ba_pairs"]:
            failures.append(f"retained BA pairs too low: {retained}")
        sentinels = (
            result.get("sentinel_before", {}),
            result.get("sentinel_after", {}),
        )
        sentinel_telemetry_valid: list[bool] = []
        for sentinel_value in sentinels:
            reasons = telemetry_reasons(
                sentinel_value.get("telemetry_before", {}), contract
            )
            reasons += telemetry_reasons(
                sentinel_value.get("telemetry_after", {}), contract
            )
            expect_equal(
                sentinel_value.get("telemetry_reasons"),
                reasons,
                "sentinel telemetry reasons",
                failures,
            )
            for sample_name in ("telemetry_before", "telemetry_after"):
                validate_gpu_binding_and_health(
                    sentinel_value.get(sample_name, {}),
                    contract,
                    aggregate,
                    f"sentinel.{sample_name}",
                    failures,
                )
            sentinel_telemetry_valid.append(not reasons)
        if all(sentinel_telemetry_valid):
            try:
                first = float(sentinels[0]["ring_us"])
                second = float(sentinels[1]["ring_us"])
                if not finite_positive(first) or not finite_positive(second):
                    raise ValueError("sentinel times must be finite and positive")
                mean = (first + second) / 2.0
                drift = abs(second - first) / mean
                if drift > contract["timing"]["sentinel_max_drift_fraction"]:
                    failures.append(f"sentinel drift {drift} exceeds contract")
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                failures.append(f"invalid sentinel duration: {error}")
        else:
            sentinel_drift_abstentions += 1
    if results and (
        any(len(values) != 1 for values in page_table_digests.values())
        or len({next(iter(values)) for values in page_table_digests.values() if values})
        != len(page_table_digests)
    ):
        failures.append(
            f"decision page-table permutation digests are invalid: {page_table_digests}"
        )
    if results and len(query_digest_sets) != 1:
        failures.append(f"decision query digests differ: {query_digest_sets}")
    raw_inputs_digest = canonical_json_digest(results)
    result = {
        "schema_version": 1,
        "status": "VALID" if not failures else "NO_DECISION",
        "experiment": contract["experiment"],
        "contract_digest": contract_digest,
        "decision_contract_digest": decision_contract.get("decision_contract_digest"),
        "raw_inputs_digest": raw_inputs_digest,
        "result_count": len(results),
        "sentinel_drift_abstentions": sentinel_drift_abstentions,
        "failures": failures,
        "note": "Automated frozen-predicate validity; raw labels/times exist, but no inter-arm timing difference is used.",
    }
    result["validity_result_digest"] = canonical_json_digest(result)
    return result


def performance_stage(
    root: Path,
    contract: dict[str, Any],
    decision_contract: dict[str, Any],
    validity: dict[str, Any],
) -> dict[str, Any]:
    if validity.get("status") != "VALID":
        raise ValueError("performance analysis refuses a non-VALID Stage-1 result")
    validity_copy = dict(validity)
    digest = validity_copy.pop("validity_result_digest", None)
    if digest != canonical_json_digest(validity_copy):
        raise ValueError("validity result digest mismatch")
    decision_copy = dict(decision_contract)
    decision_digest = decision_copy.pop("decision_contract_digest", None)
    if decision_digest != canonical_json_digest(decision_copy):
        raise ValueError("decision contract digest mismatch")
    if validity.get("decision_contract_digest") != decision_digest:
        raise ValueError("validity result is bound to a different decision contract")
    if decision_contract.get("contract_digest") != canonical_json_digest(contract):
        raise ValueError("decision contract is bound to a different parent contract")
    if validity.get("contract_digest") != canonical_json_digest(contract):
        raise ValueError("validity result is bound to a different parent contract")
    results = decision_results(root, contract, decision_contract)
    if validity.get("raw_inputs_digest") != canonical_json_digest(results):
        raise ValueError("raw decision inputs changed after Stage 1")
    selected_layers = contract["geometry"]["selected_layers"]
    analyses: list[dict[str, Any]] = []
    for context_text, context_contract in contract["contexts"].items():
        context = int(context_text)
        process_means: list[float] = []
        for result in [value for value in results if value["context"] == context]:
            process_means.append(
                balanced_process_recovery(result["pairs"], selected_layers)
            )
        lower = bootstrap_lower_bound(
            process_means,
            draws=contract["bootstrap"]["draws"],
            quantile=contract["bootstrap"]["one_sided_lower_quantile"],
            seed=contract["seeds"]["bootstrap"] + context,
        )
        threshold = context_contract["recovery_threshold_us_per_selected_layer"]
        analyses.append(
            {
                "context": context,
                "process_mean_recoveries_us_per_selected_layer": process_means,
                "mean_recovery_us_per_selected_layer": statistics.fmean(process_means),
                "process_recovery_stdev_us_per_selected_layer": statistics.stdev(
                    process_means
                ),
                "one_sided_95_lower_bound_us_per_selected_layer": lower,
                "required_lower_bound_us_per_selected_layer": threshold,
                "gate": "PASS" if lower >= threshold else "REJECT",
            }
        )
    status = "ADVANCE" if all(item["gate"] == "PASS" for item in analyses) else "REJECT"
    value = {
        "schema_version": 1,
        "status": status,
        "experiment": contract["experiment"],
        "contract_digest": canonical_json_digest(contract),
        "decision_contract_digest": decision_contract["decision_contract_digest"],
        "validity_result_digest": validity["validity_result_digest"],
        "bootstrap_draws": contract["bootstrap"]["draws"],
        "bootstrap_quantile": contract["bootstrap"]["one_sided_lower_quantile"],
        "analyses": analyses,
        "memory": contract["memory_bytes_per_rank"],
        "scope": "Reader-only screen; ADVANCE authorizes only a separately reviewed integrated implementation.",
    }
    value["performance_result_digest"] = canonical_json_digest(value)
    return value


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    contract = load_contract(args.contract.resolve())
    if args.stage == "pilot":
        if args.qualification_gates is None:
            raise ValueError("pilot stage requires --qualification-gates")
        qualification_gates = load_json(args.qualification_gates.resolve())
        value = pilot_stage(root, contract, qualification_gates)
    else:
        if args.decision_contract is None:
            raise ValueError("decision stages require --decision-contract")
        decision_contract = load_json(args.decision_contract.resolve())
        if args.stage == "validity":
            value = validity_stage(root, contract, decision_contract)
        else:
            if args.validity_result is None:
                raise ValueError("performance stage requires --validity-result")
            validity = load_json(args.validity_result.resolve())
            value = performance_stage(root, contract, decision_contract, validity)
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
