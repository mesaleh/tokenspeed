"""CPU-only checks for the frozen H43 experiment machinery."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest
from analyze_h43_codebook_ab import (
    balanced_process_recovery,
    validate_decision_contract,
    validate_gpu_binding_and_health,
    validate_gpu_sample,
    validate_qualification_gates,
)
from check_h43_no_kernel_launch import validate_ncu_no_kernel_launch
from compare_h43_ncu import load_ncu
from h43_codebook_ab_common import (
    canonical_json_digest,
    compiled_artifact_manifest,
    load_contract,
    pilot_sigma,
    projected_no_decision_probability,
    select_process_count,
)
from run_h43_maintenance import gpu_health_mismatches

ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "h43_codebook_ab_contract.json"


def contract():
    return load_contract(CONTRACT_PATH)


def test_contract_thresholds_and_memory_are_exact():
    value = contract()
    memory = value["memory_bytes_per_rank"]
    assert value["contexts"]["10219"][
        "recovery_threshold_us_per_selected_layer"
    ] == pytest.approx(-13.366256, abs=1e-12)
    assert value["contexts"]["37932"][
        "recovery_threshold_us_per_selected_layer"
    ] == pytest.approx(6.522896, abs=1e-12)
    assert (
        memory["codebook_cache"] - memory["no_codebook_cache"]
        == memory["codebook_allocation"]
    )
    assert (
        memory["dense_control_cache"] - memory["codebook_cache"]
        == memory["persistent_saving"]
    )
    assert (
        memory["persistent_saving"] - memory["writer_workspace"]
        == memory["projected_net_saving"]
    )


def test_contract_digest_is_order_independent():
    value = contract()
    reversed_value = dict(reversed(list(value.items())))
    assert canonical_json_digest(value) == canonical_json_digest(reversed_value)


def test_internal_pilot_can_only_raise_or_stop_process_count():
    value = contract()
    pairs = [
        (10219, [0.2, -0.2] * 10),
        (10219, [0.1, -0.1] * 10),
        (37932, [0.3, -0.3] * 10),
        (37932, [0.2, -0.2] * 10),
    ]
    sigma = pilot_sigma(pairs, pairs_per_process=20)
    assert sigma < 1.0
    assert select_process_count(sigma, value) == 10
    assert select_process_count(1.1, value) == 14
    assert select_process_count(1.3, value) is None


def test_pilot_sigma_is_on_process_mean_scale():
    pairs = [
        (10219, [4.0, -4.0] * 10),
        (10219, [4.0, -4.0] * 10),
        (37932, [4.0, -4.0] * 10),
        (37932, [4.0, -4.0] * 10),
    ]
    sigma = pilot_sigma(pairs, pairs_per_process=20)
    assert 0.8 < sigma < 1.0


def test_process_recovery_balances_retained_ab_and_ba_orders():
    pairs = []
    for _ in range(10):
        pairs.append(
            {
                "telemetry_valid": True,
                "order": ["no_codebook", "codebook"],
                "durations_us": {"no_codebook": 150.0, "codebook": 10.0},
            }
        )
    for _ in range(8):
        pairs.append(
            {
                "telemetry_valid": True,
                "order": ["codebook", "no_codebook"],
                "durations_us": {"no_codebook": 10.0, "codebook": 10.0},
            }
        )
    assert balanced_process_recovery(pairs, selected_layers=14) == pytest.approx(5.0)


def test_ncu_parser_models_two_kernel_launches_per_ring_call(tmp_path):
    value = contract()
    path = tmp_path / "ncu.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Value"])
        launch_id = 0
        for _ in range(value["ncu"]["expected_ring_calls"]):
            for kernel_name in value["ncu"]["launch_kernel_order"]:
                for metric in value["ncu"]["required_metrics"]:
                    writer.writerow([launch_id, f"void {kernel_name}<x>", metric, "1"])
                launch_id += 1
    selected = load_ncu(path, value)
    assert [launch["id"] for launch in selected["split_kv_kernel"]] == list(
        range(48, 76, 2)
    )
    assert [launch["id"] for launch in selected["reduction_kernel"]] == list(
        range(49, 77, 2)
    )


def test_no_kernel_ncu_proof_requires_positive_profiler_evidence():
    good = """==PROF== Connected to process 123
==WARNING== No kernels were profiled.
==PROF== Disconnected from process 123
"""
    assert validate_ncu_no_kernel_launch(good)["kernel_launch_rows"] == 0
    with pytest.raises(RuntimeError, match="unrelated profiler warning"):
        validate_ncu_no_kernel_launch(
            good.replace(
                "==WARNING== No kernels were profiled.",
                "==WARNING== No kernels were profiled.\n"
                "==WARNING== GPU performance-counter permissions deny profiling",
            )
        )
    with pytest.raises(RuntimeError, match="lacks"):
        validate_ncu_no_kernel_launch("==WARNING== No kernels were profiled.\n")


def test_restore_scripts_use_verified_predicates_and_fresh_marker_deadline():
    rank0 = (ROOT / "restore_h43_rank0.sh").read_text(encoding="utf-8")
    rank1 = (ROOT / "restore_h43_rank1.sh").read_text(encoding="utf-8")
    assert "-eo pid,args" in rank0 and "-eo args" not in rank0
    assert "-eo pid,args" in rank1 and "-eo args" not in rank1
    assert "--query-compute-apps=pid" in rank0
    assert "rank0_already_running" in rank0
    assert "attempting exact rank-0 recovery" in rank0
    assert "GPU did not become idle before rank-0 restore" not in rank0
    assert rank1.count("deadline=$((SECONDS + H43_RANK1_READY_TIMEOUT))") == 1
    assert rank1.count("deadline=$((SECONDS + H43_MARKER_READY_TIMEOUT))") == 1
    assert '--directory "${marker_root}/${H43_NONCE}"' in rank1


def test_prebuild_derives_production_variable_sequence_dispatch_flags():
    source = (ROOT / "prebuild_h43_codebook_cache.py").read_text(encoding="utf-8")
    assert "inspect.signature(tokenspeed_mla_decode)" in source
    assert "inspect.signature(tokenspeed_mla_decode_tq4)" in source
    assert '"is_persistent": not is_var_seq' in source
    assert '"is_var_seq": is_var_seq' in source


def test_preparation_has_no_privileged_predictable_tmp_staging():
    source = (ROOT / "prepare_h43_remote.sh").read_text(encoding="utf-8")
    assert '"${remote_root}/incoming"' in source
    assert "/tmp/h43-${campaign}" not in source
    assert "mesaleh@${rank0_host}:/tmp/" not in source
    assert 'base_tag="h43-${campaign}-base:accepted"' in source
    assert source.count('docker image inspect "${base_tag}" --format') == 2
    assert '--build-arg "BASE_IMAGE=${base_tag}"' in source


def test_decision_window_has_remote_atomic_single_use_marker():
    source = (ROOT / "run_h43_maintenance.py").read_text(encoding="utf-8")
    assert "DECISION_WINDOW_CONSUMED.json" in source
    assert "os.O_EXCL" in source
    assert source.index("self.consume_decision_window()") < source.index("self.arm()")


def test_remote_script_quotes_space_containing_arguments_and_timers_are_bounded():
    source = (ROOT / "run_h43_maintenance.py").read_text(encoding="utf-8")
    terminal = (ROOT / "terminal_h43_maintenance.sh").read_text(encoding="utf-8")
    assert (
        'remote_command = shlex.join(["sudo", "-n", "bash", "-s", "--", *arguments])'
        in source
    )
    assert "h43-terminal-{self.campaign}" in source
    assert "h43-restore-${campaign}.timer" in terminal
    assert "automatic actions stopped" in terminal


def test_all_gpu_health_predicate_matches_frozen_baseline():
    value = contract()
    machine = value["machine"]
    sample = {
        "name": machine["gpu_name"],
        "pstate": machine["pstate"],
        "clocks.sm": str(machine["sm_clock_mhz"]),
        "clocks.max.sm": str(machine["sm_clock_mhz"]),
        "clocks_event_reasons.hw_slowdown": value["telemetry"]["inactive_event_value"],
        "clocks_event_reasons.sw_thermal_slowdown": value["telemetry"][
            "inactive_event_value"
        ],
        "ecc.errors.uncorrected.volatile.total": "0",
        "ecc.errors.uncorrected.aggregate.total": "0",
        "gpu_recovery_action": machine["recovery_action"],
        "fabric.state": machine["fabric_state"],
        "fabric.status": machine["fabric_status"],
        "power.draw.instant": "500",
        "power.limit": str(machine["power_limit_w"]),
    }
    assert not gpu_health_mismatches(sample, value, "0")
    sample["pstate"] = "P8"
    assert "pstate" in gpu_health_mismatches(sample, value, "0")


def test_snapshot_captures_gpu_baseline_before_prestop_comparison():
    source = (ROOT / "run_h43_maintenance.py").read_text(encoding="utf-8")
    method = source[
        source.index("    def snapshot_and_verify") : source.index("    def arm")
    ]
    assert method.index("self.pre_service_telemetry[host] = rows") < method.index(
        "baseline_by_uuid ="
    )


def test_projected_no_decision_probability_is_monotonic():
    clean = projected_no_decision_probability(0, process_count=10)
    one = projected_no_decision_probability(1, process_count=10)
    two = projected_no_decision_probability(2, process_count=10)
    expanded = projected_no_decision_probability(1, process_count=14)
    assert 0.0 < clean < one < two < 1.0
    assert expanded > one
    assert one < 0.05 < two


def test_compiled_artifact_manifest_ignores_only_frozen_ephemeral_names(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kernel.cubin").write_bytes(b"cubin-a")
    (cache / ".lock").write_text("ephemeral", encoding="utf-8")
    first = compiled_artifact_manifest(cache, contract()["cache"])
    (cache / ".lock").write_text("changed", encoding="utf-8")
    second = compiled_artifact_manifest(cache, contract()["cache"])
    assert first["digest"] == second["digest"]
    (cache / "kernel.cubin").write_bytes(b"cubin-b")
    third = compiled_artifact_manifest(cache, contract()["cache"])
    assert third["digest"] != second["digest"]


def test_compiled_artifact_manifest_rejects_symlinks(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    target = cache / "kernel.cubin"
    target.write_bytes(b"x")
    (cache / "alias").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        compiled_artifact_manifest(cache, contract()["cache"])


def test_analyzer_import_is_cpu_only():
    path = ROOT / "analyze_h43_codebook_ab.py"
    spec = importlib.util.spec_from_file_location("h43_analyzer_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.pilot_stage)


def test_contract_is_strict_json():
    raw = CONTRACT_PATH.read_text(encoding="utf-8")
    value = json.loads(
        raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token))
    )
    assert math.isfinite(value["machine"]["power_limit_w"])


def test_canonical_json_rejects_non_finite_numbers():
    with pytest.raises(ValueError):
        canonical_json_digest({"bad": math.inf})


def test_qualification_gate_record_is_digest_bound():
    value = contract()
    record = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": value["experiment"],
        "contract_digest": canonical_json_digest(value),
        "physical_host": value["machine"]["physical_host"],
        "device_uuid": value["machine"]["gpu_uuid"],
        "source_manifest_digest": "b" * 64,
        "installed_mla_sha256": "c" * 64,
        "cache_artifact_digest": "d" * 64,
        "codebook_sha256": "e" * 64,
        "aggregate_ecc_baseline": "0",
        "gates": [
            {"name": name, "status": "PASS", "evidence_sha256": "a" * 64}
            for name in value["qualification"]["required_gates"]
        ],
    }
    record["qualification_gates_digest"] = canonical_json_digest(record)
    failures: list[str] = []
    validate_qualification_gates(record, value, failures)
    assert not failures
    record["gates"][0]["status"] = "FAIL"
    failures = []
    validate_qualification_gates(record, value, failures)
    assert failures


def test_decision_contract_rejects_wrong_parent_and_non_finite_runtime():
    value = contract()
    decision = {
        "schema_version": 1,
        "status": "READY",
        "experiment": value["experiment"],
        "contract_digest": "b" * 64,
        "pilot_raw_inputs_digest": "c" * 64,
        "qualification_gates_digest": "d" * 64,
        "sigma_pilot_us_per_layer": 0.5,
        "contaminated_pilot_pairs": 0,
        "projected_campaign_no_decision_probability": 0.01,
        "processes_per_context": 10,
        "decision_timeout_seconds": 1200,
        "projected_decision_runtime_seconds": math.inf,
        "cache_artifact_digest": "e" * 64,
        "source_manifest_digest": "f" * 64,
        "installed_mla_sha256": "1" * 64,
        "aggregate_ecc_baseline": "0",
        "codebook_sha256": "2" * 64,
        "failures": [],
    }
    decision["decision_contract_digest"] = "3" * 64
    failures: list[str] = []
    validate_decision_contract(decision, value, failures)
    assert any("contract_digest" in failure for failure in failures)
    assert any("runtime" in failure for failure in failures)


def test_transient_pair_telemetry_is_excludable_but_hard_health_is_not():
    value = contract()
    machine = value["machine"]
    sample = {
        "index": str(machine["gpu_index"]),
        "uuid": machine["gpu_uuid"],
        "name": machine["gpu_name"],
        "pstate": "P8",
        "clocks.sm": "1000",
        "clocks.max.sm": str(machine["sm_clock_mhz"]),
        "clocks_event_reasons.hw_slowdown": "Active",
        "clocks_event_reasons.sw_thermal_slowdown": "Not Active",
        "temperature.gpu": "30",
        "temperature.gpu.tlimit": "60",
        "power.draw.instant": "0",
        "power.limit": str(machine["power_limit_w"]),
        "ecc.errors.uncorrected.volatile.total": "0",
        "ecc.errors.uncorrected.aggregate.total": "0",
        "gpu_recovery_action": machine["recovery_action"],
        "fabric.state": machine["fabric_state"],
        "fabric.status": machine["fabric_status"],
    }
    failures: list[str] = []
    validate_gpu_binding_and_health(sample, value, "0", "pair", failures)
    assert not failures
    validate_gpu_sample(sample, value, "0", "sentinel", failures)
    assert failures
    failures = []
    sample["ecc.errors.uncorrected.volatile.total"] = "1"
    validate_gpu_binding_and_health(sample, value, "0", "pair", failures)
    assert failures
