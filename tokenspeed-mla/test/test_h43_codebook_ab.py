"""CPU-only checks for the frozen H43 experiment machinery."""

from __future__ import annotations

import ast
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
from check_h43_racecheck import load_racecheck_log, load_target_log
from check_h43_racecheck import main as check_racecheck_main
from compare_h43_ncu import load_ncu
from h43_aot_loader import (
    AOT_MANIFEST_NAME,
    adapt_h43_aot_callable,
    dispatch_key,
    load_aot_manifest,
)
from h43_aot_loader import sha256_file as aot_sha256_file
from h43_aot_loader import (
    write_aot_manifest,
)
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
        writer.writerow(["ID", "Kernel Name", *value["ncu"]["required_metrics"]])
        writer.writerow(["", "", *["unit" for _ in value["ncu"]["required_metrics"]]])
        launch_id = 0
        for _ in range(value["ncu"]["expected_ring_calls"]):
            for kernel_name in value["ncu"]["launch_kernel_order"]:
                writer.writerow(
                    [
                        launch_id,
                        f"void {kernel_name}<x>",
                        *["1" for _ in value["ncu"]["required_metrics"]],
                    ]
                )
                launch_id += 1
    selected = load_ncu(path, value)
    assert [launch["id"] for launch in selected["split_kv_kernel"]] == list(
        range(48, 76, 2)
    )
    assert [launch["id"] for launch in selected["reduction_kernel"]] == list(
        range(49, 77, 2)
    )


def test_ncu_push_pop_filter_and_empty_capture_fail_closed(tmp_path):
    value = contract()
    probe = (ROOT / "probe_h43_codebook_graph.py").read_text(encoding="utf-8")
    assert value["ncu"]["nvtx_range"] == "H43_CODEBOOK_Q5_RING/"
    assert 'torch.cuda.nvtx.range_push("H43_CODEBOOK_Q5_RING")' in probe
    assert "torch.cuda.nvtx.range_pop()" in probe
    path = tmp_path / "no-kernels.csv"
    path.write_text(
        "==PROF== Connected to process 1\n"
        "==WARNING== No kernels were profiled.\n"
        "==PROF== Disconnected from process 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no raw metric header"):
        load_ncu(path, value)


def test_ncu_uses_installed_raw_metric_contract_without_section_derived_metrics():
    value = contract()
    metrics = value["ncu"]["required_metrics"]
    assert "sm__maximum_warps_per_active_cycle_pct" in metrics
    assert not any(metric.startswith("derived__") for metric in metrics)
    assert value["timeouts_seconds"]["ncu_each"] == 90
    runner = (ROOT / "run_h43_maintenance.py").read_text(encoding="utf-8")
    method = runner[
        runner.index("    def run_ncu_resource_gate") : runner.index(
            "    def run_sanitizers"
        )
    ]
    assert '"--page",\n            "raw"' in method
    assert '"--section"' not in method


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


def test_racecheck_gate_requires_an_exact_zero_hazard_summary(tmp_path):
    value = contract()
    path = tmp_path / "racecheck.log"
    path.write_text(
        "========= COMPUTE-SANITIZER\n"
        "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n",
        encoding="utf-8",
    )
    observed = load_racecheck_log(path, value)
    assert observed["hazards"] == observed["errors"] == observed["warnings"] == 0

    path.write_text(
        "========= COMPUTE-SANITIZER\n"
        "========= Warning: Maximum number of hazards reached.\n"
        "========= RACECHECK SUMMARY: 1 hazards displayed (0 errors, 1 warnings)\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="racecheck summary"):
        load_racecheck_log(path, value)

    path.write_text(
        "========= COMPUTE-SANITIZER\n"
        "unexpected informational line\n"
        "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected racecheck diagnostic"):
        load_racecheck_log(path, value)


def test_racecheck_target_result_is_digest_and_contract_bound(tmp_path):
    value = contract()
    target = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": value["experiment"],
        "contract_digest": canonical_json_digest(value),
        "context": 37932,
        "sequence": 1,
        "query_length": value["racecheck"]["target_query_length"],
        "target_tq_calls": value["racecheck"]["target_tq_calls"],
        "codebook_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "output_finite": True,
    }
    target["result_digest"] = canonical_json_digest(target)
    path = tmp_path / "target.log"
    path.write_text(
        "NGC entrypoint banner\n" + json.dumps(target, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert load_target_log(path, value, 37932)["output_sha256"] == "b" * 64
    target["context"] = 10219
    path.write_text(json.dumps(target) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_target_log(path, value, 37932)


def test_racecheck_comparison_rejects_output_drift_and_nonzero_exit(
    tmp_path, monkeypatch
):
    value = contract()
    summary = (
        "========= COMPUTE-SANITIZER\n"
        "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
    )
    reference_log = tmp_path / "reference-racecheck.log"
    candidate_log = tmp_path / "candidate-racecheck.log"
    reference_log.write_text(summary, encoding="utf-8")
    candidate_log.write_text(summary, encoding="utf-8")

    def write_target(path, output_sha256):
        target = {
            "schema_version": 1,
            "status": "PASS",
            "experiment": value["experiment"],
            "contract_digest": canonical_json_digest(value),
            "context": 37932,
            "sequence": 1,
            "query_length": value["racecheck"]["target_query_length"],
            "target_tq_calls": value["racecheck"]["target_tq_calls"],
            "codebook_sha256": "a" * 64,
            "output_sha256": output_sha256,
            "output_finite": True,
        }
        target["result_digest"] = canonical_json_digest(target)
        path.write_text(json.dumps(target) + "\n", encoding="utf-8")

    reference_target = tmp_path / "reference-target.log"
    candidate_target = tmp_path / "candidate-target.log"
    write_target(reference_target, "b" * 64)
    write_target(candidate_target, "c" * 64)
    arguments = [
        "check_h43_racecheck.py",
        "--contract",
        str(CONTRACT_PATH),
        "--context",
        "37932",
        "--reference-log",
        str(reference_log),
        "--candidate-log",
        str(candidate_log),
        "--reference-target",
        str(reference_target),
        "--candidate-target",
        str(candidate_target),
        "--reference-exit-code",
        "0",
        "--candidate-exit-code",
        "0",
    ]
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(ValueError, match="output_sha256 mismatch"):
        check_racecheck_main()

    arguments[-1] = "86"
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(ValueError, match="exit codes must both be zero"):
        check_racecheck_main()


def test_racecheck_probe_and_runner_are_tq_only_and_preserve_failures():
    value = contract()
    probe = (ROOT / "probe_h43_tq_racecheck.py").read_text(encoding="utf-8")
    runner = (ROOT / "run_h43_maintenance.py").read_text(encoding="utf-8")
    method = runner[
        runner.index("    def run_sanitizers") : runner.index(
            "    def cache_persistence"
        )
    ]
    assert value["racecheck"]["target_tq_calls"] == 1
    assert "experiment.tq_call(" in probe
    assert "use_codebook=True" in probe
    assert "experiment.correctness()" not in probe
    assert "experiment.dense_call(" not in probe
    assert method.count("probe_h43_tq_racecheck.py") == 1
    assert "sanitizer-racecheck-reference.log" in method
    assert "sanitizer-racecheck-candidate.log" in method
    assert method.count("check=False") >= 4
    assert method.index("write_remote_root_file(") < method.index(
        "if result.returncode:"
    )


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


def test_prebuild_initializes_exactly_one_visible_cuda_device_before_compile():
    source = (ROOT / "prebuild_h43_codebook_cache.py").read_text(encoding="utf-8")
    count = source.index("torch.cuda.device_count() != 1")
    initialize = source.index("torch.cuda.init()")
    compile_call = source.index("compiled = getter(**arguments)")
    assert count < initialize < compile_call
    assert "torch.cuda.set_device(0)" in source
    assert "torch.cuda.current_device() != 0" in source


def test_aot_dispatch_key_binds_defaults_and_keyword_order():
    def sample(alpha, beta=0.5, *, enabled=True):
        return alpha, beta, enabled

    first = dispatch_key(sample, 7, enabled=False)
    second = dispatch_key(sample, enabled=False, alpha=7, beta=0.5)
    assert first == second
    assert first != dispatch_key(sample, 7, enabled=True)


def test_aot_callable_restores_dense_optional_none_arguments():
    calls = []

    def exported(*args):
        calls.append(args)
        return "called"

    adapted = adapt_h43_aot_callable(exported)
    dense = tuple(range(15))
    tq = tuple(range(18))
    assert adapted(*dense) == "called"
    assert calls[-1] == (*dense, None, None, None)
    assert adapted(*tq) == "called"
    assert calls[-1] == tq
    with pytest.raises(TypeError, match="got 17"):
        adapted(*range(17))


def test_aot_adapter_arity_matches_the_installed_mla_call_sites():
    source = (ROOT.parent / "python" / "tokenspeed_mla" / "mla_decode.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    runtime_tuples = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "args"
            for target in node.targets
        )
        and isinstance(node.value, ast.Tuple)
    ]
    assert [len(value.elts) for value in runtime_tuples] == [15]
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compiled_kernel"
    ]
    assert sorted(len(call.args) for call in calls) == [1, 4]
    assert all(isinstance(call.args[0], ast.Starred) for call in calls)


def test_aot_manifest_seals_source_and_artifacts(tmp_path):
    root = tmp_path / "aot"
    objects = root / "objects"
    libraries = root / "libraries"
    objects.mkdir(parents=True)
    libraries.mkdir()
    object_path = objects / "h43_dense_q1.o"
    library_path = libraries / "h43_dense_q1.so"
    object_path.write_bytes(b"object")
    library_path.write_bytes(b"library")
    key = "a" * 64
    entry = {
        "dispatch_key": key,
        "label": "dense-q1",
        "function_name": "h43_dense_q1",
        "object": "objects/h43_dense_q1.o",
        "object_sha256": aot_sha256_file(object_path),
        "library": "libraries/h43_dense_q1.so",
        "library_sha256": aot_sha256_file(library_path),
    }
    source_digest = "b" * 64
    installed_digest = "c" * 64
    manifest_path = root / AOT_MANIFEST_NAME
    write_aot_manifest(
        manifest_path,
        experiment="H43_D1_CODEBOOK_READER_AB",
        source_manifest_digest=source_digest,
        installed_mla_sha256=installed_digest,
        entries={key: entry},
    )
    loaded = load_aot_manifest(
        manifest_path,
        expected_source_manifest_digest=source_digest,
        expected_installed_mla_sha256=installed_digest,
        expected_entries=1,
    )
    assert loaded["entries"][key] == entry
    library_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        load_aot_manifest(
            manifest_path,
            expected_source_manifest_digest=source_digest,
            expected_installed_mla_sha256=installed_digest,
            expected_entries=1,
        )


def test_aot_manifest_rejects_artifact_path_traversal(tmp_path):
    root = tmp_path / "aot"
    root.mkdir()
    escaped_object = tmp_path / "escaped.o"
    escaped_object.write_bytes(b"object")
    library_path = root / "library.so"
    library_path.write_bytes(b"library")
    key = "d" * 64
    write_aot_manifest(
        root / AOT_MANIFEST_NAME,
        experiment="H43_D1_CODEBOOK_READER_AB",
        source_manifest_digest="e" * 64,
        installed_mla_sha256="f" * 64,
        entries={
            key: {
                "dispatch_key": key,
                "label": "dense-q1",
                "function_name": "h43_dense_q1",
                "object": "../escaped.o",
                "object_sha256": aot_sha256_file(escaped_object),
                "library": "library.so",
                "library_sha256": aot_sha256_file(library_path),
            }
        },
    )
    with pytest.raises(ValueError, match="invalid H43 AOT artifact path"):
        load_aot_manifest(
            root / AOT_MANIFEST_NAME,
            expected_source_manifest_digest="e" * 64,
            expected_installed_mla_sha256="f" * 64,
            expected_entries=1,
        )


def test_h43_runtime_is_wired_to_exact_aot_manifest():
    prebuild = (ROOT / "prebuild_h43_codebook_cache.py").read_text(encoding="utf-8")
    benchmark = (ROOT / "microbench_h43_codebook_ab.py").read_text(encoding="utf-8")
    runner = (ROOT / "run_h43_maintenance.py").read_text(encoding="utf-8")
    assert "compiled.export_to_c" in prebuild
    assert "find_runtime_libraries(enable_tvm_ffi=True)" in prebuild
    assert "install_h43_aot_from_environment()" in benchmark
    assert runner.count("H43_AOT_MANIFEST=") == 3
    assert runner.count("H43_AOT_EXPECTED_ENTRIES=") == 3
    assert runner.count("['cache_root']}:ro") == 3
    preparation = (ROOT / "prepare_h43_remote.sh").read_text(encoding="utf-8")
    assert '"${cache_root}:${cache_root}:ro"' in preparation


def test_preparation_has_no_privileged_predictable_tmp_staging():
    source = (ROOT / "prepare_h43_remote.sh").read_text(encoding="utf-8")
    assert '"${remote_root}/incoming"' in source
    assert "/tmp/h43-${campaign}" not in source
    assert "mesaleh@${rank0_host}:/tmp/" not in source
    assert 'base_tag="h43-${campaign}-base:accepted"' in source
    assert source.count('docker image inspect "${base_tag}" --format') == 2
    assert '--build-arg "BASE_IMAGE=${base_tag}"' in source


def test_preparation_capture_commands_bypass_noisy_image_entrypoint():
    source = (ROOT / "prepare_h43_remote.sh").read_text(encoding="utf-8")
    assert source.count("--entrypoint python3") == 3
    assert source.count("--entrypoint ncu") == 1
    assert source.count("--entrypoint sha256sum") == 1
    assert '"${image_id}" ncu ' not in source
    assert '"${image_id}" python3 ' not in source
    assert "installed TokenSpeed MLA digest is not one SHA-256 value" in source
    assert "invalid H43 prebuild evidence" in source


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


def test_setup_creates_the_system_helper_directory_before_installing_scripts():
    source = (ROOT / "run_h43_maintenance.py").read_text(encoding="utf-8")
    setup = source[
        source.index("    def setup") : source.index("    def verify_preparation")
    ]
    assert setup.index('"/usr/local/libexec"') < setup.index(
        'test_dir / "restore_h43_rank0.sh"'
    )


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
