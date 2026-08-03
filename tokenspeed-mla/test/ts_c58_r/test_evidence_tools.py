from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_common import EvidenceError, sha256_file, validate_execution_spec  # noqa: E402
from capture_gpu_health import parse_gpus  # noqa: E402
from capture_disassembly import parse_ptx_barriers, parse_sass_barriers  # noqa: E402
from map_barrier_pc import semantic_role  # noqa: E402
from seal_gpu_recovery import by_uuid, healthy  # noqa: E402
from seal_sanitizer_result import classify  # noqa: E402


IDENTITY_HASH = "a" * 64


def spec(**updates):
    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-execution-spec",
        "cell_id": "unaligned-wrong-count-synccheck",
        "source_identity_sha256": IDENTITY_HASH,
        "target_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "device_index": 0,
        "command": [
            "/usr/bin/python3", "/work/litmus.py",
            "--target-uuid", "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "--device-index", "0",
        ],
        "sanitizer_tool": "synccheck",
        "timeout_seconds": 30,
        "acceptable_outcomes": ["diagnosed_sync_error", "diagnosed_timeout"],
        "required_report_regex": ["Divergent thread"],
        "result_path": None,
        "result_outcomes": [],
        "result_requirements": {},
    }
    value.update(updates)
    return value


class EvidenceToolTests(unittest.TestCase):
    def test_named_barrier_parsers_preserve_exact_operands_and_locations(self):
        sass = """
        /*15520*/ BAR.SYNC.DEFER_BLOCKING 0x1, 0x120 ;
        /*a6d0*/ @P0 BAR.SYNC.DEFER_BLOCKING 0x6, 0x80 ;
        """
        self.assertEqual(
            parse_sass_barriers(sass),
            [
                {"address_hex": "0x15520", "instruction": "BAR.SYNC.DEFER_BLOCKING",
                 "barrier_id": 1, "count": 288},
                {"address_hex": "0xa6d0", "instruction": "BAR.SYNC.DEFER_BLOCKING",
                 "barrier_id": 6, "count": 128},
            ],
        )
        ptx = "\n\tbar.sync \t6, 128;\n\tbar.sync \t0x1, 0x120;\n"
        self.assertEqual(
            [(row["line"], row["barrier_id"], row["count"])
             for row in parse_ptx_barriers(ptx)],
            [(2, 6, 128), (3, 1, 288)],
        )

    def test_sass_operands_map_uniquely_to_source_contract(self):
        inventory = {
            "inferred_contract": {
                "tmem_barrier": {"id": 1, "count": 288, "intended_threads": [0, 287]},
                "tq4_conversion_barrier": {
                    "id": 6, "count": 128, "intended_threads": [384, 511]
                },
            }
        }
        self.assertEqual(
            semantic_role(inventory, 1, 288), ("tmem_pointer_handoff", [0, 287])
        )
        self.assertEqual(
            semantic_role(inventory, 6, 128), ("tq4_conversion_rendezvous", [384, 511])
        )
        with self.assertRaisesRegex(EvidenceError, "does not map uniquely"):
            semantic_role(inventory, 1, 128)

    def test_reviewed_count_cell_accepts_only_fixed_two_member_set(self):
        validate_execution_spec(spec(), IDENTITY_HASH)
        invalid = spec(cell_id="another-cell")
        with self.assertRaisesRegex(EvidenceError, "multi-outcome"):
            validate_execution_spec(invalid, IDENTITY_HASH)

    def test_result_artifact_outcomes_are_fail_closed(self):
        invalid = spec(
            result_path="/evidence/result.json",
            result_outcomes=["diagnosed_timeout"],
            result_requirements={"status": "pass"},
            command=[
                "/usr/bin/python3", "/work/litmus.py",
                "--target-uuid", "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "--device-index", "0", "--output", "/evidence/result.json",
            ],
        )
        with self.assertRaisesRegex(EvidenceError, "killed timeout"):
            validate_execution_spec(invalid, IDENTITY_HASH)
        invalid = spec(
            cell_id="aligned-full-synccheck",
            acceptable_outcomes=["clean"],
            required_report_regex=[],
        )
        with self.assertRaisesRegex(EvidenceError, "clean outcome"):
            validate_execution_spec(invalid, IDENTITY_HASH)
        invalid = spec(acceptable_outcomes=["clean", "diagnosed_timeout"])
        with self.assertRaisesRegex(EvidenceError, "multi-outcome"):
            validate_execution_spec(invalid, IDENTITY_HASH)

    def test_outcome_classifier_distinguishes_clean_sync_error_and_timeout(self):
        clean_spec = spec(
            cell_id="aligned-full-synccheck",
            acceptable_outcomes=["clean"],
            required_report_regex=[],
            command=[
                "/usr/bin/python3", "/work/litmus.py",
                "--target-uuid", "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "--device-index", "0", "--output", "/evidence/aligned-full.json",
            ],
            result_path="/evidence/aligned-full.json",
            result_outcomes=["clean"],
            result_requirements={"status": "pass"},
        )
        clean_record = {"timed_out": False, "killed_process_group": False}
        clean_report = "========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"
        self.assertEqual(classify(clean_spec, clean_record, clean_report, 0)[0], "clean")

        error_record = {"timed_out": False, "killed_process_group": False}
        error_report = (
            "========= COMPUTE-SANITIZER\n"
            "========= Divergent thread(s) in block\n"
            "========= ERROR SUMMARY: 160 errors\n"
        )
        self.assertEqual(classify(spec(), error_record, error_report, 99)[0],
                         "diagnosed_sync_error")

        timeout_record = {"timed_out": True, "killed_process_group": True}
        self.assertEqual(classify(spec(), timeout_record, "", 124)[0], "diagnosed_timeout")
        with self.assertRaisesRegex(EvidenceError, "unclassified"):
            classify(spec(), error_record, "", 1)


def gpu_row(index: int):
    return {
        "index": index,
        "uuid": f"GPU-aaaaaaaa-bbbb-cccc-dddd-{index:012d}",
        "name": "NVIDIA GB200",
        "compute_capability": "10.0",
        "driver_version": "580.95.05",
        "application_graphics_mhz": 1965,
        "application_memory_mhz": 4000,
        "ecc_corrected_volatile": 0,
        "ecc_uncorrected_volatile": 0,
        "ecc_corrected_aggregate": 1,
        "ecc_uncorrected_aggregate": 0,
        "retired_pages_single_bit": None,
        "retired_pages_double_bit": None,
        "pending_remapped_rows": False,
        "recovery_action": "None",
        "fabric_state": "state  Completed",
        "fabric_status": "status Success",
    }

class HealthToolTests(unittest.TestCase):
    def test_health_parser_types_gb200_unavailable_and_boolean_fields(self):
        row = (
            "{index}, GPU-aaaaaaaa-bbbb-cccc-dddd-{index:012d}, NVIDIA GB200, 10.0, "
            "570.158.01, 1200, 4000, 0, 0, 0, 0, [N/A], [N/A], No, None, "
            "state  Completed, status Success"
        )
        rows = parse_gpus("\n".join(row.format(index=index) for index in range(4)))
        self.assertIsNone(rows[0]["retired_pages_single_bit"])
        self.assertIsNone(rows[0]["retired_pages_double_bit"])
        self.assertIs(rows[0]["pending_remapped_rows"], False)

    def test_health_predicates_allow_monotonic_evidence_but_reject_degradation(self):
        rows = [gpu_row(index) for index in range(4)]
        self.assertEqual(len(by_uuid(rows)), 4)
        healthy(rows[0], "test")
        changed = deepcopy(rows[0])
        changed["ecc_corrected_aggregate"] += 1
        healthy(changed, "test")
        changed["ecc_uncorrected_volatile"] = 1
        with self.assertRaisesRegex(EvidenceError, "uncorrected ECC"):
            healthy(changed, "test")

    def test_recovery_seal_binds_execution_inside_snapshots(self):
        tool_root = Path(__file__).resolve().parent
        capture_hash = sha256_file(tool_root / "capture_gpu_health.py")
        rows = [gpu_row(index) for index in range(4)]
        command = {"return_code": 0, "stdout": "stable\n", "stderr": ""}
        before = {
            "schema_version": 1,
            "record_type": "ts-c58-r-gpu-health-snapshot",
            "phase_id": "aligned-full-synccheck",
            "position": "before",
            "captured_unix_ns": 100,
            "gpus": rows,
            "compute_clients": [],
            "nvlink_status": command,
            "topology": command,
            "xid_events": {"available": False, "matching_lines": []},
            "capture_sha256": capture_hash,
        }
        after = deepcopy(before)
        after["position"] = "after"
        after["captured_unix_ns"] = 400
        after["gpus"][0]["ecc_corrected_aggregate"] += 1
        execution = {
            "record_type": "ts-c58-r-execution-seal",
            "status": "pass",
            "cell_id": "aligned-full-synccheck",
            "target_uuid": rows[0]["uuid"],
            "started_unix_ns": 200,
            "finished_unix_ns": 300,
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths = {name: directory / f"{name}.json" for name in ("before", "after", "execution", "seal")}
            for name, value in (("before", before), ("after", after), ("execution", execution)):
                paths[name].write_text(json.dumps(value), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(tool_root / "seal_gpu_recovery.py"),
                    "--before", str(paths["before"]),
                    "--after", str(paths["after"]),
                    "--execution-seal", str(paths["execution"]),
                    "--output", str(paths["seal"]),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
            self.assertEqual(seal["status"], "pass")
            self.assertEqual(seal["monotonic_deltas"][rows[0]["uuid"]]["ecc_corrected_aggregate"], 1)
            self.assertIn('"status": "pass"', completed.stdout)


if __name__ == "__main__":
    unittest.main()
