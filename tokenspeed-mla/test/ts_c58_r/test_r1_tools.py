from __future__ import annotations

import copy
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_r1_results import (  # noqa: E402
    require_sanitizer_oracle,
    require_accepted_identity,
    require_baseline,
    require_disassembly,
    require_zero_recovery,
)
from evidence_common import (  # noqa: E402
    EvidenceError,
    compiler_sass_contract,
    compiler_semantic_contract,
    sha256_bytes,
    sha256_file,
)


def disassembly_fixture(arm: str) -> dict:
    ptx = [
        {
            "instruction": "barrier.sync",
            "aligned": False,
            "barrier_id": 1,
            "count": 288,
        }
        for _ in range(3)
    ]
    sass = [
        {"barrier_id": 1, "count": 288}
        for _ in range(3)
    ]
    if arm == "m128":
        ptx.extend(
            {
                "instruction": "bar.sync",
                "aligned": True,
                "barrier_id": 6,
                "count": 128,
            }
            for _ in range(6)
        )
        sass.extend({"barrier_id": 6, "count": 128} for _ in range(6))
    return {
        "record_type": "ts-c58-r-disassembly-manifest",
        "status": "pass",
        "arm": arm,
        "ptx_named_barriers": ptx,
        "sass_named_barriers": sass,
        "ptx_sha256": "a" * 64,
        "cubin_sha256": "b" * 64,
        "disassembly_sha256": "c" * 64,
    }


class R1ToolTests(unittest.TestCase):
    @staticmethod
    def _artifact_inventory(root: Path) -> list[dict]:
        return [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "suffix": path.suffix,
            }
            for path in sorted(root.iterdir())
        ]

    @staticmethod
    def _fake_nvdisasm(root: Path) -> Path:
        path = root / "nvdisasm"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            "if '--version' in sys.argv:\n"
            "    print('fake nvdisasm 1.0')\n"
            "else:\n"
            "    raw = pathlib.Path(sys.argv[-1]).read_bytes()\n"
            "    count = '0x100' if b'sass-drift' in raw else '0x120'\n"
            "    print('.global kernel')\n"
            "    print('/*0000*/ BAR.SYNC.DEFER_BLOCKING 0x1, ' + count + ';')\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def test_compiler_contract_tolerates_register_allocation_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            for directory, register, cubin in (
                (first, "%r1", b"first-cubin"),
                (second, "%r99", b"second-cubin"),
            ):
                (directory / "kernel.mlir").write_text("stable-ir\n", encoding="utf-8")
                (directory / "kernel.ptx").write_text(
                    ".version 8.7\n"
                    f"mov.u32 {register}, %tid.x;\n"
                    "barrier.sync 1, 288;\n"
                    "barrier.sync.aligned 6, 128;\n",
                    encoding="utf-8",
                )
                (directory / "kernel.cubin").write_bytes(cubin)
            first_artifacts = self._artifact_inventory(first)
            second_artifacts = self._artifact_inventory(second)
            self.assertNotEqual(first_artifacts, second_artifacts)
            self.assertEqual(
                compiler_semantic_contract(first, first_artifacts),
                compiler_semantic_contract(second, second_artifacts),
            )
            nvdisasm = self._fake_nvdisasm(root)
            self.assertEqual(
                compiler_sass_contract(first, first_artifacts, nvdisasm),
                compiler_sass_contract(second, second_artifacts, nvdisasm),
            )

    def test_compiler_contract_rejects_ir_or_barrier_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            barrier_drift = root / "barrier-drift"
            ir_drift = root / "ir-drift"
            for directory, ir, barrier in (
                (baseline, "stable-ir\n", "barrier.sync 1, 288;\n"),
                (barrier_drift, "stable-ir\n", "barrier.sync.aligned 1, 288;\n"),
                (ir_drift, "changed-ir\n", "barrier.sync 1, 288;\n"),
            ):
                directory.mkdir()
                (directory / "kernel.mlir").write_text(ir, encoding="utf-8")
                (directory / "kernel.ptx").write_text(barrier, encoding="utf-8")
                (directory / "kernel.cubin").write_bytes(b"cubin")
            baseline_contract = compiler_semantic_contract(
                baseline, self._artifact_inventory(baseline)
            )
            self.assertNotEqual(
                baseline_contract,
                compiler_semantic_contract(
                    barrier_drift, self._artifact_inventory(barrier_drift)
                ),
            )
            self.assertNotEqual(
                baseline_contract,
                compiler_semantic_contract(ir_drift, self._artifact_inventory(ir_drift)),
            )

    def test_compiler_sass_contract_rejects_named_barrier_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            drift = root / "drift"
            for directory, cubin in ((baseline, b"cubin"), (drift, b"sass-drift")):
                directory.mkdir()
                (directory / "kernel.mlir").write_text("stable-ir\n", encoding="utf-8")
                (directory / "kernel.ptx").write_text(
                    "barrier.sync 1, 288;\n", encoding="utf-8"
                )
                (directory / "kernel.cubin").write_bytes(cubin)
            nvdisasm = self._fake_nvdisasm(root)
            self.assertNotEqual(
                compiler_sass_contract(
                    baseline, self._artifact_inventory(baseline), nvdisasm
                ),
                compiler_sass_contract(drift, self._artifact_inventory(drift), nvdisasm),
            )

    def test_sanitizer_oracle_binds_exact_candidate_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "kernel.mlir").write_text("stable-ir\n", encoding="utf-8")
            (root / "kernel.ptx").write_text(
                "barrier.sync 1, 288;\n", encoding="utf-8"
            )
            (root / "kernel.cubin").write_bytes(b"cubin")
            artifacts = self._artifact_inventory(root)
            semantic_contract = compiler_semantic_contract(root, artifacts)
            candidate = {
                "cases": [{"case_id": "case"}],
                "compiler_semantic_contract": semantic_contract,
            }
            candidate_raw = b'{"candidate":true}\n'
            result = {
                "record_type": "ts-c58-r-decode-oracle",
                "status": "pass",
                "arm": "m128",
                "mode": "synccheck",
                "expected_sha256": sha256_bytes(candidate_raw),
                "hashes_match_unsanitized": True,
                "compiler_semantic_contract_matches_unsanitized": True,
                "cases": candidate["cases"],
                "compiler_semantic_contract": semantic_contract,
                "compiler_dump_dir": str(root),
                "compiler_artifacts": artifacts,
            }
            result_raw = b'{"result":true}\n'
            execution = {"result_sha256": sha256_bytes(result_raw)}
            require_sanitizer_oracle(
                result,
                result_raw,
                execution,
                candidate,
                candidate_raw,
                arm="m128",
                mode="synccheck",
            )
            wrong = copy.deepcopy(result)
            wrong["expected_sha256"] = "0" * 64
            with self.assertRaisesRegex(EvidenceError, "oracle binding differs"):
                require_sanitizer_oracle(
                    wrong,
                    result_raw,
                    execution,
                    candidate,
                    candidate_raw,
                    arm="m128",
                    mode="synccheck",
                )

    def test_candidate_cannot_substitute_for_pinned_accepted_baseline(self) -> None:
        candidate_identity = {
            "record_type": "ts-c58-r-source-identity",
            "schema_version": 1,
            "source_commit": "2f378dad7b0591b97dc7f3b682636068b0783c55",
        }
        candidate_identity_raw = b'{"source_commit":"candidate"}\n'
        with self.assertRaisesRegex(EvidenceError, "pinned baseline"):
            require_accepted_identity(
                candidate_identity,
                candidate_identity_raw,
                candidate_identity["source_commit"],
            )

        candidate_oracle = {
            "record_type": "ts-c58-r-decode-oracle",
            "status": "pass",
            "arm": "m128",
            "mode": "unsanitized",
            "source_commit": candidate_identity["source_commit"],
            "source_identity_sha256": "c" * 64,
        }
        candidate_raw = b'{"candidate":true}\n'
        candidate_seal = {
            "record_type": "ts-c58-r-execution-seal",
            "status": "pass",
            "cell_id": "accepted-target-unsanitized",
            "actual_outcome": "clean",
            "sanitizer_tool": None,
            "source_commit": candidate_identity["source_commit"],
            "source_identity_sha256": "c" * 64,
            "result_sha256": sha256_bytes(candidate_raw),
        }
        with self.assertRaisesRegex(EvidenceError, "baseline bytes differ"):
            require_baseline(
                candidate_oracle,
                candidate_raw,
                candidate_seal,
                b'{"candidate-seal":true}\n',
                "c" * 64,
                candidate_identity["source_commit"],
                "m128",
                "accepted-target-unsanitized",
            )

    def test_disassembly_requires_three_explicit_unaligned_handoff_sites(self) -> None:
        summary = require_disassembly(disassembly_fixture("m128"), "m128")
        self.assertEqual(summary["ptx_handoff_sites"], 3)
        self.assertEqual(summary["ptx_tq4_sites"], 6)

        aligned = disassembly_fixture("m128")
        aligned["ptx_named_barriers"][1]["aligned"] = True
        with self.assertRaisesRegex(EvidenceError, "not explicitly unaligned"):
            require_disassembly(aligned, "m128")

        missing = disassembly_fixture("dense")
        missing["ptx_named_barriers"].pop()
        with self.assertRaisesRegex(EvidenceError, "PTX handoff-site count"):
            require_disassembly(missing, "dense")

        wrong_count = disassembly_fixture("dense")
        wrong_count["ptx_named_barriers"][0]["count"] = 256
        with self.assertRaisesRegex(EvidenceError, "handoff operands drifted"):
            require_disassembly(wrong_count, "dense")

    def test_disassembly_preserves_m128_tq4_sites_and_dense_absence(self) -> None:
        require_disassembly(disassembly_fixture("dense"), "dense")
        missing_tq4 = disassembly_fixture("m128")
        missing_tq4["sass_named_barriers"].pop()
        with self.assertRaisesRegex(EvidenceError, "SASS TQ4-site count"):
            require_disassembly(missing_tq4, "m128")

        unaligned_tq4 = disassembly_fixture("m128")
        unaligned_tq4["ptx_named_barriers"][-1]["aligned"] = False
        with self.assertRaisesRegex(EvidenceError, "TQ4 barrier is not aligned"):
            require_disassembly(unaligned_tq4, "m128")

        dense_with_tq4 = disassembly_fixture("dense")
        dense_with_tq4["ptx_named_barriers"].append(
            {"barrier_id": 6, "count": 128}
        )
        with self.assertRaisesRegex(EvidenceError, "PTX TQ4-site count"):
            require_disassembly(dense_with_tq4, "dense")

    def test_recovery_accepts_only_zero_or_unavailable_deltas(self) -> None:
        execution_raw = b'{"status":"pass"}\n'
        gpu = {
            "ecc_corrected_aggregate": 0,
            "ecc_corrected_volatile": 0,
            "ecc_uncorrected_aggregate": 0,
            "retired_pages_double_bit": None,
            "retired_pages_single_bit": None,
        }
        recovery = {
            "record_type": "ts-c58-r-gpu-recovery-seal",
            "status": "pass",
            "phase_id": "accepted-target-synccheck",
            "execution_seal_sha256": sha256_bytes(execution_raw),
            "capture_sha256": "a" * 64,
            "sealer_sha256": "b" * 64,
            "gpu_uuids": [
                f"GPU-{index:08X}-0000-0000-0000-000000000000"
                for index in range(4)
            ],
            "monotonic_deltas": {
                f"GPU-{index:08X}-0000-0000-0000-000000000000": copy.deepcopy(gpu)
                for index in range(4)
            },
        }
        require_zero_recovery(
            recovery,
            execution_raw,
            "accepted-target-synccheck",
            target_uuid="GPU-00000000-0000-0000-0000-000000000000",
            capture_sha256="a" * 64,
            sealer_sha256="b" * 64,
        )
        changed = copy.deepcopy(recovery)
        next(iter(changed["monotonic_deltas"].values()))[
            "ecc_corrected_aggregate"
        ] = 1
        with self.assertRaisesRegex(EvidenceError, "counter increased"):
            require_zero_recovery(
                changed,
                execution_raw,
                "accepted-target-synccheck",
                target_uuid="GPU-00000000-0000-0000-0000-000000000000",
                capture_sha256="a" * 64,
                sealer_sha256="b" * 64,
            )

        wrong_tool = copy.deepcopy(recovery)
        wrong_tool["sealer_sha256"] = "d" * 64
        with self.assertRaisesRegex(EvidenceError, "tool or target identity"):
            require_zero_recovery(
                wrong_tool,
                execution_raw,
                "accepted-target-synccheck",
                target_uuid="GPU-00000000-0000-0000-0000-000000000000",
                capture_sha256="a" * 64,
                sealer_sha256="b" * 64,
            )


if __name__ == "__main__":
    unittest.main()
