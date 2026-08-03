from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_r1_results import (  # noqa: E402
    require_accepted_identity,
    require_baseline,
    require_disassembly,
    require_zero_recovery,
)
from evidence_common import EvidenceError, sha256_bytes  # noqa: E402


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
        ptx.extend({"barrier_id": 6, "count": 128} for _ in range(6))
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
