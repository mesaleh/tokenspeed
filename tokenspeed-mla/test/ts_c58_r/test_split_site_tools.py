from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_common import EvidenceError, validate_execution_spec  # noqa: E402
from analyze_split_site_results import select_decision  # noqa: E402


IDENTITY_HASH = "a" * 64


def split_spec(cell_id: str) -> dict:
    return {
        "schema_version": 1,
        "record_type": "ts-c58-r-execution-spec",
        "cell_id": cell_id,
        "source_identity_sha256": IDENTITY_HASH,
        "target_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "device_index": 0,
        "command": [
            "/usr/bin/python3",
            "/work/litmus.py",
            "--target-uuid",
            "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "--device-index",
            "0",
            "--output",
            "/evidence/result.json",
        ],
        "sanitizer_tool": "synccheck",
        "timeout_seconds": 120,
        "acceptable_outcomes": ["clean", "diagnosed_sync_error"],
        "required_report_regex": [r"Divergent thread\(s\) in block"],
        "result_path": "/evidence/result.json",
        "result_outcomes": ["clean"],
        "result_requirements": {"status": "pass"},
    }


class SplitSiteSpecTests(unittest.TestCase):
    def test_split_site_cells_seal_clean_or_sync_error_but_not_timeout(self):
        for cell_id in ("aligned-split-synccheck", "unaligned-split-synccheck"):
            candidate = split_spec(cell_id)
            validate_execution_spec(candidate, IDENTITY_HASH)
            candidate["acceptable_outcomes"] = [
                "clean", "diagnosed_sync_error", "diagnosed_timeout"
            ]
            with self.assertRaisesRegex(EvidenceError, "multi-outcome"):
                validate_execution_spec(candidate, IDENTITY_HASH)

    def test_unreviewed_multi_outcome_cell_is_rejected(self):
        candidate = split_spec("another-split-cell")
        with self.assertRaisesRegex(EvidenceError, "multi-outcome"):
            validate_execution_spec(candidate, IDENTITY_HASH)

    def test_split_site_decision_matrix_is_exhaustive(self):
        self.assertEqual(
            select_decision("diagnosed_sync_error", "clean"),
            ("select_verified_unaligned_handoff", True),
        )
        self.assertEqual(
            select_decision("diagnosed_sync_error", "diagnosed_sync_error"),
            ("select_single_pc_or_replacement_protocol", False),
        )
        self.assertEqual(
            select_decision("clean", "clean"),
            ("static_site_split_not_sufficient", False),
        )
        self.assertEqual(
            select_decision("clean", "diagnosed_sync_error"),
            ("counter_hypothesis_unaligned_only_diagnoses", False),
        )


if __name__ == "__main__":
    unittest.main()
