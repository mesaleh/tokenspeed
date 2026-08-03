from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_disassembly import (  # noqa: E402
    parse_ptx_barriers,
    parse_sass_barriers,
    parse_sass_function_barriers,
)
from evidence_common import EvidenceError, sha256_bytes  # noqa: E402
from map_barrier_pc import semantic_role  # noqa: E402


class MappingToolTests(unittest.TestCase):
    def test_thread_map_derives_complete_count_from_execution_seal(self):
        tool_root = Path(__file__).resolve().parent
        report = (
            "========= Barrier error detected. Divergent thread(s) in block.\n"
            "=========     at kernel_name+0x15520\n"
            "=========     by thread (128,0,0) in block (0,0,0)\n"
            "========= Barrier error detected. Divergent thread(s) in block.\n"
            "=========     at kernel_name+0x15520\n"
            "=========     by thread (128,0,0) in block (1,0,0)\n"
            "========= ERROR SUMMARY: 2 errors\n"
        )
        report_raw = report.encode("utf-8")
        seal = {
            "record_type": "ts-c58-r-execution-seal",
            "status": "pass",
            "cell_id": "accepted-target-synccheck-map",
            "sanitizer_tool": "synccheck",
            "actual_outcome": "diagnosed_sync_error",
            "report_sha256": sha256_bytes(report_raw),
            "tool_summary": {"error_summaries": [2]},
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            report_path = directory / "report.log"
            seal_path = directory / "seal.json"
            output_path = directory / "thread-map.json"
            report_path.write_bytes(report_raw)
            seal_path.write_text(json.dumps(seal), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(tool_root / "parse_synccheck_report.py"),
                    "--report", str(report_path),
                    "--execution-seal", str(seal_path),
                    "--require-complete",
                    "--output", str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["error_count"], 2)
            self.assertTrue(result["complete"])
            self.assertFalse(result["thread_x_contiguous"])
            self.assertEqual(result["execution_seal_sha256"], sha256_bytes(seal_path.read_bytes()))

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

    def test_sass_barriers_are_scoped_to_each_cuda_function(self):
        sass = """
        .global reduction_kernel
        /*ca60*/ BAR.SYNC.DEFER_BLOCKING 0x2, 0x40 ;
        .global split_kv_kernel
        /*ca60*/ BAR.SYNC.DEFER_BLOCKING 0x1, 0x120 ;
        /*cb00*/ BAR.SYNC.DEFER_BLOCKING 0x1, 0x120 ;
        """
        self.assertEqual(
            parse_sass_function_barriers(sass),
            [
                {
                    "function": "reduction_kernel",
                    "address_hex": "0xca60",
                    "instruction": "BAR.SYNC.DEFER_BLOCKING",
                    "barrier_id": 2,
                    "count": 64,
                },
                {
                    "function": "split_kv_kernel",
                    "address_hex": "0xca60",
                    "instruction": "BAR.SYNC.DEFER_BLOCKING",
                    "barrier_id": 1,
                    "count": 288,
                },
                {
                    "function": "split_kv_kernel",
                    "address_hex": "0xcb00",
                    "instruction": "BAR.SYNC.DEFER_BLOCKING",
                    "barrier_id": 1,
                    "count": 288,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
