from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_disassembly import parse_ptx_barriers, parse_sass_barriers  # noqa: E402
from evidence_common import EvidenceError  # noqa: E402
from map_barrier_pc import semantic_role  # noqa: E402


class MappingToolTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
