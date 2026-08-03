#!/usr/bin/env python3
"""Lock the TS-C58-R1 unaligned TMEM-handoff source scope."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python/tokenspeed_mla/mla_decode_fp8.py"
)
HELPER = "named_barrier_sync_unaligned"


def dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


class UnalignedHandoffSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))

    def test_helper_is_exact_unaligned_nvvm_sync(self) -> None:
        helpers = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == HELPER
        ]
        self.assertEqual(len(helpers), 1)
        helper = helpers[0]
        self.assertEqual([dotted_name(item) for item in helper.decorator_list], ["cute.jit"])
        calls = [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and dotted_name(node.func) == "nvvm.barrier_cta_sync"
        ]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(len(call.args), 1)
        self.assertEqual(dotted_name(call.args[0].func), "barrier_id.ir_value")
        keywords = {item.arg: item.value for item in call.keywords}
        self.assertEqual(set(keywords), {"thread_count", "aligned"})
        self.assertEqual(dotted_name(keywords["thread_count"].func), "num_threads.ir_value")
        self.assertIsInstance(keywords["aligned"], ast.Constant)
        self.assertIs(keywords["aligned"].value, False)

    def test_only_three_tmem_handoff_calls_are_replaced(self) -> None:
        helper_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and dotted_name(node.func) == HELPER
        ]
        self.assertEqual(len(helper_calls), 3)
        for call in helper_calls:
            self.assertEqual(
                [dotted_name(arg) for arg in call.args],
                [
                    "self.tmem_ptr_sync_bar.barrier_id",
                    "self.tmem_ptr_sync_bar.num_threads",
                ],
            )

        allocator_waits = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and dotted_name(node.func) == "tmem.wait_for_alloc"
        ]
        self.assertEqual(allocator_waits, [])

        conversion_waits = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and dotted_name(node.func)
            == "self.tq4_conversion_sync_bar.arrive_and_wait"
        ]
        self.assertEqual(len(conversion_waits), 5)


if __name__ == "__main__":
    unittest.main()
