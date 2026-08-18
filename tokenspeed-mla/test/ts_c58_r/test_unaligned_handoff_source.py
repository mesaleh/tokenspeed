#!/usr/bin/env python3
"""Lock the TS-C58-R1 unaligned TMEM-handoff source scope."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "python/tokenspeed_mla/mla_decode_fp8.py"
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


def int32_ir_value_argument(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "ir_value":
        return None
    conversion = node.func.value
    if (
        not isinstance(conversion, ast.Call)
        or dotted_name(conversion.func) != "Int32"
        or len(conversion.args) != 1
        or conversion.keywords
        or not isinstance(conversion.args[0], ast.Name)
    ):
        return None
    return conversion.args[0].id


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
        self.assertEqual(
            [dotted_name(item) for item in helper.decorator_list], ["cute.jit"]
        )
        calls = [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and dotted_name(node.func) == "nvvm.barrier_cta_sync"
        ]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(len(call.args), 1)
        self.assertEqual(int32_ir_value_argument(call.args[0]), "barrier_id")
        keywords = {item.arg: item.value for item in call.keywords}
        self.assertEqual(set(keywords), {"thread_count", "aligned"})
        self.assertEqual(
            int32_ir_value_argument(keywords["thread_count"]), "num_threads"
        )
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

        # Later kernels removed the old TQ conversion barrier entirely.  The
        # repair's invariant is narrower: only the three TMEM pointer handoff
        # waits use the explicit unaligned helper.  The exact helper-call count
        # above keeps future barrier changes from being swept into this fix.


if __name__ == "__main__":
    unittest.main()
