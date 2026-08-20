# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import math
import unittest

import torch

from tokenspeed_mla import reduce_mla_mixed_workspace


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (10, 0)


def _effective_splits(length: int, declared: int) -> int:
    tiles = math.ceil(length / 128)
    if tiles == 0:
        return 0
    tiles_per_cta = math.ceil(tiles / declared)
    return math.ceil(tiles / tiles_per_cta)


@unittest.skipUnless(_is_sm100(), "mixed MLA reduction requires SM100")
class TestMixedMLAReduction(unittest.TestCase):
    def test_q5_ignores_poisoned_declared_gaps_and_graph_replays(self) -> None:
        torch.manual_seed(0xA1736)
        batch, query_len, heads, latent = 3, 5, 8, 512
        hot_declared, cold_declared = 8, 9
        rows = batch * query_len * heads
        hot_lengths = torch.tensor([6400, 0, 256], dtype=torch.int32, device="cuda")
        cold_lengths = torch.tensor([4352, 384, 0], dtype=torch.int32, device="cuda")
        hot_bytes = rows * hot_declared * (latent + 1) * 4
        cold_bytes = rows * cold_declared * (latent + 1) * 4
        workspace = torch.empty(hot_bytes + cold_bytes, dtype=torch.int8, device="cuda")
        workspace.view(torch.float32).fill_(float("nan"))

        hot_float = workspace[:hot_bytes].view(torch.float32)
        cold_float = workspace[hot_bytes:].view(torch.float32)
        hot_acc = hot_float[: rows * hot_declared * latent].view(
            rows, hot_declared, latent
        )
        hot_lse = hot_float[rows * hot_declared * latent :].view(
            rows, hot_declared
        )
        cold_acc = cold_float[: rows * cold_declared * latent].view(
            rows, cold_declared, latent
        )
        cold_lse = cold_float[rows * cold_declared * latent :].view(
            rows, cold_declared
        )

        for row in range(rows):
            request = row // (query_len * heads)
            hot_effective = _effective_splits(
                int(hot_lengths[request]), hot_declared
            )
            cold_effective = _effective_splits(
                int(cold_lengths[request]), cold_declared
            )
            if hot_effective:
                hot_acc[row, :hot_effective].normal_()
                hot_lse[row, :hot_effective].normal_()
            if cold_effective:
                cold_acc[row, :cold_effective].normal_()
                cold_lse[row, :cold_effective].normal_()

        reference = torch.empty((rows, latent), dtype=torch.float32, device="cuda")
        reference_lse = torch.empty((rows,), dtype=torch.float32, device="cuda")
        for row in range(rows):
            request = row // (query_len * heads)
            hot_effective = _effective_splits(
                int(hot_lengths[request]), hot_declared
            )
            cold_effective = _effective_splits(
                int(cold_lengths[request]), cold_declared
            )
            local_lse = torch.cat(
                (
                    hot_lse[row, :hot_effective],
                    cold_lse[row, :cold_effective],
                )
            )
            local_acc = torch.cat(
                (
                    hot_acc[row, :hot_effective],
                    cold_acc[row, :cold_effective],
                )
            )
            merged_lse = torch.logsumexp(local_lse * math.log(2.0), dim=0) / math.log(
                2.0
            )
            reference_lse[row] = merged_lse
            reference[row] = torch.sum(
                local_acc * torch.exp2(local_lse - merged_lse).unsqueeze(-1),
                dim=0,
            )

        output = torch.empty(
            (batch, query_len, heads, latent), dtype=torch.bfloat16, device="cuda"
        )
        output_lse = torch.empty(
            (batch, query_len, heads), dtype=torch.float32, device="cuda"
        )

        def launch() -> None:
            reduce_mla_mixed_workspace(
                workspace,
                hot_lengths,
                cold_lengths,
                hot_declared,
                cold_declared,
                output,
                output_lse,
            )

        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.float().view(rows, latent), reference, rtol=0, atol=0.02
        )
        torch.testing.assert_close(
            output_lse.flatten(), reference_lse, rtol=0, atol=2.0e-5
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        output.zero_()
        output_lse.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.float().view(rows, latent), reference, rtol=0, atol=0.02
        )
        torch.testing.assert_close(
            output_lse.flatten(), reference_lse, rtol=0, atol=2.0e-5
        )

    def test_rejects_unaligned_workspace(self) -> None:
        output = torch.empty((1, 1, 8, 512), dtype=torch.bfloat16, device="cuda")
        output_lse = torch.empty((1, 1, 8), dtype=torch.float32, device="cuda")
        seq = torch.ones((1,), dtype=torch.int32, device="cuda")
        storage = torch.empty(8 * 2 * 513 * 4 + 1, dtype=torch.int8, device="cuda")
        with self.assertRaisesRegex(ValueError, "32-byte aligned"):
            reduce_mla_mixed_workspace(
                storage[1:], seq, seq, 1, 1, output, output_lse
            )


if __name__ == "__main__":
    unittest.main()
