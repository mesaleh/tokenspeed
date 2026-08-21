"""Initial algebra/compile probe for the physical R31 split-score mainloop."""

from __future__ import annotations

import argparse
import json
import math

import torch
from cutlass import Float32, Int32
from tokenspeed_mla import tokenspeed_mla_decode_tq_r31
from tokenspeed_mla.mla_decode_tq_r31 import _get_compiled_tq_r31_kernel


def _launch(
    *,
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    packed_latent: torch.Tensor,
    scale: torch.Tensor,
    high_rope: torch.Tensor,
    residual_rope: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    physical: bool,
    lookahead: bool = True,
    dual_tmem: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = torch.empty_like(query_latent, dtype=torch.bfloat16)
    fault_status = (
        torch.zeros((1,), device=query_latent.device, dtype=torch.int32)
        if dual_tmem
        else None
    )
    compiled = _get_compiled_tq_r31_kernel(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        reconstruction_scale=scale,
        high_rope=high_rope,
        residual_rope=residual_rope,
        block_tables=block_tables,
        output=output,
        lse=None,
        workspace=None,
        seq_lens=seq_lens,
        fold_sq_factor=1,
        causal_mask=False,
        enable_pdl=False,
        producer_only=False,
        physical_split_score=physical,
        physical_split_score_lookahead=lookahead,
        physical_split_score_dual_tmem=dual_tmem,
        fault_status=fault_status,
    )
    import tvm_ffi

    with torch.cuda.device(query_latent.device), tvm_ffi.use_torch_stream():
        compiled(
            query_latent,
            query_rope,
            packed_latent,
            high_rope,
            block_tables,
            output,
            None,
            None,
            Int32(1),
            seq_lens,
            seq_lens,
            None,
            Float32(0.125),
            Float32(1.0),
            scale,
            None,
            residual_rope,
            fault_status,
        )
    return output, fault_status


def run(
    sequence_length: int = 384, batch: int = 1, query_len: int = 1
) -> dict[str, object]:
    torch.manual_seed(20260824)
    device = torch.device("cuda:0")
    query_latent = (
        torch.randn((batch, query_len, 8, 512), device=device) * 0.25
    ).to(torch.float8_e4m3fn)
    query_rope = (
        torch.randn((batch, query_len, 8, 256), device=device) * 0.25
    ).to(
        torch.float8_e4m3fn
    )
    page_count = math.ceil(sequence_length / 32)
    total_pages = batch * page_count
    packed_latent = torch.randint(
        0, 256, (total_pages, 32, 256), device=device, dtype=torch.uint8
    )
    scale_pattern = torch.tensor(
        [0.5, 1.0, 2.0, 4.0], device=device, dtype=torch.bfloat16
    )
    row_scales = scale_pattern.repeat(math.ceil(total_pages / 4))[:total_pages]
    scale = row_scales.view(total_pages, 1).expand(total_pages, 32).contiguous()
    normalized_high = (
        torch.randn((total_pages, 32, 64), device=device) * 0.125
    ).to(
        torch.float8_e4m3fn
    )
    physical_high = (normalized_high.float() * scale.float().unsqueeze(-1)).to(
        torch.float8_e4m3fn
    )
    residual = torch.zeros((total_pages, 32, 32), device=device, dtype=torch.uint8)
    table = torch.arange(total_pages, device=device, dtype=torch.int32).view(
        batch, page_count
    )
    seq_lens = torch.full(
        (batch,), sequence_length, device=device, dtype=torch.int32
    )
    normalized, _ = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        scale=scale,
        high_rope=normalized_high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=seq_lens,
        physical=False,
    )
    serial, _ = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        scale=scale,
        high_rope=physical_high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=seq_lens,
        physical=True,
        lookahead=False,
    )
    lookahead, _ = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        scale=scale,
        high_rope=physical_high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=seq_lens,
        physical=True,
    )
    dual_tmem, dual_fault_status = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        scale=scale,
        high_rope=physical_high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=seq_lens,
        physical=True,
        lookahead=False,
        dual_tmem=True,
    )
    literal_residual = torch.randint(
        0, 256, (total_pages, 32, 32), device=device, dtype=torch.uint8
    )
    unit_scale = torch.ones_like(scale)
    residual_normalized, _ = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        scale=unit_scale,
        high_rope=normalized_high,
        residual_rope=literal_residual,
        block_tables=table,
        seq_lens=seq_lens,
        physical=False,
    )
    residual_dual, residual_fault_status = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        scale=unit_scale,
        high_rope=normalized_high,
        residual_rope=literal_residual,
        block_tables=table,
        seq_lens=seq_lens,
        physical=True,
        lookahead=False,
        dual_tmem=True,
    )
    torch.cuda.synchronize()
    serial_delta = (serial.float() - normalized.float()).abs()
    lookahead_delta = (lookahead.float() - normalized.float()).abs()
    dual_tmem_delta = (dual_tmem.float() - normalized.float()).abs()
    residual_delta = (residual_dual.float() - residual_normalized.float()).abs()
    result = {
        "sequence_length": sequence_length,
        "batch": batch,
        "query_len": query_len,
        "serial_finite": bool(torch.isfinite(serial).all()),
        "serial_max_abs": float(serial_delta.max()),
        "serial_mean_abs": float(serial_delta.mean()),
        "lookahead_finite": bool(torch.isfinite(lookahead).all()),
        "lookahead_max_abs": float(lookahead_delta.max()),
        "lookahead_mean_abs": float(lookahead_delta.mean()),
        "dual_tmem_finite": bool(torch.isfinite(dual_tmem).all()),
        "dual_tmem_max_abs": float(dual_tmem_delta.max()),
        "dual_tmem_mean_abs": float(dual_tmem_delta.mean()),
        "dual_tmem_fault_status": int(dual_fault_status.item()),
        "residual_max_abs": float(residual_delta.max()),
        "residual_mean_abs": float(residual_delta.mean()),
        "residual_mismatch_count": int(torch.count_nonzero(residual_delta)),
        "residual_fault_status": int(residual_fault_status.item()),
    }
    print(json.dumps(result, sort_keys=True))
    if (
        not result["serial_finite"]
        or not result["lookahead_finite"]
        or not result["dual_tmem_finite"]
        or result["serial_max_abs"] > 0.02
        or result["lookahead_max_abs"] > 0.02
        or result["dual_tmem_max_abs"] > 0.02
        or result["dual_tmem_fault_status"] != 0
        # Normalized R31 accumulates latent, high, and residual in TMEM order;
        # dual-TMEM forms (high + residual) before adding latent in registers.
        # The algebra is identical but FP32 reassociation can round one BF16
        # output differently.  Keep the same strict 0.02 component tolerance
        # used above while reporting the mismatch count explicitly.
        or result["residual_max_abs"] > 0.02
        or result["residual_fault_status"] != 0
    ):
        raise AssertionError(result)
    return result


def run_masked_split_graph() -> dict[str, object]:
    """Exercise q5 causal, a fully masked split, and graph replay together."""

    torch.manual_seed(20260826)
    device = torch.device("cuda:0")
    batch, query_len, sequence_length = 1, 5, 129
    page_count = 8  # Four-page TMA padding for five live pages.
    query_latent = (torch.randn((batch, query_len, 8, 512), device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    query_rope = (torch.randn((batch, query_len, 8, 256), device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    packed = torch.randint(
        0, 256, (page_count, 32, 256), device=device, dtype=torch.uint8
    )
    scale = torch.ones((page_count, 32), device=device, dtype=torch.bfloat16)
    high = (torch.randn((page_count, 32, 64), device=device) * 0.125).to(
        torch.float8_e4m3fn
    )
    residual = torch.randint(
        0, 256, (page_count, 32, 32), device=device, dtype=torch.uint8
    )
    table = torch.arange(page_count, device=device, dtype=torch.int32).view(1, -1)
    lengths = torch.tensor([sequence_length], device=device, dtype=torch.int32)
    normalized_workspace = torch.empty((8 << 20,), device=device, dtype=torch.int8)
    candidate_workspace = torch.empty_like(normalized_workspace)
    normalized_out = torch.empty_like(query_latent, dtype=torch.bfloat16)
    candidate_out = torch.empty_like(normalized_out)
    fault_status = torch.zeros((1,), device=device, dtype=torch.int32)
    common = dict(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed,
        reconstruction_scale=scale,
        high_rope=high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=lengths,
        max_seq_len=sequence_length,
        softmax_scale=0.125,
        causal_mask=True,
        split_kv_override=2,
        enable_pdl=False,
    )
    tokenspeed_mla_decode_tq_r31(
        **common, workspace_buffer=normalized_workspace, out=normalized_out
    )
    tokenspeed_mla_decode_tq_r31(
        **common,
        workspace_buffer=candidate_workspace,
        out=candidate_out,
        _physical_split_score=True,
        _physical_split_score_lookahead=False,
        _physical_split_score_dual_tmem=True,
        _physical_split_score_fault_status=fault_status,
    )
    torch.cuda.synchronize()
    eager_candidate = candidate_out.clone()
    output_pointer = candidate_out.data_ptr()
    workspace_pointer = candidate_workspace.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tokenspeed_mla_decode_tq_r31(
            **common,
            workspace_buffer=candidate_workspace,
            out=candidate_out,
            _physical_split_score=True,
            _physical_split_score_lookahead=False,
            _physical_split_score_dual_tmem=True,
            _physical_split_score_fault_status=fault_status,
        )
    torch.cuda.synchronize()
    allocated_after_capture = torch.cuda.memory_allocated()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    allocated_after_replay = torch.cuda.memory_allocated()
    delta = (candidate_out.float() - normalized_out.float()).abs()
    replay_delta = (candidate_out.float() - eager_candidate.float()).abs()
    result = {
        "candidate_finite": bool(torch.isfinite(candidate_out).all()),
        "candidate_max_abs": float(delta.max()),
        "candidate_mean_abs": float(delta.mean()),
        "candidate_mismatch_count": int(torch.count_nonzero(delta)),
        "fault_status": int(fault_status.item()),
        "graph_replay_max_abs": float(replay_delta.max()),
        "graph_replay_mean_abs": float(replay_delta.mean()),
        "output_pointer_stable": candidate_out.data_ptr() == output_pointer,
        "workspace_pointer_stable": candidate_workspace.data_ptr() == workspace_pointer,
        "allocation_stable": allocated_after_replay == allocated_after_capture,
        "fully_masked_split_exercised": True,
    }
    print(json.dumps(result, sort_keys=True))
    if (
        not result["candidate_finite"]
        or result["candidate_max_abs"] > 0.02
        or result["fault_status"] != 0
        or result["graph_replay_max_abs"] != 0.0
        or not result["output_pointer_stable"]
        or not result["workspace_pointer_stable"]
        or not result["allocation_stable"]
    ):
        raise AssertionError(result)
    return result


def run_invalid_scale() -> dict[str, object]:
    """An invalid live key must fault and equal removing that key."""

    torch.manual_seed(20260825)
    device = torch.device("cuda:0")
    query_latent = (torch.randn((1, 1, 8, 512), device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    query_rope = (torch.randn((1, 1, 8, 256), device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    packed_latent = torch.randint(
        0, 256, (4, 32, 256), device=device, dtype=torch.uint8
    )
    scale = torch.ones((4, 32), device=device, dtype=torch.bfloat16)
    high = (torch.randn((4, 32, 64), device=device) * 0.125).to(
        torch.float8_e4m3fn
    )
    residual = torch.randint(
        0, 256, (4, 32, 32), device=device, dtype=torch.uint8
    )
    table = torch.tensor([[2, 0, 3, 1]], device=device, dtype=torch.int32)
    # Logical key 96 maps to physical page 1, row 0. The rest of that page is
    # poisoned fixed-N128 padding and must never be inspected as a live scale.
    scale[1, :] = torch.tensor(float("nan"), device=device, dtype=torch.bfloat16)
    high[1, :] = torch.tensor(float("nan"), device=device).to(
        torch.float8_e4m3fn
    )
    invalid, invalid_status = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        scale=scale,
        high_rope=high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=torch.tensor([97], device=device, dtype=torch.int32),
        physical=True,
        lookahead=False,
        dual_tmem=True,
    )
    removed, removed_status = _launch(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        scale=scale,
        high_rope=high,
        residual_rope=residual,
        block_tables=table,
        seq_lens=torch.tensor([96], device=device, dtype=torch.int32),
        physical=True,
        lookahead=False,
        dual_tmem=True,
    )
    torch.cuda.synchronize()
    delta = (invalid.float() - removed.float()).abs()
    boundary_status = {}
    boundary_high = high.clone()
    boundary_high[1, 0].zero_()
    boundary_residual = residual.clone()
    boundary_residual[1, 0].zero_()
    boundary_packed = packed_latent.clone()
    boundary_packed[1, 0].zero_()
    for name, value, expected_valid in (
        ("below_min", 2.0**-17, False),
        ("at_min", 2.0**-16, True),
        ("at_max", 224.0 * (2.0**16), True),
        ("above_max", 240.0 * (2.0**16), False),
    ):
        boundary_scale = torch.ones_like(scale)
        boundary_scale[1, 0] = value
        boundary_output, status = _launch(
            query_latent=query_latent,
            query_rope=query_rope,
            packed_latent=boundary_packed,
            scale=boundary_scale,
            high_rope=boundary_high,
            residual_rope=boundary_residual,
            block_tables=table,
            seq_lens=torch.tensor([97], device=device, dtype=torch.int32),
            physical=True,
            lookahead=False,
            dual_tmem=True,
        )
        torch.cuda.synchronize()
        observed = int(status.item())
        boundary_status[name] = observed
        if not torch.isfinite(boundary_output).all() or (observed == 0) != expected_valid:
            raise AssertionError(
                {"boundary": name, "status": observed, "expected_valid": expected_valid}
            )
    result = {
        "invalid_finite": bool(torch.isfinite(invalid).all()),
        "invalid_status": int(invalid_status.item()),
        "removed_status": int(removed_status.item()),
        "max_abs_vs_removed": float(delta.max()),
        "mean_abs_vs_removed": float(delta.mean()),
        "boundary_status": boundary_status,
    }
    print(json.dumps(result, sort_keys=True))
    if (
        not result["invalid_finite"]
        or result["invalid_status"] == 0
        or result["removed_status"] != 0
        or result["max_abs_vs_removed"] > 0.0
    ):
        raise AssertionError(result)
    return result


def run_dynamic_batch_reuse() -> list[dict[str, object]]:
    """Validate one dual-TMEM module across production capture batch order."""

    return [
        run(batch=batch, query_len=query_len)
        for query_len in (1, 5)
        for batch in (8, 5, 1)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=384)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-len", type=int, choices=(1, 5), default=1)
    parser.add_argument("--invalid-scale", action="store_true")
    parser.add_argument("--masked-split-graph", action="store_true")
    parser.add_argument("--dynamic-batch-reuse", action="store_true")
    args = parser.parse_args()
    if args.invalid_scale:
        run_invalid_scale()
    elif args.masked_split_graph:
        run_masked_split_graph()
    elif args.dynamic_batch_reuse:
        run_dynamic_batch_reuse()
    else:
        run(args.sequence_length, args.batch, args.query_len)
