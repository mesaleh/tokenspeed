# Copyright (c) 2026 LightSeek Foundation

"""Single-pass reduction for mixed dense-FP8/R31 MLA split producers."""

from __future__ import annotations

import numbers

import torch
import triton
import triton.language as tl
from tokenspeed_mla.mla_decode_tq_e2m1 import _overlaps


_MAX_TOTAL_SPLITS = 128


@triton.jit
def _reduce_mixed_workspace_kernel(
    workspace,
    hot_seq_lens,
    cold_seq_lens,
    output,
    output_lse,
    ROWS: tl.constexpr,
    ROWS_PER_BATCH: tl.constexpr,
    D: tl.constexpr,
    HOT_DECLARED: tl.constexpr,
    COLD_DECLARED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch_index = row // ROWS_PER_BATCH
    hot_length = tl.load(hot_seq_lens + batch_index).to(tl.int32)
    cold_length = tl.load(cold_seq_lens + batch_index).to(tl.int32)

    hot_tiles = tl.cdiv(hot_length, 128)
    cold_tiles = tl.cdiv(cold_length, 128)
    hot_tiles_per_cta = tl.cdiv(hot_tiles, HOT_DECLARED)
    cold_tiles_per_cta = tl.cdiv(cold_tiles, COLD_DECLARED)
    hot_effective = tl.where(
        hot_tiles > 0, tl.cdiv(hot_tiles, tl.maximum(hot_tiles_per_cta, 1)), 0
    )
    cold_effective = tl.where(
        cold_tiles > 0,
        tl.cdiv(cold_tiles, tl.maximum(cold_tiles_per_cta, 1)),
        0,
    )

    hot_acc_offset = row * HOT_DECLARED * D
    hot_lse_offset = ROWS * HOT_DECLARED * D + row * HOT_DECLARED
    cold_region_offset = ROWS * HOT_DECLARED * (D + 1)
    cold_acc_offset = cold_region_offset + row * COLD_DECLARED * D
    cold_lse_offset = (
        cold_region_offset + ROWS * COLD_DECLARED * D + row * COLD_DECLARED
    )
    global_max = -float("inf")
    for split in range(HOT_DECLARED):
        active = split < hot_effective
        local_lse = tl.load(
            workspace + hot_lse_offset + split,
            mask=active,
            other=-float("inf"),
        )
        global_max = tl.maximum(global_max, local_lse)
    for split in range(COLD_DECLARED):
        active = split < cold_effective
        local_lse = tl.load(
            workspace + cold_lse_offset + split,
            mask=active,
            other=-float("inf"),
        )
        global_max = tl.maximum(global_max, local_lse)

    has_mass = global_max != -float("inf")
    safe_max = tl.where(has_mass, global_max, 0.0)
    mass = 0.0
    for split in range(HOT_DECLARED):
        active = split < hot_effective
        local_lse = tl.load(
            workspace + hot_lse_offset + split,
            mask=active,
            other=-float("inf"),
        )
        mass += tl.where(active, tl.exp2(local_lse - safe_max), 0.0)
    for split in range(COLD_DECLARED):
        active = split < cold_effective
        local_lse = tl.load(
            workspace + cold_lse_offset + split,
            mask=active,
            other=-float("inf"),
        )
        mass += tl.where(active, tl.exp2(local_lse - safe_max), 0.0)
    global_lse = tl.where(has_mass, safe_max + tl.log2(mass), -float("inf"))
    safe_global_lse = tl.where(has_mass, global_lse, 0.0)

    features = tl.arange(0, BLOCK_D)
    feature_mask = features < D
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for split in range(HOT_DECLARED):
        active = split < hot_effective
        local_lse = tl.load(
            workspace + hot_lse_offset + split,
            mask=active,
            other=-float("inf"),
        )
        partial = tl.load(
            workspace + hot_acc_offset + split * D + features,
            mask=feature_mask & active,
            other=0.0,
        )
        weight = tl.where(
            active & has_mass, tl.exp2(local_lse - safe_global_lse), 0.0
        )
        accumulator += partial * weight
    for split in range(COLD_DECLARED):
        active = split < cold_effective
        local_lse = tl.load(
            workspace + cold_lse_offset + split,
            mask=active,
            other=-float("inf"),
        )
        partial = tl.load(
            workspace + cold_acc_offset + split * D + features,
            mask=feature_mask & active,
            other=0.0,
        )
        weight = tl.where(
            active & has_mass, tl.exp2(local_lse - safe_global_lse), 0.0
        )
        accumulator += partial * weight

    accumulator = tl.where(has_mass, accumulator, 0.0)
    tl.store(output + row * D + features, accumulator, mask=feature_mask)
    tl.store(output_lse + row, global_lse)


def reduce_mla_mixed_workspace(
    workspace_buffer: torch.Tensor,
    hot_seq_lens: torch.Tensor,
    cold_seq_lens: torch.Tensor,
    hot_declared_splits: int,
    cold_declared_splits: int,
    output: torch.Tensor,
    output_lse: torch.Tensor,
) -> None:
    """Reduce disjoint hot/cold split states from one caller-owned workspace.

    The caller partitions one allocation into a contiguous hot producer region
    followed by a contiguous cold producer region. Each region preserves the
    stock ``[acc_o, acc_lse]`` layout, avoiding strided producer writes. Only
    each request's effective split count is read; trailing declared slots may
    remain unwritten and require no memset.
    """
    for name, value in (
        ("hot_declared_splits", hot_declared_splits),
        ("cold_declared_splits", cold_declared_splits),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be an integer")
        if not 1 <= int(value) <= 64:
            raise ValueError(f"{name} must be in [1, 64], got {value}")
    hot_declared_splits = int(hot_declared_splits)
    cold_declared_splits = int(cold_declared_splits)
    total_declared = hot_declared_splits + cold_declared_splits
    if total_declared > _MAX_TOTAL_SPLITS:
        raise ValueError(
            f"combined declared split count must not exceed {_MAX_TOTAL_SPLITS}"
        )
    if workspace_buffer.dtype != torch.int8 or workspace_buffer.ndim != 1:
        raise ValueError("workspace_buffer must be a contiguous 1-D int8 tensor")
    if not workspace_buffer.is_cuda or not workspace_buffer.is_contiguous():
        raise ValueError("workspace_buffer must be contiguous CUDA storage")
    if workspace_buffer.data_ptr() % 32:
        raise ValueError("workspace_buffer must be at least 32-byte aligned")
    if output.dtype != torch.bfloat16 or output.ndim != 4:
        raise ValueError("output must be a BF16 [B,Q,H,D] tensor")
    if not output.is_cuda or not output.is_contiguous():
        raise ValueError("output must be contiguous CUDA storage")
    batch, query_len, heads, latent_dim = output.shape
    if latent_dim != 512:
        raise ValueError(f"mixed MLA reduction requires D=512, got {latent_dim}")
    if output_lse.dtype != torch.float32 or tuple(output_lse.shape) != (
        batch,
        query_len,
        heads,
    ):
        raise ValueError("output_lse must be contiguous FP32 [B,Q,H]")
    if not output_lse.is_cuda or not output_lse.is_contiguous():
        raise ValueError("output_lse must be contiguous CUDA storage")
    for name, seq_lens in (
        ("hot_seq_lens", hot_seq_lens),
        ("cold_seq_lens", cold_seq_lens),
    ):
        if (
            seq_lens.dtype != torch.int32
            or tuple(seq_lens.shape) != (batch,)
            or not seq_lens.is_cuda
            or not seq_lens.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA int32 [B]")
        if seq_lens.device != output.device:
            raise ValueError(f"{name} must be on {output.device}")
    if workspace_buffer.device != output.device or output_lse.device != output.device:
        raise ValueError("workspace and outputs must be on the same CUDA device")
    writable = (output, output_lse)
    readonly = (workspace_buffer, hot_seq_lens, cold_seq_lens)
    for index, lhs in enumerate(writable):
        for rhs in writable[index + 1 :] + readonly:
            if _overlaps(lhs, rhs):
                raise ValueError("mixed output tensors must not alias inputs")

    rows = batch * query_len * heads
    required_bytes = rows * total_declared * (latent_dim + 1) * 4
    if workspace_buffer.numel() < required_bytes:
        raise ValueError(
            f"workspace_buffer has {workspace_buffer.numel()} bytes, "
            f"but mixed reduction requires {required_bytes}"
        )
    workspace_float = workspace_buffer[:required_bytes].view(torch.float32)
    _reduce_mixed_workspace_kernel[(rows,)](
        workspace_float,
        hot_seq_lens,
        cold_seq_lens,
        output,
        output_lse,
        ROWS=rows,
        ROWS_PER_BATCH=query_len * heads,
        D=latent_dim,
        HOT_DECLARED=hot_declared_splits,
        COLD_DECLARED=cold_declared_splits,
        BLOCK_D=triton.next_power_of_2(latent_dim),
        num_warps=8,
    )


__all__ = ["reduce_mla_mixed_workspace"]
