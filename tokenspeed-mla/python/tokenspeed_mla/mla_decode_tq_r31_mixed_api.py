# Copyright (c) 2026 LightSeek Foundation

"""Public one-grid mixed dense-FP8/R31 MLA decode for Blackwell SM100."""

from __future__ import annotations

import math
import numbers
import threading
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
)
from tokenspeed_mla.mla_decode_tq_e2m1 import (
    _LATENT_DIM,
    _MMA_PV_TILER,
    _MMA_QK_TILER,
    _NUM_HEADS,
    _PAGE_SIZE,
    _ROPE_DIM,
    _SUPPORTED_QUERY_LENGTHS,
    _as_cute_tensor,
    _require_tensor,
    _use_early_final_pcor,
    _use_packed_p_scale_math,
)
from tokenspeed_mla.mla_decode_tq_r31 import _validate_r31_inputs
from tokenspeed_mla.mla_decode_tq_r31_mixed import reduce_mla_mixed_workspace
from tokenspeed_mla.mla_decode_tq_r31_mixed_native import (
    BlackwellMixedFP8R31Producer,
)
from tokenspeed_mla.mla_helpers import get_mla_decode_fold_sq_factor
from tokenspeed_mla.utils import get_max_active_clusters

_MAX_SPLITS_PER_OWNER = 64
_MAX_COMPILED_MIXED_VARIANTS = 128

_COMPILED_MIXED_KERNELS: dict[tuple, Callable] = {}
_COMPILE_MIXED_LOCK = threading.Lock()


def _require_page_rows(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    row_width: int,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (_PAGE_SIZE, row_width):
        raise ValueError(
            f"{name} must have shape [num_pages,{_PAGE_SIZE},{row_width}], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one physical page")
    if tuple(tensor.stride()[1:]) != (row_width, 1):
        raise ValueError(
            f"{name} rows must be contiguous with inner strides "
            f"({row_width}, 1), got {tensor.stride()}"
        )
    minimum_page_stride = _PAGE_SIZE * row_width
    if tensor.stride(0) < minimum_page_stride:
        raise ValueError(
            f"{name} page stride must be at least {minimum_page_stride}, "
            f"got {tensor.stride(0)}"
        )
    page_stride_bytes = tensor.stride(0) * tensor.element_size()
    if page_stride_bytes % 16:
        raise ValueError(
            f"{name} page stride must be 16-byte aligned, "
            f"got {page_stride_bytes} bytes"
        )
    if tensor.data_ptr() % 16:
        raise ValueError(
            f"{name} must be 16-byte aligned, got address 0x{tensor.data_ptr():x}"
        )


def _require_query_component(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    width: int,
) -> tuple[int, int]:
    """Validate contiguous or row-padded FP8 query storage."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"{name} must have dtype {torch.float8_e4m3fn}, got {tensor.dtype}"
        )
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be 4D, got shape {tuple(tensor.shape)}")
    batch, query_len, heads, actual_width = tensor.shape
    if (heads, actual_width) != (_NUM_HEADS, width):
        raise ValueError(
            f"{name} must have shape [B,q_len,{_NUM_HEADS},{width}], got "
            f"{tuple(tensor.shape)}"
        )
    strides = tensor.stride()
    if (
        strides[3] != 1
        or strides[2] < width
        or strides[1] != heads * strides[2]
        or strides[0] < query_len * strides[1]
    ):
        raise ValueError(
            f"{name} must be row-major with a contiguous feature dimension, "
            f"got stride {strides}"
        )
    if any(stride % 16 for stride in strides[:3]):
        raise ValueError(
            f"{name} outer strides must be 16-byte aligned for TMA, "
            f"got stride {strides}"
        )
    if tensor.numel() and tensor.data_ptr() % 16:
        raise ValueError(
            f"{name} must be 16-byte aligned, got address 0x{tensor.data_ptr():x}"
        )
    return batch, query_len


def _require_table_and_lengths(
    owner: str,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    causal_seqs: torch.Tensor,
    *,
    device: torch.device,
    batch: int,
    max_seq_len: int,
) -> None:
    _require_tensor(
        f"{owner}_block_tables",
        block_tables,
        dtype=torch.int32,
        ndim=2,
        device=device,
        alignment=4,
    )
    if block_tables.shape[0] != batch or block_tables.shape[1] <= 0:
        raise ValueError(
            f"{owner}_block_tables must have shape [B,max_pages] with "
            f"B={batch}, got {tuple(block_tables.shape)}"
        )
    pages_per_tile = _MMA_QK_TILER[1] // _PAGE_SIZE
    if block_tables.shape[1] % pages_per_tile:
        raise ValueError(
            f"{owner}_block_tables width must be padded to a multiple of "
            f"{pages_per_tile}, got {block_tables.shape[1]}"
        )
    for suffix, value in (("seq_lens", seq_lens), ("causal_seqs", causal_seqs)):
        _require_tensor(
            f"{owner}_{suffix}",
            value,
            dtype=torch.int32,
            ndim=1,
            device=device,
            alignment=4,
        )
        if tuple(value.shape) != (batch,):
            raise ValueError(
                f"{owner}_{suffix} must have shape ({batch},), "
                f"got {tuple(value.shape)}"
            )
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, numbers.Integral):
        raise TypeError(f"{owner}_max_seq_len must be an integer")
    if int(max_seq_len) <= 0:
        raise ValueError(f"{owner}_max_seq_len must be positive")
    capacity = block_tables.shape[1] * _PAGE_SIZE
    if int(max_seq_len) > capacity:
        raise ValueError(
            f"{owner}_max_seq_len {max_seq_len} exceeds block-table "
            f"capacity {capacity}"
        )


def _resolve_owner_splits(name: str, value: int, *, max_seq_len: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    available_tiles = (max_seq_len + _MMA_QK_TILER[1] - 1) // _MMA_QK_TILER[1]
    upper_bound = min(_MAX_SPLITS_PER_OWNER, available_tiles)
    if not 1 <= value <= upper_bound:
        raise ValueError(
            f"{name} must be in [1, {upper_bound}] for max_seq_len="
            f"{max_seq_len}, got {value}"
        )
    return value


def _byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the addressed byte interval for a positive-stride tensor."""

    if tensor.numel() == 0:
        return (tensor.data_ptr(), tensor.data_ptr())
    if any(stride < 0 for stride in tensor.stride()):
        raise ValueError("mixed MLA tensors may not have negative strides")
    last_element_offset = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    begin = tensor.data_ptr()
    end = begin + (last_element_offset + 1) * tensor.element_size()
    return begin, end


def _overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.device != rhs.device or lhs.numel() == 0 or rhs.numel() == 0:
        return False
    lhs_begin, lhs_end = _byte_span(lhs)
    rhs_begin, rhs_end = _byte_span(rhs)
    return max(lhs_begin, rhs_begin) < min(lhs_end, rhs_end)


def _validate_mixed_inputs(
    *,
    hot_query_latent: torch.Tensor,
    hot_query_rope: torch.Tensor,
    hot_cache: torch.Tensor,
    hot_block_tables: torch.Tensor,
    hot_seq_lens: torch.Tensor,
    hot_causal_seqs: torch.Tensor,
    hot_max_seq_len: int,
    cold_query_latent: torch.Tensor,
    cold_query_rope: torch.Tensor,
    cold_packed_latent: torch.Tensor,
    cold_reconstruction_scale: torch.Tensor,
    cold_high_rope: torch.Tensor,
    cold_residual_rope: torch.Tensor,
    cold_block_tables: torch.Tensor,
    cold_seq_lens: torch.Tensor,
    cold_causal_seqs: torch.Tensor,
    cold_max_seq_len: int,
    workspace_buffer: torch.Tensor,
    hot_splits: int,
    cold_splits: int,
    out: torch.Tensor,
    lse_out: torch.Tensor,
    fault_status: Optional[torch.Tensor] = None,
) -> tuple[int, int, int, int]:
    if not isinstance(hot_query_latent, torch.Tensor):
        raise TypeError("hot_query_latent must be a torch.Tensor")
    if not hot_query_latent.is_cuda:
        raise ValueError("mixed FP8/R31 MLA decode requires CUDA tensors")
    device = hot_query_latent.device
    batch, query_len = _require_query_component(
        "hot_query_latent",
        hot_query_latent,
        device=device,
        width=_LATENT_DIM,
    )
    if batch <= 0:
        raise ValueError(f"query batch must be positive, got {batch}")
    if query_len not in _SUPPORTED_QUERY_LENGTHS:
        raise ValueError(f"mixed MLA supports only q_len 1 or 5, got {query_len}")
    rope_batch, rope_query_len = _require_query_component(
        "hot_query_rope",
        hot_query_rope,
        device=device,
        width=_ROPE_DIM,
    )
    if (rope_batch, rope_query_len) != (batch, query_len):
        raise ValueError(
            "hot query components must have matching batch/query dimensions, "
            f"got latent={(batch, query_len)} rope="
            f"{(rope_batch, rope_query_len)}"
        )
    _require_page_rows(
        "hot_cache",
        hot_cache,
        dtype=torch.float8_e4m3fn,
        device=device,
        row_width=_LATENT_DIM + _ROPE_DIM,
    )
    _require_table_and_lengths(
        "hot",
        hot_block_tables,
        hot_seq_lens,
        hot_causal_seqs,
        device=device,
        batch=batch,
        max_seq_len=hot_max_seq_len,
    )

    cold_batch, cold_query_len = _validate_r31_inputs(
        cold_query_latent,
        cold_query_rope,
        cold_packed_latent,
        cold_reconstruction_scale,
        cold_high_rope,
        cold_residual_rope,
        workspace_buffer,
        cold_block_tables,
        cold_seq_lens,
        cold_max_seq_len,
        out,
        lse_out,
        True,
        fault_status,
    )
    if (cold_batch, cold_query_len) != (batch, query_len):
        raise ValueError(
            "hot and cold queries must have matching batch/query dimensions, "
            f"got hot={(batch, query_len)} cold={(cold_batch, cold_query_len)}"
        )
    _require_tensor(
        "cold_causal_seqs",
        cold_causal_seqs,
        dtype=torch.int32,
        ndim=1,
        device=device,
        alignment=4,
    )
    if tuple(cold_causal_seqs.shape) != (batch,):
        raise ValueError(
            f"cold_causal_seqs must have shape ({batch},), "
            f"got {tuple(cold_causal_seqs.shape)}"
        )

    hot_splits = _resolve_owner_splits(
        "hot_splits", hot_splits, max_seq_len=int(hot_max_seq_len)
    )
    cold_splits = _resolve_owner_splits(
        "cold_splits", cold_splits, max_seq_len=int(cold_max_seq_len)
    )
    rows = batch * query_len * _NUM_HEADS
    required_bytes = rows * (hot_splits + cold_splits) * (_LATENT_DIM + 1) * 4
    if workspace_buffer.numel() < required_bytes:
        raise ValueError(
            f"workspace_buffer has {workspace_buffer.numel()} bytes, but mixed "
            f"decode requires {required_bytes}"
        )

    writable = tuple(
        tensor
        for tensor in (workspace_buffer, out, lse_out, fault_status)
        if tensor is not None
    )
    readonly = (
        hot_query_latent,
        hot_query_rope,
        hot_cache,
        hot_block_tables,
        hot_seq_lens,
        hot_causal_seqs,
        cold_query_latent,
        cold_query_rope,
        cold_packed_latent,
        cold_reconstruction_scale,
        cold_high_rope,
        cold_residual_rope,
        cold_block_tables,
        cold_seq_lens,
        cold_causal_seqs,
    )
    for index, lhs in enumerate(writable):
        for rhs in writable[index + 1 :] + readonly:
            if _overlaps(lhs, rhs):
                raise ValueError("mixed output/workspace tensors must not alias inputs")
    return batch, query_len, hot_splits, cold_splits


def _tensor_signature(
    tensor: torch.Tensor, *, omit_leading_extent: bool = False
) -> tuple:
    shape = tuple(tensor.shape[1:]) if omit_leading_extent else tuple(tensor.shape)
    return (shape, tuple(tensor.stride()), tensor.dtype)


def _get_compiled_mixed_kernel(
    *,
    hot_query_latent: torch.Tensor,
    hot_query_rope: torch.Tensor,
    hot_cache: torch.Tensor,
    hot_block_tables: torch.Tensor,
    hot_workspace: torch.Tensor,
    hot_seq_lens: torch.Tensor,
    hot_causal_seqs: torch.Tensor,
    cold_query_latent: torch.Tensor,
    cold_query_rope: torch.Tensor,
    cold_packed_latent: torch.Tensor,
    cold_reconstruction_scale: torch.Tensor,
    cold_high_rope: torch.Tensor,
    cold_residual_rope: torch.Tensor,
    cold_block_tables: torch.Tensor,
    cold_workspace: torch.Tensor,
    cold_seq_lens: torch.Tensor,
    cold_causal_seqs: torch.Tensor,
    out: torch.Tensor,
    hot_splits: int,
    cold_splits: int,
    fold_sq_factor: int,
    enable_pdl: bool,
    physical_r31: bool,
    fault_status: Optional[torch.Tensor],
) -> Callable:
    batch, query_len = hot_query_latent.shape[:2]
    physical_q5_packed_p = physical_r31 and query_len == 5
    dynamic_batch = physical_r31
    key = (
        hot_query_latent.device.index,
        _tensor_signature(
            hot_query_latent, omit_leading_extent=dynamic_batch
        ),
        _tensor_signature(hot_query_rope, omit_leading_extent=dynamic_batch),
        _tensor_signature(hot_cache, omit_leading_extent=dynamic_batch),
        _tensor_signature(hot_block_tables, omit_leading_extent=dynamic_batch),
        _tensor_signature(hot_seq_lens, omit_leading_extent=dynamic_batch),
        _tensor_signature(hot_causal_seqs, omit_leading_extent=dynamic_batch),
        _tensor_signature(
            cold_query_latent, omit_leading_extent=dynamic_batch
        ),
        _tensor_signature(cold_query_rope, omit_leading_extent=dynamic_batch),
        _tensor_signature(cold_packed_latent, omit_leading_extent=dynamic_batch),
        _tensor_signature(
            cold_reconstruction_scale, omit_leading_extent=dynamic_batch
        ),
        _tensor_signature(cold_high_rope, omit_leading_extent=dynamic_batch),
        _tensor_signature(cold_residual_rope, omit_leading_extent=dynamic_batch),
        _tensor_signature(cold_block_tables, omit_leading_extent=dynamic_batch),
        _tensor_signature(cold_seq_lens, omit_leading_extent=dynamic_batch),
        _tensor_signature(cold_causal_seqs, omit_leading_extent=dynamic_batch),
        True if dynamic_batch else hot_workspace.numel(),
        True if dynamic_batch else cold_workspace.numel(),
        hot_splits,
        cold_splits,
        fold_sq_factor,
        enable_pdl,
        physical_r31,
        physical_q5_packed_p,
        fault_status is not None,
    )
    compiled = _COMPILED_MIXED_KERNELS.get(key)
    if compiled is not None:
        return compiled

    with _COMPILE_MIXED_LOCK:
        compiled = _COMPILED_MIXED_KERNELS.get(key)
        if compiled is not None:
            return compiled
        if len(_COMPILED_MIXED_KERNELS) >= _MAX_COMPILED_MIXED_VARIANTS:
            raise RuntimeError(
                "mixed FP8/R31 MLA compile cache reached its 128-variant "
                "safety bound"
            )

        common = dict(
            acc_dtype=cutlass.Float32,
            lse_dtype=cutlass.Float32,
            mma_qk_tiler_mn=_MMA_QK_TILER,
            mma_pv_tiler_mn=_MMA_PV_TILER,
            max_active_clusters=get_max_active_clusters(1),
            page_size=_PAGE_SIZE,
            skip_correction_threshold=0.0,
            is_persistent=False,
            is_var_seq=True,
            is_var_split_kv=False,
            fold_sq_factor=fold_sq_factor,
            num_heads=_NUM_HEADS,
            seq_len_q=query_len,
            cp_world=1,
            use_runtime_causal_bound=True,
            producer_only=True,
            is_causal=True,
        )
        hot_kernel = BlackwellMultiHeadLatentAttentionForwardFP8(**common)
        cold_kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
            **common,
            use_tq_e2m1=True,
            use_tq_r31_rope=not physical_r31,
            use_tq_r31_physical_split_score=physical_r31,
            tq_r31_physical_split_score_lookahead=False,
            tq_r31_physical_split_score_dual_tmem=physical_r31,
            tq_s1_scale_tma=True,
            tq_s1_scale_stages=3,
            tq_s1_k_rope_stages=1,
            tq_s1_packed_p_scale_math=(
                physical_q5_packed_p
                if physical_r31
                else _use_packed_p_scale_math(batch, query_len)
            ),
            tq_s1_early_final_pcor=(
                False
                if physical_r31
                else _use_early_final_pcor(batch, query_len)
            ),
            tq_r31_async_expand=True,
        )
        kernel = BlackwellMixedFP8R31Producer(hot_kernel, cold_kernel)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        with torch.cuda.device(hot_query_latent.device):
            compiled = cute.compile(
                kernel,
                _as_cute_tensor(hot_query_latent, cutlass.Float8E4M3FN, 3, 16),
                _as_cute_tensor(hot_query_rope, cutlass.Float8E4M3FN, 3, 16),
                _as_cute_tensor(
                    hot_cache[..., :_LATENT_DIM], cutlass.Float8E4M3FN, 2, 16
                ),
                _as_cute_tensor(
                    hot_cache[..., _LATENT_DIM:], cutlass.Float8E4M3FN, 2, 16
                ),
                _as_cute_tensor(hot_block_tables, cutlass.Int32, 1, 4),
                _as_cute_tensor(hot_workspace, cutlass.Int8, 0, 32),
                Int32(hot_splits),
                _as_cute_tensor(hot_seq_lens, cutlass.Int32, 0, 4),
                _as_cute_tensor(hot_causal_seqs, cutlass.Int32, 0, 4),
                _as_cute_tensor(cold_query_latent, cutlass.Float8E4M3FN, 3, 16),
                _as_cute_tensor(cold_query_rope, cutlass.Float8E4M3FN, 3, 16),
                _as_cute_tensor(cold_packed_latent, cutlass.Uint8, 2, 16),
                _as_cute_tensor(cold_high_rope, cutlass.Float8E4M3FN, 2, 16),
                _as_cute_tensor(cold_block_tables, cutlass.Int32, 1, 4),
                _as_cute_tensor(cold_workspace, cutlass.Int8, 0, 32),
                Int32(cold_splits),
                _as_cute_tensor(cold_seq_lens, cutlass.Int32, 0, 4),
                _as_cute_tensor(cold_causal_seqs, cutlass.Int32, 0, 4),
                _as_cute_tensor(out, cutlass.BFloat16, 3, 16),
                Float32(1.0),
                Float32(1.0),
                _as_cute_tensor(cold_reconstruction_scale, cutlass.BFloat16, 1, 16),
                _as_cute_tensor(cold_residual_rope, cutlass.Uint8, 2, 16),
                (
                    _as_cute_tensor(fault_status, cutlass.Int32, 0, 4)
                    if fault_status is not None
                    else None
                ),
                stream,
                enable_pdl,
                options="--enable-tvm-ffi --opt-level 3",
            )
        _COMPILED_MIXED_KERNELS[key] = compiled
        return compiled


def _tokenspeed_mla_decode_tq_r31_mixed_split_impl(
    hot_query_latent: torch.Tensor,
    hot_query_rope: torch.Tensor,
    hot_cache: torch.Tensor,
    hot_block_tables: torch.Tensor,
    hot_seq_lens: torch.Tensor,
    hot_causal_seqs: torch.Tensor,
    hot_max_seq_len: int,
    cold_query_latent: torch.Tensor,
    cold_query_rope: torch.Tensor,
    cold_packed_latent: torch.Tensor,
    cold_reconstruction_scale: torch.Tensor,
    cold_high_rope: torch.Tensor,
    cold_residual_rope: torch.Tensor,
    cold_block_tables: torch.Tensor,
    cold_seq_lens: torch.Tensor,
    cold_causal_seqs: torch.Tensor,
    cold_max_seq_len: int,
    workspace_buffer: torch.Tensor,
    hot_splits: int,
    cold_splits: int,
    softmax_scale: float,
    out: torch.Tensor,
    lse_out: torch.Tensor,
    *,
    output_scale: float = 1.0,
    enable_pdl: bool = False,
    physical_r31: bool = False,
    fault_status: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one batch from disjoint hot-FP8 and cold-R31 page owners.

    The caller supplies graph-stable owner-compacted page tables, sequence
    lengths, owner-resolved causal bounds, workspace, output, and LSE storage.
    The operation launches one native mixed producer followed by one global
    softmax reducer. It never expands the R31 cache or allocates a shadow.

    ``hot_max_seq_len`` and ``cold_max_seq_len`` are host-known scheduling
    bounds used to validate the explicit split counts. They may be nominal
    positive bounds when every request has zero tokens for one owner.
    """
    for name, value in (
        ("softmax_scale", softmax_scale),
        ("output_scale", output_scale),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real scalar")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value}")
    if not isinstance(enable_pdl, bool):
        raise TypeError("enable_pdl must be a bool")
    if not isinstance(physical_r31, bool):
        raise TypeError("physical_r31 must be a bool")
    if physical_r31 and fault_status is None:
        raise ValueError("physical mixed R31 requires caller-owned fault status")
    if not physical_r31 and fault_status is not None:
        raise ValueError("fault status is reserved for physical mixed R31")

    batch, query_len, hot_splits, cold_splits = _validate_mixed_inputs(
        hot_query_latent=hot_query_latent,
        hot_query_rope=hot_query_rope,
        hot_cache=hot_cache,
        hot_block_tables=hot_block_tables,
        hot_seq_lens=hot_seq_lens,
        hot_causal_seqs=hot_causal_seqs,
        hot_max_seq_len=hot_max_seq_len,
        cold_query_latent=cold_query_latent,
        cold_query_rope=cold_query_rope,
        cold_packed_latent=cold_packed_latent,
        cold_reconstruction_scale=cold_reconstruction_scale,
        cold_high_rope=cold_high_rope,
        cold_residual_rope=cold_residual_rope,
        cold_block_tables=cold_block_tables,
        cold_seq_lens=cold_seq_lens,
        cold_causal_seqs=cold_causal_seqs,
        cold_max_seq_len=cold_max_seq_len,
        workspace_buffer=workspace_buffer,
        hot_splits=hot_splits,
        cold_splits=cold_splits,
        out=out,
        lse_out=lse_out,
        fault_status=fault_status,
    )
    rows = batch * query_len * _NUM_HEADS
    hot_workspace_bytes = rows * hot_splits * (_LATENT_DIM + 1) * 4
    cold_workspace_bytes = rows * cold_splits * (_LATENT_DIM + 1) * 4
    active_workspace = workspace_buffer[: hot_workspace_bytes + cold_workspace_bytes]
    hot_workspace = active_workspace[:hot_workspace_bytes]
    cold_workspace = active_workspace[hot_workspace_bytes:]
    fold_sq_factor = get_mla_decode_fold_sq_factor(
        _NUM_HEADS, query_len, _MMA_QK_TILER[0]
    )
    compiled = _get_compiled_mixed_kernel(
        hot_query_latent=hot_query_latent,
        hot_query_rope=hot_query_rope,
        hot_cache=hot_cache,
        hot_block_tables=hot_block_tables,
        hot_workspace=hot_workspace,
        hot_seq_lens=hot_seq_lens,
        hot_causal_seqs=hot_causal_seqs,
        cold_query_latent=cold_query_latent,
        cold_query_rope=cold_query_rope,
        cold_packed_latent=cold_packed_latent,
        cold_reconstruction_scale=cold_reconstruction_scale,
        cold_high_rope=cold_high_rope,
        cold_residual_rope=cold_residual_rope,
        cold_block_tables=cold_block_tables,
        cold_workspace=cold_workspace,
        cold_seq_lens=cold_seq_lens,
        cold_causal_seqs=cold_causal_seqs,
        out=out,
        hot_splits=hot_splits,
        cold_splits=cold_splits,
        fold_sq_factor=fold_sq_factor,
        enable_pdl=enable_pdl,
        physical_r31=physical_r31,
        fault_status=fault_status,
    )

    import tvm_ffi

    with torch.cuda.device(hot_query_latent.device), tvm_ffi.use_torch_stream():
        compiled(
            hot_query_latent,
            hot_query_rope,
            hot_cache[..., :_LATENT_DIM],
            hot_cache[..., _LATENT_DIM:],
            hot_block_tables,
            hot_workspace,
            Int32(hot_splits),
            hot_seq_lens,
            hot_causal_seqs,
            cold_query_latent,
            cold_query_rope,
            cold_packed_latent,
            cold_high_rope,
            cold_block_tables,
            cold_workspace,
            Int32(cold_splits),
            cold_seq_lens,
            cold_causal_seqs,
            out,
            Float32(float(softmax_scale)),
            Float32(float(output_scale)),
            cold_reconstruction_scale,
            cold_residual_rope,
            fault_status,
        )
        reduce_mla_mixed_workspace(
            active_workspace,
            hot_seq_lens,
            cold_seq_lens,
            hot_splits,
            cold_splits,
            out,
            lse_out,
        )
    return out, lse_out


def tokenspeed_mla_decode_tq_r31_mixed(
    hot_query: torch.Tensor,
    hot_cache: torch.Tensor,
    hot_block_tables: torch.Tensor,
    hot_seq_lens: torch.Tensor,
    hot_causal_seqs: torch.Tensor,
    hot_max_seq_len: int,
    cold_query_latent: torch.Tensor,
    cold_query_rope: torch.Tensor,
    cold_packed_latent: torch.Tensor,
    cold_reconstruction_scale: torch.Tensor,
    cold_high_rope: torch.Tensor,
    cold_residual_rope: torch.Tensor,
    cold_block_tables: torch.Tensor,
    cold_seq_lens: torch.Tensor,
    cold_causal_seqs: torch.Tensor,
    cold_max_seq_len: int,
    workspace_buffer: torch.Tensor,
    hot_splits: int,
    cold_splits: int,
    softmax_scale: float,
    out: torch.Tensor,
    lse_out: torch.Tensor,
    *,
    output_scale: float = 1.0,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode mixed owners from the legacy contiguous 576-byte hot query."""

    if not isinstance(hot_query, torch.Tensor):
        raise TypeError("hot_query must be a torch.Tensor")
    if not hot_query.is_cuda:
        raise ValueError("mixed FP8/R31 MLA decode requires CUDA tensors")
    _require_tensor(
        "hot_query",
        hot_query,
        dtype=torch.float8_e4m3fn,
        ndim=4,
        device=hot_query.device,
        alignment=16,
    )
    if tuple(hot_query.shape[2:]) != (_NUM_HEADS, _LATENT_DIM + _ROPE_DIM):
        raise ValueError(
            "hot_query must have shape "
            f"[B,q_len,{_NUM_HEADS},{_LATENT_DIM + _ROPE_DIM}], got "
            f"{tuple(hot_query.shape)}"
        )
    return _tokenspeed_mla_decode_tq_r31_mixed_split_impl(
        hot_query_latent=hot_query[..., :_LATENT_DIM],
        hot_query_rope=hot_query[..., _LATENT_DIM:],
        hot_cache=hot_cache,
        hot_block_tables=hot_block_tables,
        hot_seq_lens=hot_seq_lens,
        hot_causal_seqs=hot_causal_seqs,
        hot_max_seq_len=hot_max_seq_len,
        cold_query_latent=cold_query_latent,
        cold_query_rope=cold_query_rope,
        cold_packed_latent=cold_packed_latent,
        cold_reconstruction_scale=cold_reconstruction_scale,
        cold_high_rope=cold_high_rope,
        cold_residual_rope=cold_residual_rope,
        cold_block_tables=cold_block_tables,
        cold_seq_lens=cold_seq_lens,
        cold_causal_seqs=cold_causal_seqs,
        cold_max_seq_len=cold_max_seq_len,
        workspace_buffer=workspace_buffer,
        hot_splits=hot_splits,
        cold_splits=cold_splits,
        softmax_scale=softmax_scale,
        out=out,
        lse_out=lse_out,
        output_scale=output_scale,
        enable_pdl=enable_pdl,
    )


def tokenspeed_mla_decode_tq_r31_mixed_split_query(
    query_latent: torch.Tensor,
    hot_query_rope: torch.Tensor,
    hot_cache: torch.Tensor,
    hot_block_tables: torch.Tensor,
    hot_seq_lens: torch.Tensor,
    hot_causal_seqs: torch.Tensor,
    hot_max_seq_len: int,
    cold_query_rope: torch.Tensor,
    cold_packed_latent: torch.Tensor,
    cold_reconstruction_scale: torch.Tensor,
    cold_high_rope: torch.Tensor,
    cold_residual_rope: torch.Tensor,
    cold_block_tables: torch.Tensor,
    cold_seq_lens: torch.Tensor,
    cold_causal_seqs: torch.Tensor,
    cold_max_seq_len: int,
    workspace_buffer: torch.Tensor,
    hot_splits: int,
    cold_splits: int,
    softmax_scale: float,
    out: torch.Tensor,
    lse_out: torch.Tensor,
    *,
    output_scale: float = 1.0,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode mixed owners while sharing one resident rotated latent query.

    ``query_latent`` is consumed by both the dense-FP8 hot owner and the R31
    cold owner. Separate 64-byte hot and 256-byte R31 RoPE operands avoid a
    caller-side 576-byte concatenation or a duplicate latent rotation.
    """

    return _tokenspeed_mla_decode_tq_r31_mixed_split_impl(
        hot_query_latent=query_latent,
        hot_query_rope=hot_query_rope,
        hot_cache=hot_cache,
        hot_block_tables=hot_block_tables,
        hot_seq_lens=hot_seq_lens,
        hot_causal_seqs=hot_causal_seqs,
        hot_max_seq_len=hot_max_seq_len,
        cold_query_latent=query_latent,
        cold_query_rope=cold_query_rope,
        cold_packed_latent=cold_packed_latent,
        cold_reconstruction_scale=cold_reconstruction_scale,
        cold_high_rope=cold_high_rope,
        cold_residual_rope=cold_residual_rope,
        cold_block_tables=cold_block_tables,
        cold_seq_lens=cold_seq_lens,
        cold_causal_seqs=cold_causal_seqs,
        cold_max_seq_len=cold_max_seq_len,
        workspace_buffer=workspace_buffer,
        hot_splits=hot_splits,
        cold_splits=cold_splits,
        softmax_scale=softmax_scale,
        out=out,
        lse_out=lse_out,
        output_scale=output_scale,
        enable_pdl=enable_pdl,
    )


def tokenspeed_mla_decode_tq_r31_physical_mixed_split_query(
    query_latent: torch.Tensor,
    hot_query_rope: torch.Tensor,
    hot_cache: torch.Tensor,
    hot_block_tables: torch.Tensor,
    hot_seq_lens: torch.Tensor,
    hot_causal_seqs: torch.Tensor,
    hot_max_seq_len: int,
    cold_query_rope: torch.Tensor,
    cold_packed_latent: torch.Tensor,
    cold_reconstruction_scale: torch.Tensor,
    cold_high_rope: torch.Tensor,
    cold_residual_rope: torch.Tensor,
    cold_block_tables: torch.Tensor,
    cold_seq_lens: torch.Tensor,
    cold_causal_seqs: torch.Tensor,
    cold_max_seq_len: int,
    workspace_buffer: torch.Tensor,
    hot_splits: int,
    cold_splits: int,
    softmax_scale: float,
    out: torch.Tensor,
    lse_out: torch.Tensor,
    fault_status: torch.Tensor,
    *,
    output_scale: float = 1.0,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode hot FP8 and physical-unit R31 owners in one producer grid.

    This entry point is intentionally distinct from the normalized R31 mixed
    APIs: its cold cache stores RoPE in physical post-RoPE units. The caller
    owns and clears ``fault_status`` before launch, then rejects the launch if
    that sticky word is nonzero after stream synchronization.
    """

    return _tokenspeed_mla_decode_tq_r31_mixed_split_impl(
        hot_query_latent=query_latent,
        hot_query_rope=hot_query_rope,
        hot_cache=hot_cache,
        hot_block_tables=hot_block_tables,
        hot_seq_lens=hot_seq_lens,
        hot_causal_seqs=hot_causal_seqs,
        hot_max_seq_len=hot_max_seq_len,
        cold_query_latent=query_latent,
        cold_query_rope=cold_query_rope,
        cold_packed_latent=cold_packed_latent,
        cold_reconstruction_scale=cold_reconstruction_scale,
        cold_high_rope=cold_high_rope,
        cold_residual_rope=cold_residual_rope,
        cold_block_tables=cold_block_tables,
        cold_seq_lens=cold_seq_lens,
        cold_causal_seqs=cold_causal_seqs,
        cold_max_seq_len=cold_max_seq_len,
        workspace_buffer=workspace_buffer,
        hot_splits=hot_splits,
        cold_splits=cold_splits,
        softmax_scale=softmax_scale,
        out=out,
        lse_out=lse_out,
        output_scale=output_scale,
        enable_pdl=enable_pdl,
        physical_r31=True,
        fault_status=fault_status,
    )


__all__ = [
    "tokenspeed_mla_decode_tq_r31_mixed",
    "tokenspeed_mla_decode_tq_r31_mixed_split_query",
    "tokenspeed_mla_decode_tq_r31_physical_mixed_split_query",
]
