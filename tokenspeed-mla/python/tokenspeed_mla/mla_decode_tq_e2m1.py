# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Packed E2M1 TurboQuant MLA decode integration for Blackwell SM100.

This module deliberately exposes a separate API from ``tokenspeed_mla_decode``.
The packed cache is not a dense FP8 cache: latent values are fixed hardware
E2M1 codes with a per-token BF16 reconstruction scale, and RoPE coordinates are
stored divided by that same scale. The kernel returns a BF16 latent vector in
the caller-owned rotated basis; it does not apply an inverse transform.
"""

import math
import numbers
import threading
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
)
from tokenspeed_mla.mla_helpers import get_mla_decode_fold_sq_factor
from tokenspeed_mla.utils import get_max_active_clusters, get_num_sm


_PAGE_SIZE = 32
_LATENT_DIM = 512
_PACKED_LATENT_DIM = _LATENT_DIM // 2
_ROPE_DIM = 64
_NUM_HEADS = 8
_SUPPORTED_QUERY_LENGTHS = (1, 5)
_MMA_QK_TILER = (64, 128)
_MMA_PV_TILER = (64, 256)


def _use_packed_p_scale_math(query_len: int) -> bool:
    """Select packed P-scale arithmetic only for q5 verification."""
    if query_len not in _SUPPORTED_QUERY_LENGTHS:
        raise ValueError(
            "TurboQuant E2M1 MLA packed P-scale routing supports only "
            f"q_len 1 or 5, got {query_len}"
        )
    return query_len == 5


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    ndim: int,
    device: torch.device,
    alignment: int,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride {tensor.stride()}")
    if tensor.numel() and tensor.data_ptr() % alignment:
        raise ValueError(
            f"{name} must be {alignment}-byte aligned, got address "
            f"0x{tensor.data_ptr():x}"
        )


def _overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return whether two nonempty contiguous tensors overlap in device memory."""
    if lhs.device != rhs.device or lhs.numel() == 0 or rhs.numel() == 0:
        return False
    lhs_begin = lhs.data_ptr()
    lhs_end = lhs_begin + lhs.numel() * lhs.element_size()
    rhs_begin = rhs.data_ptr()
    rhs_end = rhs_begin + rhs.numel() * rhs.element_size()
    return max(lhs_begin, rhs_begin) < min(lhs_end, rhs_end)


def _validate_inputs(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    packed_latent: torch.Tensor,
    reconstruction_scale: torch.Tensor,
    reciprocal_rope: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    out: Optional[torch.Tensor],
    lse_out: Optional[torch.Tensor],
    return_lse: bool,
) -> tuple[int, int]:
    if not isinstance(query_latent, torch.Tensor):
        raise TypeError("query_latent must be a torch.Tensor")
    if not query_latent.is_cuda:
        raise ValueError("TurboQuant E2M1 MLA decode requires CUDA tensors")
    device = query_latent.device

    _require_tensor(
        "query_latent",
        query_latent,
        dtype=torch.float8_e4m3fn,
        ndim=4,
        device=device,
        alignment=16,
    )
    batch, query_len, num_heads, latent_dim = query_latent.shape
    if batch <= 0:
        raise ValueError(f"query batch must be positive, got {batch}")
    if query_len not in _SUPPORTED_QUERY_LENGTHS:
        raise ValueError(
            "TurboQuant E2M1 MLA decode supports only q_len 1 or 5, "
            f"got {query_len}"
        )
    if num_heads != _NUM_HEADS or latent_dim != _LATENT_DIM:
        raise ValueError(
            "TurboQuant E2M1 MLA decode requires query shape "
            f"[B,q_len,{_NUM_HEADS},{_LATENT_DIM}], got "
            f"{tuple(query_latent.shape)}"
        )

    _require_tensor(
        "query_rope",
        query_rope,
        dtype=torch.bfloat16,
        ndim=4,
        device=device,
        alignment=16,
    )
    expected_query_rope = (batch, query_len, num_heads, _ROPE_DIM)
    if tuple(query_rope.shape) != expected_query_rope:
        raise ValueError(
            f"query_rope must have shape {expected_query_rope}, "
            f"got {tuple(query_rope.shape)}"
        )

    _require_tensor(
        "packed_latent",
        packed_latent,
        dtype=torch.uint8,
        ndim=3,
        device=device,
        alignment=16,
    )
    num_pages, page_size, packed_dim = packed_latent.shape
    if num_pages <= 0:
        raise ValueError(f"packed_latent must contain pages, got {num_pages}")
    if page_size != _PAGE_SIZE or packed_dim != _PACKED_LATENT_DIM:
        raise ValueError(
            "packed_latent must have shape "
            f"[num_pages,{_PAGE_SIZE},{_PACKED_LATENT_DIM}], got "
            f"{tuple(packed_latent.shape)}"
        )

    _require_tensor(
        "reconstruction_scale",
        reconstruction_scale,
        dtype=torch.bfloat16,
        ndim=2,
        device=device,
        alignment=16,
    )
    expected_scale = (num_pages, _PAGE_SIZE)
    if tuple(reconstruction_scale.shape) != expected_scale:
        raise ValueError(
            f"reconstruction_scale must have shape {expected_scale}, "
            f"got {tuple(reconstruction_scale.shape)}"
        )

    _require_tensor(
        "reciprocal_rope",
        reciprocal_rope,
        dtype=torch.bfloat16,
        ndim=3,
        device=device,
        alignment=16,
    )
    expected_reciprocal_rope = (num_pages, _PAGE_SIZE, _ROPE_DIM)
    if tuple(reciprocal_rope.shape) != expected_reciprocal_rope:
        raise ValueError(
            f"reciprocal_rope must have shape {expected_reciprocal_rope}, "
            f"got {tuple(reciprocal_rope.shape)}"
        )

    _require_tensor(
        "workspace_buffer",
        workspace_buffer,
        dtype=torch.int8,
        ndim=1,
        device=device,
        alignment=32,
    )
    _require_tensor(
        "block_tables",
        block_tables,
        dtype=torch.int32,
        ndim=2,
        device=device,
        alignment=4,
    )
    if block_tables.shape[0] != batch or block_tables.shape[1] <= 0:
        raise ValueError(
            f"block_tables must have shape [B,max_pages] with B={batch}, got "
            f"{tuple(block_tables.shape)}"
        )
    _require_tensor(
        "seq_lens",
        seq_lens,
        dtype=torch.int32,
        ndim=1,
        device=device,
        alignment=4,
    )
    if tuple(seq_lens.shape) != (batch,):
        raise ValueError(
            f"seq_lens must have shape ({batch},), got {tuple(seq_lens.shape)}"
        )
    if not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool):
        raise TypeError(f"max_seq_len must be an int, got {type(max_seq_len).__name__}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    table_capacity = block_tables.shape[1] * _PAGE_SIZE
    if max_seq_len > table_capacity:
        raise ValueError(
            f"max_seq_len {max_seq_len} exceeds block-table capacity "
            f"{table_capacity}"
        )
    pages_per_tile = _MMA_QK_TILER[1] // _PAGE_SIZE
    if block_tables.shape[1] % pages_per_tile:
        raise ValueError(
            "block_tables width must be padded to a multiple of "
            f"{pages_per_tile} pages for {_MMA_QK_TILER[1]}-token TMA tiles, "
            f"got {block_tables.shape[1]}"
        )

    if out is not None:
        _require_tensor(
            "out",
            out,
            dtype=torch.bfloat16,
            ndim=4,
            device=device,
            alignment=16,
        )
        if tuple(out.shape) != tuple(query_latent.shape):
            raise ValueError(
                f"out must have shape {tuple(query_latent.shape)}, "
                f"got {tuple(out.shape)}"
            )
    if lse_out is not None:
        _require_tensor(
            "lse_out",
            lse_out,
            dtype=torch.float32,
            ndim=3,
            device=device,
            alignment=4,
        )
        expected_lse = (batch, query_len, num_heads)
        if tuple(lse_out.shape) != expected_lse:
            raise ValueError(
                f"lse_out must have shape {expected_lse}, got {tuple(lse_out.shape)}"
            )
        if not return_lse:
            raise ValueError("lse_out requires return_lse=True")

    writable = tuple(
        tensor
        for tensor in (out, lse_out, workspace_buffer)
        if tensor is not None
    )
    readonly = (
        query_latent,
        query_rope,
        packed_latent,
        reconstruction_scale,
        reciprocal_rope,
        block_tables,
        seq_lens,
    )
    for index, lhs in enumerate(writable):
        for rhs in writable[index + 1 :] + readonly:
            if _overlaps(lhs, rhs):
                raise ValueError("output/workspace tensors must not alias any input")

    compute_capability = torch.cuda.get_device_capability(device)
    if compute_capability != (10, 0):
        raise ValueError(
            "TurboQuant E2M1 MLA decode requires SM100, got compute capability "
            f"{compute_capability[0]}.{compute_capability[1]}"
        )
    return batch, query_len


_COMPILED_KERNELS: dict[tuple, Callable] = {}
_COMPILE_LOCK = threading.Lock()
_MAX_COMPILED_VARIANTS = 128


def _as_cute_tensor(
    tensor: torch.Tensor, dtype, leading_dim: int, assumed_align: int
):
    result = from_dlpack(
        tensor, assumed_align=assumed_align, enable_tvm_ffi=True
    )
    result.element_type = dtype
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _get_compiled_tq_e2m1_kernel(
    *,
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    packed_latent: torch.Tensor,
    reconstruction_scale: torch.Tensor,
    reciprocal_rope: torch.Tensor,
    block_tables: torch.Tensor,
    output: torch.Tensor,
    lse: Optional[torch.Tensor],
    workspace: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
    fold_sq_factor: int,
    enable_pdl: bool,
) -> Callable:
    """Compile/cache a shape-specialized, spill-free K0-G3 owner."""
    key = (
        query_latent.device.index,
        tuple(query_latent.shape),
        tuple(packed_latent.shape),
        tuple(block_tables.shape),
        workspace is not None,
        lse is not None,
        enable_pdl,
    )
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is not None:
        return compiled

    with _COMPILE_LOCK:
        compiled = _COMPILED_KERNELS.get(key)
        if compiled is not None:
            return compiled
        if len(_COMPILED_KERNELS) >= _MAX_COMPILED_VARIANTS:
            raise RuntimeError(
                "TurboQuant E2M1 MLA compile cache reached its 128-variant "
                "safety bound"
            )

        query_len = query_latent.shape[1]
        with torch.cuda.device(query_latent.device):
            kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
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
                # A single next-token query is already aligned to the end of
                # its cache, so its causal bound equals seq_len and the mask is
                # redundant. Keep the real causal owner only for q5 verify.
                is_causal=query_len > 1,
                num_heads=_NUM_HEADS,
                seq_len_q=query_len,
                cp_world=1,
                use_tq_e2m1=True,
                tq_s1_scale_tma=True,
                tq_s1_scale_stages=3,
                tq_s1_k_rope_stages=2,
                tq_s1_packed_p_scale_math=_use_packed_p_scale_math(query_len),
            )
            stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True
            )
            compiled = cute.compile(
                kernel,
                _as_cute_tensor(
                    query_latent, cutlass.Float8E4M3FN, 3, 16
                ),
                _as_cute_tensor(query_rope, cutlass.BFloat16, 3, 16),
                _as_cute_tensor(packed_latent, cutlass.Uint8, 2, 16),
                _as_cute_tensor(reciprocal_rope, cutlass.BFloat16, 2, 16),
                _as_cute_tensor(block_tables, cutlass.Int32, 1, 4),
                _as_cute_tensor(output, cutlass.BFloat16, 3, 16),
                (
                    _as_cute_tensor(lse, cutlass.Float32, 2, 4)
                    if lse is not None
                    else None
                ),
                (
                    _as_cute_tensor(workspace, cutlass.Int8, 0, 32)
                    if workspace is not None
                    else None
                ),
                Int32(1),
                _as_cute_tensor(seq_lens, cutlass.Int32, 0, 4),
                _as_cute_tensor(seq_lens, cutlass.Int32, 0, 4),
                None,
                Float32(1.0),
                Float32(1.0),
                stream,
                enable_pdl,
                _as_cute_tensor(
                    reconstruction_scale, cutlass.BFloat16, 1, 16
                ),
                options="--enable-tvm-ffi --opt-level 3",
            )
        _COMPILED_KERNELS[key] = compiled
        return compiled


def tokenspeed_mla_decode_tq_e2m1(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    packed_latent: torch.Tensor,
    reconstruction_scale: torch.Tensor,
    reciprocal_rope: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    output_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    *,
    enable_pdl: bool = False,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
):
    """Decode MLA attention directly from the packed N10 TurboQuant cache.

    Parameters
    ----------
    query_latent:
        Contiguous E4M3 tensor ``[B, q_len, 8, 512]`` in the signed-WHT basis.
    query_rope:
        Contiguous BF16 post-RoPE tensor ``[B, q_len, 8, 64]``.
    packed_latent:
        Contiguous uint8 tensor ``[num_pages, 32, 256]``. Each byte stores two
        fixed hardware E2M1 codes for the rotated 512-coordinate latent.
    reconstruction_scale:
        Contiguous BF16 per-token scale ``[num_pages, 32]``.
        Values must be finite and positive and must not exceed 14,680,064
        (``224 * 2**16``), the matched writer/reader carrier bound. The
        matched N10 writer satisfies this contract, including a unity scale
        for exact-zero rows.
    reciprocal_rope:
        Contiguous BF16 tensor ``[num_pages, 32, 64]`` containing post-RoPE
        coordinates divided by ``reconstruction_scale``.
    workspace_buffer:
        Contiguous int8 scratch. It must be at least
        ``B * 8 * q_len * split_kv * 513 * 4`` bytes; callers may safely use the
        existing TokenSpeed closed-form SM-count upper bound.
    block_tables:
        Contiguous int32 page table ``[B, max_pages]``. ``max_pages`` must be
        a multiple of four because the reader fetches one 128-token tile at a
        time. Every entry, including tile-padding entries past a sequence's
        logical end, must contain a valid page index; padding may repeat the
        final valid page because those token columns are masked.
    seq_lens:
        Contiguous int32 logical cache lengths ``[B]``.
    max_seq_len:
        Maximum logical sequence length in this launch.
    softmax_scale:
        Scale applied to the combined latent and RoPE score.
    output_scale:
        Scale applied to the attention output.
    out:
        Optional contiguous BF16 output ``[B, q_len, 8, 512]``. Pass this when
        capturing or replaying a CUDA graph to avoid allocation.
    enable_pdl:
        Enable Programmatic Dependent Launch in the compiled kernel.
    return_lse:
        Also return FP32 base-2 log-sum-exp ``[B, q_len, 8]``, matching the
        existing CuTe DSL MLA decode kernel contract.
    lse_out:
        Optional preallocated LSE output. It requires ``return_lse=True``.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        BF16 output in the signed-WHT latent basis. The caller owns the inverse
        WHT before passing the value to consumers that expect the original MLA
        latent basis. When ``return_lse=True``, returns ``(output, log2_lse)``.

    Notes
    -----
    This initial owner is intentionally fail-closed: SM100, page size 32,
    512/64 dimensions, eight local heads, q1 or q5 causal semantics, and
    ``cp_world=1`` only. The q1 specialization elides its redundant causal
    mask; q5 retains it. It does not support DCP or custom/tree masks. Warm a
    shape once before capturing it in a CUDA graph.
    """
    for name, value in (
        ("softmax_scale", softmax_scale),
        ("output_scale", output_scale),
    ):
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real scalar")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value}")
    if not isinstance(enable_pdl, bool):
        raise TypeError("enable_pdl must be a bool")
    if not isinstance(return_lse, bool):
        raise TypeError("return_lse must be a bool")

    batch, query_len = _validate_inputs(
        query_latent,
        query_rope,
        packed_latent,
        reconstruction_scale,
        reciprocal_rope,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_seq_len,
        out,
        lse_out,
        return_lse,
    )

    fold_sq_factor = get_mla_decode_fold_sq_factor(
        _NUM_HEADS, query_len, _MMA_QK_TILER[0]
    )
    effective_heads = _NUM_HEADS * fold_sq_factor
    effective_query_len = query_len // fold_sq_factor
    split_kv = BlackwellMultiHeadLatentAttentionForwardFP8.get_split_kv(
        batch,
        effective_query_len,
        max_seq_len,
        _MMA_QK_TILER,
        get_num_sm(query_latent.device),
        1,
    )
    workspace_size = BlackwellMultiHeadLatentAttentionForwardFP8.get_workspace_size(
        effective_heads,
        effective_query_len,
        _LATENT_DIM,
        batch,
        split_kv,
        cutlass.Float32,
    )
    if workspace_buffer.numel() < workspace_size:
        raise ValueError(
            f"workspace_buffer has {workspace_buffer.numel()} bytes, "
            f"but this launch requires {workspace_size}"
        )
    workspace = None if workspace_size == 0 else workspace_buffer[:workspace_size]

    output = (
        out
        if out is not None
        else torch.empty_like(query_latent, dtype=torch.bfloat16)
    )
    lse = None
    if return_lse:
        lse = (
            lse_out
            if lse_out is not None
            else torch.empty(
                (batch, query_len, _NUM_HEADS),
                dtype=torch.float32,
                device=query_latent.device,
            )
        )

    compiled_kernel = _get_compiled_tq_e2m1_kernel(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        reconstruction_scale=reconstruction_scale,
        reciprocal_rope=reciprocal_rope,
        block_tables=block_tables,
        output=output,
        lse=lse,
        workspace=workspace,
        seq_lens=seq_lens,
        fold_sq_factor=fold_sq_factor,
        enable_pdl=enable_pdl,
    )

    import tvm_ffi

    with torch.cuda.device(query_latent.device), tvm_ffi.use_torch_stream():
        compiled_kernel(
            query_latent,
            query_rope,
            packed_latent,
            reciprocal_rope,
            block_tables,
            output,
            lse,
            workspace,
            Int32(split_kv),
            seq_lens,
            seq_lens,
            None,
            Float32(softmax_scale),
            Float32(output_scale),
            reconstruction_scale,
        )

    if return_lse:
        return output, lse
    return output


__all__ = ["tokenspeed_mla_decode_tq_e2m1"]
