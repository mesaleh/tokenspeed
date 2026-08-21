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

"""Compact R31 TurboQuant MLA decode integration for Blackwell SM100.

R31 retains the packed E2M1 latent cache used by the N10 owner, but stores RoPE
as an E4M3 high component plus a packed E2M1 residual.  Its query carries four
E4M3 RoPE planes so the kernel can accumulate the high/residual corrections
without expanding a dense BF16 cache row.
"""

import math
import numbers
import threading
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from tokenspeed_mla.mla_decode import _resolve_split_kv_override
from tokenspeed_mla.mla_decode_fp8 import BlackwellMultiHeadLatentAttentionForwardFP8
from tokenspeed_mla.mla_decode_tq_e2m1 import (
    _LATENT_DIM,
    _MMA_PV_TILER,
    _MMA_QK_TILER,
    _NUM_HEADS,
    _PACKED_LATENT_DIM,
    _PAGE_SIZE,
    _ROPE_DIM,
    _SUPPORTED_QUERY_LENGTHS,
    _as_cute_tensor,
    _overlaps,
    _require_tensor,
    _use_early_final_pcor,
    _use_packed_p_scale_math,
)
from tokenspeed_mla.mla_helpers import get_mla_decode_fold_sq_factor
from tokenspeed_mla.utils import get_max_active_clusters, get_num_sm

_QUERY_ROPE_PLANES = 4
_QUERY_ROPE_DIM = _QUERY_ROPE_PLANES * _ROPE_DIM
_PACKED_ROPE_DIM = _ROPE_DIM // 2


def _require_paged_component(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    row_width: int,
    alignment: int,
) -> None:
    """Validate a contiguous-row, page-strided persistent cache component.

    The elastic R31 arena stores every component contiguously inside one
    layer/page envelope. CuTe receives the real page stride through DLPack;
    no gather or dense shadow is required.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    is_scalar_row = row_width == 1 and tensor.ndim == 2
    is_vector_row = tensor.ndim == 3
    if not (is_scalar_row or is_vector_row):
        expected_ndim = "2D" if row_width == 1 else "3D"
        raise ValueError(
            f"{name} must be {expected_ndim}, got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[1] != _PAGE_SIZE or (
        is_vector_row and tensor.shape[2] != row_width
    ):
        expected_shape = (
            f"[num_pages,{_PAGE_SIZE}]"
            if is_scalar_row
            else f"[num_pages,{_PAGE_SIZE},{row_width}]"
        )
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    expected_inner = (1,) if is_scalar_row else (row_width, 1)
    if tuple(tensor.stride()[1:]) != expected_inner:
        raise ValueError(
            f"{name} rows must be contiguous with inner strides "
            f"{expected_inner}, got {tensor.stride()}"
        )
    minimum_page_stride = _PAGE_SIZE * row_width
    if tensor.stride(0) < minimum_page_stride:
        raise ValueError(
            f"{name} page stride must be at least {minimum_page_stride}, "
            f"got {tensor.stride(0)}"
        )
    page_stride_bytes = tensor.stride(0) * tensor.element_size()
    if page_stride_bytes % alignment:
        raise ValueError(
            f"{name} page stride must be {alignment}-byte aligned, "
            f"got {page_stride_bytes} bytes"
        )
    if tensor.numel() and tensor.data_ptr() % alignment:
        raise ValueError(
            f"{name} must be {alignment}-byte aligned, got address "
            f"0x{tensor.data_ptr():x}"
        )


def _validate_r31_inputs(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    packed_latent: torch.Tensor,
    reconstruction_scale: torch.Tensor,
    high_rope: torch.Tensor,
    residual_rope: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    out: Optional[torch.Tensor],
    lse_out: Optional[torch.Tensor],
    return_lse: bool,
    fault_status: Optional[torch.Tensor] = None,
) -> tuple[int, int]:
    if not isinstance(query_latent, torch.Tensor):
        raise TypeError("query_latent must be a torch.Tensor")
    if not query_latent.is_cuda:
        raise ValueError("TurboQuant R31 MLA decode requires CUDA tensors")
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
            "TurboQuant R31 MLA decode supports only q_len 1 or 5, "
            f"got {query_len}"
        )
    if num_heads != _NUM_HEADS or latent_dim != _LATENT_DIM:
        raise ValueError(
            "TurboQuant R31 MLA decode requires query shape "
            f"[B,q_len,{_NUM_HEADS},{_LATENT_DIM}], got "
            f"{tuple(query_latent.shape)}"
        )

    _require_tensor(
        "query_rope",
        query_rope,
        dtype=torch.float8_e4m3fn,
        ndim=4,
        device=device,
        alignment=16,
    )
    expected_query_rope = (batch, query_len, num_heads, _QUERY_ROPE_DIM)
    if tuple(query_rope.shape) != expected_query_rope:
        raise ValueError(
            f"query_rope must have shape {expected_query_rope}, "
            f"got {tuple(query_rope.shape)}"
        )

    _require_paged_component(
        "packed_latent",
        packed_latent,
        dtype=torch.uint8,
        device=device,
        row_width=_PACKED_LATENT_DIM,
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

    _require_paged_component(
        "reconstruction_scale",
        reconstruction_scale,
        dtype=torch.bfloat16,
        device=device,
        row_width=1,
        alignment=16,
    )
    expected_scale = (num_pages, _PAGE_SIZE)
    if tuple(reconstruction_scale.shape) != expected_scale:
        raise ValueError(
            f"reconstruction_scale must have shape {expected_scale}, "
            f"got {tuple(reconstruction_scale.shape)}"
        )

    _require_paged_component(
        "high_rope",
        high_rope,
        dtype=torch.float8_e4m3fn,
        device=device,
        row_width=_ROPE_DIM,
        alignment=16,
    )
    expected_high_rope = (num_pages, _PAGE_SIZE, _ROPE_DIM)
    if tuple(high_rope.shape) != expected_high_rope:
        raise ValueError(
            f"high_rope must have shape {expected_high_rope}, "
            f"got {tuple(high_rope.shape)}"
        )

    _require_paged_component(
        "residual_rope",
        residual_rope,
        dtype=torch.uint8,
        device=device,
        row_width=_PACKED_ROPE_DIM,
        alignment=16,
    )
    expected_residual_rope = (num_pages, _PAGE_SIZE, _PACKED_ROPE_DIM)
    if tuple(residual_rope.shape) != expected_residual_rope:
        raise ValueError(
            f"residual_rope must have shape {expected_residual_rope}, "
            f"got {tuple(residual_rope.shape)}"
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
                f"lse_out must have shape {expected_lse}, "
                f"got {tuple(lse_out.shape)}"
            )
        if not return_lse:
            raise ValueError("lse_out requires return_lse=True")
    if fault_status is not None:
        _require_tensor(
            "fault_status",
            fault_status,
            dtype=torch.int32,
            ndim=1,
            device=device,
            alignment=4,
        )
        if tuple(fault_status.shape) != (1,):
            raise ValueError(
                f"fault_status must have shape (1,), got {tuple(fault_status.shape)}"
            )

    writable = tuple(
        tensor
        for tensor in (out, lse_out, workspace_buffer, fault_status)
        if tensor is not None
    )
    readonly = (
        query_latent,
        query_rope,
        packed_latent,
        reconstruction_scale,
        high_rope,
        residual_rope,
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
            "TurboQuant R31 MLA decode requires SM100, got compute capability "
            f"{compute_capability[0]}.{compute_capability[1]}"
        )
    return batch, query_len


_COMPILED_R31_KERNELS: dict[tuple, Callable] = {}
_COMPILE_R31_LOCK = threading.Lock()
_MAX_COMPILED_R31_VARIANTS = 128


def _get_compiled_tq_r31_kernel(
    *,
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    packed_latent: torch.Tensor,
    reconstruction_scale: torch.Tensor,
    high_rope: torch.Tensor,
    residual_rope: torch.Tensor,
    block_tables: torch.Tensor,
    output: torch.Tensor,
    lse: Optional[torch.Tensor],
    workspace: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
    fold_sq_factor: int,
    causal_mask: bool,
    enable_pdl: bool,
    producer_only: bool,
    physical_split_score: bool = False,
    physical_split_score_lookahead: bool = True,
    physical_split_score_dual_tmem: bool = False,
    fault_status: Optional[torch.Tensor] = None,
) -> Callable:
    """Compile/cache a policy-specialized compact R31 kernel."""
    batch, query_len = query_latent.shape[:2]
    use_packed_p_scale_math = _use_packed_p_scale_math(batch, query_len)
    use_early_final_pcor = _use_early_final_pcor(batch, query_len)
    if physical_split_score_dual_tmem:
        # The B1/q5 packed-P route would require a second q5 module after the
        # generic B8/B5 capture.  Reuse the generic dynamic-batch module: its
        # arithmetic is exact and avoids the late-capture module-state cliff.
        use_packed_p_scale_math = False
        use_early_final_pcor = False

    # Every tensor passed to CuTe below has a runtime-dynamic layout.  The
    # dual-TMEM policy is identical across batch sizes after selecting the
    # generic q5 P-scale route above, so share its compiled kernel across the
    # batch extent.
    # This matches SGLang's descending multi-shape graph capture without
    # retaining redundant batch-specialized modules.
    dynamic_batch_cache = physical_split_score_dual_tmem
    query_latent_shape_key = (
        tuple(query_latent.shape[1:])
        if dynamic_batch_cache
        else tuple(query_latent.shape)
    )
    query_rope_shape_key = (
        tuple(query_rope.shape[1:])
        if dynamic_batch_cache
        else tuple(query_rope.shape)
    )
    packed_latent_shape_key = (
        tuple(packed_latent.shape[1:])
        if dynamic_batch_cache
        else tuple(packed_latent.shape)
    )
    high_rope_shape_key = (
        tuple(high_rope.shape[1:])
        if dynamic_batch_cache
        else tuple(high_rope.shape)
    )
    residual_rope_shape_key = (
        tuple(residual_rope.shape[1:])
        if dynamic_batch_cache
        else tuple(residual_rope.shape)
    )
    block_tables_shape_key = (
        tuple(block_tables.shape[1:])
        if dynamic_batch_cache
        else tuple(block_tables.shape)
    )
    key = (
        query_latent.device.index,
        query_latent_shape_key,
        query_rope_shape_key,
        packed_latent_shape_key,
        tuple(packed_latent.stride()),
        tuple(reconstruction_scale.stride()),
        high_rope_shape_key,
        tuple(high_rope.stride()),
        residual_rope_shape_key,
        tuple(residual_rope.stride()),
        block_tables_shape_key,
        workspace is not None,
        lse is not None,
        causal_mask,
        enable_pdl,
        producer_only,
        physical_split_score,
        physical_split_score_lookahead,
        physical_split_score_dual_tmem,
        fault_status is not None,
        use_packed_p_scale_math,
        use_early_final_pcor,
    )
    compiled = _COMPILED_R31_KERNELS.get(key)
    if compiled is not None:
        return compiled

    with _COMPILE_R31_LOCK:
        compiled = _COMPILED_R31_KERNELS.get(key)
        if compiled is not None:
            return compiled
        if len(_COMPILED_R31_KERNELS) >= _MAX_COMPILED_R31_VARIANTS:
            raise RuntimeError(
                "TurboQuant R31 MLA compile cache reached its 128-variant "
                "safety bound"
            )

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
                is_causal=causal_mask,
                num_heads=_NUM_HEADS,
                seq_len_q=query_len,
                cp_world=1,
                use_tq_e2m1=True,
                use_tq_r31_rope=not physical_split_score,
                use_tq_r31_physical_split_score=physical_split_score,
                tq_r31_physical_split_score_lookahead=(
                    physical_split_score_lookahead
                ),
                tq_r31_physical_split_score_dual_tmem=(
                    physical_split_score_dual_tmem
                ),
                tq_s1_scale_tma=True,
                tq_s1_scale_stages=3,
                tq_s1_k_rope_stages=1,
                tq_s1_packed_p_scale_math=use_packed_p_scale_math,
                tq_s1_early_final_pcor=use_early_final_pcor,
                tq_r31_async_expand=True,
                producer_only=producer_only,
            )
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            compiled = cute.compile(
                kernel,
                _as_cute_tensor(query_latent, cutlass.Float8E4M3FN, 3, 16),
                _as_cute_tensor(query_rope, cutlass.Float8E4M3FN, 3, 16),
                _as_cute_tensor(packed_latent, cutlass.Uint8, 2, 16),
                _as_cute_tensor(high_rope, cutlass.Float8E4M3FN, 2, 16),
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
                _as_cute_tensor(reconstruction_scale, cutlass.BFloat16, 1, 16),
                None,
                _as_cute_tensor(residual_rope, cutlass.Uint8, 2, 16),
                (
                    _as_cute_tensor(fault_status, cutlass.Int32, 0, 4)
                    if fault_status is not None
                    else None
                ),
                options="--enable-tvm-ffi --opt-level 3",
            )
        _COMPILED_R31_KERNELS[key] = compiled
        return compiled


def tokenspeed_mla_decode_tq_r31(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    packed_latent: torch.Tensor,
    reconstruction_scale: torch.Tensor,
    high_rope: torch.Tensor,
    residual_rope: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    output_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    *,
    causal_mask: Optional[bool] = None,
    enable_pdl: bool = False,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
    split_kv_override: Optional[int] = None,
    producer_only: bool = False,
    _physical_split_score: bool = False,
    _physical_split_score_lookahead: bool = True,
    _physical_split_score_dual_tmem: bool = False,
    _physical_split_score_fault_status: Optional[torch.Tensor] = None,
):
    """Decode MLA attention directly from the compact 354-byte R31 cache.

    ``query_latent`` is E4M3 ``[B,q,8,512]``. ``query_rope`` is E4M3
    ``[B,q,8,256]`` containing four consecutive 64-coordinate planes.
    ``packed_latent``, ``reconstruction_scale``, ``high_rope``, and
    ``residual_rope`` respectively have shapes ``[pages,32,256]``,
    ``[pages,32]``, ``[pages,32,64]``, and ``[pages,32,32]``. The remaining
    scheduling, output, and CUDA-graph contracts match
    :func:`tokenspeed_mla_decode_tq_e2m1`. By default q5 is causal and q1 is
    non-causal, preserving the original standalone decode behavior. Pass
    ``causal_mask=False`` when the cache is an entirely historical prefix of a
    segmented q5 attention operation.

    ``producer_only=True`` is the mixed-cache stage-1 contract: it publishes
    split partials to ``workspace_buffer``, requires caller-owned ``out``, does
    not write ``out`` or a final LSE, and returns ``None``.

    The diagnostic dual-TMEM arm requires a caller-owned zero-initialized fault
    word.  Its output is valid only when that word remains zero after stream
    completion; a nonzero sticky value rejects the entire launch.
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
    if causal_mask is not None and not isinstance(causal_mask, bool):
        raise TypeError("causal_mask must be a bool or None")
    if not isinstance(_physical_split_score, bool):
        raise TypeError("_physical_split_score must be a bool")
    if not isinstance(_physical_split_score_lookahead, bool):
        raise TypeError("_physical_split_score_lookahead must be a bool")
    if not isinstance(_physical_split_score_dual_tmem, bool):
        raise TypeError("_physical_split_score_dual_tmem must be a bool")
    if _physical_split_score_lookahead and not _physical_split_score:
        _physical_split_score_lookahead = False
    if _physical_split_score_dual_tmem and not _physical_split_score:
        raise ValueError("dual-TMEM requires physical split-score")
    if _physical_split_score_dual_tmem and _physical_split_score_lookahead:
        raise ValueError("dual-TMEM and lookahead are mutually exclusive")
    if _physical_split_score_dual_tmem and _physical_split_score_fault_status is None:
        raise ValueError("dual-TMEM requires caller-owned fault status")
    if (
        not _physical_split_score_dual_tmem
        and _physical_split_score_fault_status is not None
    ):
        raise ValueError("fault status is reserved for dual-TMEM physical R31")

    batch, query_len = _validate_r31_inputs(
        query_latent,
        query_rope,
        packed_latent,
        reconstruction_scale,
        high_rope,
        residual_rope,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_seq_len,
        out,
        lse_out,
        return_lse,
        _physical_split_score_fault_status,
    )
    resolved_causal_mask = query_len > 1 if causal_mask is None else causal_mask

    fold_sq_factor = get_mla_decode_fold_sq_factor(
        _NUM_HEADS, query_len, _MMA_QK_TILER[0]
    )
    effective_heads = _NUM_HEADS * fold_sq_factor
    effective_query_len = query_len // fold_sq_factor
    inferred_split_kv = BlackwellMultiHeadLatentAttentionForwardFP8.get_split_kv(
        batch,
        effective_query_len,
        max_seq_len,
        _MMA_QK_TILER,
        get_num_sm(query_latent.device),
        1,
    )
    split_kv = _resolve_split_kv_override(
        inferred_split_kv,
        split_kv_override,
        max_seq_len=max_seq_len,
        tile_size=_MMA_QK_TILER[1],
        max_split_kv=64,
    )
    if not isinstance(producer_only, bool):
        raise TypeError("producer_only must be a bool")
    if producer_only:
        if split_kv <= 1:
            raise ValueError("producer-only shared workspace requires split_kv > 1")
        if return_lse:
            raise ValueError("producer-only R31 MLA does not emit a final LSE")
        if out is None:
            raise ValueError("producer-only R31 MLA requires caller-owned out storage")
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
        out if out is not None else torch.empty_like(query_latent, dtype=torch.bfloat16)
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

    compiled_kernel = _get_compiled_tq_r31_kernel(
        query_latent=query_latent,
        query_rope=query_rope,
        packed_latent=packed_latent,
        reconstruction_scale=reconstruction_scale,
        high_rope=high_rope,
        residual_rope=residual_rope,
        block_tables=block_tables,
        output=output,
        lse=lse,
        workspace=workspace,
        seq_lens=seq_lens,
        fold_sq_factor=fold_sq_factor,
        causal_mask=resolved_causal_mask,
        enable_pdl=enable_pdl,
        producer_only=producer_only,
        physical_split_score=_physical_split_score,
        physical_split_score_lookahead=_physical_split_score_lookahead,
        physical_split_score_dual_tmem=_physical_split_score_dual_tmem,
        fault_status=_physical_split_score_fault_status,
    )

    import tvm_ffi

    with torch.cuda.device(query_latent.device), tvm_ffi.use_torch_stream():
        compiled_kernel(
            query_latent,
            query_rope,
            packed_latent,
            high_rope,
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
            None,
            residual_rope,
            _physical_split_score_fault_status,
        )

    if producer_only:
        return None

    if return_lse:
        return output, lse
    return output


__all__ = ["tokenspeed_mla_decode_tq_r31"]
