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

"""Native packed TQ4 MLA decode on the reviewed M=128 reader.

The public serving entry point is an exact alias of the accepted M=128/two-CTA
control.  Keeping the alias direct ensures that promotion changes no launch,
validation, compilation, or kernel semantics.  The M=64/one-CTA specialization
remains a private correctness control.  No dense cache is reconstructed.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from .mla_decode import _get_dummy_tree_mask_tensors, _get_zero_cmask_off_tensor
from .mla_decode_fp8 import BlackwellMultiHeadLatentAttentionForwardFP8
from .mla_helpers import MAX_SPLITS, get_mla_decode_fold_sq_factor
from .tq4_contract import (
    TQ4_LATENT_DIM,
    TQ4_LEVELS,
    TQ4_PACKED_DIM,
    TQ4_ROPE_DIM,
    validate_tq4_decode_inputs,
)
from .utils import get_max_active_clusters, get_num_sm

_M128_QK_TILER = (128, 128)
_M128_PV_TILER = (128, 256)
_M128_CLUSTER = (2, 1, 1)
_M64_QK_TILER = (64, 128)
_M64_PV_TILER = (64, 256)
_M64_CLUSTER = (1, 1, 1)


@functools.cache
def _get_compiled_tq4_m128_control(
    *,
    page_size: int,
    num_heads: int,
    seq_len_q: int,
    fold_sq_factor: int,
    is_persistent: bool,
    is_workspace_size_zero: bool,
    tree_mask_mode: bool,
    fp8_rope: bool,
    tiles_per_split: int,
    use_codebook: bool,
    use_pdl: bool,
    return_lse: bool,
    use_m64: bool = False,
) -> Callable:
    """Compile one explicit packed control specialization."""

    qk_tiler = _M64_QK_TILER if use_m64 else _M128_QK_TILER
    pv_tiler = _M64_PV_TILER if use_m64 else _M128_PV_TILER
    cluster = _M64_CLUSTER if use_m64 else _M128_CLUSTER

    kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=qk_tiler,
        mma_pv_tiler_mn=pv_tiler,
        max_active_clusters=get_max_active_clusters(cluster[0]),
        page_size=page_size,
        skip_correction_threshold=0.0,
        is_persistent=is_persistent,
        is_var_seq=True,
        is_var_split_kv=False,
        fold_sq_factor=fold_sq_factor,
        is_causal=True,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        tree_mask_mode=tree_mask_mode,
        cp_world=1,
        tq4_cache=True,
        tq4_fp8_rope=fp8_rope,
        tq4_tiles_per_split=tiles_per_split,
    )

    sym_batch = cute.sym_int()
    sym_seq_q = cute.sym_int()
    sym_heads = cute.sym_int()
    sym_pages = cute.sym_int()
    sym_page_size = cute.sym_int()
    sym_page_count = cute.sym_int()
    sym_workspace = cute.sym_int()

    query_latent = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (sym_batch, sym_seq_q, sym_heads, TQ4_LATENT_DIM),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    query_rope = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (sym_batch, sym_seq_q, sym_heads, TQ4_ROPE_DIM),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    packed = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_pages, sym_page_size, TQ4_PACKED_DIM),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    rope = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN if fp8_rope else cutlass.BFloat16,
        (sym_pages, sym_page_size, TQ4_ROPE_DIM),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    page_table = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch, sym_page_count),
        stride_order=(1, 0),
        assumed_align=4,
    )
    custom_mask = cute.runtime.make_fake_compact_tensor(
        cutlass.Int8, (cute.sym_int(),), assumed_align=1
    )
    cmask_off = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_batch,), assumed_align=4
    )
    output = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_batch, sym_seq_q, sym_heads, TQ4_LATENT_DIM),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    lse = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (sym_batch, sym_seq_q, sym_heads),
            stride_order=(2, 1, 0),
            assumed_align=4,
        )
        if return_lse
        else None
    )
    workspace = (
        None
        if is_workspace_size_zero
        else cute.runtime.make_fake_compact_tensor(
            cutlass.Int8, (sym_workspace,), assumed_align=32
        )
    )
    seq_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_batch,), assumed_align=4
    )
    scale = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_pages, sym_page_size),
        stride_order=(1, 0),
        assumed_align=16,
    )
    centroids = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (TQ4_LEVELS,), assumed_align=16
    )
    codebook = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (sym_pages, sym_page_size, TQ4_LEVELS),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        if use_codebook
        else None
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel,
        query_latent,
        query_rope,
        packed,
        rope,
        page_table,
        custom_mask,
        cmask_off,
        output,
        lse,
        workspace,
        Int32(1),
        seq_lens,
        seq_lens,
        None,
        Float32(1.0),
        Float32(1.0),
        stream,
        use_pdl,
        scale,
        centroids,
        codebook,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _validate_codebook(
    codebook: Optional[torch.Tensor], scale: torch.Tensor, device: torch.device
) -> None:
    if codebook is None:
        return
    expected = (*scale.shape, TQ4_LEVELS)
    if codebook.shape != expected:
        raise ValueError(f"TQ4 codebook must have shape {expected}, got {codebook.shape}")
    if codebook.dtype != torch.uint8 or not codebook.is_contiguous():
        raise ValueError("TQ4 codebook must be contiguous uint8")
    if codebook.device != device or codebook.data_ptr() % 16:
        raise ValueError("TQ4 codebook must share the cache device and be 16-byte aligned")


def _tree_mask_args(
    query: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    custom_mask: Optional[torch.Tensor],
    cmask_off: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, q_len = query.shape[:2]
    if custom_mask is None:
        if cmask_off is not None:
            raise ValueError("cmask_off requires custom_mask")
        return _get_dummy_tree_mask_tensors(query.device, batch)
    if custom_mask.dtype != torch.bool or custom_mask.device != query.device:
        raise ValueError("custom_mask must be bool on the query device")
    if custom_mask.ndim != 1 or not custom_mask.is_contiguous():
        raise ValueError("custom_mask must be a contiguous flattened 1-D tensor")
    required_mask_elements = q_len * max_seq_len
    if custom_mask.numel() < required_mask_elements:
        raise ValueError(
            "custom_mask is too small for the longest request: "
            f"got {custom_mask.numel()} elements, need at least "
            f"{required_mask_elements}"
        )
    if cmask_off is not None:
        if (
            cmask_off.dtype != torch.int32
            or cmask_off.device != query.device
            or cmask_off.shape != (batch,)
            or not cmask_off.is_contiguous()
        ):
            raise ValueError("cmask_off must be contiguous int32 with one entry per request")
        offsets = cmask_off
    elif batch == 1:
        offsets = _get_zero_cmask_off_tensor(query.device)
    else:
        exclusive = torch.cumsum(seq_lens, dim=0, dtype=torch.int32) - seq_lens
        offsets = q_len * exclusive
    return custom_mask.view(torch.int8), offsets


def _tokenspeed_mla_decode_tq4_m128_control(
    query: torch.Tensor,
    kv_nope_packed: torch.Tensor,
    kv_nope_scale: torch.Tensor,
    kv_rope: torch.Tensor,
    centroids: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    *,
    output_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    custom_mask: Optional[torch.Tensor] = None,
    cmask_off: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
    split_kv_override: Optional[int] = None,
    kv_nope_codebook: Optional[torch.Tensor] = None,
    fp8_rope: bool = False,
    return_lse: bool = False,
    _use_m64: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the no-shadow packed reader through an explicit control tiler."""

    shape, packed, rope = validate_tq4_decode_inputs(
        query,
        kv_nope_packed,
        kv_nope_scale,
        kv_rope,
        centroids,
        workspace_buffer,
        block_tables,
        seq_lens,
        out,
        fp8_rope=fp8_rope,
    )
    if centroids.dtype != torch.float32:
        raise ValueError("native TQ4 control requires persistent float32 centroids")
    _validate_codebook(kv_nope_codebook, kv_nope_scale, query.device)
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    q_latent = query[..., :TQ4_LATENT_DIM]
    q_rope = query[..., TQ4_LATENT_DIM:]
    mask, offsets = _tree_mask_args(
        query, seq_lens, max_seq_len, custom_mask, cmask_off
    )
    qk_tiler = _M64_QK_TILER if _use_m64 else _M128_QK_TILER
    cluster = _M64_CLUSTER if _use_m64 else _M128_CLUSTER
    fold = get_mla_decode_fold_sq_factor(
        shape.num_heads, shape.query_length, qk_tiler[0]
    )
    heads_eff = shape.num_heads * fold
    q_len_eff = shape.query_length // fold
    tile_count = (max_seq_len + qk_tiler[1] - 1) // qk_tiler[1]
    required_pages = tile_count * (qk_tiler[1] // shape.page_size)
    if block_tables.shape[1] < required_pages:
        raise ValueError(
            f"TQ4 control needs {required_pages} page columns, "
            f"got {block_tables.shape[1]}"
        )
    split_kv = BlackwellMultiHeadLatentAttentionForwardFP8.get_split_kv(
        shape.batch_size,
        q_len_eff,
        max_seq_len,
        qk_tiler,
        get_num_sm(query.device),
        cluster[0],
    )
    if split_kv_override is not None:
        split_kv = split_kv_override
    max_split_kv = min(tile_count, MAX_SPLITS)
    if not 1 <= split_kv <= max_split_kv:
        raise ValueError(
            f"split_kv must be in [1, {max_split_kv}], got {split_kv}"
        )
    tiles_per_split = (tile_count + split_kv - 1) // split_kv
    workspace_size = BlackwellMultiHeadLatentAttentionForwardFP8.get_workspace_size(
        heads_eff,
        q_len_eff,
        TQ4_LATENT_DIM,
        shape.batch_size,
        split_kv,
        cutlass.Float32,
    )
    if workspace_buffer.numel() < workspace_size:
        raise ValueError(
            f"workspace requires {workspace_size} bytes, got {workspace_buffer.numel()}"
        )
    workspace = None if workspace_size == 0 else workspace_buffer[:workspace_size]
    output = (
        out
        if out is not None
        else torch.empty(
            (shape.batch_size, shape.query_length, shape.num_heads, TQ4_LATENT_DIM),
            dtype=torch.bfloat16,
            device=query.device,
        )
    )
    lse = (
        torch.zeros(
            (shape.batch_size, shape.query_length, shape.num_heads),
            dtype=torch.float32,
            device=query.device,
        )
        if return_lse
        else None
    )
    compiled = _get_compiled_tq4_m128_control(
        page_size=shape.page_size,
        num_heads=shape.num_heads,
        seq_len_q=shape.query_length,
        fold_sq_factor=fold,
        is_persistent=False,
        is_workspace_size_zero=workspace is None,
        tree_mask_mode=custom_mask is not None,
        fp8_rope=fp8_rope,
        tiles_per_split=tiles_per_split,
        use_codebook=kv_nope_codebook is not None,
        use_pdl=enable_pdl,
        return_lse=return_lse,
        use_m64=_use_m64,
    )

    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        compiled(
            q_latent,
            q_rope,
            packed,
            rope,
            block_tables,
            mask,
            offsets,
            output,
            lse,
            workspace,
            Int32(split_kv),
            seq_lens,
            seq_lens,
            None,
            Float32(softmax_scale),
            Float32(output_scale),
            kv_nope_scale,
            centroids,
            kv_nope_codebook,
        )
    return (output, lse) if return_lse else output


def _tokenspeed_mla_decode_tq4_m64_control(*args, **kwargs):
    """Run the private serial M=64 correctness specialization."""
    kwargs["_use_m64"] = True
    return _tokenspeed_mla_decode_tq4_m128_control(*args, **kwargs)


# Serving must execute the exact R1-v4-reviewed M=128 implementation.  A direct
# alias preserves its signature and avoids another Python frame in the decode
# hot path while leaving the M64 diagnostic unavailable through the public API.
tokenspeed_mla_decode_tq4 = _tokenspeed_mla_decode_tq4_m128_control

__all__ = ["tokenspeed_mla_decode_tq4"]
