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

"""TurboQuant-4 MLA cache contract and test-only dense oracle.

The serving kernel consumes the three cache tensors independently.  This module
does not provide a dense-cache compatibility path: the reconstruction helpers
are deliberately named ``reference`` and are intended only for correctness
tests and isolated benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

TQ4_BITS = 4
TQ4_LEVELS = 1 << TQ4_BITS
TQ4_LATENT_DIM = 512
TQ4_PACKED_DIM = TQ4_LATENT_DIM // 2
TQ4_ROPE_DIM = 64
TQ4_QUERY_DIM = TQ4_LATENT_DIM + TQ4_ROPE_DIM


@dataclass(frozen=True)
class TQ4DecodeShape:
    batch_size: int
    query_length: int
    num_heads: int
    num_pages: int
    page_size: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalize_paged_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize ``[pages, 1, page, dim]`` to ``[pages, page, dim]``."""

    if tensor.dim() == 4:
        _require(
            tensor.shape[1] == 1,
            f"{name} 4-D form requires a singleton head dimension, got "
            f"shape={tuple(tensor.shape)}",
        )
        tensor = tensor.squeeze(1)
    _require(
        tensor.dim() == 3,
        f"{name} must be 3-D or singleton-head 4-D, got shape={tuple(tensor.shape)}",
    )
    return tensor


def validate_tq4_decode_inputs(
    query: torch.Tensor,
    kv_nope_packed: torch.Tensor,
    kv_nope_scale: torch.Tensor,
    kv_rope: torch.Tensor,
    centroids: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    kv_lora_rank: int = TQ4_LATENT_DIM,
    qk_rope_head_dim: int = TQ4_ROPE_DIM,
    fp8_rope: bool = False,
    require_cuda: bool = True,
) -> tuple[TQ4DecodeShape, torch.Tensor, torch.Tensor]:
    """Validate and normalize the native packed MLA decode ABI.

    Returns the static shape contract plus normalized packed/RoPE tensors.  No
    tensor is copied or converted.  Value-range checks are intentionally left to
    focused tests because synchronizing device tensors in a decode hot path is
    inadmissible.
    """

    _require(kv_lora_rank == TQ4_LATENT_DIM, "TQ4 MLA requires kv_lora_rank=512")
    _require(
        qk_rope_head_dim == TQ4_ROPE_DIM,
        "TQ4 MLA requires qk_rope_head_dim=64",
    )
    _require(query.dim() == 4, f"query must be [B, q_len, H, 576], got {query.shape}")
    batch_size, query_length, num_heads, query_dim = query.shape
    _require(
        query_dim == TQ4_QUERY_DIM,
        f"query last dimension must be 576, got {query_dim}",
    )
    _require(
        query.dtype == torch.float8_e4m3fn,
        f"query must be FP8 E4M3, got {query.dtype}",
    )
    _require(query.stride(-1) == 1, "query must have a stride-1 inner dimension")
    _require(
        batch_size > 0 and query_length > 0 and num_heads > 0,
        "query dimensions must be positive",
    )
    _require(
        num_heads in (8, 16),
        "initial TQ4 MLA kernel requires 8 or 16 TP-local query heads, "
        f"got {num_heads}",
    )
    _require(
        1 <= query_length <= 5,
        f"TQ4 MLA kernel supports q_len in [1, 5], got {query_length}",
    )

    packed = _normalize_paged_tensor(kv_nope_packed, "kv_nope_packed")
    rope = _normalize_paged_tensor(kv_rope, "kv_rope")
    num_pages, page_size, packed_dim = packed.shape
    _require(num_pages > 0, "kv_nope_packed must contain at least one page")
    _require(
        packed_dim == TQ4_PACKED_DIM,
        f"packed latent dimension must be 256, got {packed_dim}",
    )
    _require(
        packed.dtype == torch.uint8,
        f"kv_nope_packed must be uint8, got {packed.dtype}",
    )
    _require(packed.is_contiguous(), "kv_nope_packed must be contiguous")
    _require(
        page_size == 32,
        f"initial TQ4 MLA kernel requires page_size=32, got {page_size}",
    )
    _require(
        rope.shape == (num_pages, page_size, TQ4_ROPE_DIM),
        f"kv_rope shape must be {(num_pages, page_size, TQ4_ROPE_DIM)}, got {tuple(rope.shape)}",
    )
    _require(
        rope.dtype == (torch.float8_e4m3fn if fp8_rope else torch.bfloat16),
        "kv_rope must be "
        f"{'FP8 E4M3' if fp8_rope else 'bfloat16'}, got {rope.dtype}",
    )
    _require(rope.is_contiguous(), "kv_rope must be contiguous")
    _require(
        packed.data_ptr() % 16 == 0,
        "kv_nope_packed must be 16-byte aligned",
    )
    _require(rope.data_ptr() % 16 == 0, "kv_rope must be 16-byte aligned")

    _require(
        kv_nope_scale.shape == (num_pages, page_size),
        f"kv_nope_scale shape must be {(num_pages, page_size)}, got {tuple(kv_nope_scale.shape)}",
    )
    _require(
        kv_nope_scale.dtype == torch.bfloat16,
        f"kv_nope_scale must be bfloat16, got {kv_nope_scale.dtype}",
    )
    _require(kv_nope_scale.is_contiguous(), "kv_nope_scale must be contiguous")

    _require(
        centroids.shape == (TQ4_LEVELS,),
        f"centroids must have shape (16,), got {centroids.shape}",
    )
    _require(
        centroids.dtype in (torch.bfloat16, torch.float32),
        f"centroids must be bfloat16 or float32, got {centroids.dtype}",
    )
    _require(centroids.is_contiguous(), "centroids must be contiguous")

    _require(workspace_buffer.dim() == 1, "workspace_buffer must be 1-D")
    _require(
        workspace_buffer.dtype == torch.int8,
        f"workspace_buffer must be int8, got {workspace_buffer.dtype}",
    )
    _require(workspace_buffer.is_contiguous(), "workspace_buffer must be contiguous")
    _require(
        block_tables.dim() == 2 and block_tables.shape[0] == batch_size,
        f"block_tables must be [B, max_pages] with B={batch_size}, got {tuple(block_tables.shape)}",
    )
    _require(
        block_tables.shape[1] > 0, "block_tables must contain at least one page column"
    )
    _require(
        block_tables.dtype == torch.int32,
        f"block_tables must be int32, got {block_tables.dtype}",
    )
    _require(block_tables.is_contiguous(), "block_tables must be contiguous")
    _require(
        seq_lens.shape == (batch_size,),
        f"seq_lens must have shape ({batch_size},), got {seq_lens.shape}",
    )
    _require(
        seq_lens.dtype == torch.int32, f"seq_lens must be int32, got {seq_lens.dtype}"
    )
    _require(seq_lens.is_contiguous(), "seq_lens must be contiguous")

    tensors = (
        query,
        packed,
        kv_nope_scale,
        rope,
        centroids,
        workspace_buffer,
        block_tables,
        seq_lens,
    )
    device = query.device
    _require(
        all(tensor.device == device for tensor in tensors),
        "all TQ4 decode tensors must share one device",
    )
    if require_cuda:
        _require(
            device.type == "cuda", f"TQ4 native decode requires CUDA, got {device}"
        )

    if out is not None:
        _require(out.device == device, "out must be on the query device")
        _require(
            out.shape == (batch_size, query_length, num_heads, TQ4_LATENT_DIM),
            "out must be [B, q_len, H, 512]",
        )
        _require(out.dtype == torch.bfloat16, f"out must be bfloat16, got {out.dtype}")
        _require(out.is_contiguous(), "out must be contiguous")

    return (
        TQ4DecodeShape(batch_size, query_length, num_heads, num_pages, page_size),
        packed,
        rope,
    )


def unpack_tq4_indices_reference(kv_nope_packed: torch.Tensor) -> torch.Tensor:
    """Test-only expansion of canonical low-even/high-odd TQ4 indices."""

    _require(
        kv_nope_packed.dtype == torch.uint8, "packed reference input must be uint8"
    )
    _require(
        kv_nope_packed.shape[-1] == TQ4_PACKED_DIM,
        "packed reference input must end in 256 bytes",
    )
    indices = torch.empty(
        (*kv_nope_packed.shape[:-1], TQ4_LATENT_DIM),
        dtype=torch.uint8,
        device=kv_nope_packed.device,
    )
    indices[..., 0::2] = kv_nope_packed & 0x0F
    indices[..., 1::2] = kv_nope_packed >> 4
    return indices


def dequantize_tq4_reference(
    kv_nope_packed: torch.Tensor,
    kv_nope_scale: torch.Tensor,
    centroids: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Test-only dense reconstruction of rotated-domain TQ4 latent rows."""

    _require(centroids.shape == (TQ4_LEVELS,), "centroids must have shape (16,)")
    _require(
        kv_nope_scale.shape == kv_nope_packed.shape[:-1],
        "scale shape must match packed token dimensions",
    )
    _require(
        centroids.device == kv_nope_packed.device
        and kv_nope_scale.device == kv_nope_packed.device,
        "packed values, scales, and centroids must share one device",
    )
    indices = unpack_tq4_indices_reference(kv_nope_packed).to(torch.long)
    values = centroids.to(torch.float32)[indices]
    values = values * kv_nope_scale.to(torch.float32).unsqueeze(-1)
    return values.to(dtype)
