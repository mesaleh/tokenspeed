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

"""Raw-word correctness probe for the vectorized TQ4 FP8 codebook lookup."""

from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor

from tokenspeed_mla.tq4_cutedsl import lookup_tq4_word_from_fp8_codebook_prmt


@cute.kernel
def lookup_kernel(
    packed: cute.Tensor,
    codebook: cute.Tensor,
    output: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    row = bidx * 8 + tidx // 16
    word = tidx % 16
    halfwarp_lane = tidx % 16
    codebook_i32 = cute.recast_ptr(codebook.iterator, dtype=cutlass.Int32)
    codebook_word = cutlass.Int32(0)
    if halfwarp_lane < 4:
        codebook_word = (codebook_i32 + row * 4 + halfwarp_lane).load()
    output[row, word, 0], output[row, word, 1] = (
        lookup_tq4_word_from_fp8_codebook_prmt(
            packed[row, word], codebook_word
        )
    )


@cute.jit
def lookup(
    packed: cute.Tensor,
    codebook: cute.Tensor,
    output: cute.Tensor,
):
    lookup_kernel(packed, codebook, output).launch(
        grid=(packed.shape[0] // 8, 1, 1), block=(128, 1, 1)
    )


def fake(dtype: type[cutlass.Numeric], shape: tuple[int | cute.Int, ...]):
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=16,
    )


def main() -> None:
    rows = cute.sym_int()
    compiled = cute.compile(
        lookup,
        fake(cutlass.Int32, (rows, 16)),
        fake(cutlass.Uint8, (rows, 16)),
        fake(cutlass.Int32, (rows, 16, 2)),
        options="--enable-tvm-ffi --opt-level 2",
    )

    torch.manual_seed(20260723)
    num_rows = 4096
    packed = torch.randint(
        -(1 << 31),
        1 << 31,
        (num_rows, 16),
        device="cuda",
        dtype=torch.int32,
    )
    codebook = torch.randint(
        0, 256, (num_rows, 16), device="cuda", dtype=torch.uint8
    )
    output = torch.empty(
        (num_rows, 16, 2), device="cuda", dtype=torch.int32
    )
    compiled(packed, codebook, output)
    torch.cuda.synchronize()

    shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)
    indices = ((packed[..., None] >> shifts) & 0xF).long()
    expected_bytes = torch.gather(
        codebook[:, None, :].expand(-1, 16, -1), 2, indices
    )
    expected = expected_bytes.contiguous().view(torch.int32)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    print(
        f"PASS rows={num_rows} words={num_rows * 16} "
        f"lookups={num_rows * 16 * 8}"
    )


if __name__ == "__main__":
    main()
