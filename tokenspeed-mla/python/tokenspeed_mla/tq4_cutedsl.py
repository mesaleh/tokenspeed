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

"""CuTe DSL register conversion primitives for canonical TurboQuant-4."""

import cutlass
import cutlass.cute as cute

from .fmha_helpers import cvt_f32x4_to_f8x4_pack_i32


@cute.jit
def dequantize_tq4_word_to_fp8(
    packed_word: cutlass.Int32,
    scale: cutlass.Float32,
    centroids: cute.Tensor,
):
    """Convert four canonical packed bytes to two packed FP8 words.

    Each input nibble is an unsigned Lloyd-codebook index. The least
    significant nibble represents the even latent coordinate.
    """

    values0 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
    values1 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
    values0[0] = cutlass.Float32(centroids[(packed_word >> 0) & 0xF]) * scale
    values0[1] = cutlass.Float32(centroids[(packed_word >> 4) & 0xF]) * scale
    values0[2] = cutlass.Float32(centroids[(packed_word >> 8) & 0xF]) * scale
    values0[3] = cutlass.Float32(centroids[(packed_word >> 12) & 0xF]) * scale
    values1[0] = cutlass.Float32(centroids[(packed_word >> 16) & 0xF]) * scale
    values1[1] = cutlass.Float32(centroids[(packed_word >> 20) & 0xF]) * scale
    values1[2] = cutlass.Float32(centroids[(packed_word >> 24) & 0xF]) * scale
    values1[3] = cutlass.Float32(centroids[(packed_word >> 28) & 0xF]) * scale
    return (
        cvt_f32x4_to_f8x4_pack_i32(values0, cutlass.Float8E4M3FN),
        cvt_f32x4_to_f8x4_pack_i32(values1, cutlass.Float8E4M3FN),
    )
