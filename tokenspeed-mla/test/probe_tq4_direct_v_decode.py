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

"""Prove the production direct packed-V address and half-warp decode map."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor
from tokenspeed_mla.tq4_cutedsl import dequantize_tq4_word_to_fp8_shfl

THREADS = 128
PAGE = 32
LATENT = 512


@cute.kernel
def direct_v_decode_kernel(
    packed: cute.Tensor,
    scales: cute.Tensor,
    centroids: cute.Tensor,
    page_table: cute.Tensor,
    output: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    halfwarp_lane = tidx % 16
    centroid_lane = cutlass.Float32(centroids[halfwarp_lane])
    packed_i32_ptr = cute.recast_ptr(packed.iterator, dtype=cutlass.Int32)
    decoded_words = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Int32)
    decoded_fp8 = cute.make_tensor(
        cute.recast_ptr(decoded_words.iterator, dtype=cutlass.Uint8),
        cute.make_layout(8),
    )

    for phase in cutlass.range_constexpr(4):
        for iteration in cutlass.range_constexpr(16):
            linear_word = iteration * THREADS + tidx
            token = linear_word // 16
            latent_word = linear_word % 16
            logical_page = token // PAGE
            page_row = token % PAGE
            physical_page = page_table[logical_page]
            packed_byte = (
                physical_page * packed.stride[0]
                + page_row * packed.stride[1]
                + (phase * 64 + latent_word * 4) * packed.stride[2]
            )
            raw_word = (packed_i32_ptr + packed_byte // 4).load()
            fp8_0, fp8_1 = dequantize_tq4_word_to_fp8_shfl(
                raw_word,
                cutlass.Float32(scales[physical_page, page_row]),
                centroid_lane,
            )
            decoded_words[0] = fp8_0
            decoded_words[1] = fp8_1
            for nibble in cutlass.range_constexpr(8):
                output[token, phase * 128 + latent_word * 8 + nibble] = decoded_fp8[
                    nibble
                ]


@cute.jit
def direct_v_decode(
    packed: cute.Tensor,
    scales: cute.Tensor,
    centroids: cute.Tensor,
    page_table: cute.Tensor,
    output: cute.Tensor,
):
    direct_v_decode_kernel(packed, scales, centroids, page_table, output).launch(
        grid=(1, 1, 1), block=(THREADS, 1, 1), min_blocks_per_mp=1
    )


def fake(dtype: type[cutlass.Numeric], shape: tuple[int, ...], align: int):
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )


def main() -> None:
    compiled = cute.compile(
        direct_v_decode,
        fake(cutlass.Uint8, (4, PAGE, LATENT // 2), 16),
        fake(cutlass.BFloat16, (4, PAGE), 16),
        fake(cutlass.Float32, (16,), 16),
        fake(cutlass.Int32, (4,), 4),
        fake(cutlass.Uint8, (128, LATENT), 16),
        options="--enable-tvm-ffi --opt-level 3",
    )
    generator = torch.Generator(device="cuda").manual_seed(20260723)
    packed = torch.randint(
        0,
        256,
        (4, PAGE, LATENT // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    scales = (
        torch.rand((4, PAGE), device="cuda", dtype=torch.float32, generator=generator)
        * 8.0
        + 16.0
    ).to(torch.bfloat16)
    centroids = torch.linspace(-0.12, 0.11, 16, device="cuda", dtype=torch.float32)
    page_table = torch.randperm(4, device="cuda", dtype=torch.int32)
    output = torch.empty((128, LATENT), device="cuda", dtype=torch.uint8)
    compiled(packed, scales, centroids, page_table, output)
    torch.cuda.synchronize()

    physical = page_table.view(4, 1).expand(4, PAGE).reshape(-1)
    rows = torch.arange(PAGE, device="cuda").repeat(4)
    packed_logical = packed[physical, rows]
    codes = torch.empty((128, LATENT), device="cuda", dtype=torch.uint8)
    codes[:, 0::2] = packed_logical & 0xF
    codes[:, 1::2] = packed_logical >> 4
    expected = (
        (centroids[codes.long()] * scales[physical, rows].float().unsqueeze(1))
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    print("PASS direct_v_decode_exact=True")


if __name__ == "__main__":
    main()
