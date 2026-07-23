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

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import dsl_user_op

from .fmha_helpers import cvt_f32x4_to_f8x4_pack_i32


@dsl_user_op
def copy_bulk_smem_to_dsmem(
    destination: Pointer,
    source: Pointer,
    byte_count: cutlass.Int32,
    completion_barrier: Pointer,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Copy one aligned SMEM segment to a remote CTA and signal its barrier."""
    i32_type = ir.IntegerType.get_signless(32)
    destination_address = llvm.ptrtoint(
        i32_type, destination.llvm_ptr, loc=loc, ip=ip
    )
    source_address = llvm.ptrtoint(i32_type, source.llvm_ptr, loc=loc, ip=ip)
    barrier_address = llvm.ptrtoint(
        i32_type, completion_barrier.llvm_ptr, loc=loc, ip=ip
    )
    llvm.inline_asm(
        res=None,
        operands_=[
            destination_address,
            source_address,
            cutlass.Int32(byte_count).ir_value(loc=loc, ip=ip),
            barrier_address,
        ],
        asm_string=(
            "cp.async.bulk.shared::cluster.shared::cta."
            "mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];"
        ),
        constraints="r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


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


@cute.jit
def dequantize_tq4_word_to_fp8_shfl(
    packed_word: cutlass.Int32,
    scale: cutlass.Float32,
    centroid_lane: cutlass.Float32,
):
    """Convert one word using each scale-uniform half warp as a codebook."""
    values0 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
    values1 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
    scaled_centroid_lane = centroid_lane * scale
    # PTX shfl.idx: segment mask 0x10 and clamp 0x0F isolate both half warps.
    values0[0] = cute.arch.shuffle_sync(
        scaled_centroid_lane, (packed_word >> 0) & 0xF, mask_and_clamp=0x100F
    )
    values0[1] = cute.arch.shuffle_sync(
        scaled_centroid_lane, (packed_word >> 4) & 0xF, mask_and_clamp=0x100F
    )
    values0[2] = cute.arch.shuffle_sync(
        scaled_centroid_lane, (packed_word >> 8) & 0xF, mask_and_clamp=0x100F
    )
    values0[3] = cute.arch.shuffle_sync(
        scaled_centroid_lane, (packed_word >> 12) & 0xF, mask_and_clamp=0x100F
    )
    values1[0] = cute.arch.shuffle_sync(
        scaled_centroid_lane, (packed_word >> 16) & 0xF, mask_and_clamp=0x100F
    )
    values1[1] = cute.arch.shuffle_sync(
        scaled_centroid_lane, (packed_word >> 20) & 0xF, mask_and_clamp=0x100F
    )
    values1[2] = cute.arch.shuffle_sync(
        scaled_centroid_lane, (packed_word >> 24) & 0xF, mask_and_clamp=0x100F
    )
    values1[3] = cute.arch.shuffle_sync(
        scaled_centroid_lane, (packed_word >> 28) & 0xF, mask_and_clamp=0x100F
    )
    return (
        cvt_f32x4_to_f8x4_pack_i32(values0, cutlass.Float8E4M3FN),
        cvt_f32x4_to_f8x4_pack_i32(values1, cutlass.Float8E4M3FN),
    )


@dsl_user_op
def lookup_tq4x4_from_fp8_codebook_prmt(
    lut0: cutlass.Int32,
    lut1: cutlass.Int32,
    lut2: cutlass.Int32,
    lut3: cutlass.Int32,
    lut_indices: cutlass.Int32,
    table_select: cutlass.Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Map four TQ4 indices through a 16-byte register LUT."""
    i32_type = ir.IntegerType.get_signless(32)
    return llvm.inline_asm(
        i32_type,
        [
            cutlass.Int32(lut0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(lut1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(lut2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(lut3).ir_value(loc=loc, ip=ip),
            cutlass.Int32(lut_indices).ir_value(loc=loc, ip=ip),
            cutlass.Int32(table_select).ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .b32 lower, upper;\n\t"
            "prmt.b32 lower, $1, $2, $5;\n\t"
            "prmt.b32 upper, $3, $4, $5;\n\t"
            "prmt.b32 $0, lower, upper, $6;\n\t"
            "}"
        ),
        "=r,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def lookup_tq4_word_from_fp8_codebook_prmt(
    packed_word: cutlass.Int32,
    codebook_word_lane: cutlass.Int32,
):
    """Expand eight TQ4 indices with four broadcasts and six byte permutes."""
    lut0 = cute.arch.shuffle_sync(
        codebook_word_lane, 0, mask_and_clamp=0x100F
    )
    lut1 = cute.arch.shuffle_sync(
        codebook_word_lane, 1, mask_and_clamp=0x100F
    )
    lut2 = cute.arch.shuffle_sync(
        codebook_word_lane, 2, mask_and_clamp=0x100F
    )
    lut3 = cute.arch.shuffle_sync(
        codebook_word_lane, 3, mask_and_clamp=0x100F
    )
    lut_indices = packed_word & 0x77777777
    table_select = ((packed_word & 0x88888888) >> 1) | 0x32103210
    return (
        cutlass.Int32(
            lookup_tq4x4_from_fp8_codebook_prmt(
                lut0, lut1, lut2, lut3, lut_indices, table_select
            )
        ),
        cutlass.Int32(
            lookup_tq4x4_from_fp8_codebook_prmt(
                lut0,
                lut1,
                lut2,
                lut3,
                lut_indices >> 16,
                table_select >> 16,
            )
        ),
    )
