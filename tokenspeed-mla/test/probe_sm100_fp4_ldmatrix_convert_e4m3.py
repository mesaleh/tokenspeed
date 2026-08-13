"""Qualify exact register-only E2M1-to-E4M3 conversion on SM100.

This is the A17-N8-R1-C0 conversion micro-gate.  It extends the accepted D0
compact-FP4 TMA -> padded narrow-LdMatrix path with two independently lowered
conversion arms:

* ``swar`` maps four byte-contained E2M1 codes to four exact E4M3 bytes with
  branch-free integer operations; and
* ``cvt`` repacks the four nibbles, uses native E2M1-to-F16 conversion, then
  native F16-to-E4M3 conversion.

Both arms publish bounded diagnostic words for an independent CPU bit oracle.
There is no decoded global cache, TMEM population, PV MMA, or endpoint claim.
The source requires the isolated CUTLASS DSL 4.7 environment recorded by R1.
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims

_ROWS = 16
_K = 128
_PACKED_ROW_BYTES = _K // 2
_PADDED_ROW_BYTES = _K
_GROUPS = _K // 16
_WARP_SIZE = 32
_WORDS_PER_LANE = 2
_E2M1_TO_E4M3 = torch.tensor(
    [
        0x00,
        0x30,
        0x38,
        0x3C,
        0x40,
        0x44,
        0x48,
        0x4C,
        0x80,
        0xB0,
        0xB8,
        0xBC,
        0xC0,
        0xC4,
        0xC8,
        0xCC,
    ],
    dtype=torch.uint8,
)


@dsl_user_op
def convert_e2m1_bytes_to_e4m3_swar(
    codes: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Convert four low-nibble E2M1 byte codes to four exact E4M3 bytes."""

    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Uint32(codes).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b32 mag, t1, t2, any, nz, zero_bit, one_bit;\n\t"
            ".reg .b32 zero_mask, one_mask, scaled, out, corr0, corr1, corr;\n\t"
            ".reg .b32 sign;\n\t"
            "and.b32 mag, $1, 0x07070707;\n\t"
            "shr.u32 t1, mag, 1;\n\t"
            "shr.u32 t2, mag, 2;\n\t"
            "or.b32 any, mag, t1;\n\t"
            "or.b32 any, any, t2;\n\t"
            "and.b32 nz, any, 0x01010101;\n\t"
            "xor.b32 zero_bit, nz, 0x01010101;\n\t"
            "or.b32 any, t1, t2;\n\t"
            "not.b32 any, any;\n\t"
            "and.b32 one_bit, mag, any;\n\t"
            "and.b32 one_bit, one_bit, 0x01010101;\n\t"
            "mul.lo.u32 zero_mask, zero_bit, 0xff;\n\t"
            "mul.lo.u32 one_mask, one_bit, 0xff;\n\t"
            "shl.b32 scaled, mag, 2;\n\t"
            "add.u32 out, scaled, 0x30303030;\n\t"
            "and.b32 corr0, zero_mask, 0x30303030;\n\t"
            "and.b32 corr1, one_mask, 0x04040404;\n\t"
            "or.b32 corr, corr0, corr1;\n\t"
            "sub.u32 out, out, corr;\n\t"
            "and.b32 sign, $1, 0x08080808;\n\t"
            "shl.b32 sign, sign, 4;\n\t"
            "or.b32 out, out, sign;\n\t"
            "mov.b32 $0, out;\n\t"
            "}\n",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def convert_e2m1_bytes_to_e4m3_cvt(
    codes: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Convert four byte codes through native E2M1->F16->E4M3 instructions."""

    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Uint32(codes).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b32 packed, tmp, h0, h1, out;\n\t"
            ".reg .b8 b0, b1, b2, b3;\n\t"
            ".reg .b16 e0, e1;\n\t"
            "and.b32 packed, $1, 0x0000000f;\n\t"
            "shr.u32 tmp, $1, 4;\n\t"
            "and.b32 tmp, tmp, 0x000000f0;\n\t"
            "or.b32 packed, packed, tmp;\n\t"
            "shr.u32 tmp, $1, 8;\n\t"
            "and.b32 tmp, tmp, 0x00000f00;\n\t"
            "or.b32 packed, packed, tmp;\n\t"
            "shr.u32 tmp, $1, 12;\n\t"
            "and.b32 tmp, tmp, 0x0000f000;\n\t"
            "or.b32 packed, packed, tmp;\n\t"
            "mov.b32 {b0, b1, b2, b3}, packed;\n\t"
            "cvt.rn.f16x2.e2m1x2 h0, b0;\n\t"
            "cvt.rn.f16x2.e2m1x2 h1, b1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n\t"
            "mov.b32 out, {e0, e1};\n\t"
            "mov.b32 $0, out;\n\t"
            "}\n",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.kernel
def kernel(
    tma_src_desc: cutlass.GridConstant[cuda.TensorMap],
    raw_dst: cute.Tensor,
    swar_dst: cute.Tensor,
    cvt_dst: cute.Tensor,
    POISON: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass.Array(
        cutlass.Int8,
        _ROWS * _PADDED_ROW_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )

    if prims.elect_sync():
        prims.mbarrier_init(mbar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(mbar, tma_src_desc.global_tx_bytes())
        prims.cp_async_bulk_tensor_shared_cta_global(
            smem,
            tma_src_desc.get_ptr(),
            (cutlass.Int32(0), cutlass.Int32(0)),
            mbar,
        )

    while not prims.mbarrier_try_wait_parity(
        mbar, cutlass.Int32(0), time_limit=10_000_000
    ):
        pass
    prims.barrier_cta_sync(0)

    for i in cutlass.range_constexpr((_ROWS * (_PADDED_ROW_BYTES // 2)) // _WARP_SIZE):
        flat = tidx + i * _WARP_SIZE
        row = flat // (_PADDED_ROW_BYTES // 2)
        within_row = flat % (_PADDED_ROW_BYTES // 2)
        group = within_row // 8
        byte = within_row % 8
        smem[row * _PADDED_ROW_BYTES + group * 16 + 8 + byte] = cutlass.Int8(POISON)
    prims.barrier_cta_sync(0)

    lane = tidx % _WARP_SIZE
    for group in cutlass.range_constexpr(_GROUPS):
        row_start = (lane % _ROWS) * _PADDED_ROW_BYTES + group * 16
        regs = prims.ldmatrix(
            smem.data_ptr() + row_start,
            _WORDS_PER_LANE,
            prims.MMALayout.COL,
            shape=prims.LoadShape.M16N16,
            src_format=prims.LoadSrcFormat.B4X16_P64,
        )
        for word in cutlass.range_constexpr(_WORDS_PER_LANE):
            raw = regs[word].to(cutlass.Uint32)
            raw_dst[group, lane, word] = raw
            swar_dst[group, lane, word] = convert_e2m1_bytes_to_e4m3_swar(raw)
            cvt_dst[group, lane, word] = convert_e2m1_bytes_to_e4m3_cvt(raw)


@cute.jit
def host(
    src: cute.Tensor,
    raw_dst: cute.Tensor,
    swar_dst: cute.Tensor,
    cvt_dst: cute.Tensor,
    POISON: cutlass.Constexpr[int],
) -> None:
    tma_src = cuda.create_tensor_map_tiled(
        src.iterator.toint(),
        cutlass.Float4E2M1FN,
        global_dims=[_K, _ROWS],
        global_strides=[_PACKED_ROW_BYTES // 16],
        box_dims=[_K, _ROWS],
        swizzle=cuda.TensorMapSwizzle.none,
    )
    kernel(tma_src, raw_dst, swar_dst, cvt_dst, POISON).launch(
        grid=(1, 1, 1),
        block=(_WARP_SIZE, 1, 1),
    )


@lru_cache(maxsize=None)
def compile_gate(poison: int):
    fake_src = make_fake_compact_tensor(
        cutlass.Uint8,
        (_ROWS, _PACKED_ROW_BYTES),
        stride_order=(1, 0),
        assumed_align=32,
    )
    fake_dst = make_fake_compact_tensor(
        cutlass.Uint32,
        (_GROUPS, _WARP_SIZE, _WORDS_PER_LANE),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    return cute.compile(
        host,
        fake_src,
        fake_dst,
        fake_dst,
        fake_dst,
        poison,
        options="--enable-tvm-ffi",
    )


def _codes() -> torch.Tensor:
    row = torch.arange(_ROWS, dtype=torch.int64).view(_ROWS, 1)
    col = torch.arange(_K, dtype=torch.int64).view(1, _K)
    return ((row * 5 + col * 3 + (col // 16) * 7) & 0xF).to(torch.uint8)


def _packed_source(codes: torch.Tensor) -> torch.Tensor:
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous().cuda()


def _expected_raw(codes: torch.Tensor) -> torch.Tensor:
    expected = torch.empty((_GROUPS, _WARP_SIZE, _WORDS_PER_LANE), dtype=torch.uint32)
    for group in range(_GROUPS):
        for lane in range(_WARP_SIZE):
            output_row = lane // 4
            output_col = (lane % 4) * 4
            for word in range(_WORDS_PER_LANE):
                latent = group * 16 + output_row + word * 8
                value = 0
                for byte in range(4):
                    token = output_col + byte
                    value |= int(codes[token, latent]) << (8 * byte)
                expected[group, lane, word] = value
    return expected


def _expected_e4m3(raw: torch.Tensor) -> torch.Tensor:
    raw_bytes = raw.view(torch.uint8)
    converted = _E2M1_TO_E4M3[raw_bytes.to(torch.int64)].contiguous()
    return converted.view(torch.uint32).reshape_as(raw)


def run(poison: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    codes = _codes()
    src = _packed_source(codes)
    shape = (_GROUPS, _WARP_SIZE, _WORDS_PER_LANE)
    outputs = [
        torch.full(shape, 0xDEADBEEF, dtype=torch.uint32, device="cuda")
        for _ in range(3)
    ]
    compiled = compile_gate(poison)
    compiled(src, *outputs)
    torch.cuda.synchronize()
    raw_expected = _expected_raw(codes)
    return *(output.cpu() for output in outputs), _expected_e4m3(raw_expected)


def verify() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("an SM100 GPU is required")

    first = run(0x00)
    second = run(0x5A)
    for observed in first[1:3]:
        torch.testing.assert_close(observed, first[3], rtol=0, atol=0)
    for observed in second[1:3]:
        torch.testing.assert_close(observed, second[3], rtol=0, atol=0)
    for index in range(3):
        torch.testing.assert_close(first[index], second[index], rtol=0, atol=0)
        assert not torch.any(first[index] == 0xDEADBEEF), "output retained poison"

    observed_bytes = first[1].view(torch.uint8)
    assert set(int(x) for x in torch.unique(first[0].view(torch.uint8))) == set(
        range(16)
    )
    assert set(int(x) for x in torch.unique(observed_bytes)) == set(
        int(x) for x in _E2M1_TO_E4M3
    )
    print(
        "PASS exact_e2m1_to_e4m3=True arms=swar,cvt codes=16 "
        "padding_poison_independent=True decoded_global_cache=False",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    verify()
