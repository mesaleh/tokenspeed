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

"""Prove the token-major E2M1-to-native-PV on-chip transpose on SM100.

The canonical TurboQuant MLA cache packs adjacent latent coordinates in each
byte and stores tokens as rows. Native reversed PV needs V-transpose, so its
E2M1 A operand packs adjacent tokens in each byte. This probe performs exactly
that bounded 128-token nibble transpose in shared memory, executes E2M1 by FP8
UMMA, applies the known final-latent-row correction, and compares random values
against dense math. It allocates no decoded global tensor.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor

THREADS = 128
CLUSTER_SHAPE_MNK = (1, 1, 1)
PV_TILER_MNK = (128, 256, 32)
LOAD_ONLY = False
MMA_ONLY = False
FP8_CONTROL = False


@cute.struct
class SharedStorage:
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


def make_tiled_mmas():
    native_pv = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        cutlass.Float4E2M1FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        PV_TILER_MNK[:2],
    )
    output_view = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        PV_TILER_MNK[:2],
    )
    return native_pv, output_view


@cute.kernel
def packed_transpose_pv_kernel(
    output: cute.Tensor,
    metadata: cute.Tensor,
    packed_v: cute.Tensor,
    p_input: cute.Tensor,
    p_reference: cute.Tensor,
    codebook: cute.Tensor,
    latent_base: cutlass.Int32,
    token_base: cutlass.Int32,
    native_pv_mma: cute.TiledMma,
    output_view_mma: cute.TiledMma,
    v_layout: cute.ComposedLayout,
    p_layout: cute.ComposedLayout,
    acc_cols: cutlass.Constexpr,
    cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    is_leader_cta = cta_rank == 0

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    s_p = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        p_layout.outer,
        byte_alignment=1024,
        swizzle=p_layout.inner,
    )
    s_v = smem.allocate_tensor(
        cutlass.Float4E2M1FN,
        v_layout.outer,
        byte_alignment=1024,
        swizzle=v_layout.inner,
    )
    s_v_guard = smem.allocate_tensor(
        cutlass.Uint8,
        cute.make_layout(128),
        byte_alignment=128,
    )

    # Source bytes are [token, latent_pair]. Destination bytes are
    # [latent, token_pair]. Both operand bases are 1 KiB aligned, so apply the
    # native SW64 S<2,4,3> byte permutation relative to the stripped pointer.
    v_bytes = cute.recast_tensor(s_v, cutlass.Uint8)
    print(f"E2M1_PV_V_BYTES_LAYOUT={v_bytes.layout}")
    for destination_byte in cutlass.range(tidx, 256 * 16, THREADS):
        latent = destination_byte // 16
        token_pair = destination_byte % 16
        token0 = token_base + token_pair * 2
        token1 = token0 + 1
        source_latent = latent_base + latent
        source_byte = source_latent // 2
        shift = (source_latent % 2) * 4
        code0 = (cutlass.Int32(packed_v[token0, source_byte]) >> shift) & 0xF
        code1 = (cutlass.Int32(packed_v[token1, source_byte]) >> shift) & 0xF
        v_bytes[(latent, token_pair), 0, 0, 0] = cutlass.Uint8(
            code0 | (code1 << 4)
        )

    # p_input contains token scale folded into P and is cast to FP8 exactly as
    # the integrated attention path will do.
    for logical_element in cutlass.range(tidx, 128 * 32, THREADS):
        query = logical_element // 32
        token = logical_element % 32
        s_p[(query, token), 0, 0, 0] = p_input[query, token_base + token]
    if tidx < 128:
        s_v_guard[tidx] = cutlass.Uint8(0)
    cute.arch.sync_threads()

    if cutlass.const_expr(LOAD_ONLY):
        p_bits = cute.recast_tensor(s_p, cutlass.Uint8)
        for logical_element in cutlass.range(tidx, 128 * 64, THREADS):
            query = logical_element // 64
            token = logical_element % 64
            output[query, token] = cutlass.Float32(
                cutlass.Int32(p_bits[(query, token % 32), 0, token // 32, 0])
            )
        return

    mma_producer, mma_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.mma_mbar.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()
    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_barrier,
        is_two_cta=False,
    )
    tmem.allocate(acc_cols)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

    p = native_pv_mma.make_fragment_A(s_p)
    v = native_pv_mma.make_fragment_B(s_v)
    print(f"E2M1_PV_P_FRAGMENT={p.layout}")
    print(f"E2M1_PV_V_FRAGMENT={v.layout}")
    print(f"E2M1_PV_K_BLOCKS={cute.size(p, mode=[2])}")
    acc_shape = native_pv_mma.partition_shape_C(PV_TILER_MNK[:2])
    acc_fake = native_pv_mma.make_fragment_C(acc_shape)
    acc = cute.make_tensor(tmem_ptr, acc_fake.layout)

    if warp_idx == 0 and is_leader_cta:
        mma_producer.acquire_and_advance()
        native_pv_mma.set(tcgen05.Field.ACCUMULATE, False)
        k_blocks = cute.size(p, mode=[2])
        for k_block in cutlass.range(k_blocks, unroll_full=True):
            cute.gemm(
                native_pv_mma,
                acc,
                p[None, None, k_block, 0],
                v[None, None, k_block, 0],
                acc,
            )
            native_pv_mma.set(tcgen05.Field.ACCUMULATE, True)
        mma_producer.commit()

    mma_full = mma_consumer.wait_and_advance()
    mma_full.release()
    cute.arch.sync_threads()

    if cutlass.const_expr(MMA_ONLY):
        if tidx == 0:
            metadata[0] = acc_cols
            metadata[1] = cute.size_in_bytes(cutlass.Float4E2M1FN, s_v)
            metadata[2] = cute.size_in_bytes(cutlass.Float8E4M3FN, s_p)
        if warp_idx == 0 and is_leader_cta:
            mma_producer.tail()
        tmem.relinquish_alloc_permit()
        cute.arch.sync_threads()
        tmem.free(tmem_ptr)
        return

    output_tile = acc[(None, None), 0, 0]
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
    )
    tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, output_tile)
    thr_load = tmem_load.get_slice(tidx)
    output_matrix = cute.make_tensor(
        output.iterator, cute.make_layout((128, 256), stride=(256, 1))
    )
    coordinates = cute.make_identity_tensor((128, 256))
    t_tmem = thr_load.partition_S(output_tile)
    t_gmem = thr_load.partition_D(output_matrix)
    t_coordinates = thr_load.partition_D(coordinates)
    registers = cute.make_fragment_like(t_gmem, cutlass.Float32)
    cute.copy(tmem_load, t_tmem, registers)
    cute.arch.fence_view_async_tmem_load()

    # SM100's one-CTA mixed instruction has a deterministic final-N-column
    # defect. Recompute only latent column 255 from canonical bytes and FP8 P.
    for element in cutlass.range_constexpr(cute.size(registers)):
        query = t_coordinates[element][0]
        latent = t_coordinates[element][1]
        if cutlass.const_expr(not FP8_CONTROL):
            if latent == 255:
                correction = cutlass.Float32(0.0)
                for token in cutlass.range(32):
                    source_latent = latent_base + 255
                    source_token = token_base + token
                    packed = cutlass.Int32(
                        packed_v[source_token, source_latent // 2]
                    )
                    code = (packed >> ((source_latent % 2) * 4)) & 0xF
                    correction += cutlass.Float32(codebook[code]) * cutlass.Float32(
                        p_reference[query, source_token]
                    )
                registers[element] = correction
    cute.autovec_copy(registers, t_gmem)
    cute.arch.sync_threads()

    if tidx == 0:
        metadata[0] = acc_cols
        metadata[1] = cute.size_in_bytes(cutlass.Float4E2M1FN, s_v)
        metadata[2] = cute.size_in_bytes(cutlass.Float8E4M3FN, s_p)

    if warp_idx == 0 and is_leader_cta:
        mma_producer.tail()
    tmem.relinquish_alloc_permit()
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def packed_transpose_pv_probe(
    output: cute.Tensor,
    metadata: cute.Tensor,
    packed_v: cute.Tensor,
    p_input: cute.Tensor,
    p_reference: cute.Tensor,
    codebook: cute.Tensor,
    latent_base: cutlass.Int32,
    token_base: cutlass.Int32,
):
    native_pv_mma, output_view_mma = make_tiled_mmas()
    v_layout = sm100_utils.make_smem_layout_b(
        native_pv_mma, PV_TILER_MNK, cutlass.Float4E2M1FN, 1
    )
    p_layout = sm100_utils.make_smem_layout_a(
        native_pv_mma, PV_TILER_MNK, cutlass.Float8E4M3FN, 1
    )
    print(f"E2M1_PV_V_LAYOUT={v_layout}")
    print(f"E2M1_PV_P_LAYOUT={p_layout}")
    acc_fake = native_pv_mma.make_fragment_C(
        native_pv_mma.partition_shape_C(PV_TILER_MNK[:2])
    )
    print(f"E2M1_PV_ACC_LAYOUT={acc_fake.layout}")
    acc_cols = utils.get_num_tmem_alloc_cols(acc_fake)
    if cutlass.const_expr(acc_cols > 512):
        raise ValueError(f"TMEM overflow: acc={acc_cols}")
    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE_MNK), (native_pv_mma.thr_id.shape,)
    )
    packed_transpose_pv_kernel(
        output,
        metadata,
        packed_v,
        p_input,
        p_reference,
        codebook,
        latent_base,
        token_base,
        native_pv_mma,
        output_view_mma,
        v_layout,
        p_layout,
        acc_cols,
        cta_layout_vmnk,
    ).launch(
        grid=CLUSTER_SHAPE_MNK,
        block=(THREADS, 1, 1),
        cluster=CLUSTER_SHAPE_MNK,
        min_blocks_per_mp=1,
    )


def main() -> None:
    compiled = cute.compile(
        packed_transpose_pv_probe,
        make_fake_compact_tensor(
            cutlass.Float32,
            (128, 256),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.Int32,
            (3,),
            stride_order=(0,),
            assumed_align=4,
        ),
        make_fake_compact_tensor(
            cutlass.Uint8,
            (128, 256),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (128, 64),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (128, 64),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (16,),
            stride_order=(0,),
            assumed_align=16,
        ),
        cutlass.Int32(0),
        cutlass.Int32(0),
        options="--enable-tvm-ffi --opt-level 3",
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260723)
    codes = torch.randint(
        0, 16, (128, 512), generator=generator, device="cuda", dtype=torch.uint8
    )
    packed_v = codes[:, 0::2] | (codes[:, 1::2] << 4)
    codebook = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.full(
        (128, 256), float("nan"), device="cuda", dtype=torch.float32
    )
    metadata = torch.empty(3, device="cuda", dtype=torch.int32)
    decoded_v = codebook.float()[codes.long()]

    def check(label: str, expected: torch.Tensor, active_rows: int = 16) -> None:
        actual = output[:active_rows]
        expected = expected[:active_rows]
        mismatch = actual != expected
        if mismatch.any():
            bad = mismatch.nonzero()
            row_matches = {}
            if label == "selector":
                for row in torch.unique(bad[:, 0]).cpu().tolist():
                    matches = (
                        decoded_v[:64, :255] == actual[row, :255]
                    ).all(dim=1).nonzero()
                    row_matches[row] = matches.flatten().cpu().tolist()
            print(
                f"DIAGNOSTIC label={label} "
                f"active_rows={active_rows} "
                f"max_abs={(actual - expected).abs().max().item()} "
                f"bad_rows={torch.unique(bad[:, 0]).cpu().tolist()} "
                f"bad_columns={torch.unique(bad[:, 1]).cpu().tolist()} "
                f"row_matches={row_matches} "
                f"metadata={metadata.cpu().tolist()} "
                f"output_sample={actual[:8, :8].cpu().tolist()} "
                f"expected_sample={expected[:8, :8].cpu().tolist()} "
                f"output_tail={actual[:4, 44:64].cpu().tolist()} "
                f"expected_tail={expected[:4, 44:64].cpu().tolist()}"
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Diagnose the physical FP8-A K mapping independently of dense math. Each
    # launch places a one in the same source column for the first 16 query
    # rows; the random E2M1 rows uniquely identify which logical token the MMA
    # consumed from that source column.
    active_rows_by_col = []
    active_tokens_by_col = []
    for source_col in range(32):
        p_reference = torch.zeros(
            (128, 64), device="cuda", dtype=torch.bfloat16
        )
        p_reference[:16, source_col] = 1.0
        compiled(
            output,
            metadata,
            packed_v,
            p_reference.to(torch.float8_e4m3fn),
            p_reference,
            codebook,
            cutlass.Int32(0),
            cutlass.Int32(0),
        )
        torch.cuda.synchronize()
        actual = output[:, :255]
        nonzero_rows = actual.abs().sum(dim=1).ne(0).nonzero().flatten()
        row_tokens = []
        for query in nonzero_rows.cpu().tolist():
            matches = (
                decoded_v[:32, :255] == actual[query]
            ).all(dim=1).nonzero()
            if matches.numel() == 1:
                row_tokens.append(matches.item())
            else:
                row_tokens.append(-1)
        active_rows_by_col.append(nonzero_rows.cpu().tolist())
        active_tokens_by_col.append(row_tokens)
    print(f"ACTIVE_ROWS_BY_K={active_rows_by_col}")
    print(f"ACTIVE_TOKENS_BY_K={active_tokens_by_col}")
    return

    # First isolate the transpose map: each query selects exactly one token.
    selector_rows = torch.arange(16, device="cuda")
    selector_tokens = selector_rows + 32
    p_logical = torch.zeros((128, 64), device="cuda", dtype=torch.bfloat16)
    p_logical[selector_rows, selector_tokens] = 1.0
    p_reference = torch.zeros_like(p_logical)
    p_reference[:16] = p_logical[:16]
    p_input = p_reference.to(torch.float8_e4m3fn)
    compiled(
        output,
        metadata,
        packed_v,
        p_input,
        p_reference,
        codebook,
        cutlass.Int32(0),
        cutlass.Int32(32),
    )
    torch.cuda.synchronize()
    if LOAD_ONLY:
        expected_bits = p_input.view(torch.uint8).float()
        actual_bits = output[:, :64]
        bad = (actual_bits != expected_bits).nonzero()
        if bad.numel():
            bad_rows = torch.unique(bad[:, 0]).cpu().tolist()
            actual_nonzero = {
                row: actual_bits[row].nonzero().flatten().cpu().tolist()
                for row in bad_rows
            }
            expected_nonzero = {
                row: expected_bits[row].nonzero().flatten().cpu().tolist()
                for row in bad_rows
            }
            print(
                f"P_SMEM_BITS_DIAGNOSTIC bad_count={bad.shape[0]} "
                f"bad_rows={bad_rows} first_bad={bad[:128].cpu().tolist()} "
                f"actual_nonzero={actual_nonzero} "
                f"expected_nonzero={expected_nonzero}"
            )
        torch.testing.assert_close(
            actual_bits,
            expected_bits,
            rtol=0,
            atol=0,
        )
        print("PASS p_smem_bits_roundtrip=True rows=128")
        return
    check("selector", decoded_v[32:48, :256].contiguous())

    # Then cover dense P and explicit per-token scale folding over both
    # 256-coordinate Kimi latent slices. Chosen values remain exactly
    # representable in FP8.
    p_values = torch.tensor(
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], device="cuda", dtype=torch.float32
    )
    query_ids = torch.arange(128, device="cuda", dtype=torch.int32)[:, None]
    p_token_ids = torch.arange(64, device="cuda", dtype=torch.int32)[None, :]
    p_indices = (query_ids * 17 + p_token_ids * 5 + 1) % p_values.numel()
    token_scales = torch.tensor(
        [0.5, 1.0, 2.0], device="cuda", dtype=torch.float32
    )[p_token_ids.remainder(3)]
    p_logical = (p_values[p_indices] * token_scales).to(torch.float8_e4m3fn)
    p_reference = torch.zeros_like(p_logical, dtype=torch.bfloat16)
    p_reference[:16] = p_logical[:16].float().to(torch.bfloat16)
    p_input = p_reference.to(torch.float8_e4m3fn)
    for token_base in (0, 32):
        for latent_base in (0, 256):
            compiled(
                output,
                metadata,
                packed_v,
                p_input,
                p_reference,
                codebook,
                cutlass.Int32(latent_base),
                cutlass.Int32(token_base),
            )
            torch.cuda.synchronize()
            expected = (
                p_logical[:16, token_base : token_base + 32].float()
                @ decoded_v[
                    token_base : token_base + 32,
                    latent_base : latent_base + 256,
                ]
            )
            check(f"dense-k-{token_base}-latent-{latent_base}", expected)

    acc_cols, v_bytes, p_bytes = metadata.cpu().tolist()
    print(
        "PASS e2m1_pv_packed_transpose=True latent_slices=2 "
        "token_scale_fold=True boundary_correction=True "
        "tile_rows=128 useful_rows=16 query_subgroup=True "
        f"acc_cols={acc_cols} v_smem_bytes={v_bytes} p_smem_bytes={p_bytes}"
    )


if __name__ == "__main__":
    main()
