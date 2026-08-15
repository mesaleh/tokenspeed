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

"""Runtime gate for D0's direct page-32 E2M1 reader.

The same logical cache is materialized once in identity physical-page order and
once under a non-affine page permutation.  Outputs must be bit-identical, which
proves that QK, RoPE, PV, and the block-table lookup agree on page ownership.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
    make_ptr,
)

MODULE_PATH = Path(__file__).parent / "tokenspeed_mla" / "mla_decode_e2m1.py"
MODULE_SPEC = importlib.util.spec_from_file_location("mla_decode_e2m1", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load D0 module from {MODULE_PATH}")
d0 = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(d0)
PHYSICAL_PAGE_COUNT = 37


def fake(dtype: type[cutlass.Numeric], shape: tuple[int, ...], align: int):
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )


def compile_reader(query_len: int, physical_page_count: int):
    return cute.compile(
        d0.ownership_probe,
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        fake(cutlass.BFloat16, (d0.SCORE_ROWS, d0.LATENT_K), 16),
        fake(cutlass.Float32, (d0.SCORE_ROWS,), 16),
        fake(cutlass.BFloat16, (physical_page_count, d0.PAGE_SIZE), 16),
        fake(cutlass.Int32, (d0.MAX_PAGES,), 16),
        cutlass.Int32(640),
        cutlass.Float32(d0.PROBE_SOFTMAX_SCALE_LOG2),
        query_len * d0.NUM_HEADS,
        query_len,
        physical_page_count,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        make_fake_stream(),
        options="--enable-tvm-ffi --opt-level 3",
    )


def to_cute_tensor_and_storage(source: torch.Tensor, dtype):
    source_cuda = source.cuda().contiguous()
    result, storage = cutlass_torch.cute_tensor_like(
        source,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    result = cutlass_torch.convert_cute_tensor(
        source_cuda,
        result,
        dtype,
        is_dynamic_layout=True,
    )
    return result, storage


def to_cute_tensor(source: torch.Tensor, dtype):
    return to_cute_tensor_and_storage(source, dtype)[0]


def make_inputs(query_len: int):
    active_rows = query_len * d0.NUM_HEADS
    row = torch.arange(active_rows, dtype=torch.int64).view(active_rows, 1)
    token = torch.arange(d0.TOKENS, dtype=torch.int64).view(d0.TOKENS, 1)
    latent = torch.arange(d0.LATENT_K, dtype=torch.int64).view(1, d0.LATENT_K)
    rope = torch.arange(d0.ROPE_K, dtype=torch.int64).view(1, d0.ROPE_K)
    query = (
        (((row + 1) * (latent + 3) * 17 + row * 37 + latent * 19) % 257) % 3
    ).float()
    query = (query - 1.0) * 0.5
    key_base = (
        (((token + 1) * (latent + 5) * 23 + token * 41 + latent * 29) % 263) % 3
    ).float()
    key_base = (key_base - 1.0) * 0.5
    key = key_base.unsqueeze(0) * torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0]).view(
        d0.TILES, 1, 1
    )
    rope_query = ((((row + 3) * (rope + 1) * 11 + row * 13) % 127) % 3).float()
    rope_query = (rope_query - 1.0) * 0.5
    rope_key_base = ((((token + 5) * (rope + 7) * 19 + token * 17) % 131) % 3).float()
    rope_key_base = (rope_key_base - 1.0) * 0.5
    rope_key = rope_key_base.unsqueeze(0) * torch.tensor(
        [1.0, 0.5, 2.0, 1.0, 1.0]
    ).view(d0.TILES, 1, 1)
    tile = torch.arange(d0.TILES, dtype=torch.int64).view(d0.TILES, 1)
    token_row = torch.arange(d0.TOKENS, dtype=torch.int64).view(1, d0.TOKENS)
    scale = (
        (0.75 + ((tile * 7 + token_row * 3) % 17).float() / 32.0)
        * torch.tensor([1.0, 2.0, 0.5, 8.0, 4.0]).view(d0.TILES, 1)
    ).to(torch.bfloat16)
    return query, key, rope_query, rope_key, scale


def physical_pages(
    logical: torch.Tensor,
    permutation: torch.Tensor,
    physical_page_count: int,
) -> torch.Tensor:
    logical_pages = logical.reshape(d0.MAX_PAGES, d0.PAGE_SIZE, logical.shape[-1])
    result = torch.zeros(
        (physical_page_count, d0.PAGE_SIZE, logical.shape[-1]),
        dtype=logical.dtype,
    )
    result[permutation] = logical_pages
    return result


def run(reader, inputs, permutation: torch.Tensor, cache_len: int):
    query, key, rope_query, rope_key, scale = inputs
    combined_query = torch.cat((query, rope_query), dim=-1)
    query_cute = to_cute_tensor(combined_query.unsqueeze(0), cutlass.Float8E4M3FN)
    key_cute = to_cute_tensor(
        physical_pages(key, permutation, PHYSICAL_PAGE_COUNT),
        cutlass.Float4E2M1FN,
    )
    rope_key_cute = to_cute_tensor(
        physical_pages(rope_key, permutation, PHYSICAL_PAGE_COUNT),
        cutlass.Float8E4M3FN,
    )
    native_dummy = to_cute_tensor(
        torch.zeros(16, dtype=torch.float32), cutlass.Float8E4M3FN
    )
    scale_pages = physical_pages(
        scale.reshape(-1, 1), permutation, PHYSICAL_PAGE_COUNT
    ).squeeze(-1)
    output = torch.full(
        (d0.SCORE_ROWS, d0.LATENT_K),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    lse = torch.full(
        (d0.SCORE_ROWS,),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    reader(
        query_cute.iterator,
        key_cute.iterator,
        native_dummy.iterator,
        query_cute.iterator + d0.LATENT_K,
        rope_key_cute.iterator,
        output,
        lse,
        scale_pages.cuda().contiguous(),
        permutation.to(device="cuda", dtype=torch.int32).contiguous(),
        cutlass.Int32(cache_len),
        cutlass.Float32(d0.PROBE_SOFTMAX_SCALE_LOG2),
        stream,
    )
    torch.cuda.synchronize()
    return output.cpu(), lse.cpu()


def run_public(
    inputs,
    permutation: torch.Tensor,
    query_len: int,
    cache_len: int,
    graph_replays: int = 0,
):
    query, key, rope_query, rope_key, scale = inputs
    combined_query = torch.cat((query, rope_query), dim=-1).reshape(
        1, query_len, d0.NUM_HEADS, d0.QUERY_DIM
    )
    _, query_storage = to_cute_tensor_and_storage(combined_query, cutlass.Float8E4M3FN)
    _, packed_storage = to_cute_tensor_and_storage(
        physical_pages(key, permutation, PHYSICAL_PAGE_COUNT),
        cutlass.Float4E2M1FN,
    )
    packed_cache = (
        packed_storage.view(torch.uint8)
        .flatten()[: PHYSICAL_PAGE_COUNT * d0.PAGE_SIZE * (d0.LATENT_K // 2)]
        .reshape(PHYSICAL_PAGE_COUNT, d0.PAGE_SIZE, d0.LATENT_K // 2)
    )
    _, rope_storage = to_cute_tensor_and_storage(
        physical_pages(rope_key, permutation, PHYSICAL_PAGE_COUNT),
        cutlass.Float8E4M3FN,
    )
    scale_pages = (
        physical_pages(scale.reshape(-1, 1), permutation, PHYSICAL_PAGE_COUNT)
        .squeeze(-1)
        .cuda()
        .contiguous()
    )
    page_table = permutation.to(device="cuda", dtype=torch.int32)[None].contiguous()
    output = torch.full(
        (1, query_len, d0.NUM_HEADS, d0.LATENT_K),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    lse = torch.full(
        (1, query_len, d0.NUM_HEADS),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )

    def launch(trusted_page_table: bool):
        d0.tokenspeed_mla_decode_e2m1(
            query_storage,
            packed_cache,
            scale_pages,
            rope_storage,
            page_table,
            cache_len,
            d0.PROBE_SOFTMAX_SCALE_LOG2 / math.log2(math.e),
            output,
            lse,
            trusted_page_table=trusted_page_table,
        )

    launch(False)
    torch.cuda.synchronize()
    if graph_replays:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(True)
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        for _ in range(graph_replays):
            graph.replay()
        torch.cuda.synchronize()
        allocated_delta = torch.cuda.memory_allocated() - allocated_before
        reserved_delta = torch.cuda.memory_reserved() - reserved_before
        if allocated_delta != 0 or reserved_delta != 0:
            raise AssertionError(
                "graph replay allocated memory: "
                f"allocated_delta={allocated_delta} reserved_delta={reserved_delta}"
            )
        print(
            "PASS_D0_D_GRAPH_REPLAY "
            f"replays={graph_replays} allocated_delta={allocated_delta} "
            f"reserved_delta={reserved_delta}",
            flush=True,
        )
    return output.reshape(-1, d0.LATENT_K).cpu(), lse.flatten().cpu()


def reference_result(inputs, query_len: int, cache_len: int):
    query, key, rope_query, rope_key, scale = inputs
    key_flat = key.reshape(-1, d0.LATENT_K).float()
    rope_key_flat = rope_key.reshape(-1, d0.ROPE_K).float()
    scale_flat = scale.reshape(-1).float()
    scores = query.float() @ key_flat.T
    scores = scores * scale_flat.unsqueeze(0)
    scores += rope_query.float() @ rope_key_flat.T
    carriers = []
    previous_carrier = torch.tensor(1.0, dtype=torch.float32)
    for tile in range(d0.TILES):
        valid_in_tile = max(0, min(d0.TOKENS, cache_len - tile * d0.TOKENS))
        if valid_in_tile:
            max_scale = scale[tile, :valid_in_tile].float().max()
            carrier = torch.tensor(2.0**-16, dtype=torch.float32)
            while max_scale > 224.0 * carrier:
                carrier *= 2.0
        else:
            carrier = previous_carrier
        carriers.append(carrier)
        previous_carrier = carrier
    output = torch.zeros_like(query, dtype=torch.float32)
    lse = torch.full((query.shape[0],), float("-inf"), dtype=torch.float32)
    for row in range(query.shape[0]):
        query_token = row // d0.NUM_HEADS
        causal_bound = max(
            0,
            min(cache_len, cache_len - (query_len - 1) + query_token),
        )
        if causal_bound == 0:
            continue
        row_max = torch.tensor(-1.0e6, dtype=torch.float32)
        row_sum = torch.tensor(0.0, dtype=torch.float32)
        accumulator = torch.zeros(d0.LATENT_K, dtype=torch.float32)
        previous_carrier = carriers[0]
        for tile in range(d0.TILES):
            begin = tile * d0.TOKENS
            end = min(begin + d0.TOKENS, causal_bound)
            carrier = carriers[tile]
            if end > begin:
                tile_scores = scores[row, begin:end]
                tile_max = tile_scores.max()
                row_max_new = torch.maximum(row_max, tile_max)
                prior_correction = torch.exp2(
                    (row_max - row_max_new) * d0.PROBE_SOFTMAX_SCALE_LOG2
                )
                if tile == 0:
                    prior_correction = torch.tensor(1.0)
                probability = torch.exp2(
                    (tile_scores - row_max_new) * d0.PROBE_SOFTMAX_SCALE_LOG2
                )
                p_quantized = (
                    (probability * scale_flat[begin:end] / carrier)
                    .to(torch.float8_e4m3fn)
                    .float()
                )
                tile_output = p_quantized @ key_flat[begin:end]
                accumulator = (
                    accumulator * prior_correction * previous_carrier / carrier
                    + tile_output
                )
                row_sum = prior_correction * row_sum + probability.sum()
                row_max = row_max_new
            elif tile > 0:
                accumulator = accumulator * previous_carrier / carrier
            previous_carrier = carrier
        output[row] = accumulator * previous_carrier / row_sum
        lse[row] = torch.log2(row_sum) + d0.PROBE_SOFTMAX_SCALE_LOG2 * row_max
    return output, lse


def main() -> None:
    torch.manual_seed(20260814)
    identity = torch.arange(d0.MAX_PAGES, dtype=torch.int64)
    permutation = torch.tensor(
        [31, 1, 36, 4, 22, 0, 26, 9, 3, 24, 6, 35, 20, 2, 25, 8, 23, 5, 27, 21],
        dtype=torch.int64,
    )
    cases = (0, 1, 31, 32, 33, 127, 128, 129, 511, 513, 639, 640)
    max_output_error = 0.0
    for query_len in (1, 5):
        reader = compile_reader(query_len, PHYSICAL_PAGE_COUNT)
        inputs = make_inputs(query_len)
        active_rows = query_len * d0.NUM_HEADS
        for cache_len in cases:
            identity_output, identity_lse = run(reader, inputs, identity, cache_len)
            permuted_output, permuted_lse = run(reader, inputs, permutation, cache_len)
            torch.testing.assert_close(
                permuted_output[:active_rows].view(torch.int16),
                identity_output[:active_rows].view(torch.int16),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                permuted_lse[:active_rows],
                identity_lse[:active_rows],
                rtol=0,
                atol=0,
            )
            actual_output = identity_output
            actual_lse = identity_lse
            expected_output, expected_lse = reference_result(
                inputs, query_len, cache_len
            )
            output_error = (actual_output[:active_rows].float() - expected_output).abs()
            allowance = expected_output.abs() * 0.01 + 0.02
            if not torch.all(output_error <= allowance):
                raise AssertionError(
                    "D0 output exceeds the dynamic causal envelope: "
                    f"q={query_len} cache={cache_len} "
                    f"max_error={output_error.max().item():.8f} "
                    f"max_allowance={allowance.max().item():.8f}"
                )
            finite = torch.isfinite(expected_lse)
            torch.testing.assert_close(
                actual_lse[:active_rows][finite],
                expected_lse[finite],
                rtol=2.0e-6,
                atol=2.0e-5,
            )
            torch.testing.assert_close(
                actual_lse[:active_rows][~finite],
                expected_lse[~finite],
                rtol=0,
                atol=0,
            )
            if not torch.isnan(actual_output[active_rows:]).all():
                raise AssertionError(
                    f"inactive output rows were written for q={query_len} "
                    f"cache={cache_len}"
                )
            if not torch.isnan(actual_lse[active_rows:]).all():
                raise AssertionError(
                    f"inactive LSE rows were written for q={query_len} "
                    f"cache={cache_len}"
                )
            max_output_error = max(max_output_error, output_error.max().item())
        direct_output, direct_lse = run(reader, inputs, permutation, 640)
        public_output, public_lse = run_public(
            inputs,
            permutation,
            query_len,
            640,
            graph_replays=100 if query_len == 5 else 0,
        )
        torch.testing.assert_close(
            public_output.view(torch.int16),
            direct_output[:active_rows].view(torch.int16),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(public_lse, direct_lse[:active_rows], rtol=0, atol=0)
    print(
        "PASS_D0_C_DYNAMIC_CAUSAL_PAGE32 "
        f"queries=1,5 cache_cases={','.join(str(value) for value in cases)} "
        f"logical_pages={d0.MAX_PAGES} physical_pages={PHYSICAL_PAGE_COUNT} "
        f"max_output_error={max_output_error:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
