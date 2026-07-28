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

"""Measure native SM100 NVFP4 GEMM ceilings at Kimi MLA matrix shapes."""

import argparse
import json

import cutlass
import cutlass.torch as cutlass_torch
import torch
from flashinfer.cute_dsl.utils import get_cutlass_dtype
from flashinfer.gemm import create_scale_factor_tensor, grouped_gemm_nt_masked

SF_VEC_SIZE = 16
GROUPS = 2  # The FlashInfer kernel currently has a CUTLASS-DSL L=1 issue.


def make_operand(rows: int, k: int, device: torch.device):
    reference = cutlass_torch.matrix(
        GROUPS, rows, k, False, cutlass.Float32, device=device
    )
    tensor, storage = cutlass_torch.cute_tensor_like(
        reference,
        get_cutlass_dtype("float4_e2m1fn"),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    rows_actual, k_actual, groups = storage.shape
    half_length = storage.numel() // 2
    storage = (
        storage.permute(2, 0, 1)
        .flatten()[:half_length]
        .reshape(groups, rows_actual, k_actual // 2)
        .permute(1, 2, 0)
    )
    scale_reference, _, scale_storage = create_scale_factor_tensor(
        GROUPS,
        rows,
        k,
        SF_VEC_SIZE,
        get_cutlass_dtype("float8_e4m3fn"),
        device,
    )
    return reference, scale_reference, storage, scale_storage, tensor


def bench(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=("qk", "pv"), required=True)
    parser.add_argument("--context", type=int, default=10240)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if args.context % 128:
        raise ValueError("--context must be a multiple of 128")

    device = torch.device("cuda:0")
    if args.op == "qk":
        m, n, k = 128, args.context, 512
        active_m = 64
    else:
        # Reversed value product V^T * P^T.
        m, n, k = 512, 128, args.context
        active_m = 512

    a_ref, sfa_ref, a, sfa, _ = make_operand(m, k, device)
    b_ref, sfb_ref, b, sfb, _ = make_operand(n, k, device)
    c_ref = cutlass_torch.matrix(GROUPS, m, n, False, cutlass.Float32, device=device)
    _, output = cutlass_torch.cute_tensor_like(
        c_ref,
        cutlass.BFloat16,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    masked_m = torch.tensor([active_m, 0], device=device, dtype=torch.int32)

    def run() -> None:
        grouped_gemm_nt_masked(
            (a, sfa),
            (b, sfb),
            output,
            masked_m,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=SF_VEC_SIZE,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )

    run()
    torch.cuda.synchronize()
    reference_a = a_ref * sfa_ref
    reference_b = b_ref * sfb_ref
    reference = torch.einsum("mkl,nkl->mnl", reference_a, reference_b)
    actual = output[:active_m, :, 0].float()
    expected = reference[:active_m, :, 0].float()
    error = actual - expected
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.02)

    group_divisor = output.shape[-1]
    result = {
        "status": "PASS",
        "op": args.op,
        "context": args.context,
        "m": m,
        "active_m": active_m,
        "n": n,
        "k": k,
        "microseconds": bench(run, args.warmup, args.iterations),
        "max_abs_diff": float(error.abs().max()),
        "mean_abs_diff": float(error.abs().mean()),
        "a_packed_bytes_per_group": a.numel() * a.element_size() // group_divisor,
        "a_scale_bytes_per_group": sfa.numel() * sfa.element_size() // group_divisor,
        "b_packed_bytes_per_group": b.numel() * b.element_size() // group_divisor,
        "b_scale_bytes_per_group": sfb.numel() * sfb.element_size() // group_divisor,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
