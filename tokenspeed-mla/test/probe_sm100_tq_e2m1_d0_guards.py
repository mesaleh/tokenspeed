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

"""Fail-closed host and device guards for D0's public E2M1 reader."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).parent / "tokenspeed_mla" / "mla_decode_e2m1.py"
MODULE_SPEC = importlib.util.spec_from_file_location("mla_decode_e2m1", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load D0 module from {MODULE_PATH}")
d0 = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(d0)


def inputs(query_len: int = 1, physical_pages: int = 37):
    return (
        torch.zeros(
            (1, query_len, d0.NUM_HEADS, d0.QUERY_DIM),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        torch.zeros(
            (physical_pages, d0.PAGE_SIZE, d0.LATENT_K // 2),
            dtype=torch.uint8,
            device="cuda",
        ),
        torch.ones(
            (physical_pages, d0.PAGE_SIZE),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        torch.zeros(
            (physical_pages, d0.PAGE_SIZE, d0.ROPE_K),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        torch.arange(d0.MAX_PAGES, dtype=torch.int32, device="cuda")[None],
    )


def require_raises(error_type, pattern: str, callback) -> None:
    try:
        callback()
    except error_type as error:
        if pattern not in str(error):
            raise AssertionError(
                f"guard raised the wrong message: {error!s}"
            ) from error
    else:
        raise AssertionError(f"expected {error_type.__name__}: {pattern}")


def run_host_guards() -> None:
    query, packed, scale, rope, pages = inputs()
    softmax_scale = d0.PROBE_SOFTMAX_SCALE_LOG2 / math.log2(math.e)
    require_raises(
        ValueError,
        "batch must be 1",
        lambda: d0.tokenspeed_mla_decode_e2m1(
            query.expand(2, -1, -1, -1),
            packed,
            scale,
            rope,
            pages,
            1,
            softmax_scale,
        ),
    )
    query_q2, *_ = inputs(query_len=2)
    require_raises(
        ValueError,
        "query length must be 1 or 5",
        lambda: d0.tokenspeed_mla_decode_e2m1(
            query_q2, packed, scale, rope, pages, 1, softmax_scale
        ),
    )
    require_raises(
        TypeError,
        "query must be FP8 E4M3FN",
        lambda: d0.tokenspeed_mla_decode_e2m1(
            query.to(torch.bfloat16),
            packed,
            scale,
            rope,
            pages,
            1,
            softmax_scale,
        ),
    )
    require_raises(
        ValueError,
        f"block_table needs at least {d0.MAX_PAGES} entries",
        lambda: d0.tokenspeed_mla_decode_e2m1(
            query,
            packed,
            scale,
            rope,
            pages[:, :-1],
            1,
            softmax_scale,
        ),
    )
    require_raises(
        ValueError,
        "cache_len must be in",
        lambda: d0.tokenspeed_mla_decode_e2m1(
            query, packed, scale, rope, pages, 641, softmax_scale
        ),
    )
    require_raises(
        TypeError,
        "unexpected keyword argument 'cp_world'",
        lambda: d0.tokenspeed_mla_decode_e2m1(
            query,
            packed,
            scale,
            rope,
            pages,
            1,
            softmax_scale,
            cp_world=2,
        ),
    )
    print("PASS_D0_D_HOST_GUARDS", flush=True)


def run_invalid_page_guard() -> None:
    query, packed, scale, rope, pages = inputs()
    softmax_scale = d0.PROBE_SOFTMAX_SCALE_LOG2 / math.log2(math.e)
    output = torch.empty(
        (1, 1, d0.NUM_HEADS, d0.LATENT_K),
        dtype=torch.bfloat16,
        device="cuda",
    )
    lse = torch.empty((1, 1, d0.NUM_HEADS), dtype=torch.float32, device="cuda")
    d0.tokenspeed_mla_decode_e2m1(
        query,
        packed,
        scale,
        rope,
        pages,
        1,
        softmax_scale,
        output,
        lse,
        trusted_page_table=True,
    )
    torch.cuda.synchronize()
    invalid_pages = pages.clone()
    invalid_pages[0, 7] = packed.shape[0]
    try:
        d0.tokenspeed_mla_decode_e2m1(
            query,
            packed,
            scale,
            rope,
            invalid_pages,
            1,
            softmax_scale,
            output,
            lse,
        )
        torch.cuda.synchronize()
    except Exception as error:
        if not any(
            marker in str(error)
            for marker in (
                "block_table contains a physical page",
                "device-side assert triggered",
            )
        ):
            raise
        print("PASS_D0_D_INVALID_PAGE_GUARD", flush=True)
        return
    raise AssertionError("invalid physical page reached the E2M1 reader")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("host", "invalid-page"), required=True)
    args = parser.parse_args()
    if args.mode == "host":
        run_host_guards()
    else:
        run_invalid_page_guard()


if __name__ == "__main__":
    main()
