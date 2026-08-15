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

"""One complete q5/page-permuted D0 launch for Compute Sanitizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

RUNTIME_PATH = Path(__file__).parent / "probe_sm100_tq_e2m1_d0_runtime.py"
RUNTIME_SPEC = importlib.util.spec_from_file_location("d0_runtime", RUNTIME_PATH)
if RUNTIME_SPEC is None or RUNTIME_SPEC.loader is None:
    raise RuntimeError(f"cannot load D0 runtime probe from {RUNTIME_PATH}")
runtime = importlib.util.module_from_spec(RUNTIME_SPEC)
RUNTIME_SPEC.loader.exec_module(runtime)


def main() -> None:
    query_len = 5
    cache_len = 640
    permutation = torch.tensor(
        [31, 1, 36, 4, 22, 0, 26, 9, 3, 24, 6, 35, 20, 2, 25, 8, 23, 5, 27, 21],
        dtype=torch.int64,
    )
    reader = runtime.compile_reader(query_len, runtime.PHYSICAL_PAGE_COUNT)
    inputs = runtime.make_inputs(query_len)
    output, lse = runtime.run(reader, inputs, permutation, cache_len)
    active_rows = query_len * runtime.d0.NUM_HEADS
    if not torch.isfinite(output[:active_rows]).all():
        raise AssertionError("sanitizer launch produced non-finite output")
    if not torch.isfinite(lse[:active_rows]).all():
        raise AssertionError("sanitizer launch produced non-finite LSE")
    print("PASS_D0_D_SANITIZER_SMOKE", flush=True)


if __name__ == "__main__":
    main()
