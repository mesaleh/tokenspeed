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

"""Compile-only gate for the diagnostic-free D0 RN extraction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Generated-code correspondence is the purpose of this test. Keep artifacts
# and bypass the artifact-free on-disk callable cache before CuTe DSL initializes.
os.environ.setdefault("CUTE_DSL_KEEP", "cubin,sass")
os.environ.setdefault("CUTE_DSL_DISABLE_FILE_CACHING", "1")

import cutlass
import cutlass.cute as cute
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

EXPECTED_SASS_COUNTS = {
    "mma": 130,
    "tma": 115,
    "ldl": 50,
    "stl": 14,
    "mufu_rcp": 10,
    "fchk": 9,
    "calls": 12,
    "ffma": 62,
}
EXPECTED_RESOURCES = {
    "reg": 168,
    "stack": 24,
    "shared": 1024,
    "local": 0,
}


def fake(dtype: type[cutlass.Numeric], shape: tuple[int, ...], align: int):
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )


def audit_generated(compiled) -> dict[str, dict[str, int]]:
    nvdisasm = shutil.which("nvdisasm")
    cuobjdump = shutil.which("cuobjdump")
    if nvdisasm is None or cuobjdump is None:
        raise RuntimeError("nvdisasm and cuobjdump are required for the D0 gates")
    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
        cubin_file.write(compiled.__cubin__)
        cubin_file.flush()
        sass = subprocess.run(
            [nvdisasm, cubin_file.name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        resource_output = subprocess.run(
            [cuobjdump, "--dump-resource-usage", cubin_file.name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    sass_counts = {
        "mma": sass.count(" UTCQMMA"),
        "tma": sass.count(" UTMALDG"),
        "ldl": sass.count(" LDL"),
        "stl": sass.count(" STL"),
        "mufu_rcp": sass.count(" MUFU.RCP"),
        "fchk": sass.count(" FCHK"),
        "calls": sass.count(" CALL.REL.NOINC"),
        "ffma": sass.count(" FFMA"),
    }
    if sass_counts != EXPECTED_SASS_COUNTS:
        raise AssertionError(
            f"D0 generated instruction profile drifted: {sass_counts} != "
            f"{EXPECTED_SASS_COUNTS}"
        )
    # The accepted A8 RN ancestor contains 69 LDL + 1 STL. Paging may change
    # the split, but it must not introduce a larger local-memory family.
    if sass_counts["ldl"] + sass_counts["stl"] > 70:
        raise AssertionError(f"D0 added local-memory traffic: {sass_counts}")

    match = re.search(
        r"REG:(?P<reg>\d+) STACK:(?P<stack>\d+) "
        r"SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)",
        resource_output,
    )
    if match is None:
        raise AssertionError(f"could not parse D0 resource usage:\n{resource_output}")
    resources = {name: int(value) for name, value in match.groupdict().items()}
    if resources != EXPECTED_RESOURCES:
        raise AssertionError(
            f"D0 generated resources drifted: {resources} != {EXPECTED_RESOURCES}"
        )
    return {"sass": sass_counts, "resources": resources}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-generated-dir")
    args = parser.parse_args()

    compiled = cute.compile(
        d0.ownership_probe,
        make_ptr(
            cutlass.Float8E4M3FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float4E2M1FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            0,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        fake(cutlass.BFloat16, (d0.SCORE_ROWS, d0.LATENT_K), 16),
        fake(cutlass.Float32, (d0.SCORE_ROWS,), 16),
        fake(cutlass.BFloat16, (d0.MAX_PAGES, d0.PAGE_SIZE), 16),
        fake(cutlass.Int32, (d0.MAX_PAGES,), 16),
        cutlass.Int32(640),
        cutlass.Float32(d0.PROBE_SOFTMAX_SCALE_LOG2),
        d0.SCORE_ROWS,
        d0.SCORE_ROWS // d0.NUM_HEADS,
        d0.MAX_PAGES,
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

    cubin = compiled.__cubin__
    audit = audit_generated(compiled)
    print(
        "PASS_D0_A_DIAGNOSTIC_FREE_COMPILE "
        f"cubin_sha256={hashlib.sha256(cubin).hexdigest()} "
        f"cubin_bytes={len(cubin)} audit={json.dumps(audit, sort_keys=True)}",
        flush=True,
    )
    if args.dump_generated_dir:
        output_dir = Path(args.dump_generated_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix, attribute, binary in (
            ("ptx", "__ptx__", False),
            ("sass", "__sass__", False),
            ("mlir", "__mlir__", False),
            ("cubin", "__cubin__", True),
        ):
            payload = getattr(compiled, attribute)
            path = output_dir / f"d0-a.{suffix}"
            if binary:
                path.write_bytes(payload)
            else:
                path.write_text(payload)


if __name__ == "__main__":
    main()
