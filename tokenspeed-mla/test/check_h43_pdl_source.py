#!/usr/bin/env python3
"""Fail closed if mutable codebook reads move ahead of the PDL pipeline wait."""

from __future__ import annotations

import json
from pathlib import Path

import tokenspeed_mla.mla_decode_fp8 as mla_decode_fp8
from h43_codebook_ab_common import sha256_file


def main() -> None:
    path = Path(mla_decode_fp8.__file__).resolve()
    source = path.read_text(encoding="utf-8")
    start = source.index("    def convert_tq4_kv(")
    end = source.index("\n    @cute.jit", start + 1)
    body = source[start:end]
    positions = {
        "immutable_centroid": body.index("common_params.mTQCentroids"),
        "consumer_wait": body.index("common_params.raw_k_pipeline.consumer_wait"),
        "page_table": body.index("page_table = common_params.mPT"),
        "codebook_pointer": body.index("codebook_i32_ptr = cute.recast_ptr"),
    }
    positions["codebook_load"] = body.index(").load()", positions["codebook_pointer"])
    if not (
        positions["immutable_centroid"]
        < positions["consumer_wait"]
        < positions["page_table"]
        < positions["codebook_pointer"]
        < positions["codebook_load"]
    ):
        raise RuntimeError(f"H43 PDL source order is unsafe: {positions}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "source": str(path),
                "source_sha256": sha256_file(path),
                "relative_positions": positions,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
