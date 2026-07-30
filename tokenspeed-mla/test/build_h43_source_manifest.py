#!/usr/bin/env python3
"""Emit the exact installed-package and staged-work SHA-256 manifest for H43."""

from __future__ import annotations

import argparse
from pathlib import Path

import tokenspeed_mla
from h43_codebook_ab_common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    package = Path(tokenspeed_mla.__file__).resolve().parent
    paths = sorted(package.rglob("*.py"))
    paths += sorted(
        path
        for path in work.rglob("*")
        if path.is_file()
        and path.name != "H43_D1_SOURCE_MANIFEST.sha256"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    if not paths or not any(path.name == "mla_decode_fp8.py" for path in paths):
        raise RuntimeError("installed TokenSpeed MLA package is incomplete")
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"source manifest refuses symlink: {path}")
        print(f"{sha256_file(path)}  {path}")


if __name__ == "__main__":
    main()
