#!/usr/bin/env python3
"""Build or run exact aligned/unaligned named-barrier CUDA litmus cells."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

from evidence_common import canonical_uuid, load_json, require, sha256_bytes, sha256_file, write_json_exclusive


EXTENSION_NAME = "ts_c58_r_barrier_litmus_v1"
CPP_SOURCE = r"""
#include <torch/extension.h>
int run_barrier_litmus(int cell);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run_barrier_litmus, "Run one TS-C58-R barrier litmus cell");
}
"""
CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void aligned_full(int* completed) {
  asm volatile("barrier.sync.aligned 7, 512;" ::: "memory");
  atomicAdd(completed, 1);
}

__global__ void aligned_partial(int* completed) {
  if (threadIdx.x < 128) {
    asm volatile("barrier.sync.aligned 7, 128;" ::: "memory");
    atomicAdd(completed, 1);
  }
}

__global__ void unaligned_partial(int* completed) {
  if (threadIdx.x < 128) {
    asm volatile("barrier.sync 7, 128;" ::: "memory");
    atomicAdd(completed, 1);
  }
}

__global__ void unaligned_wrong_count(int* completed) {
  if (threadIdx.x < 128) {
    asm volatile("barrier.sync 7, 160;" ::: "memory");
    atomicAdd(completed, 1);
  }
}

int run_barrier_litmus(int cell) {
  int* completed = nullptr;
  cudaError_t status = cudaMallocManaged(&completed, sizeof(int));
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  *completed = 0;
  switch (cell) {
    case 0: aligned_full<<<1, 512>>>(completed); break;
    case 1: aligned_partial<<<1, 512>>>(completed); break;
    case 2: unaligned_partial<<<1, 512>>>(completed); break;
    case 3: unaligned_wrong_count<<<1, 512>>>(completed); break;
    default: cudaFree(completed); throw std::invalid_argument("unknown litmus cell");
  }
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    cudaFree(completed);
    throw std::runtime_error(cudaGetErrorString(status));
  }
  int result = *completed;
  cudaFree(completed);
  return result;
}
"""
CELLS = {
    "aligned-full-512-512": (0, 512),
    "aligned-partial-128-128": (1, 128),
    "unaligned-partial-128-128": (2, 128),
    "unaligned-wrong-count-128-160": (3, None),
}


def build_extension():
    return load_inline(
        name=EXTENSION_NAME,
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--target-uuid", required=True)
    parser.add_argument("--cell", choices=tuple(CELLS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--expected-build", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")
    if args.prepare_only == (args.cell is not None):
        parser.error("choose exactly one of --prepare-only or --cell")
    if args.prepare_only == (args.expected_build is not None):
        parser.error("run mode requires --expected-build; prepare mode forbids it")
    require(0 <= args.device_index < torch.cuda.device_count(), "CUDA ordinal is not visible")
    torch.cuda.set_device(args.device_index)
    require(torch.cuda.get_device_capability() == (10, 0), "SM100 is required")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    raw_uuid = str(getattr(properties, "uuid", ""))
    device_uuid = canonical_uuid(raw_uuid if raw_uuid.lower().startswith("gpu-") else f"GPU-{raw_uuid}")
    require(device_uuid == canonical_uuid(args.target_uuid), "CUDA ordinal does not map to target UUID")
    module = build_extension()
    module_path = Path(module.__file__).resolve()
    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r-barrier-litmus",
        "status": "pass",
        "mode": "prepare" if args.prepare_only else "run",
        "cell": args.cell,
        "device_index": args.device_index,
        "target_uuid": device_uuid,
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "extension_name": EXTENSION_NAME,
        "extension_path": str(module_path),
        "extension_sha256": sha256_file(module_path),
        "cpp_source_sha256": sha256_bytes(CPP_SOURCE.encode()),
        "cuda_source_sha256": sha256_bytes(CUDA_SOURCE.encode()),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }
    if args.cell is not None:
        expected_build, expected_build_raw = load_json(args.expected_build)
        require(
            expected_build.get("record_type") == "ts-c58-r-barrier-litmus"
            and expected_build.get("status") == "pass"
            and expected_build.get("mode") == "prepare"
            and expected_build.get("cell") is None
            and expected_build.get("target_uuid") == device_uuid
            and expected_build.get("device_index") == args.device_index
            and expected_build.get("extension_sha256") == value["extension_sha256"]
            and expected_build.get("cpp_source_sha256") == value["cpp_source_sha256"]
            and expected_build.get("cuda_source_sha256") == value["cuda_source_sha256"]
            and expected_build.get("tool_sha256") == value["tool_sha256"],
            "prepared litmus build identity differs",
        )
        value["expected_build_sha256"] = sha256_bytes(expected_build_raw)
        cell, expected = CELLS[args.cell]
        completed = int(module.run(cell))
        if expected is not None:
            require(completed == expected, f"litmus completion count differs: {completed}")
        value["completed_threads"] = completed
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
