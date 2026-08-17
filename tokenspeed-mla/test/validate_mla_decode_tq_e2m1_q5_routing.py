#!/usr/bin/env python3
"""Validate q1-scalar/q5-packed routing through the public TQ MLA API."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

import benchmark_mla_decode_tq_e2m1_packed_scale_application as benchmark
import tokenspeed_mla.mla_decode_tq_e2m1 as public_reader


EXPECTED_LSE_SASS = {
    1: "5ef06bbba7ae3dd52279a7b5ad8c6da0899d11dfbf6c2475da37d25c29cb370a",
    5: "1f6858384075485016ca70cf20176ca46c5e37efa59a2d413a36c8c6b9e70e0c",
}
INPUT_NAMES = (
    "query_latent",
    "query_rope",
    "packed",
    "scale",
    "reciprocal_rope",
    "block_tables",
    "seq_lens",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(source_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _hash_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _input_state(tensors: dict[str, Any]) -> dict[str, dict[str, int | str]]:
    return {
        name: {
            "pointer": tensors[name].data_ptr(),
            "sha256": _hash_tensor(tensors[name]),
        }
        for name in INPUT_NAMES
    }


def _call_public(
    *,
    tensors: dict[str, Any],
    workspace: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor | None,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    result = public_reader.tokenspeed_mla_decode_tq_e2m1(
        query_latent=tensors["query_latent"],
        query_rope=tensors["query_rope"],
        packed_latent=tensors["packed"],
        reconstruction_scale=tensors["scale"],
        reciprocal_rope=tensors["reciprocal_rope"],
        workspace_buffer=workspace,
        block_tables=tensors["block_tables"],
        seq_lens=tensors["seq_lens"],
        max_seq_len=benchmark.PRIMARY_SEQ_LEN,
        softmax_scale=1.0 / math.sqrt(benchmark.LATENT + benchmark.ROPE),
        output_scale=1.0,
        out=output,
        enable_pdl=False,
        return_lse=return_lse,
        lse_out=lse,
    )
    if return_lse:
        observed, observed_lse = result
        return observed, observed_lse
    return result, None


def _only_compiled_kernel():
    if len(public_reader._COMPILED_KERNELS) != 1:
        raise RuntimeError("public wrapper did not create exactly one shape variant")
    return next(iter(public_reader._COMPILED_KERNELS.values()))


def main() -> None:
    if os.environ.get("CUTE_DSL_KEEP") != "all":
        raise ValueError("CUTE_DSL_KEEP=all is required")
    source_root = Path(os.environ["R8_SOURCE_ROOT"]).resolve()
    expected_commit = os.environ["R8_EXPECTED_COMMIT"]
    observed_commit = _git(source_root, "rev-parse", "HEAD")
    if observed_commit != expected_commit:
        raise RuntimeError(
            f"source commit drift: {observed_commit} != {expected_commit}"
        )
    if _git(source_root, "status", "--short"):
        raise RuntimeError("R8 routing source tree is dirty")

    query_len = int(os.environ["R8_QUERY_LEN"])
    if query_len not in EXPECTED_LSE_SASS:
        raise ValueError("R8_QUERY_LEN must be 1 or 5")
    return_lse_value = os.environ.get("R8_RETURN_LSE", "0")
    if return_lse_value not in {"0", "1"}:
        raise ValueError("R8_RETURN_LSE must be 0 or 1")
    return_lse = return_lse_value == "1"
    output_dir = Path(os.environ["R8_OUTPUT_DIR"]).resolve()
    if output_dir == source_root or source_root in output_dir.parents:
        raise ValueError("R8_OUTPUT_DIR must be outside the immutable source tree")
    output_dir.mkdir(parents=True, exist_ok=True)
    # CUTE_DSL_KEEP writes compiler intermediates to the current directory.
    # Keep those evidence files outside the immutable source checkout.
    os.chdir(output_dir)

    tensors = benchmark._build_fixture(
        query_len,
        benchmark.PRIMARY_SEQ_LEN,
        permutation_seed=2026081701,
    )
    split, workspace_size = benchmark._workspace_geometry(
        query_len, benchmark.PRIMARY_SEQ_LEN
    )
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.int8, device="cuda")
    output = torch.empty(
        (1, query_len, benchmark.HEADS, benchmark.LATENT),
        dtype=torch.bfloat16,
        device="cuda",
    )
    lse = (
        torch.empty(
            (1, query_len, benchmark.HEADS),
            dtype=torch.float32,
            device="cuda",
        )
        if return_lse
        else None
    )
    input_state_before = _input_state(tensors)

    expected_packed = query_len == 5
    if public_reader._use_packed_p_scale_math(query_len) != expected_packed:
        raise RuntimeError(f"q{query_len} selected the wrong routing arm")
    routed_arm = "packed" if expected_packed else "full"

    public_reader._COMPILED_KERNELS.clear()
    observed, observed_lse = _call_public(
        tensors=tensors,
        workspace=workspace,
        output=output,
        lse=lse,
        return_lse=return_lse,
    )
    torch.cuda.synchronize()
    eager_output = observed.clone()
    eager_lse = observed_lse.clone() if observed_lse is not None else None
    routed_kernel = _only_compiled_kernel()
    routed_artifact = benchmark._artifact_identity(
        routed_kernel,
        routed_arm,
        query_len,
        split,
        output_dir / "routed-artifact",
        False,
    )
    if return_lse and routed_artifact["sass_sha256"] != EXPECTED_LSE_SASS[query_len]:
        raise RuntimeError(
            f"q{query_len} routed LSE SASS drift: "
            f"{routed_artifact['sass_sha256']} != {EXPECTED_LSE_SASS[query_len]}"
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call_public(
            tensors=tensors,
            workspace=workspace,
            output=output,
            lse=lse,
            return_lse=return_lse,
        )
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(output, eager_output):
        raise RuntimeError(f"q{query_len} eager/graph100 output parity failed")
    if lse is not None and not torch.equal(lse, eager_lse):
        raise RuntimeError(f"q{query_len} eager/graph100 LSE parity failed")

    control_output = torch.empty_like(output)
    control_lse = torch.empty_like(lse) if lse is not None else None
    original_selector = public_reader._use_packed_p_scale_math
    try:
        public_reader._use_packed_p_scale_math = lambda candidate_q: False
        public_reader._COMPILED_KERNELS.clear()
        control, control_observed_lse = _call_public(
            tensors=tensors,
            workspace=workspace,
            output=control_output,
            lse=control_lse,
            return_lse=return_lse,
        )
        torch.cuda.synchronize()
        control_kernel = _only_compiled_kernel()
    finally:
        public_reader._use_packed_p_scale_math = original_selector
        public_reader._COMPILED_KERNELS.clear()
    control_artifact = benchmark._artifact_identity(
        control_kernel,
        "full",
        query_len,
        split,
        output_dir / "scalar-control-artifact",
        False,
    )
    if not torch.equal(observed, control):
        raise RuntimeError(f"q{query_len} routed/full output parity failed")
    if observed_lse is not None and not torch.equal(
        observed_lse, control_observed_lse
    ):
        raise RuntimeError(f"q{query_len} routed/full LSE parity failed")

    replay_output = torch.empty_like(output)
    replay_lse = torch.empty_like(lse) if lse is not None else None
    replay, replay_observed_lse = _call_public(
        tensors=tensors,
        workspace=workspace,
        output=replay_output,
        lse=replay_lse,
        return_lse=return_lse,
    )
    torch.cuda.synchronize()
    if not torch.equal(replay, eager_output):
        raise RuntimeError(f"q{query_len} post-control routed replay failed")
    if replay_observed_lse is not None and not torch.equal(
        replay_observed_lse, eager_lse
    ):
        raise RuntimeError(f"q{query_len} post-control LSE replay failed")

    input_state_after = _input_state(tensors)
    if input_state_after != input_state_before:
        raise RuntimeError(f"q{query_len} input pointer or content mutation")
    if _git(source_root, "status", "--short"):
        raise RuntimeError("R8 routing validation dirtied the source tree")

    receipt = {
        "schema_version": 2,
        "experiment": "A17-N10-E0-R8",
        "commit": observed_commit,
        "query_len": query_len,
        "seq_len": benchmark.PRIMARY_SEQ_LEN,
        "physical_seq_len": benchmark.PRIMARY_PHYSICAL_SEQ_LEN,
        "return_lse": return_lse,
        "routed_arm": routed_arm,
        "source_sha256": _sha256(
            source_root / "tokenspeed-mla/python/tokenspeed_mla/mla_decode_tq_e2m1.py"
        ),
        "reader_sha256": _sha256(
            source_root / "tokenspeed-mla/python/tokenspeed_mla/mla_decode_fp8.py"
        ),
        "harness_sha256": _sha256(Path(__file__).resolve()),
        "routed_artifact": routed_artifact,
        "scalar_control_artifact": control_artifact,
        "output_sha256": _hash_tensor(output),
        "lse_sha256": _hash_tensor(lse) if lse is not None else None,
        "input_state": input_state_after,
        "pointers": {
            "output": output.data_ptr(),
            "lse": lse.data_ptr() if lse is not None else None,
            "workspace": workspace.data_ptr(),
        },
        "checks": {
            "routing_selected_expected_arm": True,
            "eager_graph100_bitwise": True,
            "routed_full_bitwise": True,
            "post_control_routed_replay_bitwise": True,
            "inputs_immutable": True,
            "source_clean_after": True,
            "frozen_lse_sass_match": return_lse,
        },
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("R8_ROUTING_RECEIPT " + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
