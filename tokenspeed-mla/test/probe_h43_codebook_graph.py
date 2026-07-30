#!/usr/bin/env python3
"""Excluded correctness/resource/sanitizer probe for the H43 q5 ring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from h43_codebook_ab_common import canonical_json_digest, load_contract
from microbench_h43_codebook_ab import Experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--sequence", type=int, default=1)
    args = parser.parse_args()
    contract = load_contract(args.contract.resolve())
    if str(args.context) not in contract["contexts"]:
        raise ValueError("context is absent from the H43 contract")

    experiment = Experiment(contract, args.context, args.sequence)
    correctness = experiment.correctness()
    graph, outputs = experiment.codebook_probe_graph()
    torch.cuda.nvtx.range_push("H43_CODEBOOK_Q5_RING")
    graph.replay()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    if any(not bool(torch.isfinite(output.float()).all()) for output in outputs):
        raise AssertionError("profiled codebook ring produced non-finite output")

    value = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": contract["experiment"],
        "contract_digest": canonical_json_digest(contract),
        "context": args.context,
        "sequence": args.sequence,
        "correctness": correctness,
        "codebook_sha256": experiment.codebook_sha256,
        "profiled_graph_replays": 1,
        "profiled_ring_layers": contract["geometry"]["total_layers"],
        "profiled_selected_codebook_layers": contract["geometry"]["selected_layers"],
    }
    value["result_digest"] = canonical_json_digest(value)
    print(json.dumps(value, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
