#!/usr/bin/env python3
"""Run one exact H43 q5 codebook call for a focused racecheck gate."""

from __future__ import annotations

import argparse
import hashlib
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

    racecheck = contract["racecheck"]
    q_len = racecheck["target_query_length"]
    if racecheck["target_tq_calls"] != 1:
        raise ValueError("the H43 racecheck probe requires exactly one TQ call")

    experiment = Experiment(contract, args.context, args.sequence)
    output = experiment.output(q_len)
    experiment._first_call(
        f"racecheck-tq-codebook-q{q_len}",
        lambda: experiment.tq_call(
            0,
            experiment.queries[q_len],
            output,
            use_codebook=True,
        ),
    )
    if not bool(torch.isfinite(output.float()).all()):
        raise AssertionError("targeted H43 q5 call produced non-finite output")
    output_sha256 = hashlib.sha256(
        output.detach().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()

    value = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": contract["experiment"],
        "contract_digest": canonical_json_digest(contract),
        "context": args.context,
        "sequence": args.sequence,
        "query_length": q_len,
        "target_tq_calls": 1,
        "codebook_sha256": experiment.codebook_sha256,
        "output_sha256": output_sha256,
        "output_finite": True,
    }
    value["result_digest"] = canonical_json_digest(value)
    print(json.dumps(value, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
