#!/usr/bin/env python3
"""Accept or reject TS-C58-R1 from the complete sealed repair matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_common import (
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    validate_source_identity,
    write_json_exclusive,
)
from make_r1_specs import CELLS


SANITIZER_CELLS = {
    "accepted-target-racecheck": "racecheck",
    "accepted-target-synccheck": "synccheck",
    "dense-control-synccheck": "synccheck",
}

ACCEPTED_SOURCE_COMMIT = "26c6f21742024cebcb6df0df5a70a269c7f5b0f5"
ACCEPTED_IDENTITY_SHA256 = (
    "fd1ddbf4582885b63482d6e6fdccfc5df0ba50aac750419e6f321294fcf46847"
)
ACCEPTED_BASELINES = {
    "m128": {
        "oracle_sha256": (
            "28837a34b1df644a6f5bcab6caa1be2af9e397240d203d75f57f276d6dc553d6"
        ),
        "seal_sha256": (
            "c8f9331cc58077531b564fd2f3dff47222506c029b75df016b03f567f7e206a5"
        ),
    },
    "dense": {
        "oracle_sha256": (
            "4f3b0ed63d469476c6b6cdbb83f4b5d50b04bfb145f0ce2041d849202907e02d"
        ),
        "seal_sha256": (
            "a7263a9dfaa78b1308328814ced0672eada510170124989af16a66605c0af5de"
        ),
    },
}


def load_seals(paths: list[Path], record_type: str, key: str) -> dict[str, tuple[dict, bytes]]:
    result: dict[str, tuple[dict, bytes]] = {}
    for path in paths:
        value, raw = load_json(path)
        require(value.get("record_type") == record_type, f"{record_type} differs")
        name = value.get(key)
        require(isinstance(name, str) and name not in result, f"duplicate or invalid {key}")
        result[name] = (value, raw)
    return result


def require_accepted_identity(
    accepted_identity: dict, accepted_identity_raw: bytes, candidate_commit: str
) -> tuple[str, str]:
    accepted_identity_hash = sha256_bytes(accepted_identity_raw)
    accepted_commit = accepted_identity.get("source_commit")
    require(
        accepted_identity.get("record_type") == "ts-c58-r-source-identity"
        and accepted_identity.get("schema_version") == 1
        and accepted_identity_hash == ACCEPTED_IDENTITY_SHA256
        and accepted_commit == ACCEPTED_SOURCE_COMMIT
        and candidate_commit != accepted_commit,
        "accepted identity is not the pinned baseline",
    )
    return accepted_identity_hash, accepted_commit


def require_baseline(
    oracle: dict,
    oracle_raw: bytes,
    seal: dict,
    seal_raw: bytes,
    accepted_identity_hash: str,
    accepted_commit: str,
    arm: str,
    cell: str,
) -> None:
    expected = ACCEPTED_BASELINES[arm]
    require(
        sha256_bytes(oracle_raw) == expected["oracle_sha256"]
        and sha256_bytes(seal_raw) == expected["seal_sha256"],
        f"accepted {arm} baseline bytes differ",
    )
    require(
        oracle.get("record_type") == "ts-c58-r-decode-oracle"
        and oracle.get("status") == "pass"
        and oracle.get("arm") == arm
        and oracle.get("mode") == "unsanitized"
        and oracle.get("source_commit") == accepted_commit
        and oracle.get("source_identity_sha256") == accepted_identity_hash,
        f"accepted {arm} oracle differs",
    )
    require(
        seal.get("record_type") == "ts-c58-r-execution-seal"
        and seal.get("status") == "pass"
        and seal.get("cell_id") == cell
        and seal.get("actual_outcome") == "clean"
        and seal.get("sanitizer_tool") is None
        and seal.get("source_commit") == accepted_commit
        and seal.get("source_identity_sha256") == accepted_identity_hash
        and seal.get("result_sha256") == sha256_bytes(oracle_raw),
        f"accepted {arm} oracle seal differs",
    )


def barrier_rows(manifest: dict, field: str, barrier_id: int, count: int) -> list[dict]:
    rows = manifest.get(field)
    require(isinstance(rows, list), f"{field} differs")
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("barrier_id") == barrier_id
        and row.get("count") == count
    ]


def barrier_id_rows(manifest: dict, field: str, barrier_id: int) -> list[dict]:
    rows = manifest.get(field)
    require(isinstance(rows, list), f"{field} differs")
    return [
        row
        for row in rows
        if isinstance(row, dict) and row.get("barrier_id") == barrier_id
    ]


def require_disassembly(manifest: dict, arm: str) -> dict[str, object]:
    require(
        manifest.get("record_type") == "ts-c58-r-disassembly-manifest"
        and manifest.get("status") == "pass"
        and manifest.get("arm") == arm,
        f"{arm} disassembly differs",
    )
    ptx_handoff = barrier_rows(manifest, "ptx_named_barriers", 1, 288)
    sass_handoff = barrier_rows(manifest, "sass_named_barriers", 1, 288)
    require(
        ptx_handoff == barrier_id_rows(manifest, "ptx_named_barriers", 1),
        f"{arm} PTX handoff operands drifted",
    )
    require(
        sass_handoff == barrier_id_rows(manifest, "sass_named_barriers", 1),
        f"{arm} SASS handoff operands drifted",
    )
    require(len(ptx_handoff) == 3, f"{arm} PTX handoff-site count differs")
    require(len(sass_handoff) == 3, f"{arm} SASS handoff-site count differs")
    require(
        all(row.get("instruction") == "barrier.sync" and row.get("aligned") is False
            for row in ptx_handoff),
        f"{arm} PTX handoff is not explicitly unaligned",
    )
    ptx_tq4 = barrier_rows(manifest, "ptx_named_barriers", 6, 128)
    sass_tq4 = barrier_rows(manifest, "sass_named_barriers", 6, 128)
    require(
        ptx_tq4 == barrier_id_rows(manifest, "ptx_named_barriers", 6),
        f"{arm} PTX TQ4 operands drifted",
    )
    require(
        sass_tq4 == barrier_id_rows(manifest, "sass_named_barriers", 6),
        f"{arm} SASS TQ4 operands drifted",
    )
    expected_tq4_sites = 6 if arm == "m128" else 0
    require(len(ptx_tq4) == expected_tq4_sites, f"{arm} PTX TQ4-site count differs")
    require(len(sass_tq4) == expected_tq4_sites, f"{arm} SASS TQ4-site count differs")
    return {
        "ptx_handoff_sites": len(ptx_handoff),
        "sass_handoff_sites": len(sass_handoff),
        "ptx_tq4_sites": len(ptx_tq4),
        "sass_tq4_sites": len(sass_tq4),
        "ptx_sha256": manifest.get("ptx_sha256"),
        "cubin_sha256": manifest.get("cubin_sha256"),
        "disassembly_sha256": manifest.get("disassembly_sha256"),
    }


def require_zero_recovery(
    recovery: dict,
    execution_raw: bytes,
    cell: str,
    *,
    target_uuid: str,
    capture_sha256: str,
    sealer_sha256: str,
) -> None:
    require(
        recovery.get("record_type") == "ts-c58-r-gpu-recovery-seal"
        and recovery.get("status") == "pass"
        and recovery.get("phase_id") == cell
        and recovery.get("execution_seal_sha256") == sha256_bytes(execution_raw),
        f"{cell} recovery seal differs",
    )
    require(
        recovery.get("capture_sha256") == capture_sha256
        and recovery.get("sealer_sha256") == sealer_sha256
        and target_uuid in recovery.get("gpu_uuids", []),
        f"{cell} recovery tool or target identity differs",
    )
    deltas = recovery.get("monotonic_deltas")
    require(isinstance(deltas, dict) and len(deltas) == 4, f"{cell} recovery width differs")
    require(
        all(
            value is None or value == 0
            for gpu in deltas.values()
            if isinstance(gpu, dict)
            for value in gpu.values()
        )
        and all(isinstance(gpu, dict) for gpu in deltas.values()),
        f"{cell} recovery counter increased",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--accepted-identity", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--spec-suite", type=Path, required=True)
    parser.add_argument("--baseline-m128", type=Path, required=True)
    parser.add_argument("--baseline-m128-seal", type=Path, required=True)
    parser.add_argument("--baseline-dense", type=Path, required=True)
    parser.add_argument("--baseline-dense-seal", type=Path, required=True)
    parser.add_argument("--candidate-m128", type=Path, required=True)
    parser.add_argument("--candidate-dense", type=Path, required=True)
    parser.add_argument("--m128-disassembly", type=Path, required=True)
    parser.add_argument("--dense-disassembly", type=Path, required=True)
    parser.add_argument("--execution-seal", type=Path, action="append", required=True)
    parser.add_argument("--recovery-seal", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be absent")

    source_root = args.source_root.resolve()
    identity, identity_raw = load_json(args.identity)
    validate_source_identity(identity, source_root)
    identity_hash = sha256_bytes(identity_raw)
    accepted_identity, accepted_identity_raw = load_json(args.accepted_identity)
    accepted_identity_hash, accepted_commit = require_accepted_identity(
        accepted_identity, accepted_identity_raw, identity["source_commit"]
    )

    provenance, provenance_raw = load_json(args.provenance)
    suite, suite_raw = load_json(args.spec_suite)
    require(
        provenance.get("record_type") == "ts-c58-r-provenance"
        and provenance.get("status") == "pass"
        and provenance.get("source_commit") == identity["source_commit"]
        and provenance.get("source_identity_sha256") == identity_hash,
        "candidate provenance differs",
    )
    require(
        suite.get("record_type") == "ts-c58-r-execution-spec-suite"
        and suite.get("status") == "pass"
        and suite.get("campaign") == "r1-repair"
        and suite.get("source_commit") == identity["source_commit"]
        and suite.get("source_identity_sha256") == identity_hash
        and tuple(suite.get("cell_order", [])) == CELLS,
        "R1 spec suite differs",
    )
    require(
        provenance.get("execution_spec_suite_sha256") == sha256_bytes(suite_raw),
        "provenance does not bind R1 suite",
    )
    require(
        suite.get("target_uuid") == provenance.get("target_uuid")
        and suite.get("device_index") == provenance.get("device_index"),
        "provenance and R1 suite CUDA targets differ",
    )
    tool_hashes = provenance.get("tool_hashes", {})
    for name in ("analyze_r1_results.py", "capture_disassembly.py", "make_r1_specs.py"):
        require(
            tool_hashes.get(name) == sha256_file(Path(__file__).resolve().with_name(name)),
            f"provenance does not bind {name}",
        )

    baseline_m128, baseline_m128_raw = load_json(args.baseline_m128)
    baseline_m128_seal, baseline_m128_seal_raw = load_json(args.baseline_m128_seal)
    baseline_dense, baseline_dense_raw = load_json(args.baseline_dense)
    baseline_dense_seal, baseline_dense_seal_raw = load_json(args.baseline_dense_seal)
    require_baseline(
        baseline_m128,
        baseline_m128_raw,
        baseline_m128_seal,
        baseline_m128_seal_raw,
        accepted_identity_hash,
        accepted_commit,
        "m128",
        "accepted-target-unsanitized",
    )
    require_baseline(
        baseline_dense,
        baseline_dense_raw,
        baseline_dense_seal,
        baseline_dense_seal_raw,
        accepted_identity_hash,
        accepted_commit,
        "dense",
        "dense-control-unsanitized",
    )

    candidate_m128, candidate_m128_raw = load_json(args.candidate_m128)
    candidate_dense, candidate_dense_raw = load_json(args.candidate_dense)
    for arm, candidate, baseline in (
        ("m128", candidate_m128, baseline_m128),
        ("dense", candidate_dense, baseline_dense),
    ):
        require(
            candidate.get("record_type") == "ts-c58-r-decode-oracle"
            and candidate.get("status") == "pass"
            and candidate.get("arm") == arm
            and candidate.get("mode") == "unsanitized"
            and candidate.get("source_commit") == identity["source_commit"]
            and candidate.get("source_identity_sha256") == identity_hash,
            f"candidate {arm} oracle differs",
        )
        require(candidate.get("cases") == baseline.get("cases"),
                f"candidate {arm} output/LSE hashes drift from accepted baseline")

    execution = load_seals(args.execution_seal, "ts-c58-r-execution-seal", "cell_id")
    require(set(execution) == set(CELLS), "R1 execution-cell set differs")
    for cell in CELLS:
        seal, _ = execution[cell]
        expected_tool = SANITIZER_CELLS.get(cell)
        require(
            seal.get("status") == "pass"
            and seal.get("actual_outcome") == "clean"
            and seal.get("source_commit") == identity["source_commit"]
            and seal.get("source_identity_sha256") == identity_hash
            and seal.get("sanitizer_tool") == expected_tool
            and seal.get("execution_spec_sha256") == suite["spec_sha256s"][cell],
            f"{cell} execution seal differs",
        )
        require(
            seal.get("target_uuid") == provenance.get("target_uuid")
            and seal.get("device_index") == provenance.get("device_index")
            and seal.get("runner_sha256")
            == tool_hashes.get("run_compute_sanitizer.py")
            and seal.get("sealer_sha256")
            == tool_hashes.get("seal_sanitizer_result.py"),
            f"{cell} execution tool or target identity differs",
        )
    require(
        execution["accepted-target-unsanitized"][0].get("result_sha256")
        == sha256_bytes(candidate_m128_raw),
        "candidate M128 oracle is not execution-bound",
    )
    require(
        execution["dense-control-unsanitized"][0].get("result_sha256")
        == sha256_bytes(candidate_dense_raw),
        "candidate dense oracle is not execution-bound",
    )

    recoveries = load_seals(
        args.recovery_seal, "ts-c58-r-gpu-recovery-seal", "phase_id"
    )
    require(set(recoveries) == set(SANITIZER_CELLS), "R1 recovery-cell set differs")
    for cell in SANITIZER_CELLS:
        require_zero_recovery(
            recoveries[cell][0],
            execution[cell][1],
            cell,
            target_uuid=provenance["target_uuid"],
            capture_sha256=tool_hashes["capture_gpu_health.py"],
            sealer_sha256=tool_hashes["seal_gpu_recovery.py"],
        )

    m128_disassembly, m128_disassembly_raw = load_json(args.m128_disassembly)
    dense_disassembly, dense_disassembly_raw = load_json(args.dense_disassembly)
    require(
        execution["m128-disassembly"][0].get("result_sha256")
        == sha256_bytes(m128_disassembly_raw),
        "M128 disassembly is not execution-bound",
    )
    require(
        execution["dense-disassembly"][0].get("result_sha256")
        == sha256_bytes(dense_disassembly_raw),
        "dense disassembly is not execution-bound",
    )
    disassembly_summary = {
        "m128": require_disassembly(m128_disassembly, "m128"),
        "dense": require_disassembly(dense_disassembly, "dense"),
    }

    value = {
        "schema_version": 1,
        "record_type": "ts-c58-r1-analysis",
        "status": "pass",
        "decision": "accept_r1_candidate",
        "source_commit": identity["source_commit"],
        "source_identity_sha256": identity_hash,
        "accepted_source_commit": accepted_commit,
        "accepted_source_identity_sha256": accepted_identity_hash,
        "provenance_sha256": sha256_bytes(provenance_raw),
        "spec_suite_sha256": sha256_bytes(suite_raw),
        "baseline_oracle_sha256s": {
            "m128": sha256_bytes(baseline_m128_raw),
            "dense": sha256_bytes(baseline_dense_raw),
        },
        "candidate_oracle_sha256s": {
            "m128": sha256_bytes(candidate_m128_raw),
            "dense": sha256_bytes(candidate_dense_raw),
        },
        "execution_seal_sha256s": {
            cell: sha256_bytes(execution[cell][1]) for cell in CELLS
        },
        "recovery_seal_sha256s": {
            cell: sha256_bytes(recoveries[cell][1]) for cell in sorted(recoveries)
        },
        "disassembly": disassembly_summary,
        "output_lse_hashes_match_accepted": {"m128": True, "dense": True},
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_exclusive(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
