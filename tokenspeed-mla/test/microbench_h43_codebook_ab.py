#!/usr/bin/env python3
"""H43 same-residency A/B of the post-wait TQ4 codebook reader.

This process produces raw correctness, telemetry, A/A-pilot, or paired A/B
evidence. It never decides whether D1 passes; the two-stage analyzer owns that
decision after validating the complete campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from h43_codebook_ab_common import (
    canonical_json_digest,
    compiled_artifact_manifest,
    compute_apps,
    gpu_covariates,
    load_contract,
    sha256_file,
    telemetry_reasons,
)
from tokenspeed_mla import tokenspeed_mla_decode, tokenspeed_mla_decode_tq4
from tokenspeed_mla.mla_decode import _get_compiled_mla_kernel

E2M1_CENTROIDS = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("smoke", "preflight", "decision"), required=True
    )
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest-digest", required=True)
    parser.add_argument("--installed-mla-sha256", required=True)
    parser.add_argument("--aggregate-ecc-baseline", required=True)
    parser.add_argument("--expected-cache-digest")
    return parser.parse_args()


def effective_nonempty_split(context: int, requested_split: int) -> int:
    tiles = math.ceil(context / 128)
    return math.ceil(tiles / math.ceil(tiles / requested_split))


def cache_delta(before: Any, after: Any) -> dict[str, int]:
    return {
        "hits": after.hits - before.hits,
        "misses": after.misses - before.misses,
        "currsize": after.currsize - before.currsize,
    }


def timed_replays(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def capture_graph(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def cpu_codebook_reference(
    scales: torch.Tensor, centroids: torch.Tensor
) -> tuple[torch.Tensor, str]:
    scale_cpu = scales.detach().cpu().to(torch.float32)
    centroid_cpu = centroids.detach().cpu().to(torch.float32)
    reference = (
        (scale_cpu[..., None] * centroid_cpu)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .contiguous()
    )
    digest = hashlib.sha256(reference.numpy().tobytes()).hexdigest()
    return reference, digest


class Experiment:
    def __init__(self, contract: dict[str, Any], context: int, sequence: int):
        self.contract = contract
        self.geometry = contract["geometry"]
        self.timing = contract["timing"]
        self.context_contract = contract["contexts"][str(context)]
        self.context = context
        self.sequence = sequence
        self.device = torch.device("cuda", contract["machine"]["gpu_index"])
        torch.cuda.set_device(self.device)
        self.fp8 = torch.float8_e4m3fn
        self.latent = self.geometry["latent_dim"]
        self.rope_dim = self.geometry["rope_dim"]
        self.page_size = self.geometry["page_size"]
        self.cache_rows = self.geometry["cache_rows"]
        self.pages = self.cache_rows // self.page_size
        self.selected_layers = self.geometry["selected_layers"]
        self.dense_layers = (
            self.geometry["total_layers"] - self.geometry["selected_layers"]
        )
        self.requested_split = self.context_contract["requested_split"]
        observed_effective = effective_nonempty_split(context, self.requested_split)
        if observed_effective != self.context_contract["effective_nonempty_split"]:
            raise RuntimeError(
                f"effective split mismatch: {observed_effective} != "
                f"{self.context_contract['effective_nonempty_split']}"
            )

        free_bytes, self.total_device_bytes = torch.cuda.mem_get_info(self.device)
        if free_bytes < (20 << 30):
            raise RuntimeError(
                f"H43 requires 20 GiB free, found {free_bytes / 2**30:.2f}"
            )
        self.free_bytes_before_allocations = free_bytes

        content_seed = contract["seeds"]["content"]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(content_seed)
        self.centroids = torch.tensor(
            E2M1_CENTROIDS, device=self.device, dtype=torch.float32
        )

        self.dense = torch.empty(
            self.dense_layers,
            self.pages,
            self.page_size,
            self.latent + self.rope_dim,
            device=self.device,
            dtype=self.fp8,
        )
        dense_scratch = torch.empty(
            self.pages,
            self.page_size,
            self.latent + self.rope_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        for layer in range(self.dense_layers):
            dense_scratch.uniform_(-0.125, 0.125, generator=generator)
            self.dense[layer].copy_(dense_scratch)
        del dense_scratch

        self.packed = torch.empty(
            self.selected_layers,
            self.pages,
            self.page_size,
            self.latent // 2,
            device=self.device,
            dtype=torch.uint8,
        )
        self.packed.random_(0, 256, generator=generator)
        self.scales = torch.empty(
            self.selected_layers,
            self.pages,
            self.page_size,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.scales.uniform_(
            contract["correctness"]["normal_scale_min"],
            contract["correctness"]["normal_scale_max"],
            generator=generator,
        )
        rope_bf16 = torch.empty(
            self.selected_layers,
            self.pages,
            self.page_size,
            self.rope_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        rope_bf16.normal_(0.0, 0.1, generator=generator)
        self.rope = rope_bf16.to(self.fp8)
        del rope_bf16
        self.codebooks = (
            (self.scales.float()[..., None] * self.centroids)
            .to(self.fp8)
            .view(torch.uint8)
            .contiguous()
        )
        if self.codebooks.data_ptr() % 16:
            raise AssertionError("codebook allocation is not 16-byte aligned")
        cpu_reference, self.codebook_sha256 = cpu_codebook_reference(
            self.scales, self.centroids
        )
        observed_codebooks = self.codebooks.detach().cpu()
        if not torch.equal(observed_codebooks, cpu_reference):
            raise AssertionError("GPU codebooks differ from independent CPU-FP32 bytes")
        del observed_codebooks, cpu_reference

        page_generator = torch.Generator(device=self.device)
        page_generator.manual_seed(contract["seeds"]["page_table_base"] + sequence)
        self.page_table = torch.randperm(
            self.pages,
            device=self.device,
            dtype=torch.int32,
            generator=page_generator,
        ).view(1, self.pages)
        page_table_cpu = self.page_table.detach().cpu()
        if not torch.equal(
            torch.sort(page_table_cpu.flatten()).values,
            torch.arange(self.pages, dtype=torch.int32),
        ):
            raise AssertionError(
                "page table is not a complete physical-page permutation"
            )
        self.page_table_sha256 = hashlib.sha256(
            page_table_cpu.numpy().tobytes()
        ).hexdigest()
        del page_table_cpu
        self.seq_lens = torch.full((1,), context, device=self.device, dtype=torch.int32)
        self.workspace = torch.empty(64 << 20, device=self.device, dtype=torch.int8)
        self.softmax_scale = 1.0 / math.sqrt(self.latent + self.rope_dim)
        query_generator = torch.Generator(device=self.device)
        query_generator.manual_seed(content_seed + 991)
        self.queries = {
            q_len: torch.empty(
                1,
                q_len,
                self.geometry["heads"],
                self.latent + self.rope_dim,
                device=self.device,
                dtype=torch.bfloat16,
            )
            .uniform_(-0.125, 0.125, generator=query_generator)
            .to(self.fp8)
            for q_len in self.geometry["correctness_query_lengths"]
        }
        self.query_sha256 = {
            str(q_len): hashlib.sha256(
                query.detach().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()
            for q_len, query in self.queries.items()
        }
        self.reference_dense = torch.empty(
            self.pages,
            self.page_size,
            self.latent + self.rope_dim,
            device=self.device,
            dtype=self.fp8,
        )
        self._reconstruct_dense_reference()
        self.compile_evidence: list[dict[str, Any]] = []

    def _reconstruct_dense_reference(self, chunk_pages: int = 256) -> None:
        codebook = self.codebooks[0].view(self.fp8)
        for first in range(0, self.pages, chunk_pages):
            last = min(first + chunk_pages, self.pages)
            packed = self.packed[0, first:last]
            indices = torch.empty(
                (*packed.shape[:-1], self.latent),
                device=self.device,
                dtype=torch.uint8,
            )
            indices[..., 0::2] = packed & 0x0F
            indices[..., 1::2] = packed >> 4
            values = torch.gather(
                codebook[first:last].float(), -1, indices.to(torch.long)
            )
            self.reference_dense[first:last, ..., : self.latent].copy_(values)
            self.reference_dense[first:last, ..., self.latent :].copy_(
                self.rope[0, first:last]
            )

    def output(self, q_len: int) -> torch.Tensor:
        return torch.empty(
            1,
            q_len,
            self.geometry["heads"],
            self.latent,
            device=self.device,
            dtype=torch.bfloat16,
        )

    def dense_call(
        self, cache: torch.Tensor, query: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return tokenspeed_mla_decode(
            query=query,
            kv_cache=cache,
            workspace_buffer=self.workspace,
            kv_lora_rank=self.latent,
            qk_rope_head_dim=self.rope_dim,
            block_tables=self.page_table,
            seq_lens=self.seq_lens,
            max_seq_len=self.cache_rows,
            softmax_scale=self.softmax_scale,
            out=output,
            causal_mask=True,
            enable_pdl=True,
        )

    def tq_call(
        self,
        layer: int,
        query: torch.Tensor,
        output: torch.Tensor,
        *,
        use_codebook: bool,
    ) -> torch.Tensor:
        return tokenspeed_mla_decode_tq4(
            query=query,
            kv_nope_packed=self.packed[layer],
            kv_nope_scale=self.scales[layer],
            kv_rope=self.rope[layer],
            centroids=self.centroids,
            workspace_buffer=self.workspace,
            kv_lora_rank=self.latent,
            qk_rope_head_dim=self.rope_dim,
            block_tables=self.page_table,
            seq_lens=self.seq_lens,
            max_seq_len=self.cache_rows,
            softmax_scale=self.softmax_scale,
            out=output,
            causal_mask=True,
            enable_pdl=True,
            split_kv_override=self.requested_split,
            kv_nope_codebook=self.codebooks[layer] if use_codebook else None,
            fp8_rope=True,
        )

    def _first_call(self, label: str, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        before = _get_compiled_mla_kernel.cache_info()
        result = fn()
        torch.cuda.synchronize()
        after = _get_compiled_mla_kernel.cache_info()
        delta = cache_delta(before, after)
        if delta["misses"] != 1 or delta["currsize"] != 1:
            raise AssertionError(f"{label} dispatch miss delta is not one: {delta}")
        self.compile_evidence.append({"label": label, **delta})
        return result

    def correctness(self) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        atol = self.contract["correctness"]["dense_oracle_atol"]
        for q_len in self.geometry["correctness_query_lengths"]:
            query = self.queries[q_len]
            dense_output = self.output(q_len)
            no_outputs = [self.output(q_len) for _ in range(self.selected_layers)]
            code_outputs = [self.output(q_len) for _ in range(self.selected_layers)]
            self._first_call(
                f"dense-q{q_len}",
                lambda: self.dense_call(self.reference_dense, query, dense_output),
            )
            self._first_call(
                f"tq-no-codebook-q{q_len}",
                lambda: self.tq_call(0, query, no_outputs[0], use_codebook=False),
            )
            self._first_call(
                f"tq-codebook-q{q_len}",
                lambda: self.tq_call(0, query, code_outputs[0], use_codebook=True),
            )
            for layer in range(1, self.selected_layers):
                self.tq_call(layer, query, no_outputs[layer], use_codebook=False)
                self.tq_call(layer, query, code_outputs[layer], use_codebook=True)
            torch.cuda.synchronize()
            eager_equal = [
                torch.equal(no_outputs[layer], code_outputs[layer])
                for layer in range(self.selected_layers)
            ]
            if not all(eager_equal):
                raise AssertionError(f"q{q_len} eager arm outputs differ")

            def run_no() -> None:
                for layer in range(self.selected_layers):
                    self.tq_call(layer, query, no_outputs[layer], use_codebook=False)

            def run_codebook() -> None:
                for layer in range(self.selected_layers):
                    self.tq_call(layer, query, code_outputs[layer], use_codebook=True)

            no_graph = capture_graph(run_no)
            code_graph = capture_graph(run_codebook)
            allocated_before = torch.cuda.memory_allocated(self.device)
            for _ in range(5):
                no_graph.replay()
                code_graph.replay()
            torch.cuda.synchronize()
            allocated_after = torch.cuda.memory_allocated(self.device)
            if allocated_before != allocated_after:
                raise AssertionError("correctness graph replay allocated memory")
            graph_equal = [
                torch.equal(no_outputs[layer], code_outputs[layer])
                for layer in range(self.selected_layers)
            ]
            if not all(graph_equal):
                raise AssertionError(f"q{q_len} graph arm outputs differ")
            if any(
                not bool(torch.isfinite(output.float()).all())
                for output in [dense_output, *no_outputs, *code_outputs]
            ):
                raise AssertionError(f"q{q_len} produced non-finite output")
            dense_differences = [
                float((output.float() - dense_output.float()).abs().max().item())
                for output in (no_outputs[0], code_outputs[0])
            ]
            if max(dense_differences) > atol:
                raise AssertionError(
                    f"q{q_len} dense oracle difference {max(dense_differences)} > {atol}"
                )
            cases.append(
                {
                    "q_len": q_len,
                    "eager_all_14_bit_identical": all(eager_equal),
                    "graph_all_14_bit_identical": all(graph_equal),
                    "dense_oracle_layer": 0,
                    "dense_max_abs_differences": dense_differences,
                    "graph_replay_allocation_before": allocated_before,
                    "graph_replay_allocation_after": allocated_after,
                }
            )
        return cases

    def ring_graphs(self) -> tuple[dict[str, torch.cuda.CUDAGraph], dict[str, Any]]:
        q_len = self.geometry["query_length"]
        query = self.queries[q_len]
        dense_output = self.output(q_len)
        outputs = {
            "no_codebook": [self.output(q_len) for _ in range(self.selected_layers)],
            "codebook": [self.output(q_len) for _ in range(self.selected_layers)],
        }

        def ring(arm: str) -> None:
            use_codebook = arm == "codebook"
            for layer in range(self.geometry["dense_before"]):
                self.dense_call(self.dense[layer], query, dense_output)
            for layer in range(self.selected_layers):
                self.tq_call(
                    layer,
                    query,
                    outputs[arm][layer],
                    use_codebook=use_codebook,
                )
            for layer in range(self.geometry["dense_before"], self.dense_layers):
                self.dense_call(self.dense[layer], query, dense_output)

        arm_order = (
            ("no_codebook", "codebook")
            if self.sequence % 2
            else ("codebook", "no_codebook")
        )
        eager_count = self.timing["eager_pre_capture_invocations"]
        graphs: dict[str, torch.cuda.CUDAGraph] = {}
        for arm in arm_order:
            for _ in range(eager_count):
                ring(arm)
            torch.cuda.synchronize()
            graphs[arm] = capture_graph(lambda arm=arm: ring(arm))
            for _ in range(self.timing["post_capture_warmups"]):
                graphs[arm].replay()
            torch.cuda.synchronize()

        allocation_before = torch.cuda.memory_allocated(self.device)
        for _ in range(self.timing["allocation_check_replays"]):
            graphs["no_codebook"].replay()
            graphs["codebook"].replay()
        torch.cuda.synchronize()
        allocation_after = torch.cuda.memory_allocated(self.device)
        if allocation_before != allocation_after:
            raise AssertionError("decision graph replay allocated memory")
        self._assert_ring_outputs(outputs)
        return graphs, {
            "outputs": outputs,
            "capture_order": list(arm_order),
            "allocation_before": allocation_before,
            "allocation_after": allocation_after,
        }

    def codebook_probe_graph(self) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor]]:
        """Capture the exact q5 61-layer codebook ring without timing warmups."""
        q_len = self.geometry["query_length"]
        query = self.queries[q_len]
        dense_output = self.output(q_len)
        outputs = [self.output(q_len) for _ in range(self.selected_layers)]

        def ring() -> None:
            for layer in range(self.geometry["dense_before"]):
                self.dense_call(self.dense[layer], query, dense_output)
            for layer in range(self.selected_layers):
                self.tq_call(layer, query, outputs[layer], use_codebook=True)
            for layer in range(self.geometry["dense_before"], self.dense_layers):
                self.dense_call(self.dense[layer], query, dense_output)

        graph = capture_graph(ring)
        graph.replay()
        torch.cuda.synchronize()
        if any(not bool(torch.isfinite(output.float()).all()) for output in outputs):
            raise AssertionError("codebook probe graph produced non-finite output")
        return graph, outputs

    def _assert_ring_outputs(self, outputs: dict[str, list[torch.Tensor]]) -> None:
        for layer in range(self.selected_layers):
            no_output = outputs["no_codebook"][layer]
            code_output = outputs["codebook"][layer]
            if not bool(torch.isfinite(no_output.float()).all()):
                raise AssertionError(f"no-codebook layer {layer} is non-finite")
            if not bool(torch.isfinite(code_output.float()).all()):
                raise AssertionError(f"codebook layer {layer} is non-finite")
            if not torch.equal(no_output, code_output):
                raise AssertionError(f"ring layer {layer} arm outputs differ")


def telemetry_pair(contract: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    sample = gpu_covariates(contract)
    return sample, telemetry_reasons(sample, contract)


def sentinel(graph: torch.cuda.CUDAGraph, contract: dict[str, Any]) -> dict[str, Any]:
    before, before_reasons = telemetry_pair(contract)
    duration = timed_replays(graph, contract["timing"]["sentinel_replays"])
    after, after_reasons = telemetry_pair(contract)
    return {
        "ring_us": duration,
        "telemetry_before": before,
        "telemetry_after": after,
        "telemetry_reasons": before_reasons + after_reasons,
    }


def paired_samples(
    graphs: dict[str, torch.cuda.CUDAGraph],
    contract: dict[str, Any],
    sequence: int,
    *,
    aa_only: bool,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for pair_index in range(contract["timing"]["pairs_per_process"]):
        no_first = (pair_index % 2 == 0) == (sequence % 2 == 1)
        order = ["no_codebook", "codebook"] if no_first else ["codebook", "no_codebook"]
        if aa_only:
            order = (
                ["no_codebook_a", "no_codebook_b"]
                if no_first
                else ["no_codebook_b", "no_codebook_a"]
            )
        before, before_reasons = telemetry_pair(contract)
        durations: dict[str, float] = {}
        for arm in order:
            graph_key = "no_codebook" if aa_only else arm
            durations[arm] = timed_replays(
                graphs[graph_key], contract["timing"]["replays_per_sample"]
            )
        after, after_reasons = telemetry_pair(contract)
        reasons = before_reasons + after_reasons
        pair: dict[str, Any] = {
            "pair_index": pair_index,
            "order": order,
            "durations_us": durations,
            "telemetry_before": before,
            "telemetry_after": after,
            "telemetry_reasons": reasons,
            "telemetry_valid": not reasons,
        }
        if aa_only:
            pair["aa_recovery_us_per_selected_layer"] = (
                durations["no_codebook_a"] - durations["no_codebook_b"]
            ) / contract["geometry"]["selected_layers"]
        pairs.append(pair)
    return pairs


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract.resolve())
    contract_digest = canonical_json_digest(contract)
    if str(args.context) not in contract["contexts"]:
        raise ValueError(f"context is not in contract: {args.context}")
    if not 1 <= args.sequence <= contract["seeds"]["max_sequence"]:
        raise ValueError("sequence is outside the contract")
    if os.environ.get("H43_PHYSICAL_HOST") != contract["machine"]["physical_host"]:
        raise RuntimeError("H43_PHYSICAL_HOST does not identify ct13")
    if platform.node() != contract["machine"]["container_hostname"]:
        raise RuntimeError(
            f"container hostname {platform.node()!r} is not "
            f"{contract['machine']['container_hostname']!r}"
        )

    imported_mla = Path(
        __import__("tokenspeed_mla.mla_decode_fp8", fromlist=["x"]).__file__
    ).resolve()
    imported_sha256 = sha256_file(imported_mla)
    if imported_sha256 != args.installed_mla_sha256:
        raise RuntimeError("installed mla_decode_fp8.py digest does not match manifest")
    cache_before = compiled_artifact_manifest(args.cache_root, contract["cache"])
    if (
        args.expected_cache_digest
        and cache_before["digest"] != args.expected_cache_digest
    ):
        raise RuntimeError(
            "compiled-artifact digest differs before CUDA initialization"
        )
    gpu_before_cuda = gpu_covariates(contract)
    applications_before_cuda = compute_apps(contract)
    if applications_before_cuda:
        raise RuntimeError(
            f"GPU has compute applications before CUDA init: {applications_before_cuda}"
        )

    started = time.monotonic()
    experiment = Experiment(contract, args.context, args.sequence)
    if torch.cuda.get_device_name(experiment.device) != contract["machine"]["gpu_name"]:
        raise RuntimeError("unexpected CUDA device")
    correctness = experiment.correctness()

    base: dict[str, Any] = {
        "schema_version": 1,
        "status": args.mode.upper(),
        "experiment": contract["experiment"],
        "mode": args.mode,
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "physical_host": os.environ["H43_PHYSICAL_HOST"],
        "container_hostname": platform.node(),
        "device_name": torch.cuda.get_device_name(experiment.device),
        "device_uuid": gpu_before_cuda["uuid"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "contract_digest": contract_digest,
        "source_manifest_digest": args.source_manifest_digest,
        "installed_mla_path": str(imported_mla),
        "installed_mla_sha256": imported_sha256,
        "cache_artifacts_before": cache_before,
        "gpu_before_cuda": gpu_before_cuda,
        "applications_before_cuda": applications_before_cuda,
        "aggregate_ecc_baseline": args.aggregate_ecc_baseline,
        "context": args.context,
        "requested_split": experiment.requested_split,
        "effective_nonempty_split": experiment.context_contract[
            "effective_nonempty_split"
        ],
        "tq4_tiles_per_split": experiment.context_contract["tq4_tiles_per_split"],
        "sequence": args.sequence,
        "page_table_seed": contract["seeds"]["page_table_base"] + args.sequence,
        "page_table_sha256": experiment.page_table_sha256,
        "query_sha256": experiment.query_sha256,
        "codebook_sha256": experiment.codebook_sha256,
        "correctness": correctness,
        "compile_evidence": experiment.compile_evidence,
        "free_bytes_before_allocations": experiment.free_bytes_before_allocations,
        "total_device_bytes": experiment.total_device_bytes,
        "memory": contract["memory_bytes_per_rank"],
    }

    if args.mode == "smoke":
        base["gpu_final"] = gpu_covariates(contract)
    else:
        graphs, graph_state = experiment.ring_graphs()
        base.update(
            {
                "capture_order": graph_state["capture_order"],
                "graph_replay_allocation_before": graph_state["allocation_before"],
                "graph_replay_allocation_after": graph_state["allocation_after"],
            }
        )
        if args.mode == "preflight":
            base["aa_pairs"] = paired_samples(
                graphs, contract, args.sequence, aa_only=True
            )
        else:
            base["sentinel_before"] = sentinel(graphs["no_codebook"], contract)
            base["pairs"] = paired_samples(
                graphs, contract, args.sequence, aa_only=False
            )
            base["sentinel_after"] = sentinel(graphs["no_codebook"], contract)
        experiment._assert_ring_outputs(graph_state["outputs"])
        base["selected_output_checksums"] = {
            arm: [float(output.float().sum().item()) for output in outputs]
            for arm, outputs in graph_state["outputs"].items()
        }
        base["gpu_final"] = gpu_covariates(contract)

    torch.cuda.synchronize()
    cache_after = compiled_artifact_manifest(args.cache_root, contract["cache"])
    if cache_after["digest"] != cache_before["digest"]:
        raise RuntimeError(
            "runtime dispatch changed the immutable prebuilt artifact cache"
        )
    base["cache_artifacts_after"] = cache_after
    base["wall_time_seconds"] = time.monotonic() - started
    base["raw_result_digest"] = canonical_json_digest(base)
    print(json.dumps(base, allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
