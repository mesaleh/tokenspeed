# Copyright (c) 2026 LightSeek Foundation

"""Packed FP32x2 probability-scale candidate in the E2M1 reader.

This research-only harness compiles full and packed specializations from one
source and invokes them with identical runtime pointers.  Both retain the
accepted nonuniform Gather4 scale pipeline and both scale applications. The
packed arm changes only P's ordered ``(p * scale) * carrier_reciprocal`` into
two SM100 ``mul_packed_f32x2`` operations; QK, delivery, carrier selection,
barriers, and every accepted MMA/pipeline contract remain scalar-identical.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from benchmark_mla_decode_tq_e2m1_public import HEADS, LATENT, PAGE_SIZE, ROPE
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack
from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
)
from tokenspeed_mla.mla_helpers import get_mla_decode_fold_sq_factor
from tokenspeed_mla.utils import get_max_active_clusters, get_num_sm

SCHEMA_VERSION = 4
PRIMARY_SEQ_LEN = 10_219
PRIMARY_PHYSICAL_SEQ_LEN = 10_240
TRANSFER_SEQ_LEN = 32_768
SCORE_CELLS = {
    "primary": PRIMARY_SEQ_LEN,
    "transfer": TRANSFER_SEQ_LEN,
}
QUERY_LENGTHS = (1, 5)
DEFAULT_WINDOWS = 30
DEFAULT_REPLAYS = 500
DEFAULT_WARMUP_WINDOWS = 12
GRAPH_PARITY_REPLAYS = 100
POISON_PACKED = 0xFF
POISON_ROPE = 64.0
POISON_UNUSED_SCALE = 16.0
SCALE_VALUES = (0.5, 0.75, 1.0)
SCALE_TILE = 128
EXPECTED_R4_FULL_SASS_BY_QUERY = {
    1: "5ef06bbba7ae3dd52279a7b5ad8c6da0899d11dfbf6c2475da37d25c29cb370a",
    5: "d9bcb84104c25e8aa5e16cb678b327f2ebfd6ceaeee83cb9846a120b8f66f916",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    data = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return _sha256_bytes(data)


def _as_cute_tensor(tensor: torch.Tensor, dtype, leading_dim: int, align: int):
    result = from_dlpack(tensor, assumed_align=align, enable_tvm_ffi=True)
    result.element_type = dtype
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _run_text(command: list[str], *, check: bool = True) -> str:
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_deterministic_hash_seed(
    timing: bool,
    *,
    hash_randomization: int | None = None,
) -> str | None:
    """Fail closed before scored JIT compilation can inherit a random hash seed."""

    seed = os.environ.get("PYTHONHASHSEED")
    if hash_randomization is None:
        hash_randomization = sys.flags.hash_randomization
    if timing and (seed != "0" or hash_randomization != 0):
        raise ValueError(
            "scored R7 timing requires PYTHONHASHSEED=0 before interpreter startup"
        )
    return seed


def _gpu_state(stage: str) -> dict[str, Any]:
    fields = (
        "index,uuid,driver_version,pstate,clocks.sm,clocks.mem,"
        "ecc.errors.uncorrected.volatile.total"
    )
    rows = [
        row.split(", ")
        for row in _run_text(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
    ]
    expected_sm = os.environ.get("TQ_PACKED_SCALE_EXPECT_SM_CLOCK", "1965")
    expected_mem = os.environ.get("TQ_PACKED_SCALE_EXPECT_MEM_CLOCK", "4000")
    for row in rows:
        if row[3] != "P0" or int(row[6]) != 0:
            raise RuntimeError(f"GPU state is not scoreable at {stage}: {row}")
        if expected_sm and row[4] != expected_sm:
            raise RuntimeError(
                f"unexpected SM clock at {stage}: {row[4]} != {expected_sm}"
            )
        if expected_mem and row[5] != expected_mem:
            raise RuntimeError(
                f"unexpected memory clock at {stage}: {row[5]} != {expected_mem}"
            )
    compute_apps = _run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    compute_rows = [
        row
        for row in compute_apps.splitlines()
        if row.strip() and "No running processes found" not in row
    ]
    if len(compute_rows) > 1:
        raise RuntimeError(
            f"GPU allocation is not exclusive at {stage}: {compute_rows}"
        )
    result = {"stage": stage, "rows": rows, "compute_apps": compute_apps}
    print(
        "TQ_PACKED_SCALE_GPU_STATE " + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def _runtime_identity() -> dict[str, Any]:
    git_commit = _run_text(["git", "rev-parse", "HEAD"])
    git_status = _run_text(["git", "status", "--short"])
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cutlass": getattr(cutlass, "__version__", "unknown"),
        "git_commit": git_commit,
        "git_status": git_status,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "python_hash_randomization": sys.flags.hash_randomization,
    }


def _audit_scale_fixture(
    scale: torch.Tensor,
    table_ids: list[int],
    seq_len: int,
) -> dict[str, Any]:
    """Validate logical paged scales independently of the device kernel."""

    if scale.ndim != 2 or scale.shape[1] != PAGE_SIZE:
        raise AssertionError("scale plane must be [physical_page, PAGE_SIZE]")
    logical_pages = math.ceil(seq_len / PAGE_SIZE)
    if len(table_ids) < logical_pages:
        raise AssertionError("page table does not cover the logical sequence")
    host = scale.detach().cpu()
    logical = [
        float(host[table_ids[token // PAGE_SIZE], token % PAGE_SIZE].item())
        for token in range(seq_len)
    ]
    tile_maxima = [
        max(logical[begin : min(begin + SCALE_TILE, seq_len)])
        for begin in range(0, seq_len, SCALE_TILE)
    ]
    if any(value != 1.0 for value in tile_maxima):
        raise AssertionError(
            f"every logical N128 tile must have max one: {tile_maxima}"
        )
    allowed = set(SCALE_VALUES)
    if set(logical) != allowed:
        raise AssertionError(
            f"logical scale values are not the frozen nonuniform set: {set(logical)}"
        )
    counts = {str(value): logical.count(value) for value in SCALE_VALUES}
    encoded_maxima = json.dumps(tile_maxima, separators=(",", ":")).encode()
    return {
        "allowed_values": list(SCALE_VALUES),
        "value_counts": counts,
        "logical_tokens": seq_len,
        "tile_size": SCALE_TILE,
        "tile_count": len(tile_maxima),
        "tile_minimum_max": min(tile_maxima),
        "tile_maximum_max": max(tile_maxima),
        "tile_maxima_sha256": _sha256_bytes(encoded_maxima),
    }


def _build_fixture(
    query_len: int,
    seq_len: int,
    *,
    permutation_seed: int,
) -> dict[str, torch.Tensor | int | list[int]]:
    if query_len not in QUERY_LENGTHS or seq_len <= 0:
        raise ValueError("packed-arithmetic fixture supports positive K and q1/q5")
    logical_pages = math.ceil(seq_len / PAGE_SIZE)
    table_pages = math.ceil(logical_pages / 4) * 4
    num_pages = table_pages + 4
    device = torch.device("cuda")

    packed = torch.full(
        (num_pages, PAGE_SIZE, LATENT // 2),
        POISON_PACKED,
        dtype=torch.uint8,
        device=device,
    )
    scale = torch.full(
        (num_pages, PAGE_SIZE),
        POISON_UNUSED_SCALE,
        dtype=torch.bfloat16,
        device=device,
    )
    reciprocal_rope = torch.full(
        (num_pages, PAGE_SIZE, ROPE),
        POISON_ROPE,
        dtype=torch.bfloat16,
        device=device,
    )

    physical_ids = list(range(num_pages))
    random.Random(permutation_seed).shuffle(physical_ids)
    table_ids = physical_ids[:table_pages]
    unused_ids = physical_ids[table_pages:]
    block_tables = torch.tensor(
        [table_ids], dtype=torch.int32, device=device
    ).contiguous()

    pair_dim = torch.arange(LATENT // 2, device=device)[None, :]
    rope_dim = torch.arange(ROPE, device=device)[None, :]
    for logical_page in range(logical_pages):
        physical_page = table_ids[logical_page]
        begin = logical_page * PAGE_SIZE
        end = min(begin + PAGE_SIZE, seq_len)
        count = end - begin
        token = torch.arange(begin, end, device=device)[:, None]
        low = ((token * 5 + (pair_dim * 2) * 3 + 1) % 15 + 1).to(torch.uint8)
        high = ((token * 5 + (pair_dim * 2 + 1) * 3 + 1) % 15 + 1).to(torch.uint8)
        packed[physical_page, :count] = low | (high << 4)
        scale[physical_page].fill_(0.5)
        logical_token = torch.arange(begin, end, device=device)
        selector = logical_token % len(SCALE_VALUES)
        logical_scale = torch.where(
            selector == 0,
            torch.tensor(0.5, device=device),
            torch.where(
                selector == 1,
                torch.tensor(0.75, device=device),
                torch.tensor(1.0, device=device),
            ),
        ).to(torch.bfloat16)
        scale[physical_page, :count] = logical_scale
        raw_rope = (((token * 7 + rope_dim * 11 + 3) % 9) - 4) / 2.0
        reciprocal_rope[physical_page, :count] = raw_rope.to(torch.bfloat16)
        # The unused tail keeps poisoned packed/RoPE data; its bounded scale is
        # ignored by the logical-tail predicate in both scored arms.

    # Gather4 may address table-padding pages in a final partial group. Their
    # packed/RoPE data stays poisoned and their masked scale remains bounded.
    for table_index in range(logical_pages, table_pages):
        scale[table_ids[table_index]].fill_(0.5)

    torch.manual_seed(20260817 + query_len * 1_000_003 + permutation_seed)
    query_latent = (
        torch.randint(-4, 5, (1, query_len, HEADS, LATENT), device=device) / 2.0
    ).to(torch.float8_e4m3fn)
    query_rope = (
        torch.randint(-4, 5, (1, query_len, HEADS, ROPE), device=device) / 2.0
    ).to(torch.bfloat16)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    scale_audit = _audit_scale_fixture(scale, table_ids, seq_len)
    if not torch.all(
        scale[torch.tensor(unused_ids, device=device)] == POISON_UNUSED_SCALE
    ).item():
        raise AssertionError("unused physical pages lost their scale poison")
    return {
        "query_latent": query_latent,
        "query_rope": query_rope,
        "packed": packed,
        "scale": scale,
        "reciprocal_rope": reciprocal_rope,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "seq_len": seq_len,
        "logical_pages": logical_pages,
        "table_pages": table_pages,
        "num_pages": num_pages,
        "table_ids": table_ids,
        "unused_ids": unused_ids,
        "scale_audit": scale_audit,
    }


def _workspace_geometry(query_len: int, seq_len: int) -> tuple[int, int]:
    fold = get_mla_decode_fold_sq_factor(HEADS, query_len, 64)
    split = BlackwellMultiHeadLatentAttentionForwardFP8.get_split_kv(
        1,
        query_len // fold,
        seq_len,
        (64, 128),
        get_num_sm(torch.device("cuda")),
        1,
    )
    workspace_size = BlackwellMultiHeadLatentAttentionForwardFP8.get_workspace_size(
        HEADS * fold,
        query_len // fold,
        LATENT,
        1,
        split,
        cutlass.Float32,
    )
    return split, workspace_size


def _compile(
    arm: str,
    tensors: dict[str, Any],
    query_len: int,
    output: torch.Tensor,
    lse: torch.Tensor,
    workspace: torch.Tensor | None,
    split: int,
):
    fold = get_mla_decode_fold_sq_factor(HEADS, query_len, 64)
    kwargs: dict[str, Any] = {
        "acc_dtype": cutlass.Float32,
        "lse_dtype": cutlass.Float32,
        "mma_qk_tiler_mn": (64, 128),
        "mma_pv_tiler_mn": (64, 256),
        "max_active_clusters": get_max_active_clusters(1),
        "page_size": PAGE_SIZE,
        "skip_correction_threshold": 0.0,
        "is_persistent": False,
        "is_var_seq": True,
        "is_var_split_kv": False,
        "fold_sq_factor": fold,
        "is_causal": query_len > 1,
        "num_heads": HEADS,
        "seq_len_q": query_len,
        "cp_world": 1,
        "use_tq_e2m1": True,
        "tq_s1_k_rope_stages": 2,
    }
    if arm == "full":
        kwargs.update(tq_s1_scale_tma=True, tq_s1_scale_stages=3)
    elif arm == "packed":
        kwargs.update(
            tq_s1_scale_tma=True,
            tq_s1_scale_stages=3,
            tq_s1_packed_p_scale_math=True,
        )
    elif arm == "total_ceiling":
        kwargs.update(tq_s1_scale_ceiling=True)
    else:
        raise ValueError(f"unknown arm: {arm}")
    kernel = BlackwellMultiHeadLatentAttentionForwardFP8(**kwargs)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel,
        _as_cute_tensor(tensors["query_latent"], cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(tensors["query_rope"], cutlass.BFloat16, 3, 16),
        _as_cute_tensor(tensors["packed"], cutlass.Uint8, 2, 16),
        _as_cute_tensor(tensors["reciprocal_rope"], cutlass.BFloat16, 2, 16),
        _as_cute_tensor(tensors["block_tables"], cutlass.Int32, 1, 4),
        _as_cute_tensor(output, cutlass.BFloat16, 3, 16),
        _as_cute_tensor(lse, cutlass.Float32, 2, 4),
        (
            _as_cute_tensor(workspace, cutlass.Int8, 0, 32)
            if workspace is not None
            else None
        ),
        Int32(split),
        _as_cute_tensor(tensors["seq_lens"], cutlass.Int32, 0, 4),
        _as_cute_tensor(tensors["seq_lens"], cutlass.Int32, 0, 4),
        None,
        Float32(1.0),
        Float32(1.0),
        stream,
        False,
        _as_cute_tensor(tensors["scale"], cutlass.BFloat16, 1, 16),
        options="--enable-tvm-ffi --opt-level 3",
    )
    return compiled


def _invoke(
    compiled,
    tensors: dict[str, Any],
    output: torch.Tensor,
    lse: torch.Tensor,
    workspace: torch.Tensor | None,
    split: int,
) -> None:
    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        compiled(
            tensors["query_latent"],
            tensors["query_rope"],
            tensors["packed"],
            tensors["reciprocal_rope"],
            tensors["block_tables"],
            output,
            lse,
            workspace,
            Int32(split),
            tensors["seq_lens"],
            tensors["seq_lens"],
            None,
            Float32(1.0 / math.sqrt(LATENT + ROPE)),
            Float32(1.0),
            tensors["scale"],
        )


def _artifact_identity(
    compiled,
    arm: str,
    query_len: int,
    split: int,
    dump_dir: Path,
    enforce_r4_full_sass: bool,
) -> dict[str, Any]:
    artifacts = getattr(compiled, "artifacts", None)
    cubin = getattr(artifacts, "CUBIN", None)
    ptx = getattr(artifacts, "PTX", None)
    sass = getattr(artifacts, "SASS", None)
    if not isinstance(cubin, bytes) or not isinstance(ptx, str):
        raise RuntimeError("R7 requires retained CUBIN/PTX; set CUTE_DSL_KEEP=all")
    dump_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = dump_dir / f"{arm}-q{query_len}.cubin"
    ptx_path = dump_dir / f"{arm}-q{query_len}.ptx"
    cubin_path.write_bytes(cubin)
    ptx_path.write_text(ptx)
    if not isinstance(sass, str):
        sass = _run_text(["cuobjdump", "--dump-sass", str(cubin_path)])
    resource_output = _run_text(["cuobjdump", "--dump-resource-usage", str(cubin_path)])
    sass_path = dump_dir / f"{arm}-q{query_len}.sass.txt"
    resource_path = dump_dir / f"{arm}-q{query_len}.resources.txt"
    sass_path.write_text(sass)
    resource_path.write_text(resource_output)

    owner_marker = "Function : kernel_cutlass_split_kv_kernel_"
    owner_offset = sass.find(owner_marker)
    if owner_offset < 0:
        raise RuntimeError(f"{arm}/q{query_len} split-KV SASS symbol is missing")
    following_function = re.search(r"\n\s*Function :", sass[owner_offset + 1 :])
    owner_end = (
        len(sass)
        if following_function is None
        else owner_offset + 1 + following_function.start()
    )
    owner_sass = sass[owner_offset:owner_end]

    ptx_mma = ptx.count("tcgen05.mma.ws.cta_group::1.kind::f8f6f4")
    ptx_rope_mma = ptx.count("tcgen05.mma.ws.cta_group::1.kind::f16")
    sass_mma = owner_sass.count(" UTCQMMA.WS")
    sass_rope_mma = owner_sass.count(" UTCHMMA.WS")
    mixed_qk = len(re.findall(r"mov\.b32\s+%r\d+, 69211152;", ptx))
    bf16_rope = len(re.findall(r"mov\.b32\s+%r\d+, 69207184;", ptx))
    mixed_pv = len(re.findall(r"mov\.b32\s+%r\d+, 71373840;", ptx))
    ptx_gather4 = ptx.count("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4")
    sass_gather4 = owner_sass.count(" UTMALDG.2D.GATHER4")
    expected_gather4 = 1
    if (
        ptx_mma != 48
        or ptx_rope_mma != 8
        or sass_mma != 48
        or sass_rope_mma != 8
        or mixed_qk != 32
        or bf16_rope != 8
        or mixed_pv != 16
    ):
        raise RuntimeError(
            f"{arm}/q{query_len} MMA audit failed: "
            f"PTX={ptx_mma}/{ptx_rope_mma} "
            f"SASS={sass_mma}/{sass_rope_mma} "
            f"descriptors={mixed_qk}/{bf16_rope}/{mixed_pv}"
        )
    if ptx_gather4 != expected_gather4 or sass_gather4 != expected_gather4:
        raise RuntimeError(
            f"{arm}/q{query_len} Gather4 audit failed: "
            f"expected={expected_gather4} PTX={ptx_gather4} SASS={sass_gather4}"
        )
    if any(
        opcode in ptx for opcode in ("ld.global.u8", "ld.global.s8", "ld.global.b8")
    ):
        raise RuntimeError(f"{arm}/q{query_len} contains a scalar global byte load")
    if re.search(r"\bld\.global\.(?:u16|s16|b16)\b", ptx):
        raise RuntimeError(f"{arm}/q{query_len} contains a scalar BF16 scale load")
    resource_match = re.search(
        r"Function kernel_cutlass_split_kv_kernel_[^\n]*:\s*\n\s*"
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        resource_output,
    )
    if resource_match is None:
        raise RuntimeError(f"{arm}/q{query_len} owner resource record missing")
    registers = int(resource_match.group(1))
    stack = int(resource_match.group(2))
    static_shared = int(resource_match.group(3))
    local = int(resource_match.group(4))
    local_loads = len(re.findall(r"\bLDL\b", owner_sass))
    local_stores = len(re.findall(r"\bSTL\b", owner_sass))
    reciprocal_instructions = len(re.findall(r"\bMUFU\.RCP\b", owner_sass))
    barrier_sync_instructions = len(re.findall(r"\bBAR\.SYNC\b", owner_sass))
    barrier_signatures: dict[str, int] = {}
    for signature in re.findall(
        r"\bBAR\.SYNC(?:\.[A-Z_]+)*\s+([^;]+?)\s*;", owner_sass
    ):
        normalized = " ".join(signature.replace(",", ", ").split())
        barrier_signatures[normalized] = barrier_signatures.get(normalized, 0) + 1
    barrier_signatures = dict(sorted(barrier_signatures.items()))
    ptx_shared_proxy_fences = len(
        re.findall(r"\bfence\.proxy\.async\.shared::cta\b", ptx)
    )
    sass_shared_proxy_fences = len(re.findall(r"\bFENCE\.VIEW\.ASYNC\.S\b", owner_sass))
    sass_tmem_proxy_fences = len(re.findall(r"\bFENCE\.VIEW\.ASYNC\.T\b", owner_sass))
    ptx_carrier_mantissa_masks = ptx.count("8388607")
    ptx_carrier_boundaries = ptx.count("6291456")
    ptx_carrier_exp_biases = ptx.count("-134")
    ptx_packed_fmul = len(re.findall(r"\bmul(?:\.[a-z0-9_]+)*\.f32x2\b", ptx))
    fmul_instructions = len(re.findall(r"\bFMUL\b", owner_sass))
    fmnmx_instructions = len(re.findall(r"\bFMNMX\b", owner_sass))
    fsel_instructions = len(re.findall(r"\bFSEL\b", owner_sass))
    shf_instructions = len(re.findall(r"\bSHF\b", owner_sass))
    lop3_instructions = len(re.findall(r"\bLOP3\b", owner_sass))
    iadd3_instructions = len(re.findall(r"\bIADD3\b", owner_sass))
    shared_loads = len(re.findall(r"\bLDS\b", owner_sass))
    shared_stores = len(re.findall(r"\bSTS\b", owner_sass))
    total_reciprocal_instructions = len(re.findall(r"\bMUFU\.RCP\b", sass))
    expected_reduction_reciprocals = 0 if split == 1 else 2
    if (
        total_reciprocal_instructions - reciprocal_instructions
        != expected_reduction_reciprocals
    ):
        raise RuntimeError(
            f"{arm}/q{query_len} reduction reciprocal audit failed: "
            f"expected={expected_reduction_reciprocals} "
            f"owner={reciprocal_instructions} total={total_reciprocal_instructions}"
        )
    if (
        registers > 168
        or stack != 0
        or local != 0
        or local_loads != 0
        or local_stores != 0
        or reciprocal_instructions > 6
    ):
        raise RuntimeError(
            f"{arm}/q{query_len} resource audit failed: registers={registers} "
            f"stack={stack} local={local} LDL={local_loads} STL={local_stores} "
            f"RCP={reciprocal_instructions}"
        )
    sass_sha256 = _sha256_bytes(sass_path.read_bytes())
    expected_full_sass = (
        EXPECTED_R4_FULL_SASS_BY_QUERY[query_len]
        if arm == "full" and enforce_r4_full_sass
        else None
    )
    if arm == "full" and expected_full_sass and sass_sha256 != expected_full_sass:
        raise RuntimeError(
            f"full/q{query_len} inherited SASS drift: "
            f"{sass_sha256} != {expected_full_sass}"
        )
    return {
        "cubin_sha256": _sha256_bytes(cubin),
        "ptx_sha256": _sha256_bytes(ptx.encode()),
        "cubin_bytes": len(cubin),
        "ptx_bytes": len(ptx.encode()),
        "sass_sha256": sass_sha256,
        "expected_full_sass_sha256": (expected_full_sass if arm == "full" else None),
        "resource_sha256": _sha256_bytes(resource_path.read_bytes()),
        "registers": registers,
        "stack_bytes": stack,
        "static_shared_bytes": static_shared,
        "local_bytes": local,
        "sass_local_loads": local_loads,
        "sass_local_stores": local_stores,
        "sass_reciprocals": reciprocal_instructions,
        "sass_barrier_sync": barrier_sync_instructions,
        "sass_barrier_signatures": barrier_signatures,
        "ptx_shared_proxy_fences": ptx_shared_proxy_fences,
        "sass_shared_proxy_fences": sass_shared_proxy_fences,
        "sass_tmem_proxy_fences": sass_tmem_proxy_fences,
        "ptx_carrier_mantissa_masks": ptx_carrier_mantissa_masks,
        "ptx_carrier_boundaries": ptx_carrier_boundaries,
        "ptx_carrier_exp_biases": ptx_carrier_exp_biases,
        "ptx_packed_fmul": ptx_packed_fmul,
        "sass_fmul": fmul_instructions,
        "sass_fmnmx": fmnmx_instructions,
        "sass_fsel": fsel_instructions,
        "sass_shf": shf_instructions,
        "sass_lop3": lop3_instructions,
        "sass_iadd3": iadd3_instructions,
        "sass_shared_loads": shared_loads,
        "sass_shared_stores": shared_stores,
        "sass_total_reciprocals": total_reciprocal_instructions,
        "expected_reduction_reciprocals": expected_reduction_reciprocals,
        "ptx_mixed_mma": ptx_mma,
        "ptx_rope_mma": ptx_rope_mma,
        "sass_mixed_mma": sass_mma,
        "sass_rope_mma": sass_rope_mma,
        "mixed_qk_descriptors": mixed_qk,
        "bf16_rope_descriptors": bf16_rope,
        "mixed_pv_descriptors": mixed_pv,
        "ptx_gather4": ptx_gather4,
        "sass_gather4": sass_gather4,
        "paths": {
            "cubin": str(cubin_path),
            "ptx": str(ptx_path),
            "sass": str(sass_path),
            "resources": str(resource_path),
        },
    }


def _assert_static_packed_isolation(
    full: dict[str, Any], packed: dict[str, Any]
) -> dict[str, Any]:
    if full["static_shared_bytes"] != packed["static_shared_bytes"]:
        raise AssertionError("packed arm changed the retained shared-memory envelope")
    if packed["sass_fmul"] >= full["sass_fmul"]:
        raise AssertionError("packed arm did not reduce the scalar FMUL family")
    if packed["sass_barrier_sync"] != full["sass_barrier_sync"]:
        raise AssertionError("packed arm changed the accepted barrier count")
    if packed["sass_barrier_signatures"] != full["sass_barrier_signatures"]:
        raise AssertionError("packed arm changed barrier IDs or participant literals")
    for key in (
        "ptx_shared_proxy_fences",
        "sass_shared_proxy_fences",
        "sass_tmem_proxy_fences",
    ):
        if packed[key] != full[key] or full[key] <= 0:
            raise AssertionError(f"packed arm changed retained fence family {key}")
    retained_equal_families = (
        "sass_shared_stores",
        "ptx_carrier_mantissa_masks",
        "ptx_carrier_boundaries",
        "ptx_carrier_exp_biases",
        "sass_reciprocals",
        "sass_total_reciprocals",
    )
    for key in retained_equal_families:
        if packed[key] != full[key]:
            raise AssertionError(f"packed arm changed retained family {key}")
    if not 0 < packed["sass_fmnmx"] <= full["sass_fmnmx"]:
        raise AssertionError("packed arm removed or increased the scale-max family")
    if not 0 < packed["sass_shared_loads"] <= full["sass_shared_loads"]:
        raise AssertionError("packed arm removed or increased shared scale consumption")
    if packed["ptx_packed_fmul"] <= full["ptx_packed_fmul"]:
        raise AssertionError("packed arm did not add packed FP32x2 multiplies")
    return {
        "full_barrier_sync": full["sass_barrier_sync"],
        "packed_barrier_sync": packed["sass_barrier_sync"],
        "retained_barrier_signatures": full["sass_barrier_signatures"],
        "retained_ptx_shared_proxy_fences": full["ptx_shared_proxy_fences"],
        "retained_sass_shared_proxy_fences": full["sass_shared_proxy_fences"],
        "retained_sass_tmem_proxy_fences": full["sass_tmem_proxy_fences"],
        "full_sass_fmul": full["sass_fmul"],
        "packed_sass_fmul": packed["sass_fmul"],
        "removed_sass_fmul": full["sass_fmul"] - packed["sass_fmul"],
        "full_sass_fmnmx": full["sass_fmnmx"],
        "packed_sass_fmnmx": packed["sass_fmnmx"],
        "full_sass_shared_loads": full["sass_shared_loads"],
        "packed_sass_shared_loads": packed["sass_shared_loads"],
        "packed_ptx_f32x2_mul": packed["ptx_packed_fmul"],
        "full_ptx_f32x2_mul": full["ptx_packed_fmul"],
        "retained_equal_families": list(retained_equal_families),
    }


def _capture(call) -> torch.cuda.CUDAGraph:
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return graph


def _measure(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def _assert_finite_written(output: torch.Tensor, lse: torch.Tensor) -> None:
    if not torch.isfinite(output).all().item():
        raise AssertionError("reader left nonfinite data in output")
    if not torch.isfinite(lse).all().item():
        raise AssertionError("reader left nonfinite data in LSE")


def _run_and_snapshot(
    call,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output.fill_(float("nan"))
    lse.fill_(float("nan"))
    call()
    torch.cuda.synchronize()
    _assert_finite_written(output, lse)
    return output.clone(), lse.clone()


def _assert_equal(
    stage: str,
    full: tuple[torch.Tensor, torch.Tensor],
    packed: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, str]:
    full_output, full_lse = full
    ceiling_output, ceiling_lse = packed
    if not torch.equal(full_output, ceiling_output):
        difference = (full_output.float() - ceiling_output.float()).abs()
        raise AssertionError(
            f"{stage} output mismatch max={difference.max().item()} "
            f"count={torch.count_nonzero(difference).item()}"
        )
    if not torch.equal(full_lse, ceiling_lse):
        difference = (full_lse - ceiling_lse).abs()
        raise AssertionError(
            f"{stage} LSE mismatch max={difference.max().item()} "
            f"count={torch.count_nonzero(difference).item()}"
        )
    return {
        "output_sha256": _sha256_tensor(full_output),
        "lse_sha256": _sha256_tensor(full_lse),
    }


def _assert_nondegenerate(
    full: tuple[torch.Tensor, torch.Tensor],
    total_ceiling: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    """Prove that retained nonuniform scale work affects the result."""

    full_output, full_lse = full
    ceiling_output, ceiling_lse = total_ceiling
    output_difference = torch.count_nonzero(full_output != ceiling_output).item()
    lse_difference = torch.count_nonzero(full_lse != ceiling_lse).item()
    if output_difference == 0 and lse_difference == 0:
        raise AssertionError(
            "nonuniform fixture collapsed to total-scale-ceiling semantics"
        )
    return {
        "full_output_sha256": _sha256_tensor(full_output),
        "full_lse_sha256": _sha256_tensor(full_lse),
        "total_ceiling_output_sha256": _sha256_tensor(ceiling_output),
        "total_ceiling_lse_sha256": _sha256_tensor(ceiling_lse),
        "output_difference_count": output_difference,
        "lse_difference_count": lse_difference,
    }


def _run_shape(
    query_len: int,
    seq_len: int,
    *,
    compile_order: tuple[str, str],
    permutation_seed: int,
    timing: bool,
    windows: int,
    replays: int,
    warmup_windows: int,
    dump_dir: Path,
) -> dict[str, Any]:
    tensors = _build_fixture(
        query_len,
        seq_len,
        permutation_seed=permutation_seed,
    )
    split, workspace_size = _workspace_geometry(query_len, seq_len)
    output = torch.empty(
        (1, query_len, HEADS, LATENT), dtype=torch.bfloat16, device="cuda"
    )
    lse = torch.empty((1, query_len, HEADS), dtype=torch.float32, device="cuda")
    workspace = (
        torch.empty(workspace_size, dtype=torch.int8, device="cuda")
        if workspace_size
        else None
    )
    pointers = {
        name: value.data_ptr()
        for name, value in tensors.items()
        if isinstance(value, torch.Tensor)
    }
    pointers.update(output=output.data_ptr(), lse=lse.data_ptr())
    if workspace is not None:
        pointers["workspace"] = workspace.data_ptr()
    compiled: dict[str, Any] = {}
    for arm in compile_order:
        compiled[arm] = _compile(arm, tensors, query_len, output, lse, workspace, split)
    compiled["total_ceiling"] = _compile(
        "total_ceiling", tensors, query_len, output, lse, workspace, split
    )
    artifacts = {
        arm: _artifact_identity(
            compiled[arm],
            arm,
            query_len,
            split,
            dump_dir,
            seq_len in SCORE_CELLS.values(),
        )
        for arm in ("full", "packed")
    }
    if artifacts["full"]["cubin_sha256"] == artifacts["packed"]["cubin_sha256"]:
        raise AssertionError("full and packed CUBIN identities must differ")
    if artifacts["full"]["ptx_sha256"] == artifacts["packed"]["ptx_sha256"]:
        raise AssertionError("full and packed PTX identities must differ")
    static_isolation = _assert_static_packed_isolation(
        artifacts["full"], artifacts["packed"]
    )
    calls = {
        arm: (
            lambda arm=arm: _invoke(
                compiled[arm], tensors, output, lse, workspace, split
            )
        )
        for arm in ("full", "packed")
    }
    eager = {
        arm: _run_and_snapshot(calls[arm], output, lse) for arm in ("full", "packed")
    }
    parity = {"eager": _assert_equal("eager", eager["full"], eager["packed"])}
    total_ceiling = _run_and_snapshot(
        lambda: _invoke(
            compiled["total_ceiling"], tensors, output, lse, workspace, split
        ),
        output,
        lse,
    )
    nondegeneracy = _assert_nondegenerate(eager["full"], total_ceiling)
    graphs = {arm: _capture(calls[arm]) for arm in ("full", "packed")}
    captured = {
        arm: _run_and_snapshot(graphs[arm].replay, output, lse)
        for arm in ("full", "packed")
    }
    parity["captured"] = _assert_equal(
        "captured", captured["full"], captured["packed"]
    )
    graph100 = {}
    for arm in ("full", "packed"):
        for _ in range(GRAPH_PARITY_REPLAYS):
            graphs[arm].replay()
        torch.cuda.synchronize()
        _assert_finite_written(output, lse)
        graph100[arm] = (output.clone(), lse.clone())
    parity["graph100"] = _assert_equal(
        "graph100", graph100["full"], graph100["packed"]
    )

    samples: dict[str, list[float]] = {"full": [], "packed": []}
    if timing:
        for warmup in range(warmup_windows):
            order = (
                ("full", "packed")
                if warmup % 2 == 0
                else (
                    "packed",
                    "full",
                )
            )
            for arm in order:
                _measure(graphs[arm], replays)
        for window in range(windows):
            order = (
                ("full", "packed")
                if window % 2 == 0
                else (
                    "packed",
                    "full",
                )
            )
            for arm in order:
                samples[arm].append(_measure(graphs[arm], replays))

    post = {
        arm: _run_and_snapshot(graphs[arm].replay, output, lse)
        for arm in ("full", "packed")
    }
    parity["post_timing"] = _assert_equal("post_timing", post["full"], post["packed"])
    return {
        "query_len": query_len,
        "seq_len": seq_len,
        "physical_seq_len": (
            PRIMARY_PHYSICAL_SEQ_LEN
            if seq_len == PRIMARY_SEQ_LEN
            else math.ceil(seq_len / PAGE_SIZE) * PAGE_SIZE
        ),
        "permutation_seed": permutation_seed,
        "split": split,
        "workspace_size": workspace_size,
        "fold": get_mla_decode_fold_sq_factor(HEADS, query_len, 64),
        "pointers": pointers,
        "table_ids": tensors["table_ids"],
        "unused_ids": tensors["unused_ids"],
        "scale_audit": tensors["scale_audit"],
        "nondegeneracy": nondegeneracy,
        "static_isolation": static_isolation,
        "artifacts": artifacts,
        "parity": parity,
        "samples_us": samples,
        "timing": timing,
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    receipt_value = os.environ.get("TQ_PACKED_SCALE_RECEIPT")
    if not receipt_value:
        raise ValueError("TQ_PACKED_SCALE_RECEIPT is required")
    receipt_path = Path(receipt_value).resolve()
    dump_dir = Path(
        os.environ.get(
            "TQ_PACKED_SCALE_DUMP_DIR", str(receipt_path.parent / "artifacts")
        )
    ).resolve()
    role = os.environ.get("TQ_PACKED_SCALE_ROLE")
    if not role:
        raise ValueError("TQ_PACKED_SCALE_ROLE is required")
    node_id = os.environ.get("TQ_PACKED_SCALE_NODE_ID")
    if not node_id:
        raise ValueError("TQ_PACKED_SCALE_NODE_ID is required")
    attempt_id = os.environ.get("TQ_PACKED_SCALE_ATTEMPT_ID")
    if not attempt_id:
        raise ValueError("TQ_PACKED_SCALE_ATTEMPT_ID is required")
    order = tuple(
        os.environ.get("TQ_PACKED_SCALE_COMPILE_ORDER", "full,packed").split(",")
    )
    if len(order) != 2 or set(order) != {"full", "packed"}:
        raise ValueError("compile order must contain full and packed exactly once")
    compile_order = (order[0], order[1])
    timing = os.environ.get("TQ_PACKED_SCALE_TIMING", "1") == "1"
    _require_deterministic_hash_seed(timing)
    cell = os.environ.get("TQ_PACKED_SCALE_CELL", "primary")
    if cell not in SCORE_CELLS:
        raise ValueError(f"unknown R7 score cell: {cell}")
    windows = int(os.environ.get("TQ_PACKED_SCALE_WINDOWS", str(DEFAULT_WINDOWS)))
    replays = int(os.environ.get("TQ_PACKED_SCALE_REPLAYS", str(DEFAULT_REPLAYS)))
    warmup_windows = int(
        os.environ.get("TQ_PACKED_SCALE_WARMUP_WINDOWS", str(DEFAULT_WARMUP_WINDOWS))
    )
    if timing and (
        windows != DEFAULT_WINDOWS
        or replays != DEFAULT_REPLAYS
        or warmup_windows != DEFAULT_WARMUP_WINDOWS
    ):
        raise ValueError("scored R7 timing geometry is frozen at 30/500/12")
    query_lengths = tuple(
        int(value)
        for value in os.environ.get("TQ_PACKED_SCALE_QUERY_LENS", "1,5").split(",")
    )
    sequence_lengths = tuple(
        int(value)
        for value in os.environ.get(
            "TQ_PACKED_SCALE_SEQ_LENS", str(PRIMARY_SEQ_LEN)
        ).split(",")
    )
    if not query_lengths or any(value not in QUERY_LENGTHS for value in query_lengths):
        raise ValueError("query lengths must contain only q1 and/or q5")
    if not sequence_lengths or any(value <= 0 for value in sequence_lengths):
        raise ValueError("sequence lengths must be positive")
    permutation_seed = int(
        os.environ.get("TQ_PACKED_SCALE_PERMUTATION_SEED", "2026081701")
    )
    expected_seq_len = SCORE_CELLS[cell]
    if timing and (
        query_lengths != QUERY_LENGTHS or sequence_lengths != (expected_seq_len,)
    ):
        raise ValueError(
            f"scored R7 {cell} timing requires q1,q5 at exact K{expected_seq_len}"
        )

    source_path = Path(__file__).resolve()
    reader_source = inspect.getsourcefile(BlackwellMultiHeadLatentAttentionForwardFP8)
    if reader_source is None:
        raise RuntimeError("cannot resolve accepted reader source")
    reader_path = Path(reader_source).resolve()
    expected_reader = os.environ.get("TQ_PACKED_SCALE_EXPECT_READER_SHA256")
    reader_sha256 = _sha256_bytes(reader_path.read_bytes())
    if expected_reader and reader_sha256 != expected_reader:
        raise RuntimeError(f"reader SHA drift: {reader_sha256} != {expected_reader}")
    if timing and not expected_reader:
        raise ValueError(
            "scored R7 timing requires TQ_PACKED_SCALE_EXPECT_READER_SHA256"
        )

    started_ns = time.time_ns()
    runtime = _runtime_identity()
    before = _gpu_state("before")
    shapes = []
    for query_len in query_lengths:
        for sequence_index, seq_len in enumerate(sequence_lengths):
            shape_dump = dump_dir / f"q{query_len}-k{seq_len}"
            shapes.append(
                _run_shape(
                    query_len,
                    seq_len,
                    compile_order=compile_order,
                    permutation_seed=permutation_seed + sequence_index,
                    timing=timing,
                    windows=windows,
                    replays=replays,
                    warmup_windows=warmup_windows,
                    dump_dir=shape_dump,
                )
            )
    after = _gpu_state("after")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "A17-N10-E0-R7",
        "cell": cell,
        "role": role,
        "node_id": node_id,
        "attempt_id": attempt_id,
        "compile_order": list(compile_order),
        "timing": timing,
        "windows": windows,
        "replays": replays,
        "warmup_windows": warmup_windows,
        "permutation_seed": permutation_seed,
        "reader_source": str(reader_path),
        "reader_sha256": reader_sha256,
        "harness_sha256": _sha256_bytes(source_path.read_bytes()),
        "runtime": runtime,
        "gpu_state": {"before": before, "after": after},
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "shapes": shapes,
    }
    _write_receipt(receipt_path, receipt)
    print(
        "TQ_PACKED_SCALE_RECEIPT " + json.dumps(receipt, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
