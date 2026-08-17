# Copyright (c) 2026 LightSeek Foundation

"""Unit tests for the R7 packed P-scale harness contract."""

import pytest
import torch
import cutlass
from benchmark_mla_decode_tq_e2m1_packed_scale_application import (
    PAGE_SIZE,
    _audit_scale_fixture,
    _require_deterministic_hash_seed,
)
from tokenspeed_mla.mla_decode_fp8 import BlackwellMultiHeadLatentAttentionForwardFP8


def _paged_scale_fixture() -> tuple[torch.Tensor, list[int]]:
    table_ids = [2, 0, 3, 1]
    scale = torch.full((4, PAGE_SIZE), 16.0, dtype=torch.bfloat16)
    values = (0.5, 0.75, 1.0)
    for token in range(128):
        scale[table_ids[token // PAGE_SIZE], token % PAGE_SIZE] = values[token % 3]
    return scale, table_ids


def test_nonuniform_paged_scale_oracle_accepts_exact_max_one():
    scale, table_ids = _paged_scale_fixture()
    audit = _audit_scale_fixture(scale, table_ids, 128)
    assert audit["allowed_values"] == [0.5, 0.75, 1.0]
    assert audit["tile_count"] == 1
    assert audit["tile_minimum_max"] == 1.0
    assert audit["tile_maximum_max"] == 1.0


def test_nonuniform_paged_scale_oracle_rejects_max_two():
    scale, table_ids = _paged_scale_fixture()
    scale[table_ids[0], 7] = 2.0
    with pytest.raises(AssertionError, match="logical N128 tile must have max one"):
        _audit_scale_fixture(scale, table_ids, 128)


def test_scored_timing_requires_fixed_python_hash_seed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(ValueError, match="before interpreter startup"):
        _require_deterministic_hash_seed(True, hash_randomization=1)


def test_fixed_python_hash_seed_is_accepted(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    assert _require_deterministic_hash_seed(True, hash_randomization=0) == "0"


def test_late_hash_seed_mutation_is_rejected(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    with pytest.raises(ValueError, match="before interpreter startup"):
        _require_deterministic_hash_seed(True, hash_randomization=1)


def _kernel(**overrides):
    kwargs = {
        "acc_dtype": cutlass.Float32,
        "lse_dtype": cutlass.Float32,
        "mma_qk_tiler_mn": (64, 128),
        "mma_pv_tiler_mn": (64, 256),
        "max_active_clusters": 1,
        "page_size": PAGE_SIZE,
        "skip_correction_threshold": 0.0,
        "is_persistent": False,
        "is_var_seq": True,
        "is_var_split_kv": False,
        "use_tq_e2m1": True,
    }
    kwargs.update(overrides)
    return BlackwellMultiHeadLatentAttentionForwardFP8(**kwargs)


def test_packed_p_scale_application_requires_paged_scale_tma():
    with pytest.raises(ValueError, match="requires paged scale TMA"):
        _kernel(tq_s1_packed_p_scale_math=True)


def test_packed_p_scale_application_accepts_exact_tma_arm():
    kernel = _kernel(
        tq_s1_scale_tma=True,
        tq_s1_scale_stages=3,
        tq_s1_packed_p_scale_math=True,
    )
    assert kernel.tq_s1_packed_p_scale_math is True


def test_packed_p_scale_application_rejects_total_scale_ceiling():
    with pytest.raises(ValueError, match="mutually exclusive|incompatible"):
        _kernel(
            tq_s1_scale_tma=True,
            tq_s1_packed_p_scale_math=True,
            tq_s1_scale_ceiling=True,
        )
