"""CPU contract tests for the native TurboQuant-4 MLA ABI."""

import ast
from pathlib import Path

import pytest
import torch
from tokenspeed_mla.tq4_contract import (
    TQ4_LATENT_DIM,
    dequantize_tq4_reference,
    unpack_tq4_indices_reference,
    validate_tq4_decode_inputs,
)


def _valid_inputs(q_len: int = 5, heads: int = 16):
    pages, page_size, batch = 3, 32, 1
    return {
        "query": torch.empty(batch, q_len, heads, 576, dtype=torch.float8_e4m3fn),
        "kv_nope_packed": torch.empty(pages, page_size, 256, dtype=torch.uint8),
        "kv_nope_scale": torch.empty(pages, page_size, dtype=torch.bfloat16),
        "kv_rope": torch.empty(pages, page_size, 64, dtype=torch.bfloat16),
        "centroids": torch.linspace(-1, 1, 16, dtype=torch.float32),
        "workspace_buffer": torch.empty(4096, dtype=torch.int8),
        "block_tables": torch.arange(pages, dtype=torch.int32)[None],
        "seq_lens": torch.tensor([65], dtype=torch.int32),
        "out": torch.empty(batch, q_len, heads, 512, dtype=torch.bfloat16),
    }


def test_unpack_uses_low_nibble_for_even_coordinate():
    packed = torch.zeros(1, 1, 256, dtype=torch.uint8)
    packed[..., :4] = torch.tensor([0x10, 0x32, 0x54, 0x76], dtype=torch.uint8)
    packed[..., 4] = 0xFE
    indices = unpack_tq4_indices_reference(packed)
    assert indices.shape[-1] == TQ4_LATENT_DIM
    assert indices[0, 0, :8].tolist() == list(range(8))
    assert indices[0, 0, 8:10].tolist() == [14, 15]


def test_reference_dequant_applies_one_scale_per_token():
    packed = torch.zeros(1, 2, 256, dtype=torch.uint8)
    packed[0, 0, 0] = 0x10
    packed[0, 1, 0] = 0x32
    scales = torch.tensor([[2.0, 0.5]], dtype=torch.bfloat16)
    centroids = torch.arange(16, dtype=torch.float32) - 8
    values = dequantize_tq4_reference(packed, scales, centroids, dtype=torch.float32)
    assert values[0, 0, :2].tolist() == [-16.0, -14.0]
    assert values[0, 1, :2].tolist() == [-3.0, -2.5]


@pytest.mark.parametrize("q_len", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("heads", [8, 16])
def test_validate_accepts_kimi_decode_and_verify_shapes(q_len: int, heads: int):
    inputs = _valid_inputs(q_len, heads)
    shape, packed, rope = validate_tq4_decode_inputs(**inputs, require_cuda=False)
    assert shape.query_length == q_len
    assert shape.num_heads == heads
    assert shape.page_size == 32
    assert packed.shape == (3, 32, 256)
    assert rope.shape == (3, 32, 64)


def test_validate_accepts_singleton_head_cache_form_without_copy():
    inputs = _valid_inputs()
    inputs["kv_nope_packed"] = inputs["kv_nope_packed"].unsqueeze(1)
    inputs["kv_rope"] = inputs["kv_rope"].unsqueeze(1)
    _, packed, rope = validate_tq4_decode_inputs(**inputs, require_cuda=False)
    assert packed.data_ptr() == inputs["kv_nope_packed"].data_ptr()
    assert rope.data_ptr() == inputs["kv_rope"].data_ptr()


def test_validate_accepts_explicit_fp8_rope():
    inputs = _valid_inputs()
    inputs["kv_rope"] = torch.empty(3, 32, 64, dtype=torch.float8_e4m3fn)
    _, _, rope = validate_tq4_decode_inputs(**inputs, fp8_rope=True, require_cuda=False)
    assert rope.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize(
    ("dtype", "fp8_rope", "message"),
    [
        (torch.float8_e4m3fn, False, "must be bfloat16"),
        (torch.bfloat16, True, "must be FP8 E4M3"),
    ],
)
def test_validate_rejects_rope_flag_dtype_mismatch(dtype, fp8_rope, message):
    inputs = _valid_inputs()
    inputs["kv_rope"] = torch.empty(3, 32, 64, dtype=dtype)
    with pytest.raises(ValueError, match=message):
        validate_tq4_decode_inputs(**inputs, fp8_rope=fp8_rope, require_cuda=False)


@pytest.mark.parametrize("name", ["kv_nope_packed", "kv_rope"])
def test_validate_rejects_unaligned_raw_cache(name):
    inputs = _valid_inputs()
    tensor = inputs[name]
    unaligned = torch.empty(
        tensor.numel() + 1,
        dtype=tensor.dtype,
    )[
        1:
    ].view(tensor.shape)
    assert unaligned.is_contiguous()
    assert unaligned.data_ptr() % 16 != 0
    inputs[name] = unaligned
    with pytest.raises(ValueError, match="16-byte aligned"):
        validate_tq4_decode_inputs(**inputs, require_cuda=False)


def test_validate_rejects_cache_device_mismatch():
    inputs = _valid_inputs()
    inputs["centroids"] = torch.empty(16, dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="must share one device"):
        validate_tq4_decode_inputs(**inputs, require_cuda=False)


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    [
        (
            "kv_nope_packed",
            torch.empty(3, 32, 512, dtype=torch.uint8),
            "packed latent dimension",
        ),
        (
            "kv_nope_packed",
            torch.empty(3, 32, 256, dtype=torch.bfloat16),
            "must be uint8",
        ),
        ("kv_nope_scale", torch.empty(3, 31, dtype=torch.bfloat16), "scale shape"),
        ("kv_rope", torch.empty(3, 32, 63, dtype=torch.bfloat16), "kv_rope shape"),
        ("centroids", torch.empty(15, dtype=torch.float32), r"shape \(16,\)"),
        ("block_tables", torch.empty(1, 3, dtype=torch.int64), "must be int32"),
    ],
)
def test_validate_rejects_contract_drift(name, replacement, message):
    inputs = _valid_inputs()
    inputs[name] = replacement
    with pytest.raises(ValueError, match=message):
        validate_tq4_decode_inputs(**inputs, require_cuda=False)


@pytest.mark.parametrize("q_len", [6, 8])
def test_validate_rejects_unimplemented_query_lengths(q_len: int):
    with pytest.raises(ValueError, match=r"supports q_len in \[1, 5\]"):
        validate_tq4_decode_inputs(**_valid_inputs(q_len), require_cuda=False)


def test_validate_rejects_non_page32_layout():
    inputs = _valid_inputs()
    inputs["kv_nope_packed"] = torch.empty(3, 64, 256, dtype=torch.uint8)
    inputs["kv_nope_scale"] = torch.empty(3, 64, dtype=torch.bfloat16)
    inputs["kv_rope"] = torch.empty(3, 64, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires page_size=32"):
        validate_tq4_decode_inputs(**inputs, require_cuda=False)


def test_validate_rejects_non_kimi_head_geometry():
    inputs = _valid_inputs()
    inputs["query"] = torch.empty(1, 5, 32, 576, dtype=torch.float8_e4m3fn)
    inputs["out"] = torch.empty(1, 5, 32, 512, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires 8 or 16 TP-local query heads"):
        validate_tq4_decode_inputs(**inputs, require_cuda=False)


def test_validate_rejects_noncontiguous_packed_pages():
    inputs = _valid_inputs()
    inputs["kv_nope_packed"] = torch.empty(3, 32, 512, dtype=torch.uint8)[..., ::2]
    with pytest.raises(ValueError, match="kv_nope_packed must be contiguous"):
        validate_tq4_decode_inputs(**inputs, require_cuda=False)


def test_tq4_mutable_global_loads_follow_raw_pipeline_wait():
    source = (
        Path(__file__).parents[1] / "python" / "tokenspeed_mla" / "mla_decode_fp8.py"
    ).read_text(encoding="utf-8")
    start = source.index("    def convert_tq4_kv(")
    end = source.index("\n    @cute.jit", start + 1)
    body = source[start:end]

    wait = body.index("common_params.raw_k_pipeline.consumer_wait")
    page_table = body.index("page_table = common_params.mPT")
    codebook_pointer = body.index("codebook_i32_ptr = cute.recast_ptr")
    codebook_load = body.index(").load()", codebook_pointer)
    centroid = body.index("common_params.mTQCentroids")

    assert wait < centroid < page_table < codebook_pointer < codebook_load


def test_public_tq4_api_is_exact_m128_alias():
    package_dir = Path(__file__).parents[1] / "python" / "tokenspeed_mla"
    module_tree = ast.parse(
        (package_dir / "mla_decode_tq4.py").read_text(encoding="utf-8")
    )
    aliases = {
        target.id: node.value.id
        for node in module_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and isinstance(node.value, ast.Name)
    }
    assert aliases["tokenspeed_mla_decode_tq4"] == (
        "_tokenspeed_mla_decode_tq4_m128_control"
    )


def test_package_exports_public_tq4_api_with_unavailable_fallback():
    package_dir = Path(__file__).parents[1] / "python" / "tokenspeed_mla"
    init_source = (package_dir / "__init__.py").read_text(encoding="utf-8")
    init_tree = ast.parse(init_source)

    imported = {
        alias.name
        for node in ast.walk(init_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "tokenspeed_mla.mla_decode_tq4"
        for alias in node.names
    }
    fallback_targets = {
        target.id
        for node in ast.walk(init_tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Name)
        and node.value.id == "_unavailable"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    exported = next(
        ast.literal_eval(node.value)
        for node in init_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )

    assert "tokenspeed_mla_decode_tq4" in imported
    assert "tokenspeed_mla_decode_tq4" in fallback_targets
    assert "tokenspeed_mla_decode_tq4" in exported
