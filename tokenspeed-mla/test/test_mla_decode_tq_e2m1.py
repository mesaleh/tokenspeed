# Copyright (c) 2026 LightSeek Foundation

"""Correctness and contract gates for the public packed E2M1 MLA reader."""

import math

import cutlass
import pytest
import torch
from cutlass import Float32, Int32

from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
)
from tokenspeed_mla.mla_decode_tq_e2m1 import (
    _MMA_QK_TILER,
    _NUM_HEADS,
    _get_compiled_tq_e2m1_kernel,
    tokenspeed_mla_decode_tq_e2m1,
)
from tokenspeed_mla.mla_helpers import get_mla_decode_fold_sq_factor
from tokenspeed_mla.utils import get_num_sm


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0),
    reason="packed E2M1 MLA decode requires SM100",
)

_PAGE_SIZE = 32
_LATENT = 512
_ROPE = 64
_E2M1 = torch.tensor(
    [
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
    ],
    dtype=torch.float32,
)


def _pack(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def _inputs(query_len: int, seq_len: int, batch: int = 2):
    torch.manual_seed(20260816 + query_len + seq_len)
    device = torch.device("cuda")
    pages_per_request = math.ceil(seq_len / _PAGE_SIZE)
    num_pages = batch * pages_per_request

    query_latent = (
        torch.randint(
            -4,
            5,
            (batch, query_len, _NUM_HEADS, _LATENT),
            device=device,
        )
        / 2.0
    ).to(torch.float8_e4m3fn)
    query_rope = (
        torch.randint(
            -4,
            5,
            (batch, query_len, _NUM_HEADS, _ROPE),
            device=device,
        )
        / 2.0
    ).to(torch.bfloat16)

    token = torch.arange(num_pages * _PAGE_SIZE, device=device)[:, None]
    dim = torch.arange(_LATENT, device=device)[None, :]
    codes = ((token * 5 + dim * 3 + 1) % 15 + 1).to(torch.uint8)
    packed = _pack(codes.view(num_pages, _PAGE_SIZE, _LATENT)).contiguous()
    levels = _E2M1.to(device)[codes.long()].view(
        num_pages, _PAGE_SIZE, _LATENT
    )

    flat_scale = torch.pow(2.0, (-8 + token[:, 0] % 7).float())
    scale = flat_scale.to(torch.bfloat16).view(num_pages, _PAGE_SIZE).contiguous()
    rope_dim = torch.arange(_ROPE, device=device)[None, :]
    raw_rope = (((token * 7 + rope_dim * 11 + 3) % 9) - 4) / 2.0
    reciprocal_rope = (
        raw_rope / scale.view(-1, 1).float()
    ).to(torch.bfloat16).view(num_pages, _PAGE_SIZE, _ROPE).contiguous()

    rows = []
    for request in range(batch):
        pages = torch.arange(
            request * pages_per_request,
            (request + 1) * pages_per_request,
            dtype=torch.int32,
            device=device,
        )
        if request % 2:
            pages = torch.flip(pages, dims=(0,))
        padded_page_count = math.ceil(pages.numel() / 4) * 4
        if pages.numel() < padded_page_count:
            pages = torch.cat(
                (pages, pages[-1:].expand(padded_page_count - pages.numel()))
            )
        rows.append(pages)
    block_tables = torch.stack(rows).contiguous()
    seq_lens = torch.tensor(
        [seq_len - request for request in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    workspace = torch.empty(
        get_num_sm(device) * _NUM_HEADS * query_len * (_LATENT + 1) * 4,
        dtype=torch.int8,
        device=device,
    )
    return {
        "query_latent": query_latent.contiguous(),
        "query_rope": query_rope.contiguous(),
        "packed_latent": packed,
        "reconstruction_scale": scale,
        "reciprocal_rope": reciprocal_rope,
        "workspace_buffer": workspace,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "max_seq_len": seq_len,
        "softmax_scale": 1.0 / math.sqrt(_LATENT + _ROPE),
        "_levels": levels,
    }


def _reference(inputs: dict[str, torch.Tensor], query_len: int):
    results = []
    lses = []
    for request, length_tensor in enumerate(inputs["seq_lens"]):
        length = int(length_tensor.item())
        pages = inputs["block_tables"][request].long()
        levels = inputs["_levels"][pages].reshape(-1, _LATENT)[:length]
        scale = inputs["reconstruction_scale"][pages].reshape(-1).float()[:length]
        latent = levels * scale[:, None]
        reciprocal_rope = inputs["reciprocal_rope"][pages].reshape(
            -1, _ROPE
        )[:length].float()
        rope = reciprocal_rope * scale[:, None]
        request_results = []
        request_lses = []
        for token in range(query_len):
            causal_bound = length - query_len + token + 1
            score = torch.einsum(
                "hd,kd->hk",
                inputs["query_latent"][request, token].float(),
                latent[:causal_bound],
            )
            score += torch.einsum(
                "hr,kr->hk",
                inputs["query_rope"][request, token].float(),
                rope[:causal_bound],
            )
            score *= inputs["softmax_scale"]
            request_results.append(torch.softmax(score, dim=-1) @ latent[:causal_bound])
            request_lses.append(
                torch.logsumexp(score, dim=-1) * math.log2(math.e)
            )
        results.append(torch.stack(request_results))
        lses.append(torch.stack(request_lses))
    return torch.stack(results), torch.stack(lses)


def _direct_call(inputs, out, lse, return_lse):
    query_len = inputs["query_latent"].shape[1]
    batch = inputs["query_latent"].shape[0]
    fold = get_mla_decode_fold_sq_factor(_NUM_HEADS, query_len, 64)
    split = BlackwellMultiHeadLatentAttentionForwardFP8.get_split_kv(
        batch,
        query_len // fold,
        inputs["max_seq_len"],
        _MMA_QK_TILER,
        get_num_sm(inputs["query_latent"].device),
        1,
    )
    workspace_size = BlackwellMultiHeadLatentAttentionForwardFP8.get_workspace_size(
        _NUM_HEADS * fold,
        query_len // fold,
        _LATENT,
        batch,
        split,
        cutlass.Float32,
    )
    workspace = (
        None
        if workspace_size == 0
        else inputs["workspace_buffer"][:workspace_size]
    )
    compiled = _get_compiled_tq_e2m1_kernel(
        query_latent=inputs["query_latent"],
        query_rope=inputs["query_rope"],
        packed_latent=inputs["packed_latent"],
        reconstruction_scale=inputs["reconstruction_scale"],
        reciprocal_rope=inputs["reciprocal_rope"],
        block_tables=inputs["block_tables"],
        output=out,
        lse=lse,
        workspace=workspace,
        seq_lens=inputs["seq_lens"],
        fold_sq_factor=fold,
        enable_pdl=False,
    )
    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        compiled(
            inputs["query_latent"],
            inputs["query_rope"],
            inputs["packed_latent"],
            inputs["reciprocal_rope"],
            inputs["block_tables"],
            out,
            lse,
            workspace,
            Int32(split),
            inputs["seq_lens"],
            inputs["seq_lens"],
            None,
            Float32(inputs["softmax_scale"]),
            Float32(1.0),
            inputs["reconstruction_scale"],
        )


@pytest.mark.parametrize("query_len,seq_len", [(1, 33), (1, 257), (5, 257)])
def test_public_matches_direct_owner_and_reference(query_len, seq_len):
    inputs = _inputs(query_len, seq_len)
    shape = inputs["query_latent"].shape
    direct_out = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    public_out = torch.empty_like(direct_out)
    direct_lse = torch.empty(shape[:-1], dtype=torch.float32, device="cuda")
    public_lse = torch.empty_like(direct_lse)

    _direct_call(inputs, direct_out, direct_lse, return_lse=True)
    observed, observed_lse = tokenspeed_mla_decode_tq_e2m1(
        **{key: value for key, value in inputs.items() if not key.startswith("_")},
        out=public_out,
        return_lse=True,
        lse_out=public_lse,
    )
    torch.cuda.synchronize()

    assert torch.equal(observed, direct_out)
    assert torch.equal(observed_lse, direct_lse)
    expected, expected_lse = _reference(inputs, query_len)
    torch.testing.assert_close(observed.float(), expected, atol=0.75, rtol=0.35)
    torch.testing.assert_close(observed_lse, expected_lse, atol=0.2, rtol=0.02)


def test_public_fail_closed_contract():
    inputs = _inputs(query_len=1, seq_len=33, batch=1)
    bad = dict(inputs)
    bad.pop("_levels")
    bad["reconstruction_scale"] = bad["reconstruction_scale"].float()
    with pytest.raises(TypeError, match="reconstruction_scale"):
        tokenspeed_mla_decode_tq_e2m1(**bad)

    bad = {key: value for key, value in inputs.items() if not key.startswith("_")}
    bad["block_tables"] = bad["block_tables"][:, :1]
    with pytest.raises(ValueError, match="block-table capacity"):
        tokenspeed_mla_decode_tq_e2m1(**bad)

    bad = {key: value for key, value in inputs.items() if not key.startswith("_")}
    bad["block_tables"] = bad["block_tables"][:, :2].contiguous()
    with pytest.raises(ValueError, match="padded to a multiple of 4 pages"):
        tokenspeed_mla_decode_tq_e2m1(**bad)


@pytest.mark.parametrize("query_len", [1, 5])
def test_public_cuda_graph_replay_is_allocation_stable(query_len):
    inputs = _inputs(query_len=query_len, seq_len=257, batch=2)
    public_inputs = {
        key: value for key, value in inputs.items() if not key.startswith("_")
    }
    out = torch.empty_like(inputs["query_latent"], dtype=torch.bfloat16)

    tokenspeed_mla_decode_tq_e2m1(**public_inputs, out=out)
    torch.cuda.synchronize()
    output_pointer = out.data_ptr()
    workspace_pointer = inputs["workspace_buffer"].data_ptr()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tokenspeed_mla_decode_tq_e2m1(**public_inputs, out=out)
    torch.cuda.synchronize()
    allocated_after_capture = torch.cuda.memory_allocated()

    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()

    assert out.data_ptr() == output_pointer
    assert inputs["workspace_buffer"].data_ptr() == workspace_pointer
    assert torch.cuda.memory_allocated() == allocated_after_capture
    expected, _ = _reference(inputs, query_len=query_len)
    torch.testing.assert_close(out.float(), expected, atol=0.75, rtol=0.35)
