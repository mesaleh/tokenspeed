# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEST_ROOT))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=900, suite="runtime-1gpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
MLA_ROOT = REPO_ROOT / "tokenspeed-mla"
DECODE_SOURCE = MLA_ROOT / "python/tokenspeed_mla/mla_decode.py"
FP8_SOURCE = MLA_ROOT / "python/tokenspeed_mla/mla_decode_fp8.py"
FP16_SOURCE = MLA_ROOT / "python/tokenspeed_mla/mla_decode_fp16.py"
MICROBENCH_SOURCE = MLA_ROOT / "test/microbench_tree_decode.py"
MLA_PYTHON_ROOT = MLA_ROOT / "python"
sys.path.insert(0, str(MLA_PYTHON_ROOT))


def _function_args(path: Path, function_name: str, class_name: str | None = None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name is not None:
        nodes = next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    function = next(
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    return [arg.arg for arg in function.args.args]


def build_ancestor_mask(parent: list[int]) -> torch.Tensor:
    ancestor = torch.zeros(len(parent), len(parent), dtype=torch.bool)
    for query_token in range(len(parent)):
        node = query_token
        for _ in range(len(parent) + 1):
            if node == -1:
                break
            ancestor[query_token, node] = True
            node = parent[node]
        else:
            raise AssertionError(f"cycle in parent list: {parent}")
    return ancestor


def flatten_tree_masks(
    ancestors: list[torch.Tensor], seq_lens: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    q_len = ancestors[0].shape[0]
    masks = []
    offsets = []
    offset = 0
    for ancestor, seq_len in zip(ancestors, seq_lens, strict=True):
        history = seq_len - q_len
        mask = torch.ones(q_len, seq_len, dtype=torch.bool)
        mask[:, history:] = ancestor
        offsets.append(offset)
        masks.append(mask.flatten())
        offset += q_len * seq_len
    return torch.cat(masks), torch.tensor(offsets, dtype=torch.int32)


def _load_microbench():
    spec = importlib.util.spec_from_file_location(
        "microbench_tree_decode", MICROBENCH_SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.H = 16
    module.PAGE = 32
    return module


def _call_until_tree_mask_validation(
    *,
    query_dtype=torch.float8_e4m3fn,
    custom_mask=None,
    cmask_off=None,
    cp_world=1,
):
    from tokenspeed_mla import tokenspeed_mla_decode

    batch_size, q_len, max_seq_len = 2, 2, 2
    query = torch.empty(batch_size, q_len, 1, 2, dtype=query_dtype)
    kv_cache = torch.empty(4, 1, 2, dtype=query_dtype)
    return tokenspeed_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=torch.empty(1, dtype=torch.int8),
        kv_lora_rank=1,
        qk_rope_head_dim=1,
        block_tables=torch.arange(4, dtype=torch.int32).view(batch_size, 2),
        seq_lens=torch.full((batch_size,), max_seq_len, dtype=torch.int32),
        max_seq_len=max_seq_len,
        softmax_scale=1.0,
        custom_mask=custom_mask,
        cmask_off=cmask_off,
        cp_world=cp_world,
    )


def test_public_decode_abi_has_exact_tree_mask_arguments():
    args = _function_args(DECODE_SOURCE, "tokenspeed_mla_decode")
    assert "custom_mask" in args
    assert "cmask_off" in args
    assert args.index("cmask_off") == args.index("custom_mask") + 1


def test_fp8_kernel_abi_preserves_dcp_after_tree_mask_arguments():
    call_args = _function_args(
        FP8_SOURCE,
        "__call__",
        class_name="BlackwellMultiHeadLatentAttentionForwardFP8",
    )
    assert call_args[5:14] == [
        "page_table",
        "custom_mask",
        "cmask_off",
        "o",
        "lse",
        "workspace",
        "split_kv",
        "cache_seqs",
        "causal_seqs",
    ]

    kernel_args = _function_args(
        FP8_SOURCE,
        "split_kv_kernel",
        class_name="BlackwellMultiHeadLatentAttentionForwardFP8",
    )
    assert kernel_args.index("mCmask") == kernel_args.index("mPT") + 1
    assert kernel_args.index("mCmaskOff") == kernel_args.index("mCmask") + 1
    assert kernel_args.index("causal_seqs") == kernel_args.index("cache_seqs") + 1


def test_fp16_kernel_abi_remains_unchanged():
    args = _function_args(
        FP16_SOURCE,
        "__call__",
        class_name="BlackwellMultiHeadLatentAttentionForwardFP16",
    )
    assert args == [
        "self",
        "q_latent",
        "q_rope",
        "c_latent",
        "c_rope",
        "page_table",
        "o",
        "lse",
        "workspace",
        "split_kv",
        "cache_seqs",
        "block_split_kvs",
        "softmax_scale",
        "output_scale",
        "stream",
        "use_pdl",
    ]


def test_chain_tree_equals_causal_and_offsets_use_each_request_length():
    q_len = 4
    chain = build_ancestor_mask([-1, 0, 1, 2])
    branch = build_ancestor_mask([-1, 0, 0, 2])
    assert torch.equal(chain, torch.tril(torch.ones(q_len, q_len, dtype=torch.bool)))

    flattened, offsets = flatten_tree_masks([chain, branch], [9, 13])
    assert offsets.tolist() == [0, q_len * 9]
    assert flattened.numel() == q_len * (9 + 13)

    second_request = flattened[offsets[1] :].view(q_len, 13)
    assert second_request[:, : 13 - q_len].all()
    assert torch.equal(second_request[:, 13 - q_len :], branch)


def test_kernel_masks_tree_boundary_tiles_and_clamps_padding_rows():
    source = FP8_SOURCE.read_text()
    assert "self.is_causal or self.tree_mask_mode" in source
    assert source.count("within_q = cute.elem_less(q_tok, self.seq_len_q)") == 2
    assert source.count("q_tok_in =") == 2
    assert source.count("if within_q") >= 2
    assert source.count("within_mask = cute.elem_less(") == 2
    assert source.count("cm_idx_in = cm_idx_in if within_mask") == 2


def test_public_contract_is_limited_to_full_history_tree_masks():
    source = DECODE_SOURCE.read_text()
    assert "Every history column" in source
    assert "not a general sparse/window attention mask" in source


def test_tree_mask_wrapper_guards_fail_closed_before_kernel_dispatch():
    valid_mask = torch.ones(8, dtype=torch.bool)
    valid_offsets = torch.tensor([0, 4], dtype=torch.int32)

    with pytest.raises(ValueError, match="decode-context parallelism is unsupported"):
        _call_until_tree_mask_validation(custom_mask=valid_mask, cp_world=2)
    with pytest.raises(ValueError, match="supported only for the FP8"):
        _call_until_tree_mask_validation(
            query_dtype=torch.bfloat16, custom_mask=valid_mask
        )
    with pytest.raises(ValueError, match="dtype torch.bool"):
        _call_until_tree_mask_validation(
            custom_mask=valid_mask.to(torch.int8), cmask_off=valid_offsets
        )
    with pytest.raises(ValueError, match="contiguous flattened 1-D"):
        _call_until_tree_mask_validation(
            custom_mask=valid_mask.view(2, 4), cmask_off=valid_offsets
        )
    with pytest.raises(ValueError, match="too small"):
        _call_until_tree_mask_validation(
            custom_mask=valid_mask[:3], cmask_off=valid_offsets
        )
    with pytest.raises(ValueError, match="exactly 2 entries"):
        _call_until_tree_mask_validation(
            custom_mask=valid_mask, cmask_off=valid_offsets[:1]
        )
    with pytest.raises(ValueError, match="cmask_off requires custom_mask"):
        _call_until_tree_mask_validation(cmask_off=valid_offsets)


def test_tree_mask_keeps_upstream_dcp_kernel_plumbing():
    decode_source = DECODE_SOURCE.read_text()
    fp8_source = FP8_SOURCE.read_text()
    assert 'kernel_kwargs["cp_world"] = cp_world' in decode_source
    assert "call_args.append(causal_seqs_dev)" in decode_source
    assert "K_causal=causal_seqs[blk_coord[2]]" in fp8_source
    assert "if cutlass.const_expr(self.cp_world == 1)" in fp8_source


def test_upstream_h16_m64_dispatch_remains_present():
    decode_source = DECODE_SOURCE.read_text()
    fp8_source = FP8_SOURCE.read_text()
    assert "select_mla_decode_tilers(" in decode_source
    assert "mma_qk_tiler_mn[0] == 64" in decode_source
    assert "self.use_m64_ws" in fp8_source
    assert "self.cluster_shape_mnk = (1, 1, 1)" in fp8_source


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)),
    reason="TokenSpeed MLA tree-mask kernel requires Blackwell SM100",
)
def test_tree_mask_kernel_matches_reference_and_negative_control():
    module = _load_microbench()
    args = SimpleNamespace(iters=0, warmup=0)

    # q=4 selects the official 0.1.8 M64 path; K=q makes the negative
    # control load-bearing. q=6 numerically exercises padded M rows, and
    # q=8/K=129 covers the M128 padded-K path at bs=2.
    assert module.run_case(1, 4, 4, 0.2, set(), args)
    assert module.run_case(1, 6, 6, 0.2, set(), args)
    assert module.run_case(2, 8, 129, 0.2, set(), args)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)),
    reason="TokenSpeed MLA tree-mask kernel requires Blackwell SM100",
)
def test_tree_mask_cuda_graph_capture_replay_and_zero_offset_cache():
    module = _load_microbench()
    import tokenspeed_mla.mla_decode as decode_module

    q_len = seq_k = 4
    query, kv_cache, block_tables, seq_lens, workspace = module.build_inputs(
        1, q_len, seq_k, seed=404
    )
    ancestor = module.random_tree_ancestor(1, q_len).to("cuda")
    custom_mask = module.build_custom_mask(1, q_len, seq_k, ancestor)
    explicit_zero = torch.zeros(1, dtype=torch.int32, device="cuda")

    eager_reference = module.call(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace,
        seq_k,
        causal=False,
        cmask=custom_mask,
        cmask_off=explicit_zero,
    ).float()
    torch.cuda.synchronize()

    decode_module._ZERO_CMASK_OFF_CACHE.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = module.call(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            workspace,
            seq_k,
            causal=False,
            cmask=custom_mask,
        )

    # Before the first replay, an eager call must not observe an uninitialized
    # zero-offset tensor that escaped from graph capture into the eager cache.
    eager_after_capture = module.call(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace,
        seq_k,
        causal=False,
        cmask=custom_mask,
    ).float()
    torch.cuda.synchronize()
    assert torch.allclose(eager_after_capture, eager_reference, atol=0.2, rtol=0)

    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(graph_output.float(), eager_reference, atol=0.2, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
