# Copyright (c) 2026 LightSeek Foundation

"""R31-R5 F1 gate for one native FP8/R31 producer grid."""

import argparse
import json
import math
import statistics

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

import benchmark_mla_decode_tq_r31_segmented as fixture
from tokenspeed_mla import (
    reduce_mla_mixed_workspace,
    tokenspeed_mla_decode,
    tokenspeed_mla_decode_tq_r31,
    tokenspeed_mla_decode_tq_r31_mixed,
    tokenspeed_mla_decode_tq_r31_mixed_split_query,
)
from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
)
from tokenspeed_mla.mla_decode_tq_e2m1 import _as_cute_tensor
from tokenspeed_mla.mla_decode_tq_r31_mixed_native import (
    BlackwellMixedFP8R31Producer,
)
from tokenspeed_mla.mla_helpers import get_mla_decode_fold_sq_factor
from tokenspeed_mla.utils import get_max_active_clusters

PAGE = fixture.PAGE
LATENT = fixture.LATENT
ROPE = fixture.ROPE
HEADS = fixture.HEADS
QUERY_LEN = fixture.QUERY_LEN
BATCH = fixture.BATCH
SEQ_LEN = fixture.SEQ_LEN
SOFTMAX_SCALE = fixture.SOFTMAX_SCALE
HOT_SPLITS = 8
COLD_SPLITS = 9
QK_TILER = (64, 128)
PV_TILER = (64, 256)
WRITER_DELTA_US = 1.7013013164202375
Q5_COMPONENT_CEILINGS_US = {1: 13.5067, 5: 29.2079, 8: 37.5659}
NUM_LAYERS = 61
ARENA_LAYER = 17
R31_TOKEN_BYTES = 354


def _as_arena_component(
    raw: torch.Tensor,
    source: torch.Tensor,
    *,
    page_stride_bytes: int,
    offset_bytes: int,
) -> torch.Tensor:
    itemsize = source.element_size()
    if page_stride_bytes % itemsize or offset_bytes % itemsize:
        raise AssertionError("arena component offset/stride must align to dtype")
    view = torch.as_strided(
        raw.view(source.dtype),
        size=source.shape,
        stride=(page_stride_bytes // itemsize, *source.stride()[1:]),
        storage_offset=offset_bytes // itemsize,
    )
    view.copy_(source)
    return view


def _move_cold_to_layer_sharded_arena(state) -> torch.Tensor:
    """Move one R31 layer into its exact full-context arena envelope."""

    num_pages = state.packed_cold.shape[0]
    layer_bytes = PAGE * R31_TOKEN_BYTES
    layer_base = ARENA_LAYER * (num_pages + 1) * layer_bytes
    raw = torch.empty(
        NUM_LAYERS * (num_pages + 1) * layer_bytes,
        dtype=torch.uint8,
        device="cuda",
    )
    packed_offset = layer_base
    scale_offset = packed_offset + PAGE * (LATENT // 2)
    high_offset = scale_offset + PAGE * 2
    residual_offset = high_offset + PAGE * ROPE
    state.packed_cold = _as_arena_component(
        raw,
        state.packed_cold,
        page_stride_bytes=layer_bytes,
        offset_bytes=packed_offset,
    )
    state.cold_scale = _as_arena_component(
        raw,
        state.cold_scale,
        page_stride_bytes=layer_bytes,
        offset_bytes=scale_offset,
    )
    state.cold_high = _as_arena_component(
        raw,
        state.cold_high,
        page_stride_bytes=layer_bytes,
        offset_bytes=high_offset,
    )
    state.cold_residual = _as_arena_component(
        raw,
        state.cold_residual,
        page_stride_bytes=layer_bytes,
        offset_bytes=residual_offset,
    )
    return raw


def _move_hot_to_layer_sharded_arena(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one FP8 layer into its exact full-context arena envelope."""

    num_pages = source.shape[0]
    layer_bytes = PAGE * (LATENT + ROPE)
    layer_base = ARENA_LAYER * (num_pages + 1) * layer_bytes
    raw = torch.empty(
        NUM_LAYERS * (num_pages + 1) * layer_bytes,
        dtype=torch.uint8,
        device="cuda",
    )
    return (
        _as_arena_component(
            raw,
            source,
            page_stride_bytes=layer_bytes,
            offset_bytes=layer_base,
        ),
        raw,
    )


def _arena_metadata(state, hot_cache: torch.Tensor, layout: str) -> dict[str, object]:
    metadata = {
        "arena_layout": layout,
        "hot_cache_stride": tuple(hot_cache.stride()),
        "cold_cache_strides": {
            "packed": tuple(state.packed_cold.stride()),
            "scale": tuple(state.cold_scale.stride()),
            "high": tuple(state.cold_high.stride()),
            "residual": tuple(state.cold_residual.stride()),
        },
    }
    if layout == "layer-sharded":
        expected = {
            "hot": (PAGE * (LATENT + ROPE), LATENT + ROPE, 1),
            "packed": (PAGE * R31_TOKEN_BYTES, LATENT // 2, 1),
            "scale": (PAGE * R31_TOKEN_BYTES // 2, 1),
            "high": (PAGE * R31_TOKEN_BYTES, ROPE, 1),
            "residual": (PAGE * R31_TOKEN_BYTES, ROPE // 2, 1),
        }
        actual = {
            "hot": tuple(hot_cache.stride()),
            **metadata["cold_cache_strides"],
        }
        if actual != expected:
            raise AssertionError(
                f"layer-sharded arena stride contract mismatch: {actual} != {expected}"
            )
        cold_storage = state.packed_cold.untyped_storage().data_ptr()
        if any(
            component.untyped_storage().data_ptr() != cold_storage
            for component in (
                state.cold_scale,
                state.cold_high,
                state.cold_residual,
            )
        ):
            raise AssertionError("R31 components must share one cold arena backing")
        metadata["exact_stride_contract"] = True
        metadata["cold_components_share_storage"] = True
    return metadata


def _capture(function) -> torch.cuda.CUDAGraph:
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
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


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "stdev_us": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_us": min(samples),
        "max_us": max(samples),
    }


def _compile_mixed_producer(
    hot_query: torch.Tensor,
    hot_cache: torch.Tensor,
    hot_table: torch.Tensor,
    hot_workspace: torch.Tensor,
    hot_seq: torch.Tensor,
    cold_query_latent: torch.Tensor,
    cold_query_rope: torch.Tensor,
    cold_cache: torch.Tensor,
    cold_high_rope: torch.Tensor,
    cold_table: torch.Tensor,
    cold_workspace: torch.Tensor,
    cold_seq: torch.Tensor,
    output_sentinel: torch.Tensor,
    cold_scale: torch.Tensor,
    cold_residual_rope: torch.Tensor,
):
    fold_sq_factor = get_mla_decode_fold_sq_factor(HEADS, QUERY_LEN, QK_TILER[0])
    common = dict(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=QK_TILER,
        mma_pv_tiler_mn=PV_TILER,
        max_active_clusters=get_max_active_clusters(1),
        page_size=PAGE,
        skip_correction_threshold=0.0,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=False,
        fold_sq_factor=fold_sq_factor,
        num_heads=HEADS,
        seq_len_q=QUERY_LEN,
        cp_world=1,
        use_runtime_causal_bound=True,
        producer_only=True,
    )
    hot_kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
        **common,
        is_causal=True,
    )
    use_packed_p_scale_math = BATCH == 1 and QUERY_LEN == 5
    cold_kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
        **common,
        is_causal=True,
        use_tq_e2m1=True,
        use_tq_r31_rope=True,
        tq_s1_scale_tma=True,
        tq_s1_scale_stages=3,
        tq_s1_k_rope_stages=1,
        tq_s1_packed_p_scale_math=use_packed_p_scale_math,
        tq_s1_early_final_pcor=use_packed_p_scale_math,
        tq_r31_async_expand=True,
    )
    kernel = BlackwellMixedFP8R31Producer(hot_kernel, cold_kernel)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        _as_cute_tensor(hot_query[..., :LATENT], cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(hot_query[..., LATENT:], cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(hot_cache[..., :LATENT], cutlass.Float8E4M3FN, 2, 16),
        _as_cute_tensor(hot_cache[..., LATENT:], cutlass.Float8E4M3FN, 2, 16),
        _as_cute_tensor(hot_table, cutlass.Int32, 1, 4),
        _as_cute_tensor(hot_workspace, cutlass.Int8, 0, 32),
        Int32(1),
        _as_cute_tensor(hot_seq, cutlass.Int32, 0, 4),
        _as_cute_tensor(hot_seq, cutlass.Int32, 0, 4),
        _as_cute_tensor(cold_query_latent, cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(cold_query_rope, cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(cold_cache, cutlass.Uint8, 2, 16),
        _as_cute_tensor(cold_high_rope, cutlass.Float8E4M3FN, 2, 16),
        _as_cute_tensor(cold_table, cutlass.Int32, 1, 4),
        _as_cute_tensor(cold_workspace, cutlass.Int8, 0, 32),
        Int32(1),
        _as_cute_tensor(cold_seq, cutlass.Int32, 0, 4),
        _as_cute_tensor(cold_seq, cutlass.Int32, 0, 4),
        _as_cute_tensor(output_sentinel, cutlass.BFloat16, 3, 16),
        Float32(1.0),
        Float32(1.0),
        _as_cute_tensor(cold_scale, cutlass.BFloat16, 1, 16),
        _as_cute_tensor(cold_residual_rope, cutlass.Uint8, 2, 16),
        None,
        stream,
        options="--enable-tvm-ffi --opt-level 3",
    )


def run(
    windows: int,
    replays: int,
    batch: int,
    query_len: int,
    hot_splits: int,
    cold_splits: int,
    correctness_only: bool = False,
    sanitizer_fixture: str | None = None,
    causal_owner: str = "hot",
    arena_layout: str = "contiguous",
) -> dict[str, object]:
    global BATCH, QUERY_LEN, HOT_SPLITS, COLD_SPLITS
    BATCH = batch
    QUERY_LEN = query_len
    HOT_SPLITS = hot_splits
    COLD_SPLITS = cold_splits
    if causal_owner not in ("hot", "cold"):
        raise ValueError(f"unknown causal owner {causal_owner!r}")
    if arena_layout not in ("contiguous", "layer-sharded"):
        raise ValueError(f"unknown arena layout {arena_layout!r}")
    fixture.BATCH = batch
    fixture.QUERY_LEN = query_len
    fixture.HOT_CAPACITY = 2_560 * batch
    from sglang.kernels.ops.quantization.hadamard import (
        hadamard_transform_with_signs,
    )

    state, config, dense_query, dense_cache, dense_table, dense_seq = (
        fixture._build_state("balanced")
    )
    contiguous_cold = (
        state.packed_cold,
        state.cold_scale,
        state.cold_high,
        state.cold_residual,
    )
    arena_backing = []
    if arena_layout == "layer-sharded":
        arena_backing.append(_move_cold_to_layer_sharded_arena(state))
    torch.manual_seed(0xA173100)
    source = (torch.randn(BATCH, SEQ_LEN, LATENT + ROPE, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    query_source = (
        torch.randn(BATCH, QUERY_LEN, HEADS, LATENT + ROPE, device="cuda") * 0.1
    ).to(torch.bfloat16)
    hot_lengths, cold_lengths = fixture._layout_lengths("balanced")
    hot_source = torch.cat(
        [
            source[request, cold_lengths[request] :]
            for request in range(BATCH)
            if hot_lengths[request]
        ],
        dim=0,
    ).contiguous()
    rotated_hot_latent = torch.empty_like(hot_source[..., :LATENT])
    hadamard_transform_with_signs(
        hot_source[..., :LATENT],
        config.signs1,
        config.signs2,
        scale=1.0 / math.sqrt(LATENT),
        out=rotated_hot_latent,
    )
    hot_cache = torch.cat((rotated_hot_latent, hot_source[..., LATENT:]), dim=-1).to(
        torch.float8_e4m3fn
    )
    hot_cache = hot_cache.view(-1, PAGE, LATENT + ROPE)
    contiguous_hot_cache = hot_cache
    if arena_layout == "layer-sharded":
        hot_cache, hot_arena = _move_hot_to_layer_sharded_arena(hot_cache)
        arena_backing.append(hot_arena)
    arena_metadata = _arena_metadata(state, hot_cache, arena_layout)
    arena_metadata["arena_bytes_allocated"] = sum(
        backing.numel() * backing.element_size() for backing in arena_backing
    )
    hot_query = torch.cat(
        (
            state.r31_query_latent,
            query_source[..., LATENT:].to(torch.float8_e4m3fn),
        ),
        dim=-1,
    ).contiguous()

    rows = BATCH * QUERY_LEN * HEADS
    hot_workspace_bytes = rows * HOT_SPLITS * (LATENT + 1) * 4
    cold_workspace_bytes = rows * COLD_SPLITS * (LATENT + 1) * 4
    shared_workspace = torch.empty(
        hot_workspace_bytes + cold_workspace_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    hot_workspace = shared_workspace[:hot_workspace_bytes]
    cold_workspace = shared_workspace[hot_workspace_bytes:]
    reference_workspace = torch.empty_like(shared_workspace)
    reference_hot_workspace = reference_workspace[:hot_workspace_bytes]
    reference_cold_workspace = reference_workspace[hot_workspace_bytes:]
    contiguous_workspace = (
        torch.empty_like(shared_workspace) if arena_layout == "layer-sharded" else None
    )
    contiguous_hot_workspace = (
        contiguous_workspace[:hot_workspace_bytes]
        if contiguous_workspace is not None
        else None
    )
    contiguous_cold_workspace = (
        contiguous_workspace[hot_workspace_bytes:]
        if contiguous_workspace is not None
        else None
    )
    output_sentinel = torch.empty(
        BATCH, QUERY_LEN, HEADS, LATENT, dtype=torch.bfloat16, device="cuda"
    )
    reference_output = torch.empty_like(output_sentinel)
    reference_lse = torch.empty(
        BATCH, QUERY_LEN, HEADS, dtype=torch.float32, device="cuda"
    )
    contiguous_output = (
        torch.empty_like(output_sentinel) if arena_layout == "layer-sharded" else None
    )
    contiguous_lse = (
        torch.empty_like(reference_lse) if arena_layout == "layer-sharded" else None
    )
    mixed_output = torch.empty_like(output_sentinel)
    mixed_lse = torch.empty_like(reference_lse)
    public_workspace = torch.empty_like(shared_workspace)
    public_output = torch.empty_like(output_sentinel)
    public_lse = torch.empty_like(reference_lse)
    split_query_workspace = torch.empty_like(shared_workspace)
    split_query_output = torch.empty_like(output_sentinel)
    split_query_lse = torch.empty_like(reference_lse)
    hot_query_rope = hot_query[..., LATENT:].contiguous()
    baseline_workspace = torch.empty(
        rows * 64 * (LATENT + 1) * 4, dtype=torch.int8, device="cuda"
    )
    baseline_output = torch.empty_like(output_sentinel)
    baseline_lse = torch.empty_like(reference_lse)
    hot_max_seq_len = int(state.hot_seq.max().item())
    cold_max_seq_len = int(state.prefix_seq.max().item())
    hot_causal_seq = state.hot_seq + ((QUERY_LEN - 1) if causal_owner == "cold" else 0)
    cold_causal_seq = state.prefix_seq + (
        (QUERY_LEN - 1) if causal_owner == "hot" else 0
    )

    compiled = _compile_mixed_producer(
        hot_query,
        hot_cache,
        state.hot_table,
        hot_workspace,
        state.hot_seq,
        state.r31_query_latent,
        state.r31_query_rope,
        state.packed_cold,
        state.cold_high,
        state.prefix_table,
        cold_workspace,
        state.prefix_seq,
        output_sentinel,
        state.cold_scale,
        state.cold_residual,
    )

    def baseline_launch() -> None:
        tokenspeed_mla_decode(
            query=dense_query,
            kv_cache=dense_cache,
            workspace_buffer=baseline_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=dense_table,
            seq_lens=dense_seq,
            max_seq_len=SEQ_LEN,
            softmax_scale=SOFTMAX_SCALE,
            out=baseline_output,
            causal_mask=True,
            enable_pdl=True,
            return_lse=True,
            lse_out=baseline_lse,
        )

    def hot_reference_launch() -> None:
        tokenspeed_mla_decode(
            query=hot_query,
            kv_cache=hot_cache,
            workspace_buffer=reference_hot_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=state.hot_table,
            seq_lens=state.hot_seq,
            max_seq_len=hot_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=causal_owner == "hot",
            enable_pdl=True,
            split_kv_override=HOT_SPLITS,
            producer_only=True,
        )

    def cold_reference_launch() -> None:
        tokenspeed_mla_decode_tq_r31(
            query_latent=state.r31_query_latent,
            query_rope=state.r31_query_rope,
            packed_latent=state.packed_cold,
            reconstruction_scale=state.cold_scale,
            high_rope=state.cold_high,
            residual_rope=state.cold_residual,
            workspace_buffer=reference_cold_workspace,
            block_tables=state.prefix_table,
            seq_lens=state.prefix_seq,
            max_seq_len=cold_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=causal_owner == "cold",
            enable_pdl=True,
            split_kv_override=COLD_SPLITS,
            producer_only=True,
        )

    def reference_launch() -> None:
        hot_reference_launch()
        cold_reference_launch()
        reduce_mla_mixed_workspace(
            reference_workspace,
            state.hot_seq,
            state.prefix_seq,
            HOT_SPLITS,
            COLD_SPLITS,
            reference_output,
            reference_lse,
        )

    def contiguous_oracle_launch() -> None:
        if (
            contiguous_workspace is None
            or contiguous_hot_workspace is None
            or contiguous_cold_workspace is None
            or contiguous_output is None
            or contiguous_lse is None
        ):
            raise AssertionError("contiguous oracle requires layer-sharded mode")
        packed_cold, cold_scale, cold_high, cold_residual = contiguous_cold
        tokenspeed_mla_decode(
            query=hot_query,
            kv_cache=contiguous_hot_cache,
            workspace_buffer=contiguous_hot_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=state.hot_table,
            seq_lens=state.hot_seq,
            max_seq_len=hot_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=causal_owner == "hot",
            enable_pdl=True,
            split_kv_override=HOT_SPLITS,
            producer_only=True,
        )
        tokenspeed_mla_decode_tq_r31(
            query_latent=state.r31_query_latent,
            query_rope=state.r31_query_rope,
            packed_latent=packed_cold,
            reconstruction_scale=cold_scale,
            high_rope=cold_high,
            residual_rope=cold_residual,
            workspace_buffer=contiguous_cold_workspace,
            block_tables=state.prefix_table,
            seq_lens=state.prefix_seq,
            max_seq_len=cold_max_seq_len,
            softmax_scale=SOFTMAX_SCALE,
            out=output_sentinel,
            causal_mask=causal_owner == "cold",
            enable_pdl=True,
            split_kv_override=COLD_SPLITS,
            producer_only=True,
        )
        reduce_mla_mixed_workspace(
            contiguous_workspace,
            state.hot_seq,
            state.prefix_seq,
            HOT_SPLITS,
            COLD_SPLITS,
            contiguous_output,
            contiguous_lse,
        )

    hot_stream = torch.cuda.Stream()
    cold_stream = torch.cuda.Stream()
    fork_event = torch.cuda.Event()
    hot_done_event = torch.cuda.Event()
    cold_done_event = torch.cuda.Event()

    def concurrent_producers_launch() -> None:
        launch_origin = torch.cuda.current_stream()
        fork_event.record(launch_origin)
        hot_stream.wait_event(fork_event)
        cold_stream.wait_event(fork_event)
        with torch.cuda.stream(hot_stream):
            hot_reference_launch()
            hot_done_event.record(hot_stream)
        with torch.cuda.stream(cold_stream):
            cold_reference_launch()
            cold_done_event.record(cold_stream)
        launch_origin.wait_event(hot_done_event)
        launch_origin.wait_event(cold_done_event)

    def concurrent_reference_launch() -> None:
        concurrent_producers_launch()
        reduce_mla_mixed_workspace(
            reference_workspace,
            state.hot_seq,
            state.prefix_seq,
            HOT_SPLITS,
            COLD_SPLITS,
            reference_output,
            reference_lse,
        )

    def mixed_producer_launch() -> None:
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            compiled(
                hot_query[..., :LATENT],
                hot_query[..., LATENT:],
                hot_cache[..., :LATENT],
                hot_cache[..., LATENT:],
                state.hot_table,
                hot_workspace,
                Int32(HOT_SPLITS),
                state.hot_seq,
                hot_causal_seq,
                state.r31_query_latent,
                state.r31_query_rope,
                state.packed_cold,
                state.cold_high,
                state.prefix_table,
                cold_workspace,
                Int32(COLD_SPLITS),
                state.prefix_seq,
                cold_causal_seq,
                output_sentinel,
                Float32(SOFTMAX_SCALE),
                Float32(1.0),
                state.cold_scale,
                state.cold_residual,
                None,
            )

    def mixed_launch() -> None:
        mixed_producer_launch()
        reduce_mla_mixed_workspace(
            shared_workspace,
            state.hot_seq,
            state.prefix_seq,
            HOT_SPLITS,
            COLD_SPLITS,
            mixed_output,
            mixed_lse,
        )

    public_kwargs = {
        "hot_query": hot_query,
        "hot_cache": hot_cache,
        "hot_block_tables": state.hot_table,
        "hot_seq_lens": state.hot_seq,
        "hot_causal_seqs": hot_causal_seq,
        "hot_max_seq_len": hot_max_seq_len,
        "cold_query_latent": state.r31_query_latent,
        "cold_query_rope": state.r31_query_rope,
        "cold_packed_latent": state.packed_cold,
        "cold_reconstruction_scale": state.cold_scale,
        "cold_high_rope": state.cold_high,
        "cold_residual_rope": state.cold_residual,
        "cold_block_tables": state.prefix_table,
        "cold_seq_lens": state.prefix_seq,
        "cold_causal_seqs": cold_causal_seq,
        "cold_max_seq_len": cold_max_seq_len,
        "workspace_buffer": public_workspace,
        "hot_splits": HOT_SPLITS,
        "cold_splits": COLD_SPLITS,
        "softmax_scale": SOFTMAX_SCALE,
        "out": public_output,
        "lse_out": public_lse,
        "enable_pdl": True,
    }

    def public_launch() -> None:
        tokenspeed_mla_decode_tq_r31_mixed(**public_kwargs)

    split_query_kwargs = {
        key: value
        for key, value in public_kwargs.items()
        if key not in {"hot_query", "cold_query_latent"}
    }
    split_query_kwargs.update(
        {
            "query_latent": state.r31_query_latent,
            "hot_query_rope": hot_query_rope,
            "workspace_buffer": split_query_workspace,
            "out": split_query_output,
            "lse_out": split_query_lse,
        }
    )

    def split_query_launch() -> None:
        tokenspeed_mla_decode_tq_r31_mixed_split_query(**split_query_kwargs)

    reference_workspace.view(torch.float32).fill_(float("nan"))
    shared_workspace.view(torch.float32).fill_(float("nan"))
    public_workspace.view(torch.float32).fill_(float("nan"))
    split_query_workspace.view(torch.float32).fill_(float("nan"))
    if contiguous_workspace is not None:
        contiguous_workspace.view(torch.float32).fill_(float("nan"))
        contiguous_oracle_launch()
    reference_launch()
    mixed_launch()
    public_launch()
    split_query_launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(mixed_output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(mixed_lse, reference_lse, rtol=0, atol=0)
    torch.testing.assert_close(public_output, mixed_output, rtol=0, atol=0)
    torch.testing.assert_close(public_lse, mixed_lse, rtol=0, atol=0)
    torch.testing.assert_close(split_query_output, mixed_output, rtol=0, atol=0)
    torch.testing.assert_close(split_query_lse, mixed_lse, rtol=0, atol=0)
    if contiguous_output is not None and contiguous_lse is not None:
        torch.testing.assert_close(mixed_output, contiguous_output, rtol=0, atol=0)
        torch.testing.assert_close(mixed_lse, contiguous_lse, rtol=0, atol=0)
    if not torch.isfinite(mixed_output).all() or not torch.isfinite(mixed_lse).all():
        raise AssertionError("mixed producer consumed a poisoned workspace slot")

    steady_allocated_before = torch.cuda.memory_allocated()
    for _ in range(8):
        public_launch()
        split_query_launch()
    torch.cuda.synchronize()
    steady_allocated_after = torch.cuda.memory_allocated()
    if steady_allocated_after != steady_allocated_before:
        raise AssertionError(
            "public mixed API allocated persistent CUDA storage after warmup: "
            f"before={steady_allocated_before} after={steady_allocated_after}"
        )

    fail_closed_cases = {
        "short_workspace": {
            "workspace_buffer": public_workspace[:-1],
            "error": "mixed decode requires",
        },
        "zero_hot_splits": {"hot_splits": 0, "error": "hot_splits must be"},
        "wrong_hot_causal_dtype": {
            "hot_causal_seqs": hot_causal_seq.to(torch.int64),
            "error": "hot_causal_seqs must have dtype",
        },
        "hot_max_exceeds_table": {
            "hot_max_seq_len": state.hot_table.shape[1] * PAGE + 1,
            "error": "exceeds block-table capacity",
        },
        "nonfinite_softmax_scale": {
            "softmax_scale": float("nan"),
            "error": "softmax_scale must be finite",
        },
        "malformed_hot_row_stride": {
            "hot_cache": torch.as_strided(
                hot_cache,
                size=hot_cache.shape,
                stride=(hot_cache.stride(0), LATENT + ROPE - 1, 1),
            ),
            "error": "hot_cache rows must be contiguous",
        },
        "output_aliases_workspace": {
            "out": public_workspace[: public_output.numel() * 2]
            .view(torch.bfloat16)
            .view_as(public_output),
            "error": "must not alias",
        },
    }
    for case_name, case in fail_closed_cases.items():
        overrides = {key: value for key, value in case.items() if key != "error"}
        try:
            tokenspeed_mla_decode_tq_r31_mixed(**(public_kwargs | overrides))
        except (TypeError, ValueError) as exc:
            if case["error"] not in str(exc):
                raise AssertionError(
                    f"{case_name} raised unexpected error: {exc}"
                ) from exc
        else:
            raise AssertionError(f"{case_name} did not fail closed")

    padded_heads = torch.empty(
        BATCH,
        QUERY_LEN,
        HEADS * 2,
        ROPE,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )[:, :, :HEADS]
    unaligned_storage = torch.empty(
        BATCH * QUERY_LEN * HEADS * (ROPE + 1),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    unaligned_head_stride = torch.as_strided(
        unaligned_storage,
        (BATCH, QUERY_LEN, HEADS, ROPE),
        (
            QUERY_LEN * HEADS * (ROPE + 1),
            HEADS * (ROPE + 1),
            ROPE + 1,
            1,
        ),
    )
    split_fail_closed_cases = {
        "split_query_padded_token_stride": {
            "hot_query_rope": padded_heads,
            "error": "must be row-major",
        },
        "split_query_unaligned_outer_stride": {
            "hot_query_rope": unaligned_head_stride,
            "error": "outer strides must be 16-byte aligned",
        },
    }
    for case_name, case in split_fail_closed_cases.items():
        overrides = {key: value for key, value in case.items() if key != "error"}
        try:
            tokenspeed_mla_decode_tq_r31_mixed_split_query(
                **(split_query_kwargs | overrides)
            )
        except (TypeError, ValueError) as exc:
            if case["error"] not in str(exc):
                raise AssertionError(
                    f"{case_name} raised unexpected error: {exc}"
                ) from exc
        else:
            raise AssertionError(f"{case_name} did not fail closed")

    if sanitizer_fixture is not None:
        torch.save(
            {
                "batch": BATCH,
                "query_len": QUERY_LEN,
                "hot_splits": HOT_SPLITS,
                "cold_splits": COLD_SPLITS,
                "causal_owner": causal_owner,
                "arena_layout": arena_layout,
                "hot_query": hot_query.cpu(),
                "hot_cache": hot_cache.cpu(),
                "hot_table": state.hot_table.cpu(),
                "hot_seq": state.hot_seq.cpu(),
                "hot_causal_seq": hot_causal_seq.cpu(),
                "cold_query_latent": state.r31_query_latent.cpu(),
                "cold_query_rope": state.r31_query_rope.cpu(),
                "cold_cache": state.packed_cold.cpu(),
                "cold_high_rope": state.cold_high.cpu(),
                "cold_table": state.prefix_table.cpu(),
                "cold_seq": state.prefix_seq.cpu(),
                "cold_causal_seq": cold_causal_seq.cpu(),
                "cold_scale": state.cold_scale.cpu(),
                "cold_residual_rope": state.cold_residual.cpu(),
                "reference_workspace": (
                    contiguous_workspace
                    if contiguous_workspace is not None
                    else reference_workspace
                ).cpu(),
            },
            sanitizer_fixture,
        )

    if correctness_only:
        schema = (
            "r31-r5-f2-exact-arena-correctness-v1"
            if arena_layout == "layer-sharded"
            else "r31-r5-f1-mixed-native-correctness-v1"
        )
        return {
            "schema": schema,
            "shape": {
                "batch": BATCH,
                "query_len": QUERY_LEN,
                "sequence_len": SEQ_LEN,
                "hot_splits": HOT_SPLITS,
                "cold_splits": COLD_SPLITS,
                "causal_owner": causal_owner,
                **arena_metadata,
            },
            "correctness": {
                "bit_exact_vs_two_producers": True,
                "bit_exact_public_api_vs_direct_mixed": True,
                "bit_exact_split_query_api_vs_direct_mixed": True,
                "bit_exact_vs_contiguous_oracle": (
                    True if arena_layout == "layer-sharded" else None
                ),
                "poisoned_unwritten_slots_ignored": True,
                "fail_closed_validation_cases": sorted(
                    fail_closed_cases | split_fail_closed_cases
                ),
                "steady_state_cuda_allocation_delta_bytes": 0,
            },
        }

    baseline_graph = _capture(baseline_launch)
    reference_graph = _capture(reference_launch)
    concurrent_graph = _capture(concurrent_reference_launch)
    mixed_graph = _capture(mixed_launch)
    public_graph = _capture(public_launch)
    split_query_graph = _capture(split_query_launch)
    producer_graph = _capture(mixed_producer_launch)
    for _ in range(10):
        _measure(baseline_graph, replays)
        _measure(reference_graph, replays)
        _measure(concurrent_graph, replays)
        _measure(mixed_graph, replays)
        _measure(public_graph, replays)
        _measure(split_query_graph, replays)
    samples = {
        "baseline": [],
        "reference": [],
        "concurrent": [],
        "mixed": [],
        "public": [],
        "split_query": [],
    }
    graph_by_name = {
        "baseline": baseline_graph,
        "reference": reference_graph,
        "concurrent": concurrent_graph,
        "mixed": mixed_graph,
        "public": public_graph,
        "split_query": split_query_graph,
    }
    orders = (
        (
            "baseline",
            "reference",
            "concurrent",
            "mixed",
            "public",
            "split_query",
        ),
        (
            "split_query",
            "public",
            "mixed",
            "concurrent",
            "reference",
            "baseline",
        ),
    )
    for window in range(windows):
        for name in orders[window % len(orders)]:
            samples[name].append(_measure(graph_by_name[name], replays))
    producer_samples = [_measure(producer_graph, replays) for _ in range(windows)]
    timing = {name: _summary(values) for name, values in samples.items()}
    timing["mixed_producer"] = _summary(producer_samples)
    delta_us = timing["mixed"]["mean_us"] - timing["reference"]["mean_us"]
    attention_added_us = timing["public"]["mean_us"] - timing["baseline"]["mean_us"]
    charged_complete_added_us = attention_added_us + WRITER_DELTA_US
    # Only q5 has a fresh production-control TPOT budget.  q1 remains useful
    # as a descriptive shape guard, but must not emit a formal pass/fail until
    # its control is rerun under the same request contract.
    component_ceiling_us = Q5_COMPONENT_CEILINGS_US[BATCH] if QUERY_LEN == 5 else None
    schema = (
        "r31-r5-f2-exact-arena-v1"
        if arena_layout == "layer-sharded"
        else "r31-r5-f1-mixed-native-v1"
    )
    return {
        "schema": schema,
        "shape": {
            "batch": BATCH,
            "query_len": QUERY_LEN,
            "sequence_len": SEQ_LEN,
            "hot_splits": HOT_SPLITS,
            "cold_splits": COLD_SPLITS,
            "causal_owner": causal_owner,
            **arena_metadata,
        },
        "correctness": {
            "bit_exact_vs_two_producers": True,
            "bit_exact_public_api_vs_direct_mixed": True,
            "bit_exact_split_query_api_vs_direct_mixed": True,
            "bit_exact_vs_contiguous_oracle": (
                True if arena_layout == "layer-sharded" else None
            ),
            "poisoned_unwritten_slots_ignored": True,
            "fail_closed_validation_cases": sorted(
                fail_closed_cases | split_fail_closed_cases
            ),
            "steady_state_cuda_allocation_delta_bytes": 0,
        },
        "timing": timing,
        "mixed_minus_reference_us": delta_us,
        "public_minus_mixed_us": (
            timing["public"]["mean_us"] - timing["mixed"]["mean_us"]
        ),
        "split_query_minus_public_us": (
            timing["split_query"]["mean_us"] - timing["public"]["mean_us"]
        ),
        "mixed_minus_concurrent_us": (
            timing["mixed"]["mean_us"] - timing["concurrent"]["mean_us"]
        ),
        "attention_added_us": attention_added_us,
        "writer_delta_us": WRITER_DELTA_US,
        "charged_complete_added_us": charged_complete_added_us,
        "component_ceiling_us": component_ceiling_us,
        "charged_complete_within_ceiling": (
            charged_complete_added_us <= component_ceiling_us
            if component_ceiling_us is not None
            else None
        ),
        "mixed_faster": delta_us < 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--batch", type=int, choices=(1, 5, 8), default=8)
    parser.add_argument("--query-len", type=int, choices=(1, 5), default=5)
    parser.add_argument("--hot-splits", type=int, default=3)
    parser.add_argument("--cold-splits", type=int, default=14)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--sanitizer-fixture")
    parser.add_argument("--causal-owner", choices=("hot", "cold"), default="hot")
    parser.add_argument(
        "--arena-layout",
        choices=("contiguous", "layer-sharded"),
        default="contiguous",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.windows,
                args.replays,
                args.batch,
                args.query_len,
                args.hot_splits,
                args.cold_splits,
                args.correctness_only,
                args.sanitizer_fixture,
                args.causal_owner,
                args.arena_layout,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
