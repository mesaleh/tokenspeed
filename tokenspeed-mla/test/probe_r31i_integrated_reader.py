# Copyright (c) 2026 LightSeek Foundation

"""Compile and smoke-test R31's packed RoPE inside the production M64 reader."""

import math
import os
import statistics

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack
from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
)
from tokenspeed_mla.utils import get_max_active_clusters

PAGE_SIZE = 32
LATENT_DIM = 512
ROPE_DIM = 64
HEADS = 8
E2M1 = torch.tensor(
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


def _as_cute_tensor(tensor: torch.Tensor, dtype, leading_dim: int, alignment: int):
    result = from_dlpack(tensor, assumed_align=alignment, enable_tvm_ffi=True)
    result.element_type = dtype
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _pack(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def _decode(packed: torch.Tensor, *, swap_nibbles: bool = False) -> torch.Tensor:
    first = packed >> 4 if swap_nibbles else packed & 0xF
    second = packed & 0xF if swap_nibbles else packed >> 4
    codes = torch.stack((first, second), dim=-1).flatten(-2)
    return E2M1.to(packed.device)[codes.long()]


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("R31I requires SM100")

    device = torch.device("cuda")
    batch = 1
    query_len = 1
    seq_len = int(os.environ.get("R31I_SEQ_LEN", "129"))
    mode = os.environ.get("R31I_MODE", "r31")
    if mode not in ("r31", "n10"):
        raise ValueError(f"unsupported R31I_MODE={mode!r}")
    use_r31 = mode == "r31"
    pages = math.ceil(seq_len / PAGE_SIZE)

    torch.manual_seed(0xA1731)
    query_latent = (
        torch.randint(
            -8,
            9,
            (batch, query_len, HEADS, LATENT_DIM),
            device=device,
        )
        / 8.0
    ).to(torch.float8_e4m3fn)
    query_rope_values = torch.randint(
        -8,
        9,
        (batch, query_len, HEADS, (4 if use_r31 else 1) * ROPE_DIM),
        device=device,
    ) / 8.0
    query_rope = query_rope_values.to(
        torch.float8_e4m3fn if use_r31 else torch.bfloat16
    )
    latent_codes = torch.randint(
        0,
        16,
        (pages, PAGE_SIZE, LATENT_DIM),
        dtype=torch.uint8,
        device=device,
    )
    packed_latent = _pack(latent_codes).contiguous()
    cache_rope_values = torch.randint(
        -8,
        9,
        (pages, PAGE_SIZE, ROPE_DIM),
        device=device,
    ) / 8.0
    high_rope = cache_rope_values.to(
        torch.float8_e4m3fn if use_r31 else torch.bfloat16
    )
    residual_codes = torch.randint(
        0,
        16,
        (pages, PAGE_SIZE, ROPE_DIM),
        dtype=torch.uint8,
        device=device,
    )
    component = os.environ.get("R31I_COMPONENT", "both") if use_r31 else "both"
    if component == "high":
        residual_codes.zero_()
    elif component == "residual":
        high_rope.zero_()
    elif component != "both":
        raise ValueError(f"unsupported R31I_COMPONENT={component!r}")
    residual_payload = _pack(residual_codes)
    residual_rope = torch.cat(
        (residual_payload, torch.zeros_like(residual_payload)),
        dim=-1,
    ).contiguous()
    scale = torch.ones(
        (pages, PAGE_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    block_tables = torch.arange(pages, dtype=torch.int32, device=device)[None]
    padded_pages = math.ceil(pages / 4) * 4
    if pages < padded_pages:
        block_tables = torch.cat(
            (block_tables, block_tables[:, -1:].expand(1, padded_pages - pages)),
            dim=1,
        ).contiguous()
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    output = torch.empty_like(query_latent, dtype=torch.bfloat16)
    lse = torch.empty(
        (batch, query_len, HEADS),
        dtype=torch.float32,
        device=device,
    )

    kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(64, 128),
        mma_pv_tiler_mn=(64, 256),
        max_active_clusters=get_max_active_clusters(1),
        page_size=PAGE_SIZE,
        skip_correction_threshold=0.0,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=False,
        fold_sq_factor=1,
        is_causal=False,
        num_heads=HEADS,
        seq_len_q=query_len,
        cp_world=1,
        use_tq_e2m1=True,
        use_tq_r31_rope=use_r31,
        tq_s1_scale_tma=True,
        tq_s1_scale_stages=3,
        tq_s1_k_rope_stages=(
            int(os.environ.get("R31I_ROPE_STAGES", "1")) if use_r31 else 2
        ),
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel,
        _as_cute_tensor(query_latent, cutlass.Float8E4M3FN, 3, 16),
        _as_cute_tensor(
            query_rope,
            cutlass.Float8E4M3FN if use_r31 else cutlass.BFloat16,
            3,
            16,
        ),
        _as_cute_tensor(packed_latent, cutlass.Uint8, 2, 16),
        _as_cute_tensor(
            high_rope,
            cutlass.Float8E4M3FN if use_r31 else cutlass.BFloat16,
            2,
            16,
        ),
        _as_cute_tensor(block_tables, cutlass.Int32, 1, 4),
        _as_cute_tensor(output, cutlass.BFloat16, 3, 16),
        _as_cute_tensor(lse, cutlass.Float32, 2, 4),
        None,
        Int32(1),
        _as_cute_tensor(seq_lens, cutlass.Int32, 0, 4),
        _as_cute_tensor(seq_lens, cutlass.Int32, 0, 4),
        None,
        Float32(1.0 / math.sqrt(LATENT_DIM + ROPE_DIM)),
        Float32(1.0),
        stream,
        False,
        _as_cute_tensor(scale, cutlass.BFloat16, 1, 16),
        None,
        _as_cute_tensor(residual_rope, cutlass.Uint8, 2, 16) if use_r31 else None,
        options="--enable-tvm-ffi --opt-level 3",
    )

    import tvm_ffi

    def launch() -> None:
        compiled(
            query_latent,
            query_rope,
            packed_latent,
            high_rope,
            block_tables,
            output,
            lse,
            None,
            Int32(1),
            seq_lens,
            seq_lens,
            None,
            Float32(1.0 / math.sqrt(LATENT_DIM + ROPE_DIM)),
            Float32(1.0),
            scale,
            None,
            residual_rope if use_r31 else None,
        )

    with tvm_ffi.use_torch_stream():
        launch()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all() or not torch.isfinite(lse).all():
        raise AssertionError("R31I produced non-finite output")

    latent = _decode(packed_latent).reshape(-1, LATENT_DIM)[:seq_len]
    high = high_rope.reshape(-1, ROPE_DIM)[:seq_len].float()
    residual = _decode(
        residual_rope[..., : ROPE_DIM // 2],
        swap_nibbles=os.environ.get("R31I_SWAP_REFERENCE") == "1",
    ).reshape(-1, ROPE_DIM)[:seq_len]
    score = torch.einsum(
        "bqhd,kd->bqhk",
        query_latent.float(),
        latent,
    )
    if use_r31:
        q_rope_planes = query_rope.view(
            batch,
            query_len,
            HEADS,
            4,
            ROPE_DIM,
        ).float()
        score += torch.einsum("bqhd,kd->bqhk", q_rope_planes[:, :, :, 0], high)
        score += torch.einsum("bqhd,kd->bqhk", q_rope_planes[:, :, :, 1], high)
        score += torch.einsum(
            "bqhd,kd->bqhk",
            q_rope_planes[:, :, :, 2],
            residual,
        )
        score += torch.einsum(
            "bqhd,kd->bqhk",
            q_rope_planes[:, :, :, 3],
            residual,
        )
    else:
        score += torch.einsum("bqhd,kd->bqhk", query_rope.float(), high)
    score *= 1.0 / math.sqrt(LATENT_DIM + ROPE_DIM)
    expected = torch.softmax(score, dim=-1) @ latent
    expected_lse = torch.logsumexp(score, dim=-1) * math.log2(math.e)
    torch.testing.assert_close(output.float(), expected, atol=0.75, rtol=0.35)
    torch.testing.assert_close(lse, expected_lse, atol=0.2, rtol=0.02)
    print(
        f"PASS mode={mode} integrated_reader=True "
        f"output_abs_max={output.float().abs().max().item():.6f} "
        f"lse_abs_max={lse.abs().max().item():.6f}"
    )
    if os.environ.get("R31I_BENCH") == "1":
        replays = int(os.environ.get("R31I_REPLAYS", "100"))
        windows = int(os.environ.get("R31I_WINDOWS", "10"))
        with tvm_ffi.use_torch_stream():
            graph = torch.cuda.CUDAGraph()
            launch()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                launch()
        torch.cuda.synchronize()

        def measure() -> float:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) * 1000.0 / replays

        for _ in range(12):
            measure()
        samples = [measure() for _ in range(windows)]
        print(
            "R31I_TIMING "
            f"mode={mode} seq_len={seq_len} "
            f"mean_us={statistics.fmean(samples):.6f} "
            f"median_us={statistics.median(samples):.6f} "
            f"min_us={min(samples):.6f} max_us={max(samples):.6f} "
            f"windows={windows} replays={replays}",
            flush=True,
        )


if __name__ == "__main__":
    main()
