# Copyright (c) 2026 LightSeek Foundation
#
# Microbench / correctness harness for tree-mask MLA decode (EAGLE MLA tree arc).
# Run on a GB200 (SM100), e.g. inside the prod image with the worktree tokenspeed_mla
# bind-mounted. Validates the tree custom_mask path in tokenspeed_mla_decode, which
# reads SGLang's flattened custom_mask directly (per request row-major [q_len x K],
# True=attend; history cols all-True, last q_len cols carry the tree ancestry):
#
#   GATE (correctness): for BOTH a chain ancestor and a random tree, the kernel
#           output with custom_mask must match an independent absorbed-MLA float
#           SDPA reference (built from the same ancestor) within FP8_TOL. The mask
#           is built by build_custom_mask (the layout build_tree_kernel_efficient
#           writes); the reference reads the ancestor directly, so the check is NOT
#           circular — it validates the custom_mask layout, kernel indexing, and
#           predicate against float math.
#   INFORMATIONAL: chain custom_mask vs the trusted causal kernel — a chain tree IS
#           causal, so these match up to fp8 reduction-order noise (~1e-2).
#   Cases include NON-tile-aligned K (250/300/129; tile_n=128) and bs=2 to exercise
#           the tile-padding columns and the per-request custom_mask offset.
#
# Usage:
#   python microbench_tree_decode.py
#   python microbench_tree_decode.py --kimi-shape --iters 50
import argparse
import math
import torch
from tokenspeed_mla import tokenspeed_mla_decode

DEV = "cuda"
FP8 = torch.float8_e4m3fn
KV_LORA, ROPE = 512, 64
D = KV_LORA + ROPE
H = 128
PAGE = 64


def build_inputs(B, q_len, seq_k, seed=0):
    """Paged MLA decode inputs. seq_k = total KV per request (history + q_len new)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    query = (torch.randn(B, q_len, H, D, device=DEV, generator=g) * 0.3).to(FP8)
    pages_per = (seq_k + PAGE - 1) // PAGE
    num_pages = pages_per * B
    kv_cache = (torch.randn(num_pages, PAGE, D, device=DEV, generator=g) * 0.3).to(FP8)
    block_tables = torch.arange(num_pages, device=DEV, dtype=torch.int32).view(B, pages_per)
    seq_lens = torch.full((B,), seq_k, device=DEV, dtype=torch.int32)
    workspace = torch.empty(1 << 29, dtype=torch.int8, device=DEV)  # 512MB, generous
    return query, kv_cache, block_tables, seq_lens, workspace


def chain_ancestor(B, q_len):
    a = torch.tril(torch.ones(q_len, q_len, dtype=torch.int32))
    return a.unsqueeze(0).expand(B, q_len, q_len).contiguous()


def random_tree_ancestor(B, q_len, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(B, q_len, q_len, dtype=torch.int32)
    for b in range(B):
        parent = [-1] + [int(torch.randint(0, i, (1,), generator=g)) for i in range(1, q_len)]
        for qi in range(q_len):
            j = qi
            while j != -1:
                out[b, qi, j] = 1
                j = parent[j]
    return out


def build_custom_mask(B, q_len, seq_k, ancestor):
    """SGLang FULL_MASK layout: per request row-major [q_len x K] bool (True=attend),
    concatenated over requests. History cols (< hist) all True; the last q_len cols
    carry the tree ancestry (col hist+j set iff j is an ancestor of q_tok). This is
    exactly what build_tree_kernel_efficient writes into spec_info.custom_mask."""
    hist = seq_k - q_len
    m = torch.zeros(B, q_len, seq_k, dtype=torch.bool, device=DEV)
    m[:, :, :hist] = True
    for b in range(B):
        for qi in range(q_len):
            for j in range(q_len):
                if ancestor[b, qi, j]:
                    m[b, qi, hist + j] = True
    return m.reshape(-1).contiguous()


def call(query, kv_cache, block_tables, seq_lens, workspace, seq_k, *, causal, cmask=None):
    return tokenspeed_mla_decode(
        query=query, kv_cache=kv_cache, workspace_buffer=workspace,
        kv_lora_rank=KV_LORA, qk_rope_head_dim=ROPE, block_tables=block_tables,
        seq_lens=seq_lens, max_seq_len=seq_k, softmax_scale=1.0 / math.sqrt(D),
        causal_mask=causal, custom_mask=cmask,
    )


def measure_ms(fn, *, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def absorbed_ref(query, kv_cache, block_tables, seq_lens, q_len, seq_k, ancestor):
    """Absorbed-MLA decode reference with tree mask. out[b,qi,h,:KV_LORA]."""
    B = query.shape[0]
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros(B, q_len, H, KV_LORA, device=DEV, dtype=torch.float32)
    qf = query.float()
    for b in range(B):
        Kb = int(seq_lens[b]); hist = Kb - q_len
        # gather this request's KV [Kb, D] from pages
        pages = block_tables[b]
        kv = kv_cache[pages].reshape(-1, D).float()[:Kb]   # [Kb, D]
        for qi in range(q_len):
            valid = torch.zeros(Kb, dtype=torch.bool, device=DEV)
            valid[:hist] = True
            for j in range(q_len):
                if ancestor[b, qi, j]:
                    valid[hist + j] = True
            s = (qf[b, qi] @ kv.t()) * scale          # [H, Kb]
            s = s.masked_fill(~valid.view(1, -1), float("-inf"))
            p = torch.softmax(s, dim=-1)              # [H, Kb]
            out[b, qi] = p @ kv[:, :KV_LORA]          # [H, KV_LORA]
    return out


def parse_cases(cases_arg):
    cases = []
    for raw_case in cases_arg.split(";"):
        raw_case = raw_case.strip()
        if not raw_case:
            continue
        parts = [int(x.strip()) for x in raw_case.split(",")]
        if len(parts) != 3:
            raise ValueError(f"case must be B,q_len,seq_k; got {raw_case!r}")
        B, q_len, seq_k = parts
        if B <= 0 or q_len <= 0 or seq_k < q_len:
            raise ValueError(f"invalid case {raw_case!r}")
        cases.append((B, q_len, seq_k))
    if not cases:
        raise ValueError("--cases did not contain any cases")
    return cases


def parse_expected_q(raw):
    if not raw:
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def run_case(B, q_len, seq_k, fp8_tol, expected_unsupported_q, args):
    q, kv, bt, sl, ws = build_inputs(B, q_len, seq_k, seed=B * 1000 + seq_k)
    unsupported_expected = q_len in expected_unsupported_q
    # GATE: tree output vs float SDPA reference for BOTH a chain ancestor and a
    # random tree. This is the absolute correctness check (a chain is just a
    # specific ancestor pattern; the kernel applies ancestors uniformly).
    for name, anc in (
        ("chain", chain_ancestor(B, q_len).to(DEV)),
        ("random", random_tree_ancestor(B, q_len).to(DEV)),
    ):
        cmask = build_custom_mask(B, q_len, seq_k, anc)
        try:
            o_tree = call(q, kv, bt, sl, ws, seq_k, causal=False, cmask=cmask).float()
        except ValueError as exc:
            if unsupported_expected:
                print(
                    f"[B={B} q={q_len} K={seq_k}] {name}-tree unsupported as expected: {exc}"
                )
                return True
            raise
        if unsupported_expected:
            print(
                f"[B={B} q={q_len} K={seq_k}] expected q={q_len} "
                "to be unsupported, but decode ran"
            )
            return False
        o_ref = absorbed_ref(q, kv, bt, sl, q_len, seq_k, anc)
        d = (o_tree - o_ref).abs().max().item()
        ok = d < fp8_tol
        print(
            f"[B={B} q={q_len} K={seq_k}] {name}-tree vs float-ref "
            f"Delta={d:.2e} {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            return False

    # INFORMATIONAL: chain-tree vs causal kernel -- byte-identical only when the
    # two compiles share fp8 reduction order (holds for tile-aligned K); a small
    # Delta (~1e-2, same magnitude as the fp8 ref error) is reduction-order noise.
    o_causal = call(q, kv, bt, sl, ws, seq_k, causal=True).float()
    chain_cmask = build_custom_mask(B, q_len, seq_k, chain_ancestor(B, q_len).to(DEV))
    o_chain = call(q, kv, bt, sl, ws, seq_k, causal=False, cmask=chain_cmask).float()
    print(
        "            chain-tree vs causal-kernel "
        f"Delta={(o_causal - o_chain).abs().max().item():.2e} "
        "(fp8 reduction-order; informational)"
    )
    if args.iters > 0:
        random_cmask = build_custom_mask(
            B, q_len, seq_k, random_tree_ancestor(B, q_len).to(DEV)
        )
        causal_ms = measure_ms(
            lambda: call(q, kv, bt, sl, ws, seq_k, causal=True),
            warmup=args.warmup,
            iters=args.iters,
        )
        tree_ms = measure_ms(
            lambda: call(q, kv, bt, sl, ws, seq_k, causal=False, cmask=random_cmask),
            warmup=args.warmup,
            iters=args.iters,
        )
        print(
            f"            timing causal={causal_ms:.4f} ms tree={tree_ms:.4f} ms "
            f"(warmup={args.warmup}, iters={args.iters})"
        )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument(
        "--kimi-shape",
        action="store_true",
        help="Use Kimi EAGLE3 verify shape defaults: H=16, page32, q8/q16 cases.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help=(
            "Semicolon-separated B,q_len,seq_k cases. Defaults preserve the "
            "original harness, or Kimi q8/q16 cases with --kimi-shape."
        ),
    )
    parser.add_argument(
        "--expect-unsupported-q",
        default="",
        help="Comma-separated q_len values expected to fail current dispatch, e.g. 16.",
    )
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    global H, PAGE
    if args.kimi_shape:
        H = 16
        PAGE = 32
    if args.num_heads is not None:
        H = args.num_heads
    if args.page_size is not None:
        PAGE = args.page_size

    if args.cases is not None:
        cases = parse_cases(args.cases)
    elif args.kimi_shape:
        cases = [(1, 8, 250), (1, 8, 300), (2, 8, 250), (1, 16, 250), (2, 16, 250)]
    else:
        # Cases include NON-tile-aligned K (250, 300, 129; tile_n=128) to exercise the
        # tile-padding columns (kcol >= K) that the original seq_k=256 missed and that
        # caused the OOB anc_bits read on the server. Plus dt=8 (server config) and bs=2.
        cases = [(1, 4, 256), (1, 8, 250), (1, 8, 300), (1, 8, 129), (2, 8, 250)]
    expected_unsupported_q = parse_expected_q(args.expect_unsupported_q)
    print(f"shape: H={H}, PAGE={PAGE}, cases={cases}")
    FP8_TOL = 0.2  # absolute correctness tolerance vs float SDPA reference (fp8 KV/Q)
    allpass = True
    for (B, q_len, seq_k) in cases:
        allpass = (
            run_case(B, q_len, seq_k, FP8_TOL, expected_unsupported_q, args)
            and allpass
        )
    print("ALL PASS (float-ref correctness gate)" if allpass else "SOME FAILED")


if __name__ == "__main__":
    main()
