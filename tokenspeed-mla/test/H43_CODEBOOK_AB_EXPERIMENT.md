# H43 D1 same-source codebook reader A/B

Status: frozen diagnostic plan; no harness or live run yet
TokenSpeed base: `a9af4c57f682e800c42cc53063e5f013369210a9`
SGLang consumer base: `ed7b2c905` (no SGLang change in D1)
Machine scope: local source work plus CT13 GPU0 for an isolated final run. Both
accepted endpoint ranks on CT13 and CT14 must be stopped and restored together;
CT14 runs no experimental process. CT15, CT16, and SecurityLLMs are excluded.

## Corrected hypothesis

H42 D1 reproduced the historical-long N14 integrated miss at `33.099257 us`
per selected layer, but every predeclared scheduling contrast was smaller than
`0.65 us/layer`. Its post-result audit found that the component estimate was
not representation-matched:

- H41 W1 passed a materialized `[page, row, 16]` raw-FP8 codebook to the
  selected reader and measured approximately `14.11-14.19 us/layer` versus
  dense at historical long.
- H41 I1 and H42 passed `kv_nope_codebook=None` and used the scale/centroid
  reconstruction specialization.

The two TokenSpeed specializations retain the same packed indices, BF16 scale,
FP8 RoPE, query, page table, split, tensor-core schedule, and output type. The
codebook path replaces read-time FP32 scale/centroid reconstruction and
cross-lane conversion with the accepted vectorized PRMT lookup. H43 D1 tests
that single specialization difference directly. It does not assume that the
historical W1/I1 subtraction is causal.

## Prior evidence and non-duplication

Phase 17's scalar codebook lookup was neutral to slower and remains rejected.
Phase 18's vectorized PRMT path passed raw-word and q1-q5 correctness,
sanitizer, resource, and actual-10K A/B/A gates; it is the primitive currently
present in TokenSpeed. Phase 19 proved a correct fused writer/lifecycle but an
all-61-layer 402-byte deployment remained about 22% slower than FP8. H43 does
not repeat that all-layer endpoint result. It asks whether the already-selected
N14 FP8-RoPE topology gains enough at historical long to close H41 I1 while
retaining most of its memory saving.

Current public-code recheck on 2026-07-29 found no replacement: vLLM PR
`41803` remains at `f3447918`, excludes MTP/spec decode and CUDA/CUTLASS kernels,
and reports the 4-bit MLA path at about `0.75-0.76x` dense throughput; SGLang PR
`23135` remains at `7a2dc8a`, targets generic Triton attention rather than Kimi
SM100 q5 MLA. No public `kv_nope_codebook` implementation was found.

## Frozen cache and graph contract

Both timed arms share one simultaneously resident full 61-layer cache ring:

- 24 dense FP8 layers, 14 selected TQ layers `24-37`, then 23 dense FP8 layers;
- 256,000 physical rows, page size 32, content-matched finite nonzero values,
  the H41 shuffled page table, batch 1, q_len 5, and eight query heads;
- selected row before the optional codebook: 256 packed latent bytes, 2 BF16
  scale bytes, and 64 FP8 RoPE bytes (`322 B` total);
- fixed contexts/splits: `10,219/64` and `37,932/40`;
- one fixed FP8 query shared by every reader call, matching H41 W1;
- one shared preallocated reader workspace and separate preallocated outputs;
- PDL on for dense and selected readers.

The control graph passes `kv_nope_codebook=None` to all 14 selected calls. The
test graph passes the same contiguous, 16-byte-aligned codebook tensor. Both
graphs execute the same 61 reader calls in the same order and consume the same
packed/scale/rope tensors; the codebook allocation is resident during both
arms. There is no frontend or current-token writer in D1, so this experiment
isolates reader benefit and cannot authorize integration by itself.

Construct each codebook byte exactly as:

`(Float32(BF16(scale)) * Float32(centroid)).to(E4M3FN).view(uint8)`.

Require all codebook bytes to match the canonical formula and require the two
reader outputs to be bit-identical for q1 and q5 before timing. Also compare one
selected call against the exact dense FP8 cache reconstructed from those bytes
with the accepted `atol=0.002` bound.

## Timing and analysis protocol

Run one smoke process per context, then three independent decision processes
per context. Every process compiles both specializations before capture. Graph
capture order is `no-codebook, codebook` in sequences 1 and 3 and reversed in
sequence 2; timing order is always no-codebook/codebook/no-codebook. Each graph
receives at least 100 post-capture warmups and every timed arm at least 2,000
uninstrumented replays. Seed is fixed to `20260729`.

For each sequence compute:

`reader_recovery_us_per_selected_layer =
 (mean(no_codebook_flanks) - codebook_graph_us) / 14`.

Use the fresh process as the resampling unit and 50,000 deterministic bootstrap
draws. H43 D1 passes only if:

1. the historical-long recovery 95% lower bound is at least
   `6.122896 us/layer`, the minimum recovery required by H41 I1;
2. the actual-10K recovery 95% lower bound is nonnegative;
3. every no-codebook flank drifts by at most 2%, outputs are finite and
   bit-identical across specializations, dense-oracle error is at most `0.002`,
   and graph replay allocates zero bytes; and
4. every process is CT13 SM100, P0 at max SM clock, with zero uncorrectable ECC,
   recovery action `None`, successful fabric state, no unrelated GPU process,
   and no Xid in the maintenance window.

An interval crossing either performance boundary is `INDETERMINATE` and does
not permit more samples, split tuning, a new layer set, or endpoint work. No
instrumented duration can enter the decision.

## Memory and implementation decision

For N14/256K, the optional codebook costs exactly `57,344,000 B/rank`. Relative
to the 61-layer FP8 control, persistent saving becomes `852,992,000 B` or
`0.794410706 GiB/rank`, retaining `93.7008%` of the no-codebook gross saving.
After charging the accepted 4,096-row writer workspace (`16,793,600 B/rank`),
the projected net is `836,198,400 B` or `0.778770447 GiB/rank`. No dense/FP8
latent shadow or decoded workspace is allowed.

A D1 pass authorizes one separately committed implementation plan only:

1. reuse Phase 19's backend-gated allocation, page movement, CPU/HiCache copy,
   and raw-FP8 PRMT ABI;
2. add an optional codebook destination to the H41 native SM100 frontend and
   generate the 16 bytes inside its existing launch from the just-rounded BF16
   scale;
3. prove byte-exact writes for current rows, q1/q5 eager and graph replay,
   movement/reuse, allocation accounting, sticky status, and sanitizer; then
4. rerun the complete 61-layer H41 I1 gate at both contexts. The integrated
   historical-long 95% upper bound must be at most `27.6973 us/layer`, and the
   exact upper bound at most `27.0292 us/layer`.

Only a passing integrated implementation can advance to an immutable dual-node
endpoint, quality, DFlash-acceptance, realized-memory, TTFT/TPOT, concurrency,
and production-promotion campaign. If D1 fails, reject the codebook path and do
not revive Phase 17/19 lookup variants. The next direction would require a more
substantial SM100 operand-production or reader redesign.

## Evidence, rollback, and service discipline

Commit and push the frozen plan before harness work. Commit and push the
accepted harness/source manifest before stopping service. Archive commands,
source hashes, raw JSON, logs, graph contracts, byte accounting, GPU/fabric/Xid
state, and a sorted SHA-256 manifest off CT13. Any rejected code is restored
only from the last accepted Git commit and retained on its named remote branch.

Before the isolated run, capture both accepted endpoint containers and health,
stop both together, and prove CT13 GPU0 idle before CUDA initialization. After
the run, restart the exact original N2 DFlash containers rank 1 then rank 0,
require both health checks, rank-0 model information, an independent 64-token
completion, P0/max clock, ECC/recovery/fabric health, and no new Xid.

## Plan self-review

- Round 1 rejected a direct comparison between historical W1 and H42 because
  their frontend/query/buffer contracts differ. The plan now uses one shared
  cache/query/page table in the same process and changes only the optional
  reader input.
- Round 2 found that allocating codebook state only in the test arm could
  confound address/headroom. The single codebook allocation is now resident
  for both graphs, and both graphs share all other cache and workspace storage.
- Round 3 checked prior Phase 17-19 evidence, exact and long splits, no-retuning
  rules, process-level resampling, conservative bounds, byte accounting,
  machine restrictions, rollback, and service restoration. LGTM for plan-only
  commit and collaborative review; no harness, live run, implementation, image,
  production, or upstream work is approved yet.
- Claude Opus 5 collaborative review launched with the required exact model,
  max effort, read-only sandbox, and a fresh reviewer session after the stored
  session was stale. It emitted no response or finding for more than 11
  minutes and was stopped; the audit log is retained at
  `.claude/review-logs/review-20260729-141251-14081.log`. This is an unavailable
  external review, not an external LGTM. Per owner direction, the three-round
  self-review LGTM remains the gate for harness work.
