# H43 D1 deployable codebook-reader A/B

Status: AOT arity repair self-reviewed LGTM; immutable preparation pending
Machine scope: CT13 GPU0 only for CUDA work. CT14 runs no experiment but its
accepted rank is stopped and restored with CT13. CT15, CT16, and SecurityLLMs
are excluded.

## Pinned lineage

- TokenSpeed experiment base: `a9af4c57f682e800c42cc53063e5f013369210a9`.
- Accepted vectorized raw-FP8 codebook primitive: TokenSpeed `94c915c`;
  `a9af4c57` contains its current descendant.
- Accepted H41 native front-end source: SGLang `8d47c4292`; accepted integrated
  H41 harness/result head: `8abe8aa2f`; pinned native package: `029805242`;
  TokenSpeed: `a9af4c57`.
- Accepted optional codebook lifecycle primitive: SGLang `2c1c73fd5`. Its
  allocation, fused write, page movement, CPU/HiCache copy, and clear behavior
  passed correctness/lifecycle tests. The Phase 19 endpoint speed result was
  rejected, not this optional primitive. Any reuse is an explicit cherry-pick
  from `2c1c73fd5`, never reconstruction from rejected working-tree state.
- H42 closeout/consumer head: SGLang `ed7b2c905`.
- Experimental image: CT13 configuration digest
  `sha256:cc6fa1f338da513f9da20ee9310ba8dafbc964e2d9a6b7d528bdef9500c1920c`.
- CT13 GPU0 UUID:
  `GPU-9f90e004-9332-4d9d-fa34-018fb9f07fca`.
- Accepted restore identities are rank-local, not assumed identical: CT13 rank
  0 container `03b4ce2c...` uses image `sha256:cc6fa1f3...`; CT14 rank 1
  container `a64d7605...` uses image `sha256:d4171492...`. The executable
  contract contains the complete IDs and refuses drift.

Every source used in D1 must be in a pushed signed-off commit and in the
committed SHA-256 source manifest before service is stopped. `a9af4c57` is the
parent, not the eventual timed source. Because a commit cannot contain its own
hash, preparation writes the reviewed final D1 TokenSpeed commit to manifested
`H43_SOURCE_COMMIT`, verifies the matching immutable image label, and records
it in this section/evidence after implementation review. The executable
contract pins the parent; its digest and the source-manifest digest bind the
final timed tree without a self-referential hash.

## Corrected hypothesis

H42 reproduced the N14 historical-long integrated miss at `33.099257 us` per
selected layer, but every predeclared scheduling contrast was smaller than
`0.65 us/layer`. Its source audit found that the earlier component comparison
was representation-mismatched: H41 W1 supplied a materialized raw-FP8
`[page, row, 16]` codebook and used the vectorized PRMT reader, while H41 I1 and
H42 supplied `kv_nope_codebook=None` and reconstructed from BF16 scale plus
FP32 centroids.

The optional codebook may recover enough reader time while preserving most of
the no-shadow cache saving. D1 tests that mechanism within one shared cache and
query allocation. It is reader-only evidence and cannot authorize endpoint or
production use.

## Current public-code check

The 2026-07-29 GitHub sweep included vLLM PR `41803` at `f3447918`, SGLang PR
`23135` at `7a2dc8a`, `hackimov/turboquant-kv` at `12c38c7`,
`ansschh/turboquant-kv` at `08247d0`, the llama.cpp TurboQuant CUDA fork at
`67f38aa`, LMCache serialization, and recent TurboQuant code search results.
The vLLM overlay explicitly rejects MLA, disables CUDA graphs, and uses a
Python cache writer. The other CUDA readers are standard-head Triton or scalar
FP32 CUDA paths rather than fused Kimi MLA/SM100 kernels. No public
`kv_nope_codebook` implementation or deployable SM100 q5 Kimi MLA path was
found. Recheck these exact heads before any later upstream claim.

The 2026-07-02 NVIDIA [CuTe DSL JIT-caching
documentation](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html)
states that `cute.compile` deliberately bypasses the implicit/file JIT cache and
returns a process-local executor. NVIDIA's 2026-07-18 [TVM-FFI export
documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#exporting-compiled-module)
instead supports `export_to_c`, linking with the discovered CuTe/TVM runtime
libraries, and fresh-process `cute.runtime.load_module(...,
enable_tvm_ffi=True)`. H43 now uses that supported AOT route.

## PDL ordering correction before measurement

The current codebook specialization loads its 16 bytes before
`raw_k_pipeline.consumer_wait`. With a preceding PDL writer, that load is not
ordered by the TMA-load warps' `griddepcontrol_wait()` and could observe stale
bytes. D1 must not time that optimistic unsafe schedule.

Before any CUDA run, move the entire mutable codebook block—including its page
table read and global loads—after
`raw_k_pipeline.consumer_wait`. The raw-load producer performs the PDL wait;
its pipeline release and the conversion consumer wait establish the transitive
writer-to-codebook-read dependency. The no-codebook scale load is already
after this wait. The remaining pre-wait centroid read is permitted only because
the 16 FP32 centroids are immutable for the lifetime of the server. Add a
source-contract test proving both the page-table and codebook loads remain
after the consumer wait. This correction is part of the D1 source manifest and
is measured in both smoke and decision runs.

This move removes the latency hiding used by H41 W1, so W1's fast reader is an
optimistic prior rather than a deployable prediction. If the direct post-wait
path does not pass, the only still-credible codebook follow-up is a separately
reviewed ordered-prefetch design: issue the codebook load in the raw-load
producer after its PDL wait, stage it with the existing raw pipeline, and let
conversion consume it from shared memory. That H44 design requires its own
SMEM/register/occupancy gate; it is not silently substituted into H43.

D1 still has no writer, so it cannot by itself prove producer/consumer safety.
Any later integrated implementation must add a positive no-host-synchronization
ordering test with an explicit sensitivity control. A native writer stamps
step-distinguishable bytes into rows that the reader actually dereferences
through the randomized page table, then performs additional bounded work after
the stamp to widen the race. The immediately PDL-launched reader output must
encode and reveal the same step. The test must first fail against the pre-move
reader build, proving it can expose the unsafe schedule; the corrected build
must then pass many iterations. A PDL-disabled same-stream run must also pass.
Retain the static ordering test. Sanitizer remains required hygiene but is not
claimed to detect a stale-but-valid cross-grid read.

## Single-source experiment contract

`h43_codebook_ab_contract.json` is the executable source of truth for contexts,
requested/effective splits, the n-selection rule and allowed process counts,
seeds, replay budget, thresholds,
bytes, machine identity, source pins, dedicated cache path, and every timeout.
The harness, analyzer, and maintenance orchestrator load it; they do not
duplicate decision constants. Qualification materializes a generated decision
contract containing only the rule-selected `n`, corresponding timeout/timer
values, pilot digest, and parent-contract digest; it cannot alter any other
constant.

Implementation is not accepted for scheduling until the contract, harness,
analyzer, maintenance orchestrator, installed-package manifest, PDL-order
source test, raw-word probe, resource check, and all local/static tests exist,
agree with this checklist, are collaboratively reviewed, and are pushed.

Both arms share one simultaneously resident 61-layer cache ring:

- 24 dense FP8 layers, selected TQ layers `24-37`, then 23 dense FP8 layers;
- 256,000 physical rows, page size 32, batch 1, q_len 5, and eight heads;
- finite randomized content that may legitimately round to FP8 zero;
- fixed content/query seed `20260729`;
- decision page-table seed `20260729 + sequence`, for the first `n` of
  predeclared sequences `1-14`;
- requested context/split `10,219/64` and `37,932/40`; the kernel's effective
  nonempty splits are respectively `40` and `38`, and both values are emitted;
- one fixed FP8 query shared by all reader calls in a graph;
- one shared preallocated reader workspace, a dense scratch output, and one
  separate output tensor for every selected layer and arm;
- PDL on for dense and selected readers.

The no-codebook arm passes `kv_nope_codebook=None`; the codebook arm passes the
same resident contiguous, 16-byte-aligned codebook. Packed, scale, RoPE, query,
page table, workspace, and cache addresses are shared. Both graphs issue the
same 47 dense plus 14 selected calls in the same order.

The selected persistent row is 322 bytes without a codebook: 256 packed latent
bytes, 2 BF16 scale bytes, and 64 FP8 RoPE bytes. It is 338 stored bytes with a
codebook. The codebook specialization reads 336 of those bytes because its
compiled branch does not read the scale.

## Independent correctness and specialization evidence

Construct each codebook byte as
`Float32(BF16(scale)) * Float32(centroid)`, rounded to E4M3FN and viewed as raw
uint8. Compare all bytes against an independently materialized CPU-FP32
reference, then emit a SHA-256 digest; every process must report the same
digest. CPU FP32 is intentional because Float32 multiplication is the kernel's
canonical contract; Float64 would change boundary rounding.

In every process:

1. require q1 and q5 no-codebook/codebook equality for all 14 selected layers
   in eager execution and after graph replay;
2. compare at least one replay-validated selected layer with the exact dense
   FP8 reconstruction at `atol=0.002`;
3. retain one output per selected layer in the q5 61-layer graphs and require
   all 14 arm outputs to be finite and bit-identical after capture warmup,
   allocation replays, and final timing;
4. record `_get_compiled_mla_kernel.cache_info()` around every first call and
   require exactly one new in-process Python-dispatch miss for each
   `(q_len in {1,5}, codebook in {false,true})` pair—four TQ dispatch-key misses
   per fresh process—plus the separately scoped dense q1/q5 misses; and
5. require graph replay allocation to remain unchanged.

The literal declared operation count is descriptive only. It is not treated as
physical trace evidence.

Compiled AOT artifacts use dedicated CT13 bind-mount namespaces keyed by their
committed source-manifest digests. The accepted pre-move reference and D1 build
have separate source manifests, installed-package trees, artifact namespaces,
and dispatch manifests. The reference resource run is archived against its own
source hash and is exempt only from the D1-package equality predicate; it must
match its own package. Reference work finishes before the D1
qualification-close state is captured.

Before any service window, a source-only AOT build runs while the accepted
endpoint remains online. It initializes one explicitly bound CUDA context for
the SM100 occupancy query, then calls `_get_compiled_mla_kernel` with fake
tensors and a fake stream; it never invokes the returned executor, launches a
reader kernel, or allocates the experiment cache ring. Each TVM-FFI executor is
exported to an object with a unique symbol and linked to a shared library using
`cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)`. The sealed AOT
manifest binds the complete signature-derived dispatch key, label, function,
object and library paths/hashes, installed MLA hash, and source-manifest digest.
It is published only after all artifacts succeed. Temporary partial names are
excluded from evidence; any final artifact in a cold namespace fails closed.

The exact set is eight TQ entries—frozen `tq4_tiles_per_split` values 32/50 by
q1/q5 by codebook false/true—plus dense q1/q5. Their `is_var_seq=True` and
`is_persistent=False` fields are derived from the dense and TQ runtime APIs'
shared production default rather than duplicated as an independent
fixed-sequence choice. The successful namespace contains exactly ten `.o`, ten
`.so`, and one sealed JSON manifest. Its sorted immutable content manifest is
`(relative_path, size, sha256)`; lock, temporary, access-time, or index metadata
remain excluded by the frozen contract.

A second fresh process mounts this namespace read-only, validates every source,
manifest, object, and library digest, installs the H43 loader, and resolves all
ten functions with `cute.runtime.load_module(..., enable_tvm_ffi=True)`. NCU
must show connected and disconnected profiler banners, its explicit `No kernels
were profiled` result, zero metric rows, and no unrelated warning. Silence or
disabled counters fail closed. Any unexpected CUDA kernel launch or endpoint
health change aborts before maintenance. The AOT process exits before service
stop; exclusive empty-compute-app checks apply to qualification and decision
CUDA execution, not source-only build/load.

Every qualification and decision process mounts artifacts read-only. Before
CUDA initialization it revalidates the sealed manifest and installed source;
immediately before each first load it rehashes the selected library. The
signature-compatible patched dispatch remains an in-process LRU, so the frozen
fresh-process miss counts still prove all required q1/q5/dense/codebook entries
were selected, but no JIT compilation is allowed. Record the immutable artifact
digest after every process and at close. The two warm qualification preflights
must create no artifact and yield identical manifests. Stop and restart the
unchanged experiment container to prove the read-only bind mount survives. A
missing/new/changed artifact or absent dispatch key yields `NO_DECISION` before
performance analysis. The source manifest includes the AOT builder/loader and
installed packages; generated binaries are evidence rather than source.

## Timing design and statistical gate

Run smoke only in qualification. Before any codebook-arm timing, each of the
four fresh preflight processes runs 20 decision-replay-equivalent A/A pairs of
the no-codebook graph, alternating pseudo-AB/BA order. This reveals no codebook
contrast. For each context, take the larger of (a) the sample SD of its 40 A/A
recoveries `(A1_us-A2_us)/14`, divided by `sqrt(20)` to put it on the
20-pair process-mean scale, and (b) the sample SD of its two process means;
define `sigma_pilot` as the worse of the two context values.
The internal-pilot rule may only increase sample size: use `n=10` fresh
decision processes per context when `sigma_pilot <= 1.0 us/layer`, and `n=14`
when `1.0 < sigma_pilot <= 1.25`. A larger value yields pre-decision
`NO_DECISION` and a newly reviewed sample-size plan; it cannot reject D1.
At one-sided alpha 0.05, n=10 provides about 93% power for an effect 1 us/layer
beyond threshold at sigma=1, and n=14 provides about 91% at sigma=1.25. The
qualification archive seals `sigma_pilot`, selected `n`, and the generated
decision-contract digest before any codebook contrast. Stage 2 later reports
the realized paired-recovery SD, but it cannot change `n` or validity post hoc.

Capture order is balanced: no-codebook first for odd sequences and codebook
first for even sequences. Each graph receives exactly 100 eager pre-capture
invocations, exactly 100 post-capture graph warmups, and 100 excluded
allocation-check replays before timing:

- time 20 paired samples per arm, each sample containing 100 graph replays;
- alternate pair order AB, BA, AB, BA within every process, and reverse the
  first pair for even sequences;
- compute each paired recovery as
  `(no_codebook_ring_us - codebook_ring_us) / 14`;
- use the equally weighted mean of the retained AB mean and retained BA mean
  as the process resampling unit, so telemetry exclusions cannot reintroduce
  order bias;
- bracket the paired block with excluded 100-replay no-codebook sentinels and
  require absolute sentinel drift at most 0.5% of ring time when both
  sentinels have valid transient telemetry. If either sentinel has a transient
  P-state/clock/throttle/power reason, abstain from that process's drift gate;
  identity, ECC, recovery, or fabric failures remain hard failures;
- query excluded P-state, clock, throttle, power, and temperature telemetry
  before and after every pair. Exclude a pair if either sample is not P0/1965
  MHz, reports hardware-slowdown or software-thermal-slowdown active, has a
  non-finite/non-positive power draw or draw above the reported 1200 W limit.
  Temperature is recorded, including the steady-state maximum, but is not an
  invalidating threshold unless the hardware/software thermal-slowdown flag is
  active. Retain at least eight AB and eight BA pairs in every process and all
  `n` processes per context; otherwise the campaign is `NO_DECISION`; and
- use 50,000 deterministic process-level bootstrap draws and the one-sided 5th
  percentile as the 95% lower confidence bound across the frozen `n` processes.

No block-middle arm supplies the primary contrast. Instrumented or sanitizer
durations never enter the decision.

The A/A preflights also measure steady-state temperature and telemetry pair-
exclusion rate without exposing an arm contrast. If `k` of the 80 pilot pairs
are contaminated, use the predeclared Jeffreys-smoothed projection
`p = (k + 0.5) / 81` and
`F = sum(j=0..2, C(10,j)*p^j*(1-p)^(10-j))`. The projected probability that the
retained-pair rule invalidates at least one of the `2*n` processes is
`1 - F^(4*n)`; seal and report it, and require it to be at most 5% before the
decision window. This is an explicit independent-pair projection, not a claim
that telemetry events are truly independent. A miss is pre-decision
`NO_DECISION`, not a D1 rejection.

Validity is resolved before performance statistics. Stage 1 is a fully
automated pass over the contract's frozen predicates; it necessarily reads raw
ring durations and arm/order labels for sentinel and retained-order checks, so
it is not claimed to provide cryptographic or operator blinding. No Stage-1
predicate references an inter-arm duration difference. It writes and archives
the validity verdict plus a digest of the immutable raw inputs before Stage 2
may be invoked. Stage 2 refuses unless that exact digest is `VALID`, then
computes the paired recoveries, process means, and bootstrap. Power is not a
tuning covariate: only the explicit finite `(0, 1200] W` device-limit check
above is invalidating.

## Threshold provenance

Define the authoritative integrated quantity once:

`integrated_overhead_per_selected_layer =
 (mixed_candidate_61_layer_graph_us - mean(dense_61_layer_control_flanks_us))
 / 14`.

H41 W1 derived each full allowance as
`R = 0.8 * 0.05 * mean_TPOT_ms * accepted_tokens_per_forward * 1000 / 14`.
The 5% term is the owner-approved production mean-TPOT regression gate and
`0.8` reserves 20% of that endpoint budget for work/noise outside this graph.
H43 uses the lower endpoint of each W1 allowance: `27.029200` at actual 10K and
`27.697300` at historical long.

H43 thresholds come from SGLang result commit `8abe8aa2f`,
`benchmark/bench_turboquant_mla/H41_I1_INTEGRATED_EXPERIMENT.md`, and the
manifested evidence root
`phase28/h41-i1-20260729` recorded in vault note
`H41 I1 - Integrated Four-Family Graph Gate - 2026-07-29.md`, not H42's
`33.099257` replication. That harness formed every observation exactly as
`(candidate_graph_us - control_graph_us) / 14`: its control ran 61 complete
dense triplets (fused RoPE/FP8 query+KV conversion, dense FP8 writer/scatter,
then dense attention), while its candidate ran complete selected TQ triplets.

- Historical-long H41 I1 observations were `33.820196`, `33.638324`, and
  `33.804324 us/selected layer`. The conservative W1 allowance is
  `27.697300`; therefore the worst observed integrated path needs exactly
  `33.820196 - 27.697300 = 6.122896 us/layer` reader recovery.
- Actual-10K H41 I1 observations were `12.815642`, `13.262944`, and
  `12.370810 us/selected layer`. The allowance is `27.029200`, leaving a
  reader-only headroom of
  `27.029200 - 13.262944 = 13.766256 us/layer`. D1 therefore requires recovery
  no worse than `-13.366256 us/layer` after reserving the writer allowance
  below; the later integrated gate remains authoritative.

Phase 19 SGLang commit `2c1c73fd5`, file
`test/registered/attention/bench_tq4_codebook_write.py`, captures one layer's
complete three-launch writer invocation at q1, q5, and q128. Its A/B/A medians
showed no measurable codebook cost and all differences were inside the observed
`0.3-0.4 us` per-layer-graph timing granularity. H43 therefore uses
`w = 0.400000 us/selected layer` as a deliberately conservative placeholder
for the future H41 native codebook write. The long D1 screen is
`6.122896 + w = 6.522896 us/layer`. Before the integrated item-4 gate, the
native writer is measured directly and replaces `w`; neither `0.4` nor the D1
screen becomes an integrated-performance claim.

The old three-observation H41 ranges are treated as observed ranges, not valid
95% bootstrap intervals.

D1 advances only if the campaign is valid and both one-sided 95% lower bounds
meet their thresholds:

1. historical long: at least `+6.522896 us/layer`;
2. actual 10K: at least `-13.366256 us/layer`;
3. all correctness, specialization, allocation, drift, sanitizer, source,
   machine, and health gates pass.

Only a fully valid campaign whose lower bound misses a threshold is a `REJECT`;
that closes the direct post-wait D1 implementation. Any failure of a numeric or
enumerated Stage-1 predicate frozen in the contract—including correctness,
source, compiled artifacts, machine, health, sanitizer, sentinel drift,
retained-pair count, telemetry, or collection—is `NO_DECISION`, never evidence
against D1. No unstated covariate can invalidate a result after Stage 1 seals.
No extra samples, new seed, split tuning, layer-set tuning, or automatic rerun
is permitted after seeing either outcome. A future owner-approved attempt after
`NO_DECISION` requires a new reviewed plan and reports the invalid raw attempt.

## Machine, source, and health binding

Every result must contain and the analyzer must require:

- physical host environment marker `ct13`, container hostname `ct13-h43`, GPU
  index 0, the pinned GPU UUID, and device name `NVIDIA GB200`;
- an empty `nvidia-smi --query-compute-apps` result before CUDA initialization;
- the same committed source-manifest SHA-256 and codebook SHA-256;
- the SHA-256 of the actually imported installed-package
  `mla_decode_fp8.py`, matching the manifest;
- P0 and SM clock equal to 1965 MHz in samples bracketing the timed block;
- volatile uncorrectable ECC unchanged at zero, aggregate uncorrectable ECC
  unchanged from the pre-maintenance baseline, recovery action `None`, and
  successful fabric state/status; and
- no Xid/SXid in the complete maintenance window.

The parser and exact `nvidia-smi` field formatting were dry-run while the
accepted endpoint remained online on 2026-07-29. CT13 GPU0 reported P0,
1965/1965 MHz, aggregate ECC 0, recovery `None`, `state  Completed`, and
`status Success`. The live service prevented an idle-process check; that check
occurs after both ranks stop and before any CUDA process starts.

## Sanitizer precondition

Before decision timing, run excluded q1/q5/all-14 correctness plus one q5
61-layer graph replay under `compute-sanitizer` memcheck, racecheck, and
initcheck with nonzero error exit codes. Each tool has a frozen 180-second
timeout. A tool error, timeout, or unsupported invocation blocks decision
timing and triggers service restoration. This validates the H43 codebook + FP8
RoPE + PDL + graph combination; it does not replace the later adjacent-writer
positive ordering test. Sanitizer is memory-access hygiene only.

Requalify the reordered primitive before timing: run the Phase 18 exhaustive
raw-word PRMT probe, q1/q5 eager/graph checks, and an excluded Nsight Compute
LaunchStats/Occupancy capture on both the accepted pre-move build and the D1
post-wait build. Each of the 61 reader calls emits a split-KV kernel followed
by a reduction kernel, so require the exact 122-launch alternating sequence and
select calls 24-37 rather than raw launch IDs. Compare the 14 selected split-KV
launches and 14 corresponding reduction launches as separate kernel classes.
Require launch success and no D1 regression in registers per thread, static or
dynamic SMEM, achieved occupancy, or theoretical occupancy for either class;
both builds and classes must report zero local-memory spill loads and stores
and at least one resident block per SM. Archive the exact comparative Nsight
metrics in the source-manifest evidence. Nsight durations are excluded from
every timing gate.

## Memory contract

At N14 and 256K rows:

- no-codebook cache: `8,084,480,000 B/rank`;
- codebook allocation: `57,344,000 B/rank`;
- codebook cache: `8,141,824,000 B/rank`;
- 61-layer FP8 control: `8,994,816,000 B/rank`;
- persistent GPU saving: `852,992,000 B/rank = 0.794410706 GiB/rank`, retaining
  `93.7008%` of the no-codebook gross saving;
- after the accepted 4,096-row writer workspace, projected net saving:
  `836,198,400 B/rank = 0.778770447 GiB/rank`.

No decoded or dense FP8 latent shadow is allowed. If the codebook is mirrored
in host offload, HiCache, or transferred between nodes, each full N14/256K copy
adds the same `57,344,000 B/rank`; later realized-memory and DFlash campaigns
must measure those copies rather than claiming GPU accounting as end to end.

## Campaign count, runtime qualification, maintenance, and recovery

At most three service windows are pre-registered: one qualification attempt,
at most one infrastructure-only qualification retry, then one decision window.
The qualification retry is allowed only for an independently identified
transient orchestration/infrastructure failure before any arm contrast exists,
with identical committed source, contract, cache-namespace rule, and procedure.
A source, correctness, sanitizer, resource, cache-integrity, or machine-health
failure cannot be retried under H43; it requires a new reviewed plan. The
decision window is always single-shot.
Immediately before arming a decision window, CT13 atomically creates a
root-owned `O_EXCL` consumption record under the immutable preparation root,
bound to the contract and decision-contract digests. Any later campaign name
using that preparation state fails while the accepted endpoint is still live;
operator naming cannot create an extra decision sample.

1. **Qualification window**: source/idle checks, dedicated-cache verification,
   q1/q5 smoke, raw-word and comparative resource gates, all three sanitizer
   tools, a stop/start cache-persistence proof, and two replay-count-equivalent
   fresh-process runtime preflights per context (four total). Preflights load
   the same q1/q5/dense/codebook specializations and execute the decision replay
   count but emit no codebook-arm durations, recoveries, or inferential
   contrasts. Only the predeclared A/A internal pilot may affect sample size;
   neither absolute ring variance nor any post-codebook statistic can alter
   eligibility or `n`. Record every process wall time and compiled-artifact
   manifest. The projected decision runtime is
   `1.2 * (2*n) * (max(second-preflight wall time by context) + 2 seconds)`;
   the frozen two seconds conservatively cover each fresh process launch. It
   must fit the selected decision experiment budget: 20 minutes
   for n=10 or 28 minutes for n=14. Otherwise no decision window opens.
2. **Decision window**: the internally selected and sealed `n` processes per
   context exactly once. No
   smoke, profiling, sanitizer, source change, cache clearing, compilation, or
   tuning occurs in this window.

Cold AOT compilation, export, linking, and fresh-process load proof are excluded
from downtime because their source-only work and per-key cost record finish
before maintenance. The qualification experiment
budget is explicitly 23 minutes: 2 minutes for source/staging/idle and cache
persistence, 1 for smoke, 1 for the raw-word probe, 3 for comparative Nsight,
9 for three 180-second sanitizers, 6 for four preflights, and 1 for
archive/source/cache closeout. The orchestrator hard-stops
new qualification work at minute 24 and invokes restoration. An incomplete
qualification authorizes no decision. An incomplete or invalid decision is
`NO_DECISION` and is not automatically repeated. Any owner-approved future
decision attempt requires a new reviewed plan and reports every prior raw
attempt; windows cannot be silently sampled until pass.

Before stopping service, capture both container inspections, health, image
identities, journal/Xid baseline, and accepted endpoint responses. Prove while
the service is still live that `docker top -eo pid,args` contains the frozen
rank-specific `--node-rank 0/1` process marker and that logs from the current
container start contain the matching `node_rank=0/1` server marker; restore
uses those same predicates. Re-query every GPU on CT13 and CT14 at the final
pre-stop snapshot and require the same P0/1965 MHz, throttle, power-limit,
ECC/recovery/fabric, name, and UUID-baseline predicates used after restoration;
pre-existing machine drift therefore blocks the window. The source
manifest covers every imported installed `tokenspeed_mla/*.py` file, every
`/work` harness/contract/orchestrator file, the PDL source test, and the image
digest—not merely the package path.
Candidate/reference archives are uploaded only into a root-owned mode-0700
directory under the preparation root, verified root-owned mode 0600 before
privileged extraction, and deleted afterward. Build logs stream directly into
the root-owned role directory; no predictable `/tmp` archive or build-log path
feeds a privileged operation. Because Dockerfile `FROM` cannot resolve a bare
local `sha256:` image ID, preparation creates a campaign-specific local tag
from the frozen accepted rank-0 image ID and re-verifies that tag resolves to
the exact ID immediately before each candidate/reference build.

Install and verify idempotent remote fail-safe units at experiment-budget plus
five minutes: minute 25 for qualification or an n=10 decision and minute 33 for
an n=14 decision. Install an independent alert-only unit ten minutes later
(minute 35 or 43) that never starts or stops a container. Keep both restore
units armed throughout explicit restoration and full validation; cancel them
only after every restore gate passes. Both
explicit and fail-safe paths use the same rank-local `flock`, inspect the exact
container ID/state, and start only the stopped accepted container, so a
concurrent invocation is harmless.
The fail-safe repeats every 60 seconds after its first deadline until
validation disarms it. A separate terminal timer stops and resets the
restore/alert timers at the contracted 44.75/40.75/48.75-minute bound, emits a
final manual-escalation alert, and prevents an unbounded failed-unit loop. CT14
publishes a random 256-bit nonce path on a
temporary host HTTP listener only after the exact rank-1 container, process,
and new-start log marker pass; CT13 refuses to start rank 0 without the exact
nonce and rank-1 container ID. A pre-maintenance connectivity test opens and
removes the same listener while the accepted endpoint remains online. The
HTTP server's document root is the nonce-specific mode-0700 directory, so the
parent cannot expose campaign nonces through a directory listing. The
rank-0 restore script also stops only the exact manifested H43 experiment
container before starting the accepted rank; a name/ID mismatch fails closed.
After the candidate stop, the rank-0 restore script itself polls the contracted
GPU index for three consecutive empty compute-process samples within the
60-second GPU-idle timeout, covering both explicit and fail-safe restoration.
If the exact accepted rank is already running, it skips that self-defeating
idle wait. If an unknown process remains after the bound, it logs the hazard
but still attempts to start the exact accepted rank rather than vetoing all
future recovery.
CT14 starts original rank 1. CT13 starts
original rank 0 only after CT14's exact container and expected server process
remain running and its log emits the accepted `server_args` marker with
`node_rank=1`, all within 120 seconds. Marker publication then receives a fresh
15-second deadline; the dummy HTTP health server appears
only after distributed initialization and therefore cannot gate rank-0 startup.
After both ranks start, allow at most 600 seconds for full HTTP health. Every
poll loop and every `ssh`, `docker`, `curl`, completion, and health command has
a contract timeout. On timeout, stop new work, retain the fail-safe, invoke
explicit restore if safe, preserve logs, and escalate.
This uses real predicates, never a fixed delay. The local orchestrator has a
24-minute qualification timeout or the selected 20/28-minute decision timeout plus an
unconditional EXIT trap that invokes the same restore scripts.

The maintenance alert target is minute 35 for qualification/n=10 and minute 43
for n=14, without mutating service state. Qualification has a bounded
44.75-minute terminal result; n=10 decision, 40.75; n=14 decision, 48.75. Each
combines the selected 24/20/28-minute experiment timeout with independently
bounded rank-1 readiness/marker publication, rank-0 GPU-idle/marker/HTTP-health
restoration, three minutes of final validation, and one minute of terminal
margin. All command and poll timeouts guarantee a terminal
restore result or escalation by the applicable bound; the alert is not
mislabeled as a hard cap. Any
staging, source-verification, idle, smoke, sanitizer,
or decision failure skips all remaining work and restores service before result
analysis. During explicit restoration require both health checks, rank-0 model
information, an independent exact 64-token completion, P0/max clock,
ECC/recovery/fabric health, and no new Xid. Only after these pass cancel the
restore fail-safe units, then cancel the alert. If validation fails,
leave the fail-safe armed only until the terminal cleanup bound, preserve both
original containers and logs, make no replacement container, and stop for
owner escalation.

## Advancement after D1 only

A D1 pass authorizes a separately reviewed implementation plan, not endpoint
or production work:

1. explicitly cherry-pick the accepted optional lifecycle primitive from
   SGLang `2c1c73fd5` onto the accepted H41 integration lineage;
2. add an optional codebook destination to the accepted H41 native SM100
   front-end and generate 16 bytes from its just-rounded BF16 scale inside the
   existing launch;
3. retain the H43 post-wait reader ordering and prove byte-exact writes,
   current-row movement/reuse/copy/clear behavior, q1/q5 eager and graphs,
   allocation, sticky status, sanitizer hygiene, and the repeated stamped
   PDL/no-PDL positive ordering test without host synchronization; exercise
   subnormal and saturation-boundary scales against device
   `cvt.rn.satfinite.e4m3x2.f32` semantics, because D1's `0.05-0.20` scale
   distribution validates only normal-range reader bytes; then
4. rerun the complete H41 I1 gate as an isolated idle-GPU harness, not inside
   the model server. It simultaneously allocates the `8,994,816,000 B` dense
   FP8 control ring and `8,141,824,000 B` integrated mixed ring—
   `17,136,640,000 B` total before workspaces—and must prove co-residency and
   unchanged replay allocation before timing. Within each of ten fresh
   processes per context, reproduce the exact `8abe8aa2f` construct: the dense
   control includes the accepted fused RoPE/FP8 query+KV conversion, dense FP8
   writer/scatter, and dense attention at all 61 layers; the mixed arm includes
   the corresponding complete dense triplets plus selected TQ codebook writer
   and reader triplets. Pair those full graphs in balanced AB/BA order; each
   paired value is
   `(mixed_us - dense_us) / 14`. Excluded dense-control 100-replay sentinels
   bracket the paired block. Use the same retained-pair, telemetry, drift, and
   deterministic 50,000-draw process-bootstrap rules as H43. The one-sided 95%
   *upper* confidence bound for integrated overhead must be at most
   `27.6973 us/layer` at historical long and `27.0292 us/layer` at actual 10K.

Only that integrated pass can authorize an immutable dual-node endpoint,
quality, DFlash acceptance, realized memory, TTFT/TPOT, concurrency, and the
production-promotion campaign. A direct post-wait D1 failure closes that
implementation, not the predeclared ordered-prefetch H44 option. If resource
analysis rejects ordered prefetch too, the next direction requires a larger
SM100 operand-production or reader redesign.

H44 is only a named follow-up, not part of H43. It must have its own reviewed
plan, source commit, qualification/decision windows, source/cache/machine
binding, correctness and resource controls, and thresholds derived from the
same `R` formula. Its report must disclose the complete H43 outcome and raw
artifact digest. No H44 source or measurement may enter an H43 window.

The `R` allowance applies production mean TPOT and accepted tokens per target
forward to this batch-1 q5 graph screen. That assumes this isolated shape is
representative of the production verifier path; H43 cannot prove it. The later
integrated endpoint campaign explicitly tests the assumption at production
request shape and concurrency before any promotion claim.

## Review history

- Self-review rounds 1-3 corrected representation matching, shared residency,
  allocation symmetry, seed, byte accounting, and service recovery. They were
  plan-only reviews.
- The first Claude Opus 5 attempt during the collaborator outage produced no
  verdict and is retained in
  `.claude/review-logs/review-20260729-141251-14081.log`.
- Claude Opus 5 round 1 after recovery produced 15 findings in
  `.claude/review-logs/review-20260729-143351-16332.log`. This revision accepts
  the substantive corrections: ten process units, paired AB/BA timing, a
  0.5% sentinel, all-layer replay outputs, PDL-safe load ordering, explicit
  component pins and threshold arithmetic, derived exact-context margin,
  independent CPU codebook reference, compile-cache evidence, machine/source
  binding, varied page permutations, sanitizer gates, a hard maintenance cap,
  unconditional/fail-safe restore, and corrected split/byte/content language.
- Claude Opus 5 round 2 confirmed the statistical, threshold, PDL, output,
  provenance, permutation, sanitizer, memory, split, and recovery corrections.
  This revision additionally reserves the measured `0.4 us/layer` writer
  allowance, defines integrated overhead and the W1 budget formula, limits the
  campaign to one qualification plus one decision window, adds runtime/JIT
  preflight, makes restore idempotent/readiness-based, hashes the installed
  package, requalifies raw-word/resources, scopes compile misses, records
  per-pair clocks, bounds CPU-reference claims, and adds a positive stamped PDL
  ordering test for later integration.
- Claude Opus 5 round 3 confirmed the corrected arithmetic and core design but
  found remaining governance and reproducibility gaps. This revision keeps the
  fail-safe armed through validation; separates `NO_DECISION` from statistical
  `REJECT`; adds an ordering-test sensitivity control; source-keys and
  persistence-tests the compiled cache; caps qualification with a summed
  budget and one infrastructure-only retry; pins the writer-reserve units;
  compares pre/post resource metrics; validates permutation variance; bounds
  every poll; defines the integrated upper-confidence gate; isolates H44; and
  states the batch/concurrency assumption.
- Claude Opus 5 round 4 found eight interactions introduced by those fixes.
  This revision removes the invalid absolute-ring variance screen; prebuilds
  and times all ten compile keys outside downtime; hashes immutable compiled
  artifacts rather than ephemeral cache metadata; separates reference and D1
  source/cache namespaces; seals numeric validity before performance analysis;
  reconciles the 35-minute alert with a bounded 40-minute worst case; permits
  at most two telemetry-contaminated pair exclusions per process; and defines
  executable dense-vs-integrated paired/sentinel arms with proven co-residency.
- Claude Opus 5 round 5, the skill's maximum, verified the source-only compile
  path, ten-key count, no hidden LSE allocation, memory sum, budget, and restore
  arithmetic, then found five final issues. This revision adds a blinded-to-
  codebook A/A internal pilot that can only raise n from 10 to 14; derives and
  caps projected telemetry invalidation; relies on real thermal-throttle flags
  rather than an arbitrary 55 C cutoff; cites and exactly reproduces H41 I1's
  full dense-writer control; requires the first qualification process to be a
  compiled-artifact hit; and describes Stage 1 as automated predicate sealing,
  not impossible label/operator blinding.
- External review stopped at the mandatory five-iteration limit. Final
  acceptance requires a fresh line-by-line self-review LGTM of these fixes.
- Final self-review traced the A/A internal-pilot power rule, telemetry-
  exclusion probability, treatment-independent validity predicates, dynamic
  process/timer contract, ten compile keys, reference/D1 namespaces, H41 I1
  graph equivalence, threshold arithmetic, and restore bounds. The fixes are
  mutually consistent: `LGTM` for the plan. This does not approve the current
  quarantined draft code or any live run.
- The first implementation review found eight substantive issues. This revision
  corrects the two-kernel-per-call NCU model; puts pilot sigma on the
  process-mean scale; equally weights retained AB/BA orders; makes transient
  sentinel telemetry abstain rather than invalidate; proves live restore
  predicates; uses the supported `docker top -eo pid,args` form; waits for an
  idle CT13 GPU in the shared restore script; gives marker publication a fresh
  deadline; and makes the source-only NCU proof require positive attach,
  detach, and explicit no-kernel evidence. These implementation-driven plan
  corrections require renewed code/plan review before any live window.
- The fresh implementation review then found the runtime/prebuild dispatch
  mismatch, a privileged predictable-`/tmp` staging race, nonce-parent listing,
  delayed cache-miss detection, an imprecise sigma failure label, and a missing
  programmatic single-shot interlock. This revision derives variable-sequence
  dispatch flags from both runtime API defaults, uses root-owned preparation
  staging/logs, scopes the marker server to the nonce directory, fails inside
  the first process on cache drift, preserves the original pilot failure, and
  atomically consumes the decision contract before a maintenance window can be
  armed. Its restore-budget finding is also resolved by contract-derived rank
  timeouts and enlarged terminal bounds.
- The next Opus pass found remote-shell argument re-splitting, a recovery-veto
  interaction in the GPU-idle proof, unbounded periodic restore timers, and a
  too-late all-GPU clock check. This revision shell-quotes every remote-script
  argument; skips the idle proof for an already running accepted rank and logs
  rather than vetoes after a bounded busy wait; adds terminal cleanup units;
  and applies the full final GPU predicate to every CT13/CT14 GPU immediately
  before service stop.
- The following stable-tree pass found that the pre-stop comparison executed
  before its baseline capture. The baseline health/identity pass now occurs at
  the start of `snapshot_and_verify`, followed by the live container snapshot
  and second telemetry comparison; a regression test freezes that ordering.
- The first online-only preparation safely failed before CUDA because BuildKit
  treated a bare `sha256:` `FROM` value as a registry repository. Preparation
  now creates and re-verifies a campaign-local tag for the exact accepted image
  ID before both builds; the endpoint remained healthy and no service window
  or source-only prebuild was consumed.
- The second online-only preparation built the candidate image, then safely
  failed its installed-source provenance check before compilation. The NGC
  entrypoint writes its CUDA license banner to stdout, contaminating the
  captured source manifest, installed-module digest, and eventual prebuild
  JSON. Direct `--entrypoint python3` and `--entrypoint ncu` probes reproduced
  clean machine-readable output. Preparation now bypasses the image entrypoint
  for every captured Python/NCU command and immediately validates the complete
  manifest, digest shape, and phase-tagged JSON. The rejected campaign is
  retained as evidence; it consumed no service window or CUDA kernel launch,
  and the accepted endpoint remained healthy.
- Preparation-fix self-review pass 1 expanded the repair from the observed
  digest failure to every stdout-captured machine-readable artifact and added
  fail-closed format checks. Pass 2 exercised the corrected manifest, digest,
  and NCU entrypoint forms against the immutable candidate image while the
  accepted endpoint stayed healthy. Pass 3 traced verification ordering so no
  contaminated manifest can select a cache and no malformed prebuild result can
  enter identity evidence. The focused 24-test suite, Bash parse, diff checks,
  targeted format hooks, and repository-wide non-format hooks pass: `LGTM`.
- The third online-only preparation passed source provenance and reached the
  first compile key, then safely failed the SM100 occupancy query with
  `CUDA_ERROR_INVALID_CONTEXT`. Unlike the allocating microbench path, the
  source-only compiler path had never initialized PyTorch's lazy CUDA context.
  A real-image probe with exactly one visible GPU showed that explicit device-0
  binding plus `torch.cuda.init()` makes the query return 152 clusters without
  a kernel launch. Prebuild now freezes that one-device invariant and initializes
  the context before any compile key. The accepted endpoint remained healthy
  and no maintenance window was consumed.
- The initialized diagnostic compiled all ten exact keys successfully, but its
  declared cache namespace remained empty. Installed NVIDIA source and the
  current official JIT-caching guide both confirm that `cute.compile` forces
  `no_cache=True`; prior H43 persistence language was therefore invalid. This
  rules out a file-JIT prebuild and explains why the warm process could not be
  a hit. A one-key real-image probe then used the matching official TVM-FFI API
  to export `dense_q1.o`, link `dense_q1.so` against the discovered CuTe/TVM
  libraries, and load its callable symbol in a fresh process without a kernel
  invocation. The endpoint remained healthy. H43 now generalizes that proven
  AOT mechanism to all ten exact keys rather than weakening the no-JIT gate.
- AOT self-review pass 1 traced source-to-dispatch-to-library identity and made
  published artifacts read-only for warm and maintenance consumers, with a
  final selected-library rehash. Pass 2 froze the manifest/entry schemas, exact
  SHA forms, path containment, unique symbols, and fail-closed missing-key
  behavior. Pass 3 proved one-key cross-process load with one LRU miss and a
  positive NCU attach/detach/no-kernel result. Pass 4 ran all ten cold
  compile/export/link operations, obtained exactly 21 immutable artifacts,
  loaded all ten from a read-only mount in a fresh process, matched cold/warm
  digests, and again proved zero launches; endpoint health stayed green. The
  collaborative-review session is already at its mandatory five-round maximum,
  so no sixth invocation is allowed. The focused 29-test suite, Python/Bash
  parse checks, targeted format hooks, and repository-wide non-format hooks
  pass: `LGTM` for the AOT preparation implementation. This still does not
  authorize a service window without a clean signed commit and immutable
  preparation state.
- The first qualification attempt stopped safely during online setup, before
  fail-safe arming or any service stop, because CT13 did not already contain
  `/usr/local/libexec`; atomic helper installation therefore could not create
  its temporary file. Both accepted ranks and HTTP health remained unchanged.
  Setup now creates the conventional root-owned `0755` helper directory on
  both nodes before installing the restore, alert, or terminal scripts, and a
  regression check freezes that ordering. The rejected campaign and local
  evidence manifest remain preserved; a new source-bound preparation is
  required before another qualification attempt.
- Qualification-setup repair self-review pass 1 traced the exact execution
  order: immutable preparation and endpoint checks precede root creation;
  helper-directory creation and atomic script installation precede snapshot,
  fail-safe arming, and service stop. A failure on either node therefore still
  leaves both accepted ranks untouched. Pass 2 verified idempotent root-owned
  `0755` directory creation on both nodes, deterministic preservation of the
  rejected attempt, and a clean import-only AOT loader without an inapplicable
  executable shebang. Focused tests and targeted format hooks pass: `LGTM`.
- The second qualification campaign reached the guarded service window and
  passed source-manifest and PDL-source-order gates, then failed the first dense
  correctness call before any timing. The exported TVM-FFI symbol requires all
  18 parameters, including the three TQ-only parameters compiled as `None`,
  while TokenSpeed's dense runtime deliberately invokes the CuTe JIT executor's
  compact 15-argument form. A raw exported symbol does not retain that executor
  adaptation, so it rejected the call. The runner restored both exact accepted
  containers, passed endpoint completion and GPU health validation, found no
  Xid, and disarmed all timers. This is a deterministic source/correctness
  failure and `NO_DECISION`, not an infrastructure retry.
- The AOT loader now restores the frozen executor ABI at its narrow boundary:
  exactly 15 runtime arguments receive three trailing `None` values; exactly 18
  pass unchanged; every other count fails closed. A regression test covers all
  three cases. No kernel, dispatch key, manifest schema, or production module is
  changed by this adapter; a new signed source identity and immutable
  preparation are required before qualification.
- AOT-arity repair self-review pass 1 matched the failure's named 18-parameter
  TVM-FFI signature against the exact 15-element dense tuple and 15+3 TQ call
  sites. Pass 2 traced loader caching, module lifetime, compact and expanded
  paths, malformed-count rejection, and source-manifest rebinding; the adapter
  is outside measured kernel execution and adds no timing-path work after
  dispatch. An AST regression freezes both production call shapes. Focused
  tests, parse checks, targeted format hooks, and repository-wide non-format
  hooks pass: `LGTM`.
