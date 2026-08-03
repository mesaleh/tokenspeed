# TS-C58-R barrier diagnosis tooling

This directory is a fail-closed experiment harness. It maps the accepted M128
synccheck PC before any kernel repair, compares the dense ID-1 TMEM protocol,
and calibrates aligned, unaligned, and wrong-count barrier behavior. It does not
score speed or authorize production promotion.

Fixed rules:

- use the exact v0.5.16 production image digest in the source identity;
- reserve one whole four-GPU GB200 node and use one declared physical UUID;
- do not mutate clocks, drivers, or node state;
- stage a standalone Git checkout (for example, clone a bundle of the committed
  diagnosis branch) into the pod's source volume; never copy the worktree's
  `.git` indirection file because it names a host-only path;
- generate the execution spec suite before running a command;
- pass the previously sealed accepted M128 and dense oracles plus their
  execution seals as explicit read-only inputs when generating a mapping
  suite;
- wrap every sanitizer phase in machine health snapshots and bind its execution
  seal into the recovery seal;
- export evidence after every phase;
- stop on any required health failure; never reset or reboot to recover;
- do not edit `mla_decode_fp8.py` until PC/SASS/PTX mapping and the dense/litmus
  controls select R1, R2, or R3.

## Tool roles

- `identity.*.json`: exact image, semantic source commit, and locked source/
  probe/litmus hashes. A post-tooling checkout may descend from the semantic
  source commit only when the non-tool MLA tree remains byte-identical.
- `resolve_cuda_ordinal.py`: maps the target physical UUID to visible CUDA
  ordinal without assuming `nvidia-smi` index order.
- `make_execution_specs.py`: emits the complete hashed command contract.
- `run_compute_sanitizer.py` and `seal_sanitizer_result.py`: execute without a
  shell and seal only the predeclared result class plus exact result JSON.
- `capture_gpu_health.py` and `seal_gpu_recovery.py`: type strict, monotonic,
  and best-effort health fields and enclose the exact execution timestamps.
- `probe_tq4_m128_sanitizer.py`: isolates either dense FP8 or native TQ4 M128;
  sanitizer runs must reproduce their same-GPU unsanitized oracle.
- `probe_m128_synccheck_map.py`: runs one predeclared M128 mapping launch,
  catches only the expected sanitizer-induced launch failure, and preserves a
  byte-identical compiler-artifact inventory without assuming an error count.
- `probe_dense_synccheck_control.py`: runs the matched dense control and seals
  the predeclared typed CUTLASS launch-failure branch with byte-identical dense
  compiler artifacts.
- `barrier_litmus.py`: compiles explicit `barrier.sync.aligned` and unaligned
  `barrier.sync` cells; every run must match the prepared extension hash.
- `barrier_split_site_litmus.py`: compiles single-site 288-thread controls and
  participant-identical 128+128+32 split-site cells. Every run binds both the
  prepared extension and the formal disassembly result.
- `capture_split_site_disassembly.py`: fails unless the single-site kernels
  have one ID-8/count-288 barrier PC each and the split kernels have three
  distinct ID-8/count-288 PCs each.
- `make_split_site_specs.py`: predeclares the split-site build, disassembly,
  four unsanitized controls, and four synccheck cells. The two split
  synccheck cells deliberately seal either clean or diagnosed-sync-error so a
  falsified prediction remains attributable evidence.
- `analyze_split_site_results.py`: requires the complete ten-cell execution
  set plus execution-bound recovery seals for every sanitizer cell, then
  selects exactly one repair/diagnostic branch.
- `inspect_barrier_source.py`: records barrier/warp/call-site source geometry.
- `capture_disassembly.py`: for an explicit M128 or dense arm, binds its accepted
  oracle PTX/CUBIN, exact oracle execution seal, `nvdisasm` binary/version,
  arm-specific named-barrier operands, and the resulting SASS.
- `map_barrier_pc.py`: requires a complete synccheck thread map and maps its
  exact, execution-sealed report PC for the explicit M128 or dense arm through
  SASS operands and PTX candidates to one source-level named-barrier role.
- `seal_synccheck_exception.py`: can seal only the fully mapped, reviewed,
  contract-conformant tool-limitation branch with all litmus/recovery evidence.
- `make_provenance.py`: binds source, tools, pod UID/node, runtime image ID,
  physical/CUDA mapping, device scope, and spec-suite hashes.
- `analyze_b1965.py`: admission-only in TS-C58-R. It validates normal synccheck
  or the reviewed exception but deliberately makes no performance decision.

The accepted target order is: source inventory; pre-health and provenance;
unsanitized M128; racecheck; count-independent mapping synccheck with a full
single-PC report and byte-identical compiler artifacts; recovery;
dense unsanitized and synccheck; recovery; prepared litmus build; valid aligned
and unaligned cells; invalid aligned partial; recovery; wrong-count cell last;
recovery; mapping/proof decision. A result is not branch evidence until its
execution and recovery seals both pass.

For the follow-up split-site discriminator, the fixed order is: prepare;
formal disassembly; all four unsanitized cells; aligned and unaligned
single-site synccheck controls with recovery after each; aligned and unaligned
split-site synccheck cells with recovery after each; machine analysis. The
predicted branch is aligned diagnosed plus unaligned clean. Both diagnosed
selects a one-PC or replacement protocol; both clean rules out static-site
splitting alone; unaligned-only diagnosis is a counter-hypothesis. No branch
edits the accepted kernel until the analyzer result is reviewed.
