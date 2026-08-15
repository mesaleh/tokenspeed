# S5-D0 page-32 native-E2M1 reader — rejected timing evidence

Decision: `S5_D0_REJECT_TWO_CTA_OWNER`.

This branch preserves a correct page-32, no-shadow native-E2M1 reader and its
decision harness. It is research evidence, not a production integration
candidate. The accepted writer, packed pool, and N8 representation live in
their separate accepted SGLang/TokenSpeed branches.

## What passed

- Public W5-compatible packed-cache ABI and normal BF16 output/FP32 LSE.
- Exact q1/q5 output and LSE at cache lengths
  `0,1,31,32,33,127,128,129,511,513,639,640`, including a partially live
  final drain tile and partial final page.
- Arbitrary physical pages, including page IDs above 19, and poisoned padding.
- Eager execution, graph100 replay with zero allocation, public host guards,
  and device-side invalid-page protection.
- Two-node correctness reproduction of the complete expanded matrix. Both
  ranks report maximum output error `0.00755835` and graph100 zero allocation.
- Compute Sanitizer memcheck, initcheck, racecheck, and synccheck with zero
  device findings.
- Generated cubin SHA-256
  `670f083d53254f17b06801ffb7606def95552873513e3f430a86a5090f7ee291`;
  438,128 bytes; 168 registers; 24-byte stack; 1,024-byte static shared
  allocation; no reported local allocation. SASS contains 130 MMA and 115 TMA
  instructions. The saturation-only multi-request specialization adds four
  local loads over the cluster-one version but does not change resource
  dimensions.

## Decision-grade saturation rehearsal

Each arm owns 100 address-disjoint requests. D0 launches them as independent
two-CTA clusters in one graph replay; the dense arm runs the installed
production TokenSpeed FP8 MLA backend at the same batch. Six Williams-balanced
windows follow six warmup windows. Timing is microseconds per request.

| shape/node | RN mean | C0 mean | strict-native mean | dense mean | RN/C0 geometric ratio (CI95) | RN/dense geometric ratio (CI95) |
|---|---:|---:|---:|---:|---:|---:|
| q1 / r0 initial | 1.48187 | 3.07744 | 1.29989 | 0.18587 | 0.48145 (upper 0.49835) | 7.80923 (7.36455..8.28077) |
| q5 / r1 initial | 1.84160 | 3.62875 | 1.55248 | 0.19952 | 0.50564 (upper 0.59692) | 7.95482 (7.46525..8.47651) |
| q1 / r0 archival repeat | 1.45941 | 3.07909 | 1.25317 | 0.18245 | 0.47403 (0.46372..0.48457) | 7.94015 (7.36587..8.55921) |
| q5 / r0 archival repeat | 1.84581 | 3.69707 | 1.39669 | 0.22133 | 0.49725 (0.45007..0.54938) | 7.01731 (5.95174..8.27365) |
| q5 / r1 repaired-runtime repeat | 1.86331 | 3.61072 | 1.56805 | 0.19552 | 0.51402 (0.43816..0.60301) | 8.15898 (7.43878..8.94891) |

The RN/C0 `CI95 upper <=0.99` gate passes strongly: RN is a real improvement
over the inherited C0 mechanism. The required RN/dense `CI95 upper <=1.10`
gate fails by a factor far beyond plausible sampling uncertainty. The cheap
stop therefore rejects the architecture without spending 30 scored windows.

An attempted q5 archival repeat on r1 terminated with a host-side bus error
during recompilation. The node's ordinary CUDA matmul and ECC checks passed;
its temporary CUTLASS 4.7 runtime shared object differed from r0 while the
patched Python source tree was byte-identical. Pointing r1 at a separately
copied, SHA-matched r0 runtime fixed the error without changing a driver or
rebooting. The expanded correctness matrix and a q5 timing repeat then pass on
r1, with RN/dense `8.15898` (`7.43878..8.94891`).

## Mechanism retained and boundary rejected

Retain the RN scale/carrier/correction and packed-MMA mechanisms as transplant
candidates. Reject the A8/D0 two-CTA scheduler and ownership scaffold. The next
reader must retain production TokenSpeed's persistent scheduler, split, and
epilogue ownership while teaching that owner to consume a packed representation
directly. D0 must not proceed to D1 or endpoint wiring.

## Review closure

Collaborative review ran three Fable 5/xhigh iterations. The first restored
the byte-frozen A8 source after repository-wide formatting broke its runtime
SHA gate. The second confirmed the timing harness and rejection inference but
found the missing partially live final drain-tile case; the expanded 513/639
matrix now passes on both ranks. The third returned exact `LGTM`.

Logs: `review-20260814-173419-13324.log`,
`review-20260814-173630-13781.log`, and
`review-20260814-174737-14922.log`.
