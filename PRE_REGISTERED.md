# Pre-Registered Experimental Covenant - HiveMind Swarm Thesis

This file is the withdrawal line of the swarm experiment. Its
SHA-256 is recorded in seal/THRESHOLDS.sha256 and verified by CI
on every push. Editing this file without resealing breaks the
build loudly and in public.

External seal (the witness that does not trust the author):

    pip install opentimestamps-client
    ots stamp seal/THRESHOLDS.sha256

Commit the resulting .ots proof once Bitcoin confirms it. The
repo hash is the tripwire; the external timestamp is the witness.
Rewriting git history rewrites both files together, but it
cannot rewrite Bitcoin. Any third party (or CI on any fork) can
recompute the hash and compare against the published stamp.

## 1. Swarm thresholds (frozen before first measurement)

- SWARM_ADVANTAGE_FLOOR    = 1.5   (model size per dollar at >= 1 TPS)
- SCALING_EFFICIENCY_FLOOR = 0.65  (per-device share of standalone TPS)
- COORDINATION_TAX_CEILING = 0.40  (wait / compute)
- MIN_INTERACTIVE_TPS      = 1.0   (the definition of "it runs")

Verdict: THESIS HOLDS / THESIS REFUTED. Never edited after data.

## 2. The two gates before any summit talk (strict order)

Gate order: OVERFLOW first, then TWO-SHARD. Measuring the tax
of a swarm that cannot run at all is propaganda, not
measurement. Both gates are carved as constants in benchmark.py
before their first run; editing them after data is the same
crime as editing the thresholds.

### Gate 1 - Overflow gate (the heart of the thesis)

Tests the heart: running a model that no single device can
hold.

- At least MIN_DEVICES (3) devices. Fewer than 3 is a pair,
  not a swarm: no routing, no meaningful coordination tax.
- A real 4-bit MoE whose total size exceeds the smallest
  MEASURED available device memory (recorded per run via
  mem_get_info, never assumed from a spec sheet) but fits
  the aggregate.
- Each shard must stay within the MEASURED max usable shard
  memory, obtained by TRACE-REPLAY calibration
  (benchmark.py --replay-calibrate): the workload
  allocation pattern - persistent weights, transient
  activations, a KV cache that grows monotonically, and
  jitter blocks freed out of order - is journaled and
  replayed with OOM-stepping on the scale factor, and
  recorded as a gate fact. The flat contiguous ceiling
  (--calibrate-shard) survives only as a diagnostic: the
  fragmentation premium = (flat - replayed) / flat is
  measured and recorded, never estimated, and a premium
  above the derived threshold (k x sigma_noise, amendment
  #26) proves the flat ceiling unfaithful. The gate
  consumes the replayed number exclusively, and the capture
  layer is judged before it measures (amendment #26).
  Calibration
  controls are mandatory: a replay at 50% of the measured
  limit must survive and a replay at 150% must fail; if
  either misbehaves the calibration is invalid and no
  number leaves the tool. The fixed 6 GB was a round guess
  (executed in amendment #23); the 0.85 safety factor was a
  guess about a guess (executed in amendment #24); the flat
  calibration was a size test wearing the uniform of a
  pattern test (executed in amendment #25).
- The gate passes only if generation completes with at least
  MIN_COMPLETION_TOKENS (64) tokens: the shortest length
  that can expose a leak of >= 1 MB/token. The slope must be
  measured on a separate 256-token probe run and recorded.

Verdict: OVERFLOW_GATE_PASS / OVERFLOW_GATE_FAIL, printed by
benchmark.py --gate overflow from recorded run facts. A fail
kills the thesis before any anchor or tax work.

### Gate 2 - Two-shard gate (the tax floor)

A real model that runs on one device at >= 1 TPS must run on
two devices at >= 1.5x the model size per dollar at the same
TPS. A fail kills the thesis. Cost: about one week.

No talk about 100B before BOTH gates pass, in this order.

Constant law (amendment #23): every constant in benchmark.py
is either (a) derived from a named decision, (b) measured and
recorded per run, or (c) swept. No fourth kind. The 6 GB
budget and the 16-token minimum were round guesses wearing
uniforms; both were executed two hours after carving. The
0.85 safety factor was a fourth kind wearing the uniform of
a fix; it was executed in amendment #24, hours after it
survived its own audit. Amendment #25 extends the same law
to the tools that judge: a calibration is a hypothesis
until it survives its own controls, and every statistical
constant (k, the target width, the number of runs) is
derived from a named decision, never from a round number.

Amendment record: issues #22 (two-gate structure), #23
(constant audit), #24 (execution of the 0.85 safety
factor: the budget becomes a measured fact) and #25 (the
judge is also on trial: trace-replay calibration with
mandatory controls, and a margin that is an interval
which refuses to judge when under-powered) and #26 (the
ruler of the ruler: the capture layer is judged before
it measures, and the fragmentation threshold is derived
from sigma_noise - the chosen 10% is dead) and #27 (the
recorder judged in time: a live capture gate before the static
one, on-device injection with declared tolerances, and an
extension rule for LIVE_TAGS) and #29 (the blind spot measured: a coverage audit matching the journal against reality before any trace is recorded, and the ladder judged by a linter, not a gate) and #30 (the ladder enforced at runtime: every gate
refuses without its predecessor passed in the same invocation -
the linter guards the text, the enforcement test guards the
behavior) and #32 (the four published metrics,
the A-E ratio to INTERACTIVE_FLOOR = 3 tok/s, and the
deliberate memory pressure protocol - stress-ng pressing the
device while generation runs simultaneously, N = 20 runs,
median and p95, false acceptance target 0, false rejection
rate published - all frozen before any measurement, and the
README carries no number without a source) and #33 (the
missed-target fallback: a miss is published as measured -
publish, acknowledge, investigate, with no post-hoc target
adjustment - and the parity rule: the README Parity table
lists only models the code can load) and #34 (the unified
threshold law: parity judged by k x noise_floor measured by
scripts/parity_check.py - HF run twice, plus a cross-backend
run when one exists - the README parity section renamed from
a status table to a debts ledger, and every present and
future acceptance threshold - reproduction tolerance,
fragmentation threshold, Model B epsilon, L1 delta -
required to derive from a measured noise floor) and #35
(k derived, never assumed: k = Quantile(noise_dist,
1 - alpha), alpha = 0.27%, with a mandatory Gaussianity test
on the noise samples - a KS test against a fitted normal,
scipy.stats, no scipy no verdict - deciding between k = 3
with the assumption documented and the empirical quantile;
the parity threshold basis moved from the max noise floor
to sigma_model with the floor still measured and printed;
the debts paid only by the measured protocol - 10 pilots for
a sigma, N claims sized from a target CI width, 3 seeds per
claim, p95(per-prompt diff) under the threshold, all claims
pass or 95% by a decision named before the campaign; Model
B and L1 removed as commitments and kept only as a forward
reference to RESEARCH_QUESTIONS.md; and the model page
MODELS.md - the T_token bound over activated bytes, h a
measured function, h_threshold derived from T_cloud, p50
and p95 separate) and #36 (one law,
two measures: threshold = k x noise_measure per instrument -
max(2 runs) for the non-Gaussian logits of parity, std after
a mandatory Gaussianity test for the potentially Gaussian
timing of sections 6 and 8, which stay frozen until their
tools adopt the test; the parity basis moved from sigma_model
to the extreme statistic, with the non-Gaussian reason
attached; k derived from cost: k* = k_gaussian + (1/2) x
ln[((1 - pi) / pi) x (C_accept / C_reject)], approximately
4.2 under the temporary pi = 0.9 and C_accept / C_reject =
100 - k = 3 provisional, an assumption result, not a rule,
until the cost issue measures the ratio; the stability
protocol: 3 seeds = within-session consistency, 5 sessions x
3 seeds = across-session stability, ICC = sigma_across^2 /
(sigma_across^2 + sigma_within^2), ICC > 0.5 verified, ICC <
0.5 intermittent, never "stable" on consistency alone; and
RESEARCH_QUESTIONS.md created as the open frontier - Model B
(kappa), L1 (delta), I_eff, h, H_lang and C_accept / C_reject
as definitions, open questions and registered hypotheses,
never commitments), all before
the first run; no measurements existed, so no runs were
invalidated.

## 3. Anchor calibration policy

- ANCHOR_TOLERANCE_SWEEP = [0.05, 0.10, 0.25, 0.50]. The verdict
  is reported as a function of tolerance. A single tolerance was
  rejected: 0.25 could not be tied to a decision that flips at
  26% but not at 24%, so it is declared provisional and swept
  until a deployment decision is defined and a threshold can be
  derived from it.
- ANCHOR_EXTRAPOLATION_CAP = 2.0. No claim may exceed twice the
  anchored size without a second anchor.
- CORRECTION_FACTOR = FORBIDDEN. The anchor is a test, never a
  training signal. Fitting the synthetic tax to the real tax
  turns the anchor into part of the model and voids it as
  evidence.

## 4. Anchor ladder to 100B (ammo count, written before the battle)

| Anchor | Est. effort  | Earns the right to claim |
|--------|--------------|--------------------------|
| 8B     | ~1 week      | up to 16B                |
| 16B    | 1-2 weeks    | up to 32B                |
| 32B    | 2-4 weeks    | up to 64B                |
| 64B    | 1-2 months   | up to 128B               |

Total: 6-9 months of real anchors before any claim about 100B.
Whoever is surprised by this count in the middle of the road
will be tempted to drop the cap; that temptation is the third
way to move a withdrawal line after the data.

## 5. Trace-replay: two declared layers

- Content layer (captured, real): per-layer expert selection and
  token counts from a real MoE run on one device.
- Size layer (derived, from a declared placement map): message
  sizes are the aggregation over shards, not per expert.
- Timing layer (modeled, declared as modeling): arrival times
  are generated from measured RTT distributions and compute
  variance, and swept. Never claimed as captured.

## 6. Reproduction tolerance (derived, never chosen)

The acceptance margin for an independent replay is computed
from measurements, never picked - and it is an INTERVAL,
never a point (amendment #25):

- Noise floor: run the stage N times (--variance-report
  N); the floor is REPRODUCE_K (3) times the standard
  deviation. k = 3 is justified, not decorative: 3 sigma =
  0.27% false failure per test, and at ~10 reproduction
  tests per year that is ~0.027 false failures per year -
  the two named decisions that derive the constant.
- Sigma estimated from n runs is itself noisy (chi-square:
  ~24% relative uncertainty at n=10). The report therefore
  prints [k x sigma_lower, k x sigma_upper] from a
  two-sided 95% interval on sigma (Wilson-Hilferty
  quantiles, a declared approximation), with n and df on
  the line.
- The tool refuses judgment while it is UNDER-POWERED: if
  the margin interval is wider than TARGET_MARGIN_WIDTH
  (0.10, derived from the thesis floor 1.5x),
  --reproduce-tolerance prints JUDGMENT REFUSED and exits
  nonzero. An under-powered tool rejects nothing - it
  abstains.
- The number of runs is derived from the decision (the
  detection width), never from a round number: the report
  prints the plug-in n required until the interval is
  narrower than the target. Paired mode (the same workload,
  same device, standalone vs swarm, per iteration) cancels
  the shared hardware noise and shrinks the required n
  before any budget is spent.
- Hardware spread: the difference of means across the
  replayed configurations.
- Margin = max(noise floor, hardware spread), computed and
  printed by --reproduce-tolerance from recorded reports.
- The margin applies to the swarm ADVANTAGE RATIO, not to
  raw TPS: the ratio divides the same hardware noise by
  itself and survives hardware differences better than
  absolute numbers.
- Independence is a seal chain on a repository the author
  does not own (different account, documented hardware, its
  own CI and its own timestamp), never a declaration.
- A report that predates amendment #25 (no sigma interval
  on the line) is not auditable and is refused.

## 7. Capture gate - the ruler is judged before it measures

The tracing layer (`--record-trace`) is not trusted by default. Before any
`--replay-calibrate`, the `--trace-capture-gate` must pass three checks:
1. **Accounting closure:** sum(alloc) - sum(free) = 0 (or known live memory).
   Any residual != 0 outside the known live set means a missed
   allocation/free - the ruler is perforated.
2. **Known leak injection:** a synthetic 100MB leak must be captured within +/-1%.
3. **Deterministic replay:** the same trace replayed twice yields identical totals
   (or within a declared noise margin).

Calibration without the capture gate is **rejected** - the number is treated as
if it was never measured.

The journal is PyTorch-level, never OS-level: per-token activations must
close their accounting at the PyTorch allocator. An OS-level RSS trace sees
the caching allocator's reserved pools and never closes - the gate would
reject honest traces. The injection level always matches the source the
calibration reads, never a mixed one.

LIVE_TAGS is extensible only by an open issue that names the new tag and
why it is legitimately alive at journal end (MoE expert buffers retained
between tokens will need exactly such an issue when the first real MoE
trace lands).

A fourth static check judges the device itself: a real on-device injection,
witnessed by the device's own counter (CUDA: torch.cuda.memory_stats()
delta within +/-1%; no CUDA: psutil RSS delta within +/-5%, the caching
allocator's declared noise). No reliable injection path available: the
gate refuses - a fake injection is worse than none.

### Live capture gate - the recorder judged while it records (#27)

The static checks judge the journal AFTER the fact; the live gate judges
the recorder WHILE it records, and comes FIRST:
1. **Positive live:** 10 known allocations during the recording window,
   captured in order at the right sizes.
2. **Negative live:** allocations outside the recording window produce
   ZERO records - a recorder without boundaries poisons the analysis.
3. **Live determinism:** the same live pattern, recorded twice, produces
   the same journal.

Strict order of the four gates: live capture -> static capture ->
calibration (with its 50%/150% controls) -> overflow verdict. Each gate
judges one layer; none releases the next.

### Coverage audit - the journal matched against reality (#29)

The recorder's boundary is measured, not declared. Three rules:

1. The instrument is tried first. tracemalloc is expected to be
   blind to PyTorch CPU allocations (they bypass the Python
   allocator); the trial runs anyway and the verdict is recorded.
   torch.profiler must then see a known allocation, or the audit
   refuses - an instrument that cannot see a known allocation
   audits nothing, and a fake audit is worse than none.
2. The journal must be a subset of reality. One known allocation
   on the covered path (torch.empty, AUDIT_COVERED_BYTES) and one
   on an uncovered path (torch.zeros, AUDIT_UNCOVERED_BYTES) are
   journaled by the recorder and witnessed by the accepted
   instrument simultaneously; every journaled allocation must be
   corroborated entry by entry.
3. The boundary is a number, not a docstring. The uncovered-path
   allocation must escape the journal, and the escape is printed
   as a measured size.

Wiring: --record-trace refuses unless BOTH the live gate and
the coverage audit passed in the same invocation - and the two
gates are EXECUTED inside that invocation, not merely named;
--trace-capture-gate refuses unless the audit passed in the same
invocation, executing the audit and the live gate with it;
--replay-calibrate inherits the whole chain through
--trace-capture-gate (amendment #30: a predecessor's flag means
its gate EXECUTES in the same invocation - a stage that names a
predecessor carries that predecessor's flag or refuses before
doing anything). A single allocation on a
path the real pattern exercises that the journal does not claim
rejects the trace.

### The ladder is judged by a linter, not a gate (#29)

The ladder produces ORDER, not a number - a fifth gate would
open an infinite regress. The linter
(tests/unit/test_ladder_lint.py) fails the build when a layer
lacks its gate (completeness) or a gate is wired behind its own
prerequisite (order). The stopping rule is the Gate Law in
CONTRIBUTING.md: every gate needs a named decision and a public
issue; no named decision, no gate. Gates are added by decision,
never by anxiety.

The linter guards the TEXT; the behavior is guarded by
test_ladder_enforcement.py (amendment #30): every gate refuses,
loudly and nonzero, when its predecessor has not passed in the
same invocation - on a CI runner and on a real device alike,
because the refusals fire at invocation level before any
hardware is touched.

## 8. Fragmentation threshold - derived, never chosen

The threshold for discarding flat calibration (`|flat - traced| > threshold`) is
`k x sigma_noise`, where `sigma_noise` comes from `--variance-report` and `k = 3`
(justified: 0.27% false-rejection x ~10 tests/year = 0.027/year, acceptable).
The fixed 10% is **dead** - it was a type-four constant.

If no variance report exists, the threshold **cannot be derived** and the tool
exits with an explicit error. No default. No fallback.

## 9. The four published metrics and the A-E ratio (issue #32)

Frozen before the first measurement of any of them:

- INTERACTIVE_FLOOR = 3 tok/s (a named decision: a comfortable
  human reading rate - below it, "interactive" is a lie). This is
  the floor of the README's A-E levels, distinct from
  MIN_INTERACTIVE_TPS = 1.0 in section 1 (the swarm thesis's
  "it runs" definition); the two answer different questions.
- The A-E usability levels are RATIOS to that floor, not fixed
  thresholds: ratio = predicted_tps / INTERACTIVE_FLOOR;
  A: ratio > 1 (fits + fast), B: 0.5-1 (fits, slow),
  C: 0.1-0.5 (overflow, usable), D: 0.01-0.1 (overflow,
  background), E: < 0.01 (proof of concept).
- Deliberate memory pressure test: stress-ng presses the device
  to a measured budget while generation runs simultaneously,
  N = 20 runs, median and p95 recorded.
- False acceptance: target = 0 in N = 20 runs. One kill in
  twenty fails the metric.
- False rejection rate: published whatever it turns out to be.
- Peak prediction error: median and p95 of the signed relative
  error (predicted peak vs measured peak), published.
- Bandwidth efficiency: measured sustained / theoretical, as a
  percentage, published.

README status cells stay "Not measured" until the 3-device
campaign records the number, and the README honesty rule in
CONTRIBUTING.md polices every number, comparison and absolute
word. No fourth kind here either: every constant above is (a)
derived from a named decision or (b) to be measured and
recorded when the campaign runs.

### Fallback if the target is missed (#33)

If a target above is missed - a kill occurs within the N = 20
deliberate-pressure runs - the miss is published as measured:
the kill count, the run index, and the memory trace of the
killed run. The target is not adjusted, N is not re-rolled, and
the killed run is not discarded. A fix follows the amendment
rule (section 11): a new amendment, a new seal, and an open
issue announcing the change. No denial, no goal-moving.

### Why this rule exists (#33)

Adjusting the target after seeing the data is a dishonest
measurement by definition. The covenant exists to forbid
post-hoc goal-moving: a published miss is stronger evidence of
honest measurement than a met target we cannot prove. The
weaker claim is the stronger product.

## 10. The unified threshold law (issues #34, #35, #36)

No acceptance threshold in this covenant is a bare constant.
One law, not two:

    threshold = k x noise_measure(instrument)

with noise_measure measured on the same configuration the
threshold will judge. The measure belongs to the instrument,
because noise is a property of what is measured, not of the
law:

- Parity (scripts/parity_check.py - logits, non-Gaussian):
  noise_measure = max(2 runs), the maximum absolute difference
  between two noise runs. The judge runs --sessions sessions x
  --seeds seeds (defaults 5 x 3, the issue #36 stability
  protocol); the pairwise max-abs-diff run samples give the
  noise distribution and its max, and the session structure
  decomposes the noise into within-session consistency (3
  seeds) and across-session stability (5 sessions x 3 seeds):
  ICC = sigma_across^2 / (sigma_across^2 + sigma_within^2),
  with ICC > 0.5 = verified and ICC < 0.5 = intermittent - the
  word "stable" is never claimed on consistency alone
  (MODELS.md, issue #36).
- Timing instruments (the reproduction tolerance of section 6
  and the fragmentation threshold of section 8):
  noise_measure = std, legitimate only after a Gaussianity
  test passes - timing noise is potentially Gaussian. Their
  sections keep their frozen named decisions until their tools
  adopt the test.
- k is derived, never assumed (issues #35, #36):
  k = Quantile(noise_dist, 1 - alpha), with a target false
  rejection rate alpha = 0.27% per test. The Gaussianity test
  is mandatory - a KS test against a fitted normal
  (scipy.stats; no scipy, no verdict). If it passes, k = 3 is
  legitimate and the normality assumption is documented next
  to the verdict; if it fails, k comes from the empirical
  quantile of the standardized samples. A fixed k without a
  Gaussianity test is a bare constant wearing a lab coat. The
  cost derivation (MODELS.md, issue #36) moves the boundary to
  k* = k_gaussian + (1/2) x ln[((1 - pi) / pi) x
  (C_accept / C_reject)] - approximately 4.2 under the
  temporary pi = 0.9 and C_accept / C_reject = 100 (definition
  required, RESEARCH_QUESTIONS.md). Named decision: k = 3
  stays provisional until C_accept / C_reject is measured in
  a separate issue; k = 3 is an assumption result, not a
  rule.
- The tool prints diff, noise_measure, sigma_within,
  sigma_across, ICC, the stability status, k, threshold and
  PASS/FAIL, and refuses to judge on a zero noise measure, a
  degenerate noise distribution, a noise that cannot be
  decomposed into within and across, missing scipy, or an
  under-powered session/seed structure - exactly like the
  under-powered tool of section 6.
- The law is one law over every acceptance threshold in this
  covenant: the reproduction tolerance (section 6), the
  fragmentation threshold (section 8) and the parity threshold
  above are each k x a measured noise quantity, with the
  measure chosen by the instrument. The reproduction tolerance
  and the fragmentation threshold predate the per-instrument
  measure and keep their frozen named decisions under their
  own amendments until their tools adopt it. Model B (kappa)
  and L1 (delta) are defined in RESEARCH_QUESTIONS.md as open
  questions and open hypotheses - not commitments; they enter
  this law with their own derived thresholds when their
  definitions freeze into commitments, and no commitment
  exists before then.
- The rule is mirrored in CONTRIBUTING.md (Unified Threshold
  Rule) so the repository text and this covenant cannot
  drift apart.

## 11. Amendment rule

Any edit to this covenant invalidates all prior runs, requires a
new seal, and an open issue announcing the invalidation - the
same social pricing as serialization guard exceptions.

