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
from sigma_noise - the chosen 10% is dead), all before
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

## 8. Fragmentation threshold - derived, never chosen

The threshold for discarding flat calibration (`|flat - traced| > threshold`) is
`k x sigma_noise`, where `sigma_noise` comes from `--variance-report` and `k = 3`
(justified: 0.27% false-rejection x ~10 tests/year = 0.027/year, acceptable).
The fixed 10% is **dead** - it was a type-four constant.

If no variance report exists, the threshold **cannot be derived** and the tool
exits with an explicit error. No default. No fallback.

## 9. Amendment rule

Any edit to this covenant invalidates all prior runs, requires a
new seal, and an open issue announcing the invalidation - the
same social pricing as serialization guard exceptions.

