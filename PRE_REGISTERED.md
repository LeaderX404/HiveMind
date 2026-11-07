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
  memory, obtained by OOM-stepping calibration
  (benchmark.py --calibrate-shard) and recorded as a gate
  fact. The fixed 6 GB was a round guess (executed in
  amendment #23); the 0.85 safety factor was a guess about a
  guess (executed in amendment #24). No factor survives: the
  budget is a measurement, not a multiplication.
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
survived its own audit.

Amendment record: issues #22 (two-gate structure), #23
(constant audit) and #24 (execution of the 0.85 safety
factor: the budget becomes a measured fact), all before the
first run; no measurements existed, so no runs were
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
from measurements, never picked:

- Noise floor: run the stage 10 times
  (--variance-report 10); the floor is REPRODUCE_K (3,
  3-sigma: beyond it a difference is systematic, not noise)
  times the standard deviation of TPS.
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

## 7. Amendment rule

Any edit to this covenant invalidates all prior runs, requires a
new seal, and an open issue announcing the invalidation - the
same social pricing as serialization guard exceptions.

