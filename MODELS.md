# MODELS.md - the model page (issues #35, #36, #37)

One page. Every symbol on it is either measured or marked as a
hypothesis; nothing here is a measurement claim. This page
follows the unified threshold law (PRE_REGISTERED.md section
10) and the derived-k rule (CONTRIBUTING.md, issues #35, #36, #37).

## Token time

    T_token >= max(disk_time, ram_time, compute_time) + latency

- T_token: seconds per generated token, an upper bound from the
  bandwidth law. p50 and p95 are reported separately - a median
  hides the tail, a p95 hides the median, and neither is allowed
  to hide behind the other.
- disk_time = W_t_act / measured disk read bandwidth
- ram_time = W_t_act / measured RAM bandwidth
- compute_time = FLOPs per token / measured device throughput
- latency: the fixed per-token overhead that does not scale with
  bytes (measured, not assumed)

## W_t_act: activated bytes, not total bytes

W_t_act is the byte weight actually touched to produce one
token - the layers read for that step - not the total size of
the checkpoint. Out-of-core inference reads what the token
needs, so the budget and the bound are over activated bytes. A
number computed from total bytes is a different model wearing
this model's name.

## h: the quality function

h = h(C, L, policy, language) is a measured function, never an
assumed constant: C the task class, L the length scale, policy
the decoding policy, language the natural language. It is
measured on the configuration it will judge.

h_threshold is derived from T_cloud - the measured cloud
reference on the same task - never a fixed 0.9. A round quality
number chosen before measurement is a bare constant under the
unified threshold law.

Declaration: h is a simplification. Generalizing h to other
domains, other languages or other policies is an open
hypothesis - neither claimed nor measured here.

## k: derived from cost, not tradition (issues #36, #37)

The 3 in "k = 3" is a Gaussian quantile, and Gaussianity itself
is only a hypothesis until the noise is tested. The cost
derivation moves the boundary to where the money is:

    k* = k_gaussian + (1/2) * ln[((1 - pi) / pi) * (C_accept / C_reject)]

- pi = 0.9: the prior probability that a candidate engine is a
  true parity. Temporary - to be measured later.
- C_accept / C_reject = 100: the cost of accepting a false
  parity (a wrong engine ships as verified) over the cost of
  rejecting a true one (a correct engine is sent back).
  Declared as a business commitment (README's
  Verification commitment section, issue #37) - a
  declaration, not a measurement; the definition and
  measurement stay open (RESEARCH_QUESTIONS.md).
- With the temporary numbers, k* is approximately 4.2.

Named decision (issues #36, #37): k = 3 stays
  provisional - a temporary assumption, explicitly weaker
  than the declared commitment; the gap is documented, not
  hidden - until C_accept / C_reject is measured in a
  separate issue.

Declaration: k = 3 is an assumption result, not a rule. The
cost measurement issue is what would make it one.

## Stability vs consistency (issues #36, #37)

Three seeds in one session are consistency, not stability: they
show that the engine agrees with itself inside one session, and
nothing more.

- 3 seeds = within-session consistency.
- 5 sessions x 3 seeds = across-session stability.
- ICC_mean = sigma_across^2 / (sigma_across^2 +
  sigma_within^2), with sigma_within the spread of the
  within-session pairwise diffs and sigma_across the spread
  of the session-mean diffs.
- ICC_quantile = sigma_p95_across^2 / (sigma_p95_across^2 +
  sigma_within^2), with sigma_p95_across the spread of the
  per-session p95s - the quantile test that a mean-shift
  alone can hide.
- max_session_shift: the largest session-mean shift from the
  overall mean - one bad session must not impersonate
  stability.
- The verdict is "stable" only when ICC_mean > 0.5 AND
  ICC_quantile > 0.5 AND max_session_shift < k x sigma_within
  (issue #37); anything else is "intermittent or outlier".
  The boundary joins the weaker claim, and the word "stable"
  is never printed on consistency alone.
- scripts/parity_check.py prints sigma_within, sigma_across,
  ICC_mean, ICC_quantile, max_session_shift and the stability
  status next to every verdict.

## Thresholds on this page

One law, not two (issues #34, #35, #36):

    threshold = k x noise_measure(instrument)

The measure belongs to the instrument, because noise is a
property of what is measured, not of the law. For logits
(parity), noise_measure = max(2 runs) - the maximum absolute
difference between two noise runs - because logit noise is
non-Gaussian and the extreme statistic is its honest scale.
For timing, noise_measure = std, legitimate only after a
Gaussianity test passes, because timing noise is potentially
Gaussian. k is derived - k = Quantile(noise_dist, 1 - alpha),
alpha = 0.27% target false rejection per test, with the
mandatory Gaussianity test of issue #35 deciding between the
normal quantile (k = 3) and the empirical one - and k = 3
stays provisional under the issue #36 cost derivation
  (declared a commitment in README by issue #37): an
assumption result, not a rule. No fixed k, no fixed 0.9, no
exceptions.

## Status

Unmeasured. This page is the hypothesis; the measurement
campaign is the verdict. Nothing on it enters README.md as a
claim until the campaign prints the number and its source.
