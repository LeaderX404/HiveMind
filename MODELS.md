# MODELS.md - the model page (issue #35)

One page. Every symbol on it is either measured or marked as a
hypothesis; nothing here is a measurement claim. This page
follows the unified threshold law (PRE_REGISTERED.md section
10) and the derived-k rule (CONTRIBUTING.md, issue #35).

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

## Thresholds on this page

Every threshold follows one law: threshold = k x sigma_model,
with sigma_model measured on the same configuration the
threshold will judge, and k derived - k = Quantile(noise_dist,
1 - alpha), alpha = 0.27% target false rejection per test, with
the mandatory Gaussianity test of issue #35 deciding between
the normal quantile (k = 3) and the empirical one. No fixed k,
no fixed 0.9, no exceptions.

## Status

Unmeasured. This page is the hypothesis; the measurement
campaign is the verdict. Nothing on it enters README.md as a
claim until the campaign prints the number and its source.
