#!/usr/bin/env python3
"""
Parity checker: HiveMind vs Hugging Face, judged by a threshold
the tool derives from its own measurement (covenant issues #34,
#35, #36 and #37).

One law, not two (issue #36):

    threshold = k x noise_measure(instrument)

This instrument measures logits. Logit noise is non-Gaussian,
so its noise measure is the extreme statistic, not the std:

    noise_measure = max(2 runs)
                  = the maximum absolute difference between
                    two noise runs (the max over the pairwise
                    max-abs-diff run samples)

The judge runs --sessions sessions x --seeds seeds HF forwards
(defaults 5 x 3, the issue #36 stability protocol):

    within samples = pairwise run diffs inside each session
                     (3 seeds = within-session consistency)
    across samples = pairwise session-mean diffs
                     (5 sessions x 3 seeds = across-session
                     stability)
    sigma_within   = std(within samples)
    sigma_across   = std(across samples)
    ICC_mean       = sigma_across^2 / (sigma_across^2 +
                     sigma_within^2)
    per-session p95 = the p95 of each session's within diffs;
                     sigma_p95_across = std(per-session p95s)
    ICC_quantile   = sigma_p95_across^2 / (sigma_p95_across^2 +
                     sigma_within^2) - the quantile test that a
                     mean-shift alone can hide
    max_session_shift = the largest session-mean shift from
                     the overall mean of the within diffs

The verdict is "stable" only when ALL THREE gates hold
(issue #37): ICC_mean > 0.5 AND ICC_quantile > 0.5 AND
max_session_shift < k x sigma_within. Anything else is
"intermittent or outlier" (the boundary joins the weaker
claim; the word "stable" is never printed on consistency
alone).

k is derived, never assumed (issues #35, #36). A Gaussianity
test on the noise samples is mandatory: a KS test against a
normal distribution fitted to the samples (scipy.stats; no
scipy, no verdict). If it passes, k = 3 - and k = 3 is
provisional under the issue #36 cost derivation (k* =
k_gaussian + (1/2) x ln[((1 - pi) / pi) x (C_accept /
C_reject)], approximately 4.2 under the temporary pi = 0.9
and C_accept / C_reject = 100 - declared a business commitment
by README's Verification commitment section, issue #37; k = 3
is an assumption result, not a rule, until the cost ratio is
measured in a separate issue). If the Gaussianity test fails,
k comes from the empirical quantile of the standardized
samples - never the round 3. A fixed k without a Gaussianity
test is a bare constant wearing a lab coat.

Known simplifications, documented rather than hidden: the
pairwise samples share runs (correlated draws, not independent
ones); the KS test against a fitted normal is Lilliefors
territory (strictly it needs the Lilliefors correction); the
cross-backend diffs are printed and pooled into the
distribution the Gaussianity test sees, while the noise
measure itself is the max over same-backend run pairs (a
backend gap is a configuration difference, not run noise);
the across samples and max_session_shift compare session
MEANS, a scalar proxy for a tensor distance; the per-session
p95 of a small session is that session's maximum (2 seeds =
one pair, 3 seeds = the max of 3) - the quantile is a max in
disguise until seeds grow; and with few samples the
empirical quantile at 99.73% is the sample maximum - it
lowers the false rejection rate below the target and raises
the false acceptance risk. More sessions and seeds tighten
every one of these.

Refuses to judge (nonzero exit, no default, no fallback):
- transformers or torch missing (no judge, no verdict);
- scipy missing (the Gaussianity test is mandatory);
- an under-powered session/seed structure (an under-powered
  test rejects nothing);
- a zero noise measure (a bit-identical judge measured
  nothing, exactly like the reproduction tolerance of PRE
  section 6);
- a degenerate noise distribution (a constant is not a
  distribution; there is nothing to standardize) or a noise
  that cannot be decomposed into within and across (both
  consistency and stability would be undefined);
- logits whose shapes disagree (a comparison that cannot be
  made is not a comparison).

Usage:
  python scripts/parity_check.py --hf gpt2 --hive gpt2.safetensors
      --config config.json --prompt "Explain AI in simple terms"
      [--sessions 5] [--seeds 3]
"""

import argparse
import sys

# Target false rejection rate per test (issue #35). k = 3 is
# never hardcoded: it appears only as the normal quantile when
# the mandatory Gaussianity test passes, and it stays
# provisional under the issue #36 cost derivation (an
# assumption result, not a rule). KS_SIGNIFICANCE is the
# test's own named decision. The MIN_* floors are the smallest
# structure the tool accepts: fewer than 3 sessions leaves no
# spread between session means, fewer than 2 seeds leaves no
# within-session pair (and the floors imply at least 6 runs,
# above the issue #35 under-power line of 5). ICC_VERIFIED is
# the issue #37 named decision: both ICC_mean and ICC_quantile
# must clear it, and max_session_shift must stay under
# k x sigma_within, before the verdict says "stable". P95 is
# the per-session quantile the ICC_quantile gate reads.
ALPHA = 0.0027
KS_SIGNIFICANCE = 0.05
MIN_SESSIONS = 3
MIN_SEEDS = 2
ICC_VERIFIED = 0.5
P95 = 0.95


def _max_abs_diff(a, b):
    """Max absolute elementwise difference between two tensors."""
    return (a.float() - b.float()).abs().max().item()


def _hf_forward(model, input_ids):
    """One forward pass, logits detached to CPU float32."""
    import torch

    with torch.no_grad():
        return model(input_ids).logits.float().cpu()


def _load_hf(model_id, device, attn_implementation=None):
    """Load the reference model in float32, eval mode."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {"torch_dtype": torch.float32}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model.to(device).eval()


def _cross_backend_samples(model_id, input_ids, device, main_logits):
    """Measure one noise sample per second backend that exists.
    Returns a list of (label, diff); empty when none exists.
    Nothing is faked: absence is printed, not substituted.
    """
    import torch

    samples = []
    if device.startswith("cuda") and torch.cuda.is_available():
        cpu_model = _load_hf(model_id, "cpu")
        cpu_logits = _hf_forward(cpu_model, input_ids.to("cpu"))
        samples.append(("cuda-vs-cpu",
                        _max_abs_diff(main_logits, cpu_logits)))
    for impl in ("eager", "sdpa"):
        try:
            alt_model = _load_hf(model_id, device,
                                 attn_implementation=impl)
        except (ValueError, ImportError, RuntimeError):
            continue
        alt_logits = _hf_forward(alt_model, input_ids.to(device))
        samples.append(("default-vs-" + impl,
                        _max_abs_diff(main_logits, alt_logits)))
    return samples


def _empirical_quantile(values, q):
    """Nearest-rank empirical quantile of a list of floats."""
    import math

    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(q * len(ordered))))
    return ordered[rank - 1]


def _sample_stats(samples):
    """Mean and sample standard deviation of a list of floats."""
    n = len(samples)
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / (n - 1)
    return mean, var ** 0.5


def derive_k(samples):
    """The issues #35/#36 rule: k = Quantile(noise_dist,
    1 - alpha), never a chosen constant. Runs the mandatory
    Gaussianity test (KS test against a fitted normal,
    scipy.stats). k = 3 only when the test passes - provisional
    under the issue #36 cost derivation, declared a business
    commitment by README's Verification commitment section
    (issue #37) - and the empirical quantile of the
    standardized samples otherwise. Returns (k, test_line);
    test_line documents the test and the assumption next to
    the verdict.
    """
    import scipy.stats

    mean, std = _sample_stats(samples)
    stat, p = scipy.stats.kstest(samples, "norm", args=(mean, std))
    if p >= KS_SIGNIFICANCE:
        k = 3.0
        line = ("gaussianity: KS stat=%.4f, p=%.4f -> PASSED "
                "(alpha=%.2f); k=3.0 (the 3-sigma convention - "
                "provisional under the issue #36 cost derivation, "
                "declared a business commitment by README's "
                "Verification commitment section, issue #37: an "
                "assumption result, not a rule, until the cost "
                "ratio is measured; assumption documented)"
                % (stat, p, KS_SIGNIFICANCE))
    else:
        z = [(s - mean) / std for s in samples]
        k = _empirical_quantile([abs(v) for v in z], 1.0 - ALPHA)
        line = ("gaussianity: KS stat=%.4f, p=%.4f -> FAILED "
                "(alpha=%.2f); k=%.4f (empirical quantile of the "
                "standardized samples at 1-alpha, alpha=%.2f%%)"
                % (stat, p, KS_SIGNIFICANCE, k, 100 * ALPHA))
    return k, line


def stability(within_samples, across_samples, session_p95s,
              session_mean_scalars, k):
    """The issue #36/#37 decomposition: within-session
    consistency vs across-session stability, judged by three
    gates (issue #37): ICC_mean = sigma_across^2 /
    (sigma_across^2 + sigma_within^2), ICC_quantile =
    sigma_p95_across^2 / (sigma_p95_across^2 + sigma_within^2),
    and max_session_shift < k x sigma_within. The verdict is
    "stable" only when all three hold; anything else is
    "intermittent or outlier" (the boundary joins the weaker
    claim). Returns (sigma_within, sigma_across, icc_mean,
    icc_quantile, max_session_shift, status), or a tuple
    starting with None and ending in the refusal reason when
    the variance cannot be decomposed and both claims would
    be undefined.
    """
    _, sigma_within = _sample_stats(within_samples)
    _, sigma_across = _sample_stats(across_samples)
    _, sigma_p95_across = _sample_stats(session_p95s)
    if (sigma_within == 0.0 and sigma_across == 0.0
            and sigma_p95_across == 0.0):
        return (None, None, None, None, None,
                "no within- or across-session variance to "
                "decompose - consistency and stability are both "
                "undefined")
    if sigma_within == 0.0 and sigma_p95_across == 0.0:
        return (None, None, None, None, None,
                "no within-session variance and no per-session "
                "p95 spread - ICC_quantile would be undefined")
    icc_mean = sigma_across ** 2 / (sigma_across ** 2
                                    + sigma_within ** 2)
    icc_quantile = sigma_p95_across ** 2 / (
        sigma_p95_across ** 2 + sigma_within ** 2)
    overall_mean = sum(within_samples) / len(within_samples)
    max_session_shift = max(abs(m - overall_mean)
                            for m in session_mean_scalars)
    if (icc_mean > ICC_VERIFIED and icc_quantile > ICC_VERIFIED
            and max_session_shift < k * sigma_within):
        status = "stable"
    else:
        status = "intermittent or outlier"
    return (sigma_within, sigma_across, icc_mean, icc_quantile,
            max_session_shift, status)


def main():
    parser = argparse.ArgumentParser(
        description="HiveMind vs HF parity, judged by a derived "
        "threshold (k x max(2 runs) for logits, k from the noise "
        "distribution; prints the issue #37 three-gate stability "
        "verdict: ICC_mean, ICC_quantile, max_session_shift).")
    parser.add_argument("--hf", required=True,
                        help="HF model id or path: the reference side")
    parser.add_argument("--hive", required=True,
                        help="single-file Safetensors checkpoint")
    parser.add_argument("--config", required=True,
                        help="config.json path for from_hf_config")
    parser.add_argument("--tokenizer", default=None,
                        help="HF tokenizer id (default: same as --hf)")
    parser.add_argument("--prompt", required=True,
                        help="the prompt both sides run on")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sessions", type=int, default=5,
                        help="judge sessions (default 5, minimum %d; "
                        "the across-session stability protocol)"
                        % MIN_SESSIONS)
    parser.add_argument("--seeds", type=int, default=3,
                        help="seeds per session (default 3, minimum "
                        "%d; the within-session consistency "
                        "protocol)" % MIN_SEEDS)
    parser.add_argument("--max-positions", type=int, default=64,
                        help="cap on prompt tokens compared")
    parser.add_argument("--page-size-mb", type=int, default=1000)
    args = parser.parse_args()

    if args.sessions < MIN_SESSIONS or args.seeds < MIN_SEEDS:
        print("judgment refused: %d sessions x %d seeds is "
              "under-powered (minimum %d sessions, %d seeds) - an "
              "under-powered test rejects nothing"
              % (args.sessions, args.seeds, MIN_SESSIONS,
                 MIN_SEEDS), file=sys.stderr)
        return 5

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        print("judgment refused: no judge, no verdict (%s)" % exc,
              file=sys.stderr)
        return 2

    try:
        import scipy.stats  # noqa: F401 - the mandatory test needs it
    except ImportError:
        print("judgment refused: the Gaussianity test is mandatory "
              "and scipy is missing - no test, no verdict",
              file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.hf)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    if input_ids.size(1) > args.max_positions:
        input_ids = input_ids[:, : args.max_positions]

    # --- the judge measures its own noise before judging:
    # sessions x seeds forwards on the same backend ---
    model = _load_hf(args.hf, args.device)
    session_runs = []
    for _ in range(args.sessions):
        session_runs.append(
            [_hf_forward(model, input_ids.to(args.device))
             for _ in range(args.seeds)])
    flat_runs = [run for runs in session_runs for run in runs]
    main_logits = flat_runs[0]

    # all pairwise run diffs: the noise distribution
    run_samples = []
    for i in range(len(flat_runs)):
        for j in range(i + 1, len(flat_runs)):
            run_samples.append(_max_abs_diff(flat_runs[i],
                                             flat_runs[j]))

    # the issue #36/#37 decomposition: within (same session),
    # across (session means), per-session p95s, session shifts
    session_within = []
    for runs in session_runs:
        diffs = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                diffs.append(_max_abs_diff(runs[i], runs[j]))
        session_within.append(diffs)
    within_samples = [d for diffs in session_within
                      for d in diffs]
    session_p95s = [_empirical_quantile(diffs, P95)
                    for diffs in session_within]
    session_mean_scalars = [sum(diffs) / len(diffs)
                            for diffs in session_within]
    session_means = [torch.stack(runs).mean(dim=0)
                     for runs in session_runs]
    across_samples = []
    for i in range(len(session_means)):
        for j in range(i + 1, len(session_means)):
            across_samples.append(_max_abs_diff(session_means[i],
                                                session_means[j]))

    # cross-backend diffs: printed, pooled into the distribution
    # the Gaussianity test sees, never part of max(2 runs)
    cross = _cross_backend_samples(args.hf, input_ids, args.device,
                                   main_logits)
    pooled_samples = run_samples + [diff for _, diff in cross]
    cross_line = (", ".join("%s=%s" % (label, diff)
                           for label, diff in cross)
                  if cross else "unavailable (no second backend)")
    print("# noise_samples=%d (%d runs), cross: %s"
          % (len(pooled_samples), args.sessions * args.seeds,
             cross_line))

    # noise_measure = max(2 runs): the logits instrument's
    # measure (issue #36) - same-backend run pairs only
    noise_measure = max(run_samples)
    if noise_measure == 0.0:
        print("judgment refused: the judge is bit-identical across "
              "runs - nothing was measured (PRE section 6: an "
              "under-powered tool rejects nothing)", file=sys.stderr)
        return 3

    _, pooled_std = _sample_stats(pooled_samples)
    if pooled_std == 0.0:
        print("judgment refused: the noise samples are all equal - "
              "a constant is not a distribution, there is nothing "
              "to standardize", file=sys.stderr)
        return 6

    # k first (issue #37): the max_session_shift gate reads it
    k, test_line = derive_k(pooled_samples)

    stab = stability(within_samples, across_samples,
                     session_p95s, session_mean_scalars, k)
    if stab[0] is None:
        print("judgment refused: %s" % stab[5], file=sys.stderr)
        return 6
    (sigma_within, sigma_across, icc_mean, icc_quantile,
     max_session_shift, stability_status) = stab
    print("stability: sigma_within=%s, sigma_across=%s, "
          "ICC_mean=%s, ICC_quantile=%s, max_session_shift=%s, "
          "status: %s" % (sigma_within, sigma_across, icc_mean,
                          icc_quantile, max_session_shift,
                          stability_status))
    print("# " + test_line)
    threshold = k * noise_measure

    # --- the examined side: HiveMind, same prompt, same tokens ---
    from hivemind.core.file_pager import FileBackedPager
    from hivemind.core.transformer_engine import (
        PagedTransformerEngine,
        TransformerConfig,
    )

    pager = FileBackedPager(args.hive,
                            page_size_mb=args.page_size_mb,
                            device=args.device)
    config = TransformerConfig.from_hf_config(args.config)
    engine = PagedTransformerEngine(pager, tokenizer, config,
                                    device=args.device)
    with torch.no_grad():
        hive_logits = engine.forward(input_ids.to(args.device))
    hive_logits = hive_logits.float().cpu()

    if tuple(hive_logits.shape) != tuple(main_logits.shape):
        print("judgment refused: logits shapes disagree (hive %s vs "
              "hf %s) - not a comparison" % (tuple(hive_logits.shape),
              tuple(main_logits.shape)), file=sys.stderr)
        return 4

    diff = _max_abs_diff(main_logits, hive_logits)
    status = "PASS" if diff <= threshold else "FAIL"
    print("diff=%s, noise_measure=%s, sigma_within=%s, "
          "sigma_across=%s, ICC_mean=%s, ICC_quantile=%s, "
          "max_session_shift=%s, k=%s, threshold=%s, status: %s"
          % (diff, noise_measure, sigma_within, sigma_across,
             icc_mean, icc_quantile, max_session_shift, k,
             threshold, status))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
