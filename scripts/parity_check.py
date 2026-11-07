#!/usr/bin/env python3
"""
Parity checker: HiveMind vs Hugging Face, judged by a threshold
the tool derives from its own measurement (covenant issues #34
and #35).

The acceptance threshold for parity is not a constant chosen in
advance. It is derived from the measured noise of the judge
itself:

    noise samples = pairwise max |logits(HF, run i) -
                    logits(HF, run j)| over --noise-runs HF
                    runs on the same backend, plus a
                    cross-backend sample when a second backend
                    exists (CUDA vs CPU, or sdpa vs eager
                    attention - used when one exists, skipped
                    when it does not; it is never faked)
    noise_floor   = max(noise samples)      (the raw scale)
    sigma_model   = std(noise samples)     (the threshold
                    basis, issue #35)
    k             = Quantile(noise_dist, 1 - alpha),
                    alpha = 0.27% target false rejection
    threshold     = k x sigma_model

k is derived, never assumed (issue #35). A Gaussianity test on
the noise samples is mandatory: a KS test against a normal
distribution fitted to the samples (scipy.stats; no scipy, no
verdict). If the test passes, k = 3 is legitimate - the
covenant's 3-sigma convention (99.73% of a Gaussian's mass
within +/- 3 sigma, 0.27% outside) - and the normality
assumption is documented next to the verdict. If it fails, k
comes from the empirical quantile of the standardized samples -
never the round 3. A fixed k without a Gaussianity test is a
bare constant wearing a lab coat.

Known simplifications, documented rather than hidden: the
pairwise samples share runs (correlated draws, not independent
ones); the KS test against a fitted normal is Lilliefors
territory (strictly it needs the Lilliefors correction); the
cross-backend samples are pooled with the same-backend samples
into one noise distribution; and with few samples the empirical
quantile at 99.73% is the sample maximum - it lowers the false
rejection rate below the target and raises the false acceptance
risk. More noise runs tighten every one of these.

Refuses to judge (nonzero exit, no default, no fallback):
- transformers or torch missing (no judge, no verdict);
- scipy missing (the Gaussianity test is mandatory);
- fewer than MIN_NOISE_RUNS noise runs (an under-powered test
  rejects nothing);
- a zero noise floor (a bit-identical judge measured nothing,
  exactly like the reproduction tolerance of section 6);
- a zero sigma_model (one constant value is not a
  distribution; there is nothing to standardize);
- logits whose shapes disagree (a comparison that cannot be
  made is not a comparison).

Usage:
  python scripts/parity_check.py --hf gpt2 --hive gpt2.safetensors
      --config config.json --prompt "Explain AI in simple terms"
      [--noise-runs 8]
"""

import argparse
import sys

# Target false rejection rate per test (issue #35). k = 3 is
# never hardcoded: it appears only as the normal quantile when
# the mandatory Gaussianity test passes. KS_SIGNIFICANCE is the
# test's own named decision; MIN_NOISE_RUNS is the floor below
# which the test refuses to pretend it measured a distribution.
ALPHA = 0.0027
KS_SIGNIFICANCE = 0.05
MIN_NOISE_RUNS = 5


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
    """The issue #35 rule: k = Quantile(noise_dist, 1 - alpha),
    never a chosen constant. Runs the mandatory Gaussianity test
    (KS test against a fitted normal, scipy.stats). k = 3 only
    when the test passes, the empirical quantile of the
    standardized samples otherwise. Returns (k, test_line);
    test_line documents the test and the assumption next to the
    verdict.
    """
    import scipy.stats

    mean, std = _sample_stats(samples)
    stat, p = scipy.stats.kstest(samples, "norm", args=(mean, std))
    if p >= KS_SIGNIFICANCE:
        k = 3.0
        line = ("gaussianity: KS stat=%.4f, p=%.4f -> PASSED "
                "(alpha=%.2f); k=3.0 (the 3-sigma convention: "
                "99.73%% of a Gaussian within +/- 3 sigma, "
                "alpha=%.2f%% outside - assumption documented)"
                % (stat, p, KS_SIGNIFICANCE, 100 * ALPHA))
    else:
        z = [(s - mean) / std for s in samples]
        k = _empirical_quantile([abs(v) for v in z], 1.0 - ALPHA)
        line = ("gaussianity: KS stat=%.4f, p=%.4f -> FAILED "
                "(alpha=%.2f); k=%.4f (empirical quantile of the "
                "standardized samples at 1-alpha, alpha=%.2f%%)"
                % (stat, p, KS_SIGNIFICANCE, k, 100 * ALPHA))
    return k, line


def main():
    parser = argparse.ArgumentParser(
        description="HiveMind vs HF parity, judged by a derived "
        "threshold (k x sigma_model, k from the noise "
        "distribution).")
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
    parser.add_argument("--noise-runs", type=int, default=8,
                        help="HF runs used to build the noise "
                        "distribution (default 8, minimum %d; more "
                        "runs, stronger test)" % MIN_NOISE_RUNS)
    parser.add_argument("--max-positions", type=int, default=64,
                        help="cap on prompt tokens compared")
    parser.add_argument("--page-size-mb", type=int, default=1000)
    args = parser.parse_args()

    if args.noise_runs < MIN_NOISE_RUNS:
        print("judgment refused: %d noise runs is under-powered "
              "(minimum %d) - an under-powered test rejects nothing"
              % (args.noise_runs, MIN_NOISE_RUNS), file=sys.stderr)
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

    # --- the judge measures its own noise before judging ---
    model = _load_hf(args.hf, args.device)
    runs = [_hf_forward(model, input_ids.to(args.device))
            for _ in range(args.noise_runs)]

    samples = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            samples.append(_max_abs_diff(runs[i], runs[j]))
    main_logits = runs[0]

    cross = _cross_backend_samples(args.hf, input_ids, args.device,
                                   main_logits)
    samples.extend(diff for _, diff in cross)

    cross_line = (", ".join("%s=%s" % (label, diff)
                           for label, diff in cross)
                  if cross else "unavailable (no second backend)")
    print("# noise_samples=%d (%d runs), cross: %s"
          % (len(samples), args.noise_runs, cross_line))

    noise_floor = max(samples)
    if noise_floor == 0.0:
        print("judgment refused: the judge is bit-identical across "
              "runs - nothing was measured (PRE section 6: an "
              "under-powered tool rejects nothing)", file=sys.stderr)
        return 3

    _, sigma_model = _sample_stats(samples)
    if sigma_model == 0.0:
        print("judgment refused: the noise samples are all equal - "
              "a constant is not a distribution, there is nothing "
              "to standardize", file=sys.stderr)
        return 6

    k, test_line = derive_k(samples)
    print("# " + test_line)
    threshold = k * sigma_model

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
    print("diff=%s, noise_floor=%s, sigma_model=%s, k=%s, "
          "threshold=%s, status: %s"
          % (diff, noise_floor, sigma_model, k, threshold, status))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
