#!/usr/bin/env python3
"""
Parity checker: HiveMind vs Hugging Face, judged by a threshold
the tool derives from its own measurement (covenant issue #34).

The acceptance threshold for parity is not a constant chosen in
advance. It is derived from the measured noise floor of the judge
itself:

    noise_floor_same  = max |logits(HF, run 1) - logits(HF, run 2)|
                        (same model, same prompt, same backend)
    noise_floor_cross = max |logits(backend A)  - logits(backend B)|
                        (same model, same prompt, a second backend
                         such as CUDA vs CPU, or sdpa vs eager
                         attention - used when one exists, skipped
                         when it does not; it is never faked)
    noise_floor       = max(noise_floor_same, noise_floor_cross)
    threshold         = PARITY_K x noise_floor,  PARITY_K = 3

PARITY_K = 3 carries the same justification as every other k in
the covenant (REPRODUCE_K, section 6 of PRE_REGISTERED.md; the
fragmentation threshold, section 8): 3 sigma = 0.27% false
rejection per test, an accepted price for a threshold nobody
chose. Every number this tool prints comes out of the
measurement - none is chosen in advance.

Refuses to judge (nonzero exit, no default, no fallback):
- transformers or torch missing (no judge, no verdict);
- a zero noise floor (a bit-identical judge measured nothing;
  an under-powered tool rejects nothing, exactly like the
  reproduction tolerance of section 6);
- logits whose shapes disagree (a comparison that cannot be
  made is not a comparison).

Usage:
  python scripts/parity_check.py --hf gpt2 --hive gpt2.safetensors \\
      --config config.json --prompt "Explain AI in simple terms"
"""

import argparse
import sys

# k in threshold = k x noise_floor. Justified in the module
# docstring (3 sigma = 0.27% false rejection per test) - the same
# named decision as REPRODUCE_K in benchmark.py and the
# fragmentation threshold of PRE_REGISTERED.md section 8.
PARITY_K = 3


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


def _cross_backend_diff(model_id, input_ids, device, main_logits):
    """Measure the cross-backend noise floor, if a second backend
    exists. Returns (label, diff), or (None, None) when no second
    backend is available. Nothing is faked: absence is printed,
    not substituted.
    """
    import torch

    # (a) the main run on CUDA vs a CPU copy of the same model
    if device.startswith("cuda") and torch.cuda.is_available():
        cpu_model = _load_hf(model_id, "cpu")
        cpu_logits = _hf_forward(cpu_model, input_ids.to("cpu"))
        return "cuda-vs-cpu", _max_abs_diff(main_logits, cpu_logits)

    # (b) the default attention kernel vs eager/sdpa, same device
    diffs = []
    for impl in ("eager", "sdpa"):
        try:
            alt_model = _load_hf(model_id, device,
                                 attn_implementation=impl)
        except (ValueError, ImportError, RuntimeError):
            continue
        alt_logits = _hf_forward(alt_model, input_ids.to(device))
        diffs.append((impl, _max_abs_diff(main_logits, alt_logits)))
    if diffs:
        impl, diff = max(diffs, key=lambda x: x[1])
        return "default-vs-" + impl, diff

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="HiveMind vs HF parity, judged by a derived"
        " threshold (k x noise_floor).")
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
    parser.add_argument("--max-positions", type=int, default=64,
                        help="cap on prompt tokens compared")
    parser.add_argument("--page-size-mb", type=int, default=1000)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        print("judgment refused: no judge, no verdict (%s)" % exc,
              file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.hf)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    if input_ids.size(1) > args.max_positions:
        input_ids = input_ids[:, : args.max_positions]

    # --- the judge measures its own noise floor before judging ---
    model = _load_hf(args.hf, args.device)
    main_logits = _hf_forward(model, input_ids.to(args.device))
    again_logits = _hf_forward(model, input_ids.to(args.device))
    noise_floor_same = _max_abs_diff(main_logits, again_logits)

    cross_label, noise_floor_cross = _cross_backend_diff(
        args.hf, input_ids, args.device, main_logits)
    if noise_floor_cross is not None:
        noise_floor = max(noise_floor_same, noise_floor_cross)
        print("# noise_floor_same=%s, noise_floor_cross=%s (%s)"
              % (noise_floor_same, noise_floor_cross, cross_label))
    else:
        noise_floor = noise_floor_same
        print("# noise_floor_same=%s, noise_floor_cross=unavailable "
              "(no second backend)" % noise_floor_same)

    if noise_floor == 0.0:
        print("judgment refused: the judge is bit-identical across "
              "runs - nothing was measured (PRE section 6: an "
              "under-powered tool rejects nothing)", file=sys.stderr)
        return 3

    threshold = PARITY_K * noise_floor

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
    print("diff=%s, noise_floor=%s, threshold=%s, status: %s"
          % (diff, noise_floor, threshold, status))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
