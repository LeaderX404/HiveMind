# HiveMind by VALCRYPTIA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![By VALCRYPTIA](https://img.shields.io/badge/By-VALCRYPTIA-0078d7.svg)](https://github.com/LeaderX404/HiveMind)

**HiveMind computes a memory budget and an upper bound on speed from your device before touching weights, and refuses what won't fit.** (Upper bound from the bandwidth law. Measured calibration in progress.)

## 🚀 Features

- **True Out-of-Core Inference**: reads model tensors from disk on demand, so a model larger than RAM can still generate (a design property; what is and is not yet measured is tabulated below)
- **No Full Model Loading**: tensors are read page by page through a bounded cache; the full model is not held in RAM at once (by design, not a measured claim)
- **Direct File Access**: reads tensors directly from Safetensors files on demand (GGUF: not implemented yet)
- **Complete Transformer Support**: full forward pass with RoPE, RMSNorm, Causal Mask, KV Cache, GQA
- **Quantization Support**: Q2, Q4 and Q8 quantization for memory efficiency (2x-8x compression, derived from bit widths against the 16-bit FP16 baseline: 16/8, 16/4, 16/2 - arithmetic, not a benchmark claim; perplexity: not measured)
- **Tensor Caching**: LRU page cache for loaded tensors to avoid repeated file reads
- **Performance Monitoring**: Track RAM usage, GPU usage, tokens/sec, and loading times
- **Refusal at admission**: We refuse at admission if predicted peak > budget. Zero kills under deliberate memory pressure is a measured metric, not a claim (not yet measured).
- **Model Agnostic**: the transformer layouts of the Llama family, Mistral, Qwen and Phi-2 are implemented; numerical parity with Hugging Face is unmeasured for every model (see the parity ledger below)

## 📦 Installation

### Prerequisites

- Python 3.9+ (CI matrix: 3.9 / 3.10 / 3.11)
- PyTorch 2.6+
- Git
- pip

### Install from Source

```bash
# Clone the repository
git clone https://github.com/LeaderX404/HiveMind.git
cd HiveMind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install HiveMind
pip install -e .
```

### Install with pip (coming soon)

```bash
pip install hivemind-valcryptia
```

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    HiveMind by VALCRYPTIA                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ DirectFile   │    │ FileBacked   │    │PagedTransform│  │
│  │ ModelLoader  │───▶│    Pager     │───▶│   Engine     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                       │                  │                 │
│                       ▼                  ▼                 │
│              ┌─────────────────────────────────┐            │
│              │       Safetensors Files           │            │
│              │  (Not fully held in RAM)         │            │
│              └─────────────────────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

1. **DirectFileModelLoader**: Reads model tensors directly from files without loading the entire model
2. **FileBackedPager**: Manages tensor loading with configurable page sizes and LRU caching
3. **PagedTransformerEngine**: Implements full transformer forward pass with:
   - Embeddings
   - Rotary Position Embedding (RoPE)
   - RMS Normalization
   - Multi-Head Attention with Causal Mask
   - Feed-Forward Networks
   - Residual Connections
   - KV Cache for efficient generation
   - GQA (Grouped Query Attention) support

## 💡 Quick Start

### Basic Usage

```python
from hivemind.core.file_pager import FileBackedPager
from hivemind.core.transformer_engine import PagedTransformerEngine, TransformerConfig
from transformers import AutoTokenizer

# ONE Safetensors file per model. Sharded checkpoints
# (model-00001-of-000XX.safetensors) are not supported yet:
# consolidate the shards into a single file first.
MODEL = "llama-2-7b.safetensors"

# The pager owns the file loader: tensors are read from disk on
# demand, page by page (page_size_mb bounds the read burst).
pager = FileBackedPager(MODEL, page_size_mb=1000, device="cpu")

# Build the engine config from the checkpoint's own config.json
# (from_hf_config accepts an HF config object, a dict, or a path).
config = TransformerConfig.from_hf_config("config.json")

# Any tokenizer with .encode/.decode works.
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# The engine reads weights through the pager - not all at once.
engine = PagedTransformerEngine(pager, tokenizer, config, device="cpu")

response = engine.generate("What is the meaning of life?", max_new_tokens=50)
print(response)
```

### With Quantization

Continuing with the `pager` and `tokenizer` from Basic Usage:

```python
from hivemind.core.transformer_engine import PagedTransformerEngine, TransformerConfig

# Q8: quant_bits lives on the config; the engine dequantizes the
# packed weights on the fly. The checkpoint must store packed
# weights with their companion tensors (the layout produced by
# quantize_model_state_dict). Perplexity: not measured.
config = TransformerConfig(
    model_type="llama",
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=32000,
    quant_bits=8,
)

engine = PagedTransformerEngine(pager, tokenizer, config, device="cpu")
response = engine.generate("Explain AI in simple terms", max_new_tokens=50)
print(response)
```

### File format and integrity notes

- **Safetensors only, one file per model.** GGUF raises a clear "not implemented yet" error; sharded checkpoints must be consolidated into a single file first.
- **Integrity verification (optional)**: `DirectFileModelLoader(path, expected_sha256="...")` streams the file through SHA-256 before anything is parsed - a tampered file is rejected up front.
- **Reference presets (experimental)**: `TransformerConfig.from_preset("50b" | "70b" | "100b")` builds reference configs for 50B-100B class models. Not exercised on hardware (see Experimental).

## 🎯 Parity debts (unverified)

This is a ledger, not a claim. No green until measured.

The Llama-family, Mistral, Qwen and Phi-2 layouts are implemented,
but numerical parity with Hugging Face is unmeasured for every
model. Every row below is a debt to be paid with one measured
line, judged by a threshold that is derived, not chosen:

| Model | Parity debt | Target |
|-------|-------------|--------|
| Llama-3.1-8B | Owed — see #34 | p95 per-prompt logit diff vs HF <= k x max(2 runs) (k derived, #35, #36) |
| Qwen2.5-7B | Owed — see #34 | p95 per-prompt logit diff vs HF <= k x max(2 runs) (k derived, #35, #36) |

scripts/parity_check.py runs HiveMind and Hugging Face on the
same prompt and measures the judge's own noise before judging:
HF runs --sessions x --seeds times (defaults 5 x 3) on the
same backend (plus cross-backend runs when a second backend
exists), and the pairwise max-abs-diff run samples give the
noise distribution. The logits instrument's noise measure is
max(2 runs) - the maximum absolute difference between two
noise runs - because logit noise is non-Gaussian (issue #36:
one law, threshold = k x noise_measure(instrument), a measure
per instrument). A mandatory Gaussianity test (scipy.stats KS
test against a fitted normal) derives k: k = 3 when it
passes - provisional under the issue #36 cost derivation,
declared a business commitment by the Verification
commitment section below (issue #37: k* approximately 4.2
under the temporary pi = 0.9 and C_accept / C_reject = 100,
with k = 3 an assumption result, not a rule, explicitly
weaker than the declared commitment) - and the empirical
quantile otherwise. The session structure also decomposes
within-session consistency from across-session stability:
the tool prints sigma_within, sigma_across, ICC_mean,
ICC_quantile and max_session_shift, and the verdict is
"stable" only when ICC_mean > 0.5 AND ICC_quantile > 0.5 AND
max_session_shift < k x sigma_within (issue #37); anything
else is "intermittent or outlier" - the word "stable" is
never claimed on consistency alone. Every number it prints
comes out of the measurement. A debt is paid only by the
protocol below.

### Paying a debt (issue #35)

A debt leaves "Owed" only through the measured protocol, frozen
before the campaign runs:

- Pilot: 10 pilot claims; their per-claim p95 logit diffs give a
  measured sigma.
- N claims: N is sized from a target confidence-interval width on
  that sigma - N = ceil((1.96 x sigma / width)^2), the standard
  95% CI half-width for the mean of the per-claim p95 diffs. The
  target width is named in the campaign issue before the campaign
  runs.
- 3 seeds per claim: every claim runs under three seeds, and all
  three lines are published. Three seeds measure within-session
  consistency; across-session stability is a separate claim
  (5 sessions x 3 seeds, the issue #37 stability gate as
  printed by the tool, MODELS.md issues #36 and #37) and is
  never claimed on consistency alone.
- Aggregation rule: p95(per-prompt diff) < threshold, with the
  threshold from scripts/parity_check.py (k x max(2 runs), k
  derived).
- Pass condition: all claims pass, or 95% pass by a decision named
  before the campaign - never a decision named after seeing the
  numbers.
- Output: diff p95, noise_measure, the stability line
  (sigma_within, sigma_across, ICC_mean, ICC_quantile,
  max_session_shift), threshold, PASS/FAIL - all published,
  none summarized away.

## Verification commitment

We publish parity debts (see table above) rather than claiming verification.
This is a deliberate cost choice:

- C_reject (cost of rejecting a working model): ~1 engineer-hour of investigation
- C_accept (cost of shipping a broken model): loss of user trust, silent wrong outputs

We declare C_accept/C_reject = 100. This is a **business commitment**, not a measurement.
It yields k* ≈ 4.2 under π=0.9 (see MODELS.md). We currently use k=3 as a **temporary
assumption**, explicitly weaker than our declared commitment. This gap is documented,
not hidden.

If we later lower C_accept/C_reject to 10, k* drops below 3.9 — and we will say so
publicly, in this section, before any threshold change.

## 🗺️ Planned architectures (layout not implemented)

Models we intend to support but cannot load today. The Phi-3.5
layout is not implemented in this repository (the implemented
layouts are the Llama family, Mistral, Qwen and Phi-2), so
Phi-3.5-mini cannot be loaded. Implementation and a measured
max-logit-diff parity test are tracked in
[#33](https://github.com/LeaderX404/HiveMind/issues/33). A model
enters the Parity table only after the code can load it and a
measured parity number is published.

## 🧪 Experimental

Untested until the 3-device measurement campaign records
otherwise:

- **Q2 quantization**: implemented and unit-tested in code
  (tests/unit/test_quantization_q2.py); perplexity: not measured.
- **50B-100B model presets (Phase 3)**: reference presets
  (50b/70b/100b) and 4000MB adaptive pages exist in code but have
  not been exercised on hardware. Do not rely on them.

See [LIMITATIONS.md](LIMITATIONS.md) for what this engine cannot
do today.

## 📊 Performance

### File size on disk (arithmetic, not measurements)

File size per quantization level is arithmetic - parameters x
bits per parameter against the 16-bit FP16 baseline - not a
benchmark result:

| Model Size | FP16 | Q8 | Q4 | Q2 | Peak RSS (measured) |
|------------|------|----|----|----|---------------------|
| 1B | ~2GB | ~1GB | ~500MB | ~250MB | not measured |
| 7B | ~14GB | ~7GB | ~3.5GB | ~1.8GB | not measured |
| 13B | ~26GB | ~13GB | ~6.5GB | ~3.3GB | not measured |
| 70B | ~140GB | ~70GB | ~35GB | ~18GB | not measured |
| 100B | ~200GB | ~100GB | ~50GB | ~25GB | not measured |

Peak RSS is a different number from file size, and it must be
measured, not estimated:

```
peak_ram ≈ max_page_size + KV_cache + activations + torch_overhead
KV_cache = 2 × layers × kv_heads × head_dim × seq_len × bytes_per_element
```

(kv_heads, not attention heads: with GQA the cache stores keys and
values before their expansion to query heads. torch_overhead - the
allocator's reserved pools and fragmentation - is measured per run
by the tools in benchmark.py, not assumed.)

### Four measured metrics

The engine's contract. Every status cell stays "Not measured"
until the 3-device campaign records the number; the protocol -
deliberate memory pressure (stress-ng) with simultaneous
generation, N = 20 runs, median + p95 - is pre-registered in
[PRE_REGISTERED.md](PRE_REGISTERED.md):

| Metric | Target | Status |
|--------|--------|--------|
| False acceptance | 0 in N=20 runs | Not measured |
| False rejection | published | Not measured |
| Peak prediction error | median + p95 | Not measured |
| Bandwidth efficiency | published | Not measured |

### Usability levels (A-E: ratios, not fixed thresholds)

ratio = predicted_tps / INTERACTIVE_FLOOR, with
INTERACTIVE_FLOOR = 3 tok/s (a named decision: a comfortable
human reading rate; registered in PRE_REGISTERED.md before any
measurement):

| Level | Ratio | Meaning |
|-------|-------|---------|
| A | > 1 | fits + fast |
| B | 0.5 - 1 | fits, slow |
| C | 0.1 - 0.5 | overflow, usable |
| D | 0.01 - 0.1 | overflow, background |
| E | < 0.01 | proof of concept |

### Speed

No speed numbers are published until they are measured. The
measurement protocol is pre-registered in
[PRE_REGISTERED.md](PRE_REGISTERED.md); measured tokens/sec per
device returns to this section only after the 3-device campaign
records them. Until then, any tokens/sec number claimed about
HiveMind is a number without a measurement.

## 🔧 Configuration

### TransformerConfig Parameters

```python
config = TransformerConfig(
    model_type="llama",      # Model architecture (llama, mistral, qwen, phi)
    hidden_size=4096,        # Hidden layer size
    num_layers=32,           # Number of transformer layers
    num_attention_heads=32,  # Number of attention heads
    num_key_value_heads=8,   # Number of KV heads (for GQA)
    vocab_size=32000,        # Vocabulary size
    max_position_embeddings=32768,  # Maximum position embeddings
    head_dim=128,            # Head dimension
    rope_theta=500000.0,     # RoPE theta parameter
    use_rms_norm=True,       # Use RMSNorm (for Llama/Mistral)
    quant_bits=None          # Quantization bits (2, 4 or 8)
)
```

## 🧪 Testing

Run the test suites:

```bash
# Out-of-core tests
python -m examples.test_out_of_core

# Quantization tests
python -m examples.test_quantization

# Phase 3 tests (Q2 + 50B-100B models)
python -m examples.test_phase3

# Unit tests (incl. Q2 suite)
pytest tests/unit/test_quantization_q2.py
pytest tests/unit/
```

## 🔒 Covenant and the commit-dated seal

The experimental claims of this repository are governed by
[PRE_REGISTERED.md](PRE_REGISTERED.md): thresholds and protocols
frozen before the first measurement. Its SHA-256 lives in
[seal/THRESHOLDS.sha256](seal/THRESHOLDS.sha256) and is verified
by CI on every push - editing the covenant without resealing turns
the build red.

The seal is commit-dated: the git commit that carries
seal/THRESHOLDS.sha256 is its date (see
`git log --follow seal/THRESHOLDS.sha256`). The covenant was
registered and sealed before the first measurement; every later
amendment is a public, numbered entry in that history.

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Contribution Guide](docs/contributing.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers
- PyTorch
- GGUF format by gerganov
- Safetensors format

---

**HiveMind by VALCRYPTIA** - Out-of-core inference on limited hardware. Every claim in this README carries its measurement or an explicit "not yet measured" label.

*Built with ❤️ by VALCRYPTIA*
