# HiveMind by VALCRYPTIA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![By VALCRYPTIA](https://img.shields.io/badge/By-VALCRYPTIA-0078d7.svg)](https://github.com/LeaderX404/HiveMind)

**Out-of-Core Inference Engine for Running 7B+ LLMs on Single Devices with Limited RAM**

## 🚀 Features

- **True Out-of-Core Inference**: Run large models on devices with limited RAM by loading weights directly from disk
- **No Full Model Loading**: Never loads the entire model into RAM at once
- **Direct File Access**: Reads tensors directly from GGUF/Safetensors files on demand
- **Complete Transformer Support**: Full forward pass with RoPE, RMSNorm, Causal Mask, KV Cache, GQA
- **Quantization Support**: Q2, Q4 and Q8 quantization for memory efficiency (2x-8x compression, derived from bit widths against the 16-bit FP16 baseline: 16/8, 16/4, 16/2 - arithmetic, never a benchmark claim)
- **Tensor Caching**: LRU cache for loaded tensors to avoid repeated file reads
- **Performance Monitoring**: Track RAM usage, GPU usage, tokens/sec, and loading times
- **Model Agnostic**: Works with Llama, Mistral, Qwen, Phi-2, and other transformer models

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
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
│              │       GGUF/Safetensors Files      │            │
│              │  (Never fully loaded into RAM)   │            │
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
from hivemind.core.direct_loader import DirectFileModelLoader
from hivemind.core.file_pager import FileBackedPager
from hivemind.core.transformer_engine import PagedTransformerEngine, TransformerConfig
from transformers import AutoTokenizer

# Initialize components
loader = DirectFileModelLoader("model.safetensors", device="cpu")
pager = FileBackedPager("model.safetensors", page_size_mb=1000, device="cpu")

# Create transformer config (example: Llama)
config = TransformerConfig(
    model_type="llama",
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=32000,
    use_rms_norm=True
)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create engine
engine = PagedTransformerEngine(pager, tokenizer, config, device="cpu")

# Generate text
response = engine.generate("What is the meaning of life?", max_new_tokens=50)
print(response)
```

### With Quantization

```python
from hivemind.core.direct_loader import DirectFileModelLoader
from hivemind.core.transformer_engine import PagedTransformerEngine, TransformerConfig

# Use Q8 quantization
config = TransformerConfig(
    model_type="llama",
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=32000,
    quant_bits=8  # Enable Q8 quantization
)

loader = DirectFileModelLoader("model-quant.safetensors", device="cpu", quant_bits=8)
engine = PagedTransformerEngine(pager, tokenizer, config, device="cpu")

# Generate with quantized model
response = engine.generate("Explain AI in simple terms", max_new_tokens=50)
```

## 🎯 Supported Models

- ✅ **Llama** (all versions)
- ✅ **Mistral** (with GQA support)
- ✅ **Qwen**
- ✅ **Phi-2**
- ✅ **Custom transformer models**

## 🧪 Experimental

Untested until the 3-device measurement campaign records
otherwise:

- **50B-100B model presets (Phase 3)**: reference presets
  (50b/70b/100b) and 4000MB adaptive pages exist in code but have
  never been exercised on hardware. Do not rely on them.

See [LIMITATIONS.md](LIMITATIONS.md) for what this engine cannot
do today.

## 📊 Performance

### Memory Requirements (arithmetic, not measurements)

Memory per quantization level is arithmetic - parameters x bits per
parameter against the 16-bit FP16 baseline - never a benchmark
result:

| Model Size | FP16 | Q8 | Q4 | Q2 |
|------------|------|----|----|----|
| 1B | ~2GB | ~1GB | ~500MB | ~250MB |
| 7B | ~14GB | ~7GB | ~3.5GB | ~1.8GB |
| 13B | ~26GB | ~13GB | ~6.5GB | ~3.3GB |
| 70B | ~140GB | ~70GB | ~35GB | ~18GB |
| 100B | ~200GB | ~100GB | ~50GB | ~25GB |

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

**HiveMind by VALCRYPTIA** - Revolutionizing LLM Inference on Limited Hardware

*Built with ❤️ by VALCRYPTIA*
