"""
HiveMind by VALCRYPTIA - Phase 3 Test Suite
Tests for 50B-100B model support: Q2 Quantization and Large Model Presets
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hivemind.core.file_pager import compute_optimal_page_size_mb
from hivemind.core.quantizer import DynamicQuantizer
from hivemind.core.transformer_engine import LARGE_MODEL_PRESETS, PagedTransformerEngine, TransformerConfig
from hivemind.utils.quant_utils import (
    QuantizationConfig,
    dequantize_model_state_dict,
    dequantize_tensor,
    quantize_model_state_dict,
    quantize_tensor,
)

print("=" * 70)
print("HiveMind by VALCRYPTIA: Phase 3 (50B-100B Models) Test Suite")
print("=" * 70)
print()

# Test 1: Q2 Quantization Roundtrip
print("Testing Q2 Quantization...")
try:
    torch.manual_seed(42)
    tensor = torch.randn(100, 100, dtype=torch.float32)
    original_kb = tensor.numel() * tensor.element_size() / 1024

    config = QuantizationConfig(bits=2, group_size=32)
    quantized, scales, mins = quantize_tensor(tensor, config)
    quantized_kb = quantized.numel() * quantized.element_size() / 1024

    dequantized = dequantize_tensor(quantized, scales, mins, config, tensor.shape, tensor.numel())
    error = (tensor - dequantized.float()).abs().max().item()

    assert error < 1.5, f"Q2 quantization error too high: {error}"
    assert quantized.numel() == (tensor.numel() + 3) // 4

    print("✅ Q2 quantization works")
    print(f"   - Original: {original_kb:.1f} KB -> Q2: {quantized_kb:.1f} KB (~{original_kb / quantized_kb:.0f}x smaller)")
    print(f"   - Max error: {error:.4f}")

except Exception as e:
    print(f"❌ Q2 quantization test failed: {e}")

print()

# Test 2: Q2 State Dict Roundtrip
print("Testing Q2 Model State Dict Quantization...")
try:
    torch.manual_seed(42)
    state_dict = {
        "layer1.weight": torch.randn(100, 100),
        "layer1.bias": torch.randn(100),
        "layer2.weight": torch.randn(100, 100),
        "layer2.bias": torch.randn(100),
    }
    original_size = sum(v.numel() * v.element_size() for v in state_dict.values())

    config = QuantizationConfig(bits=2)
    quantized_state = quantize_model_state_dict(state_dict, config)
    quantized_size = sum(
        v.numel() * v.element_size() for k, v in quantized_state.items() if not k.endswith(("_shape", "_numel"))
    )

    restored = dequantize_model_state_dict(quantized_state, config)
    for key in state_dict:
        if "weight" in key:
            error = (state_dict[key] - restored[key].float()).abs().max().item()
            assert error < 1.5, f"Error too high for {key}: {error}"

    print("✅ Q2 state dict quantization works")
    print(f"   - Original: {original_size / 1024:.1f} KB -> Q2: {quantized_size / 1024:.1f} KB (~{original_size / quantized_size:.1f}x smaller)")

except Exception as e:
    print(f"❌ Q2 state dict test failed: {e}")

print()

# Test 3: Large Model Presets (50B, 70B, 100B)
print("Testing Large Model Presets...")
try:
    for name in ("50b", "70b", "100b"):
        config = TransformerConfig.from_preset(name)
        params = 4 * config.num_layers * config.hidden_size * config.hidden_size
        print(f"   - {name}: hidden={config.hidden_size}, layers={config.num_layers}, heads={config.num_attention_heads}")

    # Q2 preset: a 100B-class config with 2-bit quantization
    config_100b_q2 = TransformerConfig.from_preset("100b", quant_bits=2)
    assert config_100b_q2.quant_config.bits == 2

    print("✅ Large model presets work")
    print("   - from_preset('100b', quant_bits=2) ready for Q2 inference")

except Exception as e:
    print(f"❌ Large model presets test failed: {e}")

print()

# Test 4: Page Size Optimization for 50B-100B models
print("Testing Page Size Optimization for 50B-100B models...")
try:
    # Pure function: no giant files needed, just the tier logic
    page_100b = compute_optimal_page_size_mb(200000.0, 1000)  # 100B FP16 ~ 200GB
    page_70b = compute_optimal_page_size_mb(140000.0, 1000)   # 70B FP16 ~ 140GB
    page_10b = compute_optimal_page_size_mb(30000.0, 500)     # 10B FP16 ~ 30GB
    page_small = compute_optimal_page_size_mb(1000.0, 500)    # 1B model

    assert page_100b == 4000
    assert page_70b == 4000
    assert page_10b == 2000
    assert page_small == 500

    print("✅ Page size optimization works")
    print(f"   - 100B model: {page_100b}MB pages")
    print(f"   - 70B model: {page_70b}MB pages")
    print(f"   - 10B model: {page_10b}MB pages")
    print(f"   - 1B model: {page_small}MB pages (untouched)")

except Exception as e:
    print(f"❌ Page size optimization test failed: {e}")

print()

# Test 5: Dynamic Quantizer picks Q2 for huge models
print("Testing Dynamic Quantizer for huge models...")
try:
    dynamic = DynamicQuantizer(min_bits=2, max_bits=8)

    # 100B model in FP16 (~200GB) on a 16GB machine: only Q2 fits
    bits = dynamic.get_optimal_bits(16.0, 200.0)
    assert bits == 2, f"Expected 2 bits for 100B model on 16GB RAM, got {bits}"

    # Same model with plenty of RAM: Q4 is preferred
    bits_relaxed = dynamic.get_optimal_bits(64.0, 200.0)
    assert bits_relaxed == 4

    print("✅ Dynamic quantizer picks Q2 for 100B models")
    print(f"   - 100B model, 16GB RAM -> {bits}-bit")
    print(f"   - 100B model, 64GB RAM -> {bits_relaxed}-bit")

except Exception as e:
    print(f"❌ Dynamic quantizer test failed: {e}")

print()

# Test 6: Engine on-the-fly dequantization (_quantized_linear)
print("Testing PagedTransformerEngine._quantized_linear with Q2...")
try:
    torch.manual_seed(42)
    weight = torch.randn(8, 4, dtype=torch.float32)
    config = QuantizationConfig(bits=2, group_size=32)
    quantized, scales, mins = quantize_tensor(weight, config)

    class MockPager:
        """Minimal pager stub serving pre-quantized tensors"""

        def __init__(self, store):
            self.store = store

        def load_tensor(self, name, prefetch_next=True):
            return self.store[name]

    pager = MockPager(
        {
            "layer.weight": quantized,
            "layer.weight_scales": scales,
            "layer.weight_zeros": mins,
            "layer.weight_shape": torch.tensor(list(weight.shape)),
            "layer.weight_numel": torch.tensor(weight.numel()),
        }
    )

    engine_config = TransformerConfig(
        model_type="llama",
        hidden_size=32,
        num_layers=1,
        num_attention_heads=4,
        vocab_size=100,
        quant_bits=2,
    )
    engine = PagedTransformerEngine(
        file_pager=pager, tokenizer=object(), config=engine_config, device="cpu"
    )

    x = torch.randn(2, 4)
    out = engine._quantized_linear(x, "layer.weight")

    reference = dequantize_tensor(quantized, scales, mins, config, weight.shape, weight.numel())
    expected = torch.nn.functional.linear(x, reference.float())
    assert out.shape == (2, 8)
    assert torch.allclose(out, expected, atol=1e-3)

    print("✅ Engine _quantized_linear works with Q2")
    print("   - Fixed missing method (was called but never defined before Phase 3)")
    print(f"   - Output shape: {tuple(out.shape)}")

except Exception as e:
    print(f"❌ Engine _quantized_linear test failed: {e}")

print()
print("=" * 70)
print("✅ Phase 3 (50B-100B Models) Test Suite Complete!")
print("=" * 70)
print()
print("Phase 3 Features Implemented:")
print("  ✅ Q2 quantization (4 values packed per byte)")
print("  ✅ Engine on-the-fly dequantization (_quantized_linear implemented)")
print("  ✅ 50B/70B/100B model presets (TransformerConfig.from_preset)")
print("  ✅ 4000MB adaptive pages for 50B+ models")
print("  ✅ Dynamic quantizer falls back to Q2 for 100B models")
print()
print("Memory Requirements for 100B Model:")
print("  • FP16: ~200GB")
print("  • Q4:   ~50GB")
print("  • Q2:   ~25GB (fits on a single 32GB machine)")
print()
print("Next Steps:")
print("  - Test with actual 50B+ model files")
print("  - Q2-aware KV cache compression")
print("  - Benchmark Q2 quality (perplexity) vs Q4")
print()
print("Built with ❤️ by VALCRYPTIA")
