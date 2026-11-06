"""
Unit tests for Phase 3: Q2 quantization and 50B-100B model support
"""
import pytest
import torch

from hivemind.core.file_pager import compute_optimal_page_size_mb
from hivemind.core.quantizer import DynamicQuantizer
from hivemind.core.transformer_engine import (
    LARGE_MODEL_PRESETS,
    PagedTransformerEngine,
    TransformerConfig,
)
from hivemind.utils.quant_utils import (
    QuantizationConfig,
    dequantize_model_state_dict,
    dequantize_tensor,
    quantize_model_state_dict,
    quantize_tensor,
)


class TestQuantizationConfigQ2:
    """Test QuantizationConfig with 2-bit support"""

    def test_q2_config_accepted(self):
        config = QuantizationConfig(bits=2)
        assert config.bits == 2

    def test_invalid_bits_rejected(self):
        with pytest.raises(AssertionError):
            QuantizationConfig(bits=3)
        with pytest.raises(AssertionError):
            QuantizationConfig(bits=6)

    def test_q4_q8_still_supported(self):
        assert QuantizationConfig(bits=4).bits == 4
        assert QuantizationConfig(bits=8).bits == 8


class TestQ2Quantization:
    """Test Q2 quantize/dequantize roundtrips"""

    def test_q2_roundtrip(self):
        torch.manual_seed(0)
        tensor = torch.randn(100, 100, dtype=torch.float32)
        config = QuantizationConfig(bits=2, group_size=32)
        quantized, scales, mins = quantize_tensor(tensor, config)

        dequantized = dequantize_tensor(
            quantized, scales, mins, config, tensor.shape, tensor.numel()
        )
        assert dequantized.shape == tensor.shape

        error = (tensor - dequantized.float()).abs().max().item()
        # 2-bit: 4 levels per group -> max error is half the level step
        assert error < 1.5, f"Q2 quantization error too high: {error}"

    def test_q2_packing_size(self):
        tensor = torch.randn(256, dtype=torch.float32)
        config = QuantizationConfig(bits=2, group_size=32)
        quantized, scales, mins = quantize_tensor(tensor, config)
        # Four 2-bit codes are packed into a single byte
        assert quantized.dtype == torch.uint8
        assert quantized.numel() == (tensor.numel() + 3) // 4

    def test_q2_constant_tensor(self):
        tensor = torch.full((64,), 2.5)
        config = QuantizationConfig(bits=2, group_size=32)
        quantized, scales, mins = quantize_tensor(tensor, config)
        dequantized = dequantize_tensor(
            quantized, scales, mins, config, tensor.shape, tensor.numel()
        )
        assert torch.allclose(dequantized.float(), tensor)

    def test_q2_state_dict_roundtrip(self):
        torch.manual_seed(0)
        state_dict = {
            "layer1.weight": torch.randn(64, 64),
            "layer1.bias": torch.randn(64),
        }
        config = QuantizationConfig(bits=2)
        quantized_state = quantize_model_state_dict(state_dict, config)
        restored = dequantize_model_state_dict(quantized_state, config)
        assert restored["layer1.weight"].shape == state_dict["layer1.weight"].shape
        error = (
            (state_dict["layer1.weight"] - restored["layer1.weight"].float()).abs().max().item()
        )
        assert error < 1.5

    def test_q2_compresses_more_than_q4(self):
        torch.manual_seed(0)
        tensor = torch.randn(1024, dtype=torch.float32)
        q2_config = QuantizationConfig(bits=2, group_size=32)
        q4_config = QuantizationConfig(bits=4, group_size=32)
        q2_quantized, _, _ = quantize_tensor(tensor, q2_config)
        q4_quantized, _, _ = quantize_tensor(tensor, q4_config)
        assert q2_quantized.numel() < q4_quantized.numel()


class TestEngineQuantizedLinear:
    """Test the engine's on-the-fly dequantization path (Phase 3 fix)"""

    def test_engine_quantized_linear_q2(self):
        torch.manual_seed(0)
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

        reference = dequantize_tensor(
            quantized, scales, mins, config, weight.shape, weight.numel()
        )
        expected = torch.nn.functional.linear(x, reference.float())
        assert out.shape == (2, 8)
        assert torch.allclose(out, expected, atol=1e-3)


class TestLargeModelPresets:
    """Test the 50B-100B reference presets (Phase 3)"""

    def test_preset_70b(self):
        config = TransformerConfig.from_preset("70b")
        assert config.hidden_size == 8192
        assert config.num_layers == 80
        assert config.num_attention_heads == 64
        assert config.num_key_value_heads == 8

    def test_preset_100b_with_q2(self):
        config = TransformerConfig.from_preset("100b", quant_bits=2)
        assert config.hidden_size == 10240
        assert config.num_layers == 100
        assert config.quant_bits == 2
        assert config.quant_config.bits == 2

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            TransformerConfig.from_preset("does-not-exist")

    def test_all_presets_use_gqa(self):
        for name in LARGE_MODEL_PRESETS:
            assert LARGE_MODEL_PRESETS[name]["num_key_value_heads"] == 8


class TestLargeModelPageOptimization:
    """Test page size tiers for 50B-100B models (Phase 3)"""

    def test_50b_model_gets_largest_pages(self):
        # 50B params in FP16 is roughly 100GB on disk
        assert compute_optimal_page_size_mb(150000.0, 1000) == 4000

    def test_10b_model_gets_2000mb(self):
        assert compute_optimal_page_size_mb(30000.0, 500) == 2000

    def test_small_models_untouched(self):
        assert compute_optimal_page_size_mb(1000.0, 500) == 500

    def test_page_size_never_lowered(self):
        assert compute_optimal_page_size_mb(150000.0, 8000) == 8000


class TestDynamicQuantizerPhase3:
    """Test dynamic bit selection for huge models"""

    def test_picks_2bit_when_nothing_else_fits(self):
        dynamic = DynamicQuantizer(min_bits=2, max_bits=8)
        # 100B model in FP16 (~200GB) on a 16GB machine: only Q2 fits
        assert dynamic.get_optimal_bits(16.0, 200.0) == 2

    def test_prefers_higher_bits_when_memory_allows(self):
        dynamic = DynamicQuantizer(min_bits=2, max_bits=8)
        # 100GB model with 20GB available: 8-bit needs 25GB, Q4 needs 12.5GB
        assert dynamic.get_optimal_bits(20.0, 100.0) == 4

    def test_picks_8bit_when_memory_plentiful(self):
        dynamic = DynamicQuantizer(min_bits=2, max_bits=8)
        # Same 100GB model with 64GB available: even 8-bit (25GB) fits
        assert dynamic.get_optimal_bits(64.0, 100.0) == 8
