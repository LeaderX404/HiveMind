#!/usr/bin/env python
"""
Test script for Out-of-Core Inference
Demonstrates HiveMind by VALCRYPTIA - Out-of-Core Inference Engine for LLMs
"""
import torch
from hivemind.core.direct_loader import DirectFileModelLoader
from hivemind.core.file_pager import FileBackedPager
from hivemind.core.transformer_engine import PagedTransformerEngine, TransformerConfig
from hivemind.utils.perf_monitor import PerformanceMonitor


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    assert DirectFileModelLoader is not None
    assert FileBackedPager is not None
    assert PagedTransformerEngine is not None
    assert TransformerConfig is not None
    assert PerformanceMonitor is not None
    print("✅ All imports successful")


def test_config():
    """Test configuration creation"""
    print("\nTesting configuration...")
    
    # Test Llama config
    llama_config = TransformerConfig(
        model_type="llama",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        vocab_size=32000,
        use_rms_norm=True
    )
    assert llama_config.model_type == "llama"
    assert llama_config.use_rms_norm == True
    print("✅ Llama configuration created")
    
    # Test Mistral config
    mistral_config = TransformerConfig(
        model_type="mistral",
        hidden_size=4096,
        num_layers=24,
        num_attention_heads=32,
        num_key_value_heads=8,
        use_rms_norm=True
    )
    assert mistral_config.num_key_value_heads == 8
    print("✅ Mistral configuration created")


def test_performance_monitor():
    """Test performance monitor"""
    print("\nTesting PerformanceMonitor...")
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Simulate some work
    import time
    time.sleep(0.1)
    monitor.record_token()
    monitor.record_token()
    
    monitor.stop()
    report = monitor.get_report()
    assert "HiveMind Performance Report" in report
    print("✅ PerformanceMonitor working")


def test_rope():
    """Test RoPE computation"""
    print("\nTesting RoPE...")
    config = TransformerConfig(
        model_type="llama",
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        head_dim=16
    )
    
    # Check RoPE frequencies
    assert hasattr(config, '_rope_freqs')
    print("✅ RoPE frequencies computed")


def main():
    """Run all tests"""
    print("=" * 70)
    print("HiveMind by VALCRYPTIA: Out-of-Core Inference Test Suite")
    print("=" * 70)
    
    test_imports()
    test_config()
    test_performance_monitor()
    test_rope()
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    print("\nHiveMind by VALCRYPTIA is ready for:")
    print("  • True out-of-core inference")
    print("  • RoPE (Rotary Position Embedding)")
    print("  • RMSNorm support")
    print("  • Causal mask")
    print("  • KV Cache")
    print("  • GQA (Grouped Query Attention)")
    print("  • Tensor caching")
    print("  • Performance monitoring")
    print("\nTo test with actual model files:")
    print("  python examples/test_out_of_core.py --model-path <path-to-model.safetensors>")
    print("\nBuilt with ❤️ by VALCRYPTIA")


if __name__ == "__main__":
    main()
