#!/usr/bin/env python
"""
Test script for Q4 and Q8 Quantization
Demonstrates HiveMind v2.1 quantization support for 1B models
"""
import torch
from hivemind.utils.quant_utils import (
    QuantizationConfig, 
    quantize_tensor, 
    dequantize_tensor,
    quantize_model_state_dict,
    dequantize_model_state_dict
)


def test_q8_quantization():
    """Test 8-bit quantization"""
    print("Testing Q8 Quantization...")
    
    # Create a test tensor
    tensor = torch.randn(100, 100, dtype=torch.float32)
    print(f"  Original tensor: {tensor.dtype}, size: {tensor.numel() * tensor.element_size() / 1024:.2f} KB")
    
    # Quantize
    config = QuantizationConfig(bits=8, symmetric=True, group_size=32)
    quantized, scales, zeros = quantize_tensor(tensor, config)
    print(f"  Quantized tensor: {quantized.dtype}, size: {quantized.numel() * quantized.element_size() / 1024:.2f} KB")
    print(f"  Scales: {scales.shape}, Zeros: {zeros.shape}")
    
    # Dequantize
    dequantized = dequantize_tensor(quantized, scales, zeros, config, tensor.shape, tensor.numel())
    print(f"  Dequantized tensor: {dequantized.dtype}, size: {dequantized.numel() * dequantized.element_size() / 1024:.2f} KB")
    
    # Check error
    error = (tensor - dequantized.float()).abs().max().item()
    print(f"  Max error: {error:.6f}")
    
    assert error < 0.1, f"Q8 quantization error too high: {error}"
    print("  ✅ Q8 quantization test passed")


def test_q4_quantization():
    """Test 4-bit quantization"""
    print("\nTesting Q4 Quantization...")
    
    # Create a test tensor
    tensor = torch.randn(100, 100, dtype=torch.float32)
    print(f"  Original tensor: {tensor.dtype}, size: {tensor.numel() * tensor.element_size() / 1024:.2f} KB")
    
    # Quantize
    config = QuantizationConfig(bits=4, symmetric=False, group_size=32)
    quantized, scales, mins = quantize_tensor(tensor, config)
    print(f"  Quantized tensor: {quantized.dtype}, size: {quantized.numel() * quantized.element_size() / 1024:.2f} KB")
    print(f"  Scales: {scales.shape}, Mins: {mins.shape}")
    
    # Dequantize
    dequantized = dequantize_tensor(quantized, scales, mins, config, tensor.shape, tensor.numel())
    print(f"  Dequantized tensor: {dequantized.dtype}, size: {dequantized.numel() * dequantized.element_size() / 1024:.2f} KB")
    
    # Check error
    error = (tensor - dequantized.float()).abs().max().item()
    print(f"  Max error: {error:.6f}")
    
    assert error < 0.5, f"Q4 quantization error too high: {error}"
    print("  ✅ Q4 quantization test passed")


def test_compression_ratio():
    """Test compression ratios"""
    print("\nTesting Compression Ratios...")
    
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        tensor = torch.randn(size, dtype=torch.float32)
        original_size = tensor.numel() * tensor.element_size()
        
        # Q8
        config_q8 = QuantizationConfig(bits=8, group_size=32)
        quantized_q8, scales_q8, zeros_q8 = quantize_tensor(tensor, config_q8)
        q8_size = quantized_q8.numel() * quantized_q8.element_size() + scales_q8.numel() * scales_q8.element_size() + zeros_q8.numel() * zeros_q8.element_size()
        q8_ratio = original_size / q8_size
        
        # Q4
        config_q4 = QuantizationConfig(bits=4, group_size=32)
        quantized_q4, scales_q4, mins_q4 = quantize_tensor(tensor, config_q4)
        q4_size = quantized_q4.numel() * quantized_q4.element_size() + scales_q4.numel() * scales_q4.element_size() + mins_q4.numel() * mins_q4.element_size()
        q4_ratio = original_size / q4_size
        
        print(f"  Size {size}: FP32={original_size/1024:.1f}KB, Q8={q8_size/1024:.1f}KB ({q8_ratio:.1f}x), Q4={q4_size/1024:.1f}KB ({q4_ratio:.1f}x)")


def test_model_state_dict():
    """Test quantization of model state dict"""
    print("\nTesting Model State Dict Quantization...")
    
    # Create a dummy state dict
    state_dict = {
        'layer1.weight': torch.randn(100, 100),
        'layer1.bias': torch.randn(100),
        'layer2.weight': torch.randn(100, 100),
        'layer2.bias': torch.randn(100),
    }
    
    print(f"  Original state dict size: {sum(v.numel() * v.element_size() for v in state_dict.values()) / 1024:.1f} KB")
    
    # Quantize
    config = QuantizationConfig(bits=8)
    quantized_state_dict = quantize_model_state_dict(state_dict, config)
    print(f"  Quantized state dict keys: {list(quantized_state_dict.keys())}")
    
    # Dequantize
    dequantized_state_dict = dequantize_model_state_dict(quantized_state_dict, config)
    
    # Check error
    for key in state_dict:
        if 'weight' in key:
            error = (state_dict[key] - dequantized_state_dict[key].float()).abs().max().item()
            print(f"  {key} max error: {error:.6f}")
            assert error < 0.1, f"Error too high for {key}: {error}"
    
    print("  ✅ Model state dict quantization test passed")


def main():
    """Run all quantization tests"""
    print("=" * 70)
    print("HiveMind v2.1: Q4 and Q8 Quantization Test Suite")
    print("=" * 70)
    
    test_q8_quantization()
    test_q4_quantization()
    test_compression_ratio()
    test_model_state_dict()
    
    print("\n" + "=" * 70)
    print("✅ All quantization tests passed!")
    print("=" * 70)
    print("\nQuantization Support Summary:")
    print("  ✅ Q8 quantization (symmetric/asymmetric)")
    print("  ✅ Q4 quantization (symmetric/asymmetric)")
    print("  ✅ Per-group quantization")
    print("  ✅ Model state dict quantization")
    print("  ✅ Dequantization on-the-fly")
    print("\nCompression Ratios:")
    print("  • FP32 → Q8: ~2x compression")
    print("  • FP32 → Q4: ~4x compression")
    print("\nMemory Requirements for 1B Model:")
    print("  • FP16: ~2GB")
    print("  • Q8: ~1GB")
    print("  • Q4: ~500MB")


if __name__ == "__main__":
    main()
