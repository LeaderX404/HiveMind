"""
Unit tests for PagedTransformerEngine
"""
import pytest
import torch
from hivemind.core.transformer_engine import PagedTransformerEngine, TransformerConfig, LayerConfig


class TestTransformerConfig:
    """Test TransformerConfig"""
    
    def test_llama_config(self):
        """Test Llama configuration"""
        config = TransformerConfig(
            model_type="llama",
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            vocab_size=32000
        )
        assert config.model_type == "llama"
        assert config.hidden_size == 4096
        assert config.num_layers == 32
        assert config.use_rms_norm == True
    
    def test_mistral_config(self):
        """Test Mistral configuration"""
        config = TransformerConfig(
            model_type="mistral",
            hidden_size=4096,
            num_layers=24,
            num_attention_heads=32,
            num_key_value_heads=8
        )
        assert config.num_key_value_heads == 8
    

class TestLayerConfig:
    """Test LayerConfig"""
    
    def test_layer_config(self):
        """Test layer configuration"""
        config = LayerConfig(
            layer_index=0,
            prefix="model.layers.0",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128
        )
        assert config.layer_index == 0
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 8


class TestPagedTransformerEngine:
    """Test PagedTransformerEngine"""
    
    def test_config_creation(self):
        """Test engine configuration"""
        config = TransformerConfig(
            model_type="llama",
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            vocab_size=100
        )
        assert config is not None
    
    def test_rope_frequencies(self):
        """Test RoPE frequency computation"""
        config = TransformerConfig(
            model_type="llama",
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            head_dim=16
        )
        # Check that RoPE frequencies are computed
        assert hasattr(config, '_rope_freqs')
