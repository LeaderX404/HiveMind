"""
Unit tests for ModelLoader class
"""
import pytest
import torch
from unittest.mock import patch, MagicMock
from hivemind.core.model_loader import ModelLoader


class TestModelLoader:
    """Test cases for ModelLoader"""
    
    @patch('hivemind.core.model_loader.AutoTokenizer')
    @patch('hivemind.core.model_loader.AutoModelForCausalLM')
    def test_load_without_quantization(self, mock_model, mock_tokenizer):
        """Test loading model without quantization"""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        
        # Test
        loader = ModelLoader(model_name="test-model", quantize=False, bits=4)
        model, tokenizer = loader.load()
        
        # Assertions
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_model.from_pretrained.assert_called_once()
        assert model == mock_model_instance
        assert tokenizer == mock_tokenizer_instance
        assert loader.device in ["cuda", "cpu"]
    
    @patch('hivemind.core.model_loader.AutoTokenizer')
    @patch('hivemind.core.model_loader.AutoModelForCausalLM')
    @patch('hivemind.core.model_loader.BitsAndBytesConfig')
    def test_load_with_4bit_quantization(self, mock_bnb_config, mock_model, mock_tokenizer):
        """Test loading model with 4-bit quantization"""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Setup BitsAndBytesConfig mock
        mock_config_instance = MagicMock()
        mock_bnb_config.return_value = mock_config_instance
        
        loader = ModelLoader(model_name="test-model", quantize=True, bits=4)
        model, tokenizer = loader.load()
        
        # Check that BitsAndBytesConfig was called with correct parameters
        mock_bnb_config.assert_called_once()
        call_kwargs = mock_bnb_config.call_args[1]
        assert call_kwargs['load_in_4bit'] == True
        assert call_kwargs['load_in_8bit'] == False
    
    @patch('hivemind.core.model_loader.AutoTokenizer')
    @patch('hivemind.core.model_loader.AutoModelForCausalLM')
    @patch('hivemind.core.model_loader.BitsAndBytesConfig')
    def test_load_with_8bit_quantization(self, mock_bnb_config, mock_model, mock_tokenizer):
        """Test loading model with 8-bit quantization"""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Setup BitsAndBytesConfig mock
        mock_config_instance = MagicMock()
        mock_bnb_config.return_value = mock_config_instance
        
        loader = ModelLoader(model_name="test-model", quantize=True, bits=8)
        model, tokenizer = loader.load()
        
        # Check that BitsAndBytesConfig was called with correct parameters
        mock_bnb_config.assert_called_once()
        call_kwargs = mock_bnb_config.call_args[1]
        assert call_kwargs['load_in_8bit'] == True
        assert call_kwargs['load_in_4bit'] == False
    
    @patch('hivemind.core.model_loader.os.makedirs')
    @patch('hivemind.core.model_loader.Path')
    @patch('hivemind.core.model_loader.torch.save')
    def test_save_shard(self, mock_save, mock_path, mock_makedirs):
        """Test saving a shard"""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__ = MagicMock(return_value=mock_path_instance)
        
        loader = ModelLoader(model_name="test-model")
        loader.model = MagicMock()
        loader.model.config = {"test": "config"}
        
        layers = [("layer1", MagicMock()), ("layer2", MagicMock())]
        result = loader.save_shard(shard_index=0, layers=layers, output_dir="shards")
        
        mock_makedirs.assert_called_once_with("shards", exist_ok=True)
        mock_save.assert_called_once()
        assert result == mock_path_instance
    
    def test_device_detection(self):
        """Test automatic device detection"""
        loader = ModelLoader(model_name="test-model")
        
        # Device should be either 'cuda' or 'cpu'
        assert loader.device in ["cuda", "cpu"]
        
        # If CUDA is available, device should be 'cuda'
        if torch.cuda.is_available():
            assert loader.device == "cuda"
        else:
            assert loader.device == "cpu"
    
    def test_default_values(self):
        """Test default initialization values"""
        loader = ModelLoader()
        
        assert loader.model_name == "microsoft/phi-2"
        assert loader.quantize == False
        assert loader.bits == 4
        assert loader.model is None
        assert loader.tokenizer is None
