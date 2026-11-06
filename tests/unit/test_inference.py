"""
Unit tests for Inference Engines
"""
import pytest
import torch
from unittest.mock import MagicMock, patch
from hivemind.core.inference import LocalInferenceEngine, ShardedInferenceEngine


class MockTokenizedOutput:
    """Mock tokenized output that can be moved to device"""
    def __init__(self, data):
        self.data = data
    
    def to(self, device):
        return self


class TestLocalInferenceEngine:
    """Test cases for LocalInferenceEngine"""
    
    def test_init(self):
        """Test initialization"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        engine = LocalInferenceEngine(mock_model, mock_tokenizer, device="cpu")
        
        assert engine.model == mock_model
        assert engine.tokenizer == mock_tokenizer
        assert engine.device == "cpu"
    
    @patch('hivemind.core.inference.torch.no_grad')
    def test_generate(self, mock_no_grad):
        """Test text generation"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup tokenizer mock to return a dict with input_ids
        mock_tokenized = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = mock_tokenized
        mock_tokenizer.decode.return_value = "test response"
        
        # Setup model mock
        mock_outputs = torch.tensor([[1, 2, 3, 4]])
        mock_model.generate.return_value = mock_outputs
        
        engine = LocalInferenceEngine(mock_model, mock_tokenizer, device="cpu")
        
        with patch('hivemind.core.inference.torch.no_grad', mock_no_grad):
            mock_no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_no_grad.return_value.__exit__ = MagicMock(return_value=None)
            
            # Mock the to() method for the dict
            with patch.dict(mock_tokenized, {'to': MagicMock(return_value=mock_tokenized)}):
                response = engine.generate("test prompt", max_new_tokens=50)
        
        # Check that tokenizer was called
        mock_tokenizer.assert_called_once_with("test prompt", return_tensors="pt")
        
        # Check that model.generate was called
        mock_model.generate.assert_called_once()
        
        # Check that decode was called
        assert response == "test response"
    
    def test_generate_with_kwargs(self):
        """Test generation with additional kwargs"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        mock_tokenized = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = mock_tokenized
        mock_tokenizer.decode.return_value = "test response"
        
        mock_outputs = torch.tensor([[1, 2, 3, 4]])
        mock_model.generate.return_value = mock_outputs
        
        engine = LocalInferenceEngine(mock_model, mock_tokenizer, device="cpu")
        
        with patch('hivemind.core.inference.torch.no_grad') as mock_no_grad:
            mock_no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_no_grad.return_value.__exit__ = MagicMock(return_value=None)
            
            with patch.dict(mock_tokenized, {'to': MagicMock(return_value=mock_tokenized)}):
                response = engine.generate(
                    "test prompt", 
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9
                )
        
        # Check that kwargs were passed to generate
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs['temperature'] == 0.7
        assert call_kwargs['top_p'] == 0.9


class TestShardedInferenceEngine:
    """Test cases for ShardedInferenceEngine"""
    
    def test_init(self):
        """Test initialization"""
        mock_shards = [[("layer1", MagicMock())], [("layer2", MagicMock())]]
        mock_tokenizer = MagicMock()
        
        engine = ShardedInferenceEngine(mock_shards, mock_tokenizer, device="cpu")
        
        assert engine.shards == mock_shards
        assert engine.tokenizer == mock_tokenizer
        assert engine.device == "cpu"
    
    def test_forward_shard(self):
        """Test forwarding through a single shard"""
        mock_layer1 = MagicMock()
        mock_layer2 = MagicMock()
        
        # Setup layer behavior
        mock_input = torch.tensor([[1.0, 2.0, 3.0]])
        mock_layer1.return_value = torch.tensor([[4.0, 5.0, 6.0]])
        mock_layer2.return_value = torch.tensor([[7.0, 8.0, 9.0]])
        
        shards = [[("layer1", mock_layer1), ("layer2", mock_layer2)]]
        mock_tokenizer = MagicMock()
        
        engine = ShardedInferenceEngine(shards, mock_tokenizer, device="cpu")
        
        output = engine.forward_shard(0, mock_input)
        
        # Check that layers were called in sequence
        mock_layer1.assert_called_once()
        mock_layer2.assert_called_once()
        
        assert output.shape == mock_input.shape
    
    def test_generate(self):
        """Test text generation with sharded model"""
        # Create mock shards
        mock_layer1 = MagicMock()
        mock_layer2 = MagicMock()
        
        mock_input = torch.tensor([[1.0, 2.0, 3.0]])
        mock_layer1.return_value = torch.tensor([[4.0, 5.0, 6.0]])
        mock_layer2.return_value = torch.tensor([[7.0, 8.0, 9.0]])
        
        shards = [
            [("layer1", mock_layer1)],
            [("layer2", mock_layer2)]
        ]
        
        mock_tokenizer = MagicMock()
        mock_tokenized = {"input_ids": mock_input}
        mock_tokenizer.return_value = mock_tokenized
        mock_tokenizer.decode.return_value = "sharded response"
        
        engine = ShardedInferenceEngine(shards, mock_tokenizer, device="cpu")
        
        with patch.dict(mock_tokenized, {'to': MagicMock(return_value=mock_tokenized)}):
            response = engine.generate("test prompt", max_new_tokens=50)
        
        # Check that tokenizer was called
        mock_tokenizer.assert_called_once_with("test prompt", return_tensors="pt")
        
        # Check that decode was called
        assert response == "sharded response"
    
    def test_generate_with_multiple_shards(self):
        """Test generation with multiple shards"""
        # Create 3 shards
        mock_layer1 = MagicMock()
        mock_layer2 = MagicMock()
        mock_layer3 = MagicMock()
        
        mock_input = torch.tensor([[1.0, 2.0, 3.0]])
        mock_layer1.return_value = torch.tensor([[4.0, 5.0, 6.0]])
        mock_layer2.return_value = torch.tensor([[7.0, 8.0, 9.0]])
        mock_layer3.return_value = torch.tensor([[10.0, 11.0, 12.0]])
        
        shards = [
            [("layer1", mock_layer1)],
            [("layer2", mock_layer2)],
            [("layer3", mock_layer3)]
        ]
        
        mock_tokenizer = MagicMock()
        mock_tokenized = {"input_ids": mock_input}
        mock_tokenizer.return_value = mock_tokenized
        mock_tokenizer.decode.return_value = "multi-shard response"
        
        engine = ShardedInferenceEngine(shards, mock_tokenizer, device="cpu")
        
        with patch.dict(mock_tokenized, {'to': MagicMock(return_value=mock_tokenized)}):
            response = engine.generate("test prompt")
        
        # Check that all layers were called
        mock_layer1.assert_called_once()
        mock_layer2.assert_called_once()
        mock_layer3.assert_called_once()
        
        assert response == "multi-shard response"
