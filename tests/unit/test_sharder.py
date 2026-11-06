"""
Unit tests for ModelSharder class
"""
import pytest
import torch
from unittest.mock import MagicMock
from hivemind.core.sharder import ModelSharder


class TestModelSharder:
    """Test cases for ModelSharder"""
    
    def test_init(self):
        """Test initialization"""
        mock_model = MagicMock()
        sharder = ModelSharder(mock_model, num_shards=2)
        
        assert sharder.model == mock_model
        assert sharder.num_shards == 2
        assert sharder.shards == []
    
    def test_split_basic(self):
        """Test basic model splitting"""
        # Create a mock model with some linear layers
        mock_model = MagicMock()
        mock_linear1 = MagicMock(spec=torch.nn.Linear)
        mock_linear2 = MagicMock(spec=torch.nn.Linear)
        mock_linear3 = MagicMock(spec=torch.nn.Linear)
        mock_linear4 = MagicMock(spec=torch.nn.Linear)
        
        # Mock named_modules to return our linear layers
        mock_model.named_modules.return_value = [
            ("layer1", mock_linear1),
            ("layer2", mock_linear2),
            ("layer3", mock_linear3),
            ("layer4", mock_linear4),
            ("other", MagicMock())  # Non-linear layer
        ]
        
        sharder = ModelSharder(mock_model, num_shards=2)
        shards = sharder.split()
        
        # Should have 2 shards
        assert len(shards) == 2
        
        # Each shard should contain linear layers
        assert len(shards[0]) == 2  # First 2 layers
        assert len(shards[1]) == 2  # Last 2 layers
    
    def test_split_uneven(self):
        """Test splitting with uneven number of layers"""
        mock_model = MagicMock()
        mock_linear1 = MagicMock(spec=torch.nn.Linear)
        mock_linear2 = MagicMock(spec=torch.nn.Linear)
        mock_linear3 = MagicMock(spec=torch.nn.Linear)
        
        mock_model.named_modules.return_value = [
            ("layer1", mock_linear1),
            ("layer2", mock_linear2),
            ("layer3", mock_linear3),
        ]
        
        sharder = ModelSharder(mock_model, num_shards=2)
        shards = sharder.split()
        
        # Should have 2 shards
        assert len(shards) == 2
        
        # First shard should have 1 layer, second should have 2
        assert len(shards[0]) == 1
        assert len(shards[1]) == 2
    
    def test_get_shard_valid(self):
        """Test getting a valid shard"""
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("layer1", MagicMock(spec=torch.nn.Linear)),
            ("layer2", MagicMock(spec=torch.nn.Linear)),
        ]
        
        sharder = ModelSharder(mock_model, num_shards=2)
        sharder.split()
        
        shard = sharder.get_shard(0)
        assert shard == sharder.shards[0]
        
        shard = sharder.get_shard(1)
        assert shard == sharder.shards[1]
    
    def test_get_shard_invalid(self):
        """Test getting an invalid shard index"""
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("layer1", MagicMock(spec=torch.nn.Linear)),
        ]
        
        sharder = ModelSharder(mock_model, num_shards=1)
        sharder.split()
        
        with pytest.raises(ValueError, match="Shard index 5 out of range"):
            sharder.get_shard(5)
    
    def test_split_with_no_linear_layers(self):
        """Test splitting model with no linear layers"""
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("layer1", MagicMock()),  # Not a Linear layer
            ("layer2", MagicMock()),  # Not a Linear layer
        ]
        
        sharder = ModelSharder(mock_model, num_shards=2)
        shards = sharder.split()
        
        # Should have 2 empty shards
        assert len(shards) == 2
        assert len(shards[0]) == 0
        assert len(shards[1]) == 0
    
    def test_split_with_single_shard(self):
        """Test splitting into single shard"""
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("layer1", MagicMock(spec=torch.nn.Linear)),
            ("layer2", MagicMock(spec=torch.nn.Linear)),
        ]
        
        sharder = ModelSharder(mock_model, num_shards=1)
        shards = sharder.split()
        
        # Should have 1 shard with all layers
        assert len(shards) == 1
        assert len(shards[0]) == 2
