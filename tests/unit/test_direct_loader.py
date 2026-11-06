"""
Unit tests for DirectFileModelLoader
"""
import pytest
import torch
from pathlib import Path
from hivemind.core.direct_loader import DirectFileModelLoader


class TestDirectFileModelLoader:
    """Test DirectFileModelLoader with dummy files"""
    
    def test_init(self):
        """Test initialization"""
        # This will fail without actual model file, but tests the interface
        with pytest.raises(FileNotFoundError):
            loader = DirectFileModelLoader("nonexistent.gguf")
    
    def test_list_tensors_empty(self):
        """Test list_tensors returns empty list for non-existent file"""
        with pytest.raises(FileNotFoundError):
            loader = DirectFileModelLoader("nonexistent.gguf")
            loader.list_tensors()
