"""
Unit tests for FileBackedPager
"""
import pytest
import torch
from pathlib import Path
from hivemind.core.file_pager import FileBackedPager


class TestFileBackedPager:
    """Test FileBackedPager with dummy files"""
    
    def test_init(self):
        """Test initialization"""
        with pytest.raises(FileNotFoundError):
            pager = FileBackedPager("nonexistent.gguf")
    
    def test_cache_stats(self):
        """Test cache statistics"""
        with pytest.raises(FileNotFoundError):
            pager = FileBackedPager("nonexistent.gguf")
            pager.get_cache_stats()
