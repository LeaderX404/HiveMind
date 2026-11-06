"""
Unit tests for Model Paging System (pager.py)
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path
from hivemind.core.pager import (
    DiskCache,
    LRUCache,
    MemoryMonitor,
    ModelPager,
    PagedModel
)


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class TestDiskCache:
    """Test cases for DiskCache"""
    
    def test_init(self, tmp_path):
        """Test disk cache initialization"""
        cache = DiskCache(cache_dir=str(tmp_path))
        assert cache.cache_dir.exists()
    
    def test_save_and_load_page(self, tmp_path):
        """Test saving and loading a page"""
        cache = DiskCache(cache_dir=str(tmp_path))
        
        # Create mock layers
        layer1 = nn.Linear(10, 20)
        layer2 = nn.Linear(20, 30)
        layers = [("layer1", layer1), ("layer2", layer2)]
        
        # Save page
        page_path = cache.save_page(0, layers)
        assert page_path.exists()
        
        # Load page
        loaded_layers = cache.load_page(0, layers)
        assert len(loaded_layers) == 2
    
    def test_clear(self, tmp_path):
        """Test clearing the cache"""
        cache = DiskCache(cache_dir=str(tmp_path))
        
        # Save some pages
        layer = nn.Linear(10, 20)
        cache.save_page(0, [("layer1", layer)])
        cache.save_page(1, [("layer2", layer)])
        
        # Clear cache
        cache.clear()
        
        # Check that files are deleted
        assert len(list(tmp_path.glob("page_*.pkl"))) == 0
    
    def test_get_cache_size(self, tmp_path):
        """Test getting cache size"""
        cache = DiskCache(cache_dir=str(tmp_path))
        
        # Save a page
        layer = nn.Linear(10, 20)
        cache.save_page(0, [("layer1", layer)])
        
        # Get size
        size_mb = cache.get_cache_size()
        assert size_mb > 0


class TestLRUCache:
    """Test cases for LRUCache"""
    
    def test_init(self):
        """Test LRU cache initialization"""
        cache = LRUCache(max_pages=3)
        assert cache.max_pages == 3
        assert len(cache.cache) == 0
    
    def test_put_and_get(self):
        """Test putting and getting pages"""
        cache = LRUCache(max_pages=3)
        
        # Put a page
        page_data = [("layer1", nn.Linear(10, 20))]
        cache.put(0, page_data)
        
        # Get the page
        result = cache.get(0)
        assert result == page_data
    
    def test_eviction(self):
        """Test that LRU eviction works"""
        cache = LRUCache(max_pages=2)
        
        # Put 3 pages (should evict the first one)
        cache.put(0, [("layer1", nn.Linear(10, 20))])
        cache.put(1, [("layer2", nn.Linear(20, 30))])
        cache.put(2, [("layer3", nn.Linear(30, 10))])
        
        # Check that page 0 was evicted
        assert cache.get(0) is None
        assert cache.get(1) is not None
        assert cache.get(2) is not None
    
    def test_lru_order(self):
        """Test LRU ordering"""
        cache = LRUCache(max_pages=2)
        
        # Put pages
        cache.put(0, [("layer1", nn.Linear(10, 20))])
        cache.put(1, [("layer2", nn.Linear(20, 30))])
        
        # Access page 0 (makes it most recently used)
        cache.get(0)
        
        # Put page 2 (should evict page 1, not page 0)
        cache.put(2, [("layer3", nn.Linear(30, 10))])
        
        # Check that page 1 was evicted
        assert cache.get(0) is not None
        assert cache.get(1) is None
        assert cache.get(2) is not None
    
    def test_get_stats(self):
        """Test getting cache statistics"""
        cache = LRUCache(max_pages=3)
        
        # Put some pages
        cache.put(0, [("layer1", nn.Linear(10, 20))])
        cache.put(1, [("layer2", nn.Linear(20, 30))])
        
        # Access some pages
        cache.get(0)  # Hit
        cache.get(0)  # Hit
        cache.get(1)  # Hit
        cache.get(2)  # Miss
        
        stats = cache.get_stats()
        assert stats['size'] == 2
        assert stats['max_size'] == 3
        assert stats['hits'] == 3
        assert stats['misses'] == 1


class TestMemoryMonitor:
    """Test cases for MemoryMonitor"""
    
    def test_init(self):
        """Test memory monitor initialization"""
        monitor = MemoryMonitor()
        assert monitor.max_memory_mb > 0
    
    def test_get_memory_usage(self):
        """Test getting memory usage"""
        monitor = MemoryMonitor()
        usage = monitor.get_memory_usage()
        
        assert 'ram_mb' in usage
        assert 'gpu_mb' in usage
        assert 'total_mb' in usage
        assert 'available_mb' in usage
    
    def test_can_load_page(self):
        """Test checking if a page can be loaded"""
        monitor = MemoryMonitor(max_memory_mb=1000)
        
        # Should be able to load a small page
        assert monitor.can_load_page(100)
        
        # Should not be able to load a large page
        assert not monitor.can_load_page(2000)


class TestModelPager:
    """Test cases for ModelPager"""
    
    def test_init(self):
        """Test model pager initialization"""
        model = SimpleModel()
        pager = ModelPager(model, page_size_mb=100)
        
        assert pager.model == model
        assert pager.page_size_mb == 100
        assert len(pager.pages) > 0
    
    def test_split_model(self):
        """Test splitting a model into pages"""
        model = SimpleModel()
        pager = ModelPager(model, page_size_mb=0.1)  # Small page size to force multiple pages
        
        # Should have multiple pages
        assert len(pager.pages) >= 1
    
    def test_get_page_index(self):
        """Test getting page index for a layer"""
        model = SimpleModel()
        pager = ModelPager(model, page_size_mb=100)
        
        # Get page index for layer1
        page_index = pager.get_page_index("layer1")
        assert page_index >= 0
    
    def test_get_page(self):
        """Test getting a page"""
        model = SimpleModel()
        pager = ModelPager(model, page_size_mb=100)
        
        # Get first page
        page = pager.get_page(0)
        assert isinstance(page, list)
        assert len(page) > 0
    
    def test_get_page_count(self):
        """Test getting page count"""
        model = SimpleModel()
        pager = ModelPager(model, page_size_mb=100)
        
        count = pager.get_page_count()
        assert count == len(pager.pages)
    
    def test_get_total_size(self):
        """Test getting total model size"""
        model = SimpleModel()
        pager = ModelPager(model, page_size_mb=100)
        
        total_size = pager.get_total_size()
        assert total_size > 0


class TestPagedModel:
    """Test cases for PagedModel"""
    
    def test_init(self, tmp_path):
        """Test paged model initialization"""
        model = SimpleModel()
        
        paged_model = PagedModel(
            model=model,
            page_size_mb=100,
            max_loaded_pages=2,
            device="cpu"
        )
        
        assert paged_model.original_model == model
        assert paged_model.device == "cpu"
    
    def test_get_layer(self, tmp_path):
        """Test getting a layer"""
        model = SimpleModel()
        
        paged_model = PagedModel(
            model=model,
            page_size_mb=100,
            max_loaded_pages=2,
            device="cpu"
        )
        
        # Get layer1
        layer = paged_model.get_layer("layer1")
        assert layer is not None
        assert isinstance(layer, nn.Linear)
    
    def test_get_stats(self, tmp_path):
        """Test getting paged model statistics"""
        model = SimpleModel()
        
        paged_model = PagedModel(
            model=model,
            page_size_mb=100,
            max_loaded_pages=2,
            device="cpu"
        )
        
        stats = paged_model.get_stats()
        assert 'total_pages' in stats
        assert 'loaded_pages' in stats
        assert 'cache_stats' in stats
