"""
File-Backed Pager for Out-of-Core Inference
Reads tensors directly from GGUF/Safetensors files on demand with caching.
"""
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from hivemind.utils.logger import get_logger
from hivemind.core.direct_loader import DirectFileModelLoader

logger = get_logger(__name__)


class FileBackedPager:
    """
    Pager that reads tensors directly from file on demand.
    NEVER loads the entire model into RAM.
    Uses LRU caching for loaded tensors.
    """
    
    def __init__(self, 
                 model_path: str,
                 page_size_mb: int = 1000,
                 device: str = "cpu",
                 max_cached_pages: int = 10):
        """
        Initialize file-backed pager.
        
        Args:
            model_path: Path to GGUF or Safetensors file
            page_size_mb: Size of each page in MB
            device: Device to use
            max_cached_pages: Maximum number of pages to cache in RAM
        """
        self.model_path = Path(model_path)
        self.page_size_mb = page_size_mb
        self.device = device
        self.max_cached_pages = max_cached_pages
        
        # Initialize loader
        self.loader = DirectFileModelLoader(model_path, device)
        
        # Create pages
        self.pages = self._create_pages()
        self.tensor_to_page = self._map_tensors_to_pages()
        
        # Page cache: stores loaded pages
        self.page_cache = {}
        self.page_cache_order = []
        
        # Tensor cache: stores individual tensors
        self.tensor_cache = {}
        
        logger.info(f"FileBackedPager: {len(self.pages)} pages created, max {max_cached_pages} cached")
    
    def _create_pages(self) -> List[List[str]]:
        """Create pages based on tensor sizes"""
        pages = []
        current_page = []
        current_size = 0.0
        
        for tensor_name in self.loader.list_tensors():
            info = self.loader.get_tensor_info(tensor_name)
            tensor_size_mb = info['size'] / (1024 * 1024)
            
            if current_page and current_size + tensor_size_mb > self.page_size_mb:
                pages.append(current_page)
                current_page = []
                current_size = 0.0
            
            current_page.append(tensor_name)
            current_size += tensor_size_mb
        
        if current_page:
            pages.append(current_page)
        
        return pages
    
    def _map_tensors_to_pages(self) -> Dict[str, int]:
        """Map tensor names to page indices"""
        mapping = {}
        for page_index, page in enumerate(self.pages):
            for tensor_name in page:
                mapping[tensor_name] = page_index
        return mapping
    
    def get_page_index(self, tensor_name: str) -> int:
        """Get page index for a tensor"""
        return self.tensor_to_page.get(tensor_name, 0)
    
    def get_page_tensors(self, page_index: int) -> List[str]:
        """Get tensor names for a page"""
        return self.pages[page_index]
    
    def get_page_count(self) -> int:
        """Get total number of pages"""
        return len(self.pages)
    
    def _load_page(self, page_index: int) -> Dict[str, torch.Tensor]:
        """Load all tensors in a page"""
        page_tensors = self.pages[page_index]
        page_data = {}
        
        for tensor_name in page_tensors:
            page_data[tensor_name] = self.loader.load_tensor(tensor_name)
        
        return page_data
    
    def _evict_page(self):
        """Evict oldest page from cache"""
        if len(self.page_cache_order) > 0:
            oldest_page = self.page_cache_order.pop(0)
            if oldest_page in self.page_cache:
                del self.page_cache[oldest_page]
    
    def load_tensor(self, tensor_name: str) -> torch.Tensor:
        """Load a tensor from file on demand with caching"""
        # Check tensor cache first
        if tensor_name in self.tensor_cache:
            return self.tensor_cache[tensor_name].to(self.device)
        
        # Get page index
        page_index = self.get_page_index(tensor_name)
        
        # Check page cache
        if page_index in self.page_cache:
            # Move to end of order (most recently used)
            if page_index in self.page_cache_order:
                self.page_cache_order.remove(page_index)
            self.page_cache_order.append(page_index)
            
            return self.page_cache[page_index][tensor_name].to(self.device)
        
        # Load the page
        page_data = self._load_page(page_index)
        
        # Cache the page
        if len(self.page_cache) >= self.max_cached_pages:
            self._evict_page()
        
        self.page_cache[page_index] = page_data
        self.page_cache_order.append(page_index)
        
        return page_data[tensor_name].to(self.device)
    
    def get_tensor_size(self, tensor_name: str) -> float:
        """Get tensor size in MB"""
        info = self.loader.get_tensor_info(tensor_name)
        return info['size'] / (1024 * 1024)
    
    def clear_cache(self):
        """Clear all cached pages and tensors"""
        self.page_cache.clear()
        self.page_cache_order.clear()
        self.tensor_cache.clear()
        logger.info("FileBackedPager cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cached_pages': len(self.page_cache),
            'max_cached_pages': self.max_cached_pages,
            'cached_tensors': len(self.tensor_cache),
            'total_pages': len(self.pages)
        }
