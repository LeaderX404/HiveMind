"""
File-Backed Pager for Out-of-Core Inference
Reads tensors directly from GGUF/Safetensors files on demand with caching.
Phase 2: Added prefetching for 10B+ models
Phase 3: Added adaptive 4000MB pages for 50B-100B models
"""
import torch
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from hivemind.utils.logger import get_logger
from hivemind.core.direct_loader import DirectFileModelLoader

logger = get_logger(__name__)


def compute_optimal_page_size_mb(total_size_mb: float, current_page_size_mb: int) -> int:
    """Return the recommended page size (MB) for a model of a given on-disk size.

    Phase 2: 5GB+ -> 1000MB pages, 20GB+ (10B params) -> 2000MB pages.
    Phase 3: 100GB+ (50B-100B params) -> 4000MB pages for smoother prefetching.
    The current page size is never lowered.
    """
    if total_size_mb > 100000:  # 100GB+ on disk (50B+ parameters in FP16)
        return max(current_page_size_mb, 4000)
    if total_size_mb > 20000:  # 20GB+ on disk (10B+ parameters)
        return max(current_page_size_mb, 2000)
    if total_size_mb > 5000:  # 5GB+ on disk (5B+ parameters)
        return max(current_page_size_mb, 1000)
    return current_page_size_mb


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
                 max_cached_pages: int = 10,
                 prefetch_pages: int = 2,
                 prefetch_enabled: bool = True):
        """
        Initialize file-backed pager.
        
        Args:
            model_path: Path to GGUF or Safetensors file
            page_size_mb: Size of each page in MB (Phase 2: optimized for 10B models)
            device: Device to use
            max_cached_pages: Maximum number of pages to cache in RAM
            prefetch_pages: Number of pages to prefetch ahead (Phase 2 feature)
            prefetch_enabled: Enable/disable prefetching (Phase 2 feature)
        """
        self.model_path = Path(model_path)
        self.page_size_mb = page_size_mb
        self.device = device
        self.max_cached_pages = max_cached_pages
        self.prefetch_pages = prefetch_pages
        self.prefetch_enabled = prefetch_enabled
        
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
        
        # Prefetch tracking (Phase 2)
        self.prefetch_lock = threading.Lock()
        self.prefetch_thread = None
        self.prefetch_queue = []
        self.prefetching = False
        
        # Optimize page size for 10B+ models
        self._optimize_page_size_for_10b()
        
        logger.info(f"FileBackedPager: {len(self.pages)} pages created, max {max_cached_pages} cached, prefetch={prefetch_pages}")
    
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
    
    def _optimize_page_size_for_10b(self):
        """Optimize page size based on model size (Phase 2, extended in Phase 3)"""
        # Estimate model size
        total_size_mb = sum(
            self.loader.get_tensor_info(tensor_name)['size'] / (1024 * 1024)
            for tensor_name in self.loader.list_tensors()
        )
        
        # Pick the best tier for this model size (Phase 3: 50B-100B supported)
        optimal_page_size_mb = compute_optimal_page_size_mb(total_size_mb, self.page_size_mb)
        if optimal_page_size_mb > self.page_size_mb:
            logger.info(
                f"Optimized page size from {self.page_size_mb}MB to {optimal_page_size_mb}MB for large model"
            )
            self.page_size_mb = optimal_page_size_mb
            # Recreate pages with new size
            self.pages = self._create_pages()
            self.tensor_to_page = self._map_tensors_to_pages()
    
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
    
    def _prefetch_page(self, page_index: int):
        """Prefetch a page in the background (Phase 2 feature)"""
        if not self.prefetch_enabled:
            return
        
        with self.prefetch_lock:
            if page_index in self.page_cache or page_index in self.prefetch_queue:
                return
            
            self.prefetch_queue.append(page_index)
            
            if not self.prefetching:
                self.prefetching = True
                self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
                self.prefetch_thread.start()
    
    def _prefetch_worker(self):
        """Background thread for prefetching pages (Phase 2 feature)"""
        while True:
            with self.prefetch_lock:
                if not self.prefetch_queue:
                    self.prefetching = False
                    break
                
                page_index = self.prefetch_queue.pop(0)
            
            # Load the page
            try:
                page_data = self._load_page(page_index)
                
                with self.prefetch_lock:
                    # Check if we still need this page
                    if len(self.page_cache) >= self.max_cached_pages:
                        self._evict_page()
                    
                    self.page_cache[page_index] = page_data
                    if page_index in self.page_cache_order:
                        self.page_cache_order.remove(page_index)
                    self.page_cache_order.append(page_index)
                    
                    logger.debug(f"Prefetched page {page_index}")
            except Exception as e:
                logger.warning(f"Prefetch failed for page {page_index}: {e}")
                continue
    
    def load_tensor(self, tensor_name: str, prefetch_next: bool = True) -> torch.Tensor:
        """Load a tensor from file on demand with caching and prefetching (Phase 2 feature)"""
        # Check tensor cache first
        if tensor_name in self.tensor_cache:
            # Prefetch next pages
            if prefetch_next and self.prefetch_enabled:
                self._prefetch_next_pages(tensor_name)
            return self.tensor_cache[tensor_name].to(self.device)
        
        # Get page index
        page_index = self.get_page_index(tensor_name)
        
        # Check page cache
        if page_index in self.page_cache:
            # Move to end of order (most recently used)
            if page_index in self.page_cache_order:
                self.page_cache_order.remove(page_index)
            self.page_cache_order.append(page_index)
            
            # Prefetch next pages
            if prefetch_next and self.prefetch_enabled:
                self._prefetch_next_pages(tensor_name)
            
            return self.page_cache[page_index][tensor_name].to(self.device)
        
        # Load the page
        page_data = self._load_page(page_index)
        
        # Cache the page
        if len(self.page_cache) >= self.max_cached_pages:
            self._evict_page()
        
        self.page_cache[page_index] = page_data
        self.page_cache_order.append(page_index)
        
        # Prefetch next pages
        if prefetch_next and self.prefetch_enabled:
            self._prefetch_next_pages(tensor_name)
        
        return page_data[tensor_name].to(self.device)
    
    def _prefetch_next_pages(self, tensor_name: str):
        """Prefetch the next N pages after the current one (Phase 2 feature)"""
        current_page = self.get_page_index(tensor_name)
        
        for offset in range(1, self.prefetch_pages + 1):
            next_page = current_page + offset
            if next_page < len(self.pages):
                self._prefetch_page(next_page)
    
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics (Phase 2: added prefetch stats)"""
        return {
            'cached_pages': len(self.page_cache),
            'max_cached_pages': self.max_cached_pages,
            'cached_tensors': len(self.tensor_cache),
            'total_pages': len(self.pages),
            'prefetch_enabled': self.prefetch_enabled,
            'prefetch_pages': self.prefetch_pages,
            'prefetch_queue_size': len(self.prefetch_queue),
            'page_size_mb': self.page_size_mb
        }
    
    def set_prefetch_enabled(self, enabled: bool):
        """Enable or disable prefetching (Phase 2 feature)"""
        self.prefetch_enabled = enabled
        logger.info(f"Prefetching {'enabled' if enabled else 'disabled'}")
    
    def set_prefetch_pages(self, num_pages: int):
        """Set number of pages to prefetch (Phase 2 feature)"""
        self.prefetch_pages = num_pages
        logger.info(f"Prefetch pages set to {num_pages}")
