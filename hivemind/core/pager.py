"""
Model Paging System for Out-of-Core Inference
Allows running large models on devices with limited RAM by loading
model layers from disk on demand.

This is the core of HiveMind v2.0 - the real solution for running
7B+ models on devices with 16GB RAM or less.

Disk pages use a pickle-free container (HMP2): a JSON manifest
describing the layers and keys, followed by concatenated safe
tensor containers (see hivemind/net/serialization.py). Pages are
only ever read as bytes; nothing stored on disk is executed.
"""
import torch
import os
import json
import struct
import hashlib
from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Any
from hivemind.net.serialization import serialize_tensor, deserialize_tensor
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)

# Pickle-free page container: 4B magic "HMP2", 4B manifest length
# (little-endian), JSON manifest, then concatenated safe tensor
# containers. Mirrors the HMT1 wire format philosophy: data on
# disk is bytes to be read, never code to be executed.
_PAGE_MAGIC = b"HMP2"
_PAGE_HEADER = struct.Struct("<4sI")


class DiskCache:
    """
    Cache for storing model pages on disk.
    Each page contains a subset of the model's layers.
    """
    
    def __init__(self, cache_dir: str = "model_cache", use_hash: bool = True):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory to store cached pages
            use_hash: Whether to use hash-based filenames
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_hash = use_hash
        self._page_hashes = {}
    
    def _get_filename(self, page_index: int, layers: List[Tuple[str, Any]]) -> str:
        """Generate filename for a page (pickle-free .hmp container)"""
        if self.use_hash:
            # Create hash of layer names for consistent filename
            layer_names = "|".join(name for name, _ in layers)
            page_hash = hashlib.md5(layer_names.encode()).hexdigest()[:8]
            return f"page_{page_index}_{page_hash}.hmp"
        else:
            return f"page_{page_index}.hmp"
    
    def save_page(self, page_index: int, layers: List[Tuple[str, Any]]) -> Path:
        """
        Save a page to disk.
        
        Args:
            page_index: Index of the page
            layers: List of (layer_name, layer) tuples
            
        Returns:
            Path to the saved page file
        """
        filename = self._get_filename(page_index, layers)
        page_path = self.cache_dir / filename
        
        # Build a pickle-free page container: a JSON manifest
        # describing every layer and key, followed by concatenated
        # safe tensor containers. Nothing on disk is ever executed.
        meta = {"version": 2, "entries": []}
        blob = bytearray()
        for name, layer in layers:
            entry = {
                "name": name,
                "class": layer.__class__.__name__,
                "module": layer.__class__.__module__,
                "keys": [],
            }
            for key, value in layer.state_dict().items():
                wire = serialize_tensor(value.cpu())
                entry["keys"].append({"key": key, "bytes": len(wire)})
                blob.extend(wire)
            meta["entries"].append(entry)
        
        meta_bytes = json.dumps(meta).encode("utf-8")
        with open(page_path, "wb") as f:
            f.write(_PAGE_HEADER.pack(_PAGE_MAGIC, len(meta_bytes)))
            f.write(meta_bytes)
            f.write(bytes(blob))
        
        logger.debug(f"Saved page {page_index} to {page_path}")
        return page_path
    
    def load_page(self, page_index: int, layers: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """
        Load a page from disk and reconstruct layers.
        
        Args:
            page_index: Index of the page
            layers: Original layer references (for reconstruction)
            
        Returns:
            List of (layer_name, layer) tuples with loaded parameters
        """
        filename = self._get_filename(page_index, layers)
        page_path = self.cache_dir / filename
        
        if not page_path.exists():
            raise FileNotFoundError(f"Page {page_index} not found at {page_path}")
        
        with open(page_path, "rb") as f:
            raw = f.read()
        
        # Parse the pickle-free page container. Any malformed page
        # is rejected with a ValueError: page bytes are read, never
        # executed (legacy pickle pages fail the magic check).
        if len(raw) < _PAGE_HEADER.size:
            raise ValueError("page rejected: too short to be a page container")
        magic, meta_len = _PAGE_HEADER.unpack_from(raw, 0)
        if magic != _PAGE_MAGIC:
            raise ValueError(
                "page rejected: not a HiveMind page container; "
                "legacy pickle pages are no longer accepted"
            )
        manifest_start = _PAGE_HEADER.size
        manifest_end = manifest_start + meta_len
        if manifest_end > len(raw):
            raise ValueError("page rejected: truncated manifest")
        try:
            meta = json.loads(raw[manifest_start:manifest_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"page rejected: malformed manifest: {e}")
        
        offset = manifest_end
        
        # Reconstruct layers with loaded parameters
        loaded_layers = []
        for entry in meta["entries"]:
            layer_name = entry["name"]
            state_dict = {}
            for key_meta in entry["keys"]:
                size = int(key_meta["bytes"])
                if size < 0 or offset + size > len(raw):
                    raise ValueError("page rejected: truncated tensor data")
                state_dict[key_meta["key"]] = deserialize_tensor(
                    raw[offset:offset + size]
                )
                offset += size
            
            # Find the corresponding layer in the original model
            for name, layer in layers:
                if name == layer_name:
                    # Load state dict
                    layer.load_state_dict(state_dict)
                    loaded_layers.append((name, layer))
                    break
        
        logger.debug(f"Loaded page {page_index} from {page_path}")
        return loaded_layers
    
    def clear(self):
        """Clear all cached pages (current .hmp and legacy .pkl files)"""
        count = 0
        for pattern in ("page_*.hmp", "page_*.pkl"):
            for page_file in self.cache_dir.glob(pattern):
                page_file.unlink()
                count += 1
        logger.info(f"Cleared {count} cached pages")
    
    def get_cache_size(self) -> float:
        """Get total cache size in MB"""
        total_size = 0
        for pattern in ("page_*.hmp", "page_*.pkl"):
            for page_file in self.cache_dir.glob(pattern):
                total_size += page_file.stat().st_size
        return total_size / (1024 * 1024)


class LRUCache:
    """
    LRU (Least Recently Used) Cache for managing loaded pages in RAM.
    Automatically evicts least recently used pages when capacity is reached.
    """
    
    def __init__(self, max_pages: int):
        """
        Initialize LRU cache.
        
        Args:
            max_pages: Maximum number of pages to keep in RAM
        """
        self.max_pages = max_pages
        self.cache: OrderedDict[int, List[Tuple[str, Any]]] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, page_index: int) -> Optional[List[Tuple[str, Any]]]:
        """
        Get a page from cache.
        
        Args:
            page_index: Index of the page
            
        Returns:
            Page data if found, None otherwise
        """
        if page_index in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(page_index)
            self.hits += 1
            return self.cache[page_index]
        self.misses += 1
        return None
    
    def put(self, page_index: int, page_data: List[Tuple[str, Any]]):
        """
        Add a page to cache.
        
        Args:
            page_index: Index of the page
            page_data: Page data to cache
        """
        if page_index in self.cache:
            # Update existing
            self.cache.move_to_end(page_index)
        else:
            # Add new
            self.cache[page_index] = page_data
        
        # Remove oldest if over capacity
        while len(self.cache) > self.max_pages:
            oldest_index = next(iter(self.cache))
            self.cache.pop(oldest_index)
            logger.debug(f"Evicted page {oldest_index} from RAM cache")
    
    def remove(self, page_index: int):
        """Remove a page from cache"""
        if page_index in self.cache:
            self.cache.pop(page_index)
    
    def clear(self):
        """Clear all pages from cache"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_pages,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }


class MemoryMonitor:
    """
    Monitor memory usage to help with paging decisions.
    """
    
    def __init__(self, max_memory_mb: Optional[float] = None):
        """
        Initialize memory monitor.
        
        Args:
            max_memory_mb: Maximum memory to monitor (None = auto-detect)
        """
        self.max_memory_mb = max_memory_mb
        self._detect_memory()
    
    def _detect_memory(self):
        """Detect available memory"""
        try:
            import psutil
            if self.max_memory_mb is None:
                self.max_memory_mb = psutil.virtual_memory().available / (1024 * 1024) * 0.8  # Use 80% of available
                logger.info(f"Detected available memory: {self.max_memory_mb:.0f}MB")
        except ImportError:
            logger.warning("psutil not installed, using default memory limit")
            self.max_memory_mb = 16000  # Default to 16GB
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        try:
            import psutil
            import torch
            
            process = psutil.Process(os.getpid())
            ram_usage = process.memory_info().rss / (1024 * 1024)
            
            gpu_usage = 0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / (1024 * 1024)
            
            return {
                'ram_mb': ram_usage,
                'gpu_mb': gpu_usage,
                'total_mb': ram_usage + gpu_usage,
                'available_mb': self.max_memory_mb - (ram_usage + gpu_usage)
            }
        except ImportError:
            return {'ram_mb': 0, 'gpu_mb': 0, 'total_mb': 0, 'available_mb': self.max_memory_mb}
    
    def can_load_page(self, page_size_mb: float) -> bool:
        """Check if a page can be loaded based on available memory"""
        usage = self.get_memory_usage()
        return usage['available_mb'] >= page_size_mb


class ModelPager:
    """
    Splits a model into pages for out-of-core inference.
    Each page contains a subset of the model's layers.
    """
    
    def __init__(self, model: torch.nn.Module, page_size_mb: float = 1000):
        """
        Initialize model pager.
        
        Args:
            model: The model to page
            page_size_mb: Maximum size of each page in MB
        """
        self.model = model
        self.page_size_mb = page_size_mb
        self.pages: List[List[Tuple[str, torch.nn.Module]]] = []
        self.layer_to_page: Dict[str, int] = {}
        self._split_model()
    
    def _get_layer_size(self, layer: torch.nn.Module) -> float:
        """Calculate size of a layer in MB"""
        size = 0
        for param in layer.parameters():
            size += param.numel() * param.element_size()
        return size / (1024 * 1024)  # Convert to MB
    
    def _split_model(self):
        """Split model into pages based on layer sizes"""
        current_page: List[Tuple[str, torch.nn.Module]] = []
        current_size = 0.0
        page_index = 0
        
        for name, layer in self.model.named_modules():
            # Only consider trainable layers (Linear, Conv, etc.)
            if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
                layer_size = self._get_layer_size(layer)
                
                # If adding this layer exceeds page size, start new page
                if current_page and current_size + layer_size > self.page_size_mb:
                    self.pages.append(current_page)
                    for layer_name, _ in current_page:
                        self.layer_to_page[layer_name] = page_index
                    page_index += 1
                    current_page = []
                    current_size = 0.0
                
                current_page.append((name, layer))
                current_size += layer_size
        
        # Add the last page
        if current_page:
            self.pages.append(current_page)
            for layer_name, _ in current_page:
                self.layer_to_page[layer_name] = page_index
        
        logger.info(f"Split model into {len(self.pages)} pages with {self.page_size_mb}MB per page")
        for i, page in enumerate(self.pages):
            page_size = sum(self._get_layer_size(layer) for _, layer in page)
            logger.debug(f"  Page {i}: {len(page)} layers, {page_size:.1f}MB")
    
    def get_page_index(self, layer_name: str) -> int:
        """Get page index for a given layer"""
        return self.layer_to_page.get(layer_name, 0)
    
    def get_page(self, page_index: int) -> List[Tuple[str, torch.nn.Module]]:
        """Get layers for a given page"""
        if 0 <= page_index < len(self.pages):
            return self.pages[page_index]
        raise IndexError(f"Page index {page_index} out of range (0-{len(self.pages)-1})")
    
    def get_page_count(self) -> int:
        """Get total number of pages"""
        return len(self.pages)
    
    def get_total_size(self) -> float:
        """Get total model size in MB"""
        total_size = 0.0
        for page in self.pages:
            for _, layer in page:
                total_size += self._get_layer_size(layer)
        return total_size


class PagedModel:
    """
    Model wrapper that loads pages from disk on demand.
    This is the main class for out-of-core inference.
    """
    
    def __init__(self, model: torch.nn.Module, 
                 page_size_mb: float = 1000, 
                 max_loaded_pages: int = 4,
                 device: str = "cpu"):
        """
        Initialize paged model.
        
        Args:
            model: The model to page
            page_size_mb: Size of each page in MB
            max_loaded_pages: Maximum pages to keep in RAM
            device: Device to run on (cpu or cuda)
        """
        self.original_model = model
        self.device = device
        self.pager = ModelPager(model, page_size_mb)
        self.disk_cache = DiskCache()
        self.ram_cache = LRUCache(max_loaded_pages)
        self.memory_monitor = MemoryMonitor()
        
        # Save all pages to disk
        self._save_all_pages()
        
        logger.info(f"PagedModel initialized: {self.pager.get_page_count()} pages, "
                   f"max {max_loaded_pages} in RAM")
    
    def _save_all_pages(self):
        """Save all model pages to disk"""
        for page_index, page in enumerate(self.pager.pages):
            self.disk_cache.save_page(page_index, page)
        
        total_size = self.disk_cache.get_cache_size()
        logger.info(f"Saved all pages to disk: {total_size:.1f}MB")
    
    def _load_page_to_ram(self, page_index: int):
        """Load a page from disk to RAM"""
        # Check if already in RAM
        if self.ram_cache.get(page_index) is not None:
            return
        
        # Check memory before loading
        page = self.pager.get_page(page_index)
        page_size = sum(self.pager._get_layer_size(layer) for _, layer in page)
        
        if not self.memory_monitor.can_load_page(page_size):
            # Free some pages to make room
            self._free_pages_for_space(page_size)
        
        # Load from disk
        loaded_page = self.disk_cache.load_page(page_index, page)
        self.ram_cache.put(page_index, loaded_page)
        logger.debug(f"Loaded page {page_index} to RAM ({page_size:.1f}MB)")
    
    def _free_pages_for_space(self, required_size_mb: float):
        """Free pages until we have enough space"""
        usage = self.memory_monitor.get_memory_usage()
        needed = required_size_mb - usage['available_mb']
        
        if needed <= 0:
            return
        
        # Free pages until we have enough space
        while needed > 0 and len(self.ram_cache.cache) > 0:
            # Get least recently used page
            oldest_index = next(iter(self.ram_cache.cache))
            oldest_page = self.ram_cache.cache[oldest_index]
            oldest_size = sum(self.pager._get_layer_size(layer) for _, layer in oldest_page)
            
            # Free this page
            self.ram_cache.remove(oldest_index)
            needed -= oldest_size
            logger.debug(f"Freed page {oldest_index} ({oldest_size:.1f}MB")
    
    def get_layer(self, layer_name: str) -> torch.nn.Module:
        """
        Get a layer, loading its page if necessary.
        
        Args:
            layer_name: Name of the layer to get
            
        Returns:
            The layer with loaded parameters
        """
        page_index = self.pager.get_page_index(layer_name)
        self._load_page_to_ram(page_index)
        
        # Find and return the layer
        page = self.ram_cache.get(page_index)
        for name, layer in page:
            if name == layer_name:
                return layer.to(self.device)
        
        raise ValueError(f"Layer {layer_name} not found in page {page_index}")
    
    def get_all_loaded_layers(self) -> Dict[str, torch.nn.Module]:
        """Get all currently loaded layers"""
        loaded_layers = {}
        for page_index in self.ram_cache.cache:
            page = self.ram_cache.cache[page_index]
            for name, layer in page:
                loaded_layers[name] = layer
        return loaded_layers
    
    def clear_cache(self):
        """Clear both disk and RAM cache"""
        self.disk_cache.clear()
        self.ram_cache.clear()
        logger.info("Cleared all caches")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get paging statistics"""
        return {
            'total_pages': self.pager.get_page_count(),
            'loaded_pages': len(self.ram_cache.cache),
            'cache_stats': self.ram_cache.get_stats(),
            'disk_cache_size_mb': self.disk_cache.get_cache_size(),
            'memory_usage': self.memory_monitor.get_memory_usage()
        }
