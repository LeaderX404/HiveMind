"""
File-Backed Pager for Out-of-Core Inference
Reads tensors directly from GGUF/Safetensors files on demand.
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
    """
    
    def __init__(self, 
                 model_path: str,
                 page_size_mb: int = 1000,
                 device: str = "cpu"):
        """
        Initialize file-backed pager.
        
        Args:
            model_path: Path to GGUF or Safetensors file
            page_size_mb: Size of each page in MB
            device: Device to use
        """
        self.model_path = Path(model_path)
        self.page_size_mb = page_size_mb
        self.device = device
        
        # Initialize loader
        self.loader = DirectFileModelLoader(model_path, device)
        
        # Create pages
        self.pages = self._create_pages()
        self.tensor_to_page = self._map_tensors_to_pages()
        
        logger.info(f"FileBackedPager: {len(self.pages)} pages created")
    
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
    
    def load_tensor(self, tensor_name: str) -> torch.Tensor:
        """Load a tensor from file on demand"""
        return self.loader.load_tensor(tensor_name).to(self.device)
    
    def get_tensor_size(self, tensor_name: str) -> float:
        """Get tensor size in MB"""
        info = self.loader.get_tensor_info(tensor_name)
        return info['size'] / (1024 * 1024)
