"""
Direct File Model Loader for Out-of-Core Inference
Loads model tensors directly from GGUF/Safetensors files WITHOUT loading full model into RAM.
This is the TRUE out-of-core solution.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)


class DirectFileModelLoader:
    """
    Load model tensors directly from GGUF/Safetensors files.
    NEVER loads the entire model into RAM at once.
    Supports memory-mapped loading for better performance.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = "cpu",
                 use_mmap: bool = True):
        """
        Initialize direct file model loader.
        
        Args:
            model_path: Path to GGUF or Safetensors file
            device: Device to use (cpu or cuda)
            use_mmap: Use memory-mapped loading if available
        """
        self.model_path = Path(model_path)
        self.device = device
        self.use_mmap = use_mmap
        self.model_info = self._load_model_info()
        self.tensor_infos = self._extract_tensor_infos()
        self._file_handle = None
        
        logger.info(f"DirectFileModelLoader initialized for {model_path}")
        logger.info(f"  Format: {self.model_info.get('format', 'unknown')}")
        logger.info(f"  Tensors: {len(self.tensor_infos)}")
        logger.info(f"  MMAP: {'enabled' if use_mmap else 'disabled'}")
    
    def _load_model_info(self) -> Dict[str, Any]:
        """Load model metadata without loading weights"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if self.model_path.suffix == '.gguf':
            return self._load_gguf_info()
        elif self.model_path.suffix == '.safetensors':
            return self._load_safetensors_info()
        else:
            raise ValueError(f"Unsupported format: {self.model_path.suffix}")
    
    def _load_gguf_info(self) -> Dict[str, Any]:
        """Load GGUF metadata"""
        # GGUF needs: GGUFReader without a context manager, iteration over reader.tensors (a list),
        # dequantization of Q4/Q8 blocks, and mapping of names like "blk.0.attn_q.weight" to the
        # Hugging Face names used by PagedTransformerEngine. None of that is implemented yet, so
        # fail clearly instead of with an unrelated TypeError.
        raise NotImplementedError(
            "GGUF loading is not implemented yet in DirectFileModelLoader; "
            "use a .safetensors checkpoint for now.")

    def _load_safetensors_info(self) -> Dict[str, Any]:
        """Load Safetensors metadata"""
        from safetensors import safe_open
        with safe_open(self.model_path, framework="pt") as f:
            return {
                'format': 'safetensors',
                'tensors': list(f.keys())
            }
    
    def _extract_tensor_infos(self) -> Dict[str, Dict[str, Any]]:
        """Extract info about all tensors"""
        infos = {}
        
        if self.model_info['format'] == 'gguf':
            from gguf import GGUFReader
            with GGUFReader(self.model_path) as reader:
                for name, tensor in reader.tensors.items():
                    infos[name] = {
                        'shape': tuple(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'size': tensor.nbytes(),
                        'offset': getattr(tensor, 'offset', 0)
                    }
        elif self.model_info['format'] == 'safetensors':
            from safetensors import safe_open
            # Read shapes/dtypes from the header only, without materializing any tensor data
            dtype_bytes = {'F64': 8, 'I64': 8, 'U64': 8, 'F32': 4, 'I32': 4, 'U32': 4,
                           'F16': 2, 'BF16': 2, 'I16': 2, 'U16': 2,
                           'I8': 1, 'U8': 1, 'BOOL': 1, 'F8_E4M3': 1, 'F8_E5M2': 1}
            with safe_open(self.model_path, framework="pt") as f:
                for name in f.keys():
                    tensor_slice = f.get_slice(name)
                    shape = tuple(tensor_slice.get_shape())
                    dtype = tensor_slice.get_dtype()
                    numel = 1
                    for dim in shape:
                        numel *= dim
                    infos[name] = {
                        'shape': shape,
                        'dtype': dtype,
                        'size': numel * dtype_bytes.get(dtype, 4),
                        'offset': 0
                    }
        
        return infos
    
    def _gguf_to_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert GGUF dtype to PyTorch dtype"""
        mapping = {
            'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
            'I8': torch.int8, 'I16': torch.int16, 'I32': torch.int32,
            'Q4_0': torch.uint8, 'Q4_1': torch.uint8, 'Q5_0': torch.uint8,
            'Q5_1': torch.uint8, 'Q8_0': torch.uint8
        }
        return mapping.get(dtype_str, torch.float32)
    
    def load_tensor(self, tensor_name: str) -> torch.Tensor:
        """
        Load a single tensor from file WITHOUT loading entire model.
        This is the CORE of out-of-core inference.
        """
        if tensor_name not in self.tensor_infos:
            raise ValueError(f"Tensor {tensor_name} not found")
        
        if self.model_info['format'] == 'gguf':
            return self._load_gguf_tensor(tensor_name)
        elif self.model_info['format'] == 'safetensors':
            return self._load_safetensors_tensor(tensor_name)
    
    def _load_gguf_tensor(self, tensor_name: str) -> torch.Tensor:
        """Load tensor from GGUF file"""
        from gguf import GGUFReader
        with GGUFReader(self.model_path) as reader:
            tensor = reader.get_tensor(tensor_name)
            return torch.tensor(tensor, dtype=self._gguf_to_torch_dtype(tensor.dtype))
    
    def _load_safetensors_tensor(self, tensor_name: str) -> torch.Tensor:
        """Load tensor from Safetensors file with optional mmap"""
        from safetensors import safe_open
        
        if self.use_mmap:
            # Use memory mapping for better performance
            with safe_open(self.model_path, framework="pt", device=self.device) as f:
                return f.get_tensor(tensor_name)
        else:
            # Standard loading
            with safe_open(self.model_path, framework="pt") as f:
                return f.get_tensor(tensor_name)
    
    def get_tensor_info(self, tensor_name: str) -> Dict[str, Any]:
        """Get info about a tensor"""
        return self.tensor_infos.get(tensor_name, {})
    
    def list_tensors(self) -> list:
        """List all tensor names"""
        return list(self.tensor_infos.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        return self.model_info
