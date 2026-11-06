"""
Model Loader for HiveMind
Supports loading models with optional quantization and paging.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from pathlib import Path
from typing import Optional, Tuple, Any
from hivemind.utils.logger import get_logger
from hivemind.core.quantizer import AdvancedQuantizer, QuantizationConfig
from hivemind.core.pager import PagedModel

logger = get_logger(__name__)


class ModelLoader:
    """
    Load models from HuggingFace with optional quantization and paging.
    
    Supports:
    - Standard loading (float16/float32)
    - Quantization (2-bit, 3-bit, 4-bit, 8-bit)
    - Model paging for out-of-core inference
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/phi-2", 
                 quantize: bool = True, 
                 bits: int = 4, 
                 quant_method: str = "bitsandbytes",
                 enable_paging: bool = False,
                 page_size_mb: int = 1000,
                 max_loaded_pages: int = 4,
                 device: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            model_name: Name of the model to load (e.g., "microsoft/phi-2")
            quantize: Whether to quantize the model
            bits: Number of bits for quantization (2, 3, 4, 8)
            quant_method: Quantization method ("bitsandbytes", "qlora", "gptq")
            enable_paging: Whether to enable model paging
            page_size_mb: Size of each page in MB (for paging)
            max_loaded_pages: Maximum pages to keep in RAM (for paging)
            device: Device to use ("cuda", "cpu", or None for auto-detect)
        """
        self.model_name = model_name
        self.quantize = quantize
        self.bits = bits
        self.quant_method = quant_method
        self.enable_paging = enable_paging
        self.page_size_mb = page_size_mb
        self.max_loaded_pages = max_loaded_pages
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        self.paged_model = None
        
        logger.info(f"ModelLoader initialized for {model_name}")
        if quantize:
            logger.info(f"  Quantization: {bits}-bit using {quant_method}")
        if enable_paging:
            logger.info(f"  Paging: enabled with {page_size_mb}MB pages, max {max_loaded_pages} in RAM")
    
    def load(self) -> Tuple[Any, Any]:
        """
        Load the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(f"Tokenizer loaded: {self.model_name}")
        
        # Load model
        if self.quantize:
            self.model = self._load_quantized()
        else:
            self.model = self._load_standard()
        
        # Apply paging if enabled
        if self.enable_paging:
            self._apply_paging()
        
        logger.info(f"Model loaded on {self.device}")
        if self.quantize:
            logger.info(f"  Quantization: {self.bits}-bit")
        if self.enable_paging:
            logger.info(f"  Paging: {self.paged_model.pager.get_page_count()} pages")
        
        return self.model, self.tokenizer
    
    def _load_standard(self) -> torch.nn.Module:
        """Load model without quantization"""
        logger.info("Loading model without quantization")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        )
        
        # Move to device
        if "cuda" in self.device:
            model = model.to(self.device)
        
        return model
    
    def _load_quantized(self) -> torch.nn.Module:
        """Load model with quantization"""
        if self.bits in [2, 3]:
            # Use advanced quantization for 2-bit and 3-bit
            return self._load_advanced_quantized()
        else:
            # Use bitsandbytes for 4-bit and 8-bit
            return self._load_bitsandbytes_quantized()
    
    def _load_advanced_quantized(self) -> torch.nn.Module:
        """Load model with advanced quantization (2-bit, 3-bit)"""
        logger.info(f"Loading model with {self.bits}-bit {self.quant_method} quantization")
        
        try:
            config = QuantizationConfig(bits=self.bits, method=self.quant_method)
            quantizer = AdvancedQuantizer(config)
            
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            )
            
            # Apply quantization
            model = quantizer.quantize(base_model)
            
            # Move to device
            if "cuda" in self.device:
                model = model.to(self.device)
            
            return model
            
        except ImportError as e:
            logger.error(f"Required package not installed for {self.quant_method}: {e}")
            logger.info("Falling back to standard loading")
            return self._load_standard()
    
    def _load_bitsandbytes_quantized(self) -> torch.nn.Module:
        """Load model with bitsandbytes quantization (4-bit, 8-bit)"""
        logger.info(f"Loading model with {self.bits}-bit bitsandbytes quantization")
        
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.bits == 4,
                load_in_8bit=self.bits == 8,
                bnb_4bit_quant_type="nf4" if self.bits == 4 else None,
                bnb_4bit_compute_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                bnb_4bit_use_double_quant=True if self.bits == 4 else False,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                quantization_config=quantization_config,
                device_map="auto" if "cuda" in self.device else None,
            )
            
            return model
            
        except ImportError as e:
            logger.error(f"bitsandbytes not installed: {e}")
            logger.info("Falling back to standard loading")
            return self._load_standard()
    
    def _apply_paging(self):
        """Apply paging to the loaded model"""
        logger.info("Applying model paging")
        
        if self.model is None:
            raise ValueError("Model must be loaded before applying paging")
        
        self.paged_model = PagedModel(
            model=self.model,
            page_size_mb=self.page_size_mb,
            max_loaded_pages=self.max_loaded_pages,
            device=self.device
        )
        
        # Update model reference
        self.model = self.paged_model
        
        logger.info(f"Paging applied: {self.paged_model.pager.get_page_count()} pages created")
    
    def save_shard(self, shard_index: int, layers: list, output_dir: str = "shards") -> Path:
        """
        Save a shard (for distributed inference - v1.0 compatibility).
        
        Args:
            shard_index: Index of the shard
            layers: List of layers to save
            output_dir: Output directory
            
        Returns:
            Path to the saved shard file
        """
        os.makedirs(output_dir, exist_ok=True)
        shard_path = Path(output_dir) / f"shard_{shard_index}.pt"
        torch.save({"layers": layers, "config": self.model.config}, shard_path)
        logger.info(f"Saved shard {shard_index} to {shard_path}")
        return shard_path
    
    def get_model(self) -> Any:
        """Get the loaded model"""
        return self.model
    
    def get_tokenizer(self) -> Any:
        """Get the loaded tokenizer"""
        return self.tokenizer
    
    def get_paged_model(self) -> Optional[PagedModel]:
        """Get the paged model (if paging is enabled)"""
        return self.paged_model
    
    def get_stats(self) -> dict:
        """Get loading statistics"""
        stats = {
            'model_name': self.model_name,
            'device': self.device,
            'quantize': self.quantize,
            'bits': self.bits,
            'quant_method': self.quant_method,
            'enable_paging': self.enable_paging,
        }
        
        if self.enable_paging and self.paged_model:
            stats['paging_stats'] = self.paged_model.get_stats()
        
        return stats
