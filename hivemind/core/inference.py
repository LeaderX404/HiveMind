"""
Inference Engines for HiveMind
Supports:
- LocalInferenceEngine: Standard inference
- ShardedInferenceEngine: Distributed inference (v1.0)
- PagedInferenceEngine: Out-of-core inference (v2.0) ✨ NEW
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)


class LocalInferenceEngine:
    """
    Standard inference engine for running models in memory.
    """
    
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cpu"):
        """
        Initialize local inference engine.
        
        Args:
            model: The model to use for inference
            tokenizer: The tokenizer
            device: Device to run on (cpu or cuda)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to device
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class ShardedInferenceEngine:
    """
    Inference engine for distributed inference across multiple shards.
    (HiveMind v1.0 - Swarm Inference)
    """
    
    def __init__(self, shards: List[List[Tuple[str, nn.Module]]], 
                 tokenizer: Any, device: str = "cpu"):
        """
        Initialize sharded inference engine.
        
        Args:
            shards: List of shards, where each shard is a list of (name, layer) tuples
            tokenizer: The tokenizer
            device: Device to run on (cpu or cuda)
        """
        self.shards = shards
        self.tokenizer = tokenizer
        self.device = device
    
    def forward_shard(self, shard_index: int, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through a single shard.
        
        Args:
            shard_index: Index of the shard to use
            input_tensor: Input tensor
            
        Returns:
            Output tensor
        """
        shard_layers = self.shards[shard_index]
        x = input_tensor
        for name, layer in shard_layers:
            x = layer(x)
        return x
    
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate text from a prompt using sharded model.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Get input_ids and move to device
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"].to(self.device)
        else:
            input_ids = inputs.to(self.device)
        
        # Process through all shards
        x = input_ids
        for i in range(len(self.shards)):
            x = self.forward_shard(i, x)
        
        # Get output tokens
        outputs = torch.argmax(x, dim=-1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class PagedInferenceEngine:
    """
    Inference engine for out-of-core inference using model paging.
    (HiveMind v2.0 - Model Paging)
    
    This engine loads model layers from disk on demand, allowing
    large models to run on devices with limited RAM.
    """
    
    def __init__(self, paged_model: Any, tokenizer: Any, 
                 device: str = "cpu", 
                 max_loaded_pages: int = 4):
        """
        Initialize paged inference engine.
        
        Args:
            paged_model: A PagedModel instance
            tokenizer: The tokenizer
            device: Device to run on (cpu or cuda)
            max_loaded_pages: Maximum pages to keep in RAM
        """
        self.paged_model = paged_model
        self.tokenizer = tokenizer
        self.device = device
        self.paged_model.device = device
        self.paged_model.ram_cache.max_pages = max_loaded_pages
        
        # Store original model for reference
        self.original_model = paged_model.original_model
        
        logger.info(f"PagedInferenceEngine initialized with {max_loaded_pages} max loaded pages")
    
    def _get_transformer_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Get all transformer layers from the model.
        This needs to be customized for different model architectures.
        """
        layers = []
        for name, module in self.original_model.named_modules():
            # For transformer models, we typically want Linear layers
            # This is a generic implementation - specific models may need custom handling
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append((name, module))
        return layers
    
    def _get_layer_order(self) -> List[str]:
        """
        Get the order in which layers should be executed.
        For transformer models, this is typically the order they appear in the model.
        """
        layers = self._get_transformer_layers()
        return [name for name, _ in layers]
    
    def _paged_forward(self, inputs: Dict[str, torch.Tensor], 
                       max_new_tokens: int, **kwargs) -> torch.Tensor:
        """
        Forward pass with paged loading.
        
        Args:
            inputs: Tokenized inputs
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Output tensor
        """
        # Get the layer execution order
        layer_order = self._get_layer_order()
        
        # Get input embeddings (this varies by model architecture)
        # For simplicity, we'll assume inputs contains 'input_ids'
        x = inputs.get("input_ids", inputs)
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        
        # Process through each layer
        for layer_name in layer_order:
            try:
                layer = self.paged_model.get_layer(layer_name)
                x = layer(x)
            except Exception as e:
                logger.warning(f"Error processing layer {layer_name}: {e}")
                # Skip this layer and continue
                continue
        
        return x
    
    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """
        Generate text from a prompt using paged inference.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to device
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
        
        # For transformer models, we need to handle the forward pass carefully
        # This is a simplified implementation
        with torch.no_grad():
            # Try to use the model's generate method if available
            if hasattr(self.original_model, 'generate'):
                # For models with generate method, we need a different approach
                # since generate handles the forward pass internally
                outputs = self._generate_with_paging(inputs, max_new_tokens, **kwargs)
            else:
                # Fallback to manual forward pass
                outputs = self._paged_forward(inputs, max_new_tokens, **kwargs)
        
        # Decode the output
        if isinstance(outputs, torch.Tensor):
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif isinstance(outputs, dict):
            return self.tokenizer.decode(outputs["last_hidden_state"][0], skip_special_tokens=True)
        else:
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _generate_with_paging(self, inputs: Dict[str, torch.Tensor], 
                            max_new_tokens: int, **kwargs) -> torch.Tensor:
        """
        Generate with paging for models that have a generate method.
        This is more complex and requires hooking into the generation process.
        """
        # For now, we'll use a simplified approach
        # TODO: Implement proper generation with paging
        logger.warning("Paged generation with generate method not yet fully implemented")
        
        # Fallback to standard generation (this will use more memory)
        return self.original_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get paging and inference statistics"""
        paged_stats = self.paged_model.get_stats()
        return {
            **paged_stats,
            'engine': 'PagedInferenceEngine'
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.paged_model.clear_cache()
        logger.info("Cleared all caches")


class HybridInferenceEngine:
    """
    Hybrid inference engine that combines paging with CPU offloading.
    (Future enhancement)
    """
    
    def __init__(self, model: nn.Module, tokenizer: Any, 
                 gpu_memory_gb: float = 8.0, device: str = "cpu"):
        """
        Initialize hybrid inference engine.
        
        Args:
            model: The model to use
            tokenizer: The tokenizer
            gpu_memory_gb: Maximum GPU memory to use
            device: Primary device (cpu or cuda)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.gpu_memory_gb = gpu_memory_gb
        self.device = device
        
        # Split layers between GPU and CPU
        self.gpu_layers = []
        self.cpu_layers = []
        self._split_layers()
    
    def _split_layers(self):
        """Split layers between GPU and CPU based on memory"""
        current_memory = 0
        
        for name, layer in self.model.named_modules():
            layer_size = self._get_layer_size(layer)
            
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if current_memory + layer_size <= self.gpu_memory_gb * 1024:
                    self.gpu_layers.append((name, layer))
                    current_memory += layer_size
                else:
                    self.cpu_layers.append((name, layer))
    
    def _get_layer_size(self, layer: nn.Module) -> float:
        """Calculate layer size in MB"""
        size = 0
        for param in layer.parameters():
            size += param.numel() * param.element_size()
        return size / (1024 * 1024)
    
    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """Generate text with hybrid inference"""
        # TODO: Implement hybrid inference
        raise NotImplementedError("Hybrid inference not yet implemented")
