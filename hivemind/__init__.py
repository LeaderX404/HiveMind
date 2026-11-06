"""
HiveMind by VALCRYPTIA
Out-of-Core Inference Engine for Running 7B+ LLMs on Single Devices with Limited RAM

This package provides tools for true out-of-core inference, allowing you to run
large language models on devices with limited RAM by loading model weights directly
from disk files (GGUF/Safetensors) without ever loading the full model into memory.

Key Features:
- Direct file loading from GGUF/Safetensors
- File-backed paging with LRU caching
- Full transformer forward pass (RoPE, RMSNorm, Attention, FFN)
- Q4 and Q8 quantization support
- Performance monitoring
- Support for Llama, Mistral, Qwen, Phi-2 models

Author: VALCRYPTIA
License: MIT
"""

__version__ = "2.1.0"
__author__ = "VALCRYPTIA"
__description__ = "Out-of-Core Inference Engine for LLMs"

# Import core modules
from .core import (
    ModelLoader,
    ModelSharder,
    LocalInferenceEngine,
    ShardedInferenceEngine,
    PagedInferenceEngine,
    HybridInferenceEngine,
    DirectFileModelLoader,
    FileBackedPager,
    PagedTransformerEngine,
    TransformerConfig,
    LayerConfig,
)

from .utils import (
    get_logger,
    PerformanceMonitor,
    QuantizationConfig,
    quantize_tensor,
    dequantize_tensor,
)

__all__ = [
    # Core
    "ModelLoader",
    "ModelSharder",
    "LocalInferenceEngine",
    "ShardedInferenceEngine",
    "PagedInferenceEngine",
    "HybridInferenceEngine",
    "DirectFileModelLoader",
    "FileBackedPager",
    "PagedTransformerEngine",
    "TransformerConfig",
    "LayerConfig",
    # Utils
    "get_logger",
    "PerformanceMonitor",
    "QuantizationConfig",
    "quantize_tensor",
    "dequantize_tensor",
    # Version
    "__version__",
    "__author__",
    "__description__",
]
