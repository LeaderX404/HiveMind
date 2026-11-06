# User Guide

This guide walks through the core HiveMind workflows: out-of-core paging,
quantization, dynamic bit selection and large model presets.

## Adaptive page sizes

Instead of hardcoding one page size for every model, HiveMind derives it
from the model size on disk:

```python
from hivemind.core.file_pager import compute_optimal_page_size_mb

# 100B model (~200 GB on disk) -> 4000 MB pages
page_size = compute_optimal_page_size_mb(model_size_mb=200000.0, current_page_size_mb=1000)
```

The function never lowers an existing setting, and small models keep
their original page size.

## Quantization

Quantize tensors or whole state dicts to 2, 4 or 8 bits:

```python
from hivemind.utils.quant_utils import (
    QuantizationConfig,
    quantize_model_state_dict,
    dequantize_model_state_dict,
)

config = QuantizationConfig(bits=2, group_size=32)
quantized_state = quantize_model_state_dict(state_dict, config)
restored = dequantize_model_state_dict(quantized_state, config)
```

With 2-bit quantization a 100B-parameter model shrinks from roughly
200 GB to 25 GB — small enough for a single 32 GB machine.

## Dynamic bit selection

Let HiveMind pick the highest precision that fits available memory:

```python
from hivemind.core.quantizer import DynamicQuantizer

dynamic = DynamicQuantizer(min_bits=2, max_bits=8)
bits = dynamic.get_optimal_bits(available_memory_gb=16.0, model_size_gb=200.0)
```

A compressed size must fit within 90% of the available memory, so the
quantizer always leaves headroom for activations and the KV cache.

## Large model presets

Reference configurations for 50B, 70B and 100B models:

```python
from hivemind.core.transformer_engine import TransformerConfig

config = TransformerConfig.from_preset("100b", quant_bits=2)
```

Presets use grouped-query attention (8 KV heads) to keep the KV cache
small at 50B+ scale.

## Distributed inference (Neural Swarm)

Multiple low-end devices can act as one bigger machine over gRPC.
See the [API Reference](api_reference.md) for the networking entry points.
