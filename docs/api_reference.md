# API Reference

## hivemind.utils.quant_utils

Low-level quantization primitives shared by the engine and the pager.

- `QuantizationConfig(bits=8, symmetric=True, group_size=32)` — quantization settings; `bits` must be 2, 4 or 8.
- `quantize_tensor(tensor, config)` — quantize a tensor; returns `(quantized, scales, zeros_or_mins)`.
- `dequantize_tensor(quantized, scales, zeros, config, output_shape, original_numel)` — restore a quantized tensor to float precision.
- `quantize_model_state_dict(state_dict, config)` — quantize every weight in a state dict (adds `_scales`, `_zeros`, `_shape` and `_numel` companion tensors).
- `dequantize_model_state_dict(quantized_state_dict, config)` — restore a quantized state dict.
- `QuantizedLinear` — linear layer that dequantizes its weight on the fly at forward time.

## hivemind.core.quantizer

Model-level quantization and memory-aware bit selection.

- `DynamicQuantizer(min_bits=2, max_bits=8)` — picks the highest bit-width whose compressed size fits in 90% of available RAM.
- `DynamicQuantizer.get_optimal_bits(available_memory_gb, model_size_gb)` — returns the selected bit-width.
- `AdvancedQuantizer(config)` — model quantization via bitsandbytes, QLoRA or GPTQ.
- `quantize_model(model, bits=4, method="bitsandbytes")` — factory helper.

## hivemind.core.file_pager

Out-of-core paging of model weights.

- `FileBackedPager` — pages tensors in and out of memory from a model directory.
- `compute_optimal_page_size_mb(model_size_mb, current_page_size_mb)` — pure helper: 4000 MB pages for 100 GB+ models, 2000 MB for 20 GB+; never lowers an existing setting.

## hivemind.core.transformer_engine

- `TransformerConfig` — configuration for the paged engine.
- `TransformerConfig.from_preset(name, quant_bits=None)` — reference configs for `"50b"`, `"70b"` and `"100b"`. Unknown names raise `ValueError`.
- `LARGE_MODEL_PRESETS` — the raw preset table (GQA attention with 8 KV heads).
- `PagedTransformerEngine` — the inference engine; `_quantized_linear` dequantizes weight pages on the fly.

## hivemind.net

Distributed inference (Neural Swarm) over gRPC.

- `hivemind.net.server.serve` — start a swarm node.
