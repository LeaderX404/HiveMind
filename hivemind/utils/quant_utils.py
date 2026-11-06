"""
Quantization Utilities for Out-of-Core Inference
Supports Q4 and Q8 quantization for memory efficiency and speed.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)


class QuantizationConfig:
    """Configuration for quantization"""
    
    def __init__(self, 
                 bits: int = 8,
                 symmetric: bool = True,
                 group_size: int = 32):
        """
        Initialize quantization configuration.
        
        Args:
            bits: Quantization bits (4 or 8)
            symmetric: Use symmetric quantization
            group_size: Group size for per-group quantization
        """
        assert bits in [4, 8], f"Only 4 and 8 bit quantization supported, got {bits}"
        self.bits = bits
        self.symmetric = symmetric
        self.group_size = group_size
        self.scale_dtype = torch.float16


def quantize_tensor(tensor: torch.Tensor, 
                   config: QuantizationConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor.
    
    Returns:
        quantized: Quantized tensor (uint8 or int8)
        scales: Scale factors
        zeros: Zero points (for asymmetric quantization)
    """
    if config.bits == 8:
        return _quantize_q8(tensor, config)
    elif config.bits == 4:
        return _quantize_q4(tensor, config)
    else:
        raise ValueError(f"Unsupported quantization bits: {config.bits}")


def dequantize_tensor(quantized: torch.Tensor,
                     scales: torch.Tensor,
                     zeros: Optional[torch.Tensor] = None,
                     config: QuantizationConfig = None,
                     output_shape: Optional[Tuple[int, ...]] = None,
                     original_numel: Optional[int] = None) -> torch.Tensor:
    """
    Dequantize a tensor.
    
    Args:
        quantized: Quantized tensor
        scales: Scale factors
        zeros: Zero points (for asymmetric quantization)
        config: Quantization configuration
        output_shape: Expected output shape
        original_numel: Original number of elements (for truncation)
    
    Returns:
        Dequantized tensor
    """
    if config is None:
        config = QuantizationConfig(bits=8 if quantized.dtype == torch.uint8 else 4)
    
    if config.bits == 8:
        dequantized = _dequantize_q8(quantized, scales, zeros, config)
    elif config.bits == 4:
        dequantized = _dequantize_q4(quantized, scales, zeros, config)
    else:
        raise ValueError(f"Unsupported quantization bits: {config.bits}")
    
    # Truncate to original number of elements if provided
    if original_numel is not None and dequantized.numel() > original_numel:
        dequantized = dequantized[:original_numel]
    
    # Reshape to expected output shape if provided
    if output_shape is not None:
        dequantized = dequantized.reshape(output_shape)
    
    return dequantized


def _quantize_q8(tensor: torch.Tensor, 
                config: QuantizationConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize to 8-bit with per-group quantization"""
    # Store original shape
    original_numel = tensor.numel()
    
    # Flatten
    tensor_flat = tensor.reshape(-1)
    
    # Calculate number of groups
    num_groups = (original_numel + config.group_size - 1) // config.group_size
    padded_size = num_groups * config.group_size
    
    # Pad if needed
    if original_numel < padded_size:
        padding = torch.zeros(padded_size - original_numel, 
                            dtype=tensor.dtype, device=tensor.device)
        tensor_flat = torch.cat([tensor_flat, padding])
    
    # Reshape to groups
    tensor_groups = tensor_flat.reshape(num_groups, config.group_size)
    
    # Quantize each group
    quantized_list = []
    scales_list = []
    zeros_list = []
    
    for i in range(num_groups):
        group = tensor_groups[i]
        
        if config.symmetric:
            abs_max = group.abs().max()
            scale = abs_max / 127.0
            zero = torch.tensor(0.0, device=tensor.device)
            quantized = (group / scale).round().clamp(-128, 127).to(torch.int8)
        else:
            min_val = group.min()
            max_val = group.max()
            scale = (max_val - min_val) / 255.0
            zero = min_val
            quantized = ((group - zero) / scale).round().clamp(0, 255).to(torch.uint8)
        
        quantized_list.append(quantized)
        scales_list.append(scale)
        zeros_list.append(zero)
    
    # Stack and flatten
    quantized = torch.stack(quantized_list).reshape(-1)
    scales = torch.stack(scales_list).to(config.scale_dtype)
    zeros = torch.stack(zeros_list).to(config.scale_dtype)
    
    return quantized, scales, zeros


def _dequantize_q8(quantized: torch.Tensor,
                   scales: torch.Tensor,
                   zeros: torch.Tensor,
                   config: QuantizationConfig) -> torch.Tensor:
    """Dequantize from 8-bit"""
    num_groups = scales.numel()
    expected_size = num_groups * config.group_size
    
    # Ensure quantized has enough elements
    if quantized.numel() < expected_size:
        padding = torch.zeros(expected_size - quantized.numel(), 
                            dtype=quantized.dtype, device=quantized.device)
        quantized = torch.cat([quantized.reshape(-1), padding])
    
    # Reshape to groups
    quantized_groups = quantized.reshape(num_groups, config.group_size)
    
    # Dequantize each group
    dequantized_groups = []
    
    for i in range(num_groups):
        group_quant = quantized_groups[i]
        scale = scales[i]
        zero = zeros[i]
        
        if config.symmetric:
            if group_quant.dtype == torch.uint8:
                group_quant = group_quant.to(torch.int8)
            dequantized = (group_quant * scale).to(torch.float16)
        else:
            dequantized = (group_quant * scale + zero).to(torch.float16)
        
        dequantized_groups.append(dequantized)
    
    # Concatenate all groups
    dequantized = torch.cat(dequantized_groups)
    
    return dequantized


def _quantize_q4(tensor: torch.Tensor, 
                config: QuantizationConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize to 4-bit with per-group quantization"""
    # Store original shape
    original_numel = tensor.numel()
    
    # Flatten
    tensor_flat = tensor.reshape(-1)
    
    # Calculate number of groups
    num_groups = (original_numel + config.group_size - 1) // config.group_size
    padded_size = num_groups * config.group_size
    
    # Pad if needed
    if original_numel < padded_size:
        padding = torch.zeros(padded_size - original_numel, 
                            dtype=tensor.dtype, device=tensor.device)
        tensor_flat = torch.cat([tensor_flat, padding])
    
    # Reshape to groups
    tensor_groups = tensor_flat.reshape(num_groups, config.group_size)
    
    # Quantize each group to 4-bit
    all_quantized_values = []
    scales_list = []
    mins_list = []
    
    for i in range(num_groups):
        group = tensor_groups[i]
        min_val = group.min()
        max_val = group.max()
        scale = (max_val - min_val) / 15.0
        quantized = ((group - min_val) / scale).round().clamp(0, 15).to(torch.uint8)
        
        all_quantized_values.extend(quantized.tolist())
        scales_list.append(scale)
        mins_list.append(min_val)
    
    # Pack two 4-bit values into one byte
    quantized_packed = []
    for i in range(0, len(all_quantized_values), 2):
        if i + 1 < len(all_quantized_values):
            quantized_packed.append((all_quantized_values[i] << 4) | all_quantized_values[i + 1])
        else:
            quantized_packed.append(all_quantized_values[i] << 4)
    
    quantized = torch.tensor(quantized_packed, dtype=torch.uint8, device=tensor.device)
    scales = torch.stack(scales_list).to(config.scale_dtype)
    mins = torch.stack(mins_list).to(config.scale_dtype)
    
    return quantized, scales, mins


def _dequantize_q4(quantized: torch.Tensor,
                   scales: torch.Tensor,
                   mins: torch.Tensor,
                   config: QuantizationConfig) -> torch.Tensor:
    """Dequantize from 4-bit"""
    num_groups = scales.numel()
    
    # Unpack all bytes
    unpacked_values = []
    for byte in quantized:
        unpacked_values.append((byte >> 4) & 0x0F)
        unpacked_values.append(byte & 0x0F)
    
    unpacked = torch.tensor(unpacked_values, dtype=torch.uint8, device=quantized.device)
    
    # Reshape to groups
    unpacked_groups = unpacked.reshape(num_groups, config.group_size)
    
    # Dequantize each group
    dequantized_groups = []
    
    for i in range(num_groups):
        group_quant = unpacked_groups[i]
        scale = scales[i]
        min_val = mins[i]
        dequantized = (group_quant.float() * scale + min_val).to(torch.float16)
        dequantized_groups.append(dequantized)
    
    # Concatenate all groups
    dequantized = torch.cat(dequantized_groups)
    
    return dequantized


class QuantizedLinear:
    """Linear layer with quantized weights"""
    
    def __init__(self, 
                 quantized_weight: torch.Tensor,
                 scales: torch.Tensor,
                 zeros_or_mins: torch.Tensor,
                 config: QuantizationConfig,
                 out_features: int,
                 in_features: int):
        self.quantized_weight = quantized_weight
        self.scales = scales
        self.zeros_or_mins = zeros_or_mins
        self.config = config
        self.out_features = out_features
        self.in_features = in_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization on-the-fly"""
        weight = dequantize_tensor(
            self.quantized_weight,
            self.scales,
            self.zeros_or_mins,
            self.config,
            (self.out_features, self.in_features),
            self.out_features * self.in_features
        )
        return F.linear(x, weight)
    
    def to(self, device: str) -> 'QuantizedLinear':
        self.quantized_weight = self.quantized_weight.to(device)
        self.scales = self.scales.to(device)
        self.zeros_or_mins = self.zeros_or_mins.to(device)
        return self


def quantize_model_state_dict(state_dict: Dict[str, torch.Tensor],
                             config: QuantizationConfig) -> Dict[str, torch.Tensor]:
    """Quantize all weight tensors in a model state dict"""
    quantized_state_dict = {}
    
    for name, tensor in state_dict.items():
        if 'weight' in name and 'bias' not in name:
            quantized, scales, zeros = quantize_tensor(tensor, config)
            quantized_state_dict[name] = quantized
            quantized_state_dict[f"{name}_scales"] = scales
            quantized_state_dict[f"{name}_zeros"] = zeros
            quantized_state_dict[f"{name}_shape"] = torch.tensor(tensor.shape)
            quantized_state_dict[f"{name}_numel"] = torch.tensor(tensor.numel())
            logger.info(f"Quantized {name}: {tensor.shape}")
        else:
            quantized_state_dict[name] = tensor
    
    return quantized_state_dict


def dequantize_model_state_dict(quantized_state_dict: Dict[str, torch.Tensor],
                               config: QuantizationConfig) -> Dict[str, torch.Tensor]:
    """Dequantize all tensors in a quantized state dict"""
    state_dict = {}
    
    for name, tensor in quantized_state_dict.items():
        if name.endswith('_scales') or name.endswith('_zeros') or name.endswith('_shape') or name.endswith('_numel'):
            continue
        
        if f"{name}_scales" in quantized_state_dict:
            quantized = tensor
            scales = quantized_state_dict[f"{name}_scales"]
            zeros = quantized_state_dict.get(f"{name}_zeros", None)
            shape_name = f"{name}_shape"
            numel_name = f"{name}_numel"
            output_shape = None
            original_numel = None
            
            if shape_name in quantized_state_dict:
                output_shape = tuple(quantized_state_dict[shape_name].tolist())
            if numel_name in quantized_state_dict:
                original_numel = quantized_state_dict[numel_name].item()
            
            dequantized = dequantize_tensor(quantized, scales, zeros, config, output_shape, original_numel)
            state_dict[name] = dequantized
            logger.info(f"Dequantized {name}: {dequantized.shape}")
        else:
            state_dict[name] = tensor
    
    return state_dict
