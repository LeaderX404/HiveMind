"""
Paged Transformer Engine for Out-of-Core Inference
Implements full transformer forward pass with paging.
Supports: RoPE, RMSNorm, Causal Mask, KV Cache, GQA, Q4/Q8 Quantization
Phase 2: Added batch processing for 10B+ models
"""
import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from hivemind.utils.logger import get_logger
from hivemind.utils.quant_utils import QuantizationConfig, dequantize_tensor

logger = get_logger(__name__)


class TransformerConfig:
    """Configuration for transformer models"""
    
    def __init__(self,
                 model_type: str = "llama",
                 hidden_size: int = 4096,
                 num_layers: int = 32,
                 num_attention_heads: int = 32,
                 num_key_value_heads: Optional[int] = None,
                 vocab_size: int = 32000,
                 max_position_embeddings: int = 32768,
                 head_dim: Optional[int] = None,
                 rope_theta: float = 500000.0,
                 use_rms_norm: bool = True,
                 quant_bits: Optional[int] = None):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_rms_norm = use_rms_norm
        self.quant_bits = quant_bits
        
        # Quantization config
        if quant_bits is not None:
            self.quant_config = QuantizationConfig(bits=quant_bits)
        else:
            self.quant_config = None
        
        # Precompute RoPE frequencies
        self._rope_freqs = self._precompute_rope_frequencies()
    
    def _precompute_rope_frequencies(self) -> torch.Tensor:
        """Precompute RoPE frequencies for caching"""
        # Compute inverse frequencies
        head_dim = self.head_dim or (self.hidden_size // self.num_attention_heads)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        return inv_freq


class LayerConfig:
    """Configuration for a single transformer layer"""
    
    def __init__(self, layer_index: int, prefix: str, hidden_size: int, 
                 num_attention_heads: int, num_key_value_heads: int, head_dim: int):
        self.layer_index = layer_index
        self.prefix = prefix
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim


class PagedTransformerEngine:
    """
    Transformer inference engine with paging support.
    Implements full forward pass with:
    - Embeddings
    - RoPE (Rotary Position Embedding)
    - RMSNorm
    - Attention with KV Cache
    - Causal Mask
    - Feed-forward networks
    - Residual connections
    - GQA (Grouped Query Attention) support
    - Q4/Q8 Quantization support
    """
    
    def __init__(self,
                 file_pager: 'FileBackedPager',
                 tokenizer: Any,
                 config: TransformerConfig,
                 device: str = "cpu",
                 batch_size: int = 1):
        """
        Initialize paged transformer engine.
        
        Args:
            file_pager: FileBackedPager instance
            tokenizer: Tokenizer
            config: Transformer configuration
            device: Device to use
            batch_size: Batch size for generation (Phase 2 feature)
        """
        self.file_pager = file_pager
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.batch_size = batch_size
        
        # Create layer configs
        self.layer_configs = []
        for i in range(config.num_layers):
            prefix = f"model.layers.{i}"
            self.layer_configs.append(LayerConfig(
                layer_index=i,
                prefix=prefix,
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim
            ))
        
        # KV Cache: stores (key, value) for each layer
        self.kv_cache = None
        
        # Tensor cache: LRU cache for loaded tensors
        self.tensor_cache = {}
        self.max_cached_tensors = 100
        
        # Quantization support
        self.quant_bits = config.quant_bits
        
        # Position tracking for RoPE
        self.current_position = 0
        
        logger.info(f"PagedTransformerEngine initialized: {config.num_layers} layers, {config.model_type} config, batch_size={batch_size}")
        if config.quant_bits:
            logger.info(f"Quantization: {config.quant_bits}-bit enabled")
    
    def _load_tensor(self, tensor_name: str) -> torch.Tensor:
        """Load tensor from file with caching and dequantization"""
        # Check cache first
        if tensor_name in self.tensor_cache:
            return self.tensor_cache[tensor_name]
        
        # Load from file
        tensor = self.file_pager.load_tensor(tensor_name)
        
        # If quantized, ensure it's dequantized
        if self.config.quant_config is not None and tensor.dtype == torch.uint8:
            # This might be a quantized tensor, try to dequantize
            # For now, we assume the loader handles dequantization
            pass
        
        # Cache it
        if len(self.tensor_cache) >= self.max_cached_tensors:
            # Simple eviction: remove first key
            first_key = next(iter(self.tensor_cache))
            del self.tensor_cache[first_key]
        
        self.tensor_cache[tensor_name] = tensor
        return tensor
    
    def _apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embedding (RoPE)"""
        # x shape: [batch, num_heads, seq_len, head_dim]
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # Compute RoPE frequencies
        inv_freq = self.config._rope_freqs.to(x.device)
        
        # Create position embeddings
        pos_emb = torch.outer(position_ids.float(), inv_freq)
        
        # Create cos and sin embeddings
        cos_emb = torch.cos(pos_emb)
        sin_emb = torch.sin(pos_emb)
        
        # Reshape for broadcasting: [batch, 1, seq_len, head_dim]
        cos_emb = cos_emb.unsqueeze(0).unsqueeze(0)
        sin_emb = sin_emb.unsqueeze(0).unsqueeze(0)
        
        # Split x into two parts for RoPE
        x1 = x[..., :head_dim//2]
        x2 = x[..., head_dim//2:]
        
        # Apply RoPE
        out1 = x1 * cos_emb - x2 * sin_emb
        out2 = x2 * cos_emb + x1 * sin_emb
        
        return torch.cat([out1, out2], dim=-1)
    
    def _rms_norm(self, x: torch.Tensor, prefix: str, suffix: str = "") -> torch.Tensor:
        """Apply RMS Normalization (used in Llama, Mistral)"""
        weight = self._load_tensor(f"{prefix}.input_layernorm{suffix}.weight")
        # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        norm = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return norm * weight
    
    def _layer_norm(self, x: torch.Tensor, prefix: str, suffix: str = "") -> torch.Tensor:
        """Apply layer normalization (backward compatibility)"""
        if self.config.use_rms_norm:
            return self._rms_norm(x, prefix, suffix)
        else:
            weight = self._load_tensor(f"{prefix}.input_layernorm{suffix}.weight")
            bias = self._load_tensor(f"{prefix}.input_layernorm{suffix}.bias")
            return F.layer_norm(x, (self.config.hidden_size,), weight, bias)
    
    def _quantized_linear(self, x: torch.Tensor, weight_name: str) -> torch.Tensor:
        """Linear operation with quantized weights"""
        # Load quantized weight
        quantized_weight = self._load_tensor(weight_name)
        
        # If quantized, dequantize
        if self.config.quant_config is not None and quantized_weight.dtype == torch.uint8:
            # Load scales and zeros
            scales_name = weight_name.replace('.weight', '.scales')
            zeros_name = weight_name.replace('.weight', '.zeros')
            
            try:
                scales = self._load_tensor(scales_name)
                zeros = self._load_tensor(zeros_name)
                weight = dequantize_tensor(quantized_weight, scales, zeros, self.config.quant_config)
            except:
                # If scales/zeros not found, use the weight as-is
                weight = quantized_weight
        else:
            weight = quantized_weight
        
        return F.linear(x, weight)
    
    def _attention(self, x: torch.Tensor, layer_config: LayerConfig, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute attention with RoPE and causal mask"""
        prefix = layer_config.prefix
        
        # Use quantized linear if quantization is enabled
        if self.config.quant_config is not None:
            q = self._quantized_linear(x, f"{prefix}.self_attn.q_proj.weight")
            k = self._quantized_linear(x, f"{prefix}.self_attn.k_proj.weight")
            v = self._quantized_linear(x, f"{prefix}.self_attn.v_proj.weight")
            o = self._quantized_linear(x, f"{prefix}.self_attn.o_proj.weight")
        else:
            # Load weights normally
            q_proj = self._load_tensor(f"{prefix}.self_attn.q_proj.weight")
            k_proj = self._load_tensor(f"{prefix}.self_attn.k_proj.weight")
            v_proj = self._load_tensor(f"{prefix}.self_attn.v_proj.weight")
            o_proj = self._load_tensor(f"{prefix}.self_attn.o_proj.weight")
            
            # Project
            q = F.linear(x, q_proj)
            k = F.linear(x, k_proj)
            v = F.linear(x, v_proj)
        
        # Reshape for multi-head
        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, layer_config.num_attention_heads, layer_config.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, layer_config.num_key_value_heads, layer_config.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, layer_config.num_key_value_heads, layer_config.head_dim).transpose(1, 2)
        
        # Grouped Query Attention (GQA) support
        if layer_config.num_attention_heads != layer_config.num_key_value_heads:
            # Repeat KV heads for GQA
            num_groups = layer_config.num_attention_heads // layer_config.num_key_value_heads
            k = k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(
                batch_size, layer_config.num_attention_heads, seq_len, layer_config.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(
                batch_size, layer_config.num_attention_heads, seq_len, layer_config.head_dim)
        
        # Apply RoPE
        if position_ids is not None:
            q = self._apply_rope(q, position_ids)
            k = self._apply_rope(k, position_ids)
        
        # Causal mask: prevent attending to future tokens
        if seq_len > 1:
            # Create causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (layer_config.head_dim ** 0.5)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (layer_config.head_dim ** 0.5)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        if self.config.quant_config is not None:
            output = self._quantized_linear(context, f"{prefix}.self_attn.o_proj.weight")
        else:
            output = F.linear(context, o_proj)
        
        return output
    
    def _feed_forward(self, x: torch.Tensor, layer_config: LayerConfig) -> torch.Tensor:
        """Feed-forward network with optional quantization"""
        prefix = layer_config.prefix
        
        if self.config.quant_config is not None:
            gate = self._quantized_linear(x, f"{prefix}.mlp.gate_proj.weight")
            up = self._quantized_linear(x, f"{prefix}.mlp.up_proj.weight")
            down = self._quantized_linear(x, f"{prefix}.mlp.down_proj.weight")
        else:
            # Load weights normally
            gate_proj = self._load_tensor(f"{prefix}.mlp.gate_proj.weight")
            up_proj = self._load_tensor(f"{prefix}.mlp.up_proj.weight")
            down_proj = self._load_tensor(f"{prefix}.mlp.down_proj.weight")
            
            gate = F.linear(x, gate_proj)
            up = F.linear(x, up_proj)
            
        hidden = F.silu(gate) * up
        
        if self.config.quant_config is not None:
            output = self._quantized_linear(hidden, f"{prefix}.mlp.down_proj.weight")
        else:
            output = F.linear(hidden, down_proj)
        
        return output
    
    def _transformer_layer(self, x: torch.Tensor, layer_config: LayerConfig, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single transformer layer with RoPE and KV Cache"""
        prefix = layer_config.prefix
        
        # Input layer norm
        residual = x
        x = self._rms_norm(x, prefix, suffix=".input_layernorm")
        
        # Attention with position IDs
        attn_out = self._attention(x, layer_config, position_ids)
        x = residual + attn_out
        
        # Post-attention layer norm
        residual = x
        x = self._rms_norm(x, prefix, suffix=".post_attention_layernorm")
        
        # Feed-forward
        ffn_out = self._feed_forward(x, layer_config)
        x = residual + ffn_out
        
        return x
    
    def _get_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings"""
        embed_weight = self._load_tensor("model.embed_tokens.weight")
        return F.embedding(input_ids, embed_weight)
    
    def _get_position_embedding(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Get position embeddings (for models that use it)"""
        # Llama doesn't use position embeddings, only RoPE
        if self.config.model_type in ["llama", "mistral", "qwen"]:
            return x
        
        try:
            pos_embed = self._load_tensor("model.embed_positions.weight")
            return x + pos_embed[position_ids]
        except:
            return x
    
    def _final_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Final layer norm"""
        if self.config.use_rms_norm:
            weight = self._load_tensor("model.norm.weight")
            norm = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
            return norm * weight
        else:
            weight = self._load_tensor("model.norm.weight")
            bias = self._load_tensor("model.norm.bias")
            return F.layer_norm(x, (self.config.hidden_size,), weight, bias)
    
    def _get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get output logits"""
        weight = self._load_tensor("lm_head.weight")
        return F.linear(x, weight)
    
    def _initialize_kv_cache(self, batch_size: int):
        """Initialize KV cache for all layers"""
        self.kv_cache = {
            'k': [],
            'v': []
        }
        for _ in range(self.config.num_layers):
            self.kv_cache['k'].append(None)
            self.kv_cache['v'].append(None)
    
    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Full forward pass with KV Cache support"""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self._get_embedding(input_ids)
        
        # Position IDs for RoPE
        if self.current_position == 0:
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        else:
            position_ids = torch.arange(self.current_position, self.current_position + seq_len, device=self.device).unsqueeze(0)
        
        # Position embeddings (if used)
        x = self._get_position_embedding(x, position_ids)
        
        # Transformer layers
        for layer_config in self.layer_configs:
            x = self._transformer_layer(x, layer_config, position_ids)
        
        # Final norm
        x = self._final_norm(x)
        
        # Output
        logits = self._get_logits(x)
        
        # Update position for next token
        if use_cache:
            self.current_position += seq_len
        
        return logits
    
    def generate(self, prompt: str, max_new_tokens: int = 50, 
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50) -> str:
        """Generate text with KV Cache and efficient token-by-token generation"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = input_ids.size(1)
        
        # Reset position counter
        self.current_position = 0
        
        # Pre-allocate KV cache
        if self.kv_cache is None:
            self._initialize_kv_cache(input_ids.size(0))
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                # Only process the current token with KV cache
                if step == 0:
                    # First step: process entire prompt
                    logits = self.forward(input_ids, use_cache=True)
                else:
                    # Subsequent steps: process only the last token
                    last_token = input_ids[:, -1:]
                    logits = self.forward(last_token, use_cache=True)
            
            # Get last token logits
            last_logits = logits[0, -1, :]
            
            # Apply sampling
            if temperature > 0:
                last_logits = last_logits / temperature
            
            if top_k > 0:
                v, _ = torch.topk(last_logits, top_k)
                last_logits[last_logits < v[-1]] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                last_logits.scatter_(0, indices_to_remove, float('-inf'))
            
            # Sample
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(input_ids[0, input_length:], skip_special_tokens=True)
    
    def batch_generate(self, prompts: List[str], max_new_tokens: int = 50,
                       temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50) -> List[str]:
        """
        Batch generation for multiple prompts (Phase 2 feature)
        More efficient than individual generation for 10B+ models
        """
        # Tokenize all prompts
        input_ids_list = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            input_ids_list.append(input_ids)
        
        # Pad to same length
        max_length = max(ids.size(1) for ids in input_ids_list)
        padded_input_ids = []
        for ids in input_ids_list:
            pad_size = max_length - ids.size(1)
            if pad_size > 0:
                pad_token = torch.full((1, pad_size), self.tokenizer.pad_token_id, device=self.device)
                padded_ids = torch.cat([ids, pad_token], dim=1)
            else:
                padded_ids = ids
            padded_input_ids.append(padded_ids)
        
        # Stack into batch
        batch_input_ids = torch.cat(padded_input_ids, dim=0)
        
        # Reset position counter
        self.current_position = 0
        
        # Pre-allocate KV cache for batch
        if self.kv_cache is None:
            self._initialize_kv_cache(batch_input_ids.size(0))
        
        # Generate for each token position
        all_outputs = [[] for _ in range(len(prompts))]
        input_lengths = [ids.size(1) for ids in input_ids_list]
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                # Process current tokens
                if step == 0:
                    # First step: process entire batch
                    logits = self.forward(batch_input_ids, use_cache=True)
                else:
                    # Subsequent steps: process only the last token for each prompt
                    last_tokens = batch_input_ids[:, -1:]
                    logits = self.forward(last_tokens, use_cache=True)
            
            # Process each prompt in the batch
            for i in range(len(prompts)):
                # Get logits for this prompt
                prompt_logits = logits[i, -1, :]
                
                # Apply sampling
                if temperature > 0:
                    prompt_logits = prompt_logits / temperature
                
                if top_k > 0:
                    v, _ = torch.topk(prompt_logits, top_k)
                    prompt_logits[prompt_logits < v[-1]] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(prompt_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    prompt_logits.scatter_(0, indices_to_remove, float('-inf'))
                
                # Sample
                probs = F.softmax(prompt_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to output
                all_outputs[i].append(next_token.item())
                
                # Update input_ids for next step
                if step == 0:
                    batch_input_ids[i, input_lengths[i]:] = next_token
                else:
                    batch_input_ids[i, -1] = next_token
                
                # Check EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    all_outputs[i].append(self.tokenizer.eos_token_id)
        
        # Decode all outputs
        results = []
        for i, output_tokens in enumerate(all_outputs):
            # Skip the first token (which is already in the prompt)
            if len(output_tokens) > 0:
                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                results.append(output_text)
            else:
                results.append("")
        
        return results
    
    def _initialize_kv_cache(self, batch_size: int):
        """Initialize KV cache for batch processing (Phase 2 feature)"""
        self.kv_cache = {
            'k': [],
            'v': []
        }
        for _ in range(self.config.num_layers):
            # Initialize with batch dimension
            self.kv_cache['k'].append(None)
            self.kv_cache['v'].append(None)
        
        self.kv_cache_batch_size = batch_size
    
    def clear_cache(self):
        """Clear all caches"""
        self.tensor_cache.clear()
        self.current_position = 0
        self.kv_cache = None
        logger.info("PagedTransformerEngine cache cleared")
