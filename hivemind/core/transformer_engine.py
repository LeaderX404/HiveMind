"""
Paged Transformer Engine for Out-of-Core Inference
Implements full transformer forward pass with paging.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)


class TransformerConfig:
    """Configuration for transformer models"""
    
    def __init__(self,
                 model_type: str = "mistral",
                 hidden_size: int = 4096,
                 num_layers: int = 32,
                 num_attention_heads: int = 32,
                 num_key_value_heads: Optional[int] = None,
                 vocab_size: int = 32000,
                 max_position_embeddings: int = 32768,
                 head_dim: Optional[int] = None):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings


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
    - LayerNorm
    - Attention with KV Cache
    - Feed-forward networks
    - Residual connections
    """
    
    def __init__(self,
                 file_pager: 'FileBackedPager',
                 tokenizer: Any,
                 config: TransformerConfig,
                 device: str = "cpu"):
        """
        Initialize paged transformer engine.
        
        Args:
            file_pager: FileBackedPager instance
            tokenizer: Tokenizer
            config: Transformer configuration
            device: Device to use
        """
        self.file_pager = file_pager
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
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
        
        # KV Cache
        self.kv_cache = None
        
        logger.info(f"PagedTransformerEngine initialized: {config.num_layers} layers")
    
    def _load_tensor(self, tensor_name: str) -> torch.Tensor:
        """Load tensor from file"""
        return self.file_pager.load_tensor(tensor_name)
    
    def _get_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings"""
        embed_weight = self._load_tensor("model.embed_tokens.weight")
        return F.embedding(input_ids, embed_weight)
    
    def _get_position_embedding(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Get position embeddings"""
        pos_embed = self._load_tensor("model.embed_positions.weight")
        return x + pos_embed[position_ids]
    
    def _layer_norm(self, x: torch.Tensor, prefix: str, suffix: str = "") -> torch.Tensor:
        """Apply layer normalization"""
        weight = self._load_tensor(f"{prefix}.input_layernorm{suffix}.weight")
        bias = self._load_tensor(f"{prefix}.input_layernorm{suffix}.bias")
        return F.layer_norm(x, (self.config.hidden_size,), weight, bias)
    
    def _attention(self, x: torch.Tensor, layer_config: LayerConfig) -> torch.Tensor:
        """Compute attention"""
        prefix = layer_config.prefix
        
        # Load weights
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
        
        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (layer_config.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = F.linear(context, o_proj)
        
        return output
    
    def _feed_forward(self, x: torch.Tensor, layer_config: LayerConfig) -> torch.Tensor:
        """Feed-forward network"""
        prefix = layer_config.prefix
        
        # Load weights
        gate_proj = self._load_tensor(f"{prefix}.mlp.gate_proj.weight")
        up_proj = self._load_tensor(f"{prefix}.mlp.up_proj.weight")
        down_proj = self._load_tensor(f"{prefix}.mlp.down_proj.weight")
        
        # FFN
        gate = F.linear(x, gate_proj)
        up = F.linear(x, up_proj)
        hidden = F.silu(gate) * up
        output = F.linear(hidden, down_proj)
        
        return output
    
    def _transformer_layer(self, x: torch.Tensor, layer_config: LayerConfig) -> torch.Tensor:
        """Single transformer layer"""
        prefix = layer_config.prefix
        
        # Input layer norm
        residual = x
        x = self._layer_norm(x, prefix, suffix=".input_layernorm")
        
        # Attention
        attn_out = self._attention(x, layer_config)
        x = residual + attn_out
        
        # Post-attention layer norm
        residual = x
        x = self._layer_norm(x, prefix, suffix=".post_attention_layernorm")
        
        # Feed-forward
        ffn_out = self._feed_forward(x, layer_config)
        x = residual + ffn_out
        
        return x
    
    def _final_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Final layer norm"""
        weight = self._load_tensor("model.norm.weight")
        bias = self._load_tensor("model.norm.bias")
        return F.layer_norm(x, (self.config.hidden_size,), weight, bias)
    
    def _get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get output logits"""
        weight = self._load_tensor("lm_head.weight")
        return F.linear(x, weight)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full forward pass"""
        # Embeddings
        x = self._get_embedding(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(input_ids.size(1), device=self.device).unsqueeze(0)
        x = self._get_position_embedding(x, position_ids)
        
        # Transformer layers
        for layer_config in self.layer_configs:
            x = self._transformer_layer(x, layer_config)
        
        # Final norm
        x = self._final_norm(x)
        
        # Output
        logits = self._get_logits(x)
        
        return logits
    
    def generate(self, prompt: str, max_new_tokens: int = 50, 
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50) -> str:
        """Generate text"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = input_ids.size(1)
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(input_ids)
            
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
