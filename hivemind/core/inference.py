import torch
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)

class LocalInferenceEngine:
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs):
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
    def __init__(self, shards, tokenizer, device: str = "cpu"):
        self.shards = shards
        self.tokenizer = tokenizer
        self.device = device

    def forward_shard(self, shard_index: int, input_tensor: torch.Tensor):
        shard_layers = self.shards[shard_index]
        x = input_tensor
        for name, layer in shard_layers:
            x = layer(x)
        return x

    def generate(self, prompt: str, max_new_tokens: int = 50):
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
