import torch
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)

class LocalInferenceEngine:
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        x = inputs["input_ids"]
        for i in range(len(self.shards)):
            x = self.forward_shard(i, x)
        outputs = torch.argmax(x, dim=-1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
