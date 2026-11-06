import torch
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)

class ModelSharder:
    def __init__(self, model, num_shards: int = 2):
        self.model = model
        self.num_shards = num_shards
        self.shards = []

    def split(self):
        logger.info(f"Splitting model into {self.num_shards} shards")
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layers.append((name, module))

        shard_size = len(layers) // self.num_shards
        for i in range(self.num_shards):
            start = i * shard_size
            end = (i + 1) * shard_size if i < self.num_shards - 1 else len(layers)
            shard_layers = layers[start:end]
            self.shards.append(shard_layers)

        logger.info(f"Model split into {len(self.shards)} shards")
        return self.shards

    def get_shard(self, shard_index: int):
        if shard_index >= len(self.shards):
            raise ValueError(f"Shard index {shard_index} out of range")
        return self.shards[shard_index]
