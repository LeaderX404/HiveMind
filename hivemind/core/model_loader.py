from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from pathlib import Path
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)

class ModelLoader:
    def __init__(self, model_name: str = "microsoft/phi-2", quantize: bool = False, bits: int = 4):
        self.model_name = model_name
        self.quantize = quantize
        self.bits = bits
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.quantize:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                load_in_4bit=True if self.bits == 4 else False,
                load_in_8bit=True if self.bits == 8 else False,
                device_map="auto" if "cuda" in self.device else None,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            ).to(self.device)

        logger.info(f"Model loaded on {self.device}")
        return self.model, self.tokenizer

    def save_shard(self, shard_index: int, layers: list, output_dir: str = "shards"):
        os.makedirs(output_dir, exist_ok=True)
        shard_path = Path(output_dir) / f"shard_{shard_index}.pt"
        torch.save({"layers": layers, "config": self.model.config}, shard_path)
        logger.info(f"Saved shard {shard_index} to {shard_path}")
        return shard_path
