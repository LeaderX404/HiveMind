from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from pathlib import Path
from hivemind.utils.logger import get_logger
from hivemind.core.quantizer import AdvancedQuantizer, QuantizationConfig

logger = get_logger(__name__)

class ModelLoader:
    def __init__(self, model_name: str = "microsoft/phi-2", quantize: bool = False, bits: int = 4, quant_method: str = "bitsandbytes"):
        self.model_name = model_name
        self.quantize = quantize
        self.bits = bits
        self.quant_method = quant_method
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.quantize:
            # Use advanced quantization for 2-bit and 3-bit
            if self.bits in [2, 3]:
                logger.info(f"Using advanced {self.bits}-bit quantization with {self.quant_method}")
                config = QuantizationConfig(bits=self.bits, method=self.quant_method)
                quantizer = AdvancedQuantizer(config)
                
                # Load model first without quantization
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                )
                
                # Apply quantization
                self.model = quantizer.quantize(base_model)
                
                if "cuda" in self.device:
                    self.model = self.model.to(self.device)
            else:
                # Use bitsandbytes for 4-bit and 8-bit
                logger.info(f"Using bitsandbytes {self.bits}-bit quantization")
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.bits == 4,
                    load_in_8bit=self.bits == 8,
                    bnb_4bit_quant_type="nf4" if self.bits == 4 else None,
                    bnb_4bit_compute_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                    bnb_4bit_use_double_quant=True if self.bits == 4 else False,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                    quantization_config=quantization_config,
                    device_map="auto" if "cuda" in self.device else None,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            ).to(self.device)

        logger.info(f"Model loaded on {self.device} with {self.bits}-bit precision")
        return self.model, self.tokenizer

    def save_shard(self, shard_index: int, layers: list, output_dir: str = "shards"):
        os.makedirs(output_dir, exist_ok=True)
        shard_path = Path(output_dir) / f"shard_{shard_index}.pt"
        torch.save({"layers": layers, "config": self.model.config}, shard_path)
        logger.info(f"Saved shard {shard_index} to {shard_path}")
        return shard_path
