import argparse
import torch
from hivemind.core.model_loader import ModelLoader
from hivemind.core.sharder import ModelSharder
from hivemind.core.inference import LocalInferenceEngine, ShardedInferenceEngine
from hivemind.net.server import serve
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="HiveMind - Run LLMs on low-end devices")
    parser.add_argument("--mode", type=str, choices=["local", "server", "client", "swarm"], default="local")
    parser.add_argument("--model", type=str, default="microsoft/phi-2")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--prompt", type=str, default="What is the meaning of life?")
    parser.add_argument("--server-address", type=str, default="localhost:50051")
    args = parser.parse_args()

    if args.mode == "local":
        loader = ModelLoader(args.model, args.quantize, args.bits)
        model, tokenizer = loader.load()
        engine = LocalInferenceEngine(model, tokenizer)
        response = engine.generate(args.prompt)
        print(f"Response: {response}")

    elif args.mode == "server":
        loader = ModelLoader(args.model, args.quantize, args.bits)
        model, tokenizer = loader.load()
        sharder = ModelSharder(model, args.shards)
        shards = sharder.split()
        shard_index = int(input("Enter shard index (0, 1, ...): "))
        serve(shard_index, shards[shard_index], args.port)

    elif args.mode == "client":
        from hivemind.net.client import InferenceClient
        client = InferenceClient(args.server_address)
        input_tensor = torch.randn(1, 10, 2560)
        output_tensor = client.process_shard(input_tensor)
        print(f"Output shape: {output_tensor.shape}")

    elif args.mode == "swarm":
        loader = ModelLoader(args.model, args.quantize, args.bits)
        model, tokenizer = loader.load()
        sharder = ModelSharder(model, args.shards)
        shards = sharder.split()
        engine = ShardedInferenceEngine(shards, tokenizer)
        response = engine.generate(args.prompt)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
