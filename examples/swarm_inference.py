#!/usr/bin/env python3
"""
Swarm inference example using HiveMind.
This demonstrates distributed inference across multiple shards.
"""

from hivemind.core.model_loader import ModelLoader
from hivemind.core.sharder import ModelSharder
from hivemind.core.inference import ShardedInferenceEngine

def main():
    # Load model
    print("Loading model...")
    loader = ModelLoader(
        model_name="microsoft/phi-2",
        quantize=True,
        bits=4
    )
    model, tokenizer = loader.load()
    
    # Split model into shards
    num_shards = 2
    print(f"Splitting model into {num_shards} shards...")
    sharder = ModelSharder(model, num_shards)
    shards = sharder.split()
    
    # Create sharded inference engine
    engine = ShardedInferenceEngine(shards, tokenizer)
    
    # Run inference
    prompts = [
        "What is machine learning?",
        "Tell me about neural networks.",
        "How do transformers work?"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = engine.generate(prompt, max_new_tokens=50)
        print(f"Response: {response}")
        print("-" * 80)

if __name__ == "__main__":
    main()
