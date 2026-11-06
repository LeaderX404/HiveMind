#!/usr/bin/env python3
"""
Basic inference example using HiveMind.
This demonstrates loading a model and running inference locally.
"""

from hivemind.core.model_loader import ModelLoader
from hivemind.core.inference import LocalInferenceEngine

def main():
    # Load model with 4-bit quantization
    print("Loading model...")
    loader = ModelLoader(
        model_name="microsoft/phi-2",
        quantize=True,
        bits=4
    )
    model, tokenizer = loader.load()
    
    # Create inference engine
    engine = LocalInferenceEngine(model, tokenizer)
    
    # Run inference
    prompts = [
        "What is the meaning of life?",
        "Explain artificial intelligence in simple terms.",
        "Write a short poem about programming.",
        "What are the latest advancements in AI?"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = engine.generate(prompt, max_new_tokens=100)
        print(f"Response: {response}")
        print("-" * 80)

if __name__ == "__main__":
    main()
