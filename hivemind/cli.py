#!/usr/bin/env python3
"""
HiveMind CLI - Command Line Interface
Supports both v1.0 (Swarm Inference) and v2.0 (Model Paging)
"""
import argparse
import torch
import sys
from hivemind.core.model_loader import ModelLoader
from hivemind.core.inference import (
    LocalInferenceEngine,
    ShardedInferenceEngine,
    PagedInferenceEngine
)
from hivemind.core.sharder import ModelSharder
from hivemind.net.server import serve, CompressionType
from hivemind.utils.logger import get_logger, configure_hivemind_logging

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="HiveMind - Run LLMs on low-end devices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local inference (standard)
  python -m hivemind.cli --mode local --model phi-2 --prompt "Hello"

  # Local inference with 4-bit quantization
  python -m hivemind.cli --mode local --model phi-2 --quantize --bits 4 --prompt "Hello"

  # Paged inference (v2.0) - for running 7B+ on 16GB RAM
  python -m hivemind.cli --mode paged --model mistral-7b --quantize --bits 4 --paging --prompt "Hello"

  # Swarm inference (v1.0) - distributed across devices
  python -m hivemind.cli --mode server --model phi-2 --shards 2 --port 50051
  python -m hivemind.cli --mode client --server-address localhost:50051

  # Show help
  python -m hivemind.cli --help
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "paged", "server", "client", "swarm"],
        default="local",
        help="Inference mode: local, paged, server, client, or swarm (default: local)"
    )
    
    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="Model name (default: microsoft/phi-2)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable quantization"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Quantization bits (2, 3, 4, or 8) (default: 4)"
    )
    parser.add_argument(
        "--quant-method",
        type=str,
        default="bitsandbytes",
        choices=["bitsandbytes", "qlora", "gptq"],
        help="Quantization method (default: bitsandbytes)"
    )
    
    # Paging options (v2.0)
    parser.add_argument(
        "--paging",
        action="store_true",
        help="Enable model paging for out-of-core inference (v2.0)"
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Page size in MB for paging (default: 1000)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=4,
        help="Maximum pages to keep in RAM (default: 4)"
    )
    
    # Sharding options (v1.0)
    parser.add_argument(
        "--shards",
        type=int,
        default=2,
        help="Number of shards for distributed inference (default: 2)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port for gRPC server (default: 50051)"
    )
    
    # Inference options
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the meaning of life?",
        help="Prompt for inference (default: 'What is the meaning of life?')"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)"
    )
    
    # Network options
    parser.add_argument(
        "--server-address",
        type=str,
        default="localhost:50051",
        help="Server address for client mode (default: localhost:50051)"
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=2,  # CompressionType.ZLIB
        choices=[0, 1, 2],
        help="Compression type: 0=None, 1=GZIP, 2=ZLIB (default: 2)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: None)"
    )
    
    return parser.parse_args()


def setup_logging(args):
    """Setup logging based on command line arguments"""
    configure_hivemind_logging(
        log_level=args.log_level,
        log_file=args.log_file
    )


def local_mode(args):
    """Run local inference (standard)"""
    logger.info("Starting local inference mode")
    
    loader = ModelLoader(
        model_name=args.model,
        quantize=args.quantize,
        bits=args.bits,
        quant_method=args.quant_method,
        device=args.device if hasattr(args, 'device') else None
    )
    model, tokenizer = loader.load()
    
    engine = LocalInferenceEngine(model, tokenizer)
    
    logger.info(f"Generating response for: {args.prompt}")
    response = engine.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print(f"\n{'='*60}")
    print(f"PROMPT: {args.prompt}")
    print(f"{'='*60}")
    print(f"RESPONSE:\n{response}")
    print(f"{'='*60}")


def paged_mode(args):
    """Run paged inference (v2.0) - Out-of-Core Inference"""
    logger.info("Starting paged inference mode (v2.0)")
    
    loader = ModelLoader(
        model_name=args.model,
        quantize=args.quantize,
        bits=args.bits,
        quant_method=args.quant_method,
        enable_paging=args.paging,
        page_size_mb=args.page_size,
        max_loaded_pages=args.max_pages
    )
    model, tokenizer = loader.load()
    
    # Get the paged model
    paged_model = loader.get_paged_model()
    
    if paged_model:
        engine = PagedInferenceEngine(
            paged_model,
            tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_loaded_pages=args.max_pages
        )
        
        # Print paging stats
        stats = engine.get_stats()
        logger.info(f"Paging Statistics:")
        logger.info(f"  Total pages: {stats['total_pages']}")
        logger.info(f"  Loaded pages: {stats['loaded_pages']}")
        logger.info(f"  Disk cache size: {stats['disk_cache_size_mb']:.1f}MB")
    else:
        # Fallback to local engine if paging failed
        engine = LocalInferenceEngine(model, tokenizer)
        logger.warning("Paging not enabled, falling back to local inference")
    
    logger.info(f"Generating response for: {args.prompt}")
    
    try:
        response = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        print(f"\n{'='*60}")
        print(f"PROMPT: {args.prompt}")
        print(f"{'='*60}")
        print(f"RESPONSE:\n{response}")
        print(f"{'='*60}")
        
        # Print final stats
        if paged_model:
            final_stats = engine.get_stats()
            logger.info(f"Final Statistics:")
            logger.info(f"  Cache hits: {final_stats['cache_stats']['hits']}")
            logger.info(f"  Cache misses: {final_stats['cache_stats']['misses']}")
            logger.info(f"  Hit rate: {final_stats['cache_stats']['hit_rate']}")
        
    except Exception as e:
        logger.error(f"Error during paged inference: {e}")
        logger.info("Note: Paged inference for transformer models requires custom implementation")
        logger.info("      For now, consider using smaller models or standard inference")


def server_mode(args):
    """Run gRPC server for distributed inference (v1.0)"""
    logger.info("Starting server mode")
    
    loader = ModelLoader(
        model_name=args.model,
        quantize=args.quantize,
        bits=args.bits,
        quant_method=args.quant_method
    )
    model, tokenizer = loader.load()
    
    sharder = ModelSharder(model, args.shards)
    shards = sharder.split()
    
    shard_index = int(input("Enter shard index (0, 1, ...): "))
    
    # Convert compression type
    compression_map = {
        0: CompressionType.NONE,
        1: CompressionType.GZIP,
        2: CompressionType.ZLIB
    }
    compression = compression_map.get(args.compression, CompressionType.ZLIB)
    
    logger.info(f"Starting server for shard {shard_index} on port {args.port}")
    serve(
        shard_index=shard_index,
        shard_layers=shards[shard_index],
        port=args.port,
        compression=compression
    )


def client_mode(args):
    """Run gRPC client for testing distributed inference (v1.0)"""
    logger.info("Starting client mode")
    
    from hivemind.net.client import InferenceClient
    
    # Convert compression type
    compression_map = {
        0: CompressionType.NONE,
        1: CompressionType.GZIP,
        2: CompressionType.ZLIB
    }
    compression = compression_map.get(args.compression, CompressionType.ZLIB)
    
    client = InferenceClient(
        server_address=args.server_address,
        compression=compression
    )
    
    # Create a test tensor
    input_tensor = torch.randn(1, 10, 2560)  # Phi-2 hidden size
    
    logger.info(f"Sending request to {args.server_address}")
    output_tensor = client.process_shard(input_tensor)
    
    logger.info(f"Received response with shape: {output_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    
    # Get server stats
    stats = client.get_server_stats()
    logger.info(f"Server stats: {stats}")
    
    client.close()


def swarm_mode(args):
    """Run swarm inference (v1.0) - distributed across local shards"""
    logger.info("Starting swarm inference mode")
    
    loader = ModelLoader(
        model_name=args.model,
        quantize=args.quantize,
        bits=args.bits,
        quant_method=args.quant_method
    )
    model, tokenizer = loader.load()
    
    sharder = ModelSharder(model, args.shards)
    shards = sharder.split()
    
    engine = ShardedInferenceEngine(shards, tokenizer)
    
    logger.info(f"Generating response for: {args.prompt}")
    response = engine.generate(args.prompt, max_new_tokens=args.max_tokens)
    
    print(f"\n{'='*60}")
    print(f"PROMPT: {args.prompt}")
    print(f"{'='*60}")
    print(f"RESPONSE:\n{response}")
    print(f"{'='*60}")


def main():
    """Main entry point"""
    args = parse_args()
    setup_logging(args)
    
    logger.info("="*60)
    logger.info("HiveMind - Run LLMs on low-end devices")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Quantization: {args.bits}-bit" if args.quantize else "No quantization")
    logger.info(f"Paging: {'Enabled' if args.paging else 'Disabled'}")
    logger.info("="*60)
    
    # Route to appropriate mode handler
    mode_handlers = {
        "local": local_mode,
        "paged": paged_mode,
        "server": server_mode,
        "client": client_mode,
        "swarm": swarm_mode
    }
    
    handler = mode_handlers.get(args.mode)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error in {args.mode} mode: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
