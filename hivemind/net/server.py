import grpc
from concurrent import futures
import torch
import zlib
import pickle
import os
from hivemind.net.proto import inference_pb2, inference_pb2_grpc
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)


class CompressionType:
    """Compression types for gRPC communication"""
    NONE = 0
    GZIP = 1
    ZLIB = 2


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, shard_index: int, shard_layers: list, compression: int = CompressionType.ZLIB):
        self.shard_index = shard_index
        self.shard_layers = shard_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compression = compression
        self.request_count = 0
        self.error_count = 0
        logger.info(f"InferenceServicer initialized for shard {shard_index} with compression: {compression}")

    def _compress(self, data: bytes) -> bytes:
        """Compress data based on compression type"""
        if self.compression == CompressionType.NONE:
            return data
        elif self.compression == CompressionType.ZLIB:
            return zlib.compress(data)
        elif self.compression == CompressionType.GZIP:
            import gzip
            return gzip.compress(data)
        return data

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data based on compression type"""
        if self.compression == CompressionType.NONE:
            return data
        elif self.compression == CompressionType.ZLIB:
            return zlib.decompress(data)
        elif self.compression == CompressionType.GZIP:
            import gzip
            return gzip.decompress(data)
        return data

    def ProcessShard(self, request, context):
        self.request_count += 1
        try:
            # Decompress input data if needed
            input_data = self._decompress(request.input_data)
            
            # Load input tensor
            input_tensor = torch.load(input_data)
            
            # Process through shard layers
            x = input_tensor.to(self.device)
            for name, layer in self.shard_layers:
                x = layer(x)
            
            # Save output
            output_path = f"output_shard_{self.shard_index}.pt"
            torch.save(x.cpu(), output_path)
            
            # Read and compress output
            with open(output_path, "rb") as f:
                output_bytes = f.read()
            
            compressed_output = self._compress(output_bytes)
            
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)
            
            logger.debug(f"Shard {self.shard_index}: Processed request successfully")
            return inference_pb2.InferenceResponse(
                output_data=compressed_output,
                error="",
                shard_index=self.shard_index
            )
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing shard {self.shard_index}: {e}")
            return inference_pb2.InferenceResponse(
                output_data=b"",
                error=str(e),
                shard_index=self.shard_index
            )

    def GetStats(self, request, context):
        """Get server statistics"""
        return inference_pb2.ServerStats(
            request_count=self.request_count,
            error_count=self.error_count,
            shard_index=self.shard_index
        )


def serve(shard_index: int, shard_layers: list, port: int = 50051, 
          max_workers: int = 10, compression: int = CompressionType.ZLIB):
    """
    Start gRPC server for a shard
    
    Args:
        shard_index: Index of the shard
        shard_layers: List of layers in the shard
        port: Port to listen on
        max_workers: Maximum number of worker threads
        compression: Compression type (0=None, 1=GZIP, 2=ZLIB)
    """
    # Create server with thread pool
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.so_reuseport', 1),  # Enable SO_REUSEPORT
        ]
    )
    
    # Add servicer
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(shard_index, shard_layers, compression), server
    )
    
    # Add health check servicer
    try:
        from grpc_health.v1 import health_pb2, health_pb2_grpc
        health_servicer = health_pb2_grpc.HealthServicer()
        health_servicer.set("inference", health_pb2.HealthCheckResponse.SERVING)
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    except ImportError:
        logger.warning("grpc_health not available, skipping health check servicer")
    
    # Listen on port
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    
    logger.info(f"Server for shard {shard_index} started on port {port}")
    logger.info(f"Compression: {compression}, Max workers: {max_workers}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        logger.info(f"Server for shard {shard_index} stopped")
