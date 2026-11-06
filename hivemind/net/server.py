import grpc
from concurrent import futures
import torch
import zlib
from typing import Optional
from hivemind.net.proto import inference_pb2, inference_pb2_grpc
from hivemind.net.serialization import serialize_tensor, deserialize_tensor
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)


class CompressionType:
    """Compression types for gRPC communication"""
    NONE = 0
    GZIP = 1
    ZLIB = 2


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, shard_index: int, shard_layers: list, compression: int = CompressionType.ZLIB,
                 auth_token: Optional[str] = None):
        """
        Args:
            shard_index: Index of the shard this servicer owns
            shard_layers: List of (name, layer) tuples
            compression: Compression type (0=None, 1=GZIP, 2=ZLIB)
            auth_token: If set, clients must present it in the
                "x-hivemind-token" gRPC metadata header. None disables
                auth (only acceptable on trusted networks).
        """
        self.shard_index = shard_index
        self.shard_layers = shard_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compression = compression
        self.auth_token = auth_token
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

    def _authorized(self, context) -> bool:
        """Check the client auth token, if one is configured."""
        if self.auth_token is None:
            return True
        try:
            metadata = dict(context.invocation_metadata() or ())
        except Exception:
            return False
        return metadata.get("x-hivemind-token") == self.auth_token

    def ProcessShard(self, request, context):
        self.request_count += 1
        
        # Auth gate: must pass before the payload is touched at all.
        if not self._authorized(context):
            self.error_count += 1
            logger.warning(
                f"Shard {self.shard_index}: rejected unauthorized request"
            )
            return inference_pb2.InferenceResponse(
                output_data=b"",
                error="unauthorized: invalid or missing auth token",
                shard_index=self.shard_index
            )
        
        try:
            # Decompress input data if needed
            input_data = self._decompress(request.input_data)
            
            # Rebuild the tensor with the safe container format.
            # No torch.load / pickle anywhere on the network path: the
            # payload is only ever read as bytes (see serialization.py).
            input_tensor = deserialize_tensor(input_data)
            
            # Process through shard layers
            x = input_tensor.to(self.device)
            for name, layer in self.shard_layers:
                x = layer(x)
            
            # Serialize the output in memory with the safe container
            # format and compress it.
            compressed_output = self._compress(serialize_tensor(x.cpu()))
            
            logger.debug(f"Shard {self.shard_index}: Processed request successfully")
            return inference_pb2.InferenceResponse(
                output_data=compressed_output,
                error="",
                shard_index=self.shard_index
            )
        except Exception as e:
            self.error_count += 1
            # Log the full details locally, return only a generic
            # message to the client: exception text can leak file
            # paths, library versions and infrastructure details.
            logger.error(
                f"Error processing shard {self.shard_index}: {e}",
                exc_info=True,
            )
            return inference_pb2.InferenceResponse(
                output_data=b"",
                error="internal error while processing shard",
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
          max_workers: int = 10, compression: int = CompressionType.ZLIB,
          auth_token: Optional[str] = None,
          ssl_cert_file: Optional[str] = None,
          ssl_key_file: Optional[str] = None):
    """
    Start gRPC server for a shard
    
    Args:
        shard_index: Index of the shard
        shard_layers: List of layers in the shard
        port: Port to listen on
        max_workers: Maximum number of worker threads
        compression: Compression type (0=None, 1=GZIP, 2=ZLIB)
        auth_token: Shared secret required from clients in the
            "x-hivemind-token" metadata header (None disables auth)
        ssl_cert_file: Path to the TLS certificate (PEM). Together
            with ssl_key_file, enables TLS on the server.
        ssl_key_file: Path to the TLS private key (PEM)
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
        InferenceServicer(shard_index, shard_layers, compression, auth_token=auth_token), server
    )
    
    # Add health check servicer
    try:
        from grpc_health.v1 import health_pb2, health_pb2_grpc
        health_servicer = health_pb2_grpc.HealthServicer()
        health_servicer.set("inference", health_pb2.HealthCheckResponse.SERVING)
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    except ImportError:
        logger.warning("grpc_health not available, skipping health check servicer")
    
    # Listen on port: TLS when a certificate is provided, plain
    # otherwise (with a loud warning: an unauthenticated, unencrypted
    # endpoint must never be exposed to an untrusted network).
    if ssl_cert_file and ssl_key_file:
        with open(ssl_key_file, "rb") as f:
            key_bytes = f.read()
        with open(ssl_cert_file, "rb") as f:
            cert_bytes = f.read()
        server_credentials = grpc.ssl_server_credentials(
            ((key_bytes, cert_bytes),)
        )
        server.add_secure_port(f"[::]:{port}", server_credentials)
        logger.info(f"TLS enabled for shard {shard_index} on port {port}")
    else:
        logger.warning(
            "Starting gRPC server WITHOUT TLS and WITHOUT encryption: "
            "never expose this endpoint to an untrusted network"
        )
        server.add_insecure_port(f"[::]:{port}")
    server.start()
    
    logger.info(f"Server for shard {shard_index} started on port {port}")
    logger.info(f"Compression: {compression}, Max workers: {max_workers}, Auth: {'on' if auth_token else 'OFF'}, TLS: {'on' if ssl_cert_file else 'OFF'}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        logger.info(f"Server for shard {shard_index} stopped")
