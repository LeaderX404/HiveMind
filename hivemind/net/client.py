import grpc
import torch
import zlib
import time
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


class InferenceClient:
    """
    gRPC client for inference requests
    Supports compression, retries, TLS and token authentication
    """
    
    def __init__(self, server_address: str = "localhost:50051", 
                 compression: int = CompressionType.ZLIB,
                 max_retries: int = 3,
                 timeout: float = 30.0,
                 auth_token: Optional[str] = None,
                 root_certificates: Optional[str] = None):
        """
        Initialize the inference client
        
        Args:
            server_address: Address of the gRPC server
            compression: Compression type (0=None, 1=GZIP, 2=ZLIB)
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout for requests in seconds
            auth_token: Shared secret sent in the x-hivemind-token
                metadata header on every request
            root_certificates: Path to the PEM bundle of trusted CA
                certificates. When provided, the client connects over
                TLS via grpc.secure_channel.
        """
        self.server_address = server_address
        self.compression = compression
        self.max_retries = max_retries
        self.timeout = timeout
        self.auth_token = auth_token
        self.root_certificates = root_certificates
        self.channel = None
        self.stub = None
        self._connect()
        
        logger.info(f"InferenceClient connected to {server_address} with compression: {compression}")

    def _connect(self):
        """Establish connection to server"""
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.keepalive_time_ms', 10000),  # 10 seconds
            ('grpc.keepalive_timeout_ms', 5000),  # 5 seconds
        ]
        try:
            if self.root_certificates:
                # TLS: trust anchors come from the provided CA bundle.
                with open(self.root_certificates, "rb") as f:
                    ca_pem = f.read()
                credentials = grpc.ssl_channel_credentials(root_certificates=ca_pem)
                self.channel = grpc.secure_channel(self.server_address, credentials, options=options)
                logger.info(f"Connected to {self.server_address} over TLS")
            else:
                # No CA bundle configured: plain-text channel. This is
                # only acceptable on trusted networks; warn loudly.
                logger.warning(
                    "Connecting WITHOUT TLS (no root_certificates set): "
                    "traffic is unencrypted and open to interception"
                )
                self.channel = grpc.insecure_channel(self.server_address, options=options)
            self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
        except Exception as e:
            logger.error(f"Failed to connect to {self.server_address}: {e}")
            raise

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

    def _auth_metadata(self):
        """Metadata headers sent with every request."""
        if self.auth_token:
            return (("x-hivemind-token", self.auth_token),)
        return None

    def _retry_request(self, func, *args, **kwargs):
        """Retry a request with exponential backoff"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except grpc.RpcError as e:
                last_error = e
                wait_time = (2 ** attempt) * 0.1  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                logger.warning(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                
                # Reconnect if channel is closed
                if self.channel is None or self.channel._closed:
                    self._connect()
        
        raise last_error if last_error else Exception("Request failed after retries")

    def process_shard(self, input_tensor: torch.Tensor, model_id: str = "", 
                      shard_index: int = 0) -> torch.Tensor:
        """
        Process input tensor through a shard
        
        Args:
            input_tensor: Input tensor to process
            model_id: Model identifier
            shard_index: Shard index
            
        Returns:
            Output tensor from the shard
        """
        # Pack the tensor with the safe container format: no
        # torch.save/pickle anywhere on the wire, in either direction.
        compressed_input = self._compress(serialize_tensor(input_tensor))
        
        # Create request
        request = inference_pb2.InferenceRequest(
            input_data=compressed_input,
            model_id=model_id,
            shard_index=shard_index
        )
        
        # Send request with retry (auth token travels as metadata)
        def make_request():
            return self.stub.ProcessShard(
                request,
                timeout=self.timeout,
                metadata=self._auth_metadata(),
            )
        
        response = self._retry_request(make_request)
        
        # Check for errors
        if response.error:
            raise RuntimeError(f"Server error (shard {response.shard_index}): {response.error}")
        
        # Decompress and rebuild the output tensor (safe container
        # format: bytes are only read, never executed)
        output_tensor = deserialize_tensor(self._decompress(response.output_data))
        
        logger.debug(f"Processed shard {response.shard_index} successfully")
        return output_tensor

    def get_server_stats(self) -> dict:
        """Get server statistics"""
        try:
            request = inference_pb2.StatsRequest()
            response = self.stub.GetStats(
                request, timeout=5.0, metadata=self._auth_metadata()
            )
            return {
                "request_count": response.request_count,
                "error_count": response.error_count,
                "shard_index": response.shard_index
            }
        except Exception as e:
            logger.error(f"Failed to get server stats: {e}")
            return {"error": str(e)}

    def check_health(self) -> bool:
        """Check if server is healthy"""
        try:
            from grpc_health.v1 import health_pb2, health_pb2_grpc
            health_stub = health_pb2_grpc.HealthStub(self.channel)
            request = health_pb2.HealthCheckRequest(service="inference")
            response = health_stub.Check(request, timeout=5.0)
            return response.status == health_pb2.HealthCheckResponse.SERVING
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def close(self):
        """Close the connection"""
        if self.channel and not self.channel._closed:
            self.channel.close()
            logger.info(f"Disconnected from {self.server_address}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
