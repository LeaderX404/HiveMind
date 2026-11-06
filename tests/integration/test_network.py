"""
Integration tests for Network components (gRPC)
"""
import io
import os
import zlib
import pytest
import torch
import grpc
from concurrent import futures
from unittest.mock import MagicMock, patch, ANY
from hivemind.net.server import InferenceServicer, serve
from hivemind.net.client import InferenceClient
from hivemind.net.proto import inference_pb2, inference_pb2_grpc


class TestInferenceServicer:
    """Test cases for InferenceServicer"""
    
    def test_init(self):
        """Test servicer initialization"""
        mock_layers = [("layer1", MagicMock())]
        servicer = InferenceServicer(shard_index=0, shard_layers=mock_layers)
        
        assert servicer.shard_index == 0
        assert servicer.shard_layers == mock_layers
        assert servicer.device in ["cuda", "cpu"]
    
    def test_process_shard_success(self):
        """Test successful shard processing"""
        # Create mock layers
        mock_layer = MagicMock()
        mock_layer.return_value = torch.tensor([[1.0, 2.0, 3.0]])
        
        mock_layers = [("layer1", mock_layer)]
        servicer = InferenceServicer(shard_index=0, shard_layers=mock_layers)
        
        # Create request: the wire protocol is zlib-compressed (both client
        # and servicer default to CompressionType.ZLIB), so the request
        # payload must be compressed the same way a real client would.
        input_tensor = torch.tensor([[1.0, 2.0, 3.0]])
        torch.save(input_tensor, "test_input.pt")
        
        with open("test_input.pt", "rb") as f:
            input_bytes = zlib.compress(f.read())
        
        request = inference_pb2.InferenceRequest(
            input_data=input_bytes,
            model_id="test-model",
            shard_index=0
        )
        
        # Mock context
        context = MagicMock()
        
        # Process request
        response = servicer.ProcessShard(request, context)
        
        # Should return response with output_data
        assert response.error == ""
        assert len(response.output_data) > 0
        
        # The response payload is zlib-compressed as well: verify the
        # full roundtrip instead of only checking the payload size.
        output_tensor = torch.load(io.BytesIO(zlib.decompress(response.output_data)))
        assert torch.allclose(output_tensor, torch.tensor([[1.0, 2.0, 3.0]]))
        
        # Cleanup
        if os.path.exists("test_input.pt"):
            os.remove("test_input.pt")


class TestInferenceClient:
    """Test cases for InferenceClient"""
    
    @patch('hivemind.net.client.grpc.insecure_channel')
    def test_init(self, mock_channel):
        """Test client initialization"""
        mock_channel_instance = MagicMock()
        mock_channel.return_value = mock_channel_instance
        
        client = InferenceClient(server_address="localhost:50051")
        
        # The channel is created with the address plus explicit gRPC
        # options (message limits and keepalive), so pin only the
        # address here and accept any options payload.
        mock_channel.assert_called_once_with("localhost:50051", options=ANY)
        assert client.channel == mock_channel_instance
    
    @patch('hivemind.net.client.grpc.insecure_channel')
    @patch('hivemind.net.client.torch.save')
    @patch('hivemind.net.client.torch.load')
    def test_process_shard_success(self, mock_load, mock_save, mock_channel):
        """Test successful shard processing by client"""
        # Setup mocks
        mock_channel_instance = MagicMock()
        mock_channel.return_value = mock_channel_instance
        
        mock_stub = MagicMock()
        
        # Setup torch mocks
        mock_save.return_value = None
        mock_load.return_value = torch.tensor([[1.0, 2.0, 3.0]])
        
        # Setup response: the server sends zlib-compressed payloads, so
        # the mocked response must be compressed too, otherwise the
        # client-side decompression step fails.
        mock_response = MagicMock()
        mock_response.error = ""
        mock_response.shard_index = 0
        mock_response.output_data = zlib.compress(b"test_output")
        mock_stub.ProcessShard.return_value = mock_response
        
        # Patch the stub creation: the service in inference.proto is
        # InferenceService, so the generated stub class is
        # InferenceServiceStub (not InferenceStub).
        with patch('hivemind.net.client.inference_pb2_grpc.InferenceServiceStub', return_value=mock_stub):
            client = InferenceClient(server_address="localhost:50051")
            
            # Mock open for reading/writing
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_file.read.return_value = b"test_output"
                mock_file.write.return_value = None
                mock_open.return_value.__enter__ = MagicMock(return_value=mock_file)
                mock_open.return_value.__exit__ = MagicMock(return_value=None)
                
                input_tensor = torch.tensor([[1.0, 2.0, 3.0]])
                output_tensor = client.process_shard(input_tensor)
        
        # Check that stub was called
        mock_stub.ProcessShard.assert_called_once()
        
        # Check that output is a tensor
        assert isinstance(output_tensor, torch.Tensor)


class TestNetworkIntegration:
    """Integration tests for the full network stack"""
    
    def test_grpc_communication(self):
        """Test full gRPC communication between client and server"""
        # This test requires actual gRPC server setup
        # For now, we'll just verify the components can be created
        
        # Create mock layers
        mock_layer = MagicMock()
        mock_layer.return_value = torch.tensor([[1.0, 2.0, 3.0]])
        mock_layers = [("layer1", mock_layer)]
        
        # Create servicer
        servicer = InferenceServicer(shard_index=0, shard_layers=mock_layers)
        
        # Verify servicer can be created
        assert servicer is not None
        assert servicer.shard_index == 0
        
        # Create client
        client = InferenceClient(server_address="localhost:50051")
        
        # Verify client can be created
        assert client is not None
