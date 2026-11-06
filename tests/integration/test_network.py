"""
Integration tests for Network components (gRPC)
"""
import io
import zlib
import pytest
import torch
import grpc
from concurrent import futures
from unittest.mock import MagicMock, patch, ANY
from hivemind.net.server import InferenceServicer, serve
from hivemind.net.client import InferenceClient
from hivemind.net.serialization import serialize_tensor, deserialize_tensor
from hivemind.net.proto import inference_pb2, inference_pb2_grpc


def make_request(payload_tensor=None, **kwargs):
    """Build a wire-protocol request: zlib-compressed safe container."""
    tensor = payload_tensor if payload_tensor is not None else torch.tensor([[1.0, 2.0, 3.0]])
    return inference_pb2.InferenceRequest(
        input_data=zlib.compress(serialize_tensor(tensor)),
        model_id="test-model",
        shard_index=0,
        **kwargs,
    )


def make_context(token=None):
    """Build a mock gRPC context with optional auth metadata."""
    context = MagicMock()
    if token is None:
        context.invocation_metadata.return_value = ()
    else:
        context.invocation_metadata.return_value = (("x-hivemind-token", token),)
    return context


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
        """Test successful shard processing over the safe wire format"""
        mock_layer = MagicMock()
        mock_layer.return_value = torch.tensor([[1.0, 2.0, 3.0]])
        
        mock_layers = [("layer1", mock_layer)]
        servicer = InferenceServicer(shard_index=0, shard_layers=mock_layers)
        
        response = servicer.ProcessShard(make_request(), make_context())
        
        # Should return a successful response
        assert response.error == ""
        assert len(response.output_data) > 0
        
        # Verify the full compress -> process -> decompress roundtrip.
        # deserialize_tensor is pickle-free: it only reads bytes.
        output_tensor = deserialize_tensor(zlib.decompress(response.output_data))
        assert torch.allclose(output_tensor, torch.tensor([[1.0, 2.0, 3.0]]))
    
    def test_process_shard_rejects_legacy_pickle_payload(self):
        """Legacy torch.save/pickle payloads must be rejected: unpickling
        attacker-controlled bytes is remote code execution."""
        mock_layers = [("layer1", MagicMock())]
        servicer = InferenceServicer(shard_index=0, shard_layers=mock_layers)
        
        # Old-style payload: torch.save into a buffer, compressed like
        # the old wire format did.
        buffer = io.BytesIO()
        torch.save(torch.tensor([[1.0, 2.0, 3.0]]), buffer)
        request = inference_pb2.InferenceRequest(
            input_data=zlib.compress(buffer.getvalue()),
            model_id="test-model",
            shard_index=0,
        )
        
        response = servicer.ProcessShard(request, make_context())
        
        # Rejected with a generic error; details stay in server logs.
        assert response.error != ""
        assert len(response.output_data) == 0
    
    def test_process_shard_rejects_tampered_payload(self):
        """A corrupted container must not deserialize into a tensor."""
        mock_layers = [("layer1", MagicMock())]
        servicer = InferenceServicer(shard_index=0, shard_layers=mock_layers)
        
        # Valid magic and a well-formed header, but the tensor
        # bytes are shorter than shape/dtype require: the
        # byte-length check must reject the mismatch instead of
        # reading out of bounds.
        header = b'{"shape": [1, 3], "dtype": "float32"}'
        payload = zlib.compress(
            b"HMT1"
            + len(header).to_bytes(4, "little")
            + header
            + b"\x00\x00\x80",
        )
        request = inference_pb2.InferenceRequest(
            input_data=payload,
            model_id="test-model",
            shard_index=0,
        )
        
        response = servicer.ProcessShard(request, make_context())
        assert response.error != ""
        assert len(response.output_data) == 0
    
    def test_process_shard_requires_valid_token(self):
        """With auth enabled, a wrong or missing token must be rejected
        before the payload is even touched."""
        mock_layers = [("layer1", MagicMock())]
        servicer = InferenceServicer(shard_index=0, shard_layers=mock_layers, auth_token="secret")
        
        # Missing token
        response = servicer.ProcessShard(make_request(), make_context())
        assert "unauthorized" in response.error
        assert len(response.output_data) == 0
        
        # Wrong token
        response = servicer.ProcessShard(make_request(), make_context(token="wrong"))
        assert "unauthorized" in response.error
        assert len(response.output_data) == 0
        
        # The mock layer must never have run for rejected requests
        mock_layers[0][1].assert_not_called()
    
    def test_process_shard_accepts_valid_token(self):
        """With auth enabled, the correct token grants access."""
        mock_layer = MagicMock()
        mock_layer.return_value = torch.tensor([[1.0, 2.0, 3.0]])
        mock_layers = [("layer1", mock_layer)]
        servicer = InferenceServicer(shard_index=0, shard_layers=mock_layers, auth_token="secret")
        
        response = servicer.ProcessShard(make_request(), make_context(token="secret"))
        assert response.error == ""
        output_tensor = deserialize_tensor(zlib.decompress(response.output_data))
        assert torch.allclose(output_tensor, torch.tensor([[1.0, 2.0, 3.0]]))


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
    def test_process_shard_sends_token_metadata(self, mock_channel):
        """The auth token must travel in the request metadata and the
        response must roundtrip through the safe container format."""
        mock_channel_instance = MagicMock()
        mock_channel.return_value = mock_channel_instance
        
        mock_stub = MagicMock()
        
        # Server-style response: zlib-compressed safe container.
        mock_response = MagicMock()
        mock_response.error = ""
        mock_response.shard_index = 0
        mock_response.output_data = zlib.compress(
            serialize_tensor(torch.tensor([[1.0, 2.0, 3.0]]))
        )
        mock_stub.ProcessShard.return_value = mock_response
        
        with patch('hivemind.net.client.inference_pb2_grpc.InferenceServiceStub', return_value=mock_stub):
            client = InferenceClient(server_address="localhost:50051", auth_token="secret")
            input_tensor = torch.tensor([[1.0, 2.0, 3.0]])
            output_tensor = client.process_shard(input_tensor)
        
        # The stub was called exactly once, with the token in metadata
        mock_stub.ProcessShard.assert_called_once()
        kwargs = mock_stub.ProcessShard.call_args.kwargs
        assert ("x-hivemind-token", "secret") in kwargs["metadata"]
        
        # The request payload must be the safe container (and NOT a
        # pickle: pickle streams start with protocol markers, never
        # with our HMT1 magic).
        request = mock_stub.ProcessShard.call_args.args[0]
        assert zlib.decompress(request.input_data)[:4] == b"HMT1"
        
        # Output is rebuilt from the safe container
        assert isinstance(output_tensor, torch.Tensor)
        assert torch.allclose(output_tensor, torch.tensor([[1.0, 2.0, 3.0]]))


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
