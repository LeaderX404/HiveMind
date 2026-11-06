import grpc
from hivemind.net.proto import inference_pb2, inference_pb2_grpc
import torch
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)

class InferenceClient:
    def __init__(self, server_address: str = "localhost:50051"):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = inference_pb2_grpc.InferenceStub(self.channel)

    def process_shard(self, input_tensor: torch.Tensor, model_id: str = "", shard_index: int = 0):
        torch.save(input_tensor, "input.pt")
        with open("input.pt", "rb") as f:
            input_bytes = f.read()
        request = inference_pb2.InferenceRequest(
            input_data=input_bytes,
            model_id=model_id,
            shard_index=shard_index
        )
        response = self.stub.ProcessShard(request)
        if response.error:
            raise RuntimeError(f"Server error: {response.error}")
        with open("output.pt", "wb") as f:
            f.write(response.output_data)
        output_tensor = torch.load("output.pt")
        return output_tensor
