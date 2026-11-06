import grpc
from concurrent import futures
import torch
from hivemind.net.proto import inference_pb2, inference_pb2_grpc
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)

class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, shard_index: int, shard_layers: list):
        self.shard_index = shard_index
        self.shard_layers = shard_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def ProcessShard(self, request, context):
        try:
            input_tensor = torch.load(request.input_data)
            x = input_tensor
            for name, layer in self.shard_layers:
                x = layer(x.to(self.device))
            torch.save(x.cpu(), "output.pt")
            with open("output.pt", "rb") as f:
                output_bytes = f.read()
            return inference_pb2.InferenceResponse(output_data=output_bytes)
        except Exception as e:
            logger.error(f"Error processing shard {self.shard_index}: {e}")
            return inference_pb2.InferenceResponse(error=str(e))

def serve(shard_index: int, shard_layers: list, port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServicer_to_server(
        InferenceServicer(shard_index, shard_layers), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Server for shard {shard_index} started on port {port}")
    server.wait_for_termination()
