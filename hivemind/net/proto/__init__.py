# Generated protobuf modules
import sys
import os

# Add the proto directory to the path so imports work
_proto_dir = os.path.dirname(os.path.abspath(__file__))
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

# Import the generated modules
from . import inference_pb2
from . import inference_pb2_grpc

# Make them available at package level
InferenceRequest = inference_pb2.InferenceRequest
InferenceResponse = inference_pb2.InferenceResponse
ServerStats = inference_pb2.ServerStats
StatsRequest = inference_pb2.StatsRequest
InferenceService = inference_pb2_grpc.InferenceService
InferenceServiceServicer = inference_pb2_grpc.InferenceServiceServicer
add_InferenceServiceServicer_to_server = inference_pb2_grpc.add_InferenceServiceServicer_to_server

__all__ = [
    'inference_pb2',
    'inference_pb2_grpc',
    'InferenceRequest',
    'InferenceResponse',
    'ServerStats',
    'StatsRequest',
    'InferenceService',
    'InferenceServiceServicer',
    'add_InferenceServiceServicer_to_server',
]
