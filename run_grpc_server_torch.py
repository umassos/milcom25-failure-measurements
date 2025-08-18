from grpc import server

import grpc
from concurrent import futures
from inference_pb2 import InferenceRequest, InferenceResponse
from inference_pb2_grpc import InferenceServiceServicer, add_InferenceServiceServicer_to_server
import argparse
import zlib
import numpy as np
import torch
import torchvision
import time
import timeit
from run_gpu_profiling import init_nvml, model_config, device, dtype


def parse_args():
    parser = argparse.ArgumentParser(description='GRPC Server')
    parser.add_argument('-m', '--model_id', type=str, required=True, help='Model ID')
    parser.add_argument('-p', '--port', type=int, default=8180, help='Port to run the server on')
    return parser.parse_args()

class InferenceServiceServicer(InferenceServiceServicer):
    def __init__(self, model):
        self.model = model
    
    def Predict(self, request, context):
        input_data = request.input
        input_data = zlib.decompress(input_data)
        input_data = np.frombuffer(input_data, dtype=np.float32).reshape(request.shape)
        
        # Convert to PyTorch tensor and move to GPU
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        
        # Run inference with PyTorch
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(input_tensor)
        
        # Convert back to numpy and compress
        outputs_np = outputs.cpu().numpy()
        outputs_compressed = zlib.compress(outputs_np.tobytes())
        
        return InferenceResponse(output=outputs_compressed, shape=list(outputs_np.shape))

def load_model(model_variant, weights):
    model = getattr(torchvision.models, model_variant)(weights=weights).to(device).to(dtype).eval()
    return model

def run_grpc_server(model, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_InferenceServiceServicer_to_server(InferenceServiceServicer(model), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    print("PyTorch inference gRPC server started on port {}".format(port))
    server.wait_for_termination()

def main():
    args = parse_args()
    handle, device_name = init_nvml()
    model_family, model_variant, weights, input_shape = model_config[args.model_id]
    
    # Load PyTorch model
    start_time=timeit.default_timer()
    model = load_model(model_variant, weights)
    end_time=timeit.default_timer()
    print("PyTorch model {} loaded in {:.2f} ms".format(args.model_id, (end_time - start_time) * 1000))
    
    # Start the server with PyTorch model
    run_grpc_server(model, args.port)

if __name__ == "__main__":
    main()