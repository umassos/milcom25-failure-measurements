from grpc import server

import grpc
from concurrent import futures
from inference_pb2 import InferenceRequest, InferenceResponse
from inference_pb2_grpc import InferenceServiceServicer, add_InferenceServiceServicer_to_server
from model_config import model_config
import argparse
import zlib
import numpy as np
import timeit
import torch_tensorrt
from load_torch_trt import load_torch_trt_model
import torch

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
        input_data = torch.from_numpy(input_data).cuda()
        
        service_time = 0.0
        with torch.no_grad():
            start_time = timeit.default_timer()
            outputs = self.model(input_data)
            service_time = timeit.default_timer() - start_time

        outputs = zlib.compress(outputs.cpu().numpy().tobytes())
        return InferenceResponse(output=outputs, shape=request.shape, service_time=service_time)

def run_grpc_server(model, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_InferenceServiceServicer_to_server(InferenceServiceServicer(model), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    print("Inference grpc server started on port {}".format(port))
    server.wait_for_termination()


def main():
    args = parse_args()
    model_family, model_variant, weights, input_shape = model_config[args.model_id]
    start_time = timeit.default_timer()
    model = load_torch_trt_model(model_variant, input_shape)
    load_duration = timeit.default_timer() - start_time
    print("Model {} loaded in {:.2f} ms".format(args.model_id, load_duration * 1000))
    
    # Start the server
    run_grpc_server(model, args.port)

if __name__ == "__main__":
    main()