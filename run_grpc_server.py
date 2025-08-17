from grpc import server

import grpc
from concurrent import futures
from inference_pb2 import InferenceRequest, InferenceResponse
from inference_pb2_grpc import InferenceServiceServicer, add_InferenceServiceServicer_to_server
from run_gpu_profiling import load_model, init_nvml, model_config
import argparse
import zlib
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='GRPC Server')
    parser.add_argument('-m', '--model_id', type=str, required=True, help='Model ID')
    return parser.parse_args()

class InferenceServiceServicer(InferenceServiceServicer):
    def __init__(self, session):
        self.session = session
    
    def Predict(self, request, context):
        input_data = request.input
        input_data = zlib.decompress(input_data)
        input_data = np.frombuffer(input_data, dtype=np.float32).reshape(request.shape)
        outputs = self.session.run(output_names=["output"], input_feed={"input": input_data})[0]
        outputs = zlib.compress(outputs.tobytes())
        return InferenceResponse(output=outputs, shape=request.shape)

def run_grpc_server(session):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_InferenceServiceServicer_to_server(InferenceServiceServicer(session), server)
    server.add_insecure_port('[::]:8180')
    server.start()
    print("Inference grpc server started on port 8180")
    server.wait_for_termination()


def main():
    args = parse_args()
    handle, device_name = init_nvml()
    model_family, model_variant, weights, input_shape = model_config[args.model_id]
    model, load_duration_ms, mem_usage_mb, file_size_mb, session = load_model(model_variant, weights, input_shape, handle)
    print("Model {} loaded in {:.2f} ms".format(args.model_id, load_duration_ms))
    
    # Start the server
    run_grpc_server(session)

if __name__ == "__main__":
    main()