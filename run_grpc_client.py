import argparse
import grpc
from inference_pb2 import InferenceRequest, InferenceResponse
from inference_pb2_grpc import InferenceServiceStub
import timeit
import numpy as np
from model_config import model_config
import pandas as pd
import zlib

def parse_args():
    parser = argparse.ArgumentParser(description='GRPC Client')
    parser.add_argument('-m', '--model_id', type=str, required=True, help='Model ID')
    parser.add_argument('-n', '--num_iterations', type=int, default=500, help='Number of inference iterations')
    parser.add_argument('-s', '--model_server', type=str, default='obelix193:8180', help='Server address of the model')
    parser.add_argument('-o', '--output_file', type=str, default='response_latencies_jetson.csv', help='File name to save the response latencies')
    return parser.parse_args()

def run_inference(stub, input_shape, num_iterations):
    input_data = np.random.randn(*input_shape).astype(np.float32)
    input_bytes = zlib.compress(input_data.tobytes())
    request = InferenceRequest(input=input_bytes, shape=input_shape)
    
    start_time = timeit.default_timer()
    response = stub.Predict(request)
    end_time = timeit.default_timer()
    first_response_time_ms = (end_time - start_time) * 1000
    print("Time for first inference: {:.2f} ms".format(first_response_time_ms))

    # Warmup
    print("Running warmup iterations...")
    for i in range(10):
        response = stub.Predict(request)
    
    # Run inference iterations
    print("Running {} inference iterations...".format(num_iterations))
    response_latencies = []
    for i in range(num_iterations):
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input_bytes = zlib.compress(input_data.tobytes())
        request = InferenceRequest(input=input_bytes, shape=input_shape)

        start_time = timeit.default_timer()
        response = stub.Predict(request)
        end_time = timeit.default_timer()
        response_latencies.append(end_time - start_time)
        if (i + 1) % 100 == 0:
            print("Completed {}/{} iterations".format(i + 1, num_iterations))
        
    avg_response_time_ms = (sum(response_latencies) / len(response_latencies)) * 1000
    print("Average response time: {:.2f} ms".format(avg_response_time_ms))

    df = pd.read_csv("profiles.csv", index_col=0)
    if 'first_response_time_ms' not in df.columns:
        df['first_response_time_ms'] = 0.0
    if 'avg_response_time_ms' not in df.columns:
        df['avg_response_time_ms'] = 0.0
    

    df.loc[args.model_id, 'first_response_time_ms'] = first_response_time_ms
    df.loc[args.model_id, 'avg_response_time_ms'] = avg_response_time_ms

    df.to_csv("profiles.csv", index=True)
    print("Updated row in output csv: {}".format(df.loc[args.model_id]))



def get_response_times(stub, input_shape, num_iterations, model_id, file_name):
    input_data = np.random.randn(*input_shape).astype(np.float32)
    # print(input_shape)
    input_bytes = zlib.compress(input_data.tobytes())
    request = InferenceRequest(input=input_bytes, shape=input_shape)

    # Warmup
    print("Running warmup iterations...")
    for i in range(10):
        response = stub.Predict(request)
    
    # Run inference iterations
    print("Running {} inference iterations...".format(num_iterations))
    response_latencies = []
    
    # i = 0
    # while True:
    for i in range(num_iterations):
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input_bytes = zlib.compress(input_data.tobytes())
        request = InferenceRequest(input=input_bytes, shape=input_shape)

        start_time = timeit.default_timer()
        response = stub.Predict(request)
        end_time = timeit.default_timer()
        response_latencies.append(end_time - start_time)
        if (i + 1) % 100 == 0:
            print("Completed {} iterations".format(i + 1))
        # i += 1

    # write response_latencies to a csv file
    df = pd.read_csv(file_name, index_col=0)
    df.loc[model_id] = [response_latencies]
    df.to_csv(file_name, index=True)
    return



def main():
    args = parse_args()
    channel = grpc.insecure_channel(args.model_server)
    stub = InferenceServiceStub(channel)
    _, _, _, input_shape = model_config[args.model_id]
    # run_inference(stub, input_shape, args.num_iterations)
    get_response_times(stub, input_shape, args.num_iterations, args.model_id, args.output_file)
    

if __name__ == "__main__":
    main()