#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import argparse
import timeit
import pandas as pd
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import gc
import numpy as np
import copy
print(trt.__version__)

def load_trt_model(engine_path) -> tuple[trt.ICudaEngine, trt.IExecutionContext, float]:
    logger = trt.Logger(trt.Logger.ERROR)
    engine, context = None, None
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        start_time = timeit.default_timer()
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        load_time = timeit.default_timer() - start_time
    return engine, context, load_time


def allocate_buffers(engine, batch_size):
    """
    Create memory buffer for both host and device
    Args:
        engine: TensorRT engine
        batch_size: Batch size
    Returns:
        h_inputs: List of host input buffers
        d_inputs: List of device input buffers
        h_outputs: List of host output buffers
        d_outputs: List of device output buffers
        bindings: List of device bindings
    """

    h_inputs, d_inputs = [], [] 
    h_outputs, d_outputs = [], []
    bindings = []
    

    print("Acclocating buffer...")
    print("Device memory needed: %d" % engine.device_memory_size)
    print("Current Batch size: %d" % batch_size)

    for i, binding in enumerate(engine):
        size = trt.volume(engine.get_tensor_shape(binding))
        dtype = trt.nptype(engine.get_tensor_dtype(binding))

        # Allocate host and device memory
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))

        # Determine input vs output based on engine input names
        if engine[i] == 'input': 
            h_inputs.append(host_mem)
            d_inputs.append(device_mem)
        else:
            h_outputs.append(host_mem)
            d_outputs.append(device_mem)

    return h_inputs, d_inputs, h_outputs, d_outputs, bindings

def prepare_context(engine, context, batch_size=1):
    h_inputs, d_inputs, h_outputs, d_outputs, bindings = allocate_buffers(engine, batch_size)
    
    for i, binding in enumerate(bindings):
        # context.set_input_shap(binding, (batch_size, *engine.get_tensor_shape(binding)))
        context.set_tensor_address(engine.get_tensor_name(i), binding)
        # context.set_tensor_address(binding, int(d_outputs[i]))
    
    return h_inputs, d_inputs, h_outputs, d_outputs, context



def run_inference(engine, context, batch_size=1, num_iterations=1000):

    h_inputs, d_inputs, h_outputs, d_outputs, context = prepare_context(engine, context, batch_size)
    
    
    input_img = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    input_size = np.prod(input_img.shape)
    h_inputs[0][:input_size] = input_img.ravel() ### Important step to copy input to device
    
    # Calculate the first inference time (cold start)
    stream = cuda.Stream()
    [cuda.memcpy_htod_async(d_in, h_in, stream) for h_in, d_in in zip(h_inputs, d_inputs)]
    start_time = timeit.default_timer()
    context.execute_async_v3(
        stream_handle=stream.handle
    )
    first_inference_time = timeit.default_timer() - start_time
    print(f"First inference time: {first_inference_time * 1000:.2f} ms")
    [cuda.memcpy_dtoh_async(h_out, d_out, stream) for h_out, d_out in zip(h_outputs, d_outputs)]
    stream.synchronize()

    
    print("Running warmup iterations...")
    for i in range(10):
        h_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        h_inputs[0][:input_size] = input_img.ravel()
        context.execute_async_v3(
            stream_handle=stream.handle
        )
        [cuda.memcpy_dtoh_async(h_out, d_out, stream) for h_out, d_out in zip(h_outputs, d_outputs)]
        stream.synchronize()


    latencies = []
    print(f"Running {num_iterations} inference iterations...")
    for i in range(num_iterations):
        h_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        h_inputs[0][:input_size] = input_img.ravel()
        start_time = timeit.default_timer()
        context.execute_async_v3(
            stream_handle=stream.handle
        )
        latency = timeit.default_timer() - start_time
        [cuda.memcpy_dtoh_async(h_out, d_out, stream) for h_out, d_out in zip(h_outputs, d_outputs)]
        stream.synchronize()
        latencies.append(latency)
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"Average inference time: {avg_latency * 1000:.2f} ms")
    print(f"Total iterations: {len(latencies)}")

    return first_inference_time, avg_latency


def main(args):
    engine_path = f"trt_models/{args.model_id}.trt"
    if not os.path.exists(engine_path):
        print(f"TRT Engine file {engine_path} does not exist")
        return
    engine, context, load_time = load_trt_model(engine_path)
    print(engine)
    print(context)
    print(f"Model {args.model_id} loaded in {load_time * 1000:.2f} ms")

    print("Profiling inference...")
    first_inference_time, avg_latency = run_inference(engine, context, num_iterations=args.num_iterations)
    
    # df = pd.read_csv(args.output_file, index_col=0)
    # if 'orin_nano_trt_load_time_ms' not in df.columns:
    #     df['orin_nano_trt_load_time_ms'] = 0.0
    
    # df.loc[args.model_id, 'orin_nano_trt_load_time_ms'] = load_time * 1000
    # df.to_csv(args.output_file, index=True)
    # print(f"Updated row in output csv: {df.loc[args.model_id]}")

    del engine
    del context
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Profiling on GPU')
    parser.add_argument("-m", "--model_id", type=str, default="efficientnet_b0", help="Model ID to run")
    parser.add_argument("-o", "--output_file", type=str, default="profiles.csv", help="Output CSV file")
    parser.add_argument("-n", "--num_iterations", type=int, default=500, help="Number of inference iterations")
    args = parser.parse_args()
    main(args)
    
    