import os
import time
import json
import torch
import thop
# import psutil
import tracemalloc
# from huggingface_hub import login
from PIL import Image
# from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
# from transformers.image_utils import load_image
# from config import model_name, dataset, dataset_json, models_directory
import csv, argparse
import argparse
import builtins
from typing import List
from model_config import model_config
import torchvision.models
import timeit
# import onnxruntime as ort
import pandas as pd
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Workaround for broken MINICPM code that doesn't import List
builtins.List = List

# ============ Parse CLI Arguments ============
def parse_args():
    parser = argparse.ArgumentParser(description='Model Profiling on GPU')
    parser.add_argument("-m", "--model_id", type=str, default=None, help="Model ID to run")
    parser.add_argument("-o", "--output_file", type=str, default="profiles.csv", help="Output CSV file")
    parser.add_argument("-n", "--num_iterations", type=int, default=500, help="Number of inference iterations")
    return parser.parse_args()

# ============ Setup ============
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(device)


# ============ Load Model ============
def convert_to_onnx(model_variant, weights, input_shape):
    
    model = getattr(torchvision.models, model_variant)(weights=weights).eval()

    onnx_path = os.path.join("onnx_models", f"{model_variant}.onnx")
    if not os.path.exists(onnx_path):
        torch.onnx.export(
            model,
            torch.randn(input_shape),
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=13
        )

    # file_size_mb = os.path.getsize(onnx_path) / (1024**2)

    # mem_before_load_mb = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
    # try:
    #     start_time = timeit.default_timer()
    #     session = ort.InferenceSession(
    #         onnx_path, 
    #         providers=["CUDAExecutionProvider"]
    #     )
    #     load_duration_ms = (timeit.default_timer() - start_time) * 1000
    #     memory_after_load_mb = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
    #     print(f"Mem before load: {mem_before_load_mb}, Mem after load: {memory_after_load_mb}")
    #     mem_usage_mb = memory_after_load_mb - mem_before_load_mb
    # except Exception as e:
    #     raise RuntimeError(f"Error loading model {model_variant}: {e}")
 
    # return model, load_duration_ms, mem_usage_mb, file_size_mb, session


def load_trt_model(model_variant, input_shape):
    trt_path = os.path.join("trt_models", f"{model_variant}.trt")
    with open(trt_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    output_shape = (1, 1000)

    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return engine, context, d_input, d_output, stream, h_input, h_output


def run_inference(input_shape, num_iterations, context, d_input, d_output, stream, h_input, h_output):

    input_data = np.random.randn(input_shape)
    latencies = []

    print("Running warmup iterations...")
    for i in range(10):
        h_input = np.random.randn(*input_shape).astype(np.float32)
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(
            bindings=[int(d_input), int(d_output)],
            stream_handle=stream.handle
        )
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    print(f"Running {num_iterations} inference iterations...")
    for i in range(num_iterations):
        start = time.time()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(
            bindings=[int(d_input), int(d_output)],
            stream_handle=stream.handle
        )
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        latency = time.time() - start
        latencies.append(latency)
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"Average inference time: {avg_latency:.6f} seconds")
    print(f"Total iterations: {len(latencies)}")
    
    return avg_latency


# ============ Main ============
def main():
    
    args = parse_args()
    print(args)
    
    model_family, model_variant, weights, input_shape = model_config[args.model_id]
    print('Converting to ONNX... {}'.format(model_variant))
    convert_to_onnx(model_variant, weights, input_shape)
    # model, load_duration_ms, mem_usage_mb, file_size_mb, session = load_model(model_variant, weights, input_shape, handle)
    # gflops, mparams = get_flops(model, input_shape)
    # metrics = {
    #     "model_family": model_family,
    #     "model_variant": model_variant,
    #     "model_input_shape": input_shape,  # (batch_size, channels, height, width)
    #     "device": device_name,
    #     "onnx_file_size_mb": file_size_mb,  # MB (ONNX)
    #     "onnx_load_duration_ms": load_duration_ms,  # ms
    #     "onnx_gpu_load_memory_mb": mem_usage_mb,  # MB (GPU)
    #     "onnx_gflops": gflops,  # GFLOPs
    #     "onnx_mparams": mparams,  # Millions of parameters
    # }

    #  csv_path = args.output_file
    # write_header = not os.path.exists(csv_path)

    # with open(csv_path, "a", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=metrics.keys())
    #     if write_header:
    #         writer.writeheader()
    #     writer.writerow(metrics)

    # print(f"Metrics saved to {csv_path}")

    # first_inference_time, avg_latency = run_inference(model, input_shape, args.num_iterations, session, handle)
    # df = pd.read_csv(args.output_file, index_col=0)
    # # print(df.head(10))
    
    # # Fix: Properly add new columns and update values
    # if 'first_inference_time_ms' not in df.columns:
    #     df['first_inference_time_ms'] = 0.0
    # if 'avg_latency_ms' not in df.columns:
    #     df['avg_latency_ms'] = 0.0
    
    # # Update the specific row values
    # df.loc[args.model_id, 'first_inference_time_ms'] = first_inference_time * 1000  # Convert to ms
    # df.loc[args.model_id, 'avg_latency_ms'] = avg_latency * 1000  # Convert to ms
    
    # # print(df.head(5))
    # # Save the updated DataFrame
    # df.to_csv(args.output_file, index=True)
    # print("Update row in output csv: {}".format(df.loc[args.model_id]))
    

if __name__ == "__main__":
    main()