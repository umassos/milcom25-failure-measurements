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
from pynvml import *
import argparse
import builtins
from typing import List
from model_config import model_config
import torchvision.models
import timeit
import onnxruntime as ort
import pandas as pd

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


# ============ Metrics ============

def init_nvml():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # assuming single-GPU
    device_name = nvmlDeviceGetName(handle)
    return handle, device_name

def get_gpu_memory_and_util(handle):
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    util = nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_mem_used_mb": mem_info.used / 1024**2,
        "gpu_util_percent": util.gpu
    }

# ============ Load Model ============
def load_model(model_variant, weights, input_shape, handle):
    
    model = getattr(torchvision.models, model_variant)(weights=weights).eval()

    onnx_path = os.path.join("onnx_models", f"{model_variant}.onnx")
    if not os.path.exists(onnx_path):
        torch.onnx.export(
            model,
            torch.randn(input_shape),
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=12
        )

    file_size_mb = os.path.getsize(onnx_path) / (1024**2)

    mem_before_load_mb = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
    try:
        start_time = timeit.default_timer()
        session = ort.InferenceSession(
            onnx_path, 
            providers=["CUDAExecutionProvider"]
        )
        load_duration_ms = (timeit.default_timer() - start_time) * 1000
        memory_after_load_mb = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
        print(f"Mem before load: {mem_before_load_mb}, Mem after load: {memory_after_load_mb}")
        mem_usage_mb = memory_after_load_mb - mem_before_load_mb
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_variant}: {e}")
 
    return model, load_duration_ms, mem_usage_mb, file_size_mb, session


def get_flops(model, input_shape):
    dummy_input = torch.randn(input_shape)
    flops, params = thop.profile(model, inputs=(dummy_input,))
    return flops/1e9, params/1e6 # GFLOPs, Millions of parameters

# ============ Load Dataset ============
def load_dataset(json_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    return list(raw_data.values())

def run_inference(model, input_shape, num_iterations, session, handle):
    model.eval()
    total_latency = 0
    total_gpu_util = 0
    total_gpu_mem = 0
    total_cpu_util = 0
    total_cpu_mem = 0

    # tracemalloc.start()
    gpu_before = get_gpu_memory_and_util(handle)
    # print("GPU Before: ", gpu_before)
    # cpu_before = get_cpu_memory_and_util()

    input_data = torch.randn(input_shape)
    latencies = []

    # Clear GPU cache before first inference
    torch.cuda.empty_cache()
    
    # Calculate the first inference time (cold start)
    start_time = timeit.default_timer()
    outputs = session.run(output_names=["output"], input_feed={"input": input_data.numpy()})
    first_inference_time = timeit.default_timer() - start_time
    print(f"First inference time: {first_inference_time:.6f} seconds")

    # Run warmup iterations (don't measure these)
    print("Running warmup iterations...")
    for i in range(10):
        outputs = session.run(output_names=["output"], input_feed={"input": input_data.numpy()})
    
    # Run actual inference iterations and measure
    print(f"Running {num_iterations} inference iterations...")
    for i in range(num_iterations):
        start = time.time()
        outputs = session.run(output_names=["output"], input_feed={"input": input_data.numpy()})
        latency = time.time() - start
        latencies.append(latency)
        
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")

    # tracemalloc.stop()

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"Average inference time: {avg_latency:.6f} seconds")
    print(f"Total iterations: {len(latencies)}")

    return first_inference_time, avg_latency

# ============ Main ============
def main():
    
    args = parse_args()
    print(args)
    
    handle, device_name = init_nvml()
    model_family, model_variant, weights, input_shape = model_config[args.model_id]
    model, load_duration_ms, mem_usage_mb, file_size_mb, session = load_model(model_variant, weights, input_shape, handle)
    gflops, mparams = get_flops(model, input_shape)
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

    first_inference_time, avg_latency = run_inference(model, input_shape, args.num_iterations, session, handle)
    df = pd.read_csv(args.output_file, index_col=0)
    # print(df.head(10))
    
    # Fix: Properly add new columns and update values
    if 'first_inference_time_ms' not in df.columns:
        df['first_inference_time_ms'] = 0.0
    if 'avg_latency_ms' not in df.columns:
        df['avg_latency_ms'] = 0.0
    
    # Update the specific row values
    df.loc[args.model_id, 'first_inference_time_ms'] = first_inference_time * 1000  # Convert to ms
    df.loc[args.model_id, 'avg_latency_ms'] = avg_latency * 1000  # Convert to ms
    
    # print(df.head(5))
    # Save the updated DataFrame
    df.to_csv(args.output_file, index=True)
    print("Update row in output csv: {}".format(df.loc[args.model_id]))
    

if __name__ == "__main__":
    main()