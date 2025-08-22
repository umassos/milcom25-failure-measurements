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
import torch_tensorrt

# Workaround for broken MINICPM code that doesn't import List
builtins.List = List

# ============ Parse CLI Arguments ============
def parse_args():
    parser = argparse.ArgumentParser(description='Model Profiling on GPU')
    parser.add_argument("-m", "--model_id", type=str, default=None, help="Model ID to run")
    parser.add_argument("-o", "--output_file", type=str, default="profiles.csv", help="Output CSV file")
    return parser.parse_args()

# ============ Setup ============
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(device)


# ============ Load Model ============
def convert_to_torch_trt(model_variant, weights, input_shape):
    
    model = getattr(torchvision.models, model_variant)(weights=weights).cuda().eval()
    model = torch.jit.script(model)

    trt_path = os.path.join("torch_trt_models", f"{model_variant}.pt")

    # if not os.path.exists(trt_path):
    trt_model = torch_tensorrt.compile(
        model,
        ir='torchscript',
        inputs=[torch_tensorrt.Input(input_shape, dtype=torch.float32)],
        enabled_precisions={torch.float32},
        workspace_size=1 << 22
    )
    torch.jit.save(trt_model, trt_path)


def load_torch_trt_model(model_variant, input_shape):
    trt_path = os.path.join("torch_trt_models", f"{model_variant}.pt")
    model = torch.jit.load(trt_path).cuda().eval()
    return model


# ============ Main ============
def main():
    
    args = parse_args()
    print(args)
    
    model_family, model_variant, weights, input_shape = model_config[args.model_id]
    if input_shape == (1, 3, 224, 224) or model_variant == 'efficientnet_b1':
        return
        
    print('Converting to Torch TRT... {}'.format(model_variant))
    convert_to_torch_trt(model_variant, weights, input_shape)

    start_time = timeit.default_timer()
    model = load_torch_trt_model(model_variant, input_shape)
    model_load_time = timeit.default_timer() - start_time
    print(f"Model loaded in {model_load_time * 1000:.2f} ms")
    
    
    df = pd.read_csv(args.output_file, index_col=0)
    
    if 'torch_trt_load_time_ms' not in df.columns:
        df['torch_trt_load_time_ms'] = 0.0
    
    df.loc[args.model_id, 'torch_trt_load_time_ms'] = model_load_time * 1000  # Convert to ms
    df.to_csv(args.output_file, index=True)
    print("Update row in output csv: {}".format(df.loc[args.model_id]))

if __name__ == "__main__":
    main()