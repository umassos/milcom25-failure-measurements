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


# ============ Main ============
def main():
    
    args = parse_args()
    print(args)
    
    model_family, model_variant, weights, input_shape = model_config[args.model_id]
    print('Converting to ONNX... {}'.format(model_variant))
    convert_to_onnx(model_variant, weights, input_shape)

if __name__ == "__main__":
    main()