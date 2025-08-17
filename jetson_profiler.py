#!/usr/bin/env python3
"""
Jetson Orin Model Profiling Script
Optimized for NVIDIA Jetson Orin devices with TensorRT support
"""

import os
import time
import json
import csv
import argparse
import subprocess
import psutil
import numpy as np
from typing import Dict, Any, Tuple
import torch
import torchvision.models
import timeit

# Jetson-specific imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT not available. Install with: sudo apt-get install python3-tensorrt")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available")

from model_config import model_config

# ============ Jetson System Monitoring ============
class JetsonMonitor:
    """Monitor Jetson system resources"""
    
    def __init__(self):
        self.device_id = 0
        
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information using tegrastats"""
        try:
            # Run tegrastats to get GPU info
            result = subprocess.run(['tegrastats', '--interval', '1', '--count', '1'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return self._parse_tegrastats(result.stdout)
        except Exception as e:
            print(f"Warning: Could not get tegrastats: {e}")
        
        return {"gpu_util": 0, "gpu_mem": 0, "cpu_util": 0, "cpu_mem": 0, "temp": 0}
    
    def _parse_tegrastats(self, output: str) -> Dict[str, Any]:
        """Parse tegrastats output"""
        try:
            # Example output: "RAM 6.5/31.2GB (lfb 2x512KB) | SWAP 0/15.6GB (cached 0MB) | CPU [0%@1479,off,off,off] | EMC_FREQ 0%@665 | GR3D_FREQ 0%@76 | APE 25Mhz | GPU@76Mhz | PLL@38.4Mhz | CPU@1.48GHz | VDD_CPU_AVG 0.7V | VDD_GPU_AVD 0.7V | VDD_SOC_AVG 0.7V | VDD_CPU_CV 0.7V | VDD_GPU_CV 0.7V | VDD_SOC_CV 0.7V | VDD_CPU 0.7V | VDD_GPU 0.7V | VDD_SOC 0.7V | temp 35C"
            
            # Extract GPU frequency and utilization
            gpu_match = None
            if "GR3D_FREQ" in output:
                gpu_match = output.split("GR3D_FREQ")[1].split("|")[0].strip()
            
            # Extract CPU utilization
            cpu_match = None
            if "CPU [" in output:
                cpu_match = output.split("CPU [")[1].split("]")[0]
            
            # Extract temperature
            temp_match = None
            if "temp" in output:
                temp_match = output.split("temp")[1].split("C")[0].strip()
            
            return {
                "gpu_util": float(gpu_match.split("%")[0]) if gpu_match else 0,
                "gpu_freq": float(gpu_match.split("@")[1].split("M")[0]) if gpu_match else 0,
                "cpu_util": float(cpu_match.split("%")[0]) if cpu_match else 0,
                "cpu_freq": float(cpu_match.split("@")[1].split("G")[0]) if cpu_match else 0,
                "temp": float(temp_match) if temp_match else 0
            }
        except Exception as e:
            print(f"Warning: Could not parse tegrastats: {e}")
            return {"gpu_util": 0, "gpu_mem": 0, "cpu_util": 0, "cpu_mem": 0, "temp": 0}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            "cpu_util": cpu_percent,
            "cpu_mem_percent": memory.percent,
            "cpu_mem_used_gb": memory.used / (1024**3),
            "cpu_mem_total_gb": memory.total / (1024**3)
        }
    
    def get_power_info(self) -> Dict[str, Any]:
        """Get power consumption information"""
        try:
            # Read power from sysfs (Jetson specific)
            power_path = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device0/in_power0_input"
            if os.path.exists(power_path):
                with open(power_path, 'r') as f:
                    power_mw = float(f.read().strip()) / 1000.0  # Convert to watts
                return {"power_w": power_mw}
        except Exception as e:
            print(f"Warning: Could not read power info: {e}")
        
        return {"power_w": 0}

# ============ TensorRT Engine Builder ============
class TensorRTEngineBuilder:
    """Build and optimize TensorRT engines for Jetson"""
    
    def __init__(self, workspace_size: int = 1 << 30):  # 1GB workspace
        self.logger = trt.Logger(trt.Logger.WARNING) if TENSORRT_AVAILABLE else None
        self.workspace_size = workspace_size
        
    def build_engine_from_onnx(self, onnx_path: str, engine_path: str, 
                              precision: str = "fp16") -> bool:
        """Build TensorRT engine from ONNX model"""
        if not TENSORRT_AVAILABLE:
            return False
            
        try:
            builder = trt.Builder(self.logger)
            config = builder.create_builder_config()
            config.max_workspace_size = self.workspace_size
            
            # Set precision
            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
            
            # Parse ONNX
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.logger)
            
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print(f"Failed to parse ONNX model: {onnx_path}")
                    return False
            
            # Build engine
            engine = builder.build_engine(network, config)
            if engine is None:
                print("Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            print(f"TensorRT engine saved to: {engine_path}")
            return True
            
        except Exception as e:
            print(f"Error building TensorRT engine: {e}")
            return False
    
    def load_engine(self, engine_path: str):
        """Load TensorRT engine"""
        if not TENSORRT_AVAILABLE:
            return None
            
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            return engine
        except Exception as e:
            print(f"Error loading TensorRT engine: {e}")
            return None

# ============ Model Profiler ============
class JetsonModelProfiler:
    """Profile models on Jetson Orin"""
    
    def __init__(self, output_file: str = "jetson_profiles.csv"):
        self.output_file = output_file
        self.monitor = JetsonMonitor()
        self.trt_builder = TensorRTEngineBuilder()
        
        # Create directories
        os.makedirs("onnx_models", exist_ok=True)
        os.makedirs("tensorrt_engines", exist_ok=True)
        
    def profile_model(self, model_id: str) -> Dict[str, Any]:
        """Profile a single model"""
        print(f"==========================================")
        print(f"Profiling model: {model_id}")
        print(f"==========================================")
        
        # Get model configuration
        if model_id not in model_config:
            raise ValueError(f"Model {model_id} not found in model_config")
        
        model_family, model_variant, weights, input_shape = model_config[model_id]
        
        # Get system baseline
        baseline_system = self.monitor.get_system_info()
        baseline_gpu = self.monitor.get_gpu_info()
        
        # Load PyTorch model
        print(f"Loading PyTorch model: {model_variant}")
        model = getattr(torchvision.models, model_variant)(weights=weights).eval()
        
        # Calculate model metrics
        gflops, mparams = self._calculate_flops(model, input_shape)
        
        # Export to ONNX
        onnx_path = os.path.join("onnx_models", f"{model_variant}.onnx")
        onnx_file_size = self._export_to_onnx(model, model_variant, input_shape, onnx_path)
        
        # Profile ONNX Runtime
        onnx_metrics = self._profile_onnx(onnx_path, input_shape)
        
        # Profile TensorRT (if available)
        trt_metrics = self._profile_tensorrt(onnx_path, model_variant, input_shape)
        
        # Compile results
        results = {
            "model_id": model_id,
            "model_family": model_family,
            "model_variant": model_variant,
            "input_shape": str(input_shape),
            "device": "Jetson_Orin",
            "gflops": gflops,
            "mparams": mparams,
            "onnx_file_size_mb": onnx_file_size,
            "onnx_load_time_ms": onnx_metrics.get("load_time_ms", 0),
            "onnx_inference_time_ms": onnx_metrics.get("inference_time_ms", 0),
            "onnx_memory_mb": onnx_metrics.get("memory_mb", 0),
            "tensorrt_available": TENSORRT_AVAILABLE,
            "tensorrt_build_time_ms": trt_metrics.get("build_time_ms", 0),
            "tensorrt_inference_time_ms": trt_metrics.get("inference_time_ms", 0),
            "tensorrt_memory_mb": trt_metrics.get("memory_mb", 0),
            "baseline_cpu_util": baseline_system["cpu_util"],
            "baseline_cpu_mem_gb": baseline_system["cpu_mem_used_gb"],
            "baseline_gpu_util": baseline_gpu["gpu_util"],
            "baseline_temp_c": baseline_gpu["temp"]
        }
        
        return results
    
    def _calculate_flops(self, model, input_shape) -> Tuple[float, float]:
        """Calculate FLOPs and parameters"""
        try:
            dummy_input = torch.randn(input_shape)
            # Use thop if available, otherwise estimate
            try:
                import thop
                flops, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
                return flops / 1e9, params / 1e6  # GFLOPs, Millions of parameters
            except ImportError:
                # Rough estimation
                params = sum(p.numel() for p in model.parameters())
                # Estimate FLOPs based on input size and parameters
                flops = params * 2  # Rough estimate
                return flops / 1e9, params / 1e6
        except Exception as e:
            print(f"Warning: Could not calculate FLOPs: {e}")
            return 0, 0
    
    def _export_to_onnx(self, model, model_variant, input_shape, onnx_path) -> float:
        """Export PyTorch model to ONNX"""
        if os.path.exists(onnx_path):
            print(f"ONNX model already exists: {onnx_path}")
        else:
            print(f"Exporting to ONNX: {onnx_path}")
            dummy_input = torch.randn(input_shape)
            
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            )
        
        return os.path.getsize(onnx_path) / (1024**2)  # MB
    
    def _profile_onnx(self, onnx_path: str, input_shape: Tuple) -> Dict[str, Any]:
        """Profile ONNX Runtime performance"""
        if not ONNX_AVAILABLE:
            return {}
        
        try:
            # Get baseline memory
            baseline_system = self.monitor.get_system_info()
            
            # Load ONNX model
            start_time = timeit.default_timer()
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            load_time_ms = (timeit.default_timer() - start_time) * 1000
            
            # Prepare input
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            ort_inputs = {session.get_inputs()[0].name: dummy_input}
            
            # Warmup
            for _ in range(3):
                session.run(None, ort_inputs)
            
            # Benchmark inference
            times = []
            for _ in range(10):
                start_time = timeit.default_timer()
                session.run(None, ort_inputs)
                times.append((timeit.default_timer() - start_time) * 1000)
            
            # Get memory after inference
            after_system = self.monitor.get_system_info()
            memory_delta = after_system["cpu_mem_used_gb"] - baseline_system["cpu_mem_used_gb"]
            
            return {
                "load_time_ms": load_time_ms,
                "inference_time_ms": np.mean(times),
                "inference_time_std_ms": np.std(times),
                "memory_mb": memory_delta * 1024  # Convert GB to MB
            }
            
        except Exception as e:
            print(f"Warning: ONNX profiling failed: {e}")
            return {}
    
    def _profile_tensorrt(self, onnx_path: str, model_variant: str, 
                          input_shape: Tuple) -> Dict[str, Any]:
        """Profile TensorRT performance"""
        if not TENSORRT_AVAILABLE:
            return {}
        
        try:
            engine_path = os.path.join("tensorrt_engines", f"{model_variant}.engine")
            
            # Build or load engine
            if not os.path.exists(engine_path):
                print(f"Building TensorRT engine for {model_variant}")
                start_time = timeit.default_timer()
                success = self.trt_builder.build_engine_from_onnx(onnx_path, engine_path, "fp16")
                build_time_ms = (timeit.default_timer() - start_time) * 1000
                
                if not success:
                    return {"build_time_ms": build_time_ms, "error": "Build failed"}
            else:
                print(f"Loading existing TensorRT engine: {engine_path}")
                build_time_ms = 0
            
            # Load engine
            engine = self.trt_builder.load_engine(engine_path)
            if engine is None:
                return {"build_time_ms": build_time_ms, "error": "Load failed"}
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Allocate GPU memory
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            input_size = dummy_input.nbytes
            output_size = engine.get_binding_size(1)  # Assuming single output
            
            # Allocate GPU buffers
            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(output_size)
            
            # Create CUDA stream
            stream = cuda.Stream()
            
            # Warmup
            for _ in range(3):
                cuda.memcpy_htod_async(d_input, dummy_input, stream)
                context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                stream.synchronize()
            
            # Benchmark inference
            times = []
            for _ in range(10):
                start_time = timeit.default_timer()
                cuda.memcpy_htod_async(d_input, dummy_input, stream)
                context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                stream.synchronize()
                times.append((timeit.default_timer() - start_time) * 1000)
            
            # Cleanup
            del context, engine
            
            return {
                "build_time_ms": build_time_ms,
                "inference_time_ms": np.mean(times),
                "inference_time_std_ms": np.std(times),
                "memory_mb": (input_size + output_size) / (1024**2)  # GPU memory in MB
            }
            
        except Exception as e:
            print(f"Warning: TensorRT profiling failed: {e}")
            return {}
    
    def save_results(self, results: Dict[str, Any]):
        """Save profiling results to CSV"""
        file_exists = os.path.exists(self.output_file)
        
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(results)
        
        print(f"Results saved to: {self.output_file}")

# ============ Main ============
def main():
    parser = argparse.ArgumentParser(description='Jetson Orin Model Profiling')
    parser.add_argument('-m', '--model_id', type=str, required=True, 
                       help='Model ID to profile')
    parser.add_argument('-o', '--output_file', type=str, default='jetson_profiles.csv',
                       help='Output CSV file')
    parser.add_argument('--list_models', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model_id in model_config.keys():
            print(f"  {model_id}")
        return
    
    # Initialize profiler
    profiler = JetsonModelProfiler(args.output_file)
    
    try:
        # Profile the model
        results = profiler.profile_model(args.model_id)
        
        # Save results
        profiler.save_results(results)
        
        print(f"✅ Successfully profiled {args.model_id}")
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"❌ Failed to profile {args.model_id}: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
