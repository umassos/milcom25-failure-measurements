# Jetson Orin Model Profiling

This directory contains optimized profiling scripts for NVIDIA Jetson Orin devices, designed to measure model performance with TensorRT acceleration and comprehensive system monitoring.

## Features

- **TensorRT Integration**: Build and profile optimized TensorRT engines
- **ONNX Runtime Support**: Compare performance with standard ONNX inference
- **System Monitoring**: Real-time CPU, GPU, memory, temperature, and power monitoring
- **Jetson Optimization**: Performance mode settings and thermal management
- **Comprehensive Metrics**: FLOPs, parameters, inference time, memory usage, and more

## Files

- `jetson_profiler.py` - Main Python profiling script
- `run_jetson_profiling.sh` - Bash script for batch profiling with system monitoring
- `jetson_requirements.txt` - Python dependencies
- `model_config.py` - Model configurations (shared with desktop profiler)

## Prerequisites

### System Requirements
- NVIDIA Jetson Orin device (Orin NX, Orin Nano, or Orin AGX)
- JetPack 5.0 or later
- Python 3.8+

### Install Dependencies

```bash
# Install Python dependencies
pip install -r jetson_requirements.txt

# Install system packages (if not already available)
sudo apt-get update
sudo apt-get install python3-tensorrt python3-pycuda bc

# Make scripts executable
chmod +x run_jetson_profiling.sh
```

## Usage

### Single Model Profiling

```bash
# Profile a single model
python3 jetson_profiler.py -m efficientnet_b0 -o results.csv

# List available models
python3 jetson_profiler.py --list_models
```

### Batch Profiling

```bash
# Run profiling for all models in model_ids.txt
./run_jetson_profiling.sh

# Or run in background
nohup ./run_jetson_profiling.sh > profiling.log 2>&1 &
```

### Custom Configuration

```bash
# Edit model_ids.txt to specify which models to profile
nano model_ids.txt

# Example content:
# efficientnet_b0
# efficientnet_b1
# resnet18
# resnet50
```

## Output

The profiler generates a CSV file with the following metrics:

| Metric | Description |
|--------|-------------|
| `model_id` | Unique model identifier |
| `model_family` | Model architecture family |
| `model_variant` | Specific model variant |
| `input_shape` | Input tensor dimensions |
| `device` | Device identifier (Jetson_Orin) |
| `gflops` | Computational complexity in GFLOPs |
| `mparams` | Model parameters in millions |
| `onnx_file_size_mb` | ONNX model file size |
| `onnx_load_time_ms` | ONNX model loading time |
| `onnx_inference_time_ms` | ONNX inference latency |
| `onnx_memory_mb` | ONNX memory usage |
| `tensorrt_available` | Whether TensorRT is available |
| `tensorrt_build_time_ms` | TensorRT engine build time |
| `tensorrt_inference_time_ms` | TensorRT inference latency |
| `tensorrt_memory_mb` | TensorRT memory usage |
| `baseline_cpu_util` | Baseline CPU utilization |
| `baseline_cpu_mem_gb` | Baseline CPU memory usage |
| `baseline_gpu_util` | Baseline GPU utilization |
| `baseline_temp_c` | Baseline system temperature |

## System Monitoring

The bash script provides real-time system monitoring:

- **CPU/GPU Utilization**: Performance metrics during profiling
- **Memory Usage**: RAM consumption tracking
- **Temperature**: Thermal monitoring with warnings
- **Power Consumption**: Power draw measurement (if available)
- **Performance Mode**: Automatic CPU/GPU governor optimization

## TensorRT Optimization

### Precision Modes
- **FP32**: Full precision (default)
- **FP16**: Half precision (recommended for Jetson)
- **INT8**: 8-bit quantization (maximum performance)

### Engine Building
```python
# Customize TensorRT settings in jetson_profiler.py
class TensorRTEngineBuilder:
    def __init__(self, workspace_size: int = 1 << 30):  # 1GB workspace
        # Adjust workspace size based on available GPU memory
```

## Performance Tips

### 1. Thermal Management
- Ensure adequate cooling during profiling
- Monitor temperature with `tegrastats`
- Use cooldown periods between models

### 2. Memory Optimization
- Close unnecessary applications
- Monitor available RAM with `free -h`
- Adjust TensorRT workspace size if needed

### 3. Power Management
- Use external power supply for consistent performance
- Monitor power consumption during profiling
- Set appropriate power limits

### 4. TensorRT Tuning
- Use FP16 precision for better performance
- Experiment with different workspace sizes
- Profile with representative input sizes

## Troubleshooting

### Common Issues

1. **TensorRT Not Available**
   ```bash
   sudo apt-get install python3-tensorrt
   ```

2. **PyCUDA Import Error**
   ```bash
   sudo apt-get install python3-pycuda
   ```

3. **Permission Denied for tegrastats**
   ```bash
   sudo usermod -a -G video $USER
   # Log out and back in
   ```

4. **High Temperature Warnings**
   - Check cooling system
   - Reduce ambient temperature
   - Increase cooldown periods

5. **Out of Memory Errors**
   - Reduce batch size
   - Close other applications
   - Use smaller models for testing

### Performance Debugging

```bash
# Monitor system resources
tegrastats --interval 1

# Check GPU status
nvidia-smi

# Monitor power consumption
cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device0/in_power0_input
```

## Comparison with Desktop Profiler

| Feature | Desktop Profiler | Jetson Profiler |
|---------|------------------|-----------------|
| **Runtime** | ONNX Runtime (CUDA) | ONNX Runtime + TensorRT |
| **Monitoring** | Basic GPU memory | Full system monitoring |
| **Optimization** | None | TensorRT engines |
| **Power** | Not measured | Power consumption |
| **Thermal** | Not monitored | Temperature tracking |
| **Architecture** | x86_64 | ARM64 (Jetson) |

## Contributing

To extend the profiler:

1. **Add New Models**: Update `model_config.py`
2. **New Metrics**: Extend the results dictionary in `profile_model()`
3. **Additional Runtimes**: Implement new profiling methods
4. **System Monitoring**: Add new monitoring capabilities

## License

This project follows the same license as the parent repository.

## Support

For Jetson-specific issues:
- [NVIDIA Jetson Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/70)
- [JetPack Documentation](https://docs.nvidia.com/jetson/jetpack/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
