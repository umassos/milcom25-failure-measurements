#!/bin/bash

# Jetson Orin Profiling Script
# Optimized for NVIDIA Jetson Orin devices

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_IDS_FILE="model_ids.txt"
OUTPUT_FILE="jetson_profiles.csv"
LOG_FILE="jetson_profiling.log"
MAX_POWER_W=15  # Maximum power consumption in watts
COOLDOWN_TIME=30  # Cooldown time between models in seconds

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check Jetson system status
check_jetson_status() {
    print_status "Checking Jetson system status..."
    
    # Check if we're on a Jetson device
    if ! command -v tegrastats &> /dev/null; then
        print_error "tegrastats not found. This script is designed for Jetson devices."
        exit 1
    fi
    
    # Check system temperature
    temp=$(tegrastats --interval 1 --count 1 | grep -o 'temp [0-9]*C' | cut -d' ' -f2 | cut -d'C' -f1)
    if [ "$temp" -gt 80 ]; then
        print_warning "System temperature is high: ${temp}°C. Consider cooling before profiling."
    else
        print_success "System temperature: ${temp}°C"
    fi
    
    # Check available memory
    mem_available=$(free -g | grep Mem | awk '{print $7}')
    if [ "$mem_available" -lt 4 ]; then
        print_warning "Low available memory: ${mem_available}GB. Consider closing other applications."
    else
        print_success "Available memory: ${mem_available}GB"
    fi
}

# Function to set Jetson performance mode
set_performance_mode() {
    print_status "Setting Jetson to performance mode..."
    
    # Set CPU governor to performance
    if [ -f "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" ]; then
        echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
        print_success "CPU governor set to performance mode"
    fi
    
    # Set GPU to maximum frequency
    if [ -f "/sys/devices/gpu.0/devfreq/17000000.gv11b/governor" ]; then
        echo performance | sudo tee /sys/devices/gpu.0/devfreq/17000000.gv11b/governor > /dev/null
        print_success "GPU governor set to performance mode"
    fi
}

# Function to monitor system during profiling
monitor_system() {
    local model_id=$1
    local log_file="monitoring_${model_id}.log"
    
    print_status "Starting system monitoring for ${model_id}..."
    
    # Start monitoring in background
    (
        while true; do
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            
            # Get system stats
            cpu_util=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
            mem_util=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
            temp=$(tegrastats --interval 1 --count 1 | grep -o 'temp [0-9]*C' | cut -d' ' -f2 | cut -d'C' -f1)
            
            # Get power consumption if available
            power_w="N/A"
            if [ -f "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device0/in_power0_input" ]; then
                power_mw=$(cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device0/in_power0_input)
                power_w=$(echo "scale=2; $power_mw/1000" | bc)
            fi
            
            echo "[${timestamp}] CPU: ${cpu_util}%, MEM: ${mem_util}%, TEMP: ${temp}°C, POWER: ${power_w}W" >> "$log_file"
            
            # Check for critical conditions
            if [ "$temp" -gt 85 ]; then
                print_warning "Critical temperature: ${temp}°C"
            fi
            
            sleep 5
        done
    ) &
    
    MONITOR_PID=$!
}

# Function to stop system monitoring
stop_monitoring() {
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
        unset MONITOR_PID
    fi
}

# Function to profile a single model
profile_model() {
    local model_id=$1
    
    print_status "=========================================="
    print_status "Profiling model: ${model_id}"
    print_status "=========================================="
    
    # Start monitoring
    monitor_system "$model_id"
    
    # Run the profiling
    start_time=$(date +%s)
    
    if python3 jetson_profiler.py -m "$model_id" -o "$OUTPUT_FILE"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_success "Successfully profiled ${model_id} in ${duration}s"
    else
        print_error "Failed to profile ${model_id}"
        return 1
    fi
    
    # Stop monitoring
    stop_monitoring
    
    # Cooldown period
    print_status "Cooldown period: ${COOLDOWN_TIME}s..."
    sleep $COOLDOWN_TIME
    
    return 0
}

# Function to restore normal system settings
restore_system_settings() {
    print_status "Restoring normal system settings..."
    
    # Restore CPU governor to ondemand
    if [ -f "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" ]; then
        echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
        print_success "CPU governor restored to ondemand mode"
    fi
    
    # Restore GPU governor to ondemand
    if [ -f "/sys/devices/gpu.0/devfreq/17000000.gv11b/governor" ]; then
        echo ondemand | sudo tee /sys/devices/gpu.0/devfreq/17000000.gv11b/governor > /dev/null
        print_success "GPU governor restored to ondemand mode"
    fi
}

# Main execution
main() {
    print_status "Starting Jetson Orin Model Profiling"
    print_status "====================================="
    
    # Check if model_ids.txt exists
    if [ ! -f "$MODEL_IDS_FILE" ]; then
        print_error "Model IDs file not found: $MODEL_IDS_FILE"
        print_status "Please create a file with one model ID per line"
        exit 1
    fi
    
    # Check Jetson status
    check_jetson_status
    
    # Set performance mode
    set_performance_mode
    
    # Trap to restore settings on exit
    trap restore_system_settings EXIT
    
    # Read and profile each model
    local success_count=0
    local total_count=0
    
    while IFS= read -r model_id; do
        # Skip empty lines and comments
        if [[ -z "$model_id" || "$model_id" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        
        # Remove leading/trailing whitespace
        model_id=$(echo "$model_id" | xargs)
        total_count=$((total_count + 1))
        
        if profile_model "$model_id"; then
            success_count=$((success_count + 1))
        fi
        
        echo ""
        
    done < "$MODEL_IDS_FILE"
    
    # Summary
    print_status "=========================================="
    print_status "Profiling Complete!"
    print_status "=========================================="
    print_success "Successfully profiled: ${success_count}/${total_count} models"
    print_status "Results saved to: ${OUTPUT_FILE}"
    print_status "Logs saved to: ${LOG_FILE}"
    
    if [ $success_count -eq $total_count ]; then
        print_success "All models profiled successfully!"
        exit 0
    else
        print_warning "Some models failed to profile"
        exit 1
    fi
}

# Run main function
main "$@"
