#!/bin/bash

# Check if model_ids.txt exists
if [ ! -f "model_ids.txt" ]; then
    echo "Error: model_ids.txt not found!"
    echo "Please create a file with one model ID per line, e.g.:"
    echo "efficientnet_b0"
    echo "efficientnet_b1"
    echo "resnet18"
    exit 1
fi

# Read each line from model_ids.txt and run profiling
while IFS= read -r model_id; do
    # Skip empty lines and comments
    if [[ -z "$model_id" || "$model_id" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Remove leading/trailing whitespace
    model_id=$(echo "$model_id" | xargs)

    echo "=========================================="
    echo "Running inference for model: $model_id"
    echo "=========================================="
    
    echo "Starting server..."
    model_server="obelix193"
    python run_grpc_server.py -m "$model_id" &
    server_pid=$!
    echo "Server started with PID: $server_pid"
    echo "Waiting for server to start..."
    
    timeout_seconds=10
    elapsed=0
    # Wait for server to actually start listening on port 8180
    while ! nc -z $model_server 8180 2>/dev/null; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $timeout_seconds ]; then
            echo "Server failed to start within $timeout_seconds seconds. Exiting..."
            kill $server_pid
            exit 1
        fi
    done
    echo "Server started successfully"
    
    echo "Running inference..."
    python run_grpc_client.py -m "$model_id" -s "$model_server:8180"
    
    echo "Stopping server..."
    kill $server_pid
    sleep 1
    echo ""
    
done < "model_ids.txt"

echo "=========================================="
echo "All models inference completed!"
echo "Results saved to profiles.csv"
echo "=========================================="
