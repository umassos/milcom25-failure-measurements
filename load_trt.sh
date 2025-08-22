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
    echo "Profiling loading time for model: $model_id"
    echo "=========================================="
    
    python load_trt.py -m "$model_id"
    echo ""
    
done < "model_ids.txt"

echo "=========================================="
echo "All models profiled!"
echo "Results saved to profiles.csv"
echo "=========================================="
