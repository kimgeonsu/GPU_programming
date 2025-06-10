#!/bin/bash

echo "=== MNIST CUDNN Programs ==="
echo ""
echo "Available programs:"
echo "1. mnist_improved - Advanced MNIST with batch normalization"
echo "2. mnist_simple   - Simple improved MNIST"
echo ""
echo "Usage examples:"
echo "  ./mnist_improved batch           # Process all test images"
echo "  ./mnist_improved help            # Show help"
echo "  ./mnist_simple batch             # Process all test images (simple model)"
echo ""

if [ "$1" = "improved" ]; then
    shift
    ./mnist_improved "$@"
elif [ "$1" = "simple" ]; then
    shift
    ./mnist_simple "$@"
elif [ "$1" = "batch" ]; then
    echo "Running both models on all test images:"
    echo ""
    echo "=== IMPROVED MODEL ==="
    ./mnist_improved batch
    echo ""
    echo "=== SIMPLE MODEL ==="
    ./mnist_simple batch
else
    echo "Usage: $0 [improved|simple|batch] [options]"
    echo ""
    echo "  improved [options] - Run advanced model"
    echo "  simple [options]   - Run simple model" 
    echo "  batch              - Run both models"
fi 