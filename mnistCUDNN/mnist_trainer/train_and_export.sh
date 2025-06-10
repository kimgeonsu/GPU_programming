#!/bin/bash

echo "=== Improved MNIST Model Training and Export ==="
echo "This script will:"
echo "1. Install required Python packages"
echo "2. Train an improved CNN model on MNIST"
echo "3. Export weights to binary files for CUDNN"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed"
    exit 1
fi

echo "Installing required packages..."
pip3 install -r mnist_trainer/requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install required packages"
    exit 1
fi

echo ""
echo "Starting model training..."
echo "This may take several minutes depending on your hardware."
echo "GPU will be used if available, otherwise CPU."
echo ""

cd mnist_trainer
python3 improved_mnist_model.py

if [ $? -ne 0 ]; then
    echo "Error: Model training failed"
    exit 1
fi

echo ""
echo "=== Training Complete ==="
echo "Model weights have been exported to ../data_improved/"
echo "You can now compile and run the improved CUDNN implementation:"
echo ""
echo "1. Compile: make -f Makefile_improved"
echo "2. Run: ./mnistCUDNN_improved"
echo ""
echo "The improved model should achieve higher accuracy than the original!" 