#!/bin/bash
# Quick Start Script for Visitor Counting System (Linux/Mac)

set -e  # Exit on error

echo "========================================"
echo "Visitor Counting System - Quick Start"
echo "========================================"
echo

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "ERROR: Python 3.12 not found"
    echo "Please install Python 3.12 first"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3.12 -m venv venv
    echo "Virtual environment created successfully!"
    echo
else
    echo "[1/5] Virtual environment already exists"
    echo
fi

# Activate virtual environment
echo "[2/5] Activating virtual environment..."
source venv/bin/activate
echo

# Check if PyTorch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo "[3/5] Installing PyTorch with CUDA 12.1..."
    echo "This may take several minutes..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo "PyTorch installed successfully!"
    echo
else
    echo "[3/5] PyTorch already installed"
    echo
fi

# Check if ultralytics is installed
if ! python -c "import ultralytics" 2>/dev/null; then
    echo "[4/5] Installing dependencies..."
    pip install ultralytics opencv-python opencv-contrib-python numpy requests lap
    echo "Dependencies installed successfully!"
    echo
else
    echo "[4/5] Dependencies already installed"
    echo
fi

# Verify CUDA
echo "[5/5] Verifying GPU setup..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "WARNING: main.py not found in current directory!"
    echo "Please ensure main.py is in the same folder as this script."
    exit 1
fi

# Check if rtsp_urls.txt exists
if [ ! -f "rtsp_urls.txt" ]; then
    echo "WARNING: rtsp_urls.txt not found!"
    echo "Creating template file..."
    cat > rtsp_urls.txt << EOF
# Add your RTSP URLs here (one per line)
# Example: rtsp://admin:12345@192.168.5.227/71
EOF
    echo
    echo "Please edit rtsp_urls.txt with your camera URLs and run this script again."
    exit 1
fi

# Check if bytetrack.yaml exists
if [ ! -f "bytetrack.yaml" ]; then
    echo "WARNING: bytetrack.yaml not found!"
    echo "The application will use default tracking settings."
    echo "Consider creating bytetrack.yaml for optimized tracking."
    echo
fi

echo "========================================"
echo "Setup complete! Starting application..."
echo "========================================"
echo
echo "Press Ctrl+C to stop the application"
echo

# Run the application
python main.py
