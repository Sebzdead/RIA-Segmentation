#!/bin/bash
# setup.sh - Automated setup script for RIA Segmentation Pipeline

set -e  # Exit on error

echo "🔬 RIA Segmentation Pipeline Setup"
echo "=================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "📋 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "ria_segmentation" ]; then
    echo "🔨 Creating virtual environment..."
    python -m venv ria_segmentation
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source ria_segmentation/bin/activate  # Linux/Mac
# For Windows: ria_segmentation\Scripts\activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version by default)
echo "🔥 Installing PyTorch..."
read -p "Do you have CUDA installed? (y/n): " has_cuda
if [[ $has_cuda == "y" || $has_cuda == "Y" ]]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📦 Installing other dependencies..."
pip install -r requirements.txt

# Clone and install SAM2
if [ ! -d "segment-anything-2" ]; then
    echo "🤖 Installing SAM2..."
    git clone https://github.com/facebookresearch/segment-anything-2.git
    cd segment-anything-2
    pip install -e .
    cd ..
else
    echo "✅ SAM2 already installed"
fi

# Create checkpoint directory
mkdir -p sam2_checkpoints

# Download model checkpoints
echo "⬇️  Downloading SAM2 model checkpoints..."
echo "Available models:"
echo "1. Tiny (fastest, ~150MB)"
echo "2. Base+ (balanced, ~900MB)" 
echo "3. Large (most accurate, ~2.5GB)"
echo "4. All models"

read -p "Which model(s) to download? (1-4): " model_choice

case $model_choice in
    1)
        echo "Downloading Tiny model..."
        wget -O sam2_checkpoints/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt
        ;;
    2)
        echo "Downloading Base+ model..."
        wget -O sam2_checkpoints/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt
        ;;
    3)
        echo "Downloading Large model..."
        wget -O sam2_checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt
        ;;
    4)
        echo "Downloading all models..."
        wget -O sam2_checkpoints/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt
        wget -O sam2_checkpoints/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt
        wget -O sam2_checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt
        ;;
    *)
        echo "Invalid choice. Skipping model download."
        ;;
esac

# Test installation
echo "🧪 Testing installation..."
python -c "
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
print('✅ Core dependencies installed successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

try:
    from sam2.build_sam import build_sam2_video_predictor
    print('✅ SAM2 installed successfully')
except ImportError as e:
    print(f'❌ SAM2 import failed: {e}')
"

echo ""
echo "🎉 Setup completed!"
echo ""
echo "Next steps:"
echo "1. Update checkpoint paths in processing scripts (3AutoCrop.py, 4RIAMaskGen.py, 6SegmentWorm.py)"
echo "2. Place your TIFF files in the TIFF/ directory"
echo "3. Run the pipeline scripts in order (1-7)"
echo ""
echo "To activate the environment in the future:"
echo "source ria_segmentation/bin/activate"
