# Installation Guide for RIA Segmentation Pipeline

## Prerequisites
- Python 3.8 or higher
- Git
- CUDA-capable GPU (recommended for faster processing)

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/Sebzdead/RIA-Segmentation.git
cd RIA-Segmentation
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using conda (recommended)
conda create -n ria_segmentation python=3.9
conda activate ria_segmentation

# OR using venv
python -m venv ria_segmentation
# Windows:
ria_segmentation\Scripts\activate
# Linux/Mac:
source ria_segmentation/bin/activate
```

### 3. Install PyTorch
Install PyTorch with CUDA support (if available):
```bash
# For CUDA 11.8 (check your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Other Dependencies
```bash
pip install -r requirements.txt
```

### 5. Install SAM2
```bash
# Clone and install SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..
```

### 6. Download SAM2 Model Checkpoints
Download the required model checkpoints:
```bash
# Create checkpoints directory
mkdir -p sam2_checkpoints

# Download model checkpoints (choose based on your needs)
# Tiny model (fastest, less accurate)
wget -O sam2_checkpoints/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt

# Base+ model (balanced)
wget -O sam2_checkpoints/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt

# Large model (most accurate, slowest)
wget -O sam2_checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt
```

### 7. Configure Model Paths
Update the checkpoint paths in the Python scripts:
- `3AutoCrop.py`
- `4RIAMaskGen.py` 
- `6SegmentWorm.py`

Change the paths to point to your downloaded checkpoints:
```python
sam2_checkpoint = r"path/to/your/sam2_checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = r"path/to/segment-anything-2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
```

## Verification
Test your installation:
```python
import torch
import sam2
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("SAM2 installation successful!")
```

## Troubleshooting

### Common Issues:
1. **CUDA out of memory**: Reduce `CHUNK_SIZE` in processing scripts
2. **SAM2 import errors**: Ensure SAM2 is properly installed and in Python path
3. **Missing model checkpoints**: Verify checkpoint files are downloaded and paths are correct

### System Requirements:
- RAM: 16GB+ recommended
- GPU: 8GB+ VRAM for large models
- Storage: 50GB+ for models and processed data

## Alternative Installation (Docker)
For a containerized environment, consider using the provided Dockerfile (if available) or create one based on the requirements.
