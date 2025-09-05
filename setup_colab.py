#!/usr/bin/env python3
"""
Google Colab Setup Script for RIA Segmentation Pipeline

This script sets up the complete environment for running RIA segmentation
in Google Colab, including dependency installation, model downloads, and
directory structure creation.

Usage:
    1. Upload this script to your Google Drive
    2. Run it in a Colab cell to setup everything automatically
    3. Follow the prompts to configure your environment
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def check_colab_environment():
    """Check if we're running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install all required Python packages."""
    packages = [
        'segment-anything-2',
        'tifffile',
        'h5py', 
        'opencv-python',
        'matplotlib',
        'pandas',
        'scipy',
        'scikit-image',
        'ipywidgets'
    ]
    
    print("📦 Installing Python packages...")
    for package in packages:
        print(f"   Installing {package}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', package], 
                      check=True, capture_output=True)
    
    # Install SAM2 from GitHub
    print("   Installing SAM2 from GitHub...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                   'git+https://github.com/facebookresearch/segment-anything-2.git'],
                  check=True, capture_output=True)
    
    print("✅ All packages installed successfully!")

def mount_google_drive():
    """Mount Google Drive in Colab."""
    if not check_colab_environment():
        print("⚠️  Not in Colab environment, skipping Drive mount")
        return False
    
    try:
        from google.colab import drive
        print("📁 Mounting Google Drive...")
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False

def create_directory_structure(base_path='/content/drive/MyDrive/RIA_segmentation'):
    """Create the required directory structure."""
    directories = [
        base_path,
        f"{base_path}/input_videos",
        f"{base_path}/output",
        f"{base_path}/models",
        f"{base_path}/temp"
    ]
    
    print("📂 Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directory structure created!")
    return base_path

def download_sam2_model(model_dir):
    """Download SAM2 model checkpoint."""
    model_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt"
    model_path = f"{model_dir}/sam2.1_hiera_base_plus.pt"
    
    if os.path.exists(model_path):
        print("✅ SAM2 model already exists, skipping download")
        return model_path
    
    print("🤖 Downloading SAM2 model checkpoint...")
    print("   This may take a few minutes (~2.4GB download)")
    
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"✅ Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return None

def create_example_config(base_path):
    """Create an example configuration file."""
    config_content = '''
# Example RIA Segmentation Configuration
# Copy this to configure your processing parameters

# Bounding box examples for different image sizes
EXAMPLE_COORDINATES = {
    # For 512x512 images
    "small_image": {
        1: [150, 200, 250, 300],  # nrD (dorsal)
        2: [150, 320, 250, 420]   # nrV (ventral)
    },
    
    # For 1024x1024 images  
    "large_image": {
        1: [300, 400, 500, 600],  # nrD (dorsal)
        2: [300, 640, 500, 840]   # nrV (ventral)
    }
}

# Processing parameters
PROCESSING_SETTINGS = {
    "chunk_size": 200,        # Frames per chunk
    "preview_fps": 10,        # Preview video frame rate
    "overlay_alpha": 0.4,     # Mask transparency
    "use_gpu": True           # Enable GPU acceleration
}

# Object definitions
OBJECTS = {
    1: "nrD (Dorsal pharyngeal neuron ring)",
    2: "nrV (Ventral pharyngeal neuron ring)"
}
'''
    
    config_path = f"{base_path}/example_config.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"📝 Example configuration created: {config_path}")

def create_usage_instructions(base_path):
    """Create a quick reference file."""
    instructions = '''
# RIA Segmentation - Quick Reference

## 1. Upload Your Data
Place your video directories in:
{base_path}/input_videos/

Each video should be a directory containing sequential JPG frames:
video_001/
├── 000000.jpg
├── 000001.jpg
├── 000002.jpg
└── ...

## 2. Run the Pipeline
Open the notebook: RIA_Segmentation_Colab.ipynb
Execute cells in order:
1. Setup → 2. Model Init → 3. Video Selection → 4. Processing

## 3. Set Bounding Boxes
For each video, you'll specify bounding boxes for:
- Object 1 (nrD): Dorsal pharyngeal neuron ring
- Object 2 (nrV): Ventral pharyngeal neuron ring

Coordinates format: [X1, Y1, X2, Y2]
- (X1,Y1) = top-left corner
- (X2,Y2) = bottom-right corner
- Ensure X1 < X2 and Y1 < Y2

## 4. Results
Output files saved to:
{base_path}/output/
- *_segments.h5: Segmentation masks
- *_preview.mp4: Preview videos with overlays

## 5. Download Results
Use the download button in the notebook to get a ZIP file
with all results for offline analysis.

## Troubleshooting
- Enable GPU: Runtime → Change runtime type → GPU
- Restart if memory issues: Runtime → Restart runtime
- Check paths if files not found
- Ensure bounding box coordinates are valid
'''.format(base_path=base_path)
    
    instructions_path = f"{base_path}/USAGE_INSTRUCTIONS.txt"
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    print(f"📖 Usage instructions created: {instructions_path}")

def check_system_requirements():
    """Check system requirements and provide recommendations."""
    print("🔍 Checking system requirements...")
    
    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️  GPU: Not available (processing will be slower)")
    except ImportError:
        print("❌ PyTorch not available")
    
    # Check memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / 1e9
        print(f"💾 RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print("⚠️  Warning: Less than 8GB RAM may cause issues")
    except ImportError:
        print("❓ Memory check not available")
    
    # Check disk space
    if check_colab_environment():
        disk_usage = os.statvfs('/content')
        free_space_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / 1e9
        print(f"💿 Disk space: {free_space_gb:.1f} GB available")
        
        if free_space_gb < 10:
            print("⚠️  Warning: Less than 10GB disk space may cause issues")

def main():
    """Main setup function."""
    print("🚀 RIA Segmentation Pipeline - Google Colab Setup")
    print("=" * 60)
    
    # Check environment
    in_colab = check_colab_environment()
    if in_colab:
        print("✅ Running in Google Colab")
    else:
        print("⚠️  Not in Google Colab - some features may not work")
    
    # Check system requirements
    check_system_requirements()
    
    try:
        # Install dependencies
        install_dependencies()
        
        # Mount Google Drive (if in Colab)
        if in_colab:
            mount_success = mount_google_drive()
            if not mount_success:
                print("❌ Setup failed - could not mount Google Drive")
                return False
        
        # Create directory structure
        if in_colab:
            base_path = create_directory_structure()
        else:
            base_path = create_directory_structure('./RIA_segmentation')
        
        # Download SAM2 model
        model_dir = f"{base_path}/models"
        model_path = download_sam2_model(model_dir)
        
        # Create configuration files
        create_example_config(base_path)
        create_usage_instructions(base_path)
        
        # Success summary
        print("\n🎉 Setup completed successfully!")
        print("=" * 60)
        print(f"📂 Base directory: {base_path}")
        print(f"🤖 Model path: {model_path}")
        print("\n📋 Next steps:")
        print("1. Upload your video data to input_videos/")
        print("2. Open RIA_Segmentation_Colab.ipynb")
        print("3. Run the notebook cells to start processing")
        print("\n📖 See USAGE_INSTRUCTIONS.txt for detailed guidance")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        print("Please check your internet connection and try again")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
