@echo off
REM setup.bat - Windows setup script for RIA Segmentation Pipeline

echo 🔬 RIA Segmentation Pipeline Setup (Windows)
echo ===========================================

REM Check Python version
python --version
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "ria_segmentation" (
    echo 🔨 Creating virtual environment...
    python -m venv ria_segmentation
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call ria_segmentation\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch
echo 🔥 Installing PyTorch...
set /p has_cuda="Do you have CUDA installed? (y/n): "
if /i "%has_cuda%"=="y" (
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Installing PyTorch ^(CPU only^)...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

REM Install other requirements
echo 📦 Installing other dependencies...
pip install -r requirements.txt

REM Clone and install SAM2
if not exist "segment-anything-2" (
    echo 🤖 Installing SAM2...
    git clone https://github.com/facebookresearch/segment-anything-2.git
    cd segment-anything-2
    pip install -e .
    cd ..
) else (
    echo ✅ SAM2 already installed
)

REM Create checkpoint directory
if not exist "sam2_checkpoints" mkdir sam2_checkpoints

REM Download model checkpoints
echo ⬇️  Downloading SAM2 model checkpoints...
echo Available models:
echo 1. Tiny ^(fastest, ~150MB^)
echo 2. Base+ ^(balanced, ~900MB^)
echo 3. Large ^(most accurate, ~2.5GB^)
echo 4. All models

set /p model_choice="Which model(s) to download? (1-4): "

if "%model_choice%"=="1" (
    echo Downloading Tiny model...
    powershell -Command "Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt' -OutFile 'sam2_checkpoints\sam2.1_hiera_tiny.pt'"
) else if "%model_choice%"=="2" (
    echo Downloading Base+ model...
    powershell -Command "Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt' -OutFile 'sam2_checkpoints\sam2.1_hiera_base_plus.pt'"
) else if "%model_choice%"=="3" (
    echo Downloading Large model...
    powershell -Command "Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt' -OutFile 'sam2_checkpoints\sam2.1_hiera_large.pt'"
) else if "%model_choice%"=="4" (
    echo Downloading all models...
    powershell -Command "Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt' -OutFile 'sam2_checkpoints\sam2.1_hiera_tiny.pt'"
    powershell -Command "Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt' -OutFile 'sam2_checkpoints\sam2.1_hiera_base_plus.pt'"
    powershell -Command "Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt' -OutFile 'sam2_checkpoints\sam2.1_hiera_large.pt'"
) else (
    echo Invalid choice. Skipping model download.
)

REM Test installation
echo 🧪 Testing installation...
python -c "import torch; import numpy as np; import cv2; import matplotlib.pyplot as plt; print('✅ Core dependencies installed successfully'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); from sam2.build_sam import build_sam2_video_predictor; print('✅ SAM2 installed successfully')"

echo.
echo 🎉 Setup completed!
echo.
echo Next steps:
echo 1. Update checkpoint paths in processing scripts ^(3AutoCrop.py, 4RIAMaskGen.py, 6SegmentWorm.py^)
echo 2. Place your TIFF files in the TIFF\ directory
echo 3. Run the pipeline scripts in order ^(1-7^)
echo.
echo To activate the environment in the future:
echo ria_segmentation\Scripts\activate.bat

pause
