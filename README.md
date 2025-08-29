# RIA Segmentation Pipeline

![Workflow Overview](workflow.png)

## Overview

This repository contains a computer vision pipeline for analyzing the **RIA interneuron** in *C. elegans* worms in an olfactory chip using **SAM2 (Segment Anything Model 2)** for video segmentation. The pipeline processes calcium imaging data to automatically segment and analyze neuronal regions (nrD and nrV) and head movement patterns.

## Pipeline Workflow

The analysis pipeline consists of 7 steps, each handled by a numbered Python script:

### 1. **Convert TIF to Stack** (`1ConvertTIFFtoStack.py`)
- **Purpose**: Converts individual TIFF files into multi-frame TIFF stacks
- **Input**: Directory containing individual TIFF files
- **Output**: Multi-frame TIFF stacks (`*_stack.tiff`)
- **Function**: Consolidates temporal sequences into single stack files for easier processing

### 2. **TIFF to JPG Conversion** (`2TIFF2JPG.py`)
- **Purpose**: Converts TIFF stacks to JPG image sequences with proper normalization
- **Input**: TIFF stacks from step 1
- **Output**: JPG frame sequences in `JPG/` directory
- **Features**: 
  - Multiple normalization methods (minmax, percentile, global)
  - Preserves dynamic range while converting to 8-bit
  - Maintains metadata for processing verification

### 3. **Automatic Cropping** (`3AutoCrop.py`)
- **Purpose**: Automatically crops videos around the RIA region using SAM2
- **Input**: JPG frame sequences or TIFF stacks
- **Output**: Cropped frame sequences in `CROP/` and `CROP_JPG/` directories
- **Method**: 
  - Interactive bounding box selection on first frame
  - SAM2-based region tracking across frames
  - Fixed crop window calculation for consistent region extraction

### 4. **RIA Mask Generation** (`4RIAMaskGen.py`)
- **Purpose**: Performs SAM2 video segmentation with interactive prompts for nrD and nrV neurons
- **Input**: Cropped JPG frame sequences from step 3
- **Output**: 
  - Segmentation masks saved as H5 files in `Segmentation/`
  - Optional overlay videos for quality inspection
- **Features**:
  - Interactive bounding box selection on first frame
  - Temporal mask propagation across video frames
  - Quality analysis and re-prompting capabilities
  - Chunked processing for large videos

### 5. **Mask Analysis** (`5AnalyzeMasks.py`)
- **Purpose**: Extracts brightness and morphometric data from segmented regions
- **Input**: H5 mask files from step 4 and corresponding TIFF stacks
- **Output**: CSV files with quantitative measurements in `ANALYSIS/`
- **Measurements**:
  - Mean brightness values for each segmented region
  - Background-corrected brightness values
  - Top percentile brightness analysis (top 50%, 25%, 10%)
  - Pixel counts and spatial statistics
  - Left/right side determination

### 6. **Head Segmentation** (`6SegmentWorm.py`)
- **Purpose**: Segments worm head regions using SAM2 for head angle analysis
- **Input**: JPG frame sequences
- **Output**: Head segmentation masks in `HEAD_SEGMENT/` directory
- **Method**: Similar to step 4 but focused on head region segmentation

### 7. **Head Angle Analysis** (`7HeadAngle.py`)
- **Purpose**: Calculates head angles and bend positions from head segmentation masks
- **Input**: Head segmentation H5 files from step 6
- **Output**: Head angle measurements and analysis in `HEADBEND_ANALYSIS/`
- **Analysis**:
  - Skeleton extraction from head masks
  - Gaussian-weighted curvature calculation
  - Head angle computation with temporal smoothing
  - Bend location detection along head skeleton

## Directory Structure

```
RIA_segmentation/
├── 1ConvertTIFFtoStack.py       # Step 1: TIFF stack creation
├── 2TIFF2JPG.py                 # Step 2: Format conversion
├── 3AutoCrop.py                 # Step 3: Automatic cropping
├── 4RIAMaskGen.py               # Step 4: RIA segmentation
├── 5AnalyzeMasks.py             # Step 5: Brightness analysis
├── 6SegmentWorm.py              # Step 6: Head segmentation
├── 7HeadAngle.py                # Step 7: Head angle analysis
├── TIFF/                        # Input TIFF stacks
├── JPG/                         # Converted JPG sequences
├── CROP/                        # Cropped TIFF stacks
├── CROP_JPG/                    # Cropped JPG sequences
├── Segmentation/                # RIA segmentation masks (H5)
├── HEAD_SEGMENT/                # Head segmentation masks (H5)
├── ANALYSIS/                    # Final brightness measurements (CSV)
└── HEADBEND_ANALYSIS/           # Head angle measurements (CSV)
```

## Installation

### Quick Setup
```bash
# Clone repository
git clone https://github.com/Sebzdead/RIA-Segmentation.git
cd RIA-Segmentation

# Create virtual environment
conda create -n ria_segmentation python=3.9
conda activate ria_segmentation

# Install dependencies
pip install -r requirements.txt

# Install SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e . && cd ..
```

**📋 For detailed installation instructions, see [INSTALL.md](INSTALL.md)**

## Requirements

### Core Dependencies
- **Python 3.8+**
- **PyTorch** (CUDA-enabled for GPU acceleration)
- **SAM2** (Segment Anything 2)
- **OpenCV** (`cv2`)
- **NumPy, Pandas, SciPy, Matplotlib**
- **h5py, tifffile, scikit-image, tqdm**

### Hardware Recommendations
- **RAM**: 16GB+ 
- **GPU**: 8GB+ VRAM (for large SAM2 models)
- **Storage**: 50GB+ for models and processed data

## SAM2 Setup

The pipeline uses SAM2 for video segmentation. After installation:

1. **Download Model Checkpoints**:
   - Tiny: `sam2.1_hiera_tiny.pt` (fastest)
   - Base+: `sam2.1_hiera_base_plus.pt` (balanced)
   - Large: `sam2.1_hiera_large.pt` (most accurate)

2. **Configure Paths** in processing scripts:
```python
sam2_checkpoint = r"path/to/sam2_checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = r"path/to/segment-anything-2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
```

## Usage

1. **Install Dependencies**: Follow installation guide in [INSTALL.md](INSTALL.md)
2. **Prepare Data**: Place your TIFF files in the `TIFF/` directory
3. **Run Pipeline**: Execute scripts in numerical order (1-7)
4. **Interactive Steps**: Scripts 3, 4, and 6 require user interaction for initial prompting
5. **Monitor Progress**: Each script provides detailed progress feedback and quality metrics
5. **Review Results**: Check output directories for processed data and analysis results

## Key Features

- **Automated Processing**: Minimal user intervention after initial setup
- **Quality Control**: Built-in quality assessment and re-prompting capabilities
- **Scalability**: Chunked processing for large video datasets
- **Flexibility**: Multiple normalization and analysis options
- **Reproducibility**: Consistent processing parameters across datasets

## Output Data

The pipeline generates comprehensive datasets including:
- **Segmentation Masks**: Binary masks for each neuronal region
- **Brightness Measurements**: Temporal brightness profiles with background correction
- **Morphometric Data**: Pixel counts, centroids, and spatial relationships
- **Head Kinematics**: Head angles, bend positions, and movement patterns
- **Quality Metrics**: Confidence scores and processing validation data

## Citation

If you use this pipeline in your research, please cite the associated publication and acknowledge the use of SAM2 for video segmentation.

## Support

For questions or issues, please refer to the individual script documentation or contact the development team.