# Google Colab RIA Segmentation Configuration
# This file contains setup instructions and configuration for running the pipeline

# ============================================================================
# GOOGLE DRIVE FOLDER STRUCTURE
# ============================================================================
# Create this folder structure in your Google Drive:
#
# /MyDrive/RIA_segmentation/
# ├── input_videos/              # Upload your video directories here
# │   ├── video_001/             # Each video in its own folder
# │   │   ├── 000000.jpg         # Sequential frame files
# │   │   ├── 000001.jpg
# │   │   └── ...
# │   ├── video_002/
# │   │   ├── 000000.jpg
# │   │   └── ...
# │   └── ...
# └── output/                    # Results will be saved here automatically
#     ├── video_001_segments.h5  # Segmentation masks
#     ├── video_001_preview.mp4  # Preview video with overlays
#     └── ...

# ============================================================================
# COLAB SETUP COMMANDS (Run these in the first cell)
# ============================================================================

# Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Install required packages
# pip install -q segment-anything-2 tifffile h5py opencv-python matplotlib pandas scipy scikit-image ipywidgets
# pip install -q git+https://github.com/facebookresearch/segment-anything-2.git

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Model Configuration
MODEL_CONFIG = {
    'checkpoint_url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt',
    'model_type': 'sam2_hiera_b+.yaml',
    'device': 'auto'  # 'cuda', 'cpu', or 'auto'
}

# Storage Paths (Google Drive)
STORAGE_CONFIG = {
    'base_path': '/content/drive/MyDrive/RIA_segmentation',
    'input_videos_dir': '/content/drive/MyDrive/RIA_segmentation/input_videos',
    'output_dir': '/content/drive/MyDrive/RIA_segmentation/output',
    'temp_dir': '/content/temp_processing'
}

# Processing Parameters
PROCESSING_CONFIG = {
    'chunk_size': 200,              # Frames per processing chunk
    'fps': 10,                      # Preview video frame rate
    'preview_alpha': 0.4,           # Mask overlay transparency
    'compression': 'gzip'           # HDF5 compression
}

# Object Configuration
OBJECT_CONFIG = {
    1: {
        'name': 'nrD',
        'description': 'Dorsal pharyngeal neuron ring',
        'color': (255, 0, 0)        # Red
    },
    2: {
        'name': 'nrV', 
        'description': 'Ventral pharyngeal neuron ring',
        'color': (0, 0, 255)        # Blue
    }
}

# Widget Configuration
WIDGET_CONFIG = {
    'bbox_input_width': '120px',
    'button_width': '120px',
    'selector_height': '200px',
    'output_height': '300px'
}

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

USAGE_INSTRUCTIONS = """
1. SETUP PHASE:
   - Open this notebook in Google Colab
   - Run the setup cells to install dependencies and mount Google Drive
   - Upload your video data to the specified Google Drive folder structure

2. DATA PREPARATION:
   - Each video should be in its own subdirectory under input_videos/
   - Frames should be named sequentially (000000.jpg, 000001.jpg, etc.)
   - Supported formats: JPG, JPEG, PNG
   - Recommended resolution: 512x512 to 1024x1024 pixels

3. PROCESSING WORKFLOW:
   - Run the video selection cell to see available videos
   - Select one or more videos from the list
   - Click "Start Processing" to begin
   - For each video, you'll see:
     a) Reference image showing the first frame
     b) Interactive widgets to set bounding box coordinates
     c) Instructions for object identification

4. BOUNDING BOX ANNOTATION:
   - Object 1 (nrD): Dorsal pharyngeal neuron ring - typically in upper region
   - Object 2 (nrV): Ventral pharyngeal neuron ring - typically in lower region
   - Set coordinates as: (X1,Y1) = top-left corner, (X2,Y2) = bottom-right corner
   - Ensure X1 < X2 and Y1 < Y2 for valid bounding boxes

5. RESULTS:
   - Segmentation masks saved as HDF5 files (*.h5)
   - Preview videos with mask overlays (*.mp4)
   - Download results using the download button
   - Files also remain in your Google Drive for future access

6. TROUBLESHOOTING:
   - Use GPU runtime for faster processing (Runtime → Change runtime type → GPU)
   - If processing fails, try smaller batch sizes or single videos
   - Check Google Drive storage limits for large datasets
   - Restart runtime if you encounter memory issues
"""

# ============================================================================
# EXAMPLE BOUNDING BOX COORDINATES
# ============================================================================

EXAMPLE_BBOXES = {
    'small_worm_512x512': {
        1: [150, 200, 250, 300],  # nrD: center-upper region
        2: [150, 320, 250, 420]   # nrV: center-lower region  
    },
    'large_worm_1024x1024': {
        1: [300, 400, 500, 600],  # nrD: scaled up proportionally
        2: [300, 640, 500, 840]   # nrV: scaled up proportionally
    }
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_bbox(bbox, image_shape):
    """Validate bounding box coordinates."""
    x1, y1, x2, y2 = bbox
    height, width = image_shape[:2]
    
    if x1 >= x2 or y1 >= y2:
        return False, "Invalid coordinates: ensure x1 < x2 and y1 < y2"
    
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False, f"Coordinates outside image bounds (0,0) to ({width},{height})"
    
    bbox_area = (x2 - x1) * (y2 - y1)
    min_area = 100  # Minimum 10x10 pixels
    max_area = width * height * 0.5  # Maximum 50% of image
    
    if bbox_area < min_area:
        return False, f"Bounding box too small (minimum {min_area} pixels)"
    
    if bbox_area > max_area:
        return False, f"Bounding box too large (maximum 50% of image)"
    
    return True, "Valid bounding box"

def check_system_requirements():
    """Check if system meets requirements for processing."""
    import torch
    import psutil
    
    requirements = {
        'gpu_available': torch.cuda.is_available(),
        'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
        'system_memory': psutil.virtual_memory().total / 1e9,
        'disk_space': psutil.disk_usage('/').free / 1e9
    }
    
    print("System Requirements Check:")
    print(f"  GPU Available: {'✅' if requirements['gpu_available'] else '❌'}")
    if requirements['gpu_available']:
        print(f"  GPU Memory: {requirements['gpu_memory']:.1f} GB")
    print(f"  System Memory: {requirements['system_memory']:.1f} GB")
    print(f"  Disk Space: {requirements['disk_space']:.1f} GB")
    
    return requirements

# ============================================================================
# QUICK START COMMANDS
# ============================================================================

QUICK_START = """
# Copy and paste these commands into Colab cells:

# 1. Mount Drive and Install Packages
from google.colab import drive
drive.mount('/content/drive')
# Run in separate cell:
# !pip install -q segment-anything-2 tifffile h5py opencv-python matplotlib pandas scipy scikit-image ipywidgets
# !pip install -q git+https://github.com/facebookresearch/segment-anything-2.git

# 2. Import and Run Pipeline
exec(open('/content/drive/MyDrive/RIA_segmentation/colab_ria_pipeline.py').read())

# 3. Start Processing
main_pipeline()
"""

print("RIA Segmentation Colab Configuration Loaded")
print("=" * 50)
print("This configuration file provides:")
print("- Folder structure guidelines")
print("- Setup instructions")
print("- Processing parameters")
print("- Example bounding box coordinates")
print("- Validation functions")
print("\nTo get started, follow the USAGE_INSTRUCTIONS above.")
