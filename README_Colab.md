# RIA Segmentation Pipeline - Google Colab Edition

This repository contains a Google Colab-compatible version of the RIA (pharyngeal pumping) segmentation pipeline for C. elegans analysis using SAM2 (Segment Anything Model 2).

## 🚀 Quick Start

### 1. Upload to Google Drive

1. Create the following folder structure in your Google Drive:
```
/MyDrive/RIA_segmentation/
├── input_videos/              # Upload your video directories here
│   ├── video_001/             # Each video in its own folder
│   │   ├── 000000.jpg         # Sequential frame files
│   │   ├── 000001.jpg
│   │   └── ...
│   └── video_002/
└── output/                    # Results will be saved here automatically
```

2. Upload the following files to `/MyDrive/RIA_segmentation/`:
   - `RIA_Segmentation_Colab.ipynb`
   - `colab_ria_pipeline.py`
   - `colab_config.py`

### 2. Open in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Open the notebook: `File` → `Open notebook` → `Google Drive` → `RIA_Segmentation_Colab.ipynb`
3. Ensure you're using GPU runtime: `Runtime` → `Change runtime type` → `GPU`

### 3. Run the Pipeline

Execute the notebook cells in order:
1. **Environment Setup** - Installs dependencies and mounts Google Drive
2. **Model Setup** - Downloads and initializes SAM2 model
3. **Storage Configuration** - Sets up paths and checks for videos
4. **Video Selection** - Interactive interface to select videos for processing
5. **Processing** - Bounding box collection and SAM2 segmentation
6. **Results** - Download and view results

## 📋 Key Features

### ✅ Cloud-Native Design
- **Google Drive Integration**: Seamless file storage and access
- **Cloud Storage Support**: Compatible with Google Cloud Storage
- **Automatic Dependency Management**: One-click setup in Colab

### 🎛️ Interactive Widgets
- **Bounding Box Collection**: User-friendly coordinate input widgets
- **Real-time Validation**: Instant feedback on coordinate validity
- **Visual Reference**: Side-by-side image display with coordinates
- **Batch Processing**: Select and process multiple videos

### 🤖 SAM2 Integration
- **GPU Acceleration**: Automatic GPU detection and usage
- **Memory Optimization**: Efficient processing for large videos
- **Quality Monitoring**: Real-time feedback on segmentation quality
- **Chunked Processing**: Handle videos of any length

### 📊 Comprehensive Output
- **HDF5 Segmentation Files**: Compressed mask storage
- **Preview Videos**: Overlay visualization for quality check
- **Processing Logs**: Detailed progress and error reporting
- **Downloadable Results**: One-click download of all outputs

## 🎯 Object Identification

The pipeline segments two key anatomical structures:

### Object 1: nrD (Dorsal Pharyngeal Neuron Ring)
- **Location**: Upper/dorsal region of the pharynx
- **Color**: Red in preview videos
- **Function**: Controls dorsal pharyngeal muscle contractions

### Object 2: nrV (Ventral Pharyngeal Neuron Ring)  
- **Location**: Lower/ventral region of the pharynx
- **Color**: Blue in preview videos
- **Function**: Controls ventral pharyngeal muscle contractions

## 📐 Bounding Box Guidelines

### Coordinate System
- **Origin**: Top-left corner (0, 0)
- **X-axis**: Left to right (horizontal)
- **Y-axis**: Top to bottom (vertical)
- **Format**: [X1, Y1, X2, Y2] where (X1,Y1) = top-left, (X2,Y2) = bottom-right

### Best Practices
1. **Tight Boundaries**: Draw boxes closely around the target structures
2. **Consistent Sizing**: Use similar box sizes across videos for consistency
3. **Center Focus**: Place boxes to capture the center of neural activity
4. **Avoid Overlap**: Ensure nrD and nrV boxes don't overlap significantly

### Example Coordinates
For a 512x512 pixel image:
- **nrD**: [150, 200, 250, 300] (center-upper region)
- **nrV**: [150, 320, 250, 420] (center-lower region)

## 🔧 Technical Requirements

### System Requirements
- **GPU**: Recommended for processing speed (10-20x faster than CPU)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: ~1GB per video for temporary processing
- **Internet**: Stable connection for model downloads (~2.4GB)

### Data Format Requirements
- **Frame Format**: JPG, JPEG, or PNG
- **Naming Convention**: Sequential numeric (000000.jpg, 000001.jpg, ...)
- **Resolution**: 512x512 to 1024x1024 pixels recommended
- **Frame Rate**: Any (preview videos generated at 10 fps)

### Processing Performance
- **GPU Processing**: 1-5 fps depending on resolution
- **CPU Processing**: 0.1-0.5 fps (significantly slower)
- **Memory Usage**: 2-8GB depending on video length and resolution

## 📁 Output Files

### Segmentation Masks (`*_segments.h5`)
HDF5 files containing boolean masks for each frame and object:
```python
# Structure:
/masks/
  /frame_000000/
    /object_1    # nrD mask (boolean array)
    /object_2    # nrV mask (boolean array)
  /frame_000001/
    ...
```

### Preview Videos (`*_preview.mp4`)
MP4 files with mask overlays for visual verification:
- Red overlay: nrD (Object 1)
- Blue overlay: nrV (Object 2)
- Semi-transparent overlays on original frames

### Processing Logs
Detailed logs including:
- Processing time per video
- Number of frames processed
- Quality metrics and warnings
- Error messages and troubleshooting info

## 🛠️ Troubleshooting

### Common Issues

#### "No videos found"
- **Cause**: Videos not uploaded to correct Google Drive location
- **Solution**: Check folder structure and file paths
- **Path**: `/MyDrive/RIA_segmentation/input_videos/your_video_name/`

#### "Model loading failed"
- **Cause**: Insufficient GPU memory or network issues
- **Solution**: 
  - Restart runtime: `Runtime` → `Restart runtime`
  - Check GPU allocation: `Runtime` → `Change runtime type` → `GPU`
  - Verify internet connection for model download

#### "Processing too slow"
- **Cause**: Using CPU instead of GPU
- **Solution**: 
  - Enable GPU: `Runtime` → `Change runtime type` → `GPU`
  - Reduce video resolution or length
  - Process videos individually instead of batch

#### "Invalid coordinates"
- **Cause**: Incorrect bounding box format
- **Solution**: 
  - Ensure X1 < X2 and Y1 < Y2
  - Check coordinates are within image bounds
  - Use the reference image to verify positions

#### "Out of memory"
- **Cause**: Video too large or insufficient RAM
- **Solution**:
  - Restart runtime to clear memory
  - Process shorter video segments
  - Reduce batch size to 1 video at a time

### Getting Help

1. **Check the troubleshooting section** in the notebook
2. **Review processing logs** for specific error messages
3. **Try manual processing** with known-good coordinates
4. **Restart runtime** if experiencing persistent issues

## 📖 Advanced Usage

### Manual Processing
For debugging or testing specific coordinates:
```python
# Process single video with predefined coordinates
result = manual_process_video('video_name', {
    1: [100, 100, 200, 200],  # nrD bounding box
    2: [100, 250, 200, 350]   # nrV bounding box
})
```

### Batch Processing
Process all videos automatically (requires pre-defined coordinates):
```python
# Define coordinates for all videos
video_coords = {
    'video_001': {1: [150, 200, 250, 300], 2: [150, 320, 250, 420]},
    'video_002': {1: [140, 190, 240, 290], 2: [140, 310, 240, 410]},
    # ... more videos
}

# Process all videos
for video_name, coords in video_coords.items():
    manual_process_video(video_name, coords)
```

### Custom Configuration
Modify processing parameters in `colab_config.py`:
```python
PROCESSING_CONFIG = {
    'chunk_size': 100,      # Reduce for memory constraints
    'fps': 15,              # Higher frame rate for preview
    'preview_alpha': 0.6,   # More visible overlays
    'compression': 'lzf'    # Faster compression
}
```

## 🔬 Scientific Applications

This pipeline enables quantitative analysis of:
- **Pharyngeal Pumping Rates**: Automated counting of pumping events
- **Neural Activity Patterns**: Temporal analysis of nrD/nrV activation
- **Drug Response Studies**: Comparative analysis of pharyngeal function
- **Mutant Phenotyping**: Quantitative characterization of pharyngeal defects
- **Aging Studies**: Longitudinal tracking of pharyngeal decline

## 📚 References

- **SAM2**: [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)
- **C. elegans Pharynx**: [WormAtlas Pharynx](http://www.wormatlas.org/hermaphrodite/pharynx/mainframe.htm)
- **Google Colab**: [Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)

## 📄 License

This project is licensed under the MIT License - see the original repository for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test your changes in Colab
4. Submit a pull request with detailed description

## 📧 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the notebook outputs for error messages
3. Open an issue on the GitHub repository
4. Include your error logs and system information
