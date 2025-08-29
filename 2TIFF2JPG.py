"""
Convert TIFF stacks to JPG image sequences while maintaining dynamic range.
This script processes all .tif/.tiff files in a directory and converts each stack
to a series of JPG images in individual folders.
"""
import os
import cv2
import numpy as np
import tifffile
from tqdm import tqdm

def normalize_to_8bit(image, method='minmax'):
    """
    Normalize image to 8-bit while preserving dynamic range.
    
    Args:
        image (np.ndarray): Input image
        method (str): Normalization method ('minmax', 'percentile', 'global')
    
    Returns:
        np.ndarray: 8-bit normalized image
    """
    if image.dtype == np.uint8:
        return image
    
    if method == 'minmax':
        # Simple min-max normalization
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
    
    elif method == 'percentile':
        # Use percentile-based normalization to handle outliers
        p_low, p_high = np.percentile(image, [1, 99])
        if p_high > p_low:
            normalized = np.clip((image - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
    
    elif method == 'global':
        # Use global min/max across entire stack (handled externally)
        normalized = image.astype(np.uint8)
    
    return normalized

def convert_tiff_stack_to_jpg(tiff_path, output_base_dir, normalization='minmax', quality=95):
    """
    Convert a single TIFF stack to JPG sequence.
    
    Args:
        tiff_path (str): Path to TIFF stack file
        output_base_dir (str): Base directory for output
        normalization (str): Normalization method
        quality (int): JPG quality (0-100)
    """
    try:
        # Load TIFF stack
        print(f"Loading TIFF stack: {tiff_path}")
        tiff_stack = tifffile.imread(tiff_path)
        
        # Handle different stack dimensions
        if tiff_stack.ndim == 2:
            # Single image
            tiff_stack = tiff_stack[np.newaxis, ...]
        elif tiff_stack.ndim == 4:
            # Multi-channel stack - convert to grayscale or handle first channel
            print(f"Multi-channel stack detected with shape {tiff_stack.shape}")
            if tiff_stack.shape[-1] == 3:  # RGB
                tiff_stack = np.mean(tiff_stack, axis=-1).astype(tiff_stack.dtype)
            else:
                tiff_stack = tiff_stack[..., 0]  # Take first channel
        
        n_frames = tiff_stack.shape[0]
        print(f"Stack contains {n_frames} frames with shape {tiff_stack.shape[1:]}")
        print(f"Data type: {tiff_stack.dtype}, Range: {tiff_stack.min()} - {tiff_stack.max()}")
        
        # Create output directory
        base_name = os.path.splitext(os.path.basename(tiff_path))[0]
        output_dir = os.path.join(output_base_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Global normalization if requested
        if normalization == 'global':
            global_min = tiff_stack.min()
            global_max = tiff_stack.max()
            print(f"Using global normalization: {global_min} - {global_max}")
        
        # Convert each frame
        for i in tqdm(range(n_frames), desc=f"Converting {base_name}"):
            frame = tiff_stack[i]
            
            # Normalize to 8-bit
            if normalization == 'global':
                if global_max > global_min:
                    normalized_frame = ((frame - global_min) / (global_max - global_min) * 255).astype(np.uint8)
                else:
                    normalized_frame = np.zeros_like(frame, dtype=np.uint8)
            else:
                normalized_frame = normalize_to_8bit(frame, method=normalization)
            
            # Save as JPG
            output_path = os.path.join(output_dir, f"{i:06d}.jpg")
            
            # Use OpenCV for JPG saving with quality control
            success = cv2.imwrite(output_path, normalized_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not success:
                print(f"Warning: Failed to save frame {i} to {output_path}")
        
        print(f"Successfully converted {n_frames} frames to {output_dir}")
        
        # Save metadata file
        metadata_path = os.path.join(output_dir, "conversion_info.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Original file: {tiff_path}\n")
            f.write(f"Original shape: {tiff_stack.shape}\n")
            f.write(f"Original dtype: {tiff_stack.dtype}\n")
            f.write(f"Original range: {tiff_stack.min()} - {tiff_stack.max()}\n")
            f.write(f"Normalization method: {normalization}\n")
            f.write(f"JPG quality: {quality}\n")
            f.write(f"Number of frames: {n_frames}\n")
        
        return True
        
    except Exception as e:
        print(f"Error processing {tiff_path}: {str(e)}")
        return False

def process_tiff_directory(input_dir, output_dir, normalization='minmax', quality=100):
    """
    Process all TIFF files in a directory.
    
    Args:
        input_dir (str): Directory containing TIFF files
        output_dir (str): Directory to save JPG sequences
        normalization (str): Normalization method ('minmax', 'percentile', 'global')
        quality (int): JPG quality (0-100)
    """
    # Find all TIFF files
    tiff_files = []
    for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
        tiff_files.extend([f for f in os.listdir(input_dir) if f.endswith(ext)])
    
    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files to process")
    print(f"Normalization method: {normalization}")
    print(f"JPG quality: {quality}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    successful = 0
    failed = 0
    
    for tiff_file in tiff_files:
        tiff_path = os.path.join(input_dir, tiff_file)
        print(f"\n{'='*60}")
        print(f"Processing: {tiff_file}")
        print(f"{'='*60}")
        
        if convert_tiff_stack_to_jpg(tiff_path, output_dir, normalization, quality):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"{'='*60}")

def main():
    """Main function with configurable parameters."""
    
    # Configuration
    input_directory = "TIFF/good"  # Directory containing TIFF stacks
    output_directory = "JPG"  # Directory to save JPG sequences
    
    # Normalization options:
    # 'minmax' - Simple min-max normalization per frame (preserves frame-to-frame variation)
    # 'percentile' - Percentile-based normalization per frame (handles outliers)
    # 'global' - Global min-max across entire stack (maintains relative intensities)
    normalization_method = 'minmax'
    
    jpg_quality = 100  # JPG quality (0-100)
    
    # Check if input directory exists
    if not os.path.exists(input_directory):
        print(f"Input directory '{input_directory}' does not exist.")
        print("Please create the directory and place your TIFF files there.")
        return
    
    print("TIFF to JPG Converter")
    print("=" * 40)
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Normalization: {normalization_method}")
    print(f"JPG quality: {jpg_quality}")
    print("=" * 40)
    
    # Process all TIFF files
    process_tiff_directory(input_directory, output_directory, normalization_method, jpg_quality)

if __name__ == "__main__":
    main()
