import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tifffile
import random

# Directory paths
segments_dir = '5RIA_SEGMENT'
tiff_dir = '3CROP'
jpg_dir = '4CROP_JPG'
output_dir = '7VISUALIZATIONS'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_masks_from_h5(h5_path):
    """
    Load masks from an H5 file.
    
    Args:
        h5_path (str): Path to the H5 file
        
    Returns:
        tuple: (masks_dict, metadata) where masks_dict is {frame_idx: {obj_id: mask}}
    """
    masks_by_frame = {}
    metadata = {}
    original_frame_indices = []  # Track original frame order
    
    with h5py.File(h5_path, 'r') as f:
        # Load metadata
        metadata['num_frames'] = f.attrs['num_frames']
        metadata['object_ids'] = [int(obj_id) if obj_id.isdigit() else obj_id 
                                 for obj_id in f.attrs['object_ids']]
        
        masks_group = f['masks']
        
        # Debug: Check how data is stored in H5 file
        print(f"\n=== DEBUG: H5 FILE STRUCTURE ===")
        print(f"File: {os.path.basename(h5_path)}")
        print(f"Metadata num_frames: {metadata['num_frames']}")
        print(f"Object IDs: {metadata['object_ids']}")
        
        # First, determine the frame ordering by looking at the frame_indices attribute if it exists
        # or try to reconstruct it from the original video processing
        if 'frame_indices' in f.attrs:
            original_frame_indices = f.attrs['frame_indices']
            print(f"Found stored frame indices: {original_frame_indices[:10]}{'...' if len(original_frame_indices) > 10 else ''}")
        else:
            print("No frame_indices attribute found - frames were stored in reverse order")
            # Since 4RIAMaskGen.py uses sorted(..., reverse=True), we need to reverse the H5 index order
            num_frames = len(masks_group[f.attrs['object_ids'][0]])
            original_frame_indices = list(range(num_frames-1, -1, -1))  # Reverse order
            print(f"Reconstructed frame order (highest to lowest): {original_frame_indices[:10]}{'...' if len(original_frame_indices) > 10 else ''}")
        
        # Load masks for each object using corrected frame order
        for obj_id_str in f.attrs['object_ids']:
            obj_id = int(obj_id_str) if obj_id_str.isdigit() else obj_id_str
            masks_data = masks_group[obj_id_str][:]
            
            print(f"Object {obj_id}: mask array shape {masks_data.shape}")
            
            # Map H5 array indices to correct frame indices
            for h5_idx in range(len(masks_data)):
                if h5_idx < len(original_frame_indices):
                    # The actual frame index this H5 position represents
                    correct_frame_idx = original_frame_indices[h5_idx]
                    
                    if correct_frame_idx not in masks_by_frame:
                        masks_by_frame[correct_frame_idx] = {}
                    masks_by_frame[correct_frame_idx][obj_id] = masks_data[h5_idx]
                    
                    # Debug first few mappings
                    if h5_idx < 5:
                        pixel_count = np.sum(masks_data[h5_idx])
                        print(f"  H5 index {h5_idx} -> frame {correct_frame_idx}: {pixel_count} pixels")
        
        # Debug: Check final frame indices
        frame_indices = sorted(masks_by_frame.keys())
        print(f"Final frame indices: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")
        print(f"Total frames loaded: {len(frame_indices)}")
        print(f"Frame range: {min(frame_indices)} to {max(frame_indices)}")
        
        print(f"=== END DEBUG ===\n")
    
    print(f"Loaded masks: {len(masks_by_frame)} frames, objects: {metadata['object_ids']}")
    return masks_by_frame, metadata

def debug_h5_structure(h5_path):
    """
    Debug function to investigate H5 file structure and frame ordering.
    
    Args:
        h5_path (str): Path to the H5 file
    """
    print(f"\n=== DETAILED H5 STRUCTURE DEBUG ===")
    print(f"File: {os.path.basename(h5_path)}")
    
    with h5py.File(h5_path, 'r') as f:
        print(f"\nH5 File Attributes:")
        for attr_name in f.attrs.keys():
            attr_value = f.attrs[attr_name]
            print(f"  {attr_name}: {attr_value} (type: {type(attr_value)})")
        
        print(f"\nH5 File Structure:")
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}Dataset: {name}")
                print(f"{indent}  Shape: {obj.shape}")
                print(f"{indent}  Dtype: {obj.dtype}")
                if obj.size < 50:  # Only print small arrays
                    print(f"{indent}  Data: {obj[:]}")
        
        f.visititems(print_structure)
        
        # Investigate mask storage order and reverse order issue
        if 'masks' in f:
            masks_group = f['masks']
            print(f"\nMask Storage Investigation:")
            print(f"INVESTIGATING REVERSE ORDER ISSUE:")
            print(f"In 4RIAMaskGen.py, frames are sorted with reverse=True:")
            print(f"  sorted(filtered_video_segments.items(), reverse=True)")
            print(f"This means frame indices are stored in DESCENDING order in H5 file")
            
            for obj_id_str in f.attrs['object_ids']:
                if obj_id_str in masks_group:
                    mask_dataset = masks_group[obj_id_str]
                    print(f"\nObject {obj_id_str}:")
                    print(f"  Dataset shape: {mask_dataset.shape}")
                    print(f"  Dataset dtype: {mask_dataset.dtype}")
                    
                    # Check first and last few masks
                    num_frames = mask_dataset.shape[0]
                    print(f"  Total frames: {num_frames}")
                    
                    # Sample first 3 and last 3 frames
                    sample_indices = list(range(min(3, num_frames))) + list(range(max(0, num_frames-3), num_frames))
                    sample_indices = sorted(set(sample_indices))  # Remove duplicates and sort
                    
                    print(f"  Sampling frames: {sample_indices}")
                    print(f"  NOTE: Due to reverse=True in save function, these indices represent:")
                    print(f"    H5 index 0 = highest original frame number")
                    print(f"    H5 index {num_frames-1} = lowest original frame number")
                    
                    for i in sample_indices:
                        mask = mask_dataset[i]
                        pixel_count = np.sum(mask)
                        print(f"    H5 frame {i}: {pixel_count} pixels, shape {mask.shape}")
                        if pixel_count > 0:
                            y_coords, x_coords = np.where(mask.squeeze())
                            if len(x_coords) > 0:
                                centroid_x = np.mean(x_coords)
                                centroid_y = np.mean(y_coords)
                                print(f"      Centroid: ({centroid_x:.1f}, {centroid_y:.1f})")
    
    print(f"=== END DETAILED DEBUG ===\n")

def debug_mask_alignment(h5_path, image_source, video_name):
    """
    Debug mask-image alignment issues by comparing frame content.
    
    Args:
        h5_path (str): Path to the H5 file
        image_source (str): 'tiff' or 'jpg'
        video_name (str): Name of the video
    """
    print(f"\n=== DEBUGGING ALIGNMENT FOR {video_name} ===")
    
    # Load masks using our function
    masks_dict, metadata = load_masks_from_h5(h5_path)
    
    # Load images
    if image_source == 'tiff':
        tiff_path = os.path.join(tiff_dir, f"{video_name}.tif")
        image_stack = load_tiff_stack(tiff_path)
    else:
        jpg_path = os.path.join(jpg_dir, video_name)
        image_stack = load_jpg_frames(jpg_path)
    
    if image_stack is None:
        print("Failed to load images for alignment debug")
        return
    
    print(f"\nAlignment Analysis:")
    print(f"Masks: {len(masks_dict)} frames, indices: {sorted(list(masks_dict.keys())[:5])}...")
    print(f"Images: {len(image_stack)} frames")
    
    # Test first few frames for alignment
    mask_frame_indices = sorted(list(masks_dict.keys())[:5])
    
    for mask_frame_idx in mask_frame_indices:
        print(f"\n--- Testing mask frame {mask_frame_idx} ---")
        
        frame_masks = masks_dict[mask_frame_idx]
        has_masks = any(mask is not None and np.sum(mask.squeeze()) > 0 
                       for mask in frame_masks.values())
        
        if not has_masks:
            print(f"  No valid masks in frame {mask_frame_idx}")
            continue
        
        # Try different image indices
        for img_idx in [mask_frame_idx, mask_frame_idx-1, mask_frame_idx+1, 0, len(image_stack)-1-mask_frame_idx]:
            if 0 <= img_idx < len(image_stack):
                print(f"  Testing against image[{img_idx}]...")
                
                # Calculate some basic image statistics for comparison
                image = image_stack[img_idx]
                img_mean = np.mean(image)
                img_std = np.std(image)
                
                print(f"    Image stats: mean={img_mean:.1f}, std={img_std:.1f}")
                
                # Check if mask regions make sense
                for obj_id, mask in frame_masks.items():
                    if mask is not None:
                        mask_2d = mask.squeeze()
                        pixel_count = np.sum(mask_2d)
                        if pixel_count > 0:
                            # Get masked region statistics
                            masked_pixels = image[mask_2d]
                            mask_mean = np.mean(masked_pixels)
                            mask_std = np.std(masked_pixels)
                            
                            # Get background statistics (inverse mask)
                            bg_mask = ~mask_2d
                            if np.sum(bg_mask) > 0:
                                bg_pixels = image[bg_mask]
                                bg_mean = np.mean(bg_pixels)
                                contrast = abs(mask_mean - bg_mean)
                                
                                print(f"    Object {obj_id}: {pixel_count} pixels, "
                                      f"mask_mean={mask_mean:.1f}, bg_mean={bg_mean:.1f}, "
                                      f"contrast={contrast:.1f}")
                            else:
                                print(f"    Object {obj_id}: {pixel_count} pixels, mask_mean={mask_mean:.1f}")
    
    print(f"=== END ALIGNMENT DEBUG ===\n")

def load_tiff_stack(tiff_path):
    """
    Load a TIFF stack.
    
    Args:
        tiff_path (str): Path to the TIFF file
        
    Returns:
        np.ndarray: TIFF stack array
    """
    try:
        stack = tifffile.imread(tiff_path)
        print(f"Loaded TIFF stack: {stack.shape}")
        return stack
    except Exception as e:
        print(f"Error loading TIFF stack {tiff_path}: {e}")
        return None

def load_jpg_frames(jpg_folder_path):
    """
    Load JPG frames from a folder.
    
    Args:
        jpg_folder_path (str): Path to the folder containing JPG frames
        
    Returns:
        np.ndarray: Array of images with shape (num_frames, height, width)
    """
    try:
        if not os.path.exists(jpg_folder_path):
            print(f"JPG folder does not exist: {jpg_folder_path}")
            return None
            
        # Get all JPG files
        jpg_files = [f for f in os.listdir(jpg_folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        jpg_files.sort()  # Sort to ensure correct order
        
        if not jpg_files:
            print(f"No JPG files found in {jpg_folder_path}")
            return None
        
        print(f"Found {len(jpg_files)} JPG files")
        
        # Load first image to get dimensions
        first_img_path = os.path.join(jpg_folder_path, jpg_files[0])
        from PIL import Image
        first_img = np.array(Image.open(first_img_path).convert('L'))  # Convert to grayscale
        
        # Initialize array
        frames = np.zeros((len(jpg_files), first_img.shape[0], first_img.shape[1]), dtype=np.uint8)
        frames[0] = first_img
        
        # Load remaining frames
        for i, jpg_file in enumerate(jpg_files[1:], 1):
            img_path = os.path.join(jpg_folder_path, jpg_file)
            img = np.array(Image.open(img_path).convert('L'))
            frames[i] = img
        
        print(f"Loaded JPG frames: {frames.shape}")
        return frames
        
    except Exception as e:
        print(f"Error loading JPG frames from {jpg_folder_path}: {e}")
        return None

def create_interactive_viewer(tiff_stack, masks_dict, video_name, save_dir):
    """
    Create an interactive viewer with a slider to navigate through frames.
    
    Args:
        tiff_stack (np.ndarray): TIFF image stack
        masks_dict (dict): Dictionary of masks by frame
        video_name (str): Name of the video for display
        save_dir (str): Directory to save screenshots
    """
    # Colors for different objects (consistent with previous scripts)
    object_colors = {
        1: (1.0, 0.0, 0.0, 0.5),    # Red for nrD
        2: (0.0, 0.0, 1.0, 0.5),    # Blue for nrV
        3: (0.0, 1.0, 0.0, 0.5),    # Green
        4: (1.0, 0.0, 1.0, 0.5),    # Magenta
        5: (0.0, 1.0, 1.0, 0.5),    # Cyan
    }
    
    # Default colors for unexpected object IDs
    default_colors = [(0.5, 0.5, 0.5, 0.5), (0.8, 0.4, 0.2, 0.5), (0.2, 0.8, 0.4, 0.5)]
    
    num_frames = min(len(tiff_stack), len(masks_dict))
    
    # Create figure and subplots
    fig, (ax_orig, ax_overlay) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Video: {video_name}', fontsize=14, fontweight='bold')
    
    # Make room for slider
    plt.subplots_adjust(bottom=0.15)
    
    # Initialize with first frame
    current_frame = 0
    
    def get_frame_image(frame_idx):
        """Get image for given frame index, handling potential offset"""
        possible_indices = [frame_idx, frame_idx - 1, 0 if frame_idx == 1 else frame_idx]
        
        for tiff_idx in possible_indices:
            if 0 <= tiff_idx < len(tiff_stack):
                return tiff_stack[tiff_idx]
        
        print(f"Warning: Could not find image for frame {frame_idx}")
        return np.zeros((100, 100), dtype=np.uint8)
    
    def update_display(frame_idx):
        """Update the display for the given frame"""
        ax_orig.clear()
        ax_overlay.clear()
        
        # Get the image
        image = get_frame_image(frame_idx)
        
        # Display original image
        ax_orig.imshow(image, cmap='gray')
        ax_orig.set_title(f'Original Frame {frame_idx}')
        ax_orig.axis('off')
        
        # Display image with mask overlay
        ax_overlay.imshow(image, cmap='gray')
        
        # Add masks if they exist for this frame
        if frame_idx in masks_dict:
            frame_masks = masks_dict[frame_idx]
            
            for obj_id, mask in frame_masks.items():
                if mask is None:
                    continue
                    
                # Squeeze mask to 2D
                mask_2d = mask.squeeze()
                
                # Get color for this object
                if obj_id in object_colors:
                    color = object_colors[obj_id]
                else:
                    color_idx = hash(obj_id) % len(default_colors)
                    color = default_colors[color_idx]
                
                # Create colored overlay
                if mask_2d.sum() > 0:  # Only if mask has pixels
                    overlay = np.zeros((*mask_2d.shape, 4))
                    overlay[mask_2d, :] = color
                    ax_overlay.imshow(overlay, alpha=0.7)
                    
                    # Add object ID label at centroid
                    y_coords, x_coords = np.where(mask_2d)
                    if len(x_coords) > 0:
                        centroid_x = np.mean(x_coords)
                        centroid_y = np.mean(y_coords)
                        ax_overlay.text(centroid_x, centroid_y, str(obj_id), 
                                      color='white', fontsize=12, fontweight='bold',
                                      ha='center', va='center',
                                      bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor=color[:3], alpha=0.8))
        
        ax_overlay.set_title(f'Frame {frame_idx} with Masks')
        ax_overlay.axis('off')
        
        # Add frame info
        info_text = f"Frame: {frame_idx}/{num_frames-1}"
        if frame_idx in masks_dict:
            obj_ids = [oid for oid in masks_dict[frame_idx].keys() if masks_dict[frame_idx][oid] is not None]
            info_text += f"\nObjects: {obj_ids}"
            
            # Add pixel counts
            pixel_counts = []
            for oid in obj_ids:
                mask = masks_dict[frame_idx][oid]
                if mask is not None:
                    count = np.sum(mask.squeeze())
                    pixel_counts.append(f"{oid}:{count}")
            if pixel_counts:
                info_text += f"\nPixels: {', '.join(pixel_counts)}"
        
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.draw()
    
    def on_slider_change(val):
        """Callback for slider change"""
        nonlocal current_frame
        current_frame = int(val)
        update_display(current_frame)
    
    def on_save_click(event):
        """Save current frame view"""
        # Extract video name without source suffix for cleaner filename
        clean_video_name = video_name.replace(' (TIFF)', '').replace(' (JPG)', '')
        # Determine source from video_name display
        source_suffix = '_tiff' if '(TIFF)' in video_name else '_jpg' if '(JPG)' in video_name else ''
        save_path = os.path.join(save_dir, f"{clean_video_name}{source_suffix}_frame_{current_frame:06d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved frame {current_frame} to {save_path}")
    
    def on_prev_click(event):
        """Go to previous frame"""
        nonlocal current_frame
        if current_frame > 0:
            current_frame -= 1
            slider.set_val(current_frame)
    
    def on_next_click(event):
        """Go to next frame"""
        nonlocal current_frame
        if current_frame < num_frames - 1:
            current_frame += 1
            slider.set_val(current_frame)
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valfmt='%d')
    slider.on_changed(on_slider_change)
    
    # Create buttons
    ax_prev = plt.axes([0.1, 0.05, 0.08, 0.04])
    btn_prev = Button(ax_prev, 'Previous')
    btn_prev.on_clicked(on_prev_click)
    
    ax_next = plt.axes([0.72, 0.05, 0.08, 0.04])
    btn_next = Button(ax_next, 'Next')
    btn_next.on_clicked(on_next_click)
    
    ax_save = plt.axes([0.82, 0.05, 0.08, 0.04])
    btn_save = Button(ax_save, 'Save')
    btn_save.on_clicked(on_save_click)
    
    # Initial display
    update_display(current_frame)
    
    plt.show()

def create_summary_grid(tiff_stack, masks_dict, video_name, save_dir, max_frames=16):
    """
    Create a summary grid showing multiple frames.
    
    Args:
        tiff_stack (np.ndarray): TIFF image stack
        masks_dict (dict): Dictionary of masks by frame
        video_name (str): Name of the video
        save_dir (str): Directory to save the grid
        max_frames (int): Maximum number of frames to show in grid
    """
    num_frames = min(len(tiff_stack), len(masks_dict), max_frames)
    
    # Select frames evenly distributed across the video
    frame_indices = np.linspace(0, min(len(tiff_stack), len(masks_dict)) - 1, num_frames, dtype=int)
    
    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(num_frames)))
    rows = int(np.ceil(num_frames / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    fig.suptitle(f'Overview: {video_name}', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Colors for objects
    object_colors = {
        1: (1.0, 0.0, 0.0, 0.6),    # Red for nrD
        2: (0.0, 0.0, 1.0, 0.6),    # Blue for nrV
        3: (0.0, 1.0, 0.0, 0.6),    # Green
        4: (1.0, 0.0, 1.0, 0.6),    # Magenta
        5: (0.0, 1.0, 1.0, 0.6),    # Cyan
    }
    
    for i, frame_idx in enumerate(frame_indices):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Get image (handle potential offset)
        possible_indices = [frame_idx, frame_idx - 1, 0 if frame_idx == 1 else frame_idx]
        image = None
        for tiff_idx in possible_indices:
            if 0 <= tiff_idx < len(tiff_stack):
                image = tiff_stack[tiff_idx]
                break
        
        if image is None:
            ax.text(0.5, 0.5, f'Frame {frame_idx}\nNo Image', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Display image
        ax.imshow(image, cmap='gray')
        
        # Add masks
        if frame_idx in masks_dict:
            frame_masks = masks_dict[frame_idx]
            
            for obj_id, mask in frame_masks.items():
                if mask is None:
                    continue
                    
                mask_2d = mask.squeeze()
                
                if mask_2d.sum() > 0:
                    color = object_colors.get(obj_id, (0.5, 0.5, 0.5, 0.6))
                    overlay = np.zeros((*mask_2d.shape, 4))
                    overlay[mask_2d, :] = color
                    ax.imshow(overlay, alpha=0.8)
        
        ax.set_title(f'Frame {frame_idx}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(num_frames, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save the grid
    # Extract clean video name without source suffix
    clean_video_name = video_name.replace(' (TIFF)', '').replace(' (JPG)', '')
    # Determine source from video_name display
    source_suffix = '_tiff' if '(TIFF)' in video_name else '_jpg' if '(JPG)' in video_name else ''
    save_path = os.path.join(save_dir, f"{clean_video_name}{source_suffix}_overview_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved overview grid to {save_path}")
    
    plt.show()

def get_available_videos():
    """
    Get list of videos that have both H5 files and either TIFF or JPG files.
    
    Returns:
        tuple: (available_videos, tiff_available, jpg_available)
            - available_videos: List of video names (without extensions)
            - tiff_available: Dict mapping video_name -> True if TIFF exists
            - jpg_available: Dict mapping video_name -> True if JPG folder exists
    """
    print(f"Checking directories:")
    print(f"  Segments directory: {segments_dir}")
    print(f"  TIFF directory: {tiff_dir}")
    print(f"  JPG directory: {jpg_dir}")
    
    # Check if directories exist
    if not os.path.exists(segments_dir):
        print(f"ERROR: Segments directory does not exist: {segments_dir}")
        return [], {}, {}
    
    tiff_exists = os.path.exists(tiff_dir)
    jpg_exists = os.path.exists(jpg_dir)
    
    if not tiff_exists and not jpg_exists:
        print(f"ERROR: Neither TIFF directory ({tiff_dir}) nor JPG directory ({jpg_dir}) exist")
        return [], {}, {}
    
    try:
        # Get H5 files
        h5_files = [f for f in os.listdir(segments_dir) if f.endswith('.h5')]
        h5_names = [os.path.splitext(f)[0] for f in h5_files]
        print(f"Found {len(h5_files)} H5 files: {h5_files}")
        
        # Get TIFF files if directory exists
        tiff_available = {}
        if tiff_exists:
            tiff_files = [f for f in os.listdir(tiff_dir) if f.endswith('.tif')]
            tiff_names = [os.path.splitext(f)[0] for f in tiff_files]
            tiff_available = {name: True for name in tiff_names}
            print(f"Found {len(tiff_files)} TIFF files: {tiff_files}")
        else:
            print("TIFF directory not found - TIFF visualization not available")
        
        # Get JPG folders if directory exists
        jpg_available = {}
        if jpg_exists:
            jpg_folders = [d for d in os.listdir(jpg_dir) if os.path.isdir(os.path.join(jpg_dir, d))]
            jpg_available = {name: True for name in jpg_folders}
            print(f"Found {len(jpg_folders)} JPG folders: {jpg_folders}")
        else:
            print("JPG directory not found - JPG visualization not available")
        
        # Find videos that have H5 files and at least one image source
        available_videos = []
        for h5_name in h5_names:
            has_tiff = h5_name in tiff_available
            has_jpg = h5_name in jpg_available
            if has_tiff or has_jpg:
                available_videos.append(h5_name)
        
        print(f"Found {len(available_videos)} videos with H5 files and image sources:")
        for video in sorted(available_videos):
            sources = []
            if video in tiff_available:
                sources.append("TIFF")
            if video in jpg_available:
                sources.append("JPG")
            print(f"  - {video} ({', '.join(sources)})")
        
        return sorted(available_videos), tiff_available, jpg_available
        
    except Exception as e:
        print(f"ERROR scanning directories: {e}")
        return [], {}, {}

def select_video_interactive(available_videos):
    """
    Let user select a video interactively.
    
    Args:
        available_videos (list): List of available video names
        
    Returns:
        str: Selected video name, or None if cancelled
    """
    if not available_videos:
        print("No videos available!")
        return None
    
    print("\nAvailable videos:")
    for i, video in enumerate(available_videos):
        print(f"{i+1}: {video}")
    
    while True:
        try:
            choice = input(f"\nSelect a video (1-{len(available_videos)}) or 'r' for random: ").strip()
            
            if choice.lower() == 'r':
                return random.choice(available_videos)
            elif choice.lower() in ['q', 'quit', 'exit']:
                return None
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(available_videos):
                    return available_videos[idx]
                else:
                    print(f"Please enter a number between 1 and {len(available_videos)}")
        except ValueError:
            print("Please enter a valid number or 'r' for random")

def main():
    """Main function to run the visualization tool"""
    print("=== TIFF + H5 Mask Visualization Tool ===")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        available_videos, tiff_available, jpg_available = get_available_videos()
    except Exception as e:
        print(f"ERROR getting available videos: {e}")
        return
    
    if not available_videos:
        print("No videos found with H5 files and image sources!")
        print("Make sure you have:")
        print(f"  - H5 files in: {os.path.abspath(segments_dir)}")
        print(f"  - TIF files in: {os.path.abspath(tiff_dir)} OR")
        print(f"  - JPG folders in: {os.path.abspath(jpg_dir)}")
        return
    
    while True:
        try:
            video_name = select_video_interactive(available_videos)
            
            if video_name is None:
                print("Exiting...")
                break
            
            print(f"\nLoading video: {video_name}")
            
            # Check available image sources for this video
            has_tiff = video_name in tiff_available
            has_jpg = video_name in jpg_available
            
            # Let user choose image source if both are available
            image_source = None
            image_stack = None
            
            if has_tiff and has_jpg:
                print(f"\nBoth TIFF and JPG sources available for {video_name}")
                print("1: Use TIFF stack")
                print("2: Use JPG frames")
                
                while True:
                    try:
                        source_choice = input("Select image source (1-2): ").strip()
                        if source_choice == '1':
                            image_source = 'tiff'
                            break
                        elif source_choice == '2':
                            image_source = 'jpg'
                            break
                        else:
                            print("Please enter 1 or 2")
                    except KeyboardInterrupt:
                        print("\nExiting...")
                        return
            elif has_tiff:
                image_source = 'tiff'
                print("Using TIFF stack")
            elif has_jpg:
                image_source = 'jpg'
                print("Using JPG frames")
            
            # Load the selected image source
            if image_source == 'tiff':
                tiff_path = os.path.join(tiff_dir, f"{video_name}.tif")
                print(f"Loading TIFF from: {os.path.abspath(tiff_path)}")
                image_stack = load_tiff_stack(tiff_path)
            elif image_source == 'jpg':
                jpg_path = os.path.join(jpg_dir, video_name)
                print(f"Loading JPG frames from: {os.path.abspath(jpg_path)}")
                image_stack = load_jpg_frames(jpg_path)
            
            if image_stack is None:
                print(f"Failed to load {image_source.upper()} stack for {video_name}")
                continue
            
            # Load masks
            h5_path = os.path.join(segments_dir, f"{video_name}.h5")
            print(f"Loading H5 from: {os.path.abspath(h5_path)}")
            
            # Run detailed H5 debug first
            debug_h5_structure(h5_path)
            
            masks_dict, metadata = load_masks_from_h5(h5_path)
            
            # Run alignment debug
            debug_mask_alignment(h5_path, image_source, video_name)
            
            print(f"\nVisualization options:")
            print("1: Interactive viewer (navigate frame by frame)")
            print("2: Overview grid (multiple frames at once)")
            print("3: Both")
            
            while True:
                try:
                    viz_choice = input("Select visualization type (1-3): ").strip()
                    if viz_choice in ['1', '2', '3']:
                        break
                    else:
                        print("Please enter 1, 2, or 3")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    return
            
            # Create visualizations
            display_name = f"{video_name} ({image_source.upper()})"
            
            if viz_choice in ['1', '3']:
                print("Opening interactive viewer...")
                create_interactive_viewer(image_stack, masks_dict, display_name, output_dir)
            
            if viz_choice in ['2', '3']:
                print("Creating overview grid...")
                create_summary_grid(image_stack, masks_dict, display_name, output_dir)
            
            print(f"\nVisualization complete for {video_name} using {image_source.upper()}")
            
            # Ask if user wants to continue with another video
            try:
                continue_choice = input("\nVisualize another video? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    break
            except KeyboardInterrupt:
                print("\nExiting...")
                break
        
        except Exception as e:
            print(f"ERROR in main loop: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("Visualization tool finished!")

if __name__ == "__main__":
    main()