import h5py
import numpy as np
import os
from tqdm import tqdm
import random
import pandas as pd
from scipy.ndimage import distance_transform_edt

# Directory paths for input segments, output analysis data, and video frames
segments_dir = '5RIA_SEGMENT'
video_dir = '3CROP'
final_data_dir = '6ANALYSIS'

def load_cleaned_segments_from_h5(filename):
    """
    Load cleaned segment masks from an HDF5 file.
    
    Args:
        filename (str): Path to the HDF5 file containing segment masks
        
    Returns:
        dict: Dictionary with frame indices as keys and dictionaries of object masks as values
              Format: {frame_idx: {object_id: boolean_mask, ...}, ...}
    """
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        masks_group = f['masks']
        nb_frames = 0
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in object_ids:
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                frame_data[obj_id] = mask
            
            cleaned_segments[frame_idx] = frame_data
            nb_frames += 1
    
    print(f"{nb_frames} frames loaded from {filename}")
    return cleaned_segments

def get_random_unprocessed_video(cleaned_aligned_segments_dir, final_data_dir):
    """
    Select a random video that hasn't been processed yet.
    
    Args:
        cleaned_aligned_segments_dir (str): Directory containing HDF5 segment files
        final_data_dir (str): Directory containing already processed CSV files
        
    Returns:
        str: Full path to an unprocessed HDF5 file
        
    Raises:
        ValueError: If all videos have been processed
    """
    # Get all .h5 files in the segments directory
    h5_files = [f for f in os.listdir(cleaned_aligned_segments_dir) if f.endswith('.h5')]
    all_videos = [os.path.splitext(f)[0] for f in h5_files]
    
    # Get all .csv files in the final data directory
    csv_files = [f for f in os.listdir(final_data_dir) if f.endswith('.csv')]
    processed_videos = [os.path.splitext(f)[0] for f in csv_files]
    
    print(f"Found {len(h5_files)} .h5 files: {h5_files}")
    print(f"Found {len(csv_files)} .csv files: {csv_files}")
    print(f"All video basenames: {all_videos}")
    print(f"Processed video basenames: {processed_videos}")
    
    unprocessed_videos = [
        video for video in all_videos
        if video not in processed_videos
    ]
    
    print(f"Unprocessed videos: {unprocessed_videos}")
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    selected_video = random.choice(unprocessed_videos) + ".h5"
    print(f"Selected video: {selected_video}")
    
    return os.path.join(cleaned_aligned_segments_dir, selected_video)

def get_all_unprocessed_videos(cleaned_aligned_segments_dir, final_data_dir):
    """
    Get all videos that haven't been processed yet.
    
    Args:
        cleaned_aligned_segments_dir (str): Directory containing HDF5 segment files
        final_data_dir (str): Directory containing already processed CSV files
        
    Returns:
        list: List of full paths to unprocessed HDF5 files
    """
    # Get all .h5 files in the segments directory
    h5_files = [f for f in os.listdir(cleaned_aligned_segments_dir) if f.endswith('.h5')]
    all_videos = [os.path.splitext(f)[0] for f in h5_files]
    
    # Get all .csv files in the final data directory
    csv_files = [f for f in os.listdir(final_data_dir) if f.endswith('.csv')]
    processed_videos = [os.path.splitext(f)[0] for f in csv_files]
    
    print(f"Found {len(h5_files)} .h5 files: {h5_files}")
    print(f"Found {len(csv_files)} .csv files: {csv_files}")
    print(f"All video basenames: {all_videos}")
    print(f"Processed video basenames: {processed_videos}")
    
    unprocessed_videos = [
        video for video in all_videos
        if video not in processed_videos
    ]
    
    print(f"Unprocessed videos: {unprocessed_videos}")
    
    unprocessed_paths = [os.path.join(cleaned_aligned_segments_dir, video + ".h5") for video in unprocessed_videos]
    
    return unprocessed_paths

def extract_top_percent_brightness(aligned_images, masks_dict, object_id, percent):
    """
    Extract mean brightness of the top X% brightest pixels from masked regions.
    
    Args:
        aligned_images (list): List of image arrays
        masks_dict (dict): Dictionary of masks by frame and object ID
        object_id (int): ID of the object to analyze
        percent (float): Percentage of brightest pixels to use (0-100)
        
    Returns:
        pd.DataFrame: DataFrame with brightness statistics per frame
        
    Raises:
        ValueError: If percentage is not between 0 and 100
    """
    if not 0 < percent <= 100:
        raise ValueError("Percentage must be between 0 and 100")
        
    data = []
    for frame_idx, image in enumerate(aligned_images):
        if frame_idx in masks_dict and object_id in masks_dict[frame_idx]:
            mask = masks_dict[frame_idx][object_id][0]
            masked_pixels = image[mask]
            n_pixels = int(round(len(masked_pixels) * (percent / 100)))
            n_pixels = max(1, n_pixels)
            top_n_pixels = np.sort(masked_pixels)[-n_pixels:]
            mean_top_percent = np.mean(top_n_pixels)
            
            data.append({
                'frame': frame_idx,
                'mean_top_percent_brightness': mean_top_percent,
                'n_pixels_used': n_pixels,
                'total_pixels': len(masked_pixels),
                'percent_used': percent
            })
    
    return pd.DataFrame(data)

def get_background_sample(frame_masks, image_shape, num_samples=100, min_distance=40):
    """
    Sample background pixels that are far enough from any segmented objects.
    
    Args:
        frame_masks (dict): Dictionary of masks for current frame {object_id: mask}
        image_shape (tuple): Shape of the image (height, width)
        num_samples (int): Number of background pixels to sample
        min_distance (int): Minimum distance from objects in pixels
        
    Returns:
        np.ndarray: Array of (y, x) coordinates of background pixels, shape (n_samples, 2)
    """
    combined_mask = np.zeros(image_shape[1:], dtype=bool)
    for mask in frame_masks.values():
        combined_mask |= mask.squeeze()
    
    distance_map = distance_transform_edt(~combined_mask)
    valid_bg = (distance_map >= min_distance)
    valid_coords = np.column_stack(np.where(valid_bg))
    
    if len(valid_coords) < num_samples:
        print(f"Warning: Only {len(valid_coords)} valid background pixels found. Sampling all of them.")
        return valid_coords
    else:
        sampled_indices = random.sample(range(len(valid_coords)), num_samples)
        return valid_coords[sampled_indices]

def load_image(frame_idx, tiff_stack=None):
    """
    Load a single frame from a TIFF stack corresponding to the currently selected video.
    
    Args:
        frame_idx (int): Index of the frame to load
        tiff_stack (np.ndarray, optional): Pre-loaded TIFF stack array
        
    Returns:
        np.ndarray: Grayscale image array, or None if loading fails
        
    Note:
        Uses global variable 'filename' to determine video basename and loads from TIFF stack
    """
    if tiff_stack is not None:
        if frame_idx < len(tiff_stack):
            return tiff_stack[frame_idx]
        else:
            print(f"Warning: Frame {frame_idx} is out of bounds for TIFF stack")
            return None
    
    # Extract base filename from h5 file and find corresponding tiff stack
    h5_basename = os.path.splitext(os.path.basename(filename))[0]
    
    tiff_path = os.path.join(video_dir, h5_basename + ".tif")
    
    try:
        # Load the entire TIFF stack
        import tifffile
        tiff_stack = tifffile.imread(tiff_path)
        if frame_idx < len(tiff_stack):
            return tiff_stack[frame_idx]
        else:
            print(f"Warning: Frame {frame_idx} is out of bounds for TIFF stack")
            return None
    except Exception as e:
        print(f"Error loading TIFF stack {tiff_path}: {e}")
        return None

def count_mask_pixels(masks):
    """
    Count the number of pixels in each mask.
    
    Args:
        masks (dict): Dictionary of masks {object_id: boolean_mask}
        
    Returns:
        dict: Dictionary with pixel counts {object_id: pixel_count}
    """
    pixel_counts = {}
    for obj_id, mask in masks.items():
        pixel_counts[obj_id] = np.sum(mask)
    return pixel_counts

def calculate_mean_values_and_pixel_counts(image, masks, background_coordinates):
    """
    Calculate mean pixel values for each mask and background, plus pixel counts.
    
    Args:
        image (np.ndarray): Grayscale image array
        masks (dict): Dictionary of boolean masks {object_id: mask}
        background_coordinates (np.ndarray): Array of background pixel coordinates
        
    Returns:
        tuple: (mean_values_dict, pixel_counts_dict, top50_values_dict, top25_values_dict, top10_values_dict)
            - mean_values: {object_id: mean_value, 'background': bg_mean}
            - pixel_counts: {object_id: pixel_count}
            - top50_values: {object_id: mean_of_top_50_pixels}
            - top25_values: {object_id: mean_of_top_25_pixels}
            - top10_values: {object_id: mean_of_top_10_pixels}
    """
    mean_values = {}
    pixel_counts = count_mask_pixels(masks)
    top50_values = {}
    top25_values = {}
    top10_values = {}
    
    bg_pixel_values = image[background_coordinates[:, 0], background_coordinates[:, 1]]
    mean_values['background'] = np.mean(bg_pixel_values)
    
    for obj_id, mask in masks.items():
        mask_pixel_values = image[mask.squeeze()]
        mean_values[obj_id] = np.mean(mask_pixel_values)
        
        # Calculate mean of top 50 brightest pixels
        if len(mask_pixel_values) >= 50:
            top_50_pixels = np.sort(mask_pixel_values)[-50:]
            top50_values[obj_id] = np.mean(top_50_pixels)
        else:
            # If fewer than 50 pixels, use all pixels
            top50_values[obj_id] = np.mean(mask_pixel_values)
        
        # Calculate mean of top 25 brightest pixels
        if len(mask_pixel_values) >= 25:
            top_25_pixels = np.sort(mask_pixel_values)[-25:]
            top25_values[obj_id] = np.mean(top_25_pixels)
        else:
            # If fewer than 25 pixels, use all pixels
            top25_values[obj_id] = np.mean(mask_pixel_values)
        
        # Calculate mean of top 10 brightest pixels
        if len(mask_pixel_values) >= 10:
            top_10_pixels = np.sort(mask_pixel_values)[-10:]
            top10_values[obj_id] = np.mean(top_10_pixels)
        else:
            # If fewer than 10 pixels, use all pixels
            top10_values[obj_id] = np.mean(mask_pixel_values)
    
    return mean_values, pixel_counts, top50_values, top25_values, top10_values

def create_wide_format_table_with_bg_correction_and_pixel_count(mean_values, pixel_counts, top50_values, top25_values, top10_values):
    """
    Create a wide-format DataFrame with brightness values, background correction, and pixel counts.
    
    Args:
        mean_values (dict): Dictionary {frame_idx: {object_id: mean_value, 'background': bg_value}}
        pixel_counts (dict): Dictionary {frame_idx: {object_id: pixel_count}}
        top50_values (dict): Dictionary {frame_idx: {object_id: top50_mean_value}}
        top25_values (dict): Dictionary {frame_idx: {object_id: top25_mean_value}}
        top10_values (dict): Dictionary {frame_idx: {object_id: top10_mean_value}}
        
    Returns:
        pd.DataFrame: Wide-format table with columns:
            - frame: frame index
            - object_id: raw mean brightness
            - object_id_bg_corrected: background-corrected brightness
            - object_id_top50: mean of top 50 brightest pixels
            - object_id_top50_bg_corrected: background-corrected top50 brightness
            - object_id_top25: mean of top 25 brightest pixels
            - object_id_top25_bg_corrected: background-corrected top25 brightness
            - object_id_top10: mean of top 10 brightest pixels
            - object_id_top10_bg_corrected: background-corrected top10 brightness
            - object_id_pixel_count: number of pixels in mask
    """
    data = {'frame': []}
    
    all_objects = set()
    for frame_data in mean_values.values():
        all_objects.update(frame_data.keys())
    all_objects.remove('background')
    
    for obj in all_objects:
        data[obj] = []
        data[f"{obj}_bg_corrected"] = []
        data[f"{obj}_top50"] = []
        data[f"{obj}_top50_bg_corrected"] = []
        data[f"{obj}_top25"] = []
        data[f"{obj}_top25_bg_corrected"] = []
        data[f"{obj}_top10"] = []
        data[f"{obj}_top10_bg_corrected"] = []
        data[f"{obj}_pixel_count"] = []
    
    for frame_idx, frame_data in mean_values.items():
        data['frame'].append(frame_idx)
        bg_value = frame_data['background']
        frame_pixel_counts = pixel_counts[frame_idx]
        frame_top50_values = top50_values[frame_idx]
        frame_top25_values = top25_values[frame_idx]
        frame_top10_values = top10_values[frame_idx]
        
        for obj in all_objects:
            obj_value = frame_data.get(obj, np.nan)
            data[obj].append(obj_value)
            
            if pd.notnull(obj_value):
                bg_corrected = obj_value - bg_value
            else:
                bg_corrected = np.nan
            data[f"{obj}_bg_corrected"].append(bg_corrected)
            
            # Top 50
            top50_value = frame_top50_values.get(obj, np.nan)
            data[f"{obj}_top50"].append(top50_value)
            
            if pd.notnull(top50_value):
                top50_bg_corrected = top50_value - bg_value
            else:
                top50_bg_corrected = np.nan
            data[f"{obj}_top50_bg_corrected"].append(top50_bg_corrected)
            
            # Top 25
            top25_value = frame_top25_values.get(obj, np.nan)
            data[f"{obj}_top25"].append(top25_value)
            
            if pd.notnull(top25_value):
                top25_bg_corrected = top25_value - bg_value
            else:
                top25_bg_corrected = np.nan
            data[f"{obj}_top25_bg_corrected"].append(top25_bg_corrected)
            
            # Top 10
            top10_value = frame_top10_values.get(obj, np.nan)
            data[f"{obj}_top10"].append(top10_value)
            
            if pd.notnull(top10_value):
                top10_bg_corrected = top10_value - bg_value
            else:
                top10_bg_corrected = np.nan
            data[f"{obj}_top10_bg_corrected"].append(top10_bg_corrected)
            
            data[f"{obj}_pixel_count"].append(frame_pixel_counts.get(obj, 0))
    
    df = pd.DataFrame(data)
    
    return df

def process_cleaned_segments(cleaned_segments):
    """
    Process all frames in cleaned segments to extract brightness and pixel count data.
    
    Args:
        cleaned_segments (dict): Dictionary of cleaned segment masks by frame
        
    Returns:
        pd.DataFrame: Wide-format table with brightness measurements and statistics
    """
    first_frame = next(iter(cleaned_segments.values()))
    first_mask = next(iter(first_frame.values()))
    image_shape = first_mask.shape

    # Pre-load the entire TIFF stack for efficiency
    h5_basename = os.path.splitext(os.path.basename(filename))[0]
    
    tiff_path = os.path.join(video_dir, h5_basename + ".tif")
    
    try:
        import tifffile
        tiff_stack = tifffile.imread(tiff_path)
        print(f"Loaded TIFF stack with {len(tiff_stack)} frames from {tiff_path}")
    except Exception as e:
        print(f"Error loading TIFF stack {tiff_path}: {e}")
        tiff_stack = None

    mean_values = {}
    pixel_counts = {}
    top50_values = {}
    top25_values = {}
    top10_values = {}

    for frame_idx, frame_masks in tqdm(cleaned_segments.items(), desc="Processing frames"):
        bg_coordinates = get_background_sample(frame_masks, image_shape)
        image = load_image(frame_idx, tiff_stack)
        
        if image is None:
            print(f"Warning: Could not load image for frame {frame_idx}")
            continue
        
        frame_mean_values, frame_pixel_counts, frame_top50_values, frame_top25_values, frame_top10_values = calculate_mean_values_and_pixel_counts(image, frame_masks, bg_coordinates)
        mean_values[frame_idx] = frame_mean_values
        pixel_counts[frame_idx] = frame_pixel_counts
        top50_values[frame_idx] = frame_top50_values
        top25_values[frame_idx] = frame_top25_values
        top10_values[frame_idx] = frame_top10_values

    df_wide_bg_corrected = create_wide_format_table_with_bg_correction_and_pixel_count(mean_values, pixel_counts, top50_values, top25_values, top10_values)
    df_wide_bg_corrected.columns = df_wide_bg_corrected.columns.astype(str)

    if 'background' not in df_wide_bg_corrected.columns:
        background_values = [frame_data['background'] for frame_data in mean_values.values()]
        df_wide_bg_corrected['background'] = background_values

    background_column = ['background']
    original_columns = [col for col in df_wide_bg_corrected.columns if not col.endswith('_bg_corrected') and not col.endswith('_pixel_count') and not col.endswith('_top50') and not col.endswith('_top25') and not col.endswith('_top10') and col != 'frame']
    original_columns = [col for col in original_columns if col != 'background']
    bg_corrected_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_bg_corrected') and not col.endswith('_top50_bg_corrected') and not col.endswith('_top25_bg_corrected') and not col.endswith('_top10_bg_corrected')]
    top50_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_top50') and not col.endswith('_top50_bg_corrected')]
    top50_bg_corrected_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_top50_bg_corrected')]
    top25_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_top25') and not col.endswith('_top25_bg_corrected')]
    top25_bg_corrected_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_top25_bg_corrected')]
    top10_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_top10') and not col.endswith('_top10_bg_corrected')]
    top10_bg_corrected_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_top10_bg_corrected')]
    pixel_count_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_pixel_count')]

    all_columns = ['frame'] + background_column + original_columns + bg_corrected_columns + top50_columns + top50_bg_corrected_columns + top25_columns + top25_bg_corrected_columns + top10_columns + top10_bg_corrected_columns + pixel_count_columns

    print(df_wide_bg_corrected[all_columns].describe())

    return df_wide_bg_corrected

def get_centroid(mask):
    """
    Calculate the centroid (center of mass) of a binary mask.
    
    Args:
        mask (np.ndarray): Boolean mask array, shape (1, height, width)
        
    Returns:
        tuple or None: (centroid_x, centroid_y) coordinates, or None if mask is empty
    """
    y_indices, x_indices = np.where(mask[0])
    if len(x_indices) == 0:
        return None
    
    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)
    return (centroid_x, centroid_y)

def get_relative_position(first_frame):
    """
    Determine the relative position of object 4 vs object 2 to classify worm orientation.
    
    Args:
        first_frame (dict): Dictionary of masks for the first frame {object_id: mask}
        
    Returns:
        str: "left" if object 4 is left of object 2, "right" otherwise,
             or error message if objects not found
    """
    available_objects = list(first_frame.keys())
    
    if len(available_objects) < 2:
        return f"Not enough objects in first frame: {available_objects}"
    
    # Use the first two available objects
    obj1_id = available_objects[0]
    obj2_id = available_objects[1]
    
    centroid1 = get_centroid(first_frame[obj1_id])
    centroid2 = get_centroid(first_frame[obj2_id])
    
    if centroid1 is None or centroid2 is None:
        return f"Could not calculate centroids for objects {obj1_id}, {obj2_id}"
    
    if centroid1[0] < centroid2[0]:
        return "left"
    else:
        return "right"
    
def save_brightness_and_side_data(df_wide_brightness_and_background, cleaned_segments, filename, final_data_dir):
    """
    Add side position information and save the complete analysis to CSV.
    
    Args:
        df_wide_brightness_and_background (pd.DataFrame): DataFrame with brightness data
        cleaned_segments (dict): Dictionary of cleaned segment masks
        filename (str): Path to the source HDF5 file
        final_data_dir (str): Directory to save the output CSV file
        
    Returns:
        pd.DataFrame: DataFrame with added side_position column
        
    Side effects:
        - Prints data statistics
        - Saves CSV file to final_data_dir
    """
    position = get_relative_position(next(iter(cleaned_segments.values())))
    df_wide_brightness_and_background['side_position'] = position

    base_filename = os.path.splitext(os.path.basename(filename))[0]
    output_filename = os.path.join(final_data_dir, base_filename + ".csv")

    df_wide_brightness_and_background.to_csv(output_filename, index=False)

    print(df_wide_brightness_and_background.describe())
    print("side_position unique values:", df_wide_brightness_and_background['side_position'].unique())
    print(df_wide_brightness_and_background['side_position'].value_counts())
    print(f"Data saved to: {output_filename}")

    return df_wide_brightness_and_background

def verify_mask_alignment(cleaned_segments, tiff_stack, jpeg_dir=None):
    """
    Verify that masks align properly with TIFF images.
    
    Args:
        cleaned_segments (dict): Dictionary of masks from H5 file
        tiff_stack (np.ndarray): TIFF image stack
        jpeg_dir (str, optional): Directory with original JPEG frames for comparison
    
    Returns:
        dict: Alignment verification results
    """
    verification_results = {
        'dimension_match': True,
        'sample_correlations': [],
        'issues': []
    }
    
    # Check dimensions
    first_frame_masks = next(iter(cleaned_segments.values()))
    first_mask = next(iter(first_frame_masks.values()))
    mask_shape = first_mask.shape
    
    if len(tiff_stack.shape) == 3:
        tiff_shape = tiff_stack[0].shape
    else:
        tiff_shape = tiff_stack.shape
    
    print(f"Mask shape: {mask_shape}")
    print(f"TIFF shape: {tiff_shape}")
    
    if mask_shape[1:] != tiff_shape:
        verification_results['dimension_match'] = False
        verification_results['issues'].append(f"Shape mismatch: masks {mask_shape[1:]} vs TIFF {tiff_shape}")
    
    # Sample a few frames to check correlation
    sample_frames = random.sample(list(cleaned_segments.keys()), min(5, len(cleaned_segments)))
    
    for frame_idx in sample_frames:
        if frame_idx < len(tiff_stack):
            tiff_frame = tiff_stack[frame_idx]
            frame_masks = cleaned_segments[frame_idx]
            
            for obj_id, mask in frame_masks.items():
                mask_2d = mask.squeeze()
                
                # Calculate correlation between mask edges and image edges
                from scipy.ndimage import sobel
                tiff_edges = sobel(tiff_frame.astype(float))
                mask_edges = sobel(mask_2d.astype(float))
                
                correlation = np.corrcoef(tiff_edges.flatten(), mask_edges.flatten())[0,1]
                verification_results['sample_correlations'].append({
                    'frame': frame_idx,
                    'object': obj_id,
                    'correlation': correlation
                })
    
    return verification_results

# Add this before process_cleaned_segments
def load_and_verify_alignment(filename, cleaned_segments):
    """Load TIFF stack and verify mask alignment before processing"""
    h5_basename = os.path.splitext(os.path.basename(filename))[0]
    tiff_path = os.path.join(video_dir, h5_basename + ".tif")
    
    try:
        import tifffile
        tiff_stack = tifffile.imread(tiff_path)
        print(f"Loaded TIFF stack: {tiff_stack.shape}")
        
        # Verify alignment
        verification = verify_mask_alignment(cleaned_segments, tiff_stack)
        
        if not verification['dimension_match']:
            print("WARNING: Dimension mismatch detected!")
            for issue in verification['issues']:
                print(f"  - {issue}")
        
        avg_correlation = np.mean([r['correlation'] for r in verification['sample_correlations'] if not np.isnan(r['correlation'])])
        print(f"Average mask-image correlation: {avg_correlation:.3f}")
        
        if avg_correlation < 0.1:
            print("WARNING: Low correlation suggests poor mask alignment!")
        
        return tiff_stack, verification
        
    except Exception as e:
        print(f"Error loading/verifying TIFF: {e}")
        return None, None

# Main execution - process all unprocessed videos
unprocessed_files = get_all_unprocessed_videos(segments_dir, final_data_dir)

if not unprocessed_files:
    print("All videos have been processed!")
else:
    print(f"Processing {len(unprocessed_files)} unprocessed videos...")
    
    for i, filename in enumerate(unprocessed_files):
        print(f"\n{'='*60}")
        print(f"Processing video {i+1}/{len(unprocessed_files)}: {os.path.basename(filename)}")
        print(f"{'='*60}")
        
        try:
            # Load segments for this video
            cleaned_segments = load_cleaned_segments_from_h5(filename)
            
            # Verify alignment
            tiff_stack, verification = load_and_verify_alignment(filename, cleaned_segments)
            if verification and not verification['dimension_match']:
                print(f"SKIPPING {os.path.basename(filename)}: Mask alignment issues detected.")
                continue
            
            # Process the video
            df_wide_brightness_and_background = process_cleaned_segments(cleaned_segments)
            
            # Save results
            df_wide_brightness_and_side = save_brightness_and_side_data(
                df_wide_brightness_and_background, 
                cleaned_segments, 
                filename, 
                final_data_dir
            )
            
            print(f"Successfully processed {os.path.basename(filename)}")
            
        except Exception as e:
            print(f"ERROR processing {os.path.basename(filename)}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")