"""
Simple version of the visualization script that just loads and inspects data
without creating visual plots - useful for testing and debugging
"""
import h5py
import numpy as np
import os

# Directory paths
segments_dir = '5RIA_SEGMENT'
video_dir = '3CROP'

def load_and_inspect_h5(h5_path):
    """Load and inspect an H5 file"""
    print(f"\n--- Inspecting H5 file: {os.path.basename(h5_path)} ---")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"Attributes:")
            for attr_name in f.attrs.keys():
                print(f"  {attr_name}: {f.attrs[attr_name]}")
            
            print(f"\nGroups and datasets:")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"  Group: {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            
            f.visititems(print_structure)
            
            # Load actual mask data
            if 'masks' in f:
                masks_group = f['masks']
                print(f"\nMask data:")
                for obj_id in f.attrs['object_ids']:
                    if obj_id in masks_group:
                        mask_data = masks_group[obj_id]
                        print(f"  Object {obj_id}: shape {mask_data.shape}, dtype {mask_data.dtype}")
                        
                        # Check first frame
                        first_frame = mask_data[0]
                        pixel_count = np.sum(first_frame)
                        print(f"    First frame pixels: {pixel_count}")
        
        return True
        
    except Exception as e:
        print(f"ERROR loading H5 file: {e}")
        return False

def load_and_inspect_tiff(tiff_path):
    """Load and inspect a TIFF file"""
    print(f"\n--- Inspecting TIFF file: {os.path.basename(tiff_path)} ---")
    
    try:
        import tifffile
        stack = tifffile.imread(tiff_path)
        print(f"TIFF shape: {stack.shape}")
        print(f"TIFF dtype: {stack.dtype}")
        print(f"TIFF min/max values: {np.min(stack)} / {np.max(stack)}")
        
        if len(stack.shape) == 3:
            print(f"Number of frames: {stack.shape[0]}")
            print(f"Frame dimensions: {stack.shape[1]} x {stack.shape[2]}")
        
        return stack
        
    except Exception as e:
        print(f"ERROR loading TIFF file: {e}")
        return None

def find_matching_videos():
    """Find videos that have both H5 and TIFF files"""
    print("=== Finding Matching Videos ===")
    
    # Get H5 files
    h5_files = [f for f in os.listdir(segments_dir) if f.endswith('.h5')]
    h5_names = [os.path.splitext(f)[0] for f in h5_files]
    print(f"H5 files ({len(h5_files)}): {h5_names}")
    
    # Get TIFF files  
    tiff_files = [f for f in os.listdir(video_dir) if f.endswith('.tif')]
    tiff_names = [os.path.splitext(f)[0] for f in tiff_files]
    print(f"TIFF files ({len(tiff_files)}): {tiff_names}")
    
    # Find matches
    matching = list(set(h5_names) & set(tiff_names))
    print(f"Matching videos ({len(matching)}): {matching}")
    
    return matching

def compare_data_alignment(video_name):
    """Compare H5 mask data with TIFF data for alignment"""
    print(f"\n=== Comparing Data Alignment for {video_name} ===")
    
    h5_path = os.path.join(segments_dir, f"{video_name}.h5")
    tiff_path = os.path.join(video_dir, f"{video_name}.tif")
    
    # Load H5 data
    masks_by_frame = {}
    try:
        with h5py.File(h5_path, 'r') as f:
            num_frames = f.attrs['num_frames']
            object_ids = f.attrs['object_ids']
            masks_group = f['masks']
            
            for obj_id_str in object_ids:
                masks_data = masks_group[obj_id_str][:]
                for frame_idx in range(len(masks_data)):
                    if frame_idx not in masks_by_frame:
                        masks_by_frame[frame_idx] = {}
                    masks_by_frame[frame_idx][obj_id_str] = masks_data[frame_idx]
        
        print(f"H5 data: {num_frames} frames, objects: {object_ids}")
        
    except Exception as e:
        print(f"ERROR loading H5: {e}")
        return
    
    # Load TIFF data
    try:
        import tifffile
        tiff_stack = tifffile.imread(tiff_path)
        print(f"TIFF data: {tiff_stack.shape}")
        
        # Compare dimensions
        if len(masks_by_frame) > 0:
            first_mask = next(iter(next(iter(masks_by_frame.values())).values()))
            mask_shape = first_mask.shape
            print(f"Mask shape: {mask_shape}")
            
            if len(tiff_stack.shape) == 3:
                tiff_frame_shape = tiff_stack[0].shape
                print(f"TIFF frame shape: {tiff_frame_shape}")
                
                if mask_shape[1:] == tiff_frame_shape:
                    print("✓ Shapes match!")
                else:
                    print("✗ Shape mismatch!")
            
            # Check frame counts
            print(f"H5 frames: {len(masks_by_frame)}")
            print(f"TIFF frames: {len(tiff_stack) if len(tiff_stack.shape) == 3 else 1}")
        
    except Exception as e:
        print(f"ERROR loading TIFF: {e}")

def main():
    """Main function for simple inspection"""
    print("=== Simple Data Inspection Tool ===")
    print("This tool loads and inspects data without creating visualizations")
    
    try:
        # Find matching videos
        matching_videos = find_matching_videos()
        
        if not matching_videos:
            print("No matching videos found!")
            return
        
        # Inspect first matching video in detail
        video_name = matching_videos[0]
        print(f"\nDetailed inspection of: {video_name}")
        
        h5_path = os.path.join(segments_dir, f"{video_name}.h5")
        tiff_path = os.path.join(video_dir, f"{video_name}.tif")
        
        # Inspect files
        load_and_inspect_h5(h5_path)
        load_and_inspect_tiff(tiff_path)
        
        # Compare alignment
        compare_data_alignment(video_name)
        
        print(f"\nInspection complete! If this works, try the full visualization script.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()