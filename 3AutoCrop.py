"""
This script generates a cropped video around the RIA region.
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.cuda.empty_cache()
from tqdm import tqdm
import cv2
import random
from matplotlib.backend_bases import MouseButton
import tifffile

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

from sam2.build_sam import build_sam2_video_predictor # type: ignore
sam2_checkpoint = r"c:\Users\switte3\Documents\sam2\checkpoints\sam2.1_hiera_tiny.pt"
model_cfg = r"c:\Users\switte3\Documents\sam2\sam2\configs\sam2.1\sam2.1_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

parent_video_dir = '2JPG'
crop_dir = '3CROP'
tiff_dir = '1TIFF'

# Add quality thresholds and visualization constants
CONFIDENCE_THRESHOLD = 0.9  # minimum confidence score (0-1)
QUALITY_CHECK_INTERVAL = 100  # check quality every 100 frames
VIS_INTERVAL = 100       # visualize masks every 100 frames
CHUNK_SIZE = 300   # number of frames per chunk
OVERLAP = 1        # number of overlapping frames between consecutive chunks

def interactive_collect_bboxes(image_path, object_ids=None):
    """
    Collect bounding box prompts on a single image interactively.

    Controls:
    - Number keys 1..2: select object id (1:RIA)
    - Click and drag: draw bounding box for current object
    - u: undo last box for current object
    - c: clear current object's boxes
    - enter/q: finish and return
    - Use matplotlib toolbar for zoom/pan

    Returns:
        dict[int, np.ndarray] mapping obj_id -> bbox[4] (x1, y1, x2, y2)
    """
    if object_ids is None:
        object_ids = (1,)  # Single RIA object
    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Enable the navigation toolbar for zoom/pan functionality
    fig.canvas.toolbar_visible = True
    
    ax.imshow(img)
    ax.set_axis_off()

    # per-object storage
    storage = {oid: {"boxes": []} for oid in object_ids}
    current_obj = object_ids[0]
    
    # Drawing state
    drawing = False
    start_point = None
    current_rect = None

    # Define specific colors for each object: 1=red
    object_colors = {1: (1.0, 0.0, 0.0)}  # red

    def redraw():
        ax.clear()
        ax.imshow(img)
        ax.set_axis_off()
        
        # Draw existing boxes
        for oid in sorted(storage.keys()):
            boxes = storage[oid]["boxes"]
            if not boxes:
                continue
            
            # Use consistent colors: RIA=red
            color = object_colors.get(oid, (0.5, 0.5, 0.5))
            
            for box in boxes:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rect = plt.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, facecolor='none',
                                   label=f"obj {oid}")
                ax.add_patch(rect)
        
        # Display per-object counts of boxes
        obj_lines = []
        total = 0
        for oid in sorted(storage.keys()):
            boxes = storage[oid]["boxes"]
            total += len(boxes)
            name = "RIA" if oid == 1 else str(oid)
            obj_lines.append(f"{oid}:{name}  boxes:{len(boxes)}")
        obj_list = ",  ".join(obj_lines)
        
        ax.text(0.01, 0.99,
            f"Active={current_obj}:{'RIA' if current_obj == 1 else str(current_obj)} | Total boxes: {total}\n"
            f"Keys: 1 select obj, click+drag=draw box, u=undo, c=clear, enter/q=finish\n"
            f"Use toolbar above for zoom/pan. Objects: {obj_list}",
            transform=ax.transAxes, va='top', ha='left', color='white',
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        fig.canvas.draw_idle()

    def on_press(event):
        nonlocal drawing, start_point, current_rect
        if event.inaxes != ax:
            return
        if event.button != MouseButton.LEFT:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        drawing = True
        start_point = (event.xdata, event.ydata)

    def on_motion(event):
        nonlocal current_rect
        if not drawing or start_point is None:
            return
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        # Remove previous temporary rectangle
        if current_rect is not None:
            current_rect.remove()
            current_rect = None
        
        # Draw temporary rectangle
        x1, y1 = start_point
        x2, y2 = event.xdata, event.ydata
        width = x2 - x1
        height = y2 - y1
        
        color = object_colors.get(current_obj, (0.5, 0.5, 0.5))
        current_rect = plt.Rectangle((x1, y1), width, height,
                                   linewidth=2, edgecolor=color, facecolor='none',
                                   linestyle='--', alpha=0.7)
        ax.add_patch(current_rect)
        fig.canvas.draw_idle()

    def on_release(event):
        nonlocal drawing, start_point, current_rect
        if not drawing or start_point is None:
            return
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        drawing = False
        
        # Remove temporary rectangle
        if current_rect is not None:
            current_rect.remove()
            current_rect = None
        
        # Store the box
        x1, y1 = start_point
        x2, y2 = event.xdata, event.ydata
        
        # Ensure proper ordering (x1 < x2, y1 < y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Only store if box has reasonable size
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            storage[current_obj]["boxes"].append([x1, y1, x2, y2])
            redraw()
        
        start_point = None

    def on_key(event):
        nonlocal current_obj
        key = event.key
        if not key:
            return
        if len(key) == 1 and key.isdigit():
            oid = int(key)
            if oid in storage:
                current_obj = oid
                redraw()
            return
        if key == 'u':
            boxes = storage[current_obj]["boxes"]
            if boxes:
                boxes.pop()
            redraw()
            return
        if key == 'c':
            storage[current_obj] = {"boxes": []}
            redraw()
            return
        if key in ('enter', 'return', 'q'):
            plt.close(fig)
            return

    cid1 = fig.canvas.mpl_connect('button_press_event', on_press)
    cid2 = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    cid3 = fig.canvas.mpl_connect('button_release_event', on_release)
    cid4 = fig.canvas.mpl_connect('key_press_event', on_key)
    redraw()
    plt.show()
    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)
    fig.canvas.mpl_disconnect(cid3)
    fig.canvas.mpl_disconnect(cid4)

    # convert to numpy arrays and return the most recent box for each object
    out = {}
    for oid, d in storage.items():
        if len(d["boxes"]) == 0:
            continue
        # Use the last (most recent) box for each object
        bbox = np.array(d["boxes"][-1], dtype=np.float32)
        out[oid] = bbox
    return out

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def load_tiff_stack(tiff_path):
    """Load TIFF stack and return as numpy array."""
    return tifffile.imread(tiff_path)

def save_tiff_stack(frames, output_path):
    """Save numpy array as TIFF stack."""
    tifffile.imwrite(output_path, frames)

def extract_frames_from_tiff(tiff_stack, temp_dir):
    """Extract frames from TIFF stack and save as JPG for SAM2 processing."""
    os.makedirs(temp_dir, exist_ok=True)
    frame_paths = []
    
    for i, frame in enumerate(tiff_stack):
        frame_path = os.path.join(temp_dir, f"{i:06d}.jpg")
        # Convert to 8-bit if needed
        if frame.dtype != np.uint8:
            frame_normalized = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        else:
            frame_normalized = frame
        cv2.imwrite(frame_path, frame_normalized)
        frame_paths.append(frame_path)
    
    return frame_paths

def get_random_unprocessed_tiff(tiff_dir, crop_dir):
    """Get random unprocessed TIFF file."""
    all_tiffs = [f for f in os.listdir(tiff_dir) if f.lower().endswith(('.tif', '.tiff'))]
    unprocessed_tiffs = [
        tiff for tiff in all_tiffs
        if not os.path.exists(os.path.join(crop_dir, os.path.splitext(tiff)[0] + "_crop.tif"))
    ]
    
    if not unprocessed_tiffs:
        raise ValueError("All TIFF files have been processed.")
    
    return os.path.join(tiff_dir, random.choice(unprocessed_tiffs))

def get_random_unprocessed_video(parent_dir, crop_dir):
    """Modified to handle both video directories and TIFF files."""
    # First try TIFF files
    if os.path.exists(tiff_dir):
        try:
            return get_random_unprocessed_tiff(tiff_dir, crop_dir), 'tiff'
        except ValueError:
            pass
    
    # Fall back to original video directory logic
    all_videos = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(crop_dir, video + "_crop"))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(parent_dir, random.choice(unprocessed_videos)), 'video'

def calculate_fixed_crop_window(video_segments, original_size, crop_size):
    orig_height, orig_width = original_size
    centers = []
    empty_masks = 0
    total_masks = 0

    for frame_num in sorted(video_segments.keys()):
        mask = next(iter(video_segments[frame_num].values()))
        total_masks += 1
        y_coords, x_coords = np.where(mask[0])
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            # Use midpoint of the detected object
            center_x = (x_coords.min() + x_coords.max()) // 2
            center_y = (y_coords.min() + y_coords.max()) // 2
            centers.append((center_x, center_y))
        else:
            empty_masks += 1
            centers.append((orig_width // 2, orig_height // 2))

    if empty_masks > 0:
        # Use average of valid centers for empty mask frames
        valid_centers = [center for i, center in enumerate(centers) if i < total_masks - empty_masks]
        if valid_centers:
            avg_center_x = sum(center[0] for center in valid_centers) // len(valid_centers)
            avg_center_y = sum(center[1] for center in valid_centers) // len(valid_centers)
        else:
            avg_center_x, avg_center_y = orig_width // 2, orig_height // 2
        
        # Replace empty mask centers with average
        for i in range(len(centers)):
            if sum(1 for mask in list(video_segments.values())[i].values() if mask[0].sum() == 0) > 0:
                centers[i] = (avg_center_x, avg_center_y)

    crop_windows = []
    for center_x, center_y in centers:
        left = max(0, center_x - crop_size // 2)
        top = max(0, center_y - crop_size // 2)
        right = min(orig_width, left + crop_size)
        bottom = min(orig_height, top + crop_size)
        
        # Adjust if crop window is out of bounds
        if right == orig_width:
            left = right - crop_size
        if bottom == orig_height:
            top = bottom - crop_size
        
        crop_windows.append((left, top, right, bottom))

    return crop_windows, (crop_size, crop_size), empty_masks, total_masks

def process_frames_fixed_crop(input_folder, output_folder, video_segments, original_size, crop_size):
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")    
    os.makedirs(output_folder, exist_ok=True)
    
    crop_windows, (crop_height, crop_width), empty_masks, total_masks = calculate_fixed_crop_window(video_segments, original_size, crop_size) 
    print(f"Empty masks: {empty_masks}/{total_masks}")
    print(f"Crop size: {crop_height}x{crop_width}")
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        left, top, right, bottom = crop_windows[idx]
        
        cropped_frame = frame[top:bottom, left:right]
        if cropped_frame.shape[:2] != (crop_height, crop_width):
            cropped_frame = cv2.resize(cropped_frame, (crop_width, crop_height))
        
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, cropped_frame)
    
    print(f"Cropped frames saved to: {output_folder}")
    return len(frame_files), (crop_height, crop_width)

def process_tiff_stack_fixed_crop(tiff_path, output_path, video_segments, crop_size):
    """Process TIFF stack with fixed crop windows."""
    tiff_stack = load_tiff_stack(tiff_path)
    n_frames, height, width = tiff_stack.shape[:3]
    
    crop_windows, (crop_height, crop_width), empty_masks, total_masks = calculate_fixed_crop_window(
        video_segments, (height, width), crop_size
    )
    
    print(f"Empty masks: {empty_masks}/{total_masks}")
    print(f"Crop size: {crop_height}x{crop_width}")
    
    cropped_frames = []
    
    for idx in tqdm(range(n_frames), desc="Processing TIFF frames"):
        frame = tiff_stack[idx]
        left, top, right, bottom = crop_windows[idx]
        
        cropped_frame = frame[top:bottom, left:right]
        if cropped_frame.shape[:2] != (crop_height, crop_width):
            cropped_frame = cv2.resize(cropped_frame, (crop_width, crop_height))
        
        cropped_frames.append(cropped_frame)
    
    # Save as TIFF stack
    cropped_stack = np.array(cropped_frames)
    save_tiff_stack(cropped_stack, output_path)
    
    print(f"Cropped TIFF stack saved to: {output_path}")
    return len(cropped_frames), (crop_height, crop_width)

def check_mask_quality(video_segments, frame_idx, confidence_scores=None):
    """
    Check if masks meet confidence quality threshold.
    
    Args:
        video_segments (dict): Current video segments
        frame_idx (int): Frame index to check
        confidence_scores (dict): Confidence scores per object {obj_id: score}
    
    Returns:
        tuple[bool, dict]: (needs_reprompt, quality_report)
    """
    quality_report = {"frame": frame_idx, "issues": []}
    needs_reprompt = False
    
    # Check for missing masks
    if frame_idx not in video_segments or not video_segments[frame_idx]:
        quality_report["issues"].append("No masks found")
        return True, quality_report
    
    # Check confidence scores if provided
    low_confidence = []
    if confidence_scores:
        for obj_id, confidence in confidence_scores.items():
            if confidence < CONFIDENCE_THRESHOLD:
                low_confidence.append(obj_id)
                needs_reprompt = True
    
    if low_confidence:
        quality_report["issues"].append(f"Low confidence (< {CONFIDENCE_THRESHOLD}): {low_confidence}")
    
    if not quality_report["issues"]:
        quality_report["status"] = "good"
    else:
        quality_report["status"] = "needs_reprompt" if needs_reprompt else "warning"
    
    return needs_reprompt, quality_report

def overlay_predictions_on_frame(video_dir, frame_idx, video_segments, alpha=0.6):
    """
    Save a single frame image with predicted masks overlaid and display it to the user.
    """
    COLORS = {
        1: (255, 0, 0),     # Red for RIA region
        2: (0, 0, 255),     # Blue
        3: (0, 255, 0),     # Green
    }

    frame_path = os.path.join(video_dir, f"{frame_idx:06d}.jpg")
    if not os.path.exists(frame_path):
        # Try alternative naming
        frame_names = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        if frame_idx < len(frame_names):
            frame_path = os.path.join(video_dir, frame_names[frame_idx])
        else:
            print(f"Frame {frame_idx} not found")
            return

    image = cv2.imread(frame_path)
    if image is None:
        print(f"Could not read image from {frame_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    overlay = np.zeros_like(image)
    if frame_idx in video_segments:
        masks = video_segments[frame_idx]

        for mask_id, mask in masks.items():
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            if mask.dtype != bool:
                mask = mask > 0.5

            if mask.ndim > 2:
                mask = mask.squeeze()

            mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            color = COLORS.get(mask_id, (128, 128, 128))
            colored_mask = np.zeros_like(image)
            colored_mask[mask_resized == 1] = color
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)

        overlaid_image = cv2.addWeighted(image, 1, overlay, alpha, 0)
    else:
        print(f"No predictions found for frame {frame_idx}")
        overlaid_image = image

    # Display the image to the user
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(overlaid_image)
    ax.set_title(f"Frame {frame_idx:06d} - RIA Region Tracking")
    ax.set_axis_off()
    
    # Add mask legend if masks exist
    if frame_idx in video_segments:
        legend_text = []
        for mask_id in sorted(video_segments[frame_idx].keys()):
            legend_text.append(f"{mask_id}: RIA Region")
        
        if legend_text:
            ax.text(0.02, 0.98, '\n'.join(legend_text), transform=ax.transAxes, 
                   va='top', ha='left', color='white', fontsize=10,
                   bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)  # Display for 2 seconds
    plt.close(fig)

    cv2.imwrite("frame_prediction_overlay.png", cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR))
    print(f"Overlaid image saved to frame_prediction_overlay.png")

def get_all_unprocessed_tiffs(tiff_dir, crop_dir):
    """Get all unprocessed TIFF files in alphabetical order."""
    if not os.path.exists(tiff_dir):
        return []
    
    all_tiffs = [f for f in os.listdir(tiff_dir) if f.lower().endswith(('.tif', '.tiff'))]
    unprocessed_tiffs = [
        tiff for tiff in sorted(all_tiffs)  # Sort alphabetically
        if not os.path.exists(os.path.join(crop_dir, os.path.splitext(tiff)[0] + "_crop.tif"))
    ]
    
    return [os.path.join(tiff_dir, tiff) for tiff in unprocessed_tiffs]

def get_all_unprocessed_videos(parent_dir, crop_dir):
    """Get all unprocessed video directories in alphabetical order."""
    all_videos = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    unprocessed_videos = [
        video for video in sorted(all_videos)  # Sort alphabetically
        if not os.path.exists(os.path.join(crop_dir, video + "_crop"))
    ]
    
    return [os.path.join(parent_dir, video) for video in unprocessed_videos]

def get_sorted_frame_indices(names):
    return [int(os.path.splitext(n)[0]) for n in names]

def create_chunk_dir(base_dir, indices, chunk_root, chunk_id):
    """Create a temporary chunk directory with locally renumbered frames 000000.jpg, 000001.jpg, ...
    Returns (chunk_dir, local_to_global) mapping list where local_to_global[local_idx] = global_frame_idx
    """
    import shutil
    os.makedirs(chunk_root, exist_ok=True)
    chunk_dir = os.path.join(chunk_root, f"chunk_{chunk_id:04d}")
    os.makedirs(chunk_dir, exist_ok=True)
    local_to_global = []
    for local_idx, global_idx in enumerate(indices):
        cand_a = os.path.join(base_dir, f"{global_idx:06d}.jpg")
        cand_b = os.path.join(base_dir, f"{global_idx}.jpg")
        src = cand_a if os.path.exists(cand_a) else cand_b
        src_abs = os.path.abspath(src)
        dst = os.path.join(chunk_dir, f"{local_idx:06d}.jpg")
        try:
            if not os.path.exists(dst):
                os.symlink(src_abs, dst)
        except Exception:
            # Fallback to copy if symlink not permitted
            if not os.path.exists(dst):
                shutil.copy(src_abs, dst)
        local_to_global.append(global_idx)
    return chunk_dir, local_to_global

def resolve_image_path(dir_path, frame_idx):
    a = os.path.join(dir_path, f"{frame_idx:06d}.jpg")
    b = os.path.join(dir_path, f"{frame_idx}.jpg")
    if os.path.exists(a):
        return a
    if os.path.exists(b):
        return b
    # Fallback: search for any jpg starting with the idx pattern
    for fn in os.listdir(dir_path):
        name, ext = os.path.splitext(fn)
        if ext.lower() in ('.jpg', '.jpeg') and name == str(frame_idx).zfill(6):
            return os.path.join(dir_path, fn)
    return a

def build_bbox_from_existing_masks(masks_for_frame, img_shape):
    """
    Create bounding boxes from existing masks for auto-prompting.
    
    Args:
        masks_for_frame (dict): Dictionary mapping obj_id -> mask array
        img_shape (tuple): Shape of the image (height, width, channels)
    
    Returns:
        dict: Dictionary mapping obj_id -> bbox array [x1, y1, x2, y2]
    """
    bboxes = {}
    
    for obj_id, mask in masks_for_frame.items():
        if obj_id is None:
            continue
            
        # Convert mask to numpy if it's a tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
            
        # Squeeze and ensure 2D
        mask = mask.squeeze()
        if mask.ndim != 2:
            continue
            
        # Find coordinates where mask is True
        y_coords, x_coords = np.where(mask > 0.5)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            continue
            
        # Get bounding box coordinates
        x1, x2 = float(np.min(x_coords)), float(np.max(x_coords))
        y1, y2 = float(np.min(y_coords)), float(np.max(y_coords))
        
        # Scale to image dimensions if needed
        mask_h, mask_w = mask.shape
        img_h, img_w = img_shape[:2]
        
        if mask_h != img_h or mask_w != img_w:
            x1 = x1 / max(mask_w - 1, 1) * (img_w - 1)
            x2 = x2 / max(mask_w - 1, 1) * (img_w - 1)
            y1 = y1 / max(mask_h - 1, 1) * (img_h - 1)
            y2 = y2 / max(mask_h - 1, 1) * (img_h - 1)
        
        # Add small padding to ensure bbox covers the mask
        padding = 2
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w - 1, x2 + padding)
        y2 = min(img_h - 1, y2 + padding)
        
        bboxes[int(obj_id)] = np.array([x1, y1, x2, y2], dtype=np.float32)
    
    return bboxes

# Get all unprocessed files to process
unprocessed_tiffs = get_all_unprocessed_tiffs(tiff_dir, crop_dir)
unprocessed_videos = get_all_unprocessed_videos(parent_video_dir, crop_dir)

# Combine all inputs - prioritize TIFF files
all_inputs = [(path, 'tiff') for path in unprocessed_tiffs] + [(path, 'video') for path in unprocessed_videos]

if not all_inputs:
    print("All videos and TIFF files have been processed.")
    exit()

print(f"Found {len(unprocessed_tiffs)} unprocessed TIFF files and {len(unprocessed_videos)} unprocessed video directories to process.")

for file_idx, (input_path, input_type) in enumerate(all_inputs, 1):
    print(f"\n{'='*60}")
    print(f"Processing {input_type} {file_idx}/{len(all_inputs)}: {input_path}")
    print(f"{'='*60}")

    if input_type == 'tiff':
        # Handle TIFF stack
        tiff_stack = load_tiff_stack(input_path)
        temp_dir = os.path.join(os.path.dirname(input_path), "temp_frames")
        
        # Extract frames for SAM2 processing
        frame_paths = extract_frames_from_tiff(tiff_stack, temp_dir)
        frame_names = [os.path.basename(p) for p in frame_paths]
        
        random_video_dir = temp_dir
    else:
        # Original video directory logic
        random_video_dir = input_path
        frame_names = [
            p for p in os.listdir(random_video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Process video in chunks for large videos
    all_indices = get_sorted_frame_indices(frame_names)
    video_segments = {}

    # Setup chunking directories
    chunk_root = os.path.join(random_video_dir, ".tmp_chunks")
    if os.path.exists(chunk_root):
        import shutil
        shutil.rmtree(chunk_root)
    os.makedirs(chunk_root, exist_ok=True)

    # Process video in chunks
    i = 0
    chunk_id = 0
    first_frame_idx = all_indices[0] if all_indices else 0

    while i < len(all_indices):
        # Determine chunk range with overlap
        end_idx_exclusive = min(i + CHUNK_SIZE, len(all_indices))
        # Include overlap with previous: start at i if first, else i - OVERLAP
        start_i = i if chunk_id == 0 else max(i - OVERLAP, 0)
        indices = all_indices[start_i:end_idx_exclusive]
        chunk_id += 1
        chunk_dir, local_to_global = create_chunk_dir(random_video_dir, indices, chunk_root, chunk_id)

        # Initialize state for this chunk
        inference_state = predictor.init_state(video_path=chunk_dir)

        # Determine the seeding frame for prompts in this chunk
        seed_local_idx = 0
        seed_global_idx = local_to_global[seed_local_idx]
        seed_frame_path = os.path.join(random_video_dir, f"{seed_global_idx:06d}.jpg")
        print(f"\n--- Chunk {chunk_id} start (global frame {seed_global_idx}) ---")
        
        # Only prompt user for the first chunk
        if chunk_id == 1:
            print("\nDraw a bounding box around the RIA region.")
            print("- Number key 1 = select RIA object")
            print("- Click and drag = draw bounding box") 
            print("- u = undo last box")
            print("- c = clear all boxes")
            print("- enter/q = finish")
            print("- Use toolbar above for zoom/pan")

            user_bboxes = interactive_collect_bboxes(seed_frame_path, object_ids=(1,))

            if not user_bboxes or 1 not in user_bboxes:
                print("No bounding box provided for RIA region. Skipping this file...")
                break

            ann_obj_id = 1  # Single object ID
            bbox = user_bboxes[1]
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=seed_local_idx,
                obj_id=ann_obj_id,
                box=bbox,
            )

            #Visualize prompt
            plt.figure(figsize=(12, 8))
            plt.imshow(Image.open(seed_frame_path))
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
            for i, out_obj_id in enumerate(out_obj_ids):
                show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
            plt.savefig("ria_prompt.png")
            plt.close()
        else:
            # Attempt auto-seeding from previously computed masks if available
            if seed_global_idx in video_segments:
                prev_masks = video_segments[seed_global_idx]
                img_path = resolve_image_path(chunk_dir, seed_local_idx)
                img = cv2.imread(img_path)
                img_shape = img.shape if img is not None else (110, 110, 3)
                auto_bboxes = build_bbox_from_existing_masks(prev_masks, img_shape)
                if auto_bboxes:
                    for obj_id, bbox in sorted(auto_bboxes.items()):
                        print(f"[chunk {chunk_id}] Auto-seeding obj {obj_id} with bbox from previous masks")
                        predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=seed_local_idx,
                            obj_id=int(obj_id),
                            box=bbox,
                        )
                else:
                    print(f"[chunk {chunk_id}] No bboxes derived from previous masks; proceeding without seeding")
            else:
                print(f"[chunk {chunk_id}] No previous masks to auto-seed; proceeding without seeding")

        # Propagate within the chunk starting from the seed frame
        last_quality_check = seed_global_idx
        
        for out_local_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=seed_local_idx, reverse=False):
            global_idx = local_to_global[out_local_idx]
            # Skip duplicate overlapped first frame in subsequent chunks
            if chunk_id > 1 and out_local_idx == 0 and global_idx in video_segments:
                continue
            
            # Store masks and confidence scores
            video_segments[global_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
            # Calculate confidence scores
            confidence_scores = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                if isinstance(out_mask_logits[i], torch.Tensor):
                    probs = torch.sigmoid(out_mask_logits[i])
                    # Calculate confidence as mean probability of positive pixels
                    positive_mask = out_mask_logits[i] > 0.0
                    if torch.sum(positive_mask) > 0:
                        confidence = torch.mean(probs[positive_mask]).item()
                    else:
                        confidence = 0.0
                    confidence_scores[out_obj_id] = confidence

            # Check quality periodically
            if (global_idx - last_quality_check) >= QUALITY_CHECK_INTERVAL:
                needs_reprompt, quality_report = check_mask_quality(video_segments, global_idx, confidence_scores)
                
                if needs_reprompt:
                    print(f"\n!!! Quality check failed at frame {global_idx} !!!")
                    print(f"Issues: {', '.join(quality_report['issues'])}")
                    print("Re-prompting user...")
                    
                    # Re-prompt user on current frame
                    current_frame_path = os.path.join(random_video_dir, f"{global_idx:06d}.jpg")
                    if not os.path.exists(current_frame_path):
                        # Try alternative naming
                        frame_names_local = sorted([f for f in os.listdir(random_video_dir) if f.endswith('.jpg')])
                        if global_idx < len(frame_names_local):
                            current_frame_path = os.path.join(random_video_dir, frame_names_local[global_idx])
                    
                    print("Please draw a new bounding box to improve tracking quality.")
                    print("- Number key 1 = select RIA object")
                    print("- Click and drag = draw bounding box")
                    print("- u = undo last box")
                    print("- c = clear all boxes")
                    print("- enter/q = finish")
                    
                    new_bboxes = interactive_collect_bboxes(current_frame_path, object_ids=(1,))
                    
                    if new_bboxes and 1 in new_bboxes:
                        # Convert global frame index to local frame index for this chunk
                        local_frame_idx = out_local_idx
                        
                        # Add new bounding box to the current frame
                        print(f"Adding correction bbox at frame {global_idx}")
                        predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=local_frame_idx,
                            obj_id=1,
                            box=new_bboxes[1],
                        )
                        print("Bounding box added. Continuing propagation...")
                    else:
                        print("No new bounding box provided. Continuing with current tracking...")
                    
                    last_quality_check = global_idx
                else:
                    print(f"[quality] Frame {global_idx}: {quality_report['status']}")
                    if global_idx - last_quality_check >= QUALITY_CHECK_INTERVAL:
                        last_quality_check = global_idx

            # Periodic visualization every VIS_INTERVAL frames (relative to first frame)
            if ((global_idx - first_frame_idx) % VIS_INTERVAL) == 0:
                try:
                    overlay_predictions_on_frame(random_video_dir, global_idx, video_segments, alpha=0.6)
                    print(f"[viz] Saved overlay for frame {global_idx:06d}")
                    
                    # Show confidence scores in visualization
                    if confidence_scores:
                        conf_str = ", ".join([f"obj{oid}:{conf:.2f}" for oid, conf in confidence_scores.items()])
                        print(f"[confidence] {conf_str}")
                        
                except Exception as e:
                    print(f"[viz] Failed to save overlay for frame {global_idx}: {e}")

        # Advance to next non-overlapping position
        i = end_idx_exclusive

    # Cleanup chunk dirs
    try:
        import shutil
        shutil.rmtree(chunk_root)
    except Exception:
        pass

    # Only process cropping if we have video segments (user didn't skip)
    if video_segments:
        if input_type == 'tiff':
            output_path = os.path.join(crop_dir, os.path.splitext(os.path.basename(input_path))[0] + "_crop.tif")
            original_size = tiff_stack.shape[1:3]  # height, width
            
            process_tiff_stack_fixed_crop(input_path, output_path, video_segments, 110)
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)
        else:
            output_folder = os.path.join(os.path.dirname(crop_dir), os.path.basename(random_video_dir) + "_crop")
            first_frame = cv2.imread(os.path.join(random_video_dir, frame_names[0]))
            original_size = first_frame.shape[:2]
            
            process_frames_fixed_crop(random_video_dir, output_folder, video_segments, original_size, 110)
    
    print(f"Completed processing {input_type} {file_idx}/{len(all_inputs)}: {os.path.basename(input_path)}")

print(f"\nAll {len(all_inputs)} files have been processed successfully!")