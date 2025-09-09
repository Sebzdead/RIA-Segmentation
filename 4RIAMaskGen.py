'''This script performs SAM2 video segmentation with interactive prompts: you label the first frame
with positive/negative points per object, then masks are propagated across the video.

Overview
- Picks a random unprocessed video directory under CROPPED/.
- Loads the first frame and lets you click positive/negative points for nrD and nrV.
- Initializes a SAM2 video predictor and adds your prompts on that first frame.
- Propagates masks forward across the video, produces per-frame masks.
- Analyzes mask quality and optionally renders a mask-overlay video for quick inspection.
- Saves masks into an H5 file.

Autosegmentation with SAM2
- SAM2 uses user-provided prompts (points with binary labels) to initialize objects (obj_id).
- After establishing prompts on a frame, the model's video predictor propagates masks temporally.
- add_new_points registers prompts for an object on a frame; propagate_in_video runs temporal tracking and returns masks for each object per frame.

Inputs
- crop_videos_dir: directory containing per-video frame folders (each folder of JPG frames).
- SAM2 checkpoint and model config.

Outputs
- Console analysis of mask issues across the video.
- A mask-overlay MP4 (optional).
- An H5 file containing boolean masks per object per frame.
'''
import os
import cv2
import shutil
import tqdm
import torch
torch.cuda.empty_cache()
import h5py
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from scipy.ndimage import binary_dilation
from sam2.build_sam import build_sam2_video_predictor # type: ignore

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

from sam2.build_sam import build_sam2_video_predictor # type: ignore
sam2_checkpoint = r"c:\Users\User\Documents\sam2\checkpoints\sam2.1_hiera_base_plus.pt"
model_cfg = r"c:\Users\User\Documents\sam2\sam2\configs\sam2.1\sam2.1_hiera_b+.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Define the two target objects used throughout
OBJECTS = {1: 'nrD', 2: 'nrV'}
CHUNK_SIZE = 200   # Reduced from 201 to prevent memory issues
OVERLAP = 1        # number of overlapping frames between consecutive chunks (for continuity)
VIS_INTERVAL = 10  # visualize masks every N frames

# Quality thresholds for re-prompting
CONFIDENCE_THRESHOLD = 0.8  # minimum confidence score (0-1)
QUALITY_CHECK_INTERVAL = 10  # check quality every N frames

crop_videos_dir = '4CROP_JPG'
segmented_videos_dir = '5RIA_SEGMENT'
output_dir = '5RIA_SEGMENT'
# Remove the static output_video_path - will be generated dynamically
# output_video_path = "mask_overlay.mp4"

def interactive_collect_bboxes(image_path, object_ids=None):
    """
    Collect bounding box prompts on a single image interactively.

    Controls:
    - Number keys 1..2: select object id (1:nrD, 2:nrV)
    - Click and drag: draw bounding box for current object
    - u: undo last box for current object
    - c: clear current object's boxes
    - enter/q: finish and return
    - Use matplotlib toolbar for zoom/pan

    Returns:
        dict[int, np.ndarray] mapping obj_id -> bbox[4] (x1, y1, x2, y2)
    """
    if object_ids is None:
        object_ids = tuple(sorted(OBJECTS.keys()))
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

    # Define specific colors for each object: 1=red, 2=blue
    object_colors = {1: (1.0, 0.0, 0.0), 2: (0.0, 0.0, 1.0)}  # red, blue

    def redraw():
        ax.clear()
        ax.imshow(img)
        ax.set_axis_off()
        
        # Draw existing boxes
        for oid in sorted(storage.keys()):
            boxes = storage[oid]["boxes"]
            if not boxes:
                continue
            
            # Use consistent colors: nrD=red, nrV=blue
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
            name = OBJECTS.get(oid, str(oid))
            obj_lines.append(f"{oid}:{name}  boxes:{len(boxes)}")
        obj_list = ",  ".join(obj_lines)
        
        ax.text(0.01, 0.99,
            f"Active={current_obj}:{OBJECTS.get(current_obj,'')} | Total boxes: {total}\n"
            f"Keys: 1-2 select obj, click+drag=draw box, u=undo, c=clear, enter/q=finish\n"
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
    """
    Overlay a semi-transparent mask onto a Matplotlib Axes.

    Args:
        mask (np.ndarray | torch.Tensor): Binary mask of shape (H, W) or (1, H, W).
        ax (matplotlib.axes.Axes): The axes to draw on.
        obj_id (int | None): Optional object ID used to pick a stable color; None picks default.
        random_color (bool): If True, pick a random color; otherwise a tab10 color index.

    Returns:
        None. Draws into the provided axes.
    """
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else \
            np.array([*plt.get_cmap("tab10")(0 if obj_id is None else obj_id)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=26):
    """
    Plot positive (green) and negative (red) prompt points.

    Args:
        coords (np.ndarray): Array of shape (N, 2) with (x, y) in pixel coordinates.
        labels (np.ndarray): Array of shape (N,) with 1 for positive, 0 for negative.
        ax (matplotlib.axes.Axes): The axes to draw on.
        marker_size (int): Marker size in points^2.

    Returns:
        None. Draws into the provided axes.
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    if len(pos_points):
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points):
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='x', s=marker_size, linewidth=1.25)

def remove_prompt_frames_from_video(video_dir, frame_mapping):
    # No-op in interactive mode; kept for compatibility if frame_mapping is empty
    if not frame_mapping:
        return
    for frame_num in frame_mapping.keys():
        frame_name = f"{frame_num:06d}.jpg"
        frame_path = os.path.join(video_dir, frame_name)
        if os.path.exists(frame_path):
            os.remove(frame_path)
    print(f"Removed {len(frame_mapping)} prompt frames from the video directory.")

def add_prompts(inference_state, frame_idx, obj_id, points, labels):

    """
    Add prompt points to the predictor for a given frame and object, then visualize the result.

    Args:
        inference_state: Opaque state from predictor.init_state for this video.
        frame_idx (int): The frame index (in the video_dir) where the prompt is applied.
        obj_id (int | None): Target object ID; pass None to let predictor assign/create.
        points (np.ndarray): Array of shape (N, 2) in pixel coordinates (x, y).
        labels (np.ndarray): Array of shape (N,) with 1 (positive) or 0 (negative) per point.

    Returns:
        None. Updates the predictor state and saves a visualization PNG.
    """
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels
    )
    # Visualize the prompt frame, if needed
    plt.figure(figsize=(12, 8))
    plt.title(f"Frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, f"{frame_idx:06d}.jpg")))
    show_points(points, labels, plt.gca())
    
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
    
    plt.savefig(f"prompt_frame.png")
    plt.close()

def check_overlap(mask1, mask2):
    """
    Compute overlap statistics between two binary masks.

    Args:
        mask1 (np.ndarray): Boolean array (H, W).
        mask2 (np.ndarray): Boolean array (H, W).

    Returns:
        tuple[bool, float, int]:
            - has_overlap (bool): True if any overlapping pixels exist.
            - iou (float): Intersection over Union in [0, 1]; 0 if union is empty.
            - overlap_pixels (int): Count of intersecting pixels.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    overlap_pixels = np.sum(intersection)
    iou = overlap_pixels / np.sum(union) if np.sum(union) > 0 else 0
    return overlap_pixels > 0, iou, overlap_pixels

def check_distance(mask1, mask2, max_distance=10):
    """
    Check whether two masks are within a specified pixel distance using binary dilation.

    Args:
        mask1 (np.ndarray): Boolean array (H, W).
        mask2 (np.ndarray): Boolean array (H, W).
        max_distance (int): Number of dilation iterations (approximate distance threshold).

    Returns:
        bool: True if dilated masks intersect (i.e., within distance), else False.
    """
    dilated_mask1 = binary_dilation(mask1, iterations=max_distance)
    dilated_mask2 = binary_dilation(mask2, iterations=max_distance)
    return np.any(np.logical_and(dilated_mask1, dilated_mask2))

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
    
    # Check for missing objects
    missing_objects = []
    for obj_id in OBJECTS.keys():
        if obj_id not in video_segments[frame_idx] or video_segments[frame_idx][obj_id] is None:
            missing_objects.append(obj_id)
            needs_reprompt = True
    
    if missing_objects:
        quality_report["issues"].append(f"Missing objects: {missing_objects}")
    
    # Check confidence scores if provided
    low_confidence = []
    if confidence_scores:
        for obj_id, confidence in confidence_scores.items():
            if obj_id in OBJECTS and confidence < CONFIDENCE_THRESHOLD:
                low_confidence.append(obj_id)
                needs_reprompt = True
    
    if low_confidence:
        quality_report["issues"].append(f"Low confidence (< {CONFIDENCE_THRESHOLD}): {low_confidence}")
    
    if not quality_report["issues"]:
        quality_report["status"] = "good"
    else:
        quality_report["status"] = "needs_reprompt" if needs_reprompt else "warning"
    
    return needs_reprompt, quality_report

def analyze_masks(video_segments):
    """
    Analyze per-frame masks for issues: empty, large, overlapping, and distant objects.

    Args:
        video_segments (dict[int, dict[int | None, np.ndarray]]):
            Mapping frame_idx -> {obj_id -> bool mask (H, W)}.

    Returns:
        tuple:
            - results (dict): Per-category dict of issues per frame.
            - max_counts (dict[str, int]): Max count of issues per category across frames.
            - max_frames (dict[str, int | None]): Latest frame index achieving the max per category.
    """
    results = {'empty': {}, 'high': {}, 'overlapping': {}, 'distant': {}}
    max_counts = {'empty': 0, 'high': 0, 'overlapping': 0, 'distant': 0}
    max_frames = {'empty': None, 'high': None, 'overlapping': None, 'distant': None}

    for frame, mask_dict in video_segments.items():
        mask_ids = [mask_id for mask_id in mask_dict.keys() if mask_id is not None]
        for i in range(len(mask_ids)):
            mask_id = mask_ids[i]
            mask = mask_dict[mask_id]
            if mask is not None:
                mask_sum = mask.sum()
                if mask_sum == 0:
                    results['empty'].setdefault(frame, []).append(mask_id)
                elif mask_sum >= 800:
                    results['high'].setdefault(frame, []).append(mask_id)
            
                for j in range(i + 1, len(mask_ids)):
                    other_mask_id = mask_ids[j]
                    other_mask = mask_dict[other_mask_id]
                    if other_mask is not None:
                        is_overlapping, iou, overlap_pixels = check_overlap(mask, other_mask)
                        if is_overlapping:
                            results['overlapping'].setdefault(frame, []).append((mask_id, other_mask_id, iou, overlap_pixels))
        
        # Check distance between objects 1 and 2 (nrD, nrV)
        if 1 in mask_dict and 2 in mask_dict and mask_dict[1] is not None and mask_dict[2] is not None:
            if not check_distance(mask_dict[1], mask_dict[2]):
                results['distant'].setdefault(frame, []).append((1, 2))

        for category in ['empty', 'high', 'overlapping', 'distant']:
            if frame in results[category]:
                count = len(results[category][frame])
                if count > max_counts[category]:
                    max_counts[category] = count
                    max_frames[category] = frame

    return results, max_counts, max_frames

def collect_results(result_dict, condition, max_count, max_frame):
    """
    Format detailed and summary strings for a given issue category.

    Args:
        result_dict (dict): Mapping frame_idx -> list of issue details.
        condition (str): One of 'empty', 'high', 'overlapping', 'distant'.
        max_count (int): Maximum occurrences for this category.
        max_frame (int | None): Frame index with the maximum count.

    Returns:
        tuple[list[str], list[str]]: (detailed_lines, summary_lines).
    """
    detailed_output = []
    summary_output = []
    
    if result_dict:
        detailed_output.append(f"!!! Frames with masks {condition}:")
        for frame, data in result_dict.items():
            if condition == "overlapping":
                overlap_info = [f"{a}-{b} ({iou:.2%}, {pixels} pixels)" for a, b, iou, pixels in data]
                detailed_output.append(f"  Frame {frame}: Overlapping Mask ID pairs {', '.join(overlap_info)}")
            elif condition == "distant":
                detailed_output.append(f"  Frame {frame}: Distant Mask ID pairs {data}")
            else:
                detailed_output.append(f"  Frame {frame}: Mask IDs {data}")
        if max_count > 0:
            summary_output.append(f"Latest frame with highest number of {condition} masks: {max_frame} (Count: {max_count})")
    else:
        summary_output.append(f"Yay! No masks {condition} found!")
    
    return detailed_output, summary_output

def analyze_and_print_results(video_segments):
    """
    Analyze masks across all frames and print detailed and summary diagnostics.

    Args:
        video_segments (dict[int, dict[int | None, np.ndarray]]):
            Mapping frame_idx -> {obj_id -> bool mask (H, W)}.

    Returns:
        None. Prints results to stdout.
    """
    analysis_results, max_counts, max_frames = analyze_masks(video_segments)

    all_detailed_outputs = []
    all_summary_outputs = []
    problematic_frame_counts = {
        'empty': 0,
        'high': 0,
        'overlapping': 0,
        'distant': 0
    }
    total_frames = len(video_segments)

    for category in ['empty', 'high', 'overlapping', 'distant']:
        detailed, summary = collect_results(analysis_results[category], category, max_counts[category], max_frames[category])
        all_detailed_outputs.extend(detailed)
        all_summary_outputs.extend(summary)
        problematic_frame_counts[category] = len(analysis_results[category])

    for line in all_detailed_outputs:
        print(line)

    for line in all_summary_outputs:
        print(line)

    print("\nNumber of problematic frames:")
    for category, count in problematic_frame_counts.items():
        percentage = (count / total_frames) * 100
        print(f"Frames with {category} masks: {count} out of {total_frames} ({percentage:.2f}%)")

    unique_problematic_frames = set()
    for category, frames in analysis_results.items():
        unique_problematic_frames.update(frames.keys())
    
    unique_problematic_count = len(unique_problematic_frames)
    unique_problematic_percentage = (unique_problematic_count / total_frames) * 100
    print(f"\nTotal number of unique problematic frames: {unique_problematic_count} out of {total_frames} ({unique_problematic_percentage:.2f}%)")

def create_mask_overlay_video(video_dir, frame_names, video_segments, output_video_path, fps=10, alpha=0.99):
    """
    Render a video with colored mask overlays per object for visual inspection.

    Args:
        video_dir (str): Directory with frame images.
        frame_names (list[str]): Ordered list of frame filenames.
        video_segments (dict[int, dict[int | None, np.ndarray]]): Predicted masks by frame.
        output_video_path (str): Path to the output MP4 file.
        fps (int): Frames per second.
        alpha (float): Overlay strength in [0, 1]; higher means stronger mask color.

    Returns:
        None. Writes an MP4 file to disk.
    """
    # Updated colors to match object assignments: 1=red, 2=blue
    COLORS = {
        1: (255, 0, 0),     # Red for nrD
        2: (0, 0, 255),     # Blue for nrV
        3: (0, 255, 0),     # Green
        4: (255, 0, 255),   # Magenta
        5: (0, 255, 255),   # Cyan
        6: (128, 0, 0),     # Maroon
        7: (128, 0, 128),   # Purple
        8: (0, 0, 128),     # Navy
        9: (128, 128, 0),   # Olive
        10: (0, 128, 0),    # Dark Green
        11: (0, 128, 128),  # Teal
        12: (255, 128, 0),  # Orange
        13: (255, 0, 128),  # Deep Pink
        14: (128, 255, 0),  # Lime
        15: (255, 255, 0),  # Yellow
        16: (0, 255, 128)   # Spring Green
    }

    def overlay_masks_on_image(image_path, masks, colors, alpha=0.99):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        overlay = np.zeros_like(image)
        
        for mask_id, mask in masks.items():
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            if mask.dtype != bool:
                mask = mask > 0.5
            
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            color = colors[mask_id]
            colored_mask = np.zeros_like(image)
            colored_mask[mask_resized == 1] = color
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)
        
        overlaid_image = cv2.addWeighted(image, 1, overlay, alpha, 0)
        return overlaid_image

    frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    if frame is None:
        raise ValueError(f"Could not read first frame from {os.path.join(video_dir, frame_names[0])}")
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    all_mask_ids = set()
    for masks in video_segments.values():
        all_mask_ids.update(masks.keys())
    colors = {}
    for mask_id in all_mask_ids:
        if mask_id in COLORS:
            colors[mask_id] = COLORS[mask_id]
        else:
            # Fallback for unexpected mask IDs
            fallback_colors = list(COLORS.values())
            colors[mask_id] = fallback_colors[hash(mask_id) % len(fallback_colors)]

    for frame_idx in range(len(frame_names)):
        image_path = os.path.join(video_dir, frame_names[frame_idx])
        try:
            if frame_idx in video_segments:
                masks = video_segments[frame_idx]
                overlaid_frame = overlay_masks_on_image(image_path, masks, colors, alpha)
            else:
                overlaid_frame = cv2.imread(image_path)
                if overlaid_frame is None:
                    raise ValueError(f"Could not read image from {image_path}")
                overlaid_frame = cv2.cvtColor(overlaid_frame, cv2.COLOR_BGR2RGB)
            
            out.write(cv2.cvtColor(overlaid_frame, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            original_frame = cv2.imread(image_path)
            if original_frame is not None:
                out.write(original_frame)
            else:
                print(f"Could not read original frame {frame_idx}")

    out.release()
    print(f"Video saved to {output_video_path}")

def overlay_predictions_on_frame(video_dir, frame_idx, video_segments, alpha=0.99):
    """
    Save a single frame image with predicted masks overlaid and display it to the user.

    Args:
        video_dir (str): Directory with frame images.
        frame_idx (int): Frame index (zero-padded name in video_dir).
        video_segments (dict[int, dict[int | None, np.ndarray]]): Predicted masks by frame.
        alpha (float): Overlay strength in [0, 1].

    Returns:
        None. Writes PNG 'frame_prediction_overlay.png' and displays it.
    """
    # Updated colors to match object assignments: 1=red, 2=blue
    COLORS = {
        1: (255, 0, 0),     # Red for nrD
        2: (0, 0, 255),     # Blue for nrV
        3: (0, 255, 0),     # Green
        4: (255, 0, 255),   # Magenta
        5: (0, 255, 255),   # Cyan
        6: (128, 0, 0),     # Maroon
        7: (128, 0, 128),   # Purple
        8: (0, 0, 128),     # Navy
        9: (128, 128, 0),   # Olive
        10: (0, 128, 0),    # Dark Green
        11: (0, 128, 128),  # Teal
        12: (255, 128, 0),  # Orange
        13: (255, 0, 128),  # Deep Pink
        14: (128, 255, 0),  # Lime
        15: (255, 255, 0),  # Yellow
        16: (0, 255, 128)   # Spring Green
    }

    image_path = os.path.join(video_dir, f"{frame_idx:06d}.jpg")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    overlay = np.zeros_like(image)
    if frame_idx in video_segments:
        masks = video_segments[frame_idx]

        colors = {}
        for mask_id in masks.keys():
            if mask_id in COLORS:
                colors[mask_id] = COLORS[mask_id]
            else:
                # Fallback for unexpected mask IDs
                fallback_colors = list(COLORS.values())
                colors[mask_id] = fallback_colors[hash(mask_id) % len(fallback_colors)]

        for mask_id, mask in masks.items():
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            if mask.dtype != bool:
                mask = mask > 0.5

            if mask.ndim > 2:
                mask = mask.squeeze()

            mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            color = colors[mask_id]
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
    ax.set_title(f"Frame {frame_idx:06d} - Mask Overlay Preview")
    ax.set_axis_off()
    
    # Add mask legend if masks exist
    if frame_idx in video_segments:
        legend_text = []
        for mask_id in sorted(video_segments[frame_idx].keys()):
            if mask_id in OBJECTS:
                obj_name = OBJECTS[mask_id]
                color_name = "Red" if mask_id == 1 else "Blue" if mask_id == 2 else f"Color {mask_id}"
                legend_text.append(f"{mask_id}: {obj_name} ({color_name})")
        
        if legend_text:
            ax.text(0.02, 0.98, '\n'.join(legend_text), transform=ax.transAxes, 
                   va='top', ha='left', color='white', fontsize=10,
                   bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    plt.close(fig)

    cv2.imwrite("frame_prediction_overlay.png", cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR))
    print(f"Overlaid image saved to frame_prediction_overlay.png")

def analyze_prompt_frames_immediate(video_dir, frame_mapping, prompt_data, inference_state, predictor):
    """
    Re-run predictor on each appended prompt frame (no new points) to check immediate issues.

    Args:
        video_dir (str): Directory with frames.
        frame_mapping (dict[int, int]): {new_frame_idx: original_prompt_number}.
        prompt_data (dict): Loaded JSON-like structure with prompts per original prompt number.
        inference_state: Predictor state initialized for the video.
        predictor: SAM2 video predictor instance.

    Returns:
        dict[int, dict]: Per new_frame_idx -> summary:
            {
              'original_frame': int,
              'all_objects': list[int | None],
              'empty_masks': list[int],
              'large_masks': list[int],
              'overlapping_masks': list[tuple[int, int, float, int]]  # (a, b, IoU, pixels)
            }
    """
    prompt_frame_results = {}
    pbar = tqdm.tqdm(frame_mapping.items(), desc="Analyzing prompt frames", unit="frame")

    for new_frame_num, original_frame_num in pbar:
        if str(original_frame_num) in prompt_data:
            pbar.set_postfix({"Original Frame": original_frame_num})
            
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=new_frame_num,
                obj_id=None,
                points=np.empty((0, 2)),
                labels=np.empty(0)
            )

            masks = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            empty_masks = []
            large_masks = []
            overlapping_masks = []
            
            def calculate_overlap(mask1, mask2):
                intersection = np.logical_and(mask1, mask2)
                union = np.logical_or(mask1, mask2)
                overlap_pixels = np.sum(intersection)
                iou = overlap_pixels / np.sum(union)
                return iou, overlap_pixels
            
            for obj_id, mask in masks.items():
                if obj_id is not None:
                    mask_sum = mask.sum()
                    
                    if mask_sum == 0:
                        empty_masks.append(obj_id)
                    elif mask_sum >= 800:
                        large_masks.append(obj_id)
                    
                    for other_obj_id, other_mask in masks.items():
                        if other_obj_id is not None and obj_id != other_obj_id:
                            overlap, overlap_pixels = calculate_overlap(mask, other_mask)
                            if overlap > 0.01:  # 1% overlap threshold
                                overlapping_masks.append((obj_id, other_obj_id, overlap, overlap_pixels))
            
            prompt_frame_results[new_frame_num] = {
                'original_frame': original_frame_num,
                'all_objects': list(masks.keys()),
                'empty_masks': empty_masks,
                'large_masks': large_masks,
                'overlapping_masks': overlapping_masks
            }
            
            # Visualize the prompt frame results, if needed
            plt.figure(figsize=(12, 8))
            plt.title(f"Prompt Frame {new_frame_num} (Original: {original_frame_num})")
            image = Image.open(os.path.join(video_dir, f"{new_frame_num:06d}.jpg"))
            plt.imshow(image)
            
            for obj_id, mask in masks.items():
                if obj_id is not None:
                    show_mask(mask, plt.gca(), obj_id=obj_id, random_color=True)
            
            plt.savefig(f"prompt_frame_analysis_{new_frame_num}.png")
            plt.close()

    return prompt_frame_results

def print_prompt_frame_analysis(prompt_frame_results):
    """
    Print a concise summary of prompt frame issues and object coverage statistics.

    Args:
        prompt_frame_results (dict[int, dict]): Output of analyze_prompt_frames_immediate.

    Returns:
        None. Prints to stdout.
    """
    print("\nPrompt Frame Analysis Summary:")    
    problematic_frames = []
    frames_without_issues = []
    all_objects = set()

    def safe_sort(iterable):
        return sorted((item for item in iterable if item is not None), key=lambda x: (x is None, x))

    for frame_num, results in prompt_frame_results.items():
        issues = []
        frame_objects = set(obj for obj in results['all_objects'] if obj is not None)

        if results['empty_masks']:
            issues.append(f"Empty masks: {safe_sort(results['empty_masks'])}")
        if results['large_masks']:
            issues.append(f"Large masks (800+ pixels): {safe_sort(results['large_masks'])}")
        if results['overlapping_masks']:
            overlap_info = [f"{a}-{b} ({overlap:.2%}, {pixels} pixels)" for a, b, overlap, pixels in results['overlapping_masks'] if a is not None and b is not None]
            issues.append(f"Overlapping masks: {', '.join(overlap_info)}")
        
        all_objects.update(frame_objects)        
        if issues:
            problematic_frames.append((frame_num, results['original_frame'], issues, frame_objects))
        else:
            frames_without_issues.append((frame_num, frame_objects))

    if frames_without_issues:
        print("\nFrames without issues:")
        for frame_num, frame_objects in frames_without_issues:
            print(f"  Frame {frame_num}: Objects present: {safe_sort(frame_objects)}")

    if problematic_frames:
        print("Problematic frames:")
        for frame_num, original_frame, issues, frame_objects in problematic_frames:
            print(f"  Frame {frame_num} (Original: {original_frame}):")
            print(f"    Objects present: {safe_sort(frame_objects)}")
            for issue in issues:
                print(f"    - {issue}")
    else:
        print("No problematic frames detected.")


    print(f"\nTotal frames analyzed: {len(prompt_frame_results)}")
    print(f"Frames with issues: {len(problematic_frames)}")
    print(f"Frames without issues: {len(frames_without_issues)}")

    print(f"\nTotal unique object IDs detected across all frames: {safe_sort(all_objects)}")
    print(f"Number of unique objects: {len(all_objects)}")

    # Additional statistics
    objects_per_frame = [len([obj for obj in results['all_objects'] if obj is not None]) for results in prompt_frame_results.values()]
    avg_objects_per_frame = sum(objects_per_frame) / len(objects_per_frame) if objects_per_frame else 0
    print(f"\nAverage number of objects per frame: {avg_objects_per_frame:.2f}")
    print(f"Minimum objects in a frame: {min(objects_per_frame) if objects_per_frame else 0}")
    print(f"Maximum objects in a frame: {max(objects_per_frame) if objects_per_frame else 0}")

def save_video_segments_to_h5(video_segments, video_dir, output_dir, frame_mapping):
    """
    Save predicted masks to an H5 file, excluding temporary prompt frames.

    Args:
        video_segments (dict[int, dict[int | None, np.ndarray]]): Predicted masks by frame.
        video_dir (str): Video frames directory (used to derive output filename).
        output_dir (str): Directory to write the H5 file.
        frame_mapping (dict[int, int]): {new_frame_idx: original_prompt_number} to exclude.

    Returns:
        dict[int, dict[int | None, np.ndarray]]:
            Filtered video_segments without the temporary prompt frames.
    """
    last_folder = os.path.basename(os.path.normpath(video_dir))    
    filename = f"{last_folder}.h5"    
    os.makedirs(output_dir, exist_ok=True)    
    output_path = os.path.join(output_dir, filename)
    exclude_frames = set(frame_mapping.keys())
    filtered_video_segments = {
        frame: segments for frame, segments in video_segments.items()
        if frame not in exclude_frames
    }

    with h5py.File(output_path, 'w') as f:
        f.attrs['num_frames'] = len(filtered_video_segments)        
        sample_frame = next(iter(filtered_video_segments.values()))
        object_ids = list(sample_frame.keys())
        f.attrs['object_ids'] = [str(obj_id) if obj_id is not None else 'None' for obj_id in object_ids]
        
        # Get actual mask dimensions from the first available mask
        mask_height, mask_width = 110, 110  # Default fallback
        for obj_id in object_ids:
            if obj_id in sample_frame and sample_frame[obj_id] is not None:
                sample_mask = sample_frame[obj_id]
                if sample_mask.ndim >= 2:
                    mask_height, mask_width = sample_mask.shape[-2:]
                    break
        
        # Create datasets for each object
        for obj_id in object_ids:
            obj_id_str = str(obj_id) if obj_id is not None else 'None'
            # Create a dataset for each object, with the first dimension being the number of frames
            f.create_dataset(f'masks/{obj_id_str}', 
                             shape=(len(filtered_video_segments), 1, mask_height, mask_width),
                             dtype=bool,
                             compression="gzip")
        
        for i, (frame_idx, objects) in enumerate(sorted(filtered_video_segments.items(), reverse=False)):
            for obj_id, mask in objects.items():
                obj_id_str = str(obj_id) if obj_id is not None else 'None'
                f[f'masks/{obj_id_str}'][i] = mask

    print(f"Saved filtered video segments to: {output_path}")
    print(f"Number of frames saved: {len(filtered_video_segments)}")
    print(f"Number of frames excluded: {len(exclude_frames)}")
    print(f"Mask dimensions: {mask_height}x{mask_width}")
    return filtered_video_segments

def get_random_unprocessed_video(crop_videos_dir, segmented_videos_dir):
    """
    Pick a random video folder from crop_videos_dir that has not yet been segmented.

    Args:
        crop_videos_dir (str): Path containing subfolders of frames (one subfolder per video).
        segmented_videos_dir (str): Path containing H5 outputs; used to filter processed videos.

    Returns:
        str: Absolute path to a randomly chosen unprocessed video directory.

    Raises:
        ValueError: If all videos appear to be processed already.
    """
    all_videos = [d for d in os.listdir(crop_videos_dir) if os.path.isdir(os.path.join(crop_videos_dir, d))]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(segmented_videos_dir, video + ".h5"))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(crop_videos_dir, random.choice(unprocessed_videos))

def get_all_unprocessed_videos(crop_videos_dir, segmented_videos_dir):
    """
    Get all unprocessed video folders in alphabetical order.

    Args:
        crop_videos_dir (str): Path containing subfolders of frames (one subfolder per video).
        segmented_videos_dir (str): Path containing H5 outputs; used to filter processed videos.

    Returns:
        list[str]: List of absolute paths to unprocessed video directories in alphabetical order.
    """
    all_videos = [d for d in os.listdir(crop_videos_dir) if os.path.isdir(os.path.join(crop_videos_dir, d))]
    unprocessed_videos = [
        video for video in sorted(all_videos)  # Sort alphabetically
        if not os.path.exists(os.path.join(segmented_videos_dir, video + ".h5"))
    ]
    
    return [os.path.join(crop_videos_dir, video) for video in unprocessed_videos]

# Process all unprocessed videos in alphabetical order
unprocessed_videos = get_all_unprocessed_videos(crop_videos_dir, segmented_videos_dir)

if not unprocessed_videos:
    print("All videos have been processed.")
    exit()

print(f"Found {len(unprocessed_videos)} unprocessed videos to process.")

for video_idx, video_dir in enumerate(unprocessed_videos, 1):
    print(f"\n{'='*60}")
    print(f"Processing video {video_idx}/{len(unprocessed_videos)}: {video_dir}")
    print(f"{'='*60}")

    # Gather frames and choose the first as prompt frame
    frame_names = sorted([p for p in os.listdir(video_dir) if p.lower().endswith(('.jpg', '.jpeg'))],
                         key=lambda p: int(os.path.splitext(p)[0]))
    if not frame_names:
        print(f"No frames found in {video_dir}, skipping...")
        continue
    first_frame_idx = int(os.path.splitext(frame_names[0])[0])
    first_frame_path = os.path.join(video_dir, frame_names[0])

    # Remove the initial prompt collection - it will be done in the first chunk
    # print("\nLabel the first frame with positive/negative points.\n"
#     "- Number keys 1..2 select object id (1:nrD, 2:nrV)\n"
#       "- Left click = positive, Right click = negative\n"
#       "- u=undo, c=clear current object, enter/q=finish\n")
# user_prompts = interactive_collect_prompts(first_frame_path, object_ids=(1, 2))

def get_sorted_frame_indices(names):
    return [int(os.path.splitext(n)[0]) for n in names]

def create_chunk_dir(base_dir, indices, chunk_root, chunk_id):
    """Create a temporary chunk directory with locally renumbered frames 000000.jpg, 000001.jpg, ...
    Returns (chunk_dir, local_to_global) mapping list where local_to_global[local_idx] = global_frame_idx
    """
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

def sample_points_from_mask(mask, img_shape, num_points=5):
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return []
    idxs = np.linspace(0, len(xs) - 1, num=min(num_points, len(xs)), dtype=int)
    h_mask, w_mask = mask.shape[-2:] if mask.ndim >= 2 else (mask.shape[0], 1)
    img_h, img_w = img_shape[:2]
    pts = []
    for i in idxs:
        x_m, y_m = xs[i], ys[i]
        x_img = float(x_m) / max(w_mask - 1, 1) * (img_w - 1)
        y_img = float(y_m) / max(h_mask - 1, 1) * (img_h - 1)
        pts.append((x_img, y_img))
    return pts

def build_prompt_from_existing_masks(masks_for_frame, img_shape, max_points_per_obj=6):
    prompts = {}
    for obj_id, mask in masks_for_frame.items():
        if obj_id is None or obj_id not in OBJECTS:
            continue
        m = mask.squeeze() if mask.ndim > 2 else mask
        pts = sample_points_from_mask(m, img_shape, num_points=max_points_per_obj)
        if not pts:
            continue
        prompts[int(obj_id)] = (np.array(pts, dtype=np.float32), np.ones(len(pts), dtype=np.int32))
    return prompts

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
        if obj_id is None or obj_id not in OBJECTS:
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

all_indices = get_sorted_frame_indices(frame_names)
video_segments = {}

chunk_root = os.path.join(video_dir, ".tmp_chunks")
if os.path.exists(chunk_root):
    shutil.rmtree(chunk_root)
os.makedirs(chunk_root, exist_ok=True)

active_predictor = predictor  # keep GPU/MPS

i = 0
chunk_id = 0
prev_last_frame_idx = None
while i < len(all_indices):
    # Determine chunk range with overlap
    end_idx_exclusive = min(i + CHUNK_SIZE, len(all_indices))
    
    # For subsequent chunks, skip the overlap frame if it was already processed
    if chunk_id > 0 and i > 0:
        start_i = i  # Start from current position, no overlap to avoid duplicates
        indices = all_indices[start_i:end_idx_exclusive]
    else:
        # First chunk processes from the beginning
        indices = all_indices[i:end_idx_exclusive]
    
    chunk_id += 1
    
    # Skip if no frames to process
    if not indices:
        break
    
    # Clean up previous chunk directory before creating new one
    if chunk_id > 1:
        prev_chunk_dir = os.path.join(chunk_root, f"chunk_{chunk_id-1:04d}")
        if os.path.exists(prev_chunk_dir):
            try:
                shutil.rmtree(prev_chunk_dir)
            except Exception as e:
                print(f"Warning: Could not clean up previous chunk dir: {e}")
    
    chunk_dir, local_to_global = create_chunk_dir(video_dir, indices, chunk_root, chunk_id)
    
    # Clear GPU cache before processing new chunk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Processing chunk {chunk_id} with {len(indices)} frames (global frames {indices[0]}-{indices[-1]})")
    print(f"Progress: {indices[-1]}/{len(all_indices)} frames ({(indices[-1]/len(all_indices)*100):.1f}%)")

    # Initialize state for this chunk
    inference_state = active_predictor.init_state(video_path=chunk_dir)

    # Determine the seeding frame for prompts in this chunk
    seed_local_idx = 0
    seed_global_idx = local_to_global[seed_local_idx]
    seed_frame_path = os.path.join(video_dir, f"{seed_global_idx:06d}.jpg")
    print(f"\n--- Chunk {chunk_id} start (global frame {seed_global_idx}) ---")
    
    # Only prompt user for the first chunk of a new video
    if chunk_id == 1:
        print("Draw bounding boxes around objects in the first frame.")
        print("- Number keys 1..2 select object id (1:nrD, 2:nrV)")
        print("- Click and drag to draw bounding box")
        print("- u=undo, c=clear current object, enter/q=finish")
        print("- Use toolbar above for zoom/pan")
        print("- To skip this video, close the window without drawing any boxes")
        new_chunk_bboxes = interactive_collect_bboxes(seed_frame_path, object_ids=(1, 2))
        
        # Check if user wants to skip this video (no bounding boxes provided)
        if not new_chunk_bboxes:
            print(f"No bounding boxes provided. Skipping video: {os.path.basename(video_dir)}")
            # Cleanup chunk dirs before skipping
            try:
                if os.path.exists(chunk_root):
                    shutil.rmtree(chunk_root)
            except Exception:
                pass
            continue  # Skip to next video
    else:
        new_chunk_bboxes = None

    if new_chunk_bboxes:
        # Use freshly provided bounding boxes
        for obj_id, bbox in sorted(new_chunk_bboxes.items()):
            print(f"[chunk {chunk_id}] User seeding obj {obj_id} with bbox {bbox}")
            active_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=seed_local_idx,
                obj_id=int(obj_id),
                box=bbox,
            )
    else:
        # Attempt auto-seeding from previously computed masks if available
        auto_seeded = False
        
        # Try to find the most recent frame with masks for auto-seeding
        search_frames = sorted([f for f in video_segments.keys() if f < seed_global_idx], reverse=True)
        
        for search_frame in search_frames[:5]:  # Check last 5 frames with masks
            if search_frame in video_segments and video_segments[search_frame]:
                prev_masks = video_segments[search_frame]
                img_path = resolve_image_path(chunk_dir, seed_local_idx)
                img = cv2.imread(img_path)
                img_shape = img.shape if img is not None else (110, 110, 3)
                auto_bboxes = build_bbox_from_existing_masks(prev_masks, img_shape) # type: ignore
                
                if auto_bboxes:
                    print(f"[chunk {chunk_id}] Auto-seeding from frame {search_frame} masks")
                    for obj_id, bbox in sorted(auto_bboxes.items()):
                        print(f"[chunk {chunk_id}] Auto-seeding obj {obj_id} with bbox from frame {search_frame}")
                        active_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=seed_local_idx,
                            obj_id=int(obj_id),
                            box=bbox,
                        )
                    auto_seeded = True
                    break
        
        if not auto_seeded:
            print(f"[chunk {chunk_id}] No previous masks found for auto-seeding")
            print("Please provide bounding boxes for this chunk:")
            print("- Number keys 1..2 select object id (1:nrD, 2:nrV)")
            print("- Click and drag to draw bounding box")
            print("- u=undo, c=clear current object, enter/q=finish")
            print("- Use toolbar above for zoom/pan")
            
            manual_bboxes = interactive_collect_bboxes(seed_frame_path, object_ids=(1, 2))
            
            if manual_bboxes:
                for obj_id, bbox in sorted(manual_bboxes.items()):
                    print(f"[chunk {chunk_id}] Manual seeding obj {obj_id} with bbox {bbox}")
                    active_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=seed_local_idx,
                        obj_id=int(obj_id),
                        box=bbox,
                    )
            else:
                print(f"[chunk {chunk_id}] No bounding boxes provided. Skipping chunk.")
                continue  # Skip to next chunk

    # Propagate within the chunk starting from the seed frame
    last_quality_check = seed_global_idx
    restart_propagation = True
    
    while restart_propagation:
        restart_propagation = False
        
        for out_local_idx, out_obj_ids, out_mask_logits in active_predictor.propagate_in_video(
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
                    current_frame_path = os.path.join(video_dir, f"{global_idx:06d}.jpg")
                    print("Please draw new bounding boxes to improve tracking quality.")
                    print("- Number keys 1..2 select object id (1:nrD, 2:nrV)")
                    print("- Click and drag to draw bounding box")
                    print("- u=undo, c=clear current object, enter/q=finish")
                    print("- Use toolbar above for zoom/pan")
                    
                    new_bboxes = interactive_collect_bboxes(current_frame_path, object_ids=(1, 2))
                    
                    if new_bboxes:
                        # Convert global frame index to local frame index for this chunk
                        local_frame_idx = out_local_idx
                        
                        # Reset the inference state to replace existing tracking
                        print(f"Resetting tracking state and replacing with new bounding boxes at frame {global_idx}")
                        inference_state = active_predictor.init_state(video_path=chunk_dir)
                        
                        # Add new bounding boxes to replace the existing tracking
                        for obj_id, bbox in sorted(new_bboxes.items()):
                            print(f"Replacing tracking with new bbox for obj {obj_id} at frame {global_idx}")
                            active_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=local_frame_idx,
                                obj_id=int(obj_id),
                                box=bbox,
                            )
                        
                        print("Tracking state reset and new bounding boxes added. Restarting propagation...")
                        
                        # Update seed frame to current frame and restart propagation
                        seed_local_idx = local_frame_idx
                        restart_propagation = True
                        break
                    else:
                        print("No new bounding boxes provided. Continuing with current tracking...")
                    
                    last_quality_check = global_idx
                else:
                    print(f"[quality] Frame {global_idx}: {quality_report['status']}")
                    if global_idx - last_quality_check >= QUALITY_CHECK_INTERVAL:
                        last_quality_check = global_idx

            # Periodic visualization every VIS_INTERVAL frames (relative to first frame)
            if ((global_idx - first_frame_idx) % VIS_INTERVAL) == 0:
                try:
                    overlay_predictions_on_frame(video_dir, global_idx, video_segments, alpha=0.6)
                    print(f"[viz] Saved overlay for frame {global_idx:06d}")
                    
                    # Show confidence scores in visualization
                    if confidence_scores:
                        conf_str = ", ".join([f"obj{oid}:{conf:.2f}" for oid, conf in confidence_scores.items()])
                        print(f"[confidence] {conf_str}")
                        
                except Exception as e:
                    print(f"[viz] Failed to save overlay for frame {global_idx}: {e}")

    prev_last_frame_idx = indices[-1]
    
    # Clear inference state to free GPU memory
    del inference_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Advance to next non-overlapping position
    i = end_idx_exclusive

# Cleanup chunk dirs
try:
    shutil.rmtree(chunk_root)
except Exception:
    pass

analyze_and_print_results(video_segments)

# Generate unique output video path based on the input folder name
video_folder_name = os.path.basename(os.path.normpath(video_dir))
output_video_path = os.path.join(output_dir, f"{video_folder_name}_overlay.mp4")

# Make video with masks, if needed
create_mask_overlay_video(
    video_dir,
    frame_names,
    video_segments,
    output_video_path,
    fps=10,
    alpha=0.99
)

filtered_video_segments = save_video_segments_to_h5(video_segments, video_dir, output_dir, frame_mapping={})
    
print(f"Completed processing video {video_idx}/{len(unprocessed_videos)}: {os.path.basename(video_dir)}")

print(f"\nAll {len(unprocessed_videos)} videos have been processed successfully!")