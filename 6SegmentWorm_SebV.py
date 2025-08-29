import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import h5py
from tqdm import tqdm
from matplotlib.backend_bases import MouseButton
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
sam2_checkpoint = r"c:\Users\switte3\Documents\sam2\checkpoints\sam2.1_hiera_large.pt"
model_cfg = r"c:\Users\switte3\Documents\sam2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

jpg_video_dir = 'JPG'
head_segmentation_dir = 'HEAD_SEGMENT'

# Add quality thresholds and processing constants
CONFIDENCE_THRESHOLD = 0.9  # minimum confidence score (0-1)
QUALITY_CHECK_INTERVAL = 50  # check quality every 50 frames
VIS_INTERVAL = 50       # visualize masks every 50 frames
CHUNK_SIZE = 300   # number of frames per chunk
OVERLAP = 1        # number of overlapping frames between consecutive chunks

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
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='x', s=marker_size, edgecolor='white', linewidth=1.25)   

def get_random_unprocessed_video(parent_dir, head_segmentation_dir):
    all_videos = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(head_segmentation_dir, video + "_headsegmentation.h5"))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(parent_dir, random.choice(unprocessed_videos))

def create_mask_video(image_dir, masks_dict, output_path, fps=10, alpha=0.99):
    """
    Create a video with predicted masks overlaid on the images.
    """
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (128, 0, 128),  # Purple
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (0, 128, 0),    # Dark Green
        (0, 128, 128),  # Teal
        (255, 128, 0),  # Orange
        (255, 0, 128),  # Deep Pink
        (128, 255, 0),  # Lime
        (255, 255, 0),  # Yellow
        (0, 255, 128)   # Spring Green
    ]

    def overlay_masks(image, frame_masks, mask_colors, alpha):
        overlay = np.zeros_like(image)
        
        for mask_id, mask in frame_masks.items():
            if mask.dtype != bool:
                mask = mask > 0.5
            
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            mask_resized = cv2.resize(mask.astype(np.uint8), 
                                    (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)            
            colored_mask = np.zeros_like(image)
            colored_mask[mask_resized == 1] = mask_colors[mask_id]            
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)
        
        return cv2.addWeighted(image, 1, overlay, alpha, 0)

    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {image_files[0]}")
    
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_mask_ids = set()
    for masks in masks_dict.values():
        all_mask_ids.update(masks.keys())
    mask_colors = {mask_id: COLORS[i % len(COLORS)] 
                  for i, mask_id in enumerate(all_mask_ids)}

    for frame_idx, image_file in enumerate(image_files):
        try:
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_file}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_idx in masks_dict:
                frame = overlay_masks(frame, masks_dict[frame_idx], 
                                   mask_colors, alpha)

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            continue

    out.release()
    print(f"Video saved to {output_path}")

def save_cleaned_segments_to_h5(cleaned_segments, filename):
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"{head_segmentation_dir}/{name_without_ext}_headsegmentation.h5"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with h5py.File(output_filename, 'w') as f:
        num_frames = len(cleaned_segments)
        f.attrs['num_frames'] = num_frames
        f.attrs['object_ids'] = list(cleaned_segments[0].keys())

        masks_group = f.create_group('masks')

        first_frame = list(cleaned_segments.keys())[0]
        first_obj = list(cleaned_segments[first_frame].keys())[0]
        mask_shape = cleaned_segments[first_frame][first_obj].shape

        for obj_id in cleaned_segments[first_frame].keys():
            masks_group.create_dataset(str(obj_id), (num_frames, *mask_shape), dtype=np.uint8)

        sorted_frames = sorted(cleaned_segments.keys())
        
        for idx, frame in enumerate(sorted_frames):
            frame_data = cleaned_segments[frame]
            for obj_id, mask in frame_data.items():
                masks_group[str(obj_id)][idx] = mask.astype(np.uint8) * 255
            print(f"Saving frame {frame} at index {idx}")

    print(f"Cleaned segments saved to {output_filename}")
    return output_filename

def load_cleaned_segments_from_h5(filename):
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        masks_group = f['masks']
        
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in object_ids:
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                frame_data[obj_id] = mask            
            cleaned_segments[frame_idx] = frame_data
            print(f"Loading frame {frame_idx}")
    
    print(f"Cleaned segments loaded from {filename}")
    return cleaned_segments

def compare_cleaned_segments(original, loaded):
    """
    Check the segments were properly saved.
    """
    assert len(original) == len(loaded), "Number of frames doesn't match"
    original_frames = sorted(original.keys())
    loaded_frames = sorted(loaded.keys())
    
    for orig_frame, loaded_frame in zip(original_frames, loaded_frames):
        assert original[orig_frame].keys() == loaded[loaded_frame].keys(), f"Object IDs don't match in frame {orig_frame}"
        
        for obj_id in original[orig_frame]:
            original_mask = original[orig_frame][obj_id]
            loaded_mask = loaded[loaded_frame][obj_id]
            
            if not np.array_equal(original_mask, loaded_mask):
                print(f"Mismatch found in original frame {orig_frame}, loaded frame {loaded_frame}, object {obj_id}")
                print(f"Original mask shape: {original_mask.shape}")
                print(f"Loaded mask shape: {loaded_mask.shape}")
                print(f"Original mask dtype: {original_mask.dtype}")
                print(f"Loaded mask dtype: {loaded_mask.dtype}")
                print(f"Number of True values in original: {np.sum(original_mask)}")
                print(f"Number of True values in loaded: {np.sum(loaded_mask)}")
                
                diff_positions = np.where(original_mask != loaded_mask)
                print(f"Number of differing positions: {len(diff_positions[0])}")
                
                if len(diff_positions[0]) > 0:
                    print("First 5 differing positions:")
                    for i in range(min(5, len(diff_positions[0]))):
                        pos = tuple(dim[i] for dim in diff_positions)
                        print(f"  Position {pos}: Original = {original_mask[pos]}, Loaded = {loaded_mask[pos]}")
                
                return False
    
    print("All masks match exactly!")
    return True

def interactive_collect_points(image_path, object_ids=None):
    """
    Collect point prompts on a single image interactively.

    Controls:
    - Number keys 1..9: select object id
    - Left click: add positive point for current object
    - Right click: add negative point for current object
    - u: undo last point for current object
    - c: clear current object's points
    - enter/q: finish and return
    - Use matplotlib toolbar for zoom/pan

    Returns:
        dict[int, tuple[np.ndarray, np.ndarray]] mapping obj_id -> (points, labels)
    """
    if object_ids is None:
        object_ids = (2,)  # Default worm body object
    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Enable the navigation toolbar for zoom/pan functionality
    fig.canvas.toolbar_visible = True
    
    ax.imshow(img)
    ax.set_axis_off()

    # per-object storage
    storage = {oid: {"points": [], "labels": []} for oid in object_ids}
    current_obj = object_ids[0]
    
    # Define specific colors for each object
    object_colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'purple', 6: 'orange', 7: 'cyan', 8: 'magenta', 9: 'brown'}

    def redraw():
        ax.clear()
        ax.imshow(img)
        ax.set_axis_off()
        
        # Draw existing points
        for oid in sorted(storage.keys()):
            points = storage[oid]["points"]
            labels = storage[oid]["labels"]
            if not points:
                continue
            
            points_array = np.array(points)
            labels_array = np.array(labels)
            
            # Use consistent colors for each object
            color = object_colors.get(oid, 'gray')
            
            # Draw positive points (green stars)
            pos_points = points_array[labels_array == 1]
            if len(pos_points) > 0:
                ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                          s=100, edgecolor=color, linewidth=2, label=f"obj {oid} +")
            
            # Draw negative points (red X marks)
            neg_points = points_array[labels_array == 0]
            if len(neg_points) > 0:
                ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='x', 
                          s=100, edgecolor=color, linewidth=2, label=f"obj {oid} -")
        
        # Display per-object counts of points
        obj_lines = []
        total = 0
        for oid in sorted(storage.keys()):
            points = storage[oid]["points"]
            labels = storage[oid]["labels"]
            pos_count = sum(1 for l in labels if l == 1)
            neg_count = sum(1 for l in labels if l == 0)
            total += len(points)
            obj_lines.append(f"{oid}: +{pos_count}/-{neg_count}")
        obj_list = ",  ".join(obj_lines)
        
        ax.text(0.01, 0.99,
            f"Active obj={current_obj} | Total points: {total}\n"
            f"Left click=positive (green *), Right click=negative (red x), u=undo, c=clear, enter/q=finish\n"
            f"Use toolbar above for zoom/pan. Objects: {obj_list}",
            transform=ax.transAxes, va='top', ha='left', color='white',
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        if event.button == MouseButton.LEFT:
            # Positive point
            storage[current_obj]["points"].append([event.xdata, event.ydata])
            storage[current_obj]["labels"].append(1)
            redraw()
        elif event.button == MouseButton.RIGHT:
            # Negative point
            storage[current_obj]["points"].append([event.xdata, event.ydata])
            storage[current_obj]["labels"].append(0)
            redraw()

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
            points = storage[current_obj]["points"]
            labels = storage[current_obj]["labels"]
            if points:
                points.pop()
                labels.pop()
            redraw()
            return
        if key == 'c':
            storage[current_obj] = {"points": [], "labels": []}
            redraw()
            return
        if key in ('enter', 'return', 'q'):
            plt.close(fig)
            return

    cid1 = fig.canvas.mpl_connect('button_press_event', on_click)
    cid2 = fig.canvas.mpl_connect('key_press_event', on_key)
    redraw()
    plt.show()
    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)

    # convert to numpy arrays and return
    out = {}
    for oid, d in storage.items():
        if len(d["points"]) == 0:
            continue
        points = np.array(d["points"], dtype=np.float32)
        labels = np.array(d["labels"], dtype=np.int32)
        out[oid] = (points, labels)
    return out

def check_mask_quality(video_segments, frame_idx, confidence_scores=None):
    """
    Check if masks meet confidence quality threshold.
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
        1: (255, 0, 0),     # Red
        2: (0, 0, 255),     # Blue  
        3: (0, 255, 0),     # Green
        4: (255, 255, 0),   # Yellow
        5: (255, 0, 255),   # Magenta
    }

    frame_names = [f for f in os.listdir(video_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    if frame_idx >= len(frame_names):
        print(f"Frame {frame_idx} not found")
        return
        
    frame_path = os.path.join(video_dir, frame_names[frame_idx])
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
    ax.set_title(f"Frame {frame_idx:06d} - Worm Segmentation")
    ax.set_axis_off()
    
    # Add mask legend if masks exist
    if frame_idx in video_segments:
        legend_text = []
        for mask_id in sorted(video_segments[frame_idx].keys()):
            legend_text.append(f"{mask_id}: Worm Body")
        
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

def build_points_from_existing_masks(masks_for_frame, img_shape, n_points=3):
    """
    Create point prompts from existing masks for auto-prompting.
    
    Args:
        masks_for_frame (dict): Dictionary mapping obj_id -> mask array
        img_shape (tuple): Shape of the image (height, width, channels)
        n_points (int): Number of positive points to sample per mask
    
    Returns:
        dict: Dictionary mapping obj_id -> (points, labels)
    """
    prompts = {}
    
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
            
        # Scale to image dimensions if needed
        mask_h, mask_w = mask.shape
        img_h, img_w = img_shape[:2]
        
        if mask_h != img_h or mask_w != img_w:
            x_coords = x_coords / max(mask_w - 1, 1) * (img_w - 1)
            y_coords = y_coords / max(mask_h - 1, 1) * (img_h - 1)
        
        # Sample n_points positive points from the mask
        if len(x_coords) >= n_points:
            indices = np.random.choice(len(x_coords), n_points, replace=False)
        else:
            indices = np.arange(len(x_coords))
            
        points = np.column_stack([x_coords[indices], y_coords[indices]]).astype(np.float32)
        labels = np.ones(len(points), dtype=np.int32)  # All positive points
        
        prompts[int(obj_id)] = (points, labels)
    
    return prompts

def get_sorted_frame_indices(names):
    return [int(os.path.splitext(n)[0]) for n in names]

def create_chunk_dir(base_dir, indices, chunk_root, chunk_id):
    """Create a temporary chunk directory with locally renumbered frames."""
    import shutil
    os.makedirs(chunk_root, exist_ok=True)
    chunk_dir = os.path.join(chunk_root, f"chunk_{chunk_id:04d}")
    os.makedirs(chunk_dir, exist_ok=True)
    local_to_global = []
    for local_idx, global_idx in enumerate(indices):
        # Find the actual frame file
        frame_names = [f for f in os.listdir(base_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        
        if global_idx < len(frame_names):
            src = os.path.join(base_dir, frame_names[global_idx])
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

video_dir = get_random_unprocessed_video(jpg_video_dir, head_segmentation_dir)
print(f"Processing video: {video_dir}")

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Process video in chunks for large videos
all_indices = get_sorted_frame_indices(frame_names)
video_segments = {}

# Setup chunking directories
chunk_root = os.path.join(video_dir, ".tmp_chunks")
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
    chunk_dir, local_to_global = create_chunk_dir(video_dir, indices, chunk_root, chunk_id)

    # Initialize state for this chunk
    inference_state = predictor.init_state(video_path=chunk_dir)

    # Determine the seeding frame for prompts in this chunk
    seed_local_idx = 0
    seed_global_idx = local_to_global[seed_local_idx]
    seed_frame_path = os.path.join(video_dir, frame_names[seed_global_idx])
    print(f"\n--- Chunk {chunk_id} start (global frame {seed_global_idx}) ---")
    
    # Only prompt user for the first chunk
    if chunk_id == 1:
        print("\nPlace positive and negative point prompts on the worm.")
        print("- Number key 2 = select worm body object")
        print("- Left click = positive point")
        print("- Right click = negative point") 
        print("- u = undo last point")
        print("- c = clear all points")
        print("- enter/q = finish")
        print("- Use toolbar above for zoom/pan")

        user_prompts = interactive_collect_points(seed_frame_path, object_ids=(2,))

        if not user_prompts or 2 not in user_prompts:
            print("No point prompts provided for worm body. Skipping this video...")
            break

        ann_obj_id = 2  # Worm body object ID
        points, labels = user_prompts[2]
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=seed_local_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        #Visualize prompt
        plt.figure(figsize=(12, 8))
        plt.imshow(Image.open(seed_frame_path))
        show_points(points, labels, plt.gca())
        for i, out_obj_id in enumerate(out_obj_ids):
            show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
        plt.savefig("worm_prompt.png")
        plt.close()
    else:
        # Attempt auto-seeding from previously computed masks if available
        if seed_global_idx in video_segments:
            prev_masks = video_segments[seed_global_idx]
            img_path = os.path.join(chunk_dir, f"{seed_local_idx:06d}.jpg")
            img = cv2.imread(img_path)
            img_shape = img.shape if img is not None else (110, 110, 3)
            auto_prompts = build_points_from_existing_masks(prev_masks, img_shape)
            if auto_prompts:
                for obj_id, (points, labels) in sorted(auto_prompts.items()):
                    print(f"[chunk {chunk_id}] Auto-seeding obj {obj_id} with {len(points)} points from previous masks")
                    predictor.add_new_points(
                        inference_state=inference_state,
                        frame_idx=seed_local_idx,
                        obj_id=int(obj_id),
                        points=points,
                        labels=labels,
                    )
            else:
                print(f"[chunk {chunk_id}] No points derived from previous masks; proceeding without seeding")
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
                current_frame_path = os.path.join(video_dir, frame_names[global_idx])
                
                print("Please place new point prompts to improve tracking quality.")
                print("- Number key 2 = select worm body object")
                print("- Left click = positive point")
                print("- Right click = negative point")
                print("- u = undo last point")
                print("- c = clear all points")
                print("- enter/q = finish")
                
                new_prompts = interactive_collect_points(current_frame_path, object_ids=(2,))
                
                if new_prompts and 2 in new_prompts:
                    # Convert global frame index to local frame index for this chunk
                    local_frame_idx = out_local_idx
                    
                    # Add new points to the current frame
                    points, labels = new_prompts[2]
                    print(f"Adding correction points at frame {global_idx}")
                    predictor.add_new_points(
                        inference_state=inference_state,
                        frame_idx=local_frame_idx,
                        obj_id=2,
                        points=points,
                        labels=labels,
                    )
                    print("Points added. Continuing propagation...")
                else:
                    print("No new points provided. Continuing with current tracking...")
                
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

    # Advance to next non-overlapping position
    i = end_idx_exclusive

# Cleanup chunk dirs
try:
    import shutil
    shutil.rmtree(chunk_root)
except Exception:
    pass

# Only process if we have video segments (user didn't skip)
if video_segments:
    #Error flagging
    all_ok = True
    for frame_idx, frame_masks in video_segments.items():
        for obj_id in [2]:
            if obj_id not in frame_masks:
                print(f"Warning: Object {obj_id} missing in frame {frame_idx}")
                all_ok = False
            elif not np.any(frame_masks[obj_id]):
                print(f"Warning: Mask for object {obj_id} is empty in frame {frame_idx}")
                all_ok = False
    if all_ok:
        print("All frames contain valid masks for object 2")

    # Create output video with proper filename
    video_name = os.path.basename(video_dir)
    output_video_path = f"{video_name}_worm_segmentation.mp4"
    create_mask_video(video_dir, video_segments, output_video_path, fps=10, alpha=0.98)
    output_filename = save_cleaned_segments_to_h5(video_segments, video_dir)
    loaded_segments = load_cleaned_segments_from_h5(output_filename)

print(f"Completed processing video: {os.path.basename(video_dir)}")