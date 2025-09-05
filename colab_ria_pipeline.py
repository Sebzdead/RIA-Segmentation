"""
Google Colab-compatible RIA Segmentation Pipeline
This notebook performs SAM2 video segmentation with interactive prompts optimized for Colab environment.

Key modifications for Colab:
- Cloud storage integration (Google Drive/Cloud Storage)
- IPython widgets for user interaction
- Simplified prompting interface
- Batch processing capabilities
"""

# ============================================================================
# COLAB SETUP AND IMPORTS
# ============================================================================

# Mount Google Drive and install dependencies
try:
    from google.colab import drive, files
    import ipywidgets as widgets
    from IPython.display import display, clear_output, Image as IPImage
    IN_COLAB = True
    print("Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("Running locally")

# Standard imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import h5py
import json
import random
from PIL import Image
from tqdm import tqdm
import shutil
from pathlib import Path

# ============================================================================
# COLAB ENVIRONMENT SETUP
# ============================================================================

def setup_colab_environment():
    """Setup Google Colab environment with required packages and model downloads."""
    if not IN_COLAB:
        print("Not in Colab, skipping setup")
        return
    
    print("Setting up Google Colab environment...")
    
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Install required packages
    print("Installing required packages...")
    os.system('pip install segment-anything-2 tifffile h5py opencv-python matplotlib pandas scipy scikit-image ipywidgets')
    
    # Create working directory structure
    work_dir = '/content/ria_segmentation'
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    
    # Download SAM2 model if not exists
    model_dir = '/content/models'
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_path = f"{model_dir}/sam2.1_hiera_base_plus.pt"
    if not os.path.exists(checkpoint_path):
        print("Downloading SAM2 model...")
        os.system(f'wget -O {checkpoint_path} https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt')
    
    print("Colab environment setup complete!")
    return work_dir, model_dir

# ============================================================================
# CLOUD STORAGE CONFIGURATION
# ============================================================================

class CloudStorageConfig:
    """Configuration for cloud storage paths and operations."""
    
    def __init__(self, storage_type='drive'):
        self.storage_type = storage_type
        
        if storage_type == 'drive':
            # Google Drive paths
            self.base_path = '/content/drive/MyDrive/RIA_segmentation'
            self.input_videos_dir = f'{self.base_path}/input_videos'
            self.output_dir = f'{self.base_path}/output'
            self.temp_dir = '/content/temp_processing'
        elif storage_type == 'gcs':
            # Google Cloud Storage (implement if needed)
            self.bucket_name = 'your-bucket-name'
            self.base_path = f'gs://{self.bucket_name}/ria_segmentation'
        
        # Create directories
        os.makedirs(self.input_videos_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def list_input_videos(self):
        """List available input video directories."""
        if not os.path.exists(self.input_videos_dir):
            return []
        return [d for d in os.listdir(self.input_videos_dir) 
                if os.path.isdir(os.path.join(self.input_videos_dir, d))]
    
    def get_video_path(self, video_name):
        """Get full path to a video directory."""
        return os.path.join(self.input_videos_dir, video_name)
    
    def get_output_path(self, filename):
        """Get full path for output file."""
        return os.path.join(self.output_dir, filename)

# ============================================================================
# SAM2 MODEL SETUP
# ============================================================================

def setup_sam2_model(model_dir='/content/models'):
    """Initialize SAM2 model for Colab environment."""
    
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Import SAM2
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        print("Installing SAM2...")
        os.system('pip install git+https://github.com/facebookresearch/segment-anything-2.git')
        from sam2.build_sam import build_sam2_video_predictor
    
    # Model configuration
    checkpoint_path = f"{model_dir}/sam2.1_hiera_base_plus.pt"
    
    # Download config if needed
    config_dir = f"{model_dir}/configs"
    os.makedirs(config_dir, exist_ok=True)
    config_path = f"{config_dir}/sam2.1_hiera_b+.yaml"
    
    if not os.path.exists(config_path):
        # Create a basic config file
        config_content = """
model:
  _target_: sam2.modeling.sam2_base.SAM2Base
  trunk:
    _target_: sam2.modeling.backbones.hieradet.Hiera
    embed_dim: 112
    num_heads: [2, 4, 8, 16]
  sam_mask_decoder_extra_args:
    dynamic_multimask_via_stability: true
    dynamic_multimask_stability_delta: 0.05
    dynamic_multimask_stability_thresh: 0.98
  image_encoder:
    _target_: sam2.modeling.backbones.image_encoder.ImageEncoder
    trunk:
      _target_: sam2.modeling.backbones.hieradet.Hiera
      embed_dim: 112
      num_heads: [2, 4, 8, 16]
"""
        with open(config_path, 'w') as f:
            f.write(config_content)
    
    # Build predictor
    try:
        predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
        print("SAM2 model loaded successfully!")
        return predictor, device
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        print("Falling back to simplified initialization...")
        # Simplified fallback - you might need to adjust this based on your SAM2 version
        return None, device

# ============================================================================
# COLAB-COMPATIBLE INTERACTIVE WIDGETS
# ============================================================================

class ColabBboxCollector:
    """Interactive bounding box collection widget for Colab."""
    
    def __init__(self, image_path, object_ids=None):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.object_ids = object_ids or [1, 2]  # nrD and nrV
        self.bboxes = {}
        self.current_obj = self.object_ids[0]
        
        # Object names
        self.object_names = {1: 'nrD', 2: 'nrV'}
        
        # Setup widgets
        self.setup_widgets()
    
    def setup_widgets(self):
        """Create IPython widgets for bbox collection."""
        
        # Object selector
        self.obj_selector = widgets.Dropdown(
            options=[(self.object_names.get(obj_id, f'Object {obj_id}'), obj_id) 
                    for obj_id in self.object_ids],
            value=self.current_obj,
            description='Object:',
            style={'description_width': 'initial'}
        )
        
        # Coordinate input widgets
        self.x1_input = widgets.IntText(value=0, description='X1:', style={'description_width': 'initial'})
        self.y1_input = widgets.IntText(value=0, description='Y1:', style={'description_width': 'initial'})
        self.x2_input = widgets.IntText(value=100, description='X2:', style={'description_width': 'initial'})
        self.y2_input = widgets.IntText(value=100, description='Y2:', style={'description_width': 'initial'})
        
        # Buttons
        self.save_btn = widgets.Button(description='Save Bbox', button_style='success')
        self.clear_btn = widgets.Button(description='Clear All', button_style='warning')
        self.finish_btn = widgets.Button(description='Finish', button_style='primary')
        self.skip_btn = widgets.Button(description='Skip Video', button_style='danger')
        
        # Status output
        self.output = widgets.Output()
        
        # Event handlers
        self.obj_selector.observe(self.on_object_change, names='value')
        self.save_btn.on_click(self.save_bbox)
        self.clear_btn.on_click(self.clear_all)
        self.finish_btn.on_click(self.finish)
        self.skip_btn.on_click(self.skip)
        
        # Layout
        coord_box = widgets.HBox([
            self.x1_input, self.y1_input, self.x2_input, self.y2_input
        ])
        
        button_box = widgets.HBox([
            self.save_btn, self.clear_btn, self.finish_btn, self.skip_btn
        ])
        
        self.widget_box = widgets.VBox([
            widgets.HTML("<h3>Bounding Box Collection</h3>"),
            widgets.HTML("<p>Instructions: Set coordinates for bounding boxes around objects. "
                        "Check the image below for reference.</p>"),
            self.obj_selector,
            coord_box,
            button_box,
            self.output
        ])
        
        self.finished = False
        self.skipped = False
    
    def on_object_change(self, change):
        """Handle object selection change."""
        self.current_obj = change['new']
        with self.output:
            clear_output(wait=True)
            print(f"Selected object: {self.object_names.get(self.current_obj, self.current_obj)}")
    
    def save_bbox(self, btn):
        """Save current bounding box."""
        x1, y1, x2, y2 = self.x1_input.value, self.y1_input.value, self.x2_input.value, self.y2_input.value
        
        # Validate coordinates
        if x1 >= x2 or y1 >= y2:
            with self.output:
                clear_output(wait=True)
                print("Error: Invalid coordinates. Ensure x1 < x2 and y1 < y2")
            return
        
        # Save bbox
        self.bboxes[self.current_obj] = np.array([x1, y1, x2, y2], dtype=np.float32)
        
        with self.output:
            clear_output(wait=True)
            print(f"Saved bbox for {self.object_names.get(self.current_obj, self.current_obj)}: [{x1}, {y1}, {x2}, {y2}]")
            print(f"Current bboxes: {len(self.bboxes)}")
    
    def clear_all(self, btn):
        """Clear all bounding boxes."""
        self.bboxes = {}
        with self.output:
            clear_output(wait=True)
            print("Cleared all bounding boxes")
    
    def finish(self, btn):
        """Finish bbox collection."""
        self.finished = True
        with self.output:
            clear_output(wait=True)
            print(f"Finished! Collected {len(self.bboxes)} bounding boxes")
    
    def skip(self, btn):
        """Skip this video."""
        self.skipped = True
        with self.output:
            clear_output(wait=True)
            print("Skipping this video")
    
    def show_image(self):
        """Display the image for reference."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.image)
        ax.set_title("Reference Image - Note coordinates for bounding boxes")
        
        # Add coordinate grid for reference
        height, width = self.image.size[1], self.image.size[0]
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add existing bboxes
        for obj_id, bbox in self.bboxes.items():
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, 
                               edgecolor='red' if obj_id == 1 else 'blue',
                               facecolor='none',
                               label=self.object_names.get(obj_id, f'Object {obj_id}'))
            ax.add_patch(rect)
        
        if self.bboxes:
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def collect_bboxes(self):
        """Main method to collect bounding boxes."""
        # Show image first
        self.show_image()
        
        # Display widgets
        display(self.widget_box)
        
        # Wait for user to finish
        print("Use the widgets above to set bounding box coordinates.")
        print("Check the image for reference coordinates.")
        
        return self.bboxes

class ColabPointCollector:
    """Interactive point collection widget for Colab (fallback option)."""
    
    def __init__(self, image_path, object_ids=None):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.object_ids = object_ids or [1, 2]
        self.points = {obj_id: {'coords': [], 'labels': []} for obj_id in self.object_ids}
        self.current_obj = self.object_ids[0]
        
        # Object names
        self.object_names = {1: 'nrD', 2: 'nrV'}
        
        self.setup_widgets()
    
    def setup_widgets(self):
        """Setup point collection widgets."""
        
        # Object selector
        self.obj_selector = widgets.Dropdown(
            options=[(self.object_names.get(obj_id, f'Object {obj_id}'), obj_id) 
                    for obj_id in self.object_ids],
            value=self.current_obj,
            description='Object:'
        )
        
        # Point input
        self.x_input = widgets.IntText(value=0, description='X:')
        self.y_input = widgets.IntText(value=0, description='Y:')
        
        # Point type
        self.point_type = widgets.RadioButtons(
            options=[('Positive', 1), ('Negative', 0)],
            value=1,
            description='Point Type:'
        )
        
        # Buttons
        self.add_btn = widgets.Button(description='Add Point', button_style='success')
        self.clear_btn = widgets.Button(description='Clear Object', button_style='warning')
        self.finish_btn = widgets.Button(description='Finish', button_style='primary')
        
        # Output
        self.output = widgets.Output()
        
        # Event handlers
        self.obj_selector.observe(self.on_object_change, names='value')
        self.add_btn.on_click(self.add_point)
        self.clear_btn.on_click(self.clear_object)
        self.finish_btn.on_click(self.finish)
        
        # Layout
        coord_box = widgets.HBox([self.x_input, self.y_input])
        button_box = widgets.HBox([self.add_btn, self.clear_btn, self.finish_btn])
        
        self.widget_box = widgets.VBox([
            widgets.HTML("<h3>Point Collection (Fallback)</h3>"),
            self.obj_selector,
            coord_box,
            self.point_type,
            button_box,
            self.output
        ])
        
        self.finished = False
    
    def on_object_change(self, change):
        self.current_obj = change['new']
    
    def add_point(self, btn):
        x, y = self.x_input.value, self.y_input.value
        label = self.point_type.value
        
        self.points[self.current_obj]['coords'].append([x, y])
        self.points[self.current_obj]['labels'].append(label)
        
        with self.output:
            clear_output(wait=True)
            point_type = "positive" if label == 1 else "negative"
            print(f"Added {point_type} point at ({x}, {y}) for {self.object_names.get(self.current_obj)}")
            
            # Show summary
            for obj_id in self.object_ids:
                n_points = len(self.points[obj_id]['coords'])
                print(f"{self.object_names.get(obj_id)}: {n_points} points")
    
    def clear_object(self, btn):
        self.points[self.current_obj] = {'coords': [], 'labels': []}
        with self.output:
            clear_output(wait=True)
            print(f"Cleared points for {self.object_names.get(self.current_obj)}")
    
    def finish(self, btn):
        self.finished = True
        with self.output:
            clear_output(wait=True)
            print("Finished point collection!")
    
    def collect_points(self):
        """Main method to collect points."""
        # Show image
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.image)
        ax.set_title("Reference Image - Note coordinates for points")
        plt.show()
        
        # Display widgets
        display(self.widget_box)
        
        # Convert to numpy arrays
        result = {}
        for obj_id in self.object_ids:
            if len(self.points[obj_id]['coords']) > 0:
                coords = np.array(self.points[obj_id]['coords'], dtype=np.float32)
                labels = np.array(self.points[obj_id]['labels'], dtype=np.int32)
                result[obj_id] = (coords, labels)
        
        return result

# ============================================================================
# VIDEO PROCESSING FUNCTIONS
# ============================================================================

def get_video_frames(video_dir):
    """Get sorted list of frame files from video directory."""
    frame_files = [f for f in os.listdir(video_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    frame_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return frame_files

def process_video_with_sam2(video_dir, predictor, prompts, chunk_size=200):
    """Process video with SAM2 using provided prompts."""
    
    # Initialize inference state
    inference_state = predictor.init_state(video_path=video_dir)
    
    # Add prompts to first frame
    frame_idx = 0
    for obj_id, prompt_data in prompts.items():
        if 'bbox' in prompt_data:
            # Bounding box prompt
            bbox = prompt_data['bbox']
            predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=None,
                labels=None,
                clear_old_points=True,
                bbox=bbox
            )
        elif 'points' in prompt_data:
            # Point prompts
            points, labels = prompt_data['points']
            predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                clear_old_points=True
            )
    
    # Propagate masks through video
    video_segments = {}
    
    print("Propagating masks through video...")
    for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(
        predictor.propagate_in_video(inference_state), 
        desc="Processing frames"
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    return video_segments

def save_results_h5(video_segments, output_path):
    """Save segmentation results to HDF5 file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['num_frames'] = len(video_segments)
        if video_segments:
            first_frame = list(video_segments.keys())[0]
            f.attrs['object_ids'] = list(video_segments[first_frame].keys())
        
        # Create groups
        masks_group = f.create_group('masks')
        
        # Save masks
        sorted_frames = sorted(video_segments.keys())
        for frame_idx in sorted_frames:
            frame_group = masks_group.create_group(f'frame_{frame_idx:06d}')
            
            for obj_id, mask in video_segments[frame_idx].items():
                frame_group.create_dataset(
                    f'object_{obj_id}', 
                    data=mask.astype(bool),
                    compression='gzip'
                )
    
    print(f"Results saved to {output_path}")

def create_preview_video(video_dir, video_segments, output_path, fps=10):
    """Create preview video with mask overlays."""
    
    frame_files = get_video_frames(video_dir)
    if not frame_files:
        print("No frames found for preview")
        return
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(video_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width = first_frame.shape[:2]
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Colors for objects
    colors = {1: (255, 0, 0), 2: (0, 0, 255)}  # Red for nrD, Blue for nrV
    
    print("Creating preview video...")
    for frame_idx, frame_file in enumerate(tqdm(frame_files)):
        frame_path = os.path.join(video_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        # Add mask overlays if available
        if frame_idx in video_segments:
            overlay = np.zeros_like(frame)
            
            for obj_id, mask in video_segments[frame_idx].items():
                if obj_id in colors:
                    color = colors[obj_id]
                    mask_3d = np.stack([mask.squeeze()] * 3, axis=-1)
                    colored_mask = mask_3d * np.array(color)
                    overlay = cv2.addWeighted(overlay, 1, colored_mask.astype(np.uint8), 0.6, 0)
            
            frame = cv2.addWeighted(frame, 1, overlay, 0.4, 0)
        
        out.write(frame)
    
    out.release()
    print(f"Preview video saved to {output_path}")

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_single_video(video_name, storage_config, predictor, use_bbox=True):
    """Process a single video with interactive prompting."""
    
    print(f"\n{'='*60}")
    print(f"Processing video: {video_name}")
    print(f"{'='*60}")
    
    video_dir = storage_config.get_video_path(video_name)
    
    # Get frame files
    frame_files = get_video_frames(video_dir)
    if not frame_files:
        print(f"No frames found in {video_dir}")
        return None
    
    print(f"Found {len(frame_files)} frames")
    
    # Get first frame for prompting
    first_frame_path = os.path.join(video_dir, frame_files[0])
    
    # Collect prompts
    prompts = {}
    
    if use_bbox:
        print("\nCollecting bounding box prompts...")
        bbox_collector = ColabBboxCollector(first_frame_path, object_ids=[1, 2])
        user_bboxes = bbox_collector.collect_bboxes()
        
        # Wait for user to finish
        while not bbox_collector.finished and not bbox_collector.skipped:
            import time
            time.sleep(1)
        
        if bbox_collector.skipped:
            print("Video skipped by user")
            return None
        
        # Convert bboxes to prompts
        for obj_id, bbox in user_bboxes.items():
            prompts[obj_id] = {'bbox': bbox}
    
    else:
        print("\nCollecting point prompts...")
        point_collector = ColabPointCollector(first_frame_path, object_ids=[1, 2])
        user_points = point_collector.collect_points()
        
        while not point_collector.finished:
            import time
            time.sleep(1)
        
        # Convert points to prompts
        for obj_id, (points, labels) in user_points.items():
            prompts[obj_id] = {'points': (points, labels)}
    
    if not prompts:
        print("No prompts collected, skipping video")
        return None
    
    print(f"Collected prompts for {len(prompts)} objects")
    
    # Process video with SAM2
    try:
        video_segments = process_video_with_sam2(video_dir, predictor, prompts)
        
        # Save results
        output_h5_path = storage_config.get_output_path(f"{video_name}_segments.h5")
        save_results_h5(video_segments, output_h5_path)
        
        # Create preview video
        preview_path = storage_config.get_output_path(f"{video_name}_preview.mp4")
        create_preview_video(video_dir, video_segments, preview_path)
        
        print(f"✓ Successfully processed {video_name}")
        return {
            'video_name': video_name,
            'num_frames': len(video_segments),
            'num_objects': len(prompts),
            'h5_path': output_h5_path,
            'preview_path': preview_path
        }
        
    except Exception as e:
        print(f"✗ Error processing {video_name}: {str(e)}")
        return None

def main_pipeline():
    """Main processing pipeline for Colab environment."""
    
    print("🚀 Starting RIA Segmentation Pipeline for Google Colab")
    print("="*60)
    
    # Setup environment
    if IN_COLAB:
        work_dir, model_dir = setup_colab_environment()
    else:
        work_dir = os.getcwd()
        model_dir = './models'
    
    # Setup cloud storage
    storage_config = CloudStorageConfig('drive')
    
    # Setup SAM2 model
    print("\nInitializing SAM2 model...")
    predictor, device = setup_sam2_model(model_dir)
    
    if predictor is None:
        print("❌ Failed to initialize SAM2 model")
        return
    
    print(f"✓ SAM2 model loaded on {device}")
    
    # Get available videos
    available_videos = storage_config.list_input_videos()
    
    if not available_videos:
        print(f"\n❌ No videos found in {storage_config.input_videos_dir}")
        print("Please upload video directories to your Google Drive")
        return
    
    print(f"\n📹 Found {len(available_videos)} videos:")
    for i, video in enumerate(available_videos, 1):
        print(f"  {i}. {video}")
    
    # Process videos
    results = []
    
    # Create video selection widget
    video_selector = widgets.SelectMultiple(
        options=available_videos,
        description='Select videos to process:',
        style={'description_width': 'initial'}
    )
    
    process_btn = widgets.Button(description='Process Selected Videos', button_style='primary')
    output_widget = widgets.Output()
    
    def on_process_click(btn):
        with output_widget:
            clear_output(wait=True)
            selected_videos = list(video_selector.value)
            
            if not selected_videos:
                print("No videos selected")
                return
            
            print(f"Processing {len(selected_videos)} videos...")
            
            for i, video_name in enumerate(selected_videos, 1):
                print(f"\n--- Processing video {i}/{len(selected_videos)}: {video_name} ---")
                
                result = process_single_video(
                    video_name, 
                    storage_config, 
                    predictor, 
                    use_bbox=True  # Use bounding box prompts
                )
                
                if result:
                    results.append(result)
            
            # Summary
            print(f"\n🎉 Processing complete!")
            print(f"Successfully processed: {len(results)}/{len(selected_videos)} videos")
            
            for result in results:
                print(f"  ✓ {result['video_name']}: {result['num_frames']} frames, {result['num_objects']} objects")
    
    process_btn.on_click(on_process_click)
    
    # Display interface
    interface = widgets.VBox([
        widgets.HTML("<h2>Video Selection</h2>"),
        video_selector,
        process_btn,
        output_widget
    ])
    
    display(interface)
    
    print("\n📋 Instructions:")
    print("1. Select one or more videos from the list above")
    print("2. Click 'Process Selected Videos'")
    print("3. For each video, you'll be prompted to set bounding boxes")
    print("4. Results will be saved to your Google Drive")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def batch_process_all_videos(storage_config, predictor):
    """Batch process all videos with minimal interaction."""
    
    available_videos = storage_config.list_input_videos()
    results = []
    
    print(f"Batch processing {len(available_videos)} videos...")
    
    for video_name in tqdm(available_videos):
        # Check if already processed
        output_path = storage_config.get_output_path(f"{video_name}_segments.h5")
        if os.path.exists(output_path):
            print(f"Skipping {video_name} (already processed)")
            continue
        
        try:
            # Use automated prompting or simple center-based prompts
            result = process_single_video(video_name, storage_config, predictor)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
    
    return results

def download_results():
    """Download all results to local machine (if in Colab)."""
    if not IN_COLAB:
        print("Not in Colab, results are already local")
        return
    
    from google.colab import files
    import zipfile
    
    # Create zip file with all results
    output_dir = '/content/drive/MyDrive/RIA_segmentation/output'
    zip_path = '/content/results.zip'
    
    if os.path.exists(output_dir):
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, output_dir))
        
        files.download(zip_path)
        print("Results downloaded!")
    else:
        print("No results found to download")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__" or IN_COLAB:
    print("🔬 RIA Segmentation Pipeline - Google Colab Edition")
    print("Run main_pipeline() to start processing")
    
    # Auto-run in Colab
    if IN_COLAB:
        main_pipeline()
