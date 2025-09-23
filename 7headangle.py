"""
Alternative head angle extraction script with improved skeletonization and angle calculation.
Uses medial-axis skeletonization and calculates angles based on three-point method.
"""
import os
import numpy as np
import h5py
from skimage import morphology
from scipy import ndimage
import pandas as pd
import random
import re
import cv2
from collections import deque, defaultdict

head_segmentation_dir = "7HEAD_SEGMENT/MMH223"
csv_input_dir = "6ANALYSIS/MMH223"
final_data_dir = "8HEAD_ANGLE/MMH223"

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
            # print(f"Loading frame {frame_idx}")
    
    print(f"Cleaned segments loaded from {filename}")
    return cleaned_segments

def get_all_unprocessed_videos(head_segmentation_dir, csv_input_dir, final_data_dir):
    all_videos = [f for f in os.listdir(head_segmentation_dir) if f.endswith("_headsegmentation.h5")]
    
    processable_videos = []
    for video in all_videos:
        base_name = video.replace("_headsegmentation.h5", "")
        input_csv_name = base_name + "_crop.csv"
        output_csv_name = base_name + "_crop_headangles.csv"
        
        if os.path.exists(os.path.join(csv_input_dir, input_csv_name)) and \
           not os.path.exists(os.path.join(final_data_dir, output_csv_name)):
            processable_videos.append(os.path.join(head_segmentation_dir, video))
    
    if not processable_videos:
        raise ValueError("No videos found that need head angle processing.")
    
    return processable_videos

def preprocess_mask(mask):
    """
    Preprocess mask to improve skeletonization by filling gaps and aggressive smoothing.
    """
    if mask.ndim > 2:
        mask = mask.squeeze()
    mask = mask.astype(bool)
    
    if not np.any(mask):
        return mask
    
    # Fill small holes more aggressively
    area_threshold = 500  # Increased from 100
    mask_filled = morphology.remove_small_holes(mask, area_threshold=area_threshold)
    
    # Multiple rounds of morphological operations for more aggressive smoothing
    kernel_radius = 5  # Increased from 3
    ksize = 2 * kernel_radius + 1
    ell_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    
    mask_u8 = mask_filled.astype(np.uint8)
    
    # First round: Close gaps
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, ell_kernel)
    
    # Second round: Opening to remove small protrusions
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ell_kernel)
    
    # Third round: Another closing with larger kernel
    larger_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    final_morph = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, larger_kernel)
    
    # Apply more aggressive Gaussian smoothing
    float_mask = final_morph.astype(np.float32)
    blurred = ndimage.gaussian_filter(float_mask, sigma=3.0)  # Increased from 1.0
    
    # Additional smoothing pass
    blurred = ndimage.gaussian_filter(blurred, sigma=2.0)
    
    smoothed_mask = blurred >= 0.4  # Lower threshold for more inclusive smoothing
    
    return smoothed_mask

def get_medial_axis_skeleton(mask):
    """
    Extract medial axis skeleton from mask using scikit-image medial_axis.
    """
    if mask.ndim > 2:
        mask = mask.squeeze()
    mask = mask.astype(bool)
    
    if not np.any(mask):
        print("Warning: Input mask is empty")
        return mask
    
    # Preprocess the mask
    processed_mask = preprocess_mask(mask)
    
    if not np.any(processed_mask):
        print("Warning: Preprocessed mask is empty, using original")
        processed_mask = mask
    
    # Extract medial axis
    skeleton = morphology.medial_axis(processed_mask)
    
    if np.sum(skeleton) == 0:
        print("Warning: Medial axis is empty, trying with original mask")
        skeleton = morphology.medial_axis(mask)
    
    # Remove side branches to get clean midline
    if np.sum(skeleton) > 0:
        skeleton = remove_skeleton_branches(skeleton)
    
    return skeleton

def build_skeleton_graph(skeleton):
    """
    Build an 8-connected graph from skeleton pixels.
    Returns adjacency list representation of the graph.
    """
    points = np.column_stack(np.where(skeleton))
    if len(points) == 0:
        return {}, {}
    
    # Create a mapping from coordinates to indices
    coord_to_idx = {tuple(point): idx for idx, point in enumerate(points)}
    
    # Build adjacency list
    graph = defaultdict(list)
    
    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for idx, point in enumerate(points):
        y, x = point
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            neighbor_coord = (ny, nx)
            
            if neighbor_coord in coord_to_idx:
                neighbor_idx = coord_to_idx[neighbor_coord]
                graph[idx].append(neighbor_idx)
    
    return graph, coord_to_idx

def find_endpoints(graph, points):
    """
    Find endpoints (nodes with degree 1) and potential endpoints (nodes with degree 2 at extremes).
    """
    endpoints = []
    
    # Find nodes with degree 1 (true endpoints)
    for node in graph:
        if len(graph[node]) <= 1:
            endpoints.append(node)
    
    # If no true endpoints, find the topmost and bottommost points
    if len(endpoints) < 2:
        y_coords = [points[i][0] for i in range(len(points))]
        top_idx = np.argmin(y_coords)
        bottom_idx = np.argmax(y_coords)
        
        endpoints = [top_idx, bottom_idx]
    
    return endpoints

def bfs_longest_path(graph, start_node, points):
    """
    Find the longest path from start_node using BFS.
    Returns the path and its length.
    """
    if start_node not in graph:
        return [], 0
    
    visited = set()
    queue = deque([(start_node, [start_node])])
    longest_path = []
    max_length = 0
    
    while queue:
        node, path = queue.popleft()
        
        if node in visited:
            continue
            
        visited.add(node)
        
        # Update longest path if current path is longer
        if len(path) > max_length:
            max_length = len(path)
            longest_path = path.copy()
        
        # Explore neighbors
        for neighbor in graph[node]:
            if neighbor not in visited:
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
    
    return longest_path, max_length

def find_longest_geodesic_path(graph, endpoints, points):
    """
    Find the longest geodesic path between any two endpoints.
    """
    if len(endpoints) < 2:
        return []
    
    best_path = []
    max_length = 0
    
    # Try all pairs of endpoints
    for i, start in enumerate(endpoints):
        for j, end in enumerate(endpoints):
            if i >= j:
                continue
                
            # BFS from start to find path to end
            visited = set()
            queue = deque([(start, [start])])
            
            while queue:
                node, path = queue.popleft()
                
                if node in visited:
                    continue
                    
                visited.add(node)
                
                if node == end:
                    if len(path) > max_length:
                        max_length = len(path)
                        best_path = path.copy()
                    break
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
    
    # If no path found between endpoints, use longest path from any endpoint
    if not best_path:
        for endpoint in endpoints:
            path, length = bfs_longest_path(graph, endpoint, points)
            if length > max_length:
                max_length = length
                best_path = path
    
    return best_path

def remove_skeleton_branches(skeleton):
    """
    Remove side branches from skeleton by keeping only the longest geodesic path.
    """
    if not np.any(skeleton):
        return skeleton
    
    # Build graph from skeleton
    graph, coord_to_idx = build_skeleton_graph(skeleton)
    points = np.column_stack(np.where(skeleton))
    
    if len(points) < 3:
        return skeleton
    
    # Find endpoints
    endpoints = find_endpoints(graph, points)
    
    # Find longest geodesic path
    longest_path = find_longest_geodesic_path(graph, endpoints, points)
    
    if not longest_path:
        return skeleton
    
    # Create new skeleton with only the longest path
    cleaned_skeleton = np.zeros_like(skeleton, dtype=bool)
    
    for node_idx in longest_path:
        if node_idx < len(points):
            y, x = points[node_idx]
            if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]:
                cleaned_skeleton[y, x] = True
    
    return cleaned_skeleton

def order_skeleton_points(skeleton):
    """
    Order skeleton points from top to bottom using a more efficient approach.
    """
    points = np.column_stack(np.where(skeleton))
    if len(points) == 0:
        return points
    
    # Sort by row (y-coordinate) to get top-to-bottom ordering
    points = points[np.argsort(points[:, 0])]
    
    # Always subsample large skeletons for efficiency
    if len(points) > 50:
        # Group points by y-coordinate and take representative points
        unique_y = np.unique(points[:, 0])
        step_size = max(1, len(unique_y) // 50)  # Aim for ~50 points
        sampled_points = []
        
        for y in unique_y[::step_size]:
            y_points = points[points[:, 0] == y]
            if len(y_points) == 1:
                sampled_points.append(y_points[0])
            else:
                # Take the median x-coordinate point for this y-level
                median_x = np.median(y_points[:, 1])
                closest_idx = np.argmin(np.abs(y_points[:, 1] - median_x))
                sampled_points.append(y_points[closest_idx])
        
        return np.array(sampled_points)
    else:
        # For small skeletons, return as-is (already sorted by y)
        return points

def calculate_three_point_angle(p1, p2, p3):
    """
    Calculate angle at point p2 formed by points p1-p2-p3.
    Returns angle in degrees.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate magnitudes
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    # Calculate cosine of angle
    cos_angle = np.dot(v1, v2) / (mag1 * mag2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # Return the acute angle (0-180 degrees)
    return angle_deg

def find_maximum_angle_point(skeleton_points, min_distance_from_ends=10):
    """
    Find the point along the skeleton that creates the maximum angle with the endpoints.
    """
    if len(skeleton_points) < 3:
        return None, 0, 0, 0
    
    top_point = skeleton_points[0]
    bottom_point = skeleton_points[-1]
    
    max_angle = 0
    best_point_idx = len(skeleton_points) // 2  # Default to middle
    best_point = skeleton_points[best_point_idx]
    
    # Search for the point that creates the maximum angle
    start_idx = min(min_distance_from_ends, len(skeleton_points) // 4)
    end_idx = max(len(skeleton_points) - min_distance_from_ends, 3 * len(skeleton_points) // 4)
    
    for i in range(start_idx, end_idx):
        middle_point = skeleton_points[i]
        
        # Calculate angle at the middle point
        angle = calculate_three_point_angle(top_point, middle_point, bottom_point)
        
        # We want the deviation from 180 degrees (straight line)
        deviation = abs(180 - angle)
        
        if deviation > max_angle:
            max_angle = deviation
            best_point_idx = i
            best_point = middle_point
    
    # Calculate the final angle - deviation from straight
    final_angle = max_angle
    
    # Determine the direction of bending using cross product
    v1 = best_point - top_point
    v2 = bottom_point - best_point
    
    # Convert to 3D vectors for cross product calculation
    v1_3d = np.array([v1[0], v1[1], 0])
    v2_3d = np.array([v2[0], v2[1], 0])
    cross_product = np.cross(v1_3d, v2_3d)[2]  # Take z-component
    
    # Apply sign based on bending direction
    if cross_product < 0:
        final_angle = -final_angle
    
    return best_point, final_angle, best_point_idx, len(skeleton_points)

def calculate_head_angle_alternative(skeleton, prev_angle=None, min_skeleton_length=20):
    """
    Alternative head angle calculation using medial axis and three-point method.
    """
    try:
        # Handle both 2D and 3D skeleton formats
        if skeleton.ndim == 3:
            skeleton_2d = skeleton[0]
        else:
            skeleton_2d = skeleton
        
        if not np.any(skeleton_2d):
            return {
                'angle_degrees': prev_angle if prev_angle is not None else 0,
                'error': 'Empty skeleton',
                'bend_point': [0, 0],
                'bend_position_relative': 0,
                'skeleton_length': 0,
                'top_point': [0, 0],
                'bottom_point': [0, 0],
                'skeleton_points': [[0, 0]]
            }
        
        # Order skeleton points from top to bottom
        ordered_points = order_skeleton_points(skeleton_2d)
        
        if len(ordered_points) < min_skeleton_length:
            return {
                'angle_degrees': prev_angle if prev_angle is not None else 0,
                'error': f'Skeleton too short: {len(ordered_points)} < {min_skeleton_length}',
                'bend_point': [0, 0],
                'bend_position_relative': 0,
                'skeleton_length': len(ordered_points),
                'top_point': ordered_points[0].tolist() if len(ordered_points) > 0 else [0, 0],
                'bottom_point': ordered_points[-1].tolist() if len(ordered_points) > 0 else [0, 0],
                'skeleton_points': ordered_points.tolist()
            }
        
        # Find the point that creates maximum angle
        bend_point, angle_degrees, bend_idx, skeleton_length = find_maximum_angle_point(ordered_points)
        
        # Calculate relative position of bend point
        bend_position_relative = bend_idx / len(ordered_points) if len(ordered_points) > 0 else 0
        
        result = {
            'angle_degrees': float(angle_degrees),
            'error': None,
            'bend_point': bend_point.tolist(),
            'bend_position_relative': float(bend_position_relative),
            'skeleton_length': skeleton_length,
            'top_point': ordered_points[0].tolist(),
            'bottom_point': ordered_points[-1].tolist(),
            'skeleton_points': ordered_points.tolist()
        }
        
        return result
        
    except Exception as e:
        return {
            'angle_degrees': prev_angle if prev_angle is not None else 0,
            'error': f'Unexpected error: {str(e)}',
            'bend_point': [0, 0],
            'bend_position_relative': 0,
            'skeleton_length': 0,
            'top_point': [0, 0],
            'bottom_point': [0, 0],
            'skeleton_points': [[0, 0]]
        }

def process_all_frames_alternative(head_segments):
    """
    Process all frames using the alternative skeletonization method.
    """
    skeletons = {}
    smoothed_masks = {}
    skeleton_stats = []

    for frame_idx, frame_data in head_segments.items():
        frame_skeletons = {}
        frame_smoothed_masks = {}

        for obj_id, mask in frame_data.items():
            # Store the smoothed mask for visualization
            smoothed_mask = preprocess_mask(mask)
            frame_smoothed_masks[obj_id] = smoothed_mask
            
            skeleton = get_medial_axis_skeleton(mask)
            frame_skeletons[obj_id] = skeleton
            
            skeleton_length = np.sum(skeleton)
            skeleton_stats.append(skeleton_length)
        
        skeletons[frame_idx] = frame_skeletons
        smoothed_masks[frame_idx] = frame_smoothed_masks

        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}")

    stats = {
        'min_length': np.min(skeleton_stats),
        'max_length': np.max(skeleton_stats),
        'mean_length': np.mean(skeleton_stats),
        'median_length': np.median(skeleton_stats),
        'std_length': np.std(skeleton_stats)
    }

    print("\nSkeleton Statistics:")
    print(f"Minimum length: {stats['min_length']:.1f} pixels")
    print(f"Maximum length: {stats['max_length']:.1f} pixels")
    print(f"Mean length: {stats['mean_length']:.1f} pixels")
    print(f"Median length: {stats['median_length']:.1f} pixels")
    print(f"Standard deviation: {stats['std_length']:.1f} pixels")

    return skeletons, smoothed_masks, stats

def process_skeleton_batch_alternative(skeletons, min_skeleton_length=20):
    """
    Process skeleton batch using alternative angle calculation method.
    """
    results_data = []
    
    for frame_idx in sorted(skeletons.keys()):
        frame_data = skeletons[frame_idx]
        
        for obj_id, skeleton in frame_data.items():
            result = calculate_head_angle_alternative(
                skeleton,
                prev_angle=None,
                min_skeleton_length=min_skeleton_length
            )
            
            result_record = {
                'frame': frame_idx,
                'object_id': obj_id,
                'angle_degrees': result['angle_degrees'],
                'bend_point_y': result['bend_point'][0],
                'bend_point_x': result['bend_point'][1],
                'bend_position_relative': result['bend_position_relative'],
                'skeleton_length': result['skeleton_length'],
                'top_point_y': result['top_point'][0],
                'top_point_x': result['top_point'][1],
                'bottom_point_y': result['bottom_point'][0],
                'bottom_point_x': result['bottom_point'][1],
                'error': result['error']
            }
            
            results_data.append(result_record)
    
    results_df = pd.DataFrame(results_data)
    
    # Apply smoothing to reduce noise
    for obj_id in results_df['object_id'].unique():
        obj_mask = results_df['object_id'] == obj_id
        obj_angles = results_df.loc[obj_mask, 'angle_degrees'].values
        
        # Apply moving average smoothing
        smoothed_angles = pd.Series(obj_angles).rolling(window=5, center=True, min_periods=1).mean().values
        results_df.loc[obj_mask, 'angle_degrees'] = smoothed_angles
    
    return results_df

def save_head_angles(filename, results_df, csv_input_dir, final_data_dir):
    base_name = os.path.basename(filename).replace("_headsegmentation.h5", "")
    input_csv_name = base_name + "_crop.csv"
    output_csv_name = base_name + "_crop_headangles.csv"

    input_csv_path = os.path.join(csv_input_dir, input_csv_name)
    final_df = pd.read_csv(input_csv_path)

    print(f"Loaded input data from {input_csv_path}")
    print(f"Input data shape: {final_df.shape}")
    print(f"Results df shape: {results_df.shape}")

    # Merge the results with the input data
    merged_df = pd.merge(final_df, results_df, 
                        left_on=['frame'],
                        right_on=['frame'],
                        how='left',
                        suffixes=('_old', ''))

    # Remove old columns if they exist
    columns_to_remove = [col for col in merged_df.columns if col.endswith('_old')]
    if columns_to_remove:
        merged_df.drop(columns=columns_to_remove, inplace=True)

    output_path = os.path.join(final_data_dir, output_csv_name)
    merged_df.to_csv(output_path, index=False)
    
    print(f"Saved merged df to: {output_path}")
    print(f"Final merged df shape: {merged_df.shape}")
    
    return merged_df

def create_visualization_video(image_dir, head_segments, skeletons, smoothed_masks, angles_df, output_path, fps=10):
    """
    Create visualization video showing smoothed masks, skeletons, and angles.
    """
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 0),  # Yellow
    ]

    def add_skeleton_overlay(image, frame_skeletons, skeleton_colors):
        """Add skeleton visualization"""
        overlay = image.copy()
        
        for skeleton_id, skeleton_data in frame_skeletons.items():
            if skeleton_data.ndim == 3:
                skeleton = skeleton_data[0]
            else:
                skeleton = skeleton_data
            
            y_coords, x_coords = np.where(skeleton)
            if len(y_coords) == 0:
                continue
            
            color = skeleton_colors[skeleton_id]
            for y, x in zip(y_coords, x_coords):
                if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                    cv2.circle(overlay, (x, y), 2, color, -1)
        
        return overlay

    def add_angle_text(image, angle):
        """Add angle text in the center of the frame"""
        if angle is None:
            return image
            
        angle_text = f"{angle:.1f}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 3
        
        # Get text size to center it
        (text_width, text_height), baseline = cv2.getTextSize(
            angle_text, font, font_scale, font_thickness)
        
        # Calculate center position
        center_x = image.shape[1] // 2 - text_width // 2
        center_y = image.shape[0] // 2 + text_height // 2
        
        # Add background rectangle for better visibility
        padding = 10
        cv2.rectangle(image, 
                     (center_x - padding, center_y - text_height - padding),
                     (center_x + text_width + padding, center_y + baseline + padding),
                     (0, 0, 0), -1)  # Black background
        
        # Add text
        cv2.putText(image, angle_text, (center_x, center_y), 
                   font, font_scale, (255, 255, 255), font_thickness)
        
        return image

    # Get image files
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    frame_numbers = []
    for img_file in image_files:
        match = re.search(r'(\d+)', img_file)
        if match:
            frame_numbers.append((int(match.group(1)), img_file))
    
    frame_numbers.sort(key=lambda x: x[0])
    
    if not frame_numbers:
        raise ValueError(f"No image files found in {image_dir}")

    # Set up video writer
    first_image = cv2.imread(os.path.join(image_dir, frame_numbers[0][1]))
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create color mappings
    mask_ids = set()
    for masks in smoothed_masks.values():
        mask_ids.update(masks.keys())
    
    mask_colors = {mask_id: COLORS[i % len(COLORS)] for i, mask_id in enumerate(mask_ids)}
    skeleton_colors = {mask_id: (255, 255, 0) for mask_id in mask_ids}  # Yellow skeletons

    # Get angles dict
    angles_dict = angles_df.set_index('frame')['angle_degrees'].to_dict() if not angles_df.empty else {}

    for frame_number, image_file in frame_numbers:
        try:
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add smoothed mask overlay instead of original masks
            if frame_number in smoothed_masks:
                for mask_id, mask in smoothed_masks[frame_number].items():
                    if mask.dtype != bool:
                        mask = mask > 0.5
                    
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                    
                    # Create colored mask overlay
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask] = mask_colors[mask_id]
                    frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
            
            # Add skeleton overlay
            if frame_number in skeletons:
                frame = add_skeleton_overlay(frame, skeletons[frame_number], skeleton_colors)
            
            # Add angle text
            if frame_number in angles_dict:
                frame = add_angle_text(frame, angles_dict[frame_number])

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            continue

    out.release()
    print(f"Visualization video saved to {output_path}")

# Main execution
if __name__ == "__main__":
    # Get all unprocessed videos
    unprocessed_videos = get_all_unprocessed_videos(head_segmentation_dir, csv_input_dir, final_data_dir)
    print(f"Found {len(unprocessed_videos)} videos to process")
    
    for i, filename in enumerate(unprocessed_videos, 1):
        print(f"\nProcessing video {i}/{len(unprocessed_videos)}: {filename}")
        
        try:
            head_segments = load_cleaned_segments_from_h5(filename)
            
            skeletons, smoothed_masks, skeleton_stats = process_all_frames_alternative(head_segments)
            
            results_df = process_skeleton_batch_alternative(skeletons, min_skeleton_length=20)
            
            merged_df = save_head_angles(filename, results_df, csv_input_dir, final_data_dir)
            
            # Create visualization video
            base_name = os.path.basename(filename).replace("_headsegmentation.h5", "")
            video_output_path = os.path.join(final_data_dir, f"{base_name}_headangle_visualization.mp4")
            
            image_dir = os.path.join('2JPG/MMH223', base_name)
            
            if os.path.exists(image_dir):
                print(f"Creating visualization video using images from: {image_dir}")
                try:
                    create_visualization_video(
                        image_dir=image_dir,
                        head_segments=head_segments,
                        skeletons=skeletons,
                        smoothed_masks=smoothed_masks,
                        angles_df=merged_df,
                        output_path=video_output_path,
                        fps=10
                    )
                    print(f"Visualization video created: {video_output_path}")
                except Exception as e:
                    print(f"Error creating visualization video: {str(e)}")
            else:
                print(f"No image directory found for {base_name}. Checked: {image_dir}")
            
            print(f"Successfully completed processing video {i}/{len(unprocessed_videos)}")
            
        except Exception as e:
            print(f"Error processing video {filename}: {str(e)}")
            print("Continuing to next video...")
            continue
    
    print(f"\nCompleted processing all {len(unprocessed_videos)} videos!")