#!/usr/bin/env python3
"""
Script to create visualizations from head angle CSV files.
Plots 1_top25 (nrD) and 2_top25 (nrV) against frame number for each CSV file.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_visualization(csv_path, output_dir):
    """
    Create a visualization for a single CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        output_dir (str): Directory to save the PNG file
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        required_cols = ['frame', '1_top25', '2_top25', 'angle_degrees']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} in {csv_path}")
            return False
        
        # Get the base name for the output file
        base_name = Path(csv_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        
        # Normalize angle_degrees to -1 to 1 and flip the sign
        max_pos_angle = df['angle_degrees'].max()
        max_neg_angle = abs(df['angle_degrees'].min())
        max_angle = max(max_pos_angle, max_neg_angle)
        # Flip the sign by multiplying by -1
        normalized_angles = -(df['angle_degrees'] / max_angle) if max_angle > 0 else df['angle_degrees']
        
        # Normalize nrD (1_top25) to -1 to 1
        nrd_min = df['1_top25'].min()
        nrd_max = df['1_top25'].max()
        nrd_range = nrd_max - nrd_min
        if nrd_range > 0:
            # Scale to 0-1 first, then to -1 to 1
            normalized_nrd = 2 * (df['1_top25'] - nrd_min) / nrd_range - 1
        else:
            normalized_nrd = df['1_top25'] * 0  # All zeros if no range
        
        # Normalize nrV (2_top25) to -1 to 1
        nrv_min = df['2_top25'].min()
        nrv_max = df['2_top25'].max()
        nrv_range = nrv_max - nrv_min
        if nrv_range > 0:
            # Scale to 0-1 first, then to -1 to 1
            normalized_nrv = 2 * (df['2_top25'] - nrv_min) / nrv_range - 1
        else:
            normalized_nrv = df['2_top25'] * 0  # All zeros if no range
        
        # Create the plot with wider aspect ratio for better temporal visualization
        fig, ax = plt.subplots(figsize=(20, 6))
        
        # Set background color and improve aesthetics
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')
        
        # Plot all three normalized signals on the same Y-axis
        # Plot head angle first (in background) with gray color and high opacity
        line3 = ax.plot(df['frame'], normalized_angles, color='#666666', linewidth=2.5, 
                       label='Head Angle (Positive = No odour)', alpha=0.7, zorder=1)
        line1 = ax.plot(df['frame'], normalized_nrd, color='#dc3545', linewidth=2, 
                       label='nrD', alpha=0.9, zorder=3)
        line2 = ax.plot(df['frame'], normalized_nrv, color='#0d6efd', linewidth=2, 
                       label='nrV', alpha=0.9, zorder=2)
        
        # Enhance axis styling
        ax.set_xlabel('Frame', fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Values', fontsize=14, fontweight='bold')
        ax.set_ylim(-1.15, 1.15)
        
        # Improve grid appearance
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#cccccc')
        ax.axhline(y=0, color='#333333', linestyle='--', alpha=0.8, linewidth=1)
        
        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.tick_params(colors='#333333', labelsize=12)
        
        # Enhance legend with better styling
        legend = ax.legend(loc='upper right', fontsize=12, frameon=True, 
                          fancybox=True, shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#cccccc')
        
        # Improve title styling
        plt.title(f'{base_name}', fontsize=16, fontweight='bold', 
                 color='#333333', pad=20)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created visualization: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return False

def main():
    """
    Main function to process all CSV files in the 8HEAD_ANGLE directory.
    """
    # Set up directories
    head_angle_dir = "8HEAD_ANGLE/MMH223"
    visualization_dir = "8HEAD_ANGLE/MMH223"
    
    # Create output directory if it doesn't exist
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Check if input directory exists
    if not os.path.exists(head_angle_dir):
        print(f"Error: Directory {head_angle_dir} does not exist!")
        return
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(head_angle_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {head_angle_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # Process each CSV file
    successful = 0
    failed = 0
    
    for csv_file in csv_files:
        csv_path = os.path.join(head_angle_dir, csv_file)
        print(f"\nProcessing: {csv_file}")
        
        # Create only the main visualization (not the combined one)
        success = create_visualization(csv_path, visualization_dir)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Visualizations saved to: {visualization_dir}")

if __name__ == "__main__":
    main()