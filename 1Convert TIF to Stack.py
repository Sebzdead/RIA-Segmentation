import os
import tifffile
import numpy as np
import re
from tqdm import tqdm

def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

def tiff_stacker(dir_path, output_path):
    # Get a list of all TIFF files in the directory
    tiff_files = [file for file in os.listdir(dir_path) if file.endswith('.tif')]
    
    if not tiff_files:
        print(f"No TIF files found in {dir_path}")
        return

    # Sort the TIFF files numerically
    tiff_files.sort(key=numerical_sort)

    # Read the first image to determine its shape and data type
    first_image = tifffile.imread(os.path.join(dir_path, tiff_files[0]))

    # Initialize an empty stack with appropriate dimensions and data type
    stack = np.zeros((len(tiff_files), first_image.shape[0], first_image.shape[1]), dtype=first_image.dtype)

    # Read each TIFF file and add it to the stack
    for i, tiff_file in tqdm(enumerate(tiff_files), total=len(tiff_files), desc="Processing images"):
        image = tifffile.imread(os.path.join(dir_path, tiff_file))
        stack[i] = image

    # Save the stack as a multi-image TIFF file
    tifffile.imwrite(output_path, stack)
    print(f"Multi-image TIFF stack saved at {output_path}")

def process_all_folders(base_directory):
    # Get all subdirectories in the base directory
    subdirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    
    if not subdirs:
        print(f"No subdirectories found in {base_directory}")
        return

    for subdir in tqdm(subdirs, desc="Processing folders"):
        input_path = os.path.join(base_directory, subdir)
        output_filename = f"{subdir}_stack.tiff"
        output_path = os.path.join(base_directory, output_filename)
        
        print(f"\nProcessing folder: {subdir}")
        tiff_stacker(input_path, output_path)

if __name__ == "__main__":
    # Specify the base directory containing your folders
    base_directory = r"M:\Hannah's Data\raw_data\calcium imaging\Calcium Videos to Analyze_TIFF"  # Change this to your base directory path
    
    if os.path.isdir(base_directory):
        process_all_folders(base_directory)
    else:
        print(f"{base_directory} is not a directory")