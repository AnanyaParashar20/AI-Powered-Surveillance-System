# FILE: scripts/convert_avenue_mat.py

"""
Convert Avenue dataset .mat files to JSON format for evaluation.
"""

import argparse
import os
import json
import logging
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from skimage.measure import label, regionprops
from skimage.morphology import opening, disk
import cv2

logger = logging.getLogger(__name__)


def load_mat_file(mat_path: str, transpose: bool = False) -> np.ndarray:
    """
    Load .mat file and extract mask volume.
    
    Args:
        mat_path: Path to .mat file
        transpose: Whether to transpose dimensions
        
    Returns:
        3D mask array
    """
    try:
        # Load .mat file
        mat_data = loadmat(mat_path)
        
        # Print available keys for debugging
        logger.info(f"Available keys in {mat_path}: {list(mat_data.keys())}")
        
        # Find the main data key (skip matlab metadata)
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if not data_keys:
            raise ValueError(f"No data keys found in {mat_path}")
        
        # Use the first non-metadata key
        main_key = data_keys[0]
        mask_volume = mat_data[main_key]
        
        logger.info(f"Loaded {main_key} with shape: {mask_volume.shape}")
        
        # Handle different possible shapes
        if len(mask_volume.shape) == 3:
            if transpose:
                # Transpose if frames are first dimension: T×H×W -> H×W×T
                mask_volume = np.transpose(mask_volume, (1, 2, 0))
                logger.info(f"Transposed to shape: {mask_volume.shape}")
        elif len(mask_volume.shape) == 2:
            # Single frame, add time dimension
            mask_volume = mask_volume[:, :, np.newaxis]
        else:
            raise ValueError(f"Unexpected mask shape: {mask_volume.shape}")
        
        return mask_volume
        
    except Exception as e:
        logger.error(f"Failed to load {mat_path}: {e}")
        raise


def mask_to_bboxes(mask: np.ndarray, area_threshold: int = 150) -> list:
    """
    Convert binary mask to bounding boxes.
    
    Args:
        mask: Binary mask (H, W)
        area_threshold: Minimum area for valid regions
        
    Returns:
        List of bounding boxes [x, y, w, h]
    """
    if mask.max() == 0:
        return []  # No objects in mask
    
    # Ensure binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Morphological opening to clean up noise
    kernel = disk(2)
    binary_mask = opening(binary_mask, kernel)
    
    # Find connected components
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)
    
    bboxes = []
    for region in regions:
        if region.area >= area_threshold:
            # Get bounding box coordinates
            min_row, min_col, max_row, max_col = region.bbox
            
            # Convert to [x, y, w, h] format
            x = min_col
            y = min_row
            w = max_col - min_col
            h = max_row - min_row
            
            bboxes.append([float(x), float(y), float(w), float(h)])
    
    return bboxes


def convert_mat_to_json(mat_path: str, output_path: str, 
                       area_threshold: int = 150, transpose: bool = False):
    """
    Convert .mat file to JSON format.
    
    Args:
        mat_path: Path to input .mat file
        output_path: Path to output JSON file
        area_threshold: Minimum area threshold for objects
        transpose: Whether to transpose mask dimensions
    """
    logger.info(f"Converting {mat_path} to {output_path}")
    
    # Load mask volume
    mask_volume = load_mat_file(mat_path, transpose)
    h, w, num_frames = mask_volume.shape
    
    logger.info(f"Processing {num_frames} frames of size {h}x{w}")
    
    # Process each frame
    frame_annotations = {}
    total_objects = 0
    
    for frame_idx in range(num_frames):
        # Get frame mask
        frame_mask = mask_volume[:, :, frame_idx]
        
        # Convert to bounding boxes
        bboxes = mask_to_bboxes(frame_mask, area_threshold)
        
        # Store annotations with zero-padded frame key
        frame_key = f"{frame_idx:06d}"
        frame_annotations[frame_key] = bboxes
        
        total_objects += len(bboxes)
        
        if frame_idx % 100 == 0 and frame_idx > 0:
            logger.info(f"Processed {frame_idx}/{num_frames} frames, "
                       f"found {total_objects} objects so far")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(frame_annotations, f, indent=2)
    
    logger.info(f"Conversion completed: {num_frames} frames, {total_objects} total objects")
    logger.info(f"Saved to: {output_path}")


def convert_all_avenue_files(raw_mat_dir: str, output_dir: str, 
                           area_threshold: int = 150, transpose: bool = False):
    """
    Convert all .mat files in Avenue dataset.
    
    Args:
        raw_mat_dir: Directory containing .mat files
        output_dir: Output directory for JSON files  
        area_threshold: Minimum area threshold
        transpose: Whether to transpose dimensions
    """
    if not os.path.exists(raw_mat_dir):
        raise ValueError(f"Raw mat directory not found: {raw_mat_dir}")
    
    # Find all .mat files
    mat_files = [f for f in os.listdir(raw_mat_dir) if f.endswith('.mat')]
    
    if not mat_files:
        raise ValueError(f"No .mat files found in {raw_mat_dir}")
    
    logger.info(f"Found {len(mat_files)} .mat files to convert")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each file
    for mat_file in sorted(mat_files):
        mat_path = os.path.join(raw_mat_dir, mat_file)
        
        # Generate output filename (e.g., vol01.mat -> 01.json)
        base_name = os.path.splitext(mat_file)[0]
        if base_name.startswith('vol'):
            video_id = base_name[3:]  # Remove 'vol' prefix
        else:
            video_id = base_name
        
        output_file = f"{video_id}.json"
        output_path = os.path.join(output_dir, output_file)
        
        try:
            convert_mat_to_json(mat_path, output_path, area_threshold, transpose)
        except Exception as e:
            logger.error(f"Failed to convert {mat_file}: {e}")
            continue
    
    logger.info(f"Conversion completed. Results saved to: {output_dir}")


def main():
    """Main conversion script."""
    parser = argparse.ArgumentParser(description='Convert Avenue .mat files to JSON')
    
    parser.add_argument('--raw_mat_dir', 
                       default='data/avenue/raw_mat/testing_vol',
                       help='Directory containing .mat files')
    parser.add_argument('--output_dir',
                       default='data/avenue/gt', 
                       help='Output directory for JSON files')
    parser.add_argument('--area_thr', type=int, default=150,
                       help='Minimum area threshold for objects')
    parser.add_argument('--transpose', action='store_true',
                       help='Transpose mask dimensions from T×H×W to H×W×T')
    parser.add_argument('--single_file', 
                       help='Convert single .mat file instead of directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.single_file:
            # Convert single file
            output_file = os.path.splitext(os.path.basename(args.single_file))[0] + '.json'
            output_path = os.path.join(args.output_dir, output_file)
            convert_mat_to_json(args.single_file, output_path, args.area_thr, args.transpose)
        else:
            # Convert all files in directory
            convert_all_avenue_files(args.raw_mat_dir, args.output_dir, 
                                   args.area_thr, args.transpose)
            
        print("Conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
