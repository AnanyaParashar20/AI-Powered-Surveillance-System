# FILE: src/eval/avenue_eval.py

"""
Avenue dataset evaluation using frame-level IoU accuracy.
"""

import argparse
import json
import os
import logging
from typing import Dict, List, Tuple
import numpy as np
from ..utils.boxes import calculate_iou

logger = logging.getLogger(__name__)


def load_json_annotations(json_path: str) -> Dict[str, List[List[float]]]:
    """
    Load annotations from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary mapping frame keys to list of boxes
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load {json_path}: {e}")
        return {}


def calculate_frame_accuracy(gt_boxes: List[List[float]], 
                           pred_boxes: List[List[float]], 
                           iou_threshold: float = 0.5) -> float:
    """
    Calculate frame-level accuracy using Pascal VOC IoU matching.
    
    Args:
        gt_boxes: Ground truth boxes [[x,y,w,h], ...]
        pred_boxes: Predicted boxes [[x,y,w,h], ...]
        iou_threshold: IoU threshold for positive match
        
    Returns:
        Frame accuracy (0.0 to 1.0)
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0  # True negative - both empty
    
    if len(gt_boxes) == 0:
        return 0.0  # False positive - predictions on empty frame
    
    if len(pred_boxes) == 0:
        return 0.0  # False negative - no predictions on non-empty frame
    
    # Find best matches using greedy assignment
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i, j] = calculate_iou(gt_box, pred_box)
    
    # Greedy matching - assign highest IoU matches first
    matches = 0
    while True:
        # Find highest IoU among unmatched pairs
        max_iou = 0
        best_gt_idx = -1
        best_pred_idx = -1
        
        for i in range(len(gt_boxes)):
            if gt_matched[i]:
                continue
            for j in range(len(pred_boxes)):
                if pred_matched[j]:
                    continue
                if iou_matrix[i, j] > max_iou and iou_matrix[i, j] >= iou_threshold:
                    max_iou = iou_matrix[i, j]
                    best_gt_idx = i
                    best_pred_idx = j
        
        if best_gt_idx == -1:  # No more valid matches
            break
        
        # Mark as matched
        gt_matched[best_gt_idx] = True
        pred_matched[best_pred_idx] = True
        matches += 1
    
    # Calculate accuracy as ratio of matched GT boxes
    accuracy = matches / len(gt_boxes)
    return accuracy


def evaluate_video(gt_annotations: Dict[str, List[List[float]]], 
                  pred_annotations: Dict[str, List[List[float]]], 
                  iou_thresholds: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) -> Dict[str, float]:
    """
    Evaluate single video at multiple IoU thresholds.
    
    Args:
        gt_annotations: Ground truth frame annotations
        pred_annotations: Predicted frame annotations
        iou_thresholds: List of IoU thresholds to evaluate
        
    Returns:
        Dictionary mapping threshold to accuracy
    """
    results = {}
    
    # Get all frame keys (union of GT and predictions)
    all_frames = set(gt_annotations.keys()) | set(pred_annotations.keys())
    
    for threshold in iou_thresholds:
        frame_accuracies = []
        
        for frame_key in all_frames:
            gt_boxes = gt_annotations.get(frame_key, [])
            pred_boxes = pred_annotations.get(frame_key, [])
            
            accuracy = calculate_frame_accuracy(gt_boxes, pred_boxes, threshold)
            frame_accuracies.append(accuracy)
        
        # Average accuracy across frames
        video_accuracy = np.mean(frame_accuracies) if frame_accuracies else 0.0
        results[f"accuracy@{threshold:.1f}"] = video_accuracy
    
    return results


def evaluate_dataset(gt_dir: str, pred_dir: str, 
                    iou_thresholds: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) -> Dict:
    """
    Evaluate entire dataset.
    
    Args:
        gt_dir: Directory containing ground truth JSON files
        pred_dir: Directory containing prediction JSON files
        iou_thresholds: IoU thresholds for evaluation
        
    Returns:
        Evaluation results dictionary
    """
    if not os.path.exists(gt_dir):
        raise ValueError(f"Ground truth directory not found: {gt_dir}")
    
    if not os.path.exists(pred_dir):
        raise ValueError(f"Prediction directory not found: {pred_dir}")
    
    # Find all ground truth files
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.json')]
    
    if not gt_files:
        raise ValueError(f"No JSON files found in {gt_dir}")
    
    logger.info(f"Found {len(gt_files)} ground truth files")
    
    # Evaluate each video
    video_results = {}
    dataset_accuracies = {f"accuracy@{t:.1f}": [] for t in iou_thresholds}
    
    for gt_file in gt_files:
        video_id = os.path.splitext(gt_file)[0]
        
        # Load ground truth
        gt_path = os.path.join(gt_dir, gt_file)
        gt_annotations = load_json_annotations(gt_path)
        
        # Load predictions
        pred_file = f"{video_id}.json"
        pred_path = os.path.join(pred_dir, pred_file)
        
        if not os.path.exists(pred_path):
            logger.warning(f"Prediction file not found: {pred_path}")
            pred_annotations = {}
        else:
            pred_annotations = load_json_annotations(pred_path)
        
        # Evaluate video
        video_result = evaluate_video(gt_annotations, pred_annotations, iou_thresholds)
        video_results[video_id] = video_result
        
        # Accumulate for dataset average
        for key, accuracy in video_result.items():
            dataset_accuracies[key].append(accuracy)
        
        logger.info(f"Evaluated {video_id}: {video_result}")
    
    # Calculate dataset averages
    dataset_summary = {}
    for key, accuracies in dataset_accuracies.items():
        if accuracies:
            dataset_summary[key] = np.mean(accuracies)
            dataset_summary[f"{key}_std"] = np.std(accuracies)
        else:
            dataset_summary[key] = 0.0
            dataset_summary[f"{key}_std"] = 0.0
    
    results = {
        'dataset_summary': dataset_summary,
        'video_results': video_results,
        'num_videos': len(video_results),
        'iou_thresholds': iou_thresholds
    }
    
    return results


def print_results(results: Dict):
    """Print evaluation results in readable format."""
    print("\n" + "="*60)
    print("AVENUE DATASET EVALUATION RESULTS")
    print("="*60)
    
    summary = results['dataset_summary']
    print(f"Number of videos evaluated: {results['num_videos']}")
    print(f"IoU thresholds: {results['iou_thresholds']}")
    
    print(f"\nDataset Average Accuracy:")
    print("-" * 40)
    for threshold in results['iou_thresholds']:
        key = f"accuracy@{threshold:.1f}"
        mean_acc = summary[key] * 100
        std_acc = summary[f"{key}_std"] * 100
        print(f"IoU @ {threshold:.1f}: {mean_acc:6.2f}% ± {std_acc:5.2f}%")
    
    print(f"\nPer-Video Results:")
    print("-" * 40)
    for video_id, video_result in results['video_results'].items():
        print(f"{video_id}:")
        for threshold in results['iou_thresholds']:
            key = f"accuracy@{threshold:.1f}"
            accuracy = video_result[key] * 100
            print(f"  IoU @ {threshold:.1f}: {accuracy:6.2f}%")
        print()


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate Avenue dataset predictions')
    parser.add_argument('--gt_dir', required=True, help='Ground truth JSON directory')
    parser.add_argument('--pred_dir', required=True, help='Prediction JSON directory')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--thresholds', nargs='+', type=float, 
                       default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                       help='IoU thresholds for evaluation')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Run evaluation
        results = evaluate_dataset(args.gt_dir, args.pred_dir, args.thresholds)
        
        # Print results
        print_results(results)
        
        # Save results if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
