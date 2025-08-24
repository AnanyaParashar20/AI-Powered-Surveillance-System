# FILE: src/utils/boxes.py

"""
Bounding box utilities including IoU calculation and NMS.
"""

import numpy as np
from typing import List, Tuple


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes in [x, y, w, h] format
        
    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to xyxy format
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2
    
    # Calculate intersection area
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """
    Convert bounding box from [x, y, w, h] to [x1, y1, x2, y2] format.
    
    Args:
        bbox: Bounding box in [x, y, w, h] format
        
    Returns:
        Bounding box in [x1, y1, x2, y2] format
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_xywh(bbox: List[float]) -> List[float]:
    """
    Convert bounding box from [x1, y1, x2, y2] to [x, y, w, h] format.
    
    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        
    Returns:
        Bounding box in [x, y, w, h] format
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def non_max_suppression(boxes: List[List[float]], iou_threshold: float = 0.5,
                       scores: List[float] = None) -> List[List[float]]:
    """
    Apply Non-Maximum Suppression to remove overlapping bounding boxes.
    
    Args:
        boxes: List of bounding boxes in [x, y, w, h] format
        iou_threshold: IoU threshold for suppression
        scores: Optional scores for each box (higher is better)
        
    Returns:
        List of filtered bounding boxes
    """
    if not boxes:
        return []
    
    boxes = np.array(boxes)
    
    # If no scores provided, use area as score (larger boxes preferred)
    if scores is None:
        scores = boxes[:, 2] * boxes[:, 3]  # width * height
    else:
        scores = np.array(scores)
    
    # Sort boxes by score (descending)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        # Pick box with highest score
        current_idx = indices[0]
        keep.append(current_idx)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        remaining_indices = indices[1:]
        ious = []
        
        for idx in remaining_indices:
            iou = calculate_iou(boxes[current_idx].tolist(), boxes[idx].tolist())
            ious.append(iou)
        
        # Keep only boxes with IoU less than threshold
        ious = np.array(ious)
        indices = remaining_indices[ious < iou_threshold]
    
    return boxes[keep].tolist()


def filter_boxes_by_area(boxes: List[List[float]], min_area: float = 0, 
                        max_area: float = float('inf')) -> List[List[float]]:
    """
    Filter boxes by area constraints.
    
    Args:
        boxes: List of bounding boxes
        min_area: Minimum area threshold
        max_area: Maximum area threshold
        
    Returns:
        Filtered list of boxes
    """
    filtered_boxes = []
    
    for box in boxes:
        x, y, w, h = box
        area = w * h
        
        if min_area <= area <= max_area:
            filtered_boxes.append(box)
    
    return filtered_boxes


def clip_boxes_to_image(boxes: List[List[float]], img_width: int, 
                       img_height: int) -> List[List[float]]:
    """
    Clip bounding boxes to image boundaries.
    
    Args:
        boxes: List of bounding boxes [x, y, w, h]
        img_width: Image width
        img_height: Image height
        
    Returns:
        List of clipped boxes
    """
    clipped_boxes = []
    
    for box in boxes:
        x, y, w, h = box
        
        # Clip coordinates
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))
        x2 = max(0, min(x + w, img_width))
        y2 = max(0, min(y + h, img_height))
        
        # Recalculate width and height
        new_w = x2 - x
        new_h = y2 - y
        
        # Only keep boxes with positive area
        if new_w > 0 and new_h > 0:
            clipped_boxes.append([x, y, new_w, new_h])
    
    return clipped_boxes


def calculate_box_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate center point of bounding box.
    
    Args:
        bbox: Bounding box [x, y, w, h]
        
    Returns:
        Tuple of (center_x, center_y)
    """
    x, y, w, h = bbox
    return (x + w/2, y + h/2)


def calculate_box_area(bbox: List[float]) -> float:
    """
    Calculate area of bounding box.
    
    Args:
        bbox: Bounding box [x, y, w, h]
        
    Returns:
        Box area
    """
    x, y, w, h = bbox
    return w * h


def expand_box(bbox: List[float], factor: float = 1.1) -> List[float]:
    """
    Expand bounding box by given factor around center.
    
    Args:
        bbox: Bounding box [x, y, w, h]
        factor: Expansion factor
        
    Returns:
        Expanded bounding box
    """
    x, y, w, h = bbox
    
    # Calculate center
    center_x = x + w/2
    center_y = y + h/2
    
    # Expand dimensions
    new_w = w * factor
    new_h = h * factor
    
    # Calculate new top-left corner
    new_x = center_x - new_w/2
    new_y = center_y - new_h/2
    
    return [new_x, new_y, new_w, new_h]
