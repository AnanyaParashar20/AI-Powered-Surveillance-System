# FILE: src/visualize.py

"""
Visualization utilities for drawing bounding boxes and labels.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Colors:
    """Color palette for visualization."""
    
    # Class colors
    PERSON = (0, 255, 0)  # Green
    OBJECT = (255, 0, 0)  # Red
    ALERT = (0, 0, 255)   # Blue
    
    # Event colors
    LOITERING = (0, 255, 255)    # Yellow
    ABANDONMENT = (255, 0, 255)  # Magenta
    UNUSUAL = (255, 255, 0)      # Cyan
    
    # UI colors
    TEXT_BG = (0, 0, 0)      # Black
    TEXT_FG = (255, 255, 255) # White
    ROI = (128, 128, 128)     # Gray


def draw_bbox(frame: np.ndarray, bbox: List[float], label: str, 
              color: Tuple[int, int, int] = Colors.PERSON, 
              thickness: int = 2, font_scale: float = 0.6) -> np.ndarray:
    """
    Draw bounding box with label on frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box [x, y, w, h]
        label: Text label
        color: Box color (B, G, R)
        thickness: Line thickness
        font_scale: Font scale
        
    Returns:
        Frame with drawn bbox
    """
    x, y, w, h = [int(v) for v in bbox]
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
    
    cv2.rectangle(frame, (x, y - text_h - baseline - 5), 
                 (x + text_w + 5, y), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x + 2, y - baseline - 2), 
               font, font_scale, Colors.TEXT_FG, 1)
    
    return frame


def draw_track(frame: np.ndarray, track, show_trail: bool = True, 
               trail_length: int = 10) -> np.ndarray:
    """
    Draw track bounding box and trail.
    
    Args:
        frame: Input frame
        track: Track object
        show_trail: Whether to show movement trail
        trail_length: Number of trail points
        
    Returns:
        Frame with drawn track
    """
    # Choose color based on class
    if track.class_name == 'person':
        color = Colors.PERSON
    else:
        color = Colors.OBJECT
    
    # Create label
    label = f"{track.class_name} #{track.id}"
    if hasattr(track, 'confidence'):
        label += f" ({track.confidence:.2f})"
    
    # Draw bounding box
    frame = draw_bbox(frame, track.bbox, label, color)
    
    # Draw trail
    if show_trail and len(track.history) > 1:
        points = track.history[-trail_length:]
        for i in range(1, len(points)):
            alpha = i / len(points)  # Fade trail
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            
            # Draw line with fading
            line_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pt1, pt2, line_color, 2)
    
    return frame


def draw_roi(frame: np.ndarray, roi_polygon: List[List[int]], 
             alpha: float = 0.3) -> np.ndarray:
    """
    Draw region of interest overlay.
    
    Args:
        frame: Input frame
        roi_polygon: ROI polygon points [[x, y], ...]
        alpha: Overlay transparency
        
    Returns:
        Frame with ROI overlay
    """
    if not roi_polygon:
        return frame
    
    overlay = frame.copy()
    points = np.array(roi_polygon, dtype=np.int32)
    
    # Fill polygon
    cv2.fillPoly(overlay, [points], Colors.ROI)
    
    # Draw border
    cv2.polylines(overlay, [points], True, Colors.ROI, 2)
    
    # Blend with original
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame


def draw_alert_info(frame: np.ndarray, alerts: List[Dict], 
                   frame_idx: int, timestamp: float) -> np.ndarray:
    """
    Draw alert information on frame.
    
    Args:
        frame: Input frame
        alerts: List of alert dictionaries
        frame_idx: Current frame index
        timestamp: Current timestamp
        
    Returns:
        Frame with alert info
    """
    h, w = frame.shape[:2]
    
    # Draw frame info
    info_text = f"Frame: {frame_idx} | Time: {timestamp:.1f}s"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, Colors.TEXT_FG, 2)
    
    # Draw alerts
    if alerts:
        alert_y = 60
        cv2.putText(frame, f"ALERTS: {len(alerts)}", (10, alert_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, Colors.ALERT, 2)
        
        for i, alert in enumerate(alerts[:5]):  # Show max 5 alerts
            alert_y += 30
            event_type = alert.get('type', 'Unknown')
            score = alert.get('score', 0)
            track_id = alert.get('track_id', 'N/A')
            
            alert_text = f"{event_type.upper()} | Score: {score:.1f} | Track: {track_id}"
            
            # Choose color by event type
            if event_type == 'loitering':
                color = Colors.LOITERING
            elif event_type == 'abandonment':
                color = Colors.ABANDONMENT
            elif event_type == 'unusual':
                color = Colors.UNUSUAL
            else:
                color = Colors.ALERT
            
            cv2.putText(frame, alert_text, (10, alert_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame


def draw_unusual_movement_boxes(frame: np.ndarray, boxes: List[List[float]], 
                              score: float) -> np.ndarray:
    """
    Draw unusual movement detection boxes.
    
    Args:
        frame: Input frame
        boxes: List of detection boxes [x, y, w, h]
        score: Movement anomaly score
        
    Returns:
        Frame with unusual movement boxes
    """
    for bbox in boxes:
        label = f"Unusual Movement ({score:.1f})"
        frame = draw_bbox(frame, bbox, label, Colors.UNUSUAL, thickness=3)
    
    return frame


def create_summary_frame(frame_shape: Tuple[int, int], alerts_summary: Dict, 
                        video_info: Dict) -> np.ndarray:
    """
    Create summary frame with statistics.
    
    Args:
        frame_shape: (height, width) of output frame
        alerts_summary: Summary of alerts
        video_info: Video metadata
        
    Returns:
        Summary frame
    """
    h, w = frame_shape
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Title
    title = "Surveillance Analysis Summary"
    cv2.putText(frame, title, (w//2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 
               1.2, Colors.TEXT_FG, 2)
    
    # Video info
    y_offset = 100
    info_lines = [
        f"Video: {video_info.get('path', 'N/A')}",
        f"Duration: {video_info.get('duration', 0):.1f}s",
        f"FPS: {video_info.get('fps', 0):.1f}",
        f"Frames: {video_info.get('total_frames', 0)}"
    ]
    
    for line in info_lines:
        cv2.putText(frame, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, Colors.TEXT_FG, 2)
        y_offset += 30
    
    # Alert summary
    y_offset += 30
    cv2.putText(frame, "Alert Summary:", (50, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, Colors.ALERT, 2)
    y_offset += 40
    
    total_alerts = alerts_summary.get('total', 0)
    loitering_alerts = alerts_summary.get('loitering', 0)
    abandonment_alerts = alerts_summary.get('abandonment', 0)
    unusual_alerts = alerts_summary.get('unusual', 0)
    
    summary_lines = [
        f"Total Alerts: {total_alerts}",
        f"Loitering: {loitering_alerts}",
        f"Abandonment: {abandonment_alerts}",
        f"Unusual Movement: {unusual_alerts}"
    ]
    
    colors = [Colors.TEXT_FG, Colors.LOITERING, Colors.ABANDONMENT, Colors.UNUSUAL]
    
    for line, color in zip(summary_lines, colors):
        cv2.putText(frame, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, color, 2)
        y_offset += 35
    
    return frame


class FrameVisualizer:
    """Main visualizer class."""
    
    def __init__(self, config: Dict):
        """Initialize visualizer with config."""
        self.config = config
        self.roi_polygon = config.get('roi', {}).get('polygon', [])
        self.use_full_frame = config.get('roi', {}).get('use_full_frame', True)
        
    def visualize_frame(self, frame: np.ndarray, tracks: List, alerts: List[Dict],
                       frame_idx: int, timestamp: float, 
                       unusual_boxes: List[List[float]] = None,
                       unusual_score: float = 0.0) -> np.ndarray:
        """
        Visualize complete frame with tracks, alerts, and ROI.
        
        Args:
            frame: Input frame
            tracks: List of tracks
            alerts: List of alerts
            frame_idx: Frame index
            timestamp: Timestamp
            unusual_boxes: Unusual movement boxes
            unusual_score: Unusual movement score
            
        Returns:
            Visualized frame
        """
        vis_frame = frame.copy()
        
        # Draw ROI
        if not self.use_full_frame and self.roi_polygon:
            vis_frame = draw_roi(vis_frame, self.roi_polygon)
        
        # Draw tracks
        for track in tracks:
            vis_frame = draw_track(vis_frame, track)
        
        # Draw unusual movement boxes
        if unusual_boxes:
            vis_frame = draw_unusual_movement_boxes(vis_frame, unusual_boxes, unusual_score)
        
        # Draw alert info
        vis_frame = draw_alert_info(vis_frame, alerts, frame_idx, timestamp)
        
        return vis_frame
