# FILE: src/track.py

"""
Simple multi-object tracker using IoU and centroid matching.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from .detect import Detection

logger = logging.getLogger(__name__)


class Track:
    """Single object track."""
    
    def __init__(self, track_id: int, detection: Detection, frame_idx: int):
        """
        Initialize track.
        
        Args:
            track_id: Unique track ID
            detection: Initial detection
            frame_idx: Frame index when track started
        """
        self.id = track_id
        self.bbox = detection.bbox.copy()  # [x, y, w, h]
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        
        # Track history
        self.history = [self._get_center()]  # List of (x, y) centers
        self.frame_indices = [frame_idx]
        
        # State tracking
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        
        # Event-specific state
        self.seconds_in_roi = 0.0
        self.static_frames = 0
        self.owner_id = None
        self.last_owner_distance = float('inf')
    
    def _get_center(self) -> Tuple[float, float]:
        """Get bounding box center."""
        x, y, w, h = self.bbox
        return (x + w/2, y + h/2)
    
    def update(self, detection: Detection, frame_idx: int, smoothing: float = 0.8):
        """
        Update track with new detection.
        
        Args:
            detection: New detection
            frame_idx: Current frame index
            smoothing: Smoothing factor for bbox updates
        """
        # Smooth bbox update
        new_bbox = detection.bbox
        if smoothing > 0:
            self.bbox = [
                smoothing * self.bbox[i] + (1 - smoothing) * new_bbox[i]
                for i in range(4)
            ]
        else:
            self.bbox = new_bbox.copy()
        
        self.confidence = detection.confidence
        
        # Update history
        center = self._get_center()
        self.history.append(center)
        self.frame_indices.append(frame_idx)
        
        # Keep limited history
        max_history = 100
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
            self.frame_indices = self.frame_indices[-max_history:]
        
        # Update counters
        self.hits += 1
        self.time_since_update = 0
        
        # Update static frames counter
        if len(self.history) >= 2:
            prev_center = self.history[-2]
            curr_center = self.history[-1]
            movement = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                             (curr_center[1] - prev_center[1])**2)
            
            if movement < 10:  # pixels
                self.static_frames += 1
            else:
                self.static_frames = 0
    
    def predict(self):
        """Predict next position (currently just returns current bbox)."""
        self.age += 1
        self.time_since_update += 1
    
    def is_static(self, static_threshold: int = 30) -> bool:
        """Check if track has been static."""
        return self.static_frames >= static_threshold
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get average velocity over recent frames."""
        if len(self.history) < 2:
            return (0.0, 0.0)
        
        # Use last 5 frames for velocity calculation
        recent_history = self.history[-5:]
        if len(recent_history) < 2:
            return (0.0, 0.0)
        
        dx = recent_history[-1][0] - recent_history[0][0]
        dy = recent_history[-1][1] - recent_history[0][1]
        dt = len(recent_history) - 1
        
        return (dx / dt, dy / dt)


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding boxes in [x, y, w, h] format
        
    Returns:
        IoU value
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to xyxy
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_distance(center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between centers."""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


class MultiTracker:
    """Multi-object tracker."""
    
    def __init__(self, max_age: int = 15, iou_threshold: float = 0.3, 
                 distance_threshold: float = 80, smoothing: float = 0.8):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum age before deleting track
            iou_threshold: IoU threshold for matching
            distance_threshold: Distance threshold for matching
            smoothing: Bbox smoothing factor
        """
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.smoothing = smoothing
        
        self.tracks = []
        self.next_id = 1
        
        logger.info(f"MultiTracker initialized: max_age={max_age}, "
                   f"iou_threshold={iou_threshold}, distance_threshold={distance_threshold}")
    
    def update(self, detections: List[Detection], frame_idx: int) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections
            frame_idx: Current frame index
            
        Returns:
            List of active tracks
        """
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        matched_indices, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(
            detections, self.tracks
        )
        
        # Update matched tracks
        for det_idx, track_idx in matched_indices:
            self.tracks[track_idx].update(detections[det_idx], frame_idx, self.smoothing)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = Track(self.next_id, detections[det_idx], frame_idx)
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks 
                      if track.time_since_update <= self.max_age]
        
        return self.tracks.copy()
    
    def _match_detections_to_tracks(self, detections: List[Detection], tracks: List[Track]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks.
        
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Group by class for better matching
        class_groups = {}
        for i, det in enumerate(detections):
            if det.class_name not in class_groups:
                class_groups[det.class_name] = []
            class_groups[det.class_name].append(i)
        
        matched_pairs = []
        unmatched_detections = []
        unmatched_tracks = list(range(len(tracks)))
        
        for class_name, det_indices in class_groups.items():
            # Find tracks of same class
            class_tracks = []
            for i, track in enumerate(tracks):
                if track.class_name == class_name:
                    class_tracks.append(i)
            
            if not class_tracks:
                unmatched_detections.extend(det_indices)
                continue
            
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(det_indices), len(class_tracks)))
            distance_matrix = np.zeros((len(det_indices), len(class_tracks)))
            
            for i, det_idx in enumerate(det_indices):
                det_center = (detections[det_idx].bbox[0] + detections[det_idx].bbox[2]/2,
                             detections[det_idx].bbox[1] + detections[det_idx].bbox[3]/2)
                
                for j, track_idx in enumerate(class_tracks):
                    # IoU matching
                    iou = calculate_iou(detections[det_idx].bbox, tracks[track_idx].bbox)
                    iou_matrix[i, j] = iou
                    
                    # Distance matching
                    track_center = tracks[track_idx]._get_center()
                    dist = calculate_distance(det_center, track_center)
                    distance_matrix[i, j] = dist
            
            # Greedy matching - prefer IoU, fallback to distance
            used_detections = set()
            used_tracks = set()
            
            # First pass: IoU matching
            for _ in range(min(len(det_indices), len(class_tracks))):
                best_iou = 0
                best_match = None
                
                for i in range(len(det_indices)):
                    if i in used_detections:
                        continue
                    for j in range(len(class_tracks)):
                        if j in used_tracks:
                            continue
                        
                        if iou_matrix[i, j] > best_iou and iou_matrix[i, j] > self.iou_threshold:
                            best_iou = iou_matrix[i, j]
                            best_match = (i, j)
                
                if best_match is None:
                    break
                
                det_i, track_j = best_match
                matched_pairs.append((det_indices[det_i], class_tracks[track_j]))
                used_detections.add(det_i)
                used_tracks.add(track_j)
                
                # Remove from unmatched tracks
                if class_tracks[track_j] in unmatched_tracks:
                    unmatched_tracks.remove(class_tracks[track_j])
            
            # Second pass: Distance matching for remaining
            for i in range(len(det_indices)):
                if i in used_detections:
                    continue
                    
                best_dist = self.distance_threshold
                best_match = None
                
                for j in range(len(class_tracks)):
                    if j in used_tracks:
                        continue
                    
                    if distance_matrix[i, j] < best_dist:
                        best_dist = distance_matrix[i, j]
                        best_match = (i, j)
                
                if best_match is not None:
                    det_i, track_j = best_match
                    matched_pairs.append((det_indices[det_i], class_tracks[track_j]))
                    used_detections.add(det_i)
                    used_tracks.add(track_j)
                    
                    # Remove from unmatched tracks
                    if class_tracks[track_j] in unmatched_tracks:
                        unmatched_tracks.remove(class_tracks[track_j])
            
            # Add unmatched detections for this class
            for i, det_idx in enumerate(det_indices):
                if i not in used_detections:
                    unmatched_detections.append(det_idx)
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def get_tracks_by_class(self, class_name: str) -> List[Track]:
        """Get all tracks of specific class."""
        return [track for track in self.tracks if track.class_name == class_name]
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None
