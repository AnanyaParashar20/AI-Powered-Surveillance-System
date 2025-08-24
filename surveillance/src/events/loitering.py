# FILE: src/events/loitering.py

"""
Loitering detection - person dwelling in ROI beyond threshold time.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from ..fuse_score import EventScore

logger = logging.getLogger(__name__)


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm.
    
    Args:
        point: (x, y) coordinates
        polygon: List of [x, y] polygon vertices
        
    Returns:
        True if point is inside polygon
    """
    if not polygon:
        return True  # If no polygon, consider all points inside
    
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


class LoiteringDetector:
    """Detects loitering behavior of persons in ROI."""
    
    def __init__(self, config: Dict):
        """
        Initialize loitering detector.
        
        Args:
            config: Configuration dictionary
        """
        events_config = config.get('events', {})
        roi_config = config.get('roi', {})
        
        self.loitering_seconds = events_config.get('loitering_seconds', 20)
        self.roi_polygon = roi_config.get('polygon', [])
        self.use_full_frame = roi_config.get('use_full_frame', True)
        
        # FPS for time calculations
        video_config = config.get('video', {})
        self.target_fps = video_config.get('target_fps', 15)
        self.loitering_frames = int(self.loitering_seconds * self.target_fps)
        
        logger.info(f"Loitering detector initialized: {self.loitering_seconds}s "
                   f"({self.loitering_frames} frames) threshold")
        logger.info(f"ROI: {'full frame' if self.use_full_frame else 'polygon'}")
    
    def _is_in_roi(self, center: Tuple[float, float]) -> bool:
        """Check if center point is in ROI."""
        if self.use_full_frame:
            return True
        return point_in_polygon(center, self.roi_polygon)
    
    def _get_track_center(self, track) -> Tuple[float, float]:
        """Get center of track bounding box."""
        x, y, w, h = track.bbox
        return (x + w/2, y + h/2)
    
    def update_tracks_roi_time(self, tracks: List, frame_time_delta: float):
        """
        Update ROI time for all tracks.
        
        Args:
            tracks: List of tracks
            frame_time_delta: Time since last frame in seconds
        """
        for track in tracks:
            if track.class_name != 'person':
                continue
            
            center = self._get_track_center(track)
            
            if self._is_in_roi(center):
                # Person is in ROI, increment time
                track.seconds_in_roi += frame_time_delta
            else:
                # Person left ROI, reset counter
                track.seconds_in_roi = 0.0
    
    def detect_loitering(self, tracks: List, frame_idx: int) -> List[EventScore]:
        """
        Detect loitering events in current frame.
        
        Args:
            tracks: List of active tracks
            frame_idx: Current frame index
            
        Returns:
            List of loitering event scores
        """
        events = []
        
        for track in tracks:
            if track.class_name != 'person':
                continue
            
            # Check if person has been loitering
            if track.seconds_in_roi >= self.loitering_seconds:
                # Calculate score based on dwell time
                dwell_score = track.seconds_in_roi
                
                # Create event
                event = EventScore(
                    event_type='loitering',
                    raw_score=dwell_score,
                    track_id=track.id,
                    contributors=[{
                        'track_id': track.id,
                        'dwell_seconds': round(track.seconds_in_roi, 1),
                        'bbox': track.bbox.copy(),
                        'center': self._get_track_center(track)
                    }]
                )
                
                events.append(event)
                
                logger.debug(f"Loitering detected: Track {track.id} "
                           f"dwelled for {track.seconds_in_roi:.1f}s")
        
        return events
    
    def get_roi_info(self) -> Dict:
        """Get ROI information."""
        return {
            'use_full_frame': self.use_full_frame,
            'polygon': self.roi_polygon,
            'loitering_threshold_seconds': self.loitering_seconds
        }
