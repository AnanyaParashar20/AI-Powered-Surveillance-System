# FILE: src/events/abandonment.py

"""
Object abandonment detection - static objects left unattended.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from ..fuse_score import EventScore

logger = logging.getLogger(__name__)


class AbandonmentDetector:
    """Detects object abandonment events."""
    
    def __init__(self, config: Dict):
        """
        Initialize abandonment detector.
        
        Args:
            config: Configuration dictionary
        """
        events_config = config.get('events', {})
        abandonment_config = events_config.get('abandonment', {})
        
        self.static_seconds = abandonment_config.get('static_seconds', 10)
        self.owner_link_seconds = abandonment_config.get('owner_link_seconds', 4)
        self.owner_max_dist = abandonment_config.get('owner_max_dist', 120)
        self.absent_seconds = abandonment_config.get('absent_seconds', 8)
        
        # FPS for frame calculations
        video_config = config.get('video', {})
        self.target_fps = video_config.get('target_fps', 15)
        
        self.static_frames = int(self.static_seconds * self.target_fps)
        self.owner_link_frames = int(self.owner_link_seconds * self.target_fps)
        self.absent_frames = int(self.absent_seconds * self.target_fps)
        
        # Track ownership history
        self.ownership_history = {}  # object_track_id -> [(frame, person_track_id, distance)]
        
        logger.info(f"Abandonment detector initialized:")
        logger.info(f"  Static threshold: {self.static_seconds}s ({self.static_frames} frames)")
        logger.info(f"  Owner link time: {self.owner_link_seconds}s ({self.owner_link_frames} frames)")
        logger.info(f"  Owner max distance: {self.owner_max_dist} pixels")
        logger.info(f"  Absent threshold: {self.absent_seconds}s ({self.absent_frames} frames)")
    
    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between two bounding box centers."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _find_nearest_person(self, object_track, person_tracks: List) -> Tuple[Optional[int], float]:
        """
        Find nearest person to object.
        
        Args:
            object_track: Object track
            person_tracks: List of person tracks
            
        Returns:
            Tuple of (person_track_id, distance) or (None, inf)
        """
        if not person_tracks:
            return None, float('inf')
        
        min_distance = float('inf')
        nearest_person_id = None
        
        for person_track in person_tracks:
            distance = self._calculate_distance(object_track.bbox, person_track.bbox)
            if distance < min_distance:
                min_distance = distance
                nearest_person_id = person_track.id
        
        return nearest_person_id, min_distance
    
    def _update_ownership(self, object_track, person_tracks: List, frame_idx: int):
        """Update ownership history for object."""
        nearest_person_id, distance = self._find_nearest_person(object_track, person_tracks)
        
        if object_track.id not in self.ownership_history:
            self.ownership_history[object_track.id] = []
        
        history = self.ownership_history[object_track.id]
        
        # Add current ownership data
        if nearest_person_id is not None and distance <= self.owner_max_dist:
            history.append((frame_idx, nearest_person_id, distance))
        else:
            history.append((frame_idx, None, distance))
        
        # Keep limited history (2x the maximum frames we need to look back)
        max_history = max(self.owner_link_frames, self.absent_frames) * 2
        if len(history) > max_history:
            self.ownership_history[object_track.id] = history[-max_history:]
    
    def _get_established_owner(self, object_track_id: int, current_frame: int) -> Optional[int]:
        """
        Get established owner for object based on ownership history.
        
        Args:
            object_track_id: Object track ID
            current_frame: Current frame index
            
        Returns:
            Owner track ID if established, None otherwise
        """
        if object_track_id not in self.ownership_history:
            return None
        
        history = self.ownership_history[object_track_id]
        
        # Look at recent history within owner_link_frames
        link_start_frame = current_frame - self.owner_link_frames
        recent_owners = {}
        
        for frame_idx, owner_id, distance in history:
            if frame_idx >= link_start_frame and owner_id is not None:
                if owner_id not in recent_owners:
                    recent_owners[owner_id] = 0
                recent_owners[owner_id] += 1
        
        if not recent_owners:
            return None
        
        # Get most frequent recent owner
        established_owner = max(recent_owners.items(), key=lambda x: x[1])[0]
        
        # Must be present for at least half the link period
        min_presence = self.owner_link_frames // 2
        if recent_owners[established_owner] >= min_presence:
            return established_owner
        
        return None
    
    def _is_owner_absent(self, owner_id: int, person_tracks: List, 
                        object_track_id: int, current_frame: int) -> bool:
        """
        Check if established owner has been absent for sufficient time.
        
        Args:
            owner_id: Owner track ID
            person_tracks: Current person tracks
            object_track_id: Object track ID
            current_frame: Current frame index
            
        Returns:
            True if owner has been absent long enough
        """
        # Check if owner is currently present
        for person_track in person_tracks:
            if person_track.id == owner_id:
                # Owner is present, check distance
                if object_track_id in self.ownership_history:
                    history = self.ownership_history[object_track_id]
                    if history:
                        _, _, last_distance = history[-1]
                        if last_distance <= self.owner_max_dist:
                            return False  # Owner is close
                break
        
        # Owner is not present or too far, check absence duration
        if object_track_id not in self.ownership_history:
            return False
        
        history = self.ownership_history[object_track_id]
        absent_count = 0
        
        # Count recent frames where owner was absent or too far
        for frame_idx, hist_owner_id, distance in reversed(history):
            if current_frame - frame_idx > self.absent_frames:
                break
            
            if hist_owner_id != owner_id or distance > self.owner_max_dist:
                absent_count += 1
            else:
                break  # Owner was present, reset count
        
        return absent_count >= self.absent_frames
    
    def detect_abandonment(self, tracks: List, frame_idx: int) -> List[EventScore]:
        """
        Detect abandonment events.
        
        Args:
            tracks: List of all tracks
            frame_idx: Current frame index
            
        Returns:
            List of abandonment event scores
        """
        events = []
        
        # Separate person and object tracks
        person_tracks = [t for t in tracks if t.class_name == 'person']
        object_tracks = [t for t in tracks if t.class_name != 'person']
        
        for obj_track in object_tracks:
            # Update ownership history
            self._update_ownership(obj_track, person_tracks, frame_idx)
            
            # Check if object is static
            if not obj_track.is_static(self.static_frames):
                continue
            
            # Check if object has an established owner
            owner_id = self._get_established_owner(obj_track.id, frame_idx)
            if owner_id is None:
                continue
            
            # Check if owner is absent
            if not self._is_owner_absent(owner_id, person_tracks, obj_track.id, frame_idx):
                continue
            
            # Calculate abandonment score
            static_time = obj_track.static_frames / self.target_fps
            
            # Base score from static time and absence
            base_score = min(100, (static_time / self.static_seconds) * 40 + 50)
            
            # Bonus for longer absence
            absence_bonus = min(20, (static_time - self.static_seconds) / self.absent_seconds * 20)
            final_score = min(100, base_score + absence_bonus)
            
            # Create event
            contributors = [{
                'object_track_id': obj_track.id,
                'object_class': obj_track.class_name,
                'owner_track_id': owner_id,
                'static_seconds': round(static_time, 1),
                'bbox': obj_track.bbox.copy()
            }]
            
            event = EventScore(
                event_type='abandonment',
                raw_score=final_score,
                track_id=obj_track.id,
                contributors=contributors
            )
            
            events.append(event)
            
            logger.debug(f"Abandonment detected: Object {obj_track.id} "
                        f"({obj_track.class_name}) abandoned by person {owner_id} "
                        f"for {static_time:.1f}s")
        
        return events
    
    def cleanup_old_ownership(self, active_track_ids: List[int], current_frame: int):
        """Clean up ownership history for inactive tracks."""
        # Remove ownership history for tracks that are no longer active
        inactive_tracks = []
        for track_id in self.ownership_history.keys():
            if track_id not in active_track_ids:
                inactive_tracks.append(track_id)
        
        for track_id in inactive_tracks:
            del self.ownership_history[track_id]
        
        # Clean old history entries
        max_age = max(self.owner_link_frames, self.absent_frames) * 2
        for track_id, history in self.ownership_history.items():
            cutoff_frame = current_frame - max_age
            self.ownership_history[track_id] = [
                entry for entry in history if entry[0] >= cutoff_frame
            ]
