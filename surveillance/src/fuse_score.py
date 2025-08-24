# FILE: src/fuse_score.py

"""
Event score fusion and alert generation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class EventScore:
    """Individual event score."""
    
    def __init__(self, event_type: str, raw_score: float, track_id: int = None, 
                 contributors: List[Any] = None):
        """
        Initialize event score.
        
        Args:
            event_type: Type of event ('loitering', 'abandonment', 'unusual')
            raw_score: Raw event score
            track_id: Associated track ID
            contributors: Additional event data
        """
        self.event_type = event_type
        self.raw_score = raw_score
        self.track_id = track_id
        self.contributors = contributors or []
        self.normalized_score = self._normalize_score(raw_score, event_type)
    
    def _normalize_score(self, score: float, event_type: str) -> float:
        """
        Normalize raw score to 0-100 range based on event type.
        
        Args:
            score: Raw score
            event_type: Event type
            
        Returns:
            Normalized score (0-100)
        """
        if event_type == 'loitering':
            # Loitering score is typically dwell time in seconds
            # Normalize: 20s=50, 60s=80, 120s+=100
            return min(100, max(0, (score - 20) * 50 / 40 + 50))
        
        elif event_type == 'abandonment':
            # Abandonment score combines static time and absence time
            # Normalize: base score around 60-90 range
            return min(100, max(0, score))
        
        elif event_type == 'unusual':
            # Unusual movement score from z-score
            # z-score of 3.0 = 60, 5.0 = 80, 8.0+ = 100
            return min(100, max(0, (score - 3.0) * 20 + 60))
        
        else:
            return min(100, max(0, score))


class ScoreFusion:
    """Fuses multiple event scores into single alert score."""
    
    def __init__(self, config: Dict):
        """
        Initialize score fusion.
        
        Args:
            config: Fusion configuration
        """
        fusion_config = config.get('fusion', {})
        
        self.w_loiter = fusion_config.get('w_loiter', 0.33)
        self.w_unusual = fusion_config.get('w_unusual', 0.34)
        self.w_abandon = fusion_config.get('w_abandon', 0.33)
        self.alert_threshold = fusion_config.get('alert_threshold', 55)
        
        # Ensure weights sum to 1
        total_weight = self.w_loiter + self.w_unusual + self.w_abandon
        self.w_loiter /= total_weight
        self.w_unusual /= total_weight
        self.w_abandon /= total_weight
        
        logger.info(f"Score fusion weights: loiter={self.w_loiter:.3f}, "
                   f"unusual={self.w_unusual:.3f}, abandon={self.w_abandon:.3f}")
        logger.info(f"Alert threshold: {self.alert_threshold}")
    
    def fuse_scores(self, event_scores: List[EventScore]) -> Tuple[float, bool, Dict]:
        """
        Fuse multiple event scores into single alert score.
        
        Args:
            event_scores: List of event scores
            
        Returns:
            Tuple of (fused_score, is_alert, details)
        """
        if not event_scores:
            return 0.0, False, {}
        
        # Group scores by type
        scores_by_type = {
            'loitering': [],
            'unusual': [],
            'abandonment': []
        }
        
        for score in event_scores:
            if score.event_type in scores_by_type:
                scores_by_type[score.event_type].append(score.normalized_score)
        
        # Calculate maximum score per type
        max_loiter = max(scores_by_type['loitering']) if scores_by_type['loitering'] else 0
        max_unusual = max(scores_by_type['unusual']) if scores_by_type['unusual'] else 0
        max_abandon = max(scores_by_type['abandonment']) if scores_by_type['abandonment'] else 0
        
        # Weighted fusion
        fused_score = (self.w_loiter * max_loiter + 
                      self.w_unusual * max_unusual + 
                      self.w_abandon * max_abandon)
        
        # Determine if alert
        is_alert = fused_score >= self.alert_threshold
        
        # Create details
        details = {
            'loitering_score': max_loiter,
            'unusual_score': max_unusual,
            'abandonment_score': max_abandon,
            'fused_score': fused_score,
            'num_events': len(event_scores),
            'event_types': list(set(score.event_type for score in event_scores)),
            'contributing_tracks': list(set(score.track_id for score in event_scores if score.track_id))
        }
        
        return fused_score, is_alert, details
    
    def create_single_event_alert(self, event_score: EventScore) -> Tuple[float, bool, Dict]:
        """
        Create alert for single event.
        
        Args:
            event_score: Single event score
            
        Returns:
            Tuple of (score, is_alert, details)
        """
        score = event_score.normalized_score
        is_alert = score >= self.alert_threshold
        
        details = {
            'event_type': event_score.event_type,
            'raw_score': event_score.raw_score,
            'normalized_score': score,
            'track_id': event_score.track_id,
            'contributors': event_score.contributors
        }
        
        return score, is_alert, details


class AlertManager:
    """Manages alert generation and suppression."""
    
    def __init__(self, config: Dict):
        """Initialize alert manager."""
        self.fusion = ScoreFusion(config)
        self.recent_alerts = []  # List of (frame_idx, event_type, track_id)
        self.suppression_window = 30  # frames
        
    def process_frame_events(self, frame_idx: int, event_scores: List[EventScore]) -> List[Dict]:
        """
        Process events for current frame and generate alerts.
        
        Args:
            frame_idx: Current frame index
            event_scores: List of event scores
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        if not event_scores:
            return alerts
        
        # Group events by track and type to avoid duplicates
        events_by_key = {}
        for score in event_scores:
            key = (score.event_type, score.track_id)
            if key not in events_by_key:
                events_by_key[key] = []
            events_by_key[key].append(score)
        
        # Generate alerts for each unique event
        for (event_type, track_id), scores in events_by_key.items():
            # Check suppression
            if self._is_suppressed(frame_idx, event_type, track_id):
                continue
            
            # Take highest score if multiple
            best_score = max(scores, key=lambda s: s.normalized_score)
            
            # Create alert
            fused_score, is_alert, details = self.fusion.create_single_event_alert(best_score)
            
            if is_alert:
                alert = {
                    'frame_idx': frame_idx,
                    'type': event_type,
                    'track_id': track_id,
                    'score': fused_score,
                    'details': details
                }
                alerts.append(alert)
                
                # Add to suppression list
                self.recent_alerts.append((frame_idx, event_type, track_id))
        
        # Clean old alerts
        self._clean_old_alerts(frame_idx)
        
        return alerts
    
    def _is_suppressed(self, frame_idx: int, event_type: str, track_id: int) -> bool:
        """Check if event is suppressed due to recent similar alert."""
        for alert_frame, alert_type, alert_track in self.recent_alerts:
            if (event_type == alert_type and 
                track_id == alert_track and
                frame_idx - alert_frame < self.suppression_window):
                return True
        return False
    
    def _clean_old_alerts(self, current_frame: int):
        """Remove old alerts from suppression list."""
        self.recent_alerts = [
            (frame, event_type, track_id) 
            for frame, event_type, track_id in self.recent_alerts
            if current_frame - frame < self.suppression_window * 2
        ]
