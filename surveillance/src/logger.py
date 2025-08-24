# FILE: src/logger.py

"""
CSV logger for surveillance alerts and events.
"""

import csv
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EventLogger:
    """Logs surveillance events to CSV file."""
    
    def __init__(self, output_dir: str, video_name: str = "surveillance"):
        """
        Initialize event logger.
        
        Args:
            output_dir: Output directory for logs
            video_name: Video name for log file
        """
        self.output_dir = output_dir
        self.video_name = video_name
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"events_{video_name}_{timestamp}.csv")
        
        # CSV headers
        self.headers = [
            'frame_idx',
            'time_sec', 
            'type',
            'track_id',
            'score',
            'bbox',
            'contributors',
            'image_path'
        ]
        
        # Initialize CSV file
        self._init_csv()
        
        logger.info(f"Event logger initialized: {self.csv_path}")
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
    
    def log_alert(self, frame_idx: int, timestamp: float, alert: Dict, 
                  bbox: List[float] = None, image_path: str = None):
        """
        Log single alert to CSV.
        
        Args:
            frame_idx: Frame index
            timestamp: Timestamp in seconds
            alert: Alert dictionary
            bbox: Bounding box [x, y, w, h]
            image_path: Path to saved alert image
        """
        # Extract alert info
        event_type = alert.get('type', 'unknown')
        track_id = alert.get('track_id', '')
        score = alert.get('score', 0.0)
        
        # Get contributors from alert details
        details = alert.get('details', {})
        contributors = details.get('contributors', [])
        
        # Use track bbox if not provided
        if bbox is None:
            bbox = details.get('bbox', [0, 0, 0, 0])
        
        # Format data for CSV
        row_data = [
            frame_idx,
            round(timestamp, 2),
            event_type,
            track_id,
            round(float(score), 2),
            json.dumps([round(float(x), 2) for x in bbox]),
            json.dumps(self._serialize_contributors(contributors)) if contributors else '',
            image_path or ''
        ]
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        logger.debug(f"Logged alert: {event_type} at frame {frame_idx}")
    
    def log_multiple_alerts(self, frame_idx: int, timestamp: float, 
                           alerts: List[Dict], image_path: str = None):
        """
        Log multiple alerts from same frame.
        
        Args:
            frame_idx: Frame index
            timestamp: Timestamp in seconds  
            alerts: List of alert dictionaries
            image_path: Path to saved alert image
        """
        for alert in alerts:
            self.log_alert(frame_idx, timestamp, alert, image_path=image_path)
    
    def get_log_path(self) -> str:
        """Get path to CSV log file."""
        return self.csv_path
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics from log file.
        
        Returns:
            Dictionary with alert statistics
        """
        if not os.path.exists(self.csv_path):
            return {}
        
        stats = {
            'total_alerts': 0,
            'by_type': {},
            'by_track': {},
            'time_range': [float('inf'), -float('inf')],
            'frame_range': [float('inf'), -float('inf')],
            'avg_score': 0.0
        }
        
        scores = []
        
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    stats['total_alerts'] += 1
                    
                    # By type
                    event_type = row['type']
                    stats['by_type'][event_type] = stats['by_type'].get(event_type, 0) + 1
                    
                    # By track
                    track_id = row['track_id']
                    if track_id:
                        stats['by_track'][track_id] = stats['by_track'].get(track_id, 0) + 1
                    
                    # Time range
                    time_sec = float(row['time_sec'])
                    stats['time_range'][0] = min(stats['time_range'][0], time_sec)
                    stats['time_range'][1] = max(stats['time_range'][1], time_sec)
                    
                    # Frame range
                    frame_idx = int(row['frame_idx'])
                    stats['frame_range'][0] = min(stats['frame_range'][0], frame_idx)
                    stats['frame_range'][1] = max(stats['frame_range'][1], frame_idx)
                    
                    # Score
                    score = float(row['score'])
                    scores.append(score)
            
            # Calculate average score
            if scores:
                stats['avg_score'] = sum(scores) / len(scores)
            
            # Fix infinite ranges
            if stats['time_range'][0] == float('inf'):
                stats['time_range'] = [0, 0]
            if stats['frame_range'][0] == float('inf'):
                stats['frame_range'] = [0, 0]
                
        except Exception as e:
            logger.error(f"Failed to calculate stats: {e}")
        
        return stats
    
    def _serialize_contributors(self, contributors):
        """Convert numpy types to Python types for JSON serialization."""
        if not contributors:
            return []
        
        serialized = []
        for contrib in contributors:
            if isinstance(contrib, dict):
                # Convert all numpy types to Python types
                serialized_contrib = {}
                for key, value in contrib.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        serialized_contrib[key] = value.item()
                    elif isinstance(value, (list, tuple)):
                        serialized_contrib[key] = [float(x) if hasattr(x, 'item') else x for x in value]
                    else:
                        serialized_contrib[key] = value
                serialized.append(serialized_contrib)
            else:
                serialized.append(contrib)
        return serialized


class ImageSaver:
    """Saves alert frames as images."""
    
    def __init__(self, output_dir: str):
        """
        Initialize image saver.
        
        Args:
            output_dir: Directory to save images
        """
        self.frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(self.frames_dir, exist_ok=True)
        logger.info(f"Image saver initialized: {self.frames_dir}")
    
    def save_alert_frame(self, frame, frame_idx: int, alert: Dict, 
                        video_name: str = "surveillance") -> str:
        """
        Save alert frame as image.
        
        Args:
            frame: Frame array (numpy)
            frame_idx: Frame index
            alert: Alert dictionary
            video_name: Video name for filename
            
        Returns:
            Path to saved image
        """
        import cv2
        
        # Create filename
        event_type = alert.get('type', 'unknown')
        track_id = alert.get('track_id', 'none')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        
        filename = f"{video_name}_frame{frame_idx:06d}_{event_type}_track{track_id}_{timestamp}.jpg"
        filepath = os.path.join(self.frames_dir, filename)
        
        # Save image
        try:
            success = cv2.imwrite(filepath, frame)
            if success:
                logger.debug(f"Saved alert image: {filename}")
                return os.path.relpath(filepath)
            else:
                logger.error(f"Failed to save image: {filename}")
                return ""
        except Exception as e:
            logger.error(f"Error saving image {filename}: {e}")
            return ""
    
    def save_multiple_alerts(self, frame, frame_idx: int, alerts: List[Dict], 
                           video_name: str = "surveillance") -> str:
        """
        Save frame with multiple alerts.
        
        Args:
            frame: Frame array
            frame_idx: Frame index
            alerts: List of alerts
            video_name: Video name
            
        Returns:
            Path to saved image
        """
        if not alerts:
            return ""
        
        # Use first alert for filename, mention multiple in name
        primary_alert = alerts[0]
        event_types = "_".join(set(alert.get('type', 'unknown') for alert in alerts))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        filename = f"{video_name}_frame{frame_idx:06d}_multi_{event_types}_{len(alerts)}alerts_{timestamp}.jpg"
        filepath = os.path.join(self.frames_dir, filename)
        
        # Save image
        try:
            import cv2
            success = cv2.imwrite(filepath, frame)
            if success:
                logger.debug(f"Saved multi-alert image: {filename}")
                return os.path.relpath(filepath)
            else:
                logger.error(f"Failed to save multi-alert image: {filename}")
                return ""
        except Exception as e:
            logger.error(f"Error saving multi-alert image {filename}: {e}")
            return ""


class SurveillanceLogger:
    """Combined logger for events and images."""
    
    def __init__(self, output_dir: str, video_name: str = "surveillance", 
                 save_images: bool = True):
        """
        Initialize surveillance logger.
        
        Args:
            output_dir: Output directory
            video_name: Video name
            save_images: Whether to save alert images
        """
        self.event_logger = EventLogger(output_dir, video_name)
        self.image_saver = ImageSaver(output_dir) if save_images else None
        self.save_images = save_images
        self.video_name = video_name
        
    def log_frame_alerts(self, frame, frame_idx: int, timestamp: float, 
                        alerts: List[Dict]):
        """
        Log all alerts from a frame.
        
        Args:
            frame: Frame array
            frame_idx: Frame index
            timestamp: Timestamp in seconds
            alerts: List of alerts
        """
        if not alerts:
            return
        
        # Save image if enabled
        image_path = None
        if self.save_images and self.image_saver:
            if len(alerts) == 1:
                image_path = self.image_saver.save_alert_frame(
                    frame, frame_idx, alerts[0], self.video_name)
            else:
                image_path = self.image_saver.save_multiple_alerts(
                    frame, frame_idx, alerts, self.video_name)
        
        # Log events
        self.event_logger.log_multiple_alerts(frame_idx, timestamp, alerts, image_path)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get logging summary statistics."""
        return self.event_logger.get_summary_stats()
    
    def get_csv_path(self) -> str:
        """Get path to CSV log file."""
        return self.event_logger.get_log_path()
