# FILE: src/events/unusual.py

"""
Unusual movement detection using optical flow analysis.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from collections import deque
from ..fuse_score import EventScore
from ..utils.boxes import non_max_suppression

# Try to import scikit-image, fallback to OpenCV
try:
    from skimage.measure import label, regionprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class UnusualMovementDetector:
    """Detects unusual movement patterns using optical flow."""
    
    def __init__(self, config: Dict):
        """
        Initialize unusual movement detector.
        
        Args:
            config: Configuration dictionary
        """
        events_config = config.get('events', {})
        unusual_config = events_config.get('unusual', {})
        
        self.flow_history = unusual_config.get('flow_history', 60)
        self.z_thresh = unusual_config.get('z_thresh', 3.0)
        self.min_area = unusual_config.get('min_area', 800)
        self.nms_iou = unusual_config.get('nms_iou', 0.3)
        
        # Optical flow parameters
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        # State
        self.prev_gray = None
        self.flow_magnitudes = deque(maxlen=self.flow_history)
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        
        logger.info(f"Unusual movement detector initialized:")
        logger.info(f"  Flow history: {self.flow_history} frames")
        logger.info(f"  Z-score threshold: {self.z_thresh}")
        logger.info(f"  Minimum area: {self.min_area} pixels")
        logger.info(f"  NMS IoU: {self.nms_iou}")
    
    def _calculate_optical_flow(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate optical flow and mean magnitude.
        
        Args:
            frame: Current frame
            
        Returns:
            Tuple of (flow_magnitude_map, mean_magnitude)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return np.zeros_like(gray, dtype=np.float32), 0.0
        
        try:
            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, 
                gray,
                None,  # flow parameter (None for new calculation)
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Calculate magnitude from flow vectors
            if flow is not None and len(flow.shape) == 3 and flow.shape[2] == 2:
                magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            else:
                magnitude = np.zeros_like(gray, dtype=np.float32)
                
        except Exception as e:
            logger.warning(f"Optical flow calculation failed: {e}")
            magnitude = np.zeros_like(gray, dtype=np.float32)
        
        # Calculate mean magnitude
        mean_mag = np.mean(magnitude)
        
        self.prev_gray = gray.copy()
        return magnitude, mean_mag
    
    def _update_baseline(self, mean_magnitude: float):
        """Update baseline statistics with new magnitude."""
        self.flow_magnitudes.append(mean_magnitude)
        
        if len(self.flow_magnitudes) >= 10:  # Need minimum samples
            magnitudes = np.array(self.flow_magnitudes)
            self.baseline_mean = np.mean(magnitudes)
            self.baseline_std = np.std(magnitudes)
            
            # Prevent division by zero
            if self.baseline_std < 0.1:
                self.baseline_std = 0.1
    
    def _calculate_z_score(self, current_magnitude: float) -> float:
        """Calculate z-score for current magnitude."""
        if self.baseline_std <= 0:
            return 0.0
        
        z_score = (current_magnitude - self.baseline_mean) / self.baseline_std
        return max(0.0, z_score)  # Only positive anomalies
    
    def _find_anomaly_regions(self, magnitude_map: np.ndarray, 
                             threshold: float) -> List[List[float]]:
        """
        Find regions with anomalous movement.
        
        Args:
            magnitude_map: Flow magnitude map
            threshold: Magnitude threshold
            
        Returns:
            List of bounding boxes [x, y, w, h]
        """
        # Threshold magnitude map
        binary_mask = (magnitude_map > threshold).astype(np.uint8)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        boxes = []
        
        if SKIMAGE_AVAILABLE:
            # Use scikit-image for better connected component analysis
            labeled_mask = label(binary_mask)
            regions = regionprops(labeled_mask)
            
            for region in regions:
                if region.area >= self.min_area:
                    # Get bounding box
                    min_row, min_col, max_row, max_col = region.bbox
                    x, y = min_col, min_row
                    w, h = max_col - min_col, max_row - min_row
                    boxes.append([float(x), float(y), float(w), float(h)])
        else:
            # Fallback to OpenCV findContours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    boxes.append([float(x), float(y), float(w), float(h)])
        
        return boxes
    
    def detect_unusual_movement(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[EventScore], List[List[float]]]:
        """
        Detect unusual movement in frame.
        
        Args:
            frame: Current frame
            frame_idx: Frame index
            
        Returns:
            Tuple of (event_scores, anomaly_boxes)
        """
        events = []
        anomaly_boxes = []
        
        try:
            # Calculate optical flow
            magnitude_map, mean_magnitude = self._calculate_optical_flow(frame)
            
            # Update baseline
            self._update_baseline(mean_magnitude)
            
            # Calculate z-score
            z_score = self._calculate_z_score(mean_magnitude)
            
            # Check if anomalous
            if z_score >= self.z_thresh:
                # Calculate threshold for magnitude map
                mag_threshold = self.baseline_mean + (self.z_thresh * self.baseline_std)
                
                # Find anomaly regions
                raw_boxes = self._find_anomaly_regions(magnitude_map, mag_threshold)
                
                if raw_boxes:
                    # Apply non-maximum suppression
                    anomaly_boxes = non_max_suppression(raw_boxes, self.nms_iou)
                    
                    # Create event score
                    event = EventScore(
                        event_type='unusual',
                        raw_score=z_score,
                        track_id=None,  # Not associated with specific track
                        contributors=[{
                            'z_score': round(z_score, 2),
                            'mean_magnitude': round(mean_magnitude, 2),
                            'baseline_mean': round(self.baseline_mean, 2),
                            'baseline_std': round(self.baseline_std, 2),
                            'num_regions': len(anomaly_boxes),
                            'total_area': sum(box[2] * box[3] for box in anomaly_boxes)
                        }]
                    )
                    
                    events.append(event)
                    
                    logger.debug(f"Unusual movement detected: z-score={z_score:.2f}, "
                               f"{len(anomaly_boxes)} regions")
        
        except Exception as e:
            logger.error(f"Error in unusual movement detection: {e}")
        
        return events, anomaly_boxes
    
    def reset(self):
        """Reset detector state."""
        self.prev_gray = None
        self.flow_magnitudes.clear()
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        
    def get_stats(self) -> Dict:
        """Get current detector statistics."""
        return {
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
            'history_length': len(self.flow_magnitudes),
            'last_magnitude': self.flow_magnitudes[-1] if self.flow_magnitudes else 0.0
        }
