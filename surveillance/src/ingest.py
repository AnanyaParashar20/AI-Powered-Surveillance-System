# FILE: src/ingest.py

"""
Video ingestion with FPS downsampling and resizing.
"""

import cv2
import numpy as np
from typing import Iterator, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoReader:
    """Video reader with FPS downsampling and frame resizing."""
    
    def __init__(self, video_path: str, target_fps: float = 15, 
                 max_side: int = 960, frame_skip: int = 0):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
            target_fps: Target FPS for downsampling
            max_side: Maximum side length for resizing
            frame_skip: Number of frames to skip at start
        """
        self.video_path = video_path
        self.target_fps = target_fps
        self.max_side = max_side
        self.frame_skip = frame_skip
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval for downsampling
        self.frame_interval = max(1, int(self.original_fps / target_fps))
        
        logger.info(f"Video: {video_path}")
        logger.info(f"Original FPS: {self.original_fps:.2f}, Target FPS: {target_fps}")
        logger.info(f"Frame interval: {self.frame_interval}")
        logger.info(f"Total frames: {self.total_frames}")
        
        # Skip initial frames if requested
        for _ in range(frame_skip):
            self.cap.read()
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame maintaining aspect ratio."""
        h, w = frame.shape[:2]
        
        if max(h, w) <= self.max_side:
            return frame
            
        if h > w:
            new_h = self.max_side
            new_w = int(w * self.max_side / h)
        else:
            new_w = self.max_side
            new_h = int(h * self.max_side / w)
            
        return cv2.resize(frame, (new_w, new_h))
    
    def read_frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """
        Generator yielding (frame_index, timestamp, frame).
        
        Yields:
            Tuple of (frame_index, timestamp_seconds, frame_array)
        """
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Downsample based on frame interval
            if frame_count % self.frame_interval == 0:
                # Resize frame
                frame = self._resize_frame(frame)
                
                # Calculate timestamp
                timestamp = frame_count / self.original_fps
                
                yield processed_count, timestamp, frame
                processed_count += 1
            
            frame_count += 1
    
    def get_frame_at_index(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get frame at specific index."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            return self._resize_frame(frame)
        return None
    
    def get_video_info(self) -> dict:
        """Get video metadata."""
        return {
            'path': self.video_path,
            'fps': self.original_fps,
            'total_frames': self.total_frames,
            'duration': self.total_frames / self.original_fps,
            'target_fps': self.target_fps,
            'frame_interval': self.frame_interval,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
    
    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
