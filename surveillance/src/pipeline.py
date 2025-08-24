# FILE: src/pipeline.py

"""
Main surveillance processing pipeline.
"""

import argparse
import logging
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import numpy as np

from .ingest import VideoReader
from .detect import DetectorManager
from .track import MultiTracker
from .events import LoiteringDetector, AbandonmentDetector, UnusualMovementDetector
from .fuse_score import AlertManager
from .visualize import FrameVisualizer
from .logger import SurveillanceLogger
from .utils.timers import FPSCounter, ProfilerManager


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-JSON serializable types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


class SurveillancePipeline:
    """Main surveillance processing pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize surveillance pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.detector = DetectorManager(self.config)
        self.tracker = MultiTracker(
            max_age=self.config['tracker']['max_age'],
            iou_threshold=self.config['tracker']['iou_match'],
            distance_threshold=self.config['tracker']['dist_match'],
            smoothing=self.config['tracker']['smoothing']
        )
        
        # Event detectors
        self.loitering_detector = LoiteringDetector(self.config)
        self.abandonment_detector = AbandonmentDetector(self.config)
        self.unusual_detector = UnusualMovementDetector(self.config)
        
        # Alert manager
        self.alert_manager = AlertManager(self.config)
        
        # Visualizer
        self.visualizer = FrameVisualizer(self.config)
        
        # Performance monitoring
        self.fps_counter = FPSCounter()
        self.profiler = ProfilerManager()
        
        logger.info("Surveillance pipeline initialized")
    
    def process_video(self, video_path: str, save_dir: str, 
                     show_video: bool = False, save_video: bool = False,
                     avenue_pred_id: Optional[str] = None) -> Dict:
        """
        Process a video file.
        
        Args:
            video_path: Path to input video
            save_dir: Directory to save outputs
            show_video: Whether to display video during processing
            save_video: Whether to save processed video
            avenue_pred_id: ID for Avenue dataset predictions
            
        Returns:
            Processing results dictionary
        """
        # Setup paths
        os.makedirs(save_dir, exist_ok=True)
        video_name = Path(video_path).stem
        
        # Initialize logger
        save_images = self.config.get('output', {}).get('save_frames', True)
        surveillance_logger = SurveillanceLogger(save_dir, video_name, save_images)
        
        # Initialize video reader
        video_config = self.config['video']
        reader = VideoReader(
            video_path,
            target_fps=video_config['target_fps'],
            max_side=video_config['max_side'],
            frame_skip=video_config['frame_skip']
        )
        
        # Get video info
        video_info = reader.get_video_info()
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Video info: {video_info}")
        
        # Setup video writer if saving
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_path = os.path.join(save_dir, f"{video_name}_processed.mp4")
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, video_config['target_fps'],
                (video_config['max_side'], video_config['max_side'])
            )
        
        # Avenue dataset predictions storage
        avenue_predictions = {}
        
        try:
            # Process frames
            frame_count = 0
            last_timestamp = 0.0
            
            for frame_idx, timestamp, frame in reader.read_frames():
                with self.profiler.timer('total_frame'):
                    # Calculate time delta
                    time_delta = timestamp - last_timestamp if frame_count > 0 else 0.0
                    last_timestamp = timestamp
                    
                    # Object detection
                    with self.profiler.timer('detection'):
                        detections = self.detector.detect(frame)
                    
                    # Object tracking
                    with self.profiler.timer('tracking'):
                        tracks = self.tracker.update(detections, frame_idx)
                    
                    # Event detection
                    event_scores = []
                    
                    # Update ROI time for loitering
                    self.loitering_detector.update_tracks_roi_time(tracks, time_delta)
                    
                    with self.profiler.timer('loitering'):
                        loitering_events = self.loitering_detector.detect_loitering(tracks, frame_idx)
                        event_scores.extend(loitering_events)
                    
                    with self.profiler.timer('abandonment'):
                        abandonment_events = self.abandonment_detector.detect_abandonment(tracks, frame_idx)
                        event_scores.extend(abandonment_events)
                    
                    unusual_boxes = []
                    with self.profiler.timer('unusual'):
                        unusual_events, unusual_boxes = self.unusual_detector.detect_unusual_movement(frame, frame_idx)
                        event_scores.extend(unusual_events)
                    
                    # Alert generation
                    with self.profiler.timer('alerts'):
                        alerts = self.alert_manager.process_frame_events(frame_idx, event_scores)
                    
                    # Visualization
                    with self.profiler.timer('visualization'):
                        vis_frame = self.visualizer.visualize_frame(
                            frame, tracks, alerts, frame_idx, timestamp,
                            unusual_boxes, unusual_events[0].raw_score if unusual_events else 0.0
                        )
                    
                    # Logging
                    if alerts:
                        surveillance_logger.log_frame_alerts(vis_frame, frame_idx, timestamp, alerts)
                    
                    # Avenue predictions
                    if avenue_pred_id is not None:
                        # Store all detection boxes for this frame
                        frame_boxes = []
                        for track in tracks:
                            frame_boxes.append(track.bbox)
                        for box in unusual_boxes:
                            frame_boxes.append(box)
                        avenue_predictions[f"{frame_idx:06d}"] = frame_boxes
                    
                    # Display/save video
                    if show_video:
                        cv2.imshow('Surveillance', vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    if video_writer:
                        # Resize frame if necessary
                        h, w = vis_frame.shape[:2]
                        if h != video_config['max_side'] or w != video_config['max_side']:
                            vis_frame = cv2.resize(vis_frame, (video_config['max_side'], video_config['max_side']))
                        video_writer.write(vis_frame)
                    
                    # Update FPS counter
                    fps = self.fps_counter.update()
                    
                    frame_count += 1
                    
                    # Progress logging
                    if frame_count % 100 == 0:
                        logger.info(f"Processed {frame_count} frames, FPS: {fps:.1f}")
                    
                    # Cleanup old tracks
                    if frame_count % 50 == 0:
                        active_track_ids = [t.id for t in tracks]
                        self.abandonment_detector.cleanup_old_ownership(active_track_ids, frame_idx)
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise
        finally:
            # Cleanup
            reader.release()
            if video_writer:
                video_writer.release()
            if show_video:
                cv2.destroyAllWindows()
        
        # Save Avenue predictions
        if avenue_pred_id is not None:
            avenue_pred_path = os.path.join(save_dir, f"{avenue_pred_id}.json")
            with open(avenue_pred_path, 'w') as f:
                json.dump(avenue_predictions, f, indent=2)
            logger.info(f"Saved Avenue predictions: {avenue_pred_path}")
        
        # Get processing summary
        summary_stats = surveillance_logger.get_summary()
        processing_stats = {
            'frames_processed': frame_count,
            'processing_fps': fps,
            'video_info': video_info,
            'performance': self.profiler.get_stats()
        }
        
        # Print performance stats
        self.profiler.print_stats()
        
        logger.info(f"Processing completed: {frame_count} frames")
        logger.info(f"Alert summary: {summary_stats}")
        
        return convert_to_json_serializable({
            'video_path': video_path,
            'save_dir': save_dir,
            'frames_processed': frame_count,
            'alerts': summary_stats,
            'performance': processing_stats,
            'csv_path': surveillance_logger.get_csv_path()
        })


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Surveillance Anomaly Detection Pipeline')
    
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--save_dir', default='data/output', help='Output directory')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--show', action='store_true', help='Show video during processing')
    parser.add_argument('--save_video', action='store_true', help='Save processed video')
    parser.add_argument('--avenue_pred_id', help='Avenue dataset prediction ID')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SurveillancePipeline(args.config)
    
    # Process video
    results = pipeline.process_video(
        video_path=args.video,
        save_dir=args.save_dir,
        show_video=args.show,
        save_video=args.save_video,
        avenue_pred_id=args.avenue_pred_id
    )
    
    print(f"\nProcessing completed successfully!")
    print(f"Results saved to: {args.save_dir}")
    print(f"CSV log: {results['csv_path']}")


if __name__ == "__main__":
    main()
