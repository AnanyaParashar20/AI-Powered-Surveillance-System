# FILE: src/detect.py

"""
Object detection using YOLOv8 with HOG person fallback.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Detection:
    """Single detection result."""
    
    def __init__(self, bbox: List[float], confidence: float, class_id: int, class_name: str):
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name


class YOLODetector:
    """YOLOv8 object detector."""
    
    def __init__(self, weights_path: str = "yolov8n.pt", conf_threshold: float = 0.35,
                 person_labels: List = [0, 'person'], 
                 object_labels: List[str] = ['backpack', 'handbag', 'suitcase', 'bottle', 'cell phone']):
        """
        Initialize YOLO detector.
        
        Args:
            weights_path: Path to YOLO weights
            conf_threshold: Confidence threshold
            person_labels: Person class labels [id, name]
            object_labels: Object class names to track
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.available = True
            logger.info(f"YOLO model loaded: {weights_path}")
        except ImportError:
            logger.warning("Ultralytics not available, YOLO disabled")
            self.available = False
            return
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}")
            self.available = False
            return
            
        self.conf_threshold = conf_threshold
        self.person_class_id = person_labels[0]
        self.person_class_name = person_labels[1]
        self.object_labels = object_labels
        
        # Get YOLO class names
        self.class_names = self.model.names
        
        # Map object labels to class IDs
        self.object_class_ids = []
        for obj_name in object_labels:
            for class_id, class_name in self.class_names.items():
                if obj_name.lower() in class_name.lower():
                    self.object_class_ids.append(class_id)
                    break
        
        logger.info(f"Tracking person (ID {self.person_class_id}) and objects: {object_labels}")
        logger.info(f"Object class IDs: {self.object_class_ids}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of Detection objects
        """
        if not self.available:
            return []
            
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for i in range(len(boxes)):
                    # Get box coordinates and convert to xywh
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    
                    confidence = boxes.conf[i].cpu().numpy().item()
                    class_id = int(boxes.cls[i].cpu().numpy().item())
                    class_name = self.class_names[class_id]
                    
                    # Filter for person or tracked objects
                    if (class_id == self.person_class_id or 
                        class_id in self.object_class_ids):
                        
                        detections.append(Detection(
                            bbox=[x, y, w, h],
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        ))
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []


class HOGPersonDetector:
    """OpenCV HOG person detector fallback."""
    
    def __init__(self):
        """Initialize HOG detector."""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info("HOG person detector initialized")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons using HOG.
        
        Args:
            frame: Input frame
            
        Returns:
            List of Detection objects (persons only)
        """
        try:
            # Detect people
            boxes, weights = self.hog.detectMultiScale(
                frame,
                winStride=(8, 8),
                padding=(32, 32),
                scale=1.05
            )
            
            detections = []
            for i, (x, y, w, h) in enumerate(boxes):
                confidence = float(weights[i]) if len(weights) > i else 0.5
                
                detections.append(Detection(
                    bbox=[x, y, w, h],
                    confidence=confidence,
                    class_id=0,  # Person class
                    class_name='person'
                ))
            
            return detections
            
        except Exception as e:
            logger.error(f"HOG detection failed: {e}")
            return []


class DetectorManager:
    """Manages YOLO and HOG detectors with fallback."""
    
    def __init__(self, config: Dict):
        """
        Initialize detector manager.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        model_config = config.get('model', {})
        
        detector_type = model_config.get('detector', 'yolo')
        
        # Try to initialize YOLO first
        if detector_type == 'yolo':
            self.yolo = YOLODetector(
                weights_path=model_config.get('yolo_weights', 'yolov8n.pt'),
                conf_threshold=model_config.get('conf_thres', 0.35),
                person_labels=model_config.get('person_labels', [0, 'person']),
                object_labels=model_config.get('object_labels', ['backpack', 'handbag', 'suitcase', 'bottle', 'cell phone'])
            )
            
            if self.yolo.available:
                self.primary_detector = self.yolo
                logger.info("Using YOLO as primary detector")
            else:
                self.primary_detector = HOGPersonDetector()
                logger.info("Falling back to HOG detector")
        else:
            self.primary_detector = HOGPersonDetector()
            logger.info("Using HOG detector")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in frame using primary detector."""
        return self.primary_detector.detect(frame)
