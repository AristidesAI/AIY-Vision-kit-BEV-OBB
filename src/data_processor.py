#!/usr/bin/env python3
"""
Data Processing and Logging System for BEV-OBB Detection

This module handles data collection, preprocessing, annotation, and logging
for the Birds-Eye View Oriented Bounding Box detection system running on
the AIY Vision Kit.

References:
    - Lin, T. Y., et al. (2014). Microsoft coco: Common objects in context. ECCV.
    - Everingham, M., et al. (2010). The pascal visual object classes (voc) challenge. IJCV.
    - Deng, J., et al. (2009). Imagenet: A large-scale hierarchical image database. CVPR.
"""

import os
import json
import csv
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import yaml

from src.bev_obb_detector import OrientedBBox


@dataclass
class DetectionRecord:
    """
    Complete detection record with metadata.
    
    Attributes:
        timestamp (float): Unix timestamp
        frame_id (int): Frame sequence number
        detection_id (str): Unique detection identifier
        bbox (OrientedBBox): Oriented bounding box
        metadata (Dict): Additional metadata
    """
    timestamp: float
    frame_id: int
    detection_id: str
    bbox: OrientedBBox
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['bbox'] = asdict(self.bbox)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionRecord':
        """Create from dictionary."""
        bbox_data = data.pop('bbox')
        bbox = OrientedBBox(**bbox_data)
        return cls(bbox=bbox, **data)


@dataclass
class SessionMetadata:
    """
    Session metadata for tracking detection runs.
    
    Attributes:
        session_id (str): Unique session identifier
        start_time (float): Session start timestamp
        end_time (Optional[float]): Session end timestamp
        config (Dict): Session configuration
        hardware_info (Dict): Hardware information
        performance_metrics (Dict): Performance statistics
    """
    session_id: str
    start_time: float
    end_time: Optional[float]
    config: Dict[str, Any]
    hardware_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class DataProcessor:
    """
    Comprehensive data processing system for BEV-OBB detection.
    
    This class handles data collection, preprocessing, augmentation,
    and format conversion for training and evaluation datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data directories
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.annotations_dir = self.data_dir / 'annotations'
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.annotations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.image_size = tuple(config.get('image_size', [416, 416]))
        self.augmentation_enabled = config.get('enable_augmentation', True)
        
    def preprocess_image(self, image: np.ndarray,
                        target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image (np.ndarray): Input image in BGR format
            target_size (Tuple[int, int]): Target size for resizing
            
        Returns:
            np.ndarray: Preprocessed image
        """
        if target_size is None:
            target_size = self.image_size
        
        # Convert to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize with aspect ratio preservation
        h, w = image_rgb.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        processed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        
        # Place resized image in center
        processed[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        
        return processed
    
    def augment_image(self, image: np.ndarray, 
                     bboxes: List[OrientedBBox]) -> Tuple[np.ndarray, List[OrientedBBox]]:
        """
        Apply data augmentation to image and bounding boxes.
        
        Args:
            image (np.ndarray): Input image
            bboxes (List[OrientedBBox]): Original bounding boxes
            
        Returns:
            Tuple[np.ndarray, List[OrientedBBox]]: Augmented image and boxes
        """
        if not self.augmentation_enabled:
            return image, bboxes
        
        # Random horizontal flip
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)
            # Flip bounding boxes
            h, w = image.shape[:2]
            for bbox in bboxes:
                bbox.center_x = w - bbox.center_x
                bbox.angle = -bbox.angle  # Flip angle
        
        # Random brightness adjustment
        if np.random.random() < 0.3:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if np.random.random() < 0.3:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        # Random rotation (small angles to preserve BEV perspective)
        if np.random.random() < 0.2:
            angle = np.random.uniform(-5, 5)  # Small rotation angles
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            
            # Rotate bounding boxes
            for bbox in bboxes:
                # Transform center point
                center_point = np.array([[bbox.center_x, bbox.center_y, 1]]).T
                new_center = M @ center_point
                bbox.center_x = new_center[0, 0]
                bbox.center_y = new_center[1, 0]
                
                # Adjust angle
                bbox.angle += angle
        
        return image, bboxes
    
    def convert_to_yolo_format(self, image_shape: Tuple[int, int],
                              bboxes: List[OrientedBBox]) -> List[str]:
        """
        Convert oriented bounding boxes to YOLO format.
        
        Args:
            image_shape (Tuple[int, int]): Image dimensions (H, W)
            bboxes (List[OrientedBBox]): Oriented bounding boxes
            
        Returns:
            List[str]: YOLO format annotation lines
        """
        h, w = image_shape
        yolo_lines = []
        
        for bbox in bboxes:
            # Normalize coordinates
            x_center = bbox.center_x / w
            y_center = bbox.center_y / h
            width = bbox.width / w
            height = bbox.height / h
            
            # Normalize angle to [0, 1]
            angle_norm = (bbox.angle + 180) / 360
            
            # YOLO OBB format: class_id x_center y_center width height angle
            line = f"{bbox.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {angle_norm:.6f}"
            yolo_lines.append(line)
        
        return yolo_lines
    
    def convert_to_coco_format(self, detections: List[DetectionRecord],
                              image_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert detections to COCO format.
        
        Args:
            detections (List[DetectionRecord]): Detection records
            image_info (Dict): Image information
            
        Returns:
            Dict: COCO format annotation
        """
        coco_annotations = []
        
        for i, detection in enumerate(detections):
            bbox = detection.bbox
            
            # Get corner points
            corners = bbox.get_corner_points()
            
            # Calculate bounding rectangle for COCO compatibility
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            annotation = {
                'id': i + 1,
                'image_id': image_info['id'],
                'category_id': bbox.class_id + 1,  # COCO categories start from 1
                'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                'area': bbox.width * bbox.height,
                'iscrowd': 0,
                'segmentation': [corners.flatten().tolist()],  # Polygon segmentation
                'attributes': {
                    'oriented_bbox': {
                        'center': [bbox.center_x, bbox.center_y],
                        'size': [bbox.width, bbox.height],
                        'angle': bbox.angle
                    },
                    'confidence': bbox.confidence
                }
            }
            
            coco_annotations.append(annotation)
        
        return {
            'images': [image_info],
            'annotations': coco_annotations,
            'categories': [
                {'id': i + 1, 'name': name} 
                for i, name in enumerate(self.config.get('class_names', []))
            ]
        }
    
    def save_dataset(self, dataset: List[Tuple[np.ndarray, List[OrientedBBox]]],
                    output_dir: str, format_type: str = 'yolo'):
        """
        Save dataset in specified format.
        
        Args:
            dataset (List): Dataset samples
            output_dir (str): Output directory
            format_type (str): Output format ('yolo', 'coco', 'pascal')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'yolo':
            self._save_yolo_dataset(dataset, output_path)
        elif format_type == 'coco':
            self._save_coco_dataset(dataset, output_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        self.logger.info(f"Dataset saved in {format_type} format to {output_dir}")
    
    def _save_yolo_dataset(self, dataset: List[Tuple[np.ndarray, List[OrientedBBox]]],
                          output_path: Path):
        """Save dataset in YOLO format."""
        images_dir = output_path / 'images'
        labels_dir = output_path / 'labels'
        
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        for i, (image, bboxes) in enumerate(dataset):
            # Save image
            image_filename = f"image_{i:06d}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), image)
            
            # Save labels
            label_filename = f"image_{i:06d}.txt"
            label_path = labels_dir / label_filename
            
            yolo_lines = self.convert_to_yolo_format(image.shape[:2], bboxes)
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
    
    def _save_coco_dataset(self, dataset: List[Tuple[np.ndarray, List[OrientedBBox]]],
                          output_path: Path):
        """Save dataset in COCO format."""
        images_dir = output_path / 'images'
        images_dir.mkdir(exist_ok=True)
        
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': i + 1, 'name': name} 
                for i, name in enumerate(self.config.get('class_names', []))
            ]
        }
        
        annotation_id = 1
        
        for i, (image, bboxes) in enumerate(dataset):
            # Save image
            image_filename = f"image_{i:06d}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), image)
            
            # Image info
            h, w = image.shape[:2]
            image_info = {
                'id': i + 1,
                'file_name': image_filename,
                'width': w,
                'height': h
            }
            coco_data['images'].append(image_info)
            
            # Annotations
            for bbox in bboxes:
                corners = bbox.get_corner_points()
                x_coords = corners[:, 0]
                y_coords = corners[:, 1]
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                
                annotation = {
                    'id': annotation_id,
                    'image_id': i + 1,
                    'category_id': bbox.class_id + 1,
                    'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                    'area': bbox.width * bbox.height,
                    'iscrowd': 0,
                    'segmentation': [corners.flatten().tolist()],
                    'attributes': {
                        'oriented_bbox': {
                            'center': [bbox.center_x, bbox.center_y],
                            'size': [bbox.width, bbox.height],
                            'angle': bbox.angle
                        }
                    }
                }
                coco_data['annotations'].append(annotation)
                annotation_id += 1
        
        # Save annotations
        annotations_path = output_path / 'annotations.json'
        with open(annotations_path, 'w') as f:
            json.dump(coco_data, f, indent=2)


class DetectionLogger:
    """
    Comprehensive logging system for detection results and system metrics.
    
    This class handles real-time logging of detection results, performance
    metrics, and system status for analysis and debugging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detection logger.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'logs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Database connection
        self.db_path = self.output_dir / 'detections.db'
        self._init_database()
        
        # Session management
        self.current_session = None
        self.session_start_time = None
        
        # Performance tracking
        self.performance_metrics = {
            'frames_processed': 0,
            'detections_made': 0,
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': []
        }
        
        # CSV logging
        self.csv_path = self.output_dir / 'detections.csv'
        self._init_csv_logging()
        
        # Thread-safe logging
        self.log_lock = threading.Lock()
        
    def _init_database(self):
        """Initialize SQLite database for logging."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time REAL,
                    end_time REAL,
                    config TEXT,
                    hardware_info TEXT,
                    performance_metrics TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp REAL,
                    frame_id INTEGER,
                    detection_id TEXT,
                    class_name TEXT,
                    class_id INTEGER,
                    center_x REAL,
                    center_y REAL,
                    width REAL,
                    height REAL,
                    angle REAL,
                    confidence REAL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp REAL,
                    metric_name TEXT,
                    metric_value REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            conn.commit()
    
    def _init_csv_logging(self):
        """Initialize CSV logging."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 'session_id', 'frame_id', 'detection_id',
                    'class_name', 'class_id', 'center_x', 'center_y',
                    'width', 'height', 'angle', 'confidence', 'metadata'
                ])
    
    def start_session(self, config: Dict[str, Any]) -> str:
        """
        Start a new logging session.
        
        Args:
            config (Dict): Session configuration
            
        Returns:
            str: Session ID
        """
        with self.log_lock:
            session_id = f"session_{int(time.time())}"
            self.current_session = session_id
            self.session_start_time = time.time()
            
            # Reset performance metrics
            self.performance_metrics = {
                'frames_processed': 0,
                'detections_made': 0,
                'processing_times': [],
                'memory_usage': [],
                'cpu_usage': []
            }
            
            # Hardware info
            hardware_info = self._get_hardware_info()
            
            # Save session to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO sessions (session_id, start_time, config, hardware_info)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, self.session_start_time, 
                     json.dumps(config), json.dumps(hardware_info)))
                conn.commit()
            
            self.logger.info(f"Started logging session: {session_id}")
            return session_id
    
    def end_session(self):
        """End the current logging session."""
        if self.current_session is None:
            return
        
        with self.log_lock:
            end_time = time.time()
            
            # Calculate final performance metrics
            session_duration = end_time - self.session_start_time
            avg_fps = self.performance_metrics['frames_processed'] / session_duration
            
            final_metrics = {
                'session_duration': session_duration,
                'avg_fps': avg_fps,
                'total_detections': self.performance_metrics['detections_made'],
                'avg_processing_time': np.mean(self.performance_metrics['processing_times'])
                                     if self.performance_metrics['processing_times'] else 0
            }
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE sessions 
                    SET end_time = ?, performance_metrics = ?
                    WHERE session_id = ?
                ''', (end_time, json.dumps(final_metrics), self.current_session))
                conn.commit()
            
            self.logger.info(f"Ended logging session: {self.current_session}")
            self.logger.info(f"Session metrics: {final_metrics}")
            
            self.current_session = None
    
    def log_detection(self, detection: DetectionRecord):
        """
        Log a detection record.
        
        Args:
            detection (DetectionRecord): Detection to log
        """
        with self.log_lock:
            if self.current_session is None:
                self.logger.warning("No active session for logging detection")
                return
            
            # Database logging
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO detections 
                    (session_id, timestamp, frame_id, detection_id, class_name, class_id,
                     center_x, center_y, width, height, angle, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.current_session, detection.timestamp, detection.frame_id,
                    detection.detection_id, detection.bbox.class_name, detection.bbox.class_id,
                    detection.bbox.center_x, detection.bbox.center_y,
                    detection.bbox.width, detection.bbox.height, detection.bbox.angle,
                    detection.bbox.confidence, json.dumps(detection.metadata)
                ))
                conn.commit()
            
            # CSV logging
            with open(self.csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    detection.timestamp, self.current_session, detection.frame_id,
                    detection.detection_id, detection.bbox.class_name, detection.bbox.class_id,
                    detection.bbox.center_x, detection.bbox.center_y,
                    detection.bbox.width, detection.bbox.height, detection.bbox.angle,
                    detection.bbox.confidence, json.dumps(detection.metadata)
                ])
            
            # Update metrics
            self.performance_metrics['detections_made'] += 1
    
    def log_performance_metric(self, metric_name: str, metric_value: float):
        """
        Log a performance metric.
        
        Args:
            metric_name (str): Name of the metric
            metric_value (float): Metric value
        """
        with self.log_lock:
            if self.current_session is None:
                return
            
            timestamp = time.time()
            
            # Database logging
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO performance (session_id, timestamp, metric_name, metric_value)
                    VALUES (?, ?, ?, ?)
                ''', (self.current_session, timestamp, metric_name, metric_value))
                conn.commit()
            
            # Update in-memory metrics
            if metric_name == 'processing_time':
                self.performance_metrics['processing_times'].append(metric_value)
            elif metric_name == 'memory_usage':
                self.performance_metrics['memory_usage'].append(metric_value)
            elif metric_name == 'cpu_usage':
                self.performance_metrics['cpu_usage'].append(metric_value)
    
    def log_frame_processed(self):
        """Log that a frame has been processed."""
        with self.log_lock:
            self.performance_metrics['frames_processed'] += 1
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for a session.
        
        Args:
            session_id (str): Session ID (current session if None)
            
        Returns:
            Dict: Session summary
        """
        if session_id is None:
            session_id = self.current_session
        
        if session_id is None:
            return {}
        
        with sqlite3.connect(self.db_path) as conn:
            # Session info
            session_info = conn.execute('''
                SELECT * FROM sessions WHERE session_id = ?
            ''', (session_id,)).fetchone()
            
            if not session_info:
                return {}
            
            # Detection counts by class
            detection_counts = conn.execute('''
                SELECT class_name, COUNT(*) as count
                FROM detections WHERE session_id = ?
                GROUP BY class_name
            ''', (session_id,)).fetchall()
            
            # Average confidence by class
            avg_confidence = conn.execute('''
                SELECT class_name, AVG(confidence) as avg_conf
                FROM detections WHERE session_id = ?
                GROUP BY class_name
            ''', (session_id,)).fetchall()
            
            summary = {
                'session_id': session_id,
                'start_time': session_info[1],
                'end_time': session_info[2],
                'detection_counts': {row[0]: row[1] for row in detection_counts},
                'average_confidence': {row[0]: row[1] for row in avg_confidence},
                'total_detections': sum(count for _, count in detection_counts),
                'performance_metrics': json.loads(session_info[5]) if session_info[5] else {}
            }
            
            return summary
    
    def export_session_data(self, session_id: str, 
                           output_path: str, format_type: str = 'json'):
        """
        Export session data in specified format.
        
        Args:
            session_id (str): Session ID to export
            output_path (str): Output file path
            format_type (str): Export format ('json', 'csv', 'xlsx')
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get all detections for session
            detections = conn.execute('''
                SELECT * FROM detections WHERE session_id = ?
                ORDER BY timestamp
            ''', (session_id,)).fetchall()
            
            if format_type == 'json':
                data = []
                for detection in detections:
                    data.append({
                        'timestamp': detection[2],
                        'frame_id': detection[3],
                        'detection_id': detection[4],
                        'class_name': detection[5],
                        'class_id': detection[6],
                        'center_x': detection[7],
                        'center_y': detection[8],
                        'width': detection[9],
                        'height': detection[10],
                        'angle': detection[11],
                        'confidence': detection[12],
                        'metadata': json.loads(detection[13]) if detection[13] else {}
                    })
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format_type == 'csv':
                df = pd.DataFrame(detections, columns=[
                    'id', 'session_id', 'timestamp', 'frame_id', 'detection_id',
                    'class_name', 'class_id', 'center_x', 'center_y',
                    'width', 'height', 'angle', 'confidence', 'metadata'
                ])
                df.to_csv(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
        
        self.logger.info(f"Session data exported to {output_path}")
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_usage': psutil.disk_usage('/').total
        }


def main():
    """Example usage of data processing and logging systems."""
    # Configuration
    config = {
        'data_dir': 'data',
        'output_dir': 'logs',
        'image_size': [416, 416],
        'enable_augmentation': True,
        'class_names': ['vehicle', 'person', 'bicycle', 'motorcycle', 'bus', 'truck']
    }
    
    # Initialize systems
    processor = DataProcessor(config)
    logger = DetectionLogger(config)
    
    # Start logging session
    session_id = logger.start_session(config)
    
    print(f"Data processing and logging systems initialized")
    print(f"Session ID: {session_id}")


if __name__ == "__main__":
    main()