#!/usr/bin/env python3
"""
Birds-Eye View Oriented Bounding Box Detection System
AIY Vision Kit Implementation with YOLOv5-based Architecture

This module implements a real-time object detection system optimized for
aerial/drone perspectives using oriented bounding boxes. The system is
designed to run on Google AIY Vision Kit with Raspberry Pi Zero W.

References:
    - Redmon, J., & Farhadi, A. (2017). YOLO9000: better, faster, stronger. 
      Proceedings of the IEEE conference on computer vision and pattern recognition.
    - Howard, A., et al. (2017). MobileNets: Efficient convolutional neural networks 
      for mobile vision applications. arXiv preprint arXiv:1704.04861.
    - Yang, X., et al. (2021). Learning high-precision bounding box for rotated 
      object detection via kullback-leibler divergence. arXiv preprint arXiv:2106.01883.
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import yaml
from picamera import PiCamera
from picamera.array import PiRGBArray
from gpiozero import LED, Button, Buzzer

# AIY Vision Kit imports
from aiy.vision.inference import CameraInference
from aiy.vision.models import object_detection
from aiy.leds import Leds, PrivacyLed


@dataclass
class OrientedBBox:
    """
    Oriented Bounding Box representation for BEV object detection.
    
    Attributes:
        center_x (float): Center x-coordinate in image space
        center_y (float): Center y-coordinate in image space
        width (float): Bounding box width
        height (float): Bounding box height
        angle (float): Rotation angle in degrees (-90 to 90)
        confidence (float): Detection confidence score
        class_id (int): Object class identifier
        class_name (str): Object class name
    """
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float
    confidence: float
    class_id: int
    class_name: str
    
    def get_corner_points(self) -> np.ndarray:
        """
        Calculate the four corner points of the oriented bounding box.
        
        Returns:
            np.ndarray: Array of shape (4, 2) containing corner coordinates
        """
        cos_a = np.cos(np.radians(self.angle))
        sin_a = np.sin(np.radians(self.angle))
        
        # Half dimensions
        hw, hh = self.width / 2, self.height / 2
        
        # Corner points in local coordinate system
        corners = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ])
        
        # Rotation matrix
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Rotate and translate corners
        rotated_corners = corners @ R.T
        rotated_corners[:, 0] += self.center_x
        rotated_corners[:, 1] += self.center_y
        
        return rotated_corners


class YOLOv5OBB:
    """
    YOLOv5-based Oriented Bounding Box detection model.
    
    This class implements a lightweight YOLOv5 variant optimized for
    oriented bounding box detection on Raspberry Pi Zero hardware.
    """
    
    def __init__(self, model_path: str, config: Dict):
        """
        Initialize the YOLOv5 OBB model.
        
        Args:
            model_path (str): Path to the trained model weights
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cpu')  # Pi Zero only supports CPU
        self.model_path = model_path
        self.input_size = tuple(config['input']['image_size'])
        self.class_names = config['classes']['names']
        self.num_classes = config['classes']['nc']
        
        # Detection thresholds
        self.conf_threshold = config['detection']['conf_threshold']
        self.iou_threshold = config['detection']['iou_threshold']
        self.max_detections = config['detection']['max_detections']
        
        # Initialize model
        self.model = self._load_model()
        self.model.eval()
        
        # Optimization for Pi Zero
        if config['hardware']['quantization']:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        logging.info(f"YOLOv5-OBB model loaded: {model_path}")
    
    def _load_model(self) -> torch.nn.Module:
        """
        Load the YOLOv5 OBB model architecture.
        
        Returns:
            torch.nn.Module: Loaded model
        """
        # Load pre-trained YOLOv5 model adapted for OBB
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
        
        # Modify the head for oriented bounding box prediction
        # Each detection now predicts: [x, y, w, h, angle, conf, class_probs...]
        num_outputs = 6 + self.num_classes  # x,y,w,h,angle,conf + classes
        model.model[-1].nc = self.num_classes
        model.model[-1].no = num_outputs
        
        # Load custom weights if available
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model'], strict=False)
            logging.info(f"Loaded custom weights from {self.model_path}")
        
        return model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess input image for model inference.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(image_rgb, self.input_size)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array(self.config['input']['mean'])
        std = np.array(self.config['input']['std'])
        normalized = (normalized - mean) / std
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor
    
    def postprocess(self, predictions: torch.Tensor, 
                   original_shape: Tuple[int, int]) -> List[OrientedBBox]:
        """
        Post-process model predictions to extract oriented bounding boxes.
        
        Args:
            predictions (torch.Tensor): Raw model predictions
            original_shape (Tuple[int, int]): Original image shape (H, W)
            
        Returns:
            List[OrientedBBox]: List of detected oriented bounding boxes
        """
        detections = []
        predictions = predictions[0]  # Remove batch dimension
        
        # Filter by confidence threshold
        conf_mask = predictions[:, 5] >= self.conf_threshold
        predictions = predictions[conf_mask]
        
        if len(predictions) == 0:
            return detections
        
        # Scale coordinates back to original image size
        scale_x = original_shape[1] / self.input_size[0]
        scale_y = original_shape[0] / self.input_size[1]
        
        for pred in predictions:
            x, y, w, h, angle, conf = pred[:6].cpu().numpy()
            class_scores = pred[6:].cpu().numpy()
            class_id = np.argmax(class_scores)
            
            # Scale coordinates
            x *= scale_x
            y *= scale_y
            w *= scale_x
            h *= scale_y
            
            # Convert angle from radians to degrees
            angle_deg = np.degrees(angle)
            
            # Create oriented bounding box
            obb = OrientedBBox(
                center_x=float(x),
                center_y=float(y),
                width=float(w),
                height=float(h),
                angle=float(angle_deg),
                confidence=float(conf),
                class_id=int(class_id),
                class_name=self.class_names[class_id]
            )
            
            detections.append(obb)
        
        # Apply Non-Maximum Suppression for oriented boxes
        detections = self._obb_nms(detections)
        
        return detections[:self.max_detections]
    
    def _obb_nms(self, detections: List[OrientedBBox]) -> List[OrientedBBox]:
        """
        Apply Non-Maximum Suppression for oriented bounding boxes.
        
        Args:
            detections (List[OrientedBBox]): List of detections
            
        Returns:
            List[OrientedBBox]: Filtered detections after NMS
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [
                det for det in detections
                if self._obb_iou(current, det) < self.iou_threshold
            ]
        
        return keep
    
    def _obb_iou(self, obb1: OrientedBBox, obb2: OrientedBBox) -> float:
        """
        Calculate Intersection over Union for oriented bounding boxes.
        
        Args:
            obb1 (OrientedBBox): First oriented bounding box
            obb2 (OrientedBBox): Second oriented bounding box
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Get corner points
        corners1 = obb1.get_corner_points()
        corners2 = obb2.get_corner_points()
        
        # Use Sutherland-Hodgman clipping algorithm for polygon intersection
        intersection_area = self._polygon_intersection_area(corners1, corners2)
        
        # Calculate areas
        area1 = obb1.width * obb1.height
        area2 = obb2.width * obb2.height
        
        # Calculate IoU
        union_area = area1 + area2 - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _polygon_intersection_area(self, poly1: np.ndarray, 
                                 poly2: np.ndarray) -> float:
        """
        Calculate intersection area between two polygons.
        
        Args:
            poly1 (np.ndarray): First polygon vertices
            poly2 (np.ndarray): Second polygon vertices
            
        Returns:
            float: Intersection area
        """
        from shapely.geometry import Polygon
        
        try:
            p1 = Polygon(poly1)
            p2 = Polygon(poly2)
            intersection = p1.intersection(p2)
            return intersection.area
        except:
            # Fallback to approximate calculation
            return 0.0
    
    def detect(self, image: np.ndarray) -> List[OrientedBBox]:
        """
        Perform object detection on input image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            List[OrientedBBox]: List of detected objects
        """
        original_shape = image.shape[:2]
        
        # Preprocess image
        input_tensor = self.preprocess(image)
        
        # Model inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Post-process predictions
        detections = self.postprocess(predictions, original_shape)
        
        return detections


class BEVOBBDetector:
    """
    Main Birds-Eye View Oriented Bounding Box Detection System.
    
    This class integrates the YOLOv5-OBB model with the AIY Vision Kit
    hardware components for real-time object detection and tracking.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the BEV-OBB detection system.
        
        Args:
            config_path (str): Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize hardware components
        self._init_hardware()
        
        # Initialize model
        model_path = self.config.get('model_path', 'models/yolov5s_obb.pt')
        self.detector = YOLOv5OBB(model_path, self.config)
        
        # Detection state
        self.is_running = False
        self.detection_count = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Output directory
        self.output_dir = Path(self.config['logging']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info("BEV-OBB Detection System initialized")
    
    def _init_hardware(self):
        """Initialize AIY Vision Kit hardware components."""
        # Camera
        self.camera = PiCamera()
        self.camera.resolution = tuple(self.config['aiy']['camera_resolution'])
        self.camera.framerate = self.config['aiy']['framerate']
        self.camera.rotation = 180  # Adjust based on mounting orientation
        
        # GPIO components
        self.led = LED(self.config['aiy']['led_pin'])
        self.button = Button(self.config['aiy']['button_pin'])
        self.buzzer = Buzzer(self.config['aiy']['buzzer_pin'])
        
        # Privacy LED
        self.privacy_led = PrivacyLed()
        
        # Setup button callback
        self.button.when_pressed = self._button_callback
        
        logging.info("Hardware components initialized")
    
    def _button_callback(self):
        """Handle button press events."""
        if self.is_running:
            logging.info("Stopping detection (button pressed)")
            self.stop_detection()
        else:
            logging.info("Starting detection (button pressed)")
            self.start_detection()
    
    def start_detection(self):
        """Start the detection process."""
        if self.is_running:
            logging.warning("Detection already running")
            return
        
        self.is_running = True
        self.privacy_led.on()
        self.led.on()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        logging.info("Detection started")
    
    def stop_detection(self):
        """Stop the detection process."""
        self.is_running = False
        self.privacy_led.off()
        self.led.off()
        
        if hasattr(self, 'detection_thread'):
            self.detection_thread.join(timeout=5.0)
        
        logging.info("Detection stopped")
    
    def _detection_loop(self):
        """Main detection loop."""
        raw_capture = PiRGBArray(self.camera, size=self.camera.resolution)
        
        try:
            for frame in self.camera.capture_continuous(
                raw_capture, format="bgr", use_video_port=True
            ):
                if not self.is_running:
                    break
                
                # Get image array
                image = frame.array
                self.frame_count += 1
                
                # Perform detection
                detections = self.detector.detect(image)
                self.detection_count += len(detections)
                
                # Process detections
                self._process_detections(image, detections)
                
                # Clear the stream for next frame
                raw_capture.truncate(0)
                
                # LED feedback
                if detections:
                    self.led.blink(on_time=0.1, off_time=0.1, n=1, background=True)
                
                # Performance monitoring
                if self.frame_count % 30 == 0:
                    self._log_performance()
        
        except Exception as e:
            logging.error(f"Detection loop error: {e}")
        finally:
            raw_capture.close()
    
    def _process_detections(self, image: np.ndarray, 
                          detections: List[OrientedBBox]):
        """
        Process and log detection results.
        
        Args:
            image (np.ndarray): Original image
            detections (List[OrientedBBox]): Detected objects
        """
        if not detections:
            return
        
        # Draw detections on image
        annotated_image = self._draw_detections(image.copy(), detections)
        
        # Save frame if enabled
        if self.config['logging']['save_frames']:
            frame_path = self.output_dir / f"frame_{self.frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated_image)
        
        # Save detection data
        if self.config['logging']['save_detections']:
            self._save_detection_data(detections)
        
        # Log detection summary
        class_counts = {}
        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        logging.info(f"Frame {self.frame_count}: {len(detections)} objects detected - {class_counts}")
    
    def _draw_detections(self, image: np.ndarray, 
                        detections: List[OrientedBBox]) -> np.ndarray:
        """
        Draw oriented bounding boxes on image.
        
        Args:
            image (np.ndarray): Input image
            detections (List[OrientedBBox]): Detections to draw
            
        Returns:
            np.ndarray: Annotated image
        """
        for det in detections:
            # Get corner points
            corners = det.get_corner_points().astype(np.int32)
            
            # Draw oriented bounding box
            cv2.polylines(image, [corners], True, (0, 255, 0), 2)
            
            # Draw center point
            center = (int(det.center_x), int(det.center_y))
            cv2.circle(image, center, 3, (0, 0, 255), -1)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, 
                         (center[0], center[1] - label_size[1] - 10),
                         (center[0] + label_size[0], center[1]), 
                         (255, 255, 255), -1)
            cv2.putText(image, label, (center[0], center[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image
    
    def _save_detection_data(self, detections: List[OrientedBBox]):
        """
        Save detection data to file.
        
        Args:
            detections (List[OrientedBBox]): Detections to save
        """
        detection_file = self.output_dir / "detections.txt"
        
        with open(detection_file, 'a') as f:
            timestamp = time.time()
            for det in detections:
                f.write(f"{timestamp},{self.frame_count},{det.class_name},"
                       f"{det.center_x:.2f},{det.center_y:.2f},"
                       f"{det.width:.2f},{det.height:.2f},{det.angle:.2f},"
                       f"{det.confidence:.4f}\n")
    
    def _log_performance(self):
        """Log performance metrics."""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        logging.info(f"Performance: {fps:.2f} FPS, "
                    f"{self.detection_count} total detections, "
                    f"{self.frame_count} frames processed")
    
    def run(self):
        """Run the detection system."""
        logging.info("BEV-OBB Detection System starting...")
        logging.info("Press the button to start/stop detection")
        logging.info("Press Ctrl+C to exit")
        
        try:
            # Wait for button press or run automatically
            if self.config.get('auto_start', False):
                self.start_detection()
            
            # Keep main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        finally:
            self.stop_detection()
            self.camera.close()


def main():
    """Main entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/bev_obb_config.yaml"
    
    try:
        detector = BEVOBBDetector(config_path)
        detector.run()
    except Exception as e:
        logging.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()