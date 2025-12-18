"""
Multi-Camera Visitor Counting System with YOLOv11n and ByteTrack
Industry-standard implementation with interactive setup and ThingsBoard integration
"""

import cv2
import numpy as np
import json
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import requests
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CameraConfig:
    """Camera configuration handler"""
    def __init__(self, config_path='camera_config.json'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {'cameras': {}, 'settings': {'process_fps': 10, 'resize_width': 640, 'resize_height': 480}}
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_camera(self, url: str) -> Optional[Dict]:
        """Get camera configuration"""
        return self.config['cameras'].get(url)
    
    def set_camera(self, url: str, roi: List, lines: List[Dict]):
        """Set camera configuration"""
        self.config['cameras'][url] = {
            'roi': roi,
            'lines': lines,
            'thingsboard_url': ''
        }
        self.save_config()
    
    def needs_setup(self, url: str) -> bool:
        """Check if camera needs setup"""
        cam = self.get_camera(url)
        return cam is None or not cam.get('roi') or not cam.get('lines')


class InteractiveSetup:
    """Interactive setup for camera ROI and counting lines"""
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.roi_points = []
        self.lines = []
        self.current_line_points = []
        self.mode = 'roi'  # 'roi', 'line', 'direction'
        self.current_line_direction = None
        self.temp_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'roi':
                self.roi_points.append([x, y])
                logger.info(f"ROI point added: ({x}, {y})")
            elif self.mode == 'line':
                self.current_line_points.append([x, y])
                logger.info(f"Line point added: ({x}, {y})")
    
    def draw_setup(self, frame):
        """Draw current setup on frame"""
        display = frame.copy()
        
        # Draw ROI
        if len(self.roi_points) > 0:
            for i, pt in enumerate(self.roi_points):
                cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display, tuple(self.roi_points[i-1]), tuple(pt), (0, 255, 0), 2)
            if len(self.roi_points) > 2:
                cv2.line(display, tuple(self.roi_points[-1]), tuple(self.roi_points[0]), (0, 255, 0), 2)
        
        # Draw completed lines
        for idx, line_data in enumerate(self.lines):
            points = line_data['points']
            color = (255, 0, 0) if idx % 2 == 0 else (0, 0, 255)
            for i in range(len(points) - 1):
                cv2.line(display, tuple(points[i]), tuple(points[i+1]), color, 2)
            # Draw direction arrow
            mid_idx = len(points) // 2
            self._draw_direction_arrow(display, points[mid_idx], line_data['direction'], color)
            # Draw label
            cv2.putText(display, f"Line {idx+1}: {line_data['direction']}", 
                       tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw current line being drawn
        if len(self.current_line_points) > 0:
            for i, pt in enumerate(self.current_line_points):
                cv2.circle(display, tuple(pt), 5, (255, 255, 0), -1)
                if i > 0:
                    cv2.line(display, tuple(self.current_line_points[i-1]), tuple(pt), (255, 255, 0), 2)
        
        # Draw instructions
        instructions = self._get_instructions()
        y_offset = 30
        for instruction in instructions:
            cv2.putText(display, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return display
    
    def _draw_direction_arrow(self, img, point, direction, color):
        """Draw direction arrow"""
        arrow_length = 30
        if direction == 'NS':
            end_point = (point[0], point[1] + arrow_length)
        elif direction == 'SN':
            end_point = (point[0], point[1] - arrow_length)
        elif direction == 'WE':
            end_point = (point[0] + arrow_length, point[1])
        elif direction == 'EW':
            end_point = (point[0] - arrow_length, point[1])
        else:
            return
        cv2.arrowedLine(img, tuple(point), end_point, color, 2, tipLength=0.3)
    
    def _get_instructions(self) -> List[str]:
        """Get current instructions"""
        if self.mode == 'roi':
            return [
                "MODE: Drawing ROI",
                "Click to add points (minimum 3)",
                "Press 'ENTER' when done",
                "Press 'C' to clear"
            ]
        elif self.mode == 'line':
            return [
                "MODE: Drawing Counting Line",
                "Click to add points (minimum 2)",
                "Press 'ENTER' when done",
                "Press 'C' to clear current line",
                "Press 'Q' to finish setup"
            ]
        elif self.mode == 'direction':
            return [
                "MODE: Set Direction",
                "Press: N (North/Down), S (South/Up),",
                "       W (West/Left), E (East/Right)",
                f"Current line has {len(self.current_line_points)} points"
            ]
        return []
    
    def setup_camera(self, rtsp_url: str) -> Tuple[List, List[Dict]]:
        """Interactive setup for a camera"""
        logger.info(f"Starting setup for camera: {rtsp_url}")
        
        # Connect to camera
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.error(f"Failed to open camera: {rtsp_url}")
            return None, None
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Failed to read frame from: {rtsp_url}")
            cap.release()
            return None, None
        
        # Resize frame
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        window_name = f"Setup: {rtsp_url}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            display = self.draw_setup(frame)
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord('C'):
                if self.mode == 'roi':
                    self.roi_points = []
                    logger.info("ROI cleared")
                elif self.mode == 'line':
                    self.current_line_points = []
                    logger.info("Current line cleared")
            
            elif key == 13:  # Enter
                if self.mode == 'roi' and len(self.roi_points) >= 3:
                    logger.info("ROI completed, switching to line drawing mode")
                    self.mode = 'line'
                elif self.mode == 'line' and len(self.current_line_points) >= 2:
                    logger.info("Line drawn, set direction")
                    self.mode = 'direction'
            
            elif self.mode == 'direction':
                direction = None
                if key == ord('n') or key == ord('N'):
                    if key == ord('n'):
                        direction = 'NS'
                    else:
                        direction = 'SN'
                elif key == ord('s') or key == ord('S'):
                    direction = 'SN'
                elif key == ord('w') or key == ord('W'):
                    direction = 'WE'
                elif key == ord('e') or key == ord('E'):
                    direction = 'EW'
                
                if direction:
                    self.lines.append({
                        'points': self.current_line_points.copy(),
                        'direction': direction,
                        'thingsboard_key': f'line_{len(self.lines)+1}'
                    })
                    logger.info(f"Line {len(self.lines)} added with direction {direction}")
                    self.current_line_points = []
                    self.mode = 'line'
            
            elif key == ord('q') or key == ord('Q'):
                if len(self.roi_points) >= 3 and len(self.lines) > 0:
                    logger.info("Setup completed")
                    break
                else:
                    logger.warning("Setup incomplete. Need ROI and at least one line")
        
        cap.release()
        cv2.destroyWindow(window_name)
        
        return self.roi_points, self.lines


class LineCrossDetector:
    """Detect when tracked objects cross counting lines"""
    def __init__(self, line_points: List, direction: str):
        self.line_points = np.array(line_points)
        self.direction = direction
        self.crossed_ids = set()
        self.count = 0
        
    def point_position(self, point: Tuple[int, int], line_start: Tuple, line_end: Tuple) -> float:
        """Determine which side of line a point is on"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    
    def check_crossing(self, track_id: int, prev_center: Tuple, curr_center: Tuple) -> bool:
        """Check if track crossed the line in correct direction"""
        if track_id in self.crossed_ids:
            return False
        
        # Check against each segment of the line
        for i in range(len(self.line_points) - 1):
            line_start = tuple(self.line_points[i])
            line_end = tuple(self.line_points[i + 1])
            
            prev_pos = self.point_position(prev_center, line_start, line_end)
            curr_pos = self.point_position(curr_center, line_start, line_end)
            
            # Check if crossed (signs are different)
            if prev_pos * curr_pos < 0:
                # Determine crossing direction
                crossed_direction = self._get_crossing_direction(prev_center, curr_center)
                
                if self._matches_direction(crossed_direction):
                    self.crossed_ids.add(track_id)
                    self.count += 1
                    return True
        
        return False
    
    def _get_crossing_direction(self, prev_center: Tuple, curr_center: Tuple) -> str:
        """Determine the direction of crossing"""
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        
        if abs(dy) > abs(dx):
            return 'NS' if dy > 0 else 'SN'
        else:
            return 'WE' if dx > 0 else 'EW'
    
    def _matches_direction(self, crossed_dir: str) -> bool:
        """Check if crossing direction matches expected direction"""
        return crossed_dir == self.direction
    
    def reset_minute_count(self) -> int:
        """Reset and return the count for the current minute"""
        count = self.count
        # Keep track of crossed IDs but reset count
        return count


class CameraProcessor:
    """Process individual camera stream"""
    def __init__(self, rtsp_url: str, camera_config: Dict, model, 
                 process_fps: int, resize_dim: Tuple[int, int]):
        self.rtsp_url = rtsp_url
        self.config = camera_config
        self.model = model
        self.process_fps = process_fps
        self.resize_dim = resize_dim
        self.running = False
        self.thread = None
        
        # Setup ROI mask
        self.roi_mask = self._create_roi_mask(camera_config['roi'])
        
        # Setup line detectors
        self.line_detectors = []
        for line_data in camera_config['lines']:
            detector = LineCrossDetector(line_data['points'], line_data['direction'])
            detector.thingsboard_key = line_data.get('thingsboard_key', f"line_{len(self.line_detectors)+1}")
            detector.total_count = 0  # Cumulative count for display
            self.line_detectors.append(detector)
        
        # Track history for each tracked object
        self.track_history = defaultdict(list)
        
        # Frame processing
        self.frame_interval = 1.0 / process_fps
        self.last_process_time = 0
        
        # ThingsBoard posting
        self.last_post_time = time.time()
        self.post_interval = 60  # Post every minute
        
        # Retry mechanism
        self.max_retries = 5
        self.retry_delay = 5  # seconds
        
    def _create_roi_mask(self, roi_points: List) -> np.ndarray:
        """Create binary mask for ROI"""
        mask = np.zeros(self.resize_dim[::-1], dtype=np.uint8)
        roi_array = np.array(roi_points, dtype=np.int32)
        cv2.fillPoly(mask, [roi_array], 255)
        return mask
    
    def _connect_camera(self) -> Optional[cv2.VideoCapture]:
        """Connect to camera with retry logic"""
        for attempt in range(self.max_retries):
            try:
                cap = cv2.VideoCapture(self.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
                
                if cap.isOpened():
                    logger.info(f"Connected to camera: {self.rtsp_url}")
                    return cap
                else:
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {self.rtsp_url}")
                    time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Error connecting to {self.rtsp_url}: {e}")
                time.sleep(self.retry_delay)
        
        return None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """Process a single frame"""
        # Apply ROI mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=self.roi_mask)
        
        # Run YOLOv11 with tracking
        results = self.model.track(
            masked_frame,
            persist=True,
            classes=[0],  # Only detect persons (class 0)
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        crossings = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                # Store track history
                self.track_history[track_id].append(center)
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                
                # Check line crossings
                if len(self.track_history[track_id]) >= 2:
                    prev_center = self.track_history[track_id][-2]
                    curr_center = self.track_history[track_id][-1]
                    
                    for idx, detector in enumerate(self.line_detectors):
                        if detector.check_crossing(track_id, prev_center, curr_center):
                            detector.total_count += 1
                            crossings.append((idx, detector.total_count))
                            logger.info(f"Line {idx+1} crossing detected! Total: {detector.total_count}")
        
        return results[0].plot(), crossings
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI, lines, and counts on frame"""
        display = frame.copy()
        
        # Draw ROI
        roi_array = np.array(self.config['roi'], dtype=np.int32)
        cv2.polylines(display, [roi_array], True, (0, 255, 0), 2)
        
        # Draw lines and counts
        for idx, (line_data, detector) in enumerate(zip(self.config['lines'], self.line_detectors)):
            points = line_data['points']
            color = (255, 0, 0) if idx % 2 == 0 else (0, 0, 255)
            
            # Draw line
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(display, [pts], False, color, 2)
            
            # Draw count
            text_pos = tuple(points[0])
            cv2.putText(display, f"L{idx+1}:{detector.total_count}", 
                       text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return display
    
    def post_to_thingsboard(self):
        """Post counts to ThingsBoard"""
        if not self.config.get('thingsboard_url'):
            return
        
        try:
            for idx, detector in enumerate(self.line_detectors):
                minute_count = detector.reset_minute_count()
                
                data = {detector.thingsboard_key: minute_count}
                
                response = requests.post(
                    self.config['thingsboard_url'],
                    headers={'Content-Type': 'application/json'},
                    json=data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    logger.info(f"Posted to ThingsBoard - Line {idx+1}: {minute_count}")
                else:
                    logger.error(f"ThingsBoard post failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error posting to ThingsBoard: {e}")
    
    def run(self):
        """Main processing loop"""
        self.running = True
        cap = None
        
        while self.running:
            if cap is None or not cap.isOpened():
                cap = self._connect_camera()
                if cap is None:
                    logger.error(f"Failed to connect to {self.rtsp_url}, retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    continue
            
            try:
                # Read frame (keep reading to avoid buffering)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame from {self.rtsp_url}")
                    cap.release()
                    cap = None
                    time.sleep(1)
                    continue
                
                # Resize frame
                frame = cv2.resize(frame, self.resize_dim)
                
                current_time = time.time()
                
                # Process frame at specified FPS
                if current_time - self.last_process_time >= self.frame_interval:
                    processed_frame, crossings = self.process_frame(frame)
                    display_frame = self.draw_overlay(processed_frame)
                    
                    # Show frame
                    cv2.imshow(f"Camera: {self.rtsp_url}", display_frame)
                    self.last_process_time = current_time
                
                # Post to ThingsBoard every minute
                if current_time - self.last_post_time >= self.post_interval:
                    self.post_to_thingsboard()
                    self.last_post_time = current_time
                
                # Handle key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
            except Exception as e:
                logger.error(f"Error processing frame from {self.rtsp_url}: {e}")
                time.sleep(1)
        
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    
    def start(self):
        """Start processing in a separate thread"""
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop processing"""
        self.running = False
        if self.thread:
            self.thread.join()


class VisitorCountingSystem:
    """Main system coordinator"""
    def __init__(self, config_path='camera_config.json'):
        self.config_handler = CameraConfig(config_path)
        self.models = []
        self.camera_processors = []
        self.cameras_per_model = 5
        
    def load_model(self) -> YOLO:
        """Load YOLOv11n model"""
        logger.info("Loading YOLOv11n model...")
        model = YOLO('yolo11n.pt')
        logger.info("Model loaded successfully")
        return model
    
    def setup_cameras(self, rtsp_urls: List[str]):
        """Setup cameras that need configuration"""
        setup = InteractiveSetup(
            self.config_handler.config['settings']['resize_width'],
            self.config_handler.config['settings']['resize_height']
        )
        
        for url in rtsp_urls:
            if self.config_handler.needs_setup(url):
                logger.info(f"Camera needs setup: {url}")
                roi, lines = setup.setup_camera(url)
                
                if roi and lines:
                    self.config_handler.set_camera(url, roi, lines)
                    
                    # Get ThingsBoard URL
                    tb_url = input(f"\nEnter ThingsBoard URL for {url} (or press Enter to skip): ").strip()
                    if tb_url:
                        self.config_handler.config['cameras'][url]['thingsboard_url'] = tb_url
                        self.config_handler.save_config()
                else:
                    logger.error(f"Setup failed for {url}")
    
    def start_processing(self, rtsp_urls: List[str]):
        """Start processing all cameras"""
        logger.info("Starting camera processing...")
        
        process_fps = self.config_handler.config['settings']['process_fps']
        resize_dim = (
            self.config_handler.config['settings']['resize_width'],
            self.config_handler.config['settings']['resize_height']
        )
        
        # Load models (one per 5 cameras)
        num_models = (len(rtsp_urls) + self.cameras_per_model - 1) // self.cameras_per_model
        for _ in range(num_models):
            self.models.append(self.load_model())
        
        # Create processors for each camera
        for idx, url in enumerate(rtsp_urls):
            camera_config = self.config_handler.get_camera(url)
            if camera_config is None:
                logger.warning(f"No configuration for {url}, skipping")
                continue
            
            # Assign model
            model_idx = idx // self.cameras_per_model
            model = self.models[model_idx]
            
            processor = CameraProcessor(url, camera_config, model, process_fps, resize_dim)
            processor.start()
            self.camera_processors.append(processor)
        
        logger.info(f"Started processing {len(self.camera_processors)} cameras with {num_models} model(s)")
    
    def run(self, rtsp_urls: List[str]):
        """Main entry point"""
        try:
            # Setup cameras
            self.setup_cameras(rtsp_urls)
            
            # Start processing
            self.start_processing(rtsp_urls)
            
            # Keep main thread alive
            logger.info("System running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            for processor in self.camera_processors:
                processor.stop()
            logger.info("Shutdown complete")


if __name__ == "__main__":
    # Load RTSP URLs from file
    rtsp_urls = []
    rtsp_file = 'rtsp_urls.txt'
    
    if Path(rtsp_file).exists():
        with open(rtsp_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    rtsp_urls.append(line)
        
        if not rtsp_urls:
            logger.error(f"No RTSP URLs found in {rtsp_file}")
            logger.info("Add camera URLs to rtsp_urls.txt (one per line)")
            exit(1)
        
        logger.info(f"Loaded {len(rtsp_urls)} camera URL(s) from {rtsp_file}")
    else:
        logger.error(f"{rtsp_file} not found")
        logger.info("Create rtsp_urls.txt with your camera URLs (one per line)")
        logger.info("Example: rtsp://admin:12345@192.168.5.227/71")
        exit(1)
    
    # Create and run system
    system = VisitorCountingSystem()
    system.run(rtsp_urls)
