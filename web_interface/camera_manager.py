import logging
import time
import os
import logging
from datetime import datetime
import json
import threading
import cv2
import numpy as np
import base64
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CameraManager')

# Directory for storing snapshots
snapshot_dir = os.path.join(os.path.dirname(__file__), 'snapshots')
os.makedirs(snapshot_dir, exist_ok=True)

class CameraManager:
    """Implementation of CameraManager to handle camera API requests with multi-camera and stereo vision support"""
    
    def __init__(self):
        self.cameras = {}
        self.active_cameras = set()
        self.settings = {}
        self.streams = {}
        self.lock = threading.Lock()
        self.stereo_pairs = {}
        self.stereo_active = False
        self._initialize_cameras()
        self._discover_real_cameras()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.frame_buffer = {}
        self.last_frame_time = {}
        self.max_buffer_size = 5
    
    def _initialize_cameras(self):
        """Initialize mock and default cameras"""
        # Mock camera 0 - built-in webcam (Left Eye)
        self.cameras[0] = {
            'id': 0,
            'name': 'Left Eye Camera',
            'type': 'webcam',
            'description': 'Left camera for stereo vision',
            'status': 'available',
            'resolution': '1280x720',
            'fps': 30,
            'position': 'left',
            'lens_type': 'wide-angle',
            'field_of_view': 75
        }
        
        # Mock camera 1 - USB camera (Right Eye)
        self.cameras[1] = {
            'id': 1,
            'name': 'Right Eye Camera',
            'type': 'usb',
            'description': 'Right camera for stereo vision',
            'status': 'available',
            'resolution': '1280x720',
            'fps': 30,
            'position': 'right',
            'lens_type': 'wide-angle',
            'field_of_view': 75
        }
        
        # Mock camera 2 - Front view camera
        self.cameras[2] = {
            'id': 2,
            'name': 'Front View Camera',
            'type': 'usb',
            'description': 'Additional external front camera',
            'status': 'available',
            'resolution': '1920x1080',
            'fps': 60,
            'position': 'front',
            'lens_type': 'standard',
            'field_of_view': 65
        }
        
        # Mock camera 3 - Top view camera
        self.cameras[3] = {
            'id': 3,
            'name': 'Top View Camera',
            'type': 'usb',
            'description': 'Overhead view camera',
            'status': 'available',
            'resolution': '1280x720',
            'fps': 30,
            'position': 'top',
            'lens_type': 'fisheye',
            'field_of_view': 120
        }
        
        # Default settings for each camera
        for camera_id in self.cameras:
            self.settings[camera_id] = {
                'resolution': '1280x720',
                'fps': 30,
                'brightness': 50,
                'contrast': 50,
                'saturation': 50,
                'exposure': 50,
                'white_balance': 5000,
                'auto_focus': True,
                'sharpness': 50,
                'gain': 0
            }
        
        # Setup default stereo pairs
        self.stereo_pairs = {
            'default': {'left': 0, 'right': 1, 'baseline': 65, 'focal_length': 600},
            'high_res': {'left': 2, 'right': 3, 'baseline': 100, 'focal_length': 800}
        }
    
    def _discover_real_cameras(self):
        """Discover real cameras connected to the system"""
        try:
            logger.info("Discovering real cameras...")
            # Try to find up to 10 cameras
            camera_count = 0
            for i in range(10):
                try:
                    # Try different backends for better compatibility
                    backends = [cv2.CAP_DSHOW, cv2.CAP_VFW, cv2.CAP_MSMF] if os.name == 'nt' else [cv2.CAP_V4L2, cv2.CAP_FFMPEG]
                    cap = None
                    
                    # Try each backend until one works
                    for backend in backends:
                        cap = cv2.VideoCapture(i, backend)
                        if cap.isOpened():
                            break
                        cap.release()
                        cap = None
                    
                    if cap and cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Create a real camera entry
                        camera_id = 100 + camera_count  # Offset to distinguish from mock cameras
                        camera_count += 1
                        
                        # Determine position based on discovery order
                        position = 'left' if camera_count == 1 else 'right' if camera_count == 2 else 'external'
                        
                        self.cameras[camera_id] = {
                            'id': camera_id,
                            'name': f'Real {position.title()} Camera',
                            'type': 'real',
                            'description': f'Real camera device at index {i}',
                            'status': 'available',
                            'resolution': f'{width}x{height}',
                            'fps': fps if fps > 0 else 30,
                            'position': position,
                            'lens_type': 'standard',
                            'field_of_view': 70
                        }
                        
                        # Create default settings with more detailed options
                        self.settings[camera_id] = {
                            'resolution': f'{width}x{height}',
                            'fps': fps if fps > 0 else 30,
                            'brightness': 50,
                            'contrast': 50,
                            'saturation': 50,
                            'exposure': 50,
                            'white_balance': 5000,
                            'auto_focus': True,
                            'sharpness': 50,
                            'gain': 0,
                            'backend': backend
                        }
                        
                        logger.info(f"Found real camera: ID={camera_id}, Name={self.cameras[camera_id]['name']}, Resolution={width}x{height}, FPS={fps}")
                        cap.release()
                except Exception as camera_err:
                    logger.warning(f"Error checking camera {i}: {str(camera_err)}")
                    continue
        except Exception as e:
            logger.error(f"Error discovering real cameras: {str(e)}")
    
    def list_available_cameras(self):
        """List all available cameras"""
        logger.info("Listing available cameras")
        with self.lock:
            return list(self.cameras.values())
    
    def get_active_camera_ids(self):
        """Get IDs of all active cameras"""
        with self.lock:
            return list(self.active_cameras)
    
    def get_camera_status(self, camera_id):
        """Get status of a specific camera"""
        logger.info(f"Getting status for camera {camera_id}")
        with self.lock:
            if camera_id not in self.cameras:
                return {
                    'camera_id': camera_id,
                    'status': 'error',
                    'message': 'Camera not found'
                }
            
            is_active = camera_id in self.active_cameras
            camera_info = self.cameras[camera_id].copy()
            
            return {
                'camera_id': camera_id,
                'is_active': is_active,
                'status': 'active' if is_active else 'inactive',
                'start_time': datetime.now().isoformat() if is_active else None,
                'settings': self.settings.get(camera_id, {}).copy(),
                'camera_info': camera_info,
                'timestamp': datetime.now().isoformat()
            }
    
    def start_camera(self, camera_id, params=None):
        """Start a specific camera"""
        logger.info(f"Starting camera {camera_id} with params: {params}")
        with self.lock:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False
            
            # If camera is already active, return success
            if camera_id in self.active_cameras:
                logger.warning(f"Camera {camera_id} is already active")
                # Apply any new settings
                if params:
                    self.update_camera_settings(camera_id, params)
                return True
            
            # Create frame buffer for this camera
            self.frame_buffer[camera_id] = []
            self.last_frame_time[camera_id] = datetime.now()
            
            # Try to open the camera if it's a real camera (ID >= 100)
            if camera_id >= 100:
                try:
                    real_camera_index = camera_id - 100
                    
                    # Get the backend from settings if available
                    backend = self.settings[camera_id].get('backend', cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
                    cap = cv2.VideoCapture(real_camera_index, backend)
                    
                    if cap.isOpened():
                        # Apply settings if specified
                        if params:
                            if 'resolution' in params:
                                width, height = map(int, params['resolution'].split('x'))
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                                self.cameras[camera_id]['resolution'] = f'{width}x{height}'
                            
                            if 'fps' in params:
                                cap.set(cv2.CAP_PROP_FPS, params['fps'])
                                self.cameras[camera_id]['fps'] = params['fps']
                        
                        # Store the camera stream
                        self.streams[camera_id] = cap
                        logger.info(f"Successfully opened real camera: {camera_id} ({self.cameras[camera_id]['name']})")
                    else:
                        logger.warning(f"Could not open real camera {camera_id}, using mock mode")
                        cap.release()
                except Exception as cv_err:
                    logger.error(f"OpenCV error opening camera {camera_id}: {str(cv_err)}")
            
            # Add to active cameras
            self.active_cameras.add(camera_id)
            
            # Apply any settings from params
            if params:
                self.update_camera_settings(camera_id, params)
                
            logger.info(f"Camera {camera_id} started successfully")
            return True
    
    def stop_camera(self, camera_id):
        """Stop a specific camera"""
        logger.info(f"Stopping camera {camera_id}")
        with self.lock:
            if camera_id not in self.active_cameras:
                logger.warning(f"Camera {camera_id} is not active")
                return True  # Consider it a success if it's not active
            
            # Release the camera stream if it's a real camera
            if camera_id in self.streams:
                try:
                    self.streams[camera_id].release()
                    del self.streams[camera_id]
                    logger.info(f"Released camera stream: {camera_id}")
                except Exception as cv_err:
                    logger.error(f"Error releasing camera {camera_id}: {str(cv_err)}")
            
            # Clean up frame buffer and time tracking
            if camera_id in self.frame_buffer:
                del self.frame_buffer[camera_id]
            if camera_id in self.last_frame_time:
                del self.last_frame_time[camera_id]
            
            # Remove from active cameras
            self.active_cameras.remove(camera_id)
            
            # Update stereo active status if needed
            if self.stereo_active:
                for pair_name, pair in self.stereo_pairs.items():
                    if pair['left'] == camera_id or pair['right'] == camera_id:
                        self.stereo_active = False
                        logger.info(f"Stereo vision deactivated because camera {camera_id} was stopped")
                        break
            
            logger.info(f"Camera {camera_id} stopped successfully")
            return True
    
    def stop_all_cameras(self):
        """Stop all active cameras"""
        logger.info("Stopping all cameras")
        with self.lock:
            active_ids = list(self.active_cameras)
            
            for camera_id in active_ids:
                self.stop_camera(camera_id)
                
            self.stereo_active = False
            logger.info("All cameras stopped")
            return True
    
    def take_snapshot(self, camera_id):
        """Take a snapshot from a camera"""
        logger.info(f"Taking snapshot from camera {camera_id}")
        with self.lock:
            if camera_id not in self.active_cameras:
                logger.error(f"Camera {camera_id} is not active")
                return {
                    'status': 'error',
                    'message': 'Camera is not active',
                    'camera_id': camera_id
                }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            snapshot_time = datetime.now().isoformat()
            camera_name = self.cameras[camera_id]['name']
            
            # Try to get a real frame if it's a real camera
            if camera_id in self.streams:
                try:
                    cap = self.streams[camera_id]
                    ret, frame = cap.read()
                    
                    if ret:
                        # Apply camera settings with advanced image processing
                        settings = self.settings.get(camera_id, {})
                        processed_frame = self._apply_camera_settings(frame, settings)
                        
                        # Create camera-specific directory
                        camera_dir = os.path.join(snapshot_dir, f'camera_{camera_id}_{camera_name.replace(" ", "_")}')
                        os.makedirs(camera_dir, exist_ok=True)
                        
                        # Save the image
                        snapshot_path = os.path.join(camera_dir, f'snapshot_{timestamp}.jpg')
                        cv2.imwrite(snapshot_path, processed_frame)
                        
                        # Convert to base64 with quality optimization
                        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        image_data = base64.b64encode(buffer).decode('utf-8')
                        
                        logger.info(f"Snapshot saved: {snapshot_path}")
                        return {
                            'status': 'success',
                            'camera_id': camera_id,
                            'camera_name': camera_name,
                            'snapshot_time': snapshot_time,
                            'data': f'data:image/jpeg;base64,{image_data}',
                            'file_path': snapshot_path,
                            'message': 'Snapshot taken successfully'
                        }
                    else:
                        logger.warning(f"Could not capture frame from camera {camera_id}")
                        # Try to use the latest buffered frame if available
                        if camera_id in self.frame_buffer and len(self.frame_buffer[camera_id]) > 0:
                            frame = self.frame_buffer[camera_id][-1]
                            # Save and return this frame
                            camera_dir = os.path.join(snapshot_dir, f'camera_{camera_id}_{camera_name.replace(" ", "_")}')
                            os.makedirs(camera_dir, exist_ok=True)
                            snapshot_path = os.path.join(camera_dir, f'snapshot_{timestamp}_buffered.jpg')
                            cv2.imwrite(snapshot_path, frame)
                            _, buffer = cv2.imencode('.jpg', frame)
                            image_data = base64.b64encode(buffer).decode('utf-8')
                            return {
                                'status': 'success',
                                'camera_id': camera_id,
                                'camera_name': camera_name,
                                'snapshot_time': snapshot_time,
                                'data': f'data:image/jpeg;base64,{image_data}',
                                'file_path': snapshot_path,
                                'message': 'Snapshot taken from buffer successfully'
                            }
                except Exception as cv_err:
                    logger.error(f"Error capturing snapshot: {str(cv_err)}")
            
            # Return mock data if real camera not available
            # Generate a color pattern based on camera ID for better visualization
            color_palette = [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0)]
            color = color_palette[camera_id % len(color_palette)]
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_frame[:] = color
            
            # Add camera ID and timestamp text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(mock_frame, f"Camera {camera_id}", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(mock_frame, timestamp, (10, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', mock_frame)
            mock_data = base64.b64encode(buffer).decode('utf-8')
            
            # Save mock data to file for consistency
            camera_dir = os.path.join(snapshot_dir, f'camera_{camera_id}_{camera_name.replace(" ", "_")}')
            os.makedirs(camera_dir, exist_ok=True)
            mock_path = os.path.join(camera_dir, f'snapshot_{timestamp}_mock.jpg')
            cv2.imwrite(mock_path, mock_frame)
            
            return {
                'status': 'success',
                'camera_id': camera_id,
                'camera_name': camera_name,
                'snapshot_time': snapshot_time,
                'data': f'data:image/jpeg;base64,{mock_data}',
                'file_path': mock_path,
                'message': 'Snapshot taken successfully (mock data)'
            }
    
    def take_all_snapshots(self):
        """Take snapshots from all active cameras"""
        logger.info("Taking snapshots from all active cameras")
        results = []
        
        with self.lock:
            active_ids = list(self.active_cameras)
            
        for camera_id in active_ids:
            result = self.take_snapshot(camera_id)
            results.append(result)
        
        return results
    
    def _apply_camera_settings(self, frame, settings):
        """Apply camera settings to a frame"""
        # Apply brightness and contrast
        if 'brightness' in settings or 'contrast' in settings:
            alpha = settings.get('contrast', 50) / 50
            beta = settings.get('brightness', 50) - 50
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # Apply saturation
        if 'saturation' in settings:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.convertScaleAbs(hsv[:,:,1], alpha=settings['saturation']/50)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Apply sharpness
        if 'sharpness' in settings and settings['sharpness'] != 50:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpness_factor = settings['sharpness'] / 50
            sharpened = cv2.filter2D(frame, -1, kernel)
            frame = cv2.addWeighted(frame, 1 - sharpness_factor + 0.5, sharpened, sharpness_factor - 0.5, 0)
        
        return frame
    
    def get_camera_settings(self, camera_id):
        """Get settings for a specific camera"""
        logger.info(f"Getting settings for camera {camera_id}")
        with self.lock:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return {
                    'status': 'error',
                    'message': 'Camera not found',
                    'camera_id': camera_id
                }
            
            return {
                'status': 'success',
                'camera_id': camera_id,
                'settings': self.settings.get(camera_id, {}).copy()
            }
    
    def update_camera_settings(self, camera_id, settings):
        """Update settings for a specific camera"""
        logger.info(f"Updating settings for camera {camera_id} with: {settings}")
        with self.lock:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False
            
            # Update settings
            if camera_id not in self.settings:
                self.settings[camera_id] = {}
            
            self.settings[camera_id].update(settings)
            
            # Apply settings to real camera if it's active
            if camera_id in self.streams and camera_id in self.active_cameras:
                try:
                    cap = self.streams[camera_id]
                    if 'brightness' in settings:
                        cap.set(cv2.CAP_PROP_BRIGHTNESS, settings['brightness']/100)
                    if 'contrast' in settings:
                        cap.set(cv2.CAP_PROP_CONTRAST, settings['contrast']/100)
                    if 'saturation' in settings:
                        cap.set(cv2.CAP_PROP_SATURATION, settings['saturation']/100)
                    if 'exposure' in settings:
                        cap.set(cv2.CAP_PROP_EXPOSURE, settings['exposure']-100)
                    if 'white_balance' in settings:
                        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, settings['white_balance'])
                    if 'resolution' in settings:
                        width, height = map(int, settings['resolution'].split('x'))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        self.cameras[camera_id]['resolution'] = settings['resolution']
                    if 'fps' in settings:
                        cap.set(cv2.CAP_PROP_FPS, settings['fps'])
                        self.cameras[camera_id]['fps'] = settings['fps']
                except Exception as cv_err:
                    logger.warning(f"Error applying camera settings: {str(cv_err)}")
            
            logger.info(f"Settings updated for camera {camera_id}")
            return True
    
    def get_camera_frame(self, camera_id):
        """Get a frame from a camera with frame buffering and optimized performance"""
        logger.debug(f"Getting frame from camera {camera_id}")
        with self.lock:
            if camera_id not in self.active_cameras:
                logger.error(f"Camera {camera_id} is not active")
                return {
                    'status': 'error',
                    'message': 'Camera is not active',
                    'camera_id': camera_id
                }
            
            frame_time = datetime.now().isoformat()
            
            # Try to get a real frame if it's a real camera
            if camera_id in self.streams:
                try:
                    cap = self.streams[camera_id]
                    ret, frame = cap.read()
                    
                    if ret:
                        # Update frame buffer
                        if camera_id in self.frame_buffer:
                            self.frame_buffer[camera_id].append(frame.copy())
                            # Keep buffer size manageable
                            if len(self.frame_buffer[camera_id]) > self.max_buffer_size:
                                self.frame_buffer[camera_id].pop(0)
                        
                        # Apply camera settings with optimized processing
                        settings = self.settings.get(camera_id, {})
                        processed_frame = self._apply_camera_settings(frame, settings)
                        
                        # Convert to base64 with compression for faster transmission
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
                        _, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                        image_data = base64.b64encode(buffer).decode('utf-8')
                        
                        self.last_frame_time[camera_id] = datetime.now()
                        
                        return {
                            'status': 'success',
                            'camera_id': camera_id,
                            'frame_time': frame_time,
                            'data': f'data:image/jpeg;base64,{image_data}',
                            'resolution': self.cameras[camera_id]['resolution'],
                            'fps': self.cameras[camera_id]['fps'],
                            'message': 'Frame retrieved successfully'
                        }
                    else:
                        logger.warning(f"Could not capture frame from camera {camera_id}")
                        # Try to use the latest buffered frame if available
                        if camera_id in self.frame_buffer and len(self.frame_buffer[camera_id]) > 0:
                            frame = self.frame_buffer[camera_id][-1]
                            settings = self.settings.get(camera_id, {})
                            processed_frame = self._apply_camera_settings(frame, settings)
                            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
                            _, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                            image_data = base64.b64encode(buffer).decode('utf-8')
                            
                            return {
                                'status': 'success',
                                'camera_id': camera_id,
                                'frame_time': frame_time,
                                'data': f'data:image/jpeg;base64,{image_data}',
                                'resolution': self.cameras[camera_id]['resolution'],
                                'fps': self.cameras[camera_id]['fps'],
                                'message': 'Frame retrieved from buffer',
                                'from_buffer': True
                            }
                except Exception as cv_err:
                    logger.error(f"Error capturing frame: {str(cv_err)}")
            
            # Return improved mock data if real camera not available
            # Generate a dynamic mock frame with changing elements for better visualization
            width, height = map(int, self.cameras[camera_id]['resolution'].split('x'))
            width = min(width, 640)  # Limit for mock frame
            height = min(height, 480)
            
            # Create a mock frame with a gradient background
            mock_frame = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    # Create a dynamic gradient based on time and camera ID
                    t = datetime.now().timestamp()
                    r = int(128 + 64 * np.sin(t * 0.5 + camera_id))
                    g = int(128 + 64 * np.sin(t * 0.5 + camera_id + 2))
                    b = int(128 + 64 * np.sin(t * 0.5 + camera_id + 4))
                    mock_frame[y, x] = [b, g, r]  # BGR format
            
            # Add camera information text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(mock_frame, f"Camera {camera_id}: {self.cameras[camera_id]['name']}", 
                        (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(mock_frame, f"Resolution: {self.cameras[camera_id]['resolution']}", 
                        (10, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(mock_frame, datetime.now().strftime('%H:%M:%S.%f')[:-3], 
                        (10, 90), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add a moving object to simulate motion
            t = datetime.now().timestamp()
            x_pos = int((width - 100) * (0.5 + 0.5 * np.sin(t * 2 + camera_id))) + 50
            y_pos = int((height - 100) * (0.5 + 0.5 * np.cos(t * 1.5 + camera_id))) + 50
            cv2.circle(mock_frame, (x_pos, y_pos), 20, (0, 255, 0), -1)
            
            # Convert to base64
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
            _, buffer = cv2.imencode('.jpg', mock_frame, encode_params)
            mock_data = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'status': 'success',
                'camera_id': camera_id,
                'frame_time': frame_time,
                'data': f'data:image/jpeg;base64,{mock_data}',
                'resolution': self.cameras[camera_id]['resolution'],
                'fps': self.cameras[camera_id]['fps'],
                'message': 'Frame retrieved successfully (mock data)',
                'mock': True
            }
    
    def get_frame_buffer_info(self, camera_id):
        """Get information about the frame buffer for a specific camera"""
        with self.lock:
            if camera_id not in self.active_cameras:
                return {
                    'status': 'error',
                    'message': 'Camera is not active',
                    'camera_id': camera_id
                }
            
            buffer_size = len(self.frame_buffer.get(camera_id, []))
            last_frame_delta = None
            
            if camera_id in self.last_frame_time:
                delta = datetime.now() - self.last_frame_time[camera_id]
                last_frame_delta = delta.total_seconds()
            
            return {
                'status': 'success',
                'camera_id': camera_id,
                'buffer_size': buffer_size,
                'max_buffer_size': self.max_buffer_size,
                'last_frame_time': last_frame_delta,
                'message': 'Buffer info retrieved successfully'
            }
    
    # Stereo Vision Functions
    def get_stereo_pair(self, pair_name='default'):
        """Get a stereo camera pair by name"""
        logger.info(f"Getting stereo pair {pair_name}")
        with self.lock:
            if pair_name in self.stereo_pairs:
                return self.stereo_pairs[pair_name].copy()
            else:
                logger.error(f"Stereo pair {pair_name} not found")
                return None
    
    def set_stereo_pair(self, pair_name, left_camera_id, right_camera_id, baseline=65, focal_length=600):
        """Set a stereo camera pair with calibration parameters"""
        logger.info(f"Setting stereo pair {pair_name} with left={left_camera_id}, right={right_camera_id}")
        with self.lock:
            if left_camera_id in self.cameras and right_camera_id in self.cameras:
                self.stereo_pairs[pair_name] = {
                    'left': left_camera_id,
                    'right': right_camera_id,
                    'baseline': baseline,  # Distance between cameras in mm
                    'focal_length': focal_length,  # Focal length in pixels
                    'calibrated': False  # Whether the pair has been calibrated
                }
                logger.info(f"Stereo pair {pair_name} set successfully")
                return True
            else:
                logger.error(f"Invalid camera IDs for stereo pair")
                return False
    
    def list_stereo_pairs(self):
        """List all configured stereo pairs"""
        logger.info("Listing stereo pairs")
        with self.lock:
            return {pair_name: pair.copy() for pair_name, pair in self.stereo_pairs.items()}
    
    def enable_stereo_vision(self, pair_name='default'):
        """Enable stereo vision processing for a specific pair"""
        logger.info(f"Enabling stereo vision for pair {pair_name}")
        
        # Get the stereo pair
        pair = self.get_stereo_pair(pair_name)
        if not pair:
            return {
                'status': 'error',
                'message': f'Stereo pair {pair_name} not found',
                'pair_name': pair_name
            }
        
        left_id = pair['left']
        right_id = pair['right']
        
        # Check if both cameras are active
        with self.lock:
            if left_id not in self.active_cameras or right_id not in self.active_cameras:
                # Try to start them if they're not active
                if left_id not in self.active_cameras:
                    logger.info(f"Starting left camera {left_id} for stereo vision")
                    self.start_camera(left_id)
                
                if right_id not in self.active_cameras:
                    logger.info(f"Starting right camera {right_id} for stereo vision")
                    self.start_camera(right_id)
                
                # Check again after attempting to start
                if left_id not in self.active_cameras or right_id not in self.active_cameras:
                    logger.error(f'Both cameras in stereo pair {pair_name} must be active')
                    return {
                        'status': 'error',
                        'message': 'Failed to start both cameras',
                        'pair_name': pair_name,
                        'left_active': left_id in self.active_cameras,
                        'right_active': right_id in self.active_cameras
                    }
            
            # Mark stereo vision as active
            self.stereo_active = True
        
        logger.info(f"Stereo vision enabled for pair {pair_name}")
        return {
            'status': 'success',
            'pair_name': pair_name,
            'left_camera_id': left_id,
            'right_camera_id': right_id,
            'message': 'Stereo vision enabled successfully'
        }
    
    def disable_stereo_vision(self):
        """Disable stereo vision processing"""
        logger.info("Disabling stereo vision")
        with self.lock:
            self.stereo_active = False
        
        logger.info("Stereo vision disabled")
        return {
            'status': 'success',
            'message': 'Stereo vision disabled successfully'
        }
    
    def process_stereo_vision(self, pair_name='default'):
        """Process frames from a stereo camera pair to generate depth information"""
        logger.info(f"Processing stereo vision for pair {pair_name}")
        
        # Get the stereo pair
        pair = self.get_stereo_pair(pair_name)
        if not pair:
            return {
                'status': 'error',
                'message': f'Stereo pair {pair_name} not found',
                'pair_name': pair_name
            }
        
        left_id = pair['left']
        right_id = pair['right']
        
        # Check if both cameras are active
        with self.lock:
            if left_id not in self.active_cameras or right_id not in self.active_cameras:
                logger.error(f'Both cameras in stereo pair {pair_name} must be active')
                return {
                    'status': 'error',
                    'message': 'Both cameras must be active',
                    'pair_name': pair_name,
                    'left_active': left_id in self.active_cameras,
                    'right_active': right_id in self.active_cameras
                }
        
        # Get frames from both cameras
        left_frame = self.get_camera_frame(left_id)
        right_frame = self.get_camera_frame(right_id)
        
        if left_frame['status'] != 'success' or right_frame['status'] != 'success':
            logger.error(f'Failed to get frames from stereo cameras')
            return {
                'status': 'error',
                'message': 'Failed to get camera frames',
                'pair_name': pair_name,
                'left_frame_status': left_frame['status'],
                'right_frame_status': right_frame['status']
            }
        
        timestamp = datetime.now().isoformat()
        
        # For this implementation, we'll create a simulated depth map based on the mock frames
        # In a production system, this would use proper stereo matching algorithms
        try:
            # Create a mock depth map with distance information
            depth_map = self._generate_mock_depth_map(left_id, right_id)
            
            # Convert depth map to base64 for transmission
            depth_bytes = depth_map.tobytes()
            depth_data = base64.b64encode(depth_bytes).decode('utf-8')
            
            # Also create a visual representation of the depth map
            depth_visual = self._create_depth_visualization(depth_map)
            _, depth_buffer = cv2.imencode('.png', depth_visual)
            depth_visual_data = base64.b64encode(depth_buffer).decode('utf-8')
            
            # Calculate some mock 3D points for demonstration
            points_3d = self._generate_mock_3d_points(pair)
            
            return {
                'status': 'success',
                'pair_name': pair_name,
                'left_camera_id': left_id,
                'right_camera_id': right_id,
                'timestamp': timestamp,
                'depth_data': f'data:application/octet-stream;base64,{depth_data}',
                'depth_visual': f'data:image/png;base64,{depth_visual_data}',
                'points_3d': points_3d,
                'confidence': 0.85,  # Mock confidence value
                'baseline': pair.get('baseline', 65),
                'focal_length': pair.get('focal_length', 600),
                'message': 'Stereo vision processed successfully'
            }
        except Exception as e:
            logger.error(f"Error processing stereo vision: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error processing stereo vision: {str(e)}',
                'pair_name': pair_name
            }
    
    def _generate_mock_depth_map(self, left_id, right_id):
        """Generate a mock depth map for demonstration"""
        # Use a standard size for the depth map
        width, height = 640, 480
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        # Create a scene with various depth regions
        center_x, center_y = width // 2, height // 2
        radius1, radius2 = 100, 200
        
        # Generate a radial gradient for depth
        for y in range(height):
            for x in range(width):
                # Distance from center
                dx = x - center_x
                dy = y - center_y
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Create zones with different depths
                if dist < radius1:
                    # Close objects (near distance)
                    depth_map[y, x] = 1.0 + 0.2 * np.sin(x*0.1) * np.sin(y*0.1)
                elif dist < radius2:
                    # Middle distance objects
                    depth_map[y, x] = 2.0 + 0.3 * np.sin(x*0.05) * np.sin(y*0.05)
                else:
                    # Far distance objects
                    depth_map[y, x] = 3.0 + 0.5 * np.sin(x*0.02) * np.sin(y*0.02)
        
        # Add some random noise for realism
        noise = np.random.normal(0, 0.05, (height, width))
        depth_map += noise
        
        return depth_map
    
    def _create_depth_visualization(self, depth_map):
        """Create a visual representation of the depth map"""
        # Normalize depth map for visualization
        min_depth = np.min(depth_map)
        max_depth = np.max(depth_map)
        normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)
        
        # Apply a colormap for better visualization
        # Invert the depth so that closer objects appear warmer
        depth_colored = cv2.applyColorMap((1 - normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return depth_colored
    
    def _generate_mock_3d_points(self, pair):
        """Generate mock 3D points for demonstration"""
        points = []
        
        # Create a grid of points in a 3D space
        for x in range(-5, 6, 2):
            for y in range(-5, 6, 2):
                for z in range(2, 10, 2):
                    # Add some noise to make it look more realistic
                    point = {
                        'x': x + np.random.normal(0, 0.1),
                        'y': y + np.random.normal(0, 0.1),
                        'z': z + np.random.normal(0, 0.1),
                        'confidence': 0.9 + np.random.normal(0, 0.05)
                    }
                    points.append(point)
        
        return points

# Global instance cache
_camera_manager_instance = None

def get_camera_manager():
    """Get a singleton instance of CameraManager"""
    global _camera_manager_instance
    if _camera_manager_instance is None:
        _camera_manager_instance = CameraManager()
        logger.info("CameraManager initialized with enhanced multi-camera support")
    return _camera_manager_instance

# For backward compatibility
def initialize_camera_manager():
    """Initialize the camera manager"""
    return get_camera_manager()