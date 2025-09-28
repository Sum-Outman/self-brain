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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CameraManager')

class MockCameraManager:
    """Implementation of CameraManager to handle camera API requests with multi-camera support"""
    
    def __init__(self):
        self.cameras = {}
        self.active_cameras = set()
        self.settings = {}
        self.streams = {}
        self.lock = threading.Lock()
        self.stereo_pairs = {}
        self._initialize_mock_cameras()
        self._discover_real_cameras()
    
    def _initialize_mock_cameras(self):
        """Initialize mock cameras for testing"""
        # Mock camera 0 - built-in webcam
        self.cameras[0] = {
            'id': 0,
            'name': 'Left Eye Camera',
            'type': 'webcam',
            'description': 'Left camera for stereo vision',
            'status': 'available',
            'resolution': '1280x720',
            'fps': 30,
            'position': 'left'
        }
        
        # Mock camera 1 - USB camera
        self.cameras[1] = {
            'id': 1,
            'name': 'Right Eye Camera',
            'type': 'usb',
            'description': 'Right camera for stereo vision',
            'status': 'available',
            'resolution': '1920x1080',
            'fps': 60,
            'position': 'right'
        }
        
        # Mock camera 2 - External webcam
        self.cameras[2] = {
            'id': 2,
            'name': 'Front View Camera',
            'type': 'usb',
            'description': 'Additional external camera',
            'status': 'available',
            'resolution': '1280x720',
            'fps': 30,
            'position': 'front'
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
                'white_balance': 5000
            }
        
        # Setup default stereo pairs
        self.stereo_pairs['default'] = {'left': 0, 'right': 1}
    
    def _discover_real_cameras(self):
        """Discover real cameras connected to the system"""
        try:
            logger.info("Discovering real cameras...")
            # Try to find up to 10 cameras
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Create a real camera entry
                    camera_id = i + 100  # Offset to distinguish from mock cameras
                    self.cameras[camera_id] = {
                        'id': camera_id,
                        'name': f'Real Camera {i}',
                        'type': 'real',
                        'description': f'Real camera device at index {i}',
                        'status': 'available',
                        'resolution': f'{width}x{height}',
                        'fps': fps if fps > 0 else 30,
                        'position': 'unknown'
                    }
                    
                    # Create default settings
                    self.settings[camera_id] = {
                        'resolution': f'{width}x{height}',
                        'fps': fps if fps > 0 else 30,
                        'brightness': 50,
                        'contrast': 50,
                        'saturation': 50,
                        'exposure': 50,
                        'white_balance': 5000
                    }
                    
                    cap.release()
                    logger.info(f"Found real camera: ID={camera_id}, Resolution={width}x{height}, FPS={fps}")
                else:
                    cap.release()
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
            
            # Try to open the camera if it's a real camera (ID >= 100)
            if camera_id >= 100:
                try:
                    real_camera_index = camera_id - 100
                    cap = cv2.VideoCapture(real_camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
                    
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
                        logger.info(f"Successfully opened real camera: {camera_id}")
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
            
            # Remove from active cameras
            self.active_cameras.remove(camera_id)
            logger.info(f"Camera {camera_id} stopped successfully")
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
            
            # Create snapshots directory if it doesn't exist
            snapshot_dir = os.path.join(os.path.dirname(__file__), 'snapshots')
            os.makedirs(snapshot_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            snapshot_time = datetime.now().isoformat()
            
            # Try to get a real frame if it's a real camera
            if camera_id in self.streams:
                try:
                    cap = self.streams[camera_id]
                    ret, frame = cap.read()
                    
                    if ret:
                        # Apply camera settings
                        settings = self.settings.get(camera_id, {})
                        if 'brightness' in settings:
                            frame = cv2.convertScaleAbs(frame, alpha=settings['brightness']/50, beta=0)
                        if 'contrast' in settings:
                            frame = cv2.convertScaleAbs(frame, alpha=settings['contrast']/50, beta=0)
                        if 'saturation' in settings:
                            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            hsv[:,:,1] = cv2.convertScaleAbs(hsv[:,:,1], alpha=settings['saturation']/50)
                            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        
                        # Save the image
                        snapshot_path = os.path.join(snapshot_dir, f'snapshot_{camera_id}_{timestamp}.jpg')
                        cv2.imwrite(snapshot_path, frame)
                        
                        # Convert to base64
                        _, buffer = cv2.imencode('.jpg', frame)
                        image_data = base64.b64encode(buffer).decode('utf-8')
                        
                        logger.info(f"Snapshot saved: {snapshot_path}")
                        return {
                            'status': 'success',
                            'camera_id': camera_id,
                            'snapshot_time': snapshot_time,
                            'data': f'data:image/jpeg;base64,{image_data}',
                            'file_path': snapshot_path,
                            'message': 'Snapshot taken successfully'
                        }
                    else:
                        logger.warning(f"Could not capture frame from camera {camera_id}")
                except Exception as cv_err:
                    logger.error(f"Error capturing snapshot: {str(cv_err)}")
            
            # Return mock data if real camera not available
            mock_data = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCABkAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9XX/9k='
            
            # Save mock data to file for consistency
            mock_path = os.path.join(snapshot_dir, f'snapshot_mock_{camera_id}_{timestamp}.txt')
            with open(mock_path, 'w') as f:
                f.write(mock_data)
            
            return {
                'status': 'success',
                'camera_id': camera_id,
                'snapshot_time': snapshot_time,
                'data': mock_data,
                'file_path': mock_path,
                'message': 'Snapshot taken successfully (mock data)'
            }
    
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
        """Get a frame from a camera"""
        logger.info(f"Getting frame from camera {camera_id}")
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
                        # Apply camera settings
                        settings = self.settings.get(camera_id, {})
                        if 'brightness' in settings:
                            frame = cv2.convertScaleAbs(frame, alpha=settings['brightness']/50, beta=0)
                        if 'contrast' in settings:
                            frame = cv2.convertScaleAbs(frame, alpha=settings['contrast']/50, beta=0)
                        if 'saturation' in settings:
                            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            hsv[:,:,1] = cv2.convertScaleAbs(hsv[:,:,1], alpha=settings['saturation']/50)
                            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        
                        # Convert to base64
                        _, buffer = cv2.imencode('.jpg', frame)
                        image_data = base64.b64encode(buffer).decode('utf-8')
                        
                        return {
                            'status': 'success',
                            'camera_id': camera_id,
                            'frame_time': frame_time,
                            'data': f'data:image/jpeg;base64,{image_data}',
                            'message': 'Frame retrieved successfully'
                        }
                    else:
                        logger.warning(f"Could not capture frame from camera {camera_id}")
                except Exception as cv_err:
                    logger.error(f"Error capturing frame: {str(cv_err)}")
            
            # Return mock data if real camera not available
            mock_data = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCABkAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9XX/9k='
            
            return {
                'status': 'success',
                'camera_id': camera_id,
                'frame_time': frame_time,
                'data': mock_data,
                'message': 'Frame retrieved successfully (mock data)'
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
    
    def set_stereo_pair(self, pair_name, left_camera_id, right_camera_id):
        """Set a stereo camera pair"""
        logger.info(f"Setting stereo pair {pair_name} with left={left_camera_id}, right={right_camera_id}")
        with self.lock:
            if left_camera_id in self.cameras and right_camera_id in self.cameras:
                self.stereo_pairs[pair_name] = {
                    'left': left_camera_id,
                    'right': right_camera_id
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
        
        # In a real implementation, this would use OpenCV's stereo vision functions
        # to compute depth maps and 3D points
        # For now, we'll just return a mock depth map
        timestamp = datetime.now().isoformat()
        
        # Mock depth data generation
        # In a real implementation, this would be computed using stereo matching algorithms
        mock_depth_data = base64.b64encode(b'mock_depth_map_data').decode('utf-8')
        
        return {
            'status': 'success',
            'pair_name': pair_name,
            'left_camera_id': left_id,
            'right_camera_id': right_id,
            'timestamp': timestamp,
            'depth_data': f'data:application/octet-stream;base64,{mock_depth_data}',
            'confidence': 0.85,  # Mock confidence value
            'message': 'Stereo vision processed successfully (mock data)'
        }

# Global instance cache
_camera_manager_instance = None

def get_camera_manager():
    """Get a singleton instance of CameraManager"""
    global _camera_manager_instance
    if _camera_manager_instance is None:
        _camera_manager_instance = MockCameraManager()
        logger.info("CameraManager initialized with mock implementation")
    return _camera_manager_instance

# For backward compatibility
def initialize_camera_manager():
    """Initialize the camera manager"""
    return get_camera_manager()