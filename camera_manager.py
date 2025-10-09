<<<<<<< HEAD
# Copyright 2025 AGI System Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import threading
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CameraManager")

class CameraError(Exception):
    """Base exception for camera operations"""
    pass

class CameraNotFoundError(CameraError):
    """Raised when a camera is not found"""
    pass

class CameraAccessError(CameraError):
    """Raised when camera access is denied"""
    pass

class CameraStream:
    """Camera stream wrapper class"""
    def __init__(self, camera_id: int, camera_info: Dict[str, Any] = None):
        """Initialize camera stream
        
        Args:
            camera_id: Camera device ID
            camera_info: Optional camera information dictionary
        """
        self.camera_id = camera_id
        self.info = camera_info or {
            'name': f'Camera {camera_id}',
            'device_id': camera_id,
            'resolution': '640x480',
            'fps': 30
        }
        self.stream = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.last_frame_time = 0
        self.last_error = None
        self.settings = {
            'exposure': 50,
            'brightness': 50,
            'contrast': 50,
            'saturation': 50,
            'gain': 50,
            'zoom': 1.0
        }
        self.processing_callback = None
        
    def start(self, resolution: Tuple[int, int] = (640, 480), fps: int = 30) -> bool:
        """Start the camera stream
        
        Args:
            resolution: Camera resolution (width, height)
            fps: Frames per second
            
        Returns:
            True if stream started successfully, False otherwise
        """
        try:
            # Open camera
            self.stream = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW if cv2.OS_WIN32 else cv2.CAP_V4L2)
            
            if not self.stream.isOpened():
                raise CameraAccessError(f"Failed to open camera {self.camera_id}")
            
            # Set resolution
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            
            # Set FPS
            self.stream.set(cv2.CAP_PROP_FPS, fps)
            
            # Update camera info
            self.info['resolution'] = f'{resolution[0]}x{resolution[1]}'
            self.info['fps'] = fps
            
            # Start streaming thread
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"Started camera {self.camera_id} ({resolution[0]}x{resolution[1]}, {fps} FPS)")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to start camera {self.camera_id}: {e}")
            if self.stream:
                self.stream.release()
                self.stream = None
            return False
            
    def stop(self) -> bool:
        """Stop the camera stream
        
        Returns:
            True if stream stopped successfully, False otherwise
        """
        try:
            self.running = False
            if hasattr(self, 'thread') and self.thread.is_alive():
                self.thread.join(timeout=2.0)
            
            if self.stream:
                self.stream.release()
                self.stream = None
                
            with self.frame_lock:
                self.current_frame = None
                
            logger.info(f"Stopped camera {self.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping camera {self.camera_id}: {e}")
            return False
            
    def _capture_loop(self):
        """Internal camera capture loop"""
        while self.running:
            try:
                ret, frame = self.stream.read()
                if ret:
                    self.last_frame_time = time.time()
                    
                    # Apply processing if callback is set
                    if self.processing_callback:
                        frame = self.processing_callback(frame)
                    
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                else:
                    logger.warning(f"Failed to read frame from camera {self.camera_id}")
                    time.sleep(0.1)  # Small delay to prevent CPU usage spike
                    
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Error capturing frame from camera {self.camera_id}: {e}")
                time.sleep(0.1)
                
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the camera
        
        Returns:
            The latest frame as a numpy array, or None if no frame is available
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
            
    def take_snapshot(self) -> Optional[Dict[str, Any]]:
        """Take a snapshot from the camera
        
        Returns:
            Dictionary containing the snapshot data and metadata, or None if snapshot failed
        """
        frame = self.get_frame()
        if frame is None:
            logger.error(f"No frame available to take snapshot from camera {self.camera_id}")
            return None
            
        timestamp = int(time.time())
        
        # Encode frame to JPEG
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            logger.error(f"Failed to encode snapshot from camera {self.camera_id}")
            return None
            
        # Convert to bytes
        image_bytes = encoded_image.tobytes()
        
        return {
            'status': 'success',
            'camera_id': self.camera_id,
            'timestamp': timestamp,
            'image_data': image_bytes,
            'resolution': f'{frame.shape[1]}x{frame.shape[0]}',
            'format': 'jpeg'
        }
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the camera
        
        Returns:
            Dictionary containing camera status information
        """
        is_active = self.running and self.stream is not None and self.stream.isOpened()
        
        # Calculate frame rate
        current_time = time.time()
        time_diff = current_time - self.last_frame_time if self.last_frame_time > 0 else 1.0
        frame_rate = 1.0 / time_diff if time_diff > 0 else 0.0
        
        return {
            'is_active': is_active,
            'camera_id': self.camera_id,
            'camera_info': self.info,
            'settings': self.settings,
            'last_error': self.last_error,
            'frame_rate': frame_rate,
            'last_frame_time': self.last_frame_time
        }
        
    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """Update camera settings
        
        Args:
            settings: Dictionary of settings to update
            
        Returns:
            True if settings updated successfully, False otherwise
        """
        try:
            # Update settings dictionary
            self.settings.update(settings)
            
            # Apply settings to camera if active
            if self.running and self.stream:
                # Map settings to OpenCV properties
                setting_map = {
                    'exposure': cv2.CAP_PROP_EXPOSURE,
                    'brightness': cv2.CAP_PROP_BRIGHTNESS,
                    'contrast': cv2.CAP_PROP_CONTRAST,
                    'saturation': cv2.CAP_PROP_SATURATION,
                    'gain': cv2.CAP_PROP_GAIN
                }
                
                for key, value in settings.items():
                    if key in setting_map and hasattr(cv2, setting_map[key]):
                        self.stream.set(setting_map[key], value)
                        
            logger.info(f"Updated settings for camera {self.camera_id}: {settings}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update settings for camera {self.camera_id}: {e}")
            return False
            
    def set_processing_callback(self, callback: Callable[[np.ndarray], np.ndarray]) -> None:
        """Set a callback function for processing frames
        
        Args:
            callback: Function that takes a frame and returns a processed frame
        """
        self.processing_callback = callback
        logger.info(f"Set processing callback for camera {self.camera_id}")

class CameraManager:
    """Camera management class that handles multiple camera devices"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CameraManager, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        """Initialize the camera manager"""
        self.cameras = {}
        self.active_camera_streams = {}
        self.lock = threading.Lock()
        self.stereo_pairs = {}
        self.mock_cameras_enabled = False
        
        logger.info("CameraManager initialized")
        
    def list_available_cameras(self) -> List[Dict[str, Any]]:
        """List all available camera devices
        
        Returns:
            List of dictionaries containing camera information
        """
        max_cameras = 10
        cameras = []
        
        # Check for available cameras
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if cv2.OS_WIN32 else cv2.CAP_V4L2)
                if cap.isOpened():
                    # Get camera info
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Create camera info dictionary
                    camera_info = {
                        'id': i,
                        'name': f'Camera {i}',
                        'device_id': i,
                        'resolution': f'{int(width)}x{int(height)}',
                        'fps': int(fps) if fps > 0 else 30
                    }
                    
                    cameras.append(camera_info)
                    cap.release()
                    
                    # Small delay to prevent resource issues
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.debug(f"Camera {i} not available: {e}")
                continue
                
        # If no real cameras found, add mock cameras
        if not cameras and self.mock_cameras_enabled:
            cameras = self._get_mock_cameras()
            
        logger.info(f"Found {len(cameras)} available cameras")
        return cameras
        
    def _get_mock_cameras(self) -> List[Dict[str, Any]]:
        """Get mock camera data for testing
        
        Returns:
            List of mock camera information dictionaries
        """
        mock_cameras = [
            {
                'id': 0,
                'name': 'Mock Front Camera',
                'device_id': 0,
                'resolution': '1280x720',
                'fps': 30
            },
            {
                'id': 1,
                'name': 'Mock Rear Camera',
                'device_id': 1,
                'resolution': '1920x1080',
                'fps': 30
            },
            {
                'id': 2,
                'name': 'Mock Depth Camera',
                'device_id': 2,
                'resolution': '640x480',
                'fps': 15
            }
        ]
        
        logger.info("Returning mock cameras (no real cameras found)")
        return mock_cameras
        
    def get_active_camera_ids(self) -> List[int]:
        """Get list of active camera IDs
        
        Returns:
            List of active camera IDs
        """
        with self.lock:
            return list(self.active_camera_streams.keys())
            
    def start_camera(self, camera_id: int, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start a camera stream
        
        Args:
            camera_id: Camera device ID
            params: Dictionary of parameters (resolution, fps, etc.)
            
        Returns:
            Dictionary containing operation status and message
        """
        with self.lock:
            # Default parameters
            if params is None:
                params = {}
                
            # Parse resolution from params
            resolution_str = params.get('resolution', '640x480')
            try:
                width, height = map(int, resolution_str.split('x'))
                resolution = (width, height)
            except (ValueError, AttributeError):
                resolution = (640, 480)
                logger.warning(f"Invalid resolution format '{resolution_str}', using default 640x480")
                
            # Get FPS
            fps = params.get('framerate', params.get('fps', 30))
            try:
                fps = int(fps)
            except ValueError:
                fps = 30
                logger.warning(f"Invalid FPS value '{fps}', using default 30")
                
            # Check if camera is already active
            if camera_id in self.active_camera_streams:
                return {
                    'status': 'error',
                    'message': f'Camera {camera_id} is already active'
                }
                
            # Create camera stream
            camera_stream = CameraStream(camera_id)
            
            # Start the stream
            success = camera_stream.start(resolution, fps)
            
            if success:
                self.active_camera_streams[camera_id] = camera_stream
                return {
                    'status': 'success',
                    'message': f'Started camera {camera_id}',
                    'camera_id': camera_id,
                    'resolution': resolution,
                    'fps': fps
                }
            else:
                return {
                    'status': 'error',
                    'message': f'Failed to start camera {camera_id}: {camera_stream.last_error}'
                }
                
    def stop_camera(self, camera_id: int) -> bool:
        """Stop a camera stream
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            True if stopped successfully, False otherwise
        """
        with self.lock:
            if camera_id not in self.active_camera_streams:
                logger.warning(f'Camera {camera_id} is not active')
                return False
                
            camera_stream = self.active_camera_streams[camera_id]
            success = camera_stream.stop()
            
            if success:
                del self.active_camera_streams[camera_id]
                logger.info(f'Stopped camera {camera_id}')
                return True
            else:
                logger.error(f'Failed to stop camera {camera_id}')
                return False
                
    def take_snapshot(self, camera_id: int) -> Dict[str, Any]:
        """Take a snapshot from a camera
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            Dictionary containing snapshot data or error message
        """
        with self.lock:
            if camera_id not in self.active_camera_streams:
                return {
                    'status': 'error',
                    'message': f'Camera {camera_id} is not active'
                }
                
            camera_stream = self.active_camera_streams[camera_id]
            snapshot = camera_stream.take_snapshot()
            
            if snapshot:
                return snapshot
            else:
                return {
                    'status': 'error',
                    'message': f'Failed to take snapshot from camera {camera_id}'
                }
                
    def get_camera_status(self, camera_id: int) -> Dict[str, Any]:
        """Get the status of a specific camera
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            Dictionary containing camera status information
        """
        with self.lock:
            if camera_id not in self.active_camera_streams:
                return {
                    'is_active': False,
                    'camera_id': camera_id,
                    'message': 'Camera is not active'
                }
                
            camera_stream = self.active_camera_streams[camera_id]
            return camera_stream.get_status()
            
    def get_camera_settings(self, camera_id: int) -> Dict[str, Any]:
        """Get the current settings of a camera
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            Dictionary containing camera settings or error message
        """
        with self.lock:
            if camera_id not in self.active_camera_streams:
                return {
                    'status': 'error',
                    'message': f'Camera {camera_id} is not active'
                }
                
            camera_stream = self.active_camera_streams[camera_id]
            return {
                'status': 'success',
                'camera_id': camera_id,
                'settings': camera_stream.settings
            }
            
    def update_camera_settings(self, camera_id: int, settings: Dict[str, Any]) -> bool:
        """Update the settings of a camera
        
        Args:
            camera_id: Camera device ID
            settings: Dictionary of settings to update
            
        Returns:
            True if settings updated successfully, False otherwise
        """
        with self.lock:
            if camera_id not in self.active_camera_streams:
                logger.warning(f'Camera {camera_id} is not active')
                return False
                
            camera_stream = self.active_camera_streams[camera_id]
            success = camera_stream.update_settings(settings)
            
            if success:
                logger.info(f'Updated settings for camera {camera_id}: {settings}')
                return True
            else:
                logger.error(f'Failed to update settings for camera {camera_id}')
                return False
                
    def get_camera_frame(self, camera_id: int) -> Dict[str, Any]:
        """Get the latest frame from a camera
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            Dictionary containing the frame data or error message
        """
        with self.lock:
            if camera_id not in self.active_camera_streams:
                return {
                    'status': 'error',
                    'message': f'Camera {camera_id} is not active'
                }
                
            camera_stream = self.active_camera_streams[camera_id]
            frame = camera_stream.get_frame()
            
            if frame is not None:
                # Encode frame to JPEG
                success, encoded_image = cv2.imencode('.jpg', frame)
                if success:
                    image_bytes = encoded_image.tobytes()
                    return {
                        'status': 'success',
                        'camera_id': camera_id,
                        'image_data': image_bytes,
                        'format': 'jpeg',
                        'timestamp': int(time.time())
                    }
                else:
                    return {
                        'status': 'error',
                        'message': f'Failed to encode frame from camera {camera_id}'
                    }
            else:
                return {
                    'status': 'error',
                    'message': f'No frame available from camera {camera_id}'
                }
                
    def list_stereo_pairs(self) -> Dict[str, Any]:
        """List all stereo vision pairs
        
        Returns:
            Dictionary of stereo pairs
        """
        return self.stereo_pairs
        
    def get_stereo_pair(self, pair_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific stereo vision pair
        
        Args:
            pair_id: Stereo pair ID
            
        Returns:
            Dictionary containing stereo pair information, or None if not found
        """
        return self.stereo_pairs.get(pair_id)
        
    def set_stereo_pair(self, pair_id: str, left_camera_id: int, right_camera_id: int) -> Dict[str, Any]:
        """Set up a stereo vision pair
        
        Args:
            pair_id: Unique ID for the stereo pair
            left_camera_id: Left camera device ID
            right_camera_id: Right camera device ID
            
        Returns:
            Dictionary containing operation status and message
        """
        # Check if cameras are active
        if left_camera_id not in self.active_camera_streams:
            return {
                'status': 'error',
                'message': f'Left camera {left_camera_id} is not active'
            }
            
        if right_camera_id not in self.active_camera_streams:
            return {
                'status': 'error',
                'message': f'Right camera {right_camera_id} is not active'
            }
            
        # Create stereo pair
        self.stereo_pairs[pair_id] = {
            'id': pair_id,
            'left_camera_id': left_camera_id,
            'right_camera_id': right_camera_id,
            'enabled': False,
            'calibration_data': None,
            'created_at': int(time.time())
        }
        
        logger.info(f"Created stereo pair {pair_id} with cameras {left_camera_id} and {right_camera_id}")
        return {
            'status': 'success',
            'message': f'Created stereo pair {pair_id}',
            'pair_id': pair_id,
            'left_camera_id': left_camera_id,
            'right_camera_id': right_camera_id
        }
        
    def enable_stereo_pair(self, pair_id: str) -> Dict[str, Any]:
        """Enable a stereo vision pair
        
        Args:
            pair_id: Stereo pair ID
            
        Returns:
            Dictionary containing operation status and message
        """
        if pair_id not in self.stereo_pairs:
            return {
                'status': 'error',
                'message': f'Stereo pair {pair_id} not found'
            }
            
        stereo_pair = self.stereo_pairs[pair_id]
        
        # Check if cameras are still active
        if stereo_pair['left_camera_id'] not in self.active_camera_streams:
            return {
                'status': 'error',
                'message': f'Left camera {stereo_pair['left_camera_id']} is not active'
            }
            
        if stereo_pair['right_camera_id'] not in self.active_camera_streams:
            return {
                'status': 'error',
                'message': f'Right camera {stereo_pair['right_camera_id']} is not active'
            }
            
        # Enable the pair
        stereo_pair['enabled'] = True
        
        logger.info(f"Enabled stereo pair {pair_id}")
        return {
            'status': 'success',
            'message': f'Enabled stereo pair {pair_id}',
            'pair_id': pair_id
        }
        
    def disable_stereo_pair(self, pair_id: str) -> Dict[str, Any]:
        """Disable a stereo vision pair
        
        Args:
            pair_id: Stereo pair ID
            
        Returns:
            Dictionary containing operation status and message
        """
        if pair_id not in self.stereo_pairs:
            return {
                'status': 'error',
                'message': f'Stereo pair {pair_id} not found'
            }
            
        self.stereo_pairs[pair_id]['enabled'] = False
        
        logger.info(f"Disabled stereo pair {pair_id}")
        return {
            'status': 'success',
            'message': f'Disabled stereo pair {pair_id}',
            'pair_id': pair_id
        }
        
    def process_stereo_vision(self, pair_id: str) -> Dict[str, Any]:
        """Process stereo vision data for a pair
        
        Args:
            pair_id: Stereo pair ID
            
        Returns:
            Dictionary containing stereo vision processing results or error message
        """
        if pair_id not in self.stereo_pairs:
            return {
                'status': 'error',
                'message': f'Stereo pair {pair_id} not found'
            }
            
        stereo_pair = self.stereo_pairs[pair_id]
        
        if not stereo_pair['enabled']:
            return {
                'status': 'error',
                'message': f'Stereo pair {pair_id} is not enabled'
            }
            
        # Get frames from both cameras
        left_frame_result = self.get_camera_frame(stereo_pair['left_camera_id'])
        right_frame_result = self.get_camera_frame(stereo_pair['right_camera_id'])
        
        if left_frame_result['status'] != 'success':
            return left_frame_result
            
        if right_frame_result['status'] != 'success':
            return right_frame_result
            
        # Decode frames
        left_frame = cv2.imdecode(np.frombuffer(left_frame_result['image_data'], np.uint8), cv2.IMREAD_GRAYSCALE)
        right_frame = cv2.imdecode(np.frombuffer(right_frame_result['image_data'], np.uint8), cv2.IMREAD_GRAYSCALE)
        
        try:
            # Compute depth map (simplified implementation)
            # In a real application, you would use proper stereo calibration and more sophisticated algorithms
            stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
            disparity = stereo.compute(left_frame, right_frame)
            
            # Normalize disparity for visualization
            disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Encode depth map to JPEG
            success, encoded_depth_map = cv2.imencode('.jpg', disparity_norm)
            if not success:
                return {
                    'status': 'error',
                    'message': 'Failed to encode depth map'
                }
                
            depth_map_bytes = encoded_depth_map.tobytes()
            
            return {
                'status': 'success',
                'pair_id': pair_id,
                'depth_map_data': depth_map_bytes,
                'format': 'jpeg',
                'timestamp': int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Error processing stereo vision for pair {pair_id}: {e}")
            return {
                'status': 'error',
                'message': f'Failed to process stereo vision: {str(e)}'
            }
            
    def get_depth_data(self, pair_id: str) -> Dict[str, Any]:
        """Get depth data for a stereo pair
        
        Args:
            pair_id: Stereo pair ID
            
        Returns:
            Dictionary containing depth data or error message
        """
        # This is similar to process_stereo_vision but returns more detailed depth information
        # For simplicity, we'll just call process_stereo_vision here
        return self.process_stereo_vision(pair_id)
        
    def enable_mock_cameras(self) -> None:
        """Enable mock cameras for testing"""
        self.mock_cameras_enabled = True
        logger.info("Mock cameras enabled")
        
    def disable_mock_cameras(self) -> None:
        """Disable mock cameras"""
        self.mock_cameras_enabled = False
        logger.info("Mock cameras disabled")
        
    def cleanup(self) -> None:
        """Cleanup all camera resources"""
        with self.lock:
            # Stop all active camera streams
            for camera_id in list(self.active_camera_streams.keys()):
                try:
                    self.stop_camera(camera_id)
                except Exception as e:
                    logger.error(f"Error stopping camera {camera_id} during cleanup: {e}")
                    
            # Clear stereo pairs
            self.stereo_pairs.clear()
            
            logger.info("CameraManager cleanup completed")

# Singleton instance getter
def get_camera_manager() -> CameraManager:
    """Get the singleton instance of CameraManager
    
    Returns:
        The CameraManager instance
    """
    return CameraManager()

# Example usage
if __name__ == "__main__":
    try:
        # Get camera manager instance
        manager = get_camera_manager()
        
        # List available cameras
        cameras = manager.list_available_cameras()
        print(f"Available cameras: {cameras}")
        
        # If cameras are available, start the first one
        if cameras:
            camera_id = cameras[0]['id']
            print(f"Starting camera {camera_id}...")
            result = manager.start_camera(camera_id)
            print(f"Start result: {result}")
            
            # Take a snapshot
            if result['status'] == 'success':
                print("Taking snapshot...")
                snapshot = manager.take_snapshot(camera_id)
                print(f"Snapshot result: {'success' if snapshot['status'] == 'success' else 'failed'}")
                
                # Save snapshot to file if successful
                if snapshot['status'] == 'success':
                    snapshot_file = f"camera_{camera_id}_snapshot.jpg"
                    with open(snapshot_file, 'wb') as f:
                        f.write(snapshot['image_data'])
                    print(f"Snapshot saved to {snapshot_file}")
                
                # Stop the camera
                print(f"Stopping camera {camera_id}...")
                stop_result = manager.stop_camera(camera_id)
                print(f"Stop result: {stop_result}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        manager.cleanup()
=======
import os
import cv2
import numpy as np
import base64
import datetime
import logging
import threading
import os
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CameraManager')

# Determine snapshot directory
snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_interface', 'snapshots')
os.makedirs(snapshot_dir, exist_ok=True)

class CameraManager:
    def __init__(self):
        """Initialize the CameraManager with multi-camera support"""
        self.cameras = {}
        self.lock = threading.RLock()
        self.camera_status = {}
        logger.info("CameraManager initialized")
    
    def list_available_cameras(self, max_cameras=10):
        """List all available cameras on the system"""
        available_cameras = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to ensure camera is working
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        logger.info(f"Found {len(available_cameras)} available cameras: {available_cameras}")
        return available_cameras
    
    def initialize_camera(self, camera_id=0, resolution=(640, 480), fps=30):
        """Initialize a camera with the given ID"""
        with self.lock:
            if camera_id in self.cameras:
                logger.warning(f"Camera {camera_id} is already initialized")
                return False
            
            try:
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW if os.name == 'nt' else 0)
                if not cap.isOpened():
                    logger.error(f"Failed to open camera {camera_id}")
                    return False
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                cap.set(cv2.CAP_PROP_FPS, fps)
                
                self.cameras[camera_id] = cap
                self.camera_status[camera_id] = {
                    'resolution': resolution,
                    'fps': fps,
                    'active': True,
                    'last_frame_time': datetime.datetime.now()
                }
                
                logger.info(f"Camera {camera_id} initialized with resolution {resolution} and {fps} FPS")
                return True
            except Exception as e:
                logger.error(f"Error initializing camera {camera_id}: {str(e)}")
                return False
    
    def get_frame(self, camera_id=0, encoding='jpg'):
        """Get a frame from the specified camera"""
        with self.lock:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} is not initialized")
                return None
            
            cap = self.cameras[camera_id]
            if not cap.isOpened():
                logger.error(f"Camera {camera_id} is not open")
                return None
            
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {camera_id}")
                    return None
                
                self.camera_status[camera_id]['last_frame_time'] = datetime.datetime.now()
                
                if encoding == 'jpg':
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        return buffer.tobytes()
                    else:
                        logger.error(f"Failed to encode frame from camera {camera_id}")
                        return None
                else:
                    # Return raw frame for further processing
                    return frame
            except Exception as e:
                logger.error(f"Error getting frame from camera {camera_id}: {str(e)}")
                return None
    
    def get_frame_base64(self, camera_id=0):
        """Get a base64 encoded frame from the specified camera"""
        frame = self.get_frame(camera_id)
        if frame is None:
            return None
        return base64.b64encode(frame).decode('utf-8')
    
    def get_stereo_frames(self, left_camera_id=0, right_camera_id=1):
        """Get frames from both cameras for stereo vision"""
        left_frame = self.get_frame(left_camera_id)
        right_frame = self.get_frame(right_camera_id)
        return left_frame, right_frame
    
    def take_snapshot(self, camera_id=0, filename=None):
        """Take a snapshot from the specified camera and save it"""
        frame = self.get_frame(camera_id, encoding=None)
        if frame is None:
            logger.error(f"Failed to take snapshot from camera {camera_id}")
            return None
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"camera_{camera_id}_snapshot_{timestamp}.jpg"
        
        filepath = os.path.join(snapshot_dir, filename)
        try:
            cv2.imwrite(filepath, frame)
            logger.info(f"Snapshot saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving snapshot: {str(e)}")
            return None
    
    def release_camera(self, camera_id=0):
        """Release the specified camera"""
        with self.lock:
            if camera_id in self.cameras:
                try:
                    self.cameras[camera_id].release()
                    del self.cameras[camera_id]
                    if camera_id in self.camera_status:
                        del self.camera_status[camera_id]
                    logger.info(f"Camera {camera_id} released")
                    return True
                except Exception as e:
                    logger.error(f"Error releasing camera {camera_id}: {str(e)}")
                    return False
            else:
                logger.warning(f"Camera {camera_id} is not initialized")
                return False
    
    def release_all_cameras(self):
        """Release all initialized cameras"""
        camera_ids = list(self.cameras.keys())
        for camera_id in camera_ids:
            self.release_camera(camera_id)
        logger.info("All cameras released")
    
    def get_camera_info(self, camera_id=0):
        """Get information about the specified camera"""
        with self.lock:
            if camera_id in self.camera_status:
                return self.camera_status[camera_id].copy()
            else:
                return None
    
    def get_all_cameras_status(self):
        """Get status of all initialized cameras"""
        with self.lock:
            return self.camera_status.copy()

# Create a singleton instance of CameraManager
def get_camera_manager():
    """Get the singleton instance of CameraManager"""
    if not hasattr(get_camera_manager, '_instance'):
        get_camera_manager._instance = CameraManager()
    return get_camera_manager._instance

# Initialize camera manager instance on module load
get_camera_manager()
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
