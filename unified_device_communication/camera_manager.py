# -*- coding: utf-8 -*-
"""
Camera Manager Module
Manages multiple camera streams and provides access to camera feeds.
"""

import cv2
import threading
import time
import logging
import numpy as np
from datetime import datetime
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CameraManager")

class CameraManager:
    """Manages multiple camera streams"""
    
    def __init__(self):
        """Initialize the camera manager"""
        self.cameras = {}
        self.camera_locks = {}
        self.running = False
        self.max_cameras = 10  # Maximum number of cameras to support
        self.frame_buffer_size = 5  # Number of frames to buffer per camera
        
    def start(self):
        """Start the camera manager"""
        self.running = True
        logger.info("Camera Manager started")
        
    def stop(self):
        """Stop the camera manager and release all cameras"""
        self.running = False
        # Release all cameras
        for camera_id in list(self.cameras.keys()):
            self.release_camera(camera_id)
        logger.info("Camera Manager stopped")
    
    def list_available_cameras(self):
        """List all available cameras"""
        available_cameras = []
        
        # Try to open cameras with IDs from 0 to max_cameras-1
        for i in range(self.max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
            if cap.isOpened():
                # Get basic camera info
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available_cameras.append({
                    'id': i,
                    'width': int(width),
                    'height': int(height),
                    'fps': int(fps) if fps > 0 else 30
                })
                cap.release()
            
            # Don't try too many cameras if not found
            if i >= 3 and len(available_cameras) == 0:
                break
        
        logger.info(f"Found {len(available_cameras)} available cameras")
        return available_cameras
    
    def open_camera(self, camera_id, width=None, height=None, fps=None):
        """Open a camera with specified parameters"""
        try:
            if not self.running:
                return {'status': 'error', 'message': 'Camera Manager is not running'}
            
            if camera_id in self.cameras:
                return {'status': 'error', 'message': f'Camera {camera_id} is already open'}
            
            # Try to open the camera
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
            
            if not cap.isOpened():
                return {'status': 'error', 'message': f'Failed to open camera {camera_id}'}
            
            # Set camera parameters if specified
            if width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps is not None:
                cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Create a lock for this camera
            lock = threading.RLock()
            
            # Create a frame buffer
            frame_buffer = []
            
            # Store camera information
            self.cameras[camera_id] = {
                'capture': cap,
                'frame_buffer': frame_buffer,
                'running': True,
                'thread': None,
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30,
                'last_frame_time': time.time(),
                'error_count': 0
            }
            
            self.camera_locks[camera_id] = lock
            
            # Start a thread to capture frames
            thread = threading.Thread(target=self._capture_frames, args=(camera_id,))
            thread.daemon = True
            thread.start()
            
            self.cameras[camera_id]['thread'] = thread
            
            logger.info(f"Camera {camera_id} opened successfully ({width}x{height} @ {fps}fps)")
            return {'status': 'success', 'message': f'Camera {camera_id} opened'}
        except Exception as e:
            logger.error(f"Error opening camera {camera_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def release_camera(self, camera_id):
        """Release a camera"""
        try:
            if camera_id not in self.cameras:
                return {'status': 'error', 'message': f'Camera {camera_id} not found'}
            
            with self.camera_locks[camera_id]:
                camera_info = self.cameras[camera_id]
                camera_info['running'] = False
                
                # Release the camera capture
                if camera_info['capture'] is not None:
                    camera_info['capture'].release()
                
                # Wait for the thread to finish
                if camera_info['thread'] is not None and camera_info['thread'].is_alive():
                    camera_info['thread'].join(timeout=2.0)
            
            # Remove camera from dictionaries
            del self.cameras[camera_id]
            del self.camera_locks[camera_id]
            
            logger.info(f"Camera {camera_id} released")
            return {'status': 'success', 'message': f'Camera {camera_id} released'}
        except Exception as e:
            logger.error(f"Error releasing camera {camera_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _capture_frames(self, camera_id):
        """Thread function to capture frames from a camera"""
        while self.running:
            with self.camera_locks.get(camera_id, threading.RLock()):
                if camera_id not in self.cameras or not self.cameras[camera_id]['running']:
                    break
                
                camera_info = self.cameras[camera_id]
                cap = camera_info['capture']
                
                try:
                    # Read a frame from the camera
                    ret, frame = cap.read()
                    
                    if ret:
                        # Update last frame time
                        camera_info['last_frame_time'] = time.time()
                        camera_info['error_count'] = 0
                        
                        # Add frame to buffer
                        frame_buffer = camera_info['frame_buffer']
                        frame_buffer.append(frame.copy())
                        
                        # Remove old frames if buffer is full
                        if len(frame_buffer) > self.frame_buffer_size:
                            frame_buffer.pop(0)
                    else:
                        # Increment error count
                        camera_info['error_count'] += 1
                        logger.warning(f"Error reading frame from camera {camera_id}, error count: {camera_info['error_count']}")
                        
                        # If too many errors, consider the camera disconnected
                        if camera_info['error_count'] > 10:
                            logger.error(f"Too many errors reading from camera {camera_id}, releasing")
                            self.release_camera(camera_id)
                            break
                except Exception as e:
                    camera_info['error_count'] += 1
                    logger.error(f"Exception capturing frame from camera {camera_id}: {str(e)}")
                    
                    # If too many errors, consider the camera disconnected
                    if camera_info['error_count'] > 10:
                        logger.error(f"Too many exceptions with camera {camera_id}, releasing")
                        self.release_camera(camera_id)
                        break
            
            # Sleep for a short time to reduce CPU usage
            time.sleep(0.01)
    
    def get_frame(self, camera_id, format='bgr'):
        """Get the latest frame from a camera"""
        try:
            if camera_id not in self.cameras:
                return {'status': 'error', 'message': f'Camera {camera_id} not found'}
            
            with self.camera_locks[camera_id]:
                camera_info = self.cameras[camera_id]
                frame_buffer = camera_info['frame_buffer']
                
                if not frame_buffer:
                    return {'status': 'error', 'message': f'No frames available from camera {camera_id}'}
                
                # Get the latest frame
                frame = frame_buffer[-1].copy()
                
                # Convert format if needed
                if format == 'rgb':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif format == 'gray':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            return {'status': 'success', 'frame': frame}
        except Exception as e:
            logger.error(f"Error getting frame from camera {camera_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_camera_status(self):
        """Get the status of all cameras"""
        status = {}
        current_time = time.time()
        
        for camera_id, camera_info in self.cameras.items():
            with self.camera_locks[camera_id]:
                # Calculate time since last frame
                time_since_last_frame = current_time - camera_info['last_frame_time']
                
                status[camera_id] = {
                    'running': camera_info['running'],
                    'width': camera_info['width'],
                    'height': camera_info['height'],
                    'fps': camera_info['fps'],
                    'frame_buffer_size': len(camera_info['frame_buffer']),
                    'time_since_last_frame': time_since_last_frame,
                    'error_count': camera_info['error_count'],
                    'is_healthy': time_since_last_frame < 1.0 and camera_info['error_count'] < 5
                }
        
        return status
    
    def get_stereo_pair(self, left_camera_id, right_camera_id):
        """Get a stereo pair of frames from two cameras"""
        try:
            # Get frames from both cameras
            left_result = self.get_frame(left_camera_id)
            right_result = self.get_frame(right_camera_id)
            
            if left_result['status'] != 'success':
                return {'status': 'error', 'message': f'Left camera error: {left_result.get("message", "Unknown")}'}
            
            if right_result['status'] != 'success':
                return {'status': 'error', 'message': f'Right camera error: {right_result.get("message", "Unknown")}'}
            
            return {
                'status': 'success',
                'left_frame': left_result['frame'],
                'right_frame': right_result['frame']
            }
        except Exception as e:
            logger.error(f"Error getting stereo pair: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def set_camera_parameter(self, camera_id, param_name, value):
        """Set a camera parameter"""
        try:
            if camera_id not in self.cameras:
                return {'status': 'error', 'message': f'Camera {camera_id} not found'}
            
            with self.camera_locks[camera_id]:
                camera_info = self.cameras[camera_id]
                cap = camera_info['capture']
                
                # Map parameter name to OpenCV constant
                param_map = {
                    'width': cv2.CAP_PROP_FRAME_WIDTH,
                    'height': cv2.CAP_PROP_FRAME_HEIGHT,
                    'fps': cv2.CAP_PROP_FPS,
                    'brightness': cv2.CAP_PROP_BRIGHTNESS,
                    'contrast': cv2.CAP_PROP_CONTRAST,
                    'saturation': cv2.CAP_PROP_SATURATION,
                    'hue': cv2.CAP_PROP_HUE,
                    'gain': cv2.CAP_PROP_GAIN,
                    'exposure': cv2.CAP_PROP_EXPOSURE
                }
                
                if param_name not in param_map:
                    return {'status': 'error', 'message': f'Unsupported parameter: {param_name}'}
                
                # Set the parameter
                cap.set(param_map[param_name], value)
                
                # Update camera info
                if param_name in ['width', 'height', 'fps']:
                    camera_info[param_name] = int(cap.get(param_map[param_name]))
            
            logger.info(f"Set camera {camera_id} parameter {param_name} to {value}")
            return {'status': 'success', 'message': f'Parameter {param_name} set to {value}'}
        except Exception as e:
            logger.error(f"Error setting camera {camera_id} parameter {param_name}: {str(e)}")
            return {'status': 'error', 'message': str(e)}

# Global camera manager instance
_camera_manager = None

# Initialize camera manager
def init_camera_manager():
    """Initialize the camera manager"""
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = CameraManager()
        _camera_manager.start()
    return _camera_manager

# Cleanup camera manager
def cleanup_camera_manager():
    """Cleanup the camera manager"""
    global _camera_manager
    if _camera_manager:
        _camera_manager.stop()
        _camera_manager = None

# Get camera manager instance
def get_camera_manager():
    """Get the camera manager instance"""
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = init_camera_manager()
    return _camera_manager

# Initialize camera manager on module load
init_camera_manager()