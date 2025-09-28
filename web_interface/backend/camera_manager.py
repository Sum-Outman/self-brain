# -*- coding: utf-8 -*-
import threading
import time
import os
from datetime import datetime
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from manager_model.data_bus import get_data_bus, DataBus

class CameraManager:
    """Camera Manager class for managing multiple cameras and stereo vision functionality"""
    
    def __init__(self):
        # Store active camera streams
        self.camera_streams: Dict[str, Dict] = {}
        # Store camera settings
        self.camera_settings: Dict[str, Dict] = {}
        # Store stereo vision pairs
        self.stereo_pairs: Dict[str, Dict] = {}
        # Lock for thread safety
        self.lock = threading.Lock()
        # Directory for storing snapshots
        self.snapshot_dir = os.path.join(os.path.dirname(__file__), '../../data/snapshots')
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # DataBus integration
        self.data_bus: DataBus = get_data_bus()
        self.component_id = "camera_manager"
        
        # Register with data bus
        self._register_with_data_bus()
        
        # Subscribe to relevant channels
        self._subscribe_to_channels()
        
    def get_available_cameras(self) -> List[Dict]:
        """Get list of available cameras"""
        available_cameras = []
        # Try to open cameras from index 0 to 9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available_cameras.append({
                    'id': str(i),
                    'name': f'Camera {i}',
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                # Release the camera
                cap.release()
        return available_cameras
    
    def get_active_camera_inputs(self) -> Dict[str, Dict]:
        """Get list of active camera inputs"""
        with self.lock:
            return self.camera_streams.copy()
    
    def start_camera(self, camera_id: str, settings: Optional[Dict] = None) -> Dict:
        """Start a camera stream"""
        try:
            with self.lock:
                # Check if camera is already active
                if camera_id in self.camera_streams:
                    return {'status': 'error', 'message': f'Camera {camera_id} is already active'}
                
                # Convert camera_id to integer
                cam_id = int(camera_id)
                
                # Create VideoCapture object
                cap = cv2.VideoCapture(cam_id)
                
                if not cap.isOpened():
                    return {'status': 'error', 'message': f'Failed to open camera {camera_id}'}
                
                # Apply settings if provided
                if settings:
                    if 'resolution' in settings:
                        width, height = map(int, settings['resolution'].split('x'))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    if 'framerate' in settings:
                        cap.set(cv2.CAP_PROP_FPS, settings['framerate'])
                    if 'brightness' in settings:
                        cap.set(cv2.CAP_PROP_BRIGHTNESS, settings['brightness'] / 100.0)
                    if 'contrast' in settings:
                        cap.set(cv2.CAP_PROP_CONTRAST, settings['contrast'] / 100.0)
                    if 'saturation' in settings:
                        cap.set(cv2.CAP_PROP_SATURATION, settings['saturation'] / 100.0)
                    if 'exposure' in settings:
                        cap.set(cv2.CAP_PROP_EXPOSURE, settings['exposure'] / 100.0)
                    
                    # Save settings
                    self.camera_settings[camera_id] = settings
                
                # Get actual properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Create thread for capturing frames
                stop_event = threading.Event()
                frame_queue = []
                
                def capture_thread():
                    while not stop_event.is_set():
                        ret, frame = cap.read()
                        if ret:
                            # Convert BGR to RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # Keep only the latest frame
                            with self.lock:
                                frame_queue.clear()
                                frame_queue.append(frame)
                        time.sleep(0.01)  # Small delay to prevent CPU overload
                
                # Start the capture thread
                thread = threading.Thread(target=capture_thread)
                thread.daemon = True
                thread.start()
                
                # Store camera stream information
                self.camera_streams[camera_id] = {
                    'cap': cap,
                    'thread': thread,
                    'stop_event': stop_event,
                    'frame_queue': frame_queue,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'active': True
                }
                
                return {'status': 'success', 'message': f'Camera {camera_id} started successfully'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def stop_camera(self, camera_id: str) -> Dict:
        """Stop a camera stream"""
        try:
            with self.lock:
                if camera_id not in self.camera_streams:
                    return {'status': 'error', 'message': f'Camera {camera_id} is not active'}
                
                camera = self.camera_streams[camera_id]
                # Signal thread to stop
                camera['stop_event'].set()
                # Join thread
                camera['thread'].join(timeout=1.0)
                # Release camera
                camera['cap'].release()
                # Remove from active cameras
                del self.camera_streams[camera_id]
                
                return {'status': 'success', 'message': f'Camera {camera_id} stopped successfully'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def start_multiple_cameras(self, camera_ids: List[str]) -> Dict:
        """Start multiple cameras"""
        results = {}
        for camera_id in camera_ids:
            results[camera_id] = self.start_camera(camera_id)
        return {'status': 'success', 'results': results}
    
    def stop_all_cameras(self) -> Dict:
        """Stop all active cameras"""
        results = {}
        with self.lock:
            camera_ids = list(self.camera_streams.keys())
        for camera_id in camera_ids:
            results[camera_id] = self.stop_camera(camera_id)
        return {'status': 'success', 'results': results}
    
    def get_camera_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get the latest frame from a camera"""
        with self.lock:
            if camera_id not in self.camera_streams:
                return None
            camera = self.camera_streams[camera_id]
            if not camera['frame_queue']:
                return None
            return camera['frame_queue'][0].copy()
    
    def take_snapshot(self, camera_id: str) -> Dict:
        """Take a snapshot from a camera"""
        try:
            frame = self.get_camera_frame(camera_id)
            if frame is None:
                return {'status': 'error', 'message': f'Failed to get frame from camera {camera_id}'}
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'camera_{camera_id}_{timestamp}.jpg'
            filepath = os.path.join(self.snapshot_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            return {
                'status': 'success',
                'message': 'Snapshot taken successfully',
                'filename': filename,
                'filepath': filepath
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def create_stereo_pair(self, pair_id: str, left_camera_id: str, right_camera_id: str) -> Dict:
        """Create a stereo vision pair"""
        try:
            with self.lock:
                # Check if pair already exists
                if pair_id in self.stereo_pairs:
                    return {'status': 'error', 'message': f'Stereo pair {pair_id} already exists'}
                
                # Check if cameras are active
                if left_camera_id not in self.camera_streams or right_camera_id not in self.camera_streams:
                    return {'status': 'error', 'message': 'Both cameras must be active to create a stereo pair'}
                
                # Create stereo pair
                self.stereo_pairs[pair_id] = {
                    'id': pair_id,
                    'left_camera_id': left_camera_id,
                    'right_camera_id': right_camera_id,
                    'enabled': False
                }
                
                return {'status': 'success', 'message': f'Stereo pair {pair_id} created successfully'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def enable_stereo_vision(self, pair_id: str) -> Dict:
        """Enable stereo vision for a pair"""
        try:
            with self.lock:
                if pair_id not in self.stereo_pairs:
                    return {'status': 'error', 'message': f'Stereo pair {pair_id} not found'}
                
                pair = self.stereo_pairs[pair_id]
                # Check if cameras are active
                if pair['left_camera_id'] not in self.camera_streams or pair['right_camera_id'] not in self.camera_streams:
                    return {'status': 'error', 'message': 'Both cameras must be active to enable stereo vision'}
                
                # Enable stereo vision
                pair['enabled'] = True
                
                return {'status': 'success', 'message': f'Stereo vision enabled for pair {pair_id}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def disable_stereo_vision(self, pair_id: str) -> Dict:
        """Disable stereo vision for a pair"""
        try:
            with self.lock:
                if pair_id not in self.stereo_pairs:
                    return {'status': 'error', 'message': f'Stereo pair {pair_id} not found'}
                
                # Disable stereo vision
                self.stereo_pairs[pair_id]['enabled'] = False
                
                return {'status': 'success', 'message': f'Stereo vision disabled for pair {pair_id}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_stereo_pairs(self) -> List[Dict]:
        """Get all stereo pairs"""
        with self.lock:
            return list(self.stereo_pairs.values())
    
    def get_depth_data(self, pair_id: str) -> Dict:
        """Get depth data for a stereo pair"""
        try:
            with self.lock:
                if pair_id not in self.stereo_pairs:
                    return {'status': 'error', 'message': f'Stereo pair {pair_id} not found'}
                
                pair = self.stereo_pairs[pair_id]
                
                if not pair['enabled']:
                    return {'status': 'error', 'message': f'Stereo vision not enabled for pair {pair_id}'}
                
                # Get frames from both cameras
                left_frame = self.get_camera_frame(pair['left_camera_id'])
                right_frame = self.get_camera_frame(pair['right_camera_id'])
                
                if left_frame is None or right_frame is None:
                    return {'status': 'error', 'message': 'Failed to get frames from cameras'}
                
                # Convert to grayscale
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_RGB2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_RGB2GRAY)
                
                # Create StereoSGBM object for depth map calculation
                window_size = 3
                min_disp = 0
                num_disp = 112 - min_disp
                
                stereo = cv2.StereoSGBM_create(
                    minDisparity=min_disp,
                    numDisparities=num_disp,
                    blockSize=window_size,
                    P1=8 * 3 * window_size ** 2,
                    P2=32 * 3 * window_size ** 2,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32
                )
                
                # Compute disparity map
                disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
                
                # Normalize for display
                disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                disp_normalized = disp_normalized.astype(np.uint8)
                
                # Apply colormap
                depth_colormap = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
                
                # Convert to base64 for JSON transmission
                _, buffer = cv2.imencode('.png', depth_colormap)
                depth_data = buffer.tobytes()
                
                return {
                    'status': 'success',
                    'message': 'Depth data computed successfully',
                    'depth_data': depth_data.hex()  # Convert bytes to hex string for JSON
                }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def update_camera_settings(self, camera_id: str, settings: Dict) -> Dict:
        """Update camera settings"""
        try:
            with self.lock:
                if camera_id not in self.camera_streams:
                    return {'status': 'error', 'message': f'Camera {camera_id} is not active'}
                
                camera = self.camera_streams[camera_id]
                cap = camera['cap']
                
                # Apply settings
                if 'resolution' in settings:
                    width, height = map(int, settings['resolution'].split('x'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    camera['width'] = width
                    camera['height'] = height
                if 'framerate' in settings:
                    cap.set(cv2.CAP_PROP_FPS, settings['framerate'])
                    camera['fps'] = settings['framerate']
                if 'brightness' in settings:
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, settings['brightness'] / 100.0)
                if 'contrast' in settings:
                    cap.set(cv2.CAP_PROP_CONTRAST, settings['contrast'] / 100.0)
                if 'saturation' in settings:
                    cap.set(cv2.CAP_PROP_SATURATION, settings['saturation'] / 100.0)
                if 'exposure' in settings:
                    cap.set(cv2.CAP_PROP_EXPOSURE, settings['exposure'] / 100.0)
                
                # Save settings
                if camera_id not in self.camera_settings:
                    self.camera_settings[camera_id] = {}
                self.camera_settings[camera_id].update(settings)
                
                return {'status': 'success', 'message': f'Camera {camera_id} settings updated successfully'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_camera_settings(self, camera_id: str) -> Dict:
        """Get camera settings"""
        with self.lock:
            if camera_id not in self.camera_settings:
                return {'status': 'error', 'message': f'No settings found for camera {camera_id}'}
            
            return {'status': 'success', 'settings': self.camera_settings[camera_id].copy()}
    
    def _register_with_data_bus(self):
        """Register this component with the data bus"""
        try:
            # Register the component with its capabilities
            self.data_bus.register_component(
                component_id=self.component_id,
                metadata={
                    "name": "Camera Manager",
                    "version": "1.0",
                    "type": "vision",
                    "capabilities": [
                        "camera_management",
                        "multi_camera_support",
                        "stereo_vision",
                        "depth_perception",
                        "snapshot_capture"
                    ],
                    "dependencies": []
                }
            )
        except Exception as e:
            print(f"Error registering with data bus: {e}")
    
    def _subscribe_to_channels(self):
        """Subscribe to relevant data bus channels"""
        try:
            # Subscribe to camera control channel
            self.data_bus.subscribe(
                channel_id="camera_control",
                component_id=self.component_id,
                handler=self._handle_camera_control
            )
            
            # Subscribe to stereo vision control channel
            self.data_bus.subscribe(
                channel_id="stereo_vision_control",
                component_id=self.component_id,
                handler=self._handle_stereo_vision_control
            )
        except Exception as e:
            print(f"Error subscribing to channels: {e}")
    
    def _handle_camera_control(self, message: Dict):
        """Handle camera control messages from the data bus"""
        try:
            action = message.get("action", "")
            
            if action == "start":
                camera_id = message.get("camera_id", "")
                settings = message.get("settings", None)
                result = self.start_camera(camera_id, settings)
                
                # Publish the result back to the data bus
                self.data_bus.publish(
                    channel_id="camera_status",
                    message={
                        "request_id": message.get("request_id", ""),
                        "camera_id": camera_id,
                        "status": "started",
                        "result": result
                    }
                )
            elif action == "stop":
                camera_id = message.get("camera_id", "")
                result = self.stop_camera(camera_id)
                
                # Publish the result back to the data bus
                self.data_bus.publish(
                    channel_id="camera_status",
                    message={
                        "request_id": message.get("request_id", ""),
                        "camera_id": camera_id,
                        "status": "stopped",
                        "result": result
                    }
                )
            elif action == "update_settings":
                camera_id = message.get("camera_id", "")
                settings = message.get("settings", {})
                result = self.update_camera_settings(camera_id, settings)
                
                # Publish the result back to the data bus
                self.data_bus.publish(
                    channel_id="camera_status",
                    message={
                        "request_id": message.get("request_id", ""),
                        "camera_id": camera_id,
                        "status": "settings_updated",
                        "result": result
                    }
                )
        except Exception as e:
            print(f"Error handling camera control message: {e}")
    
    def _handle_stereo_vision_control(self, message: Dict):
        """Handle stereo vision control messages from the data bus"""
        try:
            action = message.get("action", "")
            
            if action == "create_pair":
                pair_id = message.get("pair_id", "")
                left_camera_id = message.get("left_camera_id", "")
                right_camera_id = message.get("right_camera_id", "")
                result = self.create_stereo_pair(pair_id, left_camera_id, right_camera_id)
                
                # Publish the result back to the data bus
                self.data_bus.publish(
                    channel_id="stereo_vision_status",
                    message={
                        "request_id": message.get("request_id", ""),
                        "pair_id": pair_id,
                        "status": "created",
                        "result": result
                    }
                )
            elif action == "enable":
                pair_id = message.get("pair_id", "")
                result = self.enable_stereo_vision(pair_id)
                
                # Publish the result back to the data bus
                self.data_bus.publish(
                    channel_id="stereo_vision_status",
                    message={
                        "request_id": message.get("request_id", ""),
                        "pair_id": pair_id,
                        "status": "enabled",
                        "result": result
                    }
                )
            elif action == "disable":
                pair_id = message.get("pair_id", "")
                result = self.disable_stereo_vision(pair_id)
                
                # Publish the result back to the data bus
                self.data_bus.publish(
                    channel_id="stereo_vision_status",
                    message={
                        "request_id": message.get("request_id", ""),
                        "pair_id": pair_id,
                        "status": "disabled",
                        "result": result
                    }
                )
            elif action == "get_depth":
                pair_id = message.get("pair_id", "")
                result = self.get_depth_data(pair_id)
                
                # Publish the depth data back to the data bus
                if result.get("status") == "success":
                    self.data_bus.publish(
                        channel_id="depth_data_update",
                        message={
                            "request_id": message.get("request_id", ""),
                            "pair_id": pair_id,
                            "data": result.get("depth_data", ""),
                            "timestamp": time.time()
                        }
                    )
                
                # Publish the result to status channel
                self.data_bus.publish(
                    channel_id="stereo_vision_status",
                    message={
                        "request_id": message.get("request_id", ""),
                        "pair_id": pair_id,
                        "status": "depth_data",
                        "result": result
                    }
                )
        except Exception as e:
            print(f"Error handling stereo vision control message: {e}")
    
    def close(self):
        """Close all cameras and clean up"""
        # Stop all cameras
        self.stop_all_cameras()
        
        # Unregister from data bus
        try:
            self.data_bus.unregister_component(self.component_id)
        except Exception as e:
            print(f"Error unregistering from data bus: {e}")

# Create a global instance of CameraManager
global_camera_manager = CameraManager()