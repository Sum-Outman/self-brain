# -*- coding: utf-8 -*-
"""
Unified Device Communication Module
Provides a unified interface for communicating with various hardware devices.
"""

from flask import Blueprint, request, jsonify, Response
import logging
import threading
import time
from datetime import datetime
import json
import os
import sys
import serial
import serial.tools.list_ports
import cv2
import numpy as np
import psutil
import platform
from pathlib import Path

# Import Camera Manager
try:
    from camera_manager import get_camera_manager
    camera_manager = get_camera_manager()
    logger.info("Camera manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize camera manager: {str(e)}")
    camera_manager = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeviceCommunication")

# Create blueprint for device communication API
device_bp = Blueprint('device_communication', __name__)

# Global device manager instance
_device_manager = None

class DeviceManager:
    """Manager for all device communications"""
    
    def __init__(self):
        """Initialize the device manager"""
        self.devices = {}
        self.connected_devices = {}
        # Use the module-level camera_manager
        self.serial_ports = {}
        self.is_running = False
        self.monitor_thread = None
        self.device_status = {}
        self.stereo_pairs = {}
        self.mock_cameras_enabled = False
        
    def start(self):
        """Start the device manager"""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_devices)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Device Manager started")
    
    def stop(self):
        """Stop the device manager"""
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        # Close all connected devices
        for device_id, device in self.connected_devices.items():
            self.disconnect_device(device_id)
        logger.info("Device Manager stopped")
    
    def _monitor_devices(self):
        """Monitor devices for connection/disconnection"""
        while self.is_running:
            # Update serial ports
            self._update_serial_ports()
            
            # Update device status
            self._update_device_status()
            
            # Sleep for a while
            time.sleep(5)
    
    def _update_serial_ports(self):
        """Update the list of available serial ports"""
        try:
            available_ports = serial.tools.list_ports.comports()
            current_ports = {port.device: port for port in available_ports}
            
            # Check for new ports
            for port_name, port_info in current_ports.items():
                if port_name not in self.serial_ports:
                    self.serial_ports[port_name] = port_info
                    logger.info(f"New serial port detected: {port_name} ({port_info.description})")
                    # Emit event via data bus if available
                    try:
                        from data_bus import get_data_bus
                        data_bus = get_data_bus()
                        data_bus.publish('device_event', {
                            'type': 'serial_port_connected',
                            'port': port_name,
                            'description': port_info.description
                        })
                    except:
                        pass
            
            # Check for removed ports
            for port_name in list(self.serial_ports.keys()):
                if port_name not in current_ports:
                    # Disconnect if this port was connected
                    if port_name in self.connected_devices:
                        self.disconnect_device(port_name)
                    del self.serial_ports[port_name]
                    logger.info(f"Serial port removed: {port_name}")
                    # Emit event via data bus if available
                    try:
                        from data_bus import get_data_bus
                        data_bus = get_data_bus()
                        data_bus.publish('device_event', {
                            'type': 'serial_port_disconnected',
                            'port': port_name
                        })
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error updating serial ports: {str(e)}")
    
    def _update_device_status(self):
        """Update the status of all devices"""
        global camera_manager
        # Update camera status
        if camera_manager:
            try:
                camera_status = camera_manager.get_camera_status()
                self.device_status['cameras'] = camera_status
            except Exception as e:
                logger.error(f"Error updating camera status: {str(e)}")
                self.device_status['cameras'] = {'status': 'error', 'message': str(e)}
        else:
            self.device_status['cameras'] = {'status': 'not_initialized'}
        
        # Update serial port status
        try:
            serial_status = {}
            for port_name, port_info in self.serial_ports.items():
                serial_status[port_name] = {
                    'connected': port_name in self.connected_devices,
                    'baudrate': port_info.get('baudrate', None),
                    'status': self.connected_devices.get(port_name, {}).get('status', 'disconnected')
                }
            self.device_status['serial_ports'] = serial_status
        except Exception as e:
            logger.error(f"Error updating serial port status: {str(e)}")
            self.device_status['serial_ports'] = {'status': 'error', 'message': str(e)}
        
        # Update stereo vision status
        if camera_manager:
            try:
                stereo_pairs = camera_manager.list_stereo_pairs()
                self.device_status['stereo_vision'] = {
                    'status': 'active',
                    'stereo_pairs': stereo_pairs,
                    'mock_cameras_enabled': getattr(camera_manager, 'mock_cameras_enabled', False)
                }
            except Exception as e:
                logger.error(f"Error updating stereo vision status: {str(e)}")
                self.device_status['stereo_vision'] = {'status': 'error', 'message': str(e)}
        else:
            self.device_status['stereo_vision'] = {'status': 'not_initialized'}
    
    def connect_device(self, device_id, device_type, **kwargs):
        """Connect to a device"""
        try:
            if device_id in self.connected_devices:
                return {'status': 'error', 'message': f'Device {device_id} is already connected'}
            
            if device_type == 'serial':
                port = kwargs.get('port')
                baudrate = kwargs.get('baudrate', 9600)
                
                if not port:
                    return {'status': 'error', 'message': 'Port is required for serial device'}
                
                # Check if port exists
                if port not in [p.device for p in serial.tools.list_ports.comports()]:
                    return {'status': 'error', 'message': f'Port {port} not found'}
                
                # Connect to serial port
                ser = serial.Serial(
                    port=port,
                    baudrate=baudrate,
                    timeout=1
                )
                
                if ser.is_open:
                    self.connected_devices[device_id] = ser
                    logger.info(f"Connected to serial device {device_id} on port {port}")
                    return {'status': 'success', 'message': f'Connected to {device_id}'}
                else:
                    return {'status': 'error', 'message': f'Failed to open port {port}'}
            
            return {'status': 'error', 'message': f'Unknown device type: {device_type}'}
        except Exception as e:
            logger.error(f"Error connecting to device {device_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def disconnect_device(self, device_id):
        """Disconnect from a device"""
        try:
            if device_id not in self.connected_devices:
                return {'status': 'error', 'message': f'Device {device_id} not found'}
            
            device = self.connected_devices[device_id]
            
            if isinstance(device, serial.Serial) and device.is_open:
                device.close()
            
            del self.connected_devices[device_id]
            logger.info(f"Disconnected from device {device_id}")
            return {'status': 'success', 'message': f'Disconnected from {device_id}'}
        except Exception as e:
            logger.error(f"Error disconnecting from device {device_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def send_command(self, device_id, command, **kwargs):
        """Send a command to a device"""
        try:
            if device_id not in self.connected_devices:
                return {'status': 'error', 'message': f'Device {device_id} not connected'}
            
            device = self.connected_devices[device_id]
            
            if isinstance(device, serial.Serial):
                # Send command to serial device
                if not device.is_open:
                    return {'status': 'error', 'message': f'Device {device_id} port is closed'}
                
                # Ensure command ends with newline
                if not command.endswith('\n'):
                    command += '\n'
                
                # Send command
                device.write(command.encode())
                
                # Read response if requested
                if kwargs.get('read_response', False):
                    timeout = kwargs.get('timeout', 1.0)
                    start_time = time.time()
                    response = ''
                    
                    while time.time() - start_time < timeout:
                        if device.in_waiting:
                            response += device.read(device.in_waiting).decode()
                            if '\n' in response:
                                break
                        time.sleep(0.1)
                    
                    return {'status': 'success', 'response': response.strip()}
                
                return {'status': 'success', 'message': 'Command sent'}
            
            return {'status': 'error', 'message': f'Command not supported for device type'}
        except Exception as e:
            logger.error(f"Error sending command to device {device_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def list_available_devices(self):
        """List all available devices"""
        global camera_manager
        devices = {
            'serial_ports': list(self.serial_ports.values())
        }
        
        # Add camera information if camera manager is available
        if camera_manager:
            try:
                cameras = camera_manager.get_camera_list()
                devices['cameras'] = cameras
            except Exception as e:
                logger.error(f"Error getting camera list: {str(e)}")
                devices['cameras'] = {'status': 'error', 'message': str(e)}
        else:
            devices['cameras'] = {'status': 'not_initialized'}
            
        return devices
        
    def get_camera_list(self):
        """Get list of all available cameras"""
        global camera_manager
        if camera_manager:
            try:
                return camera_manager.get_camera_list()
            except Exception as e:
                logger.error(f"Error getting camera list: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'not_initialized'}
            
    def start_camera(self, camera_id):
        """Start a specific camera"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.start_camera(camera_id)
                return result
            except Exception as e:
                logger.error(f"Error starting camera {camera_id}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def stop_camera(self, camera_id):
        """Stop a specific camera"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.stop_camera(camera_id)
                return result
            except Exception as e:
                logger.error(f"Error stopping camera {camera_id}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def get_camera_frame(self, camera_id):
        """Get a frame from a specific camera"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.get_frame(camera_id)
                return result
            except Exception as e:
                logger.error(f"Error getting frame from camera {camera_id}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def capture_snapshot(self, camera_id):
        """Capture a snapshot from a specific camera"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.capture_snapshot(camera_id)
                return result
            except Exception as e:
                logger.error(f"Error capturing snapshot from camera {camera_id}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def list_stereo_pairs(self):
        """List all configured stereo pairs"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.list_stereo_pairs()
                return result
            except Exception as e:
                logger.error(f"Error listing stereo pairs: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def get_stereo_pair(self, pair_name):
        """Get a specific stereo pair"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.get_stereo_pair(pair_name)
                return result
            except Exception as e:
                logger.error(f"Error getting stereo pair {pair_name}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def set_stereo_pair(self, pair_name, left_camera_id, right_camera_id):
        """Set a specific stereo pair"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.set_stereo_pair(pair_name, left_camera_id, right_camera_id)
                return result
            except Exception as e:
                logger.error(f"Error setting stereo pair {pair_name}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def enable_stereo_pair(self, pair_name):
        """Enable a specific stereo pair"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.enable_stereo_pair(pair_name)
                return result
            except Exception as e:
                logger.error(f"Error enabling stereo pair {pair_name}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def disable_stereo_pair(self, pair_name):
        """Disable a specific stereo pair"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.disable_stereo_pair(pair_name)
                return result
            except Exception as e:
                logger.error(f"Error disabling stereo pair {pair_name}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def process_stereo_vision(self, pair_name):
        """Process stereo vision for a specific pair"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.process_stereo_vision(pair_name)
                return result
            except Exception as e:
                logger.error(f"Error processing stereo vision for pair {pair_name}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def get_depth_data(self, pair_name):
        """Get depth data for a specific stereo pair"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.get_depth_data(pair_name)
                return result
            except Exception as e:
                logger.error(f"Error getting depth data for pair {pair_name}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def enable_mock_cameras(self):
        """Enable mock cameras for testing"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.enable_mock_cameras()
                return result
            except Exception as e:
                logger.error(f"Error enabling mock cameras: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def disable_mock_cameras(self):
        """Disable mock cameras"""
        global camera_manager
        if camera_manager:
            try:
                result = camera_manager.disable_mock_cameras()
                return result
            except Exception as e:
                logger.error(f"Error disabling mock cameras: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': 'Camera manager not initialized'}
            
    def get_device_list(self):
        """Get list of all devices"""
        try:
            devices = {
                'serial_ports': [{
                    'port': port.device,
                    'description': port.description,
                    'hwid': port.hwid
                } for port in serial.tools.list_ports.comports()],
                'connected_devices': list(self.connected_devices.keys()),
                'device_status': self.device_status
            }
            return {'status': 'success', 'devices': devices}
        except Exception as e:
            logger.error(f"Error getting device list: {str(e)}")
            return {'status': 'error', 'message': str(e)}

# Initialize device communication
def init_device_communication():
    """Initialize the device communication module"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
        _device_manager.start()
    return _device_manager

# Cleanup device communication
def cleanup_device_communication():
    """Cleanup the device communication module"""
    global _device_manager
    if _device_manager:
        _device_manager.stop()
        _device_manager = None

# Get device manager instance
def get_device_manager():
    """Get the device manager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = init_device_communication()
    return _device_manager

# API endpoints
@device_bp.route('/api/devices/list', methods=['GET'])
def list_devices():
    """List all available devices"""
    device_manager = get_device_manager()
    return jsonify(device_manager.get_device_list())

@device_bp.route('/api/devices/connect', methods=['POST'])
def connect_device_api():
    """Connect to a device"""
    data = request.json
    device_id = data.get('device_id')
    device_type = data.get('device_type')
    
    if not device_id or not device_type:
        return jsonify({'status': 'error', 'message': 'device_id and device_type are required'})
    
    device_manager = get_device_manager()
    result = device_manager.connect_device(device_id, device_type, **data)
    return jsonify(result)

@device_bp.route('/api/devices/disconnect/<device_id>', methods=['POST'])
def disconnect_device_api(device_id):
    """Disconnect from a device"""
    device_manager = get_device_manager()
    result = device_manager.disconnect_device(device_id)
    return jsonify(result)

@device_bp.route('/api/devices/send/<device_id>', methods=['POST'])
def send_command_api(device_id):
    """Send a command to a device"""
    data = request.json
    command = data.get('command')
    
    if not command:
        return jsonify({'status': 'error', 'message': 'Command is required'})
    
    device_manager = get_device_manager()
    result = device_manager.send_command(device_id, command, **data)
    return jsonify(result)

@device_bp.route('/api/devices/status', methods=['GET'])
def get_device_status():
    """Get the status of all devices"""
    device_manager = get_device_manager()
    return jsonify({'status': 'success', 'status_data': device_manager.device_status})

# Camera related API endpoints
@device_bp.route('/api/cameras/list', methods=['GET'])
def list_cameras():
    """List all available cameras"""
    device_manager = get_device_manager()
    result = device_manager.get_camera_list()
    return jsonify(result)

@device_bp.route('/api/cameras/start/<int:camera_id>', methods=['POST'])
def start_camera_api(camera_id):
    """Start a specific camera"""
    device_manager = get_device_manager()
    result = device_manager.start_camera(camera_id)
    return jsonify(result)

@device_bp.route('/api/cameras/stop/<int:camera_id>', methods=['POST'])
def stop_camera_api(camera_id):
    """Stop a specific camera"""
    device_manager = get_device_manager()
    result = device_manager.stop_camera(camera_id)
    return jsonify(result)

@device_bp.route('/api/cameras/frame/<int:camera_id>', methods=['GET'])
def get_camera_frame_api(camera_id):
    """Get a frame from a specific camera"""
    device_manager = get_device_manager()
    result = device_manager.get_camera_frame(camera_id)
    return jsonify(result)

@device_bp.route('/api/cameras/snapshot/<int:camera_id>', methods=['POST'])
def capture_snapshot_api(camera_id):
    """Capture a snapshot from a specific camera"""
    device_manager = get_device_manager()
    result = device_manager.capture_snapshot(camera_id)
    return jsonify(result)

# Stereo vision related API endpoints
@device_bp.route('/api/stereo/pairs', methods=['GET'])
def list_stereo_pairs_api():
    """List all configured stereo pairs"""
    device_manager = get_device_manager()
    result = device_manager.list_stereo_pairs()
    return jsonify(result)

@device_bp.route('/api/stereo/pairs/<string:pair_name>', methods=['GET'])
def get_stereo_pair_api(pair_name):
    """Get a specific stereo pair"""
    device_manager = get_device_manager()
    result = device_manager.get_stereo_pair(pair_name)
    if result:
        return jsonify({'status': 'success', 'stereo_pair': result})
    else:
        return jsonify({'status': 'error', 'message': f'Stereo pair {pair_name} not found'}), 404

@device_bp.route('/api/stereo/pairs/<string:pair_name>', methods=['POST'])
def set_stereo_pair_api(pair_name):
    """Set a specific stereo pair"""
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    
    data = request.json
    left_camera_id = data.get('left_camera_id') or data.get('left')
    right_camera_id = data.get('right_camera_id') or data.get('right')
    
    if left_camera_id is None or right_camera_id is None:
        return jsonify({'status': 'error', 'message': 'Both left and right camera IDs are required'}), 400
    
    # Convert camera IDs to integers if they are string representations of integers
    if isinstance(left_camera_id, str) and left_camera_id.isdigit():
        left_camera_id = int(left_camera_id)
    if isinstance(right_camera_id, str) and right_camera_id.isdigit():
        right_camera_id = int(right_camera_id)
    
    device_manager = get_device_manager()
    result = device_manager.set_stereo_pair(pair_name, left_camera_id, right_camera_id)
    return jsonify(result)

@device_bp.route('/api/stereo/pairs/<string:pair_name>/enable', methods=['POST'])
def enable_stereo_pair_api(pair_name):
    """Enable a specific stereo pair"""
    device_manager = get_device_manager()
    result = device_manager.enable_stereo_pair(pair_name)
    return jsonify(result)

@device_bp.route('/api/stereo/pairs/<string:pair_name>/disable', methods=['POST'])
def disable_stereo_pair_api(pair_name):
    """Disable a specific stereo pair"""
    device_manager = get_device_manager()
    result = device_manager.disable_stereo_pair(pair_name)
    return jsonify(result)

@device_bp.route('/api/stereo/process/<string:pair_name>', methods=['GET'])
def process_stereo_vision_api(pair_name):
    """Process stereo vision for a specific pair"""
    device_manager = get_device_manager()
    result = device_manager.process_stereo_vision(pair_name)
    return jsonify(result)

@device_bp.route('/api/stereo/depth/<string:pair_name>', methods=['GET'])
def get_depth_data_api(pair_name):
    """Get depth data for a specific stereo pair"""
    device_manager = get_device_manager()
    result = device_manager.get_depth_data(pair_name)
    return jsonify(result)

# Mock cameras API endpoints
@device_bp.route('/api/cameras/mock/enable', methods=['POST'])
def enable_mock_cameras_api():
    """Enable mock cameras"""
    device_manager = get_device_manager()
    result = device_manager.enable_mock_cameras()
    return jsonify(result)

@device_bp.route('/api/cameras/mock/disable', methods=['POST'])
def disable_mock_cameras_api():
    """Disable mock cameras"""
    device_manager = get_device_manager()
    result = device_manager.disable_mock_cameras()
    return jsonify(result)

# Initialize device communication on module load
init_device_communication()