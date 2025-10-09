# -*- coding: utf-8 -*-
"""
Enhanced Unified Device Communication Module
Provides a comprehensive interface for communicating with various hardware devices.
"""

from flask import Blueprint, request, jsonify
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
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedDeviceCommunication")

# Create blueprint for device communication API
device_bp = Blueprint('device_communication', __name__)

# Global device manager instance
_device_manager = None

class DeviceManager:
    """Enhanced Manager for all device communications"""
    
    def __init__(self):
        """Initialize the device manager"""
        self.devices = {}
        self.connected_devices = {}
        self.camera_manager = None
        self.serial_ports = {}
        self.is_running = False
        self.monitor_thread = None
        self.device_status = {}
        self.lock = threading.RLock()
        self.sensor_readings = {}
        self.data_buffer_size = 100
        self.event_callbacks = {}
        self.data_receivers = {}
        
        # Initialize system sensor data collection
        self.system_sensors = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'temperature': None,
            'process_count': 0,
            'network_stats': {}
        }
        
        # Support for multiple cameras
        self.camera_connections = {}
    
    def set_camera_manager(self, camera_manager):
        """Set the camera manager instance"""
        self.camera_manager = camera_manager
        logger.info("Camera manager set")
    
    def start(self):
        """Start the device manager"""
        with self.lock:
            if self.is_running:
                return
            
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_devices)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            # Start system sensor monitoring thread
            self.system_sensor_thread = threading.Thread(target=self._monitor_system_sensors)
            self.system_sensor_thread.daemon = True
            self.system_sensor_thread.start()
            
            logger.info("Enhanced Device Manager started")
    
    def stop(self):
        """Stop the device manager"""
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2.0)
            
            # Close all connected devices
            for device_id in list(self.connected_devices.keys()):
                self.disconnect_device(device_id)
            
            # Close all camera connections
            for camera_id in list(self.camera_connections.keys()):
                self.close_camera(camera_id)
            
            logger.info("Enhanced Device Manager stopped")
    
    def _monitor_devices(self):
        """Monitor devices for connection/disconnection"""
        while self.is_running:
            try:
                # Update serial ports
                self._update_serial_ports()
                
                # Update device status
                self._update_device_status()
                
                # Check for data from connected devices
                self._check_device_data()
                
            except Exception as e:
                logger.error(f"Error in device monitoring: {str(e)}")
            
            # Sleep for a while
            time.sleep(0.5)  # More frequent checks for better responsiveness
    
    def _monitor_system_sensors(self):
        """Monitor system sensors"""
        while self.is_running:
            try:
                # Get CPU usage
                cpu_usage = psutil.cpu_percent(interval=0.1)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_usage = memory.percent / 100.0  # Convert to decimal
                
                # Get disk usage
                disk = psutil.disk_usage('/')
                disk_usage = disk.percent / 100.0  # Convert to decimal
                
                # Get temperature (if available)
                temperature = None
                try:
                    if hasattr(psutil, 'sensors_temperatures'):
                        temps = psutil.sensors_temperatures()
                        if temps:
                            # Try different temperature sensors
                            for sensor_name in ['cpu-thermal', 'coretemp', 'acpitz']:
                                if sensor_name in temps:
                                    temperature = temps[sensor_name][0].current
                                    break
                except Exception as e:
                    logger.debug(f"Error getting temperature: {e}")
                
                # Get process count
                process_count = len(psutil.pids())
                
                # Get network stats
                net_io = psutil.net_io_counters()
                network_stats = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
                
                with self.lock:
                    self.system_sensors = {
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage,
                        'disk_usage': disk_usage,
                        'temperature': temperature,
                        'process_count': process_count,
                        'network_stats': network_stats,
                        'timestamp': time.time()
                    }
                
            except Exception as e:
                logger.error(f"Error in system sensor monitoring: {str(e)}")
            
            time.sleep(1.0)
    
    def _update_serial_ports(self):
        """Update the list of available serial ports"""
        try:
            available_ports = serial.tools.list_ports.comports()
            current_ports = {port.device: port for port in available_ports}
            
            with self.lock:
                # Check for new ports
                for port_name, port_info in current_ports.items():
                    if port_name not in self.serial_ports:
                        self.serial_ports[port_name] = port_info
                        logger.info(f"New serial port detected: {port_name} ({port_info.description})")
                        self._trigger_event('serial_port_connected', {
                            'port': port_name,
                            'description': port_info.description
                        })
                
                # Check for removed ports
                for port_name in list(self.serial_ports.keys()):
                    if port_name not in current_ports:
                        # Disconnect if this port was connected
                        if port_name in self.connected_devices:
                            self.disconnect_device(port_name)
                        del self.serial_ports[port_name]
                        logger.info(f"Serial port removed: {port_name}")
                        self._trigger_event('serial_port_disconnected', {
                            'port': port_name
                        })
        except Exception as e:
            logger.error(f"Error updating serial ports: {str(e)}")
    
    def _update_device_status(self):
        """Update the status of all connected devices"""
        try:
            # Update camera status
            camera_status = {}
            if self.camera_manager:
                camera_status = self.camera_manager.get_camera_status()
            
            # Update serial device status
            serial_status = {}
            with self.lock:
                for device_id, device in self.connected_devices.items():
                    if isinstance(device, serial.Serial):
                        serial_status[device_id] = {
                            'connected': device.is_open,
                            'port': device.port,
                            'baudrate': device.baudrate,
                            'in_waiting': device.in_waiting if device.is_open else 0
                        }
                
                # Update system sensors status
                system_sensors_status = self.system_sensors.copy()
                
                # Update camera connections status
                camera_connections_status = {}
                for camera_id, cap in self.camera_connections.items():
                    camera_connections_status[camera_id] = {
                        'connected': cap.isOpened(),
                        'camera_id': camera_id
                    }
                
                self.device_status = {
                    'cameras': camera_status,
                    'serial': serial_status,
                    'system_sensors': system_sensors_status,
                    'camera_connections': camera_connections_status,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            logger.error(f"Error updating device status: {str(e)}")
    
    def _check_device_data(self):
        """Check for data from connected devices"""
        try:
            with self.lock:
                for device_id, device in list(self.connected_devices.items()):
                    if isinstance(device, serial.Serial) and device.is_open and device.in_waiting:
                        # Read data from serial device
                        try:
                            data = device.readline().decode('utf-8', errors='ignore').strip()
                            if data:
                                self._process_device_data(device_id, data)
                        except Exception as e:
                            logger.error(f"Error reading data from device {device_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Error checking device data: {str(e)}")
    
    def _process_device_data(self, device_id, data):
        """Process data received from a device"""
        try:
            # Try to parse JSON data
            try:
                parsed_data = json.loads(data)
                data_type = parsed_data.get('type', 'unknown')
                
                # Add timestamp if not present
                if 'timestamp' not in parsed_data:
                    parsed_data['timestamp'] = time.time()
                
                # Store the data
                if device_id not in self.sensor_readings:
                    self.sensor_readings[device_id] = []
                
                self.sensor_readings[device_id].append(parsed_data)
                
                # Limit the buffer size
                if len(self.sensor_readings[device_id]) > self.data_buffer_size:
                    self.sensor_readings[device_id].pop(0)
                
                # Trigger data received event
                self._trigger_event('device_data_received', {
                    'device_id': device_id,
                    'data': parsed_data
                })
                
            except json.JSONDecodeError:
                # Handle non-JSON data
                if device_id not in self.sensor_readings:
                    self.sensor_readings[device_id] = []
                
                raw_data = {
                    'raw': data,
                    'timestamp': time.time(),
                    'type': 'raw'
                }
                
                self.sensor_readings[device_id].append(raw_data)
                
                # Limit the buffer size
                if len(self.sensor_readings[device_id]) > self.data_buffer_size:
                    self.sensor_readings[device_id].pop(0)
                
                # Trigger raw data received event
                self._trigger_event('raw_device_data_received', {
                    'device_id': device_id,
                    'data': raw_data
                })
        except Exception as e:
            logger.error(f"Error processing device data: {str(e)}")
    
    def _trigger_event(self, event_type, data):
        """Trigger an event with the given data"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in event callback for {event_type}: {str(e)}")
    
    def register_event_callback(self, event_type, callback):
        """Register a callback for a specific event type"""
        with self.lock:
            if event_type not in self.event_callbacks:
                self.event_callbacks[event_type] = []
            
            if callback not in self.event_callbacks[event_type]:
                self.event_callbacks[event_type].append(callback)
    
    def unregister_event_callback(self, event_type, callback):
        """Unregister a callback for a specific event type"""
        with self.lock:
            if event_type in self.event_callbacks and callback in self.event_callbacks[event_type]:
                self.event_callbacks[event_type].remove(callback)
    
    def connect_device(self, device_id, device_type, **kwargs):
        """Connect to a device"""
        try:
            with self.lock:
                if device_id in self.connected_devices:
                    return {'status': 'error', 'message': f'Device {device_id} is already connected'}
                
                if device_type == 'serial':
                    port = kwargs.get('port')
                    baudrate = kwargs.get('baudrate', 9600)
                    timeout = kwargs.get('timeout', 1)
                    
                    if not port:
                        return {'status': 'error', 'message': 'Port is required for serial device'}
                    
                    # Check if port exists
                    if port not in [p.device for p in serial.tools.list_ports.comports()]:
                        return {'status': 'error', 'message': f'Port {port} not found'}
                    
                    # Connect to serial port
                    ser = serial.Serial(
                        port=port,
                        baudrate=baudrate,
                        timeout=timeout
                    )
                    
                    if ser.is_open:
                        self.connected_devices[device_id] = ser
                        
                        # Initialize sensor readings buffer
                        self.sensor_readings[device_id] = []
                        
                        logger.info(f"Connected to serial device {device_id} on port {port}")
                        
                        # Create and start data receiver thread
                        self._start_data_receiver(device_id, ser)
                        
                        return {'status': 'success', 'message': f'Connected to {device_id}', 'device_id': device_id}
                    else:
                        return {'status': 'error', 'message': f'Failed to open port {port}'}
                
                return {'status': 'error', 'message': f'Unknown device type: {device_type}'}
        except Exception as e:
            logger.error(f"Error connecting to device {device_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _start_data_receiver(self, device_id, serial_device):
        """Start a dedicated thread to receive data from a serial device"""
        def data_receiver_thread():
            while self.is_running and device_id in self.connected_devices:
                try:
                    if serial_device.is_open and serial_device.in_waiting:
                        data = serial_device.readline().decode('utf-8', errors='ignore').strip()
                        if data:
                            self._process_device_data(device_id, data)
                    time.sleep(0.01)  # Small sleep to prevent CPU overuse
                except Exception as e:
                    logger.error(f"Error in data receiver thread for {device_id}: {str(e)}")
                    break
        
        # Create thread
        thread = threading.Thread(target=data_receiver_thread)
        thread.daemon = True
        thread.start()
        
        with self.lock:
            self.data_receivers[device_id] = thread
    
    def disconnect_device(self, device_id):
        """Disconnect from a device"""
        try:
            with self.lock:
                if device_id not in self.connected_devices:
                    return {'status': 'error', 'message': f'Device {device_id} not found'}
                
                device = self.connected_devices[device_id]
                
                if isinstance(device, serial.Serial) and device.is_open:
                    device.close()
                
                # Remove device from connected devices
                del self.connected_devices[device_id]
                
                # Remove sensor readings for this device
                if device_id in self.sensor_readings:
                    del self.sensor_readings[device_id]
                
                # Remove data receiver entry (thread will exit on its own)
                if device_id in self.data_receivers:
                    del self.data_receivers[device_id]
                
                logger.info(f"Disconnected from device {device_id}")
                return {'status': 'success', 'message': f'Disconnected from {device_id}'}
        except Exception as e:
            logger.error(f"Error disconnecting from device {device_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def send_command(self, device_id, command, **kwargs):
        """Send a command to a device with timeout support"""
        try:
            with self.lock:
                if device_id not in self.connected_devices:
                    return {'status': 'error', 'message': f'Device {device_id} not connected'}
                
                device = self.connected_devices[device_id]
                
                if isinstance(device, serial.Serial):
                    # Send command to serial device
                    if not device.is_open:
                        return {'status': 'error', 'message': f'Device {device_id} port is closed'}
                    
                    # Ensure command ends with newline if needed
                    if kwargs.get('append_newline', True) and not command.endswith('\n'):
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
                                response += device.read(device.in_waiting).decode('utf-8', errors='ignore')
                                if '\n' in response and not kwargs.get('read_until_timeout', False):
                                    break
                            time.sleep(0.05)
                        
                        return {'status': 'success', 'response': response.strip()}
                    
                    return {'status': 'success', 'message': 'Command sent'}
                
                return {'status': 'error', 'message': f'Command not supported for device type'}
        except Exception as e:
            logger.error(f"Error sending command to device {device_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_device_list(self):
        """Get list of all devices"""
        try:
            with self.lock:
                devices = {
                    'serial_ports': [{
                        'port': port.device,
                        'description': port.description,
                        'hwid': port.hwid
                    } for port in serial.tools.list_ports.comports()],
                    'connected_devices': [{
                        'id': device_id,
                        'type': 'serial',
                        'port': device.port if isinstance(device, serial.Serial) else 'unknown',
                        'connected': device.is_open if isinstance(device, serial.Serial) else False
                    } for device_id, device in self.connected_devices.items()],
                    'device_status': self.device_status,
                    'camera_connections': self.camera_connections
                }
                return {'status': 'success', 'devices': devices}
        except Exception as e:
            logger.error(f"Error getting device list: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_device_status(self, device_id):
        """Get status of a specific device"""
        try:
            with self.lock:
                if device_id not in self.connected_devices:
                    return {'status': 'error', 'message': f'Device {device_id} not connected'}
                
                device = self.connected_devices[device_id]
                
                if isinstance(device, serial.Serial):
                    status = {
                        'connected': device.is_open,
                        'port': device.port,
                        'baudrate': device.baudrate,
                        'timeout': device.timeout,
                        'in_waiting': device.in_waiting if device.is_open else 0,
                        'has_data': device_id in self.sensor_readings and len(self.sensor_readings[device_id]) > 0,
                        'data_points': len(self.sensor_readings[device_id]) if device_id in self.sensor_readings else 0
                    }
                    
                    return {'status': 'success', 'device_status': status}
                
                return {'status': 'error', 'message': 'Unknown device type'}
        except Exception as e:
            logger.error(f"Error getting device status: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_all_devices_status(self):
        """Get status of all connected devices"""
        try:
            with self.lock:
                all_status = {}
                for device_id, device in self.connected_devices.items():
                    device_status = self.get_device_status(device_id)
                    if device_status.get('status') == 'success':
                        all_status[device_id] = device_status.get('device_status')
                
                return {'status': 'success', 'all_devices_status': all_status}
        except Exception as e:
            logger.error(f"Error getting all devices status: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_sensor_readings(self, device_id, sensor_type=None, limit=None):
        """Get sensor readings for a specific device and optional sensor type"""
        try:
            with self.lock:
                if device_id not in self.sensor_readings:
                    return {'status': 'error', 'message': f'No sensor readings for device {device_id}'}
                
                readings = self.sensor_readings[device_id]
                
                # Filter by sensor type if specified
                if sensor_type:
                    filtered_readings = [r for r in readings if r.get('type') == sensor_type]
                else:
                    filtered_readings = readings
                
                # Apply limit if specified
                if limit and len(filtered_readings) > limit:
                    filtered_readings = filtered_readings[-limit:]
                
                return {'status': 'success', 'readings': filtered_readings}
        except Exception as e:
            logger.error(f"Error getting sensor readings: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def connect_camera(self, camera_id, camera_index=0):
        """Connect to a camera by index (supports multiple cameras)"""
        try:
            with self.lock:
                if camera_id in self.camera_connections:
                    if self.camera_connections[camera_id].isOpened():
                        return {'status': 'success', 'message': f'Camera {camera_id} already connected', 'camera_id': camera_id}
                    else:
                        # Close and reconnect
                        self.close_camera(camera_id)
                
                # Try to connect to the camera
                cap = cv2.VideoCapture(camera_index)
                
                if cap.isOpened():
                    self.camera_connections[camera_id] = cap
                    logger.info(f"Connected to camera {camera_id} (index: {camera_index})")
                    return {'status': 'success', 'message': f'Connected to camera {camera_id}', 'camera_id': camera_id}
                else:
                    cap.release()
                    return {'status': 'error', 'message': f'Failed to connect to camera at index {camera_index}'}
        except Exception as e:
            logger.error(f"Error connecting to camera {camera_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def close_camera(self, camera_id):
        """Close a camera connection"""
        try:
            with self.lock:
                if camera_id in self.camera_connections:
                    cap = self.camera_connections[camera_id]
                    if cap.isOpened():
                        cap.release()
                    del self.camera_connections[camera_id]
                    logger.info(f"Closed camera {camera_id}")
                    return {'status': 'success', 'message': f'Camera {camera_id} closed'}
                return {'status': 'error', 'message': f'Camera {camera_id} not found'}
        except Exception as e:
            logger.error(f"Error closing camera {camera_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_camera_frame(self, camera_id):
        """Get a frame from a connected camera"""
        try:
            with self.lock:
                if camera_id not in self.camera_connections:
                    return {'status': 'error', 'message': f'Camera {camera_id} not connected'}
                
                cap = self.camera_connections[camera_id]
                if not cap.isOpened():
                    return {'status': 'error', 'message': f'Camera {camera_id} is not open'}
                
                # Read frame without holding the lock
                self.lock.release()
                try:
                    ret, frame = cap.read()
                finally:
                    self.lock.acquire()
                
                if ret:
                    # Convert frame to base64 or return as numpy array
                    return {'status': 'success', 'frame': frame}
                else:
                    return {'status': 'error', 'message': f'Failed to read frame from camera {camera_id}'}
        except Exception as e:
            logger.error(f"Error getting camera frame: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_system_sensors(self):
        """Get system sensor data"""
        try:
            with self.lock:
                return {'status': 'success', 'sensors': self.system_sensors}
        except Exception as e:
            logger.error(f"Error getting system sensors: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def test_microphone(self):
        """Test microphone availability"""
        try:
            # Try to import pyaudio to test microphone
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                
                # Check if there are any input devices
                input_devices = []
                for i in range(p.get_device_count()):
                    dev_info = p.get_device_info_by_index(i)
                    if dev_info.get('maxInputChannels', 0) > 0:
                        input_devices.append({
                            'index': i,
                            'name': dev_info.get('name'),
                            'channels': dev_info.get('maxInputChannels')
                        })
                
                p.terminate()
                
                if input_devices:
                    return {'status': 'success', 'message': f'Microphone test passed. Found {len(input_devices)} input device(s).', 'devices': input_devices}
                else:
                    return {'status': 'warning', 'message': 'Microphone test: No input devices found.'}
            except ImportError:
                return {'status': 'warning', 'message': 'Microphone test: pyaudio library not available. Cannot test microphone properly.'}
        except Exception as e:
            logger.error(f"Error testing microphone: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def test_serial_permission(self):
        """Test serial port permissions"""
        try:
            # List serial ports to test permissions
            ports = serial.tools.list_ports.comports()
            if ports:
                return {'status': 'success', 'message': f'Serial permission test passed. Found {len(ports)} serial port(s).', 'ports': [p.device for p in ports]}
            else:
                return {'status': 'warning', 'message': 'Serial permission test passed but no serial ports found.'}
        except Exception as e:
            logger.error(f"Serial permission test failed: {str(e)}")
            return {'status': 'error', 'message': f'Serial permission test failed: {str(e)}'}

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
def get_all_devices_status_api():
    """Get the status of all devices"""
    device_manager = get_device_manager()
    return jsonify(device_manager.get_all_devices_status())

@device_bp.route('/api/devices/status/<device_id>', methods=['GET'])
def get_device_status_api(device_id):
    """Get the status of a specific device"""
    device_manager = get_device_manager()
    return jsonify(device_manager.get_device_status(device_id))

@device_bp.route('/api/devices/sensors/<device_id>', methods=['GET'])
def get_sensor_readings_api(device_id):
    """Get sensor readings for a device"""
    sensor_type = request.args.get('type')
    limit = request.args.get('limit', type=int)
    
    device_manager = get_device_manager()
    result = device_manager.get_sensor_readings(device_id, sensor_type, limit)
    return jsonify(result)

@device_bp.route('/api/device/ping', methods=['GET'])
def ping_device():
    """Ping the device API to check if it's available"""
    return jsonify({
        'status': 'success',
        'message': 'Enhanced Device API is available',
        'timestamp': datetime.now().isoformat()
    })

@device_bp.route('/api/device/test_microphone', methods=['GET'])
def test_microphone_api():
    """Test microphone functionality"""
    device_manager = get_device_manager()
    return jsonify(device_manager.test_microphone())

@device_bp.route('/api/device/test_serial_permission', methods=['GET'])
def test_serial_permission_api():
    """Test serial port permission"""
    device_manager = get_device_manager()
    return jsonify(device_manager.test_serial_permission())

@device_bp.route('/api/cameras/connect', methods=['POST'])
def connect_camera_api():
    """Connect to a camera"""
    data = request.json
    camera_id = data.get('camera_id', 'default')
    camera_index = data.get('camera_index', 0)
    
    device_manager = get_device_manager()
    result = device_manager.connect_camera(camera_id, camera_index)
    return jsonify(result)

@device_bp.route('/api/cameras/disconnect/<camera_id>', methods=['POST'])
def disconnect_camera_api(camera_id):
    """Disconnect from a camera"""
    device_manager = get_device_manager()
    result = device_manager.close_camera(camera_id)
    return jsonify(result)

@device_bp.route('/api/system/sensors', methods=['GET'])
def get_system_sensors_api():
    """Get system sensor data"""
    device_manager = get_device_manager()
    return jsonify(device_manager.get_system_sensors())

@device_bp.route('/api/devices/available', methods=['GET'])
def get_available_devices_api():
    """Get all available devices for selection"""
    try:
        # Get serial ports
        serial_ports = serial.tools.list_ports.comports()
        
        devices = {
            'serial': [{
                'id': port.device,
                'name': port.description,
                'type': 'serial'
            } for port in serial_ports],
            'cameras': [],
            'system': ['cpu', 'memory', 'disk', 'temperature', 'network']
        }
        
        # Try to detect cameras
        try:
            # Simple camera detection by trying to open a few indices
            max_cameras_to_check = 4
            for i in range(max_cameras_to_check):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    devices['cameras'].append({
                        'id': f'camera_{i}',
                        'name': f'Camera {i}',
                        'type': 'camera',
                        'index': i
                    })
                    cap.release()
        except Exception as e:
            logger.debug(f"Error detecting cameras: {str(e)}")
        
        return jsonify({'status': 'success', 'available_devices': devices})
    except Exception as e:
        logger.error(f"Error getting available devices: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# Initialize device communication on module load
init_device_communication()