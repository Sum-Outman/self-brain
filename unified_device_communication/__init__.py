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
        self.camera_manager = None
        self.serial_ports = {}
        self.is_running = False
        self.monitor_thread = None
        self.device_status = {}
        
    def set_camera_manager(self, camera_manager):
        """Set the camera manager instance"""
        self.camera_manager = camera_manager
        
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
        """Update the status of all connected devices"""
        try:
            # Update camera status
            if self.camera_manager:
                camera_status = self.camera_manager.get_camera_status()
                self.device_status['cameras'] = camera_status
                
            # Update serial device status
            serial_status = {}
            for device_id, device in self.connected_devices.items():
                if isinstance(device, serial.Serial):
                    serial_status[device_id] = {
                        'connected': device.is_open,
                        'port': device.port,
                        'baudrate': device.baudrate
                    }
            self.device_status['serial'] = serial_status
            
        except Exception as e:
            logger.error(f"Error updating device status: {str(e)}")
    
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

# Initialize device communication on module load
init_device_communication()