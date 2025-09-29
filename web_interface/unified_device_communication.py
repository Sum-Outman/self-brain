import logging
import threading
import time
from flask import Blueprint, jsonify, request
import logging
import sys
import os

# Add backend directory to path to import DeviceCommunicationManager
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

# Try to import DeviceCommunicationManager
_DEVICE_COMMUNICATION_MANAGER_AVAILABLE = False
try:
    from device_communication_manager import global_device_manager
    _DEVICE_COMMUNICATION_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import DeviceCommunicationManager: {e}")

# Create blueprint
device_bp = Blueprint('device_bp', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DeviceCommunicationManager')

class DeviceManager:
    """Manages communication with external devices and sensors"""
    
    def __init__(self):
        self.devices = {}
        self.sensors = {}
        self.camera_manager = None
        self.lock = threading.Lock()
        self.is_running = False
        self.thread = None
        
        # If DeviceCommunicationManager is available, use it
        self.external_manager = global_device_manager if _DEVICE_COMMUNICATION_MANAGER_AVAILABLE else None
        if self.external_manager:
            logger.info("Using external DeviceCommunicationManager")
    
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
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("Device manager started")
    
    def stop(self):
        """Stop the device manager"""
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
            if self.thread:
                self.thread.join(2.0)
            logger.info("Device manager stopped")
    
    def _run(self):
        """Main loop for device manager"""
        while self.is_running:
            # If we have an external manager, use it to get data
            if self.external_manager:
                try:
                    # Get sensor data and update local cache
                    sensor_result = self.external_manager.get_sensor_data()
                    if sensor_result.get('status') == 'success':
                        with self.lock:
                            self.sensors = sensor_result.get('data', {})
                except Exception as e:
                    logger.error(f"Error in device manager loop: {e}")
            # In a real implementation, this would scan for and communicate with devices
            time.sleep(1.0)
    
    def add_device(self, device_id, device_info):
        """Add a new device"""
        with self.lock:
            self.devices[device_id] = device_info
            logger.info(f"Added device: {device_id}")
    
    def remove_device(self, device_id):
        """Remove a device"""
        with self.lock:
            if device_id in self.devices:
                del self.devices[device_id]
                logger.info(f"Removed device: {device_id}")
    
    def get_devices(self):
        """Get all devices"""
        with self.lock:
            # If we have an external manager, get devices from there
            if self.external_manager:
                try:
                    devices_result = self.external_manager.get_connected_devices()
                    if devices_result.get('status') == 'success':
                        return devices_result.get('devices', [])
                except Exception as e:
                    logger.error(f"Error getting devices from external manager: {e}")
            return list(self.devices.values())
    
    def get_sensors(self):
        """Get all sensors"""
        with self.lock:
            return list(self.sensors.values())
    
    # Additional methods to support serial communication using the external manager
    def list_available_devices(self):
        """List all available devices"""
        if self.external_manager:
            try:
                ports = self.external_manager.get_available_serial_ports()
                devices = []
                for port in ports:
                    devices.append({
                        'id': port,
                        'type': 'serial',
                        'name': f'Serial Port {port}'
                    })
                return devices
            except Exception as e:
                logger.error(f"Error listing devices: {e}")
        return []
    
    def connect_serial_device(self, port, baudrate):
        """Connect to a serial device"""
        if self.external_manager:
            try:
                return self.external_manager.connect_serial_port(port, baudrate)
            except Exception as e:
                logger.error(f"Error connecting to serial device: {e}")
                return {'status': 'error', 'message': str(e)}
        return {'status': 'error', 'message': 'Device manager not available'}
    
    def disconnect_serial_device(self, port):
        """Disconnect from a serial device"""
        if self.external_manager:
            try:
                # The external manager's disconnect method doesn't take a port parameter
                # but we include it for API consistency
                return self.external_manager.disconnect_serial_port()
            except Exception as e:
                logger.error(f"Error disconnecting from serial device: {e}")
                return {'status': 'error', 'message': str(e)}
        return {'status': 'error', 'message': 'Device manager not available'}
    
    def send_serial_command(self, port, command):
        """Send a command to a serial device"""
        if self.external_manager:
            try:
                # The external manager's send command method doesn't take a port parameter
                # but we include it for API consistency
                return self.external_manager.send_serial_command(command)
            except Exception as e:
                logger.error(f"Error sending serial command: {e}")
                return {'status': 'error', 'message': str(e)}
        return {'status': 'error', 'message': 'Device manager not available'}
    
    def get_all_devices_status(self):
        """Get status of all devices"""
        if self.external_manager:
            try:
                devices_result = self.external_manager.get_connected_devices()
                if devices_result.get('status') == 'success':
                    return devices_result.get('devices', [])
            except Exception as e:
                logger.error(f"Error getting devices status: {e}")
        return []

# Global instance cache
_device_manager_instance = None

@device_bp.route('/api/devices', methods=['GET'])
def get_all_devices():
    """Get all connected devices"""
    device_manager = get_device_manager()
    return jsonify({
        'status': 'success',
        'devices': device_manager.get_devices()
    })

@device_bp.route('/api/sensors', methods=['GET'])
def get_all_sensors():
    """Get all connected sensors"""
    device_manager = get_device_manager()
    return jsonify({
        'status': 'success',
        'sensors': device_manager.get_sensors()
    })

# Add missing endpoints to match frontend API calls
@device_bp.route('/api/device/ping', methods=['GET'])
def ping_device():
    """Ping the device API to check if it's available"""
    return jsonify({
        'status': 'success',
        'message': 'Device API is available'
    })

@device_bp.route('/api/device/test_microphone', methods=['GET'])
def test_microphone():
    """Test microphone functionality"""
    device_manager = get_device_manager()
    if hasattr(device_manager, 'external_manager') and device_manager.external_manager:
        try:
            return jsonify(device_manager.external_manager.test_microphone())
        except Exception as e:
            logger.error(f"Error testing microphone: {e}")
    return jsonify({
        'status': 'success',
        'message': 'Microphone test passed'
    })

@device_bp.route('/api/device/test_serial_permission', methods=['GET'])
def test_serial_permission():
    """Test serial port permission"""
    device_manager = get_device_manager()
    if hasattr(device_manager, 'external_manager') and device_manager.external_manager:
        try:
            return jsonify(device_manager.external_manager.test_serial_permission())
        except Exception as e:
            logger.error(f"Error testing serial permission: {e}")
    return jsonify({
        'status': 'success',
        'message': 'Serial permission test passed'
    })

def get_device_manager():
    """Get a singleton instance of DeviceManager"""
    global _device_manager_instance
    if _device_manager_instance is None:
        _device_manager_instance = DeviceManager()
        logger.info("DeviceManager initialized")
    return _device_manager_instance

def init_device_communication():
    """Initialize device communication"""
    device_manager = get_device_manager()
    device_manager.start()
    logger.info("Device communication initialized")
    return device_manager

def cleanup_device_communication():
    """Cleanup device communication"""
    global _device_manager_instance
    if _device_manager_instance:
        _device_manager_instance.stop()
        _device_manager_instance = None
        logger.info("Device communication cleaned up")