import logging
import threading
import time
from flask import Blueprint, jsonify, request

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
            return list(self.devices.values())
    
    def get_sensors(self):
        """Get all sensors"""
        with self.lock:
            return list(self.sensors.values())

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