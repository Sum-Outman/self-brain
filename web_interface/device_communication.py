import sys
import os

# Add the backend directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and re-export the global_device_manager from backend.device_communication_manager
from backend.device_communication_manager import global_device_manager

# Add alias for app.py compatibility
device_manager = global_device_manager

# For backward compatibility
list_available_serial_ports = global_device_manager.get_available_serial_ports
connect_serial_device = global_device_manager.connect_serial_port
disconnect_serial_device = global_device_manager.disconnect_serial_port
send_serial_command = global_device_manager.send_serial_command
list_available_devices = global_device_manager.get_connected_devices
get_all_devices_status = global_device_manager.get_connected_devices
get_sensor_data = global_device_manager.get_sensor_data

__all__ = [
    'global_device_manager',
    'device_manager',
    'list_available_serial_ports',
    'connect_serial_device',
    'disconnect_serial_device',
    'send_serial_command',
    'list_available_devices',
    'get_all_devices_status',
    'get_sensor_data'
]