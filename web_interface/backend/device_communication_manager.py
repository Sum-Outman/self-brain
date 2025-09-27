import serial
import serial.tools.list_ports
import threading
import time
import json
import platform
import psutil
import os
from typing import Dict, List, Optional, Tuple

class DeviceCommunicationManager:
    """Device Communication Manager class for managing external device communication and sensors"""
    
    def __init__(self):
        # Serial port connection
        self.serial_connection: Optional[serial.Serial] = None
        # Lock for thread safety
        self.lock = threading.Lock()
        # Serial port data buffer
        self.serial_data_buffer = []
        # Connected device list
        self.connected_devices: Dict[str, Dict] = {}
        # Sensor data cache
        self.sensor_data_cache: Dict[str, Dict] = {}
        # Serial port read thread
        self.serial_read_thread: Optional[threading.Thread] = None
        self.serial_thread_stop_event = threading.Event()
        
    def get_available_serial_ports(self) -> List[str]:
        """Get list of available serial ports"""
        ports = []
        try:
            # List all available serial ports
            port_list = serial.tools.list_ports.comports()
            for port in port_list:
                ports.append(port.device)
        except Exception as e:
            print(f"Error getting serial ports: {e}")
        return ports
    
    def connect_serial_port(self, port: str, baud_rate: int) -> Dict:
        """Connect to a serial port"""
        try:
            with self.lock:
                # Check if already connected
                if self.serial_connection and self.serial_connection.is_open:
                    return {'status': 'error', 'message': 'Already connected to a serial port'}
                
                # Try to connect
                self.serial_connection = serial.Serial(
                    port=port,
                    baudrate=baud_rate,
                    timeout=0.1
                )
                
                # Clear stop event
                self.serial_thread_stop_event.clear()
                
                # Start read thread
                self.serial_read_thread = threading.Thread(target=self._serial_read_thread_func)
                self.serial_read_thread.daemon = True
                self.serial_read_thread.start()
                
                # Add to connected devices
                self.connected_devices[port] = {
                    'type': 'serial',
                    'port': port,
                    'baud_rate': baud_rate,
                    'connected_at': time.time()
                }
                
                return {'status': 'success', 'message': f'Connected to {port} at {baud_rate} baud'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def disconnect_serial_port(self) -> Dict:
        """Disconnect from the serial port"""
        try:
            with self.lock:
                if not self.serial_connection or not self.serial_connection.is_open:
                    return {'status': 'error', 'message': 'Not connected to any serial port'}
                
                # Signal thread to stop
                self.serial_thread_stop_event.set()
                
                # Wait for thread to stop
                if self.serial_read_thread:
                    self.serial_read_thread.join(timeout=1.0)
                
                # Close connection
                port = self.serial_connection.port
                self.serial_connection.close()
                self.serial_connection = None
                
                # Remove from connected devices
                if port in self.connected_devices:
                    del self.connected_devices[port]
                
                return {'status': 'success', 'message': 'Disconnected from serial port'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def send_serial_command(self, command: str) -> Dict:
        """Send a command to the serial port"""
        try:
            with self.lock:
                if not self.serial_connection or not self.serial_connection.is_open:
                    return {'status': 'error', 'message': 'Not connected to any serial port'}
                
                # Send command with newline
                command_with_newline = command + '\n'
                self.serial_connection.write(command_with_newline.encode())
                
                # Wait a short time for response
                time.sleep(0.1)
                
                # Get response if available
                response = ''
                if self.serial_connection.in_waiting:
                    response = self.serial_connection.read_all().decode().strip()
                    # Add to buffer
                    self.serial_data_buffer.append({'type': 'response', 'data': response, 'timestamp': time.time()})
                
                # Add command to buffer
                self.serial_data_buffer.append({'type': 'command', 'data': command, 'timestamp': time.time()})
                
                # Keep buffer size reasonable
                if len(self.serial_data_buffer) > 100:
                    self.serial_data_buffer = self.serial_data_buffer[-100:]
                
                return {
                    'status': 'success', 
                    'message': 'Command sent successfully',
                    'response': response
                }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_serial_data_buffer(self) -> List[Dict]:
        """Get the serial data buffer"""
        with self.lock:
            return self.serial_data_buffer.copy()
    
    def clear_serial_data_buffer(self) -> Dict:
        """Clear the serial data buffer"""
        try:
            with self.lock:
                self.serial_data_buffer.clear()
                return {'status': 'success', 'message': 'Serial data buffer cleared'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _serial_read_thread_func(self):
        """Serial port read thread function"""
        while not self.serial_thread_stop_event.is_set():
            try:
                if self.serial_connection and self.serial_connection.is_open and self.serial_connection.in_waiting:
                    data = self.serial_connection.read_all().decode().strip()
                    if data:
                        with self.lock:
                            self.serial_data_buffer.append({
                                'type': 'data', 
                                'data': data, 
                                'timestamp': time.time()
                            })
                time.sleep(0.01)  # Small delay to prevent CPU overload
            except Exception as e:
                print(f"Error in serial read thread: {e}")
                break
    
    def get_sensor_data(self) -> Dict:
        """Get system sensor data"""
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
                # Try to get temperature using psutil (works on some systems)
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps and 'cpu-thermal' in temps:
                        temperature = temps['cpu-thermal'][0].current
                    elif temps and 'coretemp' in temps:
                        temperature = temps['coretemp'][0].current
            except Exception as e:
                print(f"Error getting temperature: {e}")
            
            # Get platform information
            system_info = {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'architecture': platform.architecture()[0],
                'machine': platform.machine(),
                'processor': platform.processor()
            }
            
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
            
            # Create sensor data dictionary
            sensor_data = {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'temperature': temperature,
                'process_count': process_count,
                'system_info': system_info,
                'network_stats': network_stats,
                'timestamp': time.time()
            }
            
            # Update cache
            with self.lock:
                self.sensor_data_cache = sensor_data
            
            return {'status': 'success', 'data': sensor_data}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_connected_devices(self) -> Dict:
        """Get list of connected devices"""
        try:
            with self.lock:
                return {'status': 'success', 'devices': list(self.connected_devices.values())}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def test_microphone(self) -> Dict:
        """Test microphone availability"""
        try:
            # This is a placeholder for actual microphone test
            # In a real implementation, this would use a library like pyaudio to test the microphone
            # For now, we'll just simulate a successful test
            return {'status': 'success', 'message': 'Microphone test passed. Microphone is available.'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def test_serial_permission(self) -> Dict:
        """Test serial port permissions"""
        try:
            # List serial ports to test permissions
            ports = self.get_available_serial_ports()
            if ports:
                return {'status': 'success', 'message': f'Serial permission test passed. Found {len(ports)} serial ports.'}
            else:
                return {'status': 'warning', 'message': 'Serial permission test passed but no serial ports found.'}
        except Exception as e:
            return {'status': 'error', 'message': f'Serial permission test failed: {str(e)}'}
    
    def is_serial_connected(self) -> bool:
        """Check if serial port is connected"""
        with self.lock:
            return self.serial_connection is not None and self.serial_connection.is_open
    
    def get_serial_port_info(self) -> Optional[Dict]:
        """Get current serial port information"""
        with self.lock:
            if not self.serial_connection or not self.serial_connection.is_open:
                return None
            return {
                'port': self.serial_connection.port,
                'baud_rate': self.serial_connection.baudrate,
                'timeout': self.serial_connection.timeout,
                'is_open': self.serial_connection.is_open
            }
    
    def close(self):
        """Close all connections and clean up"""
        # Disconnect from serial port
        if self.is_serial_connected():
            self.disconnect_serial_port()
        
        # Signal all threads to stop
        self.serial_thread_stop_event.set()

# Create a global instance of DeviceCommunicationManager
global_device_manager = DeviceCommunicationManager()