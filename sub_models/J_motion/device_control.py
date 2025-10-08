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

import serial
import paho.mqtt.client as mqtt
from pymodbus.client import ModbusTcpClient
import time
import threading

class MotionController:
    def __init__(self):
        self.protocols = {
            'RS485': self._control_rs485,
            'MQTT': self._control_mqtt,
            'ModbusTCP': self._control_modbus_tcp
        }
        # Sensor data cache
        self.sensor_data = {}
        # Spatial positioning data cache
        self.spatial_data = {}
        # Device status monitoring
        self.device_status = {}
        
        # Start data listener thread
        self.data_listener_thread = threading.Thread(target=self._data_listener)
        self.data_listener_thread.daemon = True
        self.data_listener_thread.start()
        
        # Modbus client
        self.modbus_client = None
    
    def control_device(self, protocol, command):
        if protocol in self.protocols:
            return self.protocols[protocol](command)
        return {"error": "Unsupported protocol"}
    
    def _control_rs485(self, command):
        # RS485 device control implementation
        try:
            ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
            ser.write(command.encode())
            response = ser.readline().decode()
            ser.close()
            return {"status": "success", "response": response}
        except Exception as e:
            return {"error": str(e)}
    
    def _control_mqtt(self, command):
        # MQTT device control implementation
        client = mqtt.Client()
        client.connect("iot.eclipse.org", 1883, 60)
        client.publish("motion/control", command)
        return {"status": "command_sent"}
    
    def _control_modbus_tcp(self, command):
        """Modbus TCP设备控制实现"""
        try:
            if not self.modbus_client or not self.modbus_client.is_socket_open():
                self.modbus_client = ModbusTcpClient('127.0.0.1', port=502)
                self.modbus_client.connect()
            
            # Parse command (example: "write_register:40001=100")
            parts = command.split(':')
            if len(parts) != 2:
                return {"error": "Invalid Modbus command format"}
                
            action, params = parts[0], parts[1]
            
            if action == 'write_register':
                addr, value = params.split('=')
                result = self.modbus_client.write_register(int(addr), int(value))
                return {"status": "success", "response": str(result)}
                
            elif action == 'read_register':
                result = self.modbus_client.read_holding_registers(int(params), 1)
                return {"status": "success", "value": result.registers[0]}
                
            else:
                return {"error": "Unsupported Modbus action"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def update_sensor_data(self, sensor_type, data):
        """Update sensor data"""
        self.sensor_data[sensor_type] = {
            'value': data,
            'timestamp': time.time()
        }
    
    def update_spatial_data(self, position, velocity):
        """Update spatial positioning data"""
        self.spatial_data = {
            'position': position,
            'velocity': velocity,
            'timestamp': time.time()
        }
    
    def _data_listener(self):
        """Data listener thread for real-time response to sensor and spatial data changes"""
        while True:
            # Check sensor data and adjust control strategy
            if 'pressure' in self.sensor_data and self.sensor_data['pressure']['value'] > 100:
                self._adjust_pressure_control()
                
            # Check spatial data and adjust motion trajectory
            if 'velocity' in self.spatial_data and self.spatial_data['velocity'] > 0.5:
                self._adjust_motion_trajectory()
                
            time.sleep(0.1)  # 100ms polling interval
    
    def _adjust_pressure_control(self):
        """Adjust control based on pressure sensor data"""
        pressure = self.sensor_data['pressure']['value']
        # Implement pressure control logic
        print(f"Adjusting pressure control: current pressure={pressure}")
        
    def _adjust_motion_trajectory(self):
        """Adjust motion trajectory based on spatial velocity data"""
        velocity = self.spatial_data['velocity']
        # Implement motion trajectory adjustment logic
        print(f"Adjusting motion trajectory: current velocity={velocity}")
    
    def get_device_status(self, device_id):
        """Get device status"""
        return self.device_status.get(device_id, "unknown")
    
    def set_device_status(self, device_id, status):
        """Set device status"""
        self.device_status[device_id] = status
        # Publish status update
        self._publish_device_status(device_id, status)
    
    def _publish_device_status(self, device_id, status):
        """Publish device status to MQTT"""
        client = mqtt.Client()
        client.connect("localhost", 1883, 60)
        client.publish(f"motion/device/{device_id}/status", status)
        client.disconnect()
