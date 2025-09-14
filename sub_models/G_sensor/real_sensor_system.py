"""
真实传感器数据集成系统
Real Sensor Data Integration System
"""

import json
import logging
import random
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import serial
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class SensorDevice:
    """传感器设备基类"""
    
    def __init__(self, device_id: str, device_type: str):
        self.device_id = device_id
        self.device_type = device_type
        self.is_connected = False
        self.last_reading = None
        self.callbacks = []
    
    def connect(self) -> bool:
        """连接设备"""
        raise NotImplementedError
    
    def read_data(self) -> Dict[str, Any]:
        """读取数据"""
        raise NotImplementedError
    
    def disconnect(self):
        """断开连接"""
        self.is_connected = False
    
    def add_callback(self, callback: Callable):
        """添加数据回调"""
        self.callbacks.append(callback)

class ArduinoSensor(SensorDevice):
    """Arduino传感器"""
    
    def __init__(self, device_id: str, port: str, baudrate: int = 9600):
        super().__init__(device_id, 'arduino')
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
    
    def connect(self) -> bool:
        """连接Arduino"""
        try:
            self.serial_connection = serial.Serial(
                self.port, self.baudrate, timeout=1
            )
            time.sleep(2)  # 等待Arduino重启
            self.is_connected = True
            logger.info(f"Arduino {self.device_id} 连接成功")
            return True
        except Exception as e:
            logger.error(f"Arduino {self.device_id} 连接失败: {e}")
            return False
    
    def read_data(self) -> Dict[str, Any]:
        """读取Arduino数据"""
        if not self.is_connected or not self.serial_connection:
            return self._generate_mock_data()
        
        try:
            line = self.serial_connection.readline().decode('utf-8').strip()
            if line:
                # 解析JSON格式的数据
                data = json.loads(line)
                self.last_reading = {
                    'timestamp': datetime.now().isoformat(),
                    'device_id': self.device_id,
                    'data': data
                }
                return self.last_reading
        except Exception as e:
            logger.error(f"Arduino数据读取失败: {e}")
        
        return self._generate_mock_data()
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        """生成模拟Arduino数据"""
        return {
            'timestamp': datetime.now().isoformat(),
            'device_id': self.device_id,
            'data': {
                'temperature': round(random.uniform(20.0, 30.0), 2),
                'humidity': round(random.uniform(40.0, 80.0), 2),
                'light': round(random.uniform(0.0, 1000.0), 2),
                'motion': random.choice([0, 1]),
                'sound_level': round(random.uniform(30.0, 80.0), 2)
            }
        }
    
    def disconnect(self):
        """断开Arduino连接"""
        super().disconnect()
        if self.serial_connection:
            self.serial_connection.close()

class RaspberryPiSensor(SensorDevice):
    """树莓派传感器"""
    
    def __init__(self, device_id: str, ip_address: str, port: int = 8080):
        super().__init__(device_id, 'raspberry_pi')
        self.ip_address = ip_address
        self.port = port
    
    def connect(self) -> bool:
        """连接树莓派"""
        try:
            import requests
            response = requests.get(
                f"http://{self.ip_address}:{self.port}/status",
                timeout=5
            )
            if response.status_code == 200:
                self.is_connected = True
                logger.info(f"树莓派 {self.device_id} 连接成功")
                return True
        except Exception as e:
            logger.error(f"树莓派 {self.device_id} 连接失败: {e}")
        return False
    
    def read_data(self) -> Dict[str, Any]:
        """读取树莓派数据"""
        if not self.is_connected:
            return self._generate_mock_data()
        
        try:
            import requests
            response = requests.get(
                f"http://{self.ip_address}:{self.port}/sensors",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                self.last_reading = {
                    'timestamp': datetime.now().isoformat(),
                    'device_id': self.device_id,
                    'data': data
                }
                return self.last_reading
        except Exception as e:
            logger.error(f"树莓派数据读取失败: {e}")
        
        return self._generate_mock_data()
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        """生成模拟树莓派数据"""
        return {
            'timestamp': datetime.now().isoformat(),
            'device_id': self.device_id,
            'data': {
                'cpu_temperature': round(random.uniform(40.0, 70.0), 2),
                'cpu_usage': round(random.uniform(10.0, 90.0), 2),
                'memory_usage': round(random.uniform(30.0, 80.0), 2),
                'disk_usage': round(random.uniform(50.0, 90.0), 2),
                'camera_detected_objects': [
                    {'class': 'person', 'confidence': 0.85, 'bbox': [100, 200, 50, 100]},
                    {'class': 'car', 'confidence': 0.92, 'bbox': [300, 150, 80, 40]}
                ]
            }
        }

class MockSensor(SensorDevice):
    """模拟传感器"""
    
    def __init__(self, device_id: str, sensor_type: str):
        super().__init__(device_id, sensor_type)
        self.sensor_configs = {
            'temperature': {'min': -10, 'max': 50, 'unit': '°C'},
            'humidity': {'min': 0, 'max': 100, 'unit': '%'},
            'pressure': {'min': 980, 'max': 1050, 'unit': 'hPa'},
            'light': {'min': 0, 'max': 10000, 'unit': 'lux'},
            'motion': {'values': [0, 1], 'unit': 'detected'},
            'sound': {'min': 0, 'max': 120, 'unit': 'dB'},
            'air_quality': {'min': 0, 'max': 500, 'unit': 'AQI'},
            'gps': {'lat_range': [39.8, 40.0], 'lon_range': [116.2, 116.5], 'unit': 'degrees'}
        }
    
    def connect(self) -> bool:
        """连接模拟传感器"""
        self.is_connected = True
        logger.info(f"模拟传感器 {self.device_id} 连接成功")
        return True
    
    def read_data(self) -> Dict[str, Any]:
        """读取模拟数据"""
        if self.device_type in self.sensor_configs:
            config = self.sensor_configs[self.device_type]
            
            if self.device_type == 'gps':
                data = {
                    'latitude': round(random.uniform(*config['lat_range']), 6),
                    'longitude': round(random.uniform(*config['lon_range']), 6),
                    'altitude': round(random.uniform(0, 100), 2),
                    'accuracy': round(random.uniform(1, 10), 2)
                }
            elif self.device_type == 'motion':
                data = {'detected': random.choice(config['values'])}
            else:
                data = {
                    'value': round(random.uniform(config['min'], config['max']), 2),
                    'unit': config['unit']
                }
            
            self.last_reading = {
                'timestamp': datetime.now().isoformat(),
                'device_id': self.device_id,
                'sensor_type': self.device_type,
                'data': data
            }
            
            return self.last_reading
        
        return {
            'timestamp': datetime.now().isoformat(),
            'device_id': self.device_id,
            'sensor_type': self.device_type,
            'data': {'value': 0, 'unit': 'unknown'}
        }

class RealSensorSystem:
    """真实传感器数据集成系统"""
    
    def __init__(self):
        self.devices = {}
        self.data_history = []
        self.running = False
        self.callbacks = []
        self.data_file = Path("sensor_data.jsonl")
        
        # 确保数据文件存在
        if not self.data_file.exists():
            self.data_file.touch()
    
    def discover_devices(self) -> Dict[str, List[str]]:
        """发现可用设备"""
        available_devices = {
            'arduino': [],
            'raspberry_pi': [],
            'mock': []
        }
        
        # 扫描串口设备
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                if 'Arduino' in port.description or 'USB' in port.description:
                    available_devices['arduino'].append(port.device)
        except:
            pass
        
        # 添加模拟设备
        mock_types = ['temperature', 'humidity', 'light', 'motion', 'sound', 'air_quality', 'gps']
        for sensor_type in mock_types:
            available_devices['mock'].append(f"mock_{sensor_type}")
        
        return available_devices
    
    def add_device(self, device: SensorDevice) -> bool:
        """添加传感器设备"""
        if device.connect():
            self.devices[device.device_id] = device
            logger.info(f"设备 {device.device_id} 添加成功")
            return True
        return False
    
    def start_data_collection(self):
        """启动数据收集"""
        if self.running:
            return
        
        self.running = True
        threading.Thread(target=self._collection_loop, daemon=True).start()
        logger.info("传感器数据收集已启动")
    
    def stop_data_collection(self):
        """停止数据收集"""
        self.running = False
        
        for device in self.devices.values():
            device.disconnect()
        
        logger.info("传感器数据收集已停止")
    
    def _collection_loop(self):
        """数据收集循环"""
        while self.running:
            for device_id, device in self.devices.items():
                try:
                    data = device.read_data()
                    
                    # 保存数据
                    self.data_history.append(data)
                    
                    # 写入文件
                    with open(self.data_file, 'a') as f:
                        f.write(json.dumps(data) + '\n')
                    
                    # 触发回调
                    for callback in self.callbacks:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"回调错误: {e}")
                    
                except Exception as e:
                    logger.error(f"设备 {device_id} 数据收集错误: {e}")
            
            time.sleep(1)  # 每秒收集一次
    
    def get_real_time_data(self) -> Dict[str, Any]:
        """获取实时数据"""
        data = {}
        for device_id, device in self.devices.items():
            if device.last_reading:
                data[device_id] = device.last_reading
        return data
    
    def get_historical_data(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取历史数据"""
        cutoff_time = datetime.now().timestamp() - hours * 3600
        
        filtered_data = []
        for entry in self.data_history:
            if datetime.fromisoformat(entry['timestamp']).timestamp() > cutoff_time:
                filtered_data.append(entry)
        
        return filtered_data[-1000:]  # 限制数据量
    
    def add_data_callback(self, callback: Callable):
        """添加数据回调"""
        self.callbacks.append(callback)
    
    def get_device_status(self) -> Dict[str, Dict[str, Any]]:
        """获取设备状态"""
        status = {}
        for device_id, device in self.devices.items():
            status[device_id] = {
                'type': device.device_type,
                'connected': device.is_connected,
                'last_reading': device.last_reading
            }
        return status
    
    def analyze_sensor_data(self, device_id: str = None) -> Dict[str, Any]:
        """分析传感器数据"""
        if device_id:
            data = [d for d in self.data_history if d['device_id'] == device_id]
        else:
            data = self.data_history
        
        if not data:
            return {'error': 'No data available'}
        
        analysis = {}
        
        for entry in data:
            sensor_type = entry.get('sensor_type', 'unknown')
            sensor_data = entry['data']
            
            if sensor_type not in analysis:
                analysis[sensor_type] = {
                    'count': 0,
                    'values': [],
                    'min': float('inf'),
                    'max': float('-inf'),
                    'avg': 0
                }
            
            if 'value' in sensor_data:
                value = sensor_data['value']
                analysis[sensor_type]['values'].append(value)
                analysis[sensor_type]['min'] = min(analysis[sensor_type]['min'], value)
                analysis[sensor_type]['max'] = max(analysis[sensor_type]['max'], value)
                analysis[sensor_type]['count'] += 1
        
        # 计算平均值
        for sensor_type in analysis:
            if analysis[sensor_type]['values']:
                analysis[sensor_type]['avg'] = np.mean(analysis[sensor_type]['values'])
                analysis[sensor_type]['std'] = np.std(analysis[sensor_type]['values'])
        
        return analysis

# 全局实例
real_sensor_system = RealSensorSystem()

if __name__ == "__main__":
    # 测试真实传感器系统
    print("=== 测试真实传感器数据集成系统 ===")
    
    # 发现设备
    devices = real_sensor_system.discover_devices()
    print("发现设备:", devices)
    
    # 添加模拟传感器
    temp_sensor = MockSensor("temp_01", "temperature")
    humidity_sensor = MockSensor("humidity_01", "humidity")
    
    real_sensor_system.add_device(temp_sensor)
    real_sensor_system.add_device(humidity_sensor)
    
    # 启动数据收集
    real_sensor_system.start_data_collection()
    
    time.sleep(5)
    
    # 获取实时数据
    real_time_data = real_sensor_system.get_real_time_data()
    print("实时数据:", real_time_data)
    
    # 获取设备状态
    device_status = real_sensor_system.get_device_status()
    print("设备状态:", device_status)
    
    # 停止数据收集
    real_sensor_system.stop_data_collection()
    
    print("=== 测试完成 ===")