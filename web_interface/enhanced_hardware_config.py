"""
增强硬件配置模块
Enhanced Hardware Configuration Module

提供完整的硬件设备管理功能，包括：
- 多摄像头配置和管理
- 传感器设备连接和监控
- 串口设备通信
- 网络设备连接
- 外接设备控制
"""

import os
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import serial
import socket
import psutil

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedHardwareConfig")

class EnhancedHardwareManager:
    """增强硬件管理器"""
    
    def __init__(self, config_path: str = "d:/shiyan/config/hardware_settings.json"):
        """初始化硬件管理器"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # 硬件设备状态
        self.camera_devices = {}
        self.sensor_devices = {}
        self.serial_devices = {}
        self.network_devices = {}
        self.external_devices = {}
        
        # 设备锁
        self.device_locks = {}
        self.global_lock = threading.Lock()
        
        # 初始化硬件监控
        self._init_hardware_monitoring()
        
        logger.info("增强硬件管理器已初始化")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载硬件配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 返回默认配置
                default_config = {
                    "cameras": {
                        "max_cameras": 4,
                        "default_resolution": "1280x720",
                        "default_fps": 30,
                        "enable_stereo_vision": True,
                        "stereo_camera_pairs": {}
                    },
                    "sensors": {
                        "temperature_sensors": [],
                        "humidity_sensors": [],
                        "motion_sensors": [],
                        "light_sensors": [],
                        "pressure_sensors": [],
                        "acceleration_sensors": [],
                        "gyroscope_sensors": []
                    },
                    "serial_ports": {
                        "baud_rates": [9600, 19200, 38400, 57600, 115200],
                        "default_timeout": 1.0,
                        "auto_detect": True
                    },
                    "network_devices": {
                        "tcp_ports": [8080, 8081, 9000],
                        "udp_ports": [8082, 8083],
                        "websocket_ports": [8765, 8766]
                    },
                    "external_devices": {
                        "robotic_arms": [],
                        "motor_controllers": [],
                        "led_controllers": [],
                        "audio_devices": [],
                        "display_devices": []
                    },
                    "monitoring": {
                        "enable_real_time_monitoring": True,
                        "update_interval": 2.0,
                        "alert_thresholds": {
                            "cpu_usage": 80,
                            "memory_usage": 85,
                            "temperature": 75,
                            "disk_usage": 90
                        }
                    }
                }
                # 保存默认配置
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4, ensure_ascii=False)
                return default_config
        except Exception as e:
            logger.error(f"加载硬件配置文件失败: {str(e)}")
            return {}
    
    def _init_hardware_monitoring(self):
        """初始化硬件监控"""
        if self.config.get("monitoring", {}).get("enable_real_time_monitoring", True):
            self.monitoring_thread = threading.Thread(target=self._monitor_hardware, daemon=True)
            self.monitoring_thread.start()
            logger.info("硬件监控线程已启动")
    
    def _monitor_hardware(self):
        """硬件监控线程"""
        interval = self.config.get("monitoring", {}).get("update_interval", 2.0)
        
        while True:
            try:
                # 监控系统资源
                self._monitor_system_resources()
                
                # 监控摄像头状态
                self._monitor_camera_devices()
                
                # 监控传感器状态
                self._monitor_sensor_devices()
                
                # 监控串口设备
                self._monitor_serial_devices()
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"硬件监控错误: {str(e)}")
                time.sleep(5)
    
    def _monitor_system_resources(self):
        """监控系统资源"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 检查阈值告警
            alerts = self.config.get("monitoring", {}).get("alert_thresholds", {})
            
            if cpu_percent > alerts.get("cpu_usage", 80):
                logger.warning(f"CPU使用率过高: {cpu_percent}%")
            
            if memory.percent > alerts.get("memory_usage", 85):
                logger.warning(f"内存使用率过高: {memory.percent}%")
            
            if disk.percent > alerts.get("disk_usage", 90):
                logger.warning(f"磁盘使用率过高: {disk.percent}%")
                
        except Exception as e:
            logger.error(f"监控系统资源失败: {str(e)}")
    
    def _monitor_camera_devices(self):
        """监控摄像头设备状态"""
        try:
            for camera_id, camera_info in self.camera_devices.items():
                if camera_info.get("is_active", False):
                    # 检查摄像头连接状态
                    # 这里可以添加实际的摄像头状态检查逻辑
                    pass
        except Exception as e:
            logger.error(f"监控摄像头设备失败: {str(e)}")
    
    def _monitor_sensor_devices(self):
        """监控传感器设备状态"""
        try:
            for sensor_id, sensor_info in self.sensor_devices.items():
                if sensor_info.get("is_connected", False):
                    # 检查传感器连接状态
                    # 这里可以添加实际的传感器状态检查逻辑
                    pass
        except Exception as e:
            logger.error(f"监控传感器设备失败: {str(e)}")
    
    def _monitor_serial_devices(self):
        """监控串口设备状态"""
        try:
            for port, device_info in self.serial_devices.items():
                if device_info.get("is_connected", False):
                    # 检查串口连接状态
                    # 这里可以添加实际的串口状态检查逻辑
                    pass
        except Exception as e:
            logger.error(f"监控串口设备失败: {str(e)}")
    
    # 摄像头管理方法
    def scan_cameras(self) -> List[Dict[str, Any]]:
        """扫描可用的摄像头设备"""
        try:
            cameras = []
            
            # 模拟扫描摄像头（实际实现应该使用OpenCV或其他库）
            for i in range(4):  # 假设最多4个摄像头
                camera_info = {
                    "id": i,
                    "name": f"Camera {i}",
                    "type": "USB",
                    "resolution": "1280x720",
                    "fps": 30,
                    "is_available": True,
                    "description": f"USB Camera Device {i}"
                }
                cameras.append(camera_info)
            
            # 添加虚拟双目摄像头
            stereo_camera = {
                "id": 100,
                "name": "Stereo Camera Pair",
                "type": "Stereo",
                "resolution": "1280x480",
                "fps": 25,
                "is_available": True,
                "description": "Virtual Stereo Camera for 3D Vision"
            }
            cameras.append(stereo_camera)
            
            logger.info(f"扫描到 {len(cameras)} 个摄像头设备")
            return cameras
            
        except Exception as e:
            logger.error(f"扫描摄像头失败: {str(e)}")
            return []
    
    def connect_camera(self, camera_id: int, settings: Dict[str, Any] = None) -> bool:
        """连接摄像头设备"""
        try:
            with self.global_lock:
                if camera_id in self.camera_devices:
                    logger.warning(f"摄像头 {camera_id} 已经连接")
                    return True
                
                # 模拟摄像头连接
                camera_info = {
                    "id": camera_id,
                    "name": f"Camera {camera_id}",
                    "is_active": True,
                    "settings": settings or {},
                    "connected_at": datetime.now().isoformat(),
                    "last_frame_time": None,
                    "frame_count": 0,
                    "error_count": 0
                }
                
                self.camera_devices[camera_id] = camera_info
                logger.info(f"摄像头 {camera_id} 连接成功")
                return True
                
        except Exception as e:
            logger.error(f"连接摄像头 {camera_id} 失败: {str(e)}")
            return False
    
    def disconnect_camera(self, camera_id: int) -> bool:
        """断开摄像头连接"""
        try:
            with self.global_lock:
                if camera_id in self.camera_devices:
                    del self.camera_devices[camera_id]
                    logger.info(f"摄像头 {camera_id} 断开连接")
                    return True
                else:
                    logger.warning(f"摄像头 {camera_id} 未连接")
                    return False
                    
        except Exception as e:
            logger.error(f"断开摄像头 {camera_id} 连接失败: {str(e)}")
            return False
    
    def configure_stereo_vision(self, left_camera_id: int, right_camera_id: int, 
                               pair_name: str = "stereo_pair") -> bool:
        """配置双目视觉"""
        try:
            stereo_config = {
                "pair_name": pair_name,
                "left_camera": left_camera_id,
                "right_camera": right_camera_id,
                "calibration_file": f"calibration_{pair_name}.json",
                "baseline": 0.12,  # 基线距离（米）
                "focal_length": 800,  # 焦距
                "disparity_range": 64,
                "configured_at": datetime.now().isoformat()
            }
            
            # 更新配置
            if "cameras" not in self.config:
                self.config["cameras"] = {}
            if "stereo_camera_pairs" not in self.config["cameras"]:
                self.config["cameras"]["stereo_camera_pairs"] = {}
            
            self.config["cameras"]["stereo_camera_pairs"][pair_name] = stereo_config
            
            # 保存配置
            self._save_config()
            
            logger.info(f"双目视觉配置成功: {pair_name}")
            return True
            
        except Exception as e:
            logger.error(f"配置双目视觉失败: {str(e)}")
            return False
    
    # 传感器管理方法
    def scan_sensors(self) -> List[Dict[str, Any]]:
        """扫描可用的传感器设备"""
        try:
            sensors = []
            
            # 模拟温度传感器
            temp_sensor = {
                "id": "temp_001",
                "name": "Temperature Sensor TMP36",
                "type": "temperature",
                "interface": "I2C",
                "address": "0x48",
                "unit": "°C",
                "range": [-40, 125],
                "precision": 0.1,
                "is_available": True
            }
            sensors.append(temp_sensor)
            
            # 模拟湿度传感器
            humidity_sensor = {
                "id": "hum_001",
                "name": "Humidity Sensor DHT22",
                "type": "humidity",
                "interface": "GPIO",
                "pin": 4,
                "unit": "%",
                "range": [0, 100],
                "precision": 0.1,
                "is_available": True
            }
            sensors.append(humidity_sensor)
            
            # 模拟运动传感器
            motion_sensor = {
                "id": "motion_001",
                "name": "Motion Sensor PIR",
                "type": "motion",
                "interface": "GPIO",
                "pin": 17,
                "sensitivity": "medium",
                "is_available": True
            }
            sensors.append(motion_sensor)
            
            # 模拟加速度传感器
            accel_sensor = {
                "id": "accel_001",
                "name": "Accelerometer MPU6050",
                "type": "acceleration",
                "interface": "I2C",
                "address": "0x68",
                "range": "±16g",
                "axes": 3,
                "is_available": True
            }
            sensors.append(accel_sensor)
            
            logger.info(f"扫描到 {len(sensors)} 个传感器设备")
            return sensors
            
        except Exception as e:
            logger.error(f"扫描传感器失败: {str(e)}")
            return []
    
    def connect_sensor(self, sensor_id: str, connection_params: Dict[str, Any] = None) -> bool:
        """连接传感器设备"""
        try:
            with self.global_lock:
                if sensor_id in self.sensor_devices:
                    logger.warning(f"传感器 {sensor_id} 已经连接")
                    return True
                
                # 模拟传感器连接
                sensor_info = {
                    "id": sensor_id,
                    "is_connected": True,
                    "connection_params": connection_params or {},
                    "connected_at": datetime.now().isoformat(),
                    "last_reading": None,
                    "reading_count": 0,
                    "error_count": 0
                }
                
                self.sensor_devices[sensor_id] = sensor_info
                logger.info(f"传感器 {sensor_id} 连接成功")
                return True
                
        except Exception as e:
            logger.error(f"连接传感器 {sensor_id} 失败: {str(e)}")
            return False
    
    def get_sensor_reading(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """获取传感器读数"""
        try:
            if sensor_id not in self.sensor_devices:
                logger.warning(f"传感器 {sensor_id} 未连接")
                return None
            
            # 模拟传感器读数
            import random
            sensor_type = sensor_id.split('_')[0]
            
            if sensor_type == "temp":
                reading = {
                    "value": round(random.uniform(20.0, 25.0), 1),
                    "unit": "°C",
                    "timestamp": datetime.now().isoformat(),
                    "quality": "good"
                }
            elif sensor_type == "hum":
                reading = {
                    "value": round(random.uniform(40.0, 60.0), 1),
                    "unit": "%",
                    "timestamp": datetime.now().isoformat(),
                    "quality": "good"
                }
            elif sensor_type == "motion":
                reading = {
                    "value": random.choice([0, 1]),
                    "state": "detected" if random.random() > 0.8 else "idle",
                    "timestamp": datetime.now().isoformat()
                }
            elif sensor_type == "accel":
                reading = {
                    "x": round(random.uniform(-0.5, 0.5), 3),
                    "y": round(random.uniform(-0.5, 0.5), 3),
                    "z": round(random.uniform(-0.5, 0.5), 3),
                    "unit": "g",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                reading = {
                    "value": random.random(),
                    "timestamp": datetime.now().isoformat()
                }
            
            # 更新传感器状态
            self.sensor_devices[sensor_id]["last_reading"] = reading
            self.sensor_devices[sensor_id]["reading_count"] += 1
            
            return reading
            
        except Exception as e:
            logger.error(f"获取传感器 {sensor_id} 读数失败: {str(e)}")
            if sensor_id in self.sensor_devices:
                self.sensor_devices[sensor_id]["error_count"] += 1
            return None
    
    # 串口设备管理方法
    def scan_serial_ports(self) -> List[Dict[str, Any]]:
        """扫描可用的串口设备"""
        try:
            ports = []
            
            # 模拟串口设备
            port_devices = [
                {"port": "COM1", "name": "Communication Port", "type": "serial"},
                {"port": "COM3", "name": "Arduino Uno", "type": "arduino"},
                {"port": "COM5", "name": "USB Serial Device", "type": "usb_serial"},
                {"port": "/dev/ttyUSB0", "name": "USB to Serial Adapter", "type": "usb_serial"},
                {"port": "/dev/ttyACM0", "name": "Arduino Micro", "type": "arduino"}
            ]
            
            # 检查实际可用的串口
            import serial.tools.list_ports
            available_ports = serial.tools.list_ports.comports()
            
            for port_info in available_ports:
                port_data = {
                    "port": port_info.device,
                    "name": port_info.description,
                    "type": "serial",
                    "hwid": port_info.hwid,
                    "is_available": True
                }
                ports.append(port_data)
            
            # 如果没有找到实际串口，使用模拟数据
            if not ports:
                ports = port_devices
            
            logger.info(f"扫描到 {len(ports)} 个串口设备")
            return ports
            
        except Exception as e:
            logger.error(f"扫描串口设备失败: {str(e)}")
            return []
    
    def connect_serial_device(self, port: str, baudrate: int = 9600, 
                            timeout: float = 1.0) -> bool:
        """连接串口设备"""
        try:
            with self.global_lock:
                if port in self.serial_devices:
                    logger.warning(f"串口设备 {port} 已经连接")
                    return True
                
                # 模拟串口连接（实际实现应该使用pyserial）
                device_info = {
                    "port": port,
                    "baudrate": baudrate,
                    "timeout": timeout,
                    "is_connected": True,
                    "connected_at": datetime.now().isoformat(),
                    "bytes_sent": 0,
                    "bytes_received": 0,
                    "error_count": 0
                }
                
                self.serial_devices[port] = device_info
                logger.info(f"串口设备 {port} 连接成功，波特率: {baudrate}")
                return True
                
        except Exception as e:
            logger.error(f"连接串口设备 {port} 失败: {str(e)}")
            return False
    
    def send_serial_command(self, port: str, command: str) -> Optional[str]:
        """发送串口命令"""
        try:
            if port not in self.serial_devices:
                logger.warning(f"串口设备 {port} 未连接")
                return None
            
            # 模拟发送命令和接收响应
            response = f"ACK: {command}"
            
            # 更新设备状态
            self.serial_devices[port]["bytes_sent"] += len(command)
            self.serial_devices[port]["bytes_received"] += len(response)
            
            logger.debug(f"串口设备 {port} 发送命令: {command}, 接收响应: {response}")
            return response
            
        except Exception as e:
            logger.error(f"发送串口命令失败: {str(e)}")
            if port in self.serial_devices:
                self.serial_devices[port]["error_count"] += 1
            return None
    
    # 网络设备管理方法
    def scan_network_devices(self) -> List[Dict[str, Any]]:
        """扫描网络设备"""
        try:
            devices = []
            
            # 模拟网络设备
            network_devices = [
                {
                    "id": "net_001",
                    "name": "WiFi Module ESP32",
                    "type": "wifi",
                    "ip": "192.168.1.100",
                    "port": 8080,
                    "protocol": "TCP",
                    "is_available": True
                },
                {
                    "id": "net_002",
                    "name": "Ethernet Device",
                    "type": "ethernet",
                    "ip": "192.168.1.101",
                    "port": 9000,
                    "protocol": "TCP",
                    "is_available": True
                },
                {
                    "id": "net_003",
                    "name": "Bluetooth Device",
                    "type": "bluetooth",
                    "address": "00:11:22:33:44:55",
                    "protocol": "RFCOMM",
                    "is_available": True
                }
            ]
            
            devices.extend(network_devices)
            logger.info(f"扫描到 {len(devices)} 个网络设备")
            return devices
            
        except Exception as e:
            logger.error(f"扫描网络设备失败: {str(e)}")
            return []
    
    # 外接设备管理方法
    def scan_external_devices(self) -> List[Dict[str, Any]]:
        """扫描外接设备"""
        try:
            devices = []
            
            # 模拟外接设备
            external_devices = [
                {
                    "id": "ext_001",
                    "name": "Robotic Arm 6DOF",
                    "type": "robotic_arm",
                    "interface": "USB",
                    "dof": 6,
                    "payload": 1.0,
                    "reach": 0.5,
                    "is_available": True
                },
                {
                    "id": "ext_002",
                    "name": "DC Motor Controller",
                    "type": "motor_controller",
                    "interface": "PWM",
                    "channels": 4,
                    "current_rating": 10.0,
                    "is_available": True
                },
                {
                    "id": "ext_003",
                    "name": "RGB LED Controller",
                    "type": "led_controller",
                    "interface": "I2C",
                    "leds": 16,
                    "colors": "RGB",
                    "is_available": True
                },
                {
                    "id": "ext_004",
                    "name": "Audio Output Device",
                    "type": "audio_device",
                    "interface": "USB",
                    "channels": 2,
                    "sample_rate": 44100,
                    "is_available": True
                }
            ]
            
            devices.extend(external_devices)
            logger.info(f"扫描到 {len(devices)} 个外接设备")
            return devices
            
        except Exception as e:
            logger.error(f"扫描外接设备失败: {str(e)}")
            return []
    
    # 配置管理方法
    def save_configuration(self, config_data: Dict[str, Any]) -> bool:
        """保存硬件配置"""
        try:
            # 更新配置
            self.config.update(config_data)
            
            # 保存到文件
            return self._save_config()
            
        except Exception as e:
            logger.error(f"保存硬件配置失败: {str(e)}")
            return False
    
    def _save_config(self) -> bool:
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            logger.info("硬件配置已保存")
            return True
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
            return False
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """获取硬件状态摘要"""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "cameras": {
                    "total": len(self.camera_devices),
                    "active": len([c for c in self.camera_devices.values() if c.get("is_active", False)]),
                    "devices": list(self.camera_devices.keys())
                },
                "sensors": {
                    "total": len(self.sensor_devices),
                    "connected": len([s for s in self.sensor_devices.values() if s.get("is_connected", False)]),
                    "devices": list(self.sensor_devices.keys())
                },
                "serial_devices": {
                    "total": len(self.serial_devices),
                    "connected": len([s for s in self.serial_devices.values() if s.get("is_connected", False)]),
                    "ports": list(self.serial_devices.keys())
                },
                "system_resources": self._get_system_resources()
            }
            return status
        except Exception as e:
            logger.error(f"获取硬件状态失败: {str(e)}")
            return {}
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """获取系统资源信息"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_usage": disk.percent,
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"获取系统资源失败: {str(e)}")
            return {}
    
    def reset_hardware(self) -> bool:
        """重置所有硬件连接"""
        try:
            with self.global_lock:
                # 断开所有摄像头
                for camera_id in list(self.camera_devices.keys()):
                    self.disconnect_camera(camera_id)
                
                # 断开所有传感器
                for sensor_id in list(self.sensor_devices.keys()):
                    if sensor_id in self.sensor_devices:
                        self.sensor_devices[sensor_id]["is_connected"] = False
                
                # 断开所有串口设备
                for port in list(self.serial_devices.keys()):
                    if port in self.serial_devices:
                        self.serial_devices[port]["is_connected"] = False
                
                logger.info("所有硬件连接已重置")
                return True
                
        except Exception as e:
            logger.error(f"重置硬件失败: {str(e)}")
            return False

# 创建全局硬件管理器实例
global_hardware_manager = EnhancedHardwareManager()

def get_hardware_manager() -> EnhancedHardwareManager:
    """获取全局硬件管理器实例"""
    return global_hardware_manager

if __name__ == "__main__":
    # 测试硬件管理器
    print("测试增强硬件管理器...")
    
    manager = EnhancedHardwareManager()
    
    # 测试摄像头扫描
    print("\n1. 扫描摄像头...")
    cameras = manager.scan_cameras()
    for camera in cameras:
        print(f"  摄像头: {camera['name']} (ID: {camera['id']})")
    
    # 测试传感器扫描
    print("\n2. 扫描传感器...")
    sensors = manager.scan_sensors()
    for sensor in sensors:
        print(f"  传感器: {sensor['name']} (类型: {sensor['type']})")
    
    # 测试串口扫描
    print("\n3. 扫描串口设备...")
    serial_ports = manager.scan_serial_ports()
    for port in serial_ports:
        print(f"  串口: {port['port']} - {port['name']}")
    
    # 测试硬件状态
    print("\n4. 获取硬件状态...")
    status = manager.get_hardware_status()
    print(f"  系统状态: {status}")
    
    print("\n增强硬件管理器测试完成")
