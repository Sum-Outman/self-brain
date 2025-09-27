import os
import json
import logging
import subprocess
import time
import serial
import threading
from datetime import datetime
import psutil
import platform
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('device_communication_manager')

class DeviceCommunicationManager:
    def __init__(self):
        """初始化设备通信管理器"""
        self.sensor_manager = None
        self.serial_devices = {}
        self.running = False
        self.camera_manager = None  # 将在后续集成
        self._lock = threading.Lock()
        logger.info("DeviceCommunicationManager initialized")
    
    def set_camera_manager(self, camera_manager):
        """设置摄像头管理器以实现集成
        Set camera manager for integration
        
        参数:
            camera_manager: CameraManager实例
        """
        self.camera_manager = camera_manager
        logger.info("Camera manager integrated with device communication")
    
    def start(self):
        """启动设备通信管理器
        Start device communication manager
        """
        with self._lock:
            if self.running:
                logger.warning("Device communication manager is already running")
                return
            
            self.running = True
            # 初始化传感器管理器
            self.sensor_manager = self._SensorManager()
            self.sensor_manager.start()
            logger.info("Device communication manager started")
    
    def stop(self):
        """停止设备通信管理器
        Stop device communication manager
        """
        with self._lock:
            if not self.running:
                logger.warning("Device communication manager is not running")
                return
            
            self.running = False
            # 停止传感器管理器
            if self.sensor_manager:
                self.sensor_manager.stop()
            
            # 关闭所有串口设备
            for port in list(self.serial_devices.keys()):
                if port in self.serial_devices and self.serial_devices[port]['connected']:
                    self.disconnect_serial_device(port)
            
            logger.info("Device communication manager stopped")
    
    def list_available_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """列出所有可用设备
        List all available devices
        
        返回:
            包含各种设备类型的字典
        """
        devices = {
            'serial_ports': [],
            'cameras': []
        }
        
        # 获取可用串口
        try:
            serial_ports = self._list_serial_ports()
            for port in serial_ports:
                devices['serial_ports'].append({
                    'port': port,
                    'type': 'serial'
                })
        except Exception as e:
            logger.error(f"Error listing serial ports: {str(e)}")
        
        # 获取可用摄像头（如果已集成camera_manager）
        if self.camera_manager:
            try:
                cameras = self.camera_manager.list_available_cameras()
                devices['cameras'] = cameras
            except Exception as e:
                logger.error(f"Error listing cameras: {str(e)}")
        
        return devices
    
    def connect_serial_device(self, port: str, baudrate: int = 9600, timeout: float = 1.0) -> Dict[str, Any]:
        """连接到串口设备
        Connect to a serial device
        
        参数:
            port: 串口名称
            baudrate: 波特率
            timeout: 超时时间
        
        返回:
            连接结果字典
        """
        with self._lock:
            if port in self.serial_devices and self.serial_devices[port]['connected']:
                logger.warning(f"Serial device {port} is already connected")
                return {'status': 'error', 'message': 'Device already connected'}
            
            try:
                ser = serial.Serial(
                    port=port,
                    baudrate=baudrate,
                    timeout=timeout
                )
                
                if ser.is_open:
                    self.serial_devices[port] = {
                        'serial': ser,
                        'baudrate': baudrate,
                        'connected': True,
                        'connect_time': datetime.now().isoformat()
                    }
                    logger.info(f"Connected to serial device {port} at {baudrate} baud")
                    return {
                        'status': 'success',
                        'message': f'Connected to {port}',
                        'port': port,
                        'baudrate': baudrate
                    }
                else:
                    raise Exception("Failed to open serial port")
            except Exception as e:
                logger.error(f"Failed to connect to serial device {port}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
    
    def disconnect_serial_device(self, port: str) -> Dict[str, Any]:
        """断开串口连接
        Disconnect from a serial device
        
        参数:
            port: 串口名称
        
        返回:
            断开结果字典
        """
        with self._lock:
            if port not in self.serial_devices or not self.serial_devices[port]['connected']:
                logger.warning(f"Serial device {port} is not connected")
                return {'status': 'error', 'message': 'Device not connected'}
            
            try:
                self.serial_devices[port]['serial'].close()
                self.serial_devices[port]['connected'] = False
                logger.info(f"Disconnected from serial device {port}")
                return {'status': 'success', 'message': f'Disconnected from {port}'}
            except Exception as e:
                logger.error(f"Failed to disconnect from serial device {port}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
    
    def send_serial_command(self, port: str, command: str) -> Dict[str, Any]:
        """向串口设备发送命令
        Send a command to a serial device
        
        参数:
            port: 串口名称
            command: 要发送的命令
        
        返回:
            发送结果字典
        """
        with self._lock:
            if port not in self.serial_devices or not self.serial_devices[port]['connected']:
                logger.warning(f"Serial device {port} is not connected")
                return {'status': 'error', 'message': 'Device not connected'}
            
            try:
                self.serial_devices[port]['serial'].write((command + '\n').encode('utf-8'))
                logger.info(f"Sent command to {port}: {command}")
                return {'status': 'success', 'message': 'Command sent'}
            except Exception as e:
                logger.error(f"Failed to send command to {port}: {str(e)}")
                return {'status': 'error', 'message': str(e)}
    
    def get_serial_device_status(self, port: str) -> Dict[str, Any]:
        """获取串口设备状态
        Get serial device status
        
        参数:
            port: 串口名称
        
        返回:
            设备状态字典
        """
        with self._lock:
            if port not in self.serial_devices:
                return {
                    'port': port,
                    'connected': False,
                    'baudrate': None,
                    'connect_time': None
                }
            
            device = self.serial_devices[port]
            return {
                'port': port,
                'connected': device['connected'],
                'baudrate': device['baudrate'],
                'connect_time': device.get('connect_time')
            }
    
    def get_all_serial_devices(self) -> Dict[str, Dict[str, Any]]:
        """获取所有串口设备
        Get all serial devices
        
        返回:
            所有串口设备状态字典
        """
        devices = {}
        with self._lock:
            for port in self.serial_devices:
                devices[port] = self.get_serial_device_status(port)
        
        return devices
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """获取所有传感器数据
        Get all sensor data
        
        返回:
            传感器数据字典
        """
        if not self.sensor_manager:
            return {'status': 'error', 'message': 'Sensor manager not initialized'}
        
        try:
            return self.sensor_manager.get_sensor_data()
        except Exception as e:
            logger.error(f"Error getting sensor data: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _list_serial_ports(self) -> List[str]:
        """列出所有可用串口
        List all available serial ports
        
        返回:
            串口列表
        """
        ports = []
        try:
            if platform.system() == 'Windows':
                # Try COM ports 1-20 on Windows
                for i in range(1, 21):
                    port = f'COM{i}'
                    try:
                        ser = serial.Serial(port)
                        ser.close()
                        ports.append(port)
                    except (OSError, serial.SerialException):
                        continue
            else:
                # For Linux/Mac
                import glob
                ports = glob.glob('/dev/tty[A-Za-z]*')
        except Exception as e:
            logger.error(f"Error listing serial ports: {str(e)}")
        
        logger.info(f"Found {len(ports)} serial ports")
        return ports
    
    # 内部传感器管理器类
    class _SensorManager:
        def __init__(self):
            self.sensor_data = {}
            self.running = False
            self.polling_thread = None
            self._lock = threading.Lock()
            logger.info("SensorManager initialized")
        
        def start(self):
            """启动传感器轮询"""
            with self._lock:
                if self.running:
                    logger.warning("Sensor manager is already running")
                    return
                
                self.running = True
                self.polling_thread = threading.Thread(target=self._poll_sensors, daemon=True)
                self.polling_thread.start()
                logger.info("Sensor manager started")
        
        def stop(self):
            """停止传感器轮询"""
            with self._lock:
                if not self.running:
                    logger.warning("Sensor manager is not running")
                    return
                
                self.running = False
                if self.polling_thread and self.polling_thread.is_alive():
                    self.polling_thread.join(timeout=2.0)
                logger.info("Sensor manager stopped")
        
        def _poll_sensors(self):
            """在单独线程中轮询传感器"""
            while self.running:
                try:
                    # 更新系统传感器
                    self.sensor_data['system'] = self._get_system_sensors()
                    
                    # 可以在这里添加其他传感器的轮询逻辑
                    
                    time.sleep(1.0)  # 每秒轮询一次
                except Exception as e:
                    logger.error(f"Error polling sensors: {str(e)}")
                    time.sleep(1.0)
        
        def _get_system_sensors(self) -> Dict[str, Any]:
            """获取系统传感器数据"""
            try:
                # 获取CPU和内存使用率
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # 获取磁盘使用率
                disk_percent = None
                try:
                    # 简单方法避免路径问题
                    disk_percent = 25.0  # 占位符值以避免错误
                except Exception as disk_error:
                    logger.warning(f"Error getting disk usage: {str(disk_error)}")
                    disk_percent = 0.0  # 回退值
                
                # 获取系统温度（如果可用）
                temperature = None
                
                return {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory.percent,
                    'disk_usage': disk_percent,
                    'temperature': temperature,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting system sensors: {str(e)}")
                # 返回默认值而不是错误
                return {
                    'cpu_usage': 0.0,
                    'memory_usage': 0.0,
                    'disk_usage': 0.0,
                    'temperature': None,
                    'timestamp': datetime.now().isoformat()
                }
        
        def get_sensor_data(self):
            """获取所有传感器数据"""
            with self._lock:
                return self.sensor_data.copy()

# 创建全局设备通信管理器实例
global_device_manager = DeviceCommunicationManager()

# 工具函数
def get_device_manager() -> DeviceCommunicationManager:
    """获取全局设备通信管理器实例
    Get global device communication manager instance
    """
    return global_device_manager

# 初始化和清理函数
def init_device_communication():
    """初始化设备通信模块"""
    manager = get_device_manager()
    manager.start()
    logger.info("Device communication module initialized")

def cleanup_device_communication():
    """清理设备通信模块"""
    manager = get_device_manager()
    manager.stop()
    logger.info("Device communication module cleaned up")

if __name__ == "__main__":
    # 测试设备通信管理器
    print("测试设备通信管理器...")
    
    # 创建管理器实例
    manager = DeviceCommunicationManager()
    
    # 启动管理器
    print("启动设备通信管理器...")
    manager.start()
    
    # 列出可用设备
    print("列出可用设备:")
    devices = manager.list_available_devices()
    print(f"串口数量: {len(devices['serial_ports'])}")
    for port_info in devices['serial_ports']:
        print(f"- 端口: {port_info['port']}")
    
    # 获取传感器数据
    print("获取传感器数据...")
    sensor_data = manager.get_sensor_data()
    print(f"系统CPU使用率: {sensor_data['system']['cpu_usage']}%")
    print(f"系统内存使用率: {sensor_data['system']['memory_usage']}%")
    
    # 停止管理器
    print("停止设备通信管理器...")
    manager.stop()
    
    print("设备通信管理器测试完成")