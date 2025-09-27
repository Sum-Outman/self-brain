# -*- coding: utf-8 -*-
"""
设备通信模块
Device Communication Module

负责管理与外部设备的通信，提供设备控制、传感器数据获取和设备状态监控等功能
"""

import serial
import socket
import threading
import time
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
import concurrent.futures
from flask import Blueprint, request, jsonify

# 设置日志 | Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeviceCommunication")

# 创建设备通信蓝图
device_bp = Blueprint('device', __name__, url_prefix='/api/devices')

class DeviceManager:
    """设备管理器
    Device Manager
    
    负责管理所有连接的外部设备，提供统一的接口进行设备控制和数据获取
    """
    def __init__(self):
        """初始化设备管理器"""
        self.devices = {}
        self.device_locks = {}
        self.active_devices = set()
        self.global_lock = threading.Lock()
        self.device_configs = {}
        self.sensor_readings = {}
        logger.info("设备管理器已初始化 | Device Manager initialized")
        
    def list_available_serial_ports(self) -> List[Dict[str, Any]]:
        """列出所有可用的串口设备
        List all available serial port devices
        """
        available_ports = []
        
        # 尝试不同平台的串口检测方法
        if sys.platform.startswith('win'):
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                try:
                    available_ports.append({
                        'port': port.device,
                        'name': port.description,
                        'type': 'serial',
                        'hwid': port.hwid
                    })
                except Exception as e:
                    logger.warning(f"无法获取端口信息: {str(e)}")
        else:
            # Linux/MacOS平台的简单检测
            import glob
            port_patterns = ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/tty.*']
            for pattern in port_patterns:
                for port in glob.glob(pattern):
                    available_ports.append({
                        'port': port,
                        'name': f"Serial Device ({port})",
                        'type': 'serial'
                    })
        
        logger.info(f"找到 {len(available_ports)} 个可用串口设备 | Found {len(available_ports)} available serial devices")
        return available_ports
    
    def connect_serial_device(self, device_id: str, port: str, baudrate: int = 9600, 
                             timeout: float = 1.0) -> bool:
        """连接串口设备
        Connect to serial device
        
        参数:
            device_id: 设备唯一标识符
            port: 串口名称
            baudrate: 波特率
            timeout: 超时时间
        
        返回:
            连接成功返回True，否则返回False
        """
        with self.global_lock:
            if device_id in self.active_devices:
                logger.warning(f"设备 {device_id} 已经在运行 | Device {device_id} is already running")
                return True
            
            try:
                # 创建设备锁
                if device_id not in self.device_locks:
                    self.device_locks[device_id] = threading.Lock()
                
                with self.device_locks[device_id]:
                    # 初始化串口连接
                    ser = serial.Serial(
                        port=port,
                        baudrate=baudrate,
                        timeout=timeout,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS
                    )
                    
                    if not ser.is_open:
                        ser.open()
                        
                    # 保存设备配置
                    self.device_configs[device_id] = {
                        'type': 'serial',
                        'port': port,
                        'baudrate': baudrate,
                        'timeout': timeout
                    }
                    
                    # 创建设备数据
                    self.devices[device_id] = {
                        'connection': ser,
                        'last_data': None,
                        'last_error': None,
                        'is_connected': True,
                        'connect_time': time.time(),
                        'config': self.device_configs[device_id]
                    }
                    
                    self.active_devices.add(device_id)
                    
                    # 初始化传感器读数存储
                    self.sensor_readings[device_id] = {}
                    
                    # 启动数据接收线程
                    self._start_data_receiver(device_id)
                    
                    logger.info(f"串口设备 {device_id} 已连接到 {port} | Serial device {device_id} connected to {port}")
                    return True
                    
            except Exception as e:
                logger.error(f"连接串口设备 {device_id} 时出错: {str(e)} | Error connecting serial device {device_id}: {str(e)}")
                if device_id in self.devices:
                    self.devices[device_id]["last_error"] = str(e)
                return False
    
    def disconnect_device(self, device_id: str) -> bool:
        """断开设备连接
        Disconnect device
        
        参数:
            device_id: 设备唯一标识符
        
        返回:
            断开成功返回True，否则返回False
        """
        with self.global_lock:
            if device_id not in self.active_devices:
                logger.warning(f"设备 {device_id} 未连接 | Device {device_id} is not connected")
                return True
            
            try:
                if device_id in self.device_locks:
                    with self.device_locks[device_id]:
                        if device_id in self.devices:
                            conn = self.devices[device_id]["connection"]
                            if hasattr(conn, 'is_open') and conn.is_open:
                                conn.close()
                                
                            # 清理资源
                            del self.devices[device_id]
                            
                    # 移除锁
                    del self.device_locks[device_id]
                
                # 从活动设备集合中移除
                self.active_devices.remove(device_id)
                
                # 清理配置
                if device_id in self.device_configs:
                    del self.device_configs[device_id]
                    
                # 清理传感器读数
                if device_id in self.sensor_readings:
                    del self.sensor_readings[device_id]
                
                logger.info(f"设备 {device_id} 已断开连接 | Device {device_id} disconnected successfully")
                return True
                
            except Exception as e:
                logger.error(f"断开设备 {device_id} 连接时出错: {str(e)} | Error disconnecting device {device_id}: {str(e)}")
                return False
    
    def _start_data_receiver(self, device_id: str):
        """启动数据接收线程
        Start data receiver thread
        """
        def receive_data():
            device = self.devices.get(device_id)
            if not device or not device['is_connected']:
                return
            
            ser = device['connection']
            buffer = ""
            
            while device_id in self.active_devices:
                try:
                    if ser.in_waiting:
                        data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                        buffer += data
                        
                        # 解析完整的数据包（假设每行是一个完整的数据包）
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if line:
                                self._process_device_data(device_id, line)
                    
                    time.sleep(0.01)  # 短暂休眠避免CPU占用过高
                except Exception as e:
                    logger.error(f"接收设备 {device_id} 数据时出错: {str(e)}")
                    device['last_error'] = str(e)
                    time.sleep(0.5)
        
        thread = threading.Thread(target=receive_data, daemon=True)
        thread.start()
    
    def _process_device_data(self, device_id: str, data: str):
        """处理设备数据
        Process device data
        """
        try:
            # 尝试解析JSON格式数据
            if data.startswith('{') and data.endswith('}'):
                parsed_data = json.loads(data)
                self.devices[device_id]['last_data'] = parsed_data
                
                # 更新传感器读数
                if isinstance(parsed_data, dict):
                    for key, value in parsed_data.items():
                        if key not in self.sensor_readings[device_id]:
                            self.sensor_readings[device_id][key] = []
                        
                        # 保留最近100个读数
                        self.sensor_readings[device_id][key].append({
                            'value': value,
                            'timestamp': time.time()
                        })
                        
                        if len(self.sensor_readings[device_id][key]) > 100:
                            self.sensor_readings[device_id][key].pop(0)
            else:
                # 处理非JSON格式数据
                self.devices[device_id]['last_data'] = data
                logger.debug(f"接收到设备 {device_id} 的非JSON数据: {data}")
        except Exception as e:
            logger.error(f"处理设备 {device_id} 数据时出错: {str(e)}")
    
    def send_command(self, device_id: str, command: str, timeout: float = 2.0) -> Tuple[bool, Optional[str]]:
        """向设备发送命令
        Send command to device
        
        参数:
            device_id: 设备唯一标识符
            command: 要发送的命令
            timeout: 等待响应的超时时间
        
        返回:
            (成功标志, 响应数据或错误信息)
        """
        if device_id not in self.active_devices:
            return False, f"设备 {device_id} 未连接"
        
        try:
            with self.device_locks[device_id]:
                device = self.devices[device_id]
                conn = device['connection']
                
                if not conn or not conn.is_open:
                    return False, "设备连接已关闭"
                
                # 发送命令（确保以换行符结束）
                if not command.endswith('\n'):
                    command += '\n'
                
                conn.write(command.encode('utf-8'))
                conn.flush()
                
                logger.info(f"向设备 {device_id} 发送命令: {command.strip()}")
                
                # 等待响应
                start_time = time.time()
                response = ""
                while time.time() - start_time < timeout:
                    if conn.in_waiting:
                        response += conn.read(conn.in_waiting).decode('utf-8', errors='ignore')
                        # 检查是否接收到完整的响应（假设以换行符结束）
                        if '\n' in response:
                            break
                    time.sleep(0.01)
                
                if response:
                    return True, response.strip()
                return True, None  # 命令发送成功但没有响应
        except Exception as e:
            error_msg = str(e)
            logger.error(f"向设备 {device_id} 发送命令时出错: {error_msg}")
            return False, error_msg
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """获取设备状态
        Get device status
        
        参数:
            device_id: 设备唯一标识符
        
        返回:
            设备状态信息
        """
        if device_id not in self.active_devices:
            return None
        
        try:
            device = self.devices[device_id]
            config = self.device_configs[device_id]
            
            status = {
                'device_id': device_id,
                'type': config['type'],
                'connected': device['is_connected'],
                'connect_time': device['connect_time'],
                'uptime': time.time() - device['connect_time'],
                'last_data': device['last_data'],
                'last_error': device['last_error'],
                'config': config
            }
            
            # 添加传感器读数摘要
            if device_id in self.sensor_readings and self.sensor_readings[device_id]:
                sensor_summary = {}
                for sensor_name, readings in self.sensor_readings[device_id].items():
                    if readings:
                        # 获取最新读数
                        latest_reading = readings[-1]
                        sensor_summary[sensor_name] = {
                            'value': latest_reading['value'],
                            'timestamp': latest_reading['timestamp']
                        }
                status['sensors'] = sensor_summary
            
            return status
        except Exception as e:
            logger.error(f"获取设备 {device_id} 状态时出错: {str(e)}")
            return None
    
    def get_all_devices_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有设备的状态
        Get status of all devices
        
        返回:
            所有设备的状态信息
        """
        statuses = {}
        for device_id in list(self.active_devices):
            status = self.get_device_status(device_id)
            if status:
                statuses[device_id] = status
        return statuses
    
    def get_sensor_readings(self, device_id: str, sensor_name: str = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """获取传感器读数
        Get sensor readings
        
        参数:
            device_id: 设备唯一标识符
            sensor_name: 传感器名称，None表示所有传感器
            limit: 返回的最大读数数量
        
        返回:
            传感器读数列表
        """
        if device_id not in self.sensor_readings:
            return []
        
        if sensor_name:
            if sensor_name not in self.sensor_readings[device_id]:
                return []
            # 返回指定传感器的读数，限制数量
            return self.sensor_readings[device_id][sensor_name][-limit:]
        else:
            # 返回所有传感器的最新读数
            all_readings = []
            for name, readings in self.sensor_readings[device_id].items():
                if readings:
                    all_readings.append({
                        'sensor': name,
                        'value': readings[-1]['value'],
                        'timestamp': readings[-1]['timestamp']
                    })
            return all_readings

# 全局设备管理器实例
global_device_manager = None

# API端点路由
@device_bp.route('/list_serial_ports', methods=['GET'])
def list_serial_ports():
    """列出所有可用的串口设备"""
    try:
        ports = global_device_manager.list_available_serial_ports()
        return jsonify({
            'status': 'success',
            'ports': ports
        })
    except Exception as e:
        logger.error(f"列出串口设备时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@device_bp.route('/connect', methods=['POST'])
def connect_device():
    """连接设备"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': '无效的JSON数据'
            }), 400
        
        device_id = data.get('device_id')
        device_type = data.get('type', 'serial')
        
        if not device_id:
            return jsonify({
                'status': 'error',
                'message': '设备ID不能为空'
            }), 400
        
        if device_type == 'serial':
            port = data.get('port')
            baudrate = data.get('baudrate', 9600)
            timeout = data.get('timeout', 1.0)
            
            if not port:
                return jsonify({
                    'status': 'error',
                    'message': '串口名称不能为空'
                }), 400
            
            success = global_device_manager.connect_serial_device(device_id, port, baudrate, timeout)
            if success:
                return jsonify({
                    'status': 'success',
                    'message': f'设备 {device_id} 连接成功'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'设备 {device_id} 连接失败'
                }), 500
        else:
            return jsonify({
                'status': 'error',
                'message': f'不支持的设备类型: {device_type}'
            }), 400
    except Exception as e:
        logger.error(f"连接设备时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@device_bp.route('/disconnect/<device_id>', methods=['POST'])
def disconnect_device_route(device_id):
    """断开设备连接"""
    try:
        success = global_device_manager.disconnect_device(device_id)
        if success:
            return jsonify({
                'status': 'success',
                'message': f'设备 {device_id} 断开连接成功'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'设备 {device_id} 断开连接失败'
            }), 500
    except Exception as e:
        logger.error(f"断开设备连接时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@device_bp.route('/status/<device_id>', methods=['GET'])
def get_device_status_route(device_id):
    """获取设备状态"""
    try:
        status = global_device_manager.get_device_status(device_id)
        if status:
            return jsonify({
                'status': 'success',
                'device_status': status
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'设备 {device_id} 未连接'
            }), 404
    except Exception as e:
        logger.error(f"获取设备状态时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@device_bp.route('/status', methods=['GET'])
def get_all_devices_status_route():
    """获取所有设备的状态"""
    try:
        statuses = global_device_manager.get_all_devices_status()
        return jsonify({
            'status': 'success',
            'devices_count': len(statuses),
            'devices_status': statuses
        })
    except Exception as e:
        logger.error(f"获取所有设备状态时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@device_bp.route('/command/<device_id>', methods=['POST'])
def send_device_command(device_id):
    """向设备发送命令"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': '无效的JSON数据'
            }), 400
        
        command = data.get('command')
        timeout = data.get('timeout', 2.0)
        
        if not command:
            return jsonify({
                'status': 'error',
                'message': '命令不能为空'
            }), 400
        
        success, response = global_device_manager.send_command(device_id, command, timeout)
        if success:
            return jsonify({
                'status': 'success',
                'command': command,
                'response': response
            })
        else:
            return jsonify({
                'status': 'error',
                'message': response
            }), 500
    except Exception as e:
        logger.error(f"向设备发送命令时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@device_bp.route('/sensors/<device_id>', methods=['GET'])
def get_device_sensors(device_id):
    """获取设备的传感器读数"""
    try:
        sensor_name = request.args.get('sensor')
        limit = int(request.args.get('limit', 100))
        
        readings = global_device_manager.get_sensor_readings(device_id, sensor_name, limit)
        return jsonify({
            'status': 'success',
            'readings_count': len(readings),
            'readings': readings
        })
    except Exception as e:
        logger.error(f"获取传感器读数时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# 初始化和清理函数
def init_device_communication():
    """初始化设备通信系统"""
    global global_device_manager
    global_device_manager = DeviceManager()
    logger.info("设备通信系统已初始化 | Device communication system initialized")


def cleanup_device_communication():
    """清理设备通信系统"""
    global global_device_manager
    if global_device_manager:
        # 断开所有设备连接
        for device_id in list(global_device_manager.active_devices):
            global_device_manager.disconnect_device(device_id)
        logger.info("设备通信系统已清理 | Device communication system cleaned up")

# 导入sys模块
try:
    import sys
except ImportError:
    logger.error("无法导入sys模块 | Failed to import sys module")