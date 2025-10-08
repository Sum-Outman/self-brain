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

"""
设备通讯接口模块
提供与外部设备的通讯能力，支持多种通讯协议
"""

import os
import sys
import time
import serial
import socket
import threading
import logging
import json
import asyncio
import subprocess
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DeviceInterface')

class DeviceProtocol(Enum):
    SERIAL = "serial"
    TCP = "tcp"
    UDP = "udp"
    MQTT = "mqtt"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    USB_HID = "usb_hid"
    BLUETOOTH = "bluetooth"
    WIFI_DIRECT = "wifi_direct"

class DeviceStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CLOSED = "closed"

class DeviceInterfaceException(Exception):
    """设备接口异常类"""
    pass

class DeviceInterface:
    """
    设备通讯接口基类
    提供设备连接、断开、数据收发的基本接口
    """
    def __init__(self, device_id: str, protocol: DeviceProtocol, config: Dict[str, Any]):
        """
        初始化设备接口
        
        参数:
            device_id: 设备ID
            protocol: 通讯协议
            config: 配置参数
        """
        self.device_id = device_id
        self.protocol = protocol
        self.config = config
        self.status = DeviceStatus.DISCONNECTED
        self.error_message = None
        self.connection = None
        self.lock = threading.Lock()
        
        # 数据回调函数
        self.data_received_callback = None
        self.status_changed_callback = None
        
        # 接收线程
        self.receive_thread = None
        self.stop_event = threading.Event()
        
        logger.info(f"设备接口初始化 - ID: {device_id}, 协议: {protocol.value}")
        
    def connect(self) -> bool:
        """连接设备"""
        with self.lock:
            if self.status == DeviceStatus.CONNECTED:
                logger.warning(f"设备已连接 - ID: {self.device_id}")
                return True
                
            try:
                self.status = DeviceStatus.CONNECTING
                self._notify_status_change()
                
                # 具体连接逻辑由子类实现
                self.connection = self._create_connection()
                
                self.status = DeviceStatus.CONNECTED
                self.error_message = None
                
                # 启动接收线程
                self.stop_event.clear()
                self.receive_thread = threading.Thread(target=self._receive_loop)
                self.receive_thread.daemon = True
                self.receive_thread.start()
                
                logger.info(f"设备连接成功 - ID: {self.device_id}")
                self._notify_status_change()
                return True
                
            except Exception as e:
                self.status = DeviceStatus.ERROR
                self.error_message = str(e)
                self.connection = None
                
                logger.error(f"设备连接失败 - ID: {self.device_id}, 错误: {str(e)}")
                self._notify_status_change()
                return False
                
    def disconnect(self) -> bool:
        """断开设备连接"""
        with self.lock:
            if self.status == DeviceStatus.DISCONNECTED or self.status == DeviceStatus.CLOSED:
                logger.warning(f"设备未连接 - ID: {self.device_id}")
                return True
                
            try:
                # 停止接收线程
                self.stop_event.set()
                if self.receive_thread:
                    self.receive_thread.join(timeout=2.0)
                    
                # 关闭连接
                if self.connection:
                    self._close_connection()
                    self.connection = None
                    
                self.status = DeviceStatus.CLOSED
                self.error_message = None
                
                logger.info(f"设备断开连接 - ID: {self.device_id}")
                self._notify_status_change()
                return True
                
            except Exception as e:
                self.status = DeviceStatus.ERROR
                self.error_message = str(e)
                
                logger.error(f"设备断开失败 - ID: {self.device_id}, 错误: {str(e)}")
                self._notify_status_change()
                return False
                
    def send_data(self, data: Union[bytes, str, Dict]) -> bool:
        """
        发送数据到设备
        
        参数:
            data: 要发送的数据
        
        返回:
            是否发送成功
        """
        with self.lock:
            if self.status != DeviceStatus.CONNECTED:
                logger.warning(f"设备未连接，无法发送数据 - ID: {self.device_id}")
                return False
                
            try:
                # 具体发送逻辑由子类实现
                self._send_data(data)
                
                logger.debug(f"数据发送成功 - ID: {self.device_id}, 数据: {data}")
                return True
                
            except Exception as e:
                self.status = DeviceStatus.ERROR
                self.error_message = str(e)
                
                logger.error(f"数据发送失败 - ID: {self.device_id}, 错误: {str(e)}")
                self._notify_status_change()
                return False
                
    def receive_data(self, timeout: float = None) -> Optional[Any]:
        """
        接收设备数据
        
        参数:
            timeout: 超时时间（秒）
        
        返回:
            接收到的数据
        """
        # 注意：该方法仅用于同步接收，异步接收通过回调实现
        raise NotImplementedError("子类必须实现receive_data方法")
        
    def get_status(self) -> Dict[str, Any]:
        """获取设备状态"""
        return {
            'device_id': self.device_id,
            'protocol': self.protocol.value,
            'status': self.status.value,
            'error_message': self.error_message,
            'config': self.config
        }
        
    def set_data_received_callback(self, callback: Callable[[str, Any], None]):
        """
        设置数据接收回调函数
        
        参数:
            callback: 回调函数，参数为(device_id, data)
        """
        self.data_received_callback = callback
        
    def set_status_changed_callback(self, callback: Callable[[str, Dict], None]):
        """
        设置状态变更回调函数
        
        参数:
            callback: 回调函数，参数为(device_id, status)
        """
        self.status_changed_callback = callback
        
    def _notify_data_received(self, data: Any):
        """通知数据接收"""
        if self.data_received_callback:
            try:
                self.data_received_callback(self.device_id, data)
            except Exception as e:
                logger.error(f"数据接收回调出错 - ID: {self.device_id}, 错误: {str(e)}")
                
    def _notify_status_change(self):
        """通知状态变更"""
        if self.status_changed_callback:
            try:
                self.status_changed_callback(self.device_id, self.get_status())
            except Exception as e:
                logger.error(f"状态变更回调出错 - ID: {self.device_id}, 错误: {str(e)}")
                
    def _create_connection(self):
        """创建连接"""
        raise NotImplementedError("子类必须实现_create_connection方法")
        
    def _close_connection(self):
        """关闭连接"""
        raise NotImplementedError("子类必须实现_close_connection方法")
        
    def _send_data(self, data: Any):
        """发送数据"""
        raise NotImplementedError("子类必须实现_send_data方法")
        
    def _receive_loop(self):
        """接收循环"""
        raise NotImplementedError("子类必须实现_receive_loop方法")

class SerialDeviceInterface(DeviceInterface):
    """串口设备接口"""
    def __init__(self, device_id: str, config: Dict[str, Any]):
        # 默认串口配置
        default_config = {
            'port': 'COM1',
            'baudrate': 9600,
            'bytesize': 8,
            'parity': 'N',
            'stopbits': 1,
            'timeout': 0.1,
            'xonxoff': False,
            'rtscts': False,
            'dsrdtr': False
        }
        
        # 合并配置
        merged_config = {**default_config, **config}
        
        super().__init__(device_id, DeviceProtocol.SERIAL, merged_config)
        
    def _create_connection(self) -> serial.Serial:
        """创建串口连接"""
        try:
            # 创建串口连接
            ser = serial.Serial(
                port=self.config['port'],
                baudrate=self.config['baudrate'],
                bytesize=self.config['bytesize'],
                parity=self.config['parity'],
                stopbits=self.config['stopbits'],
                timeout=self.config['timeout'],
                xonxoff=self.config['xonxoff'],
                rtscts=self.config['rtscts'],
                dsrdtr=self.config['dsrdtr']
            )
            
            if not ser.is_open:
                ser.open()
                
            return ser
            
        except Exception as e:
            logger.error(f"串口连接创建失败 - 端口: {self.config['port']}, 错误: {str(e)}")
            raise
            
    def _close_connection(self):
        """关闭串口连接"""
        if self.connection and hasattr(self.connection, 'close'):
            try:
                self.connection.close()
            except Exception as e:
                logger.error(f"串口关闭失败 - 错误: {str(e)}")
                
    def _send_data(self, data: Union[bytes, str, Dict]):
        """发送数据到串口"""
        if isinstance(data, dict):
            # 转换字典为JSON字符串
            data_str = json.dumps(data) + '\n'
            data_bytes = data_str.encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
            
        self.connection.write(data_bytes)
        self.connection.flush()
        
    def receive_data(self, timeout: float = None) -> Optional[Union[bytes, str]]:
        """接收串口数据"""
        if timeout is not None:
            original_timeout = self.connection.timeout
            self.connection.timeout = timeout
            
        try:
            if self.connection.in_waiting > 0:
                data = self.connection.read(self.connection.in_waiting)
                return data
            return None
            
        finally:
            if timeout is not None:
                self.connection.timeout = original_timeout
                
    def _receive_loop(self):
        """串口接收循环"""
        buffer = b''
        
        while not self.stop_event.is_set():
            try:
                if self.status != DeviceStatus.CONNECTED:
                    time.sleep(0.1)
                    continue
                    
                # 读取数据
                data = self.receive_data(timeout=0.1)
                
                if data:
                    buffer += data
                    
                    # 尝试解析完整的数据包（假设以换行符分隔）
                    while b'\n' in buffer:
                        packet, buffer = buffer.split(b'\n', 1)
                        
                        try:
                            # 尝试解析为JSON
                            packet_str = packet.decode('utf-8')
                            packet_data = json.loads(packet_str)
                        except:
                            # 如果不是有效的JSON，返回原始字符串
                            packet_data = packet_str
                            
                        # 通知数据接收
                        self._notify_data_received(packet_data)
                        
            except Exception as e:
                logger.error(f"串口接收循环出错 - ID: {self.device_id}, 错误: {str(e)}")
                self.status = DeviceStatus.ERROR
                self.error_message = str(e)
                self._notify_status_change()
                break
                
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.01)

class TCPDeviceInterface(DeviceInterface):
    """TCP设备接口"""
    def __init__(self, device_id: str, config: Dict[str, Any]):
        # 默认TCP配置
        default_config = {
            'host': '127.0.0.1',
            'port': 8080,
            'buffer_size': 4096,
            'reconnect_interval': 5.0,
            'max_reconnect_attempts': 3
        }
        
        # 合并配置
        merged_config = {**default_config, **config}
        
        super().__init__(device_id, DeviceProtocol.TCP, merged_config)
        
        # 重连计数
        self.reconnect_attempts = 0
        
    def _create_connection(self) -> socket.socket:
        """创建TCP连接"""
        try:
            # 创建TCP套接字
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            
            # 连接到服务器
            sock.connect((self.config['host'], self.config['port']))
            
            # 重置重连计数
            self.reconnect_attempts = 0
            
            return sock
            
        except Exception as e:
            logger.error(f"TCP连接创建失败 - {self.config['host']}:{self.config['port']}, 错误: {str(e)}")
            raise
            
    def _close_connection(self):
        """关闭TCP连接"""
        if self.connection and hasattr(self.connection, 'close'):
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except:
                pass  # 忽略可能的错误
            try:
                self.connection.close()
            except Exception as e:
                logger.error(f"TCP关闭失败 - 错误: {str(e)}")
                
    def _send_data(self, data: Union[bytes, str, Dict]):
        """发送数据到TCP服务器"""
        if isinstance(data, dict):
            # 转换字典为JSON字符串
            data_str = json.dumps(data) + '\n'
            data_bytes = data_str.encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
            
        self.connection.sendall(data_bytes)
        
    def receive_data(self, timeout: float = None) -> Optional[Union[bytes, str]]:
        """接收TCP数据"""
        if timeout is not None:
            original_timeout = self.connection.gettimeout()
            self.connection.settimeout(timeout)
            
        try:
            data = self.connection.recv(self.config['buffer_size'])
            if data:
                return data
            return None
            
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"TCP接收数据出错 - 错误: {str(e)}")
            raise
        finally:
            if timeout is not None:
                self.connection.settimeout(original_timeout)
                
    def _receive_loop(self):
        """TCP接收循环"""
        buffer = b''
        
        while not self.stop_event.is_set():
            try:
                if self.status != DeviceStatus.CONNECTED:
                    time.sleep(0.1)
                    continue
                    
                # 读取数据
                data = self.receive_data(timeout=0.1)
                
                if data:
                    buffer += data
                    
                    # 尝试解析完整的数据包（假设以换行符分隔）
                    while b'\n' in buffer:
                        packet, buffer = buffer.split(b'\n', 1)
                        
                        try:
                            # 尝试解析为JSON
                            packet_str = packet.decode('utf-8')
                            packet_data = json.loads(packet_str)
                        except:
                            # 如果不是有效的JSON，返回原始字符串
                            packet_data = packet_str
                            
                        # 通知数据接收
                        self._notify_data_received(packet_data)
                        
            except (socket.error, ConnectionResetError) as e:
                logger.error(f"TCP连接错误 - ID: {self.device_id}, 错误: {str(e)}")
                
                # 尝试重连
                if self.reconnect_attempts < self.config['max_reconnect_attempts']:
                    self.reconnect_attempts += 1
                    logger.info(f"尝试重连... ({self.reconnect_attempts}/{self.config['max_reconnect_attempts']})")
                    
                    self.status = DeviceStatus.DISCONNECTED
                    self._notify_status_change()
                    
                    time.sleep(self.config['reconnect_interval'])
                    self.connect()
                    
                else:
                    logger.error(f"重连失败，已达到最大重连次数 - ID: {self.device_id}")
                    self.status = DeviceStatus.ERROR
                    self.error_message = str(e)
                    self._notify_status_change()
                    break
                    
            except Exception as e:
                logger.error(f"TCP接收循环出错 - ID: {self.device_id}, 错误: {str(e)}")
                self.status = DeviceStatus.ERROR
                self.error_message = str(e)
                self._notify_status_change()
                break
                
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.01)

class DeviceInterfaceManager:
    """\设备接口管理器，管理所有设备接口"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceInterfaceManager, cls).__new__(cls)
            cls._instance.devices = {}
            cls._instance.lock = threading.Lock()
            cls._instance.global_data_callback = None
            cls._instance.global_status_callback = None
        return cls._instance
        
    def create_device_interface(self, device_id: str, protocol: Union[str, DeviceProtocol], config: Dict[str, Any]) -> DeviceInterface:
        """
        创建设备接口
        
        参数:
            device_id: 设备ID
            protocol: 通讯协议
            config: 配置参数
        
        返回:
            设备接口实例
        """
        with self.lock:
            if device_id in self.devices:
                logger.warning(f"设备ID已存在，将返回已有设备接口 - ID: {device_id}")
                return self.devices[device_id]
                
            # 转换协议为枚举类型
            if isinstance(protocol, str):
                try:
                    protocol_enum = DeviceProtocol(protocol.lower())
                except ValueError:
                    raise DeviceInterfaceException(f"不支持的协议: {protocol}")
            else:
                protocol_enum = protocol
                
            # 根据协议创建对应的设备接口
            if protocol_enum == DeviceProtocol.SERIAL:
                device_interface = SerialDeviceInterface(device_id, config)
            elif protocol_enum == DeviceProtocol.TCP:
                device_interface = TCPDeviceInterface(device_id, config)
            else:
                # 其他协议可以在这里扩展
                raise DeviceInterfaceException(f"协议尚未实现: {protocol_enum.value}")
                
            # 设置全局回调
            if self.global_data_callback:
                device_interface.set_data_received_callback(self.global_data_callback)
            if self.global_status_callback:
                device_interface.set_status_changed_callback(self.global_status_callback)
                
            # 保存设备接口
            self.devices[device_id] = device_interface
            
            logger.info(f"设备接口创建成功 - ID: {device_id}, 协议: {protocol_enum.value}")
            return device_interface
            
    def get_device_interface(self, device_id: str) -> Optional[DeviceInterface]:
        """获取设备接口"""
        with self.lock:
            return self.devices.get(device_id)
            
    def connect_device(self, device_id: str) -> bool:
        """连接设备"""
        device_interface = self.get_device_interface(device_id)
        if device_interface:
            return device_interface.connect()
        logger.error(f"设备接口不存在 - ID: {device_id}")
        return False
        
    def disconnect_device(self, device_id: str) -> bool:
        """断开设备连接"""
        device_interface = self.get_device_interface(device_id)
        if device_interface:
            return device_interface.disconnect()
        logger.error(f"设备接口不存在 - ID: {device_id}")
        return False
        
    def send_data_to_device(self, device_id: str, data: Any) -> bool:
        """发送数据到设备"""
        device_interface = self.get_device_interface(device_id)
        if device_interface:
            return device_interface.send_data(data)
        logger.error(f"设备接口不存在 - ID: {device_id}")
        return False
        
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """获取设备状态"""
        device_interface = self.get_device_interface(device_id)
        if device_interface:
            return device_interface.get_status()
        logger.error(f"设备接口不存在 - ID: {device_id}")
        return None
        
    def get_all_devices_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有设备状态"""
        statuses = {}
        with self.lock:
            for device_id, device_interface in self.devices.items():
                statuses[device_id] = device_interface.get_status()
        return statuses
        
    def set_global_data_callback(self, callback: Callable[[str, Any], None]):
        """设置全局数据接收回调"""
        with self.lock:
            self.global_data_callback = callback
            for device_interface in self.devices.values():
                device_interface.set_data_received_callback(callback)
                
    def set_global_status_callback(self, callback: Callable[[str, Dict], None]):
        """设置全局状态变更回调"""
        with self.lock:
            self.global_status_callback = callback
            for device_interface in self.devices.values():
                device_interface.set_status_changed_callback(callback)
                
    def remove_device_interface(self, device_id: str) -> bool:
        """移除设备接口"""
        with self.lock:
            if device_id in self.devices:
                # 断开连接
                self.devices[device_id].disconnect()
                # 移除设备接口
                del self.devices[device_id]
                logger.info(f"设备接口已移除 - ID: {device_id}")
                return True
            logger.error(f"设备接口不存在 - ID: {device_id}")
            return False
            
# 创建全局实例
device_interface_manager = DeviceInterfaceManager()

# 系统兼容性工具
class SystemCompatibility:
    """系统兼容性工具类"""
    @staticmethod
    def get_current_os() -> str:
        """获取当前操作系统"""
        if sys.platform.startswith('win'):
            return 'windows'
        elif sys.platform.startswith('linux'):
            return 'linux'
        elif sys.platform.startswith('darwin'):
            return 'macos'
        else:
            return 'unknown'
            
    @staticmethod
    def execute_command(command: str, shell: bool = False) -> Tuple[bool, str]:
        """
        执行系统命令
        
        参数:
            command: 要执行的命令
            shell: 是否使用shell执行
        
        返回:
            (是否成功, 输出结果)
        """
        try:
            # 执行命令
            process = subprocess.Popen(
                command if shell else command.split(),
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 获取输出
            stdout, stderr = process.communicate(timeout=30)
            
            # 检查返回码
            if process.returncode == 0:
                return True, stdout
            else:
                return False, stderr
                
        except Exception as e:
            return False, str(e)
            
    @staticmethod
    def list_serial_ports() -> List[Dict[str, str]]:
        """列出可用的串口"""
        ports = []
        
        current_os = SystemCompatibility.get_current_os()
        
        if current_os == 'windows':
            # Windows系统
            import winreg
            
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'HARDWARE\\DEVICEMAP\\SERIALCOMM')
                i = 0
                while True:
                    try:
                        value_name, port, _ = winreg.EnumValue(key, i)
                        ports.append({'port': port, 'description': value_name})
                        i += 1
                    except OSError:
                        break
            except Exception as e:
                logger.error(f"列出Windows串口出错: {str(e)}")
                
        elif current_os == 'linux':
            # Linux系统
            import glob
            
            try:
                for port in glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'):
                    ports.append({'port': port, 'description': port})
            except Exception as e:
                logger.error(f"列出Linux串口出错: {str(e)}")
                
        elif current_os == 'macos':
            # macOS系统
            import glob
            
            try:
                for port in glob.glob('/dev/tty.*'):
                    ports.append({'port': port, 'description': port})
            except Exception as e:
                logger.error(f"列出macOS串口出错: {str(e)}")
                
        return ports