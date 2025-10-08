# -*- coding: utf-8 -*-
"""
传感器管理器模块
Sensor Management Module

负责管理各种传感器设备，提供传感器数据获取、处理和通信功能
"""

import time
import json
import logging
import threading
import serial
import socket
import paho.mqtt.client as mqtt
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import os
import yaml

# 设置日志 | Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SensorManager")

# 从配置文件加载传感器配置 | Load sensor configuration from config file
def load_sensor_config(config_path: str = "d:\\shiyan\\config\\sensor_config.json") -> Dict[str, Any]:
    """加载传感器配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功加载传感器配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载传感器配置文件失败: {str(e)}")
        # 返回默认配置
        return {
            "communication_protocols": {
                "serial": {"enabled": True, "default_baudrate": 9600},
                "tcp": {"enabled": True, "default_port": 8080},
                "udp": {"enabled": True, "default_port": 8081},
                "i2c": {"enabled": True, "default_address": 0x40},
                "spi": {"enabled": True, "max_speed": 1000000},
                "can": {"enabled": True, "bitrate": 500000},
                "modbus": {"enabled": True, "default_port": 502},
                "mqtt": {"enabled": True, "broker": "localhost", "port": 1883},
                "websocket": {"enabled": True, "default_port": 8765}
            },
            "sensor_configuration": {
                "temperature": {"unit": "°C", "range": [-40, 125], "precision": 0.1},
                "humidity": {"unit": "%", "range": [0, 100], "precision": 0.1},
                "acceleration": {"unit": "m/s²", "range": [-16, 16], "precision": 0.01},
                "gyroscope": {"unit": "rad/s", "range": [-2000, 2000], "precision": 0.1}
            }
        }

class SensorManager:
    """传感器管理器
    Sensor Manager
    
    管理各种传感器设备，提供统一的接口进行传感器数据获取和控制
    """
    def __init__(self, config: Dict[str, Any] = None):
        """初始化传感器管理器"""
        # 加载配置
        self.config = config or load_sensor_config()
        
        # 存储连接的传感器
        self.sensors = {}
        self.sensor_locks = {}
        self.active_sensors = set()
        self.global_lock = threading.Lock()
        
        # 通信协议客户端
        self.serial_connections = {}
        self.tcp_clients = {}
        self.udp_clients = {}
        self.mqtt_client = None
        
        # 模拟传感器数据生成器
        self.mock_sensor_data = {
            "temperature": lambda: np.random.uniform(20, 25),
            "humidity": lambda: np.random.uniform(40, 60),
            "acceleration": lambda: {
                "x": np.random.uniform(-0.5, 0.5),
                "y": np.random.uniform(-0.5, 0.5),
                "z": np.random.uniform(-0.5, 0.5)
            },
            "gyroscope": lambda: {
                "x": np.random.uniform(-1, 1),
                "y": np.random.uniform(-1, 1),
                "z": np.random.uniform(-1, 1)
            },
            "pressure": lambda: np.random.uniform(980, 1020),
            "light": lambda: np.random.uniform(100, 1000)
        }
        
        # 启动MQTT客户端（如果启用）
        if self.config.get("communication_protocols", {}).get("mqtt", {}).get("enabled", False):
            self._init_mqtt_client()
        
        logger.info("传感器管理器已初始化")
    
    def _init_mqtt_client(self):
        """初始化MQTT客户端"""
        try:
            mqtt_config = self.config["communication_protocols"]["mqtt"]
            self.mqtt_client = mqtt.Client()
            
            # 设置回调函数
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            
            # 连接到MQTT broker
            broker = mqtt_config.get("broker", "localhost")
            port = mqtt_config.get("port", 1883)
            self.mqtt_client.connect(broker, port, 60)
            
            # 开始后台线程处理MQTT消息
            self.mqtt_client.loop_start()
            logger.info(f"MQTT客户端已连接到 {broker}:{port}")
        except Exception as e:
            logger.error(f"初始化MQTT客户端失败: {str(e)}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT连接回调函数"""
        if rc == 0:
            logger.info("MQTT连接成功")
            # 订阅传感器数据主题
            client.subscribe("sensors/#")
        else:
            logger.error(f"MQTT连接失败，错误代码: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT消息接收回调函数"""
        try:
            # 解析传感器数据
            sensor_id = msg.topic.split("/")[1]
            data = json.loads(msg.payload.decode('utf-8'))
            
            # 存储接收到的数据
            with self.global_lock:
                if sensor_id in self.sensors:
                    self.sensors[sensor_id]["last_data"] = data
                    self.sensors[sensor_id]["last_update"] = datetime.now().isoformat()
            
            logger.debug(f"从MQTT接收到传感器数据: {sensor_id} -> {data}")
        except Exception as e:
            logger.error(f"处理MQTT消息失败: {str(e)}")
    
    def connect_sensor(self, sensor_id: str, sensor_type: str, protocol: str,
                      connection_params: Dict[str, Any]) -> bool:
        """连接传感器设备
        Connect to a sensor device
        
        参数:
            sensor_id: 传感器ID
            sensor_type: 传感器类型
            protocol: 通信协议
            connection_params: 连接参数
        
        返回:
            连接成功返回True，否则返回False
        """
        with self.global_lock:
            if sensor_id in self.active_sensors:
                logger.warning(f"传感器 {sensor_id} 已经连接")
                return True
            
            try:
                # 创建传感器锁
                if sensor_id not in self.sensor_locks:
                    self.sensor_locks[sensor_id] = threading.Lock()
                
                with self.sensor_locks[sensor_id]:
                    connection = None
                    
                    # 根据协议类型连接传感器
                    if protocol == "serial" and self.config["communication_protocols"]["serial"]["enabled"]:
                        connection = self._connect_serial_sensor(connection_params)
                    elif protocol == "tcp" and self.config["communication_protocols"]["tcp"]["enabled"]:
                        connection = self._connect_tcp_sensor(connection_params)
                    elif protocol == "udp" and self.config["communication_protocols"]["udp"]["enabled"]:
                        connection = self._connect_udp_sensor(connection_params)
                    elif protocol == "mqtt" and self.config["communication_protocols"]["mqtt"]["enabled"]:
                        connection = "mqtt"  # MQTT连接已在初始化时建立
                    elif protocol == "mock":
                        connection = "mock"  # 模拟传感器
                    else:
                        logger.error(f"不支持的通信协议: {protocol} 或协议未启用")
                        return False
                    
                    if connection:
                        # 存储传感器信息
                        self.sensors[sensor_id] = {
                            "type": sensor_type,
                            "protocol": protocol,
                            "connection": connection,
                            "connection_params": connection_params,
                            "last_data": None,
                            "last_update": None,
                            "is_connected": True,
                            "error": None,
                            "sampling_rate": connection_params.get("sampling_rate", 1),
                            "start_time": datetime.now().isoformat()
                        }
                        
                        # 添加到活动传感器集合
                        self.active_sensors.add(sensor_id)
                        
                        logger.info(f"传感器 {sensor_id} (类型: {sensor_type}) 已成功连接")
                        return True
                    else:
                        logger.error(f"连接传感器 {sensor_id} 失败")
                        return False
            except Exception as e:
                logger.error(f"连接传感器 {sensor_id} 时发生错误: {str(e)}")
                return False
    
    def _connect_serial_sensor(self, params: Dict[str, Any]) -> Optional[serial.Serial]:
        """连接串口传感器"""
        try:
            port = params.get("port", "COM1")
            baudrate = params.get("baudrate", 9600)
            timeout = params.get("timeout", 1)
            
            ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout
            )
            
            if ser.is_open:
                self.serial_connections[port] = ser
                logger.info(f"成功打开串口: {port}，波特率: {baudrate}")
                return ser
            else:
                logger.error(f"无法打开串口: {port}")
                return None
        except Exception as e:
            logger.error(f"串口连接失败: {str(e)}")
            return None
    
    def _connect_tcp_sensor(self, params: Dict[str, Any]) -> Optional[socket.socket]:
        """连接TCP传感器"""
        try:
            host = params.get("host", "localhost")
            port = params.get("port", 8080)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            
            conn_id = f"{host}:{port}"
            self.tcp_clients[conn_id] = sock
            logger.info(f"成功连接到TCP传感器: {host}:{port}")
            return sock
        except Exception as e:
            logger.error(f"TCP连接失败: {str(e)}")
            return None
    
    def _connect_udp_sensor(self, params: Dict[str, Any]) -> Optional[socket.socket]:
        """连接UDP传感器"""
        try:
            host = params.get("host", "localhost")
            port = params.get("port", 8081)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # UDP是无连接的，所以这里只创建套接字，不进行连接
            
            conn_id = f"{host}:{port}"
            self.udp_clients[conn_id] = (sock, (host, port))
            logger.info(f"创建UDP套接字用于与传感器通信: {host}:{port}")
            return sock
        except Exception as e:
            logger.error(f"UDP套接字创建失败: {str(e)}")
            return None
    
    def disconnect_sensor(self, sensor_id: str) -> bool:
        """断开传感器连接
        Disconnect from a sensor
        
        参数:
            sensor_id: 传感器ID
        
        返回:
            断开成功返回True，否则返回False
        """
        with self.global_lock:
            if sensor_id not in self.active_sensors:
                logger.warning(f"传感器 {sensor_id} 未连接")
                return True
            
            try:
                if sensor_id in self.sensor_locks:
                    with self.sensor_locks[sensor_id]:
                        if sensor_id in self.sensors:
                            sensor = self.sensors[sensor_id]
                            
                            # 关闭连接
                            if sensor["protocol"] == "serial" and isinstance(sensor["connection"], serial.Serial):
                                if sensor["connection"].is_open:
                                    sensor["connection"].close()
                                    port = sensor["connection_params"].get("port", "")
                                    if port in self.serial_connections:
                                        del self.serial_connections[port]
                            elif sensor["protocol"] == "tcp" and isinstance(sensor["connection"], socket.socket):
                                sensor["connection"].close()
                                host = sensor["connection_params"].get("host", "localhost")
                                port = sensor["connection_params"].get("port", 8080)
                                conn_id = f"{host}:{port}"
                                if conn_id in self.tcp_clients:
                                    del self.tcp_clients[conn_id]
                            elif sensor["protocol"] == "udp" and isinstance(sensor["connection"], socket.socket):
                                sensor["connection"].close()
                                host = sensor["connection_params"].get("host", "localhost")
                                port = sensor["connection_params"].get("port", 8081)
                                conn_id = f"{host}:{port}"
                                if conn_id in self.udp_clients:
                                    del self.udp_clients[conn_id]
                            
                            # 清理资源
                            del self.sensors[sensor_id]
                        
                    # 移除锁
                    del self.sensor_locks[sensor_id]
                
                # 从活动传感器集合中移除
                self.active_sensors.remove(sensor_id)
                
                logger.info(f"传感器 {sensor_id} 已断开连接")
                return True
            except Exception as e:
                logger.error(f"断开传感器 {sensor_id} 连接时发生错误: {str(e)}")
                return False
    
    def get_sensor_data(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """获取传感器数据
        Get data from a sensor
        
        参数:
            sensor_id: 传感器ID
        
        返回:
            传感器数据字典，失败返回None
        """
        if sensor_id not in self.active_sensors:
            logger.warning(f"传感器 {sensor_id} 未连接，无法获取数据")
            return None
        
        try:
            if sensor_id in self.sensor_locks:
                with self.sensor_locks[sensor_id]:
                    if sensor_id in self.sensors and self.sensors[sensor_id]["is_connected"]:
                        sensor = self.sensors[sensor_id]
                        
                        # 根据协议类型获取数据
                        if sensor["protocol"] == "serial":
                            data = self._read_serial_sensor(sensor)
                        elif sensor["protocol"] == "tcp":
                            data = self._read_tcp_sensor(sensor)
                        elif sensor["protocol"] == "udp":
                            data = self._read_udp_sensor(sensor)
                        elif sensor["protocol"] == "mqtt":
                            data = sensor.get("last_data", None)
                        elif sensor["protocol"] == "mock":
                            data = self._generate_mock_sensor_data(sensor["type"])
                        else:
                            logger.error(f"不支持的协议类型: {sensor['protocol']}")
                            data = None
                        
                        # 更新传感器数据
                        if data:
                            sensor["last_data"] = data
                            sensor["last_update"] = datetime.now().isoformat()
                            sensor["error"] = None
                        
                        return {
                            "sensor_id": sensor_id,
                            "type": sensor["type"],
                            "data": data,
                            "timestamp": sensor["last_update"],
                            "error": sensor["error"]
                        }
            
            return None
        except Exception as e:
            logger.error(f"获取传感器 {sensor_id} 数据时发生错误: {str(e)}")
            if sensor_id in self.sensors:
                self.sensors[sensor_id]["error"] = str(e)
            return None
    
    def _read_serial_sensor(self, sensor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """读取串口传感器数据"""
        try:
            ser = sensor["connection"]
            if ser and ser.is_open:
                # 尝试读取一行数据
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    # 假设数据格式为JSON
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        # 如果不是JSON格式，尝试解析为简单的键值对
                        data = {}
                        parts = line.split(',')
                        for part in parts:
                            if ':' in part:
                                key, value = part.split(':', 1)
                                try:
                                    data[key.strip()] = float(value.strip())
                                except ValueError:
                                    data[key.strip()] = value.strip()
                        return data if data else {"raw_data": line}
            return None
        except Exception as e:
            logger.error(f"读取串口传感器数据失败: {str(e)}")
            return None
    
    def _read_tcp_sensor(self, sensor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """读取TCP传感器数据"""
        try:
            sock = sensor["connection"]
            if sock:
                # 尝试接收数据
                data = sock.recv(1024).decode('utf-8', errors='ignore').strip()
                if data:
                    # 假设数据格式为JSON
                    try:
                        return json.loads(data)
                    except json.JSONDecodeError:
                        return {"raw_data": data}
            return None
        except Exception as e:
            logger.error(f"读取TCP传感器数据失败: {str(e)}")
            return None
    
    def _read_udp_sensor(self, sensor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """读取UDP传感器数据"""
        try:
            sock = sensor["connection"]
            host = sensor["connection_params"].get("host", "localhost")
            port = sensor["connection_params"].get("port", 8081)
            
            if sock:
                # 设置超时
                sock.settimeout(0.5)
                try:
                    # 尝试接收数据
                    data, addr = sock.recvfrom(1024)
                    if addr == (host, port):
                        data_str = data.decode('utf-8', errors='ignore').strip()
                        if data_str:
                            # 假设数据格式为JSON
                            try:
                                return json.loads(data_str)
                            except json.JSONDecodeError:
                                return {"raw_data": data_str}
                except socket.timeout:
                    # 超时，返回最后一次数据
                    return sensor.get("last_data", None)
                finally:
                    # 重置超时
                    sock.settimeout(None)
            return None
        except Exception as e:
            logger.error(f"读取UDP传感器数据失败: {str(e)}")
            return None
    
    def _generate_mock_sensor_data(self, sensor_type: str) -> Dict[str, Any]:
        """生成模拟传感器数据"""
        if sensor_type in self.mock_sensor_data:
            try:
                data = self.mock_sensor_data[sensor_type]()
                return {"value": data, "is_mock": True}
            except Exception as e:
                logger.error(f"生成模拟数据失败: {str(e)}")
                return {"value": 0, "is_mock": True, "error": str(e)}
        else:
            # 对于未知类型的传感器，生成随机数据
            return {
                "value": np.random.random(),
                "is_mock": True,
                "sensor_type": sensor_type
            }
    
    def send_command_to_sensor(self, sensor_id: str, command: Dict[str, Any]) -> bool:
        """向传感器发送命令
        Send command to a sensor
        
        参数:
            sensor_id: 传感器ID
            command: 命令字典
        
        返回:
            发送成功返回True，否则返回False
        """
        if sensor_id not in self.active_sensors:
            logger.warning(f"传感器 {sensor_id} 未连接，无法发送命令")
            return False
        
        try:
            if sensor_id in self.sensor_locks:
                with self.sensor_locks[sensor_id]:
                    if sensor_id in self.sensors and self.sensors[sensor_id]["is_connected"]:
                        sensor = self.sensors[sensor_id]
                        command_str = json.dumps(command) + "\n"
                        
                        # 根据协议类型发送命令
                        if sensor["protocol"] == "serial" and isinstance(sensor["connection"], serial.Serial):
                            if sensor["connection"].is_open:
                                sensor["connection"].write(command_str.encode('utf-8'))
                                logger.debug(f"向串口传感器 {sensor_id} 发送命令: {command_str}")
                                return True
                        elif sensor["protocol"] == "tcp" and isinstance(sensor["connection"], socket.socket):
                            sensor["connection"].sendall(command_str.encode('utf-8'))
                            logger.debug(f"向TCP传感器 {sensor_id} 发送命令: {command_str}")
                            return True
                        elif sensor["protocol"] == "udp" and isinstance(sensor["connection"], socket.socket):
                            host = sensor["connection_params"].get("host", "localhost")
                            port = sensor["connection_params"].get("port", 8081)
                            sensor["connection"].sendto(command_str.encode('utf-8'), (host, port))
                            logger.debug(f"向UDP传感器 {sensor_id} 发送命令: {command_str}")
                            return True
                        elif sensor["protocol"] == "mqtt" and self.mqtt_client:
                            topic = f"sensors/{sensor_id}/command"
                            self.mqtt_client.publish(topic, command_str)
                            logger.debug(f"通过MQTT向传感器 {sensor_id} 发送命令: {command_str}")
                            return True
                        elif sensor["protocol"] == "mock":
                            # 模拟传感器只记录命令
                            logger.debug(f"向模拟传感器 {sensor_id} 发送命令: {command_str}")
                            return True
            
            logger.error(f"无法向传感器 {sensor_id} 发送命令")
            return False
        except Exception as e:
            logger.error(f"向传感器 {sensor_id} 发送命令时发生错误: {str(e)}")
            return False
    
    def get_sensor_status(self, sensor_id: str) -> Dict[str, Any]:
        """获取传感器状态
        Get sensor status
        
        参数:
            sensor_id: 传感器ID
        
        返回:
            传感器状态字典
        """
        status = {
            "sensor_id": sensor_id,
            "is_connected": sensor_id in self.active_sensors,
            "type": None,
            "protocol": None,
            "last_update": None,
            "error": None,
            "has_data": False,
            "connection_params": None
        }
        
        if sensor_id in self.active_sensors and sensor_id in self.sensors:
            sensor = self.sensors[sensor_id]
            status["type"] = sensor["type"]
            status["protocol"] = sensor["protocol"]
            status["last_update"] = sensor["last_update"]
            status["error"] = sensor["error"]
            status["has_data"] = sensor["last_data"] is not None
            status["connection_params"] = sensor["connection_params"]
        
        return status
    
    def get_active_sensors(self) -> List[Dict[str, Any]]:
        """获取所有活动传感器列表
        Get list of all active sensors
        
        返回:
            活动传感器列表
        """
        active_sensors_list = []
        with self.global_lock:
            for sensor_id in self.active_sensors:
                if sensor_id in self.sensors:
                    sensor = self.sensors[sensor_id]
                    active_sensors_list.append({
                        "sensor_id": sensor_id,
                        "type": sensor["type"],
                        "protocol": sensor["protocol"],
                        "last_update": sensor["last_update"],
                        "has_error": sensor["error"] is not None
                    })
        
        return active_sensors_list
    
    def disconnect_all_sensors(self) -> bool:
        """断开所有传感器连接
        Disconnect all sensors
        
        返回:
            全部断开成功返回True，部分失败返回False
        """
        all_success = True
        active_sensor_ids = list(self.active_sensors)
        
        for sensor_id in active_sensor_ids:
            if not self.disconnect_sensor(sensor_id):
                all_success = False
        
        logger.info(f"尝试断开所有传感器连接 ({len(active_sensor_ids)} 个)")
        return all_success

# 创建全局传感器管理器实例
global_sensor_manager = SensorManager()

# 工具函数
def get_sensor_manager() -> SensorManager:
    """获取全局传感器管理器实例"""
    return global_sensor_manager

if __name__ == "__main__":
    # 测试传感器管理器
    print("测试传感器管理器...")
    
    # 创建管理器实例
    manager = SensorManager()
    
    # 连接模拟温度传感器
    print("\n连接模拟温度传感器...")
    success = manager.connect_sensor(
        sensor_id="temp_sensor_1",
        sensor_type="temperature",
        protocol="mock",
        connection_params={"sampling_rate": 1}
    )
    
    if success:
        print("传感器连接成功!")
        
        # 获取传感器数据
        print("\n获取传感器数据...")
        for i in range(3):
            data = manager.get_sensor_data("temp_sensor_1")
            if data:
                print(f"传感器数据 #{i+1}: {data}")
            time.sleep(1)
        
        # 获取传感器状态
        print("\n获取传感器状态...")
        status = manager.get_sensor_status("temp_sensor_1")
        print(f"传感器状态: {status}")
        
        # 获取所有活动传感器
        print("\n获取所有活动传感器...")
        active_sensors = manager.get_active_sensors()
        print(f"活动传感器: {active_sensors}")
        
        # 断开传感器连接
        print("\n断开传感器连接...")
        manager.disconnect_sensor("temp_sensor_1")
    else:
        print("传感器连接失败")
    
    print("\n传感器管理器测试完成")