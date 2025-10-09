#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
传感器集成管理模块
Sensor Integration Management Module
负责将各种传感器数据集成到主系统，并提供与外接设备的通讯接口
"""

import os
import sys
import json
import time
import threading
import logging
import redis
import numpy as np
import cv2
import pyaudio
import serial
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/sensor_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置加载器
try:
    from config.config_loader import get_config, get_model_port
    logger.info("成功导入配置加载器")
except ImportError:
    logger.error("无法导入配置加载器，使用默认配置")
    
    # 定义备用配置函数
    def get_config(key, default=None):
        config = {
            'ports.sensor': 5006,
            'models.local_models.G_sensor': True,
            'data_bus.host': 'localhost',
            'data_bus.port': 6379,
        }
        keys = key.split('.')
        value = config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_port(model_name):
        return get_config(f'ports.{model_name.lower()}')

# 导入必要的库
import queue
from collections import deque

# 传感器集成类
class SensorIntegrationManager:
    """传感器集成管理器 - 负责外部设备通信和数据集成"""
    
    def __init__(self, config_path=None):
        """初始化传感器集成管理器"""
        
        # 配置路径
        self.config_path = config_path or "config/sensor_config.json"
        
        # 加载配置
        self.config = self.load_config(self.config_path)
        
        # 数据总线连接
        self.data_bus = None
        self.connect_to_main_model()
        
        # 实时接口
        self.camera_stream = None
        self.microphone_stream = None
        self.audio = None
        self.network_streams = {}
        self.serial_ports = {}
        
        # 实时处理控制
        self.realtime_active = False
        self.realtime_thread = None
        self.realtime_lock = threading.RLock()  # 线程安全锁
        
        # 外部API客户端
        self.external_api_clients = {}
        self.initialize_external_apis()
        
        # 传感器数据队列
        self.sensor_data_queue = queue.Queue(maxsize=1000)
        
        # 传感器类型映射
        self.sensor_types = {
            "temperature": {"unit": "°C", "range": [-40, 125]},
            "humidity": {"unit": "%", "range": [0, 100]},
            "acceleration": {"unit": "m/s²", "range": [-16, 16]},
            "velocity": {"unit": "m/s", "range": [0, 100]},
            "displacement": {"unit": "m", "range": [0, 100]},
            "gyroscope": {"unit": "rad/s", "range": [-2000, 2000]},
            "pressure": {"unit": "hPa", "range": [300, 1100]},
            "barometric": {"unit": "hPa", "range": [300, 1100]},
            "distance": {"unit": "m", "range": [0, 100]},
            "infrared": {"unit": "lux", "range": [0, 10000]},
            "gas": {"unit": "ppm", "range": [0, 5000]},
            "smoke": {"unit": "ppm", "range": [0, 1000]},
            "light": {"unit": "lux", "range": [0, 100000]},
            "taste": {"unit": "intensity", "range": [0, 10]}
        }
        
        # 数据历史记录
        self.data_history = deque(maxlen=1000)  # 保存历史数据
        
        # 异常检测阈值
        self.anomaly_thresholds = {
            "temperature": 2.0,  # 温度变化阈值
            "humidity": 10.0,    # 湿度变化阈值
            "acceleration": 5.0, # 加速度变化阈值
            "default": 3.0       # 默认阈值
        }
        
        logger.info("传感器集成管理器初始化完成")
    
    def load_config(self, config_path):
        """加载配置文件"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            logger.warning(f"配置文件不存在: {config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
        
        # 默认配置
        return {
            "data_bus_host": "localhost",
            "data_bus_port": 6379,
            "realtime_interfaces": {
                "camera": {"enabled": True, "device_index": 0},
                "microphone": {"enabled": True, "device_index": 0},
                "network_streams": {"enabled": True},
                "serial_ports": {"enabled": True}
            },
            "external_api": {
                "enabled": False,
                "apis": []
            }
        }
    
    def start_realtime_processing(self):
        """启动实时数据处理"""
        with self.realtime_lock:
            if self.realtime_active:
                return {'status': 'error', 'message': 'Real-time processing already running'}
            
            self.realtime_active = True
            self.realtime_thread = threading.Thread(target=self._realtime_processing_loop)
            self.realtime_thread.daemon = True
            self.realtime_thread.start()
            
            logger.info("实时数据处理已启动")
            return {'status': 'success', 'message': 'Real-time processing started'}
    
    def stop_realtime_processing(self):
        """停止实时处理"""
        with self.realtime_lock:
            if self.realtime_active:
                self.realtime_active = False
                if self.realtime_thread and self.realtime_thread.is_alive():
                    self.realtime_thread.join(timeout=5.0)  # 等待线程结束，最多等待5秒
                logger.info("实时处理已停止")
            else:
                logger.warning("实时处理未启动")
    
    def process_sensor_data(self, data):
        """处理传感器数据"""
        try:
            # 提取传感器类型和值
            sensor_type = data.get('type')
            value = data.get('value')
            timestamp = data.get('timestamp', time.time())
            
            if sensor_type is None or value is None:
                logger.warning("无效的传感器数据: 缺少类型或值")
                return None
            
            # 验证传感器类型
            if sensor_type not in self.sensor_types:
                logger.warning(f"未知的传感器类型: {sensor_type}")
                return None
            
            # 归一化数据
            normalized_value = self.normalize_sensor_data(sensor_type, value)
            
            # 创建处理后的数据结构
            processed_data = {
                'type': sensor_type,
                'value': value,
                'normalized_value': normalized_value,
                'timestamp': timestamp,
                'unit': self.sensor_types[sensor_type]['unit']
            }
            
            # 保存到历史记录
            self.data_history.append(processed_data)
            
            # 解释输出
            interpretation = self.interpret_output(sensor_type, normalized_value)
            processed_data['interpretation'] = interpretation
            
            # 检测异常
            anomaly = self.detect_anomalies(sensor_type, value)
            if anomaly:
                processed_data['anomaly'] = anomaly
                logger.warning(f"检测到异常: {anomaly}")
            
            return processed_data
        except Exception as e:
            logger.error(f"处理传感器数据时出错: {str(e)}")
            return None
    
    def normalize_sensor_data(self, sensor_type, value):
        """归一化传感器数据"""
        try:
            if sensor_type not in self.sensor_types:
                return value
            
            sensor_config = self.sensor_types[sensor_type]
            min_val, max_val = sensor_config['range']
            
            # 处理边界情况
            if max_val == min_val:
                return 0.5
            
            # 归一化到[0, 1]范围
            normalized = (value - min_val) / (max_val - min_val)
            # 确保值在[0, 1]范围内
            return max(0.0, min(1.0, normalized))
        except Exception as e:
            logger.error(f"归一化传感器数据时出错: {str(e)}")
            return value
    
    def interpret_output(self, sensor_type, normalized_value):
        """解释模型输出"""
        try:
            # 根据归一化值生成解释
            if normalized_value < 0.2:
                level = "低"
            elif normalized_value < 0.4:
                level = "较低"
            elif normalized_value < 0.6:
                level = "中等"
            elif normalized_value < 0.8:
                level = "较高"
            else:
                level = "高"
            
            interpretations = {
                "temperature": f"当前温度{level}",
                "humidity": f"当前湿度{level}",
                "acceleration": f"当前加速度{level}",
                "velocity": f"当前速度{level}",
                "displacement": f"当前位移{level}",
                "gyroscope": f"当前角速度{level}",
                "pressure": f"当前压力{level}",
                "barometric": f"当前气压{level}",
                "distance": f"当前距离{level}",
                "infrared": f"当前红外强度{level}",
                "gas": f"当前气体浓度{level}",
                "smoke": f"当前烟雾浓度{level}",
                "light": f"当前光照强度{level}",
                "taste": f"当前味道强度{level}"
            }
            
            return interpretations.get(sensor_type, f"传感器读数{level}")
        except Exception as e:
            logger.error(f"解释模型输出时出错: {str(e)}")
            return "无法解释"
    
    def detect_anomalies(self, sensor_type, value):
        """检测传感器异常"""
        try:
            # 如果历史数据不足，不进行异常检测
            if len(self.data_history) < 10:
                return None
            
            # 获取历史数据中相同类型的传感器值
            historical_values = [d['value'] for d in self.data_history if d['type'] == sensor_type][-10:]
            
            if not historical_values:
                return None
            
            # 计算历史数据的均值和标准差
            mean = sum(historical_values) / len(historical_values)
            if len(historical_values) > 1:
                std = (sum((x - mean) ** 2 for x in historical_values) / (len(historical_values) - 1)) ** 0.5
            else:
                std = 0
            
            # 获取阈值
            threshold = self.anomaly_thresholds.get(sensor_type, self.anomaly_thresholds['default'])
            
            # 检测异常
            if abs(value - mean) > threshold * (std if std > 0 else 1):
                return {
                    'type': 'anomaly',
                    'sensor_type': sensor_type,
                    'current_value': value,
                    'mean_value': mean,
                    'deviation': abs(value - mean),
                    'message': f"{sensor_type}值偏离正常值过多: 当前值={value}, 平均值={mean:.2f}"
                }
            
            return None
        except Exception as e:
            logger.error(f"检测异常时出错: {str(e)}")
            return None
    
    def fuse_sensor_data(self, data_list):
        """多传感器数据融合"""
        try:
            if not data_list:
                return None
            
            # 按类型分组数据
            data_by_type = {}
            for data in data_list:
                sensor_type = data.get('type')
                if sensor_type not in data_by_type:
                    data_by_type[sensor_type] = []
                data_by_type[sensor_type].append(data)
            
            # 对每种类型的数据进行融合
            fused_data = {
                'timestamp': time.time(),
                'sensors': {}
            }
            
            for sensor_type, type_data in data_by_type.items():
                # 计算平均值作为融合结果
                values = [d.get('value', 0) for d in type_data if 'value' in d]
                if values:
                    fused_value = sum(values) / len(values)
                    fused_data['sensors'][sensor_type] = {
                        'value': fused_value,
                        'count': len(values),
                        'unit': self.sensor_types.get(sensor_type, {}).get('unit', '')
                    }
            
            return fused_data
        except Exception as e:
            logger.error(f"融合传感器数据时出错: {str(e)}")
            return None
    
    def send_to_main_model(self, data_type, data):
        """发送数据到主模型"""
        try:
            if not self.data_bus:
                logger.warning("数据总线未连接，无法发送数据到主模型")
                return False
            
            # 数据类型映射
            data_channels = {
                'sensor': 'sensor_data',
                'camera': 'camera_frames',
                'audio': 'audio_data',
                'serial': 'serial_data',
                'network': 'network_data'
            }
            
            channel = data_channels.get(data_type, 'other_data')
            message = {
                'type': data_type,
                'data': data,
                'timestamp': time.time(),
                'source': 'G_sensor_module'
            }
            
            # 转换为JSON字符串
            message_json = json.dumps(message)
            
            # 发送到数据总线
            self.data_bus.publish(channel, message_json)
            logger.debug(f"成功发送{data_type}数据到主模型通道: {channel}")
            return True
        except Exception as e:
            logger.error(f"发送数据到主模型时出错: {str(e)}")
            return False
    
    def analyze_frame(self, frame):
        """分析视频帧"""
        try:
            # 这里可以添加更复杂的帧分析逻辑
            # 例如：物体检测、人脸识别等
            
            # 提取帧的基本信息
            height, width, channels = frame.shape
            
            # 计算帧的平均亮度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean() / 255.0
            
            # 检测是否有运动（简单的背景差分）
            motion_detected = False
            if hasattr(self, 'prev_frame'):
                diff = cv2.absdiff(self.prev_frame, gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                motion_pixels = cv2.countNonZero(thresh)
                if motion_pixels > (width * height * 0.01):  # 如果超过1%的像素变化，则认为有运动
                    motion_detected = True
            self.prev_frame = gray.copy()
            
            analysis_result = {
                'dimensions': {'width': width, 'height': height, 'channels': channels},
                'brightness': brightness,
                'motion_detected': motion_detected,
                'timestamp': time.time()
            }
            
            return analysis_result
        except Exception as e:
            logger.error(f"分析视频帧时出错: {str(e)}")
            return None
    
    def analyze_audio(self, audio_data):
        """分析音频数据"""
        try:
            # 计算音频数据的基本特征
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # 计算音频的能量
            energy = np.sum(audio_np ** 2) / len(audio_np)
            
            # 计算音频的振幅
            amplitude = np.max(np.abs(audio_np)) / 32768.0  # 归一化到[0, 1]
            
            # 检测是否有声音
            is_silent = amplitude < 0.01
            
            analysis_result = {
                'energy': energy,
                'amplitude': amplitude,
                'is_silent': is_silent,
                'length': len(audio_data),
                'timestamp': time.time()
            }
            
            return analysis_result
        except Exception as e:
            logger.error(f"分析音频数据时出错: {str(e)}")
            return None
    
    def parse_serial_data(self, serial_data):
        """解析串口数据"""
        try:
            # 尝试将串口数据解析为JSON格式
            try:
                data = json.loads(serial_data)
                return data
            except json.JSONDecodeError:
                # 如果不是JSON格式，尝试其他解析方式
                pass
            
            # 尝试解析为CSV格式
            if ',' in serial_data:
                parts = serial_data.strip().split(',')
                if len(parts) >= 2:
                    return {
                        'type': parts[0].strip(),
                        'value': float(parts[1].strip()) if parts[1].strip().replace('.', '', 1).isdigit() else parts[1].strip()
                    }
            
            # 返回原始数据
            return {
                'type': 'raw',
                'value': serial_data.strip()
            }
        except Exception as e:
            logger.error(f"解析串口数据时出错: {str(e)}")
            return {
                'type': 'error',
                'value': str(e)
            }
    
    def _realtime_processing_loop(self):
        """实时处理循环 - 从各种接口收集数据并发送到数据总线"""
        logger.info("开始实时数据处理")
        
        # 收集的数据列表，用于数据融合
        collected_data = []
        collection_interval = 0.5  # 数据融合间隔（秒）
        last_fusion_time = time.time()
        
        while self.realtime_active:
            try:
                current_time = time.time()
                
                # 处理传感器数据队列
                while not self.sensor_data_queue.empty():
                    try:
                        sensor_data = self.sensor_data_queue.get_nowait()
                        
                        # 处理传感器数据
                        processed_data = self.process_sensor_data(sensor_data)
                        if processed_data:
                            # 发送处理后的数据到数据总线
                            self.send_sensor_data_to_bus(processed_data)
                            
                            # 添加到融合数据列表
                            collected_data.append(processed_data)
                            
                            # 发送到主模型
                            self.send_to_main_model('sensor', processed_data)
                        
                        self.sensor_data_queue.task_done()
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"处理传感器数据队列时出错: {str(e)}")
                
                # 处理摄像头数据
                if self.camera_stream and self.camera_stream.isOpened():
                    frame_data = self.capture_frame()
                    if frame_data['status'] == 'success':
                        frame = frame_data['frame']
                        # 分析视频帧
                        frame_analysis = self.analyze_frame(frame)
                        
                        # 发送摄像头数据到数据总线
                        self.send_camera_data_to_bus(frame_data)
                        
                        # 发送分析结果到主模型
                        self.send_to_main_model('camera', frame_analysis)
                
                # 处理麦克风数据
                if self.microphone_stream:
                    audio_data = self.read_audio_data()
                    if audio_data['status'] == 'success':
                        # 分析音频数据
                        audio_analysis = self.analyze_audio(audio_data['audio_data'])
                        
                        # 发送音频数据到数据总线
                        self.send_audio_data_to_bus(audio_data)
                        
                        # 发送分析结果到主模型
                        self.send_to_main_model('audio', audio_analysis)
                
                # 处理网络流数据
                for url, stream in list(self.network_streams.items()):
                    if stream['type'] == 'video' and 'capture' in stream:
                        try:
                            ret, frame = stream['capture'].read()
                            if ret:
                                # 分析视频帧
                                frame_analysis = self.analyze_frame(frame)
                                
                                # 发送网络视频数据到数据总线
                                self.send_network_video_data_to_bus(url, frame)
                                
                                # 发送分析结果到主模型
                                self.send_to_main_model('network', {'stream_id': url, 'type': 'video', 'analysis': frame_analysis})
                        except Exception as e:
                            logger.error(f"处理网络流{url}时出错: {str(e)}")
                
                # 处理串口数据
                for port in list(self.serial_ports.keys()):
                    serial_data = self.read_serial_data(port)
                    if serial_data['status'] == 'success' and serial_data['data']:
                        # 解析串口数据
                        parsed_data = self.parse_serial_data(serial_data['data'])
                        
                        # 如果解析后的数据包含传感器类型和值，进行处理
                        if 'type' in parsed_data and 'value' in parsed_data:
                            processed_data = self.process_sensor_data(parsed_data)
                            if processed_data:
                                # 添加到融合数据列表
                                collected_data.append(processed_data)
                                
                                # 发送到主模型
                                self.send_to_main_model('serial', processed_data)
                        
                        # 发送串口数据到数据总线
                        self.send_serial_data_to_bus(port, serial_data)
                
                # 定期执行数据融合
                if current_time - last_fusion_time >= collection_interval and collected_data:
                    try:
                        # 执行数据融合
                        fused_data = self.fuse_sensor_data(collected_data)
                        if fused_data:
                            # 发送融合后的数据到主模型
                            self.send_to_main_model('fused', fused_data)
                            
                            logger.debug(f"执行数据融合，处理了{len(collected_data)}个数据点")
                    except Exception as e:
                        logger.error(f"执行数据融合时出错: {str(e)}")
                    
                    # 清空收集的数据列表
                    collected_data = []
                    last_fusion_time = current_time
                
                # 避免CPU占用过高
                time.sleep(0.01)  # 10ms
                
            except Exception as e:
                logger.error(f"实时处理循环出错: {str(e)}")
                # 短暂暂停后继续
                time.sleep(0.1)
    def connect_to_main_model(self):
        """连接到主模型数据总线"""
        try:
            host = self.config.get('data_bus_host', 'localhost')
            port = self.config.get('data_bus_port', 6379)
            db = self.config.get('data_bus_db', 0)
            
            self.data_bus = redis.Redis(
                host=host,
                port=port,
                db=db
            )
            self.data_bus.ping()
            logger.info(f"已连接到主模型数据总线: {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"连接主模型数据总线失败: {str(e)}")
            self.data_bus = None
            return False
    
    def send_to_data_bus(self, channel, data):
        """发送数据到数据总线"""
        try:
            if self.data_bus:
                self.data_bus.publish(channel, json.dumps(data))
                return True
            return False
        except Exception as e:
            logger.error(f"发送数据到数据总线失败 ({channel}): {str(e)}")
            return False
    
    def send_sensor_data_to_bus(self, sensor_data):
        """发送传感器数据到数据总线"""
        enriched_data = {
            'type': 'sensor_data',
            'data': sensor_data,
            'timestamp': datetime.now().isoformat()
        }
        return self.send_to_data_bus('sensor_data', enriched_data)
    
    def send_camera_data_to_bus(self, camera_data):
        """发送摄像头数据到数据总线"""
        enriched_data = {
            'type': 'camera_data',
            'data': camera_data,
            'timestamp': datetime.now().isoformat()
        }
        return self.send_to_data_bus('camera_data', enriched_data)
    
    def send_audio_data_to_bus(self, audio_data):
        """发送音频数据到数据总线"""
        enriched_data = {
            'type': 'audio_data',
            'data': audio_data,
            'timestamp': datetime.now().isoformat()
        }
        return self.send_to_data_bus('audio_data', enriched_data)
    
    def send_network_video_data_to_bus(self, url, frame):
        """发送网络视频数据到数据总线"""
        # 注意：为了避免数据过大，这里不直接发送帧数据，而是发送帧信息
        enriched_data = {
            'type': 'network_video',
            'url': url,
            'frame_info': {
                'shape': frame.shape,
                'dtype': str(frame.dtype)
            },
            'timestamp': datetime.now().isoformat()
        }
        return self.send_to_data_bus('network_video_data', enriched_data)
    
    def send_serial_data_to_bus(self, port, serial_data):
        """发送串口数据到数据总线"""
        enriched_data = {
            'type': 'serial_data',
            'port': port,
            'data': serial_data,
            'timestamp': datetime.now().isoformat()
        }
        return self.send_to_data_bus('serial_data', enriched_data)
    
    def initialize_external_apis(self):
        """初始化外部API客户端"""
        if 'external_api' in self.config and self.config['external_api'].get('enabled', False):
            for api in self.config['external_api'].get('apis', []):
                if api.get('enabled', False):
                    result = self.connect_external_api(api)
                    if result['status'] == 'success':
                        self.external_api_clients[api['name']] = {
                            'client': result['client'],
                            'type': api['type'],
                            'config': api
                        }
    
    def connect_external_api(self, api_config):
        """连接外部API"""
        try:
            api_type = api_config.get('type')
            api_key = api_config.get('api_key')
            endpoint = api_config.get('endpoint')
            
            # 尝试导入必要的库以避免在未使用时占用资源
            if api_type == 'openai' and 'openai' not in sys.modules:
                import openai
                openai.api_key = api_key
                if endpoint:
                    openai.api_base = endpoint
                return {'status': 'success', 'client': openai}
            
            elif api_type == 'azure' and 'azure.cognitiveservices.vision.computervision' not in sys.modules:
                from azure.cognitiveservices.vision.computervision import ComputerVisionClient
                from msrest.authentication import CognitiveServicesCredentials
                client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(api_key))
                return {'status': 'success', 'client': client}
            
            elif api_type == 'google_cloud' and 'google.cloud.vision' not in sys.modules:
                from google.cloud import vision
                client = vision.ImageAnnotatorClient.from_service_account_json(api_key)
                return {'status': 'success', 'client': client}
            
            elif api_type == 'aws' and 'boto3' not in sys.modules:
                import boto3
                client = boto3.client('rekognition', 
                                    aws_access_key_id=api_config.get('access_key'),
                                    aws_secret_access_key=api_config.get('secret_key'),
                                    region_name=api_config.get('region', 'us-east-1'))
                return {'status': 'success', 'client': client}
            
            else:
                return {'status': 'error', 'message': f'不支持的API类型: {api_type}'}
            
        except Exception as e:
            logger.error(f"连接外部API失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}

# 实时接口功能
    def get_camera_stream(self, device_index=0):
        """获取摄像头流"""
        try:
            self.camera_stream = cv2.VideoCapture(device_index)
            if self.camera_stream.isOpened():
                logger.info(f"摄像头已连接 (设备 {device_index})")
                return {'status': 'success', 'message': f'Camera connected (device {device_index})'}
            else:
                self.camera_stream = None
                return {'status': 'error', 'message': f'Failed to open camera (device {device_index})'}
        except Exception as e:
            self.camera_stream = None
            logger.error(f"打开摄像头失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def release_camera_stream(self):
        """释放摄像头流"""
        if self.camera_stream:
            self.camera_stream.release()
            self.camera_stream = None
            logger.info("摄像头已释放")
    
    def capture_frame(self):
        """捕获一帧图像"""
        if self.camera_stream and self.camera_stream.isOpened():
            ret, frame = self.camera_stream.read()
            if ret:
                return {'status': 'success', 'frame': frame}
            else:
                return {'status': 'error', 'message': 'Failed to capture frame'}
        return {'status': 'error', 'message': 'Camera not initialized'}
    
    def get_microphone_stream(self, device_index=0, channels=1, rate=44100):
        """获取麦克风流"""
        try:
            if 'pyaudio' not in sys.modules:
                import pyaudio
            
            self.audio = pyaudio.PyAudio()
            self.microphone_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024
            )
            logger.info(f"麦克风已连接 (设备 {device_index})")
            return {'status': 'success', 'message': f'Microphone connected (device {device_index})'}
        except Exception as e:
            self.microphone_stream = None
            logger.error(f"打开麦克风失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def release_microphone_stream(self):
        """释放麦克风流"""
        if self.microphone_stream:
            self.microphone_stream.stop_stream()
            self.microphone_stream.close()
            self.microphone_stream = None
        if self.audio:
            self.audio.terminate()
            self.audio = None
            logger.info("麦克风已释放")
    
    def read_audio_data(self, num_frames=1024):
        """读取音频数据"""
        if self.microphone_stream:
            try:
                data = self.microphone_stream.read(num_frames, exception_on_overflow=False)
                return {'status': 'success', 'audio_data': data}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        return {'status': 'error', 'message': 'Microphone not initialized'}
    
    def connect_network_stream(self, stream_type, url, config=None):
        """连接网络流"""
        try:
            if stream_type == 'video':
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    self.network_streams[url] = {
                        'type': 'video',
                        'capture': cap,
                        'config': config or {}
                    }
                    logger.info(f"网络视频流已连接: {url}")
                    return {'status': 'success', 'message': f'Network video stream connected: {url}'}
                else:
                    return {'status': 'error', 'message': f'Failed to open network video stream: {url}'}
            
            elif stream_type == 'audio':
                # 连接网络音频流
                self.network_streams[url] = {
                    'type': 'audio',
                    'config': config or {}
                }
                logger.info(f"网络音频流已连接: {url}")
                return {'status': 'success', 'message': f'Network audio stream connected: {url}'}
            
            return {'status': 'error', 'message': 'Unsupported stream type'}
        except Exception as e:
            logger.error(f"连接网络流失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def disconnect_network_stream(self, url):
        """断开网络流"""
        if url in self.network_streams:
            stream = self.network_streams[url]
            if stream['type'] == 'video' and 'capture' in stream:
                stream['capture'].release()
            del self.network_streams[url]
            logger.info(f"网络流已断开: {url}")
    
    def open_serial_port(self, port, baudrate=9600, timeout=1):
        """打开串口"""
        try:
            if 'serial' not in sys.modules:
                import serial
            
            ser = serial.Serial(port, baudrate, timeout=timeout)
            self.serial_ports[port] = ser
            logger.info(f"串口已打开: {port}")
            return {'status': 'success', 'message': f'Serial port opened: {port}'}
        except Exception as e:
            logger.error(f"打开串口失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def close_serial_port(self, port):
        """关闭串口"""
        if port in self.serial_ports:
            self.serial_ports[port].close()
            del self.serial_ports[port]
            logger.info(f"串口已关闭: {port}")
    
    def read_serial_data(self, port):
        """读取串口数据"""
        if port in self.serial_ports:
            try:
                data = self.serial_ports[port].readline().decode('utf-8', errors='replace').strip()
                return {'status': 'success', 'data': data}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        return {'status': 'error', 'message': f'Serial port not open: {port}'}
    
    def write_serial_data(self, port, data):
        """写入串口数据"""
        if port in self.serial_ports:
            try:
                self.serial_ports[port].write(data.encode('utf-8'))
                return {'status': 'success', 'message': 'Data sent'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        return {'status': 'error', 'message': f'Serial port not open: {port}'}
    
    def get_interfaces_status(self):
        """获取所有接口状态"""
        try:
            camera_active = False
            if self.camera_stream:
                camera_active = self.camera_stream.isOpened()
        except:
            camera_active = False
        
        return {
            'camera': {
                'active': camera_active
            },
            'microphone': {
                'active': self.microphone_stream is not None
            },
            'network_streams': {
                'count': len(self.network_streams),
                'urls': list(self.network_streams.keys())
            },
            'serial_ports': {
                'count': len(self.serial_ports),
                'ports': list(self.serial_ports.keys())
            },
            'realtime_processing': {
                'active': self.realtime_active
            },
            'data_bus': {
                'connected': self.data_bus is not None
            }
        }
    
    def add_sensor_data(self, sensor_data):
        """添加传感器数据到处理队列"""
        try:
            self.sensor_data_queue.put_nowait(sensor_data)
            return {'status': 'success', 'message': 'Sensor data added to queue'}
        except queue.Full:
            return {'status': 'error', 'message': 'Sensor data queue is full'}
    
    def cleanup(self):
        """清理所有资源"""
        self.stop_realtime_processing()
        self.release_camera_stream()
        self.release_microphone_stream()
        
        for url in list(self.network_streams.keys()):
            self.disconnect_network_stream(url)
        
        for port in list(self.serial_ports.keys()):
            self.close_serial_port(port)
        
        logger.info("所有资源已清理")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 全局传感器集成管理器实例
sensor_manager = None

@app.route('/api/sensor_integration/status', methods=['GET'])
def api_get_status():
    """获取传感器集成管理器状态"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        # 获取接口状态
        interfaces_status = sensor_manager.get_interfaces_status()
        
        # 获取配置概览
        config_overview = {
            'data_bus_host': sensor_manager.config.get('data_bus_host'),
            'data_bus_port': sensor_manager.config.get('data_bus_port'),
            'external_apis': list(sensor_manager.external_api_clients.keys())
        }
        
        return jsonify({
            'status': 'success',
            'interfaces': interfaces_status,
            'config': config_overview
        })
    except Exception as e:
        logger.error(f"获取状态API失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensor_integration/add_data', methods=['POST'])
def api_add_sensor_data():
    """添加传感器数据"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': '未提供数据'}), 400
        
        result = sensor_manager.add_sensor_data(data)
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        logger.error(f"添加传感器数据失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensor_integration/realtime/start', methods=['POST'])
def api_start_realtime():
    """启动实时处理"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        result = sensor_manager.start_realtime_processing()
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        logger.error(f"启动实时处理失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensor_integration/realtime/stop', methods=['POST'])
def api_stop_realtime():
    """停止实时处理"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        sensor_manager.stop_realtime_processing()
        return jsonify({'status': 'success', 'message': '实时处理已停止'})
    except Exception as e:
        logger.error(f"停止实时处理失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensor_integration/camera/connect', methods=['POST'])
def api_connect_camera():
    """连接摄像头"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        data = request.json or {}
        device_index = data.get('device_index', 0)
        
        result = sensor_manager.get_camera_stream(device_index)
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        logger.error(f"连接摄像头失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensor_integration/camera/disconnect', methods=['POST'])
def api_disconnect_camera():
    """断开摄像头"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        sensor_manager.release_camera_stream()
        return jsonify({'status': 'success', 'message': '摄像头已断开'})
    except Exception as e:
        logger.error(f"断开摄像头失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensor_integration/serial/open', methods=['POST'])
def api_open_serial_port():
    """打开串口"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        data = request.json
        if not data or 'port' not in data:
            return jsonify({'status': 'error', 'message': '缺少端口信息'}), 400
        
        port = data['port']
        baudrate = data.get('baudrate', 9600)
        timeout = data.get('timeout', 1)
        
        result = sensor_manager.open_serial_port(port, baudrate, timeout)
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        logger.error(f"打开串口失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensor_integration/serial/close', methods=['POST'])
def api_close_serial_port():
    """关闭串口"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        data = request.json
        if not data or 'port' not in data:
            return jsonify({'status': 'error', 'message': '缺少端口信息'}), 400
        
        port = data['port']
        sensor_manager.close_serial_port(port)
        return jsonify({'status': 'success', 'message': f'串口 {port} 已关闭'})
    except Exception as e:
        logger.error(f"关闭串口失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensor_integration/serial/write', methods=['POST'])
def api_write_serial_data():
    """写入串口数据"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        data = request.json
        if not data or 'port' not in data or 'data' not in data:
            return jsonify({'status': 'error', 'message': '缺少端口或数据信息'}), 400
        
        port = data['port']
        write_data = data['data']
        
        result = sensor_manager.write_serial_data(port, write_data)
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        logger.error(f"写入串口数据失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 缺少的导入
from flask import Flask, request, jsonify
from flask_cors import CORS

@app.route('/api/sensor_integration/history', methods=['GET'])
def api_get_sensor_history():
    """获取传感器历史数据"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        # 获取查询参数
        sensor_type = request.args.get('type', None)
        limit = request.args.get('limit', 100, type=int)
        limit = min(1000, max(1, limit))  # 限制在1-1000之间
        
        # 获取历史数据
        history = list(sensor_manager.data_history)
        
        # 根据类型过滤
        if sensor_type:
            history = [d for d in history if d.get('type') == sensor_type]
        
        # 限制数量并按时间戳排序
        history = sorted(history, key=lambda x: x.get('timestamp', 0), reverse=True)[:limit]
        
        return jsonify({'status': 'success', 'data': history, 'total': len(history)})
    except Exception as e:
        logger.error(f"获取传感器历史数据失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensor_integration/anomalies', methods=['GET'])
def api_get_anomalies():
    """获取异常检测结果"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        # 获取查询参数
        sensor_type = request.args.get('type', None)
        limit = request.args.get('limit', 50, type=int)
        limit = min(500, max(1, limit))  # 限制在1-500之间
        
        # 从历史数据中提取异常
        anomalies = []
        for data in sensor_manager.data_history:
            if 'anomaly' in data:
                if not sensor_type or data.get('type') == sensor_type:
                    anomalies.append({
                        'timestamp': data.get('timestamp'),
                        'sensor_type': data.get('type'),
                        'value': data.get('value'),
                        'anomaly': data.get('anomaly')
                    })
        
        # 限制数量并按时间戳排序
        anomalies = sorted(anomalies, key=lambda x: x.get('timestamp', 0), reverse=True)[:limit]
        
        return jsonify({'status': 'success', 'data': anomalies, 'total': len(anomalies)})
    except Exception as e:
        logger.error(f"获取异常检测结果失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensor_integration/fusion_config', methods=['GET'])
def api_get_fusion_config():
    """获取传感器融合配置"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        config = {
            'sensor_types': sensor_manager.sensor_types,
            'anomaly_thresholds': sensor_manager.anomaly_thresholds
        }
        return jsonify({'status': 'success', 'data': config})
    except Exception as e:
        logger.error(f"获取融合配置失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensor_integration/fusion_config', methods=['POST'])
def api_update_fusion_config():
    """更新传感器融合配置"""
    try:
        if not sensor_manager:
            return jsonify({'status': 'error', 'message': '传感器集成管理器未初始化'}), 500
        
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': '未提供数据'}), 400
        
        # 更新异常检测阈值
        if 'anomaly_thresholds' in data:
            for sensor_type, threshold in data['anomaly_thresholds'].items():
                if sensor_type in sensor_manager.anomaly_thresholds or sensor_type in sensor_manager.sensor_types:
                    sensor_manager.anomaly_thresholds[sensor_type] = float(threshold)
        
        return jsonify({'status': 'success', 'message': '融合配置已更新'})
    except Exception as e:
        logger.error(f"更新融合配置失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 主函数
def main():
    global sensor_manager
    
    try:
        # 初始化传感器集成管理器
        sensor_manager = SensorIntegrationManager()
        logger.info("传感器集成管理器初始化成功")
        
        # 获取端口号
        port = get_model_port('sensor') or 5006
        logger.info(f"将在端口 {port} 启动传感器集成API服务")
        
        # 启动实时处理（如果配置启用）
        realtime_config = sensor_manager.config.get('realtime_interfaces', {})
        if realtime_config.get('enabled', False):
            sensor_manager.start_realtime_processing()
        
        # 启动Flask服务器
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("传感器集成服务已停止")
    except Exception as e:
        logger.error(f"启动传感器集成服务失败: {str(e)}")
    finally:
        # 清理资源
        if sensor_manager:
            sensor_manager.cleanup()
        logger.info("所有资源已清理")

if __name__ == '__main__':
    main()