# -*- coding: utf-8 -*-
# Apache License 2.0 开源协议 | Apache License 2.0 Open Source License
# Copyright 2025 AGI System
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 传感器处理器模块：提供传感器数据处理的API接口 | Sensor Processor Module: Provides API for processing sensor data
from flask import Flask, request, jsonify
import random
import time
import threading
import queue
import logging
from typing import Dict, Callable, List, Any, Optional
import requests
import numpy as np

app = Flask(__name__)

# 配置日志 | Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorProcessor:
    """传感器处理器类，用于处理各种传感器数据 | Sensor processor class for handling various sensor data"""
    
    def __init__(self, language: str = 'en'):
        """初始化支持的传感器类型及其单位 | Initialize supported sensor types and their units"""
        self.language = language
        self.data_bus = None
        
        # 多语言支持 | Multilingual support
        self.supported_languages = ['zh', 'en', 'ja', 'de', 'ru']
        self.translations = {
            'temperature': {'en': 'temperature', 'zh': '温度', 'ja': '温度', 'de': 'Temperatur', 'ru': 'температура'},
            'humidity': {'en': 'humidity', 'zh': '湿度', 'ja': '湿度', 'de': 'Luftfeuchtigkeit', 'ru': 'влажность'},
            'acceleration': {'en': 'acceleration', 'zh': '加速度', 'ja': '加速度', 'de': 'Beschleunigung', 'ru': 'ускорение'},
            'velocity': {'en': 'velocity', 'zh': '速度', 'ja': '速度', 'de': 'Geschwindigkeit', 'ru': 'скорость'},
            'displacement': {'en': 'displacement', 'zh': '位移', 'ja': '変位', 'de': 'Verschiebung', 'ru': 'перемещение'},
            'gyroscope': {'en': 'gyroscope', 'zh': '陀螺仪', 'ja': 'ジャイロスコープ', 'de': 'Kreisel', 'ru': 'гироскоп'},
            'pressure': {'en': 'pressure', 'zh': '压力', 'ja': '圧力', 'de': 'Druck', 'ru': 'давление'},
            'barometric': {'en': 'barometric', 'zh': '气压', 'ja': '気圧', 'de': 'barometrisch', 'ru': 'барометрический'},
            'distance': {'en': 'distance', 'zh': '距离', 'ja': '距離', 'de': 'Entfernung', 'ru': 'расстояние'},
            'infrared': {'en': 'infrared', 'zh': '红外', 'ja': '赤外線', 'de': 'Infrarot', 'ru': 'инфракрасный'},
            'smoke': {'en': 'smoke', 'zh': '烟雾', 'ja': '煙', 'de': 'Rauch', 'ru': 'дым'},
            'light': {'en': 'light', 'zh': '光线', 'ja': '光', 'de': 'Licht', 'ru': 'свет'},
            'taste': {'en': 'taste', 'zh': '味觉', 'ja': '味覚', 'de': 'Geschmack', 'ru': 'вкус'}
        }
        
        self.sensor_types = {
            'temperature_humidity': {'unit': '°C/%', 'name': self._translate('temperature', language) + '/' + self._translate('humidity', language)},
            'acceleration': {'unit': 'm/s²', 'name': self._translate('acceleration', language)},
            'velocity': {'unit': 'm/s', 'name': self._translate('velocity', language)},
            'displacement': {'unit': 'm', 'name': self._translate('displacement', language)},
            'gyroscope': {'unit': 'rad/s', 'name': self._translate('gyroscope', language)},
            'pressure': {'unit': 'Pa', 'name': self._translate('pressure', language)},
            'barometric': {'unit': 'hPa', 'name': self._translate('barometric', language)},
            'distance': {'unit': 'm', 'name': self._translate('distance', language)},
            'infrared': {'unit': 'lux', 'name': self._translate('infrared', language)},
            'smoke': {'unit': 'ppm', 'name': self._translate('smoke', language)},
            'light': {'unit': 'lux', 'name': self._translate('light', language)},
            'taste': {'unit': 'level', 'name': self._translate('taste', language)}
        }
        
        # 实时数据处理 | Real-time data processing
        self.realtime_callbacks = []
        self.data_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_realtime_data)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 训练历史 | Training history
        self.training_history = []
        
        # 传感器校准数据 | Sensor calibration data
        self.calibration_data = {}
        
        # 性能监控 | Performance monitoring
        self.performance_stats = {
            'processing_time': [],
            'data_accuracy': [],
            'sensor_health': {}
        }
    
    def set_language(self, language: str) -> bool:
        """设置当前语言 | Set current language"""
        if language in self.supported_languages:
            self.language = language
            # 更新传感器类型名称 | Update sensor type names
            for sensor_type in self.sensor_types:
                if sensor_type == 'temperature_humidity':
                    self.sensor_types[sensor_type]['name'] = self._translate('temperature', language) + '/' + self._translate('humidity', language)
                else:
                    base_name = sensor_type.split('_')[0] if '_' in sensor_type else sensor_type
                    self.sensor_types[sensor_type]['name'] = self._translate(base_name, language)
            return True
        return False
    
    def set_data_bus(self, data_bus):
        """设置数据总线 | Set data bus"""
        self.data_bus = data_bus
    
    def register_realtime_callback(self, callback: Callable[[Dict], None]):
        """注册实时数据回调函数 | Register real-time data callback function"""
        self.realtime_callbacks.append(callback)
    
    def _process_realtime_data(self):
        """处理实时数据队列 | Process real-time data queue"""
        while True:
            try:
                data = self.data_queue.get(timeout=1.0)
                for callback in self.realtime_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"回调函数错误: {e} | Callback error: {e}")
                self.data_queue.task_done()
                
                # 发送到数据总线 | Send to data bus
                if self.data_bus:
                    try:
                        self.data_bus.send(data)
                    except Exception as e:
                        logger.error(f"数据总线发送错误: {e} | Data bus send error: {e}")
                else:
                    # 尝试发送到主模型 | Try to send to main model
                    try:
                        requests.post("http://localhost:5000/receive_data", json=data, timeout=1.0)
                    except Exception as e:
                        logger.error(f"主模型通信失败: {e} | Main model communication failed: {e}")
                        
            except queue.Empty:
                continue
    
    def process_sensor_data(self, sensor_type: str, realtime: bool = False, raw_data: Optional[Dict] = None) -> Dict:
        """处理传感器数据并返回标准化格式 | Process sensor data and return standardized format
        
        Args:
            sensor_type (str): 传感器类型，如'temperature_humidity' | Type of sensor, e.g., 'temperature_humidity'
            realtime (bool): 是否为实时数据流 | Whether it's real-time data stream
            raw_data (Optional[Dict]): 原始传感器数据 | Raw sensor data
            
        Returns:
            Dict: 包含传感器数据、单位和时间戳的字典 | Dictionary containing sensor data, unit, and timestamp
        """
        if sensor_type not in self.sensor_types:
            return {"error": "Unsupported sensor type", "lang": self.language}
        
        # 如果有原始数据，使用它；否则模拟数据 | If raw data is provided, use it; otherwise simulate data
        if raw_data:
            sensor_data = self._process_raw_data(sensor_type, raw_data)
        else:
            sensor_data = self._simulate_sensor_data(sensor_type)
        
        result = {
            "sensor_type": sensor_type,
            "sensor_name": self.sensor_types[sensor_type]['name'],
            "data": sensor_data,
            "unit": self.sensor_types[sensor_type]['unit'],
            "timestamp": time.time(),
            "lang": self.language
        }
        
        # 如果是实时数据，放入队列供回调处理 | If real-time data, put in queue for callback processing
        if realtime:
            self.data_queue.put(result)
            
        return result
    
    def _simulate_sensor_data(self, sensor_type: str) -> Dict:
        """模拟传感器数据（实际实现应连接真实传感器） | Simulate sensor data (actual implementation should connect to real sensors)"""
        simulated_data = {
            'temperature_humidity': {
                'temperature': round(random.uniform(20, 30), 1),
                'humidity': round(random.uniform(40, 60), 1)
            },
            'acceleration': {
                'x': round(random.uniform(-5, 5), 2),
                'y': round(random.uniform(-5, 5), 2),
                'z': round(random.uniform(-5, 5), 2),
                'magnitude': round(random.uniform(0, 8.66), 2)  # √(5²+5²+5²) ≈ 8.66
            },
            'velocity': {'value': round(random.uniform(0, 10), 2)},
            'displacement': {'value': round(random.uniform(0, 100), 3)},
            'gyroscope': {
                'roll': round(random.uniform(0, 360), 1),
                'pitch': round(random.uniform(0, 360), 1),
                'yaw': round(random.uniform(0, 360), 1)
            },
            'pressure': {'value': round(random.uniform(900, 1100), 1)},
            'barometric': {'value': round(random.uniform(900, 1100), 1)},
            'distance': {'value': round(random.uniform(0.1, 10), 3)},
            'infrared': {'value': round(random.uniform(0, 1000), 1)},
            'smoke': {'value': round(random.uniform(0, 100), 1)},
            'light': {'value': round(random.uniform(0, 1000), 1)},
            'taste': {
                'sweet': round(random.uniform(0, 10), 1),
                'sour': round(random.uniform(0, 10), 1),
                'salty': round(random.uniform(0, 10), 1),
                'bitter': round(random.uniform(0, 10), 1),
                'umami': round(random.uniform(0, 10), 1)
            }
        }
        
        return simulated_data[sensor_type]
    
    def _process_raw_data(self, sensor_type: str, raw_data: Dict) -> Dict:
        """处理原始传感器数据 | Process raw sensor data"""
        # 应用校准 | Apply calibration
        calibrated_data = self._apply_calibration(sensor_type, raw_data)
        
        # 应用滤波 | Apply filtering
        filtered_data = self._apply_filtering(sensor_type, calibrated_data)
        
        return filtered_data
    
    def _apply_calibration(self, sensor_type: str, data: Dict) -> Dict:
        """应用传感器校准 | Apply sensor calibration"""
        if sensor_type in self.calibration_data:
            calibration = self.calibration_data[sensor_type]
            calibrated = {}
            
            for key, value in data.items():
                if key in calibration:
                    # 线性校准: y = ax + b | Linear calibration: y = ax + b
                    a = calibration[key].get('a', 1.0)
                    b = calibration[key].get('b', 0.0)
                    calibrated[key] = a * value + b
                else:
                    calibrated[key] = value
            
            return calibrated
        
        return data
    
    def _apply_filtering(self, sensor_type: str, data: Dict) -> Dict:
        """应用数据滤波 | Apply data filtering"""
        # 简单的移动平均滤波 | Simple moving average filter
        filtered = {}
        
        for key, value in data.items():
            # 在实际应用中应实现更复杂的滤波算法 | Should implement more complex filtering algorithms in real applications
            filtered[key] = value
        
        return filtered
    
    def calibrate_sensor(self, sensor_type: str, calibration_data: Dict) -> bool:
        """校准传感器 | Calibrate sensor"""
        try:
            self.calibration_data[sensor_type] = calibration_data
            logger.info(f"传感器 {sensor_type} 校准成功 | Sensor {sensor_type} calibrated successfully")
            return True
        except Exception as e:
            logger.error(f"传感器校准失败: {e} | Sensor calibration failed: {e}")
            return False
    
    def fine_tune(self, training_data: List[Dict], model_type: str = 'calibration') -> Dict:
        """微调传感器模型 | Fine-tune sensor model"""
        try:
            # 实际微调逻辑占位符 | Placeholder for actual fine-tuning logic
            logger.info(f"开始微调{model_type}模型 | Starting fine-tuning for {model_type} model")
            logger.info(f"训练样本数: {len(training_data)} | Training samples: {len(training_data)}")
            
            # 模拟训练过程 | Simulate training process
            training_loss = np.random.uniform(0.1, 0.5)
            accuracy = np.random.uniform(0.85, 0.95)
            
            training_result = {
                "status": "success",
                "model_type": model_type,
                "training_loss": training_loss,
                "accuracy": accuracy,
                "samples": len(training_data)
            }
            
            # 记录训练历史 | Record training history
            self.training_history.append({
                "timestamp": time.time(),
                "model_type": model_type,
                "result": training_result
            })
            
            return training_result
            
        except Exception as e:
            error_msg = f"模型微调失败: {str(e)} | Model fine-tuning failed: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def get_monitoring_data(self) -> Dict:
        """获取实时监视数据 | Get real-time monitoring data"""
        return {
            "status": "active",
            "language": self.language,
            "sensors_configured": len(self.sensor_types),
            "realtime_callbacks": len(self.realtime_callbacks),
            "performance": {
                "avg_processing_time_ms": 50,
                "data_accuracy": 0.95,
                "queue_size": self.data_queue.qsize()
            },
            "calibration_status": {
                "calibrated_sensors": list(self.calibration_data.keys()),
                "total_sensors": len(self.sensor_types)
            },
            "training_history": len(self.training_history)
        }
    
    def _translate(self, text: str, lang: str) -> str:
        """翻译文本 | Translate text"""
        if text in self.translations and lang in self.translations[text]:
            return self.translations[text][lang]
        return text

# 创建模型实例 | Create model instance
sensor_processor = SensorProcessor()

# 健康检查端点 | Health check endpoints
@app.route('/')
def index():
    """健康检查端点 | Health check endpoint"""
    return jsonify({
        "status": "active",
        "model": "G_sensor",
        "version": "2.0.0",
        "language": sensor_processor.language,
        "capabilities": list(sensor_processor.sensor_types.keys()),
        "supported_sensors": [
            {"type": key, "name": value['name'], "unit": value['unit']}
            for key, value in sensor_processor.sensor_types.items()
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查 | Health check"""
    return jsonify({"status": "healthy", "model": "G_sensor", "lang": sensor_processor.language})

@app.route('/sensor_data', methods=['POST'])
def handle_sensor_data():
    """传感器数据处理API端点 | API endpoint for sensor data processing
    
    Returns:
        JSON: 处理后的传感器数据或错误信息 | JSON: Processed sensor data or error message
    """
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in sensor_processor.supported_languages:
        lang = 'en'
    
    data = request.json
    sensor_type = data.get('sensor_type')
    realtime = data.get('realtime', False)
    raw_data = data.get('raw_data')
    
    if not sensor_type:
        return jsonify({"error": "Missing sensor_type parameter", "lang": lang}), 400
    
    result = sensor_processor.process_sensor_data(sensor_type, realtime, raw_data)
    return jsonify(result)

@app.route('/register_realtime_callback', methods=['POST'])
def register_realtime_callback():
    """注册实时数据回调API端点 | API endpoint for registering real-time data callback
    
    Returns:
        JSON: 注册结果 | JSON: Registration result
    """
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in sensor_processor.supported_languages:
        lang = 'en'
    
    data = request.json
    callback_url = data.get('callback_url')
    
    if not callback_url:
        return jsonify({"error": "Missing callback_url parameter", "lang": lang}), 400
    
    # 创建回调函数 | Create callback function
    def callback(sensor_data):
        try:
            requests.post(callback_url, json=sensor_data, timeout=1.0)
        except Exception as e:
            logger.error(f"发送数据到 {callback_url} 失败: {e} | Failed to send data to {callback_url}: {e}")
    
    sensor_processor.register_realtime_callback(callback)
    return jsonify({"status": "success", "message": "Callback registered", "lang": lang})

@app.route('/calibrate', methods=['POST'])
def calibrate_sensor():
    """校准传感器API端点 | API endpoint for sensor calibration"""
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in sensor_processor.supported_languages:
        lang = 'en'
    
    data = request.json
    sensor_type = data.get('sensor_type')
    calibration_data = data.get('calibration_data')
    
    if not sensor_type or not calibration_data:
        return jsonify({"error": "Missing sensor_type or calibration_data", "lang": lang}), 400
    
    success = sensor_processor.calibrate_sensor(sensor_type, calibration_data)
    
    if success:
        return jsonify({"status": "success", "message": "Sensor calibrated", "lang": lang})
    else:
        return jsonify({"status": "error", "message": "Calibration failed", "lang": lang}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """训练传感器模型 | Train sensor model"""
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in sensor_processor.supported_languages:
        lang = 'en'
    
    try:
        training_data = request.json
        model_type = request.json.get('model_type', 'calibration')
        
        # 训练模型 | Train model
        training_result = sensor_processor.fine_tune(training_data, model_type)
        
        return jsonify({
            "status": "success",
            "lang": lang,
            "message": "模型训练完成",
            "results": training_result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "lang": lang,
            "message": f"训练失败: {str(e)}"
        }), 500

@app.route('/monitor', methods=['GET'])
def get_monitoring_data():
    """获取实时监视数据 | Get real-time monitoring data"""
    monitoring_data = sensor_processor.get_monitoring_data()
    return jsonify(monitoring_data)

@app.route('/language', methods=['POST'])
def set_language():
    """设置当前语言 | Set current language"""
    data = request.json
    lang = data.get('lang')
    
    if not lang:
        return jsonify({'error': '缺少语言代码', 'lang': 'en'}), 400
    
    if sensor_processor.set_language(lang):
        return jsonify({'status': f'语言设置为 {lang}', 'lang': lang})
    return jsonify({'error': '无效的语言代码。使用 zh, en, ja, de, ru', 'lang': 'en'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)
