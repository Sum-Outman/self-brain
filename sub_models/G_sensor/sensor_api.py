# Copyright 2025 The AI Management System Authors
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

# 传感器模型API服务
# Sensor Model API Service

from flask import Flask, jsonify, request
import threading
import time
import random
from datetime import datetime, timedelta
import json
import os
import numpy as np
from collections import defaultdict, deque
import hashlib

app = Flask(__name__)

# 支持的传感器类型
SUPPORTED_SENSORS = [
    "temperature", "humidity", "acceleration", "velocity", "displacement",
    "gyroscope", "pressure", "barometric", "distance", "infrared",
    "taste", "smoke", "light"
]

# 传感器模型状态
sensor_status = {
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "sensor_data_points": 0,
    "data_received": 0,
    "queries_processed": 0,
    "analyses_performed": 0,
    "anomalies_detected": 0,
    "predictions_made": 0,
    "current_version": "1.0.0",
    "sensors_supported": SUPPORTED_SENSORS
}

# 传感器数据存储
SENSOR_DATA_FILE = "sensor_data.json"
sensor_data = defaultdict(lambda: defaultdict(list))  # sensor_type -> device_id -> [data points]

# 数据缓存（用于实时分析）
DATA_CACHE_SIZE = 1000
sensor_cache = defaultdict(lambda: defaultdict(lambda: deque(maxlen=DATA_CACHE_SIZE)))

def load_sensor_data():
    """加载传感器数据"""
    global sensor_data
    if os.path.exists(SENSOR_DATA_FILE):
        try:
            with open(SENSOR_DATA_FILE, 'r', encoding='utf-8') as f:
                sensor_data = json.load(f)
            print("传感器数据已加载")
            # 更新数据点计数
            count = 0
            for sensor_type in sensor_data:
                for device_id in sensor_data[sensor_type]:
                    count += len(sensor_data[sensor_type][device_id])
            sensor_status["sensor_data_points"] = count
        except Exception as e:
            print(f"加载传感器数据失败: {str(e)}")
    else:
        print("使用空传感器数据集")

def save_sensor_data():
    """保存传感器数据到文件"""
    try:
        with open(SENSOR_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(sensor_data, f, ensure_ascii=False, indent=2)
        print("传感器数据已保存")
    except Exception as e:
        print(f"保存传感器数据失败: {str(e)}")

def sensor_maintenance():
    """定期维护传感器模型"""
    while True:
        # 每30分钟自动保存一次
        save_sensor_data()
        time.sleep(1800)

def generate_device_id(sensor_type, location):
    """生成设备ID"""
    return hashlib.md5(f"{sensor_type}_{location}".encode('utf-8')).hexdigest()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "data_points": sensor_status["sensor_data_points"],
        "version": sensor_status["current_version"]
    })

@app.route('/get_status', methods=['GET'])
def get_status():
    """获取传感器模型状态"""
    return jsonify({
        "status": "success",
        "status_info": {
            "last_updated": sensor_status["last_updated"],
            "sensor_data_points": sensor_status["sensor_data_points"],
            "data_received": sensor_status["data_received"],
            "queries_processed": sensor_status["queries_processed"],
            "analyses_performed": sensor_status["analyses_performed"],
            "anomalies_detected": sensor_status["anomalies_detected"],
            "predictions_made": sensor_status["predictions_made"],
            "current_version": sensor_status["current_version"],
            "sensors_supported": sensor_status["sensors_supported"]
        }
    })

@app.route('/receive_data', methods=['POST'])
def receive_data():
    """接收传感器数据"""
    data = request.json
    if not data or 'sensor_type' not in data or 'value' not in data:
        return jsonify({"status": "error", "message": "缺少传感器类型或值参数"}), 400
    
    sensor_type = data['sensor_type']
    value = data['value']
    device_id = data.get('device_id', "default_device")
    timestamp = data.get('timestamp', datetime.now().isoformat())
    location = data.get('location', "unknown")
    unit = data.get('unit', "")
    
    if sensor_type not in SUPPORTED_SENSORS:
        return jsonify({"status": "error", "message": f"不支持的传感器类型: {sensor_type}"}), 400
    
    # 创建数据点
    data_point = {
        "value": value,
        "timestamp": timestamp,
        "location": location,
        "unit": unit
    }
    
    # 存储到内存数据结构
    sensor_data[sensor_type][device_id].append(data_point)
    
    # 添加到缓存
    sensor_cache[sensor_type][device_id].append(data_point)
    
    # 更新状态
    sensor_status["sensor_data_points"] += 1
    sensor_status["data_received"] += 1
    
    # 生成数据点ID
    data_str = f"{sensor_type}{device_id}{timestamp}{value}{location}"
    data_point_id = hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    return jsonify({
        "status": "success",
        "message": "数据点已接收",
        "data_point_id": data_point_id,
        "data_point": data_point
    })

@app.route('/query_data', methods=['POST'])
def query_data():
    """查询传感器数据"""
    data = request.json
    sensor_type = data.get('sensor_type', "")
    device_id = data.get('device_id', "")
    start_time = data.get('start_time', "")
    end_time = data.get('end_time', datetime.now().isoformat())
    max_results = data.get('max_results', 100)
    
    sensor_status["queries_processed"] += 1
    
    results = []
    
    # 如果没有指定传感器类型，搜索所有类型
    sensor_types = [sensor_type] if sensor_type else SUPPORTED_SENSORS
    
    for st in sensor_types:
        # 如果没有指定设备ID，搜索该类型的所有设备
        device_ids = [device_id] if device_id else list(sensor_data.get(st, {}).keys())
        
        for dev_id in device_ids:
            if st in sensor_data and dev_id in sensor_data[st]:
                for point in sensor_data[st][dev_id]:
                    point_time = point["timestamp"]
                    if (not start_time or point_time >= start_time) and point_time <= end_time:
                        results.append({
                            "sensor_type": st,
                            "device_id": dev_id,
                            **point
                        })
                        if len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break
    
    return jsonify({
        "status": "success",
        "results": results,
        "count": len(results)
    })

@app.route('/analyze_data', methods=['POST'])
def analyze_data():
    """分析传感器数据"""
    data = request.json
    sensor_type = data.get('sensor_type', "")
    device_id = data.get('device_id', "")
    start_time = data.get('start_time', "")
    end_time = data.get('end_time', datetime.now().isoformat())
    
    sensor_status["analyses_performed"] += 1
    
    # 获取数据点
    points = []
    if sensor_type and device_id and sensor_type in sensor_data and device_id in sensor_data[sensor_type]:
        for point in sensor_data[sensor_type][device_id]:
            point_time = point["timestamp"]
            if (not start_time or point_time >= start_time) and point_time <= end_time:
                points.append(point["value"])
    
    # 计算统计数据
    if points:
        values = np.array(points)
        analysis_result = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std_dev": float(np.std(values)),
            "variance": float(np.var(values))
        }
    else:
        analysis_result = {
            "message": "没有找到匹配的数据点"
        }
    
    return jsonify({
        "status": "success",
        "analysis_result": analysis_result
    })

@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    """检测传感器数据中的异常"""
    data = request.json
    sensor_type = data.get('sensor_type', "")
    device_id = data.get('device_id', "")
    threshold = data.get('threshold', 2.0)  # 标准差阈值
    
    sensor_status["anomalies_detected"] += 1
    
    # 获取数据点
    points = []
    timestamps = []
    if sensor_type and device_id and sensor_type in sensor_cache and device_id in sensor_cache[sensor_type]:
        for point in sensor_cache[sensor_type][device_id]:
            points.append(point["value"])
            timestamps.append(point["timestamp"])
    
    anomalies = []
    
    if len(points) > 10:  # 需要足够的数据点
        values = np.array(points)
        mean = np.mean(values)
        std = np.std(values)
        
        # 检测异常（超过阈值标准差）
        for i, value in enumerate(values):
            if abs(value - mean) > threshold * std:
                anomalies.append({
                    "value": value,
                    "timestamp": timestamps[i],
                    "deviation": abs(value - mean),
                    "threshold": threshold * std
                })
    
    return jsonify({
        "status": "success",
        "anomalies": anomalies,
        "count": len(anomalies)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """基于历史数据预测未来值"""
    data = request.json
    sensor_type = data.get('sensor_type', "")
    device_id = data.get('device_id', "")
    steps = data.get('steps', 5)  # 预测未来5个点
    
    sensor_status["predictions_made"] += 1
    
    # 获取数据点
    points = []
    if sensor_type and device_id and sensor_type in sensor_cache and device_id in sensor_cache[sensor_type]:
        points = [point["value"] for point in sensor_cache[sensor_type][device_id]]
    
    predictions = []
    
    if len(points) > 10:  # 需要足够的数据点
        # 简单预测：使用移动平均
        window_size = min(5, len(points))
        last_values = points[-window_size:]
        avg = sum(last_values) / len(last_values)
        
        # 生成预测
        for i in range(steps):
            predictions.append(avg + random.uniform(-0.1, 0.1) * avg)  # 添加一些随机变化
    
    return jsonify({
        "status": "success",
        "predictions": predictions
    })

if __name__ == '__main__':
    # 加载传感器数据
    load_sensor_data()
    
    # 启动传感器模型维护线程
    maintenance_thread = threading.Thread(target=sensor_maintenance, daemon=True)
    maintenance_thread.start()
    
    # 启动API服务
    app.run(host='0.0.0.0', port=5006)
