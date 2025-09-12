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

# 传感器感知模型API服务
# Sensor Perception Model API Service

from flask import Flask, request, jsonify
from .model import SensorModel
import numpy as np

app = Flask(__name__)
model = SensorModel()

@app.route('/process', methods=['POST'])
def process_sensor_data():
    """处理传感器数据API | Process sensor data API"""
    data = request.json
    sensor_data = data.get('sensor_data')
    
    if not sensor_data:
        return jsonify({'error': 'Sensor data is required'}), 400
    
    try:
        # 确保传感器数据是数值列表
        # Ensure sensor data is a list of numbers
        sensor_data = [float(x) for x in sensor_data]
        result = model.process_sensor_data(sensor_data)
        return jsonify({'status': 'success', 'processed_data': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    """检测传感器异常API | Detect sensor anomalies API"""
    data = request.json
    sensor_data = data.get('sensor_data')
    
    if not sensor_data:
        return jsonify({'error': 'Sensor data is required'}), 400
    
    try:
        # 确保传感器数据是数值列表
        # Ensure sensor data is a list of numbers
        sensor_data = [float(x) for x in sensor_data]
        result = model.detect_anomalies(sensor_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fuse', methods=['POST'])
def fuse_sensor_data():
    """融合多传感器数据API | Fuse multi-sensor data API"""
    data = request.json
    sensor_data_list = data.get('sensor_data_list')
    
    if not sensor_data_list or not isinstance(sensor_data_list, list):
        return jsonify({'error': 'Sensor data list is required'}), 400
    
    try:
        # 确保每个传感器数据集都是数值列表
        # Ensure each sensor dataset is a list of numbers
        processed_list = []
        for sensor_data in sensor_data_list:
            processed_list.append([float(x) for x in sensor_data])
        
        result = model.fuse_sensor_data(processed_list)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
