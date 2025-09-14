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

# 运动控制模型API服务
# Motion Control Model API Service

from flask import Flask, request, jsonify
from .model import MotionModel

app = Flask(__name__)
model = MotionModel()

@app.route('/plan', methods=['POST'])
def plan_motion():
    """规划运动轨迹API | Plan motion trajectory API"""
    data = request.json
    target = data.get('target')
    constraints = data.get('constraints', {})
    
    if not target:
        return jsonify({'error': 'Target is required'}), 400
    
    try:
        result = model.plan_motion(target, constraints)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/execute', methods=['POST'])
def execute_motion():
    """执行运动API | Execute motion API"""
    data = request.json
    trajectory = data.get('trajectory')
    
    if not trajectory:
        return jsonify({'error': 'Trajectory is required'}), 400
    
    try:
        result = model.execute_motion(trajectory)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/monitor', methods=['GET'])
def monitor_motion():
    """监控运动状态API | Monitor motion status API"""
    try:
        result = model.monitor_motion()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/calibrate', methods=['POST'])
def calibrate_system():
    """系统校准API | System calibration API"""
    data = request.json
    calibration_data = data.get('calibration_data', {})
    
    try:
        result = model.calibrate_system(calibration_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/emergency_stop', methods=['POST'])
def emergency_stop():
    """紧急停止API | Emergency stop API"""
    try:
        result = model.emergency_stop()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查 | Health check"""
    return jsonify({"status": "healthy", "model": "J_motion"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5010))
    app.run(host='0.0.0.0', port=port, debug=True)