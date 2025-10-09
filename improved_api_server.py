"""
改进的统一API服务器
Improved Unified API Server
集成所有真实功能
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import threading

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入改进的系统
from unified_system_improvements import (
    health_checker, training_system, emotion_analyzer, 
    hardware_integration, data_bus
)
from sub_models.K_programming.real_programming_system import real_programming_system
from sub_models.F_spatial.real_stereo_vision import stereo_system
from sub_models.G_sensor.real_sensor_system import real_sensor_system

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Improved_API")

app = Flask(__name__, template_folder='templates')
CORS(app)

# 启动所有改进系统
def initialize_systems():
    """初始化所有系统"""
    logger.info("正在初始化改进系统...")
    
    # 启动传感器系统
    devices = real_sensor_system.discover_devices()
    for device_type, device_list in devices.items():
        for device_id in device_list:
            if device_type == 'mock':
                sensor_type = device_id.replace('mock_', '')
                device = real_sensor_system.devices.get(device_id) or \
                        real_sensor_system.devices.__class__.__bases__[0](device_id, sensor_type)
                real_sensor_system.add_device(device)
    
    # 启动双目视觉系统
    stereo_system.calibrate_stereo()
    
    # 启动实时处理
    real_sensor_system.start_data_collection()
    stereo_system.start_real_time_processing()
    
    logger.info("所有系统初始化完成")

# 初始化系统
initialize_systems()

@app.route('/')
def index():
    """改进系统主页"""
    return render_template('improved_system.html')

@app.route('/api/health/improved', methods=['GET'])
def improved_health_check():
    """改进的健康检查"""
    try:
        # 检查所有服务状态
        services = health_checker.check_all_services()
        
        # 检查系统状态
        system_status = {
            'timestamp': datetime.now().isoformat(),
            'services': services,
            'sensor_system': {
                'running': real_sensor_system.running,
                'device_count': len(real_sensor_system.devices)
            },
            'stereo_vision': {
                'running': stereo_system.running,
                'calibrated': stereo_system.is_calibrated
            },
            'programming_system': {
                'optimizations': len(real_programming_system.learning_history)
            }
        }
        
        return jsonify({
            'status': 'healthy',
            'system': system_status
        })
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/services/start', methods=['POST'])
def start_missing_services():
    """启动缺失服务"""
    try:
        started = health_checker.start_missing_services()
        return jsonify({
            'status': 'success',
            'started_services': started
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """启动真实训练"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        model_type = data.get('model_type')
        dataset_path = data.get('dataset_path', 'default_dataset')
        config = data.get('config', {})
        
        job_id = training_system.start_training(model_type, dataset_path, config)
        
        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'model_type': model_type
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/training/status/<job_id>', methods=['GET'])
def get_training_status(job_id):
    """获取训练状态"""
    try:
        status = training_system.get_training_status(job_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/emotion/analyze', methods=['POST'])
def analyze_emotion():
    """分析情感"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        emotions = emotion_analyzer.analyze_emotion(text)
        
        return jsonify({
            'status': 'success',
            'text': text,
            'emotions': emotions
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensors/realtime', methods=['GET'])
def get_realtime_sensors():
    """获取实时传感器数据"""
    try:
        data = real_sensor_system.get_real_time_data()
        return jsonify({
            'status': 'success',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensors/devices', methods=['GET'])
def get_sensor_devices():
    """获取传感器设备状态"""
    try:
        status = real_sensor_system.get_device_status()
        return jsonify({
            'status': 'success',
            'devices': status
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensors/discover', methods=['GET'])
def discover_sensors():
    """发现可用传感器"""
    try:
        devices = real_sensor_system.discover_devices()
        return jsonify({
            'status': 'success',
            'available_devices': devices
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stereo/vision', methods=['GET'])
def get_stereo_vision_data():
    """获取双目视觉数据"""
    try:
        results = stereo_system.get_latest_results()
        
        # 转换numpy数组为列表以便JSON序列化
        processed_results = {
            'objects': results.get('objects', []),
            'calibration': stereo_system.get_calibration_info(),
            'timestamp': results.get('timestamp', datetime.now().isoformat())
        }
        
        return jsonify({
            'status': 'success',
            'data': processed_results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/programming/generate', methods=['POST'])
def generate_programming_solution():
    """生成编程解决方案"""
    try:
        data = request.get_json()
        if not data or 'requirements' not in data:
            return jsonify({'error': 'Requirements are required'}), 400
        
        requirements = data['requirements']
        result = real_programming_system.generate_and_optimize(requirements)
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/programming/history', methods=['GET'])
def get_programming_history():
    """获取编程历史"""
    try:
        summary = real_programming_system.get_learning_summary()
        return jsonify({
            'status': 'success',
            'history': summary
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/hardware/status', methods=['GET'])
def get_hardware_status():
    """获取硬件状态"""
    try:
        import psutil
        
        # CPU和内存信息
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 传感器数据
        sensor_data = hardware_integration.get_sensor_data()
        
        return jsonify({
            'status': 'success',
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'cpu_count': psutil.cpu_count(),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
            },
            'sensors': sensor_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/data/publish', methods=['POST'])
def publish_data():
    """发布数据到统一数据总线"""
    try:
        data = request.get_json()
        if not data or 'channel' not in data or 'data' not in data:
            return jsonify({'error': 'Channel and data are required'}), 400
        
        channel = data['channel']
        payload = data['data']
        
        data_bus.publish(channel, payload)
        
        return jsonify({
            'status': 'success',
            'channel': channel,
            'published_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/data/subscribe/<channel>', methods=['POST'])
def subscribe_to_channel(channel):
    """订阅数据频道"""
    try:
        # 获取该频道的历史数据
        data = data_bus.get_data(channel)
        
        return jsonify({
            'status': 'success',
            'channel': channel,
            'data': data,
            'subscribed_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/system/summary', methods=['GET'])
def get_system_summary():
    """获取系统综合摘要"""
    try:
        # 收集所有系统状态
        summary = {
            'timestamp': datetime.now().isoformat(),
            'services': health_checker.check_all_services(),
            'sensors': {
                'active_devices': len(real_sensor_system.devices),
                'realtime_data': real_sensor_system.get_real_time_data()
            },
            'stereo_vision': {
                'running': stereo_system.running,
                'objects_detected': len(stereo_system.get_latest_results().get('objects', []))
            },
            'training_system': {
                'active_jobs': len(training_system.training_jobs),
                'completed_jobs': len([j for j in training_system.training_jobs.values() 
                                     if j['status'] == 'completed'])
            },
            'programming_system': {
                'total_optimizations': len(real_programming_system.learning_history),
                'recent_optimizations': real_programming_system.get_learning_summary()
            },
            'data_bus': {
                'active_channels': len(data_bus.data_queue),
                'total_messages': sum(len(q) for q in data_bus.data_queue.values())
            }
        }
        
        return jsonify({
            'status': 'success',
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 创建改进的HTML模板
@app.route('/improved_system')
def improved_system_ui():
    """改进系统UI"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Improved AGI System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 5px 0; border-radius: 4px; }
        .status.running { background: #d4edda; color: #155724; }
        .status.stopped { background: #f8d7da; color: #721c24; }
        .status.error { background: #fff3cd; color: #856404; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .data-display { background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 10px 0; }
        pre { background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Improved AGI System Dashboard</h1>
        
        <div class="section">
            <h2>System Health</h2>
            <button onclick="checkHealth()">Check Health</button>
            <div id="health-status" class="data-display">Loading...</div>
        </div>
        
        <div class="section">
            <h2>Real-time Sensors</h2>
            <button onclick="getSensorData()">Get Sensor Data</button>
            <div id="sensor-data" class="data-display">Loading...</div>
        </div>
        
        <div class="section">
            <h2>Training System</h2>
            <button onclick="getTrainingStatus()">Check Training</button>
            <div id="training-status" class="data-display">Loading...</div>
        </div>
        
        <div class="section">
            <h2>Stereo Vision</h2>
            <button onclick="getStereoData()">Get Vision Data</button>
            <div id="stereo-data" class="data-display">Loading...</div>
        </div>
        
        <div class="section">
            <h2>Programming System</h2>
            <textarea id="requirements" placeholder="Enter programming requirements..." rows="4" style="width: 100%"></textarea>
            <button onclick="generateCode()">Generate Code</button>
            <div id="code-result" class="data-display"></div>
        </div>
    </div>

    <script>
        async function checkHealth() {
            const response = await fetch('/api/health/improved');
            const data = await response.json();
            document.getElementById('health-status').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }

        async function getSensorData() {
            const response = await fetch('/api/sensors/realtime');
            const data = await response.json();
            document.getElementById('sensor-data').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }

        async function getTrainingStatus() {
            const response = await fetch('/api/training/status/sample_job');
            const data = await response.json();
            document.getElementById('training-status').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }

        async function getStereoData() {
            const response = await fetch('/api/stereo/vision');
            const data = await response.json();
            document.getElementById('stereo-data').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }

        async function generateCode() {
            const requirements = document.getElementById('requirements').value;
            if (!requirements) {
                alert('Please enter requirements');
                return;
            }

            const response = await fetch('/api/programming/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ requirements })
            });
            const data = await response.json();
            document.getElementById('code-result').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }

        // 自动加载初始数据
        setTimeout(() => {
            checkHealth();
            getSensorData();
            getStereoData();
        }, 1000);
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    print("=== 启动改进的统一API服务器 ===")
    print("改进功能包括:")
    print("1. 真实服务健康检查")
    print("2. 真实训练系统")
    print("3. 真实情感分析")
    print("4. 真实传感器数据")
    print("5. 真实双目视觉")
    print("6. 真实自主编程优化")
    print("7. 统一数据总线")
    print()
    print("访问 http://localhost:5005/improved_system 查看改进系统")
    
    app.run(host='0.0.0.0', port=5005, debug=True)