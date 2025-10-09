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

# 双目空间定位感知模型API服务
# Binocular Spatial Localization Perception Model API Service

from flask import Flask, request, jsonify
from .model import SpatialModel
import os
import uuid
import numpy as np
import cv2
import json
import logging
import requests
import base64
from io import BytesIO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("F_spatial_API")

app = Flask(__name__)
model = SpatialModel()

# 外部API配置
EXTERNAL_API_CONFIG = {
    'enabled': False,
    'base_url': 'https://api.external-spatial-service.com/v1',
    'api_key': os.environ.get('EXTERNAL_SPATIAL_API_KEY', ''),
    'timeout': 30
}

@app.route('/config', methods=['POST'])
def configure_api():
    """配置API模式（本地/外部）| Configure API mode (local/external)"""
    data = request.json
    if data.get('external_api') is not None:
        EXTERNAL_API_CONFIG['enabled'] = data['external_api']
    if data.get('base_url'):
        EXTERNAL_API_CONFIG['base_url'] = data['base_url']
    if data.get('api_key'):
        EXTERNAL_API_CONFIG['api_key'] = data['api_key']
    
    return jsonify({
        'status': 'success',
        'current_config': EXTERNAL_API_CONFIG
    })

@app.route('/status', methods=['GET'])
def get_status():
    """获取API状态 | Get API status"""
    return jsonify({
        'status': 'active',
        'mode': 'external' if EXTERNAL_API_CONFIG['enabled'] else 'local',
        'external_api_available': check_external_api_health() if EXTERNAL_API_CONFIG['enabled'] else False
    })

def check_external_api_health():
    """检查外部API健康状态 | Check external API health"""
    try:
        headers = {'Authorization': f'Bearer {EXTERNAL_API_CONFIG["api_key"]}'} if EXTERNAL_API_CONFIG['api_key'] else {}
        response = requests.get(f"{EXTERNAL_API_CONFIG['base_url']}/health", 
                              headers=headers, 
                              timeout=EXTERNAL_API_CONFIG['timeout'])
        return response.status_code == 200
    except:
        return False

def call_external_api(endpoint, method='POST', data=None, files=None):
    """调用外部API | Call external API"""
    try:
        url = f"{EXTERNAL_API_CONFIG['base_url']}/{endpoint}"
        headers = {'Authorization': f'Bearer {EXTERNAL_API_CONFIG["api_key"]}'} if EXTERNAL_API_CONFIG['api_key'] else {}
        
        if method == 'POST':
            if files:
                response = requests.post(url, headers=headers, files=files, timeout=EXTERNAL_API_CONFIG['timeout'])
            else:
                response = requests.post(url, headers=headers, json=data, timeout=EXTERNAL_API_CONFIG['timeout'])
        else:
            response = requests.get(url, headers=headers, params=data, timeout=EXTERNAL_API_CONFIG['timeout'])
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"External API error: {response.status_code}, {response.text}")
            return {'error': f'External API error: {response.status_code}'}
    except Exception as e:
        logger.error(f"External API request failed: {str(e)}")
        return {'error': str(e)}

@app.route('/localize', methods=['POST'])
def localize():
    """空间定位API | Spatial localization API"""
    if EXTERNAL_API_CONFIG['enabled']:
        # 使用外部API
        if 'left_image' not in request.files or 'right_image' not in request.files:
            return jsonify({'error': 'Both left and right images are required'}), 400
        
        left_file = request.files['left_image']
        right_file = request.files['right_image']
        
        files = {
            'left_image': (left_file.filename, left_file.stream, left_file.mimetype),
            'right_image': (right_file.filename, right_file.stream, right_file.mimetype)
        }
        
        return jsonify(call_external_api('localize', files=files))
    else:
        # 使用本地模型
        if 'left_image' not in request.files or 'right_image' not in request.files:
            return jsonify({'error': 'Both left and right images are required'}), 400
        
        left_file = request.files['left_image']
        right_file = request.files['right_image']
        
        temp_left = f"/tmp/{uuid.uuid4()}.png"
        temp_right = f"/tmp/{uuid.uuid4()}.png"
        
        left_file.save(temp_left)
        right_file.save(temp_right)
        
        try:
            result = model.localize_objects(temp_left, temp_right)
            return jsonify(result)
        finally:
            os.remove(temp_left)
            os.remove(temp_right)

@app.route('/modeling', methods=['POST'])
def modeling():
    """空间建模API | Spatial modeling API"""
    if EXTERNAL_API_CONFIG['enabled']:
        # 使用外部API
        if 'left_image' not in request.files or 'right_image' not in request.files:
            return jsonify({'error': 'Both left and right images are required'}), 400
        
        left_file = request.files['left_image']
        right_file = request.files['right_image']
        
        files = {
            'left_image': (left_file.filename, left_file.stream, left_file.mimetype),
            'right_image': (right_file.filename, right_file.stream, right_file.mimetype)
        }
        
        return jsonify(call_external_api('modeling', files=files))
    else:
        # 使用本地模型
        if 'left_image' not in request.files or 'right_image' not in request.files:
            return jsonify({'error': 'Both left and right images are required'}), 400
        
        left_file = request.files['left_image']
        right_file = request.files['right_image']
        
        temp_left = f"/tmp/{uuid.uuid4()}.png"
        temp_right = f"/tmp/{uuid.uuid4()}.png"
        
        left_file.save(temp_left)
        right_file.save(temp_right)
        
        try:
            result = model.create_spatial_model(temp_left, temp_right)
            return jsonify(result)
        finally:
            os.remove(temp_left)
            os.remove(temp_right)

@app.route('/track', methods=['POST'])
def track():
    """运动物体追踪API | Moving object tracking API"""
    if EXTERNAL_API_CONFIG['enabled']:
        # 使用外部API
        if 'left_images' not in request.files or 'right_images' not in request.files:
            return jsonify({'error': 'Both left and right image sequences are required'}), 400
        
        left_files = request.files.getlist('left_images')
        right_files = request.files.getlist('right_images')
        
        if len(left_files) != len(right_files):
            return jsonify({'error': 'Left and right image sequences must have the same length'}), 400
        
        files = {}
        for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
            files[f'left_image_{i}'] = (left_file.filename, left_file.stream, left_file.mimetype)
            files[f'right_image_{i}'] = (right_file.filename, right_file.stream, right_file.mimetype)
        
        return jsonify(call_external_api('track', files=files))
    else:
        # 使用本地模型
        if 'left_images' not in request.files or 'right_images' not in request.files:
            return jsonify({'error': 'Both left and right image sequences are required'}), 400
        
        left_files = request.files.getlist('left_images')
        right_files = request.files.getlist('right_images')
        
        if len(left_files) != len(right_files):
            return jsonify({'error': 'Left and right image sequences must have the same length'}), 400
        
        # 创建临时文件列表
        temp_left_files = []
        temp_right_files = []
        
        try:
            # 保存所有图像文件
            for left_file, right_file in zip(left_files, right_files):
                temp_left = f"/tmp/{uuid.uuid4()}.png"
                temp_right = f"/tmp/{uuid.uuid4()}.png"
                
                left_file.save(temp_left)
                right_file.save(temp_right)
                
                temp_left_files.append(temp_left)
                temp_right_files.append(temp_right)
            
            # 执行追踪
            result = model.track_objects(temp_left_files, temp_right_files)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Object tracking error: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # 清理临时文件
            for temp_file in temp_left_files + temp_right_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查 | Health check"""
    return jsonify({
        'status': 'healthy',
        'model': 'F_spatial',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
