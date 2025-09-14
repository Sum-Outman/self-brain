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

app = Flask(__name__)
model = SpatialModel()

@app.route('/localize', methods=['POST'])
def localize():
    """空间定位API | Spatial localization API"""
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
    # 实现运动物体追踪API
    # Implement moving object tracking API
    return jsonify({'status': 'not implemented'}), 501

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006)
