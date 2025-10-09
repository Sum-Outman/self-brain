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

# 计算机控制模型API服务
# Computer Control Model API Service

from flask import Flask, request, jsonify
from .model import ComputerControlModel
import os

app = Flask(__name__)
model = ComputerControlModel()

@app.route('/execute', methods=['POST'])
def execute_command():
    """执行系统命令API | Execute system command API"""
    data = request.json
    command = data.get('command')
    
    if not command:
        return jsonify({'error': 'Command is required'}), 400
    
    try:
        result = model.execute_command(command)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/manage_process', methods=['POST'])
def manage_process():
    """管理进程API | Manage process API"""
    data = request.json
    process_name = data.get('process_name')
    action = data.get('action')  # start, stop, restart, status
    
    if not process_name or not action:
        return jsonify({'error': 'Process name and action are required'}), 400
    
    try:
        result = model.manage_process(process_name, action)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/configure', methods=['POST'])
def configure_system():
    """系统配置API | System configuration API"""
    data = request.json
    config = data.get('config')
    
    if not config:
        return jsonify({'error': 'Configuration data is required'}), 400
    
    try:
        result = model.system_configuration(config)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/file_operation', methods=['POST'])
def file_operation():
    """文件操作API | File operation API"""
    data = request.json
    operation = data.get('operation')  # create, read, update, delete
    path = data.get('path')
    content = data.get('content', None)
    
    if not operation or not path:
        return jsonify({'error': 'Operation and path are required'}), 400
    
    try:
        result = model.file_operation(operation, path, content)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/network', methods=['POST'])
def network_operation():
    """网络操作API | Network operation API"""
    data = request.json
    operation = data.get('operation')  # ping, scan, connect
    target = data.get('target')  # IP, hostname, URL
    
    if not operation or not target:
        return jsonify({'error': 'Operation and target are required'}), 400
    
    try:
        result = model.network_operation(operation, target)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006)
