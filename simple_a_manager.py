#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版A_management模型API服务 - 使用端口5002
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("A_Manager")

# 创建Flask应用
app = Flask(__name__)
CORS(app)

class AManager:
    """A_management模型"""
    
    def __init__(self):
        self.models = {
            'A_management': '主管理模型',
            'B_language': '语言处理模型',
            'C_vision': '视觉处理模型',
            'D_audio': '音频处理模型',
            'E_reasoning': '推理模型',
            'F_emotion': '情感分析模型',
            'G_sensor': '传感器模型',
            'H_computer_control': '计算机控制模型',
            'I_knowledge': '知识库模型',
            'J_motion': '运动控制模型',
            'K_programming': '编程模型'
        }
        self.task_count = 0
    
    def process_message(self, message: str, task_type: str = 'general') -> Dict[str, Any]:
        """处理消息"""
        self.task_count += 1
        
        # 根据任务类型生成响应
        responses = {
            'general': f"收到你的消息: {message}。我将为你提供帮助。",
            'programming': f"关于编程问题: {message}。我可以帮你解决编程相关问题。",
            'knowledge': f"知识查询: {message}。让我为你查找相关信息。",
            'creative': f"创意请求: {message}。我可以为你生成创意内容。",
            'analysis': f"分析请求: {message}。让我为你分析这个问题。"
        }
        
        response = responses.get(task_type, responses['general'])
        
        return {
            'response': response,
            'task_id': f"task_{self.task_count}",
            'models_used': ['A_management'],
            'processing_time': 0.5,
            'confidence': 0.9,
            'timestamp': datetime.now().isoformat()
        }

# 创建全局实例
manager = AManager()

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'service': 'A_Manager',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取模型列表"""
    return jsonify({
        'models': list(manager.models.keys()),
        'count': len(manager.models)
    })

@app.route('/process_message', methods=['POST'])
def process_message():
    """处理消息"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        task_type = data.get('task_type', 'general')
        
        if not message:
            return jsonify({'error': 'message is required'}), 400
        
        result = manager.process_message(message, task_type)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotion/analyze', methods=['POST'])
def analyze_emotion():
    """情感分析"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'text is required'}), 400
        
        # 简单情感分析
        positive_words = ['good', 'happy', 'great', 'wonderful', 'love', 'excellent']
        negative_words = ['bad', 'sad', 'terrible', 'hate', 'awful', 'horrible']
        
        positive = sum(1 for word in positive_words if word in text.lower())
        negative = sum(1 for word in negative_words if word in text.lower())
        
        if positive > negative:
            emotion = 'positive'
            score = 0.8
        elif negative > positive:
            emotion = 'negative'
            score = 0.8
        else:
            emotion = 'neutral'
            score = 0.5
        
        return jsonify({
            'emotion': emotion,
            'score': score,
            'text': text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = 5002  # 使用不同的端口
    print(f"Starting A Manager on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)