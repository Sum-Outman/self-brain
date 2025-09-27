#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立运行的A_management模型API服务 - 使用端口5014
"""

import json
import os
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class AManager:
    def __init__(self):
        self.models = [
            'A_management', 'B_language', 'C_vision', 'D_audio',
            'E_reasoning', 'F_emotion', 'G_sensor', 'H_computer_control',
            'I_knowledge', 'J_motion', 'K_programming'
        ]
        self.task_count = 0
        
    def process_message(self, message: str, task_type: str = 'general') -> dict:
        self.task_count += 1
        
        responses = {
            'general': f"你好！我收到了你的消息：'{message}'。作为A_management模型，我将为你提供智能服务。",
            'programming': f"我分析了你的编程问题：'{message}'。我可以帮助你解决Python、JavaScript等编程问题。",
            'knowledge': f"关于你的知识查询：'{message}'。让我为你提供相关的知识和信息。",
            'creative': f"我收到了你的创意请求：'{message}'。我可以为你生成创意内容。",
            'analysis': f"我分析了你的问题：'{message}'。让我为你提供深入的分析和建议。"
        }
        
        response = responses.get(task_type, responses['general'])
        
        return {
            'response': response,
            'task_id': f"task_{int(time.time() * 1000)}",
            'models_used': ['A_management'],
            'processing_time': 0.3,
            'confidence': 0.9,
            'timestamp': datetime.now().isoformat(),
            'original_message': message,
            'task_type': task_type
        }

manager = AManager()

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'A_Manager_Standalone',
        'timestamp': datetime.now().isoformat(),
        'uptime': time.time()
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': manager.models,
        'count': len(manager.models),
        'status': 'success'
    })

@app.route('/process_message', methods=['POST'])
def process_message():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'message is required'}), 400
            
        message = data.get('message')
        task_type = data.get('task_type', 'general')
        
        result = manager.process_message(message, task_type)
        return jsonify({**result, 'status': 'success'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotion/analyze', methods=['POST'])
def analyze_emotion():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'text is required'}), 400
            
        # 简单情感分析
        positive = sum(1 for word in ['good', 'happy', 'great', 'love', 'excellent'] if word in text.lower())
        negative = sum(1 for word in ['bad', 'sad', 'terrible', 'hate', 'awful'] if word in text.lower())
        
        if positive > negative:
            emotion, score = 'positive', 0.8
        elif negative > positive:
            emotion, score = 'negative', 0.8
        else:
            emotion, score = 'neutral', 0.5
            
        return jsonify({
            'status': 'success',
            'emotion': emotion,
            'score': score,
            'text': text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'status': 'success',
        'stats': {
            'total_tasks': manager.task_count,
            'active_models': len(manager.models),
            'system_uptime': time.time(),
            'timestamp': datetime.now().isoformat()
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5014))
    print(f"A Manager Standalone running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)