#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Management Model - Self Brain AGI Core Manager
管理模型：具有情感分析能力的主模型，负责协调所有下属模型

Features:
- 情感分析与表达
- 多模型任务调度
- 实时系统监控
- Web界面交互
- 语音/文字/图像反馈
- 外部API集成
"""

import os
import json
import time
import torch
import asyncio
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests
import websockets
import numpy as np
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import speech_recognition as sr
import cv2
from PIL import Image
import io
import base64

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionalState:
    """情感状态管理器"""
    
    def __init__(self):
        self.emotions = {
            'joy': 0.0, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0,
            'surprise': 0.0, 'disgust': 0.0, 'trust': 0.0, 'anticipation': 0.0
        }
        self.emotion_history = []
        
    def analyze_text_emotion(self, text: str) -> Dict[str, float]:
        """分析文本情感"""
        try:
            # 使用预训练情感分析模型
            emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            results = emotion_analyzer(text)[0]
            emotion_scores = {item['label']: item['score'] for item in results}
            
            # 更新当前情感状态
            self.update_emotions(emotion_scores)
            return emotion_scores
            
        except Exception as e:
            logger.warning(f"情感分析失败，使用备用方案: {e}")
            return self._fallback_emotion_analysis(text)
    
    def _fallback_emotion_analysis(self, text: str) -> Dict[str, float]:
        """备用情感分析"""
        text_lower = text.lower()
        
        # 定义关键词
        positive_words = ['good', 'great', 'excellent', 'happy', 'joy', 'love', 'like', 'wonderful', 'amazing', 'perfect']
        negative_words = ['bad', 'terrible', 'sad', 'hate', 'dislike', 'awful', 'horrible', 'pain', 'suffer']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {'joy': 0.8, 'trust': 0.6, 'anticipation': 0.4}
        elif negative_count > positive_count:
            return {'sadness': 0.7, 'anger': 0.5, 'fear': 0.3}
        else:
            return {'neutral': 0.8, 'anticipation': 0.3}
    
    def update_emotions(self, new_emotions: Dict[str, float]):
        """更新情感状态"""
        for emotion, score in new_emotions.items():
            if emotion in self.emotions:
                # 情感衰减和更新
                self.emotions[emotion] = self.emotions[emotion] * 0.9 + score * 0.1
                self.emotions[emotion] = max(0.0, min(1.0, self.emotions[emotion]))
        
        # 记录情感历史
        self.emotion_history.append({
            'timestamp': datetime.now().isoformat(),
            'emotions': self.emotions.copy()
        })
    
    def get_dominant_emotion(self) -> str:
        """获取主导情感"""
        return max(self.emotions, key=self.emotions.get) if self.emotions else "neutral"
    
    def express_emotion(self) -> Dict[str, Any]:
        """表达当前情感状态"""
        dominant = self.get_dominant_emotion()
        return {
            'current_emotions': self.emotions,
            'dominant_emotion': dominant,
            'intensity': self.emotions.get(dominant, 0.0),
            'expression': self._get_emotion_expression(dominant)
        }
    
    def _get_emotion_expression(self, emotion: str) -> str:
        """获取情感表达文本"""
        expressions = {
            'joy': "I feel happy and positive about this!",
            'sadness': "This makes me feel a bit down.",
            'anger': "I'm concerned about this situation.",
            'fear': "This seems a bit worrying.",
            'surprise': "That's unexpected!",
            'disgust': "That doesn't seem right.",
            'trust': "I feel confident about this.",
            'anticipation': "I'm looking forward to what's next."
        }
        return expressions.get(emotion, "I'm processing this information.")

class SubModelManager:
    """下属模型管理器"""
    
    def __init__(self):
        self.models = {
            'B_language': {'port': 5016, 'status': 'ready', 'type': 'nlp'},
            'C_audio': {'port': 5017, 'status': 'ready', 'type': 'audio'},
            'D_image': {'port': 5018, 'status': 'ready', 'type': 'vision'},
            'E_video': {'port': 5019, 'status': 'ready', 'type': 'video'},
            'F_spatial': {'port': 5020, 'status': 'ready', 'type': 'spatial'},
            'G_sensor': {'port': 5021, 'status': 'ready', 'type': 'sensor'},
            'H_computer': {'port': 5022, 'status': 'ready', 'type': 'control'},
            'I_motion': {'port': 5023, 'status': 'ready', 'type': 'motion'},
            'J_knowledge': {'port': 5024, 'status': 'ready', 'type': 'knowledge'},
            'K_programming': {'port': 5025, 'status': 'ready', 'type': 'programming'}
        }
        self.model_performance = {}
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """获取模型状态"""
        if model_id not in self.models:
            return {'error': f'Model {model_id} not found'}
        
        try:
            # 检查模型服务状态
            response = requests.get(f"http://localhost:{self.models[model_id]['port']}/health", timeout=5)
            status = 'online' if response.status_code == 200 else 'offline'
            
            return {
                'model_id': model_id,
                'status': status,
                'port': self.models[model_id]['port'],
                'type': self.models[model_id]['type'],
                'last_check': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    def assign_task(self, model_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """分配任务给下属模型"""
        if model_id not in self.models:
            return {'error': f'Model {model_id} not available'}
        
        try:
            # 发送任务到下属模型
            response = requests.post(
                f"http://localhost:{self.models[model_id]['port']}/task",
                json=task,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.model_performance[model_id] = {
                    'last_task': task,
                    'completion_time': datetime.now().isoformat(),
                    'status': 'completed'
                }
                return result
            else:
                return {'error': f'Task failed: {response.status_code}'}
                
        except Exception as e:
            return {'error': str(e)}

class AManagementSystem:
    """A管理模型主系统"""
    
    def __init__(self):
        self.emotional_state = EmotionalState()
        self.model_manager = SubModelManager()
        self.system_status = {
            'started_at': datetime.now().isoformat(),
            'total_requests': 0,
            'successful_tasks': 0,
            'failed_tasks': 0
        }
        self.app = Flask(__name__)
        self._setup_routes()
        
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/health')
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'started_at': self.system_status['started_at'],
                'uptime': str(datetime.now() - datetime.fromisoformat(self.system_status['started_at'])),
                'total_requests': self.system_status['total_requests']
            })
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            """与A管理模型对话"""
            try:
                data = request.get_json()
                message = data.get('message', '').strip()
                
                if not message:
                    return jsonify({'error': 'Message is required'}), 400
                
                # 更新系统统计
                self.system_status['total_requests'] += 1
                
                # 情感分析
                emotion_analysis = self.emotional_state.analyze_text_emotion(message)
                
                # 任务解析和分配
                task_result = self._parse_and_assign_task(message)
                
                # 生成响应
                response = self._generate_response(message, emotion_analysis, task_result)
                
                # 更新情感状态
                self.emotional_state.update_emotions(emotion_analysis)
                
                self.system_status['successful_tasks'] += 1
                
                return jsonify({
                    'response': response,
                    'emotion_analysis': emotion_analysis,
                    'task_result': task_result,
                    'emotional_state': self.emotional_state.express_emotion(),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.system_status['failed_tasks'] += 1
                logger.error(f"Chat error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/emotion/analyze', methods=['POST'])
        def analyze_emotion():
            """情感分析"""
            try:
                data = request.get_json()
                text = data.get('text', '')
                
                if not text:
                    return jsonify({'error': 'Text is required'}), 400
                
                emotion_result = self.emotional_state.analyze_text_emotion(text)
                return jsonify({
                    'emotion_analysis': emotion_result,
                    'dominant_emotion': self.emotional_state.get_dominant_emotion(),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/models/status')
        def get_models_status():
            """获取所有模型状态"""
            status = {}
            for model_id in self.model_manager.models.keys():
                status[model_id] = self.model_manager.get_model_status(model_id)
            return jsonify(status)
        
        @self.app.route('/api/models/<model_id>/task', methods=['POST'])
        def assign_model_task(model_id):
            """分配任务给指定模型"""
            try:
                task = request.get_json()
                result = self.model_manager.assign_task(model_id, task)
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _parse_and_assign_task(self, message: str) -> Dict[str, Any]:
        """解析消息并分配任务"""
        message_lower = message.lower()
        
        # 任务关键词映射
        task_mapping = {
            'language': ['translate', 'language', 'text', 'write', 'summarize'],
            'audio': ['audio', 'sound', 'speech', 'voice', 'listen'],
            'image': ['image', 'picture', 'photo', 'visual', 'see'],
            'video': ['video', 'movie', 'clip', 'record', 'stream'],
            'spatial': ['space', '3d', 'location', 'distance', 'position'],
            'sensor': ['sensor', 'temperature', 'humidity', 'data', 'measure'],
            'computer': ['computer', 'system', 'control', 'execute', 'run'],
            'motion': ['motion', 'move', 'motor', 'robot', 'actuator'],
            'knowledge': ['knowledge', 'information', 'research', 'learn', 'understand'],
            'programming': ['code', 'program', 'develop', 'debug', 'software']
        }
        
        assigned_tasks = []
        
        for model_type, keywords in task_mapping.items():
            if any(keyword in message_lower for keyword in keywords):
                model_id = f"{model_type.upper()[0]}_{model_type.lower()}"
                if model_id in self.model_manager.models:
                    task = {
                        'type': 'process',
                        'input': message,
                        'model_type': model_type,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    result = self.model_manager.assign_task(model_id, task)
                    assigned_tasks.append({
                        'model_id': model_id,
                        'task': task,
                        'result': result
                    })
        
        return {
            'assigned_tasks': assigned_tasks,
            'task_count': len(assigned_tasks)
        }
    
    def _generate_response(self, message: str, emotion_analysis: Dict, task_result: Dict) -> str:
        """生成响应"""
        dominant_emotion = max(emotion_analysis, key=emotion_analysis.get) if emotion_analysis else "neutral"
        
        # 根据情感和任务结果生成响应
        task_count = task_result.get('task_count', 0)
        
        response_templates = {
            'joy': f"I'm glad to help! I've assigned {task_count} tasks to the appropriate models.",
            'sadness': f"I understand your concern. I've processed your request through {task_count} models.",
            'anger': f"I acknowledge your frustration. I've initiated {task_count} processes to address this.",
            'neutral': f"I've analyzed your request and assigned {task_count} tasks to relevant models."
        }
        
        base_response = response_templates.get(dominant_emotion, response_templates['neutral'])
        
        # 添加情感表达
        emotion_expression = self.emotional_state._get_emotion_expression(dominant_emotion)
        
        return f"{base_response} {emotion_expression}"
    
    def run(self, host='0.0.0.0', port=5015, debug=False):
        """启动A管理模型服务"""
        logger.info(f"Starting A Management Model on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    # 启动A管理模型
    manager = AManagementSystem()
    manager.run()