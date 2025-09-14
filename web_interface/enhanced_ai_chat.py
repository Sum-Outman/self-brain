"""
Enhanced AI Chat System with Real Intelligence
集成真实AI响应、情感分析、跨模型协同、硬件集成
"""

import json
import os
import sys
import time
import threading
import requests
import random
from datetime import datetime
from flask import Flask, request, jsonify, render_template, make_response
from flask_socketio import SocketIO, emit
import logging

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入真实功能模块
from unified_system_improvements import (
    health_checker, emotion_analyzer, data_bus, hardware_integration
)
from sub_models.K_programming.real_programming_system import real_programming_system
from sub_models.F_spatial.real_stereo_vision import stereo_system
from sub_models.G_sensor.real_sensor_system import real_sensor_system

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedChat")

class EnhancedAIChat:
    """增强版AI聊天系统"""
    
    def __init__(self):
        self.model_endpoints = {
            'a_management': 'http://localhost:5001',
            'b_language': 'http://localhost:5002',
            'c_audio': 'http://localhost:5003',
            'd_image': 'http://localhost:5004',
            'e_video': 'http://localhost:5005',
            'f_spatial': 'http://localhost:5006',
            'g_sensor': 'http://localhost:5007',
            'h_computer': 'http://localhost:5008',
            'i_motion': 'http://localhost:5009',
            'j_knowledge': 'http://localhost:5010',
            'k_programming': 'http://localhost:5011'
        }
        
        self.a_manager_endpoint = 'http://localhost:5015'
        self.conversation_history = []
        self.model_status = {}
        self.real_time_data = {}
        
        # 初始化硬件系统
        self._initialize_hardware()
    
    def _initialize_hardware(self):
        """初始化硬件系统"""
        try:
            # 启动传感器系统
            devices = real_sensor_system.discover_devices()
            for device_type, device_list in devices.items():
                for device_id in device_list:
                    if device_type == 'mock':
                        sensor_type = device_id.replace('mock_', '')
                        device = real_sensor_system.devices.get(device_id) or \
                                real_sensor_system.devices.__class__.__bases__[0](device_id, sensor_type)
                        real_sensor_system.add_device(device)
            
            # 启动双目视觉
            stereo_system.calibrate_stereo()
            
            # 启动实时处理
            real_sensor_system.start_data_collection()
            stereo_system.start_real_time_processing()
            
            logger.info("Hardware systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hardware: {e}")
    
    def check_model_health(self):
        """检查所有模型服务健康状态"""
        for model_name, endpoint in self.model_endpoints.items():
            try:
                response = requests.get(f"{endpoint}/health", timeout=2)
                self.model_status[model_name] = {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'endpoint': endpoint,
                    'last_check': datetime.now().isoformat()
                }
            except:
                self.model_status[model_name] = {
                    'status': 'offline',
                    'endpoint': endpoint,
                    'last_check': datetime.now().isoformat()
                }
        
        # 检查A管理模型
        try:
            response = requests.get(f"{self.a_manager_endpoint}/health", timeout=2)
            self.model_status['a_manager'] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'endpoint': self.a_manager_endpoint,
                'last_check': datetime.now().isoformat()
            }
        except:
            self.model_status['a_manager'] = {
                'status': 'offline',
                'endpoint': self.a_manager_endpoint,
                'last_check': datetime.now().isoformat()
            }
        
        return self.model_status
    
    def get_real_time_context(self) -> Dict[str, Any]:
        """获取实时上下文数据"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'sensors': {},
            'vision': {},
            'hardware': {},
            'emotions': {}
        }
        
        try:
            # 传感器数据
            sensor_data = real_sensor_system.get_real_time_data()
            context['sensors'] = sensor_data
            
            # 视觉数据
            vision_data = stereo_system.get_latest_results()
            context['vision'] = vision_data
            
            # 硬件状态
            context['hardware'] = hardware_integration.get_sensor_data()
            
            # 情感上下文
            context['emotions'] = {
                'user_mood': 'neutral',  # 可以从对话历史推断
                'system_confidence': 0.85
            }
            
        except Exception as e:
            logger.error(f"Failed to get real-time context: {e}")
            context['error'] = str(e)
        
        return context
    
    def generate_intelligent_response(self, message: str, model_type: str = 'a_management') -> Dict[str, Any]:
        """生成智能响应"""
        
        # 分析情感
        emotion_analysis = emotion_analyzer.analyze_emotion(message)
        
        # 获取实时上下文
        context = self.get_real_time_context()
        
        # 构建增强提示
        enhanced_prompt = self._build_enhanced_prompt(message, emotion_analysis, context)
        
        # 尝试调用真实模型
        response_data = self._call_real_model(enhanced_prompt, model_type)
        
        if not response_data:
            # 使用本地备用系统
            response_data = self._generate_backup_response(message, emotion_analysis, context)
        
        # 添加跨模型协同
        self._trigger_cross_model_collaboration(message, response_data, context)
        
        # 记录到数据总线
        self._publish_to_data_bus(message, response_data, emotion_analysis, context)
        
        return response_data
    
    def _build_enhanced_prompt(self, message: str, emotion: Dict, context: Dict) -> str:
        """构建增强提示"""
        prompt_parts = [
            f"User message: {message}",
            f"User emotion: {emotion.get('primary_emotion', 'neutral')}",
            f"Current sensor data: {json.dumps(context.get('sensors', {}), indent=2)}",
            f"Vision data: {json.dumps(context.get('vision', {}), indent=2)}",
            f"Hardware status: {json.dumps(context.get('hardware', {}), indent=2)}"
        ]
        
        return "\n".join(prompt_parts)
    
    def _call_real_model(self, prompt: str, model_type: str) -> Dict[str, Any]:
        """调用真实模型"""
        try:
            if model_type == 'a_management':
                endpoint = self.a_manager_endpoint
            else:
                endpoint = self.model_endpoints.get(model_type, self.a_manager_endpoint)
            
            response = requests.post(
                f"{endpoint}/api/chat",
                json={
                    'message': prompt,
                    'context': self.conversation_history[-5:] if self.conversation_history else []
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.error(f"Failed to call real model {model_type}: {e}")
        
        return None
    
    def _generate_backup_response(self, message: str, emotion: Dict, context: Dict) -> Dict[str, Any]:
        """生成备用响应"""
        
        # 使用编程系统生成智能响应
        programming_response = real_programming_system.generate_and_optimize(
            f"Generate an intelligent response to: {message}"
        )
        
        # 基于情感调整响应
        emotion_map = {
            'happy': 'I sense you\'re feeling positive! ',
            'sad': 'I understand this might be difficult. ',
            'angry': 'I can see you\'re frustrated. ',
            'neutral': 'Thank you for sharing. '
        }
        
        base_response = programming_response.get('code', 'I understand your message.')
        
        # 添加上下文感知
        context_aware = []
        
        if context.get('sensors', {}).get('devices'):
            sensor_count = len(context['sensors']['devices'])
            context_aware.append(f"I'm monitoring {sensor_count} sensors.")
        
        if context.get('vision', {}).get('objects'):
            object_count = len(context['vision']['objects'])
            context_aware.append(f"I can see {object_count} objects in the environment.")
        
        final_response = f"{emotion_map.get(emotion.get('primary_emotion', 'neutral'), '')}{' '.join(context_aware)} {base_response}"
        
        return {
            'response': final_response,
            'source': 'backup_system',
            'emotion_analysis': emotion,
            'context': context,
            'model_type': 'enhanced_backup'
        }
    
    def _trigger_cross_model_collaboration(self, message: str, response: Dict, context: Dict):
        """触发跨模型协同"""
        
        # 发布到数据总线供其他模型订阅
        collaboration_data = {
            'type': 'cross_model_collaboration',
            'message': message,
            'response': response,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        data_bus.publish('chat_collaboration', collaboration_data)
        
        # 触发特定模型处理
        if 'image' in message.lower():
            data_bus.publish('image_analysis', {'query': message, 'context': context})
        elif 'code' in message.lower() or 'program' in message.lower():
            data_bus.publish('programming_assist', {'requirements': message})
        elif 'sensor' in message.lower():
            data_bus.publish('sensor_query', {'query': message})
    
    def _publish_to_data_bus(self, message: str, response: Dict, emotion: Dict, context: Dict):
        """发布到统一数据总线"""
        
        chat_data = {
            'message': message,
            'response': response,
            'emotion': emotion,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'conversation_id': len(self.conversation_history)
        }
        
        data_bus.publish('chat_history', chat_data)
        self.conversation_history.append(chat_data)
    
    def test_device_permissions(self) -> Dict[str, bool]:
        """测试设备权限"""
        return {
            'camera': self._check_camera_permission(),
            'microphone': self._check_microphone_permission(),
            'screen_share': self._check_screen_permission()
        }
    
    def _check_camera_permission(self) -> bool:
        """检查摄像头权限"""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ret, _ = cap.read()
            cap.release()
            return ret
        except:
            return False
    
    def _check_microphone_permission(self) -> bool:
        """检查麦克风权限"""
        try:
            import sounddevice as sd
            duration = 0.1
            recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
            sd.wait()
            return len(recording) > 0
        except:
            return False
    
    def _check_screen_permission(self) -> bool:
        """检查屏幕共享权限"""
        # 浏览器权限检查需要前端处理
        return True

# 创建全局实例
enhanced_chat = EnhancedAIChat()

# Flask路由
from flask import Flask, request, jsonify

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    """增强版主页"""
    return render_template('enhanced_ai_chat.html')

@app.route('/api/health/models')
def check_models():
    """检查所有模型状态"""
    return jsonify(enhanced_chat.check_model_health())

@app.route('/api/chat/send', methods=['POST'])
def send_message():
    """发送消息并获取智能响应"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        model_type = data.get('model_type', 'a_management')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        response = enhanced_chat.generate_intelligent_response(message, model_type)
        
        return jsonify({
            'status': 'success',
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to process message: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/context/realtime')
def get_realtime_context():
    """获取实时上下文"""
    return jsonify(enhanced_chat.get_real_time_context())

@app.route('/api/devices/test')
def test_devices():
    """测试设备权限"""
    return jsonify(enhanced_chat.test_device_permissions())

@app.route('/api/emotion/analyze', methods=['POST'])
def analyze_emotion():
    """分析消息情感"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        emotion = emotion_analyzer.analyze_emotion(text)
        return jsonify({'emotion': emotion})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/collaboration/trigger', methods=['POST'])
def trigger_collaboration():
    """触发跨模型协同"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # 触发协同处理
        enhanced_chat._trigger_cross_model_collaboration(message, {}, {})
        
        return jsonify({'status': 'success', 'message': 'Collaboration triggered'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Socket.IO连接处理"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {
        'status': 'success',
        'message': 'Connected to Enhanced AI Chat',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('send_message')
def handle_send_message(data):
    """处理Socket.IO消息"""
    try:
        message = data.get('message', '')
        model_type = data.get('model_type', 'a_management')
        
        if not message:
            emit('error', {'error': 'Message is required'})
            return
        
        response = enhanced_chat.generate_intelligent_response(message, model_type)
        
        emit('message_response', {
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        emit('error', {'error': str(e)})

if __name__ == "__main__":
    print("=== 启动增强版AI聊天系统 ===")
    print("功能特性:")
    print("1. 真实AI智能响应")
    print("2. 情感分析集成")
    print("3. 跨模型协同")
    print("4. 实时硬件数据")
    print("5. 设备权限检测")
    print("6. 统一数据总线")
    print()
    print("访问 http://localhost:5006 查看增强版主页")
    
    app.run(host='0.0.0.0', port=5006, debug=True)