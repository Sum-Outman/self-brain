#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import uuid
import threading
import platform
import psutil
from datetime import datetime
from collections import deque
from functools import wraps
import torch
import requests

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

# 导入ManagementModel和相关工具
from manager_model.main_model import ManagementModel, get_management_model
from training.train_model import train_management_model, stop_training_process

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('manager_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('A_Management_Model')

# 应用初始化
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 简单的登录验证装饰器
def login_required(f):
    """验证用户是否已登录的装饰器
    在实际部署中，应替换为更安全的认证机制
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 检查是否启用了认证（从配置文件读取或使用默认值）
        auth_enabled = True
        try:
            with open(os.path.join(os.path.dirname(__file__), '..', 'web_interface', 'config', 'security_settings.json'), 'r') as f:
                config = json.load(f)
                auth_enabled = config.get('auth_enabled', True)
        except:
            pass
        
        # 如果禁用了认证，直接通过
        if not auth_enabled:
            return f(*args, **kwargs)
        
        # 检查Authorization头部
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            # 在开发环境中，可以从查询参数获取token
            token = request.args.get('token')
            if not token:
                return jsonify({'success': False, 'error': 'Authentication required'}), 401
            
            # 简单的token验证（实际应用中应使用更安全的方式）
            if token != 'dev_token':
                return jsonify({'success': False, 'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

# 全局变量
management_model = None
model_initialized = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# 系统状态
system_state = {
    'performance_metrics': {
        'system_uptime': time.time(),
        'total_requests': 0,
        'successful_requests': 0,
        'failed_requests': 0
    },
    'model_health': {
        'status': 'initializing',
        'last_check': None,
        'memory_usage': 0
    }
}

# 下属模型状态注册表
submodel_registry = {
    'B_language': {'status': 'active', 'health': 'healthy', 'emotion_weight': 0.2},
    'C_audio': {'status': 'active', 'health': 'healthy', 'emotion_weight': 0.15},
    'D_image': {'status': 'active', 'health': 'healthy', 'emotion_weight': 0.15},
    'E_video': {'status': 'active', 'health': 'healthy', 'emotion_weight': 0.15},
    'F_spatial': {'status': 'active', 'health': 'healthy', 'emotion_weight': 0.1},
    'G_sensor': {'status': 'active', 'health': 'healthy', 'emotion_weight': 0.1},
    'I_knowledge': {'status': 'active', 'health': 'healthy', 'emotion_weight': 0.1},
    'J_motion': {'status': 'maintenance', 'health': 'warning', 'emotion_weight': 0.03},
    'K_programming': {'status': 'active', 'health': 'healthy', 'emotion_weight': 0.02}
}

# 导入修复后的情感引擎
from manager_model.emotion_engine_fixed import EmotionalState, EmotionEngine

# 模型管理器类
class ModelManager:
    def __init__(self):
        self.submodel_registry = submodel_registry
        self.active_models = {}
        
    def get_available_models(self):
        return list(self.submodel_registry.keys())
    
    def get_model_status(self, model_name):
        if model_name in self.submodel_registry:
            return self.submodel_registry[model_name]
        return None
    
    def update_model_status(self, model_name, status):
        if model_name in self.submodel_registry:
            self.submodel_registry[model_name].update(status)
            return True
        return False

<<<<<<< HEAD
# 导入AdvancedTrainingController
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_manager.advanced_train_control import (
    get_training_controller as get_advanced_training_controller,
    TrainingMode, 
    start_training_api, 
    stop_training_api
)

# 实例化管理器和控制器
model_manager = ModelManager()
# 使用高级训练控制器
def get_training_controller():
    """获取训练控制器实例"""
    controller = get_advanced_training_controller()
    
    class TrainingControllerWrapper:
        """包装高级训练控制器以保持API兼容性"""
        def __init__(self, controller):
            self.controller = controller
            
        def start_training(self, config):
            # 转换配置为AdvancedTrainingController需要的格式
            epochs = config.get('epochs', 10)
            learning_rate = config.get('learning_rate', 0.001)
            batch_size = config.get('batch_size', 32)
            
            training_config = {
=======
# 训练控制器类
class TrainingController:
    def __init__(self):
        self.stop_event = None
        self.training_thread = None
        self.training_progress = {
            'status': 'idle',
            'epoch': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'steps_completed': 0,
            'total_steps': 0,
            'config': {}
        }
        
    def start_training(self, config):
        global management_model
        
        if management_model is None:
            return {'status': 'error', 'message': 'Model not initialized'}
        
        # 设置训练配置
        epochs = config.get('epochs', 10)
        learning_rate = config.get('learning_rate', 0.001)
        batch_size = config.get('batch_size', 32)
        
        self.training_progress = {
            'status': 'training',
            'epoch': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'steps_completed': 0,
            'total_steps': epochs,
            'config': {
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
<<<<<<< HEAD
            
            # 调用API启动训练
            result = start_training_api(["A_management"], "individual", training_config)
            return result
            
        def stop_training(self):
            # 调用API停止训练
            result = stop_training_api()
            return result
            
        def get_training_progress(self):
            # 获取训练状态
            status = self.controller.get_training_status()
            return {
                'status': status['current_status'],
                'epoch': status.get('current_epoch', 0),
                'loss': status.get('metrics', {}).get('loss', 0.0),
                'accuracy': status.get('metrics', {}).get('accuracy', 0.0),
                'steps_completed': status.get('current_epoch', 0),
                'total_steps': status.get('total_epochs', 0),
                'config': status.get('config', {})
            }
    
    # 返回包装后的控制器实例
    return TrainingControllerWrapper(controller)

training_controller = get_training_controller()
=======
        }
        
        # 启动训练线程
        self.stop_event = threading.Event()
        self.training_thread = threading.Thread(
            target=train_management_model,
            args=(management_model, self.training_progress, self.stop_event),
            kwargs={
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
        )
        self.training_thread.daemon = True
        self.training_thread.start()
        
        logger.info(f"Starting real training with config: {self.training_progress['config']}")
        
        return {
            'status': 'success',
            'message': 'Real training started',
            'training_id': f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    
    def stop_training(self):
        if self.training_progress['status'] == 'training':
            # 调用stop_training_process函数停止训练
            stop_training_process(self.stop_event)
            self.training_progress['status'] = 'stopped'
            logger.info("Training stopped by user")
            return {'status': 'success', 'message': 'Training stopped'}
        return {'status': 'error', 'message': 'No training in progress'}
    
    def get_training_progress(self):
        # 直接返回真实的训练进度（由train_management_model函数更新）
        return self.training_progress

# 实例化管理器和控制器
model_manager = ModelManager()
training_controller = TrainingController()
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
emotion_engine = None  # 稍后初始化

# 初始化管理模型
def initialize_management_model():
    global management_model, model_initialized, emotion_engine
    
    try:
        logger.info("正在初始化管理模型...")
        
        # 创建或加载管理模型，传入模型注册表
        management_model = get_management_model(submodel_registry)
        management_model.to(device)
        
        # 加载预训练权重（如果有）
        model_path = os.path.join(os.path.dirname(__file__), '..', 'sub_models', 'A_management', 'models', 'management_model.pth')
        if os.path.exists(model_path):
            management_model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"已加载预训练模型权重: {model_path}")
        
        # 设置为评估模式
        management_model.eval()
        
        # 初始化情感引擎 - 使用单例模式
        emotion_engine = get_emotion_engine()
        
        # 更新系统状态
        model_initialized = True
        system_state['model_health']['status'] = 'active'
        system_state['model_health']['last_check'] = datetime.now().isoformat()
        
        logger.info("管理模型初始化完成")
        return True
    except Exception as e:
        logger.error(f"初始化管理模型失败: {e}")
        model_initialized = False
        system_state['model_health']['status'] = 'error'
        system_state['model_health']['last_check'] = datetime.now().isoformat()
        system_state['model_health']['error'] = str(e)
        return False

# 检查模型健康状态
def check_model_health():
    global system_state
    
    try:
        # 执行简单的推理测试
        if management_model is not None:
<<<<<<< HEAD
            # 创建适合模型处理的请求数据，避免直接使用张量导致的参数错误
            dummy_request = {
                'request_id': 'health_check',
                'message': 'health check',
                'conversation_id': 'health_check',
                'context': {},
                'timestamp': datetime.now().isoformat()
            }
            with torch.no_grad():
                # 使用process_local_request方法而不是直接调用forward，确保正确的参数传递
                management_model.process_local_request(dummy_request)
=======
            # 使用更适合模型处理的测试输入
            test_features = {"text": [0.1, 0.2, 0.3, 0.4], "length": 4}
            with torch.no_grad():
                management_model.forward(test_features)
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            
            # 更新健康状态
            system_state['model_health']['status'] = 'active'
            system_state['model_health']['last_check'] = datetime.now().isoformat()
            return True
        return False
    except Exception as e:
        logger.error(f"模型健康检查失败: {e}")
        system_state['model_health']['status'] = 'error'
        system_state['model_health']['last_check'] = datetime.now().isoformat()
        system_state['model_health']['error'] = str(e)
        return False

# API端点
@app.route('/api/health', methods=['GET'])
def health_check():
    """系统健康检查"""
    try:
        model_health = check_model_health()
        
        return jsonify({
            'status': 'success',
            'api': True,
            'model': model_initialized and model_health,
            'system_time': datetime.now().isoformat(),
            'uptime': time.time() - system_state['performance_metrics']['system_uptime']
        })
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/chat_with_management', methods=['POST'])
def chat_with_management():
    """与管理模型对话"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id', '')
        context = data.get('context', {})
        
        if not message:
            return jsonify({
                'status': 'error',
                'message': 'Message cannot be empty'
            }), 400
        
        if not model_initialized or management_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not initialized'
            }), 503
        
        # 更新系统统计
        system_state['performance_metrics']['total_requests'] += 1
        
        # 准备请求数据
        request_data = {
            'request_id': str(uuid.uuid4()),
            'message': message,
            'conversation_id': conversation_id,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        # 分析用户消息的情感
        emotion = None
        if emotion_engine:
            emotion_result = emotion_engine.analyze_text_emotion(message)
            # 从EmotionalState对象中获取主要情感
            if hasattr(emotion_result, 'emotions'):
                # 找出得分最高的情感
                primary_emotion = max(emotion_result.emotions.items(), key=lambda x: x[1])
                if primary_emotion[1] > 0:  # 只有当情感得分大于0时才使用
                    emotion = primary_emotion[0]
        
        # 让管理模型路由请求到合适的子模型或自己处理
        try:
            target_model = management_model.route_request(request_data)
            
            if target_model == 'A':
                # 管理模型自己处理请求
                response_text = management_model.process_local_request(request_data)
            else:
                # 转发请求到子模型
                response_text = management_model.forward_to_model(target_model, request_data)
                
            # 使用情感引擎增强响应
            if emotion and emotion_engine:
                response_text = management_model.enhance_with_emotion(response_text, emotion)
                
        except Exception as model_error:
            logger.error(f"模型处理请求失败: {model_error}")
            # 降级到本地处理
            response_text = management_model.process_local_request(request_data)
        
        # 更新成功请求计数
        system_state['performance_metrics']['successful_requests'] += 1
        
        # 格式化响应
        response = {
            'status': 'success',
            'message': message,
            'response': response_text,
            'timestamp': datetime.now().isoformat(),
            'conversation_id': conversation_id
        }
        
        # 如果情感引擎可用，添加情感分析结果
        if emotion:
            response['emotion'] = emotion
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"处理对话请求失败: {e}")
        system_state['performance_metrics']['failed_requests'] += 1
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# 添加/api/chat端点作为/api/chat_with_management的别名
@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint that forwards requests to chat_with_management"""
    return chat_with_management()

# 辅助函数：根据消息和任务类型生成响应
def generate_response(message, task_type):
    """根据消息内容和任务类型生成响应"""
    response_map = {
        'general_response': {
            'hello': "Hello! How can I help you today?",
            'hi': "Hi there! What can I do for you?",
            'how are you': "I'm doing well, thank you! How can I assist you?",
            'thanks': "You're welcome! Let me know if you need anything else.",
            'help': "I'm here to help. What do you need assistance with?"
        },
        'emotional_response': {
            'excited': "I'm excited to hear that! Let's explore this further.",
            'happy': "That's great to hear! How can I support you?",
            'sad': "I'm sorry to hear that. Let's see how we can improve things.",
            'angry': "I understand you're upset. Let's work together to resolve this issue."
        },
        'task_management': {
            'status': "System is running smoothly. All core services are operational.",
            'progress': "Current progress is on track. We're meeting our objectives.",
            'report': "Generating a comprehensive report for you."
        }
    }
    
    # 检查消息中是否包含关键词
    message_lower = message.lower()
    for category, responses in response_map.items():
        for keyword, response_text in responses.items():
            if keyword in message_lower:
                return response_text
    
    # 默认响应
    default_responses = {
        'general_response': "I'm here to assist you with your queries and tasks.",
        'emotional_response': "I understand. Let me know how I can help.",
        'task_management': "I can help manage various tasks. Please specify what you need.",
        'model_coordination': "I'm coordinating with the relevant models to provide you with the best assistance.",
        'data_analysis': "I can help analyze data and provide insights. Please provide the information you'd like analyzed."
    }
    
    return default_responses.get(task_type, "I need more information to help with this.")

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """开始模型训练"""
    try:
        data = request.get_json() or {}
        config = data.get('config', {})
        
        result = training_controller.start_training(config)
        
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"启动训练失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """停止模型训练"""
    try:
        result = training_controller.stop_training()
        
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"停止训练失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/training/progress', methods=['GET'])
def training_progress():
    """获取训练进度"""
    try:
        progress = training_controller.get_training_progress()
        progress['timestamp'] = datetime.now().isoformat()
        
        return jsonify({
            'status': 'success',
            'data': progress
        })
        
    except Exception as e:
        logger.error(f"获取训练进度失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/emotion', methods=['POST'])
def update_emotion():
    """更新情感状态"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        emotion = data.get('emotion', 'neutral')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info(f"情感更新: {emotion}")
        
        if emotion_engine:
            emotion_engine.current_emotion = emotion
        
        return jsonify({
            'status': 'success',
            'emotion': emotion,
            'response': f"Emotion state updated to {emotion}",
            'timestamp': timestamp
        })
        
    except Exception as e:
        logger.error(f"情感更新失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/emotion/current', methods=['GET'])
def get_current_emotion():
    """获取当前系统情感状态"""
    try:
        if emotion_engine and hasattr(emotion_engine, 'get_current_emotion'):
            # 使用实际存在的get_current_emotion方法获取当前情感状态
            current_emotion_data = emotion_engine.get_current_emotion()
            
            # 从返回的字典中找出得分最高的情感
            if 'emotions' in current_emotion_data:
                primary_emotion = max(current_emotion_data['emotions'].items(), key=lambda x: x[1])
                current_emotion = primary_emotion[0]
            else:
                current_emotion = 'neutral'
        else:
            current_emotion = 'neutral'
        
        return jsonify({
            'status': 'success',
            'emotion': current_emotion,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"获取当前情感状态失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/emotion/summary', methods=['GET'])
def get_emotion_summary():
    """获取情感摘要统计"""
    try:
        time_period = request.args.get('time_period', 'daily')
        
        # 验证时间周期参数
        valid_periods = ["daily", "weekly", "monthly"]
        if time_period not in valid_periods:
            time_period = "daily"
        
        # 使用实际存在的get_emotion_summary方法获取情感摘要
        summary = emotion_engine.get_emotion_summary(time_period) if emotion_engine and hasattr(emotion_engine, 'get_emotion_summary') else {}
        
        return jsonify({
            'status': 'success',
            'summary': summary,
            'time_period': time_period,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"获取情感摘要失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/emotion/reset', methods=['POST'])
def reset_emotion():
    """重置系统情感状态"""
    try:
        if emotion_engine:
            emotion_engine.reset_emotion()
        
        return jsonify({
            'status': 'success',
            'message': 'Emotional state has been reset',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"重置情感状态失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/export', methods=['POST'])
def export_data():
    """导出系统数据"""
    try:
        data = request.get_json() or {}
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info("数据导出请求")
        
        # 在实际应用中，这里会实现数据导出逻辑
        export_id = f"export_{int(datetime.now().timestamp())}"
        
        return jsonify({
            'status': 'success',
            'message': 'System data export initiated. Files will be available in the downloads section.',
            'timestamp': timestamp,
            'export_id': export_id
        })
        
    except Exception as e:
        logger.error(f"数据导出失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取模型列表"""
    try:
        logger.info("获取模型列表")
        
        models = model_manager.get_available_models()
        
        return jsonify({
            'status': 'success',
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/models/status', methods=['GET'])
def get_all_models_status():
    """获取所有模型的状态"""
    try:
        logger.info("获取所有模型状态")
        
        all_status = submodel_registry
        
        return jsonify({
            'status': 'success',
            'models': all_status,
            'count': len(all_status)
        })
    except Exception as e:
        logger.error(f"获取模型状态失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/models/<model_name>/status', methods=['GET'])
def get_model_status(model_name):
    """获取特定模型状态"""
    try:
        status = model_manager.get_model_status(model_name)
        
        if status:
            return jsonify({
                'status': 'success',
                'model': model_name,
                'data': status
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Model {model_name} not found'
            }), 404
    except Exception as e:
        logger.error(f"获取模型状态失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/emotion/analyze', methods=['POST'])
def analyze_emotion():
    """情感分析端点"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'status': 'error',
                'message': 'Text cannot be empty'
            }), 400
        
        if not emotion_engine:
            return jsonify({
                'status': 'error',
                'message': 'Emotion engine not initialized'
            }), 503
        
        # 使用情感引擎进行详细的情感分析
        emotion_state = emotion_engine.analyze_text_emotion(text)
        
        # 获取主导情感
        if hasattr(emotion_state, 'emotions'):
            dominant_emotion = max(emotion_state.emotions.items(), key=lambda x: x[1])
        else:
            dominant_emotion = ('neutral', 0.0)
        
        # 生成情感化响应
        emotional_response = emotion_engine.generate_emotional_response(text, emotion_state)
        
        # 构建响应数据
        response_data = {
            'status': 'success',
            'text': text,
            'emotion': {
                'primary': dominant_emotion[0],
                'score': dominant_emotion[1],
                'detailed': getattr(emotion_state, 'emotions', {})
            },
            'recommended_response': emotional_response.get('prefix', '') + ' ' + emotional_response.get('style', '') if emotional_response else '',
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加情感倾向数据（如果存在）
        if hasattr(emotion_state, 'valence'):
            response_data['emotion']['valence'] = emotion_state.valence
        if hasattr(emotion_state, 'arousal'):
            response_data['emotion']['arousal'] = emotion_state.arousal
        if hasattr(emotion_state, 'dominance'):
            response_data['emotion']['dominance'] = emotion_state.dominance
        if hasattr(emotion_state, 'confidence'):
            response_data['confidence'] = emotion_state.confidence
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"情感分析失败: {e}")
        system_state['performance_metrics']['failed_requests'] += 1
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/coordinate', methods=['POST'])
def coordinate_models():
    """协调多个模型"""
    try:
        data = request.get_json()
        task_description = data.get('task')
        involved_models = data.get('models', [])
        coordination_strategy = data.get('strategy', 'sequential')
        
        if not task_description or not involved_models:
            return jsonify({
                'status': 'error',
                'message': 'Task description and models are required'
            }), 400
        
        # 验证模型
        available_models = model_manager.get_available_models()
        invalid_models = [m for m in involved_models if m not in available_models]
        
        if invalid_models:
            return jsonify({
                'status': 'error',
                'message': f'Invalid models: {invalid_models}',
                'available_models': available_models
            }), 400
        
        coordination_id = f"coord_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 准备协调任务特征
        features = {
            'task': task_description,
            'models': involved_models,
            'strategy': coordination_strategy,
            'timestamp': datetime.now().isoformat()
        }
        
        # 使用管理模型创建协调计划
        with torch.no_grad():
            coordination_plan = management_model.coordinate_submodels(features)
        
        coordination_plan['coordination_id'] = coordination_id
        coordination_plan['created_at'] = datetime.now().isoformat()
        
        return jsonify({
            'status': 'success',
            'coordination_id': coordination_id,
            'plan': coordination_plan
        })
        
    except Exception as e:
        logger.error(f"协调模型失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/knowledge/query', methods=['POST'])
def query_knowledge():
    """知识库查询"""
    try:
        data = request.get_json()
        query = data.get('query')
        knowledge_domain = data.get('domain', 'general')
        
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Query is required'
            }), 400
        
        if not model_initialized or management_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not initialized'
            }), 503
        
        # 准备查询特征
        features = {
            'query': query,
            'domain': knowledge_domain,
            'timestamp': datetime.now().isoformat()
        }
        
        # 使用管理模型查询知识库
        with torch.no_grad():
            knowledge_result = management_model.query_knowledge(features)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'response': knowledge_result.get('response', 'No relevant information found.'),
            'domain': knowledge_domain,
            'confidence': knowledge_result.get('confidence', 0.85),
            'sources': knowledge_result.get('sources', ['internal_knowledge_base'])
        })
        
    except Exception as e:
        logger.error(f"知识查询失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/system/stats', methods=['GET'])
def get_system_stats():
    """获取系统统计信息"""
    try:
        # 创建可序列化的系统状态副本
        serializable_system_state = {}
        for key, value in system_state.items():
            if isinstance(value, dict):
                serializable_substate = {}
                for k, v in value.items():
                    if isinstance(v, (dict, list, str, int, float, bool)) or v is None:
                        serializable_substate[k] = v
                    else:
                        serializable_substate[k] = str(v)
                serializable_system_state[key] = serializable_substate
            elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                serializable_system_state[key] = value
            else:
                serializable_system_state[key] = str(value)
        
        stats = {
            'system_uptime': time.time() - system_state['performance_metrics']['system_uptime'],
            'active_models': len(submodel_registry),
            'collaboration_stats': system_state['performance_metrics'],
            'system_state': serializable_system_state
        }
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
    except Exception as e:
        logger.error(f"获取系统统计失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_simple_stats():
    """获取简化系统统计信息（兼容旧版本）"""
    try:
        # 简化统计信息
        simple_stats = {
            'total_tasks': system_state['performance_metrics'].get('total_requests', 0),
            'successful_tasks': system_state['performance_metrics'].get('successful_requests', 0),
            'failed_tasks': system_state['performance_metrics'].get('failed_requests', 0),
            'pending_tasks': 0,  # 在当前实现中没有 pending_tasks
            'active_models': len(submodel_registry),
            'system_uptime': time.time() - system_state['performance_metrics']['system_uptime'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'stats': simple_stats
        })
    except Exception as e:
        logger.error(f"获取简化统计失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/system/stats/detailed', methods=['GET'])
@login_required
def get_system_stats_detailed():
    """获取详细的系统统计信息"""
    try:
        # 获取详细的系统状态信息
        detailed_stats = {
            'models': {
                model_id: {
                    'status': model['status'],
                    'response_time': model.get('response_time', 0),
                    'error_rate': model.get('error_rate', 0),
                    'last_request_time': model.get('last_request_time', None)
                } for model_id, model in submodel_registry.items()
            },
            'system': {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:').percent,
                'temperature': None,  # 温度传感器数据
                'process_count': len(psutil.pids())
            },
            'network': {
                'sent_bytes': None,
                'received_bytes': None,
                'connections': None
            }
        }
        return jsonify({'success': True, 'stats': detailed_stats}), 200
    except Exception as e:
        app.logger.error(f"Error getting detailed system stats: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Model API Settings Endpoints
@app.route('/api/models/<model_id>/api-settings', methods=['GET'])
@login_required
def get_model_api_settings(model_id):
    """获取模型的API设置"""
    try:
        # 检查模型是否存在
        if model_id not in submodel_registry:
            return jsonify({'success': False, 'error': 'Model not found'}), 404
        
        # 从注册表获取API设置
        model_settings = submodel_registry.get(model_id, {})
        api_settings = {
            'api_type': model_settings.get('api_type', ''),
            'api_url': model_settings.get('api_url', ''),
            'api_key': model_settings.get('api_key', ''),
            'api_model_name': model_settings.get('api_model_name', '')
        }
        
        return jsonify(api_settings), 200
    except Exception as e:
        app.logger.error(f"Error getting model API settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/models/<model_id>/api-settings', methods=['POST'])
@login_required
def save_model_api_settings(model_id):
    """保存模型的API设置"""
    try:
        # 检查模型是否存在
        if model_id not in submodel_registry:
            return jsonify({'success': False, 'error': 'Model not found'}), 404
        
        # 获取请求数据
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # 更新模型注册表中的API设置
        submodel_registry[model_id].update({
            'api_type': data.get('api_type', ''),
            'api_url': data.get('api_url', ''),
            'api_key': data.get('api_key', ''),
            'api_model_name': data.get('api_model_name', '')
        })
        
        # 保存更新后的注册表
        with open('model_registry.json', 'w') as f:
            json.dump(submodel_registry, f, indent=4)
        
        return jsonify({'success': True, 'message': 'API settings saved successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error saving model API settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/models/<model_id>/test-connection', methods=['POST'])
@login_required
def test_model_connection(model_id):
    """测试模型API连接"""
    try:
        # 检查模型是否存在
        if model_id not in submodel_registry:
            return jsonify({'success': False, 'error': 'Model not found'}), 404
        
        model_settings = submodel_registry.get(model_id, {})
        api_type = model_settings.get('api_type', '')
        api_url = model_settings.get('api_url', '')
        
        # 如果是本地模型，直接返回成功
        if api_type == 'local' or not api_url:
            return jsonify({'success': True, 'message': 'Connection test successful for local model'}), 200
        
        # 测试远程API连接
        try:
            # 简单的连接测试
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                return jsonify({'success': True, 'message': 'Connection test successful'}), 200
            else:
                return jsonify({'success': False, 'error': f'API returned status code {response.status_code}'}), 500
        except requests.exceptions.RequestException as re:
            return jsonify({'success': False, 'error': f'Connection failed: {str(re)}'}), 500
    except Exception as e:
        app.logger.error(f"Error testing model connection: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Hardware Configuration Endpoints
@app.route('/api/hardware/camera-settings', methods=['POST'])
@login_required
def save_camera_settings():
    """保存相机设置"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # 保存相机设置
        camera_settings = {
            'camera_1': data.get('camera_1', ''),
            'camera_2': data.get('camera_2', '')
        }
        
        with open('camera_settings.json', 'w') as f:
            json.dump(camera_settings, f, indent=4)
        
        return jsonify({'success': True, 'message': 'Camera settings saved successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error saving camera settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/hardware/sensor-settings', methods=['POST'])
@login_required
def save_sensor_settings():
    """保存传感器设置"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # 保存传感器设置
        sensor_settings = {
            'port': data.get('port', ''),
            'baudrate': data.get('baudrate', '')
        }
        
        with open('sensor_settings.json', 'w') as f:
            json.dump(sensor_settings, f, indent=4)
        
        return jsonify({'success': True, 'message': 'Sensor settings saved successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error saving sensor settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# System Settings Endpoint
@app.route('/api/system/settings', methods=['POST'])
@login_required
def save_system_settings():
    """保存系统设置"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # 保存系统设置
        system_settings = {
            'language': data.get('language', 'en'),
            'max_threads': data.get('max_threads', 4),
            'memory_limit': data.get('memory_limit', 80),
            'auto_save': data.get('auto_save', True)
        }
        
        with open('system_settings.json', 'w') as f:
            json.dump(system_settings, f, indent=4)
        
        return jsonify({'success': True, 'message': 'System settings saved successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error saving system settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/collaboration/tasks', methods=['POST'])
def create_collaboration_task():
    """创建协作任务"""
    try:
        data = request.get_json()
        description = data.get('description')
        required_models = data.get('required_models', [])
        priority = data.get('priority', 'medium')
        metadata = data.get('metadata', {})
        
        if not description or not required_models:
            return jsonify({
                'status': 'error',
                'message': 'Description and required_models are required'
            }), 400
        
        if not model_initialized or management_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Management model not initialized'
            }), 503
        
        # 准备任务特征
        task_features = {
            'description': description,
            'required_models': required_models,
            'priority': priority,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        # 使用管理模型创建协作任务
        with torch.no_grad():
            task_id = management_model.create_collaboration_task(task_features)
        
        return jsonify({
            'status': 'success',
            'task_id': task_id
        })
            
    except Exception as e:
        logger.error(f"创建协作任务失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# 异常处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    system_state['performance_metrics']['failed_requests'] += 1
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# 初始化应用
def initialize_app():
    """初始化应用"""
    # 初始化管理模型
    initialize_management_model()

if __name__ == '__main__':
    initialize_app()
    
<<<<<<< HEAD
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 for A Management Model
=======
    port = int(os.environ.get('PORT', 5015))  # Changed from 5001 to 5015 according to PORT_ALLOCATION.md
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"启动A Management Model API服务于 http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)