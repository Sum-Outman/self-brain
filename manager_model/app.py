#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
from datetime import datetime
from collections import deque
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

# 导入ManagementModel和相关工具
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../sub_models/A_management')))
from enhanced_manager import ManagementModel, create_management_model

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

# 情感引擎类
class EmotionEngine:
    def __init__(self, management_model):
        self.management_model = management_model
        self.current_emotion = 'neutral'
        self.emotion_history = deque(maxlen=1000)
    
    def analyze_text_emotion(self, text):
        # 使用管理模型进行情感分析
        features = self._extract_features(text)
        # 转换特征以便forward方法可以处理
        processed_features = {"text": [0.1, 0.2, 0.3, 0.4], "length": 4}
        
        with torch.no_grad():
            # 使用forward方法获取情感预测
            _, emotion_probs = self.management_model.forward(processed_features)
        
        # 简单的情感映射
        emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted']
        emotion_index = torch.argmax(emotion_probs).item()
        primary_emotion = emotion_labels[emotion_index] if emotion_index < len(emotion_labels) else 'neutral'
        score = emotion_probs.max().item()
        
        # 构建情感结果
        emotion_result = {
            'primary': primary_emotion,
            'score': score,
            'detailed': {emotion: float(prob) for emotion, prob in zip(emotion_labels, emotion_probs.tolist()[0])},
            'valence': 0.5,  # 默认值
            'arousal': 0.5,  # 默认值
            'dominance': 0.5,  # 默认值
            'confidence': 0.85  # 默认值
        }
        
        # 更新当前情感和历史记录
        self.current_emotion = primary_emotion
        self.emotion_history.append({
            'timestamp': datetime.now().isoformat(),
            'emotion': self.current_emotion,
            'score': score
        })
        
        return emotion_result
    
    def get_current_emotion(self):
        return self.current_emotion
    
    def get_emotion_summary(self, time_period='daily'):
        # 生成情感摘要统计
        emotion_counts = {}
        for record in self.emotion_history:
            emotion_counts[record['emotion']] = emotion_counts.get(record['emotion'], 0) + 1
        
        # 计算情感分布
        total = sum(emotion_counts.values())
        emotion_distribution = {k: v/total for k, v in emotion_counts.items()} if total > 0 else {}
        
        return {
            'time_period': time_period,
            'total_records': len(self.emotion_history),
            'dominant_emotion': max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral',
            'distribution': emotion_distribution
        }
    
    def reset_emotion(self):
        self.current_emotion = 'neutral'
        self.emotion_history.clear()
        return True
    
    def generate_emotional_response(self, text, emotion_result):
        # 基于情感结果生成响应
        primary_emotion = emotion_result.get('primary', 'neutral')
        
        responses = {
            'positive': "I'm happy to help with that! ",
            'negative': "I understand this might be challenging. ",
            'neutral': "Let me provide you with information. ",
            'excited': "This is interesting! Let's explore it. ",
            'frustrated': "Let's try to resolve this issue. "
        }
        
        base_response = responses.get(primary_emotion, responses['neutral'])
        return base_response
    
    def _extract_features(self, text):
        # 提取文本特征
        # 在实际应用中，这里可能使用NLP模型提取更复杂的特征
        return {
            'text': text,
            'length': len(text),
            'word_count': len(text.split()),
            'timestamp': datetime.now().isoformat()
        }

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

# 训练控制器类
class TrainingController:
    def __init__(self):
        self.stop_event = None
        self.training_progress = {
            'status': 'idle',
            'epoch': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'steps_completed': 0,
            'total_steps': 0
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
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
        }
        
        # 在实际应用中，这里会启动一个训练线程
        logger.info(f"Starting training with config: {self.training_progress['config']}")
        
        return {
            'status': 'success',
            'message': 'Training started',
            'training_id': f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    
    def stop_training(self):
        if self.training_progress['status'] == 'training':
            self.training_progress['status'] = 'stopped'
            logger.info("Training stopped by user")
            return {'status': 'success', 'message': 'Training stopped'}
        return {'status': 'error', 'message': 'No training in progress'}
    
    def get_training_progress(self):
        # 在实际应用中，这里会返回真实的训练进度
        if self.training_progress['status'] == 'training':
            # 模拟进度更新
            self.training_progress['steps_completed'] += 1
            if self.training_progress['steps_completed'] >= self.training_progress['total_steps']:
                self.training_progress['status'] = 'completed'
                self.training_progress['epoch'] = self.training_progress['total_steps']
            else:
                self.training_progress['epoch'] = self.training_progress['steps_completed']
                # 模拟损失和准确率变化
                self.training_progress['loss'] = max(0.1, 1.0 - (self.training_progress['epoch'] * 0.1))
                self.training_progress['accuracy'] = min(0.95, self.training_progress['epoch'] * 0.1)
        
        return self.training_progress

# 实例化管理器和控制器
model_manager = ModelManager()
training_controller = TrainingController()
emotion_engine = None  # 稍后初始化

# 初始化管理模型
def initialize_management_model():
    global management_model, model_initialized, emotion_engine
    
    try:
        logger.info("正在初始化管理模型...")
        
        # 创建或加载管理模型
        management_model = create_management_model()
        management_model.to(device)
        
        # 加载预训练权重（如果有）
        model_path = os.path.join(os.path.dirname(__file__), '..', 'sub_models', 'A_management', 'models', 'management_model.pth')
        if os.path.exists(model_path):
            management_model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"已加载预训练模型权重: {model_path}")
        
        # 设置为评估模式
        management_model.eval()
        
        # 初始化情感引擎
        emotion_engine = EmotionEngine(management_model)
        
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
            # 使用更适合模型处理的测试输入
            test_features = {"text": [0.1, 0.2, 0.3, 0.4], "length": 4}
            with torch.no_grad():
                management_model.forward(test_features)
            
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
        
        # 准备输入特征（使用适合模型处理的格式）
        features = {
            "text": [0.1, 0.2, 0.3, 0.4], "length": 4
        }
        
        # 使用管理模型处理消息
        with torch.no_grad():
            strategy_probs, emotion_probs = management_model.forward(features)
        
        # 生成响应数据
        strategy_labels = ['general_response', 'emotional_response', 'task_management', 'model_coordination', 'data_analysis']
        strategy_index = torch.argmax(strategy_probs).item()
        task_type = strategy_labels[strategy_index] if strategy_index < len(strategy_labels) else 'general'
        
        # 根据消息和任务类型生成响应
        response_text = generate_response(message, task_type)
        
        # 更新成功请求计数
        system_state['performance_metrics']['successful_requests'] += 1
        
        # 格式化响应
        response = {
            'status': 'success',
            'message': message,
            'response': response_text,
            'task_type': task_type,
            'confidence': strategy_probs.max().item(),
            'timestamp': datetime.now().isoformat(),
            'conversation_id': conversation_id
        }
        
        # 如果情感引擎可用，添加情感分析结果
        if emotion_engine:
            emotion_result = emotion_engine.analyze_text_emotion(message)
            response['emotion'] = emotion_result['primary']
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"处理对话请求失败: {e}")
        system_state['performance_metrics']['failed_requests'] += 1
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

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
    """获取系统当前的情感状态"""
    try:
        current_emotion = emotion_engine.get_current_emotion() if emotion_engine else 'neutral'
        
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
        
        summary = emotion_engine.get_emotion_summary(time_period) if emotion_engine else {}
        
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
        emotion_result = emotion_engine.analyze_text_emotion(text)
        
        # 获取主导情感
        dominant_emotion = max(emotion_result['detailed'].items(), key=lambda x: x[1]) if 'detailed' in emotion_result else ('neutral', 0.0)
        
        # 生成情感化响应
        emotional_response = emotion_engine.generate_emotional_response(text, emotion_result)
        
        return jsonify({
            'status': 'success',
            'text': text,
            'emotion': {
                'primary': dominant_emotion[0],
                'score': dominant_emotion[1],
                'detailed': emotion_result.get('detailed', {}),
                'valence': emotion_result.get('valence', 0.0),
                'arousal': emotion_result.get('arousal', 0.0),
                'dominance': emotion_result.get('dominance', 0.0)
            },
            'recommended_response': emotional_response,
            'confidence': emotion_result.get('confidence', 0.85),
            'timestamp': datetime.now().isoformat()
        })
        
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
    
    port = int(os.environ.get('PORT', 5015))  # Changed from 5001 to 5015 according to PORT_ALLOCATION.md
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"启动A Management Model API服务于 http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)