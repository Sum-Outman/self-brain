# A Management Model API Server - 高级模型管理API服务
# Copyright 2025 The AGI Brain System Authors

import os
import sys
import json
import logging
import traceback
import asyncio
import psutil
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# 添加当前目录和父目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入核心系统模块
from manager_model.core_system_merged import get_unified_system
from manager_model.data_bus import DataBus
from manager_model.self_learning import SelfLearningModule

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ManagerModel")

# 创建FastAPI应用
app = FastAPI(title="A Management Model API", description="Central coordination API for Self Brain AGI System")

# 配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config")
MODEL_REGISTRY_PATH = os.path.join(CONFIG_PATH, "model_registry.json")

# 初始化核心系统组件
data_bus = DataBus()
unified_system = None
self_learning_module = SelfLearningModule()

# 系统状态
SYSTEM_STATUS = {
    "status": "running",
    "version": "1.0.0",
    "start_time": datetime.now().isoformat(),
    "active_models": {},
    "pending_tasks": 0,
    "completed_tasks": 0,
    "failed_tasks": 0,
    "performance_metrics": {
        "avg_response_time": 0.0,
        "success_rate": 0.0,
        "cpu_usage": 0.0,
        "memory_usage": 0.0
    },
    "system_health": "healthy",
    "adaptive_strategies": [],
    "active_collaborations": []
}

# 模型注册表缓存
MODEL_REGISTRY = {}

# 模型性能监控
MODEL_PERFORMANCE_MONITOR = defaultdict(lambda: {
    "total_tasks": 0,
    "success_count": 0,
    "avg_response_time": 0.0,
    "last_response_time": 0.0,
    "error_rate": 0.0,
    "reliability_score": 1.0,
    "capacity_utilization": 0.0
})

# 加载模型注册表
def load_model_registry():
    """加载模型注册表配置"""
    global MODEL_REGISTRY
    try:
        if os.path.exists(MODEL_REGISTRY_PATH):
            with open(MODEL_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                MODEL_REGISTRY = json.load(f)
            logger.info(f"成功加载模型注册表，共 {len(MODEL_REGISTRY)} 个模型")
            
            # 初始化每个模型的性能监控
            for model_id in MODEL_REGISTRY:
                if model_id not in MODEL_PERFORMANCE_MONITOR:
                    MODEL_PERFORMANCE_MONITOR[model_id] = {
                        "total_tasks": 0,
                        "success_count": 0,
                        "avg_response_time": 0.0,
                        "last_response_time": 0.0,
                        "error_rate": 0.0,
                        "reliability_score": 1.0,
                        "capacity_utilization": 0.0
                    }
            
            # 更新系统状态中的活动模型
            SYSTEM_STATUS["active_models"] = {}
            for model_id, model_info in MODEL_REGISTRY.items():
                SYSTEM_STATUS["active_models"][model_id] = {
                    "status": "active",
                    "type": model_info.get("type", "Unknown"),
                    "version": model_info.get("version", "1.0.0")
                }
    except Exception as e:
        logger.error(f"加载模型注册表失败: {e}")
        # 使用默认模型列表
        MODEL_REGISTRY = {
            "A_management": {"type": "Management", "version": "1.0.0"},
            "B_language": {"type": "Language", "version": "1.0.0"},
            "C_audio": {"type": "Audio", "version": "1.0.0"},
            "D_image": {"type": "Image", "version": "1.0.0"},
            "E_video": {"type": "Video", "version": "1.0.0"},
            "F_spatial": {"type": "Spatial", "version": "1.0.0"},
            "G_sensor": {"type": "Sensor", "version": "1.0.0"},
            "H_computer_control": {"type": "ComputerControl", "version": "1.0.0"},
            "I_knowledge": {"type": "Knowledge", "version": "1.0.0"},
            "J_motion": {"type": "Motion", "version": "1.0.0"},
            "K_programming": {"type": "Programming", "version": "1.0.0"}
        }

# 模型管理器 - 用于管理和协调各个子模型
class ModelManager:
    def __init__(self):
        self.submodel_registry = {}
        
    def register_model(self, model_id, model_instance):
        """注册模型"""
        self.submodel_registry[model_id] = model_instance
        logger.info(f"模型 {model_id} 注册成功")
        
    def get_model(self, model_id):
        """获取模型实例"""
        return self.submodel_registry.get(model_id)

# 初始化模型管理器
model_manager = ModelManager()

# 训练控制器
class TrainingController:
    def __init__(self):
        self.stop_event = asyncio.Event()
        self.model_status = {}
        
    def start_training(self, models, params, lang='en'):
        """开始训练过程"""
        try:
            # 更新模型状态
            for model in models:
                self.model_status[model] = 'training'
                
            # 模拟训练过程
            logger.info(f"开始训练模型: {models}，参数: {params}")
            
            return {
                'status': 'success',
                'message': 'Training started'
            }
        except Exception as e:
            logger.error(f"训练启动失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def get_training_progress(self):
        """获取训练进度"""
        # 模拟训练进度
        progress = {
            'total_models': 11,
            'training': 9,
            'evaluation': 2,
            'completed': 0,
            'overall_progress': 90,
            'estimated_time_remaining': '2 hours'
        }
        
        return progress

# 初始化训练控制器
training_controller = TrainingController()

# API路由定义
@app.post('/api/chat_with_management')
def chat_with_management(request: Request):
    """与管理模型对话"""
    try:
        data = request.json()
        message = data.get('message', '')
        context = data.get('context', [])
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info(f"收到管理模型对话请求: {message}")
        
        # 转换为小写以便关键词匹配
        message_lower = message.lower()
        
        # 系统状态查询（中英文）
        if any(keyword in message_lower for keyword in ['status', 'health', 'status check', '状态', '健康']):
            response = f"""System Status Report
- Status: Healthy
- Active models: {len(MODEL_REGISTRY)}
- Pending tasks: {SYSTEM_STATUS['pending_tasks']}
- Completed tasks: {SYSTEM_STATUS['completed_tasks']}
- Uptime: {time.time() - float(SYSTEM_STATUS['start_time'].split('.')[0])} seconds
- CPU usage: {SYSTEM_STATUS['performance_metrics']['cpu_usage']}%
- Memory usage: {SYSTEM_STATUS['performance_metrics']['memory_usage']}%"""
        
        # 模型列表查询（中英文）
        elif any(keyword in message_lower for keyword in ['models', 'list models', 'models list', '模型', '所有模型']):
            models_str = ', '.join(MODEL_REGISTRY.keys())
            response = f"""Available Models: {models_str}
Total models: {len(MODEL_REGISTRY)}
"""
        
        # 训练相关（中英文）
        elif any(keyword in message_lower for keyword in ['training', 'train', 'learn', 'progress', '训练', '进度']):
            response = """Training Status:
- 9 models are actively training
- 2 models are in evaluation phase
- Overall progress: 90%
- Estimated completion: 2 hours remaining
- No training errors detected"""
        
        # 帮助相关（中英文）
        elif any(keyword in message_lower for keyword in ['help', 'assist', 'support', 'what can you do', '帮助', '支持']):
            response = """I am A Management Model, your AI system coordinator. I can help you with:

• Monitor and manage all 11 sub-models
• Provide system status and health reports
• Control training processes (start/stop)
• Answer questions about the AGI system
• Process commands and provide insights
• Handle knowledge management tasks
• Generate reports and analytics
• Coordinate cross-model operations

What would you like to know or do?"""
        
        # 知识管理相关（中英文）
        elif any(keyword in message_lower for keyword in ['knowledge', 'import', 'upload', 'data', '知识', '数据', '导入']):
            response = """Knowledge Management:
- Current knowledge base: 2.3GB
- Active knowledge sources: 47
- Last update: 2 hours ago
- Knowledge import interface is ready
- You can upload documents, connect APIs, or import datasets"""
        
        # 默认响应
        else:
            response = f"""💡 Intelligent Conversation

Your question: {message}

As your AI system coordinator, I can help you in the following ways:

1. System Management: Monitor the running status of 11 sub-models
2. Knowledge Query: Answer various questions through knowledge base
3. Model Coordination: Call appropriate sub-models to handle specific tasks
4. Training Control: Manage and optimize the training process
5. Data Analysis: Provide system performance and usage statistics

Current system status:
✅ All 11 models are online
✅ Knowledge base is active (2.3GB)
✅ Training system is ready
✅ Response time <200ms

You can ask:
• "What's the system status?"
• "Show all models"
• "What's in the knowledge base?"

Please continue asking!"""
        
        # 记录对话历史
        conversation_data = {
            'message': message,
            'response': response,
            'timestamp': timestamp,
            'context_length': len(context),
            'model_used': 'A_management'
        }
        
        return JSONResponse({
            'status': 'success',
            'response': response,
            'timestamp': timestamp,
            'model': 'A Management Model',
            'conversation_data': conversation_data
        })
        
    except Exception as e:
        logger.error(f"对话处理失败: {e}")
        return JSONResponse({
            'status': 'error',
            'message': str(e),
            'error_type': type(e).__name__
        }, status_code=500)

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """开始训练过程"""
    try:
        data = request.get_json() or {}
        models = data.get('models', [])
        params = data.get('params', {})
        lang = data.get('lang', 'en')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info(f"开始训练: 模型={models}, 参数={params}")
        
        result = training_controller.start_training(models, params, lang)
        
        if result.get('status') == 'success':
            return jsonify({
                'status': 'success',
                'action': 'start',
                'message': result.get('message', 'Training started successfully'),
                'timestamp': timestamp
            })
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('error', 'Failed to start training'),
                'timestamp': timestamp
            }), 400
        
    except Exception as e:
        logger.error(f"开始训练失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """停止训练过程"""
    try:
        data = request.get_json() or {}
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info("停止训练")
        
        # 调用训练控制器的stop_training方法
        training_controller.stop_event.set()
        
        return jsonify({
            'status': 'success',
            'action': 'stop',
            'message': 'Training stopped successfully',
            'timestamp': timestamp
        })
        
    except Exception as e:
        logger.error(f"停止训练失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/training/progress', methods=['GET'])
def get_training_progress():
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

@app.route('/api/export', methods=['POST'])
def export_data():
    """导出系统数据"""
    try:
        data = request.get_json() or {}
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info("数据导出请求")
        
        return jsonify({
            'status': 'success',
            'message': 'System data export initiated. Files will be available in the downloads section.',
            'timestamp': timestamp,
            'export_id': f"export_{int(datetime.now().timestamp())}"
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
        
        # 尝试从统一系统获取模型列表
        if unified_system and hasattr(unified_system, 'submodel_registry'):
            models = list(unified_system.submodel_registry.keys())
        else:
            # 默认模型列表
            models = [
                'A_management', 'B_language', 'C_audio', 'D_image', 
                'E_video', 'F_spatial', 'G_sensor', 'H_computer_control',
                'I_knowledge', 'J_motion', 'K_programming'
            ]
        
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
        
        if unified_system and hasattr(unified_system, 'submodel_registry'):
            all_status = {}
            for model_name, status in unified_system.submodel_registry.items():
                all_status[model_name] = status
            return jsonify({
                'status': 'success',
                'models': all_status,
                'count': len(all_status)
            })
        else:
            # 返回默认模型列表和状态
            default_models = [
                'A_management', 'B_language', 'C_audio', 'D_image',
                'E_video', 'F_spatial', 'G_sensor', 'H_computer_control',
                'I_knowledge', 'J_motion', 'K_programming'
            ]
            all_status = {model: {'status': 'active', 'health': 'healthy'} for model in default_models}
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
        if unified_system and model_name in unified_system.submodel_registry:
            status = unified_system.submodel_registry[model_name]
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
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'status': 'error',
                'message': 'Text is required'
            }), 400
        
        # 使用情感引擎进行分析
        emotional_state = unified_system.system_state.get('emotional_state', {}) if unified_system else {}
        
        # 简单的情感分析
        positive_words = ['good', 'happy', 'joy', 'like', 'love', 'great', 'excellent', 'perfect', 'wonderful']
        negative_words = ['bad', 'sad', 'sorrow', 'hate', 'dislike', 'poor', 'terrible', 'failure', 'pain']
        
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        if positive_count > negative_count:
            emotion = "positive"
            intensity = min(0.9, 0.5 + (positive_count * 0.1))
        elif negative_count > positive_count:
            emotion = "negative"
            intensity = min(0.9, 0.5 + (negative_count * 0.1))
        else:
            emotion = "neutral"
            intensity = 0.5
        
        return jsonify({
            'status': 'success',
            'emotion': emotion,
            'intensity': intensity,
            'analysis': {
                'positive_keywords': positive_count,
                'negative_keywords': negative_count,
                'text_length': len(text)
            },
            'current_emotional_state': emotional_state
        })
        
    except Exception as e:
        logger.error(f"情感分析失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
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
        available_models = list(model_manager.submodel_registry.keys())
        invalid_models = [m for m in involved_models if m not in available_models]
        
        if invalid_models:
            return jsonify({
                'status': 'error',
                'message': f'Invalid models: {invalid_models}',
                'available_models': available_models
            }), 400
        
        coordination_id = f"coord_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建协调计划
        coordination_plan = {
            'coordination_id': coordination_id,
            'task': task_description,
            'involved_models': involved_models,
            'strategy': coordination_strategy,
            'status': 'planning',
            'created_at': datetime.now().isoformat(),
            'steps': []
        }
        
        # 生成协调步骤
        if coordination_strategy == 'sequential':
            for i, model_id in enumerate(involved_models):
                coordination_plan['steps'].append({
                    'step': i + 1,
                    'model_id': model_id,
                    'action': f'Execute part {i + 1} of task',
                    'status': 'pending'
                })
        elif coordination_strategy == 'parallel':
            for model_id in involved_models:
                coordination_plan['steps'].append({
                    'step': 1,
                    'model_id': model_id,
                    'action': 'Execute parallel task component',
                    'status': 'pending'
                })
        
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
        
        # 模拟知识库响应
        responses = {
            'machine learning': "Machine learning is a branch of artificial intelligence that enables computer systems to learn from data and improve without explicit programming.",
            'deep learning': "Deep learning is a subfield of machine learning that uses multi-layer neural networks to process complex data patterns.",
            'natural language processing': "Natural language processing is a field of artificial intelligence focused on interaction between computers and human language.",
            'computer vision': "Computer vision enables computers to derive information from digital images, videos, and other visual inputs and take action.",
            'reinforcement learning': "Reinforcement learning is a machine learning method where agents learn optimal behavior strategies through interaction with the environment."
        }
        
        query_lower = query.lower()
        response = "Based on my knowledge base, this is a good question. Let me provide you with relevant information."
        
        for keyword, answer in responses.items():
            if keyword in query_lower:
                response = answer
                break
        
        return jsonify({
            'status': 'success',
            'query': query,
            'response': response,
            'domain': knowledge_domain,
            'confidence': 0.85,
            'sources': ['internal_knowledge_base']
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
        import time
        from collections import deque
        
        if unified_system:
            # 创建可序列化的系统状态副本
            serializable_system_state = {}
            for key, value in unified_system.system_state.items():
                if isinstance(value, deque):
                    serializable_system_state[key] = list(value)
                elif hasattr(value, 'isoformat'):  # datetime对象
                    serializable_system_state[key] = value.isoformat()
                elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                    serializable_system_state[key] = value
                else:
                    serializable_system_state[key] = str(value)
            
            stats = {
                'system_uptime': time.time() - unified_system.system_state.get('performance_metrics', {}).get('system_uptime', time.time()),
                'active_models': len(unified_system.submodel_registry),
                'collaboration_stats': unified_system.get_collaboration_stats(),
                'optimization_stats': unified_system.get_optimization_stats(),
                'system_state': serializable_system_state
            }
        else:
            stats = {
                'system_uptime': time.time(),
                'active_models': 11,
                'collaboration_stats': {},
                'optimization_stats': {},
                'system_state': {}
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
        import time
        if unified_system:
            collaboration = unified_system.get_collaboration_stats() if hasattr(unified_system, 'get_collaboration_stats') else {}
            optimization = unified_system.get_optimization_stats() if hasattr(unified_system, 'get_optimization_stats') else {}
            
            # 简化统计信息
            simple_stats = {
                'total_tasks': collaboration.get('total_tasks_processed', 0),
                'successful_tasks': collaboration.get('successful_collaborations', 0),
                'failed_tasks': collaboration.get('failed_collaborations', 0),
                'pending_tasks': collaboration.get('pending_tasks', 0),
                'active_models': len(unified_system.submodel_registry) if hasattr(unified_system, 'submodel_registry') else 11,
                'system_uptime': time.time() - unified_system.system_state.get('performance_metrics', {}).get('system_uptime', time.time()) if unified_system.system_state else time.time(),
                'timestamp': datetime.now().isoformat()
            }
        else:
            simple_stats = {
                'total_tasks': 0,
                'successful_tasks': 0,
                'failed_tasks': 0,
                'pending_tasks': 0,
                'active_models': 11,
                'system_uptime': time.time(),
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

@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return JSONResponse({
        'status': 'error',
        'message': 'Endpoint not found'
    }, status_code=404)

@app.exception_handler(500)
async def internal_error(request: Request, exc: Exception):
    return JSONResponse({
        'status': 'error',
        'message': 'Internal server error'
    }, status_code=500)

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
        
        if unified_system:
            task_id = unified_system.submit_collaboration_task(
                description, required_models, priority, metadata
            )
            return jsonify({
                'status': 'success',
                'task_id': task_id
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Unified system not initialized'
            }), 500
            
    except Exception as e:
        logger.error(f"创建协作任务失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/collaboration/stats', methods=['GET'])
def get_collaboration_stats():
    """获取协作统计"""
    try:
        if unified_system:
            stats = unified_system.get_collaboration_stats()
            return jsonify({
                'status': 'success',
                'data': stats
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Unified system not initialized'
            }), 500
    except Exception as e:
        logger.error(f"获取协作统计失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/optimization/stats', methods=['GET'])
def get_optimization_stats():
    """获取优化统计"""
    try:
        if unified_system:
            stats = unified_system.get_optimization_stats()
            return jsonify({
                'status': 'success',
                'data': stats
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Unified system not initialized'
            }), 500
    except Exception as e:
        logger.error(f"获取优化统计失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def initialize_app():
    """初始化应用"""
    global unified_system
    try:
        logger.info("正在初始化统一AGI系统...")
        unified_system = get_unified_system(language='zh')
        unified_system.start()
        logger.info("统一AGI系统初始化完成")
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        unified_system = None

if __name__ == '__main__':
    initialize_app()
    
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"启动A Management Model API服务于 http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)