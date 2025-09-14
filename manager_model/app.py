# -*- coding: utf-8 -*-
# A Management Model API Server - 高级模型管理API服务
# Copyright 2025 The AGI Brain System Authors

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import threading

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core_system_merged import get_unified_system

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("A_Management_API")

# 创建Flask应用
app = Flask(__name__, template_folder='templates')
CORS(app)

# 全局统一系统实例
unified_system = None
model_manager = None  # 新增全局 model_manager

# 全局统一系统实例
unified_system = None
model_manager = None  # 新增全局 model_manager

@app.route('/')
def index():
    """Web界面首页"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'A_Management_Model_API',
        'version': '2.0.0'
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    try:
        if unified_system:
            status = unified_system.get_system_status()
            return jsonify({
                'status': 'success',
                'data': status
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Unified system not initialized'
            }), 500
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/process_message', methods=['POST'])
def process_message():
    """处理消息的核心端点"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        message = data.get('message', '')
        task_type = data.get('task_type', 'general')
        emotional_context = data.get('emotional_context', {})
        attachments = data.get('attachments', [])
        knowledge_base = data.get('knowledge_base', 'all')
        
        if not message:
            return jsonify({
                'status': 'error',
                'message': 'Message content is required'
            }), 400
        
        logger.info(f"处理消息: {message[:100]}...")
        
        # 模拟处理结果
        import time
        task_id = f"task_{int(time.time() * 1000)}"
        models_used = ['A_management']
        
        # 根据任务类型生成不同的响应
        if task_type == 'programming':
            response = f"I've analyzed your programming question: {message}. Based on my knowledge, I can help you with Python, JavaScript, and other programming languages."
        elif task_type == 'knowledge':
            response = f"Regarding your knowledge query: {message}. Here's what I know based on my training data and knowledge base."
        elif task_type == 'creative':
            response = f"Let me help you with your creative request: {message}. I can generate creative content and ideas."
        else:
            response = f"I understand your message: {message}. I'm here to help you with various tasks and questions."
        
        # 模拟情感状态
        emotional_state = {
            'happiness': 0.8,
            'sadness': 0.1,
            'anger': 0.05,
            'fear': 0.05,
            'surprise': 0.3
        }
        
        return jsonify({
            'status': 'success',
            'response': response,
            'task_id': task_id,
            'models_used': models_used,
            'processing_time': 0.5,
            'emotional_state': emotional_state,
            'data': {
                'original_message': message,
                'task_type': task_type,
                'confidence': 0.85
            }
        })
            
    except Exception as e:
        logger.error(f"处理消息失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取所有可用模型"""
    try:
        if unified_system and hasattr(unified_system, 'submodel_registry'):
            models = list(unified_system.submodel_registry.keys())
            return jsonify({
                'status': 'success',
                'models': models,
                'count': len(models)
            })
        else:
            # 返回默认模型列表
            default_models = [
                'A_management', 'B_language', 'C_vision', 'D_audio',
                'E_reasoning', 'F_emotion', 'G_sensor', 'H_computer_control',
                'I_knowledge', 'J_motion', 'K_programming'
            ]
            return jsonify({
                'status': 'success',
                'models': default_models,
                'count': len(default_models)
            })
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
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

@app.route('/api/models/status', methods=['GET'])
def get_all_models_status():
    """获取所有模型状态"""
    try:
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
        logger.error(f"获取所有模型状态失败: {e}")
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
        emotional_state = unified_system.system_state.get('emotional_state', {})
        
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
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
    
    port = int(os.environ.get('PORT', 5015))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"启动A Management Model API服务于 http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)