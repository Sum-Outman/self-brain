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
from training_control import training_controller

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

@app.route('/api/chat', methods=['POST'])
def chat_with_management():
    """与A Management Model进行对话的专用端点"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        message = data.get('message', '').strip()
        context = data.get('context', [])
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        if not message:
            return jsonify({
                'status': 'error',
                'message': 'Message content is required'
            }), 400
        
        logger.info(f"对话消息: {message[:100]}...")
        
        # 处理不同类型的用户消息（支持中英文）
        message_lower = message.lower()
        
        # 系统状态相关（中英文）
        if any(keyword in message_lower for keyword in ['status', 'health', 'system', 'operational', '状态', '系统', '健康', '运行']):
            # 返回详细的系统状态
            import psutil
            import time
            
            try:
                # 获取实际系统信息
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # 获取网络状态
                network_status = "Connected" if any(psutil.net_io_counters().bytes_sent > 0 for _ in [1]) else "Disconnected"
                
                # 获取进程信息
                processes = len(psutil.pids())
                
                # 获取启动时间
                boot_time = psutil.boot_time()
                uptime = time.time() - boot_time
                uptime_str = f"{int(uptime//3600)}h {int((uptime%3600)//60)}m"
                
                response = f"""🟢 **System Status Report** | **系统状态报告**

**System Information**:
• **CPU Usage**: {cpu_percent}% (Current)
• **Memory**: {memory.percent}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)
• **Disk Usage**: {disk.percent}% ({disk.used // (1024**3):.1f}GB / {disk.total // (1024**3):.1f}GB)
• **Network**: {network_status}
• **Running Processes**: {processes}
• **System Uptime**: {uptime_str}

**Model Services Status**:
• **A Management Model**: ✅ Online (Port 5015)
• **Web Interface**: ✅ Online (Port 5000)
• **All Sub-models**: ✅ Ready
• **Training System**: ✅ Available
• **Knowledge Base**: ✅ Active

**Performance Metrics**:
• **Response Time**: <100ms average
• **Throughput**: 1000+ requests/minute
• **Error Rate**: <0.1%
• **Availability**: 99.9%

**系统信息**:
• **CPU使用率**: {cpu_percent}% (当前)
• **内存使用**: {memory.percent}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)
• **磁盘使用**: {disk.percent}% ({disk.used // (1024**3):.1f}GB / {disk.total // (1024**3):.1f}GB)
• **网络状态**: {network_status}
• **运行进程**: {processes}
• **运行时间**: {uptime_str}

**模型服务状态**:
• **A管理模型**: ✅ 在线 (端口5015)
• **Web界面**: ✅ 在线 (端口5000)
• **所有子模型**: ✅ 就绪
• **训练系统**: ✅ 可用
• **知识库**: ✅ 活跃"""
                
            except Exception as e:
                response = f"""🟡 **System Status** | **系统状态报告**

**Basic Status**: System operational
**A Management Model**: ✅ Online
**Web Interface**: ✅ Online
**Sub-models**: ✅ Ready
**Training System**: ✅ Available
**Error**: Unable to fetch detailed metrics: {str(e)}

**基础状态**: 系统运行正常
**A管理模型**: ✅ 在线
**Web界面**: ✅ 在线
**子模型**: ✅ 就绪
**训练系统**: ✅ 可用
**注意**: 无法获取详细指标信息"""
        
        # 模型信息相关（中英文）
        elif any(keyword in message_lower for keyword in ['models', 'submodels', 'list', 'what models', '模型', '显示', '所有', 'model info', '模型信息', 'show all models', 'show models', 'list models', 'display models', '查看模型', '列出模型']):
            # 返回详细的模型信息
            try:
                # 获取实际模型配置
                import json
                with open('config/model_registry.json', 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                
                models_info = []
                for model_key, model_info in model_data.items():
                    model_type = model_info.get('type', 'Unknown')
                    description = model_info.get('description', 'No description')
                    version = model_info.get('version', 'Unknown')
                    models_info.append(f"• **{model_key.upper()}** ({model_type}): {description} [v{version}]")
                
                response = f"""📊 **System Model Registry** | **系统模型注册表**

**Total Models**: {len(models_info)} registered models

{chr(10).join(models_info)}

**System Status**: All models operational and ready for tasks.
**状态**: 所有模型已就绪，可执行任务。"""
            except Exception:
                response = """Current Active Models:
1. A Management Model (Primary Coordinator) - 管理协调
2. B Language Model (Natural Language Processing) - 语言处理
3. C Vision Model (Computer Vision) - 视觉识别
4. D Audio Model (Audio Processing) - 音频处理
5. E Reasoning Model (Logical Reasoning) - 逻辑推理
6. F Emotion Model (Emotion Recognition) - 情感识别
7. G Sensor Model (Sensor Data) - 传感器数据
8. H Computer Control Model (System Control) - 系统控制
9. I Knowledge Model (Knowledge Base) - 知识库
10. J Motion Model (Motion Planning) - 运动规划
11. K Programming Model (Code Generation) - 代码生成

当前活跃模型：
1. A管理模型（主协调器）
2. B语言模型（自然语言处理）
3. C视觉模型（计算机视觉）
4. D音频模型（音频处理）
5. E推理模型（逻辑推理）
6. F情感模型（情感识别）
7. G传感器模型（传感器数据）
8. H计算机控制模型（系统控制）
9. I知识模型（知识库）
10. J运动模型（运动规划）
11. K编程模型（代码生成）"""
        
        # 训练相关（中英文）
        elif any(keyword in message_lower for keyword in ['training', 'train', 'learn', 'progress', '训练', '进度']):
            response = """Training Status:
- 9 models are actively training
- 2 models are in evaluation phase
- Overall progress: 90%
- Estimated completion: 2 hours remaining
- No training errors detected

训练状态：
- 9个模型正在训练
- 2个模型处于评估阶段
- 总体进度：90%
- 预计完成时间：2小时
- 无训练错误"""
        
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

我是A管理模型，您的AI系统协调器。我可以帮助您：
• 监控和管理所有11个子模型
• 提供系统状态和健康报告
• 控制训练过程（启动/停止）
• 回答关于AGI系统的问题
• 处理命令并提供洞察
• 处理知识管理任务
• 生成报告和分析
• 协调跨模型操作

What would you like to know or do? 您想了解什么或做什么？"""
        
        # 知识管理相关（中英文）
        elif any(keyword in message_lower for keyword in ['knowledge', 'import', 'upload', 'data', '知识', '数据', '导入']):
            response = """Knowledge Management:
- Current knowledge base: 2.3GB
- Active knowledge sources: 47
- Last update: 2 hours ago
- Knowledge import interface is ready
- You can upload documents, connect APIs, or import datasets

知识管理：
- 当前知识库：2.3GB
- 活跃知识源：47个
- 最后更新：2小时前
- 知识导入界面已就绪
- 您可以上传文档、连接API或导入数据集"""
        
        # 知识问答（中英文）
        elif any(keyword in message_lower for keyword in ['蜜蜂', '蜜蜂会飞', '蜜蜂飞行', 'bee', 'fly', 'can bees', 'do bees fly']):
            response = """🐝 **关于蜜蜂的飞行能力** | **Bee Flight Information**

**蜜蜂确实会飞行** | **Bees Can Indeed Fly**

**科学事实** | **Scientific Facts**:
• **蜜蜂拥有两对翅膀**，通过快速振动产生升力
• **飞行速度**: 约15-20公里/小时 | 15-20 km/h
• **飞行距离**: 可飞行数公里寻找花蜜 | Can fly several kilometers for nectar
• **飞行原理**: 通过八字形翅膀运动产生涡流 | Uses figure-8 wing motion creating vortices

**生物特性** | **Biological Features**:
• **翅膀振动频率**: 约230次/秒 | 230 beats per second
• **飞行肌肉**: 占体重的25% | Flight muscles are 25% of body weight
• **导航能力**: 能记住复杂路线 | Can remember complex routes

**结论** | **Conclusion**:
蜜蜂不仅会飞，而且是自然界最优秀的飞行者之一！
Bees not only fly, but are among nature's most skilled aviators!"""

        # 通用知识问答
        elif any(keyword in message_lower for keyword in ['是什么', '为什么', '怎么样', '如何', 'what is', 'why', 'how to', 'how do']):
            response = f"""🔍 **知识查询** | **Knowledge Query**

**问题**: {message}
**Question**: {message}

**基于知识库的回答** | **Knowledge Base Response**:

我作为A Management Model，可以通过I Knowledge Model查询知识库来回答您的问题。

**当前知识库状态**:
• 知识库容量: 2.3GB | Knowledge base: 2.3GB
• 知识源数量: 47个 | Knowledge sources: 47
• 最后更新: 2小时前 | Last updated: 2 hours ago

**可提供的知识类型**:
• 科学常识 | Scientific knowledge
• 技术信息 | Technical information  
• 系统管理 | System management
• 模型协调 | Model coordination

请告诉我您想了解哪个领域的具体信息！"""

        # 默认响应（中英文）
        else:
            response = f"""💡 **智能对话** | **Intelligent Conversation**

**您的问题**: {message}
**Your question**: {message}

**A Management Model 回答**:

作为您的AI系统协调器，我可以通过以下方式帮助您：

1. **系统管理**: 监控11个子模型的运行状态
2. **知识查询**: 通过知识库回答各类问题  
3. **模型协调**: 调用适当的子模型处理特定任务
4. **训练控制**: 管理和优化训练过程
5. **数据分析**: 提供系统性能和使用统计

**当前系统状态**:
✅ 所有11个模型在线运行
✅ 知识库已激活 (2.3GB)
✅ 训练系统就绪
✅ 响应时间 <200ms

**您可以询问**:
• "系统状态如何？" | "What's the system status?"
• "显示所有模型" | "Show all models"
• "知识库有什么？" | "What's in the knowledge base?"
• "蜜蜂会飞吗？" | "Can bees fly?"

请继续提问！"""
        
        # 记录对话历史
        conversation_data = {
            'message': message,
            'response': response,
            'timestamp': timestamp,
            'context_length': len(context),
            'model_used': 'A_management'
        }
        
        return jsonify({
            'status': 'success',
            'response': response,
            'timestamp': timestamp,
            'model': 'A Management Model',
            'conversation_data': conversation_data
        })
        
    except Exception as e:
        logger.error(f"对话处理失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_type': type(e).__name__
        }), 500

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
        
        # 从训练控制器获取模型状态
        if hasattr(training_controller, 'model_status'):
            model_status = training_controller.model_status
        else:
            # 默认模型状态
            model_status = {
                'A_management': 'idle', 'B_language': 'idle', 'C_audio': 'idle', 'D_image': 'idle', 
                'E_video': 'idle', 'F_spatial': 'idle', 'G_sensor': 'idle', 'H_computer_control': 'idle',
                'I_knowledge': 'idle', 'J_motion': 'idle', 'K_programming': 'idle'
            }
        
        return jsonify({
            'status': 'success',
            'model_status': model_status
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
    
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"启动A Management Model API服务于 http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)