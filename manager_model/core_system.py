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

# 管理模型核心系统 - 增强版AGI大脑
# Management Model Core System - Enhanced AGI Brain
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License")

import requests
import json
import threading
import time
import logging
import psutil  # 系统监控库 | System monitoring library
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from .emotion_engine import EnhancedEmotionEngine
from .task_scheduler import AdvancedTaskScheduler
from .model_registry import ModelRegistry
from .data_bus import DataBus  # 数据总线类 | DataBus class
from .output_processor import OutputProcessor
from .realtime_input import EnhancedRealtimeInputInterface
from .self_learning import SelfLearningModule
from .optimization_engine import OptimizationEngine


class CoreSystem:
    def __init__(self, language='zh'):
        """初始化增强型AGI核心系统 | Initialize enhanced AGI core system
        
        参数:
            language: 系统语言 (zh/en/de/ja/ru) | System language (zh/en/de/ja/ru)
        """
        self.language = language
        # 验证支持的语言
        if language not in ['zh', 'en', ]:
            raise ValueError(f"不支持的语言: {language} | Unsupported language: {language}")
        
        # 初始化核心组件
        self.model_registry = ModelRegistry()  # 模型注册表
        self.data_bus = DataBus()  # 数据总线
        self.emotion_engine = EnhancedEmotionEngine(self.model_registry, language=language)
        self.task_scheduler = AdvancedTaskScheduler(self.model_registry, self.data_bus)
        self.output_processor = OutputProcessor(language=language)
        self.realtime_input = EnhancedRealtimeInputInterface()
        self.self_learning = SelfLearningModule()
        self.optimization_engine = OptimizationEngine()

        # 动态语言切换支持
        self.language_change_callbacks = []

        # 获取模型端口配置
        self.submodel_ports = self.model_registry.get_model_ports()

        # WebSocket服务器引用
        self.websocket_server = None
        
        # 情感表达状态
        self.emotional_expression = {
            "current": "neutral",
            "intensity": 0.5,
            "confidence": 0.8,
            "context_awareness": 0.7
        }
        
        # 自主学习状态
        self.learning_state = {
            "total_learning_cycles": 0,
            "last_optimization": datetime.now(),
            "knowledge_growth_rate": 0.0,
            "skill_acquisition": {}
        }
        
        # 性能监控
        self.performance_metrics = deque(maxlen=1000)  # 保存最近1000个性能指标
        
        # 集成多语言资源
        from .language_resources import get_string
        self.get_string = get_string
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        print(f"AGI核心系统初始化完成 | AGI Core System initialized")

    def start(self):
        """启动增强型AGI核心系统 | Start enhanced AGI core system"""
        print("AGI核心系统启动中... | AGI Core system starting...")
        
        # 加载模型注册表
        self.model_registry.load_registry()
        
        # 启动任务调度器
        self.task_scheduler.start()
        
        # 启动实时输入接口
        self.realtime_input.start()
        
        # 启动自主学习模块
        self.self_learning.start()
        
        # 启动优化引擎
        self.optimization_engine.start_optimization_cycle()
        
        # 初始化性能监控
        self._start_performance_monitoring()
        
        print("AGI核心系统已启动 | AGI Core system started")
        self.logger.info("AGI系统完全初始化并运行")

    def process_user_input(self, input_data, input_type="text"):
        """处理用户输入 | Process user input"""
        # 分析情感（增强情感分析能力）
        emotion = self.emotion_engine.analyze_emotion(
            input_data,
            context=self.data_bus.get_recent_context()
        )

        # 确定任务类型
        task_type = self._determine_task_type(input_data, emotion)

        # 知识库主动辅助（需求10 - 增强知识库辅助）
        knowledge_assist_result = self._invoke_knowledge_assistance(
            input_data,
            task_type,
            emotion,
            user_context=self.data_bus.get_user_profile()
        )
        self.logger.info(f"知识库辅助结果: {knowledge_assist_result}")

        # 更新情感状态（基于知识库反馈）
        if 'suggested_emotion' in knowledge_assist_result:
            self.update_emotion_state(
                knowledge_assist_result['suggested_emotion'])

        # 分配任务（包含知识库建议）
        task_config = {
            'input': input_data,
            'type': task_type,
            'emotion': emotion,
            'knowledge_suggestions': knowledge_assist_result.get(
                'suggestions',
                [])}
        task_id = self.task_scheduler.schedule_task(task_config)

        # 使用情感表达生成响应
        response_message = self._generate_emotional_response(emotion)

        return {
            'status': 'success',
            'task_id': task_id,
            'emotion': emotion,
            'knowledge_assist': knowledge_assist_result,
            'message': response_message,
            'emotional_expression': self.emotional_expression
        }

    def _generate_emotional_response(self, emotion):
        """生成带有情感表达的响应 | Generate response with emotional expression"""
        # 根据情感状态选择不同的响应
        emotion_type = emotion.get('type', 'neutral')
        intensity = emotion.get('intensity', 0.5)

        # 获取情感表达字符串
        if emotion_type == 'happy':
            base_response = self.get_string('response_happy')
        elif emotion_type == 'sad':
            base_response = self.get_string('response_sad')
        elif emotion_type == 'angry':
            base_response = self.get_string('response_angry')
        elif emotion_type == 'excited':
            base_response = self.get_string('response_excited')
        else:
            base_response = self.get_string('response_neutral')

        # 根据强度调整表达
        if intensity > 0.7:
            intensity_modifier = self.get_string('intensity_high')
        elif intensity > 0.4:
            intensity_modifier = self.get_string('intensity_medium')
        else:
            intensity_modifier = self.get_string('intensity_low')

        # 更新情感表达状态
        self.emotional_expression = {
            "current": emotion_type,
            "intensity": intensity
        }

        return f"{intensity_modifier} {base_response}"

    def _determine_task_type(self, input_data, emotion):
        """确定任务类型 | Determine task type"""
        # 根据系统语言使用不同的关键词
        # Use different keywords based on system language
        # 多语言任务类型识别 | Multilingual task type recognition
        if self.language == 'zh':
            keywords = {
                "image_processing": ["图片", "图像", "照片", "画图", "视觉"],
                "video_processing": ["视频", "影片", "录像", "电影", "剪辑"],
                "computer_control": ["控制", "操作", "命令", "执行", "运行", "系统"],
                "motion_control": ["运动", "动作", "机械", "电机", "舵机", "执行器"],
                "knowledge_query": ["知识", "信息", "资料", "学习", "教育", "教学"],
                "programming": ["编程", "代码", "程序", "开发", "软件", "算法"],
                "audio_processing": ["音频", "声音", "音乐", "录音", "语音", "识别"],
                "sensor_processing": ["传感器", "感应", "检测", "测量", "监控"],
                "spatial_processing": ["空间", "定位", "距离", "三维", "3D", "深度"]
            }
        elif self.language == 'en':
            keywords = {
                "image_processing": ["image", "picture", "photo", "visual", "graphic"],
                "video_processing": ["video", "movie", "film", "clip", "edit"],
                "computer_control": ["control", "operate", "command", "execute", "run", "system"],
                "motion_control": ["motion", "actuator", "mechanical", "motor", "servo"],
                "knowledge_query": ["knowledge", "information", "learn", "education", "teach"],
                "programming": ["program", "code", "develop", "software", "algorithm"],
                "audio_processing": ["audio", "sound", "music", "record", "voice", "speech"],
                "sensor_processing": ["sensor", "detect", "measure", "monitor", "sense"],
                "spatial_processing": ["spatial", "location", "distance", "3d", "depth", "position"]
            }
        
        # 增强的任务类型检测逻辑 | Enhanced task type detection logic
        input_lower = input_data.lower()
        
        # 首先检查精确匹配 | First check exact matches
        for task_type, words in keywords.items():
            for word in words:
                if word in input_lower:
                    return task_type
        
        # 如果没有精确匹配，使用情感和上下文辅助决策 | If no exact match, use emotion and context to assist decision
        emotion_type = emotion.get('type', 'neutral')
        emotion_intensity = emotion.get('intensity', 0.5)
        
        # 基于情感的任务类型推断 | Task type inference based on emotion
        if emotion_type == 'curious' and emotion_intensity > 0.6:
            return "knowledge_query"
        elif emotion_type == 'creative' and emotion_intensity > 0.6:
            return "image_processing"
        elif emotion_type == 'analytical' and emotion_intensity > 0.6:
            return "programming"
        
        # 默认返回语言处理 | Default to language processing
        return "language_processing"

    def _invoke_knowledge_assistance(
            self,
            input_data,
            task_type,
            emotion,
            user_context=None):
        """调用知识库模型进行主动辅助（增强版） | Invoke knowledge model for active assistance (enhanced)"""
        # 构建增强知识查询请求
        request = {
            "query": input_data,
            "context": {
                "task_type": task_type,
                "current_emotion": emotion,
                "system_state": self.collect_dashboard_data(),
                "user_context": user_context or {}
            },
            "request_type": "active_assistance"
        }

        # 通过数据总线发送到知识库模型（模型I）
        channel_id = f"knowledge_assist-A-I"
        if channel_id not in self.data_bus.channels:
            self.data_bus.create_channel(
                channel_id, capacity=10, priority=0)  # 最高优先级

        # 发送请求（使用异步回调避免阻塞）
        future = self.data_bus.send_message_async(channel_id, request)

        # 设置回调处理
        def knowledge_callback(response):
            if response:
                self.logger.info(f"知识库深度辅助: {response.get('summary', '')}")
                # 实时应用知识库建议（需求10）
                self._apply_knowledge_suggestions(response)
                return response
            return None

        future.add_done_callback(lambda f: knowledge_callback(f.result()))

        # 立即返回任务ID，不等待结果
        return {
            "status": "processing",
            "message": "知识库辅助已启动"
        }

    def _apply_knowledge_suggestions(self, knowledge_response):
        """实时应用知识库建议 | Apply knowledge suggestions in real-time"""
        suggestions = knowledge_response.get('suggestions', [])
        for suggestion in suggestions:
            # 根据建议类型执行操作
            if suggestion['type'] == 'model_optimization':
                self._optimize_model(
                    suggestion['target'],
                    suggestion['parameters'])
            elif suggestion['type'] == 'task_reprioritization':
                self.task_scheduler.reprioritize_tasks(
                    suggestion['new_priority'])
            # 添加更多建议类型处理...

        # 更新情感状态（基于知识库反馈）
        if 'suggested_emotion' in knowledge_response:
            self.update_emotion_state(knowledge_response['suggested_emotion'])
            # 推送情感状态到前端（需求5）
            self._push_emotion_update()

    def get_task_result(self, task_id):
        """获取任务结果 | Get task result"""
        return self.task_scheduler.get_task_result(task_id)

    def communicate_with_submodel(self, model_name, command, data=None):
        """与子模型通信 | Communicate with submodel
        
        支持本地模型和外部API模型 | Supports both local and external API models
        """
        # 检查是否为外部API模型 | Check if it's an external API model
        model_info = self.model_registry.get_model_info(model_name)
        if not model_info:
            return {'status': 'error', 'message': '无效的模型名称'}
        
        # 外部API模型处理 | External API model handling
        if model_info.get('type') == 'external':
            return self._call_external_api(model_info, command, data)
        
        # 本地模型处理 | Local model handling
        if model_name not in self.submodel_ports:
            return {'status': 'error', 'message': '本地模型端口未配置'}
        
        url = f"http://localhost:{self.submodel_ports[model_name]}/{command}"
        try:
            if data:
                response = requests.post(url, json=data)
            else:
                response = requests.get(url)
            return response.json()
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _call_external_api(self, model_info, command, data):
        """调用外部API模型 | Call external API model"""
        api_url = model_info.get('api_url', '')
        api_key = model_info.get('api_key', '')
        
        if not api_url:
            return {'status': 'error', 'message': 'API URL未配置'}
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "command": command,
            "data": data
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            return response.json()
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def update_emotion_state(self, new_state):
        """更新情感状态 | Update emotion state"""
        self.emotion_engine.set_emotion_state(new_state)
        # 添加情感状态推送（需求5）
        self._push_emotion_update()
        return {'status': 'success', 'new_state': new_state}

    def set_language(self, language):
        """设置系统语言（需求4） | Set system language (Requirement 4)"""
        self.language = language
        self.emotion_engine.language = language
        self.output_processor.language = language
        # 通知所有回调
        for callback in self.language_change_callbacks:
            callback(language)
        return {
            'status': 'success',
            'message': f'Language changed to {language}'}

    def register_language_callback(self, callback):
        """注册语言切换回调 | Register language change callback"""
        self.language_change_callbacks.append(callback)

    def _push_emotion_update(self):
        """推送情感状态更新到前端 | Push emotion update to frontend"""
        if self.websocket_server:
            emotion_data = {
                'type': 'emotion_update',
                'data': self.emotion_engine.get_detailed_state()
            }
            self.websocket_server.broadcast(json.dumps(emotion_data))

    def enable_realtime_dashboard(self, websocket_server):
        """启用实时仪表盘（需求13） | Enable real-time dashboard (Requirement 13)"""
        self.websocket_server = websocket_server
        # 启动仪表盘数据推送线程
        threading.Thread(
            target=self._dashboard_data_push_loop,
            daemon=True).start()

    def _dashboard_data_push_loop(self):
        """仪表盘数据推送循环 | Dashboard data push loop"""
        while self.websocket_server:
            dashboard_data = self.collect_dashboard_data()
            payload = {
                'type': 'dashboard_update',
                'data': dashboard_data
            }
            self.websocket_server.broadcast(json.dumps(payload))
            time.sleep(1)  # 每秒更新一次

    def collect_dashboard_data(self):
        """收集仪表盘所需的所有数据（增强版） | Collect all data needed for dashboard (enhanced)"""
        # 获取任务统计数据
        task_stats = self.task_scheduler.get_task_statistics()

        # 计算模型协作效率
        collab_efficiency = self._calculate_collaboration_efficiency()

        # 获取实时输入状态
        realtime_status = self.realtime_input.get_status()

        # 获取传感器数据（多源）
        sensor_data = self._get_sensor_data()

        # 获取知识库状态（详细）
        knowledge_status = self._get_knowledge_status()

        # 获取API连接状态
        api_status = self._get_api_status()

        # 获取情感状态（详细）
        emotion_state = self.emotion_engine.get_detailed_state()

        # 获取训练指标（实时）
        training_metrics = self._get_training_metrics()

        # 新增：模型资源使用情况
        model_resources = self._get_model_resource_usage()

        # 新增：数据总线状态
        data_bus_status = self.data_bus.get_status()

        # 新增：协作效率历史趋势
        collab_history = self.task_scheduler.get_collaboration_history()

        return {
            "system_metrics": {
                "cpu_usage": self._get_cpu_usage(),
                "memory_usage": self._get_memory_usage(),
                "disk_usage": self._get_disk_usage(),
                "network_usage": self._get_network_usage(),
                "uptime": self._get_uptime_seconds()
            },
            "task_metrics": {
                "stats": task_stats,
                "throughput": task_stats.get("throughput", 0),
                "avg_response_time": task_stats.get("avg_response_time", 0),
                "error_rate": task_stats.get("error_rate", 0),
                "recent_tasks": task_stats.get("recent_tasks", []),
                "collab_efficiency": collab_efficiency,
                "collab_history": collab_history
            },
            "model_metrics": {
                "resources": model_resources,
                "api_status": api_status,
                "training": training_metrics
            },
            "knowledge_metrics": knowledge_status,
            "emotion_metrics": emotion_state,
            "realtime_metrics": {
                "input": realtime_status,
                "sensors": sensor_data,
                "data_bus": data_bus_status
            }
        }

    def _calculate_collaboration_efficiency(self):
        """计算模型协作效率 - 增强版 | Calculate model collaboration efficiency - enhanced version"""
        # 获取所有任务 | Get all tasks
        all_tasks = self.task_scheduler.get_all_tasks()

        if not all_tasks:
            return 0.0

        # 初始化统计变量 | Initialize statistical variables
        multi_model_tasks = 0
        total_quality_score = 0
        total_collaboration_benefit = 0
        total_data_sharing_efficiency = 0
        total_resource_utilization = 0
        
        # 分析每个任务的协作效果 | Analyze collaboration effect for each task
        for task in all_tasks.values():
            involved_models = task.get("involved_models", [])
            if len(involved_models) > 1:
                multi_model_tasks += 1
                
                # 获取详细的协作指标 | Get detailed collaboration metrics
                accuracy_val = task.get("accuracy", 0.8)
                duration_val = task.get("duration", 10)
                collaboration_level = task.get("collaboration_level", 1.0)
                data_sharing_efficiency = task.get("data_sharing_efficiency", 0.7)  # 数据共享效率
                resource_utilization = task.get("resource_utilization", 0.6)  # 资源利用率

                # 计算准确率因子 | Calculate accuracy factor
                accuracy_factor = min(1.0, accuracy_val)

                # 计算持续时间因子 | Calculate duration factor
                duration_factor = min(1.0, duration_val / 30.0)

                # 计算协作效益 | Calculate collaboration benefit
                collaboration_benefit = (1 - duration_factor) * collaboration_level
                total_collaboration_benefit += collaboration_benefit

                # 计算质量得分 | Calculate quality score
                quality_score = accuracy_factor * collaboration_benefit
                total_quality_score += quality_score

                # 累计数据共享和资源利用指标 | Accumulate data sharing and resource utilization metrics
                total_data_sharing_efficiency += data_sharing_efficiency
                total_resource_utilization += resource_utilization

        # 基础协作比例 | Base collaboration ratio
        base_efficiency = multi_model_tasks / len(all_tasks) if all_tasks else 0.0

        # 质量调整因子 | Quality adjustment factor
        quality_factor = total_quality_score / multi_model_tasks if multi_model_tasks > 0 else 1.0

        # 协作效益因子 | Collaboration benefit factor
        collaboration_benefit_factor = total_collaboration_benefit / multi_model_tasks if multi_model_tasks > 0 else 1.0

        # 数据共享效率因子 | Data sharing efficiency factor
        data_sharing_factor = total_data_sharing_efficiency / multi_model_tasks if multi_model_tasks > 0 else 1.0

        # 资源利用因子 | Resource utilization factor
        resource_utilization_factor = total_resource_utilization / multi_model_tasks if multi_model_tasks > 0 else 1.0

        # 实时策略调整因子 | Real-time strategy adjustment factor
        recent_tasks = list(all_tasks.values())[-10:] if len(all_tasks) >= 10 else list(all_tasks.values())
        trend_factor = 1.0
        
        if len(recent_tasks) > 1:
            recent_efficiencies = [self._calculate_task_efficiency(task) for task in recent_tasks]
            trend = sum(recent_efficiencies[i] - recent_efficiencies[i-1] for i in range(1, len(recent_efficiencies))) / (len(recent_efficiencies) - 1)
            trend_factor = 1.0 + (trend * 0.5)

        # 知识库辅助因子 | Knowledge assistance factor
        knowledge_factor = 1.0
        knowledge_model = self.model_registry.get_model('I_knowledge')
        if knowledge_model and knowledge_model.get('active', False):
            # 根据知识库活跃程度调整因子 | Adjust factor based on knowledge base activity level
            knowledge_activity = knowledge_model.get('activity_level', 0.5)
            knowledge_factor = 1.0 + (knowledge_activity * 0.2)  # 最高提高20%效率

        # 自主学习优化因子 | Self-learning optimization factor
        learning_factor = 1.0
        if self.self_learning and hasattr(self.self_learning, 'learning_state'):
            learning_efficiency = self.self_learning.learning_state.get("learning_efficiency", 0.7)
            learning_factor = 0.8 + (learning_efficiency * 0.4)  # 学习效率影响协作效率

        # 计算最终效率 - 综合考虑所有因素 | Calculate final efficiency - considering all factors
        efficiency = (
            base_efficiency * 
            quality_factor * 
            collaboration_benefit_factor * 
            data_sharing_factor * 
            resource_utilization_factor * 
            trend_factor * 
            knowledge_factor * 
            learning_factor
        )

        # 确保效率在合理范围内 | Ensure efficiency is within reasonable range
        efficiency = min(1.0, max(0.0, efficiency))
        
        # 实时调整任务调度策略 | Real-time adjustment of task scheduling strategy
        self._adjust_scheduling_strategy(efficiency)
        
        return round(efficiency, 4)  # 精确到4位小数

    def _calculate_task_efficiency(self, task):
        """计算单个任务效率 | Calculate single task efficiency"""
        # 综合考量准确率和完成时间
        accuracy = task.get("accuracy", 0.8)
        duration_val = task.get("duration", 10)

        # 标准化持续时间到0-1范围
        normalized_duration = min(1.0, duration_val / 30)

        # 计算效率：准确率 * (1 - 标准化持续时间)
        efficiency = accuracy * (1 - normalized_duration)

        return efficiency

    def _get_cpu_usage(self):
        """获取CPU使用率 | Get CPU usage"""
        return psutil.cpu_percent(interval=0.1)  # 获取真实CPU使用率

    def _get_memory_usage(self):
        """获取内存使用率 | Get memory usage"""
        return psutil.virtual_memory().percent  # 获取真实内存使用率

    def _get_disk_usage(self):
        """获取磁盘使用率 | Get disk usage"""
        return psutil.disk_usage('/').percent  # 获取真实磁盘使用率

    def _get_uptime_seconds(self):
        """获取系统运行时间（秒） | Get system uptime in seconds"""
        boot_time = psutil.boot_time()
        return int(time.time() - boot_time)

    def _get_network_usage(self):
        """获取网络使用情况 | Get network usage"""
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }

    def _get_model_resource_usage(self):
        """获取各模型资源使用情况 | Get resource usage per model"""
        model_resources = {}
        for model_name in self.submodel_ports.keys():
            try:
                response = self.communicate_with_submodel(
                    model_name, "resource_usage")
                if response.get('status') == 'success':
                    model_resources[model_name] = response.get('usage', {})
            except Exception as e:
                logging.error(f"获取{model_name}资源使用失败: {str(e)}")
                model_resources[model_name] = {"error": str(e)}
        return model_resources

    def _get_sensor_data(self):
        """获取传感器数据 | Get sensor data"""
        try:
            # 通过传感器模型(G)获取实时数据
            response = self.communicate_with_submodel(
                "G_sensor", "get_sensor_data")
            if response.get('status') == 'success':
                return response.get('data', {})
        except Exception as e:
            logging.error(f"获取传感器数据失败: {str(e)}")
        return {
            "temperature": 0,
            "humidity": 0,
            "pressure": 0
        }

    def _get_knowledge_status(self):
        """获取知识库状态 | Get knowledge base status"""
        try:
            # 通过知识库模型(I)获取状态
            response = self.communicate_with_submodel(
                "I_knowledge", "get_status")
            if response.get('status') == 'success':
                return response.get('status_info', {})
        except Exception as e:
            logging.error(f"获取知识库状态失败: {str(e)}")
        return {
            "domain_count": 0,
            "last_updated": "1970-01-01 00:00:00"
        }

    def _get_api_status(self):
        """获取API连接状态 | Get API connection status"""
        api_status = {}
        for model_name, port in self.submodel_ports.items():
            try:
                # 检查模型端口是否响应
                response = requests.get(
                    f"http://localhost:{port}/health", timeout=1)
                api_status[model_name] = response.status_code == 200
            except Exception:
                api_status[model_name] = False
        return api_status

    def _get_training_metrics(self):
        """获取训练指标 | Get training metrics"""
        try:
            # 通过训练管理器获取指标
            response = requests.get(
                "http://localhost:5050/training_metrics", timeout=2)
            if response.status_code == 200:
                return response.json().get('metrics', [])
        except Exception as e:
            logging.error(f"获取训练指标失败: {str(e)}")
        return [
            {"name": "loss", "value": 0.0},
            {"name": "accuracy", "value": 0.0},
            {"name": "learning_rate", "value": 0.0}
        ]

    def _start_performance_monitoring(self):
        """启动性能监控线程 | Start performance monitoring thread"""
        def monitoring_loop():
            while True:
                try:
                    # 收集系统性能指标
                    cpu_usage = self._get_cpu_usage()
                    memory_usage = self._get_memory_usage()
                    disk_usage = self._get_disk_usage()
                    network_usage = self._get_network_usage()
                    
                    # 收集任务性能指标
                    task_stats = self.task_scheduler.get_task_statistics()
                    
                    # 记录性能数据
                    performance_data = {
                        "timestamp": datetime.now().isoformat(),
                        "system": {
                            "cpu": cpu_usage,
                            "memory": memory_usage,
                            "disk": disk_usage,
                            "network": network_usage
                        },
                        "tasks": task_stats,
                        "learning_cycles": self.learning_state["total_learning_cycles"]
                    }
                    
                    self.performance_metrics.append(performance_data)
                    
                    # 每5秒收集一次
                    time.sleep(5)
                    
                except Exception as e:
                    self.logger.error(f"性能监控错误: {str(e)}")
                    time.sleep(10)
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        self.logger.info("性能监控已启动")

    def analyze_dependencies(self):
        """分析所有模型的依赖关系 | Analyze dependencies for all models"""
        try:
            # 获取编程模型(K)实例
            programming_model = self.model_registry.get_model('K')
            if programming_model is None:
                return {'status': 'error', 'message': '编程模型未加载'}

            # 调用编程模型的依赖分析功能
            dependencies = programming_model.get_all_model_dependencies()
            return {'status': 'success', 'dependencies': dependencies}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _optimize_model(self, target_model, parameters):
        """优化指定模型 | Optimize specified model"""
        try:
            self.logger.info(f"开始优化模型: {target_model}")
            # 调用优化引擎进行模型优化
            result = self.optimization_engine.optimize_model(target_model, parameters)
            
            # 记录优化结果
            self.learning_state["total_learning_cycles"] += 1
            self.learning_state["last_optimization"] = datetime.now()
            
            if "skill_acquisition" not in self.learning_state:
                self.learning_state["skill_acquisition"] = {}
            
            if target_model not in self.learning_state["skill_acquisition"]:
                self.learning_state["skill_acquisition"][target_model] = {
                    "optimization_count": 0,
                    "last_optimized": datetime.now(),
                    "performance_gain": 0.0
                }
            
            self.learning_state["skill_acquisition"][target_model]["optimization_count"] += 1
            self.learning_state["skill_acquisition"][target_model]["last_optimized"] = datetime.now()
            
            if "performance_gain" in result:
                self.learning_state["skill_acquisition"][target_model]["performance_gain"] = result["performance_gain"]
            
            self.logger.info(f"模型 {target_model} 优化完成，性能提升: {result.get('performance_gain', 0)}%")
            return result
            
        except Exception as e:
            self.logger.error(f"模型优化失败: {str(e)}")
            return {"status": "error", "message": str(e)}

    def shutdown(self):
        """关闭系统 | Shutdown system"""
        print("关闭核心系统... | Shutting down core system...")
        self.task_scheduler.stop()
        self.realtime_input.stop()
        print("核心系统已关闭 | Core system shut down")


if __name__ == '__main__':
    # 测试核心系统
    # Test core system
    system = CoreSystem()
    system.start()

    # 模拟用户输入
    user_input = "请帮我生成一张日落的图片"
    response = system.process_user_input(user_input)
    print("用户输入处理结果 | User input processing result:", response)

    # 获取任务结果
    time.sleep(2)
    task_result = system.get_task_result(response['task_id'])
    print("任务结果 | Task result:", task_result)

    system.shutdown()
