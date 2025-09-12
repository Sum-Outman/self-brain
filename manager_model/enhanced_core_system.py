# -*- coding: utf-8 -*-
# 增强型管理模型核心系统 - 支持五种语言的AGI大脑
# Enhanced Management Model Core System - AGI Brain with 5 Language Support
# Copyright 2025 Self Brain AGI System Authors
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
import asyncio
from typing import Dict, List, Any, Optional, Set, Callable
import os
import yaml

# 导入增强的多语言管理器
from enhanced_multilingual_manager import EnhancedMultilingualManager

class EnhancedCoreSystem:
    """增强型管理模型核心系统 - 支持汉语、英文、德语、日语、俄语
    Enhanced Management Model Core System - Supports Chinese, English, German, Japanese, Russian
    """
    
    def __init__(self, default_language='zh'):
        """初始化增强型AGI核心系统 | Initialize enhanced AGI core system
        
        参数 Parameters:
            default_language: 默认语言代码 (zh/en/de/ja/ru) | Default language code (zh/en/de/ja/ru)
        """
        # 初始化增强的多语言管理器
        self.multilingual_manager = EnhancedMultilingualManager(default_language)
        self.current_language = default_language
        
        # 初始化核心组件
        self.model_registry = self._create_model_registry()  # 模型注册表
        self.data_bus = self._create_data_bus()  # 数据总线
        self.emotion_engine = self._create_emotion_engine()  # 情感引擎
        self.task_scheduler = self._create_task_scheduler()  # 任务调度器
        self.output_processor = self._create_output_processor()  # 输出处理器
        self.realtime_input = self._create_realtime_input()  # 实时输入接口
        self.self_learning = self._create_self_learning()  # 自主学习模块
        self.optimization_engine = self._create_optimization_engine()  # 优化引擎
        
        # 动态语言切换支持
        self.language_change_callbacks = []
        
        # 获取模型端口配置
        self.submodel_ports = self._load_model_ports()
        
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
        
        # 外部API配置
        self.external_api_configs = self._load_external_api_configs()
        
        # 训练管理器配置
        self.training_manager_config = self._load_training_config()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 注册语言切换回调
        self.multilingual_manager.register_language_callback(self._on_language_change)
        
        print(self.multilingual_manager.get_text('system.initialization_complete', 
                                               default="AGI核心系统初始化完成 | AGI Core System initialized"))
    
    def _create_model_registry(self):
        """创建模型注册表实例 | Create model registry instance"""
        # 这里应该返回实际的模型注册表实例
        # 为了演示，返回一个模拟对象
        class MockModelRegistry:
            def load_registry(self):
                return {"status": "success"}
            def get_model_ports(self):
                return {
                    "B_language": 5001, "C_audio": 5002, "D_image": 5003,
                    "E_video": 5004, "F_spatial": 5005, "G_sensor": 5006,
                    "H_computer_control": 5007, "I_knowledge": 5008,
                    "J_motion": 5009, "K_programming": 5010
                }
            def get_model_info(self, model_name):
                return {"type": "local", "port": self.get_model_ports().get(model_name)}
        
        return MockModelRegistry()
    
    def _create_data_bus(self):
        """创建数据总线实例 | Create data bus instance"""
        # 模拟数据总线
        class MockDataBus:
            def __init__(self):
                self.channels = {}
            def create_channel(self, channel_id, capacity=10, priority=0):
                self.channels[channel_id] = {"capacity": capacity, "priority": priority}
            def send_message_async(self, channel_id, message):
                # 模拟异步发送
                future = asyncio.Future()
                future.set_result({"status": "success", "message": "sent"})
                return future
            def get_status(self):
                return {"active_channels": len(self.channels), "total_messages": 0}
        
        return MockDataBus()
    
    def _create_emotion_engine(self):
        """创建情感引擎实例 | Create emotion engine instance"""
        # 模拟情感引擎
        class MockEmotionEngine:
            def __init__(self, model_registry, language='zh'):
                self.language = language
                self.model_registry = model_registry
            
            def analyze_emotion(self, input_data, context=None):
                return {"type": "neutral", "intensity": 0.5, "confidence": 0.8}
            
            def set_emotion_state(self, new_state):
                return {"status": "success", "new_state": new_state}
            
            def get_detailed_state(self):
                return {
                    "current_emotion": "neutral",
                    "intensity": 0.5,
                    "confidence": 0.8,
                    "emotional_trend": "stable"
                }
        
        return MockEmotionEngine(self.model_registry, self.current_language)
    
    def _create_task_scheduler(self):
        """创建任务调度器实例 | Create task scheduler instance"""
        # 模拟任务调度器
        class MockTaskScheduler:
            def __init__(self, model_registry, data_bus):
                self.model_registry = model_registry
                self.data_bus = data_bus
                self.tasks = {}
                self.task_counter = 0
            
            def start(self):
                return {"status": "started"}
            
            def schedule_task(self, task_config):
                self.task_counter += 1
                task_id = f"task_{self.task_counter}"
                self.tasks[task_id] = {
                    "id": task_id,
                    "config": task_config,
                    "status": "scheduled",
                    "created_at": datetime.now()
                }
                return task_id
            
            def get_task_result(self, task_id):
                return self.tasks.get(task_id, {"status": "not_found"})
            
            def get_task_statistics(self):
                return {
                    "total_tasks": len(self.tasks),
                    "completed_tasks": sum(1 for t in self.tasks.values() if t.get("status") == "completed"),
                    "failed_tasks": sum(1 for t in self.tasks.values() if t.get("status") == "failed"),
                    "throughput": 10.5,
                    "avg_response_time": 2.3,
                    "error_rate": 0.05
                }
            
            def get_all_tasks(self):
                return self.tasks
            
            def get_collaboration_history(self):
                return []
        
        return MockTaskScheduler(self.model_registry, self.data_bus)
    
    def _create_output_processor(self):
        """创建输出处理器实例 | Create output processor instance"""
        # 模拟输出处理器
        class MockOutputProcessor:
            def __init__(self, language='zh'):
                self.language = language
            
            def process_output(self, output_data, output_type="text"):
                return {"status": "success", "output": output_data}
        
        return MockOutputProcessor(self.current_language)
    
    def _create_realtime_input(self):
        """创建实时输入接口实例 | Create realtime input interface instance"""
        # 模拟实时输入接口
        class MockRealtimeInput:
            def start(self):
                return {"status": "started"}
            
            def get_status(self):
                return {
                    "camera_connected": True,
                    "microphone_connected": True,
                    "sensors_connected": True,
                    "network_streams": 2
                }
        
        return MockRealtimeInput()
    
    def _create_self_learning(self):
        """创建自主学习模块实例 | Create self-learning module instance"""
        # 模拟自主学习模块
        class MockSelfLearning:
            def start(self):
                return {"status": "started"}
            
            def learning_state(self):
                return {
                    "learning_efficiency": 0.7,
                    "recent_improvements": [],
                    "knowledge_integration_rate": 0.8
                }
        
        return MockSelfLearning()
    
    def _create_optimization_engine(self):
        """创建优化引擎实例 | Create optimization engine instance"""
        # 模拟优化引擎
        class MockOptimizationEngine:
            def start_optimization_cycle(self):
                return {"status": "started"}
            
            def optimize_model(self, target_model, parameters):
                return {
                    "status": "success",
                    "target_model": target_model,
                    "performance_gain": 15.5,
                    "optimization_time": 2.3
                }
        
        return MockOptimizationEngine()
    
    def _load_model_ports(self) -> Dict[str, int]:
        """加载模型端口配置 | Load model port configuration"""
        config_path = 'config/model_ports.yaml'
        default_ports = {
            "B_language": 5001, "C_audio": 5002, "D_image": 5003,
            "E_video": 5004, "F_spatial": 5005, "G_sensor": 5006,
            "H_computer_control": 5007, "I_knowledge": 5008,
            "J_motion": 5009, "K_programming": 5010
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger.error(f"加载模型端口配置错误: {str(e)}")
                return default_ports
        return default_ports
    
    def _load_external_api_configs(self) -> Dict[str, Dict[str, Any]]:
        """加载外部API配置 | Load external API configurations"""
        config_path = 'config/external_apis.yaml'
        default_configs = {
            "B_language": {
                "api_url": "https://api.example.com/language",
                "api_key": "",
                "model_name": "gpt-4",
                "enabled": False
            },
            "C_audio": {
                "api_url": "https://api.example.com/audio",
                "api_key": "",
                "model_name": "whisper",
                "enabled": False
            },
            # 其他模型的默认配置...
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger.error(f"加载外部API配置错误: {str(e)}")
                return default_configs
        return default_configs
    
    def _load_training_config(self) -> Dict[str, Any]:
        """加载训练配置 | Load training configuration"""
        config_path = 'config/training_config.yaml'
        default_config = {
            "training_manager_port": 5050,
            "max_concurrent_trainings": 3,
            "default_training_epochs": 100,
            "validation_split": 0.2,
            "batch_size": 32
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger.error(f"加载训练配置错误: {str(e)}")
                return default_config
        return default_config
    
    def _on_language_change(self, new_language: str, old_language: str) -> None:
        """处理语言切换回调 | Handle language change callback"""
        self.current_language = new_language
        self.logger.info(f"系统语言已从 {old_language} 切换到 {new_language}")
        
        # 更新所有组件的语言设置
        if hasattr(self.emotion_engine, 'language'):
            self.emotion_engine.language = new_language
        if hasattr(self.output_processor, 'language'):
            self.output_processor.language = new_language
        
        # 通知注册的回调函数
        for callback in self.language_change_callbacks:
            try:
                callback(new_language, old_language)
            except Exception as e:
                self.logger.error(f"语言切换回调错误: {str(e)}")
    
    async def start(self):
        """启动增强型AGI核心系统 | Start enhanced AGI core system"""
        start_message = self.multilingual_manager.get_text(
            'system.starting', 
            default="AGI核心系统启动中... | AGI Core system starting..."
        )
        print(start_message)
        
        # 加载模型注册表
        self.model_registry.load_registry()
        
        # 启动所有核心组件
        self.task_scheduler.start()
        self.realtime_input.start()
        self.self_learning.start()
        self.optimization_engine.start_optimization_cycle()
        
        # 启动性能监控
        self._start_performance_monitoring()
        
        # 启动仪表盘数据推送（如果WebSocket服务器已设置）
        if self.websocket_server:
            self._start_dashboard_push_loop()
        
        completed_message = self.multilingual_manager.get_text(
            'system.started', 
            default="AGI核心系统已启动 | AGI Core system started"
        )
        print(completed_message)
        self.logger.info("AGI系统完全初始化并运行")
    
    async def process_user_input(self, input_data: str, input_type: str = "text") -> Dict[str, Any]:
        """处理用户输入 | Process user input"""
        # 分析情感（增强情感分析能力）
        emotion = self.emotion_engine.analyze_emotion(
            input_data,
            context=self.data_bus.get_recent_context() if hasattr(self.data_bus, 'get_recent_context') else {}
        )
        
        # 确定任务类型
        task_type = self._determine_task_type(input_data, emotion)
        
        # 知识库主动辅助（需求10 - 增强知识库辅助）
        knowledge_assist_result = await self._invoke_knowledge_assistance(
            input_data,
            task_type,
            emotion,
            user_context=self.data_bus.get_user_profile() if hasattr(self.data_bus, 'get_user_profile') else {}
        )
        
        self.logger.info(f"知识库辅助结果: {knowledge_assist_result}")
        
        # 更新情感状态（基于知识库反馈）
        if 'suggested_emotion' in knowledge_assist_result:
            self.update_emotion_state(knowledge_assist_result['suggested_emotion'])
        
        # 分配任务（包含知识库建议）
        task_config = {
            'input': input_data,
            'type': task_type,
            'emotion': emotion,
            'knowledge_suggestions': knowledge_assist_result.get('suggestions', [])
        }
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
    
    def _determine_task_type(self, input_data: str, emotion: Dict[str, Any]) -> str:
        """确定任务类型 | Determine task type"""
        # 多语言任务类型识别 | Multilingual task type recognition
        task_keywords = {
            "zh": {
                "image_processing": ["图片", "图像", "照片", "画图", "视觉"],
                "video_processing": ["视频", "影片", "录像", "电影", "剪辑"],
                "computer_control": ["控制", "操作", "命令", "执行", "运行", "系统"],
                "motion_control": ["运动", "动作", "机械", "电机", "舵机", "执行器"],
                "knowledge_query": ["知识", "信息", "资料", "学习", "教育", "教学"],
                "programming": ["编程", "代码", "程序", "开发", "软件", "算法"],
                "audio_processing": ["音频", "声音", "音乐", "录音", "语音", "识别"],
                "sensor_processing": ["传感器", "感应", "检测", "测量", "监控"],
                "spatial_processing": ["空间", "定位", "距离", "三维", "3D", "深度"]
            },
            "en": {
                "image_processing": ["image", "picture", "photo", "visual", "graphic"],
                "video_processing": ["video", "movie", "film", "clip", "edit"],
                "computer_control": ["control", "operate", "command", "execute", "run", "system"],
                "motion_control": ["motion", "actuator", "mechanical", "motor", "servo"],
                "knowledge_query": ["knowledge", "information", "learn", "education", "teach"],
                "programming": ["program", "code", "develop", "software", "algorithm"],
                "audio_processing": ["audio", "sound", "music", "record", "voice", "speech"],
                "sensor_processing": ["sensor", "detect", "measure", "monitor", "sense"],
                "spatial_processing": ["spatial", "location", "distance", "3d", "depth", "position"]
            },
            "de": {
                "image_processing": ["bild", "foto", "abbildung", "visual", "grafik"],
                "video_processing": ["video", "film", "aufnahme", "clip", "bearbeiten"],
                "computer_control": ["steuern", "bedienen", "befehl", "ausführen", "laufen", "system"],
                "motion_control": ["bewegung", "aktuator", "mechanisch", "motor", "servo"],
                "knowledge_query": ["wissen", "information", "lernen", "bildung", "lehren"],
                "programming": ["programm", "code", "entwickeln", "software", "algorithmus"],
                "audio_processing": ["audio", "sound", "musik", "aufnahme", "stimme", "sprache"],
                "sensor_processing": ["sensor", "erkennen", "messen", "überwachen", "fühlen"],
                "spatial_processing": ["räumlich", "ort", "entfernung", "3d", "tiefe", "position"]
            },
            "ja": {
                "image_processing": ["画像", "写真", "ピクチャ", "ビジュアル", "グラフィック"],
                "video_processing": ["动画", "ビデオ", "映画", "クリップ", "編集"],
                "computer_control": ["制御", "操作", "コマンド", "実行", "実行", "システム"],
                "motion_control": ["運動", "アクチュエータ", "機械", "モーター", "サーボ"],
                "knowledge_query": ["知識", "情報", "学習", "教育", "教える"],
                "programming": ["プログラム", "コード", "開発", "ソフトウェア", "アルゴリズム"],
                "audio_processing": ["オーディオ", "音声", "音楽", "録音", "ボイス", "音声認識"],
                "sensor_processing": ["センサー", "検出", "測定", "監視", "感知"],
                "spatial_processing": ["空間", "位置", "距離", "3D", "深度", "位置"]
            },
            "ru": {
                "image_processing": ["изображение", "фото", "картинка", "визуальный", "графика"],
                "video_processing": ["видео", "фильм", "запись", "клип", "редактировать"],
                "computer_control": ["управление", "оперировать", "команда", "выполнить", "запустить", "система"],
                "motion_control": ["движение", "привод", "механический", "мотор", "серво"],
                "knowledge_query": ["знание", "информация", "учиться", "образование", "учить"],
                "programming": ["программа", "код", "разрабатывать", "программное обеспечение", "алгоритм"],
                "audio_processing": ["аудио", "звук", "музыка", "запись", "голос", "речь"],
                "sensor_processing": ["сенсор", "обнаружить", "измерять", "мониторить", "чувствовать"],
                "spatial_processing": ["пространственный", "местоположение", "расстояние", "3D", "глубина", "позиция"]
            }
        }
        
        # 获取当前语言的关键词
        keywords = task_keywords.get(self.current_language, task_keywords["en"])
        input_lower = input_data.lower()
        
        # 检查精确匹配
        for task_type, words in keywords.items():
            for word in words:
                if word in input_lower:
                    return task_type
        
        # 基于情感的任务类型推断
        emotion_type = emotion.get('type', 'neutral')
        emotion_intensity = emotion.get('intensity', 0.5)
        
        if emotion_type == 'curious' and emotion_intensity > 0.6:
            return "knowledge_query"
        elif emotion_type == 'creative' and emotion_intensity > 0.6:
            return "image_processing"
        elif emotion_type == 'analytical' and emotion_intensity > 0.6:
            return "programming"
        
        # 默认返回语言处理
        return "language_processing"
    
    async def _invoke_knowledge_assistance(self, input_data: str, task_type: str, 
                                         emotion: Dict[str, Any], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """调用知识库模型进行主动辅助 | Invoke knowledge model for active assistance"""
        # 构建知识查询请求
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
        
        try:
            # 通过数据总线发送到知识库模型（模型I）
            response = await self.communicate_with_submodel_async("I_knowledge", "assist", request)
            return response
        except Exception as e:
            self.logger.error(f"知识库辅助调用失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _generate_emotional_response(self, emotion: Dict[str, Any]) -> str:
        """生成带有情感表达的响应 | Generate response with emotional expression"""
        emotion_type = emotion.get('type', 'neutral')
        intensity = emotion.get('intensity', 0.5)
        
        # 获取情感表达文本
        emotion_key = f"emotion.{emotion_type}"
        base_response = self.multilingual_manager.get_text(
            emotion_key, 
            default=self.multilingual_manager.get_text('emotion.neutral', default="我明白了")
        )
        
        # 根据强度调整表达
        if intensity > 0.7:
            intensity_modifier = self.multilingual_manager.get_text('intensity.high', default="非常")
        elif intensity > 0.4:
            intensity_modifier = self.multilingual_manager.get_text('intensity.medium', default="很")
        else:
            intensity_modifier = self.multilingual_manager.get_text('intensity.low', default="稍微")
        
        # 更新情感表达状态
        self.emotional_expression = {
            "current": emotion_type,
            "intensity": intensity
        }
        
        return f"{intensity_modifier} {base_response}"
    
    async def communicate_with_submodel_async(self, model_name: str, command: str, data: Any = None) -> Dict[str, Any]:
        """异步与子模型通信 | Asynchronously communicate with submodel"""
        # 检查是否为外部API模型
        model_info = self.model_registry.get_model_info(model_name)
        if not model_info:
            return {'status': 'error', 'message': '无效的模型名称'}
        
        # 外部API模型处理
        if model_info.get('type') == 'external':
            return await self._call_external_api_async(model_info, command, data)
        
        # 本地模型处理
        if model_name not in self.submodel_ports:
            return {'status': 'error', 'message': '本地模型端口未配置'}
        
        url = f"http://localhost:{self.submodel_ports[model_name]}/{command}"
        try:
            loop = asyncio.get_event_loop()
            if data:
                response = await loop.run_in_executor(None, lambda: requests.post(url, json=data, timeout=10))
            else:
                response = await loop.run_in_executor(None, lambda: requests.get(url, timeout=10))
            return response.json()
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def _call_external_api_async(self, model_info: Dict[str, Any], command: str, data: Any) -> Dict[str, Any]:
        """异步调用外部API模型 | Asynchronously call external API model"""
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
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(api_url, headers=headers, json=payload, timeout=10)
            )
            return response.json()
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def update_emotion_state(self, new_state: Dict[str, Any]) -> Dict[str, Any]:
        """更新情感状态 | Update emotion state"""
        self.emotion_engine.set_emotion_state(new_state)
        self._push_emotion_update()
        return {'status': 'success', 'new_state': new_state}
    
    async def set_language(self, language: str) -> Dict[str, Any]:
        """设置系统语言 | Set system language"""
        if language not in self.multilingual_manager.get_supported_languages():
            return {'status': 'error', 'message': f'不支持的语言: {language}'}
        
        success = await self.multilingual_manager.switch_language(language)
        if success:
            return {'status': 'success', 'message': f'Language changed to {language}'}
        else:
            return {'status': 'error', 'message': '语言切换失败'}
    
    def register_language_callback(self, callback: Callable) -> None:
        """注册语言切换回调 | Register language change callback"""
        self.language_change_callbacks.append(callback)
    
    def _push_emotion_update(self) -> None:
        """推送情感状态更新到前端 | Push emotion update to frontend"""
        if self.websocket_server:
            emotion_data = {
                'type': 'emotion_update',
                'data': self.emotion_engine.get_detailed_state()
            }
            self.websocket_server.broadcast(json.dumps(emotion_data))
    
    def enable_realtime_dashboard(self, websocket_server) -> None:
        """启用实时仪表盘 | Enable real-time dashboard"""
        self.websocket_server = websocket_server
        self._start_dashboard_push_loop()
    
    def _start_dashboard_push_loop(self) -> None:
        """启动仪表盘数据推送循环 | Start dashboard data push loop"""
        def push_loop():
            while self.websocket_server:
                dashboard_data = self.collect_dashboard_data()
                payload = {
                    'type': 'dashboard_update',
                    'data': dashboard_data
                }
                self.websocket_server.broadcast(json.dumps(payload))
                time.sleep(1)  # 每秒更新一次
        
        threading.Thread(target=push_loop, daemon=True).start()
    
    def collect_dashboard_data(self) -> Dict[str, Any]:
        """收集仪表盘数据 | Collect dashboard data"""
        return {
            "system_metrics": {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_usage": self._get_network_usage(),
                "uptime": self._get_uptime_seconds()
            },
            "task_metrics": self.task_scheduler.get_task_statistics(),
            "model_metrics": {
                "api_status": self._get_api_status(),
                "resource_usage": self._get_model_resource_usage()
            },
            "emotion_metrics": self.emotion_engine.get_detailed_state(),
            "learning_metrics": self.learning_state
        }
    
    def _get_network_usage(self) -> Dict[str, int]:
        """获取网络使用情况 | Get network usage"""
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv
        }
    
    def _get_uptime_seconds(self) -> int:
        """获取系统运行时间（秒） | Get system uptime in seconds"""
        boot_time = psutil.boot_time()
        return int(time.time() - boot_time)
    
    def _get_api_status(self) -> Dict[str, bool]:
        """获取API连接状态 | Get API connection status"""
        api_status = {}
        for model_name, port in self.submodel_ports.items():
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                api_status[model_name] = response.status_code == 200
            except Exception:
                api_status[model_name] = False
        return api_status
    
    def _get_model_resource_usage(self) -> Dict[str, Any]:
        """获取模型资源使用情况 | Get model resource usage"""
        model_resources = {}
        for model_name in self.submodel_ports.keys():
            try:
                response = self.communicate_with_submodel(model_name, "resource_usage")
                if response.get('status') == 'success':
                    model_resources[model_name] = response.get('usage', {})
            except Exception as e:
                self.logger.error(f"获取{model_name}资源使用失败: {str(e)}")
                model_resources[model_name] = {"error": str(e)}
        return model_resources
    
    def _start_performance_monitoring(self) -> None:
        """启动性能监控 | Start performance monitoring"""
        def monitoring_loop():
            while True:
                try:
                    performance_data = {
                        "timestamp": datetime.now().isoformat(),
                        "system": {
                            "cpu": psutil.cpu_percent(interval=0.1),
                            "memory": psutil.virtual_memory().percent,
                            "disk": psutil.disk_usage('/').percent
                        },
                        "tasks": self.task_scheduler.get_task_statistics(),
                        "learning_cycles": self.learning_state["total_learning_cycles"]
                    }
                    self.performance_metrics.append(performance_data)
                    time.sleep(5)
                except Exception as e:
                    self.logger.error(f"性能监控错误: {str(e)}")
                    time.sleep(10)
        
        threading.Thread(target=monitoring_loop, daemon=True).start()
    
    async def shutdown(self) -> None:
        """关闭系统 | Shutdown system"""
        shutdown_message = self.multilingual_manager.get_text(
            'system.shutting_down',
            default="关闭核心系统... | Shutting down core system..."
        )
        print(shutdown_message)
        
        # 停止所有组件
        if hasattr(self.task_scheduler, 'stop'):
            self.task_scheduler.stop()
        if hasattr(self.realtime_input, 'stop'):
            self.realtime_input.stop()
        
        completed_message = self.multilingual_manager.get_text(
            'system.shut_down',
            default="核心系统已关闭 | Core system shut down"
        )
        print(completed_message)

# 创建系统实例的工厂函数
def create_enhanced_core_system(default_language='zh'):
    """创建增强型核心系统实例 | Create enhanced core system instance"""
    return EnhancedCoreSystem(default_language)

# 示例使用
async def demo_enhanced_system():
    """演示增强型系统功能 | Demonstrate enhanced system functionality"""
    print("初始化增强型AGI核心系统... | Initializing enhanced AGI core system...")
    
    system = create_enhanced_core_system()
    await system.start()
    
    # 测试语言切换
    print("测试语言切换到英文... | Testing language switch to English...")
    result = await system.set_language('en')
    print(f"语言切换结果: {result}")
    
    # 模拟用户输入
    user_input = "Please help me generate a sunset image"
    response = await system.process_user_input(user_input)
    print("用户输入处理结果 | User input processing result:", response)
    
    # 切换回中文
    print("切换回中文... | Switching back to Chinese...")
    await system.set_language('zh')
    
    # 测试中文输入
    user_input_cn = "请帮我生成一张日落的图片"
    response_cn = await system.process_user_input(user_input_cn)
    print("中文输入处理结果 | Chinese input processing result:", response_cn)
    
    # 关闭系统
    await system.shutdown()
    print("系统演示完成 | System demonstration completed")

if __name__ == '__main__':
    asyncio.run(demo_enhanced_system())
