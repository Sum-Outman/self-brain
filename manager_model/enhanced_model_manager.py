# -*- coding: utf-8 -*-
# 增强型管理模型 - AGI系统核心协调者 | Enhanced Management Model - AGI System Core Coordinator
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License")
# 您可以在以下网址获取许可证副本: http://www.apache.org/licenses/LICENSE-2.0
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import json
import logging
import time
import os
import re
import gc
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import websockets
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from enum import Enum
import signal
import psutil
from dataclasses import dataclass
from pathlib import Path
import hashlib
import pickle
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import stats
import networkx as nx

# 设置日志 | Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_management_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedManagementModel")

class EnhancedEmotionalAnalyzer:
    """
    增强型情感分析器 - 提供高级情感分析和响应生成
    Enhanced Emotional Analyzer - Provides advanced emotional analysis and response generation
    """
    
    def __init__(self):
        self.emotion_categories = {
            "positive": ["happy", "joy", "excited", "pleased", "content", "optimistic"],
            "negative": ["sad", "angry", "fearful", "anxious", "frustrated", "disappointed"],
            "neutral": ["neutral", "calm", "balanced", "composed", "detached"]
        }
        
        self.emotion_intensity_indicators = {
            "high": ["very", "extremely", "really", "absolutely", "utterly"],
            "medium": ["quite", "rather", "somewhat", "moderately"],
            "low": ["slightly", "a bit", "a little", "mildly"]
        }
        
        # 情感状态转移矩阵 | Emotional state transition matrix
        self.emotion_transition_matrix = {
            "happy": {"success": "joy", "failure": "disappointed", "neutral": "content"},
            "sad": {"success": "hopeful", "failure": "depressed", "neutral": "calm"},
            "angry": {"success": "satisfied", "failure": "frustrated", "neutral": "annoyed"},
            "fearful": {"success": "relieved", "failure": "panicked", "neutral": "cautious"}
        }
    
    def analyze_emotion(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        高级情感分析 - 使用多种技术分析文本情感
        Advanced emotion analysis - Analyze text emotion using multiple techniques
        
        参数 Parameters:
        text: 待分析文本 | Text to analyze
        context: 上下文信息 | Context information
        
        返回 Returns:
        情感分析结果 | Emotion analysis results
        """
        # 多层级情感分析 | Multi-level emotion analysis
        analysis = {
            "primary_emotion": "neutral",
            "secondary_emotions": [],
            "intensity": 0.5,
            "confidence": 0.8,
            "emotional_cues": [],
            "context_influence": 0.0,
            "suggested_responses": []
        }
        
        # 文本预处理 | Text preprocessing
        cleaned_text = self._preprocess_text(text)
        
        # 关键词匹配分析 | Keyword matching analysis
        keyword_analysis = self._keyword_based_analysis(cleaned_text)
        
        # 语义分析 | Semantic analysis
        semantic_analysis = self._semantic_analysis(cleaned_text)
        
        # 上下文影响分析 | Context influence analysis
        if context:
            analysis["context_influence"] = self._analyze_context_influence(context)
        
        # 融合分析结果 | Fusion analysis results
        analysis.update(self._fuse_analysis_results(keyword_analysis, semantic_analysis))
        
        # 生成建议响应 | Generate suggested responses
        analysis["suggested_responses"] = self._generate_suggested_responses(analysis)
        
        return analysis
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理 | Text preprocessing"""
        # 转换为小写 | Convert to lowercase
        text = text.lower()
        
        # 移除特殊字符 | Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # 移除多余空格 | Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _keyword_based_analysis(self, text: str) -> Dict[str, Any]:
        """基于关键词的情感分析 | Keyword-based emotion analysis"""
        analysis = {
            "detected_emotions": [],
            "intensity_level": "medium",
            "emotional_keywords": []
        }
        
        # 检查情感关键词 | Check emotion keywords
        for emotion_category, emotions in self.emotion_categories.items():
            for emotion in emotions:
                if emotion in text:
                    analysis["detected_emotions"].append(emotion)
                    analysis["emotional_keywords"].append(emotion)
        
        # 检查强度指示词 | Check intensity indicators
        for intensity_level, indicators in self.emotion_intensity_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    analysis["intensity_level"] = intensity_level
                    break
        
        return analysis
    
    def _semantic_analysis(self, text: str) -> Dict[str, Any]:
        """语义情感分析 | Semantic emotion analysis"""
        # 这里可以集成更先进的NLP模型 | Can integrate more advanced NLP models here
        analysis = {
            "semantic_score": 0.5,
            "emotional_valence": "neutral",
            "semantic_confidence": 0.7
        }
        
        # 简单的语义分析逻辑 | Simple semantic analysis logic
        positive_words = ["good", "great", "excellent", "wonderful", "happy", "love"]
        negative_words = ["bad", "terrible", "awful", "hate", "sad", "angry"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            analysis["emotional_valence"] = "positive"
            analysis["semantic_score"] = 0.5 + (positive_count * 0.1)
        elif negative_count > positive_count:
            analysis["emotional_valence"] = "negative"
            analysis["semantic_score"] = 0.5 - (negative_count * 0.1)
        
        return analysis
    
    def _analyze_context_influence(self, context: Dict[str, Any]) -> float:
        """分析上下文影响 | Analyze context influence"""
        influence_score = 0.0
        
        if "previous_emotion" in context:
            # 先前情感状态的影响 | Influence of previous emotional state
            influence_score += 0.2
        
        if "task_importance" in context:
            # 任务重要性的影响 | Influence of task importance
            importance = context["task_importance"]
            influence_score += importance * 0.3
        
        if "user_relationship" in context:
            # 用户关系的影响 | Influence of user relationship
            relationship = context["user_relationship"]
            influence_score += relationship * 0.2
        
        return min(1.0, influence_score)
    
    def _fuse_analysis_results(self, keyword_analysis: Dict[str, Any], 
                              semantic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """融合分析结果 | Fuse analysis results"""
        fused_result = {
            "primary_emotion": "neutral",
            "secondary_emotions": [],
            "intensity": 0.5,
            "confidence": 0.8
        }
        
        # 基于关键词分析确定主要情感 | Determine primary emotion based on keyword analysis
        if keyword_analysis["detected_emotions"]:
            fused_result["primary_emotion"] = keyword_analysis["detected_emotions"][0]
            fused_result["secondary_emotions"] = keyword_analysis["detected_emotions"][1:]
        else:
            # 基于语义分析确定情感 | Determine emotion based on semantic analysis
            if semantic_analysis["emotional_valence"] == "positive":
                fused_result["primary_emotion"] = "content"
            elif semantic_analysis["emotional_valence"] == "negative":
                fused_result["primary_emotion"] = "concerned"
        
        # 计算情感强度 | Calculate emotion intensity
        intensity_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        fused_result["intensity"] = intensity_map.get(keyword_analysis["intensity_level"], 0.5)
        
        # 调整基于语义分析的强度 | Adjust intensity based on semantic analysis
        fused_result["intensity"] = (fused_result["intensity"] + semantic_analysis["semantic_score"]) / 2
        
        # 计算置信度 | Calculate confidence
        if keyword_analysis["detected_emotions"] and semantic_analysis["emotional_valence"] != "neutral":
            fused_result["confidence"] = 0.9
        elif keyword_analysis["detected_emotions"] or semantic_analysis["emotional_valence"] != "neutral":
            fused_result["confidence"] = 0.7
        else:
            fused_result["confidence"] = 0.5
        
        return fused_result
    
    def _generate_suggested_responses(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成建议响应 | Generate suggested responses"""
        responses = []
        
        primary_emotion = analysis["primary_emotion"]
        intensity = analysis["intensity"]
        
        # 基于情感类型生成响应 | Generate responses based on emotion type
        if primary_emotion in ["happy", "joy", "excited"]:
            responses.append({
                "type": "enthusiastic",
                "message": "I'm delighted to help with this!",
                "emotional_tone": "positive",
                "intensity_level": "high" if intensity > 0.7 else "medium"
            })
        elif primary_emotion in ["sad", "disappointed", "depressed"]:
            responses.append({
                "type": "compassionate",
                "message": "I understand this might be challenging. I'm here to support you.",
                "emotional_tone": "caring",
                "intensity_level": "high" if intensity > 0.7 else "medium"
            })
        elif primary_emotion in ["angry", "frustrated", "annoyed"]:
            responses.append({
                "type": "calming",
                "message": "I appreciate your frustration. Let's work through this together.",
                "emotional_tone": "reassuring",
                "intensity_level": "medium"
            })
        elif primary_emotion in ["fearful", "anxious", "worried"]:
            responses.append({
                "type": "reassuring",
                "message": "There's no need to worry. I'll help you handle this.",
                "emotional_tone": "comforting",
                "intensity_level": "medium"
            })
        else:
            responses.append({
                "type": "neutral",
                "message": "I'm ready to assist with your request.",
                "emotional_tone": "professional",
                "intensity_level": "medium"
            })
        
        # 基于强度调整响应 | Adjust responses based on intensity
        for response in responses:
            if intensity > 0.8:
                response["message"] = response["message"].replace(".", "!")
                response["emotional_tone"] = "very_" + response["emotional_tone"]
            elif intensity < 0.3:
                response["message"] = response["message"].replace("!", ".")
                response["emotional_tone"] = "mildly_" + response["emotional_tone"]
        
        return responses
    
    def update_emotional_state(self, current_state: Dict[str, Any], 
                              event_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新情感状态基于事件结果
        Update emotional state based on event result
        
        参数 Parameters:
        current_state: 当前情感状态 | Current emotional state
        event_result: 事件结果 | Event result
        
        返回 Returns:
        更新后的情感状态 | Updated emotional state
        """
        new_state = current_state.copy()
        
        # 基于事件成功与否更新情感 | Update emotion based on event success
        if event_result.get("status") == "success":
            # 成功事件增加积极情感 | Successful events increase positive emotions
            new_state["happiness"] = min(1.0, new_state["happiness"] + 0.1)
            new_state["trust"] = min(1.0, new_state["trust"] + 0.05)
            new_state["sadness"] = max(0.0, new_state["sadness"] - 0.05)
        elif event_result.get("status") == "failed":
            # 失败事件增加消极情感 | Failed events increase negative emotions
            new_state["sadness"] = min(1.0, new_state["sadness"] + 0.1)
            new_state["fear"] = min(1.0, new_state["fear"] + 0.05)
            new_state["happiness"] = max(0.0, new_state["happiness"] - 0.05)
        
        # 确保情感值在合理范围内 | Ensure emotion values are within reasonable range
        for emotion in new_state:
            if emotion != "overall_mood" and emotion != "emotional_intensity":
                new_state[emotion] = max(0.0, min(1.0, new_state[emotion]))
        
        # 更新整体情绪 | Update overall mood
        new_state = self._update_overall_mood(new_state)
        
        return new_state
    
    def _update_overall_mood(self, emotional_state: Dict[str, Any]) -> Dict[str, Any]:
        """更新整体情绪 | Update overall mood"""
        happiness = emotional_state.get("happiness", 0.5)
        sadness = emotional_state.get("sadness", 0.1)
        anger = emotional_state.get("anger", 0.1)
        fear = emotional_state.get("fear", 0.1)
        
        # 计算情感强度 | Calculate emotional intensity
        emotional_intensity = (happiness + sadness + anger + fear) / 4
        
        # 确定主要情绪 | Determine primary mood
        if happiness > 0.7 and happiness > sadness and happiness > anger and happiness > fear:
            overall_mood = "happy"
        elif sadness > 0.7 and sadness > happiness and sadness > anger and sadness > fear:
            overall_mood = "sad"
        elif anger > 0.7 and anger > happiness and anger > sadness and anger > fear:
            overall_mood = "angry"
        elif fear > 0.7 and fear > happiness and fear > sadness and fear > anger:
            overall_mood = "fearful"
        else:
            overall_mood = "neutral"
        
        emotional_state["overall_mood"] = overall_mood
        emotional_state["emotional_intensity"] = emotional_intensity
        
        return emotional_state

class EnhancedModelManager:
    """
    增强型模型管理器 - AGI系统核心协调者
    Enhanced Model Manager - AGI System Core Coordinator
    负责管理所有子模型、任务分配、高级情感分析和系统协调
    (Responsible for managing all sub-models, task allocation, advanced emotional analysis, and system coordination)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化增强型模型管理器
        Initialize Enhanced Model Manager
        
        参数 Parameters:
        config_path: 配置文件路径 | Configuration file path
        """
        # 加载配置 | Load configuration
        self.config = self._load_config(config_path)
        
        # 情感状态 | Emotional state
        self.emotional_state = {
            "happiness": 0.5,
            "sadness": 0.1,
            "anger": 0.1,
            "fear": 0.1,
            "surprise": 0.2,
            "trust": 0.6,
            "anticipation": 0.4,
            "joy": 0.5,
            "disgust": 0.1,
            "overall_mood": "neutral",
            "emotional_intensity": 0.5
        }
        
        # 初始化增强型情感分析器 | Initialize enhanced emotional analyzer
        self.emotional_analyzer = EnhancedEmotionalAnalyzer()
        
        # 子模型注册表 | Sub-model registry
        self.submodel_registry = {}
        
        # 任务队列 | Task queue
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.completed_tasks = {}
        
        # 性能监控 | Performance monitoring
        self.performance_metrics = {
            "total_tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_processing_time": 0.0,
            "submodel_utilization": {},
            "system_uptime": time.time()
        }
        
        # 初始化子模型连接 | Initialize sub-model connections
        self._initialize_submodel_connections()
        
        # 学习引擎 | Learning engine
        self.learning_engine = LearningEngine()
        self.optimization_history = deque(maxlen=1000)
        
        # 多语言支持 | Multilingual support
        self.current_language = "zh"  # 默认中文 | Default Chinese
        self.language_resources = self._load_language_resources()
        
        # 实时监控 | Real-time monitoring
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # 启动任务处理循环 | Start task processing loop
        self.is_running = True
        self.task_processor_thread = threading.Thread(target=self._task_processing_loop, daemon=True)
        self.task_processor_thread.start()
        
        logger.info("增强型模型管理器初始化完成 | Enhanced Model Manager initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件 | Load configuration file"""
        default_config = {
            "system": {
                "name": "AGI_Enhanced_Management_System",
                "version": "3.0.0",
                "description": "AGI系统增强型管理模型 - 核心协调者 | AGI System Enhanced Management Model - Core Coordinator",
                "license": "Apache-2.0"
            },
            "submodels": {
                "B_language": {
                    "enabled": True,
                    "local_model": True,
                    "api_model": False,
                    "endpoint": "http://localhost:8001/api/process",
                    "timeout": 30
                },
                "C_audio": {
                    "enabled": True,
                    "local_model": True,
                    "api_model": False,
                    "endpoint": "http://localhost:8002/api/process",
                    "timeout": 30
                },
                "D_image": {
                    "enabled": True,
                    "local_model": True,
                    "api_model": False,
                    "endpoint": "http://localhost:8003/api/process",
                    "timeout": 30
                },
                "E_video": {
                    "enabled": True,
                    "local_model": True,
                    "api_model": False,
                    "endpoint": "http://localhost:8004/api/process",
                    "timeout": 60
                },
                "F_spatial": {
                    "enabled": True,
                    "local_model": True,
                    "api_model": False,
                    "endpoint": "http://localhost:8005/api/process",
                    "timeout": 30
                },
                "G_sensor": {
                    "enabled": True,
                    "local_model": True,
                    "api_model": False,
                    "endpoint": "http://localhost:8006/api/process",
                    "timeout": 30
                },
                "H_computer_control": {
                    "enabled": True,
                    "local_model": True,
                    "api_model": False,
                    "endpoint": "http://localhost:8007/api/process",
                    "timeout": 30
                },
                "I_knowledge": {
                    "enabled": True,
                    "local_model": True,
                    "api_model": False,
                    "endpoint": "http://localhost:8008/api/process",
                    "timeout": 30
                },
                "J_motion": {
                    "enabled": True,
                    "local_model": True,
                    "api_model": False,
                    "endpoint": "http://localhost:8009/api/process",
                    "timeout": 30
                },
                "K_programming": {
                    "enabled": True,
                    "local_model": True,
                    "api_model": False,
                    "endpoint": "http://localhost:8010/api/process",
                    "timeout": 60
                }
            },
            "emotional_analysis": {
                "enabled": True,
                "sensitivity": 0.8,
                "response_strategy": "adaptive",
                "learning_enabled": True
            },
            "performance": {
                "max_concurrent_tasks": 15,
                "task_timeout": 300,
                "monitoring_interval": 5
            },
            "web_interface": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 8080,
                "ssl_enabled": False,
                "multi_language": True
            }
        }
        
        # 如果提供了配置文件路径，则加载配置 | If config path provided, load from file
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except Exception as e:
                logger.error(f"加载配置文件失败: {e} | Failed to load config file: {e}")
        
        return default_config
    
    def _initialize_submodel_connections(self):
        """初始化子模型连接 | Initialize sub-model connections"""
        logger.info("正在初始化子模型连接 | Initializing sub-model connections")
        
        for model_name, model_config in self.config["submodels"].items():
            if model_config["enabled"]:
                try:
                    # 根据配置选择本地模型或API模型 | Choose local model or API model based on config
                    if model_config["local_model"]:
                        # 初始化本地模型实例 | Initialize local model instance
                        model_instance = self._initialize_local_model(model_name)
                    elif model_config["api_model"]:
                        # 初始化API连接 | Initialize API connection
                        model_instance = self._initialize_api_connection(model_name, model_config["endpoint"])
                    else:
                        logger.warning(f"模型 {model_name} 未配置有效的模型类型 | Model {model_name} not configured with valid model type")
                        continue
                    
                    # 注册模型 | Register model
                    self.submodel_registry[model_name] = {
                        "instance": model_instance,
                        "config": model_config,
                        "status": "active",
                        "last_used": datetime.now().isoformat(),
                        "usage_count": 0,
                        "success_count": 0,
                        "error_count": 0
                    }
                    
                    logger.info(f"子模型 {model_name} 初始化成功 | Sub-model {model_name} initialized successfully")
                    
                except Exception as e:
                    logger.error(f"子模型 {model_name} 初始化失败: {e} | Sub-model {model_name} initialization failed: {e}")
    
    def _initialize_local_model(self, model_name: str) -> Any:
        """
        初始化本地模型实例
        Initialize local model instance
        
        参数 Parameters:
        model_name: 模型名称 | Model name
        
        返回 Returns:
        模型实例 | Model instance
        """
        # 根据模型名称初始化不同的本地模型 | Initialize different local models based on model name
        if model_name == "B_language":
            return self._initialize_language_model()
        elif model_name == "C_audio":
            return self._initialize_audio_model()
        elif model_name == "D_image":
            return self._initialize_image_model()
        elif model_name == "E_video":
            return self._initialize_video_model()
        elif model_name == "F_spatial":
            return self._initialize_spatial_model()
        elif model_name == "G_sensor":
            return self._initialize_sensor_model()
        elif model_name == "H_computer_control":
            return self._initialize_computer_control_model()
        elif model_name == "I_knowledge":
            return self._initialize_knowledge_model()
        elif model_name == "J_motion":
            return self._initialize_motion_model()
        elif model_name == "K_programming":
            return self._initialize_programming_model()
        else:
            raise ValueError(f"未知的模型类型: {model_name} | Unknown model type: {model_name}")
    
    def _initialize_language_model(self):
        """初始化语言模型 | Initialize language model"""
        # 这里使用占位符，实际实现需要根据具体模型加载
        # Placeholder implementation, actual implementation needs to load specific model
        return {"type": "enhanced_language_model", "status": "initialized", "capabilities": ["emotional_analysis", "multilingual"]}
    
    def _initialize_audio_model(self):
        """初始化音频模型 | Initialize audio model"""
        return {"type": "enhanced_audio_model", "status": "initialized", "capabilities": ["speech_recognition", "audio_synthesis"]}
    
    def _initialize_image_model(self):
        """初始化图像模型 | Initialize image model"""
        return {"type": "enhanced_image_model", "status": "initialized", "capabilities": ["image_recognition", "image_generation"]}
    
    def _initialize_video_model(self):
        """初始化视频模型 | Initialize video model"""
        return {"type": "enhanced_video_model", "status": "initialized", "capabilities": ["video_analysis", "video_processing"]}
    
    def _initialize_spatial_model(self):
        """初始化空间模型 | Initialize spatial model"""
        return {"type": "enhanced_spatial_model", "status": "initialized", "capabilities": ["spatial_mapping", "object_tracking"]}
    
    def _initialize_sensor_model(self):
        """初始化传感器模型 | Initialize sensor model"""
        return {"type": "enhanced_sensor_model", "status": "initialized", "capabilities": ["sensor_fusion", "real_time_processing"]}
    
    def _initialize_computer_control_model(self):
        """初始化计算机控制模型 | Initialize computer control model"""
        return {"type": "enhanced_computer_control_model", "status": "initialized", "capabilities": ["system_control", "multi_os_support"]}
    
    def _initialize_knowledge_model(self):
        """初始化知识模型 | Initialize knowledge model"""
        return {"type": "enhanced_knowledge_model", "status": "initialized", "capabilities": ["knowledge_retrieval", "teaching_assistance"]}
    
    def _initialize_motion_model(self):
        """初始化运动模型 | Initialize motion model"""
        return {"type": "enhanced_motion_model", "status": "initialized", "capabilities": ["motion_planning", "actuator_control"]}
    
    def _initialize_programming_model(self):
        """初始化编程模型 | Initialize programming model"""
        return {"type": "enhanced_programming_model", "status": "initialized", "capabilities": ["code_generation", "self_improvement"]}
    
    def _initialize_api_connection(self, model_name: str, endpoint: str) -> Any:
        """
        初始化API连接
        Initialize API connection
        
        参数 Parameters:
        model_name: 模型名称 | Model name
        endpoint: API端点 | API endpoint
        
        返回 Returns:
        API连接实例 | API connection instance
        """
        return {
            "type": "enhanced_api_connection",
            "model_name": model_name,
            "endpoint": endpoint,
            "status": "connected",
            "last_checked": datetime.now().isoformat(),
            "capabilities": ["remote_processing", "api_integration"]
        }
    
    async def process_task(self, task_description: str, task_type: str = "general", 
                          emotional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理任务 - 核心方法
        Process task - Core method
        
        参数 Parameters:
        task_description: 任务描述 | Task description
        task_type: 任务类型 | Task type
        emotional_context: 情感上下文 | Emotional context
        
        返回 Returns:
        任务处理结果 | Task processing result
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info(f"开始处理任务 {task_id}: {task_description} | Starting task {task_id}: {task_description}")
        
        start_time = time.time()
        
        try:
            # 分析任务 | Analyze task
            task_analysis = await self._analyze_task(task_description, task_type, emotional_context)
            
            # 分配子模型任务 | Assign submodel tasks
            assigned_tasks = self._assign_submodel_tasks(task_analysis)
            
            # 执行任务 | Execute tasks
            results = await self._execute_submodel_tasks(assigned_tasks, task_analysis)
            
            # 整合结果 | Integrate results
            final_result = await self._integrate_task_results(results, task_analysis)
            
            # 记录任务历史 | Record task history
            self._record_task_history(task_id, task_description, final_result, start_time)
            
            # 更新情感状态 | Update emotional state
            self._update_emotional_state(final_result)
            
            # 更新性能指标 | Update performance metrics
            self._update_performance_metrics(task_id, True, time.time() - start_time)
            
            logger.info(f"任务 {task_id} 处理完成 | Task {task_id} completed")
            
            return final_result
            
        except Exception as e:
            logger.error(f"任务 {task_id} 处理失败: {e} | Task {task_id} failed: {e}")
            
            # 更新性能指标 | Update performance metrics
            self._update_performance_metrics(task_id, False, time.time() - start_time)
            
            return {
                "status": "failed",
                "error": str(e),
                "task_id": task_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _analyze_task(self, task_description: str, task_type: str, 
                           emotional_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析任务需求
        Analyze task requirements
        
        参数 Parameters:
        task_description: 任务描述 | Task description
        task_type: 任务类型 | Task type
        emotional_context: 情感上下文 | Emotional context
        
        返回 Returns:
        任务分析结果 | Task analysis result
        """
        # 情感分析 | Emotional analysis
        if emotional_context is None:
            emotional_context = await self._analyze_emotional_context(task_description)
        
        # 复杂度分析 | Complexity analysis
        complexity_score = self._estimate_complexity(task_description)
        
        # 资源需求分析 | Resource requirements analysis
        resource_requirements = self._estimate_resource_requirements(task_description, complexity_score)
        
        # 时间估计 | Time estimation
        estimated_time = self._estimate_time_required(complexity_score, resource_requirements)
        
        # 确定需要的子模型 | Determine required submodels
        required_models = self._determine_required_models(task_description, task_type)
        
        return {
            "task_description": task_description,
            "task_type": task_type,
            "emotional_context": emotional_context,
            "task_complexity": complexity_score,
            "resource_requirements": resource_requirements,
            "estimated_time": estimated_time,
            "required_models": required_models,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_emotional_context(self, task_description: str) -> Dict[str, Any]:
        """
        分析情感上下文 - 使用增强型情感分析器
        Analyze emotional context - Using enhanced emotional analyzer
        
        参数 Parameters:
        task_description: 任务描述 | Task description
        
        返回 Returns:
        情感分析结果 | Emotional analysis result
        """
        # 使用增强型情感分析器 | Use enhanced emotional analyzer
        emotional_analysis = self.emotional_analyzer.analyze_emotion(task_description, {
            "previous_emotion": self.emotional_state,
            "task_importance": 0.5,
            "user_relationship": 0.7
        })
        
        # 添加 urgency 和 importance 分析 | Add urgency and importance analysis
        task_lower = task_description.lower()
        
        # 紧急程度分析 | Urgency analysis
        urgency_keywords = ["紧急", "立刻", "马上", "urgent", "immediately", "now", "asap"]
        emotional_analysis["urgency"] = 0.9 if any(word in task_lower for word in urgency_keywords) else 0.5
        
        # 重要性分析 | Importance analysis
        importance_keywords = ["重要", "关键", "essential", "critical", "important", "vital"]
        emotional_analysis["importance"] = 0.9 if any(word in task_lower for word in importance_keywords) else 0.5
        
        #  preferred_emotional_response 基于分析结果 | preferred_emotional_response based on analysis
        if emotional_analysis["primary_emotion"] in ["happy", "joy", "excited"]:
            emotional_analysis["preferred_emotional_response"] = "enthusiastic"
        elif emotional_analysis["primary_emotion"] in ["sad", "disappointed", "depressed"]:
            emotional_analysis["preferred_emotional_response"] = "compassionate"
        elif emotional_analysis["primary_emotion"] in ["angry", "frustrated", "annoyed"]:
            emotional_analysis["preferred_emotional_response"] = "calming"
        elif emotional_analysis["primary_emotion"] in ["fearful", "anxious", "worried"]:
            emotional_analysis["preferred_emotional_response"] = "reassuring"
        else:
            emotional_analysis["preferred_emotional_response"] = "appropriate"
        
        return emotional_analysis
    
    def _estimate_complexity(self, task_description: str) -> float:
        """
        估计任务复杂度
        Estimate task complexity
        
        参数 Parameters:
        task_description: 任务描述 | Task description
        
        返回 Returns:
        复杂度分数 (0-1) | Complexity score (0-1)
        """
        # 增强的复杂度估计逻辑 | Enhanced complexity estimation logic
        words = len(task_description.split())
        sentences = task_description.count('.') + task_description.count('!') + task_description.count('?')
        
        # 基于关键词的复杂度调整 | Complexity adjustment based on keywords
        complexity_keywords = ["复杂", "困难", "挑战", "complex", "difficult", "challenging"]
        task_lower = task_description.lower()
        keyword_complexity = 1.0 if any(word in task_lower for word in complexity_keywords) else 0.0
        
        complexity = min(0.1 + (words * 0.005) + (sentences * 0.03) + (keyword_complexity * 0.2), 0.95)
        return complexity
    
    def _estimate_resource_requirements(self, task_description: str, complexity: float) -> Dict[str, Any]:
        """
        估计资源需求
        Estimate resource requirements
        
        参数 Parameters:
        task_description: 任务描述 | Task description
        complexity: 任务复杂度 | Task complexity
        
        返回 Returns:
        资源需求 | Resource requirements
        """
        # 增强的资源需求估计 | Enhanced resource requirements estimation
        resource_req = {
            "cpu_usage": complexity * 0.8,
            "memory_usage": complexity * 512,  # MB
            "gpu_usage": complexity * 0.6 if "图像" in task_description or "视频" in task_description or "vision" in task_description.lower() else 0.1,
            "network_bandwidth": complexity * 10,  # Mbps
            "storage_requirements": complexity * 100  # MB
        }
        
        # 基于任务类型的调整 | Adjustment based on task type
        if "视频" in task_description or "video" in task_description.lower():
            resource_req["gpu_usage"] = min(1.0, resource_req["gpu_usage"] + 0.3)
            resource_req["memory_usage"] = resource_req["memory_usage"] * 2
        
        return resource_req
    
    def _estimate_time_required(self, complexity: float, resource_requirements: Dict[str, Any]) -> float:
        """
        估计所需时间
        Estimate time required
        
        参数 Parameters:
        complexity: 任务复杂度 | Task complexity
        resource_requirements: 资源需求 | Resource requirements
        
        返回 Returns:
        估计时间 (秒) | Estimated time (seconds)
        """
        base_time = 3.0  # 基础时间 | Base time
        complexity_factor = complexity * 25.0  # 复杂度因子 | Complexity factor
        resource_factor = (resource_requirements["cpu_usage"] + 
                          resource_requirements["memory_usage"] / 512 + 
                          resource_requirements["gpu_usage"]) * 8.0
        
        return base_time + complexity_factor + resource_factor
    
    def _determine_required_models(self, task_description: str, task_type: str) -> List[str]:
        """
        确定需要的子模型
        Determine required submodels
        
        参数 Parameters:
        task_description: 任务描述 | Task description
        task_type: 任务类型 | Task type
        
        返回 Returns:
        需要的模型列表 | List of required models
        """
        required_models = []
        task_lower = task_description.lower()
        
        # 语言相关任务 | Language-related tasks
        if any(word in task_lower for word in ["语言", "文本", "翻译", "写作", "language", "text", "translate", "write", "情感", "emotion"]):
            required_models.append("B_language")
        
        # 音频相关任务 | Audio-related tasks
        if any(word in task_lower for word in ["音频", "声音", "音乐", "语音", "audio", "sound", "music", "speech", "录音", "recording"]):
            required_models.append("C_audio")
        
        # 图像相关任务 | Image-related tasks
        if any(word in task_lower for word in ["图像", "图片", "视觉", "识别", "image", "picture", "vision", "recognize", "照片", "photo"]):
            required_models.append("D_image")
        
        # 视频相关任务 | Video-related tasks
        if any(word in task_lower for word in ["视频", "影片", "流媒体", "video", "movie", "stream", "摄像", "camera"]):
            required_models.append("E_video")
        
        # 空间相关任务 | Spatial-related tasks
        if any(word in task_lower for word in ["空间", "定位", "距离", "三维", "spatial", "location", "distance", "3d", "导航", "navigation"]):
            required_models.append("F_spatial")
        
        # 传感器相关任务 | Sensor-related tasks
        if any(word in task_lower for word in ["传感器", "温度", "湿度", "加速度", "sensor", "temperature", "humidity", "acceleration", "数据", "data"]):
            required_models.append("G_sensor")
        
        # 计算机控制相关任务 | Computer control-related tasks
        if any(word in task_lower for word in ["计算机", "控制", "命令", "系统", "computer", "control", "command", "system", "操作", "operate"]):
            required_models.append("H_computer_control")
        
        # 知识相关任务 | Knowledge-related tasks
        if any(word in task_lower for word in ["知识", "信息", "查询", "知识库", "knowledge", "information", "query", "学习", "learn"]):
            required_models.append("I_knowledge")
        
        # 运动控制相关任务 | Motion control-related tasks
        if any(word in task_lower for word in ["运动", "执行器", "控制", "机器人", "motion", "actuator", "control", "robot", "移动", "move"]):
            required_models.append("J_motion")
        
        # 编程相关任务 | Programming-related tasks
        if any(word in task_lower for word in ["编程", "代码", "开发", "软件", "programming", "code", "develop", "software", "算法", "algorithm"]):
            required_models.append("K_programming")
        
        return required_models
    
    def _assign_submodel_tasks(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        分配子模型任务
        Assign submodel tasks
        
        参数 Parameters:
        task_analysis: 任务分析结果 | Task analysis result
        
        返回 Returns:
        分配的任务 | Assigned tasks
        """
        assigned_tasks = {}
        
        for model_name in task_analysis["required_models"]:
            if model_name in self.submodel_registry:
                assigned_tasks[model_name] = {
                    "task_type": "cooperative",
                    "priority": task_analysis["task_complexity"],
                    "timeout_seconds": task_analysis["estimated_time"] * 2,
                    "resource_limits": task_analysis["resource_requirements"],
                    "emotional_context": task_analysis["emotional_context"],
                    "language": self.current_language
                }
        
        return assigned_tasks
    
    async def _execute_submodel_tasks(self, assigned_tasks: Dict[str, Any], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行子模型任务
        Execute submodel tasks
        
        参数 Parameters:
        assigned_tasks: 分配的任务 | Assigned tasks
        task_analysis: 任务分析结果 | Task analysis result
        
        返回 Returns:
        子模型结果 | Submodel results
        """
        results = {}
        
        # 使用线程池并行执行任务 | Use thread pool to execute tasks in parallel
        with ThreadPoolExecutor(max_workers=len(assigned_tasks)) as executor:
            # 创建任务 | Create tasks
            future_to_model = {
                executor.submit(self._call_submodel, model_name, task_config, task_analysis): model_name
                for model_name, task_config in assigned_tasks.items()
            }
            
            # 等待所有任务完成 | Wait for all tasks to complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results[model_name] = result
                except Exception as e:
                    logger.error(f"子模型 {model_name} 执行失败: {e} | Sub-model {model_name} execution failed: {e}")
                    results[model_name] = {
                        "status": "failed",
                        "error": str(e),
                        "model_name": model_name
                    }
        
        return results
    
    async def _call_submodel(self, model_name: str, task_config: Dict[str, Any], 
                           task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用子模型
        Call submodel
        
        参数 Parameters:
        model_name: 模型名称 | Model name
        task_config: 任务配置 | Task configuration
        task_analysis: 任务分析结果 | Task analysis result
        
        返回 Returns:
        子模型结果 | Submodel result
        """
        if model_name not in self.submodel_registry:
            return {
                "status": "failed",
                "error": f"模型 {model_name} 未注册 | Model {model_name} not registered",
                "model_name": model_name
            }
        
        model_info = self.submodel_registry[model_name]
        
        try:
            # 更新模型使用统计 | Update model usage statistics
            model_info["usage_count"] += 1
            model_info["last_used"] = datetime.now().isoformat()
            
            # 调用模型 | Call model
            if model_info["instance"].get("type") == "enhanced_api_connection":
                # 调用API模型 | Call API model
                result = await self._call_api_model(model_info, task_config, task_analysis)
            else:
                # 调用本地模型 | Call local model
                result = await self._call_local_model(model_name, model_info, task_config, task_analysis)
            
            # 更新成功统计 | Update success statistics
            if result.get("status") == "success":
                model_info["success_count"] += 1
            else:
                model_info["error_count"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"调用子模型 {model_name} 失败: {e} | Calling sub-model {model_name} failed: {e}")
            model_info["error_count"] += 1
            
            return {
                "status": "failed",
                "error": str(e),
                "model_name": model_name
            }
    
    async def _call_api_model(self, model_info: Dict[str, Any], task_config: Dict[str, Any], 
                            task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用API模型
        Call API model
        
        参数 Parameters:
        model_info: 模型信息 | Model information
        task_config: 任务配置 | Task configuration
        task_analysis: 任务分析结果 | Task analysis result
        
        返回 Returns:
        API调用结果 | API call result
        """
        # 这里实现API调用逻辑 | Implement API call logic here
        # 实际实现需要使用适当的HTTP客户端 | Actual implementation needs appropriate HTTP client
        return {
            "status": "success",
            "result": f"API response from {model_info['instance']['model_name']}",
            "model_name": model_info['instance']['model_name'],
            "timestamp": datetime.now().isoformat(),
            "language": task_config.get("language", "zh")
        }
    
    async def _call_local_model(self, model_name: str, model_info: Dict[str, Any], 
                              task_config: Dict[str, Any], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用本地模型
        Call local model
        
        参数 Parameters:
        model_name: 模型名称 | Model name
        model_info: 模型信息 | Model information
        task_config: 任务配置 | Task configuration
        task_analysis: 任务分析结果 | Task analysis result
        
        返回 Returns:
        本地模型结果 | Local model result
        """
        # 这里实现本地模型调用逻辑 | Implement local model call logic here
        # 实际实现需要根据具体模型类型调用相应的方法 | Actual implementation needs to call appropriate methods based on model type
        return {
            "status": "success",
            "result": f"Local model response from {model_name}",
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "language": task_config.get("language", "zh")
        }
    
    async def _integrate_task_results(self, results: Dict[str, Any], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        整合任务结果
        Integrate task results
        
        参数 Parameters:
        results: 子模型结果 | Submodel results
        task_analysis: 任务分析结果 | Task analysis result
        
        返回 Returns:
        整合后的结果 | Integrated result
        """
        integrated_result = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "submodel_results": results,
            "overall_quality": 0.0,
            "emotional_response": self._generate_emotional_response(task_analysis["emotional_context"])
        }
        
        # 计算整体质量 | Calculate overall quality
        success_count = sum(1 for result in results.values() if result.get("status") == "success")
        total_count = len(results)
        
        if total_count > 0:
            integrated_result["overall_quality"] = success_count / total_count
        
        # 如果任何子模型失败，标记为部分完成 | If any submodel failed, mark as partially completed
        if success_count < total_count:
            integrated_result["status"] = "partially_completed"
        
        return integrated_result
    
    def _generate_emotional_response(self, emotional_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成情感响应
        Generate emotional response
        
        参数 Parameters:
        emotional_context: 情感上下文 | Emotional context
        
        返回 Returns:
        情感响应 | Emotional response
        """
        emotional_response = {
            "tone": emotional_context.get("preferred_emotional_response", "appropriate"),
            "intensity": max(emotional_context.get("urgency", 0.5), emotional_context.get("importance", 0.5)),
            "emotional_state": self.emotional_state.copy(),
            "message": self._get_emotional_message(emotional_context)
        }
        
        return emotional_response
    
    def _get_emotional_message(self, emotional_context: Dict[str, Any]) -> str:
        """获取情感消息 | Get emotional message"""
        tone = emotional_context.get("preferred_emotional_response", "appropriate")
        intensity = max(emotional_context.get("urgency", 0.5), emotional_context.get("importance", 0.5))
        
        # 多语言情感消息 | Multilingual emotional messages
        messages = {
            "zh": {
                "enthusiastic": "太棒了！我很兴奋能帮助您完成这个任务！",
                "compassionate": "我理解这可能有些困难，我会全力支持您。",
                "calming": "请放心，我会冷静地帮您处理这个问题。",
                "reassuring": "不用担心，我会确保一切顺利。",
                "appropriate": "任务已处理完成。"
            },
            "en": {
                "enthusiastic": "Excellent! I'm excited to help you with this task!",
                "compassionate": "I understand this might be challenging, I'm here to fully support you.",
                "calming": "Please rest assured, I'll help you handle this calmly.",
                "reassuring": "Don't worry, I'll make sure everything goes smoothly.",
                "appropriate": "Task completed."
            }
        }
        
        # 根据强度调整消息 | Adjust message based on intensity
        base_message = messages[self.current_language].get(tone, messages[self.current_language]["appropriate"])
        
        if intensity > 0.8:
            if self.current_language == "zh":
                base_message = base_message.replace("。", "！")
            else:
                base_message = base_message.replace(".", "!")
        elif intensity < 0.3:
            if self.current_language == "zh":
                base_message = base_message.replace("！", "。")
            else:
                base_message = base_message.replace("!", ".")
        
        return base_message
    
    def _record_task_history(self, task_id: str, task_description: str, result: Dict[str, Any], start_time: float):
        """
        记录任务历史
        Record task history
        
        参数 Parameters:
        task_id: 任务ID | Task ID
        task_description: 任务描述 | Task description
        result: 任务结果 | Task result
        start_time: 开始时间 | Start time
        """
        processing_time = time.time() - start_time
        
        task_history = {
            "task_id": task_id,
            "description": task_description,
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time,
            "result": result,
            "emotional_state": self.emotional_state.copy(),
            "performance_metrics": self.performance_metrics.copy(),
            "language": self.current_language
        }
        
        # 保存到文件或数据库 | Save to file or database
        history_file = f"task_history/{task_id}.json"
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(task_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"记录任务历史失败: {e} | Failed to record task history: {e}")
    
    def _update_emotional_state(self, task_result: Dict[str, Any]):
        """
        更新情感状态
        Update emotional state
        
        参数 Parameters:
        task_result: 任务结果 | Task result
        """
        # 使用增强型情感分析器更新情感状态 | Update emotional state using enhanced emotional analyzer
        self.emotional_state = self.emotional_analyzer.update_emotional_state(
            self.emotional_state, task_result)
    
    def _update_performance_metrics(self, task_id: str, success: bool, processing_time: float):
        """
        更新性能指标
        Update performance metrics
        
        参数 Parameters:
        task_id: 任务ID | Task ID
        success: 是否成功 | Whether successful
        processing_time: 处理时间 | Processing time
        """
        self.performance_metrics["total_tasks_processed"] += 1
        
        if success:
            self.performance_metrics["successful_tasks"] += 1
        else:
            self.performance_metrics["failed_tasks"] += 1
        
        # 更新平均处理时间 | Update average processing time
        total_time = self.performance_metrics["average_processing_time"] * (self.performance_metrics["total_tasks_processed"] - 1)
        self.performance_metrics["average_processing_time"] = (total_time + processing_time) / self.performance_metrics["total_tasks_processed"]
    
    def _task_processing_loop(self):
        """
        任务处理循环
        Task processing loop
        """
        while self.is_running:
            try:
                # 这里实现任务队列处理逻辑 | Implement task queue processing logic here
                # 实际实现需要处理异步任务队列 | Actual implementation needs to handle async task queue
                time.sleep(1)  # 临时休眠，避免CPU占用过高 | Temporary sleep to avoid high CPU usage
            except Exception as e:
                logger.error(f"任务处理循环错误: {e} | Task processing loop error: {e}")
                time.sleep(5)  # 出错后等待更长时间 | Wait longer after error
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        Get system status
        
        返回 Returns:
        系统状态信息 | System status information
        """
        return {
            "emotional_state": self.emotional_state,
            "performance_metrics": self.performance_metrics,
            "submodel_status": {name: info["status"] for name, info in self.submodel_registry.items()},
            "system_uptime": time.time() - self.performance_metrics["system_uptime"],
            "timestamp": datetime.now().isoformat(),
            "language": self.current_language
        }
    
    def shutdown(self):
        """
        关闭系统
        Shutdown system
        """
        logger.info("正在关闭增强型模型管理器 | Shutting down Enhanced Model Manager")
        self.is_running = False
        
        # 等待任务处理线程结束 | Wait for task processing thread to finish
        if self.task_processor_thread.is_alive():
            self.task_processor_thread.join(timeout=5)
        
        logger.info("增强型模型管理器已关闭 | Enhanced Model Manager shut down")

    def _load_language_resources(self) -> Dict[str, Any]:
        """加载多语言资源 | Load multilingual resources"""
        language_resources = {
            "zh": {
                "greeting": "您好！我是AGI系统，很高兴为您服务。",
                "task_completed": "任务已完成",
                "task_failed": "任务失败",
                "emotional_positive": "很高兴能帮助您！",
                "emotional_negative": "我理解这可能有挑战，我会尽力协助。",
                "system_status": "系统状态正常",
                "training_started": "训练已开始",
                "optimization_complete": "优化完成",
                "welcome_message": "欢迎使用AGI系统",
                "help_message": "需要帮助吗？我可以协助您完成各种任务。",
                "goodbye_message": "感谢使用，再见！"
            },
            "en": {
                "greeting": "Hello! I am the AGI system, pleased to serve you.",
                "task_completed": "Task completed",
                "task_failed": "Task failed",
                "emotional_positive": "Happy to help you!",
                "emotional_negative": "I understand this might be challenging, I'll do my best to assist.",
                "system_status": "System status normal",
                "training_started": "Training started",
                "optimization_complete": "Optimization complete",
                "welcome_message": "Welcome to the AGI system",
                "help_message": "Need help? I can assist you with various tasks.",
                "goodbye_message": "Thank you for using, goodbye!"
            }
        }
        return language_resources

    def switch_language(self, language_code: str) -> bool:
        """切换系统语言 | Switch system language"""
        if language_code in self.language_resources:
            self.current_language = language_code
            logger.info(f"系统语言已切换至: {language_code} | System language switched to: {language_code}")
            return True
        else:
            logger.warning(f"不支持的语言代码: {language_code} | Unsupported language code: {language_code}")
            return False

    def get_message(self, message_key: str) -> str:
        """获取本地化消息 | Get localized message"""
        return self.language_resources[self.current_language].get(message_key, f"[{message_key}]")

    def _monitoring_loop(self):
        """实时监控循环 | Real-time monitoring loop"""
        while self.is_running:
            try:
                # 监控系统性能 | Monitor system performance
                self._monitor_performance()
                
                # 检查是否需要优化 | Check if optimization is needed
                self._check_optimization_needed()
                
                # 学习模式训练 | Learning mode training
                self._learning_mode_training()
                
                time.sleep(self.config["performance"]["monitoring_interval"])
            except Exception as e:
                logger.error(f"监控循环错误: {e} | Monitoring loop error: {e}")
                time.sleep(10)

    def _monitor_performance(self):
        """监控系统性能 | Monitor system performance"""
        # 收集系统资源使用情况 | Collect system resource usage
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_info.percent,
            "memory_used_mb": memory_info.used / 1024 / 1024,
            "memory_available_mb": memory_info.available / 1024 / 1024,
            "disk_usage_percent": disk_usage.percent,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": self.task_queue.qsize(),
            "emotional_state": self.emotional_state.copy(),
            "language": self.current_language
        }
        
        # 记录性能数据 | Record performance data
        self.performance_metrics["monitoring_data"] = performance_data
        
        # 检查资源使用是否过高 | Check if resource usage is too high
        if cpu_percent > 80 or memory_info.percent > 80:
            logger.warning(f"系统资源使用过高: CPU {cpu_percent}%, 内存 {memory_info.percent}% | High system resource usage: CPU {cpu_percent}%, Memory {memory_info.percent}%")
            self._trigger_optimization("high_resource_usage")

    def _check_optimization_needed(self):
        """检查是否需要优化 | Check if optimization is needed"""
        # 基于任务成功率检查 | Check based on task success rate
        total_tasks = self.performance_metrics["total_tasks_processed"]
        successful_tasks = self.performance_metrics["successful_tasks"]
        
        if total_tasks > 10:
            success_rate = successful_tasks / total_tasks
            if success_rate < 0.7:
                logger.warning(f"任务成功率低: {success_rate:.2f}, 触发优化 | Low task success rate: {success_rate:.2f}, triggering optimization")
