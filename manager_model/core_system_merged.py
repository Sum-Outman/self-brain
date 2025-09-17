# -*- coding: utf-8 -*-
# 合并后的核心系统 - 统一AGI大脑 | Unified AGI Core System
# Copyright 2025 Unified AGI System Authors
# Licensed under the Apache License, Version 2.0

import json
import logging
import time
import os
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque
import requests
import psutil
from concurrent.futures import ThreadPoolExecutor

# 设置统一日志 | Unified logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UnifiedCoreSystem")

class MetaCognitionSystem:
    """元认知系统 - 监控和调节AGI自身的认知过程"""
    def __init__(self):
        self.cognitive_state = {
            "attention_level": 0.8,
            "focus_area": None,
            "cognitive_load": 0.5,
            "learning_efficiency": 0.7,
            "decision_confidence": 0.6
        }
        self.self_awareness_level = 0.0
        self.meta_knowledge_base = {}
        self.cognitive_history = deque(maxlen=1000)
        
    def monitor_cognition(self, thought_process, result, confidence):
        """监控并分析认知过程"""
        timestamp = datetime.now().isoformat()
        cognitive_snapshot = {
            "timestamp": timestamp,
            "thought_process": thought_process,
            "result": result,
            "confidence": confidence,
            "state_before": self.cognitive_state.copy()
        }
        
        # 分析认知过程并更新状态
        if confidence < 0.5:
            self.cognitive_state["attention_level"] = min(1.0, self.cognitive_state["attention_level"] + 0.1)
            self.cognitive_state["cognitive_load"] = min(1.0, self.cognitive_state["cognitive_load"] + 0.05)
        
        cognitive_snapshot["state_after"] = self.cognitive_state.copy()
        self.cognitive_history.append(cognitive_snapshot)
        
        # 更新自我意识水平
        self._update_self_awareness()
        
        return cognitive_snapshot
        
    def optimize_thinking(self, task_complexity):
        """根据任务复杂度优化思考策略"""
        # 基于任务复杂度调整认知参数
        if task_complexity == "high":
            return {
                "attention_budget": 0.9,
                "processing_depth": "deep",
                "resource_allocation": {"memory": 0.8, "cpu": 0.7, "time": "extended"}
            }
        elif task_complexity == "medium":
            return {
                "attention_budget": 0.7,
                "processing_depth": "balanced",
                "resource_allocation": {"memory": 0.6, "cpu": 0.5, "time": "normal"}
            }
        else:
            return {
                "attention_budget": 0.5,
                "processing_depth": "shallow",
                "resource_allocation": {"memory": 0.4, "cpu": 0.3, "time": "fast"}
            }
            
    def _update_self_awareness(self):
        """更新自我意识水平"""
        # 基于认知历史和系统状态计算自我意识水平
        if len(self.cognitive_history) < 10:
            return
            
        recent_history = list(self.cognitive_history)[-10:]
        confidence_changes = [abs(h["result"].get("confidence", 0.5) - h["confidence"]) for h in recent_history]
        avg_confidence_change = sum(confidence_changes) / len(confidence_changes) if confidence_changes else 0
        
        # 自我意识水平与认知调整的准确性相关
        self.self_awareness_level = max(0.0, min(1.0, 0.5 + (0.5 - avg_confidence_change)))

class LongTermMemorySystem:
    """长期记忆系统 - 存储和检索长期知识和经验"""
    def __init__(self):
        self.memory_storage = {}
        self.retrieval_strategies = {
            "semantic": self._semantic_retrieval,
            "temporal": self._temporal_retrieval,
            "contextual": self._contextual_retrieval
        }
        self.memory_index = {}
        self.memory_decay_rates = {}
        
    def store_experience(self, experience, relevance_score):
        """存储经验到长期记忆"""
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.now().isoformat()
        
        memory = {
            "id": memory_id,
            "experience": experience,
            "timestamp": timestamp,
            "relevance_score": relevance_score,
            "access_count": 0,
            "last_accessed": None,
            "decay_factor": 1.0 - (relevance_score * 0.3)  # 相关性越高，衰减越慢
        }
        
        self.memory_storage[memory_id] = memory
        
        # 更新记忆索引以支持高效检索
        self._update_memory_index(memory_id, memory)
        
        return memory_id
        
    def retrieve_memory(self, query, context=None, strategy="semantic"):
        """基于查询和上下文检索相关记忆"""
        if strategy in self.retrieval_strategies:
            return self.retrieval_strategies[strategy](query, context)
        return []
        
    def _semantic_retrieval(self, query, context):
        """基于语义相似度检索记忆"""
        # 简化实现 - 实际应用中应使用更复杂的语义相似度计算
        results = []
        query_lower = query.lower()
        
        for memory_id, memory in self.memory_storage.items():
            content = str(memory["experience"]).lower()
            if query_lower in content:
                # 计算简单的匹配分数
                match_score = len(query_lower) / len(content) if content else 0
                # 考虑相关性和衰减因素
                final_score = match_score * memory["relevance_score"] * memory["decay_factor"]
                results.append((memory_id, memory, final_score))
        
        # 按分数排序并返回前5个结果
        results.sort(key=lambda x: x[2], reverse=True)
        return [r[1] for r in results[:5]]
        
    def _temporal_retrieval(self, query, context):
        """基于时间接近性检索记忆"""
        # 简化实现 - 返回最近的记忆
        recent_memories = sorted(
            self.memory_storage.values(),
            key=lambda x: x["timestamp"],
            reverse=True
        )
        return recent_memories[:5]
        
    def _contextual_retrieval(self, query, context):
        """基于上下文检索记忆"""
        # 如果没有上下文，使用语义检索
        if not context:
            return self._semantic_retrieval(query, context)
        
        # 简化实现 - 同时考虑查询和上下文
        results = []
        query_lower = query.lower()
        context_lower = str(context).lower()
        
        for memory_id, memory in self.memory_storage.items():
            content = str(memory["experience"]).lower()
            if query_lower in content or context_lower in content:
                # 计算简单的匹配分数
                query_match = 1 if query_lower in content else 0
                context_match = 1 if context_lower in content else 0
                match_score = (query_match * 0.6 + context_match * 0.4)
                # 考虑相关性和衰减因素
                final_score = match_score * memory["relevance_score"] * memory["decay_factor"]
                results.append((memory_id, memory, final_score))
        
        # 按分数排序并返回前5个结果
        results.sort(key=lambda x: x[2], reverse=True)
        return [r[1] for r in results[:5]]
        
    def _update_memory_index(self, memory_id, memory):
        """更新记忆索引"""
        # 简化实现 - 实际应用中应使用更复杂的索引结构
        pass
        
    def decay_memories(self):
        """应用记忆衰减机制"""
        for memory_id, memory in self.memory_storage.items():
            # 应用衰减因子
            memory["decay_factor"] = max(0.1, memory["decay_factor"] * 0.999)
            
    def update_memory_access(self, memory_id):
        """更新记忆访问信息"""
        if memory_id in self.memory_storage:
            memory = self.memory_storage[memory_id]
            memory["access_count"] += 1
            memory["last_accessed"] = datetime.now().isoformat()
            # 访问频率高的记忆衰减更慢
            memory["decay_factor"] = min(1.0, memory["decay_factor"] * 1.01)

class UnifiedCoreSystem:
    """
    统一核心系统 - 合并所有重复功能后的单一AGI大脑
    Unified Core System - Single AGI Brain after merging all duplicate functions
    整合AdvancedModelManager、CollaborationEngine、OptimizationEngine的所有功能
    (Integrates all functionality from AdvancedModelManager, CollaborationEngine, and OptimizationEngine)
    """
    
    def __init__(self, language='zh', config_path: Optional[str] = None):
        """初始化统一核心系统 | Initialize unified core system"""
        self.language = language
        self.config = self._load_config(config_path)
        
        # 统一的状态管理 | Unified state management
        self.system_state = {
            "status": "initializing",
            "language": language,
            "uptime": 0,
            "emotional_state": self._init_emotional_state(),
            "performance_metrics": self._init_performance_metrics(),
            "submodel_status": {},
            "active_tasks": {},
            "completed_tasks": deque(maxlen=1000),
            "cognitive_state": {}
        }
        
        # 子模型注册表 | Sub-model registry (enhanced)
        self.submodel_registry = self._init_submodel_registry()
        
        # 任务队列 | Task queue
        self.task_queue = asyncio.Queue()
        self.task_executor = ThreadPoolExecutor(max_workers=10)
        
        # 协作引擎功能 | Collaboration engine features
        self.pending_tasks = {}
        self.failed_tasks = {}
        self.collaboration_stats = {
            "total_tasks_processed": 0,
            "successful_collaborations": 0,
            "failed_collaborations": 0,
            "average_completion_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # 优化引擎功能 | Optimization engine features
        self.optimization_config = {
            "cpu_threshold": 80.0,
            "memory_threshold": 75.0,
            "network_threshold": 50.0,
            "disk_threshold": 85.0,
            "optimization_cooldown": 30.0
        }
        self.optimization_history = deque(maxlen=1000)
        self.optimization_state = {
            "last_optimization": None,
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "average_improvement": 0.0,
            "current_efficiency": 0.8
        }
        
        # 监控循环 | Monitoring loop
        self.monitoring_thread = None
        self.optimization_thread = None
        self.is_running = False
        
        # 增强认知功能 | Enhanced cognitive functions
        self.meta_cognition = MetaCognitionSystem()
        self.long_term_memory = LongTermMemorySystem()
        
        # 系统学习参数
        self.learning_enabled = True
        self.adaptation_rate = 0.1
        
        logger.info("统一核心系统初始化完成 | Unified core system initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载统一配置 | Load unified configuration"""
        default_config = {
            "system": {
                "name": "Unified_AGI_System",
                "version": "3.0.0",
                "max_tasks": 100,
                "monitoring_interval": 5
            },
            "submodels": {
                "B_language": {"endpoint": "http://localhost:8001/process", "enabled": True},
                "C_audio": {"endpoint": "http://localhost:8002/process", "enabled": True},
                "D_image": {"endpoint": "http://localhost:8003/process", "enabled": True},
                "E_video": {"endpoint": "http://localhost:8004/process", "enabled": True},
                "F_spatial": {"endpoint": "http://localhost:8005/process", "enabled": True},
                "G_sensor": {"endpoint": "http://localhost:8006/process", "enabled": True},
                "H_computer_control": {"endpoint": "http://localhost:8007/process", "enabled": True},
                "I_knowledge": {"endpoint": "http://localhost:8008/process", "enabled": True},
                "J_motion": {"endpoint": "http://localhost:8009/process", "enabled": True},
                "K_programming": {"endpoint": "http://localhost:8010/process", "enabled": True}
            },
            "emotional_analysis": {"enabled": True, "sensitivity": 0.7}
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except Exception as e:
                logger.error(f"配置加载失败: {e}")
        
        return default_config
    
    def _init_emotional_state(self) -> Dict[str, float]:
        """初始化情感状态 | Initialize emotional state"""
        return {
            "happiness": 0.5, "sadness": 0.1, "anger": 0.1,
            "fear": 0.1, "surprise": 0.2, "trust": 0.6,
            "anticipation": 0.4, "overall_mood": "neutral",
            "emotional_intensity": 0.5
        }
    
    def _init_performance_metrics(self) -> Dict[str, Any]:
        """初始化性能指标 | Initialize performance metrics"""
        return {
            "total_tasks": 0, "successful_tasks": 0, "failed_tasks": 0,
            "average_processing_time": 0.0, "system_uptime": time.time(),
            "cpu_usage": 0.0, "memory_usage": 0.0, "active_connections": 0
        }
    
    def _init_submodel_registry(self) -> Dict[str, Dict[str, Any]]:
        """初始化子模型注册表 | Initialize sub-model registry"""
        registry = {}
        for model_name, model_config in self.config["submodels"].items():
            if model_config["enabled"]:
                registry[model_name] = {
                    "status": "active",
                    "endpoint": model_config["endpoint"],
                    "last_used": None,
                    "usage_count": 0,
                    "success_count": 0,
                    "error_count": 0
                }
        return registry
    
    async def process_message(self, message: str, task_type: str = "general") -> Dict[str, Any]:
        """
        统一消息处理接口 | Unified message processing interface
        合并了所有重复的处理逻辑 | Merged all duplicate processing logic
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = time.time()
        
        try:
            # 情感分析 | Emotional analysis
            emotional_context = await self._analyze_emotion(message)
            
            # 任务分析 | Task analysis
            task_analysis = await self._analyze_task(message, task_type, emotional_context)
            
            # 分配子模型 | Assign sub-models
            assigned_models = self._select_submodels(task_analysis)
            
            # 执行任务 | Execute tasks
            results = await self._execute_tasks(assigned_models, task_analysis)
            
            # 整合结果 | Integrate results
            final_result = await self._integrate_results(results, task_analysis)
            
            # 更新状态 | Update state
            processing_time = time.time() - start_time
            await self._update_system_state(task_id, True, processing_time, final_result)
            
            return {
                "status": "success",
                "task_id": task_id,
                "result": final_result,
                "processing_time": processing_time,
                "emotional_context": emotional_context
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            await self._update_system_state(task_id, False, processing_time, {"error": str(e)})
            
            return {
                "status": "failed",
                "task_id": task_id,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def _analyze_emotion(self, message: str) -> Dict[str, Any]:
        """统一情感分析 | Unified emotional analysis"""
        # 简化的情感分析逻辑 | Simplified emotional analysis
        positive_words = ["好", "棒", "优秀", "great", "good", "excellent"]
        negative_words = ["坏", "差", "糟糕", "bad", "terrible", "awful"]
        
        message_lower = message.lower()
        positive_score = sum(1 for word in positive_words if word in message_lower)
        negative_score = sum(1 for word in negative_words if word in message_lower)
        
        if positive_score > negative_score:
            emotion = "positive"
        elif negative_score > positive_score:
            emotion = "negative"
        else:
            emotion = "neutral"
        
        return {
            "emotion": emotion,
            "confidence": 0.7,
            "intensity": abs(positive_score - negative_score) * 0.1
        }
    
    async def _analyze_task(self, message: str, task_type: str, emotional_context: Dict[str, Any]) -> Dict[str, Any]:
        """统一任务分析 | Unified task analysis"""
        # 任务类型映射 | Task type mapping
        task_keywords = {
            "image": ["图片", "图像", "photo", "image", "picture"],
            "video": ["视频", "影片", "video", "movie", "film"],
            "audio": ["音频", "声音", "audio", "sound", "music"],
            "programming": ["编程", "代码", "programming", "code", "develop"],
            "knowledge": ["知识", "信息", "knowledge", "information", "learn"]
        }
        
        message_lower = message.lower()
        detected_types = []
        
        for task_type_key, keywords in task_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_types.append(task_type_key)
        
        if not detected_types:
            detected_types = ["general"]
        
        return {
            "message": message,
            "task_type": task_type,
            "detected_types": detected_types,
            "emotional_context": emotional_context,
            "complexity": "medium"
        }
    
    def _select_submodels(self, task_analysis: Dict[str, Any]) -> List[str]:
        """统一子模型选择 | Unified sub-model selection"""
        detected_types = task_analysis["detected_types"]
        
        # 映射到子模型 | Map to sub-models
        type_to_model = {
            "image": "D_image",
            "video": "E_video",
            "audio": "C_audio",
            "programming": "K_programming",
            "knowledge": "I_knowledge"
        }
        
        selected_models = []
        for task_type in detected_types:
            if task_type in type_to_model:
                model = type_to_model[task_type]
                if model in self.submodel_registry:
                    selected_models.append(model)
        
        if not selected_models:
            selected_models = ["B_language"]  # 默认使用语言模型 | Default to language model
        
        return selected_models
    
    async def _execute_tasks(self, models: List[str], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """统一任务执行 | Unified task execution"""
        results = {}
        
        for model in models:
            try:
                if model in self.submodel_registry:
                    # 调用子模型 | Call sub-model
                    response = await self._call_submodel(model, task_analysis)
                    results[model] = response
                    
                    # 更新注册表 | Update registry
                    self.submodel_registry[model]["usage_count"] += 1
                    self.submodel_registry[model]["last_used"] = datetime.now().isoformat()
                    
            except Exception as e:
                logger.error(f"子模型 {model} 执行失败: {e}")
                results[model] = {"error": str(e)}
        
        return results
    
    async def _call_submodel(self, model: str, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """增强版子模型调用 | Enhanced sub-model calling"""
        endpoint = self.submodel_registry[model]["endpoint"]
        
        # 准备调用参数
        call_params = {
            "task_id": f"subtask_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "timestamp": datetime.now().isoformat(),
            "payload": task_analysis,
            "priority": self._determine_task_priority(task_analysis)
        }
        
        try:
            # 实际应用中应使用异步HTTP客户端
            # 这里使用同步请求作为简化实现
            response = requests.post(
                endpoint,
                json=call_params,
                timeout=self._determine_timeout(model, task_analysis)
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 添加调用元数据
            result["_meta"] = {
                "model": model,
                "endpoint": endpoint,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except requests.Timeout:
            logger.error(f"调用子模型 {model} 超时")
            raise TimeoutError(f"Submodel {model} timeout")
        except requests.HTTPError as e:
            logger.error(f"调用子模型 {model} HTTP错误: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"调用子模型 {model} 失败: {str(e)}")
            raise
            
    def _determine_task_priority(self, task_analysis: Dict[str, Any]) -> str:
        """确定任务优先级"""
        complexity = task_analysis.get("complexity", "medium")
        emotional_intensity = task_analysis.get("emotional_context", {}).get("intensity", 0.0)
        
        if complexity == "complex" or emotional_intensity > 0.7:
            return "high"
        elif complexity == "simple" and emotional_intensity < 0.3:
            return "low"
        else:
            return "medium"
            
    def _determine_timeout(self, model: str, task_analysis: Dict[str, Any]) -> int:
        """根据模型和任务确定超时时间"""
        # 基础超时时间
        base_timeout = 30
        
        # 根据模型类型调整
        model_timeouts = {
            "D_image": 60,  # 图像处理可能需要更长时间
            "E_video": 120, # 视频处理需要最长时间
            "K_programming": 60, # 编程任务可能需要较长时间
            "I_knowledge": 45   # 知识处理需要适中时间
        }
        
        # 根据任务复杂度调整
        complexity_multipliers = {
            "simple": 0.7,
            "medium": 1.0,
            "complex": 1.5
        }
        
        # 获取模型特定的超时时间
        model_timeout = model_timeouts.get(model, base_timeout)
        
        # 获取复杂度乘数
        complexity = task_analysis.get("complexity", "medium")
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        # 计算最终超时时间
        final_timeout = int(model_timeout * multiplier)
        
        # 确保超时时间在合理范围内
        return max(10, min(300, final_timeout))  # 10秒到5分钟之间
        
    async def _integrate_results(self, results: Dict[str, Any], task_analysis: Dict[str, Any], relevant_memories: List = None) -> Dict[str, Any]:
        """增强版结果整合 | Enhanced result integration"""
        # 移除执行统计信息，单独处理
        execution_stats = results.pop("_execution_stats", {})
        
        # 基础结果整合
        integrated_result = {
            "summary": "",
            "details": {},
            "confidence": 0.0,
            "sources": [],
            "execution_stats": execution_stats
        }
        
        # 计算整体置信度
        confidence_scores = []
        
        # 整合各模型结果
        for model, result in results.items():
            if isinstance(result, dict):
                # 提取置信度
                if "confidence" in result:
                    confidence_scores.append(result["confidence"])
                
                # 根据模型类型整合结果
                if model == "B_language" and "text" in result:
                    integrated_result["summary"] += result["text"] + "\n"
                elif model == "I_knowledge" and "knowledge" in result:
                    integrated_result["details"]["knowledge"] = result["knowledge"]
                elif model == "D_image" and "image_analysis" in result:
                    integrated_result["details"]["image_analysis"] = result["image_analysis"]
                elif model == "K_programming" and "code" in result:
                    integrated_result["details"]["code"] = result["code"]
                
                # 添加来源信息
                integrated_result["sources"].append({
                    "model": model,
                    "contribution": result
                })
        
        # 计算平均置信度
        if confidence_scores:
            integrated_result["confidence"] = sum(confidence_scores) / len(confidence_scores)
        else:
            integrated_result["confidence"] = 0.7  # 默认置信度
        
        # 如果有历史记忆，增强结果
        if relevant_memories:
            integrated_result["historical_references"] = len(relevant_memories)
            
        # 清理摘要
        integrated_result["summary"] = integrated_result["summary"].strip()
        
        # 如果没有生成摘要，创建默认摘要
        if not integrated_result["summary"]:
            integrated_result["summary"] = "Task processed successfully with multiple models."
        
        return integrated_result
    

    
    async def _update_system_state(self, task_id: str, success: bool, processing_time: float, result: Dict[str, Any]):
        """统一系统状态更新 | Unified system state update"""
        self.system_state["performance_metrics"]["total_tasks"] += 1
        
        if success:
            self.system_state["performance_metrics"]["successful_tasks"] += 1
        else:
            self.system_state["performance_metrics"]["failed_tasks"] += 1
        
        # 更新平均处理时间 | Update average processing time
        total_tasks = self.system_state["performance_metrics"]["total_tasks"]
        current_avg = self.system_state["performance_metrics"]["average_processing_time"]
        self.system_state["performance_metrics"]["average_processing_time"] = (
            (current_avg * (total_tasks - 1) + processing_time) / total_tasks
        )
        
        # 记录任务 | Record task
        task_record = {
            "task_id": task_id,
            "success": success,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        self.system_state["completed_tasks"].append(task_record)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取统一系统状态 | Get unified system status"""
        return {
            "system": {
                "status": self.system_state["status"],
                "language": self.system_state["language"],
                "uptime": time.time() - self.system_state["performance_metrics"]["system_uptime"]
            },
            "performance": self.system_state["performance_metrics"],
            "submodels": self.submodel_registry,
            "emotional_state": self.system_state["emotional_state"],
            "recent_tasks": list(self.system_state["completed_tasks"])[-10:]  # 最近10个任务
        }
    
    def start(self):
        """启动统一核心系统 | Start unified core system"""
        self.is_running = True
        self.system_state["status"] = "running"
        self.system_state["performance_metrics"]["system_uptime"] = time.time()
        
        # 启动监控线程 | Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # 启动优化线程 | Start optimization thread
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        logger.info("统一核心系统已启动 | Unified core system started")
    
    def stop(self):
        """停止统一核心系统 | Stop unified core system"""
        self.is_running = False
        self.system_state["status"] = "stopped"
        logger.info("统一核心系统已停止 | Unified core system stopped")

    # 协作引擎方法 | Collaboration engine methods
    
    def submit_collaboration_task(self, description: str, required_models: List[str],
                                 priority: str = "medium", metadata: Optional[Dict] = None) -> str:
        """提交协作任务 | Submit collaboration task"""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        task = {
            "task_id": task_id,
            "description": description,
            "required_models": required_models,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.pending_tasks[task_id] = task
        self.collaboration_stats["total_tasks_processed"] += 1
        
        # 自动分配任务 | Auto-assign tasks
        self._assign_collaboration_tasks()
        
        logger.info(f"协作任务已提交: {task_id}")
        return task_id
    
    def _assign_collaboration_tasks(self):
        """分配协作任务 | Assign collaboration tasks"""
        if not self.pending_tasks:
            return
        
        # 按优先级排序 | Sort by priority
        sorted_tasks = sorted(self.pending_tasks.values(),
                            key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x["priority"], 2),
                            reverse=True)
        
        for task in sorted_tasks:
            available_models = [m for m in task["required_models"] if m in self.submodel_registry]
            
            if available_models:
                self._execute_collaboration_task(task, available_models)

    def _execute_collaboration_task(self, task: Dict[str, Any], models: List[str]):
        """执行协作任务 | Execute collaboration task"""
        task_id = task["task_id"]
        
        # 移动到活跃任务 | Move to active tasks
        task["status"] = "active"
        task["started_at"] = datetime.now().isoformat()
        self.system_state["active_tasks"][task_id] = self.pending_tasks.pop(task_id)
        
        # 模拟异步执行 | Simulate async execution
        threading.Thread(target=self._process_collaboration_task, 
                        args=(task_id, models), daemon=True).start()
    
    def _process_collaboration_task(self, task_id: str, models: List[str]):
        """处理协作任务 | Process collaboration task"""
        try:
            import time
            time.sleep(1)  # 模拟处理时间
            
            # 更新任务状态 | Update task status
            if task_id in self.system_state["active_tasks"]:
                task = self.system_state["active_tasks"].pop(task_id)
                task["status"] = "completed"
                task["completed_at"] = datetime.now().isoformat()
                task["result"] = f"任务完成，使用了模型: {', '.join(models)}"
                
                self.system_state["completed_tasks"].append(task)
                self.collaboration_stats["successful_collaborations"] += 1
                
                logger.info(f"协作任务完成: {task_id}")
                
        except Exception as e:
            if task_id in self.system_state["active_tasks"]:
                task = self.system_state["active_tasks"].pop(task_id)
                task["status"] = "failed"
                task["error"] = str(e)
                self.failed_tasks[task_id] = task
                
                self.collaboration_stats["failed_collaborations"] += 1
                
                logger.error(f"协作任务失败: {task_id}, 错误: {e}")

    # 优化引擎方法 | Optimization engine methods
    
    def _optimization_loop(self):
        """优化循环 | Optimization loop"""
        while self.is_running:
            try:
                # 收集性能数据 | Collect performance data
                performance_data = self._collect_optimization_data()
                
                # 检查是否需要优化 | Check if optimization needed
                if self._needs_optimization(performance_data):
                    self._execute_optimization(performance_data)
                
                time.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                logger.error(f"优化循环错误: {e}")
                time.sleep(60)
    
    def _collect_optimization_data(self) -> Dict[str, Any]:
        """收集优化数据 | Collect optimization data"""
        import psutil
        import os
        
        try:
            # 使用当前工作目录作为磁盘检查路径
            current_dir = os.getcwd()
            disk_usage = psutil.disk_usage(current_dir).percent
        except Exception:
            # 如果失败，使用C盘
            try:
                disk_usage = psutil.disk_usage('C:\\').percent
            except:
                disk_usage = 0.0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": disk_usage,
            "active_tasks": len(self.active_tasks),
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks)
        }
    
    def _needs_optimization(self, data: Dict[str, Any]) -> bool:
        """检查是否需要优化 | Check if optimization needed"""
        return (
            data["cpu_usage"] > self.optimization_config["cpu_threshold"] or
            data["memory_usage"] > self.optimization_config["memory_threshold"] or
            data["disk_usage"] > self.optimization_config["disk_threshold"] or
            len(self.pending_tasks) > 50
        )
    
    def _execute_optimization(self, data: Dict[str, Any]):
        """执行优化 | Execute optimization"""
        try:
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "triggered_by": data,
                "actions_taken": []
            }
            
            # 简单的优化策略 | Simple optimization strategies
            if data["cpu_usage"] > self.optimization_config["cpu_threshold"]:
                optimization_record["actions_taken"].append("CPU负载过高，减少并发任务")
            
            if data["memory_usage"] > self.optimization_config["memory_threshold"]:
                optimization_record["actions_taken"].append("内存使用过高，清理缓存")
            
            if len(self.pending_tasks) > 50:
                optimization_record["actions_taken"].append("任务队列过长，优先处理高优先级任务")
            
            self.optimization_history.append(optimization_record)
            self.optimization_state["total_optimizations"] += 1
            self.optimization_state["successful_optimizations"] += 1
            self.optimization_state["last_optimization"] = datetime.now().isoformat()
            
            logger.info(f"优化执行完成: {len(optimization_record['actions_taken'])} 个动作")
            
        except Exception as e:
            logger.error(f"优化执行失败: {e}")

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """获取协作统计 | Get collaboration statistics"""
        # 获取最近完成的任务（从completed_tasks字典中获取最后20个）
        recent_tasks = []
        if hasattr(self, 'completed_tasks') and self.completed_tasks:
            # 按完成时间排序，获取最近的20个任务
            sorted_tasks = sorted(
                self.completed_tasks.items(),
                key=lambda x: x[1].get('completed_at', ''),
                reverse=True
            )
            for task_id, task in sorted_tasks[:20]:
                serializable_task = {
                    "task_id": str(task_id),
                    "status": task.get("status", "completed"),
                    "completed_at": task.get("completed_at", ""),
                    "task_type": task.get("task_type", "unknown")
                }
                # 确保所有值都是JSON可序列化的
                for key, value in serializable_task.items():
                    if isinstance(value, datetime):
                        serializable_task[key] = value.isoformat()
                    elif not isinstance(value, (dict, list, str, int, float, bool)) and value is not None:
                        serializable_task[key] = str(value)
                recent_tasks.append(serializable_task)
        
        # 确保协作统计中的所有值都是JSON可序列化的
        serializable_stats = {}
        for key, value in self.collaboration_stats.items():
            if isinstance(value, datetime):
                serializable_stats[key] = value.isoformat()
            elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                serializable_stats[key] = value
            else:
                serializable_stats[key] = str(value)
        
        return {
            **serializable_stats,
            "pending_tasks": len(self.pending_tasks),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "recent_tasks": recent_tasks
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计 | Get optimization statistics"""
        # 将deque转换为可序列化的列表
        recent_optimizations = []
        for opt in list(self.optimization_history)[-10:]:
            if isinstance(opt, dict):
                # 确保所有值都是JSON可序列化的
                serializable_opt = {}
                for key, value in opt.items():
                    if isinstance(value, datetime):
                        serializable_opt[key] = value.isoformat()
                    elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                        serializable_opt[key] = value
                    else:
                        serializable_opt[key] = str(value)
                recent_optimizations.append(serializable_opt)
            else:
                recent_optimizations.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "optimization",
                    "status": "completed",
                    "data": str(opt)
                })
        
        # 处理优化状态中的datetime对象
        optimization_state = {}
        for key, value in self.optimization_state.items():
            if isinstance(value, datetime):
                optimization_state[key] = value.isoformat()
            elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                optimization_state[key] = value
            else:
                optimization_state[key] = str(value)
        
        return {
            **optimization_state,
            "total_optimizations": optimization_state.get("total_optimizations", 0),
            "recent_optimizations": recent_optimizations
        }
    
    def _monitoring_loop(self):
        """统一监控循环 | Unified monitoring loop"""
        while self.is_running:
            try:
                # 更新系统指标 | Update system metrics
                self.system_state["performance_metrics"]["cpu_usage"] = psutil.cpu_percent()
                self.system_state["performance_metrics"]["memory_usage"] = psutil.virtual_memory().percent
                
                # 检查子模型状态 | Check sub-model status
                self._check_submodel_health()
                
                time.sleep(self.config["system"]["monitoring_interval"])
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
    
    def _check_submodel_health(self):
        """统一健康检查 | Unified health check"""
        for model_name, model_info in self.submodel_registry.items():
            try:
                # 简化的健康检查 | Simplified health check
                # 实际实现会发送HTTP请求 | Actual implementation would send HTTP request
                model_info["status"] = "healthy"
            except Exception:
                model_info["status"] = "unhealthy"

# 全局实例 | Global instance
_unified_system = None

def get_unified_system(language='zh', config_path=None) -> UnifiedCoreSystem:
    """获取统一系统实例 | Get unified system instance"""
    global _unified_system
    if _unified_system is None:
        _unified_system = UnifiedCoreSystem(language=language, config_path=config_path)
    return _unified_system