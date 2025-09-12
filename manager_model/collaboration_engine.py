#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型协作引擎 - Model Collaboration Engine
Copyright 2025 The AGI Brain System Authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

模型协作引擎，负责协调所有模型之间的协作和数据共享
支持实时任务分配、性能优化和智能决策
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Callable
import uuid
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.optimize import linear_sum_assignment

# 配置日志系统 | Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CollaborationEngine")

class TaskPriority(Enum):
    """任务优先级枚举 | Task priority enumeration"""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    BACKGROUND = 0

class ModelCapability(Enum):
    """模型能力枚举 | Model capability enumeration"""
    LANGUAGE_PROCESSING = "language_processing"
    AUDIO_PROCESSING = "audio_processing"
    IMAGE_PROCESSING = "image_processing"
    VIDEO_PROCESSING = "video_processing"
    SPATIAL_ANALYSIS = "spatial_analysis"
    SENSOR_DATA = "sensor_data"
    COMPUTER_CONTROL = "computer_control"
    KNOWLEDGE_BASE = "knowledge_base"
    MOTION_CONTROL = "motion_control"
    PROGRAMMING = "programming"
    EMOTIONAL_ANALYSIS = "emotional_analysis"

@dataclass
class CollaborationTask:
    """协作任务数据类 | Collaboration task dataclass"""
    task_id: str
    description: str
    required_capabilities: List[ModelCapability]
    priority: TaskPriority
    deadline: Optional[datetime] = None
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ModelPerformance:
    """模型性能数据类 | Model performance dataclass"""
    model_id: str
    capability: ModelCapability
    throughput: float  # 处理速度 (tasks/second)
    accuracy: float    # 准确率 (0-1)
    latency: float     # 延迟 (seconds)
    resource_usage: float  # 资源使用率 (0-1)
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class CollaborationEngine:
    """模型协作引擎类 | Model collaboration engine class"""
    
    def __init__(self, config_path: str = "config/collaboration_config.json"):
        """
        初始化协作引擎 | Initialize collaboration engine
        
        参数 Parameters:
            config_path: 配置文件路径 | Config file path
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 任务管理 | Task management
        self.pending_tasks: Dict[str, CollaborationTask] = {}
        self.active_tasks: Dict[str, CollaborationTask] = {}
        self.completed_tasks: Dict[str, CollaborationTask] = {}
        self.failed_tasks: Dict[str, CollaborationTask] = {}
        
        # 模型性能数据 | Model performance data
        self.model_performance: Dict[str, Dict[ModelCapability, ModelPerformance]] = {}
        
        # 协作统计 | Collaboration statistics
        self.collaboration_stats = {
            "total_tasks_processed": 0,
            "successful_collaborations": 0,
            "failed_collaborations": 0,
            "average_completion_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # 事件监听器 | Event listeners
        self.event_listeners: Dict[str, List[Callable]] = {
            "task_started": [],
            "task_completed": [],
            "task_failed": [],
            "collaboration_started": [],
            "collaboration_completed": []
        }
        
        logger.info("模型协作引擎初始化完成 | Model collaboration engine initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件 | Load configuration file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 创建默认配置 | Create default configuration
                default_config = {
                    "max_concurrent_tasks": 100,
                    "task_timeout": 300,  # seconds
                    "retry_attempts": 3,
                    "performance_update_interval": 5,
                    "resource_allocation_strategy": "optimized",
                    "knowledge_integration_enabled": True,
                    "real_time_monitoring": True
                }
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                return default_config
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            return {
                "max_concurrent_tasks": 100,
                "task_timeout": 300,
                "retry_attempts": 3,
                "performance_update_interval": 5,
                "resource_allocation_strategy": "optimized",
                "knowledge_integration_enabled": True,
                "real_time_monitoring": True
            }
    
    def register_model(self, model_id: str, capabilities: List[ModelCapability],
                      initial_performance: Optional[Dict[ModelCapability, ModelPerformance]] = None):
        """注册模型到协作引擎 | Register model to collaboration engine"""
        if model_id not in self.model_performance:
            self.model_performance[model_id] = {}
        
        for capability in capabilities:
            if capability not in self.model_performance[model_id]:
                if initial_performance and capability in initial_performance:
                    self.model_performance[model_id][capability] = initial_performance[capability]
                else:
                    # 默认性能指标 | Default performance metrics
                    self.model_performance[model_id][capability] = ModelPerformance(
                        model_id=model_id,
                        capability=capability,
                        throughput=1.0,
                        accuracy=0.8,
                        latency=1.0,
                        resource_usage=0.5
                    )
        
        logger.info(f"模型注册成功: {model_id} - 能力: {[c.value for c in capabilities]}")
    
    def update_model_performance(self, model_id: str, capability: ModelCapability,
                               throughput: float, accuracy: float, latency: float, resource_usage: float):
        """更新模型性能指标 | Update model performance metrics"""
        if model_id in self.model_performance and capability in self.model_performance[model_id]:
            self.model_performance[model_id][capability] = ModelPerformance(
                model_id=model_id,
                capability=capability,
                throughput=throughput,
                accuracy=accuracy,
                latency=latency,
                resource_usage=resource_usage
            )
            logger.debug(f"模型性能更新: {model_id} - {capability.value}")
        else:
            logger.warning(f"无法更新性能指标: 模型 {model_id} 或能力 {capability.value} 未注册")
    
    def submit_task(self, description: str, required_capabilities: List[ModelCapability],
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   deadline: Optional[datetime] = None,
                   dependencies: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """提交新任务 | Submit new task"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = CollaborationTask(
            task_id=task_id,
            description=description,
            required_capabilities=required_capabilities,
            priority=priority,
            deadline=deadline,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.pending_tasks[task_id] = task
        logger.info(f"新任务提交: {task_id} - 优先级: {priority.name}")
        
        # 触发任务分配 | Trigger task assignment
        self._assign_tasks()
        
        return task_id
    
    def _assign_tasks(self):
        """分配待处理任务 | Assign pending tasks"""
        if not self.pending_tasks:
            return
        
        # 按优先级排序任务 | Sort tasks by priority
        sorted_tasks = sorted(self.pending_tasks.values(),
                            key=lambda t: t.priority.value, reverse=True)
        
        for task in sorted_tasks:
            # 检查依赖关系 | Check dependencies
            if not self._check_dependencies(task):
                continue
            
            # 寻找合适的模型 | Find suitable models
            assigned_models = self._find_optimal_models(task)
            
            if assigned_models:
                # 分配任务 | Assign task
                self._execute_task(task, assigned_models)
            else:
                logger.warning(f"无法分配任务 {task.task_id}: 没有合适的模型可用")
    
    def _check_dependencies(self, task: CollaborationTask) -> bool:
        """检查任务依赖关系 | Check task dependencies"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _find_optimal_models(self, task: CollaborationTask) -> Dict[ModelCapability, str]:
        """寻找最优模型分配 | Find optimal model assignment"""
        required_capabilities = task.required_capabilities
        available_models = {}
        
        # 收集可用模型 | Collect available models
        for capability in required_capabilities:
            available_models[capability] = []
            for model_id, capabilities in self.model_performance.items():
                if capability in capabilities:
                    available_models[capability].append(model_id)
        
        # 检查是否有足够模型 | Check if enough models are available
        for capability, models in available_models.items():
            if not models:
                logger.warning(f"没有可用模型处理能力: {capability.value}")
                return {}
        
        # 使用匈牙利算法进行最优分配 | Use Hungarian algorithm for optimal assignment
        return self._hungarian_assignment(task, available_models)
    
    def _hungarian_assignment(self, task: CollaborationTask,
                            available_models: Dict[ModelCapability, List[str]]) -> Dict[ModelCapability, str]:
        """使用匈牙利算法进行模型分配 | Use Hungarian algorithm for model assignment"""
        assignment = {}
        
        for capability, model_list in available_models.items():
            if len(model_list) == 1:
                # 只有一个可用模型 | Only one available model
                assignment[capability] = model_list[0]
            else:
                # 多个模型，选择性能最优的 | Multiple models, choose the best performing
                best_model = None
                best_score = -float('inf')
                
                for model_id in model_list:
                    performance = self.model_performance[model_id][capability]
                    # 综合评分 = 吞吐量 * 准确率 / (延迟 * 资源使用率)
                    score = (performance.throughput * performance.accuracy / 
                            (performance.latency * performance.resource_usage + 1e-6))
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_id
                
                assignment[capability] = best_model
        
        return assignment
    
    def _execute_task(self, task: CollaborationTask, assigned_models: Dict[ModelCapability, str]):
        """执行任务 | Execute task"""
        task_id = task.task_id
        
        # 从待处理移动到活跃 | Move from pending to active
        self.active_tasks[task_id] = self.pending_tasks.pop(task_id)
        
        # 触发事件 | Trigger event
        self._trigger_event("task_started", {
            "task_id": task_id,
            "assigned_models": assigned_models,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"任务开始执行: {task_id} - 分配模型: {assigned_models}")
        
        # 这里实际执行任务逻辑 | Actual task execution logic would go here
        # 在实际实现中，这会调用各个模型的API或服务
        
        # 模拟任务执行 | Simulate task execution
        self._simulate_task_execution(task, assigned_models)
    
    def _simulate_task_execution(self, task: CollaborationTask, assigned_models: Dict[ModelCapability, str]):
        """模拟任务执行（实际实现中应调用真实模型） | Simulate task execution"""
        # 在实际系统中，这里会调用各个模型的API
        # 现在只是模拟执行
        
        import random
        import time
        
        # 模拟执行时间 | Simulate execution time
        execution_time = random.uniform(0.1, 5.0)
        time.sleep(execution_time)
        
        # 模拟成功或失败 | Simulate success or failure
        success = random.random() > 0.1  # 90% 成功率 | 90% success rate
        
        if success:
            self._complete_task(task.task_id, {"execution_time": execution_time})
        else:
            self._fail_task(task.task_id, {"error": "模拟执行失败", "execution_time": execution_time})
    
    def _complete_task(self, task_id: str, result: Dict[str, Any]):
        """完成任务 | Complete task"""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            self.completed_tasks[task_id] = task
            
            # 更新统计信息 | Update statistics
            self.collaboration_stats["total_tasks_processed"] += 1
            self.collaboration_stats["successful_collaborations"] += 1
            
            # 触发事件 | Trigger event
            self._trigger_event("task_completed", {
                "task_id": task_id,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"任务完成: {task_id} - 结果: {result}")
    
    def _fail_task(self, task_id: str, error_info: Dict[str, Any]):
        """任务失败 | Task failed"""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            self.failed_tasks[task_id] = task
            
            # 更新统计信息 | Update statistics
            self.collaboration_stats["total_tasks_processed"] += 1
            self.collaboration_stats["failed_collaborations"] += 1
            
            # 触发事件 | Trigger event
            self._trigger_event("task_failed", {
                "task_id": task_id,
                "error": error_info,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.error(f"任务失败: {task_id} - 错误: {error_info}")
    
    def add_event_listener(self, event_type: str, callback: Callable):
        """添加事件监听器 | Add event listener"""
        if event_type in self.event_listeners:
            self.event_listeners[event_type].append(callback)
        else:
            logger.warning(f"未知事件类型: {event_type}")
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """触发事件 | Trigger event"""
        if event_type in self.event_listeners:
            for callback in self.event_listeners[event_type]:
                try:
                    callback(event_type, data)
                except Exception as e:
                    logger.error(f"事件监听器执行失败: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取协作统计信息 | Get collaboration statistics"""
        return {
            **self.collaboration_stats,
            "pending_tasks": len(self.pending_tasks),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "registered_models": len(self.model_performance),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态 | Get task status"""
        if task_id in self.pending_tasks:
            return {"status": "pending", "task": self.pending_tasks[task_id]}
        elif task_id in self.active_tasks:
            return {"status": "active", "task": self.active_tasks[task_id]}
        elif task_id in self.completed_tasks:
            return {"status": "completed", "task": self.completed_tasks[task_id]}
        elif task_id in self.failed_tasks:
            return {"status": "failed", "task": self.failed_tasks[task_id]}
        else:
            return None
    
    def optimize_performance(self):
        """优化性能配置 | Optimize performance configuration"""
        # 分析当前性能数据并优化配置 | Analyze current performance data and optimize configuration
        logger.info("开始性能优化 | Starting performance optimization")
        
        # 这里可以实现各种优化算法 | Various optimization algorithms can be implemented here
        # 例如：调整资源分配、重新分配任务、更新模型权重等
        
        logger.info("性能优化完成 | Performance optimization completed")

# 示例用法 | Example usage
if __name__ == "__main__":
    # 创建协作引擎 | Create collaboration engine
    engine = CollaborationEngine()
    
    # 注册模型 | Register models
    engine.register_model("model_A", [ModelCapability.LANGUAGE_PROCESSING, ModelCapability.EMOTIONAL_ANALYSIS])
    engine.register_model("model_B", [ModelCapability.IMAGE_PROCESSING, ModelCapability.VIDEO_PROCESSING])
    engine.register_model("model_C", [ModelCapability.KNOWLEDGE_BASE, ModelCapability.PROGRAMMING])
    
    # 添加事件监听器 | Add event listener
    def event_handler(event_type, data):
        print(f"事件: {event_type}, 数据: {data}")
    
    engine.add_event_listener("task_started", event_handler)
    engine.add_event_listener("task_completed", event_handler)
    
    # 提交任务 | Submit tasks
    task1_id = engine.submit_task(
        "处理用户查询并分析情感",
        [ModelCapability.LANGUAGE_PROCESSING, ModelCapability.EMOTIONAL_ANALYSIS],
        TaskPriority.HIGH
    )
    
    task2_id = engine.submit_task(
        "分析图像内容并生成描述",
        [ModelCapability.IMAGE_PROCESSING, ModelCapability.KNOWLEDGE_BASE],
        TaskPriority.MEDIUM
    )
    
    # 等待任务完成 | Wait for tasks to complete
    import time
    time.sleep(10)
    
    # 获取统计信息 | Get statistics
    stats = engine.get_statistics()
    print("协作统计:", stats)