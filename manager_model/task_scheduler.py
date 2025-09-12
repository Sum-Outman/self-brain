# 高级任务调度器 - AGI任务管理与协调
# Advanced Task Scheduler - AGI Task Management and Coordination
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License")

import threading
import time
import json
import logging
import uuid
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
from typing import Dict, List, Any, Optional

class AdvancedTaskScheduler:
    """高级任务调度器，支持智能任务分配和模型协调 | Advanced task scheduler supporting intelligent task allocation and model coordination"""
    
    def __init__(self, model_registry, data_bus):
        """初始化任务调度器 | Initialize task scheduler
        
        参数:
            model_registry: 模型注册表实例 | Model registry instance
            data_bus: 数据总线实例 | Data bus instance
        """
        self.model_registry = model_registry
        self.data_bus = data_bus
        self.tasks = {}  # 存储所有任务 | Store all tasks
        self.task_queue = deque()  # 任务队列 | Task queue
        self.running = False
        self.worker_thread = None
        
        # 任务统计和性能指标
        self.task_statistics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_completion_time": 0.0,
            "task_types": defaultdict(int),
            "model_usage": defaultdict(int),
            "collaboration_efficiency": 0.0,
            "recent_tasks": deque(maxlen=100)
        }
        
        # 模型性能历史
        self.model_performance = defaultdict(lambda: deque(maxlen=100))
        
        # 协作历史记录
        self.collaboration_history = deque(maxlen=1000)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 任务优先级配置
        self.priority_levels = {
            "critical": 10,
            "high": 7,
            "medium": 5,
            "low": 3,
            "background": 1
        }
    
    def start(self):
        """启动任务调度器 | Start task scheduler"""
        if self.running:
            self.logger.warning("任务调度器已在运行 | Task scheduler already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info("任务调度器已启动 | Task scheduler started")
    
    def stop(self):
        """停止任务调度器 | Stop task scheduler"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.logger.info("任务调度器已停止 | Task scheduler stopped")
    
    def schedule_task(self, task_config: Dict[str, Any]) -> str:
        """调度新任务 | Schedule new task
        
        参数:
            task_config: 任务配置 | Task configuration
            
        返回:
            任务ID | Task ID
        """
        task_id = str(uuid.uuid4())
        
        # 创建任务对象
        task = {
            "id": task_id,
            "config": task_config,
            "status": "pending",
            "priority": self._calculate_priority(task_config),
            "created_at": datetime.now(),
            "assigned_models": [],
            "progress": 0.0,
            "result": None,
            "error": None,
            "start_time": None,
            "end_time": None,
            "duration": None,
            "collaboration_level": 0.0,
            "accuracy": 0.8  # 默认准确率
        }
        
        # 添加到任务列表和队列
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        # 更新统计信息
        self.task_statistics["total_tasks"] += 1
        task_type = task_config.get("type", "unknown")
        self.task_statistics["task_types"][task_type] += 1
        
        self.logger.info(f"新任务已调度: {task_id}, 类型: {task_type}, 优先级: {task['priority']}")
        
        return task_id
    
    def _calculate_priority(self, task_config: Dict[str, Any]) -> int:
        """计算任务优先级 | Calculate task priority"""
        # 基于任务类型、情感强度和紧急程度计算优先级
        task_type = task_config.get("type", "unknown")
        emotion = task_config.get("emotion", {})
        emotion_intensity = emotion.get("intensity", 0.5)
        emotion_type = emotion.get("type", "neutral")
        
        # 基础优先级
        base_priority = self.priority_levels.get("medium", 5)
        
        # 情感强度调整
        priority_adjustment = int(emotion_intensity * 4)  # 0-4的调整
        
        # 情感类型调整
        if emotion_type in ["angry", "excited"]:
            priority_adjustment += 2
        elif emotion_type in ["sad", "happy"]:
            priority_adjustment += 1
        
        # 任务类型调整
        if task_type in ["critical_operation", "emergency"]:
            priority_adjustment += 3
        elif task_type in ["time_sensitive", "real_time"]:
            priority_adjustment += 2
        
        final_priority = min(10, max(1, base_priority + priority_adjustment))
        
        return final_priority
    
    def _scheduler_loop(self):
        """调度器主循环 | Scheduler main loop"""
        while self.running:
            try:
                if self.task_queue:
                    # 获取最高优先级的任务
                    task_id = self._get_highest_priority_task()
                    if task_id:
                        self._process_task(task_id)
                
                # 清理已完成的任务
                self._cleanup_completed_tasks()
                
                # 更新性能指标
                self._update_performance_metrics()
                
                time.sleep(0.1)  # 避免CPU占用过高
                
            except Exception as e:
                self.logger.error(f"调度器循环错误: {str(e)}")
                time.sleep(1)
    
    def _get_highest_priority_task(self) -> Optional[str]:
        """获取最高优先级的任务 | Get highest priority task"""
        if not self.task_queue:
            return None
        
        # 找到优先级最高的任务
        highest_priority = -1
        highest_priority_task = None
        
        for task_id in list(self.task_queue):
            task = self.tasks.get(task_id)
            if task and task["status"] == "pending":
                if task["priority"] > highest_priority:
                    highest_priority = task["priority"]
                    highest_priority_task = task_id
        
        return highest_priority_task
    
    def _process_task(self, task_id: str):
        """处理任务 | Process task"""
        task = self.tasks.get(task_id)
        if not task or task["status"] != "pending":
            return
        
        try:
            # 更新任务状态
            task["status"] = "processing"
            task["start_time"] = datetime.now()
            
            # 分配模型
            assigned_models = self._assign_models(task["config"])
            task["assigned_models"] = assigned_models
            
            # 执行任务
            result = self._execute_task(task, assigned_models)
            
            # 更新任务结果
            task["result"] = result
            task["status"] = "completed"
            task["end_time"] = datetime.now()
            task["duration"] = (task["end_time"] - task["start_time"]).total_seconds()
            
            # 计算协作级别
            task["collaboration_level"] = self._calculate_collaboration_level(assigned_models)
            
            # 更新统计信息
            self.task_statistics["completed_tasks"] += 1
            self.task_statistics["recent_tasks"].append(task)
            
            # 记录协作历史
            self.collaboration_history.append({
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "models": assigned_models,
                "collaboration_level": task["collaboration_level"],
                "duration": task["duration"]
            })
            
            self.logger.info(f"任务完成: {task_id}, 耗时: {task['duration']:.2f}s, 协作级别: {task['collaboration_level']:.2f}")
            
        except Exception as e:
            # 任务失败处理
            task["status"] = "failed"
            task["error"] = str(e)
            task["end_time"] = datetime.now()
            if task["start_time"]:
                task["duration"] = (task["end_time"] - task["start_time"]).total_seconds()
            
            self.task_statistics["failed_tasks"] += 1
            self.logger.error(f"任务失败: {task_id}, 错误: {str(e)}")
        
        finally:
            # 从队列中移除任务
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
    
    def _assign_models(self, task_config: Dict[str, Any]) -> List[str]:
        """分配模型执行任务 | Assign models to execute task"""
        task_type = task_config.get("type", "unknown")
        emotion = task_config.get("emotion", {})
        knowledge_suggestions = task_config.get("knowledge_suggestions", [])
        
        # 基于任务类型分配主要模型
        model_mapping = {
            "image_processing": ["D_image"],
            "video_processing": ["E_video"],
            "computer_control": ["H_computer_control"],
            "motion_control": ["J_motion"],
            "knowledge_query": ["I_knowledge"],
            "programming": ["K_programming"],
            "language_processing": ["B_language"],
            "audio_processing": ["C_audio"],
            "sensor_processing": ["G_sensor"],
            "spatial_processing": ["F_spatial"]
        }
        
        primary_models = model_mapping.get(task_type, ["B_language"])
        
        # 根据知识库建议添加辅助模型
        auxiliary_models = []
        for suggestion in knowledge_suggestions:
            if suggestion.get("type") == "model_assistance":
                auxiliary_models.extend(suggestion.get("models", []))
        
        # 情感分析可能需要的模型
        if emotion.get("intensity", 0) > 0.6:
            auxiliary_models.append("B_language")  # 语言模型用于情感表达
        
        # 去除重复模型
        all_models = list(set(primary_models + auxiliary_models))
        
        # 更新模型使用统计
        for model in all_models:
            self.task_statistics["model_usage"][model] += 1
        
        return all_models
    
    def _execute_task(self, task: Dict, models: List[str]) -> Any:
        """执行任务 | Execute task"""
        task_config = task["config"]
        results = {}
        
        # 通过数据总线协调模型执行
        for model_name in models:
            try:
                # 创建数据通道
                channel_id = f"task-{task['id']}-{model_name}"
                if channel_id not in self.data_bus.channels:
                    self.data_bus.create_channel(channel_id, capacity=10, priority=task["priority"])
                
                # 发送任务数据
                task_data = {
                    "task_id": task["id"],
                    "config": task_config,
                    "model": model_name,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 发送消息并等待响应
                response = self.data_bus.send_message(channel_id, task_data, timeout=30)
                
                if response:
                    results[model_name] = response
                    # 更新任务进度
                    task["progress"] = (models.index(model_name) + 1) / len(models) * 100
                
                # 记录模型性能
                self._record_model_performance(model_name, response)
                
            except Exception as e:
                self.logger.error(f"模型 {model_name} 执行失败: {str(e)}")
                results[model_name] = {"status": "error", "message": str(e)}
        
        return results
    
    def _record_model_performance(self, model_name: str, response: Any):
        """记录模型性能 | Record model performance"""
        if response and isinstance(response, dict):
            performance_data = {
                "timestamp": datetime.now().isoformat(),
                "response_time": response.get("response_time", 0),
                "accuracy": response.get("accuracy", 0.8),
                "success": response.get("status") == "success"
            }
            self.model_performance[model_name].append(performance_data)
    
    def _calculate_collaboration_level(self, models: List[str]) -> float:
        """计算协作级别 | Calculate collaboration level"""
        if len(models) <= 1:
            return 0.0  # 单模型任务无协作
        
        # 基于模型数量和类型计算协作级别
        base_level = min(1.0, len(models) / 5.0)  # 最多5个模型
        
        # 模型多样性加分
        model_types = set(model.split('_')[0] for model in models)
        diversity_bonus = len(model_types) / len(models)
        
        # 最终协作级别
        collaboration_level = base_level * 0.7 + diversity_bonus * 0.3
        
        return round(collaboration_level, 2)
    
    def _cleanup_completed_tasks(self):
        """清理已完成的任务 | Clean up completed tasks"""
        # 保留最近1000个任务，清理旧任务
        if len(self.tasks) > 1000:
            # 按完成时间排序，移除最旧的任务
            completed_tasks = [task for task in self.tasks.values() if task["status"] in ["completed", "failed"]]
            completed_tasks.sort(key=lambda x: x.get("end_time", datetime.min))
            
            for task in completed_tasks[:max(0, len(completed_tasks) - 900)]:
                if task["id"] in self.tasks:
                    del self.tasks[task["id"]]
    
    def _update_performance_metrics(self):
        """更新性能指标 | Update performance metrics"""
        # 计算平均完成时间
        completed_tasks = [task for task in self.tasks.values() if task["status"] == "completed" and task.get("duration")]
        if completed_tasks:
            avg_time = sum(task["duration"] for task in completed_tasks) / len(completed_tasks)
            self.task_statistics["avg_completion_time"] = avg_time
        
        # 计算协作效率
        self.task_statistics["collaboration_efficiency"] = self._calculate_overall_collaboration_efficiency()
    
    def _calculate_overall_collaboration_efficiency(self) -> float:
        """计算整体协作效率 | Calculate overall collaboration efficiency"""
        collaborative_tasks = [task for task in self.tasks.values() 
                              if task["status"] == "completed" and len(task.get("assigned_models", [])) > 1]
        
        if not collaborative_tasks:
            return 0.0
        
        total_efficiency = 0.0
        for task in collaborative_tasks:
            # 基于持续时间、准确率和协作级别计算效率
            duration_factor = min(1.0, 30.0 / max(1.0, task.get("duration", 30.0)))
            accuracy = task.get("accuracy", 0.8)
            collaboration_level = task.get("collaboration_level", 0.5)
            
            efficiency = accuracy * duration_factor * collaboration_level
            total_efficiency += efficiency
        
        return round(total_efficiency / len(collaborative_tasks), 3)
    
    def get_task_result(self, task_id: str) -> Optional[Dict]:
        """获取任务结果 | Get task result"""
        return self.tasks.get(task_id)
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息 | Get task statistics"""
        # 计算实时指标
        current_time = datetime.now()
        recent_tasks = [task for task in self.tasks.values() 
                       if task["status"] == "completed" 
                       and (current_time - task.get("end_time", current_time)).total_seconds() < 300]  # 最近5分钟
        
        throughput = len(recent_tasks) / 300  # 任务/秒
        
        return {
            **self.task_statistics,
            "throughput": round(throughput, 3),
            "active_tasks": len([task for task in self.tasks.values() if task["status"] == "processing"]),
            "pending_tasks": len(self.task_queue),
            "model_performance": {model: list(perf) for model, perf in self.model_performance.items()}
        }
    
    def get_all_tasks(self) -> Dict[str, Dict]:
        """获取所有任务 | Get all tasks"""
        return self.tasks
    
    def get_collaboration_history(self) -> List[Dict]:
        """获取协作历史 | Get collaboration history"""
        return list(self.collaboration_history)
    
    def reprioritize_tasks(self, new_priority_config: Dict[str, Any]):
        """重新优先级任务 | Reprioritize tasks"""
        # 实现动态任务重新优先级
        for task_id, task in self.tasks.items():
            if task["status"] == "pending":
                # 基于新配置更新优先级
                task["priority"] = self._calculate_priority_with_config(task["config"], new_priority_config)
        
        self.logger.info("任务优先级已更新 | Task priorities updated")
    
    def _calculate_priority_with_config(self, task_config: Dict[str, Any], priority_config: Dict[str, Any]) -> int:
        """使用配置计算优先级 | Calculate priority with config"""
        # 简化版本，实际中可以更复杂
        base_priority = self.priority_levels.get("medium", 5)
        
        # 应用配置调整
        if "priority_boost" in priority_config:
            base_priority += priority_config["priority_boost"]
        
        return min(10, max(1, base_priority))

# 兼容旧版本的类名
TaskScheduler = AdvancedTaskScheduler
