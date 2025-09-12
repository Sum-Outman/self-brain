# 优化引擎 - AGI系统性能优化与资源管理
# Optimization Engine - AGI System Performance Optimization and Resource Management
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License")

import logging
import threading
import time
import json
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
import psutil
from typing import Dict, List, Any, Optional
from .language_resources import LanguageManager  # 新增语言管理器导入

class OptimizationEngine:
    """优化引擎，负责系统性能优化和资源管理 | Optimization engine responsible for system performance optimization and resource management"""
    
    def __init__(self, model_registry, data_bus, language_manager=None):
        """初始化优化引擎 | Initialize optimization engine
        
        参数:
            model_registry: 模型注册表实例 | Model registry instance
            data_bus: 数据总线实例 | Data bus instance
            language_manager: 语言管理器实例 | Language manager instance
        """
        self.model_registry = model_registry
        self.data_bus = data_bus
        self.language = language_manager or LanguageManager()  # 默认创建语言管理器
        
        # 性能监控配置
        self.monitoring_config = {
            "cpu_threshold": 80.0,   # CPU使用率阈值(%)
            "memory_threshold": 75.0,  # 内存使用率阈值(%)
            "network_threshold": 50.0,  # 网络使用率阈值(Mbps)
            "disk_threshold": 85.0,   # 磁盘使用率阈值(%)
            "check_interval": 5.0,    # 检查间隔(秒)
            "optimization_cooldown": 30.0  # 优化冷却时间(秒)
        }
        
        # 性能历史记录
        self.performance_history = deque(maxlen=1000)
        
        # 优化策略配置
        self.optimization_strategies = {
            "load_balancing": {
                "enabled": True,
                "weight": 0.4,
                "effectiveness": 0.8
            },
            "resource_reallocation": {
                "enabled": True,
                "weight": 0.3,
                "effectiveness": 0.7
            },
            "model_prioritization": {
                "enabled": True,
                "weight": 0.2,
                "effectiveness": 0.9
            },
            "caching_optimization": {
                "enabled": True,
                "weight": 0.1,
                "effectiveness": 0.6
            }
        }
        
        # 优化状态
        self.optimization_state = {
            "last_optimization": None,
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "average_improvement": 0.0,
            "current_efficiency": 0.8,
            "resource_utilization": {
                "cpu": 0.0,
                "memory": 0.0,
                "network": 0.0,
                "disk": 0.0
            }
        }
        
        # 设置多语言日志
        self.logger = logging.getLogger(__name__)
        self.logger = self.language.get_logger(self.logger)  # 包装日志器支持多语言
        
        # 优化线程控制
        self.running = False
        self.optimization_thread = None
        
        # 最后优化时间（用于冷却）
        self.last_optimization_time = datetime.min
    
    def start(self):
        """启动优化引擎 | Start optimization engine"""
        if self.running:
            self.logger.warning(self.language.get_text("optimization_engine_already_running"))
            return
        
        self.running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        self.logger.info("优化引擎已启动 | Optimization engine started")
    
    def stop(self):
        """停止优化引擎 | Stop optimization engine"""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        self.logger.info(self.language.get_text("optimization_engine_stopped"))
    
    def _optimization_loop(self):
        """优化主循环 | Optimization main loop"""
        while self.running:
            try:
                # 收集系统性能数据（包含多语言标签）
                performance_data = self._collect_performance_data()
                performance_data['language'] = self.language.current_language  # 添加当前语言标签
                
                # 检查是否需要优化
                if self._needs_optimization(performance_data):
                    # 执行优化
                    optimization_result = self._execute_optimization(performance_data)
                    
                    # 记录优化结果
                    self._record_optimization(performance_data, optimization_result)
                
                # 等待下一次检查
                time.sleep(self.monitoring_config["check_interval"])
                
            except Exception as e:
                self.logger.error(f"优化循环错误: {str(e)}")
                time.sleep(10)  # 错误时等待更长时间
    
    def _collect_performance_data(self) -> Dict[str, Any]:
        """收集系统性能数据 | Collect system performance data"""
        try:
            # 获取系统资源使用情况
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
            network_io = psutil.net_io_counters()
            
            # 获取模型性能数据
            model_performance = self._get_model_performance()
            
            # 获取任务队列状态
            task_queue_status = self._get_task_queue_status()
            
            performance_data = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory_info.percent,
                    "memory_available": memory_info.available / (1024 ** 3),  # GB
                    "disk_usage": disk_usage.percent,
                    "disk_free": disk_usage.free / (1024 ** 3),  # GB
                    "network_sent": network_io.bytes_sent / (1024 ** 2),  # MB
                    "network_received": network_io.bytes_recv / (1024 ** 2)  # MB
                },
                "model_performance": model_performance,
                "task_queue": task_queue_status,
                "process_count": len(psutil.pids()),
                "system_load": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            }
            
            # 更新性能历史
            self.performance_history.append(performance_data)
            
            # 更新当前资源利用率
            self.optimization_state["resource_utilization"].update({
                "cpu": cpu_percent,
                "memory": memory_info.percent,
                "network": (network_io.bytes_sent + network_io.bytes_recv) / (1024 ** 2),
                "disk": disk_usage.percent
            })
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"收集性能数据失败: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_usage": 0.0,
                    "memory_usage": 0.0,
                    "memory_available": 0.0,
                    "disk_usage": 0.0,
                    "disk_free": 0.0,
                    "network_sent": 0.0,
                    "network_received": 0.0
                },
                "model_performance": {},
                "task_queue": {"pending": 0, "processing": 0},
                "process_count": 0,
                "system_load": 0.0
            }
    
    def _get_model_performance(self) -> Dict[str, Any]:
        """获取模型性能数据 | Get model performance data"""
        # 这里应该从模型注册表获取模型性能数据
        # 暂时返回模拟数据
        model_performance = {}
        
        for model_name in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]:
            model_performance[model_name] = {
                "cpu_usage": np.random.uniform(5, 25),
                "memory_usage": np.random.uniform(10, 50),
                "response_time": np.random.uniform(0.1, 1.0),
                "throughput": np.random.uniform(10, 100),
                "error_rate": np.random.uniform(0.01, 0.1)
            }
        
        return model_performance
    
    def _get_task_queue_status(self) -> Dict[str, int]:
        """获取任务队列状态 | Get task queue status"""
        # 这里应该从任务调度器获取任务队列状态
        # 暂时返回模拟数据
        return {
            "pending": np.random.randint(0, 20),
            "processing": np.random.randint(0, 10),
            "completed": np.random.randint(0, 100),
            "failed": np.random.randint(0, 5)
        }
    
    def _needs_optimization(self, performance_data: Dict[str, Any]) -> bool:
        """检查是否需要优化 | Check if optimization is needed"""
        # 检查冷却时间
        current_time = datetime.now()
        time_since_last_opt = (current_time - self.last_optimization_time).total_seconds()
        if time_since_last_opt < self.monitoring_config["optimization_cooldown"]:
            return False
        
        system_data = performance_data["system"]
        
        # 检查各项指标是否超过阈值
        needs_optimization = (
            system_data["cpu_usage"] > self.monitoring_config["cpu_threshold"] or
            system_data["memory_usage"] > self.monitoring_config["memory_threshold"] or
            system_data["disk_usage"] > self.monitoring_config["disk_threshold"] or
            (system_data["network_sent"] + system_data["network_received"]) > 
            self.monitoring_config["network_threshold"]
        )
        
        # 检查任务队列状态
        task_queue = performance_data["task_queue"]
        if task_queue["pending"] > 50 or task_queue["failed"] > 10:
            needs_optimization = True
        
        # 检查模型性能
        for model_name, model_data in performance_data["model_performance"].items():
            if (model_data["error_rate"] > 0.2 or 
                model_data["response_time"] > 2.0 or
                model_data["cpu_usage"] > 30.0):
                needs_optimization = True
                break
        
        return needs_optimization
    
    def _execute_optimization(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行优化 | Execute optimization"""
        optimization_result = {
            "timestamp": datetime.now().isoformat(),
            "strategies_applied": [],
            "improvements": {},
            "resources_saved": {},
            "execution_time": 0.0,
            "success": True
        }
        
        start_time = time.time()
        
        try:
            # 分析性能问题
            performance_issues = self._analyze_performance_issues(performance_data)
            
            # 选择优化策略
            selected_strategies = self._select_optimization_strategies(performance_issues)
            
            # 应用优化策略
            for strategy in selected_strategies:
                strategy_result = self._apply_optimization_strategy(strategy, performance_data)
                
                if strategy_result["success"]:
                    optimization_result["strategies_applied"].append(strategy)
                    optimization_result["improvements"].update(strategy_result.get("improvements", {}))
                    optimization_result["resources_saved"].update(strategy_result.get("resources_saved", {}))
            
            # 计算执行时间
            optimization_result["execution_time"] = time.time() - start_time
            
            # 更新优化状态
            self.optimization_state["total_optimizations"] += 1
            self.optimization_state["successful_optimizations"] += 1
            self.optimization_state["last_optimization"] = datetime.now().isoformat()
            
            # 更新最后优化时间
            self.last_optimization_time = datetime.now()
            
        except Exception as e:
            optimization_result["success"] = False
            optimization_result["error"] = str(e)
            self.logger.error(f"优化执行失败: {str(e)}")
        
        return optimization_result

    def _analyze_performance_issues(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """分析性能问题 | Analyze performance issues"""
        issues = {
            "cpu_bottleneck": 0.0,
            "memory_bottleneck": 0.0,
            "network_bottleneck": 0.0,
            "disk_bottleneck": 0.0,
            "model_inefficiency": 0.0,
            "task_overload": 0.0
        }
        
        system = performance_data["system"]
        model_perf = performance_data["model_performance"]
        task_queue = performance_data["task_queue"]
        
        # CPU瓶颈分析
        if system["cpu_usage"] > self.monitoring_config["cpu_threshold"]:
            issues["cpu_bottleneck"] = min(1.0, (system["cpu_usage"] - self.monitoring_config["cpu_threshold"]) / 20.0)
        
        # 内存瓶颈分析
        if system["memory_usage"] > self.monitoring_config["memory_threshold"]:
            issues["memory_bottleneck"] = min(1.0, (system["memory_usage"] - self.monitoring_config["memory_threshold"]) / 25.0)
        
        # 网络瓶颈分析
        total_network = system["network_sent"] + system["network_received"]
        if total_network > self.monitoring_config["network_threshold"]:
            issues["network_bottleneck"] = min(1.0, (total_network - self.monitoring_config["network_threshold"]) / 20.0)
        
        # 磁盘瓶颈分析
        if system["disk_usage"] > self.monitoring_config["disk_threshold"]:
            issues["disk_bottleneck"] = min(1.0, (system["disk_usage"] - self.monitoring_config["disk_threshold"]) / 15.0)
        
        # 模型效率问题分析
        inefficient_models = 0
        for model_data in model_perf.values():
            if model_data["error_rate"] > 0.15 or model_data["response_time"] > 1.5:
                inefficient_models += 1
        
        if inefficient_models > 0:
            issues["model_inefficiency"] = min(1.0, inefficient_models / len(model_perf))
        
        # 任务过载分析
        if task_queue["pending"] > 30 or task_queue["processing"] > 15:
            pending_score = min(1.0, task_queue["pending"] / 50.0)
            processing_score = min(1.0, task_queue["processing"] / 20.0)
            issues["task_overload"] = max(pending_score, processing_score)
        
        return issues
    
    def _select_optimization_strategies(self, performance_issues: Dict[str, float]) -> List[str]:
        """选择优化策略 | Select optimization strategies"""
        selected_strategies = []
        threshold = 0.3  # 问题严重度阈值
        
        # 根据问题类型选择策略
        if performance_issues["cpu_bottleneck"] > threshold:
            selected_strategies.append("load_balancing")
            selected_strategies.append("model_prioritization")
        
        if performance_issues["memory_bottleneck"] > threshold:
            selected_strategies.append("resource_reallocation")
            selected_strategies.append("caching_optimization")
        
        if performance_issues["network_bottleneck"] > threshold:
            selected_strategies.append("load_balancing")
        
        if performance_issues["disk_bottleneck"] > threshold:
            selected_strategies.append("caching_optimization")
        
        if performance_issues["model_inefficiency"] > threshold:
            selected_strategies.append("model_prioritization")
        
        if performance_issues["task_overload"] > threshold:
            selected_strategies.append("load_balancing")
            selected_strategies.append("model_prioritization")
        
        # 去重并确保策略已启用
        selected_strategies = list(set(selected_strategies))
        return [s for s in selected_strategies 
                if s in self.optimization_strategies and self.optimization_strategies[s]["enabled"]]
    
    def _apply_optimization_strategy(self, strategy: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """应用优化策略 | Apply optimization strategy"""
        result = {
            "success": True,
            "improvements": {},
            "resources_saved": {}
        }
        
        try:
            if strategy == "load_balancing":
                # 负载均衡：重新分配任务到不同模型实例
                # 模拟效果：减少CPU使用5-10%，减少内存使用3-5%
                result["improvements"] = {
                    "cpu_usage": -np.random.uniform(5.0, 10.0),
                    "memory_usage": -np.random.uniform(3.0, 5.0)
                }
                result["resources_saved"] = {
                    "cpu": np.random.uniform(5.0, 10.0),
                    "memory": np.random.uniform(3.0, 5.0)
                }
                self.logger.info("应用负载均衡策略 | Applied load balancing strategy")
                
            elif strategy == "resource_reallocation":
                # 资源重分配：调整模型资源配额
                # 模拟效果：减少CPU使用3-8%，减少内存使用5-10%
                result["improvements"] = {
                    "cpu_usage": -np.random.uniform(3.0, 8.0),
                    "memory_usage": -np.random.uniform(5.0, 10.0)
                }
                result["resources_saved"] = {
                    "cpu": np.random.uniform(3.0, 8.0),
                    "memory": np.random.uniform(5.0, 10.0)
                }
                self.logger.info("应用资源重分配策略 | Applied resource reallocation strategy")
                
            elif strategy == "model_prioritization":
                # 模型优先级：设置任务优先级队列
                # 模拟效果：减少任务队列长度20-40%，提高处理速度10-20%
                result["improvements"] = {
                    "pending_tasks": -np.random.uniform(20.0, 40.0),
                    "processing_time": -np.random.uniform(10.0, 20.0)
                }
                result["resources_saved"] = {
                    "tasks": np.random.uniform(20.0, 40.0),
                    "time": np.random.uniform(10.0, 20.0)
                }
                self.logger.info("应用模型优先级策略 | Applied model prioritization strategy")
                
            elif strategy == "caching_optimization":
                # 缓存优化：管理数据缓存
                # 模拟效果：减少磁盘使用5-15%，减少网络流量10-20%
                result["improvements"] = {
                    "disk_usage": -np.random.uniform(5.0, 15.0),
                    "network_usage": -np.random.uniform(10.0, 20.0)
                }
                result["resources_saved"] = {
                    "disk": np.random.uniform(5.0, 15.0),
                    "network": np.random.uniform(10.0, 20.0)
                }
                self.logger.info("应用缓存优化策略 | Applied caching optimization strategy")
                
            else:
                result["success"] = False
                result["error"] = f"未知策略: {strategy} | Unknown strategy"
                
        except Exception as e:
            result["success"] = False
            result["error"] = f"策略应用失败: {str(e)} | Strategy application failed"
            self.logger.error(f"应用策略 {strategy} 失败: {str(e)}")
            
        return result
    
    def _record_optimization(self, performance_data: Dict[str, Any], optimization_result: Dict[str, Any]):
        """记录优化结果 | Record optimization result"""
        try:
            # 创建日志条目（包含多语言元数据）
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "language": self.language.current_language,
                "pre_optimization": performance_data,
                "optimization": optimization_result,
                "post_optimization": self._collect_performance_data(),  # 收集优化后数据
                "knowledge_ref": self._get_knowledge_reference()  # 添加知识库参考
            }
            
            # 计算效率提升
            pre_cpu = performance_data["system"]["cpu_usage"]
            post_cpu = log_entry["post_optimization"]["system"]["cpu_usage"]
            cpu_improvement = pre_cpu - post_cpu
            
            # 更新优化状态
            self.optimization_state["average_improvement"] = (
                (self.optimization_state["average_improvement"] * 
                 (self.optimization_state["total_optimizations"] - 1) + 
                 cpu_improvement) / 
                self.optimization_state["total_optimizations"]
            )
            
            self.optimization_state["current_efficiency"] = min(1.0, 
                self.optimization_state["current_efficiency"] + 0.05 * cpu_improvement / 100)
            
            # 保存到日志文件
            log_file = "logs/optimization_log.json"
            try:
                # 读取现有日志
                existing_logs = []
                try:
                    with open(log_file, "r") as f:
                        existing_logs = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                
                # 添加新条目
                existing_logs.append(log_entry)
                
                # 写入文件
                with open(log_file, "w") as f:
                    json.dump(existing_logs, f, indent=2)
                    
            except Exception as e:
                self.logger.error(f"保存优化日志失败: {str(e)}")
                
            # 多语言日志输出
            improvement_msg = self.language.get_text("optimization_recorded", 
                                                    improvement=f"{cpu_improvement:.2f}%")
            self.logger.info(improvement_msg)
            
            # 发送到仪表盘
            self._send_to_dashboard(log_entry)
            
        except Exception as e:
            self.logger.error(self.language.get_text("optimization_record_failed", error=str(e)))
            
    def _get_knowledge_reference(self) -> Dict[str, Any]:
        """获取知识库参考数据 | Get knowledge base reference data"""
        try:
            # 连接知识库专家模型
            knowledge_model = self.model_registry.get_model("I_knowledge")
            if knowledge_model:
                return knowledge_model.get_optimization_reference()
        except Exception:
            pass
        return {"error": "Knowledge model not available"}
    
    def _send_to_dashboard(self, data: Dict[str, Any]):
        """发送数据到实时仪表盘 | Send data to real-time dashboard"""
        try:
            # 通过数据总线发送到Web仪表盘
            self.data_bus.publish("optimization_metrics", data)
        except Exception as e:
            self.logger.error(self.language.get_text("dashboard_update_failed", error=str(e)))
