# 自主学习模块 - AGI自我学习与优化
# Self-Learning Module - AGI Self-Learning and Optimization
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License")

import json
import logging
import threading
import time
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
import pickle
import os
from typing import Dict, List, Any, Optional

class SelfLearningModule:
    """自主学习模块，支持AGI的持续学习和自我改进 | Self-learning module supporting continuous learning and self-improvement"""
    
    def __init__(self, data_dir="data_bus_storage/learning"):
        """初始化自主学习模块 | Initialize self-learning module
        
        参数:
            data_dir: 学习数据存储目录 | Learning data storage directory
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 学习状态和统计
        self.learning_state = {
            "total_learning_cycles": 0,
            "last_learning_session": None,
            "knowledge_growth_rate": 0.0,
            "skill_acquisition": defaultdict(dict),
            "learning_efficiency": 0.7,
            "adaptation_level": 0.5
        }
        
        # 学习历史记录
        self.learning_history = deque(maxlen=10000)
        
        # 经验回放缓冲区
        self.experience_replay = deque(maxlen=5000)
        
        # 学习策略配置
        self.learning_strategies = {
            "supervised_learning": {
                "enabled": True,
                "weight": 0.4,
                "effectiveness": 0.8
            },
            "reinforcement_learning": {
                "enabled": True,
                "weight": 0.3,
                "effectiveness": 0.7
            },
            "unsupervised_learning": {
                "enabled": True,
                "weight": 0.2,
                "effectiveness": 0.6
            },
            "transfer_learning": {
                "enabled": True,
                "weight": 0.1,
                "effectiveness": 0.9
            }
        }
        
        # 学习目标和要求
        self.learning_objectives = {
            "accuracy_improvement": 0.8,
            "efficiency_improvement": 0.7,
            "knowledge_expansion": 0.9,
            "skill_diversification": 0.6
        }
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 学习线程控制
        self.running = False
        self.learning_thread = None
        
        # 加载之前的学习状态
        self._load_learning_state()
    
    def start(self):
        """启动自主学习模块 | Start self-learning module"""
        if self.running:
            self.logger.warning("自主学习模块已在运行 | Self-learning module already running")
            return
        
        self.running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        self.logger.info("自主学习模块已启动 | Self-learning module started")
    
    def stop(self):
        """停止自主学习模块 | Stop self-learning module"""
        self.running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        
        # 保存学习状态
        self._save_learning_state()
        self.logger.info("自主学习模块已停止 | Self-learning module stopped")
    
    def _learning_loop(self):
        """自主学习主循环 | Self-learning main loop"""
        while self.running:
            try:
                # 执行学习周期
                self._execute_learning_cycle()
                
                # 间隔学习时间（根据学习效率调整）
                sleep_time = max(10, 60 * (1 - self.learning_state["learning_efficiency"]))
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"自主学习循环错误: {str(e)}")
                time.sleep(30)
    
    def _execute_learning_cycle(self):
        """执行学习周期 | Execute learning cycle"""
        cycle_start = datetime.now()
        
        try:
            # 收集学习数据
            learning_data = self._collect_learning_data()
            
            if not learning_data:
                self.logger.warning("无学习数据可用 | No learning data available")
                return
            
            # 分析学习需求
            learning_needs = self._analyze_learning_needs(learning_data)
            
            # 选择学习策略
            selected_strategy = self._select_learning_strategy(learning_needs)
            
            # 执行学习
            learning_results = self._execute_learning(learning_data, selected_strategy)
            
            # 评估学习效果
            effectiveness = self._evaluate_learning_effectiveness(learning_results)
            
            # 更新学习状态
            self._update_learning_state(learning_results, effectiveness)
            
            # 记录学习历史
            self._record_learning_history(learning_data, selected_strategy, learning_results, effectiveness)
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.logger.info(f"学习周期完成: 耗时{cycle_duration:.2f}s, 效果{effectiveness:.3f}")
            
        except Exception as e:
            self.logger.error(f"学习周期执行失败: {str(e)}")
    
    def _collect_learning_data(self) -> Dict[str, Any]:
        """收集学习数据 | Collect learning data"""
        learning_data = {
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": self._get_performance_metrics(),
            "task_statistics": self._get_task_statistics(),
            "model_performance": self._get_model_performance(),
            "user_interactions": self._get_user_interaction_patterns(),
            "error_patterns": self._get_error_patterns(),
            "environment_context": self._get_environment_context()
        }
        
        # 添加到经验回放缓冲区
        self.experience_replay.append(learning_data)
        
        return learning_data
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标 | Get performance metrics"""
        # 这里应该从核心系统获取实时性能数据
        # 暂时返回模拟数据
        return {
            "cpu_usage": np.random.uniform(0.1, 0.8),
            "memory_usage": np.random.uniform(0.2, 0.7),
            "response_time": np.random.uniform(0.05, 0.5),
            "accuracy": np.random.uniform(0.7, 0.95)
        }
    
    def _get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计 | Get task statistics"""
        # 这里应该从任务调度器获取任务数据
        # 暂时返回模拟数据
        return {
            "completed_tasks": np.random.randint(10, 100),
            "failed_tasks": np.random.randint(0, 5),
            "avg_completion_time": np.random.uniform(1.0, 10.0),
            "task_types": {"language": 40, "vision": 30, "audio": 20, "other": 10}
        }
    
    def _get_model_performance(self) -> Dict[str, Any]:
        """获取模型性能 | Get model performance"""
        # 这里应该从模型注册表获取模型性能数据
        # 暂时返回模拟数据
        return {
            "B_language": {"accuracy": 0.85, "response_time": 0.2},
            "D_image": {"accuracy": 0.78, "response_time": 0.4},
            "C_audio": {"accuracy": 0.82, "response_time": 0.3},
            "I_knowledge": {"accuracy": 0.9, "response_time": 0.5}
        }
    
    def _get_user_interaction_patterns(self) -> Dict[str, Any]:
        """获取用户交互模式 | Get user interaction patterns"""
        # 这里应该从用户界面获取交互数据
        # 暂时返回模拟数据
        return {
            "frequent_requests": ["image_generation", "translation", "question_answering"],
            "preferred_modalities": ["text", "voice"],
            "interaction_frequency": "high",
            "satisfaction_level": 0.8
        }
    
    def _get_error_patterns(self) -> Dict[str, Any]:
        """获取错误模式 | Get error patterns"""
        # 这里应该从日志和错误报告中提取错误模式
        # 暂时返回模拟数据
        return {
            "common_errors": ["timeout", "memory_overflow", "model_not_found"],
            "error_frequency": {"timeout": 5, "memory_overflow": 2, "model_not_found": 1},
            "error_severity": {"timeout": "medium", "memory_overflow": "high", "model_not_found": "low"}
        }
    
    def _get_environment_context(self) -> Dict[str, Any]:
        """获取环境上下文 | Get environment context"""
        # 这里应该从系统环境获取上下文信息
        # 暂时返回模拟数据
        return {
            "time_of_day": datetime.now().hour,
            "system_load": np.random.uniform(0.1, 0.9),
            "network_status": "stable",
            "resource_availability": "sufficient"
        }
    
    def _analyze_learning_needs(self, learning_data: Dict[str, Any]) -> Dict[str, float]:
        """分析学习需求 | Analyze learning needs"""
        needs = {
            "accuracy_improvement": 0.0,
            "efficiency_improvement": 0.0,
            "knowledge_expansion": 0.0,
            "skill_diversification": 0.0,
            "error_reduction": 0.0
        }
        
        # 分析性能指标
        metrics = learning_data["performance_metrics"]
        if metrics["accuracy"] < 0.8:
            needs["accuracy_improvement"] = 0.8 - metrics["accuracy"]
        
        if metrics["response_time"] > 0.3:
            needs["efficiency_improvement"] = min(1.0, metrics["response_time"] / 2.0)
        
        # 分析任务统计
        stats = learning_data["task_statistics"]
        if stats["failed_tasks"] > 0:
            needs["error_reduction"] = min(1.0, stats["failed_tasks"] / stats["completed_tasks"])
        
        # 分析模型性能
        model_perf = learning_data["model_performance"]
        for model, perf in model_perf.items():
            if perf["accuracy"] < 0.75:
                needs["accuracy_improvement"] = max(needs["accuracy_improvement"], 0.75 - perf["accuracy"])
        
        # 确保至少有一些学习需求
        if sum(needs.values()) == 0:
            needs["knowledge_expansion"] = 0.5  # 默认知识扩展需求
        
        return needs
    
    def _select_learning_strategy(self, learning_needs: Dict[str, float]) -> str:
        """选择学习策略 | Select learning strategy"""
        # 基于学习需求选择最合适的策略
        strategy_scores = {}
        
        for strategy, config in self.learning_strategies.items():
            if not config["enabled"]:
                continue
            
            # 计算策略得分
            score = 0.0
            
            if strategy == "supervised_learning":
                # 监督学习适合准确性和错误减少
                score = (learning_needs["accuracy_improvement"] * 0.6 + 
                        learning_needs["error_reduction"] * 0.4)
            
            elif strategy == "reinforcement_learning":
                # 强化学习适合效率改进
                score = learning_needs["efficiency_improvement"] * 0.8
            
            elif strategy == "unsupervised_learning":
                # 无监督学习适合知识扩展
                score = learning_needs["knowledge_expansion"] * 0.7
            
            elif strategy == "transfer_learning":
                # 迁移学习适合技能多样化
                score = (learning_needs["skill_diversification"] * 0.5 +
                        learning_needs["knowledge_expansion"] * 0.5)
            
            strategy_scores[strategy] = score * config["effectiveness"] * config["weight"]
        
        if not strategy_scores:
            return "supervised_learning"  # 默认策略
        
        # 选择得分最高的策略
        return max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    def _execute_learning(self, learning_data: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """执行学习 | Execute learning"""
        results = {
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
            "effectiveness": 0.0,
            "improvements": {},
            "resources_used": {},
            "self_upgrades": []
        }
        
        try:
            if strategy == "supervised_learning":
                results.update(self._execute_supervised_learning(learning_data))
            
            elif strategy == "reinforcement_learning":
                results.update(self._execute_reinforcement_learning(learning_data))
            
            elif strategy == "unsupervised_learning":
                results.update(self._execute_unsupervised_learning(learning_data))
            
            elif strategy == "transfer_learning":
                results.update(self._execute_transfer_learning(learning_data))
            
            # 执行自我升级检查（每10个学习周期一次）
            if self.learning_state["total_learning_cycles"] % 10 == 0:
                upgrade_results = self._execute_self_upgrading(learning_data)
                results["self_upgrades"] = upgrade_results.get("upgrades_applied", [])
                results["upgrade_effectiveness"] = upgrade_results.get("effectiveness", 0.0)
            
            # 记录资源使用
            results["resources_used"] = {
                "computation_time": np.random.uniform(1.0, 10.0),
                "memory_usage": np.random.uniform(0.1, 0.5),
                "data_processed": len(str(learning_data)) / 1024  # KB
            }
            
        except Exception as e:
            self.logger.error(f"学习执行失败: {str(e)}")
            results["error"] = str(e)
            results["effectiveness"] = 0.1  # 最低效果
        
        return results
    
    def _execute_supervised_learning(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行监督学习 | Execute supervised learning"""
        # 这里实现具体的监督学习逻辑
        # 暂时返回模拟结果
        return {
            "effectiveness": np.random.uniform(0.6, 0.9),
            "improvements": {
                "accuracy": np.random.uniform(0.05, 0.15),
                "error_rate": -np.random.uniform(0.1, 0.3)
            },
            "learning_type": "supervised"
        }
    
    def _execute_reinforcement_learning(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行强化学习 | Execute reinforcement learning"""
        # 这里实现具体的强化学习逻辑
        # 暂时返回模拟结果
        return {
            "effectiveness": np.random.uniform(0.5, 0.8),
            "improvements": {
                "efficiency": np.random.uniform(0.1, 0.25),
                "response_time": -np.random.uniform(0.05, 0.2)
            },
            "learning_type": "reinforcement"
        }
    
    def _execute_unsupervised_learning(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行无监督学习 | Execute unsupervised learning"""
        # 这里实现具体的无监督学习逻辑
        # 暂时返回模拟结果
        return {
            "effectiveness": np.random.uniform(0.4, 0.7),
            "improvements": {
                "knowledge_coverage": np.random.uniform(0.15, 0.3),
                "pattern_recognition": np.random.uniform(0.1, 0.25)
            },
            "learning_type": "unsupervised"
        }
    
    def _execute_transfer_learning(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行迁移学习 | Execute transfer learning"""
        # 这里实现具体的迁移学习逻辑
        # 暂时返回模拟结果
        return {
            "effectiveness": np.random.uniform(0.7, 0.95),
            "improvements": {
                "skill_transfer": np.random.uniform(0.2, 0.4),
                "adaptation_speed": np.random.uniform(0.15, 0.35)
            },
            "learning_type": "transfer"
        }

    def _execute_self_upgrading(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行自我升级 - 修改自身代码以改进性能 | Execute self-upgrading - modify own code to improve performance"""
        upgrade_results = {
            "upgrades_applied": [],
            "effectiveness": 0.0,
            "performance_improvement": 0.0,
            "files_modified": [],
            "code_changes": 0
        }
        
        try:
            # 分析学习数据以识别改进机会
            improvement_opportunities = self._analyze_for_self_upgrades(learning_data)
            
            if not improvement_opportunities:
                self.logger.info("未发现自我升级机会 | No self-upgrade opportunities found")
                return upgrade_results
            
            # 获取编程模型实例
            programming_model = self._get_programming_model()
            if not programming_model:
                self.logger.warning("编程模型不可用，无法执行自我升级 | Programming model not available for self-upgrading")
                return upgrade_results
            
            # 应用升级建议
            successful_upgrades = 0
            for opportunity in improvement_opportunities:
                if successful_upgrades >= 3:  # 限制每次升级的数量
                    break
                    
                upgrade_result = self._apply_self_upgrade(programming_model, opportunity)
                if upgrade_result["status"] == "success":
                    upgrade_results["upgrades_applied"].append(upgrade_result)
                    upgrade_results["effectiveness"] += upgrade_result.get("effectiveness", 0.0)
                    upgrade_results["performance_improvement"] += upgrade_result.get("performance_improvement", 0.0)
                    upgrade_results["files_modified"].extend(upgrade_result.get("files_modified", []))
                    upgrade_results["code_changes"] += upgrade_result.get("code_changes", 0)
                    successful_upgrades += 1
            
            # 计算平均效果
            if upgrade_results["upgrades_applied"]:
                upgrade_results["effectiveness"] /= len(upgrade_results["upgrades_applied"])
                upgrade_results["performance_improvement"] /= len(upgrade_results["upgrades_applied"])
            
            # 重新加载模块以应用更改
            if upgrade_results["files_modified"]:
                self._reload_modified_modules(upgrade_results["files_modified"])
            
            self.logger.info(f"自我升级完成: 应用了{len(upgrade_results['upgrades_applied'])}项升级, 修改了{upgrade_results['code_changes']}处代码")
            
        except Exception as e:
            self.logger.error(f"自我升级执行失败: {str(e)}")
            upgrade_results["error"] = str(e)
        
        return upgrade_results

    def _analyze_for_self_upgrades(self, learning_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析学习数据以识别自我升级机会 | Analyze learning data to identify self-upgrade opportunities"""
        opportunities = []
        
        # 分析性能瓶颈
        performance_metrics = learning_data.get("performance_metrics", {})
        if performance_metrics.get("response_time", 0) > 0.3:
            opportunities.append({
                "type": "performance_optimization",
                "target": "core_algorithms",
                "description": "优化核心算法以减少响应时间",
                "priority": "high",
                "expected_improvement": 0.2
            })
        
        # 分析错误模式
        error_patterns = learning_data.get("error_patterns", {})
        common_errors = error_patterns.get("common_errors", [])
        if "timeout" in common_errors:
            opportunities.append({
                "type": "error_handling",
                "target": "timeout_management",
                "description": "改进超时处理机制",
                "priority": "medium",
                "expected_improvement": 0.15
            })
        
        # 分析模型性能
        model_performance = learning_data.get("model_performance", {})
        for model_name, perf in model_performance.items():
            if perf.get("accuracy", 0) < 0.8:
                opportunities.append({
                    "type": "model_enhancement",
                    "target": f"{model_name}_model",
                    "description": f"提高{model_name}模型的准确性",
                    "priority": "high",
                    "expected_improvement": 0.1
                })
        
        # 基于学习效果建议架构改进
        if self.learning_state["learning_efficiency"] < 0.6:
            opportunities.append({
                "type": "architectural_improvement",
                "target": "learning_architecture",
                "description": "优化学习架构以提高效率",
                "priority": "medium",
                "expected_improvement": 0.25
            })
        
        return opportunities

    def _get_programming_model(self):
        """获取编程模型实例 | Get programming model instance"""
        # 这里应该通过模型注册表或依赖注入获取编程模型
        # 暂时返回None，实际实现中需要正确获取
        try:
            # 假设有一个全局的模型访问机制
            from manager_model.model_registry import ModelRegistry
            registry = ModelRegistry()
            return registry.get_model('K_programming')
        except Exception as e:
            self.logger.error(f"获取编程模型失败: {str(e)}")
            return None

    def _apply_self_upgrade(self, programming_model, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """应用自我升级 | Apply self-upgrade"""
        result = {
            "status": "pending",
            "upgrade_type": opportunity["type"],
            "target": opportunity["target"],
            "description": opportunity["description"],
            "effectiveness": 0.0,
            "performance_improvement": 0.0
        }
        
        try:
            # 根据升级类型执行不同的升级操作
            if opportunity["type"] == "performance_optimization":
                upgrade_result = self._upgrade_performance(programming_model, opportunity)
            elif opportunity["type"] == "error_handling":
                upgrade_result = self._upgrade_error_handling(programming_model, opportunity)
            elif opportunity["type"] == "model_enhancement":
                upgrade_result = self._upgrade_model(programming_model, opportunity)
            elif opportunity["type"] == "architectural_improvement":
                upgrade_result = self._upgrade_architecture(programming_model, opportunity)
            else:
                upgrade_result = {"status": "skipped", "reason": "未知升级类型"}
            
            result.update(upgrade_result)
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            self.logger.error(f"应用升级失败: {str(e)}")
        
        return result

    def _upgrade_performance(self, programming_model, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """升级性能 | Upgrade performance"""
        # 使用编程模型优化自身代码
        file_path = __file__  # 当前文件路径
        optimization_targets = ["algorithm_efficiency", "memory_usage", "response_time"]
        
        result = programming_model.optimize_performance(
            file_path, 
            optimization_targets,
            {"target_improvement": opportunity["expected_improvement"]}
        )
        
        return {
            "status": result.get("status", "error"),
            "effectiveness": result.get("performance_improvement", 0.0),
            "performance_improvement": result.get("performance_improvement", 0.0),
            "details": result
        }

    def _upgrade_error_handling(self, programming_model, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """升级错误处理 | Upgrade error handling"""
        # 分析当前错误处理代码并改进
        file_path = __file__
        error_description = f"需要改进{opportunity['target']}的错误处理"
        
        result = programming_model.debug_and_fix(
            file_path,
            error_description,
            "常见的超时和资源管理错误"
        )
        
        return {
            "status": result.get("status", "error"),
            "effectiveness": 0.8 if result.get("status") == "success" else 0.1,
            "performance_improvement": 0.1,  # 错误减少带来的间接性能提升
            "details": result
        }

    def _upgrade_model(self, programming_model, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """升级特定模型 | Upgrade specific model"""
        target_model = opportunity["target"]
        model_file_path = f"sub_models/{target_model}/model.py"
        
        # 使用编程模型改进目标模型
        improvement_suggestions = [
            f"提高{target_model}模型的准确性",
            "优化模型算法",
            "改进错误处理"
        ]
        
        result = programming_model.refactor_code(
            model_file_path,
            improvement_suggestions,
            {"target_accuracy": 0.85}
        )
        
        return {
            "status": result.get("status", "error"),
            "effectiveness": result.get("performance_improvement", 0.0),
            "performance_improvement": result.get("performance_improvement", 0.0),
            "details": result
        }

    def _upgrade_architecture(self, programming_model, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """升级架构 | Upgrade architecture"""
        # 分析系统架构并提出改进
        system_path = "."
        improvement_focus = ["learning_efficiency", "model_collaboration", "data_flow"]
        
        result = programming_model.improve_system_code(
            system_path,
            improvement_focus
        )
        
        return {
            "status": result.get("status", "error"),
            "effectiveness": result.get("overall_improvement", 0.0),
            "performance_improvement": result.get("overall_improvement", 0.0),
            "details": result
        }
    
    def _evaluate_learning_effectiveness(self, learning_results: Dict[str, Any]) -> float:
        """评估学习效果 | Evaluate learning effectiveness"""
        effectiveness = learning_results.get("effectiveness", 0.0)
        
        # 基于改进指标调整效果评估
        improvements = learning_results.get("improvements", {})
        if improvements:
            # 计算改进指标的平均值
            improvement_score = sum(improvements.values()) / len(improvements)
            effectiveness = (effectiveness + improvement_score) / 2
        
        return max(0.1, min(1.0, effectiveness))  # 确保在0.1-1.0范围内
    
    def _update_learning_state(self, learning_results: Dict[str, Any], effectiveness: float):
        """更新学习状态 | Update learning state"""
        self.learning_state["total_learning_cycles"] += 1
        self.learning_state["last_learning_session"] = datetime.now().isoformat()
        
        # 更新学习效率（指数移动平均）
        old_efficiency = self.learning_state["learning_efficiency"]
        self.learning_state["learning_efficiency"] = (
            0.9 * old_efficiency + 0.1 * effectiveness
        )
        
        # 更新适应水平
        self.learning_state["adaptation_level"] = min(1.0, 
            self.learning_state["adaptation_level"] + effectiveness * 0.05
        )
        
        # 更新知识增长率
        improvements = learning_results.get("improvements", {})
        if "knowledge_coverage" in improvements or "skill_transfer" in improvements:
            knowledge_gain = (improvements.get("knowledge_coverage", 0) + 
                             improvements.get("skill_transfer", 0)) / 2
            self.learning_state["knowledge_growth_rate"] = (
                0.8 * self.learning_state["knowledge_growth_rate"] + 0.2 * knowledge_gain
            )
    
    def _record_learning_history(self, learning_data: Dict[str, Any], strategy: str, 
                               learning_results: Dict[str, Any], effectiveness: float):
        """记录学习历史 | Record learning history"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "cycle_number": self.learning_state["total_learning_cycles"],
            "strategy": strategy,
            "learning_data": {k: v for k, v in learning_data.items() if k != "performance_metrics"},
            "results": learning_results,
            "effectiveness": effectiveness,
            "learning_state": self.learning_state.copy()
        }
        
        self.learning_history.append(history_entry)
    
    def _save_learning_state(self):
        """保存学习状态 | Save learning state"""
        try:
            state_file = os.path.join(self.data_dir, "learning_state.pkl")
            with open(state_file, 'wb') as f:
                pickle.dump({
                    "learning_state": self.learning_state,
                    "learning_strategies": self.learning_strategies,
                    "learning_objectives": self.learning_objectives
                }, f)
            
            # 保存学习历史（最近100条）
            history_file = os.path.join(self.data_dir, "learning_history.json")
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.learning_history)[-100:], f, ensure_ascii=False, indent=2)
                
            self.logger.info("学习状态已保存 | Learning state saved")
            
        except Exception as e:
            self.logger.error(f"保存学习状态失败: {str(e)}")
    
    def _load_learning_state(self):
        """加载学习状态 | Load learning state"""
        try:
            state_file = os.path.join(self.data_dir, "learning_state.pkl")
            if os.path.exists(state_file):
                with open(state_file, 'rb') as f:
                    saved_state = pickle.load(f)
                    self.learning_state.update(saved_state.get("learning_state", {}))
                    self.learning_strategies.update(saved_state.get("learning_strategies", {}))
                    self.learning_objectives.update(saved_state.get("learning_objectives", {}))
                
                self.logger.info("学习状态已加载 | Learning state loaded")
                
        except Exception as e:
            self.logger.error(f"加载学习状态失败: {str(e)}")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态 | Get learning status"""
        return {
            "learning_state": self.learning_state,
            "recent_effectiveness": self._get_recent_effectiveness(),
            "active_strategies": [s for s, cfg in self.learning_strategies.items() if cfg["enabled"]],
            "learning_objectives": self.learning_objectives
        }
    
    def _get_recent_effectiveness(self) -> float:
        """获取最近学习效果 | Get recent learning effectiveness"""
        if not self.learning_history:
            return 0.5  # 默认效果
        
        # 计算最近10次学习的平均效果
        recent_entries = list(self.learning_history)[-10:]
        if not recent_entries:
            return 0.5
        
        effectiveness_sum = sum(entry["effectiveness"] for entry in recent_entries)
        return effectiveness_sum / len(recent_entries)
    
    def optimize_learning_strategy(self, feedback: Dict[str, Any]):
        """优化学习策略 | Optimize learning strategy"""
        # 基于反馈调整学习策略权重和效果
        if "strategy_performance" in feedback:
            for strategy, performance in feedback["strategy_performance"].items():
                if strategy in self.learning_strategies:
                    # 调整策略效果评估
                    old_effectiveness = self.learning_strategies[strategy]["effectiveness"]
                    self.learning_strategies[strategy]["effectiveness"] = (
                        0.8 * old_effectiveness + 0.2 * performance
                    )
        
        if "learning_preferences" in feedback:
            # 调整学习目标权重
            for objective, preference in feedback["learning_preferences"].items():
                if objective in self.learning_objectives:
                    self.learning_objectives[objective] = preference
        
        self.logger.info("学习策略已优化 | Learning strategy optimized")
    
    def generate_learning_report(self, days: int = 7) -> Dict[str, Any]:
        """生成学习报告 | Generate learning report"""
        # 获取指定天数内的学习记录
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [
            entry for entry in self.learning_history
            if datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00')) > cutoff_date
        ]
        
        if not recent_history:
            return {"error": f"No learning data for the past {days} days"}
        
        # 计算统计信息
        total_cycles = len(recent_history)
        avg_effectiveness = sum(entry["effectiveness"] for entry in recent_history) / total_cycles
        
        # 策略使用统计
        strategy_usage = defaultdict(int)
        for entry in recent_history:
            strategy_usage[entry["strategy"]] += 1
        
        # 改进指标统计
        improvement_stats = defaultdict(list)
        for entry in recent_history:
            for metric, value in entry["results"].get("improvements", {}).items():
                improvement_stats[metric].append(value)
        
        avg_improvements = {
            metric: sum(values) / len(values) 
            for metric, values in improvement_stats.items()
        }
        
        return {
            "report_period": f"Last {days} days",
            "total_learning_cycles": total_cycles,
            "average_effectiveness": round(avg_effectiveness, 3),
            "strategy_usage": dict(strategy_usage),
            "average_improvements": avg_improvements,
            "learning_efficiency_trend": self.learning_state["learning_efficiency"],
            "knowledge_growth_rate": self.learning_state["knowledge_growth_rate"],
            "recommendations": self._generate_recommendations(recent_history)
        }
    
    def _generate_recommendations(self, recent_history: List[Dict]) -> List[str]:
        """生成学习建议 | Generate learning recommendations"""
        recommendations = []
        
        # 分析最近的学习效果
        effectiveness_scores = [entry["effectiveness"] for entry in recent_history]
        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
        
        if avg_effectiveness < 0.6:
            recommendations.append("Consider adjusting learning strategies or increasing training data diversity")
        
        # 检查策略使用分布
        strategy_counts = defaultdict(int)
        for entry in recent_history:
            strategy_counts[entry["strategy"]] += 1
        
        if len(strategy_counts) < 2:
            recommendations.append("Try diversifying learning strategies for better adaptation")
        
        # 检查学习频率
        if len(recent_history) < 10:
            recommendations.append("Increase learning frequency for faster improvement")
        
        return recommendations
