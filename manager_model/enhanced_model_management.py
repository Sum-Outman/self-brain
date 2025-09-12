# -*- coding: utf-8 -*-
# 增强型模型管理系统 - 提供完整的模型管理界面和交互能力
# Enhanced Model Management System - Complete model management interface and interaction capabilities
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License")

import json
import logging
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from flask import Blueprint, render_template, request, jsonify, session, flash, redirect, url_for
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

# 导入必要的模块
from .model_registry import ModelRegistry, get_model_registry
from .language_resources import get_string
from .data_bus import DataBus
from .emotion_engine import AdvancedEmotionEngine

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedModelManagement")

# 创建蓝图
enhanced_model_bp = Blueprint('enhanced_model_management', __name__,
                            template_folder='templates',
                            static_folder='static')

# 全局实例
model_registry = get_model_registry()
data_bus = DataBus()
socketio = SocketIO(message_queue='redis://localhost:6379')

class EnhancedModelManager:
    """增强型模型管理器 - 提供高级模型管理功能"""
    
    def __init__(self):
        self.registry = model_registry
        self.model_performance = {}
        self.model_dependencies = self._load_model_dependencies()
        self.model_interaction_graph = self._build_interaction_graph()
        self.performance_history = {}
        self._initialize_performance_tracking()
        
    def _load_model_dependencies(self) -> Dict[str, List[str]]:
        """加载模型依赖关系"""
        return {
            "B_language": ["I_knowledge", "K_programming"],
            "C_audio": ["B_language", "I_knowledge"],
            "D_image": ["B_language", "I_knowledge"],
            "E_video": ["D_image", "C_audio", "B_language"],
            "F_spatial": ["D_image", "I_knowledge"],
            "G_sensor": ["I_knowledge", "K_programming"],
            "H_computer_control": ["K_programming", "I_knowledge"],
            "I_knowledge": ["B_language", "K_programming"],
            "J_motion": ["F_spatial", "G_sensor", "I_knowledge"],
            "K_programming": ["I_knowledge", "B_language"]
        }
    
    def _build_interaction_graph(self) -> Dict[str, Any]:
        """构建模型交互图"""
        graph = {
            "nodes": [],
            "edges": [],
            "interaction_strength": {}
        }
        
        for model in self.registry.get_all_models():
            graph["nodes"].append({
                "id": model,
                "name": self.registry.get_model_info(model).get("name", model),
                "type": self.registry.get_model_info(model).get("type", "local"),
                "status": self.registry.get_health_status().get(model, {}).get("status", "unknown")
            })
        
        for source, targets in self.model_dependencies.items():
            for target in targets:
                if source in self.registry.get_all_models() and target in self.registry.get_all_models():
                    graph["edges"].append({
                        "source": source,
                        "target": target,
                        "strength": 0.8  # 默认交互强度
                    })
        
        return graph
    
    def _initialize_performance_tracking(self):
        """初始化性能跟踪"""
        for model in self.registry.get_all_models():
            self.performance_history[model] = {
                "response_times": [],
                "success_rates": [],
                "utilization": [],
                "timestamps": []
            }
    
    def update_performance_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """更新性能指标"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = {
                "response_times": [],
                "success_rates": [],
                "utilization": [],
                "timestamps": []
            }
        
        timestamp = datetime.now().isoformat()
        self.performance_history[model_name]["timestamps"].append(timestamp)
        
        if "response_time" in metrics:
            self.performance_history[model_name]["response_times"].append(metrics["response_time"])
        
        if "success_rate" in metrics:
            self.performance_history[model_name]["success_rates"].append(metrics["success_rate"])
        
        if "utilization" in metrics:
            self.performance_history[model_name]["utilization"].append(metrics["utilization"])
        
        # 保持历史数据大小
        max_history = 1000
        for key in ["response_times", "success_rates", "utilization", "timestamps"]:
            if len(self.performance_history[model_name][key]) > max_history:
                self.performance_history[model_name][key] = self.performance_history[model_name][key][-max_history:]
    
    def get_model_performance_report(self, model_name: str) -> Dict[str, Any]:
        """获取模型性能报告"""
        if model_name not in self.performance_history:
            return {"error": "Model not found in performance history"}
        
        data = self.performance_history[model_name]
        
        if not data["response_times"]:
            return {"status": "no_data", "message": "No performance data available"}
        
        # 计算统计指标
        response_times = np.array(data["response_times"])
        success_rates = np.array(data["success_rates"]) if data["success_rates"] else np.array([1.0])
        utilization = np.array(data["utilization"]) if data["utilization"] else np.array([0.0])
        
        report = {
            "response_time": {
                "avg": float(np.mean(response_times)),
                "min": float(np.min(response_times)),
                "max": float(np.max(response_times)),
                "p95": float(np.percentile(response_times, 95)),
                "std": float(np.std(response_times))
            },
            "success_rate": {
                "avg": float(np.mean(success_rates)) if success_rates.size > 0 else 1.0,
                "min": float(np.min(success_rates)) if success_rates.size > 0 else 1.0,
                "max": float(np.max(success_rates)) if success_rates.size > 0 else 1.0
            },
            "utilization": {
                "avg": float(np.mean(utilization)) if utilization.size > 0 else 0.0,
                "max": float(np.max(utilization)) if utilization.size > 0 else 0.0
            },
            "data_points": len(data["response_times"]),
            "last_updated": data["timestamps"][-1] if data["timestamps"] else None
        }
        
        return report
    
    def get_performance_trends(self, model_name: str, metric: str = "response_time") -> Dict[str, Any]:
        """获取性能趋势数据"""
        if model_name not in self.performance_history:
            return {"error": "Model not found"}
        
        data = self.performance_history[model_name]
        
        if not data["timestamps"]:
            return {"error": "No data available"}
        
        # 准备趋势数据
        if metric == "response_time" and data["response_times"]:
            values = data["response_times"]
        elif metric == "success_rate" and data["success_rates"]:
            values = data["success_rates"]
        elif metric == "utilization" and data["utilization"]:
            values = data["utilization"]
        else:
            return {"error": f"Metric {metric} not available"}
        
        # 创建趋势图表数据
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data["timestamps"],
            y=values,
            mode='lines+markers',
            name=metric,
            line=dict(color='blue', width=2)
        ))
        
        # 添加移动平均线
        if len(values) > 10:
            window_size = min(20, len(values) // 5)
            moving_avg = pd.Series(values).rolling(window=window_size).mean().tolist()
            
            fig.add_trace(go.Scatter(
                x=data["timestamps"][window_size-1:],
                y=moving_avg[window_size-1:],
                mode='lines',
                name=f'{window_size}-point Moving Average',
                line=dict(color='red', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title=f"{model_name} - {metric.replace('_', ' ').title()} Trend",
            xaxis_title="Time",
            yaxis_title=metric.replace('_', ' ').title(),
            showlegend=True
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    
    def optimize_model_configuration(self, model_name: str) -> Dict[str, Any]:
        """优化模型配置"""
        model_info = self.registry.get_model_info(model_name)
        if not model_info:
            return {"status": "error", "message": f"Model {model_name} not found"}
        
        performance_report = self.get_model_performance_report(model_name)
        
        # 基于性能数据生成优化建议
        recommendations = []
        
        if "response_time" in performance_report:
            rt = performance_report["response_time"]
            if rt["avg"] > 2.0:  # 响应时间超过2秒
                recommendations.append({
                    "type": "performance",
                    "priority": "high",
                    "message": "响应时间较高，建议优化模型或增加资源分配",
                    "suggestion": "考虑模型量化、缓存策略或硬件加速"
                })
        
        if "success_rate" in performance_report:
            sr = performance_report["success_rate"]
            if sr["avg"] < 0.8:  # 成功率低于80%
                recommendations.append({
                    "type": "reliability",
                    "priority": "high",
                    "message": "成功率较低，建议检查模型健康状态",
                    "suggestion": "运行诊断测试或重新训练模型"
                })
        
        if not recommendations:
            recommendations.append({
                "type": "maintenance",
                "priority": "low",
                "message": "模型性能良好，无需立即优化",
                "suggestion": "继续监控性能指标"
            })
        
        return {
            "status": "success",
            "model": model_name,
            "performance_report": performance_report,
            "recommendations": recommendations,
            "optimization_timestamp": datetime.now().isoformat()
        }
    
    def get_model_interaction_analysis(self) -> Dict[str, Any]:
        """获取模型交互分析"""
        interaction_matrix = np.zeros((len(self.registry.get_all_models()), 
                                     len(self.registry.get_all_models())))
        
        models = sorted(self.registry.get_all_models())
        model_index = {model: idx for idx, model in enumerate(models)}
        
        # 构建交互矩阵
        for source, targets in self.model_dependencies.items():
            if source in model_index:
                for target in targets:
                    if target in model_index:
                        interaction_matrix[model_index[source]][model_index[target]] = 1
        
        # 计算中心性指标
        centrality = {}
        for model in models:
            # 简单的度中心性计算
            in_degree = np.sum(interaction_matrix[:, model_index[model]])
            out_degree = np.sum(interaction_matrix[model_index[model], :])
            centrality[model] = {
                "in_degree": int(in_degree),
                "out_degree": int(out_degree),
                "total_degree": int(in_degree + out_degree)
            }
        
        # 识别关键模型
        critical_models = sorted(centrality.items(), 
                               key=lambda x: x[1]["total_degree"], 
                               reverse=True)[:3]
        
        return {
            "interaction_matrix": interaction_matrix.tolist(),
            "model_index": model_index,
            "centrality": centrality,
            "critical_models": [model[0] for model in critical_models],
            "analysis_timestamp": datetime.now().isoformat()
        }

# 创建增强型模型管理器实例
enhanced_manager = EnhancedModelManager()

# 路由定义
@enhanced_model_bp.route('/enhanced_model_management', methods=['GET'])
def enhanced_model_management():
    """增强型模型管理主界面"""
    lang = request.args.get('lang', 'zh')
    
    # 获取所有模型信息
    models = model_registry.get_all_models()
    model_info = {model: model_registry.get_model_info(model) for model in models}
    health_status = model_registry.get_health_status()
    
    # 获取性能报告
    performance_reports = {}
    for model in models:
        performance_reports[model] = enhanced_manager.get_model_performance_report(model)
    
    # 获取交互分析
    interaction_analysis = enhanced_manager.get_model_interaction_analysis()
    
    return render_template('enhanced_model_management.html',
                         models=models,
                         model_info=model_info,
                         health_status=health_status,
                         performance_reports=performance_reports,
                         interaction_analysis=interaction_analysis,
                         lang=lang,
                         get_string=get_string)

@enhanced_model_bp.route('/api/model_performance/<model_name>', methods=['GET'])
def get_model_performance_api(model_name):
    """获取模型性能数据API"""
    lang = request.args.get('lang', 'zh')
    
    try:
        report = enhanced_manager.get_model_performance_report(model_name)
        return jsonify({
            "status": "success",
            "data": report
        })
    except Exception as e:
        logger.error(f"获取模型性能数据失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": get_string("model_performance_failed", lang).format(error=str(e))
        }), 500

@enhanced_model_bp.route('/api/performance_trends/<model_name>', methods=['GET'])
def get_performance_trends_api(model_name):
    """获取性能趋势数据API"""
    lang = request.args.get('lang', 'zh')
    metric = request.args.get('metric', 'response_time')
    
    try:
        trends = enhanced_manager.get_performance_trends(model_name, metric)
        return jsonify({
            "status": "success",
            "data": trends
        })
    except Exception as e:
        logger.error(f"获取性能趋势数据失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": get_string("performance_trends_failed", lang).format(error=str(e))
        }), 500

@enhanced_model_bp.route('/api/optimize_model/<model_name>', methods=['POST'])
def optimize_model_api(model_name):
    """优化模型配置API"""
    lang = request.json.get('lang', 'zh') if request.json else 'zh'
    
    try:
        result = enhanced_manager.optimize_model_configuration(model_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"模型优化失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": get_string("model_optimization_failed", lang).format(error=str(e))
        }), 500

@enhanced_model_bp.route('/api/model_interaction_analysis', methods=['GET'])
def get_model_interaction_analysis_api():
    """获取模型交互分析API"""
    lang = request.args.get('lang', 'zh')
    
    try:
        analysis = enhanced_manager.get_model_interaction_analysis()
        return jsonify({
            "status": "success",
            "data": analysis
        })
    except Exception as e:
        logger.error(f"获取模型交互分析失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": get_string("interaction_analysis_failed", lang).format(error=str(e))
        }), 500

@enhanced_model_bp.route('/api/update_model_config/<model_name>', methods=['POST'])
def update_model_config_api(model_name):
    """更新模型配置API"""
    lang = request.json.get('lang', 'zh') if request.json else 'zh'
    config_data = request.json.get('config', {})
    
    try:
        success = model_registry.update_model_config(model_name, config_data)
        if success:
            return jsonify({
                "status": "success",
                "message": get_string("model_config_updated", lang)
            })
        else:
            return jsonify({
                "status": "error",
                "message": get_string("model_config_update_failed", lang)
            }), 400
    except Exception as e:
        logger.error(f"更新模型配置失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": get_string("model_config_update_failed", lang).format(error=str(e))
        }), 500

@enhanced_model_bp.route('/api/test_model_connection/<model_name>', methods=['POST'])
def test_model_connection_api(model_name):
    """测试模型连接API"""
    lang = request.json.get('lang', 'zh') if request.json else 'zh'
    
    try:
        result = model_registry.test_connection(model_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"测试模型连接失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": get_string("connection_test_failed", lang).format(error=str(e))
        }), 500

@enhanced_model_bp.route('/api/switch_model_type/<model_name>', methods=['POST'])
def switch_model_type_api(model_name):
    """切换模型类型API"""
    lang = request.json.get('lang', 'zh') if request.json else 'zh'
    model_type = request.json.get('type', 'local')
    api_url = request.json.get('api_url', '')
    api_key = request.json.get('api_key', '')
    
    try:
        if model_type == 'external':
            success = model_registry.switch_to_external(model_name, api_url, api_key)
        else:
            success = model_registry.switch_to_local(model_name)
        
        if success:
            return jsonify({
                "status": "success",
                "message": get_string("model_type_switched", lang)
            })
        else:
            return jsonify({
                "status": "error",
                "message": get_string("model_type_switch_failed", lang)
            }), 400
    except Exception as e:
        logger.error(f"切换模型类型失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": get_string("model_type_switch_failed", lang).format(error=str(e))
        }), 500

@enhanced_model_bp.route('/api/enable_model/<model_name>', methods=['POST'])
def enable_model_api(model_name):
    """启用模型API"""
    lang = request.json.get('lang', 'zh') if request.json else 'zh'
    
    try:
        success = model_registry.enable_model(model_name)
        if success:
            return jsonify({
                "status": "success",
                "message": get_string("model_enabled", lang)
            })
        else:
            return jsonify({
                "status": "error",
                "message": get_string("model_enable_failed", lang)
            }), 400
    except Exception as e:
        logger.error(f"启用模型失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": get_string("model_enable_failed", lang).format(error=str(e))
        }), 500

@enhanced_model_bp.route('/api/disable_model/<model_name>', methods=['POST'])
def disable_model_api(model_name):
    """禁用模型API"""
    lang = request.json.get('lang', 'zh') if request.json else 'zh'
    
    try:
        success = model_registry.disable_model(model_name)
        if success:
            return jsonify({
                "status": "success",
                "message": get_string("model_disabled", lang)
            })
        else:
            return jsonify({
                "status": "error",
                "message": get_string("model_disable_failed", lang)
            }), 400
    except Exception as e:
        logger.error(f"禁用模型失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": get_string("model_disable_failed", lang).format(error=str(e))
        }), 500

# WebSocket 事件处理
@socketio.on('request_model_performance')
def handle_model_performance_request(data):
    """处理模型性能数据请求"""
    model_name = data.get('model_name')
    if model_name:
        report = enhanced_manager.get_model_performance_report(model_name)
        emit('model_performance_update', {
            'model_name': model_name,
            'data': report
        })

@socketio.on('request_performance_trends')
def handle_performance_trends_request(data):
    """处理性能趋势数据请求"""
    model_name = data.get('model_name')
    metric = data.get('metric', 'response_time')
    if model_name:
        trends = enhanced_manager.get_performance_trends(model_name, metric)
        emit('performance_trends_update', {
            'model_name': model_name,
            'metric': metric,
            'data': trends
        })

@socketio.on('request_model_interaction_analysis')
def handle_interaction_analysis_request():
    """处理交互分析请求"""
    analysis = enhanced_manager.get_model_interaction_analysis()
    emit('model_interaction_analysis_update', {
        'data': analysis
    })

# 后台性能监控线程
def performance_monitoring_loop():
    """性能监控循环"""
    while True:
        try:
            # 获取所有模型的健康状态和性能数据
            health_status = model_registry.get_health_status()
            
            for model_name, status in health_status.items():
                if status.get('status') == 'success':
                    # 模拟性能数据收集
                    metrics = {
                        "response_time": status.get('response_time', 0.5) + np.random.normal(0, 0.1),
                        "success_rate": 0.95 + np.random.normal(0, 0.05),
                        "utilization": np.random.uniform(0.1, 0.9)
                    }
                    
                    # 更新性能指标
                    enhanced_manager.update_performance_metrics(model_name, metrics)
            
            # 每分钟更新一次
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"性能监控循环错误: {str(e)}")
            time.sleep(30)

# 启动性能监控线程
monitoring_thread = threading.Thread(target=performance_monitoring_loop, daemon=True)
monitoring_thread.start()

# 注册蓝图
def register_enhanced_model_management(app):
    """注册增强型模型管理系统"""
    app.register_blueprint(enhanced_model_bp, url_prefix='/enhanced_model_management')
    logger.info("增强型模型管理系统已注册")

if __name__ == '__main__':
    # 测试代码
    manager = EnhancedModelManager()
    
    # 模拟一些性能数据
    for model in manager.registry.get_all_models():
        for i in range(10):
            metrics = {
                "response_time": np.random.uniform(0.1, 2.0),
                "success_rate": np.random.uniform(0.7, 1.0),
                "utilization": np.random.uniform(0.1, 0.8)
            }
            manager.update_performance_metrics(model, metrics)
            time.sleep(0.1)
    
    # 生成性能报告
    for model in manager.registry.get_all_models():
        report = manager.get_model_performance_report(model)
        print(f"\n{model} 性能报告:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # 生成交互分析
    analysis = manager.get_model_interaction_analysis()
    print("\n模型交互分析:")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
