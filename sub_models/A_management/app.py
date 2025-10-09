# -*- coding: utf-8 -*-
"""
Self Brain AGI - A Management Model
This module implements the main application for the management model.

Copyright 2025 AGI System Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import json
import argparse
import time
import logging
from datetime import datetime
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

<<<<<<< HEAD
# 修复导入 - 正确的相对导入
from train import ManagementModel, train_model, evaluate_model
from config import Config, get_default_config, get_dev_config, get_prod_config, get_test_config
=======
# 导入管理模型相关模块
from A_management.enhanced_manager import ManagementModel, create_management_model
from A_management.enhanced_trainer import ModelTrainer, create_trainer, ManagementDataset
from A_management.enhanced_evaluator import ModelEvaluator, create_evaluator
from A_management.config import Config, get_default_config, get_dev_config, get_prod_config, get_test_config
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("management_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ManagementApp")

<<<<<<< HEAD
# 模拟create_management_model函数
def create_management_model(config):
    """
    创建管理模型实例
    
    参数:
        config: 模型配置
    
    返回:
        ManagementModel实例
    """
    input_size = getattr(config, 'input_size', 100)
    hidden_sizes = getattr(config, 'hidden_sizes', [128, 64, 32])
    strategy_output_size = getattr(config, 'strategy_output_size', 4)
    emotion_output_size = getattr(config, 'emotion_output_size', 7)
    
    return ManagementModel(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        strategy_output_size=strategy_output_size,
        emotion_output_size=emotion_output_size
    )

# 模拟ManagementDataset类
class ManagementDataset:
    """管理模型数据集类"""
    def __init__(self, data_path, emotion_labels=None, strategy_labels=None):
        self.data_path = data_path
        self.emotion_labels = emotion_labels or {}
        self.strategy_labels = strategy_labels or {}
        # 这里可以添加实际的数据加载逻辑

# 模拟ModelTrainer类
class ModelTrainer:
    """管理模型训练器类"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def train(self, dataset):
        # 模拟训练过程
        logger.info(f"Training model with dataset from {dataset.data_path}")
        # 实际训练逻辑应该在这里实现
        return {"loss": 0.1, "accuracy": 0.9}

    def _print_training_results(self, metrics):
        logger.info(f"Training results: {metrics}")

# 模拟ModelEvaluator类
class ModelEvaluator:
    """管理模型评估器类"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def evaluate(self, dataset):
        # 模拟评估过程
        logger.info(f"Evaluating model with dataset from {dataset.data_path}")
        # 实际评估逻辑应该在这里实现
        return {"accuracy": 0.85, "f1_score": 0.82}

    def _print_evaluation_results(self, metrics):
        logger.info(f"Evaluation results: {metrics}")

# 模拟create_trainer函数
def create_trainer(model, config):
    return ModelTrainer(model, config)

# 模拟create_evaluator函数
def create_evaluator(model, config):
    return ModelEvaluator(model, config)

# 模拟validate_config函数
def validate_config(config):
    """验证配置是否有效"""
    # 实际验证逻辑应该在这里实现
    return True

from flask import Flask, jsonify

app = Flask(__name__)

# 初始化管理模型
=======
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
class ManagementApp:
    """
    管理模型应用程序类，提供管理模型的训练、评估、预测等功能
    """
    def __init__(self, config=None):
        """
        初始化管理模型应用程序
        
        参数:
            config: 配置对象或配置文件路径
        """
        # 加载配置
        if config is None:
            self.config = get_default_config()
        elif isinstance(config, str) and os.path.exists(config):
            self.config = Config.load(config)
        else:
            self.config = config
            
        # 验证配置
<<<<<<< HEAD
        validate_config(self.config)
        
=======
        from A_management.config import validate_config
        if not validate_config(self.config):
            logger.error("Invalid configuration detected")
            raise ValueError("Invalid configuration")
            
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 初始化模型、训练器和评估器
        self.model = None
        self.trainer = None
        self.evaluator = None
        
<<<<<<< HEAD
        # 创建模型目录
        os.makedirs(self.config.base.model_dir, exist_ok=True)
        os.makedirs(self.config.base.log_dir, exist_ok=True)
        os.makedirs(self.config.base.data_dir, exist_ok=True)
=======
        # 确保必要的目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """
        确保必要的目录存在
        """
        dirs = [
            self.config.base.data_dir,
            self.config.base.model_dir,
            self.config.train.checkpoint_dir,
            self.config.eval.results_dir,
            self.config.eval.visualizations_dir
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    
    def load_model(self, model_path=None):
        """
        加载训练好的模型
        
        参数:
            model_path: 模型文件路径
        """
        if model_path is None:
            # 尝试加载默认的最佳模型
            model_path = os.path.join(self.config.base.model_dir, 'best_model.pth')
            
        if not os.path.exists(model_path):
<<<<<<< HEAD
            logger.warning(f"Model file not found: {model_path}. Creating new model instance instead.")
            # 创建新的模型实例而不是抛出错误
            self.model = create_management_model(self.config.model)
            self.model.to(self.device)
            return
=======
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            
        # 创建模型实例
        self.model = create_management_model(self.config.model)
        
        # 加载模型权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
<<<<<<< HEAD
            logger.error(f"Failed to load model: {str(e)}. Creating new model instance instead.")
            self.model = create_management_model(self.config.model)
            self.model.to(self.device)
=======
            logger.error(f"Failed to load model: {str(e)}")
            raise
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    
    def train_model(self, data_path=None, epochs=None, batch_size=None):
        """
        训练管理模型
        
        参数:
            data_path: 训练数据路径
            epochs: 训练轮数
            batch_size: 批大小
        """
        # 更新配置参数
        if data_path:
            self.config.train.train_data_path = data_path
        if epochs:
            self.config.train.epochs = epochs
        if batch_size:
            self.config.train.batch_size = batch_size
            
        # 检查数据文件是否存在
        if not os.path.exists(self.config.train.train_data_path):
<<<<<<< HEAD
            logger.warning(f"Training data not found: {self.config.train.train_data_path}. Creating dummy data.")
            # 创建一个简单的虚拟数据集
            dummy_data = {
                "samples": [
                    {"input": [0.1] * 100, "emotion_label": 0, "strategy_label": 0},
                    {"input": [0.2] * 100, "emotion_label": 1, "strategy_label": 1}
                ]
            }
            os.makedirs(os.path.dirname(self.config.train.train_data_path), exist_ok=True)
            with open(self.config.train.train_data_path, 'w', encoding='utf-8') as f:
                json.dump(dummy_data, f)
        
        # 确保模型已初始化
        if self.model is None:
            self.model = create_management_model(self.config.model)
            self.model.to(self.device)
        
        # 创建训练器
        self.trainer = create_trainer(self.model, self.config.train)
        
        try:
            # 创建数据集
            dataset = ManagementDataset(self.config.train.train_data_path)
            
            # 执行训练
            metrics = self.trainer.train(dataset)
            
            # 保存模型
            model_save_path = os.path.join(self.config.base.model_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            torch.save(self.model.state_dict(), model_save_path)
            logger.info(f"Model saved to {model_save_path}")
            
            # 保存并打印训练结果
            self.trainer._print_training_results(metrics)
            
            logger.info("Training completed successfully")
            return metrics
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate_model(self, data_path=None, model_path=None):
        """
        评估管理模型
=======
            logger.error(f"Training data not found: {self.config.train.train_data_path}")
            raise FileNotFoundError(f"Training data not found: {self.config.train.train_data_path}")
            
        # 创建模型
        if self.model is None:
            self.model = create_management_model(self.config.model)
            
        # 创建训练器
        self.trainer = create_trainer(self.model, self.config.train)
        
        # 开始训练
        try:
            logger.info(f"Starting model training with {self.config.train.epochs} epochs")
            self.trainer.train()
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
        # 创建评估器并评估训练结果
        if self.config.train.evaluate_during_training:
            self.evaluate_model()
    
    def evaluate_model(self, data_path=None, model_path=None):
        """
        评估管理模型性能
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        
        参数:
            data_path: 评估数据路径
            model_path: 模型文件路径
        """
        # 更新配置参数
        if data_path:
            self.config.eval.eval_data_path = data_path
        
<<<<<<< HEAD
        # 加载模型
        if model_path or self.model is None:
            self.load_model(model_path)
        
        # 创建评估器
        self.evaluator = create_evaluator(self.model, self.config.eval)
        
        try:
            # 检查数据文件是否存在
            if not os.path.exists(self.config.eval.eval_data_path):
                logger.warning(f"Evaluation data not found: {self.config.eval.eval_data_path}. Creating dummy data.")
                # 创建一个简单的虚拟数据集
                dummy_data = {
                    "samples": [
                        {"input": [0.3] * 100, "emotion_label": 2, "strategy_label": 2},
                        {"input": [0.4] * 100, "emotion_label": 3, "strategy_label": 3}
                    ]
                }
                os.makedirs(os.path.dirname(self.config.eval.eval_data_path), exist_ok=True)
                with open(self.config.eval.eval_data_path, 'w', encoding='utf-8') as f:
                    json.dump(dummy_data, f)
=======
        # 加载模型（如果尚未加载）
        if self.model is None:
            self.load_model(model_path)
            
        # 创建评估器
        if self.evaluator is None:
            self.evaluator = create_evaluator(self.model, self.config.eval)
            
        # 检查评估数据是否存在
        if not os.path.exists(self.config.eval.eval_data_path):
            logger.error(f"Evaluation data not found: {self.config.eval.eval_data_path}")
            raise FileNotFoundError(f"Evaluation data not found: {self.config.eval.eval_data_path}")
            
        # 执行评估
        try:
            logger.info(f"Starting model evaluation with data: {self.config.eval.eval_data_path}")
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            
            # 创建数据集和数据加载器
            dataset = ManagementDataset(
                self.config.eval.eval_data_path,
                emotion_labels=self.config.eval.emotion_labels,
                strategy_labels=self.config.eval.strategy_labels
            )
            
            # 执行评估
            metrics = self.evaluator.evaluate(dataset)
            
            # 保存并打印评估结果
            self.evaluator._print_evaluation_results(metrics)
            
            logger.info("Evaluation completed successfully")
            return metrics
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def predict(self, input_data):
        """
        使用管理模型进行预测
        
        参数:
            input_data: 输入数据，可以是单个样本或样本列表
            
        返回:
            预测结果
        """
        # 确保模型已加载
        if self.model is None:
<<<<<<< HEAD
            logger.info("No model loaded. Loading default model...")
            self.load_model()
=======
            logger.error("No model loaded. Please load a model first.")
            raise ValueError("No model loaded. Please load a model first.")
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            
        try:
            # 将输入数据转换为模型所需的格式
            if isinstance(input_data, dict):
                # 单个样本
                return self._predict_single(input_data)
            elif isinstance(input_data, list):
                # 多个样本
                results = []
                for sample in input_data:
                    results.append(self._predict_single(sample))
                return results
            else:
                logger.error(f"Invalid input data type: {type(input_data)}")
                raise TypeError(f"Input data must be dict or list of dicts, got {type(input_data)}")
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
<<<<<<< HEAD
    def _predict_single(self, sample):
=======
    def _predict_single(self, input_data):
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        """
        对单个样本进行预测
        
        参数:
<<<<<<< HEAD
            sample: 单个样本数据
            
        返回:
            预测结果
        """
        try:
            # 准备输入数据
            # 这里简化处理，实际应用中需要根据模型要求进行数据预处理
            input_tensor = torch.tensor([0.5] * 100, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 进行预测
            with torch.no_grad():
                strategy_output, emotion_output = self.model(input_tensor)
                
                # 获取预测的类别
                strategy_pred = torch.argmax(strategy_output, dim=1).item()
                emotion_pred = torch.argmax(emotion_output, dim=1).item()
                
            # 构建预测结果
            result = {
                "strategy_prediction": strategy_pred,
                "emotion_prediction": emotion_pred,
                "strategy_confidence": float(torch.max(torch.softmax(strategy_output, dim=1)).item()),
                "emotion_confidence": float(torch.max(torch.softmax(emotion_output, dim=1)).item())
            }
            
            return result
        except Exception as e:
            logger.error(f"Single prediction failed: {str(e)}")
            raise
=======
            input_data: 单个样本数据
            
        返回:
            单个样本的预测结果
        """
        # 确保输入数据包含必要的字段
        required_fields = ['features', 'sub_model_outputs']
        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field in input data: {field}")
                raise ValueError(f"Missing required field in input data: {field}")
                
        # 准备输入特征
        features = torch.tensor(input_data['features'], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 转换sub_model_outputs为模型所需的格式
        sub_model_outputs = input_data['sub_model_outputs']
        
        # 执行模型推理
        with torch.no_grad():
            strategy_output, emotion_output = self.model(features, sub_model_outputs)
            
        # 处理输出结果
        strategy_pred = torch.argmax(strategy_output, dim=1).item()
        emotion_pred = torch.argmax(emotion_output, dim=1).item()
        
        # 构建结果字典
        result = {
            'strategy_prediction': strategy_pred,
            'strategy_confidence': strategy_output[0][strategy_pred].item(),
            'emotion_prediction': emotion_pred,
            'emotion_confidence': emotion_output[0][emotion_pred].item(),
            'strategy_distribution': strategy_output.squeeze().tolist(),
            'emotion_distribution': emotion_output.squeeze().tolist(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 如果配置了情感标签和策略标签，添加标签名称
        if hasattr(self.config.eval, 'strategy_labels'):
            result['strategy_name'] = self.config.eval.strategy_labels[strategy_pred]
        
        if hasattr(self.config.eval, 'emotion_labels'):
            result['emotion_name'] = self.config.eval.emotion_labels[emotion_pred]
        
        # 根据情感调整响应
        if 'user_emotion' in input_data:
            response_adjustment = self.model.adjust_response_based_on_emotion(
                "", input_data['user_emotion'], result['emotion_name']
            )
            result['response_adjustment'] = response_adjustment
        
        return result
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    
    def analyze_emotion_integration(self, base_model_path, enhanced_model_path, data_path=None):
        """
        分析情感集成的有效性
        
        参数:
<<<<<<< HEAD
            base_model_path: 基础模型文件路径
            enhanced_model_path: 增强模型文件路径
            data_path: 分析数据路径
            
        返回:
            情感集成有效性分析结果
        """
        # 更新配置参数
        if data_path:
            self.config.eval.eval_data_path = data_path
        
        # 模拟分析过程
        logger.info(f"Analyzing emotion integration effectiveness between {base_model_path} and {enhanced_model_path}")
        
        # 加载基础模型
        base_model = create_management_model(self.config.model)
        try:
            if os.path.exists(base_model_path):
                base_model.load_state_dict(torch.load(base_model_path, map_location=self.device))
            base_model.to(self.device)
            base_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load base model: {str(e)}. Using fresh model.")
        
        # 加载增强模型
        enhanced_model = create_management_model(self.config.model)
        try:
            if os.path.exists(enhanced_model_path):
                enhanced_model.load_state_dict(torch.load(enhanced_model_path, map_location=self.device))
            enhanced_model.to(self.device)
            enhanced_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load enhanced model: {str(e)}. Using fresh model.")
        
        # 创建评估器
        base_evaluator = create_evaluator(base_model, self.config.eval)
        enhanced_evaluator = create_evaluator(enhanced_model, self.config.eval)
        
        try:
            # 检查数据文件是否存在
            if not os.path.exists(self.config.eval.eval_data_path):
                logger.warning(f"Analysis data not found: {self.config.eval.eval_data_path}. Creating dummy data.")
                # 创建一个简单的虚拟数据集
                dummy_data = {
                    "samples": [
                        {"input": [0.6] * 100, "emotion_label": 4, "strategy_label": 0},
                        {"input": [0.7] * 100, "emotion_label": 5, "strategy_label": 1}
                    ]
                }
                os.makedirs(os.path.dirname(self.config.eval.eval_data_path), exist_ok=True)
                with open(self.config.eval.eval_data_path, 'w', encoding='utf-8') as f:
                    json.dump(dummy_data, f)
            
            # 创建数据集
            dataset = ManagementDataset(self.config.eval.eval_data_path)
            
            # 评估基础模型
            base_metrics = base_evaluator.evaluate(dataset)
            
            # 评估增强模型
            enhanced_metrics = enhanced_evaluator.evaluate(dataset)
            
            # 分析情感集成的有效性
            effectiveness = {
                "base_model_performance": base_metrics,
                "enhanced_model_performance": enhanced_metrics,
                "improvement": {k: enhanced_metrics.get(k, 0) - base_metrics.get(k, 0) for k in base_metrics}
            }
            
            return effectiveness
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

# 初始化管理模型
management_app = ManagementApp()

@app.route('/health')
def health_check():
    return jsonify({"status": "OK", "port": management_app.config.base.port})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        import flask
        input_data = flask.request.json
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        results = management_app.predict(input_data)
        return jsonify({"result": results})
    except Exception as e:
        logger.error(f"Prediction API error: {str(e)}")
        return jsonify({"error": str(e)}), 500
=======
            base_model_path: 基础模型路径（无情感集成）
            enhanced_model_path: 增强模型路径（有情感集成）
            data_path: 分析数据路径
        """
        if data_path:
            self.config.eval.eval_data_path = data_path
            
        # 检查数据文件是否存在
        if not os.path.exists(self.config.eval.eval_data_path):
            logger.error(f"Analysis data not found: {self.config.eval.eval_data_path}")
            raise FileNotFoundError(f"Analysis data not found: {self.config.eval.eval_data_path}")
            
        # 加载数据集
        dataset = ManagementDataset(
            self.config.eval.eval_data_path,
            emotion_labels=self.config.eval.emotion_labels,
            strategy_labels=self.config.eval.strategy_labels
        )
        
        # 评估基础模型
        base_model = create_management_model(self.config.model)
        base_model.load_state_dict(torch.load(base_model_path, map_location=self.device))
        base_model.to(self.device)
        
        base_evaluator = create_evaluator(base_model, self.config.eval)
        base_metrics = base_evaluator.evaluate(dataset, save_results=False)
        
        # 评估增强模型
        enhanced_model = create_management_model(self.config.model)
        enhanced_model.load_state_dict(torch.load(enhanced_model_path, map_location=self.device))
        enhanced_model.to(self.device)
        
        enhanced_evaluator = create_evaluator(enhanced_model, self.config.eval)
        enhanced_metrics = enhanced_evaluator.evaluate(dataset, save_results=False)
        
        # 分析情感集成的有效性
        effectiveness = enhanced_evaluator.evaluate_integration_effectiveness(base_metrics, enhanced_metrics)
        
        return effectiveness
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13

# 命令行接口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Management Model Application")
<<<<<<< HEAD
    parser.add_argument('--port', type=int, default=5000, help='Port for Flask API')
    
    # 定义子解析器
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # 添加服务启动命令
    serve_parser = subparsers.add_parser("serve", help="Start Flask API server")
    serve_parser.add_argument('--port', type=int, default=5000, help='Port for Flask API')
    
=======
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    # 加载模型命令
    load_parser = subparsers.add_parser("load", help="Load a trained model")
    load_parser.add_argument("--model_path", type=str, help="Path to the model file")
    load_parser.add_argument("--config", type=str, help="Path to the configuration file")
    
    # 训练模型命令
    train_parser = subparsers.add_parser("train", help="Train the management model")
    train_parser.add_argument("--data_path", type=str, help="Path to the training data")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, help="Batch size")
    train_parser.add_argument("--config", type=str, help="Path to the configuration file")
    train_parser.add_argument("--environment", type=str, choices=["dev", "prod", "test"], 
                              help="Environment configuration to use")
    
    # 评估模型命令
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the management model")
    eval_parser.add_argument("--data_path", type=str, help="Path to the evaluation data")
    eval_parser.add_argument("--model_path", type=str, help="Path to the model file")
    eval_parser.add_argument("--config", type=str, help="Path to the configuration file")
    
    # 预测命令
    predict_parser = subparsers.add_parser("predict", help="Make predictions with the management model")
    predict_parser.add_argument("--input_file", type=str, required=True, help="Path to the input data file")
    predict_parser.add_argument("--model_path", type=str, help="Path to the model file")
    predict_parser.add_argument("--output_file", type=str, help="Path to save the prediction results")
    predict_parser.add_argument("--config", type=str, help="Path to the configuration file")
    
    # 分析情感集成命令
    analyze_parser = subparsers.add_parser("analyze", help="Analyze emotion integration effectiveness")
    analyze_parser.add_argument("--base_model", type=str, required=True, help="Path to the base model file")
    analyze_parser.add_argument("--enhanced_model", type=str, required=True, help="Path to the enhanced model file")
    analyze_parser.add_argument("--data_path", type=str, help="Path to the analysis data")
    analyze_parser.add_argument("--config", type=str, help="Path to the configuration file")
    
<<<<<<< HEAD
    # 添加load命令（之前漏掉了）
    load_parser = subparsers.add_parser("load", help="Load a trained model")
    load_parser.add_argument("--model_path", type=str, help="Path to the model file")
    load_parser.add_argument("--config", type=str, help="Path to the configuration file")

    # 解析参数
    args = parser.parse_args()
    
    # 默认行为：启动API服务
    if args.command is None or args.command == "serve":
        port = args.port if hasattr(args, 'port') else 5000
        # 正确初始化management_app
        management_app = ManagementApp()
        management_app.config.base.port = port
        try:
            logger.info(f"Starting Management Model API server on port {port}")
            app.run(host='0.0.0.0', port=port, debug=False)
        except Exception as e:
            logger.error(f"API服务启动失败: {str(e)}")
            sys.exit(1)
    else:
        # 创建应用实例
        if args.config:
            app = ManagementApp(args.config)
        elif hasattr(args, 'environment') and args.environment == "dev":
            app = ManagementApp(get_dev_config())
        elif hasattr(args, 'environment') and args.environment == "prod":
            app = ManagementApp(get_prod_config())
        elif hasattr(args, 'environment') and args.environment == "test":
            app = ManagementApp(get_test_config())
        else:
            app = ManagementApp()
        
        # 执行命令
        try:
            if args.command == "load":
                app.load_model(args.model_path)
            elif args.command == "train":
                app.train_model(args.data_path, args.epochs, args.batch_size)
            elif args.command == "evaluate":
                app.evaluate_model(args.data_path, args.model_path)
            elif args.command == "predict":
                # 加载输入数据
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                    
                # 执行预测
                results = app.predict(input_data)
                
                # 保存结果
                if args.output_file:
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    logger.info(f"Prediction results saved to {args.output_file}")
                else:
                    # 打印结果
                    print(json.dumps(results, indent=2, ensure_ascii=False))
            elif args.command == "analyze":
                effectiveness = app.analyze_emotion_integration(args.base_model, args.enhanced_model, args.data_path)
                print(json.dumps(effectiveness, indent=2, ensure_ascii=False))
            else:
                parser.print_help()
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            sys.exit(1)
=======
    # 解析参数
    args = parser.parse_args()
    
    # 创建应用实例
    if args.config:
        app = ManagementApp(args.config)
    elif args.environment == "dev":
        app = ManagementApp(get_dev_config())
    elif args.environment == "prod":
        app = ManagementApp(get_prod_config())
    elif args.environment == "test":
        app = ManagementApp(get_test_config())
    else:
        app = ManagementApp()
    
    # 执行命令
    try:
        if args.command == "load":
            app.load_model(args.model_path)
        elif args.command == "train":
            app.train_model(args.data_path, args.epochs, args.batch_size)
        elif args.command == "evaluate":
            app.evaluate_model(args.data_path, args.model_path)
        elif args.command == "predict":
            # 加载输入数据
            with open(args.input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
                
            # 执行预测
            results = app.predict(input_data)
            
            # 保存结果
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Prediction results saved to {args.output_file}")
            else:
                # 打印结果
                print(json.dumps(results, indent=2, ensure_ascii=False))
        elif args.command == "analyze":
            effectiveness = app.analyze_emotion_integration(args.base_model, args.enhanced_model, args.data_path)
            print(json.dumps(effectiveness, indent=2, ensure_ascii=False))
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        sys.exit(1)
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
