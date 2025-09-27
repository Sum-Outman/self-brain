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

# 导入管理模型相关模块
from A_management.enhanced_manager import ManagementModel, create_management_model
from A_management.enhanced_trainer import ModelTrainer, create_trainer, ManagementDataset
from A_management.enhanced_evaluator import ModelEvaluator, create_evaluator
from A_management.config import Config, get_default_config, get_dev_config, get_prod_config, get_test_config

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
        from A_management.config import validate_config
        if not validate_config(self.config):
            logger.error("Invalid configuration detected")
            raise ValueError("Invalid configuration")
            
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 初始化模型、训练器和评估器
        self.model = None
        self.trainer = None
        self.evaluator = None
        
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
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # 创建模型实例
        self.model = create_management_model(self.config.model)
        
        # 加载模型权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
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
        
        参数:
            data_path: 评估数据路径
            model_path: 模型文件路径
        """
        # 更新配置参数
        if data_path:
            self.config.eval.eval_data_path = data_path
        
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
            logger.error("No model loaded. Please load a model first.")
            raise ValueError("No model loaded. Please load a model first.")
            
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
    
    def _predict_single(self, input_data):
        """
        对单个样本进行预测
        
        参数:
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
    
    def analyze_emotion_integration(self, base_model_path, enhanced_model_path, data_path=None):
        """
        分析情感集成的有效性
        
        参数:
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

# 命令行接口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Management Model Application")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
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