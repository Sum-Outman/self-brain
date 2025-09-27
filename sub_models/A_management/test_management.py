# Copyright 2025 The AI Management System Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
测试文件，用于测试管理模型的各个功能模块
"""

import os
import sys
import unittest
import json
import tempfile
import numpy as np
import torch
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入测试所需的模块
from A_management.enhanced_manager import ManagementModel
from A_management.enhanced_trainer import ManagementDataset, ModelTrainer
from A_management.enhanced_evaluator import ModelEvaluator
from A_management.config import Config, get_default_config

# 配置测试环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestManagementModel(unittest.TestCase):
    """
    管理模型测试类
    """
    def setUp(self):
        """
        测试前的准备工作
        """
        # 获取默认配置
        self.config = get_default_config()
        
        # 创建临时文件目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建管理模型实例 - 使用字典而不是ModelConfig对象
        model_config = {
            'hidden_dim': 256,
            'num_strategies': 5,
            'num_emotions': 7
        }
        self.model = ManagementModel(model_config)
        
        # 创建训练器实例 - 使用字典而不是Config对象
        trainer_config = {
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001
        }
        self.trainer = ModelTrainer(self.model, trainer_config)
        
        # 创建测试数据集
        self.test_data = [
            {
                'features': np.random.rand(100),
                'strategy_label': 0,
                'emotion_label': 3,
                'sub_model_outputs': {
                    'B_language': {
                        'emotion_pred': 3,
                        'confidence': 0.92
                    }
                }
            },
            {
                'features': np.random.rand(100),
                'strategy_label': 1,
                'emotion_label': 1,
                'sub_model_outputs': {
                    'B_language': {
                        'emotion_pred': 1,
                        'confidence': 0.95
                    }
                }
            }
        ]
        
        # 保存测试数据到临时文件，将numpy数组转换为列表
        self.train_data_path = os.path.join(self.temp_dir, 'train_data.json')
        # 将numpy数组转换为Python列表
        serializable_data = []
        for item in self.test_data:
            serializable_item = item.copy()
            # 转换numpy数组为列表
            if isinstance(serializable_item['features'], np.ndarray):
                serializable_item['features'] = serializable_item['features'].tolist()
            serializable_data.append(serializable_item)
        with open(self.train_data_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    def tearDown(self):
        """
        测试后的清理工作
        """
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """
        测试训练器初始化
        """
        # 检查训练器是否成功初始化
        self.assertIsNotNone(self.trainer)
        
        # 检查模型是否正确
        self.assertEqual(self.trainer.model, self.model)
        
        # 检查配置是否包含预期的键，而不是直接比较对象
        self.assertIn('learning_rate', self.trainer.config)
        self.assertIn('batch_size', self.trainer.config)
        self.assertIn('epochs', self.trainer.config)
        self.assertEqual(self.trainer.config['learning_rate'], 0.001)  # 检查特定配置值
    
    def test_prepare_data(self):
        """
        测试数据准备功能
        """
        # 准备数据
        self.trainer.prepare_data(self.train_data_path)
        
        # 检查数据加载器是否创建
        self.assertIsNotNone(self.trainer.train_loader)
    
    def test_optimizer_scheduler_creation(self):
        """
        测试优化器和学习率调度器创建
        """
        # 直接访问已创建的优化器和调度器
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.scheduler)

class TestModelEvaluator(unittest.TestCase):
    """
    模型评估器测试类
    """
    def setUp(self):
        """
        测试前的准备工作
        """
        # 获取默认配置
        self.config = get_default_config()
        
        # 创建临时文件目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建管理模型实例 - 使用字典而不是ModelConfig对象
        model_config = {
            'hidden_dim': 256,
            'num_strategies': 5,
            'num_emotions': 7
        }
        self.model = ManagementModel(model_config)
        
        # 创建评估器实例 - 使用字典而不是Config对象
        evaluator_config = {
            'batch_size': 32,
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score']
        }
        self.evaluator = ModelEvaluator(self.model, evaluator_config)
        
        # 创建测试数据集
        self.test_data = [
            {
                'features': np.random.rand(100),
                'strategy_label': 0,
                'emotion_label': 3,
                'sub_model_outputs': {
                    'B_language': {
                        'emotion_pred': 3,
                        'confidence': 0.92
                    }
                }
            },
            {
                'features': np.random.rand(100),
                'strategy_label': 1,
                'emotion_label': 1,
                'sub_model_outputs': {
                    'B_language': {
                        'emotion_pred': 1,
                        'confidence': 0.95
                    }
                }
            }
        ]
        
        # 保存测试数据到临时文件，将numpy数组转换为列表
        self.test_data_path = os.path.join(self.temp_dir, 'test_data.json')
        # 将numpy数组转换为Python列表
        serializable_data = []
        for item in self.test_data:
            serializable_item = item.copy()
            # 转换numpy数组为列表
            if isinstance(serializable_item['features'], np.ndarray):
                serializable_item['features'] = serializable_item['features'].tolist()
            serializable_data.append(serializable_item)
        with open(self.test_data_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    def tearDown(self):
        """
        测试后的清理工作
        """
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_evaluator_initialization(self):
        """
        测试评估器初始化
        """
        # 检查评估器是否成功初始化
        self.assertIsNotNone(self.evaluator)
        
        # 检查模型是否正确
        self.assertEqual(self.evaluator.model, self.model)
        
        # 检查配置是否包含预期的键，而不是直接比较对象
        self.assertIn('batch_size', self.evaluator.config)
        self.assertIn('metrics_dir', self.evaluator.config)
        self.assertIn('figures_dir', self.evaluator.config)
        self.assertEqual(self.evaluator.config['batch_size'], 32)  # 检查特定配置值
    
    def test_calculate_metrics(self):
        """
        测试指标计算功能
        """
        # 创建测试预测和目标
        predictions = [0, 1, 2, 0]
        targets = [0, 1, 1, 0]
        # 为了使用内部的_calculate_metrics方法，我们需要创建概率和标签列表
        probabilities = [np.array([0.9, 0.1, 0.0]), np.array([0.1, 0.8, 0.1]), 
                        np.array([0.0, 0.2, 0.8]), np.array([0.9, 0.1, 0.0])]
        labels = ['class_0', 'class_1', 'class_2']
        
        # 计算指标 - 注意：这是直接调用内部方法进行测试
        metrics = self.evaluator._calculate_metrics(targets, predictions, probabilities, labels)
        
        # 检查指标是否正确计算
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_weighted', metrics)
        self.assertIn('recall_weighted', metrics)
        self.assertIn('f1_weighted', metrics)
        
        # 检查指标值范围
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['precision_weighted'] <= 1)
        self.assertTrue(0 <= metrics['recall_weighted'] <= 1)
        self.assertTrue(0 <= metrics['f1_weighted'] <= 1)

# 运行所有测试
if __name__ == '__main__':
    unittest.main()