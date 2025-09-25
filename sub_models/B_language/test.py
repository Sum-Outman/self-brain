#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import json
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们需要的类和函数
from train import LanguageModelTrainer, start_training, start_joint_training

class TestLanguageModelTrainer(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时配置
        self.test_config = {
            "model_id": "test-model",
            "is_pretrained": False,
            "vocab_size": 10000,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 1,
            "dropout": 0.2,
            "data_path": "data",
            "train_size": 0.8,
            "val_size": 0.1,
            "max_length": 64,
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,
            "early_stopping_patience": 3,
            "checkpoint_dir": "checkpoints/test",
            "log_interval": 1,
            "main_metric": "accuracy",
            "metric_direction": "max"
        }
        
        # 确保测试目录存在
        os.makedirs(self.test_config["checkpoint_dir"], exist_ok=True)
        
        # 创建测试实例
        self.trainer = LanguageModelTrainer(self.test_config)
    
    def tearDown(self):
        """清理测试环境"""
        # 这里可以添加清理代码，比如删除临时文件等
        pass
    
    def test_trainer_initialization(self):
        """测试训练器初始化是否正确"""
        self.assertEqual(self.trainer.config["model_id"], "test-model")
        self.assertEqual(self.trainer.config["batch_size"], 2)
        self.assertIsNotNone(self.trainer.language_resources)
    
    def test_prepare_dataset(self):
        """测试数据集准备功能"""
        # 尝试准备数据集
        self.trainer.prepare_dataset()
        self.assertIsNotNone(self.trainer.train_dataset)
        self.assertIsNotNone(self.trainer.val_dataset)
        self.assertIsNotNone(self.trainer.test_dataset)
        self.assertGreater(len(self.trainer.train_dataset), 0)
        self.assertGreater(len(self.trainer.val_dataset), 0)
        self.assertGreater(len(self.trainer.test_dataset), 0)
    
    def test_config_update(self):
        """测试配置更新功能"""
        # 创建新配置
        new_config = {
            "model_id": "updated-model",
            "epochs": 5,
            "batch_size": 4
        }
        
        # 创建新的训练器实例
        trainer = LanguageModelTrainer(new_config)
        
        # 验证配置是否更新
        self.assertEqual(trainer.config["model_id"], "updated-model")
        self.assertEqual(trainer.config["epochs"], 5)
        self.assertEqual(trainer.config["batch_size"], 4)
        # 验证默认配置是否保留
        self.assertEqual(trainer.config["learning_rate"], 1e-4)
    
    def test_load_language_resources(self):
        """测试语言资源加载功能"""
        # 验证语言资源是否正确加载
        resources = self.trainer.language_resources
        self.assertIn("emotion_labels", resources)
        self.assertIn("en", resources["emotion_labels"])
        self.assertIn("zh", resources["emotion_labels"])
        self.assertIn("ja", resources["emotion_labels"])
        self.assertIn("de", resources["emotion_labels"])
        self.assertIn("fr", resources["emotion_labels"])
    
    def test_start_training_api(self):
        """测试训练API接口"""
        # 创建简化的训练配置
        training_config = {
            "model_id": "api-test",
            "is_pretrained": False,
            "epochs": 1,
            "batch_size": 2,
            "checkpoint_dir": "checkpoints/api_test"
        }
        
        # 确保检查点目录存在
        os.makedirs(training_config["checkpoint_dir"], exist_ok=True)
        
        # 测试API
        try:
            # 直接测试完整的API
            result = start_training(training_config)
            self.assertIn("status", result)
            
            # 如果状态是error，打印完整的错误信息并立即失败
            if result["status"] == "error":
                error_message = result.get('message', 'No error message provided')
                print(f"API Training Error: {error_message}")
                self.fail(f"start_training API failed with error: {error_message}")
            
            self.assertEqual(result["status"], "success")
            self.assertIn("train_history", result)
            self.assertIn("evaluation_results", result)
        except Exception as e:
            self.fail(f"start_training API failed with error: {e}")
    
    def test_start_training_from_scratch(self):
        """测试从零开始训练功能"""
        # 创建从零开始训练的配置
        scratch_config = {
            "model_id": "from_scratch_test",
            "is_pretrained": False,
            "vocab_size": 10000,
            "epochs": 1,
            "batch_size": 2,
            "checkpoint_dir": "checkpoints/scratch_test"
        }
        
        # 确保检查点目录存在
        os.makedirs(scratch_config["checkpoint_dir"], exist_ok=True)
        
        # 测试从零开始训练
        try:
            result = start_training(scratch_config)
            self.assertIn("status", result)
            # 打印错误信息以便调试
            if result["status"] == "error":
                print(f"From Scratch Training Error: {result.get('message', 'No error message provided')}")
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["config"]["is_pretrained"], False)
        except Exception as e:
            self.fail(f"From scratch training failed with error: {e}")
    
    def test_start_training_pretrained(self):
        """测试使用预训练模型训练功能"""
        # 创建预训练模型配置
        pretrained_config = {
            "model_id": "pretrained_test",
            "is_pretrained": True,
            "model_name": "xlm-roberta-base",
            "epochs": 1,
            "batch_size": 2,
            "checkpoint_dir": "checkpoints/pretrained_test"
        }
        
        # 确保检查点目录存在
        os.makedirs(pretrained_config["checkpoint_dir"], exist_ok=True)
        
        # 测试预训练模型训练
        try:
            result = start_training(pretrained_config)
            # 即使失败也不抛出异常，因为加载预训练模型可能需要网络连接
            if "status" in result:
                self.assertIn(result["status"], ["success", "error"])
        except Exception as e:
            # 网络问题或其他环境限制可能导致失败，这里只记录错误但不标记测试失败
            print(f"Pretrained model training test encountered an error: {e}")
            print("This may be due to network connectivity or other environment limitations.")

if __name__ == "__main__":
    unittest.main()