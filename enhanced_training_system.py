# -*- coding: utf-8 -*-
# 增强型训练控制系统 - 支持单独训练和联合训练
# Enhanced Training Control System - Support Individual and Joint Training
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import threading
from concurrent.futures import ThreadPoolExecutor

class EnhancedTrainingController:
    """增强型训练控制器 - 支持所有模型的单独训练和联合训练
    Enhanced Training Controller - Support individual and joint training for all models
    """
    
    def __init__(self, model_registry, language='zh'):
        """初始化训练控制器 | Initialize training controller
        
        参数 Parameters:
            model_registry: 模型注册表实例 | Model registry instance
            language: 系统语言 (zh/en) | System language (zh/en)
        """
        self.model_registry = model_registry
        self.language = language
        
        # 训练状态
        self.training_status = {
            "active_trainings": {},
            "training_history": [],
            "performance_metrics": {},
            "resource_usage": {}
        }
        
        # 训练配置
        self.training_configs = self._load_default_configs()
        
        # 训练数据集
        self.training_datasets = {}
        
        # 线程池用于并行训练
        self.thread_pool = ThreadPoolExecutor(max_workers=5)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def _load_default_configs(self) -> Dict[str, Any]:
        """加载默认训练配置 | Load default training configurations"""
        configs = {
            "B_language": {
                "learning_rate": 1e-4,
                "batch_size": 32,
                "epochs": 10,
                "optimizer": "adam",
                "loss_function": "cross_entropy",
                "validation_split": 0.2,
                "early_stopping_patience": 3
            },
            "C_audio": {
                "learning_rate": 1e-3,
                "batch_size": 16,
                "epochs": 15,
                "optimizer": "adam",
                "loss_function": "mse",
                "validation_split": 0.15,
                "early_stopping_patience": 5
            },
            "D_image": {
                "learning_rate": 2e-4,
                "batch_size": 8,
                "epochs": 20,
                "optimizer": "adam",
                "loss_function": "binary_cross_entropy",
                "validation_split": 0.1,
                "early_stopping_patience": 4
            },
            "E_video": {
                "learning_rate": 1e-4,
                "batch_size": 4,
                "epochs": 12,
                "optimizer": "adam",
                "loss_function": "mse",
                "validation_split": 0.15,
                "early_stopping_patience": 3
            },
            "joint_training": {
                "learning_rate": 1e-4,
                "batch_size": 8,
                "epochs": 25,
                "optimizer": "adam",
                "loss_function": "weighted_sum",
                "validation_split": 0.2,
                "early_stopping_patience": 5,
                "model_weights": {
                    "B_language": 0.3,
                    "C_audio": 0.2,
                    "D_image": 0.25,
                    "E_video": 0.25
                }
            }
        }
        return configs
    
    def start_individual_training(self, model_name: str, 
                                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """启动单个模型训练 | Start individual model training
        
        参数 Parameters:
            model_name: 模型名称 | Model name
            config: 训练配置 | Training configuration
            
        返回 Returns:
            训练结果 | Training results
        """
        if config is None:
            config = self.training_configs.get(model_name, {})
        
        # 检查模型是否存在
        if not self.model_registry.get_model(model_name):
            return {
                "status": "error",
                "message": f"模型 {model_name} 未找到 | Model {model_name} not found"
            }
        
        # 准备训练数据
        dataset = self._prepare_training_data(model_name)
        if not dataset:
            return {
                "status": "error",
                "message": f"无法准备 {model_name} 的训练数据 | Cannot prepare training data for {model_name}"
            }
        
        # 创建训练任务
        training_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 启动训练线程
        future = self.thread_pool.submit(
            self._train_model, model_name, dataset, config, training_id
        )
        
        # 记录训练状态
        self.training_status["active_trainings"][training_id] = {
            "model": model_name,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "config": config,
            "future": future
        }
        
        return {
            "status": "started",
            "training_id": training_id,
            "model": model_name,
            "start_time": self.training_status["active_trainings"][training_id]["start_time"]
        }
    
    def start_joint_training(self, model_names: List[str],
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """启动联合训练 | Start joint training
        
        参数 Parameters:
            model_names: 参与训练的模型名称列表 | List of model names for training
            config: 训练配置 | Training configuration
            
        返回 Returns:
            训练结果 | Training results
        """
        if config is None:
            config = self.training_configs.get("joint_training", {})
        
        # 验证所有模型都存在
        for model_name in model_names:
            if not self.model_registry.get_model(model_name):
                return {
                    "status": "error",
                    "message": f"模型 {model_name} 未找到 | Model {model_name} not found"
                }
        
        # 准备联合训练数据
        joint_dataset = self._prepare_joint_training_data(model_names)
        if not joint_dataset:
            return {
                "status": "error",
                "message": "无法准备联合训练数据 | Cannot prepare joint training data"
            }
        
        # 创建联合训练任务
        training_id = f"joint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 启动联合训练线程
        future = self.thread_pool.submit(
            self._train_joint_models, model_names, joint_dataset, config, training_id
        )
        
        # 记录训练状态
        self.training_status["active_trainings"][training_id] = {
            "models": model_names,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "config": config,
            "future": future
        }
        
        return {
            "status": "started",
            "training_id": training_id,
            "models": model_names,
            "start_time": self.training_status["active_trainings"][training_id]["start_time"]
        }
    
    def _train_model(self, model_name: str, dataset: Dataset, 
                    config: Dict[str, Any], training_id: str) -> Dict[str, Any]:
        """训练单个模型 | Train individual model"""
        try:
            # 获取模型实例
            model = self.model_registry.get_model_instance(model_name)
            if model is None:
                return {
                    "status": "error",
                    "message": f"无法获取模型实例 | Cannot get model instance: {model_name}"
                }
            
            # 准备数据加载器
            dataloader = DataLoader(
                dataset,
                batch_size=config.get("batch_size", 32),
                shuffle=True,
                num_workers=2
            )
            
            # 设置优化器和损失函数
            optimizer = self._get_optimizer(model, config)
            criterion = self._get_loss_function(config)
            
            # 训练循环
            best_loss = float('inf')
            patience_counter = 0
            early_stopping_patience = config.get("early_stopping_patience", 3)
            
            for epoch in range(config.get("epochs", 10)):
                epoch_loss = 0.0
                model.train()
                
                for batch_idx, (data, target) in enumerate(dataloader):
                    optimizer.zero_grad()
                    
                    # 前向传播
                    output = model(data)
                    
                    # 计算损失
                    loss = criterion(output, target)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # 更新训练状态
                    self._update_training_status(
                        training_id,
                        {
                            "epoch": epoch + 1,
                            "batch": batch_idx + 1,
                            "current_loss": loss.item(),
                            "average_loss": epoch_loss / (batch_idx + 1)
                        }
                    )
                
                # 检查早停
                avg_epoch_loss = epoch_loss / len(dataloader)
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self._save_model_checkpoint(model, model_name, epoch)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info(f"早停触发于 epoch {epoch+1} | Early stopping triggered at epoch {epoch+1}")
                        break
                
                # 记录epoch结果
                self._record_epoch_result(training_id, epoch + 1, avg_epoch_loss)
            
            # 完成训练
            final_result = {
                "status": "completed",
                "model": model_name,
                "final_loss": best_loss,
                "total_epochs": epoch + 1,
                "training_time": time.time() - start_time
            }
            
            # 更新训练状态
            self._complete_training(training_id, final_result)
            
            return final_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "model": model_name,
                "error": str(e),
                "training_time": time.time() - start_time
            }
            self._complete_training(training_id, error_result)
            return error_result
    
    def _train_joint_models(self, model_names: List[str], dataset: Dataset,
                           config: Dict[str, Any], training_id: str) -> Dict[str, Any]:
        """训练多个模型（联合训练） | Train multiple models (joint training)"""
        try:
            # 获取所有模型实例
            models = {}
            for model_name in model_names:
                model = self.model_registry.get_model_instance(model_name)
                if model is None:
                    return {
                        "status": "error",
                        "message": f"无法获取模型实例 | Cannot get model instance: {model_name}"
                    }
                models[model_name] = model
            
            # 准备数据加载器
            dataloader = DataLoader(
                dataset,
                batch_size=config.get("batch_size", 8),
                shuffle=True,
                num_workers=2
            )
            
            # 为每个模型设置优化器和损失函数
            optimizers = {}
            criteria = {}
            model_weights = config.get("model_weights", {})
            
            for model_name, model in models.items():
                optimizers[model_name] = self._get_optimizer(model, config)
                criteria[model_name] = self._get_loss_function(config)
                # 设置默认权重
                if model_name not in model_weights:
                    model_weights[model_name] = 1.0 / len(model_names)
            
            # 联合训练循环
            best_total_loss = float('inf')
            patience_counter = 0
            early_stopping_patience = config.get("early_stopping_patience", 5)
            
            for epoch in range(config.get("epochs", 25)):
                total_epoch_loss = 0.0
                
                for batch_idx, (data, targets) in enumerate(dataloader):
                    # 为每个模型清零梯度
                    for optimizer in optimizers.values():
                        optimizer.zero_grad()
                    
                    batch_loss = 0.0
                    individual_losses = {}
                    
                    # 每个模型的前向传播和损失计算
                    for model_name, model in models.items():
                        model.train()
                        output = model(data[model_name])
                        loss = criteria[model_name](output, targets[model_name])
                        weighted_loss = loss * model_weights[model_name]
                        weighted_loss.backward()
                        
                        individual_losses[model_name] = loss.item()
                        batch_loss += weighted_loss.item()
                    
                    # 更新所有模型的参数
                    for optimizer in optimizers.values():
                        optimizer.step()
                    
                    total_epoch_loss += batch_loss
                    
                    # 更新训练状态
                    self._update_training_status(
                        training_id,
                        {
                            "epoch": epoch + 1,
                            "batch": batch_idx + 1,
                            "current_loss": batch_loss,
                            "average_loss": total_epoch_loss / (batch_idx + 1),
                            "individual_losses": individual_losses
                        }
                    )
                
                # 检查早停
                avg_epoch_loss = total_epoch_loss / len(dataloader)
                if avg_epoch_loss < best_total_loss:
                    best_total_loss = avg_epoch_loss
                    patience_counter = 0
                    # 保存所有模型的最佳检查点
                    for model_name, model in models.items():
                        self._save_model_checkpoint(model, model_name, epoch)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info(f"联合训练早停触发于 epoch {epoch+1} | Joint training early stopping triggered at epoch {epoch+1}")
                        break
                
                # 记录epoch结果
                self._record_epoch_result(training_id, epoch + 1, avg_epoch_loss)
            
            # 完成训练
            final_result = {
                "status": "completed",
                "models": model_names,
                "final_loss": best_total_loss,
                "total_epochs": epoch + 1,
                "training_time": time.time() - start_time
            }
            
            # 更新训练状态
            self._complete_training(training_id, final_result)
            
            return final_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "models": model_names,
                "error": str(e),
                "training_time": time.time() - start_time
            }
            self._complete_training(training_id, error_result)
            return error_result
    
    def _prepare_training_data(self, model_name: str) -> Optional[Dataset]:
        """准备训练数据 | Prepare training data"""
        # 根据模型类型准备相应的数据集
        # 这里需要实现具体的数据准备逻辑
        # 目前返回None，实际中应该返回相应的Dataset实例
        
        # 示例实现框架
        if model_name == "B_language":
            return LanguageDataset()
        elif model_name == "C_audio":
            return AudioDataset()
        elif model_name == "D_image":
            return ImageDataset()
        elif model_name == "E_video":
            return VideoDataset()
        else:
            self.logger.warning(f"未知模型类型: {model_name} | Unknown model type: {model_name}")
            return None
    
    def _prepare_joint_training_data(self, model_names: List[str]) -> Optional[Dataset]:
        """准备联合训练数据 | Prepare joint training data"""
        # 实现多模态联合训练数据准备
        # 这里需要根据参与的模型类型组合相应的数据
        
        # 检查是否所有模型都有对应的数据
        for model_name in model_names:
            if not self._has_training_data(model_name):
                self.logger.error(f"模型 {model_name} 缺少训练数据 | Model {model_name} lacks training data")
                return None
        
        # 创建联合数据集
        return JointDataset(model_names)
    
    def _get_optimizer(self, model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        """获取优化器 | Get optimizer"""
        optimizer_name = config.get("optimizer", "adam").lower()
        learning_rate = config.get("learning_rate", 1e-4)
        
        if optimizer_name == "adam":
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            self.logger.warning(f"未知优化器: {optimizer_name}, 使用默认Adam | Unknown optimizer: {optimizer_name}, using default Adam")
            return optim.Adam(model.parameters(), lr=learning_rate)
    
    def _get_loss_function(self, config: Dict[str, Any]) -> nn.Module:
        """获取损失函数 | Get loss function"""
        loss_name = config.get("loss_function", "cross_entropy").lower()
        
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "binary_cross_entropy":
            return nn.BCELoss()
        else:
            self.logger.warning(f"未知损失函数: {loss_name}, 使用默认交叉熵 | Unknown loss function: {loss_name}, using default cross entropy")
            return nn.CrossEntropyLoss()
    
    def _update_training_status(self, training_id: str, status_update: Dict[str, Any]):
        """更新训练状态 | Update training status"""
        if training_id in self.training_status["active_trainings"]:
            current_status = self.training_status["active_trainings"][training_id]
            current_status.update(status_update)
            current_status["last_update"] = datetime.now().isoformat()
    
    def _record_epoch_result(self, training_id: str, epoch: int, loss: float):
        """记录epoch结果 | Record epoch result"""
        # 这里可以记录每个epoch的详细结果用于分析和可视化
        pass
    
    def _complete_training(self, training_id: str, result: Dict[str, Any]):
        """完成训练任务 | Complete training task"""
        if training_id in self.training_status["active_trainings"]:
            training_info = self.training_status["active_trainings"][training_id]
            training_info["status"] = result["status"]
            training_info["end_time"] = datetime.now().isoformat()
            training_info["result"] = result
            
            # 移动到历史记录
            self.training_status["training_history"].append(training_info)
            del self.training_status["active_trainings"][training_id]
    
    def _save_model_checkpoint(self, model: nn.Module, model_name: str, epoch: int):
        """保存模型检查点 | Save model checkpoint"""
        # 实现模型保存逻辑
        checkpoint_path = f"checkpoints/{model_name}/epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,  # 实际中应该保存优化器状态
            'loss': None  # 实际中应该保存损失值
        }, checkpoint_path)
        
        self.logger.info(f"模型 {model_name} 检查点已保存: {checkpoint_path} | Model {model_name} checkpoint saved: {checkpoint_path}")
    
    def get_training_status(self, training_id: Optional[str] = None) -> Dict[str, Any]:
        """获取训练状态 | Get training status"""
        if training_id:
            if training_id in self.training_status["active_trainings"]:
                return self.training_status["active_trainings"][training_id]
            else:
                # 在历史记录中查找
                for history in self.training_status["training_history"]:
                    if history.get("training_id") == training_id:
                        return history
                return {"status": "error", "message": "训练ID未找到 | Training ID not found"}
        else:
            return self.training_status
    
    def stop_training(self, training_id: str) -> Dict[str, Any]:
        """停止训练 | Stop training"""
        if training_id in self.training_status["active_trainings"]:
            training_info = self.training_status["active_trainings"][training_id]
            future = training_info.get("future")
            
            if future and not future.done():
                future.cancel()
                training_info["status"] = "cancelled"
                training_info["end_time"] = datetime.now().isoformat()
                
                # 移动到历史记录
                self.training_status["training_history"].append(training_info)
                del self.training_status["active_trainings"][training_id]
                
                return {
                    "status": "success",
                    "message": f"训练 {training_id} 已停止 | Training {training_id} stopped"
                }
            else:
                return {
                    "status": "error",
                    "message": f"训练 {training_id} 已完成或无法停止 | Training {training_id} completed or cannot be stopped"
                }
        else:
            return {
                "status": "error",
                "message": f"训练ID未找到: {training_id} | Training ID not found: {training_id}"
            }

# 数据集类定义
class LanguageDataset(Dataset):
    """语言数据集 | Language dataset"""
    def __init__(self):
        # 实现语言数据加载逻辑
        pass
    
    def __len__(self):
        return 0  # 实际中返回数据数量
    
    def __getitem__(self, idx):
        # 返回语言数据和标签
        return None, None

class AudioDataset(Dataset):
    """音频数据集 | Audio dataset"""
    def __init__(self):
        # 实现音频数据加载逻辑
        pass
    
    def __len__(self):
        return 0  # 实际中返回数据数量
    
    def __getitem__(self, idx):
        # 返回音频数据和标签
        return None, None

class ImageDataset(Dataset):
    """图像数据集 | Image dataset"""
    def __init__(self):
        # 实现图像数据加载逻辑
        pass
    
    def __len__(self):
        return 0  # 实际中返回数据数量
    
    def __getitem__(self, idx):
        # 返回图像数据和标签
        return None, None

class VideoDataset(Dataset):
    """视频数据集 | Video dataset"""
    def __init__(self):
        # 实现视频数据加载逻辑
        pass
    
    def __len__(self):
        return 0  # 实际中返回数据数量
    
    def __getitem__(self, idx):
        # 返回视频数据和标签
        return None, None

class JointDataset(Dataset):
    """联合训练数据集 | Joint training dataset"""
    def __init__(self, model_names):
        self.model_names = model_names
        # 实现多模态数据加载和组合逻辑
        pass
    
    def __len__(self):
        return 0  # 实际中返回数据数量
    
    def __getitem__(self, idx):
        # 返回多模态数据和标签
        data = {}
        targets = {}
        
        for model_name in self.model_names:
            # 为每个模型准备相应的数据和标签
            data[model_name] = None  # 实际中加载相应数据
            targets[model_name] = None  # 实际中加载相应标签
        
        return data, targets

# 工具函数
def create_training_controller(model_registry, language='zh'):
    """创建训练控制器实例 | Create training controller instance"""
    return EnhancedTrainingController(model_registry, language)

if __name__ == '__main__':
    # 测试训练控制系统
    # Test training control system
    print("初始化训练控制器... | Initializing training controller...")
    
    # 创建模拟模型注册表
    class MockModelRegistry:
        def get_model(self, name):
            return {"status": "active"}
        
        def get_model_instance(self, name):
            # 返回模拟模型
            return nn.Linear(10, 2)
    
    model_registry = MockModelRegistry()
    controller = create_training_controller(model_registry)
    
    # 测试单个模型训练
    print("测试单个模型训练... | Testing individual model training...")
    result = controller.start_individual_training("B_language")
    print(f"训练启动结果: {result}")
    
    # 测试联合训练
    print("测试联合训练... | Testing joint training...")
    joint_result = controller.start_joint_training(["B_language", "C_audio"])
    print(f"联合训练启动结果: {joint_result}")
    
    # 获取训练状态
    time.sleep(2)
    status = controller.get_training_status()
    print(f"训练状态: {status}")
    
    print("训练控制系统测试完成! | Training control system testing completed!")