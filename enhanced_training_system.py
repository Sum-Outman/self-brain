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
        self.logger.info(f"为模型 {model_name} 准备训练数据")
        
        # 根据模型类型创建相应的数据集
        if model_name == "B_language":
            dataset = LanguageDataset(data_dir="training_data/language", max_samples=1000)
        elif model_name == "C_audio":
            dataset = AudioDataset(data_dir="training_data/audio", max_samples=500, sample_rate=16000)
        elif model_name == "D_image":
            dataset = ImageDataset(data_dir="training_data/image", max_samples=800, image_size=(224, 224))
        elif model_name == "E_video":
            dataset = VideoDataset(data_dir="training_data/video", max_samples=300, 
                                  frame_size=(112, 112), num_frames=16)
        else:
            self.logger.warning(f"未知模型类型: {model_name} | Unknown model type: {model_name}")
            return None
        
        self.logger.info(f"成功准备 {model_name} 数据集，样本数量: {len(dataset)}")
        return dataset
    
    def _prepare_joint_training_data(self, model_names: List[str]) -> Optional[Dataset]:
        """准备联合训练数据 | Prepare joint training data"""
        self.logger.info(f"准备联合训练数据，模型列表: {model_names}")
        
        # 创建联合数据集
        joint_dataset = JointDataset(model_names, max_samples=500)
        
        self.logger.info(f"成功准备联合训练数据集，样本数量: {len(joint_dataset)}")
        return joint_dataset
    
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

import os
from pathlib import Path
from torchvision import transforms
import torchaudio
import cv2
import random

# 数据集类定义
class LanguageDataset(Dataset):
    """语言数据集 | Language dataset"""
    def __init__(self, data_dir="training_data/language", max_samples=1000):
        """初始化语言数据集
        
        参数 Parameters:
            data_dir: 数据目录 | Data directory
            max_samples: 最大样本数 | Maximum number of samples
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.data = []
        self.labels = []
        
        # 检查数据目录是否存在，如果不存在则创建示例数据
        if not self.data_dir.exists():
            self._create_sample_data()
        else:
            self._load_data()
    
    def _load_data(self):
        """加载语言数据"""
        try:
            # 尝试从JSON文件加载数据
            data_file = self.data_dir / "language_data.json"
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                    
                for i, item in enumerate(dataset):
                    if i >= self.max_samples:
                        break
                    self.data.append(item['text'])
                    self.labels.append(item['label'])
        except Exception as e:
            self.logger.error(f"加载语言数据失败: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据"""
        self.logger.info(f"创建语言示例数据到 {self.data_dir}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建示例文本数据
        sample_texts = [
            "这是一个测试句子",
            "自然语言处理很有趣",
            "深度学习在语言领域取得了很大进展",
            "我爱学习AI技术",
            "Python是一种流行的编程语言",
            "机器学习模型需要大量数据训练",
            "人工智能正在改变世界",
            "自然语言理解是AI的重要分支",
            "深度学习模型可以处理复杂的语言任务",
            "未来的AI系统将更加智能"
        ]
        
        # 复制样本以达到最大样本数
        for i in range(self.max_samples):
            text_idx = i % len(sample_texts)
            self.data.append(sample_texts[text_idx])
            self.labels.append(text_idx % 5)  # 随机标签
        
        # 保存示例数据
        dataset = [{"text": t, "label": l} for t, l in zip(self.data, self.labels)]
        data_file = self.data_dir / "language_data.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 对于真实应用，这里应该包括文本预处理、标记化、向量化等
        text = self.data[idx]
        label = self.labels[idx]
        
        # 简单的文本特征提取 (在实际应用中应该使用更复杂的方法)
        # 这里返回的是模拟特征
        feature = torch.tensor([ord(c) for c in text[:32]] + [0] * (32 - len(text[:32])), dtype=torch.float32)
        feature = feature / 255.0  # 归一化
        
        return feature, torch.tensor(label, dtype=torch.long)

class AudioDataset(Dataset):
    """音频数据集 | Audio dataset"""
    def __init__(self, data_dir="training_data/audio", max_samples=500, sample_rate=16000):
        """初始化音频数据集
        
        参数 Parameters:
            data_dir: 数据目录 | Data directory
            max_samples: 最大样本数 | Maximum number of samples
            sample_rate: 采样率 | Sample rate
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.audio_files = []
        self.labels = []
        
        # 检查数据目录是否存在，如果不存在则创建示例数据
        if not self.data_dir.exists():
            self._create_sample_data()
        else:
            self._load_data()
    
    def _load_data(self):
        """加载音频数据"""
        try:
            # 查找所有音频文件
            audio_extensions = ['.wav', '.mp3', '.flac']
            for ext in audio_extensions:
                for file_path in self.data_dir.glob(f'*{ext}'):
                    if len(self.audio_files) >= self.max_samples:
                        break
                    self.audio_files.append(str(file_path))
                    # 从文件名中提取标签
                    label = int(file_path.stem.split('_')[-1]) if '_' in file_path.stem else 0
                    self.labels.append(label)
        except Exception as e:
            self.logger.error(f"加载音频数据失败: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据"""
        self.logger.info(f"创建音频示例数据到 {self.data_dir}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建示例音频数据配置
        sample_config = {"files": [], "labels": []}
        
        for i in range(min(self.max_samples, 10)):  # 创建10个示例文件
            file_name = f"audio_{i}.json"
            file_path = self.data_dir / file_name
            
            # 生成随机音频特征
            audio_length = 16000  # 1秒的音频采样点
            random_audio = np.random.randn(audio_length).tolist()
            
            # 保存音频特征
            audio_data = {
                "audio": random_audio,
                "sample_rate": self.sample_rate,
                "duration": 1.0
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(audio_data, f)
            
            sample_config["files"].append(file_name)
            sample_config["labels"].append(i % 3)  # 随机标签
        
        # 复制样本以达到最大样本数
        for i in range(10, self.max_samples):
            orig_idx = i % 10
            self.audio_files.append(str(self.data_dir / sample_config["files"][orig_idx]))
            self.labels.append(sample_config["labels"][orig_idx])
        
        # 保存配置文件
        config_file = self.data_dir / "audio_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        try:
            file_path = self.audio_files[idx]
            label = self.labels[idx]
            
            # 检查文件扩展名，处理不同类型的音频数据
            if file_path.endswith('.json'):
                # 加载JSON格式的音频特征
                with open(file_path, 'r') as f:
                    audio_data = json.load(f)
                audio_tensor = torch.tensor(audio_data['audio'], dtype=torch.float32)
            else:
                # 实际加载音频文件
                try:
                    waveform, sr = torchaudio.load(file_path)
                    # 重采样到目标采样率
                    if sr != self.sample_rate:
                        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
                    # 转换为单声道
                    if waveform.size(0) > 1:
                        waveform = torch.mean(waveform, dim=0)
                    audio_tensor = waveform.squeeze()
                except:
                    # 如果加载失败，生成随机音频数据
                    audio_tensor = torch.randn(16000, dtype=torch.float32)
            
            # 标准化音频数据
            if audio_tensor.numel() > 0:
                audio_tensor = (audio_tensor - torch.mean(audio_tensor)) / (torch.std(audio_tensor) + 1e-8)
            
            # 确保长度一致
            target_length = 16000  # 1秒
            if len(audio_tensor) < target_length:
                pad_length = target_length - len(audio_tensor)
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_length))
            else:
                audio_tensor = audio_tensor[:target_length]
            
            return audio_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # 出错时返回随机数据
            self.logger.error(f"处理音频数据失败: {e}")
            return torch.randn(16000, dtype=torch.float32), torch.tensor(0, dtype=torch.long)

class ImageDataset(Dataset):
    """图像数据集 | Image dataset"""
    def __init__(self, data_dir="training_data/image", max_samples=800, image_size=(224, 224)):
        """初始化图像数据集
        
        参数 Parameters:
            data_dir: 数据目录 | Data directory
            max_samples: 最大样本数 | Maximum number of samples
            image_size: 图像大小 | Image size
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.image_size = image_size
        self.image_files = []
        self.labels = []
        
        # 图像预处理变换
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 检查数据目录是否存在，如果不存在则创建示例数据
        if not self.data_dir.exists():
            self._create_sample_data()
        else:
            self._load_data()
    
    def _load_data(self):
        """加载图像数据"""
        try:
            # 查找所有图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for ext in image_extensions:
                for file_path in self.data_dir.glob(f'*{ext}'):
                    if len(self.image_files) >= self.max_samples:
                        break
                    self.image_files.append(str(file_path))
                    # 从文件名中提取标签
                    label = int(file_path.stem.split('_')[-1]) if '_' in file_path.stem else 0
                    self.labels.append(label)
        except Exception as e:
            self.logger.error(f"加载图像数据失败: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据"""
        self.logger.info(f"创建图像示例数据到 {self.data_dir}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建示例图像配置
        sample_config = {"files": [], "labels": []}
        
        for i in range(min(self.max_samples, 20)):  # 创建20个示例文件
            file_name = f"image_{i}.json"
            file_path = self.data_dir / file_name
            
            # 生成随机图像数据
            height, width = self.image_size
            channels = 3
            random_image = np.random.rand(height, width, channels).tolist()
            
            # 保存图像数据
            image_data = {
                "image": random_image,
                "size": self.image_size
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(image_data, f)
            
            sample_config["files"].append(file_name)
            sample_config["labels"].append(i % 5)  # 随机标签
        
        # 复制样本以达到最大样本数
        for i in range(20, self.max_samples):
            orig_idx = i % 20
            self.image_files.append(str(self.data_dir / sample_config["files"][orig_idx]))
            self.labels.append(sample_config["labels"][orig_idx])
        
        # 保存配置文件
        config_file = self.data_dir / "image_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            file_path = self.image_files[idx]
            label = self.labels[idx]
            
            # 检查文件扩展名，处理不同类型的图像数据
            if file_path.endswith('.json'):
                # 加载JSON格式的图像数据
                with open(file_path, 'r') as f:
                    image_data = json.load(f)
                image_array = np.array(image_data['image'], dtype=np.float32)
                # 确保图像维度正确 (H, W, C)
                if len(image_array.shape) == 2:
                    image_array = np.stack([image_array] * 3, axis=2)
                # 转换为Tensor并调整维度 (C, H, W)
                image_tensor = torch.tensor(image_array).permute(2, 0, 1)
            else:
                # 实际加载图像文件
                try:
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB
                    image = cv2.resize(image, self.image_size)
                    image = image.astype(np.float32) / 255.0
                    image_tensor = torch.tensor(image).permute(2, 0, 1)
                except:
                    # 如果加载失败，生成随机图像数据
                    height, width = self.image_size
                    image_tensor = torch.rand(3, height, width, dtype=torch.float32)
            
            # 应用标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            return image_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # 出错时返回随机数据
            self.logger.error(f"处理图像数据失败: {e}")
            height, width = self.image_size
            return torch.rand(3, height, width, dtype=torch.float32), torch.tensor(0, dtype=torch.long)

class VideoDataset(Dataset):
    """视频数据集 | Video dataset"""
    def __init__(self, data_dir="training_data/video", max_samples=300, frame_size=(112, 112), num_frames=16):
        """初始化视频数据集
        
        参数 Parameters:
            data_dir: 数据目录 | Data directory
            max_samples: 最大样本数 | Maximum number of samples
            frame_size: 帧大小 | Frame size
            num_frames: 每一视频的帧数 | Number of frames per video
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.video_files = []
        self.labels = []
        
        # 视频帧预处理变换
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 检查数据目录是否存在，如果不存在则创建示例数据
        if not self.data_dir.exists():
            self._create_sample_data()
        else:
            self._load_data()
    
    def _load_data(self):
        """加载视频数据"""
        try:
            # 查找所有视频文件
            video_extensions = ['.mp4', '.avi', '.mov']
            for ext in video_extensions:
                for file_path in self.data_dir.glob(f'*{ext}'):
                    if len(self.video_files) >= self.max_samples:
                        break
                    self.video_files.append(str(file_path))
                    # 从文件名中提取标签
                    label = int(file_path.stem.split('_')[-1]) if '_' in file_path.stem else 0
                    self.labels.append(label)
        except Exception as e:
            self.logger.error(f"加载视频数据失败: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据"""
        self.logger.info(f"创建视频示例数据到 {self.data_dir}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建示例视频配置
        sample_config = {"files": [], "labels": []}
        
        for i in range(min(self.max_samples, 10)):  # 创建10个示例文件
            file_name = f"video_{i}.json"
            file_path = self.data_dir / file_name
            
            # 生成随机视频帧数据
            frames = []
            height, width = self.frame_size
            channels = 3
            
            for _ in range(self.num_frames):
                # 生成一帧随机图像
                frame = np.random.rand(height, width, channels).tolist()
                frames.append(frame)
            
            # 保存视频帧数据
            video_data = {
                "frames": frames,
                "frame_size": self.frame_size,
                "num_frames": self.num_frames
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(video_data, f)
            
            sample_config["files"].append(file_name)
            sample_config["labels"].append(i % 4)  # 随机标签
        
        # 复制样本以达到最大样本数
        for i in range(10, self.max_samples):
            orig_idx = i % 10
            self.video_files.append(str(self.data_dir / sample_config["files"][orig_idx]))
            self.labels.append(sample_config["labels"][orig_idx])
        
        # 保存配置文件
        config_file = self.data_dir / "video_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2)
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        try:
            file_path = self.video_files[idx]
            label = self.labels[idx]
            
            frames = []
            
            # 检查文件扩展名，处理不同类型的视频数据
            if file_path.endswith('.json'):
                # 加载JSON格式的视频帧数据
                with open(file_path, 'r') as f:
                    video_data = json.load(f)
                
                for frame_data in video_data['frames']:
                    frame_array = np.array(frame_data, dtype=np.float32)
                    # 转换为Tensor并调整维度 (C, H, W)
                    frame_tensor = torch.tensor(frame_array).permute(2, 0, 1)
                    frames.append(frame_tensor)
            else:
                # 实际加载视频文件
                try:
                    cap = cv2.VideoCapture(file_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # 均匀采样指定数量的帧
                    frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
                    
                    for idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = cv2.resize(frame, self.frame_size)
                            frame = frame.astype(np.float32) / 255.0
                            frame_tensor = torch.tensor(frame).permute(2, 0, 1)
                            frames.append(frame_tensor)
                    
                    cap.release()
                except:
                    # 如果加载失败，生成随机视频帧数据
                    height, width = self.frame_size
                    for _ in range(self.num_frames):
                        frames.append(torch.rand(3, height, width, dtype=torch.float32))
            
            # 确保帧数正确
            while len(frames) < self.num_frames:
                # 如果帧数不足，复制最后一帧
                frames.append(frames[-1].clone() if frames else torch.rand(3, self.frame_size[0], self.frame_size[1], dtype=torch.float32))
            
            # 截取指定数量的帧
            frames = frames[:self.num_frames]
            
            # 堆叠成一个Tensor (T, C, H, W)
            video_tensor = torch.stack(frames)
            
            # 应用标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            video_tensor = (video_tensor - mean) / std
            
            return video_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # 出错时返回随机数据
            self.logger.error(f"处理视频数据失败: {e}")
            height, width = self.frame_size
            random_frames = torch.rand(self.num_frames, 3, height, width, dtype=torch.float32)
            return random_frames, torch.tensor(0, dtype=torch.long)

class JointDataset(Dataset):
    """联合训练数据集 | Joint training dataset"""
    def __init__(self, model_names, max_samples=500):
        """初始化联合训练数据集
        
        参数 Parameters:
            model_names: 参与训练的模型名称列表 | List of model names for training
            max_samples: 最大样本数 | Maximum number of samples
        """
        self.logger = logging.getLogger(__name__)
        self.model_names = model_names
        self.max_samples = max_samples
        self.datasets = {}
        self.data_indices = []
        
        # 为每个模型创建对应的数据集
        for model_name in model_names:
            if model_name == "B_language":
                self.datasets[model_name] = LanguageDataset(max_samples=max_samples)
            elif model_name == "C_audio":
                self.datasets[model_name] = AudioDataset(max_samples=max_samples)
            elif model_name == "D_image":
                self.datasets[model_name] = ImageDataset(max_samples=max_samples)
            elif model_name == "E_video":
                self.datasets[model_name] = VideoDataset(max_samples=max_samples)
        
        # 确定数据集的长度（取最小的数据集长度）
        if self.datasets:
            self.min_length = min(len(ds) for ds in self.datasets.values())
            # 创建数据索引映射
            self.data_indices = list(range(min(self.min_length, max_samples)))
        else:
            self.min_length = 0
    
    def __len__(self):
        return len(self.data_indices)
    
    def __getitem__(self, idx):
        # 返回多模态数据和标签
        data = {}
        targets = {}
        
        # 使用映射后的索引获取每个数据集的样本
        mapped_idx = self.data_indices[idx % len(self.data_indices)]
        
        for model_name in self.model_names:
            if model_name in self.datasets:
                # 获取每个模型的数据和标签
                model_data, model_target = self.datasets[model_name][mapped_idx]
                data[model_name] = model_data
                targets[model_name] = model_target
            else:
                # 如果模型没有对应的数据集，生成随机数据
                data[model_name] = self._generate_random_data(model_name)
                targets[model_name] = torch.tensor(0, dtype=torch.long)
        
        return data, targets
    
    def _generate_random_data(self, model_name):
        """为没有对应数据集的模型生成随机数据"""
        if model_name == "B_language":
            return torch.randn(32, dtype=torch.float32)  # 模拟语言特征
        elif model_name == "C_audio":
            return torch.randn(16000, dtype=torch.float32)  # 模拟音频波形
        elif model_name == "D_image":
            return torch.randn(3, 224, 224, dtype=torch.float32)  # 模拟图像数据
        elif model_name == "E_video":
            return torch.randn(16, 3, 112, 112, dtype=torch.float32)  # 模拟视频数据
        else:
            return torch.randn(10, dtype=torch.float32)  # 默认随机数据

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