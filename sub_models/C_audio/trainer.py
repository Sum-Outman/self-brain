# -*- coding: utf-8 -*-
"""
音频处理模型训练器 - Audio Processing Model Trainer
实现模型训练、评估、优化和自主学习功能
Implements model training, evaluation, optimization, and self-learning functions

Apache License 2.0
Copyright 2025 The AGI Brain System Authors
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 配置日志 | Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_trainer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AudioModelTrainer")

class AudioDataset(Dataset):
    """
    音频数据集类 - Audio Dataset Class
    支持多种音频任务的数据加载和预处理
    Supports data loading and preprocessing for various audio tasks
    """
    
    def __init__(self, audio_files: List[str], labels: List[int] = None, 
                 task_type: str = "classification", sample_rate: int = 16000,
                 max_length: int = 16000, augment: bool = True):
        """
        初始化音频数据集
        Initialize audio dataset
        
        参数 Parameters:
        audio_files: 音频文件路径列表 | List of audio file paths
        labels: 标签列表 | List of labels
        task_type: 任务类型 | Task type
        sample_rate: 采样率 | Sample rate
        max_length: 最大长度 | Maximum length
        augment: 是否数据增强 | Whether to augment data
        """
        self.audio_files = audio_files
        self.labels = labels
        self.task_type = task_type
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.augment = augment
        self.has_labels = labels is not None and len(labels) == len(audio_files)
        
        logger.info(f"音频数据集初始化: {len(audio_files)} 个样本 | Audio dataset initialized: {len(audio_files)} samples")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        try:
            # 加载音频文件 | Load audio file
            audio_path = self.audio_files[idx]
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 预处理音频 | Preprocess audio
            audio = self._preprocess_audio(audio)
            
            # 数据增强 | Data augmentation
            if self.augment:
                audio = self._augment_audio(audio)
            
            # 转换为张量 | Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            if self.has_labels:
                label = self.labels[idx]
                if self.task_type == "classification":
                    label_tensor = torch.tensor(label, dtype=torch.long)
                else:
                    label_tensor = torch.tensor(label, dtype=torch.float)
                return audio_tensor, label_tensor
            else:
                return audio_tensor
                
        except Exception as e:
            logger.error(f"加载音频文件失败: {audio_path}, 错误: {e} | Failed to load audio file: {audio_path}, error: {e}")
            # 返回空数据 | Return empty data
            empty_audio = torch.zeros(self.max_length)
            if self.has_labels:
                return empty_audio, torch.tensor(-1, dtype=torch.long)
            else:
                return empty_audio
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        预处理音频数据
        Preprocess audio data
        
        参数 Parameters:
        audio: 原始音频数据 | Raw audio data
        
        返回 Returns:
        预处理后的音频 | Preprocessed audio
        """
        # 标准化音频 | Normalize audio
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        # 裁剪或填充到固定长度 | Crop or pad to fixed length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            padding = self.max_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        音频数据增强
        Audio data augmentation
        
        参数 Parameters:
        audio: 原始音频 | Original audio
        
        返回 Returns:
        增强后的音频 | Augmented audio
        """
        augmented_audio = audio.copy()
        
        # 随机选择增强技术 | Randomly select augmentation techniques
        augmentation_type = np.random.choice([
            'none', 'noise', 'pitch_shift', 'time_stretch', 'volume_change'
        ], p=[0.3, 0.2, 0.2, 0.15, 0.15])
        
        try:
            if augmentation_type == 'noise' and np.random.random() < 0.5:
                # 添加随机噪音 | Add random noise
                noise = np.random.normal(0, 0.01, len(audio))
                augmented_audio = audio + noise
            
            elif augmentation_type == 'pitch_shift' and np.random.random() < 0.5:
                # 音高变换 | Pitch shift
                n_steps = np.random.uniform(-2, 2)
                augmented_audio = librosa.effects.pitch_shift(
                    audio, sr=self.sample_rate, n_steps=n_steps
                )
                # 确保长度一致 | Ensure consistent length
                if len(augmented_audio) > len(audio):
                    augmented_audio = augmented_audio[:len(audio)]
                else:
                    padding = len(audio) - len(augmented_audio)
                    augmented_audio = np.pad(augmented_audio, (0, padding), mode='constant')
            
            elif augmentation_type == 'time_stretch' and np.random.random() < 0.5:
                # 时间拉伸 | Time stretch
                rate = np.random.uniform(0.8, 1.2)
                augmented_audio = librosa.effects.time_stretch(audio, rate=rate)
                # 确保长度一致 | Ensure consistent length
                if len(augmented_audio) > len(audio):
                    augmented_audio = augmented_audio[:len(audio)]
                else:
                    padding = len(audio) - len(augmented_audio)
                    augmented_audio = np.pad(augmented_audio, (0, padding), mode='constant')
            
            elif augmentation_type == 'volume_change' and np.random.random() < 0.5:
                # 音量变化 | Volume change
                gain = np.random.uniform(0.5, 1.5)
                augmented_audio = audio * gain
        
        except Exception as e:
            logger.warning(f"数据增强失败: {e}, 使用原始音频 | Data augmentation failed: {e}, using original audio")
            augmented_audio = audio
        
        return augmented_audio

class ModelTrainer:
    """
    模型训练器类 - Model Trainer Class
    负责音频模型的训练、评估和优化
    Responsible for audio model training, evaluation, and optimization
    """
    
    def __init__(self, model: nn.Module = None, device: str = "auto"):
        """
        初始化训练器
        Initialize trainer
        
        参数 Parameters:
        model: 要训练的模型 | Model to train
        device: 训练设备 | Training device
        """
        self.model = model
        self.device = self._setup_device(device)
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_history = []
        self.best_model_state = None
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        
        logger.info(f"模型训练器初始化完成 | Model trainer initialized")
        logger.info(f"使用设备: {self.device} | Using device: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """
        设置训练设备
        Setup training device
        
        参数 Parameters:
        device: 设备选择 | Device selection
        
        返回 Returns:
        torch设备 | torch device
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("检测到CUDA，使用GPU训练 | CUDA detected, using GPU for training")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("检测到MPS，使用Apple Silicon GPU训练 | MPS detected, using Apple Silicon GPU for training")
            else:
                device = torch.device("cpu")
                logger.info("使用CPU训练 | Using CPU for training")
        else:
            device = torch.device(device)
        
        return device
    
    def prepare_training(self, config: Dict):
        """
        准备训练配置
        Prepare training configuration
        
        参数 Parameters:
        config: 训练配置字典 | Training configuration dictionary
        """
        if self.model is None:
            logger.error("未设置模型，无法准备训练 | Model not set, cannot prepare training")
            return False
        
        # 设置优化器 | Setup optimizer
        optimizer_name = config.get("optimizer", "adamw").lower()
        learning_rate = config.get("learning_rate", 1e-4)
        weight_decay = config.get("weight_decay", 0.01)
        
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=learning_rate, 
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            logger.warning(f"不支持的优化器: {optimizer_name}, 使用默认AdamW | Unsupported optimizer: {optimizer_name}, using default AdamW")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        
        # 设置学习率调度器 | Setup learning rate scheduler
        scheduler_type = config.get("scheduler", "cosine").lower()
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=config.get("epochs", 20)
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=config.get("step_size", 5),
                gamma=config.get("gamma", 0.1)
            )
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',
                patience=config.get("patience", 3),
                factor=config.get("factor", 0.5)
            )
        else:
            logger.warning(f"不支持的调度器: {scheduler_type}, 不使用调度器 | Unsupported scheduler: {scheduler_type}, not using scheduler")
            self.scheduler = None
        
        # 设置损失函数 | Setup loss function
        task_type = config.get("task_type", "classification")
        if task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif task_type == "regression":
            self.criterion = nn.MSELoss()
        else:
            logger.warning(f"不支持的任务类型: {task_type}, 使用默认分类损失 | Unsupported task type: {task_type}, using default classification loss")
            self.criterion = nn.CrossEntropyLoss()
        
        # 将模型移动到设备 | Move model to device
        self.model.to(self.device)
        
        logger.info(f"训练准备完成 | Training preparation completed")
        logger.info(f"优化器: {optimizer_name} | Optimizer: {optimizer_name}")
        logger.info(f"学习率: {learning_rate} | Learning rate: {learning_rate}")
        logger.info(f"任务类型: {task_type} | Task type: {task_type}")
        
        return True
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, 
              epochs: int = 10, early_stopping: bool = True, patience: int = 5,
              checkpoint_dir: str = "checkpoints") -> Dict:
        """
        训练模型
        Train model
        
        参数 Parameters:
        train_loader: 训练数据加载器 | Training data loader
        val_loader: 验证数据加载器 | Validation data loader
        epochs: 训练轮数 | Training epochs
        early_stopping: 是否早停 | Whether to use early stopping
        patience: 早停耐心值 | Early stopping patience
        checkpoint_dir: 检查点目录 | Checkpoint directory
        
        返回 Returns:
        训练结果字典 | Training results dictionary
        """
        if self.optimizer is None or self.criterion is None:
            logger.error("训练器未准备就绪，请先调用prepare_training | Trainer not ready, please call prepare_training first")
            return {"status": "error", "message": "Trainer not ready"}
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        train_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        logger.info(f"开始训练，共 {epochs} 轮 | Starting training, total {epochs} epochs")
        
        for epoch in range(epochs):
            # 训练阶段 | Training phase
            train_loss, train_metrics = self._train_epoch(train_loader, epoch)
            
            # 验证阶段 | Validation phase
            val_loss, val_metrics = 0.0, {}
            if val_loader is not None:
                val_loss, val_metrics = self._validate_epoch(val_loader, epoch)
            
            # 更新学习率 | Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loader is not None else train_loss)
                else:
                    self.scheduler.step()
            
            # 记录训练历史 | Record training history
            epoch_history = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_metrics": train_metrics,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "timestamp": datetime.now().isoformat()
            }
            train_history.append(epoch_history)
            
            # 早停检查 | Early stopping check
            if val_loader is not None and early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch + 1
                    
                    # 保存最佳模型 | Save best model
                    self.best_model_state = self.model.state_dict().copy()
                    self.best_accuracy = val_metrics.get("accuracy", 0) if val_metrics else 0
                    self.best_loss = val_loss
                    
                    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss,
                        'metrics': val_metrics
                    }, checkpoint_path)
                    
                    logger.info(f"保存最佳模型: {checkpoint_path} | Saved best model: {checkpoint_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"早停触发，在第 {epoch+1} 轮停止训练 | Early stopping triggered, stopping training at epoch {epoch+1}")
                        break
            
            # 每5轮保存检查点 | Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss if val_loader is not None else train_loss,
                    'metrics': val_metrics if val_loader is not None else train_metrics
                }, checkpoint_path)
                logger.info(f"保存检查点: {checkpoint_path} | Saved checkpoint: {checkpoint_path}")
        
        self.train_history = train_history
        
        # 恢复最佳模型 | Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"恢复最佳模型 (第 {best_epoch} 轮) | Restored best model (epoch {best_epoch})")
        
        # 保存最终模型 | Save final model
        final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        logger.info(f"保存最终模型: {final_model_path} | Saved final model: {final_model_path}")
        
        # 生成训练报告 | Generate training report
        report = self._generate_training_report(train_history, best_epoch)
        
        logger.info("训练完成！ | Training completed!")
        return report
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """
        训练一个epoch
        Train one epoch
        
        参数 Parameters:
        train_loader: 训练数据加载器 | Training data loader
        epoch: 当前epoch | Current epoch
        
        返回 Returns:
        平均损失和指标 | Average loss and metrics
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            # 移动到设备 | Move to device
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播 | Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # 计算损失 | Calculate loss
            loss = self.criterion(outputs, labels)
            
            # 反向传播 | Backward pass
            loss.backward()
            
            # 梯度裁剪 | Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数 | Update parameters
            self.optimizer.step()
            
            # 记录损失 | Record loss
            total_loss += loss.item()
            
            # 收集预测和标签 | Collect predictions and labels
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 每100个batch打印进度 | Print progress every 100 batches
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # 计算平均损失和指标 | Calculate average loss and metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(all_preds, all_labels)
        
        logger.info(f"Epoch {epoch+1} | 训练损失: {avg_loss:.4f} | 准确率: {metrics.get('accuracy', 0):.4f} | "
                   f"Train Loss: {avg_loss:.4f} | Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """
        验证一个epoch
        Validate one epoch
        
        参数 Parameters:
        val_loader: 验证数据加载器 | Validation data loader
        epoch: 当前epoch | Current epoch
        
        返回 Returns:
        平均损失和指标 | Average loss and metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                # 移动到设备 | Move to device
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播 | Forward pass
                outputs = self.model(data)
                
                # 计算损失 | Calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # 收集预测和标签 | Collect predictions and labels
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算平均损失和指标 | Calculate average loss and metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(all_preds, all_labels)
        
        logger.info(f"Epoch {epoch+1} | 验证损失: {avg_loss:.4f} | 准确率: {metrics.get('accuracy', 0):.4f} | "
                   f"Val Loss: {avg_loss:.4f} | Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, predictions: List, labels: List) -> Dict:
        """
        计算评估指标
        Calculate evaluation metrics
        
        参数 Parameters:
        predictions: 预测列表 | List of predictions
        labels: 标签列表 | List of labels
        
        返回 Returns:
        指标字典 | Metrics dictionary
        """
        if len(predictions) == 0 or len(labels) == 0:
            return {}
        
        try:
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        except Exception as e:
            logger.error(f"计算指标失败: {e} | Failed to calculate metrics: {e}")
            return {}
    
    def _generate_training_report(self, train_history: List[Dict], best_epoch: int) -> Dict:
        """
        生成训练报告
        Generate training report
        
        参数 Parameters:
        train_history: 训练历史 | Training history
        best_epoch: 最佳epoch | Best epoch
        
        返回 Returns:
        训练报告字典 | Training report dictionary
        """
        if not train_history:
            return {"status": "error", "message": "无训练历史 | No training history"}
        
        best_result = None
        for history in train_history:
            if history["epoch"] == best_epoch:
                best_result = history
                break
        
        if best_result is None:
            best_result = train_history[-1]
        
        report = {
            "status": "success",
            "total_epochs": len(train_history),
            "best_epoch": best_epoch,
            "best_accuracy": best_result["val_metrics"].get("accuracy", 0) if "val_metrics" in best_result else 0,
            "best_loss": best_result["val_loss"] if "val_loss" in best_result else best_result["train_loss"],
            "final_accuracy": train_history[-1]["val_metrics"].get("accuracy", 0) if "val_metrics" in train_history[-1] else 0,
            "final_loss": train_history[-1]["val_loss"] if "val_loss" in train_history[-1] else train_history[-1]["train_loss"],
            "training_time": f"{(datetime.fromisoformat(train_history[-1]['timestamp']) - datetime.fromisoformat(train_history[0]['timestamp'])).total_seconds():.2f}秒",
            "hardware_used": str(self.device),
            "detailed_history": train_history
        }
        
        return report
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        评估模型性能
        Evaluate model performance
        
        参数 Parameters:
        test_loader: 测试数据加载器 | Test data loader
        
        返回 Returns:
        评估结果字典 | Evaluation results dictionary
        """
        if self.model is None:
            return {"status": "error", "message": "未设置模型 | Model not set"}
        
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                # 移动到设备 | Move to device
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播 | Forward pass
                outputs = self.model(data)
                
                # 计算损失 | Calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # 收集预测和标签 | Collect predictions and labels
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算平均损失和指标 | Calculate average loss and metrics
        avg_loss = total_loss / len(test_loader)
        metrics = self._calculate_metrics(all_preds, all_labels)
        
        logger.info(f"评估完成 | Evaluation completed")
        logger.info(f"测试损失: {avg_loss:.4f} | 准确率: {metrics.get('accuracy', 0):.4f} | "
                   f"Test Loss: {avg_loss:.4f} | Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return {
            "status": "success",
            "loss": avg_loss,
            "metrics": metrics,
            "predictions": all_preds,
            "labels": all_labels
        }
    
    def optimize_hyperparameters(self, train_loader: DataLoader, val_loader: DataLoader,
                                param_grid: Dict, n_trials: int = 10) -> Dict:
        """
        超参数优化
        Hyperparameter optimization
        
        参数 Parameters:
        train_loader: 训练数据加载器 | Training data loader
        val_loader: 验证数据加载器 | Validation data loader
        param_grid: 参数网格 | Parameter grid
        n_trials: 试验次数 | Number of trials
        
        返回 Returns:
        优化结果字典 | Optimization results dictionary
        """
        best_params = None
        best_score = float('inf')
        results = []
        
        logger.info(f"开始超参数优化，共 {n_trials} 次试验 | Starting hyperparameter optimization, total {n_trials} trials")
        
        for trial in range(n_trials):
            # 随机选择参数 | Randomly select parameters
            trial_params = {}
            for param_name, param_values in param_grid.items():
                trial_params[param_name] = np.random.choice(param_values)
            
            logger.info(f"试验 {trial+1}/{n_trials} | 参数: {trial_params} | Trial {trial+1}/{n_trials} | Params: {trial_params}")
            
            # 使用新参数重新准备训练 | Re-prepare training with new parameters
            self.prepare_training(trial_params)
            
            # 训练一个简化的epoch进行评估 | Train a simplified epoch for evaluation
            self.model.train()
            total_loss = 0.0
            
            for data, labels in train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # 在验证集上评估 | Evaluate on validation set
            val_loss, val_metrics = self._validate_epoch(val_loader, 0)
            
            # 记录结果 | Record results
            trial_result = {
                "trial": trial + 1,
                "params": trial_params,
                "train_loss": total_loss / len(train_loader),
                "val_loss": val_loss,
                "val_accuracy": val_metrics.get("accuracy", 0)
            }
            results.append(trial_result)
            
            # 更新最佳参数 | Update best parameters
            if val_loss < best_score:
                best_score = val_loss
                best_params = trial_params
                self.best_model_state = self.model.state_dict().copy()
        
        # 恢复最佳模型 | Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        logger.info(f"超参数优化完成 | Hyperparameter optimization completed")
        logger.info(f"最佳参数: {best_params} | Best params: {best_params}")
        logger.info(f"最佳验证损失: {best_score:.4f} | Best validation loss: {best_score:.4f}")
        
        return {
            "status": "success",
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results
        }
    
    def save_training_history(self, file_path: str):
        """
        保存训练历史
        Save training history
        
        参数 Parameters:
        file_path: 文件路径 | File path
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.train_history, f, indent=2, ensure_ascii=False)
            logger.info(f"训练历史已保存到: {file_path} | Training history saved to: {file_path}")
        except Exception as e:
            logger.error(f"保存训练历史失败: {e} | Failed to save training history: {e}")
    
    def load_training_history(self, file_path: str):
        """
        加载训练历史
        Load training history
        
        参数 Parameters:
        file_path: 文件路径 | File path
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.train_history = json.load(f)
                logger.info(f"训练历史已从 {file_path} 加载 | Training history loaded from {file_path}")
            else:
                logger.warning(f"训练历史文件不存在: {file_path} | Training history file does not exist: {file_path}")
        except Exception as e:
            logger.error(f"加载训练历史失败: {e} | Failed to load training history: {e}")

# 工具函数 | Utility functions
def create_data_loaders(audio_files: List[str], labels: List[int] = None,
                       batch_size: int = 32, test_size: float = 0.2,
                       val_size: float = 0.2, random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    Create data loaders
    
    参数 Parameters:
    audio_files: 音频文件列表 | List of audio files
    labels: 标签列表 | List of labels
    batch_size: 批次大小 | Batch size
    test_size: 测试集比例 | Test set ratio
    val_size: 验证集比例 | Validation set ratio
    random_state: 随机种子 | Random seed
    
    返回 Returns:
    训练、验证、测试数据加载器 | Train, validation, test data loaders
    """
    if labels is None or len(labels) != len(audio_files):
        # 创建伪标签用于无监督学习 | Create pseudo labels for unsupervised learning
        labels = [0] * len(audio_files)
    
    # 划分数据集 | Split dataset
    train_files, test_files, train_labels, test_labels = train_test_split(
        audio_files, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=val_size, random_state=random_state, stratify=train_labels
    )
    
    # 创建数据集 | Create datasets
    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)
    test_dataset = AudioDataset(test_files, test_labels)
    
    # 创建数据加载器 | Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"数据加载器创建完成 | Data loaders created")
    logger.info(f"训练集: {len(train_files)} 样本 | Train set: {len(train_files)} samples")
    logger.info(f"验证集: {len(val_files)} 样本 | Validation set: {len(val_files)} samples")
    logger.info(f"测试集: {len(test_files)} 样本 | Test set: {len(test_files)} samples")
    
    return train_loader, val_loader, test_loader

def find_audio_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    查找音频文件
    Find audio files
    
    参数 Parameters:
    directory: 目录路径 | Directory path
    extensions: 文件扩展名列表 | List of file extensions
    
    返回 Returns:
    音频文件路径列表 | List of audio file paths
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    logger.info(f"在 {directory} 中找到 {len(audio_files)} 个音频文件 | Found {len(audio_files)} audio files in {directory}")
    return audio_files

# 测试函数 | Test function
def test_trainer():
    """测试训练器 | Test trainer"""
    # 创建模拟数据 | Create mock data
    sample_rate = 16000
    duration = 1.0
    num_samples = 100
    
    audio_files = []
    labels = []
    
    # 创建临时目录和文件 | Create temporary directory and files
    temp_dir = "temp_test"
    os.makedirs(temp_dir, exist_ok=True)
    
    for i in range(num_samples):
        # 生成测试音频 | Generate test audio
        t = np.linspace(0, duration, int(sample_rate * duration))
        freq = 440 + i * 10  # 不同频率 | Different frequencies
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # 保存音频文件 | Save audio file
        file_path = os.path.join(temp_dir, f"test_{i}.wav")
        sf.write(file_path, audio, sample_rate)
        audio_files.append(file_path)
        labels.append(i % 5)  # 5个类别 | 5 classes
    
    # 创建数据加载器 | Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(audio_files, labels, batch_size=8)
    
    # 创建简单模型 | Create simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_size=16000, num_classes=5):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    model = SimpleModel()
    trainer = ModelTrainer(model)
    
    # 准备训练 | Prepare training
    config = {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "task_type": "classification",
        "epochs": 3
    }
    trainer.prepare_training(config)
    
    # 训练模型 | Train model
    report = trainer.train(train_loader, val_loader, epochs=3, early_stopping=False)
    print(f"训练报告: {report}")
    
    # 评估模型 | Evaluate model
    eval_result = trainer.evaluate(test_loader)
    print(f"评估结果: {eval_result}")
    
    # 清理临时文件 | Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    print("测试完成！ | Test completed!")

if __name__ == "__main__":
    test_trainer()
