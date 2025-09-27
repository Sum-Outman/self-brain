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
增强的训练器，用于训练管理模型的情感分析功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import time
from datetime import datetime
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('A_management_trainer')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ManagementDataset(Dataset):
    """
    管理模型的数据集类
    """
    def __init__(self, data_path, config=None):
        """
        初始化数据集
        
        参数:
            data_path: 数据文件路径或目录
            config: 数据集配置
        """
        self.config = config or {}
        self.data = []
        
        # 加载数据
        self._load_data(data_path)
        
        # 情感标签映射
        self.emotion_mapping = {
            'neutral': 0,
            'joy': 1,
            'sadness': 2,
            'anger': 3,
            'fear': 4,
            'surprise': 5,
            'disgust': 6
        }
    
    def _load_data(self, data_path):
        """
        加载数据
        """
        if os.path.isfile(data_path):
            # 单个文件
            self._load_file(data_path)
        elif os.path.isdir(data_path):
            # 目录，加载所有JSON文件
            for filename in os.listdir(data_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(data_path, filename)
                    self._load_file(file_path)
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_file(self, file_path):
        """
        加载单个文件
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                
            # 处理不同的数据格式
            if isinstance(file_data, list):
                self.data.extend(file_data)
            elif isinstance(file_data, dict) and 'samples' in file_data:
                self.data.extend(file_data['samples'])
            else:
                self.data.append(file_data)
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {str(e)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        """
        sample = self.data[idx]
        
        # 提取特征
        features = self._extract_features(sample)
        
        # 提取标签
        labels = self._extract_labels(sample)
        
        return {
            'features': features,
            'strategy_label': labels['strategy'],
            'emotion_label': labels['emotion'],
            'sub_model_outputs': labels.get('sub_model_outputs', {})
        }
    
    def _extract_features(self, sample):
        """
        从样本中提取特征
        """
        # 根据不同的输入类型处理特征
        if isinstance(sample, dict):
            # 如果样本已经是字典格式，直接返回
            return sample
        elif isinstance(sample, str):
            # 如果是文本，转换为字典
            return {'text': sample}
        else:
            # 其他类型，转换为字符串
            return {'text': str(sample)}
    
    def _extract_labels(self, sample):
        """
        从样本中提取标签
        """
        labels = {
            'strategy': 0,  # 默认策略标签
            'emotion': 0   # 默认情感标签（中性）
        }
        
        if isinstance(sample, dict):
            # 提取策略标签
            if 'strategy_label' in sample:
                labels['strategy'] = int(sample['strategy_label'])
            elif 'strategy' in sample:
                labels['strategy'] = int(sample['strategy'])
            
            # 提取情感标签
            if 'emotion_label' in sample:
                if isinstance(sample['emotion_label'], str):
                    labels['emotion'] = self.emotion_mapping.get(sample['emotion_label'], 0)
                else:
                    labels['emotion'] = int(sample['emotion_label'])
            elif 'emotion' in sample:
                if isinstance(sample['emotion'], str):
                    labels['emotion'] = self.emotion_mapping.get(sample['emotion'], 0)
                else:
                    labels['emotion'] = int(sample['emotion'])
            
            # 提取下属模型输出（如果有）
            if 'sub_model_outputs' in sample:
                labels['sub_model_outputs'] = sample['sub_model_outputs']
        
        return labels

class ModelTrainer:
    """
    模型训练器类，支持策略和情感双任务训练
    """
    def __init__(self, model, config=None):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            config: 训练配置
        """
        self.model = model.to(device)
        
        # 默认配置
        default_config = {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'batch_size': 32,
            'epochs': 50,
            'patience': 5,
            'early_stopping': True,
            'val_split': 0.2,
            'test_split': 0.1,
            'strategy_weight': 0.5,
            'emotion_weight': 0.5,
            'checkpoint_dir': './checkpoints',
            'log_interval': 10,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_patience': 3,
            'grad_clip': 1.0
        }
        
        # 合并配置
        self.config = {**default_config, **(config or {})}
        
        # 创建检查点目录
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # 定义损失函数
        self.strategy_criterion = nn.CrossEntropyLoss()
        self.emotion_criterion = nn.CrossEntropyLoss()
        
        # 定义优化器
        self.optimizer = self._create_optimizer()
        
        # 定义学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 初始化数据加载器
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # 训练历史记录
        self.train_history = {'train': [], 'val': []}
        
        # 最佳模型记录
        self.best_score = float('-inf')
        self.best_epoch = 0
        
    def _create_optimizer(self):
        """
        创建优化器
        """
        if self.config['optimizer'] == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
    
    def _create_scheduler(self):
        """
        创建学习率调度器
        """
        if self.config['scheduler'] == 'ReduceLROnPlateau':
            # PyTorch版本可能不支持verbose参数，移除该参数
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=self.config['scheduler_patience']
            )
        elif self.config['scheduler'] == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif self.config['scheduler'] == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs']
            )
        else:
            logger.warning(f"Unsupported scheduler: {self.config['scheduler']}, using No scheduler")
            return None
    
    def prepare_data(self, data_path):
        """
        准备训练、验证和测试数据
        
        参数:
            data_path: 数据路径
        """
        # 创建数据集
        dataset = ManagementDataset(data_path, self.config)
        
        # 分割数据集
        dataset_size = len(dataset)
        test_size = int(np.floor(self.config['test_split'] * dataset_size))
        val_size = int(np.floor(self.config['val_split'] * (dataset_size - test_size)))
        train_size = dataset_size - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"Data prepared: {train_size} train, {val_size} validation, {test_size} test samples")
    
    def _collate_fn(self, batch):
        """
        自定义数据批处理函数
        """
        features = [item['features'] for item in batch]
        strategy_labels = torch.tensor([item['strategy_label'] for item in batch], dtype=torch.long).to(device)
        emotion_labels = torch.tensor([item['emotion_label'] for item in batch], dtype=torch.long).to(device)
        
        # 处理下属模型输出
        sub_model_outputs = []
        for item in batch:
            sub_model_outputs.append(item['sub_model_outputs'])
        
        return {
            'features': features,
            'strategy_labels': strategy_labels,
            'emotion_labels': emotion_labels,
            'sub_model_outputs': sub_model_outputs
        }
    
    def train_epoch(self):
        """
        训练一个轮次
        """
        self.model.train()
        total_loss = 0
        total_strategy_loss = 0
        total_emotion_loss = 0
        correct_strategy = 0
        correct_emotion = 0
        total_samples = 0
        
        # 使用进度条
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            features = batch['features']
            strategy_labels = batch['strategy_labels']
            emotion_labels = batch['emotion_labels']
            sub_model_outputs = batch['sub_model_outputs']
            
            # 重置梯度
            self.optimizer.zero_grad()
            
            # 批量处理每个样本
            batch_loss = 0
            batch_strategy_loss = 0
            batch_emotion_loss = 0
            batch_correct_strategy = 0
            batch_correct_emotion = 0
            
            for i in range(len(features)):
                # 处理单个样本
                with torch.set_grad_enabled(True):
                    # 前向传播
                    strategy_probs, emotion_probs = self.model(features[i], sub_model_outputs[i])
                    
                    # 计算策略损失
                    if len(strategy_probs.shape) == 1:
                        strategy_probs = strategy_probs.unsqueeze(0)
                    strategy_loss = self.strategy_criterion(strategy_probs, strategy_labels[i].unsqueeze(0))
                    
                    # 计算情感损失
                    if len(emotion_probs.shape) == 1:
                        emotion_probs = emotion_probs.unsqueeze(0)
                    emotion_loss = self.emotion_criterion(emotion_probs, emotion_labels[i].unsqueeze(0))
                    
                    # 组合损失
                    loss = self.config['strategy_weight'] * strategy_loss + self.config['emotion_weight'] * emotion_loss
                    
                    # 反向传播
                    loss.backward(retain_graph=True)
                    
                    # 累加损失
                    batch_loss += loss.item()
                    batch_strategy_loss += strategy_loss.item()
                    batch_emotion_loss += emotion_loss.item()
                    
                    # 统计准确率
                    _, predicted_strategy = torch.max(strategy_probs, 1)
                    _, predicted_emotion = torch.max(emotion_probs, 1)
                    
                    if predicted_strategy == strategy_labels[i]:
                        batch_correct_strategy += 1
                    if predicted_emotion == emotion_labels[i]:
                        batch_correct_emotion += 1
            
            # 梯度裁剪
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            # 更新参数
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += batch_loss
            total_strategy_loss += batch_strategy_loss
            total_emotion_loss += batch_emotion_loss
            correct_strategy += batch_correct_strategy
            correct_emotion += batch_correct_emotion
            total_samples += len(features)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{batch_loss/len(features):.4f}",
                'strat_acc': f"{batch_correct_strategy/len(features):.4f}",
                'emo_acc': f"{batch_correct_emotion/len(features):.4f}"
            })
        
        # 计算平均损失和准确率
        avg_loss = total_loss / total_samples
        avg_strategy_loss = total_strategy_loss / total_samples
        avg_emotion_loss = total_emotion_loss / total_samples
        strategy_accuracy = correct_strategy / total_samples
        emotion_accuracy = correct_emotion / total_samples
        
        return {
            'loss': avg_loss,
            'strategy_loss': avg_strategy_loss,
            'emotion_loss': avg_emotion_loss,
            'strategy_acc': strategy_accuracy,
            'emotion_acc': emotion_accuracy
        }
    
    def validate(self, loader=None):
        """
        验证模型性能
        
        参数:
            loader: 数据加载器，如果不提供则使用验证集加载器
        """
        self.model.eval()
        total_loss = 0
        total_strategy_loss = 0
        total_emotion_loss = 0
        correct_strategy = 0
        correct_emotion = 0
        total_samples = 0
        
        # 使用指定的数据加载器或默认的验证集加载器
        loader = loader or self.val_loader
        
        with torch.no_grad():
            # 使用进度条
            progress_bar = tqdm(loader, desc="Validation", leave=False)
            
            for batch in progress_bar:
                features = batch['features']
                strategy_labels = batch['strategy_labels']
                emotion_labels = batch['emotion_labels']
                sub_model_outputs = batch['sub_model_outputs']
                
                # 批量处理每个样本
                batch_loss = 0
                batch_strategy_loss = 0
                batch_emotion_loss = 0
                batch_correct_strategy = 0
                batch_correct_emotion = 0
                
                for i in range(len(features)):
                    # 前向传播
                    strategy_probs, emotion_probs = self.model(features[i], sub_model_outputs[i])
                    
                    # 计算策略损失
                    if len(strategy_probs.shape) == 1:
                        strategy_probs = strategy_probs.unsqueeze(0)
                    strategy_loss = self.strategy_criterion(strategy_probs, strategy_labels[i].unsqueeze(0))
                    
                    # 计算情感损失
                    if len(emotion_probs.shape) == 1:
                        emotion_probs = emotion_probs.unsqueeze(0)
                    emotion_loss = self.emotion_criterion(emotion_probs, emotion_labels[i].unsqueeze(0))
                    
                    # 组合损失
                    loss = self.config['strategy_weight'] * strategy_loss + self.config['emotion_weight'] * emotion_loss
                    
                    # 累加损失
                    batch_loss += loss.item()
                    batch_strategy_loss += strategy_loss.item()
                    batch_emotion_loss += emotion_loss.item()
                    
                    # 统计准确率
                    _, predicted_strategy = torch.max(strategy_probs, 1)
                    _, predicted_emotion = torch.max(emotion_probs, 1)
                    
                    if predicted_strategy == strategy_labels[i]:
                        batch_correct_strategy += 1
                    if predicted_emotion == emotion_labels[i]:
                        batch_correct_emotion += 1
                
                # 更新统计信息
                total_loss += batch_loss
                total_strategy_loss += batch_strategy_loss
                total_emotion_loss += batch_emotion_loss
                correct_strategy += batch_correct_strategy
                correct_emotion += batch_correct_emotion
                total_samples += len(features)
        
        # 计算平均损失和准确率
        avg_loss = total_loss / total_samples
        avg_strategy_loss = total_strategy_loss / total_samples
        avg_emotion_loss = total_emotion_loss / total_samples
        strategy_accuracy = correct_strategy / total_samples
        emotion_accuracy = correct_emotion / total_samples
        
        # 计算综合分数（用于早停和模型选择）
        combined_score = (strategy_accuracy + emotion_accuracy) / 2
        
        return {
            'loss': avg_loss,
            'strategy_loss': avg_strategy_loss,
            'emotion_loss': avg_emotion_loss,
            'strategy_acc': strategy_accuracy,
            'emotion_acc': emotion_accuracy,
            'combined_score': combined_score
        }
    
    def train(self):
        """
        主训练循环
        """
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data loaders not initialized. Call prepare_data() first.")
        
        # 初始化早停计数器
        no_improvement_count = 0
        
        logger.info(f"Starting training with {self.config['epochs']} epochs")
        
        # 主训练循环
        for epoch in range(1, self.config['epochs'] + 1):
            start_time = time.time()
            
            # 训练一个轮次
            train_metrics = self.train_epoch()
            
            # 验证模型
            val_metrics = self.validate()
            
            # 更新学习率调度器
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['combined_score'])
                else:
                    self.scheduler.step()
            
            # 记录训练历史
            self.train_history['train'].append(train_metrics)
            self.train_history['val'].append(val_metrics)
            
            # 计算训练时间
            elapsed_time = time.time() - start_time
            
            # 打印训练进度
            print(f"\nEpoch {epoch}/{self.config['epochs']} | Time: {elapsed_time:.2f}s")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Strategy Acc: {train_metrics['strategy_acc']:.4f}, Emotion Acc: {train_metrics['emotion_acc']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Strategy Acc: {val_metrics['strategy_acc']:.4f}, Emotion Acc: {val_metrics['emotion_acc']:.4f}")
            print(f"Combined Score: {val_metrics['combined_score']:.4f}")
            
            # 检查是否是最佳模型
            if val_metrics['combined_score'] > self.best_score:
                self.best_score = val_metrics['combined_score']
                self.best_epoch = epoch
                no_improvement_count = 0
                
                # 保存最佳模型
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model saved with score: {self.best_score:.4f}")
            else:
                no_improvement_count += 1
            
            # 早停检查
            if self.config['early_stopping'] and no_improvement_count >= self.config['patience']:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # 加载最佳模型
        self.load_checkpoint(is_best=True)
        
        # 在测试集上评估最佳模型
        if self.test_loader is not None:
            test_metrics = self.validate(self.test_loader)
            logger.info(f"\nFinal Test Results:")
            logger.info(f"Test Loss: {test_metrics['loss']:.4f}, Strategy Acc: {test_metrics['strategy_acc']:.4f}, Emotion Acc: {test_metrics['emotion_acc']:.4f}")
            logger.info(f"Test Combined Score: {test_metrics['combined_score']:.4f}")
        
        logger.info(f"Training completed! Best epoch: {self.best_epoch}, Best score: {self.best_score:.4f}")
        
        return self.train_history
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        保存模型检查点
        
        参数:
            epoch: 当前轮次
            is_best: 是否是最佳模型
        """
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_score': self.best_score,
            'train_history': self.train_history
        }
        
        # 保存当前轮次的检查点
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，保存为best_model.pth
        if is_best:
            best_checkpoint_path = os.path.join(self.config['checkpoint_dir'], "best_model.pth")
            torch.save(checkpoint, best_checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path=None, is_best=False):
        """
        加载模型检查点
        
        参数:
            checkpoint_path: 检查点文件路径
            is_best: 是否加载最佳模型
        """
        # 如果没有指定路径且要求加载最佳模型，使用默认的最佳模型路径
        if checkpoint_path is None and is_best:
            checkpoint_path = os.path.join(self.config['checkpoint_dir'], "best_model.pth")
        
        # 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return False
        
        try:
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 加载模型状态字典
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 如果有优化器状态字典，加载它
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 如果有训练历史，加载它
            if 'train_history' in checkpoint:
                self.train_history = checkpoint['train_history']
            
            # 如果有最佳分数，更新它
            if 'best_score' in checkpoint:
                self.best_score = checkpoint['best_score']
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
            return False
    
    def evaluate(self, loader=None):
        """
        评估模型性能（简化版）
        
        参数:
            loader: 数据加载器
        
        返回:
            损失、策略准确率、情感准确率
        """
        metrics = self.validate(loader)
        
        return metrics['loss'], metrics['strategy_acc'], metrics['emotion_acc']

# 工具函数
def create_trainer(model, config_path=None):
    """
    创建训练器实例
    
    参数:
        model: 要训练的模型
        config_path: 配置文件路径
    
    返回:
        ModelTrainer实例
    """
    config = {}
    
    # 如果提供了配置文件路径，加载配置
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Trainer config loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load trainer config from {config_path}: {str(e)}")
    
    # 创建训练器实例
    trainer = ModelTrainer(model, config)
    
    return trainer

def evaluate_model(model, data_path, batch_size=32):
    """
    评估模型性能
    
    参数:
        model: 要评估的模型
        data_path: 评估数据路径
        batch_size: 批大小
    
    返回:
        评估指标字典
    """
    # 创建数据集和数据加载器
    dataset = ManagementDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            'features': [item['features'] for item in batch],
            'strategy_labels': torch.tensor([item['strategy_label'] for item in batch], dtype=torch.long).to(device),
            'emotion_labels': torch.tensor([item['emotion_label'] for item in batch], dtype=torch.long).to(device),
            'sub_model_outputs': [item['sub_model_outputs'] for item in batch]
        }
    )
    
    # 创建临时训练器用于评估
    trainer = ModelTrainer(model, {'batch_size': batch_size})
    
    # 评估模型
    loss, strategy_acc, emotion_acc = trainer.evaluate(loader)
    
    # 计算综合准确率
    total_acc = (strategy_acc + emotion_acc) / 2
    
    return {
        'loss': loss,
        'strategy_accuracy': strategy_acc,
        'emotion_accuracy': emotion_acc,
        'total_accuracy': total_acc
    }

def generate_training_report(train_history, model_config, report_path):
    """
    生成训练报告
    
    参数:
        train_history: 训练历史记录
        model_config: 模型配置
        report_path: 报告保存路径
    """
    # 准备报告数据
    report = {
        'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_config': model_config,
        'training_history': train_history,
        'summary': {
            'epochs_trained': len(train_history['train']),
            'final_train_loss': train_history['train'][-1]['loss'] if train_history['train'] else 0,
            'final_val_loss': train_history['val'][-1]['loss'] if train_history['val'] else 0,
            'final_train_strategy_acc': train_history['train'][-1]['strategy_acc'] if train_history['train'] else 0,
            'final_val_strategy_acc': train_history['val'][-1]['strategy_acc'] if train_history['val'] else 0,
            'final_train_emotion_acc': train_history['train'][-1]['emotion_acc'] if train_history['train'] else 0,
            'final_val_emotion_acc': train_history['val'][-1]['emotion_acc'] if train_history['val'] else 0
        }
    }
    
    # 保存报告
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training report saved to {report_path}")
    
    return report

# 主函数演示
if __name__ == "__main__":
    from enhanced_manager import ManagementModel
    
    # 创建模型
    model = ManagementModel()
    
    # 创建训练器
    trainer = ModelTrainer(model)
    
    # 注意：这里只是演示，实际使用时需要提供真实的数据路径
    print("This is a demo of the enhanced trainer for A_management model.")
    print("In a real scenario, you would call trainer.prepare_data() with a valid data path.")
    print("Then call trainer.train() to start the training process.")