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

# 知识库专家模型训练程序
# Knowledge Expert Model Training Program

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from .knowledge_model import KnowledgeModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("I_knowledge_Trainer")

class KnowledgeDataset(Dataset):
    """知识库数据集类 | Knowledge Dataset Class"""
    def __init__(self, data_dir: str, sequence_length: int = 10):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.data_files = []
        self.dataset_info = {}
        self.domain_map = {}
        
        # 加载数据集信息
        self._load_dataset_info()
    
    def _load_dataset_info(self):
        """加载数据集信息 | Load dataset information"""
        info_file = os.path.join(self.data_dir, "dataset_info.json")
        
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    self.dataset_info = json.load(f)
                    self.data_files = self.dataset_info.get("data_files", [])
                    self.domain_map = self.dataset_info.get("domain_map", {})
            except Exception as e:
                logger.error(f"加载数据集信息错误: {str(e)}")
        
        # 如果没有数据集信息，扫描目录中的数据文件
        if not self.data_files:
            self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            # 如果没有数据文件，创建一些模拟数据信息
            if not self.data_files:
                self.data_files = [f"knowledge_data_{i}.csv" for i in range(8)]
                self.domain_map = {
                    0: "physics",
                    1: "math",
                    2: "chemistry",
                    3: "biology",
                    4: "history",
                    5: "geography",
                    6: "literature",
                    7: "art",
                    8: "medicine",
                    9: "law",
                    10: "sociology",
                    11: "humanities",
                    12: "psychology",
                    13: "economics",
                    14: "management",
                    15: "mechanical_engineering",
                    16: "electronic_engineering",
                    17: "food_engineering",
                    18: "chemical_engineering"
                }
                self.dataset_info = {
                    "data_files": self.data_files,
                    "domain_map": self.domain_map,
                    "description": "模拟知识库数据 | Simulated knowledge base data",
                    "created": time.time()
                }
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data_file = self.data_files[idx]
        data_path = os.path.join(self.data_dir, data_file)
        
        try:
            # 尝试加载数据文件
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
            else:
                # 生成模拟数据
                data = self._generate_dummy_data()
            
            # 提取知识特征
            knowledge_features = self._extract_knowledge_features(data)
            
            # 提取领域标签
            domain_labels = self._extract_domain_labels(data)
            
            # 转换为张量
            knowledge_features = torch.tensor(knowledge_features, dtype=torch.float32)
            domain_labels = torch.tensor(domain_labels, dtype=torch.long)
            
            return knowledge_features, domain_labels
            
        except Exception as e:
            logger.error(f"处理知识库数据错误: {str(e)}")
            # 返回空数据
            dummy_features = torch.zeros((self.sequence_length, 8), dtype=torch.float32)
            dummy_labels = torch.zeros(self.sequence_length, dtype=torch.long)
            return dummy_features, dummy_labels
    
    def _extract_knowledge_features(self, data: pd.DataFrame) -> np.ndarray:
        """提取知识特征 | Extract knowledge features"""
        # 定义知识特征列
        feature_columns = ['physics', 'math', 'chemistry', 'biology', 
                          'history', 'geography', 'literature', 'art']
        
        # 如果数据中缺少某些列，使用默认值填充
        for col in feature_columns:
            if col not in data.columns:
                data[col] = np.random.uniform(0, 1, len(data))
        
        # 提取特征并确保序列长度
        features = data[feature_columns].values
        if len(features) < self.sequence_length:
            # 填充到所需长度
            padding = np.zeros((self.sequence_length - len(features), len(feature_columns)))
            features = np.vstack([features, padding])
        elif len(features) > self.sequence_length:
            # 截断到所需长度
            features = features[:self.sequence_length]
        
        return features
    
    def _extract_domain_labels(self, data: pd.DataFrame) -> np.ndarray:
        """提取领域标签 | Extract domain labels"""
        if 'domain' in data.columns:
            # 将领域字符串映射到数字标签
            labels = []
            for domain in data['domain']:
                if domain in self.domain_map.values():
                    # 找到领域对应的数字标签
                    label = [k for k, v in self.domain_map.items() if v == domain][0]
                    labels.append(label)
                else:
                    # 使用随机标签
                    labels.append(np.random.randint(0, len(self.domain_map)))
            
            labels = np.array(labels)
        else:
            # 生成随机标签
            labels = np.random.randint(0, len(self.domain_map), self.sequence_length)
        
        # 确保序列长度
        if len(labels) < self.sequence_length:
            labels = np.pad(labels, (0, self.sequence_length - len(labels)), 'constant')
        elif len(labels) > self.sequence_length:
            labels = labels[:self.sequence_length]
        
        return labels
    
    def _generate_dummy_data(self) -> pd.DataFrame:
        """生成模拟数据 | Generate dummy data"""
        # 创建模拟知识数据
        num_samples = self.sequence_length
        data = {
            'physics': np.random.uniform(0, 1, num_samples),
            'math': np.random.uniform(0, 1, num_samples),
            'chemistry': np.random.uniform(0, 1, num_samples),
            'biology': np.random.uniform(0, 1, num_samples),
            'history': np.random.uniform(0, 1, num_samples),
            'geography': np.random.uniform(0, 1, num_samples),
            'literature': np.random.uniform(0, 1, num_samples),
            'art': np.random.uniform(0, 1, num_samples),
            'domain': np.random.choice(list(self.domain_map.values()), num_samples)
        }
        return pd.DataFrame(data)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int = 10, lr: float = 0.001, device: str = 'cpu') -> Dict:
    """训练知识库模型 | Train knowledge model
    参数:
        model: 知识库模型 | Knowledge model
        train_loader: 训练数据加载器 | Training data loader
        val_loader: 验证数据加载器 | Validation data loader
        epochs: 训练轮数 | Number of epochs
        lr: 学习率 | Learning rate
        device: 训练设备 | Training device
    返回:
        训练历史字典 | Training history dictionary
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    model.to(device)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.numel()
                correct += (predicted == targets).sum().item()
        
        # 更新学习率
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        avg_train_loss = running_loss / len(train_loader)
        avg_val_accuracy = 100 * correct / total if total > 0 else 0
        
        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_accuracy'].append(avg_val_accuracy)
        train_history['learning_rate'].append(current_lr)
        
        logger.info(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Acc: {avg_val_accuracy:.2f}%, '
              f'LR: {current_lr:.8f}')
    
    return train_history

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> Dict:
    """评估知识库模型 | Evaluate knowledge model
    参数:
        model: 知识库模型 | Knowledge model
        test_loader: 测试数据加载器 | Test data loader
        device: 评估设备 | Evaluation device
    返回:
        评估结果字典 | Evaluation result dictionary
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.numel()
            correct += (predicted == targets).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total if total > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

def save_training_results(model: nn.Module, history: Dict, results: Dict, 
                         save_path: str = 'models/i_knowledge_model.pth'):
    """保存训练结果 | Save training results
    参数:
        model: 训练好的模型 | Trained model
        history: 训练历史 | Training history
        results: 评估结果 | Evaluation results
        save_path: 保存路径 | Save path
    """
    # 创建模型目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'training_history': history,
        'evaluation_results': results,
        'timestamp': time.time()
    }, save_path)
    
    # 保存训练日志
    log_path = save_path.replace('.pth', '_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'training_history': history,
            'evaluation_results': results,
            'timestamp': time.time()
        }, f, indent=2)
    
    logger.info(f"模型和训练结果已保存: {save_path}")

def train_jointly(models: List[nn.Module], train_loaders: List[DataLoader], val_loaders: List[DataLoader],
                  epochs: int = 10, lr: float = 0.001, device: str = 'cpu',
                  loss_weights: Optional[List[float]] = None) -> List[Dict]:
    """联合训练多个模型 | Jointly train multiple models
    参数:
        models: 模型列表 | List of models
        train_loaders: 训练数据加载器列表 | List of training data loaders
        val_loaders: 验证数据加载器列表 | List of validation data loaders
        epochs: 训练轮数 | Number of epochs
        lr: 学习率 | Learning rate
        device: 训练设备 | Training device
        loss_weights: 各模型损失权重 | Loss weights for each model
    返回:
        各模型的训练历史字典列表 | List of training history dictionaries for each model
    """
    # 检查参数
    if len(models) != len(train_loaders) or len(models) != len(val_loaders):
        raise ValueError("模型数量必须与加载器数量匹配 | Number of models must match number of loaders")
    
    # 如果未提供损失权重，使用相等权重
    if loss_weights is None:
        loss_weights = [1.0] * len(models)
    elif len(loss_weights) != len(models):
        raise ValueError("损失权重数量必须与模型数量匹配 | Number of loss weights must match number of models")
    
    # 为每个模型创建优化器和学习率调度器
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]
    schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
                 for optimizer in optimizers]
    
    # 为每个模型创建训练历史
    histories = [{
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    } for _ in range(len(models))]
    
    # 将所有模型移动到设备上
    for model in models:
        model.to(device)
    
    for epoch in range(epochs):
        # 联合训练阶段
        for model in models:
            model.train()
        
        running_losses = [0.0] * len(models)
        
        # 迭代训练数据
        for i, (inputs, targets) in enumerate(zip(*train_loaders)):
            # 梯度清零
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            # 前向传播和损失计算
            total_loss = 0.0
            for j, (model, input_data, target_data, weight) in enumerate(
                    zip(models, inputs, targets, loss_weights)):
                input_data = input_data.to(device)
                target_data = target_data.to(device)
                
                outputs = model(input_data)
                loss = nn.CrossEntropyLoss()(outputs, target_data)
                weighted_loss = loss * weight
                running_losses[j] += loss.item()
                total_loss += weighted_loss
            
            # 反向传播和优化
            total_loss.backward()
            for optimizer in optimizers:
                optimizer.step()
        
        # 联合验证阶段
        for model in models:
            model.eval()
        
        val_losses = [0.0] * len(models)
        correct = [0] * len(models)
        total = [0] * len(models)
        
        with torch.no_grad():
            for inputs, targets in zip(*val_loaders):
                for j, (model, input_data, target_data) in enumerate(
                        zip(models, inputs, targets)):
                    input_data = input_data.to(device)
                    target_data = target_data.to(device)
                    
                    outputs = model(input_data)
                    loss = nn.CrossEntropyLoss()(outputs, target_data)
                    val_losses[j] += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total[j] += target_data.numel()
                    correct[j] += (predicted == target_data).sum().item()
        
        # 更新学习率
        avg_val_losses = [val_loss / len(val_loaders[j]) for j, val_loss in enumerate(val_losses)]
        for j, scheduler in enumerate(schedulers):
            scheduler.step(avg_val_losses[j])
        
        # 记录历史
        avg_train_losses = [running_loss / len(train_loaders[j]) for j, running_loss in enumerate(running_losses)]
        avg_val_accuracies = [100 * corr / total_j if total_j > 0 else 0 
                             for corr, total_j in zip(correct, total)]
        current_lrs = [optimizer.param_groups[0]['lr'] for optimizer in optimizers]
        
        for j in range(len(models)):
            histories[j]['train_loss'].append(avg_train_losses[j])
            histories[j]['val_loss'].append(avg_val_losses[j])
            histories[j]['val_accuracy'].append(avg_val_accuracies[j])
            histories[j]['learning_rate'].append(current_lrs[j])
        
        # 记录日志
        logger.info(f'联合训练 - Epoch {epoch+1}/{epochs}')
        for j in range(len(models)):
            logger.info(f'  模型 {j+1} - 训练损失: {avg_train_losses[j]:.4f}, ' \
                       f'验证损失: {avg_val_losses[j]:.4f}, ' \
                       f'验证准确率: {avg_val_accuracies[j]:.2f}%, ' \
                       f'学习率: {current_lrs[j]:.8f}')
    
    return histories

def main():
    """主训练函数 | Main training function"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建数据集
    train_dataset = KnowledgeDataset('data/train')
    val_dataset = KnowledgeDataset('data/val')
    test_dataset = KnowledgeDataset('data/test')
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    logger.info(f"领域映射: {train_dataset.domain_map}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 初始化模型
    num_classes = len(train_dataset.domain_map) if train_dataset.domain_map else 19
    model = KnowledgeModel(input_size=8, hidden_size=128, num_classes=num_classes, num_layers=3)
    
    # 训练模型（单独训练模式）
    logger.info("开始训练知识库模型（单独训练模式） | Starting knowledge model training (individual mode)")
    history = train_model(model, train_loader, val_loader, epochs=25, lr=0.001, device=device)
    
    # 评估模型
    logger.info("开始评估知识库模型 | Starting knowledge model evaluation")
    results = evaluate_model(model, test_loader, device=device)
    
    # 保存结果
    save_training_results(model, history, results)
    
    logger.info("知识库模型训练完成 | Knowledge model training completed")
    logger.info(f"最终评估结果 - 损失: {results['loss']:.4f}, 准确率: {results['accuracy']:.2f}%")
    
    # 示例：如何使用联合训练功能
    # 注意：在实际应用中，您需要提供多个不同的模型和相应的数据集
    # 以下代码仅作为示例，用于展示如何调用联合训练功能
    logger.info("\n--- 联合训练功能示例 ---\n")
    try:
        # 为了示例目的，创建另一个相同类型的模型
        secondary_model = KnowledgeModel(input_size=8, hidden_size=128, num_classes=num_classes, num_layers=3)
        
        # 准备模型列表和数据加载器列表
        models_list = [model, secondary_model]
        train_loaders_list = [train_loader, train_loader]  # 在实际应用中应使用不同数据集
        val_loaders_list = [val_loader, val_loader]        # 在实际应用中应使用不同数据集
        
        # 配置损失权重
        loss_weights = [0.6, 0.4]
        
        # 执行联合训练
        logger.info("开始联合训练示例 | Starting joint training example")
        joint_histories = train_jointly(
            models=models_list,
            train_loaders=train_loaders_list,
            val_loaders=val_loaders_list,
            epochs=5,  # 使用较少的轮数用于演示
            lr=0.001,
            device=device,
            loss_weights=loss_weights
        )
        
        # 评估联合训练后的模型
        logger.info("评估联合训练后的模型 | Evaluating jointly trained models")
        for i, trained_model in enumerate(models_list):
            joint_results = evaluate_model(trained_model, test_loader, device=device)
            logger.info(f"模型 {i+1} - 联合训练后评估结果: 损失: {joint_results['loss']:.4f}, 准确率: {joint_results['accuracy']:.2f}%")
            
            # 保存联合训练后的模型
            joint_save_path = f'models/i_knowledge_joint_model_{i+1}.pth'
            save_training_results(trained_model, joint_histories[i], joint_results, joint_save_path)
    except Exception as e:
        logger.error(f"联合训练示例执行错误: {str(e)}")
        logger.info("在实际应用中，请确保提供正确的模型列表和数据加载器列表")

if __name__ == '__main__':
    main()