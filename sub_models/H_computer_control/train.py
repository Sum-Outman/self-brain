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

# 计算机控制模型训练程序
# Computer Control Model Training Program

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
from .model import ComputerControlModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("H_computer_control_Trainer")

class ComputerControlDataset(Dataset):
    """计算机控制数据集类 | Computer Control Dataset Class"""
    def __init__(self, data_dir: str, sequence_length: int = 10):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.data_files = []
        self.dataset_info = {}
        self.command_map = {}
        
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
                    self.command_map = self.dataset_info.get("command_map", {})
            except Exception as e:
                logger.error(f"加载数据集信息错误: {str(e)}")
        
        # 如果没有数据集信息，扫描目录中的数据文件
        if not self.data_files:
            self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            # 如果没有数据文件，创建一些模拟数据信息
            if not self.data_files:
                self.data_files = [f"computer_control_data_{i}.csv" for i in range(5)]
                self.command_map = {
                    0: "start_process",
                    1: "stop_process",
                    2: "allocate_memory",
                    3: "free_memory",
                    4: "optimize_cpu",
                    5: "network_request",
                    6: "disk_operation",
                    7: "system_scan",
                    8: "security_check",
                    9: "backup_data"
                }
                self.dataset_info = {
                    "data_files": self.data_files,
                    "command_map": self.command_map,
                    "description": "模拟计算机控制数据 | Simulated computer control data",
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
            
            # 提取系统状态特征
            system_features = self._extract_system_features(data)
            
            # 提取命令标签
            command_labels = self._extract_command_labels(data)
            
            # 转换为张量
            system_features = torch.tensor(system_features, dtype=torch.float32)
            command_labels = torch.tensor(command_labels, dtype=torch.long)
            
            return system_features, command_labels
            
        except Exception as e:
            logger.error(f"处理计算机控制数据错误: {str(e)}")
            # 返回空数据
            dummy_features = torch.zeros((self.sequence_length, 5), dtype=torch.float32)
            dummy_labels = torch.zeros(self.sequence_length, dtype=torch.long)
            return dummy_features, dummy_labels
    
    def _extract_system_features(self, data: pd.DataFrame) -> np.ndarray:
        """提取系统状态特征 | Extract system state features"""
        # 定义系统状态特征列
        feature_columns = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_traffic', 'process_count']
        
        # 如果数据中缺少某些列，使用默认值填充
        for col in feature_columns:
            if col not in data.columns:
                data[col] = np.random.uniform(0, 100, len(data))
        
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
    
    def _extract_command_labels(self, data: pd.DataFrame) -> np.ndarray:
        """提取命令标签 | Extract command labels"""
        if 'command' in data.columns:
            # 将命令字符串映射到数字标签
            labels = []
            for cmd in data['command']:
                if cmd in self.command_map.values():
                    # 找到命令对应的数字标签
                    label = [k for k, v in self.command_map.items() if v == cmd][0]
                    labels.append(label)
                else:
                    # 使用随机标签
                    labels.append(np.random.randint(0, len(self.command_map)))
            
            labels = np.array(labels)
        else:
            # 生成随机标签
            labels = np.random.randint(0, len(self.command_map), self.sequence_length)
        
        # 确保序列长度
        if len(labels) < self.sequence_length:
            labels = np.pad(labels, (0, self.sequence_length - len(labels)), 'constant')
        elif len(labels) > self.sequence_length:
            labels = labels[:self.sequence_length]
        
        return labels
    
    def _generate_dummy_data(self) -> pd.DataFrame:
        """生成模拟数据 | Generate dummy data"""
        # 创建模拟系统状态数据
        num_samples = self.sequence_length
        data = {
            'cpu_usage': np.random.uniform(0, 100, num_samples),
            'memory_usage': np.random.uniform(0, 100, num_samples),
            'disk_usage': np.random.uniform(0, 100, num_samples),
            'network_traffic': np.random.uniform(0, 1000, num_samples),
            'process_count': np.random.randint(1, 100, num_samples),
            'command': np.random.choice(list(self.command_map.values()), num_samples)
        }
        return pd.DataFrame(data)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int = 10, lr: float = 0.001, device: str = 'cpu') -> Dict:
    """训练计算机控制模型 | Train computer control model
    参数:
        model: 计算机控制模型 | Computer control model
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
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
    """评估计算机控制模型 | Evaluate computer control model
    参数:
        model: 计算机控制模型 | Computer control model
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
                         save_path: str = 'models/h_computer_control_model.pth'):
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

def main():
    """主训练函数 | Main training function"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建数据集
    train_dataset = ComputerControlDataset('data/train')
    val_dataset = ComputerControlDataset('data/val')
    test_dataset = ComputerControlDataset('data/test')
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    logger.info(f"命令映射: {train_dataset.command_map}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 初始化模型
    num_classes = len(train_dataset.command_map) if train_dataset.command_map else 10
    model = ComputerControlModel(input_size=5, hidden_size=64, num_classes=num_classes, num_layers=2)
    
    # 训练模型
    logger.info("开始训练计算机控制模型 | Starting computer control model training")
    history = train_model(model, train_loader, val_loader, epochs=15, lr=0.001, device=device)
    
    # 评估模型
    logger.info("开始评估计算机控制模型 | Starting computer control model evaluation")
    results = evaluate_model(model, test_loader, device=device)
    
    # 保存结果
    save_training_results(model, history, results)
    
    logger.info("计算机控制模型训练完成 | Computer control model training completed")
    logger.info(f"最终评估结果 - 损失: {results['loss']:.4f}, 准确率: {results['accuracy']:.2f}%")

if __name__ == '__main__':
    main()