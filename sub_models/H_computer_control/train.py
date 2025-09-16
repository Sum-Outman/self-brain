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
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
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
               epochs: int = 10, lr: float = 0.001, device: str = 'cpu',
               joint_training: bool = False, shared_optimizer: Optional[optim.Optimizer] = None,
               loss_weight: float = 1.0) -> Dict:
    """训练计算机控制模型 | Train computer control model
    参数:
        model: 计算机控制模型 | Computer control model
        train_loader: 训练数据加载器 | Training data loader
        val_loader: 验证数据加载器 | Validation data loader
        epochs: 训练轮数 | Number of epochs
        lr: 学习率 | Learning rate
        device: 训练设备 | Training device
        joint_training: 是否为联合训练模式 | Whether in joint training mode
        shared_optimizer: 共享优化器 (用于联合训练) | Shared optimizer (for joint training)
        loss_weight: 损失权重 (用于联合训练) | Loss weight (for joint training)
    返回:
        训练历史字典 | Training history dictionary
    """
    criterion = nn.CrossEntropyLoss()
    # 如果提供了共享优化器并且处于联合训练模式，则使用它
    if joint_training and shared_optimizer is not None:
        optimizer = shared_optimizer
        # 联合训练中不使用学习率调度器，由外部统一管理
        scheduler = None
    else:
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
            
            # 只有在非联合训练模式下才执行optimizer.zero_grad()
            if not joint_training:
                optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 应用损失权重
            if joint_training:
                loss = loss * loss_weight
                # 在联合训练模式下，返回loss而不执行backward
                return {'loss': loss}
            else:
                # 非联合训练模式下正常执行backward和优化步骤
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

def train_jointly(models: List[nn.Module], model_names: List[str], 
                  train_loaders: List[DataLoader], val_loaders: List[DataLoader], 
                  epochs: int = 10, lr: float = 0.001, device: str = 'cpu',
                  loss_weights: Optional[List[float]] = None, shared_optimizer: bool = False) -> Dict:
    """联合训练多个模型 | Jointly train multiple models
    参数:
        models: 模型列表 | List of models
        model_names: 模型名称列表 | List of model names
        train_loaders: 训练数据加载器列表 | List of training data loaders
        val_loaders: 验证数据加载器列表 | List of validation data loaders
        epochs: 训练轮数 | Number of epochs
        lr: 学习率 | Learning rate
        device: 训练设备 | Training device
        loss_weights: 损失权重列表 | List of loss weights
        shared_optimizer: 是否使用共享优化器 | Whether to use a shared optimizer
    返回:
        训练历史字典 | Training history dictionary
    """
    criterion = nn.CrossEntropyLoss()
    
    # 参数检查
    if len(models) != len(model_names) or len(models) != len(train_loaders) or len(models) != len(val_loaders):
        raise ValueError("模型数量、名称数量、训练加载器数量和验证加载器数量必须相同")
    
    # 设置损失权重
    if loss_weights is None:
        loss_weights = [1.0] * len(models)
    elif len(loss_weights) != len(models):
        raise ValueError("损失权重数量必须与模型数量相同")
    
    # 归一化损失权重
    total_weight = sum(loss_weights)
    loss_weights = [w / total_weight for w in loss_weights]
    
    # 配置优化器
    if shared_optimizer:
        # 共享优化器：合并所有模型的参数
        all_params = []
        for model in models:
            all_params.extend(list(model.parameters()))
        optimizer = optim.Adam(all_params, lr=lr)
        optimizers = [optimizer] * len(models)  # 所有模型共享同一个优化器
        schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)]
    else:
        # 独立优化器：每个模型有自己的优化器
        optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]
        schedulers = [optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2) for opt in optimizers]
    
    # 初始化历史记录
    history = {name: {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'learning_rate': []} for name in model_names}
    shared_history = {'joint_loss': [], 'loss_weights': loss_weights, 'shared_optimizer': shared_optimizer}
    
    # 将模型移动到设备
    for model in models:
        model.to(device)
    
    for epoch in range(epochs):
        # 训练阶段
        for model in models:
            model.train()
        
        total_joint_loss = 0.0
        individual_epoch_losses = [0.0 for _ in models]
        
        # 获取最大的数据集长度进行迭代
        max_loader_len = max(len(loader) for loader in train_loaders)
        
        # 为每个数据加载器创建迭代器
        loaders_iter = [iter(loader) for loader in train_loaders]
        
        for batch_idx in range(max_loader_len):
            # 对每个模型进行前向传播
            individual_losses = []
            
            for i, (model, train_loader_iter) in enumerate(zip(models, loaders_iter)):
                # 获取批次数据，如果加载器用尽则重置
                try:
                    inputs, targets = next(train_loader_iter)
                except StopIteration:
                    # 重置数据加载器
                    loaders_iter[i] = iter(train_loaders[i])
                    inputs, targets = next(loaders_iter[i])
                    
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 应用损失权重
                weighted_loss = loss * loss_weights[i]
                individual_losses.append(weighted_loss)
                individual_epoch_losses[i] += loss.item()
            
            # 计算联合损失
            joint_loss = sum(individual_losses)
            total_joint_loss += joint_loss.item()
            
            # 反向传播和优化
            joint_loss.backward()
            
            # 如果使用共享优化器，只需调用一次step
            if shared_optimizer:
                optimizers[0].step()
                optimizers[0].zero_grad()
            else:
                # 独立优化器，每个都需要step和zero_grad
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
        
        # 计算每个模型的平均训练损失
        avg_train_losses = [loss / max_loader_len for loss in individual_epoch_losses]
        
        # 验证阶段
        for model in models:
            model.eval()
        
        # 记录每个模型的验证结果
        for i, (model, name, val_loader) in enumerate(zip(models, model_names, val_loaders)):
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
            avg_val_loss = val_loss / len(val_loader) if val_loader else 0
            if shared_optimizer:
                if epoch % len(schedulers) == i:
                    schedulers[0].step(avg_val_loss)
                current_lr = optimizers[0].param_groups[0]['lr']
            else:
                if schedulers[i]:
                    schedulers[i].step(avg_val_loss)
                current_lr = optimizers[i].param_groups[0]['lr']
            
            # 记录历史
            val_acc = 100 * correct / total if total > 0 else 0
            
            history[name]['train_loss'].append(avg_train_losses[i])
            history[name]['val_loss'].append(avg_val_loss)
            history[name]['val_accuracy'].append(val_acc)
            history[name]['learning_rate'].append(current_lr)
        
        # 记录联合损失
        avg_joint_loss = total_joint_loss / max_loader_len
        shared_history['joint_loss'].append(avg_joint_loss)
        
        logger.info(f'Epoch {epoch+1}/{epochs}, Joint Loss: {avg_joint_loss:.4f}')
        for name in model_names:
            logger.info(f'  {name} - Train Loss: {history[name]["train_loss"][-1]:.4f}, '
                       f'Val Loss: {history[name]["val_loss"][-1]:.4f}, '
                       f'Val Acc: {history[name]["val_accuracy"][-1]:.2f}%, '
                       f'LR: {history[name]["learning_rate"][-1]:.8f}')
    
    # 合并历史记录
    full_history = {'individual': history, 'shared': shared_history}
    return full_history

class MockComputerControlDataset(Dataset):
    """模拟计算机控制数据集，用于测试 | Mock computer control dataset for testing"""
    def __init__(self, num_samples: int = 100, sequence_length: int = 10, num_classes: int = 10):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.command_map = {i: f"command_{i}" for i in range(num_classes)}
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # 生成随机系统状态特征 (序列长度, 特征维度)
        system_features = torch.randn(self.sequence_length, 5)
        # 生成随机命令标签
        command_labels = torch.randint(0, self.num_classes, (self.sequence_length,))
        return system_features, command_labels


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
    
    # 训练模型 (单独训练)
    logger.info("开始单独训练计算机控制模型 | Starting individual computer control model training")
    history = train_model(model, train_loader, val_loader, epochs=15, lr=0.001, device=device)
    
    # 评估模型
    logger.info("开始评估计算机控制模型 | Starting computer control model evaluation")
    results = evaluate_model(model, test_loader, device=device)
    
    # 保存结果
    save_training_results(model, history, results)
    
    logger.info("计算机控制模型训练完成 | Computer control model training completed")
    logger.info(f"最终评估结果 - 损失: {results['loss']:.4f}, 准确率: {results['accuracy']:.2f}%")
    
    # 联合训练演示
    if True:  # 设置为True可测试联合训练
        # 创建Mock数据集用于演示
        mock_train_dataset = MockComputerControlDataset(num_samples=50, sequence_length=10, num_classes=num_classes)
        mock_val_dataset = MockComputerControlDataset(num_samples=20, sequence_length=10, num_classes=num_classes)
        mock_test_dataset = MockComputerControlDataset(num_samples=20, sequence_length=10, num_classes=num_classes)
        
        # 创建加载器
        mock_train_loader = DataLoader(mock_train_dataset, batch_size=8, shuffle=True)
        mock_val_loader = DataLoader(mock_val_dataset, batch_size=8, shuffle=False)
        mock_test_loader = DataLoader(mock_test_dataset, batch_size=8, shuffle=False)
        
        # 创建第二个模型用于演示
        second_model = ComputerControlModel(input_size=5, hidden_size=64, num_classes=num_classes, num_layers=2)
        
        # 准备联合训练参数
        models = [model, second_model]
        model_names = ['computer_control_model_1', 'computer_control_model_2']
        train_loaders = [train_loader, mock_train_loader]
        val_loaders = [val_loader, mock_val_loader]
        
        # 损失权重和共享优化器选项
        loss_weights = [0.6, 0.4]  # 可以调整权重
        use_shared_optimizer = True  # 设置为True使用共享优化器
        
        logger.info("开始联合训练模型 | Starting joint model training")
        joint_history = train_jointly(
            models=models,
            model_names=model_names,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            epochs=5,  # 可以调整轮数
            lr=0.001,
            device=device,
            loss_weights=loss_weights,
            shared_optimizer=use_shared_optimizer
        )
        
        # 评估联合训练后的模型
        for i, (trained_model, name) in enumerate(zip(models, model_names)):
            # 使用各自的测试集评估
            test_loader = test_loader if i == 0 else mock_test_loader
            eval_results = evaluate_model(trained_model, test_loader, device=device)
            
            # 生成带时间戳的保存路径
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f'models/{name}_joint_trained_{timestamp}.pth'
            
            # 保存结果
            save_training_results(
                trained_model,
                joint_history['individual'][name],
                eval_results,
                save_path=save_path
            )
            
            logger.info(f"模型 {name} 联合训练后评估结果 - 损失: {eval_results['loss']:.4f}, 准确率: {eval_results['accuracy']:.2f}%")
        
        # 打印联合训练摘要
        logger.info("联合训练完成 | Joint training completed")
        logger.info(f"联合训练摘要 - 模型数量: {len(models)}, 损失权重: {loss_weights}, 训练轮次: {5}")
        logger.info(f"最终联合损失: {joint_history['shared']['joint_loss'][-1]:.4f}")

if __name__ == '__main__':
    main()