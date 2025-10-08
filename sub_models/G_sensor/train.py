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

# 传感器感知模型训练程序
# Sensor Perception Model Training Program

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
from .model import SensorModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("G_sensor_Trainer")

class SensorDataset(Dataset):
    """传感器数据集类 | Sensor Dataset Class"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = []
        self.dataset_info = {}
        
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
            except Exception as e:
                logger.error(f"加载数据集信息错误: {str(e)}")
        
        # 如果没有数据集信息，扫描目录中的数据文件
        if not self.data_files:
            self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            # 如果没有数据文件，创建一些模拟数据信息
            if not self.data_files:
                self.data_files = [f"sensor_data_{i}.csv" for i in range(8)]
                self.dataset_info = {
                    "data_files": self.data_files,
                    "description": "模拟传感器数据 | Simulated sensor data",
                    "created": time.time()
                }
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        data_path = os.path.join(self.data_dir, data_file)
        
        try:
            # 尝试加载数据文件
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
            else:
                # 生成模拟数据
                data = self._generate_dummy_data()
            
            # 提取传感器数据特征
            # Extract sensor data features
            features = self._extract_sensor_features(data)
            
            # 目标值
            # Target values
            targets = self._extract_targets(data)
            
            return features, targets
            
        except Exception as e:
            logger.error(f"处理传感器数据错误: {str(e)}")
            # 返回空数据
            dummy_features = torch.zeros((8,), dtype=torch.float32)
            dummy_targets = torch.tensor(0, dtype=torch.long)
            return dummy_features, dummy_targets
    
    def _extract_sensor_features(self, data: pd.DataFrame) -> torch.Tensor:
        """提取传感器特征 | Extract sensor features"""
        # 定义传感器特征列
        feature_columns = ['temperature', 'humidity', 'acceleration_x', 'acceleration_y', 
                          'acceleration_z', 'pressure', 'distance', 'light_intensity']
        
        # 如果数据中缺少某些列，使用默认值填充
        for col in feature_columns:
            if col not in data.columns:
                if col == 'temperature':
                    data[col] = np.random.uniform(20, 30, len(data))
                elif col == 'humidity':
                    data[col] = np.random.uniform(40, 80, len(data))
                elif col.startswith('acceleration'):
                    data[col] = np.random.uniform(-5, 5, len(data))
                elif col == 'pressure':
                    data[col] = np.random.uniform(950, 1050, len(data))
                elif col == 'distance':
                    data[col] = np.random.uniform(0, 100, len(data))
                elif col == 'light_intensity':
                    data[col] = np.random.uniform(0, 1000, len(data))
        
        # 提取特征
        features = data[feature_columns].values[0]  # 取第一行作为样本
        
        # 转换为张量
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_targets(self, data: pd.DataFrame) -> torch.Tensor:
        """提取目标值 | Extract targets"""
        if 'target' in data.columns:
            targets = data['target'].values[0]  # 取第一行的目标值
        else:
            # 生成随机目标值
            targets = np.random.randint(0, 3)  # 假设有3个类别
        
        # 转换为张量
        return torch.tensor(targets, dtype=torch.long)
    
    def _generate_dummy_data(self) -> pd.DataFrame:
        """生成模拟数据 | Generate dummy data"""
        # 创建模拟传感器数据
        data = {
            'temperature': [np.random.uniform(20, 30)],
            'humidity': [np.random.uniform(40, 80)],
            'acceleration_x': [np.random.uniform(-5, 5)],
            'acceleration_y': [np.random.uniform(-5, 5)],
            'acceleration_z': [np.random.uniform(-5, 5)],
            'pressure': [np.random.uniform(950, 1050)],
            'distance': [np.random.uniform(0, 100)],
            'light_intensity': [np.random.uniform(0, 1000)],
            'target': [np.random.randint(0, 3)]
        }
        return pd.DataFrame(data)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int = 10, lr: float = 0.001, device: str = 'cpu',
               joint_training: bool = False, shared_optimizer: Optional[optim.Optimizer] = None,
               loss_weight: float = 1.0) -> Dict:
    """训练传感器模型 | Train sensor model
    参数:
        model: 传感器模型 | Sensor model
        train_loader: 训练数据加载器 | Training data loader
        val_loader: 验证数据加载器 | Validation data loader
        epochs: 训练轮数 | Number of epochs
        lr: 学习率 | Learning rate
        device: 训练设备 | Training device
        joint_training: 是否为联合训练模式 | Whether it's joint training mode
        shared_optimizer: 共享优化器（联合训练时使用） | Shared optimizer (used in joint training)
        loss_weight: 损失权重（联合训练时使用） | Loss weight (used in joint training)
    返回:
        训练历史字典 | Training history dictionary
    """
    criterion = nn.CrossEntropyLoss()
    
    # 如果是联合训练且提供了共享优化器，则使用共享优化器
    if joint_training and shared_optimizer is not None:
        optimizer = shared_optimizer
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 训练历史记录
    # Training history records
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            if not joint_training:
                optimizer.zero_grad()
            
            # 前向传播
            # Forward pass
            outputs = model(inputs)
            
            # 计算损失
            # Calculate loss
            loss = criterion(outputs, targets) * loss_weight
            
            if joint_training:
                # 联合训练时，返回loss而不立即backward
                return loss
            else:
                # 单独训练时，正常backward和优化
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
        
        # 验证阶段
        # Validation phase
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
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # 更新学习率
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录训练统计信息
        # Record training statistics
        avg_train_loss = running_loss / len(train_loader)
        val_acc = 100 * correct / total if total > 0 else 0
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        logger.info(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Acc: {val_acc:.2f}%, '
              f'LR: {current_lr:.8f}')
    
    return history

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> Dict:
    """评估传感器模型 | Evaluate sensor model
    参数:
        model: 传感器模型 | Sensor model
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
            total += targets.size(0)
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
                         save_path: str = 'models/g_sensor_model.pth'):
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
                  loss_weights: Optional[List[float]] = None, shared_optimizer: bool = True) -> Dict:
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
        shared_optimizer: 是否使用共享优化器 | Whether to use shared optimizer
    返回:
        训练历史字典 | Training history dictionary
    """
    # 参数检查
    if len(models) != len(model_names) or len(models) != len(train_loaders) or len(models) != len(val_loaders):
        raise ValueError("模型数量、模型名称数量、训练加载器数量和验证加载器数量必须相同")
    
    if loss_weights is None:
        loss_weights = [1.0 / len(models)] * len(models)
    elif len(loss_weights) != len(models):
        raise ValueError("损失权重数量必须与模型数量相同")
    
    # 归一化权重
    total_weight = sum(loss_weights)
    loss_weights = [w / total_weight for w in loss_weights]
    
    criterion = nn.CrossEntropyLoss()
    
    # 创建优化器
    if shared_optimizer:
        # 共享优化器 - 所有模型参数合并到一个优化器
        all_params = []
        for model in models:
            all_params.extend(model.parameters())
        optimizer = optim.Adam(all_params, lr=lr)
        optimizers = [optimizer] * len(models)
    else:
        # 独立优化器 - 每个模型有自己的优化器
        optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]
    
    # 创建学习率调度器
    schedulers = [optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2) for opt in optimizers]
    
    # 初始化历史记录
    history = {name: {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'learning_rate': []} for name in model_names}
    shared_history = {
        'joint_loss': [],
        'loss_weights': loss_weights,
        'shared_optimizer': shared_optimizer
    }
    
    # 将模型移动到设备
    for model in models:
        model.to(device)
    
    for epoch in range(epochs):
        # 训练阶段
        for i, model in enumerate(models):
            model.train()
        
        total_joint_loss = 0.0
        individual_epoch_losses = {name: 0.0 for name in model_names}
        
        # 获取最大数据加载器长度以确保遍历所有数据
        max_loader_len = max(len(loader) for loader in train_loaders)
        
        for batch_idx in range(max_loader_len):
            optimizer.zero_grad()
            
            # 对每个模型进行前向传播
            individual_losses = []
            
            for i, (model, model_name, train_loader, weight) in enumerate(zip(models, model_names, train_loaders, loss_weights)):
                # 处理数据加载器的迭代
                try:
                    inputs, targets = next(iter(train_loader))
                except StopIteration:
                    # 如果数据加载器用尽，重新创建迭代器
                    train_iter = iter(train_loader)
                    inputs, targets = next(train_iter)
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets) * weight
                
                individual_losses.append(loss)
                individual_epoch_losses[model_name] += loss.item()
            
            # 计算联合损失
            joint_loss = sum(individual_losses)
            total_joint_loss += joint_loss.item()
            
            # 反向传播和优化
            joint_loss.backward()
            optimizer.step()
        
        # 验证阶段
        for model in models:
            model.eval()
        
        # 记录每个模型的验证结果
        for i, (model, name, val_loader, scheduler) in enumerate(zip(models, model_names, val_loaders, schedulers)):
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
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            # 更新学习率
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            current_lr = optimizers[i].param_groups[0]['lr']
            
            # 记录历史
            avg_train_loss = individual_losses[i].item()
            val_acc = 100 * correct / total if total > 0 else 0
            
            history[name]['train_loss'].append(avg_train_loss)
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

# 模拟传感器数据集类
class MockSensorDataset(SensorDataset):
    """模拟传感器数据集，用于演示联合训练功能"""
    def __init__(self, mode='train', size=1000):
        # 这里不调用父类的初始化，而是直接设置模拟数据
        self.size = size
        self.generate_mock_data()
    
    def generate_mock_data(self):
        """生成模拟传感器数据"""
        # 使用随机数据模拟传感器读数
        self.data = {
            'features': torch.randn(self.size, 8),  # 8维特征
            'targets': torch.randint(0, 3, (self.size,))  # 3分类
        }
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 重写__getitem__方法，直接返回模拟数据
        return self.data['features'][idx], self.data['targets'][idx]
    
    # 以下方法是为了兼容父类接口
    def _load_dataset_info(self):
        pass
    
    def _extract_sensor_features(self, data):
        pass
    
    def _extract_targets(self, data):
        pass
    
    def _generate_dummy_data(self):
        pass

def main():
    """主训练函数 | Main training function"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建数据集
    # Create datasets
    train_dataset = SensorDataset('data/train')
    val_dataset = SensorDataset('data/val')
    test_dataset = SensorDataset('data/test')
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    # Initialize model
    model = SensorModel()
    
    # 训练模型 (单独训练)
    # Train model (individual training)
    logger.info("开始单独训练传感器模型 | Starting individual sensor model training")
    history = train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device=device)
    
    # 评估模型
    # Evaluate model
    logger.info("开始评估传感器模型 | Starting sensor model evaluation")
    results = evaluate_model(model, test_loader, device=device)
    
    # 保存结果
    # Save results
    save_training_results(model, history, results)
    
    logger.info("传感器模型训练完成并保存 | Sensor model training completed and saved")
    logger.info(f"最终评估结果 - 损失: {results['loss']:.4f}, 准确率: {results['accuracy']:.2f}%")
    
    # 联合训练演示
    print("\n======== 联合训练演示 ========")
    
    # 创建两个传感器模型
    model1 = SensorModel()
    model2 = SensorModel()
    
    # 创建模拟数据集用于联合训练
    mock_train_dataset1 = MockSensorDataset(mode='train', size=500)
    mock_val_dataset1 = MockSensorDataset(mode='val', size=100)
    
    mock_train_dataset2 = MockSensorDataset(mode='train', size=500)
    mock_val_dataset2 = MockSensorDataset(mode='val', size=100)
    
    # 创建数据加载器
    joint_train_loader1 = DataLoader(mock_train_dataset1, batch_size=32, shuffle=True)
    joint_val_loader1 = DataLoader(mock_val_dataset1, batch_size=32, shuffle=False)
    
    joint_train_loader2 = DataLoader(mock_train_dataset2, batch_size=32, shuffle=True)
    joint_val_loader2 = DataLoader(mock_val_dataset2, batch_size=32, shuffle=False)
    
    # 执行联合训练
    joint_history = train_jointly(
        models=[model1, model2],
        model_names=["sensor_model1", "sensor_model2"],
        train_loaders=[joint_train_loader1, joint_train_loader2],
        val_loaders=[joint_val_loader1, joint_val_loader2],
        epochs=5,
        lr=0.001,
        device=device,
        loss_weights=[0.6, 0.4],  # 模型1权重60%，模型2权重40%
        shared_optimizer=True  # 使用共享优化器
    )
    
    # 评估联合训练后的模型
    model1.eval()
    model2.eval()
    
    test_loss1 = evaluate_model(model1, test_loader, device=device)
    test_loss2 = evaluate_model(model2, test_loader, device=device)
    
    print(f"\n联合训练后模型1评估结果:")
    print(f"Test Loss: {test_loss1['loss']:.4f}, Test Accuracy: {test_loss1['accuracy']:.4f}%")
    print(f"联合训练后模型2评估结果:")
    print(f"Test Loss: {test_loss2['loss']:.4f}, Test Accuracy: {test_loss2['accuracy']:.4f}%")
    
    # 保存联合训练结果
    joint_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型1
    save_training_results(
        model=model1,
        history=joint_history['individual']['sensor_model1'],
        results=test_loss1,
        save_path=f"models/sensor_joint_1_{joint_timestamp}.pth"
    )
    
    # 保存模型2
    save_training_results(
        model=model2,
        history=joint_history['individual']['sensor_model2'],
        results=test_loss2,
        save_path=f"models/sensor_joint_2_{joint_timestamp}.pth"
    )
    
    # 保存联合训练历史
    joint_history_path = f"models/joint_training_history_{joint_timestamp}.json"
    os.makedirs("models", exist_ok=True)
    with open(joint_history_path, "w") as f:
        json.dump(joint_history, f, indent=4)
    
    print(f"\n联合训练历史已保存至: {joint_history_path}")
    
    # 打印联合训练摘要
    print("\n======== 联合训练摘要 ========")
    print(f"模型数量: {len(joint_history['individual'])}")
    print(f"损失权重: {joint_history['shared']['loss_weights']}")
    print(f"训练轮次: {len(joint_history['shared']['joint_loss'])}")
    print(f"最终联合损失: {joint_history['shared']['joint_loss'][-1]:.4f}")
    print("=============================")

if __name__ == '__main__':
    main()