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

# 双目空间定位感知模型训练程序
# Binocular Spatial Localization Perception Model Training Program

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
import json
import logging
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from .model import SpatialModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("F_spatial_Trainer")

class SpatialDataset(Dataset):
    """空间数据集类 | Spatial Dataset Class"""
    def __init__(self, data_dir: str, image_size: int = 224):
        self.data_dir = data_dir
        self.image_size = image_size
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
            self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
            # 如果没有数据文件，创建一些模拟数据信息
            if not self.data_files:
                self.data_files = [f"spatial_data_{i}.npz" for i in range(10)]
                self.dataset_info = {
                    "data_files": self.data_files,
                    "description": "模拟空间定位数据 | Simulated spatial localization data",
                    "created": time.time()
                }
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        data_file = self.data_files[idx]
        data_path = os.path.join(self.data_dir, data_file)
        
        try:
            # 尝试加载数据文件
            if os.path.exists(data_path):
                data = np.load(data_path)
                # 加载双目图像和深度图
                left_img = data['left_img'] if 'left_img' in data else self._generate_dummy_image()
                right_img = data['right_img'] if 'right_img' in data else self._generate_dummy_image()
                depth_map = data['depth_map'] if 'depth_map' in data else self._generate_dummy_depth_map()
            else:
                # 生成模拟数据
                left_img = self._generate_dummy_image()
                right_img = self._generate_dummy_image()
                depth_map = self._generate_dummy_depth_map()
            
            # 预处理
            left_img = cv2.resize(left_img, (self.image_size, self.image_size))
            right_img = cv2.resize(right_img, (self.image_size, self.image_size))
            depth_map = cv2.resize(depth_map, (self.image_size, self.image_size))
            
            # 转换为张量并归一化
            left_img = torch.tensor(left_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            right_img = torch.tensor(right_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            depth_map = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0) / 255.0
            
            return (left_img, right_img), depth_map
            
        except Exception as e:
            logger.error(f"处理空间数据错误: {str(e)}")
            # 返回空数据
            dummy_img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            dummy_depth = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            return (dummy_img, dummy_img), dummy_depth
    
    def _generate_dummy_image(self) -> np.ndarray:
        """生成虚拟图像 | Generate dummy image"""
        return np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
    
    def _generate_dummy_depth_map(self) -> np.ndarray:
        """生成虚拟深度图 | Generate dummy depth map"""
        # 创建简单的深度梯度
        depth_map = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        for i in range(self.image_size):
            for j in range(self.image_size):
                depth_map[i, j] = int(255 * (i + j) / (2 * self.image_size))
        return depth_map

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int = 10, lr: float = 0.001, device: str = 'cpu',
               joint_training: bool = False, shared_optimizer: Optional[optim.Optimizer] = None,
               loss_weight: float = 1.0) -> Dict:
    """训练空间模型 | Train spatial model
    参数:
        model: 空间模型 | Spatial model
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
    criterion = nn.MSELoss()
    
    # 如果是联合训练且提供了共享优化器，则使用共享优化器
    if joint_training and shared_optimizer is not None:
        optimizer = shared_optimizer
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    model.to(device)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            left_img, right_img = inputs
            left_img = left_img.to(device)
            right_img = right_img.to(device)
            targets = targets.to(device)
            
            if not joint_training:
                optimizer.zero_grad()
            
            outputs = model(left_img, right_img)
            loss = criterion(outputs, targets) * loss_weight
            
            if joint_training:
                # 联合训练时，返回loss而不立即backward
                return loss
            else:
                # 单独训练时，正常backward和优化
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        
        # 验证阶段（只有非联合训练时执行）
        if not joint_training:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    left_img, right_img = inputs
                    left_img = left_img.to(device)
                    right_img = right_img.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(left_img, right_img)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            # 更新学习率
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            avg_train_loss = running_loss / len(train_loader)
            train_history['train_loss'].append(avg_train_loss)
            train_history['val_loss'].append(avg_val_loss)
            train_history['learning_rate'].append(current_lr)
            
            logger.info(f'Epoch {epoch+1}/{epochs}, '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, '
                  f'LR: {current_lr:.8f}')
    
    return train_history

def train_jointly(models: List[nn.Module], train_loaders: List[DataLoader], val_loaders: List[DataLoader],
                  epochs: int = 10, lr: float = 0.001, device: str = 'cpu',
                  loss_weights: Optional[List[float]] = None, shared_optimizer: bool = True) -> Dict:
    """联合训练多个空间模型 | Jointly train multiple spatial models
    参数:
        models: 模型列表 | List of models
        train_loaders: 训练数据加载器列表 | List of training data loaders
        val_loaders: 验证数据加载器列表 | List of validation data loaders
        epochs: 训练轮数 | Number of epochs
        lr: 学习率 | Learning rate
        device: 训练设备 | Training device
        loss_weights: 损失权重列表 | List of loss weights
        shared_optimizer: 是否使用共享优化器 | Whether to use shared optimizer
    返回:
        联合训练历史字典 | Joint training history dictionary
    """
    # 参数检查
    if len(models) != len(train_loaders) or len(models) != len(val_loaders):
        raise ValueError("模型数量、训练加载器数量和验证加载器数量必须相同")
    
    if loss_weights is None:
        loss_weights = [1.0 / len(models)] * len(models)
    elif len(loss_weights) != len(models):
        raise ValueError("损失权重数量必须与模型数量相同")
    
    # 归一化权重
    total_weight = sum(loss_weights)
    loss_weights = [w / total_weight for w in loss_weights]
    
    # 移动模型到设备
    for model in models:
        model.to(device)
    
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
    schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3) 
                 for optim in optimizers]
    
    # 初始化历史记录
    joint_history = {
        'total_loss': [],
        'individual_losses': [[] for _ in range(len(models))],
        'learning_rate': [],
        'models_info': [str(model) for model in models],
        'loss_weights': loss_weights
    }
    
    criterion = nn.MSELoss()
    
    # 联合训练循环
    for epoch in range(epochs):
        for i, model in enumerate(models):
            model.train()
        
        total_loss = 0.0
        individual_epoch_losses = [0.0 for _ in range(len(models))]
        
        # 获取最大数据加载器长度以确保遍历所有数据
        max_loader_len = max(len(loader) for loader in train_loaders)
        
        for batch_idx in range(max_loader_len):
            optimizer.zero_grad()
            
            batch_losses = []
            
            for i, (model, train_loader) in enumerate(zip(models, train_loaders)):
                # 处理数据加载器的迭代
                try:
                    inputs, targets = next(iter(train_loader))
                except StopIteration:
                    # 如果数据加载器用尽，重新创建迭代器
                    train_iter = iter(train_loader)
                    inputs, targets = next(train_iter)
                
                left_img, right_img = inputs
                left_img = left_img.to(device)
                right_img = right_img.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(left_img, right_img)
                loss = criterion(outputs, targets) * loss_weights[i]
                
                batch_losses.append(loss)
                individual_epoch_losses[i] += loss.item()
            
            # 计算总损失并反向传播
            joint_loss = sum(batch_losses)
            joint_loss.backward()
            optimizer.step()
            
            total_loss += joint_loss.item()
        
        # 验证阶段
        with torch.no_grad():
            val_total_loss = 0.0
            val_individual_losses = [0.0 for _ in range(len(models))]
            
            for i, (model, val_loader) in enumerate(zip(models, val_loaders)):
                model.eval()
                val_loss = 0.0
                
                for inputs, targets in val_loader:
                    left_img, right_img = inputs
                    left_img = left_img.to(device)
                    right_img = right_img.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(left_img, right_img)
                    loss = criterion(outputs, targets) * loss_weights[i]
                    val_loss += loss.item()
                
                val_individual_losses[i] = val_loss / len(val_loader)
                val_total_loss += val_individual_losses[i]
            
        # 更新学习率
        schedulers[0].step(val_total_loss)
        current_lr = optimizers[0].param_groups[0]['lr']
        
        # 记录历史
        avg_total_loss = total_loss / max_loader_len
        for i in range(len(models)):
            individual_epoch_losses[i] /= max_loader_len
            joint_history['individual_losses'][i].append(individual_epoch_losses[i])
        
        joint_history['total_loss'].append(avg_total_loss)
        joint_history['learning_rate'].append(current_lr)
        
        logger.info(f'联合训练 Epoch {epoch+1}/{epochs}, '
              f'Total Loss: {avg_total_loss:.6f}, '
              f'Val Total Loss: {val_total_loss:.6f}, '
              f'LR: {current_lr:.8f}')
        
        # 记录每个模型的损失
        for i in range(len(models)):
            logger.info(f'  模型 {i+1} - 训练损失: {individual_epoch_losses[i]:.6f}, 验证损失: {val_individual_losses[i]:.6f}')
    
    return joint_history

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> Dict:
    """评估空间模型 | Evaluate spatial model
    参数:
        model: 空间模型 | Spatial model
        test_loader: 测试数据加载器 | Test data loader
        device: 评估设备 | Evaluation device
    返回:
        评估结果字典 | Evaluation result dictionary
    """
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            left_img, right_img = inputs
            left_img = left_img.to(device)
            right_img = right_img.to(device)
            targets = targets.to(device)
            
            outputs = model(left_img, right_img)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    avg_loss = test_loss / len(test_loader)
    
    return {
        'loss': avg_loss,
        'rmse': np.sqrt(avg_loss),  # 均方根误差
        'samples': len(test_loader.dataset)
    }

def save_training_results(model: nn.Module, history: Dict, results: Dict, 
                         save_path: str = 'models/f_spatial_model.pth'):
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

class MockSpatialDataset(Dataset):
    """模拟空间数据集，用于测试联合训练 | Mock spatial dataset for testing joint training"""
    def __init__(self, num_samples: int = 50, image_size: int = 224):
        self.num_samples = num_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # 生成随机的左图和右图
        left_img = torch.rand((3, self.image_size, self.image_size), dtype=torch.float32)
        right_img = torch.rand((3, self.image_size, self.image_size), dtype=torch.float32)
        
        # 生成深度图（简单的梯度）
        depth_map = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        for i in range(self.image_size):
            for j in range(self.image_size):
                depth_map[0, i, j] = (i + j) / (2 * self.image_size)
        
        return (left_img, right_img), depth_map

def main():
    """主训练函数 | Main training function"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建数据集
    train_dataset = SpatialDataset('data/train')
    val_dataset = SpatialDataset('data/val')
    test_dataset = SpatialDataset('data/test')
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 初始化模型
    model = SpatialModel()
    
    # 训练模型
    logger.info("开始训练空间模型 | Starting spatial model training")
    history = train_model(model, train_loader, val_loader, epochs=15, lr=0.001, device=device)
    
    # 评估模型
    logger.info("开始评估空间模型 | Starting spatial model evaluation")
    results = evaluate_model(model, test_loader, device=device)
    
    # 保存结果
    save_training_results(model, history, results)
    
    logger.info("空间模型训练完成 | Spatial model training completed")
    logger.info(f"最终评估结果 - 损失: {results['loss']:.6f}, RMSE: {results['rmse']:.6f}")
    
    # 演示联合训练
    logger.info("\n===== 开始联合训练演示 =====")
    
    # 创建用于联合训练的模拟数据集
    mock_train_dataset1 = MockSpatialDataset(num_samples=50)
    mock_val_dataset1 = MockSpatialDataset(num_samples=20)
    mock_train_dataset2 = MockSpatialDataset(num_samples=50)
    mock_val_dataset2 = MockSpatialDataset(num_samples=20)
    
    # 创建数据加载器
    mock_train_loader1 = DataLoader(mock_train_dataset1, batch_size=4, shuffle=True)
    mock_val_loader1 = DataLoader(mock_val_dataset1, batch_size=4, shuffle=False)
    mock_train_loader2 = DataLoader(mock_train_dataset2, batch_size=4, shuffle=True)
    mock_val_loader2 = DataLoader(mock_val_dataset2, batch_size=4, shuffle=False)
    
    # 初始化两个不同的空间模型
    model1 = SpatialModel()
    model2 = SpatialModel()
    
    # 设置损失权重
    loss_weights = [0.6, 0.4]  # 模型1的权重为0.6，模型2的权重为0.4
    
    # 执行联合训练
    logger.info(f"开始联合训练两个空间模型，损失权重: {loss_weights}")
    joint_history = train_jointly(
        models=[model1, model2],
        train_loaders=[mock_train_loader1, mock_train_loader2],
        val_loaders=[mock_val_loader1, mock_val_loader2],
        epochs=5,  # 演示用较少的epochs
        lr=0.001,
        device=device,
        loss_weights=loss_weights,
        shared_optimizer=True
    )
    
    # 保存联合训练结果
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存模型1
    save_path1 = f'models/f_spatial_joint_model1_{timestamp}.pth'
    save_training_results(model1, {'joint_training': True}, {}, save_path1)
    
    # 保存模型2
    save_path2 = f'models/f_spatial_joint_model2_{timestamp}.pth'
    save_training_results(model2, {'joint_training': True}, {}, save_path2)
    
    # 保存联合训练历史
    history_save_path = f'models/f_spatial_joint_history_{timestamp}.json'
    os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
    with open(history_save_path, 'w', encoding='utf-8') as f:
        json.dump(joint_history, f, indent=2, default=str)
    
    logger.info("联合训练演示完成 | Joint training demonstration completed")
    logger.info(f"模型1保存路径: {save_path1}")
    logger.info(f"模型2保存路径: {save_path2}")
    logger.info(f"联合训练历史保存路径: {history_save_path}")
    
    # 打印联合训练摘要
    logger.info("\n联合训练摘要 | Joint Training Summary:")
    logger.info(f"- 模型数量: {len(joint_history['models_info'])}")
    logger.info(f"- 损失权重: {joint_history['loss_weights']}")
    logger.info(f"- 训练轮次: {len(joint_history['total_loss'])}")
    logger.info(f"- 最终总损失: {joint_history['total_loss'][-1]:.6f}")
    for i in range(len(joint_history['individual_losses'])):
        logger.info(f"- 模型 {i+1} 最终损失: {joint_history['individual_losses'][i][-1]:.6f}")

if __name__ == '__main__':
    main()
