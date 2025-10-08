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

# 图片视觉处理模型训练程序
# Image Visual Processing Model Training Program

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import json
import datetime
from typing import List, Dict, Tuple, Optional, Any
from .model import ImageModel

class ImageDataset(Dataset):
    """图像数据集类 | Image Dataset Class"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 返回图像和文件名作为标签
        # Return image and filename as label
        return image, self.image_files[idx]

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                epochs: int = 10, 
                lr: float = 0.001, 
                is_joint_training: bool = False, 
                joint_training_info: Optional[Dict[str, Any]] = None) -> Dict[str, List[float]]:
    """训练图像模型 | Train image model
    
    Args:
        model: 要训练的图像模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        is_joint_training: 是否为联合训练模式
        joint_training_info: 联合训练相关信息
        
    Returns:
        训练历史记录，包含训练损失、验证损失和验证准确率
    """
    # 定义损失函数和优化器
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练历史记录
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # 训练循环
    # Training loop
    for epoch in range(epochs):
        if not is_joint_training:
            print(f'开始训练轮次 {epoch+1}/{epochs} | Starting epoch {epoch+1}/{epochs}')
        
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            # Forward pass
            outputs = model(inputs)
            
            # 计算损失
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
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
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 记录训练历史
        # Record training history
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # 打印训练统计信息
        # Print training statistics
        if not is_joint_training:
            print(f'Epoch {epoch+1}/{epochs}, '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Val Acc: {val_accuracy:.2f}%')
    
    return history

def save_training_results(model: nn.Module, 
                          history: Dict[str, List[float]], 
                          model_name: str, 
                          is_joint: bool = False) -> str:
    """保存训练结果 | Save training results"""
    # 创建保存目录
    # Create save directory
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成保存文件名
    # Generate save filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "joint" if is_joint else "single"
    model_filename = os.path.join(save_dir, f"{model_name}_{suffix}_{timestamp}.pth")
    history_filename = os.path.join(save_dir, f"{model_name}_{suffix}_{timestamp}_history.json")
    
    # 保存模型
    # Save model
    torch.save(model.state_dict(), model_filename)
    
    # 保存历史记录
    # Save history
    with open(history_filename, 'w') as f:
        json.dump(history, f)
    
    print(f"模型已保存至: {model_filename}")
    print(f"训练历史已保存至: {history_filename}")
    
    return model_filename


def train_jointly(models: List[nn.Module], 
                   train_loaders: List[DataLoader], 
                   val_loaders: List[DataLoader], 
                   epochs: int = 10, 
                   lrs: Optional[List[float]] = None, 
                   loss_weights: Optional[List[float]] = None) -> Dict[str, Any]:
    """多模型联合训练函数 | Multi-model joint training function
    
    Args:
        models: 要联合训练的模型列表
        train_loaders: 对应每个模型的训练数据加载器列表
        val_loaders: 对应每个模型的验证数据加载器列表
        epochs: 训练轮数
        lrs: 对应每个模型的学习率列表
        loss_weights: 每个模型损失的权重列表
        
    Returns:
        联合训练结果，包含每个模型的训练历史和联合训练统计信息
    """
    # 检查输入参数
    # Check input parameters
    if len(models) != len(train_loaders) or len(models) != len(val_loaders):
        raise ValueError("模型数量必须与数据加载器数量匹配 | Number of models must match number of data loaders")
    
    num_models = len(models)
    
    # 设置默认学习率
    # Set default learning rates
    if lrs is None:
        lrs = [0.001] * num_models
    elif len(lrs) != num_models:
        raise ValueError("学习率数量必须与模型数量匹配 | Number of learning rates must match number of models")
    
    # 设置默认损失权重
    # Set default loss weights
    if loss_weights is None:
        loss_weights = [1.0 / num_models] * num_models
    elif len(loss_weights) != num_models:
        raise ValueError("损失权重数量必须与模型数量匹配 | Number of loss weights must match number of models")
    
    # 归一化损失权重
    # Normalize loss weights
    total_weight = sum(loss_weights)
    loss_weights = [w / total_weight for w in loss_weights]
    
    # 为每个模型创建优化器
    # Create optimizers for each model
    optimizers = []
    criterion = nn.CrossEntropyLoss()
    
    for i, model in enumerate(models):
        optimizer = optim.Adam(model.parameters(), lr=lrs[i])
        optimizers.append(optimizer)
    
    # 为每个模型创建学习率调度器
    # Create learning rate schedulers for each model
    schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
                  for optimizer in optimizers]
    
    # 存储每个模型的历史记录
    # Store history for each model
    all_histories = {}
    for i, model in enumerate(models):
        all_histories[f'model_{i+1}'] = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    # 联合训练循环
    # Joint training loop
    for epoch in range(epochs):
        print(f'开始联合训练轮次 {epoch+1}/{epochs} | Starting joint training epoch {epoch+1}/{epochs}')
        
        # 训练阶段
        # Training phase
        joint_train_loss = 0.0
        
        for i, model in enumerate(models):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loaders[i]:
                optimizers[i].zero_grad()
                
                # 前向传播
                # Forward pass
                outputs = model(inputs)
                
                # 计算损失
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # 应用权重并反向传播
                # Apply weight and backward pass
                (loss * loss_weights[i]).backward()
                optimizers[i].step()
                
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loaders[i])
            all_histories[f'model_{i+1}']['train_loss'].append(avg_train_loss)
            joint_train_loss += avg_train_loss * loss_weights[i]
        
        # 验证阶段
        # Validation phase
        joint_val_loss = 0.0
        joint_val_accuracy = 0.0
        
        for i, model in enumerate(models):
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loaders[i]:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loaders[i])
            val_accuracy = 100 * correct / total
            
            all_histories[f'model_{i+1}']['val_loss'].append(avg_val_loss)
            all_histories[f'model_{i+1}']['val_accuracy'].append(val_accuracy)
            
            joint_val_loss += avg_val_loss * loss_weights[i]
            joint_val_accuracy += val_accuracy * loss_weights[i]
            
            # 更新学习率调度器
            # Update learning rate scheduler
            schedulers[i].step(avg_val_loss)
        
        # 打印联合训练统计信息
        # Print joint training statistics
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Joint Train Loss: {joint_train_loss:.4f}, '
              f'Joint Val Loss: {joint_val_loss:.4f}, '
              f'Joint Val Acc: {joint_val_accuracy:.2f}%')
    
    # 准备联合训练结果
    # Prepare joint training results
    joint_results = {
        'joint_training': True,
        'num_models': num_models,
        'loss_weights': loss_weights,
        'epochs': epochs,
        'histories': all_histories,
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    return joint_results


def main():
    """主训练函数 | Main training function"""
    # 数据预处理
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    # Create datasets
    try:
        train_dataset = ImageDataset('data/train', transform=transform)
        val_dataset = ImageDataset('data/val', transform=transform)
        
        # 创建数据加载器
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 初始化模型
        # Initialize model
        model = ImageModel()
        
        print("开始单独训练模式 | Starting individual training mode")
        # 训练模型
        # Train model
        history = train_model(model, train_loader, val_loader, epochs=20)
        
        # 保存模型和训练结果
        # Save model and training results
        save_training_results(model, history, "d_image")
        print("模型训练完成并保存 | Model training completed and saved")
    except Exception as e:
        print(f"无法加载数据集，使用示例数据进行演示 | Failed to load dataset, using example for demonstration: {e}")
        
        # 示例：展示联合训练功能
        # Example: Show joint training functionality
        print("\n===== 联合训练演示 =====")
        
        # 定义模拟图像数据集类
        # Define mock image dataset class
        class MockImageDataset(Dataset):
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # 生成随机图像数据和标签
                # Generate random image data and labels
                image = torch.randn(3, 224, 224)
                label = torch.randint(0, 10, (1,)).item()  # 假设有10个类别
                return image, label
        
        # 创建模拟数据集和加载器
        # Create mock datasets and loaders
        mock_train_dataset1 = MockImageDataset()
        mock_val_dataset1 = MockImageDataset(size=50)
        mock_train_loader1 = DataLoader(mock_train_dataset1, batch_size=16, shuffle=True)
        mock_val_loader1 = DataLoader(mock_val_dataset1, batch_size=16, shuffle=False)
        
        mock_train_dataset2 = MockImageDataset()
        mock_val_dataset2 = MockImageDataset(size=50)
        mock_train_loader2 = DataLoader(mock_train_dataset2, batch_size=16, shuffle=True)
        mock_val_loader2 = DataLoader(mock_val_dataset2, batch_size=16, shuffle=False)
        
        # 初始化两个模型进行联合训练
        # Initialize two models for joint training
        models = [ImageModel(), ImageModel()]
        train_loaders = [mock_train_loader1, mock_train_loader2]
        val_loaders = [mock_val_loader1, mock_val_loader2]
        
        # 设置学习率和损失权重
        # Set learning rates and loss weights
        lrs = [0.001, 0.001]
        loss_weights = [0.6, 0.4]  # 模型1权重稍高
        
        # 执行联合训练
        # Perform joint training
        print("开始联合训练 | Starting joint training")
        joint_results = train_jointly(
            models=models,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            epochs=5,
            lrs=lrs,
            loss_weights=loss_weights
        )
        
        # 评估联合训练结果
        # Evaluate joint training results
        print("\n联合训练结果评估 | Joint Training Results Evaluation:")
        print(f"模型数量: {joint_results['num_models']}")
        print(f"损失权重: {joint_results['loss_weights']}")
        print(f"总训练轮次: {joint_results['epochs']}")
        
        # 保存联合训练结果
        # Save joint training results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存每个模型
        # Save each model
        for i, model in enumerate(models):
            model_filename = os.path.join(results_dir, f"d_image_joint_model_{i+1}_{timestamp}.pth")
            torch.save(model.state_dict(), model_filename)
            print(f"联合训练模型 {i+1} 已保存至: {model_filename}")
        
        # 保存联合训练历史
        # Save joint training history
        history_filename = os.path.join(results_dir, f"d_image_joint_history_{timestamp}.json")
        with open(history_filename, 'w') as f:
            json.dump(joint_results, f)
        print(f"联合训练历史已保存至: {history_filename}")

if __name__ == '__main__':
    main()
