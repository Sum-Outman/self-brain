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
from .model import SensorModel

class SensorDataset(Dataset):
    """传感器数据集类 | Sensor Dataset Class"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        data = pd.read_csv(data_path)
        
        # 提取传感器数据特征
        # Extract sensor data features
        features = data[['temperature', 'humidity', 'acceleration_x', 'acceleration_y', 
                        'acceleration_z', 'pressure', 'distance', 'light_intensity']].values
        
        # 转换为张量
        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32)
        
        # 目标值（假设最后一列是目标）
        # Target values (assuming last column is target)
        targets = data['target'].values
        targets = torch.tensor(targets, dtype=torch.long)
        
        return features, targets

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """训练传感器模型 | Train sensor model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练历史记录
    # Training history records
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            # Forward pass
            outputs = model(inputs)
            
            # 计算损失
            # Calculate loss
            loss = criterion(outputs, targets)
            
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
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # 记录训练统计信息
        # Record training statistics
        train_loss = running_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss_avg)
        history['val_accuracy'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss_avg:.4f}, '
              f'Val Acc: {val_acc:.2f}%')
    
    return history

def main():
    """主训练函数 | Main training function"""
    # 创建数据集
    # Create datasets
    train_dataset = SensorDataset('data/train')
    val_dataset = SensorDataset('data/val')
    
    # 创建数据加载器
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    # Initialize model
    model = SensorModel()
    
    # 训练模型
    # Train model
    history = train_model(model, train_loader, val_loader, epochs=20)
    
    # 保存模型
    # Save model
    torch.save(model.state_dict(), 'g_sensor_model.pth')
    
    # 保存训练历史
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('g_sensor_training_history.csv', index=False)
    
    print("传感器模型训练完成并保存 | Sensor model training completed and saved")
    print(f"最终验证准确率: {history['val_accuracy'][-1]:.2f}% | Final validation accuracy: {history['val_accuracy'][-1]:.2f}%")

if __name__ == '__main__':
    main()