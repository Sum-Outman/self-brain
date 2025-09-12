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

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """训练图像模型 | Train image model"""
    # 定义损失函数和优化器
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    # Training loop
    for epoch in range(epochs):
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
        
        # 打印训练统计信息
        # Print training statistics
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {100*correct/total:.2f}%')

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
    train_dataset = ImageDataset('data/train', transform=transform)
    val_dataset = ImageDataset('data/val', transform=transform)
    
    # 创建数据加载器
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    # Initialize model
    model = ImageModel()
    
    # 训练模型
    # Train model
    train_model(model, train_loader, val_loader, epochs=20)
    
    # 保存模型
    # Save model
    torch.save(model.state_dict(), 'd_image_model.pth')
    print("模型训练完成并保存 | Model training completed and saved")

if __name__ == '__main__':
    main()
