#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A_Management模型训练脚本
"""

import os
import sys
import torch
import json
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("A_Management_Trainer")

class ManagementDataset(Dataset):
    """管理模型训练数据集"""
    
    def __init__(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'input': sample['input'],
            'task_type': sample['task_type'],
            'expected_output': sample['expected_output']
        }

def collate_fn(batch):
    """数据批处理函数"""
    inputs = [item['input'] for item in batch]
    task_types = [item['task_type'] for item in batch]
    expected_outputs = [item['expected_output'] for item in batch]
    
    return {
        'inputs': inputs,
        'task_types': task_types,
        'expected_outputs': expected_outputs
    }

class ManagementModel(nn.Module):
    """管理模型的简化版本用于训练"""
    
    def __init__(self, hidden_dim=512):
        super(ManagementModel, self).__init__()
        # 假设输入是768维的嵌入向量
        self.layer1 = nn.Linear(768, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 256)
        self.output = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x
    
    def process_task(self, input_features):
        """处理任务的简化版本"""
        # 这里应该有真实的处理逻辑，但为了示例我们返回一个模拟的响应
        return {
            'manager_decision': {
                'response': 'This is a simulated response from the management model',
                'confidence': 0.9
            }
        }
    
    def save_model(self, path):
        """保存模型权重"""
        torch.save({
            'model_state_dict': self.state_dict()
        }, path)
    
    def load_model(self, path):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])

def create_management_model(hidden_dim=512):
    """创建管理模型"""
    return ManagementModel(hidden_dim=hidden_dim)

def train_model(model, dataloader, optimizer, criterion, device, epochs=10):
    """训练模型"""
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # 模拟输入特征（在实际应用中应该有真实的特征提取）
            batch_size = len(batch['inputs'])
            # 随机生成输入特征作为示例
            input_features = torch.randn(batch_size, 768).to(device)
            
            # 前向传播
            outputs = model(input_features)
            
            # 模拟目标（在实际应用中应该使用真实的标签）
            targets = torch.tensor([0.9] * batch_size, dtype=torch.float32).unsqueeze(1).to(device)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size
        
        avg_epoch_loss = epoch_loss / len(dataloader.dataset)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs']
            task_types = batch['task_types']
            expected_outputs = batch['expected_outputs']
            
            batch_loss = 0.0
            for input_text, task_type, expected in zip(inputs, task_types, expected_outputs):
                try:
                    input_features = {'text': input_text}
                    output = model.process_task(input_features)
                    
                    if 'manager_decision' in output and 'confidence' in output['manager_decision']:
                        confidence_diff = (output['manager_decision']['confidence'] - expected.get('confidence', 0.9)) ** 2
                        batch_loss += confidence_diff
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
            
            total_loss += batch_loss
    
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Evaluation completed. Average Loss: {avg_loss:.4f}")

def main():
    """主函数"""
    # 加载配置
    config_file = os.path.join(os.path.dirname(__file__), 'model_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {
            'hidden_dim': 512,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        }
    
    # 创建模型
    model = create_management_model(hidden_dim=config.get('hidden_dim', 512))
    
    # 加载现有权重（如果有）
    weights_file = os.path.join(os.path.dirname(__file__), 'model_weights', 'a_management_model.pth')
    if os.path.exists(weights_file):
        try:
            model.load_model(weights_file)
            logger.info(f"Loaded existing weights from {weights_file}")
        except Exception as e:
            logger.warning(f"Failed to load existing weights: {e}")
    
    # 准备数据集
    data_file = os.path.join(os.path.dirname(__file__), 'training_data', 'sample_training_data.json')
    if not os.path.exists(data_file):
        logger.error(f"Training data file not found at {data_file}")
        logger.error("Please run initialize_a_management_model.py first to create sample training data")
        sys.exit(1)
    
    dataset = ManagementDataset(data_file)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.get('batch_size', 32), 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    criterion = nn.MSELoss()
    
    # 开始训练
    logger.info(f"Starting training with {len(dataset)} samples")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, dataloader, optimizer, criterion, device, epochs=config.get('epochs', 10))
    
    # 保存最终模型
    final_weights_file = os.path.join(os.path.dirname(__file__), 'model_weights', 'a_management_model_trained.pth')
    model.save_model(final_weights_file)
    logger.info(f"Training completed! Final model saved to {final_weights_file}")
    
    # 评估模型
    evaluate_model(model, dataloader, device)

if __name__ == '__main__':
    main()
