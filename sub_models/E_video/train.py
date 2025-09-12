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

# 视频流视觉处理模型训练程序
# Video Stream Visual Processing Model Training Program

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import json
import logging
from typing import Dict, List, Any, Optional
from .model import VideoModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("E_video_Trainer")

class VideoDataset(Dataset):
    """视频数据集类 | Video Dataset Class"""
    def __init__(self, data_dir, frame_count=16, transform=None):
        self.data_dir = data_dir
        self.frame_count = frame_count
        self.transform = transform
        self.video_files = []
        self.labels = []
        self.label_map = {}
        
        # 加载数据集信息
        self._load_dataset_info()
    
    def _load_dataset_info(self):
        """加载数据集信息 | Load dataset information"""
        info_file = os.path.join(self.data_dir, "dataset_info.json")
        
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.video_files = data.get("video_files", [])
                    self.labels = data.get("labels", [])
                    self.label_map = data.get("label_map", {})
            except Exception as e:
                logger.error(f"加载数据集信息错误: {str(e)}")
        
        # 如果没有数据集信息，扫描目录中的视频文件
        if not self.video_files:
            self.video_files = [f for f in os.listdir(self.data_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            # 为每个视频分配一个随机标签（在实际应用中应有真实的标签）
            self.labels = [hash(f) % 5 for f in self.video_files]  # 5个类别
            self.label_map = {i: f"category_{i}" for i in range(5)}
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.data_dir, self.video_files[idx])
        label = self.labels[idx]
        
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # 提取固定数量的帧
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                # 生成模拟视频帧
                frames = self._generate_dummy_frames()
            else:
                frame_indices = np.linspace(0, total_frames-1, self.frame_count, dtype=int)
                
                for i in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (224, 224))
                        if self.transform:
                            frame = self.transform(frame)
                        frames.append(frame)
                    else:
                        # 如果读取失败，添加黑色帧
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            cap.release()
            
            # 转换为张量
            frames = np.array(frames)
            frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
            
            return frames, label
            
        except Exception as e:
            logger.error(f"处理视频数据错误: {str(e)}")
            # 返回空数据
            return torch.zeros((self.frame_count, 3, 224, 224), dtype=torch.float32), 0
    
    def _generate_dummy_frames(self):
        """生成虚拟视频帧 | Generate dummy video frames"""
        frames = []
        for _ in range(self.frame_count):
            # 生成随机颜色的帧
            frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            frames.append(frame)
        return frames

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """训练视频模型 | Train video model
    参数:
        model: 视频模型 | Video model
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    model.to(device)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录历史
        train_history['train_loss'].append(running_loss / len(train_loader))
        train_history['val_loss'].append(val_loss / len(val_loader))
        train_history['val_acc'].append(100 * correct / total)
        train_history['learning_rate'].append(current_lr)
        
        logger.info(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {100*correct/total:.2f}%, '
              f'LR: {current_lr:.6f}')
    
    return train_history

def evaluate_model(model, test_loader, device='cpu'):
    """评估视频模型 | Evaluate video model
    参数:
        model: 视频模型 | Video model
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
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

def save_training_results(model, history, results, save_path='models/e_video_model.pth'):
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
        'training_history': history,
        'evaluation_results': results
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
    train_dataset = VideoDataset('data/train')
    val_dataset = VideoDataset('data/val')
    test_dataset = VideoDataset('data/test')
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 初始化模型
    num_classes = len(set(train_dataset.labels)) if train_dataset.labels else 5
    model = VideoModel(num_classes=num_classes)
    
    # 训练模型
    logger.info("开始训练视频模型 | Starting video model training")
    history = train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device=device)
    
    # 评估模型
    logger.info("开始评估视频模型 | Starting video model evaluation")
    results = evaluate_model(model, test_loader, device=device)
    
    # 保存结果
    save_training_results(model, history, results)
    
    logger.info("视频模型训练完成 | Video model training completed")
    logger.info(f"最终评估结果 - 损失: {results['loss']:.4f}, 准确率: {results['accuracy']:.2f}%")

if __name__ == '__main__':
    import time
    main()
