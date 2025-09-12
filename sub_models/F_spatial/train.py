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
from typing import Dict, List, Any, Optional, Tuple
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
               epochs: int = 10, lr: float = 0.001, device: str = 'cpu') -> Dict:
    """训练空间模型 | Train spatial model
    参数:
        model: 空间模型 | Spatial model
        train_loader: 训练数据加载器 | Training data loader
        val_loader: 验证数据加载器 | Validation data loader
        epochs: 训练轮数 | Number of epochs
        lr: 学习率 | Learning rate
        device: 训练设备 | Training device
    返回:
        训练历史字典 | Training history dictionary
    """
    criterion = nn.MSELoss()
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
            
            optimizer.zero_grad()
            outputs = model(left_img, right_img)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 验证阶段
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

if __name__ == '__main__':
    main()
