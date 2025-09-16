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

# Motion and Actuator Control Model Training Program

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
from .model import MotionModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("J_motion_Trainer")

class MotionDataset(Dataset):
    """Motion Control Dataset Class"""
    def __init__(self, data_dir: str, sequence_length: int = 10):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.data_files = []
        self.dataset_info = {}
        self.command_map = {}
        
        # 加载数据集信息
        self._load_dataset_info()
    
    def _load_dataset_info(self):
        """Load dataset information"""
        info_file = os.path.join(self.data_dir, "dataset_info.json")
        
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    self.dataset_info = json.load(f)
                    self.data_files = self.dataset_info.get("data_files", [])
                    self.command_map = self.dataset_info.get("command_map", {})
            except Exception as e:
                logger.error(f"Error loading dataset information: {str(e)}")
        
        # 如果没有数据集信息，扫描目录中的数据文件
        if not self.data_files:
            self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            # 如果没有数据文件，创建一些模拟数据信息
            if not self.data_files:
                self.data_files = [f"motion_data_{i}.csv" for i in range(10)]
                self.command_map = {
                    0: "move_forward",
                    1: "move_backward",
                    2: "turn_left",
                    3: "turn_right",
                    4: "stop",
                    5: "accelerate",
                    6: "decelerate",
                    7: "rotate_clockwise",
                    8: "rotate_counterclockwise",
                    9: "hold_position"
                }
                self.dataset_info = {
                    "data_files": self.data_files,
                    "command_map": self.command_map,
                    "description": "Simulated motion control data",
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
            
            # 提取传感器数据特征
            sensor_features = self._extract_sensor_features(data)
            
            # 提取控制命令标签
            command_labels = self._extract_command_labels(data)
            
            # 转换为张量
            sensor_features = torch.tensor(sensor_features, dtype=torch.float32)
            command_labels = torch.tensor(command_labels, dtype=torch.long)
            
            return sensor_features, command_labels
            
        except Exception as e:
            logger.error(f"Error processing motion control data: {str(e)}")
            # 返回空数据
            dummy_features = torch.zeros((self.sequence_length, 9), dtype=torch.float32)
            dummy_labels = torch.zeros(self.sequence_length, dtype=torch.long)
            return dummy_features, dummy_labels
    
    def _extract_sensor_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract sensor features"""
        # 定义传感器特征列
        feature_columns = ['position_x', 'position_y', 'position_z', 
                          'velocity_x', 'velocity_y', 'velocity_z',
                          'acceleration_x', 'acceleration_y', 'acceleration_z']
        
        # 如果数据中缺少某些列，使用默认值填充
        for col in feature_columns:
            if col not in data.columns:
                data[col] = np.random.uniform(-1, 1, len(data))
        
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
        """Extract command labels"""
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
        """Generate dummy data"""
        # 创建模拟运动控制数据
        num_samples = self.sequence_length
        data = {
            'position_x': np.random.uniform(-10, 10, num_samples),
            'position_y': np.random.uniform(-10, 10, num_samples),
            'position_z': np.random.uniform(-5, 5, num_samples),
            'velocity_x': np.random.uniform(-2, 2, num_samples),
            'velocity_y': np.random.uniform(-2, 2, num_samples),
            'velocity_z': np.random.uniform(-1, 1, num_samples),
            'acceleration_x': np.random.uniform(-0.5, 0.5, num_samples),
            'acceleration_y': np.random.uniform(-0.5, 0.5, num_samples),
            'acceleration_z': np.random.uniform(-0.2, 0.2, num_samples),
            'command': np.random.choice(list(self.command_map.values()), num_samples)
        }
        return pd.DataFrame(data)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int = 10, lr: float = 0.001, device: str = 'cpu') -> Dict:
    """Train motion control model
    
    Args:
        model: Motion control model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Training device
    
    Returns:
        Training history dictionary
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
        # Training phase
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
                total += targets.numel()
                correct += (predicted == targets).sum().item()
        
        # Update learning rate
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
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
    """Evaluate motion control model
    
    Args:
        model: Motion control model
        test_loader: Test data loader
        device: Evaluation device
    
    Returns:
        Evaluation result dictionary
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
                         save_path: str = 'models/j_motion_model.pth'):
    """Save training results
    
    Args:
        model: Trained model
        history: Training history
        results: Evaluation results
        save_path: Save path
    """
    # Create model directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'training_history': history,
        'evaluation_results': results,
        'timestamp': time.time()
    }, save_path)
    
    # Save training log
    log_path = save_path.replace('.pth', '_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'training_history': history,
            'evaluation_results': results,
            'timestamp': time.time()
        }, f, indent=2)
    
    logger.info(f"Model and training results saved: {save_path}")

def train_jointly(models: List[nn.Module], train_loaders: List[DataLoader], val_loaders: List[DataLoader],
                  epochs: int = 10, lr: float = 0.001, device: str = 'cpu',
                  loss_weights: Optional[List[float]] = None) -> List[Dict]:
    """Jointly train multiple models
    
    Args:
        models: List of models
        train_loaders: List of training data loaders
        val_loaders: List of validation data loaders
        epochs: Number of epochs
        lr: Learning rate
        device: Training device
        loss_weights: Loss weights for each model
    
    Returns:
        List of training history dictionaries for each model
    """
    # Check parameters
    if len(models) != len(train_loaders) or len(models) != len(val_loaders):
        raise ValueError("Number of models must match number of loaders")
    
    # If no loss weights provided, use equal weights
    if loss_weights is None:
        loss_weights = [1.0] * len(models)
    elif len(loss_weights) != len(models):
        raise ValueError("Number of loss weights must match number of models")
    
    # Create optimizers and schedulers for each model
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]
    schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
                 for optimizer in optimizers]
    
    # Create training history for each model
    histories = [{
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    } for _ in range(len(models))]
    
    # Move all models to device
    for model in models:
        model.to(device)
    
    for epoch in range(epochs):
        # Joint training phase
        for model in models:
            model.train()
        
        running_losses = [0.0] * len(models)
        
        # Iterate through training data
        for i, (inputs, targets) in enumerate(zip(*train_loaders)):
            # Zero gradients
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            # Forward pass and loss calculation
            total_loss = 0.0
            for j, (model, input_data, target_data, weight) in enumerate(
                    zip(models, inputs, targets, loss_weights)):
                input_data = input_data.to(device)
                target_data = target_data.to(device)
                
                outputs = model(input_data)
                loss = nn.CrossEntropyLoss()(outputs, target_data)
                weighted_loss = loss * weight
                running_losses[j] += loss.item()
                total_loss += weighted_loss
            
            # Backward pass and optimization
            total_loss.backward()
            for optimizer in optimizers:
                optimizer.step()
        
        # Joint validation phase
        for model in models:
            model.eval()
        
        val_losses = [0.0] * len(models)
        correct = [0] * len(models)
        total = [0] * len(models)
        
        with torch.no_grad():
            for inputs, targets in zip(*val_loaders):
                for j, (model, input_data, target_data) in enumerate(
                        zip(models, inputs, targets)):
                    input_data = input_data.to(device)
                    target_data = target_data.to(device)
                    
                    outputs = model(input_data)
                    loss = nn.CrossEntropyLoss()(outputs, target_data)
                    val_losses[j] += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total[j] += target_data.numel()
                    correct[j] += (predicted == target_data).sum().item()
        
        # Update learning rate
        avg_val_losses = [val_loss / len(val_loaders[j]) for j, val_loss in enumerate(val_losses)]
        for j, scheduler in enumerate(schedulers):
            scheduler.step(avg_val_losses[j])
        
        # Record history
        avg_train_losses = [running_loss / len(train_loaders[j]) for j, running_loss in enumerate(running_losses)]
        avg_val_accuracies = [100 * corr / total_j if total_j > 0 else 0 
                             for corr, total_j in zip(correct, total)]
        current_lrs = [optimizer.param_groups[0]['lr'] for optimizer in optimizers]
        
        for j in range(len(models)):
            histories[j]['train_loss'].append(avg_train_losses[j])
            histories[j]['val_loss'].append(avg_val_losses[j])
            histories[j]['val_accuracy'].append(avg_val_accuracies[j])
            histories[j]['learning_rate'].append(current_lrs[j])
        
        # Logging
        logger.info(f'Joint Training - Epoch {epoch+1}/{epochs}')
        for j in range(len(models)):
            logger.info(f'  Model {j+1} - Train Loss: {avg_train_losses[j]:.4f}, ' \
                       f'Val Loss: {avg_val_losses[j]:.4f}, ' \
                       f'Val Acc: {avg_val_accuracies[j]:.2f}%, ' \
                       f'LR: {current_lrs[j]:.8f}')
    
    return histories

def main():
    """Main training function"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # Create datasets
    train_dataset = MotionDataset('data/train')
    val_dataset = MotionDataset('data/val')
    test_dataset = MotionDataset('data/test')
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    logger.info(f"命令映射: {train_dataset.command_map}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    num_classes = len(train_dataset.command_map) if train_dataset.command_map else 10
    model = MotionModel(input_size=9, hidden_size=64, num_classes=num_classes, num_layers=2)
    
    # Train model (individual training mode)
    logger.info("开始训练运动控制模型（单独训练模式） | Starting motion control model training (individual mode)")
    history = train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device=device)
    
    # Evaluate model
    logger.info("开始评估运动控制模型 | Starting motion control model evaluation")
    results = evaluate_model(model, test_loader, device=device)
    
    # Save results
    save_training_results(model, history, results)
    
    logger.info("运动控制模型训练完成 | Motion control model training completed")
    logger.info(f"Final evaluation results - Loss: {results['loss']:.4f}, Accuracy: {results['accuracy']:.2f}%")
    
    # Example: How to use the joint training functionality
    # Note: In a real application, you would provide multiple different models and corresponding datasets
    # The following code is only an example to demonstrate how to call the joint training function
    logger.info("\n--- Joint Training Functionality Example ---")
    try:
        # Create another model of the same type for demonstration purposes
        secondary_model = MotionModel(input_size=9, hidden_size=64, num_classes=num_classes, num_layers=2)
        
        # Prepare model list and data loader lists
        models_list = [model, secondary_model]
        train_loaders_list = [train_loader, train_loader]  # In a real app, use different datasets
        val_loaders_list = [val_loader, val_loader]        # In a real app, use different datasets
        
        # Configure loss weights
        loss_weights = [0.6, 0.4]
        
        # Execute joint training
        logger.info("Starting joint training example")
        joint_histories = train_jointly(
            models=models_list,
            train_loaders=train_loaders_list,
            val_loaders=val_loaders_list,
            epochs=5,  # Use fewer epochs for demonstration
            lr=0.001,
            device=device,
            loss_weights=loss_weights
        )
        
        # Evaluate jointly trained models
        logger.info("Evaluating jointly trained models")
        for i, trained_model in enumerate(models_list):
            joint_results = evaluate_model(trained_model, test_loader, device=device)
            logger.info(f"Model {i+1} - Post-joint training evaluation results: Loss: {joint_results['loss']:.4f}, Accuracy: {joint_results['accuracy']:.2f}%")
            
            # Save jointly trained model
            joint_save_path = f'models/j_motion_joint_model_{i+1}.pth'
            save_training_results(trained_model, joint_histories[i], joint_results, joint_save_path)
    except Exception as e:
        logger.error(f"Error executing joint training example: {str(e)}")
        logger.info("In a real application, ensure you provide correct model lists and data loader lists")

if __name__ == '__main__':
    main()
