# Copyright 2025 AGI System Team
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

# 音频处理模型训练程序 / Audio Processing Model Training Program

"""
音频处理模型训练模块
负责训练音频识别、合成和处理的神经网络模型

Audio Processing Model Training Module
Responsible for training neural network models for audio recognition, synthesis and processing
"""

import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from .model import AudioProcessingModel  # 从当前目录导入模型
from .dataset import AudioDataset  # 假设有数据集模块

def train_audio_model(config):
    """
    训练音频处理模型 (增强版)
    支持多任务学习：语音识别、音频合成、音乐处理等
    支持联合训练和外部API集成
    
    Enhanced Audio Processing Model Training
    Supports multi-task learning: speech recognition, audio synthesis, music processing, etc.
    Supports joint training and external API integration
    :param config: 训练配置字典 / Training configuration dictionary
    :return: 训练好的模型 / Trained model
    """
    # 初始化多任务模型 / Initialize multi-task model
    model = AudioProcessingModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dims=config['output_dims'],  # 改为多输出维度 / Changed to multiple output dimensions
        task_types=config['task_types']     # 添加任务类型参数 / Added task types parameter
    )
    
    # 外部API集成 / External API integration
    if config.get('use_external_api', False):
        print("使用外部API模型进行训练 / Using external API model for training")
        model.load_external_model(
            api_url=config['api_url'],
            api_key=config['api_key'],
            model_name=config['model_name']
        )
    
    # 联合训练准备 / Joint training preparation
    if config.get('joint_training', False):
        print(f"联合训练模式: 与{config['joint_models']}模型协同训练 / Joint training with {config['joint_models']}")
        # 初始化数据共享接口 / Initialize data sharing interface
        model.init_joint_training_interface(config['joint_models'])
    
    # 设置优化器和损失函数 / Configure optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    
    # 加载数据集 / Load dataset
    train_dataset = AudioDataset(
        data_dir=config['data_path'],
        sample_rate=config['sample_rate'],
        transform=config.get('transform', None)
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 训练循环 / Training loop
    for epoch in range(config['epochs']):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                # 每100批次打印损失 / Print loss every 100 batches
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")
        
        # 完成epoch训练 / Epoch training completed
        print(f"Epoch {epoch} Completed. Avg Loss: {total_loss/len(train_loader):.4f}")
    
    # 保存训练好的模型 / Save trained model
    torch.save(model.state_dict(), config['save_path'])
    return model

if __name__ == "__main__":
    # 增强版训练配置 / Enhanced training configuration
    config = {
        'data_path': 'data/audio_samples',       # 音频样本路径 / Path to audio samples
        'save_path': 'models/audio_model.pth',   # 模型保存路径 / Model save path
        'input_dim': 128,                        # 输入维度 / Input dimension
        'hidden_dim': 512,                       # 增大隐藏层维度 / Increased hidden layer dimension
        'output_dims': [10, 8, 5],               # 多任务输出维度 [语音识别, 音频合成, 音乐处理] / Multi-task outputs [speech, synthesis, music]
        'task_types': ['recognition', 'synthesis', 'music'], # 任务类型 / Task types
        'sample_rate': 44100,                    # 更高采样率支持 / Higher sample rate support
        'batch_size': 64,                        # 增大批次大小 / Increased batch size
        'learning_rate': 0.0005,                 # 调整学习率 / Adjusted learning rate
        'epochs': 50,                            # 增加训练轮数 / Increased training epochs
        
        # 新增配置选项 / New configuration options
        'use_external_api': False,               # 是否使用外部API / Use external API
        'api_url': '',                           # API地址 / API URL
        'api_key': '',                           # API密钥 / API Key
        'model_name': '',                        # 外部模型名称 / External model name
        'joint_training': False,                 # 联合训练开关 / Joint training switch
        'joint_models': ['B_language', 'D_image'] # 联合训练模型列表 / Joint training models
    }
    
    # 启动训练 / Start training
    trained_model = train_audio_model(config)
    # 训练完成提示 / Training completion notification
    print("Audio model training completed!")
