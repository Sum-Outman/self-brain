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
import json
import torch
import torchaudio
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
from .model import AudioProcessingModel  # 从当前目录导入模型
from .dataset import AudioDataset  # 假设有数据集模块

def train_audio_model(config, joint_training_info=None):
    """
    训练音频处理模型 (增强版)
    支持多任务学习：语音识别、音频合成、音乐处理等
    支持联合训练和外部API集成
    
    Enhanced Audio Processing Model Training
    Supports multi-task learning: speech recognition, audio synthesis, music processing, etc.
    Supports joint training and external API integration
    :param config: 训练配置字典 / Training configuration dictionary
    :param joint_training_info: 联合训练信息 / Joint training information
    :return: 训练结果字典 / Training result dictionary
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
    if config.get('joint_training', False) or joint_training_info:
        models_to_joint = config.get('joint_models', []) if config.get('joint_training', False) else []
        if joint_training_info:
            # 从联合训练信息中提取伙伴模型 / Extract partner models from joint training info
            models_to_joint = list(joint_training_info.keys())
        
        print(f"联合训练模式: 与{models_to_joint}模型协同训练 / Joint training with {models_to_joint}")
        # 初始化数据共享接口 / Initialize data sharing interface
        model.init_joint_training_interface(models_to_joint)
    
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
    
    # 如果是联合训练模式，确保日志中明确标识 / Ensure clear identification in logs for joint training
    training_mode = "联合训练" if (config.get('joint_training', False) or joint_training_info) else "单独训练"
    print(f"开始{training_mode}音频模型 / Starting {training_mode} for audio model")
    
    # 训练循环 / Training loop
    train_loss_history = []
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
                print(f"{training_mode} - Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")
        
        avg_loss = total_loss/len(train_loader)
        train_loss_history.append(avg_loss)
        
        # 完成epoch训练 / Epoch training completed
        print(f"{training_mode} - Epoch {epoch} Completed. Avg Loss: {avg_loss:.4f}")
    
    # 保存训练好的模型 / Save trained model
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    torch.save(model.state_dict(), config['save_path'])
    
    # 创建训练结果字典 / Create training result dictionary
    training_date = datetime.now().isoformat()
    result = {
        "status": "success",
        "message": f"音频模型{training_mode}完成",
        "model_path": config['save_path'],
        "training_date": training_date,
        "config": config,
        "metrics": {
            "final_loss": train_loss_history[-1],
            "min_loss": min(train_loss_history),
            "epochs": len(train_loss_history)
        },
        "training_history": {
            "loss": train_loss_history
        }
    }
    
    # 保存训练结果日志 / Save training result log
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/audio_model_{training_mode}_{training_date.replace(':', '_')}.json"
    
    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    result["log_path"] = log_filename
    
    return result

def train_jointly(models: List[Any], train_datasets: List[Any], 
                 val_datasets: List[Any] = None, config: Dict = None, 
                 loss_weights: List[float] = None) -> Dict:
    """与其他模型联合训练 / Joint training with other models
    
    参数:
        models: 参与联合训练的模型列表 / List of models for joint training
        train_datasets: 每个模型对应的训练数据集 / Training datasets for each model
        val_datasets: 每个模型对应的验证数据集 / Validation datasets for each model
        config: 训练配置 / Training configuration
        loss_weights: 每个模型的损失权重 / Loss weights for each model
    
    返回:
        训练结果字典 / Training result dictionary
    """
    if config is None:
        config = {}
    
    # 检查输入一致性 / Check input consistency
    if len(models) != len(train_datasets):
        raise ValueError("模型数量和训练数据集数量必须匹配 / Number of models and training datasets must match")
    
    if val_datasets and len(val_datasets) != len(models):
        raise ValueError("验证数据集数量必须与模型数量匹配 / Number of validation datasets must match number of models")
    
    # 初始化默认损失权重 / Initialize default loss weights if not provided
    if loss_weights is None:
        loss_weights = [1.0] * len(models)
    elif len(loss_weights) != len(models):
        raise ValueError("损失权重数量必须与模型数量匹配 / Number of loss weights must match number of models")
    
    # 归一化损失权重 / Normalize loss weights
    total_weight = sum(loss_weights)
    loss_weights = [w / total_weight for w in loss_weights]
    
    # 记录联合训练信息 / Log joint training information
    print(f"开始多模型联合训练 / Starting joint training with {len(models)} models")
    print(f"损失权重分配 / Loss weight distribution: {loss_weights}")
    
    # 初始化联合训练结果 / Initialize joint training results
    joint_results = {
        "status": "success",
        "message": "联合训练完成 / Joint training completed",
        "individual_results": [],
        "joint_metrics": {}
    }
    
    # 为每个模型准备联合训练数据 / Prepare joint training data for each model
    for i, (model, train_dataset) in enumerate(zip(models, train_datasets)):
        print(f"处理模型 {i+1} 的训练数据 / Processing training data for model {i+1}")
        
        # 为每个模型创建联合训练伙伴信息 / Create joint training partner info for each model
        joint_training_info = {}
        for j, partner_model in enumerate(models):
            if i != j:  # 不包含自身 / Exclude self
                # 根据伙伴模型类型添加相应信息 / Add appropriate info based on partner model type
                partner_name = f"model_{j+1}"
                joint_training_info[partner_name] = {
                    "model_type": "audio" if j == 0 else "partner",
                    "weight": loss_weights[j]
                }
        
        # 训练当前模型，传入其他模型的信息 / Train current model with other models' info
        # 注意：C_audio模型的train函数与其他模型不同，我们需要调用对应的训练函数
        # Note: C_audio model's train function is different from other models, we need to call the corresponding training function
        if hasattr(model, 'train'):
            result = model.train(train_dataset, val_datasets[i] if val_datasets else None, config, joint_training_info)
        elif hasattr(model, 'train_audio_model'):
            # 如果是AudioProcessingModel的实例，使用现有的训练配置
            # If it's an instance of AudioProcessingModel, use the existing training configuration
            model_config = {
                'data_path': config.get('data_path', 'data/audio_samples'),
                'save_path': f"models/joint_audio_model_{i+1}.pth",
                'input_dim': config.get('input_dim', 128),
                'hidden_dim': config.get('hidden_dim', 512),
                'output_dims': config.get('output_dims', [10, 8, 5]),
                'task_types': config.get('task_types', ['recognition', 'synthesis', 'music']),
                'sample_rate': config.get('sample_rate', 44100),
                'batch_size': config.get('batch_size', 64),
                'learning_rate': config.get('learning_rate', 0.0005),
                'epochs': config.get('epochs', 20),
                'joint_training': True,
                'joint_models': [f"model_{j+1}" for j in range(len(models)) if j != i]
            }
            result = train_audio_model(model_config, joint_training_info)
        else:
            # 假设其他模型有自己的训练方法 / Assume other models have their own training methods
            try:
                # 根据模型类型使用适当的训练方法 / Use appropriate training method based on model type
                if hasattr(model, 'fit'):
                    result = model.fit(train_dataset, val_datasets[i] if val_datasets else None)
                else:
                    print(f"警告: 模型 {i+1} 缺少标准训练方法，跳过训练 / Warning: Model {i+1} lacks standard training method, skipping training")
                    result = {"status": "warning", "message": "缺少标准训练方法 / Missing standard training method"}
            except Exception as e:
                print(f"训练模型 {i+1} 失败: {e} / Training model {i+1} failed: {e}")
                joint_results["status"] = "partial_failure"
                joint_results["message"] = f"部分模型训练失败 / Some models training failed"
                result = {"status": "error", "message": str(e)}
        
        joint_results["individual_results"].append(result)
    
    # 计算联合指标 / Calculate joint metrics
    if all(res.get("status") == "success" for res in joint_results["individual_results"]):
        # 计算加权平均损失 / Calculate weighted average loss
        valid_results = [res for res in joint_results["individual_results"] if "metrics" in res]
        if valid_results:
            avg_loss = sum(res["metrics"].get("final_loss", 0) * w for res, w in zip(valid_results, loss_weights[:len(valid_results)]))
            min_loss = sum(res["metrics"].get("min_loss", 0) * w for res, w in zip(valid_results, loss_weights[:len(valid_results)]))
            
            joint_results["joint_metrics"] = {
                "avg_loss": avg_loss,
                "min_loss": min_loss,
                "loss_weights": loss_weights
            }
            
            print(f"联合训练完成，平均损失: {avg_loss:.4f}, 最小损失: {min_loss:.4f} / Joint training completed, average loss: {avg_loss:.4f}, min loss: {min_loss:.4f}")
    
    # 保存联合训练报告 / Save joint training report
    report_dir = "training_reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = f"{report_dir}/joint_audio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "training_date": datetime.now().isoformat(),
            "joint_training": True,
            "model_count": len(models),
            "models": [{"model_type": "audio", "index": 1}] + [{"model_type": "partner", "index": i+2} for i in range(len(models)-1)],
            "loss_weights": loss_weights,
            "training_config": config,
            "joint_results": joint_results
        }, f, indent=2, ensure_ascii=False)
    
    joint_results["report_path"] = report_path
    return joint_results

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
    
    # 启动单独训练 / Start standalone training
    training_result = train_audio_model(config)
    # 训练完成提示 / Training completion notification
    print("Audio model standalone training completed!")
    print(f"训练结果已保存到: {training_result['log_path']} / Training result saved to: {training_result['log_path']}")
    
    # 示例：如何使用联合训练功能 (实际使用时需要准备多个模型和数据集)
    # Example: How to use joint training functionality (need to prepare multiple models and datasets for actual use)
    print("\n联合训练功能演示 (示例代码) / Joint training demonstration (example code)")
    
    # 这里仅作为示例，实际使用时需要准备真实的模型和数据集
    # This is just an example, real models and datasets are needed for actual use
    try:
        # 假设我们有两个音频模型要进行联合训练
        # Assume we have two audio models for joint training
        model1 = AudioProcessingModel(
            input_dim=128,
            hidden_dim=512,
            output_dims=[10, 8, 5],
            task_types=['recognition', 'synthesis', 'music']
        )
        
        model2 = AudioProcessingModel(
            input_dim=128,
            hidden_dim=512,
            output_dims=[8, 6, 4],
            task_types=['recognition', 'synthesis', 'music']
        )
        
        # 模拟数据集 (实际使用时需要准备真实数据)
        # Mock datasets (real data is needed for actual use)
        class MockDataset:
            def __getitem__(self, index):
                return torch.randn(128, 100), torch.randint(0, 10, (1,))
            def __len__(self):
                return 1000
        
        mock_dataset1 = MockDataset()
        mock_dataset2 = MockDataset()
        
        # 联合训练配置
        joint_config = {
            'input_dim': 128,
            'hidden_dim': 512,
            'epochs': 10,  # 减少联合训练的轮数以加快演示 / Reduce epochs for faster demonstration
            'batch_size': 32,
            'learning_rate': 0.0001
        }
        
        # 执行联合训练
        # Note: 这只是演示代码，实际执行可能需要额外的修改和适配
        print("准备执行联合训练演示 / Preparing to execute joint training demonstration")
        print("注意: 完整的联合训练功能需要与其他模型模块正确集成 / Note: Complete joint training requires proper integration with other model modules")
        
        # 以下代码仅作为示例，不建议直接执行
        # The following code is for demonstration only and is not recommended to run directly
        # joint_result = train_jointly(
        #     models=[model1, model2],
        #     train_datasets=[mock_dataset1, mock_dataset2],
        #     config=joint_config,
        #     loss_weights=[0.5, 0.5]
        # )
        
        # print(f"联合训练结果已保存到: {joint_result['report_path']} / Joint training result saved to: {joint_result['report_path']}")
        
    except Exception as e:
        print(f"联合训练演示失败: {e} / Joint training demonstration failed: {e}")
        print("请确保所有依赖都已正确安装并配置 / Please ensure all dependencies are properly installed and configured")
