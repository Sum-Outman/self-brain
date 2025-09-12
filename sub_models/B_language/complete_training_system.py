# Copyright 2025 Self Brain AGI System Authors
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

# B语言模型完整训练系统 | B Language Model Complete Training System
# 支持单独训练、联合训练和实时学习 | Supports individual training, joint training, and real-time learning

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

from .enhanced_language_model import EnhancedMultilingualEmotionalLLM, EmotionType, LanguageSupport

class LanguageTrainingDataset(Dataset):
    """多语言情感训练数据集 | Multilingual Emotional Training Dataset"""
    
    def __init__(self, data_path: str, language: str = "en", max_length: int = 512):
        """
        初始化训练数据集 | Initialize training dataset
        
        参数 Parameters:
        data_path: 数据文件路径 | Data file path
        language: 目标语言 | Target language
        max_length: 最大序列长度 | Maximum sequence length
        """
        self.language = language
        self.max_length = max_length
        self.data = self._load_data(data_path)
        self.tokenizer = None  # 将在训练时设置 | Will be set during training
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载训练数据 | Load training data"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 过滤指定语言的数据 | Filter data for specified language
            filtered_data = [
                item for item in data 
                if item.get('language', 'en') == self.language
            ]
            
            logging.info(f"Loaded {len(filtered_data)} samples for language {self.language}")
            return filtered_data
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return []
    
    def __len__(self) -> int:
        """返回数据集大小 | Return dataset size"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个训练样本 | Get single training sample"""
        sample = self.data[idx]
        
        # 编码输入文本 | Encode input text
        inputs = self.tokenizer(
            sample['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 获取情感标签 | Get emotion label
        emotion = sample.get('emotion', 'neutral')
        emotion_idx = self._emotion_to_index(emotion)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'emotion_label': torch.tensor(emotion_idx, dtype=torch.long),
            'language': sample.get('language', 'en'),
            'text': sample['text']
        }
    
    def _emotion_to_index(self, emotion: str) -> int:
        """将情感字符串转换为索引 | Convert emotion string to index"""
        emotion_mapping = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 
            'neutral': 4, 'sadness': 5, 'surprise': 6,
            'excitement': 7, 'confusion': 8, 'anticipation': 9
        }
        return emotion_mapping.get(emotion.lower(), 4)  # 默认为中性 | Default to neutral

class LanguageModelTrainer:
    """语言模型训练器 | Language Model Trainer"""
    
    def __init__(self, model: EnhancedMultilingualEmotionalLLM):
        """
        初始化训练器 | Initialize trainer
        
        参数 Parameters:
        model: 要训练的模型 | Model to train
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 训练状态 | Training state
        self.training_state = {
            'status': 'idle',
            'current_epoch': 0,
            'total_epochs': 0,
            'current_step': 0,
            'total_steps': 0,
            'current_loss': 0.0,
            'average_loss': 0.0,
            'learning_rate': 0.0,
            'start_time': None,
            'estimated_completion': None,
            'performance_metrics': {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        }
        
        # 训练历史 | Training history
        self.training_history = []
        
        logging.info(f"Language model trainer initialized on device: {self.device}")
    
    def prepare_training(self, 
                       data_path: str, 
                       languages: List[str],
                       batch_size: int = 16,
                       learning_rate: float = 1e-5,
                       num_epochs: int = 10) -> bool:
        """
        准备训练 | Prepare training
        
        参数 Parameters:
        data_path: 训练数据路径 | Training data path
        languages: 要训练的语言列表 | List of languages to train
        batch_size: 批次大小 | Batch size
        learning_rate: 学习率 | Learning rate
        num_epochs: 训练轮数 | Number of epochs
        
        返回 Returns:
        是否准备成功 | Whether preparation was successful
        """
        try:
            # 创建数据加载器 | Create data loaders
            self.data_loaders = {}
            for lang in languages:
                dataset = LanguageTrainingDataset(data_path, lang)
                dataset.tokenizer = self.model.tokenizer
                self.data_loaders[lang] = DataLoader(
                    dataset, batch_size=batch_size, shuffle=True
                )
            
            # 设置优化器 | Setup optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=0.01
            )
            
            # 设置损失函数 | Setup loss function
            self.loss_fn = nn.CrossEntropyLoss()
            
            # 更新训练状态 | Update training state
            self.training_state.update({
                'status': 'prepared',
                'total_epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'languages': languages,
                'data_path': data_path
            })
            
            logging.info(f"Training prepared for languages: {languages}")
            return True
            
        except Exception as e:
            logging.error(f"Error preparing training: {e}")
            return False
    
    def start_training(self) -> bool:
        """开始训练 | Start training"""
        if self.training_state['status'] != 'prepared':
            logging.error("Training not prepared")
            return False
        
        # 更新训练状态 | Update training state
        self.training_state.update({
            'status': 'training',
            'current_epoch': 0,
            'current_step': 0,
            'start_time': datetime.now().isoformat(),
            'estimated_completion': self._calculate_estimated_completion()
        })
        
        # 启动训练线程 | Start training thread
        import threading
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        logging.info("Training started")
        return True
    
    def stop_training(self):
        """停止训练 | Stop training"""
        self.training_state['status'] = 'stopping'
        logging.info("Training stopping...")
    
    def _training_loop(self):
        """训练循环 | Training loop"""
        try:
            total_steps = sum(len(loader) for loader in self.data_loaders.values()) * self.training_state['total_epochs']
            self.training_state['total_steps'] = total_steps
            
            for epoch in range(self.training_state['total_epochs']):
                if self.training_state['status'] == 'stopping':
                    break
                
                self.training_state['current_epoch'] = epoch + 1
                
                # 训练所有语言 | Train all languages
                for lang, data_loader in self.data_loaders.items():
                    self._train_epoch(data_loader, lang, epoch)
                
                # 每轮结束后验证 | Validate after each epoch
                self._validate()
                
                # 保存检查点 | Save checkpoint
                if (epoch + 1) % 5 == 0:
                    self._save_checkpoint(epoch + 1)
            
            # 训练完成 | Training completed
            if self.training_state['status'] != 'stopping':
                self.training_state['status'] = 'completed'
                self._save_final_model()
                logging.info("Training completed successfully")
            
        except Exception as e:
            logging.error(f"Training error: {e}")
            self.training_state['status'] = 'failed'
            self.training_state['error'] = str(e)
        
        finally:
            # 清理训练状态 | Cleanup training state
            if self.training_state['status'] == 'stopping':
                self.training_state['status'] = 'stopped'
            elif self.training_state['status'] == 'training':
                self.training_state['status'] = 'completed'
    
    def _train_epoch(self, data_loader: DataLoader, language: str, epoch: int):
        """训练单个epoch | Train single epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(data_loader):
            if self.training_state['status'] == 'stopping':
                break
            
            # 准备输入数据 | Prepare input data
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            emotion_labels = batch['emotion_label'].to(self.device)
            
            # 前向传播 | Forward pass
            self.optimizer.zero_grad()
            lm_logits, emotion_logits, intensity = self.model(
                input_ids, attention_mask, language
            )
            
            # 计算损失 | Calculate loss
            loss = self.loss_fn(emotion_logits, emotion_labels)
            
            # 反向传播 | Backward pass
            loss.backward()
            self.optimizer.step()
            
            # 更新统计 | Update statistics
            total_loss += loss.item()
            total_samples += input_ids.size(0)
            
            # 更新训练状态 | Update training state
            self.training_state['current_step'] += 1
            self.training_state['current_loss'] = loss.item()
            self.training_state['average_loss'] = total_loss / (batch_idx + 1)
            
            # 记录训练历史 | Record training history
            self.training_history.append({
                'epoch': epoch + 1,
                'step': self.training_state['current_step'],
                'loss': loss.item(),
                'language': language,
                'timestamp': datetime.now().isoformat()
            })
            
            # 每100步打印进度 | Print progress every 100 steps
            if batch_idx % 100 == 0:
                logging.info(
                    f"Epoch {epoch+1}/{self.training_state['total_epochs']}, "
                    f"Language: {language}, "
                    f"Batch {batch_idx}/{len(data_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
    
    def _validate(self):
        """验证模型性能 | Validate model performance"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for lang, data_loader in self.data_loaders.items():
                for batch in data_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['emotion_label'].numpy()
                    
                    # 预测 | Predict
                    _, emotion_logits, _ = self.model(input_ids, attention_mask, lang)
                    predictions = torch.argmax(emotion_logits, dim=-1).cpu().numpy()
                    
                    all_predictions.extend(predictions)
                    all_labels.extend(labels)
        
        # 计算指标 | Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # 更新性能指标 | Update performance metrics
        self.training_state['performance_metrics'].update({
            'accuracy': accuracy,
            'f1_score': f1
        })
        
        logging.info(f"Validation - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    def _calculate_estimated_completion(self) -> str:
        """计算预计完成时间 | Calculate estimated completion time"""
        # 基于经验估算训练时间 | Estimate training time based on experience
        estimated_seconds = self.training_state['total_epochs'] * 3600  # 假设每轮1小时 | Assume 1 hour per epoch
        
        completion_time = datetime.now() + timedelta(seconds=estimated_seconds)
        return completion_time.isoformat()
    
    def _save_checkpoint(self, epoch: int):
        """保存训练检查点 | Save training checkpoint"""
        checkpoint_path = f"checkpoints/language_model_epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_state': self.training_state,
            'training_history': self.training_history
        }, checkpoint_path)
        
        logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self):
        """保存最终模型 | Save final model"""
        model_path = "models/enhanced_language_model_final.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_state': self.training_state,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        logging.info(f"Final model saved: {model_path}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        获取训练状态 | Get training status
        
        返回 Returns:
        训练状态字典 | Training status dictionary
        """
        return self.training_state
    
    def get_training_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取训练历史 | Get training history
        
        参数 Parameters:
        limit: 返回的历史记录数量 | Number of history records to return
        
        返回 Returns:
        训练历史列表 | Training history list
        """
        return self.training_history[-limit:]
    
    def generate_training_report(self) -> Dict[str, Any]:
        """
        生成训练报告 | Generate training report
        
        返回 Returns:
        训练报告字典 | Training report dictionary
        """
        report = {
            'training_summary': {
                'status': self.training_state['status'],
                'total_epochs': self.training_state['total_epochs'],
                'completed_epochs': self.training_state['current_epoch'],
                'total_steps': self.training_state['total_steps'],
                'completed_steps': self.training_state['current_step'],
                'start_time': self.training_state['start_time'],
                'end_time': datetime.now().isoformat() if self.training_state['status'] in ['completed', 'stopped', 'failed'] else None,
                'duration_seconds': self._calculate_duration()
            },
            'performance_metrics': self.training_state['performance_metrics'],
            'language_statistics': {
                lang: len(self.data_loaders[lang].dataset) 
                for lang in self.training_state.get('languages', [])
            },
            'hardware_info': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        return report

class JointTrainingCoordinator:
    """联合训练协调器 | Joint Training Coordinator"""
    
    def __init__(self):
        """初始化联合训练协调器 | Initialize joint training coordinator"""
        self.training_sessions = {}
        self.coordination_state = {
            'status': 'idle',
            'active_sessions': 0,
            'total_models': 0,
            'performance_metrics': {}
        }
    
    def start_joint_training(self, model_configs: Dict[str, Dict]) -> bool:
        """
        开始联合训练 | Start joint training
        
        参数 Parameters:
        model_configs: 模型配置字典 | Model configuration dictionary
        
        返回 Returns:
        是否成功启动 | Whether started successfully
        """
        try:
            # 初始化训练会话 | Initialize training sessions
            for model_id, config in model_configs.items():
                if model_id == 'B':  # 语言模型
                    model = EnhancedMultilingualEmotionalLLM()
                    trainer = LanguageModelTrainer(model)
                    
                    if trainer.prepare_training(
                        config['data_path'],
                        config['languages'],
                        config.get('batch_size', 16),
                        config.get('learning_rate', 1e-5),
                        config.get('num_epochs', 10)
                    ):
                        self.training_sessions[model_id] = {
                            'trainer': trainer,
                            'config': config,
                            'status': 'prepared'
                        }
            
            # 更新协调状态 | Update coordination state
            self.coordination_state.update({
                'status': 'training',
                'active_sessions': len(self.training_sessions),
                'total_models': len(model_configs),
                'start_time': datetime.now().isoformat()
            })
            
            # 启动所有训练会话 | Start all training sessions
            for model_id, session in self.training_sessions.items():
                if session['trainer'].start_training():
                    session['status'] = 'training'
            
            logging.info("Joint training started")
            return True
            
        except Exception as e:
            logging.error(f"Error starting joint training: {e}")
            return False
    
    def stop_joint_training(self):
        """停止联合训练 | Stop joint training"""
        for session in self.training_sessions.values():
            session['trainer'].stop_training()
        
        self.coordination_state['status'] = 'stopping'
        logging.info("Joint training stopping...")
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """
        获取协调状态 | Get coordination status
        
        返回 Returns:
        协调状态字典 | Coordination status dictionary
        """
        status = self.coordination_state.copy()
        
        # 添加各个模型的训练状态 | Add training status for each model
        status['model_statuses'] = {}
        for model_id, session in self.training_sessions.items():
            status['model_statuses'][model_id] = session['trainer'].get_training_status()
        
        return status

# Web API 接口 | Web API Interface
class TrainingAPI:
    """训练API接口 | Training API Interface"""
    
    def __init__(self, trainer: LanguageModelTrainer):
        """
        初始化训练API | Initialize training API
        
        参数 Parameters:
        trainer: 语言模型训练器 | Language model trainer
        """
        self.trainer = trainer
        self.joint_coordinator = JointTrainingCoordinator()
    
    def handle_training_request(self, request_data: Dict) -> Dict[str, Any]:
        """
        处理训练请求 | Handle training request
        
        参数 Parameters:
        request_data: 请求数据 | Request data
        
        返回 Returns:
        响应字典 | Response dictionary
        """
        try:
            training_type = request_data.get('type', 'individual')
            
            if training_type == 'individual':
                return self._handle_individual_training(request_data)
            elif training_type == 'joint':
                return self._handle_joint_training(request_data)
            else:
                return {'success': False, 'error': 'Invalid training type'}
                
        except Exception as e:
            logging.error(f"Error handling training request: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_individual_training(self, request_data: Dict) -> Dict[str, Any]:
        """处理单独训练请求 | Handle individual training request"""
        # 准备训练 | Prepare training
        success = self.trainer.prepare_training(
            request_data['data_path'],
            request_data['languages'],
            request_data.get('batch_size', 16),
            request_data.get('learning_rate', 1e-5),
            request_data.get('num_epochs', 10)
        )
        
        if not success:
            return {'success': False, 'error': 'Failed to prepare training'}
        
        # 开始训练 | Start training
        if self.trainer.start_training():
            return {
                'success': True,
                'message': 'Individual training started',
                'training_id': f"indiv_{int(time.time())}"
            }
        else:
            return {'success': False, 'error': 'Failed to start training'}
    
    def _handle_joint_training(self, request_data: Dict) -> Dict[str, Any]:
        """处理联合训练请求 | Handle joint training request"""
        model_configs = request_data.get('model_configs', {})
        
        if not model_configs:
            return {'success': False, 'error': 'No model configurations provided'}
        
        if self.joint_coordinator.start_joint_training(model_configs):
            return {
                'success': True,
                'message': 'Joint training started',
                'training_id': f"joint_{int(time.time())}"
            }
        else:
            return {'success': False, 'error': 'Failed to start joint training'}
    
    def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """
        获取训练状态 | Get training status
        
        参数 Parameters:
        training_id: 训练ID | Training ID
        
        返回 Returns:
        训练状态字典 | Training status dictionary
        """
        if training_id.startswith('indiv_'):
            return self.trainer.get_training_status()
        elif training_id.startswith('joint_'):
            return self.joint_coordinator.get_coordination_status()
        else:
            return {'success': False, 'error': 'Invalid training ID'}

# 工具函数 | Utility functions
def create_sample_training_data(output_path: str, num_samples: int = 1000):
    """
    创建示例训练数据 | Create sample training data
    
    参数 Parameters:
    output_path: 输出文件路径 | Output file path
    num_samples: 样本数量 | Number of samples
    """
    emotions = ['joy', 'sadness', 'anger', 'surprise', 'neutral', 'excitement', 'confusion']
    languages = ['en', 'zh', 'es', 'fr', 'de']
    
    sample_data = []
    
    for i in range(num_samples):
        emotion = np.random.choice(emotions)
        language = np.random.choice(languages)
        
        # 生成示例文本 | Generate sample text
        if language == 'en':
            text = f"This is sample text showing {emotion} emotion. Sample #{i+1}"
        elif language == 'zh':
            text = f"这是显示{emotion}情感的示例文本。样本 #{i+1}"
        elif language == 'es':
            text = f"Este es texto de ejemplo que muestra emoción {emotion}. Muestra #{i+1}"
        else:
            text = f"Sample text in {language} showing {emotion}. #{i+1}"
        
        sample_data.append({
            'text': text,
            'emotion': emotion,
            'language': language,
            'source': 'synthetic',
            'created_at': datetime.now().isoformat()
        })
    
    # 保存数据 | Save data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Sample training data created: {output_path}")

def visualize_training_progress(training_history: List[Dict], output_path: str):
    """
    可视化训练进度 | Visualize training progress
    
    参数 Parameters:
    training_history: 训练历史 | Training history
    output_path: 输出文件路径 | Output file path
    """
    if not training_history:
        logging.warning("No training history to visualize")
        return
    
    # 提取损失数据 | Extract loss data
    steps = [item['step'] for item in training_history]
    losses = [item['loss'] for item in training_history]
    
    # 创建图表 | Create chart
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, 'b-', alpha=0.7)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Progress')
    plt.grid(True, alpha=0.3)
    
    # 保存图表 | Save chart
    plt.savefig(output_path)
    plt.close()
    
    logging.info(f"Training progress visualization saved: {output_path}")

# 主训练函数 | Main training function
def main():
    """主训练函数 | Main training function"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型 | Create model
    model = EnhancedMultilingualEmotionalLLM()
    
    # 创建训练器 | Create trainer
    trainer = LanguageModelTrainer(model)
    
    # 准备训练 | Prepare training
    success = trainer.prepare_training(
        data_path="data/training_data.json",
        languages=["en", "zh", "es"],
        batch_size=8,
        learning_rate=1e-5,
        num_epochs=3
    )
    
    if success:
        # 开始训练 | Start training
        trainer.start_training()
        
        # 等待训练完成 | Wait for training to complete
        while trainer.get_training_status()['status'] == 'training':
            time.sleep(10)
            status = trainer.get_training_status()
            logging.info(f"Training progress: Epoch {status['current_epoch']}/{status['total_epochs']}")
        
        # 生成报告 | Generate report
        report = trainer.generate_training_report()
        logging.info(f"Training completed. Final accuracy: {report['performance_metrics']['accuracy']:.4f}")
        
        # 可视化训练进度 | Visualize training progress
        visualize_training_progress(
            trainer.get_training_history(),
            "training_progress.png"
        )
    else:
        logging.error("Failed to prepare training")

if __name__ == "__main__":
    main()