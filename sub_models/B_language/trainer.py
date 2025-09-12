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

# B_language模型训练器
# B_language Model Trainer

import logging
import time
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any, Optional
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("B_language_Trainer")

class LanguageDataset(Dataset):
    """语言模型数据集 | Language model dataset"""
    
    def __init__(self, data_path: str, max_length: int = 512):
        self.data_path = data_path
        self.max_length = max_length
        self.samples = self._load_data()
        
    def _load_data(self):
        """加载训练数据 | Load training data"""
        samples = []
        data_file = os.path.join(self.data_path, "training_data.json")
        
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    samples = data.get("samples", [])
            except Exception as e:
                logger.error(f"加载训练数据错误: {str(e)}")
        
        # 如果没有数据，创建一些示例数据
        if not samples:
            samples = [
                {"text": "你好，世界！", "label": "greeting"},
                {"text": "今天天气很好", "label": "weather"},
                {"text": "人工智能很有趣", "label": "technology"},
                {"text": "我喜欢编程", "label": "hobby"},
                {"text": "机器学习很强大", "label": "technology"}
            ]
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.get("text", "")
        label = sample.get("label", "")
        
        # 简单的文本编码（在实际应用中应使用tokenizer）
        text_tensor = torch.tensor([ord(c) for c in text[:self.max_length]], dtype=torch.long)
        
        # 简单的标签编码
        label_map = {"greeting": 0, "weather": 1, "technology": 2, "hobby": 3}
        label_tensor = torch.tensor(label_map.get(label, 0), dtype=torch.long)
        
        return text_tensor, label_tensor

class LanguageModel(nn.Module):
    """AGI语言模型 | AGI Language Model"""
    
    def __init__(self, 
                 vocab_size: int = 50000,
                 embedding_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_classes: int = 10,
                 emotion_dim: int = 5):
        super(LanguageModel, self).__init__()
        
        # 多语言嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 情感分析模块
        self.emotion_analyzer = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, emotion_dim)
        )
        
        # 任务分类器
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
    def forward(self, x):
        # 嵌入层
        embedded = self.embedding(x)
        
        # 添加位置编码
        embedded = self.positional_encoding(embedded)
        
        # Transformer处理
        transformer_out = self.transformer_encoder(embedded)
        
        # 获取序列的聚合表示（使用最后时间步）
        aggregated = transformer_out[:, -1, :]
        
        # 情感分析
        emotion_output = self.emotion_analyzer(aggregated)
        
        # 任务分类
        class_output = self.classifier(aggregated)
        
        return class_output, emotion_output

class PositionalEncoding(nn.Module):
    """位置编码 | Positional Encoding"""
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class ModelTrainer:
    def __init__(self, model_path: str = None):
        """初始化语言模型训练器 | Initialize language model trainer
        参数:
            model_path: 模型保存路径 | Model save path
        """
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or "models/b_language_model.pth"
        self.training_history = []
        
        # 创建模型目录
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
    def initialize_model(self, vocab_size: int = 10000, embedding_dim: int = 128, 
                        hidden_dim: int = 256, num_classes: int = 10):
        """初始化模型 | Initialize model"""
        self.model = SimpleLanguageModel(vocab_size, embedding_dim, hidden_dim, num_classes)
        self.model.to(self.device)
        logger.info(f"模型初始化完成，设备: {self.device}")
        
    def load_model(self, model_path: str = None):
        """加载预训练模型 | Load pretrained model"""
        path = model_path or self.model_path
        if os.path.exists(path):
            try:
                self.model = torch.load(path, map_location=self.device)
                self.model.to(self.device)
                logger.info(f"模型加载成功: {path}")
                return True
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
        return False
    
    def save_model(self, model_path: str = None):
        """保存模型 | Save model"""
        path = model_path or self.model_path
        try:
            torch.save(self.model, path)
            logger.info(f"模型保存成功: {path}")
            return True
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
            return False
    
    def train(self, epochs: int, batch_size: int, learning_rate: float, 
             data_path: str = None, callback: callable = None) -> Dict:
        """训练语言模型 | Train language model
        参数:
            epochs: 训练轮数 | Number of epochs
            batch_size: 批量大小 | Batch size
            learning_rate: 学习率 | Learning rate
            data_path: 数据路径 | Data path
            callback: 回调函数 | Callback function
        返回:
            训练结果字典 | Training result dictionary
        """
        logger.info("开始训练语言模型 | Starting language model training")
        logger.info(f"配置: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
        
        # 初始化模型（如果尚未初始化）
        if self.model is None:
            self.initialize_model()
        
        # 准备数据
        dataset = LanguageDataset(data_path or "data/b_language")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            epoch_samples = 0
            
            for batch_idx, (texts, labels) in enumerate(dataloader):
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(texts)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                acc = (predicted == labels).float().mean()
                
                # 更新统计
                batch_size = texts.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_acc += acc.item() * batch_size
                epoch_samples += batch_size
                
                # 调用回调函数（用于进度更新）
                if callback:
                    progress = (epoch * len(dataloader) + batch_idx + 1) / (epochs * len(dataloader)) * 100
                    callback(progress, epoch + 1, {
                        'loss': loss.item(),
                        'accuracy': acc.item(),
                        'batch': batch_idx + 1
                    })
            
            # 计算epoch统计
            avg_loss = epoch_loss / epoch_samples
            avg_acc = epoch_acc / epoch_samples
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
            
            # 保存训练历史
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': avg_acc,
                'samples': epoch_samples
            })
            
            # 更新总统计
            total_loss += epoch_loss
            total_acc += epoch_acc
            total_samples += epoch_samples
        
        # 计算总统计
        final_loss = total_loss / total_samples
        final_acc = total_acc / total_samples
        
        # 保存模型
        self.save_model()
        
        logger.info("语言模型训练完成 | Language model training completed")
        
        return {
            "loss": final_loss,
            "accuracy": final_acc,
            "perplexity": np.exp(final_loss),  # 困惑度
            "epochs": epochs,
            "samples": total_samples,
            "history": self.training_history
        }
    
    def evaluate(self, data_path: str = None) -> Dict:
        """评估模型 | Evaluate model
        参数:
            data_path: 数据路径 | Data path
        返回:
            评估结果字典 | Evaluation result dictionary
        """
        if self.model is None:
            logger.error("模型未初始化 | Model not initialized")
            return {"error": "Model not initialized"}
        
        # 准备数据
        dataset = LanguageDataset(data_path or "data/b_language")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 评估循环
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for texts, labels in dataloader:
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(texts)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                acc = (predicted == labels).float().mean()
                
                batch_size = texts.size(0)
                total_loss += loss.item() * batch_size
                total_acc += acc.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        
        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
            "perplexity": np.exp(avg_loss),
            "samples": total_samples
        }
    
    def predict(self, text: str, language: str = "zh") -> Dict:
        """预测文本（支持多语言） | Predict text (multilingual support)
        参数:
            text: 输入文本 | Input text
            language: 文本语言代码 | Text language code (zh/en/de/ja/ru)
        返回:
            预测结果字典 | Prediction result dictionary
        """
        if self.model is None:
            logger.error("模型未初始化 | Model not initialized")
            return {"error": "Model not initialized"}
        
        self.model.eval()
        
        # 多语言tokenization
        token_ids = self.tokenize(text, language)
        text_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            class_output, emotion_output = self.model(text_tensor)
            
            # 任务分类概率
            class_probs = torch.softmax(class_output, dim=1)
            class_conf, class_pred = torch.max(class_probs, 1)
            
            # 情感分析结果
            emotion_probs = torch.softmax(emotion_output, dim=1)
            emotion_conf, emotion_pred = torch.max(emotion_probs, 1)
        
        # 标签映射
        label_map = {0: "greeting", 1: "weather", 2: "technology", 3: "hobby", 4: "question"}
        emotion_map = {0: "happy", 1: "sad", 2: "angry", 3: "surprised", 4: "neutral"}
        
        return {
            "text": text,
            "language": language,
            "predicted_label": label_map.get(class_pred.item(), "unknown"),
            "confidence": class_conf.item(),
            "emotion": emotion_map.get(emotion_pred.item(), "neutral"),
            "emotion_confidence": emotion_conf.item(),
            "class_probabilities": class_probs.cpu().numpy().tolist(),
            "emotion_probabilities": emotion_probs.cpu().numpy().tolist()
        }
        
    def tokenize(self, text: str, language: str) -> List[int]:
        """多语言tokenization | Multilingual tokenization"""
        # 实际实现应使用多语言tokenizer
        # 这里简化实现
        if language == "zh":
            return [ord(c) for c in text[:512]]
        else:
            # 对非中文语言使用空格分词
            tokens = text.split()[:512]
            return [hash(token) % 50000 for token in tokens]

class JointTrainingManager:
    """联合训练管理器 | Joint training manager"""
    
    def __init__(self):
        self.trainers = {}
        self.joint_training_history = []
    
    def register_trainer(self, model_id: str, trainer: ModelTrainer):
        """注册训练器 | Register trainer"""
        self.trainers[model_id] = trainer
        logger.info(f"训练器注册成功: {model_id}")
    
    def joint_train(self, model_ids: List[str], epochs: int, batch_size: int, 
                   learning_rate: float, callback: callable = None) -> Dict:
        """执行联合训练 | Execute joint training
        参数:
            model_ids: 模型ID列表 | List of model IDs
            epochs: 训练轮数 | Number of epochs
            batch_size: 批量大小 | Batch size
            learning_rate: 学习率 | Learning rate
            callback: 回调函数 | Callback function
        返回:
            联合训练结果字典 | Joint training result dictionary
        """
        logger.info(f"开始联合训练: {model_ids}")
        
        results = {}
        total_progress = 0
        
        for model_id in model_ids:
            if model_id not in self.trainers:
                logger.warning(f"找不到训练器: {model_id}")
                continue
            
            trainer = self.trainers[model_id]
            
            def progress_callback(progress, epoch, metrics):
                # 调整进度以反映所有模型的联合训练
                adjusted_progress = (total_progress + progress / len(model_ids)) / 100
                if callback:
                    callback(adjusted_progress * 100, epoch, {
                        'model': model_id,
                        **metrics
                    })
            
            # 训练单个模型
            result = trainer.train(epochs, batch_size, learning_rate, callback=progress_callback)
            results[model_id] = result
            total_progress += 100  # 每个模型完成100%
        
        # 记录联合训练历史
        self.joint_training_history.append({
            'models': model_ids,
            'epochs': epochs,
            'results': results,
            'timestamp': time.time()
        })
        
        logger.info("联合训练完成 | Joint training completed")
        return results

# 全局训练器实例
global_trainer = ModelTrainer()
global_joint_manager = JointTrainingManager()

def get_trainer():
    """获取全局训练器实例 | Get global trainer instance"""
    return global_trainer

def get_joint_manager():
    """获取全局联合训练管理器实例 | Get global joint training manager instance"""
    return global_joint_manager

# 测试代码
if __name__ == '__main__':
    # 测试训练器
    trainer = ModelTrainer()
    
    # 训练模型
    result = trainer.train(
        epochs=5,
        batch_size=16,
        learning_rate=0.001
    )
    
    print(f"训练结果: {result}")
    
    # 测试预测
    prediction = trainer.predict("你好，人工智能")
    print(f"预测结果: {prediction}")
    
    # 测试评估
    evaluation = trainer.evaluate()
    print(f"评估结果: {evaluation}")
