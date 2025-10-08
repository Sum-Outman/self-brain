# -*- coding: utf-8 -*-
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

"""
语言模型训练模块
整合从零开始训练和预训练模型功能
提供统一的训练API接口
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from datetime import datetime
from collections import deque
import psutil

# 设置随机种子以确保可重复性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 全局的collate_fn函数，用于DataLoader在多进程环境中使用
def collate_fn(batch):
    """将批次数据转换为模型输入格式"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'labels': labels.to(device)
    }

class Vocabulary:
    """
    词汇表类，用于处理文本数据的词汇映射
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = {}
        self.total_words = 0
        
        # 添加特殊标记
        self.add_word('<pad>')
        self.add_word('<unk>')
        self.add_word('<bos>')
        self.add_word('<eos>')
    
    def add_word(self, word):
        """添加单词到词汇表"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1
        self.total_words += 1
    
    def get_idx(self, word):
        """获取单词对应的索引，未找到则返回unknown的索引"""
        return self.word2idx.get(word, self.word2idx['<unk>'])
    
    def get_word(self, idx):
        """获取索引对应的单词"""
        return self.idx2word.get(idx, '<unk>')
    
    def __len__(self):
        """返回词汇表大小"""
        return len(self.word2idx)

class LanguageCorpusDataset(Dataset):
    """
    语言语料数据集类，支持从零开始训练
    """
    def __init__(self, data_path=None, max_length=128, vocab_size=None):
        """
        初始化数据集
        
        参数:
            data_path: 数据文件路径，如果为None则生成合成数据
            max_length: 序列最大长度
            vocab_size: 词汇表大小
        """
        self.max_length = max_length
        
        # 从零开始训练模式，构建自定义词汇表
        self.vocab = Vocabulary()
        self.tokenizer = None  # 不使用预训练分词器
        
        # 加载或生成数据
        if data_path and os.path.exists(data_path):
            self.data = self._load_data(data_path)
        else:
            self.data = self._generate_synthetic_data()
        
        # 构建并可能扩展词汇表
        self._build_vocabulary()
        if vocab_size and vocab_size > len(self.vocab):
            self._expand_vocabulary(vocab_size)
        
        # 编码数据
        self.encoded_data = self._encode_data()
    
    def _generate_synthetic_data(self):
        """生成合成训练数据"""
        # 基础词汇表（支持5种语言）
        vocabularies = {
            'en': ['hello', 'world', 'i', 'am', 'a', 'language', 'model', 'learning', 'to', 'understand', 'you'],
            'zh': ['你好', '世界', '我', '是', '一个', '语言', '模型', '学习', '理解', '你'],
            'ja': ['こんにちは', '世界', '私', 'は', '言語', 'モデル', '学習', '理解', 'し', 'ます'],
            'de': ['hallo', 'welt', 'ich', 'bin', 'ein', 'sprach', 'modell', 'lernen', 'zu', 'verstehen', 'dich'],
            'ru': ['привет', 'мир', 'я', 'являюсь', 'языковой', 'моделью', 'учусь', 'понимать', 'вас']
        }
        
        # 基础句子结构
        sentence_structures = [
            '{word1} {word2} {word3}',
            '{word1} {word2} {word3} {word4}',
            '{word1} {word2} {word3} {word4} {word5}',
            '{word1} {word2} {word3} {word4} {word5} {word6}'
        ]
        
        data = []
        # 为每种语言生成数据
        for lang, words in vocabularies.items():
            for _ in range(1000):  # 每种语言生成1000个样本
                # 随机选择句子结构
                structure = random.choice(sentence_structures)
                # 填充随机单词
                sentence_parts = {'word{}'.format(i+1): random.choice(words) 
                                for i in range(structure.count('{'))}
                # 格式化句子
                sentence = structure.format(**sentence_parts)
                # 添加情感标签（简单模拟）
                label = random.choice(['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'])
                # 添加到数据中
                data.append({'text': sentence, 'label': label, 'language': lang})
        
        return data
    
    def _load_data(self, data_path):
        """从文件加载数据"""
        data = []
        
        # 如果是目录，遍历所有子目录中的JSON文件
        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_data = json.load(f)
                                if isinstance(file_data, list):
                                    data.extend(file_data)
                        except Exception as e:
                            print(f"Error loading file {file_path}: {e}")
        else:
            # 如果是单个文件，直接加载
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
            except Exception as e:
                print(f"Error loading file {data_path}: {e}")
        
        return data
    
    def _build_vocabulary(self):
        """构建词汇表"""
        for item in self.data:
            # 简单分词（按空格分割）
            words = item['text'].split()
            for word in words:
                self.vocab.add_word(word)
    
    def _expand_vocabulary(self, target_size):
        """
        通过生成随机组合扩展词汇表到目标大小
        这对于小数据集训练很有帮助
        """
        current_size = len(self.vocab)
        if current_size >= target_size:
            return
        
        base_words = list(self.vocab.word2idx.keys())
        # 移除特殊标记
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        base_words = [word for word in base_words if word not in special_tokens]
        
        # 生成新单词直到达到目标大小
        while len(self.vocab) < target_size and base_words:
            # 随机选择基础单词数量
            n = random.randint(2, min(4, len(base_words)))
            # 随机选择n个基础单词
            selected_words = random.sample(base_words, n)
            # 组合成新单词
            new_word = ''.join(selected_words)
            # 添加到词汇表
            if new_word not in self.vocab.word2idx:
                self.vocab.add_word(new_word)
    
    def _encode_data(self):
        """将文本数据编码为模型可用的格式"""
        encoded_data = []
        
        for item in self.data:
            text = item['text']
            label = item['label']
            language = item.get('language', 'en')
            
            # 从零开始训练模式：使用自定义词汇表
            words = text.split()
            # 添加开始和结束标记
            words = ['<bos>'] + words + ['<eos>']
            # 截断或填充到最大长度
            if len(words) > self.max_length:
                words = words[:self.max_length]
            else:
                words += ['<pad>'] * (self.max_length - len(words))
            # 转换为索引
            input_ids = [self.vocab.get_idx(word) for word in words]
            # 生成注意力掩码
            attention_mask = [1 if word != '<pad>' else 0 for word in words]
            
            # 将情感标签转换为索引
            emotion_labels = {'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'fear': 4, 'surprise': 5, 'disgust': 6}
            label_idx = emotion_labels.get(label, 0)  # 默认中性
            
            encoded_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label_idx,
                'language': language
            })
        
        return encoded_data
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        """获取指定索引的数据项"""
        item = self.encoded_data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'label': torch.tensor(item['label']),
            'language': item['language']
        }

class LanguageModel(nn.Module):
    """
    语言模型类，从零开始训练，增强了情感推理能力
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, 
                 num_layers=2, dropout=0.3):
        """
        初始化语言模型
        
        参数:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super(LanguageModel, self).__init__()
        
        self.dropout = dropout
        
        # 定义情感类别数量
        self.num_emotion_classes = 7  # 7种基本情绪
        self.num_sub_emotion_classes = 21  # 细粒度子情绪类别
        
        # 从零开始训练模式：构建自定义模型
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
        # 情感特征提取层
        self.emotion_feature_extractor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, 512),  # *2 因为是双向LSTM
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # 主情感分类头
        self.emotion_head = nn.Linear(256, self.num_emotion_classes)
        
        # 细粒度子情感分类头
        self.sub_emotion_head = nn.Linear(256, self.num_sub_emotion_classes)
        
        # 情感强度回归头
        self.emotion_intensity_head = nn.Linear(256, 1)
        
        # 语言建模头
        self.lm_head = nn.Linear(self.hidden_dim * 2, vocab_size)
        
        # 初始化统计跟踪属性
        self.__post_init__()
    
    def __post_init__(self):
        """初始化统计跟踪属性"""
        self.input_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'avg_response_time': 0,
            'last_hour_requests': 0,
            'language_distribution': {}
        }
        self.performance_metrics = {
            'inference_speed': 0,
            'accuracy': 0
        }
        self.last_activity = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._request_times = deque(maxlen=100)  # 存储最近100个请求的处理时间
    
    def forward(self, input_ids, attention_mask=None):
        """前向传播，增强了情感推理能力"""
        # 从零开始训练模式
        embedded = self.embedding(input_ids)
        embedded = self.dropout_layer(embedded)
        lstm_out, _ = self.lstm(embedded)
        # 使用最后一个时间步的输出作为句子表示
        cls_representation = lstm_out[:, -1, :]
        sequence_output = self.dropout_layer(lstm_out)
        
        # 提取情感特征
        emotion_features = self.emotion_feature_extractor(cls_representation)
        
        # 主情感分类
        emotion_logits = self.emotion_head(emotion_features)
        
        # 细粒度子情感分类
        sub_emotion_logits = self.sub_emotion_head(emotion_features)
        
        # 情感强度回归（使用sigmoid将输出限制在0-1之间）
        emotion_intensity = torch.sigmoid(self.emotion_intensity_head(emotion_features))
        
        # 语言建模预测
        lm_logits = self.lm_head(sequence_output)
        
        return lm_logits, emotion_logits, sub_emotion_logits, emotion_intensity
    
    def _update_stats(self, success=True, process_time=None, language=None):
        """更新模型统计信息"""
        # 更新总请求数
        self.input_stats['total_requests'] += 1
        
        # 更新成功请求数
        if success:
            self.input_stats['successful_requests'] += 1
        
        # 更新响应时间统计
        if process_time:
            self._request_times.append(process_time)
            self.input_stats['avg_response_time'] = sum(self._request_times) / len(self._request_times)
            # 计算推理速度
            if hasattr(self, 'last_prompt_length') and self.last_prompt_length > 0:
                self.performance_metrics['inference_speed'] = self.last_prompt_length / process_time
        
        # 更新语言分布
        if language:
            if language not in self.input_stats['language_distribution']:
                self.input_stats['language_distribution'][language] = 0
            self.input_stats['language_distribution'][language] += 1
        
        # 更新最后活动时间
        self.last_activity = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def get_status(self):
        """获取模型状态信息"""
        # 获取内存使用情况
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # 获取GPU内存使用情况（如果可用）
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
        # 计算模型参数数量
        param_count = sum(p.numel() for p in self.parameters())
        
        # 获取实际性能指标
        inference_speed = self.performance_metrics.get('inference_speed', 0)
        accuracy = self.performance_metrics.get('accuracy', 0)
        
        return {
            "status": "active",
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "gpu_memory_mb": gpu_memory,
            "parameters_count": param_count,
            "last_activity": self.last_activity,
            "performance": {
                "inference_speed": f"{inference_speed:.2f} tokens/sec" if inference_speed > 0 else "measuring...",
                "accuracy": f"{accuracy:.2%}" if accuracy > 0 else "measuring..."
            }
        }
    
    def get_input_stats(self):
        """获取输入统计信息"""
        # 从实际使用中收集统计数据
        total_requests = self.input_stats.get('total_requests', 0)
        successful_requests = self.input_stats.get('successful_requests', 0)
        failed_requests = total_requests - successful_requests
        avg_response_time = self.input_stats.get('avg_response_time', 0)
        last_hour_requests = self.input_stats.get('last_hour_requests', 0)
        language_distribution = self.input_stats.get('language_distribution', {})
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "average_response_time_ms": avg_response_time,
            "last_hour_requests": last_hour_requests,
            "language_distribution": language_distribution
        }

class ModelTrainer:
    """
    模型训练器类，负责模型的训练、评估和保存，支持增强的情感推理功能
    """
    def __init__(self, model, config=None):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            config: 训练配置
        """
        self.model = model.to(device)
        
        # 设置默认配置
        self.config = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'gradient_clipping': 1.0,
            'early_stopping_patience': 10,
            'checkpoint_dir': './checkpoints',
            'log_interval': 10,
            'main_metric': 'accuracy',
            'metric_direction': 'max',
            # 情感任务相关配置
            'emotion_weight': 0.6,        # 主情感分类损失权重
            'sub_emotion_weight': 0.3,    # 子情感分类损失权重
            'intensity_weight': 0.1,      # 情感强度回归损失权重
            'has_sub_emotion_labels': False  # 是否有子情感标签
        }
        
        # 更新用户配置
        if config:
            self.config.update(config)
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 创建损失函数
        self.criterion_emotion = nn.CrossEntropyLoss()
        self.criterion_sub_emotion = nn.CrossEntropyLoss()
        self.criterion_intensity = nn.MSELoss()
        
        if hasattr(self.model, 'lm_head'):
            self.criterion_lm = nn.CrossEntropyLoss(ignore_index=self.model.vocab.get_idx('<pad>') if hasattr(self.model, 'vocab') else 0)
        
        # 创建学习率调度器
        # 注意：移除verbose参数以兼容旧版本PyTorch
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.config['metric_direction'],
            factor=0.1,
            patience=3
        )
        
        # 初始化早停机制
        self.early_stopping_counter = 0
        self.best_score = None
        
        # 创建检查点目录
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # 初始化训练历史，扩展以支持更多情感指标
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_emotion_acc': [],
            'val_emotion_acc': [],
            'train_sub_emotion_acc': [],
            'val_sub_emotion_acc': [],
            'train_intensity_mse': [],
            'val_intensity_mse': []
        }
    
    def create_data_loaders(self, train_dataset, val_dataset=None, test_dataset=None):
        """
        创建数据加载器
        """
        # 创建训练数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # 创建验证数据加载器
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4
            )
        
        # 创建测试数据加载器
        self.test_loader = None
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4
            )
    
    def _calculate_accuracy(self, logits, labels):
        """计算准确率"""
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).sum().item()
        return correct / len(labels)
    
    def train_epoch(self):
        """训练一个epoch，支持增强的情感推理功能"""
        self.model.train()
        total_loss = 0
        total_emotion_accuracy = 0
        total_sub_emotion_accuracy = 0
        total_intensity_mse = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            emotion_labels = batch['labels']
            
            # 初始化子情感标签和强度标签
            sub_emotion_labels = None
            intensity_labels = None
            
            # 如果批次中包含子情感标签
            if 'sub_emotion_labels' in batch and self.config['has_sub_emotion_labels']:
                sub_emotion_labels = batch['sub_emotion_labels']
            
            # 如果批次中包含强度标签
            if 'intensity_labels' in batch:
                intensity_labels = batch['intensity_labels']
            
            # 前向传播
            self.optimizer.zero_grad()
            lm_logits, emotion_logits, sub_emotion_logits, emotion_intensity = self.model(input_ids, attention_mask)
            
            # 初始化损失
            loss = 0
            
            # 计算主情感分类损失
            emotion_loss = self.criterion_emotion(emotion_logits, emotion_labels)
            loss += emotion_loss * self.config['emotion_weight']
            
            # 计算子情感分类损失（如果有标签）
            if sub_emotion_labels is not None:
                sub_emotion_loss = self.criterion_sub_emotion(sub_emotion_logits, sub_emotion_labels)
                loss += sub_emotion_loss * self.config['sub_emotion_weight']
            
            # 计算情感强度回归损失（如果有标签）
            if intensity_labels is not None:
                # 确保强度标签形状匹配
                intensity_labels = intensity_labels.view(-1, 1).float()
                intensity_loss = self.criterion_intensity(emotion_intensity, intensity_labels)
                loss += intensity_loss * self.config['intensity_weight']
            
            # 如果有语言建模头，添加语言建模损失
            if lm_logits is not None:
                lm_labels = input_ids[:, 1:]  # 预测下一个token
                lm_logits = lm_logits[:, :-1, :]  # 移除最后一个输出
                lm_loss = self.criterion_lm(lm_logits.reshape(-1, lm_logits.size(-1)), lm_labels.reshape(-1))
                loss += lm_loss
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
            
            self.optimizer.step()
            
            # 累积损失和准确率
            total_loss += loss.item()
            emotion_accuracy = self._calculate_accuracy(emotion_logits, emotion_labels)
            total_emotion_accuracy += emotion_accuracy
            
            # 如果有子情感标签，计算子情感准确率
            if sub_emotion_labels is not None:
                sub_emotion_accuracy = self._calculate_accuracy(sub_emotion_logits, sub_emotion_labels)
                total_sub_emotion_accuracy += sub_emotion_accuracy
            
            # 如果有强度标签，计算强度MSE
            if intensity_labels is not None:
                intensity_mse = self.criterion_intensity(emotion_intensity, intensity_labels).item()
                total_intensity_mse += intensity_mse
            
            # 打印进度
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                log_msg = f'Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}, Emotion Acc: {emotion_accuracy:.4f}'
                if sub_emotion_labels is not None:
                    log_msg += f', Sub-Emotion Acc: {sub_emotion_accuracy:.4f}'
                if intensity_labels is not None:
                    log_msg += f', Intensity MSE: {intensity_mse:.4f}'
                print(log_msg)
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        avg_emotion_accuracy = total_emotion_accuracy / len(self.train_loader)
        avg_sub_emotion_accuracy = total_sub_emotion_accuracy / len(self.train_loader) if self.config['has_sub_emotion_labels'] else 0
        avg_intensity_mse = total_intensity_mse / len(self.train_loader) if total_intensity_mse > 0 else 0
        
        return avg_loss, avg_emotion_accuracy, avg_sub_emotion_accuracy, avg_intensity_mse
    
    def evaluate(self, loader=None):
        """评估模型，支持增强的情感推理功能"""
        self.model.eval()
        total_loss = 0
        total_emotion_accuracy = 0
        total_sub_emotion_accuracy = 0
        total_intensity_mse = 0
        
        # 默认使用验证加载器
        if loader is None:
            loader = self.val_loader
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                emotion_labels = batch['labels']
                
                # 初始化子情感标签和强度标签
                sub_emotion_labels = None
                intensity_labels = None
                
                # 如果批次中包含子情感标签
                if 'sub_emotion_labels' in batch and self.config['has_sub_emotion_labels']:
                    sub_emotion_labels = batch['sub_emotion_labels']
                
                # 如果批次中包含强度标签
                if 'intensity_labels' in batch:
                    intensity_labels = batch['intensity_labels']
                
                # 前向传播
                lm_logits, emotion_logits, sub_emotion_logits, emotion_intensity = self.model(input_ids, attention_mask)
                
                # 初始化损失
                loss = 0
                
                # 计算主情感分类损失
                emotion_loss = self.criterion_emotion(emotion_logits, emotion_labels)
                loss += emotion_loss * self.config['emotion_weight']
                
                # 计算子情感分类损失（如果有标签）
                if sub_emotion_labels is not None:
                    sub_emotion_loss = self.criterion_sub_emotion(sub_emotion_logits, sub_emotion_labels)
                    loss += sub_emotion_loss * self.config['sub_emotion_weight']
                
                # 计算情感强度回归损失（如果有标签）
                if intensity_labels is not None:
                    # 确保强度标签形状匹配
                    intensity_labels = intensity_labels.view(-1, 1).float()
                    intensity_loss = self.criterion_intensity(emotion_intensity, intensity_labels)
                    loss += intensity_loss * self.config['intensity_weight']
                
                # 如果有语言建模头，添加语言建模损失
                if lm_logits is not None:
                    lm_labels = input_ids[:, 1:]
                    lm_logits = lm_logits[:, :-1, :]
                    lm_loss = self.criterion_lm(lm_logits.reshape(-1, lm_logits.size(-1)), lm_labels.reshape(-1))
                    loss += lm_loss
                
                # 累积损失和准确率
                total_loss += loss.item()
                emotion_accuracy = self._calculate_accuracy(emotion_logits, emotion_labels)
                total_emotion_accuracy += emotion_accuracy
                
                # 如果有子情感标签，计算子情感准确率
                if sub_emotion_labels is not None:
                    sub_emotion_accuracy = self._calculate_accuracy(sub_emotion_logits, sub_emotion_labels)
                    total_sub_emotion_accuracy += sub_emotion_accuracy
                
                # 如果有强度标签，计算强度MSE
                if intensity_labels is not None:
                    intensity_mse = self.criterion_intensity(emotion_intensity, intensity_labels).item()
                    total_intensity_mse += intensity_mse
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(loader)
        avg_emotion_accuracy = total_emotion_accuracy / len(loader)
        avg_sub_emotion_accuracy = total_sub_emotion_accuracy / len(loader) if self.config['has_sub_emotion_labels'] else 0
        avg_intensity_mse = total_intensity_mse / len(loader) if total_intensity_mse > 0 else 0
        
        return avg_loss, avg_emotion_accuracy, avg_sub_emotion_accuracy, avg_intensity_mse
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pt')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': self.best_score,
            'train_history': self.train_history,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
            torch.save(checkpoint, best_checkpoint_path)
            print(f'Best model saved to {best_checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        try:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # 加载模型状态
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # 加载优化器状态
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # 加载调度器状态
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # 加载最佳分数和训练历史
                if 'best_score' in checkpoint:
                    self.best_score = checkpoint['best_score']
                
                if 'train_history' in checkpoint:
                    self.train_history = checkpoint['train_history']
                
                # 加载配置
                if 'config' in checkpoint:
                    self.config.update(checkpoint['config'])
                
                epoch = checkpoint.get('epoch', 0)
                print(f'Loaded checkpoint from epoch {epoch}')
                return epoch
            else:
                print(f'Checkpoint file not found: {checkpoint_path}')
                return 0
        except Exception as e:
            print(f'Error loading checkpoint: {e}')
            return 0
    
    def train(self):
        """训练模型，支持增强的情感推理功能"""
        # 初始化最佳分数
        if self.best_score is None:
            if self.config['metric_direction'] == 'max':
                self.best_score = -float('inf')
            else:
                self.best_score = float('inf')
        
        # 主训练循环
        for epoch in range(self.config['epochs']):
            print(f'\nEpoch {epoch+1}/{self.config['epochs']}')
            print('-' * 50)
            
            # 训练一个epoch - 新的返回值格式: loss, emotion_acc, sub_emotion_acc, intensity_mse
            train_loss, train_emotion_acc, train_sub_emotion_acc, train_intensity_mse = self.train_epoch()
            
            # 评估模型
            val_loss, val_emotion_acc, val_sub_emotion_acc, val_intensity_mse = None, None, None, None
            if self.val_loader:
                val_loss, val_emotion_acc, val_sub_emotion_acc, val_intensity_mse = self.evaluate()
            
            # 更新训练历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_emotion_acc'].append(train_emotion_acc)
            self.train_history['train_sub_emotion_acc'].append(train_sub_emotion_acc)
            self.train_history['train_intensity_mse'].append(train_intensity_mse)
            
            if val_loss is not None:
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_emotion_acc'].append(val_emotion_acc)
                self.train_history['val_sub_emotion_acc'].append(val_sub_emotion_acc)
                self.train_history['val_intensity_mse'].append(val_intensity_mse)
            
            # 打印epoch总结
            print(f'\nEpoch Summary:')
            print(f'Train Loss: {train_loss:.4f}, Train Emotion Acc: {train_emotion_acc:.4f}')
            if self.config['has_sub_emotion_labels']:
                print(f'Train Sub-Emotion Acc: {train_sub_emotion_acc:.4f}')
            if train_intensity_mse > 0:
                print(f'Train Intensity MSE: {train_intensity_mse:.4f}')
            
            if val_loss is not None:
                print(f'Validation Loss: {val_loss:.4f}, Validation Emotion Acc: {val_emotion_acc:.4f}')
                if self.config['has_sub_emotion_labels']:
                    print(f'Validation Sub-Emotion Acc: {val_sub_emotion_acc:.4f}')
                if val_intensity_mse > 0:
                    print(f'Validation Intensity MSE: {val_intensity_mse:.4f}')
            
            # 检查是否是最佳模型
            current_score = val_emotion_acc if val_emotion_acc is not None else train_emotion_acc
            is_best = False
            
            if self.config['metric_direction'] == 'max':
                if current_score > self.best_score:
                    self.best_score = current_score
                    is_best = True
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
            else:
                if current_score < self.best_score:
                    self.best_score = current_score
                    is_best = True
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
            
            # 更新学习率
            if val_loss is not None:
                self.scheduler.step(val_loss)
            
            # 保存检查点
            self.save_checkpoint(epoch+1, is_best)
            
            # 检查早停条件
            if self.config['early_stopping_patience'] > 0 and self.early_stopping_counter >= self.config['early_stopping_patience']:
                print(f'\nEarly stopping triggered after {self.early_stopping_counter} epochs without improvement')
                break
        
        print('\nTraining completed!')
        return self.train_history

class LanguageModelTrainer:
    """
    语言模型训练管理器类，提供高级训练接口
    整合从零开始训练和预训练模型功能
    """
    def __init__(self, config=None):
        """
        初始化训练管理器
        
        参数:
            config: 训练配置
        """
        # 设置默认配置
        self.config = self._get_default_config()
        
        # 更新用户配置
        if config:
            self.config.update(config)
        
        # 初始化数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 初始化模型和训练器
        self.model = None
        self.trainer = None
        
        # 加载语言资源
        self.language_resources = self._load_language_resources()
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'model_id': 'B_language',
            'vocab_size': 30000,
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.3,
            'data_path': './training_data',
            'train_size': 0.8,
            'val_size': 0.1,
            'max_length': 128,
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'gradient_clipping': 1.0,
            'early_stopping_patience': 10,
            'checkpoint_dir': './checkpoints/B_language',
            'log_interval': 10,
            'main_metric': 'accuracy',
            'metric_direction': 'max',
            'has_sub_emotion_labels': False,  # 默认不包含子情感标签
            'emotion_weight': 0.6,
            'sub_emotion_weight': 0.3,
            'intensity_weight': 0.1
        }
    
    def _load_language_resources(self):
        """加载语言资源"""
        # 定义情感标签映射（多语言）
        emotion_labels = {
            'en': {
                'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'fear': 4, 'surprise': 5, 'disgust': 6
            },
            'zh': {
                '中性': 0, '快乐': 1, '悲伤': 2, '愤怒': 3, '恐惧': 4, '惊讶': 5, '厌恶': 6
            },
            'ja': {
                'ニュートラル': 0, '喜び': 1, '悲しみ': 2, '怒り': 3, '恐怖': 4, '驚き': 5, '嫌悪': 6
            },
            'de': {
                'neutral': 0, 'freude': 1, 'traurigkeit': 2, 'wut': 3, 'angst': 4, 'überraschung': 5, 'abscheu': 6
            },
            'fr': {
                'neutre': 0, 'joie': 1, 'tristesse': 2, 'colère': 3, 'peur': 4, 'surprise': 5, 'dégoût': 6
            }
        }
        
        return {'emotion_labels': emotion_labels}
    
    def prepare_dataset(self, data_path=None, task_type='sentiment'):
        """
        准备数据集
        
        参数:
            data_path: 数据路径
            task_type: 任务类型 (sentiment, ner, etc.)
        """
        if data_path is None:
            data_path = self.config['data_path']
        
        # 创建主数据集
        dataset = LanguageCorpusDataset(
            data_path=data_path,
            max_length=self.config['max_length'],
            vocab_size=self.config['vocab_size']
        )
        
        # 分割数据集
        dataset_size = len(dataset)
        train_size = int(self.config['train_size'] * dataset_size)
        val_size = int(self.config['val_size'] * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 根据任务类型进行特定处理
        if task_type == 'sentiment':
            # 情感分析任务特定处理
            pass  # 基本实现已包含在LanguageCorpusDataset中
        elif task_type == 'ner':
            # 命名实体识别任务特定处理
            pass  # 可根据需要扩展
        
        print(f'Dataset prepared: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test samples')
        
        if dataset.vocab:
            print(f'Vocabulary size: {len(dataset.vocab)}')
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def initialize_model(self):
        """
        初始化模型
        """
        # 确保数据集已准备好以获取词汇表大小
        if self.train_dataset is None:
            self.prepare_dataset()
        
        # 获取词汇表大小
        vocab_size = len(self.train_dataset.dataset.vocab)
        
        self.model = LanguageModel(
            vocab_size=vocab_size,
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        # 打印模型架构摘要
        print('\nModel Architecture:')
        print(self.model)
        
        # 计算模型参数数量
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f'Total parameters: {param_count:,}')
        
        # 初始化训练器
        self.trainer = ModelTrainer(self.model, self.config)
        
        # 创建数据加载器
        self.trainer.create_data_loaders(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset
        )
    
    def train(self, resume_from_checkpoint=None):
        """
        训练模型
        
        参数:
            resume_from_checkpoint: 从哪个检查点恢复训练
        """
        # 确保模型已初始化
        if self.model is None:
            self.initialize_model()
        
        # 如果指定了检查点，加载它
        if resume_from_checkpoint:
            self.trainer.load_checkpoint(resume_from_checkpoint)
        
        # 开始训练
        train_history = self.trainer.train()
        
        # 保存最终训练报告
        self.save_training_report(train_history)
        
        return train_history
    
    def evaluate(self, dataset_type='test'):
        """评估模型
        
        参数:
            dataset_type: 评估数据集类型 (train, val, test)
        """
        # 确保模型已初始化
        if self.model is None:
            print('Model not initialized. Please call initialize_model() first.')
            return None
        
        # 选择评估数据集
        if dataset_type == 'train':
            loader = self.trainer.train_loader
        elif dataset_type == 'val':
            loader = self.trainer.val_loader
        elif dataset_type == 'test':
            loader = self.trainer.test_loader
        else:
            print(f'Invalid dataset type: {dataset_type}')
            return None
        
        # 评估模型 - 新的返回值格式: loss, emotion_acc, sub_emotion_acc, intensity_mse
        loss, emotion_acc, sub_emotion_acc, intensity_mse = self.trainer.evaluate(loader)
        
        print(f'\n{dataset_type.capitalize()} Evaluation:')
        print(f'Loss: {loss:.4f}, Emotion Accuracy: {emotion_acc:.4f}')
        if self.config['has_sub_emotion_labels']:
            print(f'Sub-Emotion Accuracy: {sub_emotion_acc:.4f}')
        if intensity_mse > 0:
            print(f'Intensity MSE: {intensity_mse:.4f}')
        
        # 计算额外指标
        metrics = self._compute_metrics(loader)
        
        return {
            'loss': loss,
            'emotion_accuracy': emotion_acc,
            'sub_emotion_accuracy': sub_emotion_acc,
            'intensity_mse': intensity_mse,
            **metrics
        }
    
    def _compute_metrics(self, loader):
        """
        计算额外评估指标
        """
        self.model.eval()
        all_emotion_predictions = []
        all_emotion_labels = []
        all_emotion_logits = []
        all_sub_emotion_predictions = []
        all_sub_emotion_labels = []
        all_sub_emotion_logits = []
        all_intensity_predictions = []
        all_intensity_labels = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                emotion_labels = batch['labels']
                
                # 初始化子情感标签和强度标签
                sub_emotion_labels = None
                intensity_labels = None
                
                # 如果批次中包含子情感标签
                if 'sub_emotion_labels' in batch and self.config.get('has_sub_emotion_labels', False):
                    sub_emotion_labels = batch['sub_emotion_labels']
                
                # 如果批次中包含强度标签
                if 'intensity_labels' in batch:
                    intensity_labels = batch['intensity_labels']
                
                # 前向传播
                _, emotion_logits, sub_emotion_logits, emotion_intensity = self.model(input_ids, attention_mask)
                
                # 获取主情感预测
                emotion_predictions = torch.argmax(emotion_logits, dim=1)
                
                # 收集主情感结果
                all_emotion_predictions.extend(emotion_predictions.cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_emotion_logits.extend(emotion_logits.cpu().numpy())
                
                # 如果有子情感标签，收集子情感结果
                if sub_emotion_labels is not None:
                    sub_emotion_predictions = torch.argmax(sub_emotion_logits, dim=1)
                    all_sub_emotion_predictions.extend(sub_emotion_predictions.cpu().numpy())
                    all_sub_emotion_labels.extend(sub_emotion_labels.cpu().numpy())
                    all_sub_emotion_logits.extend(sub_emotion_logits.cpu().numpy())
                
                # 如果有强度标签，收集强度结果
                if intensity_labels is not None:
                    all_intensity_predictions.extend(emotion_intensity.cpu().numpy())
                    all_intensity_labels.extend(intensity_labels.cpu().numpy())
        
        # 转换为numpy数组
        all_emotion_predictions = np.array(all_emotion_predictions)
        all_emotion_labels = np.array(all_emotion_labels)
        all_emotion_logits = np.array(all_emotion_logits)
        
        # 计算主情感精确率、召回率和F1分数
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        emotion_precision = precision_score(all_emotion_labels, all_emotion_predictions, average='weighted')
        emotion_recall = recall_score(all_emotion_labels, all_emotion_predictions, average='weighted')
        emotion_f1 = f1_score(all_emotion_labels, all_emotion_predictions, average='weighted')
        
        # 计算主情感强度分析结果
        emotion_intensity = self._calculate_emotion_intensity(all_emotion_logits)
        
        metrics = {
            'emotion_precision': emotion_precision,
            'emotion_recall': emotion_recall,
            'emotion_f1_score': emotion_f1,
            'emotion_intensity': emotion_intensity
        }
        
        # 如果有子情感标签，计算子情感指标
        if len(all_sub_emotion_labels) > 0:
            all_sub_emotion_predictions = np.array(all_sub_emotion_predictions)
            all_sub_emotion_labels = np.array(all_sub_emotion_labels)
            all_sub_emotion_logits = np.array(all_sub_emotion_logits)
            
            sub_emotion_precision = precision_score(all_sub_emotion_labels, all_sub_emotion_predictions, average='weighted')
            sub_emotion_recall = recall_score(all_sub_emotion_labels, all_sub_emotion_predictions, average='weighted')
            sub_emotion_f1 = f1_score(all_sub_emotion_labels, all_sub_emotion_predictions, average='weighted')
            
            metrics.update({
                'sub_emotion_precision': sub_emotion_precision,
                'sub_emotion_recall': sub_emotion_recall,
                'sub_emotion_f1_score': sub_emotion_f1
            })
        
        # 如果有强度标签，计算强度指标
        if len(all_intensity_labels) > 0:
            all_intensity_predictions = np.array(all_intensity_predictions)
            all_intensity_labels = np.array(all_intensity_labels)
            
            # 计算MSE和MAE
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            intensity_mse = mean_squared_error(all_intensity_labels, all_intensity_predictions)
            intensity_mae = mean_absolute_error(all_intensity_labels, all_intensity_predictions)
            
            metrics.update({
                'intensity_mse': intensity_mse,
                'intensity_mae': intensity_mae
            })
        
        return metrics
    
    def _calculate_emotion_intensity(self, logits, emotion_type='main', sub_emotion_map=None):
        """
        计算情感强度分析结果
        
        参数:
            logits: 模型输出的原始预测值
            emotion_type: 情感类型，'main'表示主情感，'sub'表示子情感
            sub_emotion_map: 子情感映射字典，键为主情感ID，值为子情感名称列表
        
        返回:
            情感强度字典
        """
        # 转换为概率
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        
        # 计算每种情感的平均强度
        avg_intensity = {}
        
        if emotion_type == 'main':
            # 主情感名称
            emotion_names = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            for i, emotion in enumerate(emotion_names):
                avg_intensity[emotion] = float(np.mean(probs[:, i]))
        elif emotion_type == 'sub' and sub_emotion_map is not None:
            # 子情感强度计算
            for main_emotion_id, sub_emotions in sub_emotion_map.items():
                # 确保main_emotion_id是有效索引
                if isinstance(main_emotion_id, int) and main_emotion_id < len(probs[0]):
                    sub_emotion_key = f"sub_emotion_{main_emotion_id}"
                    avg_intensity[sub_emotion_key] = float(np.mean(probs[:, main_emotion_id]))
        
        return avg_intensity
    
    def save_training_report(self, train_history):
        """
        保存训练报告，包括主情感和子情感的所有指标
        """
        report = {
            'model_id': self.config['model_id'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'training_history': train_history,
            'best_score': self.trainer.best_score,
            'model_summary': {
                'total_parameters': sum(p.numel() for p in self.model.parameters())
            }
        }
        
        # 计算训练过程中的平均指标
        report['training_metrics_summary'] = {
            'avg_train_loss': float(np.mean([h['loss'] for h in train_history['train']])),
            'avg_val_loss': float(np.mean([h['loss'] for h in train_history['val']])),
            'avg_train_emotion_acc': float(np.mean([h['emotion_acc'] for h in train_history['train']])),
            'avg_val_emotion_acc': float(np.mean([h['emotion_acc'] for h in train_history['val']]))
        }
        
        # 如果训练历史中包含子情感指标，也计算平均值
        if len(train_history['train']) > 0 and 'sub_emotion_acc' in train_history['train'][0]:
            report['training_metrics_summary'].update({
                'avg_train_sub_emotion_acc': float(np.mean([h['sub_emotion_acc'] for h in train_history['train']])),
                'avg_val_sub_emotion_acc': float(np.mean([h['sub_emotion_acc'] for h in train_history['val']]))
            })
        
        # 如果训练历史中包含情感强度指标，也计算平均值
        if len(train_history['train']) > 0 and 'intensity_mse' in train_history['train'][0]:
            report['training_metrics_summary'].update({
                'avg_train_intensity_mse': float(np.mean([h['intensity_mse'] for h in train_history['train']])),
                'avg_val_intensity_mse': float(np.mean([h['intensity_mse'] for h in train_history['val']]))
            })
        
        # 保存报告
        report_path = os.path.join(self.config['checkpoint_dir'], 'training_report.json')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f'Training report saved to {report_path}')

# 模型保存和加载函数
def save_model(model, path):
    """保存模型到文件，包含主情感、子情感和情感强度组件"""
    # 准备配置信息
    config = {
        'hidden_dim': model.hidden_dim,
        'num_emotion_classes': model.num_emotion_classes,
        'num_sub_emotion_classes': model.num_sub_emotion_classes,
        'has_sub_emotions': hasattr(model, 'sub_emotion_head') and model.sub_emotion_head is not None,
        'has_emotion_intensity': hasattr(model, 'emotion_intensity_head') and model.emotion_intensity_head is not None
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, path)

def load_model(path, vocab_size=None):
    """从文件加载模型，支持包含子情感和情感强度组件的模型"""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get('config', {})
    
    # 构建模型配置
    model_config = {
        'vocab_size': vocab_size,
        'hidden_dim': config.get('hidden_dim', 512)
    }
    
    # 加载自定义模型
    model = LanguageModel(**model_config)
    
    # 加载模型状态字典
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

# 训练API接口
def start_training(config=None):
    """
    启动模型训练的API接口
    
    参数:
        config: 训练配置
    
    返回:
        训练结果
    """
    try:
        # 创建训练管理器
        trainer = LanguageModelTrainer(config)
        
        # 准备数据集
        trainer.prepare_dataset()
        
        # 初始化模型
        trainer.initialize_model()
        
        # 开始训练
        train_history = trainer.train()
        
        # 评估模型
        evaluation_results = trainer.evaluate()
        
        return {
            'status': 'success',
            'message': 'Training completed successfully',
            'train_history': train_history,
            'evaluation_results': evaluation_results,
            'config': trainer.config
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

def resume_training(checkpoint_path, config=None):
    """
    从检查点恢复训练的API接口
    
    参数:
        checkpoint_path: 检查点路径
        config: 训练配置
    
    返回:
        训练结果
    """
    try:
        # 创建训练管理器
        trainer = LanguageModelTrainer(config)
        
        # 准备数据集
        trainer.prepare_dataset()
        
        # 初始化模型
        trainer.initialize_model()
        
        # 从检查点恢复训练
        train_history = trainer.train(resume_from_checkpoint=checkpoint_path)
        
        # 评估模型
        evaluation_results = trainer.evaluate()
        
        return {
            'status': 'success',
            'message': 'Training resumed successfully',
            'train_history': train_history,
            'evaluation_results': evaluation_results,
            'config': trainer.config
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

# 联合训练API接口
def start_joint_training(model_configs, data_paths, config=None, loss_weights=None):
    """
    启动联合训练的API接口
    
    参数:
        model_configs: 模型配置列表
        data_paths: 数据路径列表
        config: 联合训练配置
        loss_weights: 损失权重列表
    
    返回:
        联合训练结果
    """
    try:
        # 确保模型配置和数据路径数量匹配
        if len(model_configs) != len(data_paths):
            raise ValueError('Number of model configurations must match number of data paths')
        
        # 设置默认配置
        default_config = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'checkpoint_dir': './checkpoints',
            'joint_training_type': 'parallel',  # parallel 或 sequential
            'dominant_model_index': 0  # 主导模型索引
        }
        
        if config:
            default_config.update(config)
        
        # 设置损失权重（如果未提供，默认平均）
        if loss_weights is None:
            loss_weights = [1.0 / len(model_configs)] * len(model_configs)
        else:
            # 归一化损失权重
            total_weight = sum(loss_weights)
            loss_weights = [w / total_weight for w in loss_weights]
        
        # 实现简化的联合训练逻辑
        # 对于真实场景，这里需要更复杂的实现来处理多模型交互
        
        print(f'Starting joint training with {len(model_configs)} models')
        
        # 为每个模型创建训练器
        trainers = []
        for i, (model_config, data_path) in enumerate(zip(model_configs, data_paths)):
            print(f'Preparing model {i+1}/{len(model_configs)}')
            
            # 合并配置
            combined_config = default_config.copy()
            if model_config:
                combined_config.update(model_config)
            
            # 设置数据路径
            combined_config['data_path'] = data_path
            
            # 创建训练管理器
            trainer = LanguageModelTrainer(combined_config)
            
            # 准备数据集
            trainer.prepare_dataset()
            
            # 初始化模型
            trainer.initialize_model()
            
            trainers.append(trainer)
        
        # 简化实现：分别训练每个模型
        results = []
        for i, trainer in enumerate(trainers):
            print(f'\nTraining model {i+1}/{len(trainers)}')
            
            # 训练模型
            train_history = trainer.train()
            
            # 评估模型
            evaluation_results = trainer.evaluate()
            
            # 记录结果
            results.append({
                'model_index': i,
                'train_history': train_history,
                'evaluation_results': evaluation_results,
                'config': trainer.config
            })
        
        # 计算联合指标
        # 初始化所有可能的指标
        joint_metrics = {
            'loss_weights': loss_weights
        }
        
        # 检查是否所有结果都包含情感准确率
        has_emotion_acc = all('emotion_accuracy' in r['evaluation_results'] for r in results)
        if has_emotion_acc:
            joint_metrics.update({
                'average_emotion_accuracy': np.mean([r['evaluation_results']['emotion_accuracy'] for r in results]),
                'weighted_emotion_accuracy': np.sum([r['evaluation_results']['emotion_accuracy'] * loss_weights[i] for i, r in enumerate(results)])
            })
        elif all('accuracy' in r['evaluation_results'] for r in results):
            # 向后兼容
            joint_metrics.update({
                'average_accuracy': np.mean([r['evaluation_results']['accuracy'] for r in results]),
                'weighted_accuracy': np.sum([r['evaluation_results']['accuracy'] * loss_weights[i] for i, r in enumerate(results)])
            })
        
        # 检查是否所有结果都包含F1分数
        has_f1_score = all('emotion_f1_score' in r['evaluation_results'] for r in results)
        if has_f1_score:
            joint_metrics.update({
                'average_emotion_f1_score': np.mean([r['evaluation_results']['emotion_f1_score'] for r in results]),
                'weighted_emotion_f1_score': np.sum([r['evaluation_results']['emotion_f1_score'] * loss_weights[i] for i, r in enumerate(results)])
            })
        elif all('f1_score' in r['evaluation_results'] for r in results):
            # 向后兼容
            joint_metrics.update({
                'average_f1_score': np.mean([r['evaluation_results']['f1_score'] for r in results]),
                'weighted_f1_score': np.sum([r['evaluation_results']['f1_score'] * loss_weights[i] for i, r in enumerate(results)])
            })
        
        # 检查是否所有结果都包含子情感准确率
        has_sub_emotion_acc = all('sub_emotion_accuracy' in r['evaluation_results'] for r in results)
        if has_sub_emotion_acc:
            joint_metrics.update({
                'average_sub_emotion_accuracy': np.mean([r['evaluation_results']['sub_emotion_accuracy'] for r in results]),
                'weighted_sub_emotion_accuracy': np.sum([r['evaluation_results']['sub_emotion_accuracy'] * loss_weights[i] for i, r in enumerate(results)])
            })
        
        # 检查是否所有结果都包含情感强度MSE
        has_intensity_mse = all('intensity_mse' in r['evaluation_results'] for r in results)
        if has_intensity_mse:
            joint_metrics.update({
                'average_intensity_mse': np.mean([r['evaluation_results']['intensity_mse'] for r in results]),
                'weighted_intensity_mse': np.sum([r['evaluation_results']['intensity_mse'] * loss_weights[i] for i, r in enumerate(results)])
            })
        
        # 保存联合训练报告
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': default_config,
            'joint_metrics': joint_metrics,
            'individual_results': results
        }
        
        report_path = os.path.join(default_config['checkpoint_dir'], 'joint_training_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f'Joint training report saved to {report_path}')
        
        return {
            'status': 'success',
            'message': 'Joint training completed successfully',
            'joint_metrics': joint_metrics,
            'individual_results': results,
            'config': default_config
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

# 主函数，用于测试训练功能
if __name__ == '__main__':
    # 测试从零开始训练
    print("\n--- 测试从零开始训练 ---")
    from_scratch_config = {
        'model_id': 'from_scratch',
        'vocab_size': 10000,
        'epochs': 2,
        'batch_size': 8,
        'learning_rate': 1e-3
    }
    
    from_scratch_result = start_training(from_scratch_config)
    print("从零开始训练结果 | From scratch training result:", json.dumps(from_scratch_result, indent=2, ensure_ascii=False))
