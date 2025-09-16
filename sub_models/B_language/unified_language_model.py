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
统一大语言模型实现 | Unified Large Language Model Implementation
整合基础功能和增强功能，支持配置驱动的功能开关
(Integrates basic and enhanced features with configuration-driven feature toggles)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, get_scheduler
from torch.utils.data import Dataset, DataLoader, IterableDataset, ConcatDataset
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
import json
import logging
from datetime import datetime
import requests
import psutil
import gc
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from torch.utils.tensorboard import SummaryWriter

# 设置日志 | Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_language_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UnifiedLanguageModel")

class LanguageDataset(Dataset):
    """语言模型数据集类 | Language Model Dataset Class"""
    def __init__(self, texts, emotion_labels, lm_labels, tokenizer, max_length=512):
        self.texts = texts
        self.emotion_labels = emotion_labels
        self.lm_labels = lm_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        emotion_label = self.emotion_labels[idx]
        
        # 分词和编码 | Tokenize and encode
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移除批次维度 | Remove batch dimension
        item = {
            key: val.squeeze(0) for key, val in encoding.items()
        }
        item["emotion_labels"] = torch.tensor(emotion_label, dtype=torch.long)
        
        # 为语言建模创建标签 | Create labels for language modeling
        if isinstance(self.lm_labels[idx], torch.Tensor):
            item["lm_labels"] = self.lm_labels[idx].clone()
        else:
            # 如果LM标签是文本，对其进行编码 | If LM labels are text, encode them
            lm_encoding = self.tokenizer(
                self.lm_labels[idx],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            item["lm_labels"] = lm_encoding["input_ids"].squeeze(0)
        
        return item

class StreamDataLoader(IterableDataset):
    """流式数据加载器 | Streaming Data Loader"""
    def __init__(self, data_paths: List[str], tokenizer, emotion_map, batch_size=16, max_length=512):
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.emotion_map = emotion_map
        self.batch_size = batch_size
        self.max_length = max_length
        self.current_file_index = 0
        self.data_buffer = []
        self.buffer_size = 1000  # 缓冲区大小
        self.processed_files = set()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        # 如果缓冲区为空，加载下一个文件 | If buffer is empty, load next file
        if not self.data_buffer:
            self._load_next_file()
        
        if not self.data_buffer:
            raise StopIteration
        
        # 从缓冲区获取批次 | Get batch from buffer
        batch_size = min(self.batch_size, len(self.data_buffer))
        batch = self.data_buffer[:batch_size]
        self.data_buffer = self.data_buffer[batch_size:]
        
        # 处理批次数据 | Process batch data
        texts = [item["text"] for item in batch]
        emotions = [item["emotion"] for item in batch]
        responses = [item.get("response", "") for item in batch]
        
        # 转换情感标签为ID | Convert emotion labels to IDs
        emotion_ids = [self.emotion_map.get(emotion, 0) for emotion in emotions]
        
        # 分词和编码 | Tokenize and encode
        input_encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 为语言建模创建标签 | Create labels for language modeling
        lm_encoding = self.tokenizer(
            responses,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 返回批次 | Return batch
        return {
            "input_ids": input_encoding["input_ids"],
            "attention_mask": input_encoding["attention_mask"],
            "emotion_labels": torch.tensor(emotion_ids, dtype=torch.long),
            "lm_labels": lm_encoding["input_ids"]
        }
    
    def _load_next_file(self):
        """加载下一个数据文件 | Load next data file"""
        while self.current_file_index < len(self.data_paths):
            file_path = self.data_paths[self.current_file_index]
            
            # 如果文件已处理过，跳过 | Skip if file has been processed
            if file_path in self.processed_files:
                self.current_file_index += 1
                continue
            
            try:
                # 加载文件内容 | Load file content
                if file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                elif file_path.endswith('.csv'):
                    data = pd.read_csv(file_path).to_dict('records')
                else:
                    logger.warning(f"不支持的文件格式: {file_path} | Unsupported file format: {file_path}")
                    self.current_file_index += 1
                    continue
                
                # 过滤有效的数据条目 | Filter valid data entries
                valid_entries = []
                for item in data:
                    if isinstance(item, dict) and 'text' in item and 'emotion' in item:
                        valid_entries.append(item)
                
                # 如果有有效条目，添加到缓冲区 | Add to buffer if there are valid entries
                if valid_entries:
                    self.data_buffer.extend(valid_entries)
                    logger.info(f"已加载文件: {file_path}, 添加了 {len(valid_entries)} 条数据 | Loaded file: {file_path}, added {len(valid_entries)} entries")
                else:
                    logger.warning(f"文件不包含有效数据: {file_path} | File contains no valid data: {file_path}")
                
                # 标记文件为已处理 | Mark file as processed
                self.processed_files.add(file_path)
                
                # 如果缓冲区已满或已处理完所有文件，停止加载 | Stop loading if buffer is full or all files processed
                if len(self.data_buffer) >= self.buffer_size:
                    break
                    
            except Exception as e:
                logger.error(f"加载文件失败: {file_path}, 错误: {str(e)} | Failed to load file: {file_path}, error: {str(e)}")
            
            # 移动到下一个文件 | Move to next file
            self.current_file_index += 1

class UnifiedLanguageModel(nn.Module):
    """
    统一多语言情感大语言模型
    (Unified Multilingual Emotional Large Language Model)
    支持配置驱动的功能级别控制
    (Supports configuration-driven feature level control)
    """
    
    def __init__(self, model_config: Dict = None, mode: str = "enhanced"):
        """
        初始化统一语言模型
        (Initialize unified language model)
        
        参数 Parameters:
        model_config: 模型配置字典 | Model configuration dictionary
        mode: 运行模式 "basic" | "enhanced" | Running mode
        """
        super().__init__()
        
        # 加载配置 | Load configuration
        self.config = self._get_default_config()
        if model_config:
            self.config.update(model_config)
        
        self.mode = mode
        self.is_enhanced = (mode == "enhanced")
        
        # 基础属性 | Basic attributes
        self.external_models = {}
        self.training_history = []
        self.performance_metrics = {}
        self.best_metrics = None
        self.validation_history = []
        self.last_evaluation = None
        
        # 情感分类 | Emotion categories
        self.emotion_categories = {
            "anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4,
            "sadness": 5, "surprise": 6
        }
        
        if self.is_enhanced:
            self.emotion_categories.update({
                "excitement": 7, "confusion": 8, "curiosity": 9, "love": 10,
                "gratitude": 11, "pride": 12, "shame": 13, "anxiety": 14
            })
        
        # 支持的语言 | Supported languages
        self.supported_languages = ["zh", "en", "de", "ja", "ru", "fr", "es", "it"]
        
        # 初始化知识迁移和增量学习组件 | Initialize knowledge transfer and incremental learning components
        self.language_adapters = {}  # 语言适配器
        self.knowledge_embeddings = None  # 知识嵌入
        self.important_samples = []  # 重要样本记忆库
        self.memory_size = 1000  # 记忆库大小
        self.memory_update_threshold = 0.6  # 更新记忆库的置信度阈值
        
        # 加载基础模型 | Load base model
        self._load_base_model()
        
        if self.is_enhanced:
            self._init_enhanced_features()
        
        logger.info(f"统一语言模型初始化完成 | Unified language model initialized in {mode} mode")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置 | Get default configuration"""
        return {
            "base_model": "xlm-roberta-base",
            "max_seq_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "num_beams": 4,
            "early_stopping": True,
            "emotion_analysis": True,
            "multilingual_support": True,
            "external_apis": {
                "openai": {"enabled": False},
                "local": {"enabled": True}
            },
            "data_paths": ["./data"],  # 默认数据路径
            "batch_size": 16,
            "learning_rate": 1e-5,
            "epochs": 3,
            "eval_split": 0.2,
            "save_interval": 1,
            "enable_incremental_learning": False,
            "enable_knowledge_transfer": False,
            "early_stopping_patience": 3
        }
    
    def _load_base_model(self):
        """加载基础预训练模型 | Load base pretrained model"""
        try:
            model_name = self.config["base_model"]
            
            # 加载tokenizer | Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型 | Load model
            self.base_model = AutoModel.from_pretrained(model_name)
            
            # 情感分析头 | Emotion analysis head
            self.emotion_head = nn.Linear(
                self.base_model.config.hidden_size, 
                len(self.emotion_categories)
            )
            
            # 语言建模头 | Language modeling head
            self.lm_head = nn.Linear(
                self.base_model.config.hidden_size, 
                self.tokenizer.vocab_size
            )
            
            # 初始化语言适配器 | Initialize language adapters
            self._init_language_adapters()
            
        except Exception as e:
            logger.error(f"模型加载失败: {e} | Model loading failed: {e}")
            raise
    
    def _init_language_adapters(self):
        """初始化语言适配器 | Initialize language adapters"""
        self.language_adapters = nn.ModuleDict({
            lang: nn.Sequential(
                nn.Linear(self.base_model.config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, self.base_model.config.hidden_size),
                nn.LayerNorm(self.base_model.config.hidden_size)
            )
            for lang in self.supported_languages
        })
    
    def _init_enhanced_features(self):
        """初始化增强功能 | Initialize enhanced features"""
        # 启用增量学习和知识迁移 | Enable incremental learning and knowledge transfer
        self.config["enable_incremental_learning"] = True
        self.config["enable_knowledge_transfer"] = True
        
        # 情感强度回归器 | Emotion intensity regressor
        self.intensity_regressor = nn.Linear(
            self.base_model.config.hidden_size, 1
        )
        
        # 加载知识嵌入 | Load knowledge embeddings
        self._load_knowledge_embeddings()
        
        # 初始化TensorBoard记录器 | Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.self_learning_data = []
        self.emotion_intensity_cache = {}
    
    def _load_knowledge_embeddings(self):
        """加载知识嵌入 | Load knowledge embeddings"""
        knowledge_path = Path("knowledge_embeddings.json")
        if knowledge_path.exists():
            try:
                with open(knowledge_path, "r", encoding="utf-8") as f:
                    self.knowledge_embeddings = json.load(f)
                logger.info("知识嵌入加载成功 | Knowledge embeddings loaded successfully")
            except Exception as e:
                logger.warning(f"知识嵌入加载失败: {e} | Failed to load knowledge embeddings: {e}")
                self.knowledge_embeddings = None
        else:
            logger.info("知识嵌入文件不存在 | Knowledge embeddings file not found")
            self.knowledge_embeddings = None
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言 | Detect text language"""
        # 简化的语言检测 | Simplified language detection
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u30ff')
        
        if chinese_chars / len(text) > 0.3:
            return "zh"
        elif japanese_chars / len(text) > 0.3:
            return "ja"
        elif any(c.isalpha() for c in text):
            # 简单检测欧洲语言 | Simple European language detection
            if 'é' in text or 'è' in text or 'à' in text:
                return "fr"
            elif 'ü' in text or 'ä' in text or 'ö' in text:
                return "de"
            elif 'ñ' in text or '¿' in text:
                return "es"
            elif 'ì' in text or 'ò' in text:
                return "it"
            elif 'ы' in text or 'ъ' in text or 'ё' in text:
                return "ru"
            else:
                return "en"
        
        return "unknown"
    
    def _apply_language_adapter(self, sequence_output, language: str):
        """应用语言适配器 | Apply language adapter"""
        if language in self.language_adapters:
            # 将适配器应用于序列输出 | Apply adapter to sequence output
            return self.language_adapters[language](sequence_output)
        return sequence_output
    
    def forward(self, input_ids, attention_mask, language: str = "en"):
        """
        前向传播
        (Forward propagation)
        """
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 应用语言适配器 | Apply language adapter
        if self.config["multilingual_support"] and language in self.language_adapters:
            sequence_output = self._apply_language_adapter(sequence_output, language)
        
        # 情感预测 (Emotion prediction)
        emotion_logits = self.emotion_head(sequence_output[:, 0, :])
        
        # 语言建模 (Language modeling)
        lm_logits = self.lm_head(sequence_output)
        
        if self.is_enhanced:
            # 情感强度预测
            intensity = self.intensity_regressor(sequence_output[:, 0, :])
            return lm_logits, emotion_logits, intensity
        
        return lm_logits, emotion_logits
    
    def predict(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """
        统一预测接口
        (Unified prediction interface)
        
        参数 Parameters:
        text: 输入文本 (Input text)
        language: 语言代码 (Language code)
        
        返回 Returns:
        预测结果字典 (Prediction result dictionary)
        """
        # 如果未指定语言，自动检测 | Auto-detect language if not specified
        if language == "auto":
            language = self._detect_language(text)
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            if self.is_enhanced:
                lm_logits, emotion_logits, intensity = self.forward(**inputs, language=language)
            else:
                lm_logits, emotion_logits = self.forward(**inputs, language=language)
        
        # 情感推理
        emotion_probs = torch.softmax(emotion_logits, dim=-1)
        emotion_id = torch.argmax(emotion_probs).item()
        emotions = list(self.emotion_categories.keys())
        emotion = emotions[emotion_id]
        
        # 生成响应
        generated_ids = torch.argmax(lm_logits, dim=-1)
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        result = {
            "response": response,
            "emotion": emotion,
            "confidence": float(torch.max(emotion_probs)),
            "language": language,
            "mode": self.mode
        }
        
        if self.is_enhanced:
            result["intensity"] = float(intensity[0])
            
            # 记录低置信度预测用于增量学习 | Record low-confidence predictions for incremental learning
            if self.config["enable_incremental_learning"] and result["confidence"] < self.memory_update_threshold:
                self._update_memory_bank(text, emotion_id, result["confidence"], language)
        
        return result
    
    def _update_memory_bank(self, text: str, predicted_label: int, confidence: float, language: str):
        """更新记忆库 | Update memory bank"""
        if len(self.important_samples) >= self.memory_size:
            # 移除置信度最高的样本 | Remove the sample with highest confidence
            self.important_samples.sort(key=lambda x: x["confidence"], reverse=True)
            self.important_samples.pop()
        
        # 添加新样本 | Add new sample
        self.important_samples.append({
            "text": text,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "language": language,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_status(self) -> Dict[str, Any]:
        """获取模型状态信息 | Get model status information"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
        return {
            "status": "active",
            "mode": self.mode,
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "gpu_memory_mb": gpu_memory,
            "parameters_count": sum(p.numel() for p in self.parameters()),
            "supported_languages": self.supported_languages,
            "emotion_categories": list(self.emotion_categories.keys()),
            "last_activity": datetime.now().isoformat(),
            "performance": {
                "inference_speed": "measurement_pending",
                "accuracy": "measurement_pending"
            },
            "training": {
                "has_history": len(self.training_history) > 0,
                "best_metrics": self.best_metrics,
                "last_evaluation": self.last_evaluation.isoformat() if self.last_evaluation else None
            },
            "enhanced_features": {
                "incremental_learning_enabled": self.config["enable_incremental_learning"],
                "knowledge_transfer_enabled": self.config["enable_knowledge_transfer"],
                "memory_bank_size": len(self.important_samples)
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息 | Get model information"""
        return self.get_status()
    
    def _create_mock_data(self, num_samples: int = 100) -> Tuple[List[str], List[int], List[str]]:
        """创建模拟训练数据 | Create mock training data"""
        # 模拟文本数据 | Mock text data
        mock_texts = [
            "我很高兴今天能见到你。", "今天天气真糟糕。", "这部电影很有趣。", "这项服务很差劲。",
            "I'm very happy to see you today.", "The weather is terrible today.", "This movie is very interesting.", "This service is very bad.",
            "Je suis très heureux de vous voir aujourd'hui.", "Le temps est terrible aujourd'hui.", "Ce film est très intéressant.", "Ce service est très mauvais.",
            "Ich bin sehr glücklich, dich heute zu sehen.", "Das Wetter ist heute schrecklich.", "Dieser Film ist sehr interessant.", "Dieser Service ist sehr schlecht.",
            "我感到非常愤怒。", "这让我很惊讶。", "我感到很伤心。", "我很害怕。",
            "I feel very angry.", "This surprises me.", "I feel sad.", "I'm scared.",
            "Estoy muy enojado.", "Esto me sorprende.", "Me siento triste.", "Tengo miedo."
        ]
        
        # 模拟情感标签 | Mock emotion labels
        emotions = list(self.emotion_categories.keys())
        mock_emotion_labels = [
            self.emotion_categories["joy"], self.emotion_categories["sadness"], self.emotion_categories["joy"], self.emotion_categories["disgust"],
            self.emotion_categories["joy"], self.emotion_categories["sadness"], self.emotion_categories["joy"], self.emotion_categories["disgust"],
            self.emotion_categories["joy"], self.emotion_categories["sadness"], self.emotion_categories["joy"], self.emotion_categories["disgust"],
            self.emotion_categories["joy"], self.emotion_categories["sadness"], self.emotion_categories["joy"], self.emotion_categories["disgust"],
            self.emotion_categories["anger"], self.emotion_categories["surprise"], self.emotion_categories["sadness"], self.emotion_categories["fear"],
            self.emotion_categories["anger"], self.emotion_categories["surprise"], self.emotion_categories["sadness"], self.emotion_categories["fear"],
            self.emotion_categories["anger"], self.emotion_categories["surprise"], self.emotion_categories["sadness"], self.emotion_categories["fear"]
        ]
        
        # 模拟响应 | Mock responses
        mock_responses = [
            "我也很高兴见到你！", "确实很糟糕，希望明天会更好。", "很高兴你喜欢这部电影！", "很抱歉听到这个消息，我们会改进。",
            "I'm happy to see you too!", "It's really terrible, hope it gets better tomorrow.", "Glad you liked this movie!", "Sorry to hear that, we'll improve.",
            "Je suis aussi heureux de vous voir !", "C'est vraiment terrible, j'espère que ça ira mieux demain.", "Heureux que vous ayez aimé ce film !", "Désolé d'entendre ça, nous améliorerons.",
            "Ich freue mich auch, dich zu sehen!", "Es ist wirklich schrecklich, hoffentlich wird es morgen besser.", "Freut mich, dass dir dieser Film gefallen hat!", "Es tut mir leid, das zu hören, wir werden uns verbessern.",
            "我理解你的感受，愤怒是一种自然的情绪。", "惊喜可以带来新的体验。", "伤心的时候记得照顾好自己。", "恐惧有时是保护我们的本能。",
            "I understand how you feel, anger is a natural emotion.", "Surprises can bring new experiences.", "Take care of yourself when you're sad.", "Fear is sometimes an instinct to protect us.",
            "Entiendo cómo te sientes, la ira es una emoción natural.", "Las sorpresas pueden traer nuevas experiencias.", "Cuídate cuando estés triste.", "El miedo a veces es un instinto que nos protege."
        ]
        
        # 扩展到指定数量的样本 | Extend to specified number of samples
        texts = []
        emotion_labels = []
        responses = []
        
        for i in range(num_samples):
            idx = i % len(mock_texts)
            texts.append(mock_texts[idx])
            emotion_labels.append(mock_emotion_labels[idx])
            responses.append(mock_responses[idx])
        
        return texts, emotion_labels, responses
    
    def _prepare_datasets(self, training_data=None, validation_data=None):
        """准备训练和验证数据集 | Prepare training and validation datasets"""
        # 如果没有提供数据，尝试从数据路径加载 | If no data provided, try to load from data paths
        if training_data is None:
            training_data = self._load_data_from_paths()
        
        # 如果数据已经是Dataset对象，直接返回 | If data is already Dataset objects, return directly
        if isinstance(training_data, Dataset) and isinstance(validation_data, Dataset):
            return training_data, validation_data
        
        # 提取文本、情感标签和语言模型标签 | Extract texts, emotion labels and language model labels
        texts, emotion_labels, lm_labels = self._extract_data_components(training_data)
        
        # 如果提供了验证数据，则使用它 | If validation data is provided, use it
        if validation_data:
            val_texts, val_emotion_labels, val_lm_labels = self._extract_data_components(validation_data)
        else:
            # 分割训练数据为训练集和验证集 | Split training data into train and validation sets
            train_texts, val_texts, train_emotion_labels, val_emotion_labels, train_lm_labels, val_lm_labels = train_test_split(
                texts, emotion_labels, lm_labels, test_size=self.config["eval_split"], random_state=42
            )
        
        # 创建Dataset对象 | Create Dataset objects
        train_dataset = LanguageDataset(
            train_texts, train_emotion_labels, train_lm_labels, 
            self.tokenizer, self.config["max_seq_length"]
        )
        val_dataset = LanguageDataset(
            val_texts, val_emotion_labels, val_lm_labels, 
            self.tokenizer, self.config["max_seq_length"]
        )
        
        # 如果启用了增量学习，合并记忆库中的样本 | If incremental learning is enabled, merge samples from memory bank
        if self.config["enable_incremental_learning"] and self.important_samples:
            # 从记忆库中提取样本 | Extract samples from memory bank
            memory_texts = [sample["text"] for sample in self.important_samples]
            memory_emotion_labels = [sample["predicted_label"] for sample in self.important_samples]
            # 对于记忆库样本，使用文本本身作为语言模型标签 | For memory bank samples, use text itself as LM labels
            memory_lm_labels = memory_texts.copy()
            
            # 创建记忆库数据集 | Create memory bank dataset
            memory_dataset = LanguageDataset(
                memory_texts, memory_emotion_labels, memory_lm_labels, 
                self.tokenizer, self.config["max_seq_length"]
            )
            
            # 合并数据集 | Merge datasets
            train_dataset = ConcatDataset([train_dataset, memory_dataset])
            logger.info(f"合并了 {len(memory_dataset)} 个记忆库样本到训练集 | Merged {len(memory_dataset)} memory bank samples into training set")
        
        return train_dataset, val_dataset
    
    def _extract_data_components(self, data):
        """从数据中提取文本、情感标签和语言模型标签 | Extract texts, emotion labels and language model labels from data"""
        texts = []
        emotion_labels = []
        lm_labels = []
        
        if isinstance(data, tuple) and len(data) == 3:
            # 如果数据是(texts, emotion_labels, lm_labels)的元组 | If data is a tuple of (texts, emotion_labels, lm_labels)
            texts, emotion_labels, lm_labels = data
        elif isinstance(data, list):
            # 如果数据是字典列表 | If data is a list of dictionaries
            for item in data:
                if isinstance(item, dict):
                    if 'text' in item and 'emotion' in item:
                        texts.append(item['text'])
                        # 转换情感文本标签为ID | Convert emotion text labels to IDs
                        if isinstance(item['emotion'], str):
                            emotion_labels.append(self.emotion_categories.get(item['emotion'], 0))
                        else:
                            emotion_labels.append(item['emotion'])
                        # 使用响应作为语言模型标签，或者默认使用原始文本 | Use response as LM labels, or default to original text
                        lm_labels.append(item.get('response', item['text']))
                elif isinstance(item, tuple) and len(item) >= 2:
                    # 如果是(text, emotion)元组 | If it's a (text, emotion) tuple
                    texts.append(item[0])
                    if isinstance(item[1], str):
                        emotion_labels.append(self.emotion_categories.get(item[1], 0))
                    else:
                        emotion_labels.append(item[1])
                    # 默认使用文本作为语言模型标签 | Default to text as LM labels
                    lm_labels.append(item[0])
        elif isinstance(data, dict):
            # 如果数据是包含多个字段的字典 | If data is a dictionary with multiple fields
            if 'texts' in data and 'emotion_labels' in data:
                texts = data['texts']
                # 转换情感文本标签为ID | Convert emotion text labels to IDs
                emotion_labels = [
                    self.emotion_categories.get(label, 0) if isinstance(label, str) else label
                    for label in data['emotion_labels']
                ]
                # 使用响应作为语言模型标签，或者默认使用原始文本 | Use responses as LM labels, or default to original text
                lm_labels = data.get('responses', texts)
        
        # 如果没有提取到数据，创建模拟数据 | If no data extracted, create mock data
        if not texts:
            logger.warning("没有找到有效训练数据，使用模拟数据 | No valid training data found, using mock data")
            texts, emotion_labels, lm_labels = self._create_mock_data()
        
        return texts, emotion_labels, lm_labels
    
    def _load_data_from_paths(self):
        """从数据路径加载数据 | Load data from paths"""
        all_data = []
        
        # 检查数据路径是否存在 | Check if data paths exist
        for data_path in self.config["data_paths"]:
            path = Path(data_path)
            if path.is_dir():
                # 遍历目录下的所有JSON和CSV文件 | Iterate over all JSON and CSV files in directory
                for file_path in path.glob("**/*"):
                    if file_path.suffix in ('.json', '.csv'):
                        all_data.extend(self._load_data_from_file(str(file_path)))
            elif path.is_file():
                # 加载单个文件 | Load single file
                all_data.extend(self._load_data_from_file(str(path)))
        
        return all_data
    
    def _load_data_from_file(self, file_path):
        """从单个文件加载数据 | Load data from single file"""
        data = []
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data = file_data
                    elif isinstance(file_data, dict) and 'data' in file_data:
                        data = file_data['data']
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                data = df.to_dict('records')
            
            logger.info(f"从 {file_path} 加载了 {len(data)} 条数据 | Loaded {len(data)} entries from {file_path}")
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {str(e)} | Failed to load file: {file_path}, error: {str(e)}")
        
        return data
    
    def _compute_metrics(self, logits, labels):
        """计算评估指标 | Compute evaluation metrics"""
        # 计算准确率 | Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        accuracy = correct.mean().item()
        
        # 计算精确率、召回率和F1分数 | Calculate precision, recall and F1 score
        unique_labels = torch.unique(labels)
        precision = 0.0
        recall = 0.0
        
        for label in unique_labels:
            tp = ((predictions == label) & (labels == label)).sum().item()
            fp = ((predictions == label) & (labels != label)).sum().item()
            fn = ((predictions != label) & (labels == label)).sum().item()
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precision += p
            recall += r
        
        # 计算宏平均指标 | Calculate macro average metrics
        precision /= len(unique_labels)
        recall /= len(unique_labels)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def _evaluate_model(self, val_dataset):
        """评估模型性能 | Evaluate model performance"""
        self.eval()
        
        total_loss = 0
        all_emotion_logits = []
        all_emotion_labels = []
        
        # 创建数据加载器 | Create data loader
        data_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=DataCollatorWithPadding(self.tokenizer)
        )
        
        criterion_lm = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        criterion_emotion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                emotion_labels = batch['emotion_labels']
                lm_labels = batch['lm_labels']
                
                # 前向传播 | Forward pass
                if self.is_enhanced:
                    lm_logits, emotion_logits, _ = self.forward(input_ids, attention_mask)
                else:
                    lm_logits, emotion_logits = self.forward(input_ids, attention_mask)
                
                # 计算损失 | Calculate loss
                lm_loss = criterion_lm(lm_logits.view(-1, self.tokenizer.vocab_size), 
                                      lm_labels.view(-1))
                emotion_loss = criterion_emotion(emotion_logits, emotion_labels)
                loss = lm_loss + emotion_loss
                
                total_loss += loss.item()
                
                # 收集用于指标计算的预测和标签 | Collect predictions and labels for metrics calculation
                all_emotion_logits.append(emotion_logits)
                all_emotion_labels.append(emotion_labels)
        
        # 计算平均损失 | Calculate average loss
        avg_loss = total_loss / len(data_loader)
        
        # 计算情感分类指标 | Calculate emotion classification metrics
        all_emotion_logits = torch.cat(all_emotion_logits, dim=0)
        all_emotion_labels = torch.cat(all_emotion_labels, dim=0)
        emotion_metrics = self._compute_metrics(all_emotion_logits, all_emotion_labels)
        
        # 合并所有指标 | Combine all metrics
        metrics = {
            "loss": avg_loss,
            "emotion_accuracy": emotion_metrics["accuracy"],
            "emotion_precision": emotion_metrics["precision"],
            "emotion_recall": emotion_metrics["recall"],
            "emotion_f1": emotion_metrics["f1"],
            "timestamp": datetime.now().isoformat()
        }
        
        # 记录评估结果 | Record evaluation results
        self.validation_history.append(metrics)
        self.last_evaluation = datetime.now()
        
        # 更新最佳指标 | Update best metrics
        if self.best_metrics is None or metrics["emotion_f1"] > self.best_metrics["emotion_f1"]:
            self.best_metrics = metrics
            logger.info(f"新的最佳性能: F1 = {metrics['emotion_f1']:.4f}, 损失 = {metrics['loss']:.4f} | New best performance: F1 = {metrics['emotion_f1']:.4f}, loss = {metrics['loss']:.4f}")
        
        # 切换回训练模式 | Switch back to training mode
        self.train()
        
        return metrics
    
    def _self_improvement(self):
        """自我改进机制 | Self-improvement mechanism"""
        # 基于历史性能进行自我优化 | Self-optimization based on historical performance
        if len(self.validation_history) < 2:
            return  # 历史数据不足 | Not enough historical data
        
        # 分析最近的性能变化 | Analyze recent performance changes
        recent_metrics = self.validation_history[-1]
        prev_metrics = self.validation_history[-2]
        
        # 如果性能下降，调整学习率 | Adjust learning rate if performance degrades
        if recent_metrics["emotion_f1"] < prev_metrics["emotion_f1"] * 0.95:
            for param_group in self.optimizer.param_groups:
                new_lr = param_group["lr"] * 0.5  # 学习率减半 | Halve the learning rate
                param_group["lr"] = new_lr
                logger.info(f"性能下降，学习率从 {param_group["lr"] * 2} 调整为 {new_lr} | Performance degraded, learning rate adjusted from {param_group["lr"] * 2} to {new_lr}")
        
        # 定期清理内存 | Periodic memory cleanup
        if self.current_epoch % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 如果性能稳定但没有提高，尝试调整权重衰减 | If performance stabilizes but doesn't improve, try adjusting weight decay
        elif abs(recent_metrics["emotion_f1"] - prev_metrics["emotion_f1"]) < 0.005:
            # 微调语言适配器权重 | Fine-tune language adapter weights
            if self.config["multilingual_support"]:
                # 简单实现：增加语言适配器的学习率 | Simple implementation: increase learning rate of language adapters
                pass
    
    def train_model(self, training_data=None, validation_data=None, epochs: int = None, lr: float = None):
        """
        统一训练接口
        (Unified training interface)
        
        参数 Parameters:
        training_data: 训练数据集 (Training dataset)
        validation_data: 验证数据集 (Validation dataset)
        epochs: 训练轮数 (Number of training epochs)
        lr: 学习率 (Learning rate)
        """
        # 使用配置中的默认值或传入的参数 | Use default values from config or passed parameters
        epochs = epochs or self.config["epochs"]
        lr = lr or self.config["learning_rate"]
        
        # 准备训练和验证数据集 | Prepare training and validation datasets
        train_dataset, val_dataset = self._prepare_datasets(training_data, validation_data)
        
        # 创建数据加载器 | Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=DataCollatorWithPadding(self.tokenizer)
        )
        
        # 初始化优化器和损失函数 | Initialize optimizer and loss functions
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        criterion_lm = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        criterion_emotion = nn.CrossEntropyLoss()
        
        # 学习率调度器 | Learning rate scheduler
        num_training_steps = epochs * len(train_loader)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        # 设置为训练模式 | Set to training mode
        self.train()
        self.current_epoch = 0
        
        # 记录最佳性能指标 | Track best performance metrics
        best_f1 = 0.0
        patience_counter = 0
        
        logger.info(f"开始训练模型 | Starting model training")
        logger.info(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)} | Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        try:
            for epoch in range(epochs):
                self.current_epoch = epoch + 1
                total_loss = 0
                total_emotion_loss = 0
                total_lm_loss = 0
                
                for batch in train_loader:
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    emotion_labels = batch['emotion_labels']
                    lm_labels = batch['lm_labels']
                    
                    self.optimizer.zero_grad()
                    
                    if self.is_enhanced:
                        lm_logits, emotion_logits, _ = self.forward(input_ids, attention_mask)
                    else:
                        lm_logits, emotion_logits = self.forward(input_ids, attention_mask)
                    
                    lm_loss = criterion_lm(lm_logits.view(-1, self.tokenizer.vocab_size), 
                                          lm_labels.view(-1))
                    emotion_loss = criterion_emotion(emotion_logits, emotion_labels)
                    loss = lm_loss + emotion_loss
                    
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    
                    total_loss += loss.item()
                    total_emotion_loss += emotion_loss.item()
                    total_lm_loss += lm_loss.item()
                
                # 计算平均损失 | Calculate average losses
                avg_loss = total_loss / len(train_loader)
                avg_emotion_loss = total_emotion_loss / len(train_loader)
                avg_lm_loss = total_lm_loss / len(train_loader)
                
                # 记录训练历史 | Record training history
                train_metrics = {
                    "epoch": self.current_epoch,
                    "loss": avg_loss,
                    "emotion_loss": avg_emotion_loss,
                    "lm_loss": avg_lm_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "timestamp": datetime.now().isoformat()
                }
                self.training_history.append(train_metrics)
                
                # 记录到TensorBoard | Log to TensorBoard
                if self.is_enhanced and hasattr(self, 'writer'):
                    self.writer.add_scalar('Loss/train', avg_loss, epoch)
                    self.writer.add_scalar('Loss/emotion', avg_emotion_loss, epoch)
                    self.writer.add_scalar('Loss/lm', avg_lm_loss, epoch)
                    self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]["lr"], epoch)
                
                logger.info(f"Epoch {self.current_epoch}/{epochs} | Loss: {avg_loss:.4f} | Emotion Loss: {avg_emotion_loss:.4f} | LM Loss: {avg_lm_loss:.4f}")
                
                # 每n个epoch进行一次评估 | Evaluate every n epochs
                if self.current_epoch % self.config["eval_interval"] == 0:
                    val_metrics = self._evaluate_model(val_dataset)
                    
                    # 记录到TensorBoard | Log to TensorBoard
                    if self.is_enhanced and hasattr(self, 'writer'):
                        self.writer.add_scalar('Loss/validation', val_metrics["loss"], epoch)
                        self.writer.add_scalar('Metrics/accuracy', val_metrics["emotion_accuracy"], epoch)
                        self.writer.add_scalar('Metrics/f1', val_metrics["emotion_f1"], epoch)
                    
                    logger.info(f"验证结果 | Validation Results - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['emotion_f1']:.4f}")
                    
                    # 早停机制 | Early stopping mechanism
                    if self.config.get("early_stopping_patience", 0) > 0:
                        if val_metrics["emotion_f1"] > best_f1:
                            best_f1 = val_metrics["emotion_f1"]
                            patience_counter = 0
                            # 保存最佳模型 | Save best model
                            if self.current_epoch % self.config["save_interval"] == 0:
                                self.save_model(f"model_epoch_{self.current_epoch}_best")
                        else:
                            patience_counter += 1
                            if patience_counter >= self.config["early_stopping_patience"]:
                                logger.info(f"早停触发，在第 {self.current_epoch} 轮停止训练 | Early stopping triggered, stopping training at epoch {self.current_epoch}")
                                break
                
                # 自我改进 | Self-improvement
                if self.is_enhanced:
                    self._self_improvement()
            
            # 保存最终模型 | Save final model
            self.save_model()
            
            logger.info("训练完成! | Training completed!")
            
            # 返回训练结果 | Return training results
            return {
                "status": "success",
                "epochs_completed": self.current_epoch,
                "best_metrics": self.best_metrics,
                "training_history": self.training_history,
                "validation_history": self.validation_history
            }
            
        except Exception as e:
            logger.error(f"训练失败: {str(e)} | Training failed: {str(e)}")
            self.training_history.append({
                "epoch": self.current_epoch,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return {
                "status": "error",
                "message": str(e),
                "epochs_completed": self.current_epoch
            }
    
    def save_model(self, save_path: str = "./models/unified_language_model"):
        """
        保存模型
        (Save model)
        
        参数 Parameters:
        save_path: 保存路径 (Save path)
        """
        try:
            # 确保目录存在 | Ensure directory exists
            os.makedirs(save_path, exist_ok=True)
            
            # 保存模型和分词器 | Save model and tokenizer
            torch.save(self.state_dict(), os.path.join(save_path, "model_state_dict.pth"))
            self.tokenizer.save_pretrained(save_path)
            
            # 保存配置 | Save configuration
            with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            # 保存训练历史 | Save training history
            with open(os.path.join(save_path, "training_history.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "training_history": self.training_history,
                    "validation_history": self.validation_history,
                    "best_metrics": self.best_metrics
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"模型成功保存到: {save_path} | Model successfully saved to: {save_path}")
            return True
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)} | Model saving failed: {str(e)}")
            return False
    
    def load_model(self, load_path: str = "./models/unified_language_model"):
        """
        加载模型
        (Load model)
        
        参数 Parameters:
        load_path: 加载路径 (Load path)
        """
        try:
            # 加载模型状态字典 | Load model state dict
            self.load_state_dict(torch.load(os.path.join(load_path, "model_state_dict.pth")))
            
            # 加载分词器 | Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            
            # 加载配置 | Load configuration
            config_path = os.path.join(load_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            
            # 加载训练历史 | Load training history
            history_path = os.path.join(load_path, "training_history.json")
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
                    self.training_history = history_data.get("training_history", [])
                    self.validation_history = history_data.get("validation_history", [])
                    self.best_metrics = history_data.get("best_metrics", None)
            
            logger.info(f"模型成功从 {load_path} 加载 | Model successfully loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)} | Model loading failed: {str(e)}")
            return False
    
    def incremental_train(self, new_data, epochs: int = 2, lr_scale: float = 0.5):
        """
        增量训练
        (Incremental training)
        
        参数 Parameters:
        new_data: 新的训练数据 (New training data)
        epochs: 训练轮数 (Number of training epochs)
        lr_scale: 学习率缩放因子 (Learning rate scale factor)
        """
        if not self.config["enable_incremental_learning"]:
            logger.warning("增量学习功能未启用 | Incremental learning feature is not enabled")
            return {
                "status": "error",
                "message": "增量学习功能未启用 | Incremental learning feature is not enabled"
            }
        
        logger.info(f"开始增量训练，新样本数: {len(new_data) if hasattr(new_data, '__len__') else 'unknown'} | Starting incremental training with {len(new_data) if hasattr(new_data, '__len__') else 'unknown'} new samples")
        
        # 减小学习率以避免过拟合 | Reduce learning rate to avoid overfitting
        incremental_lr = self.config["learning_rate"] * lr_scale
        
        # 执行增量训练 | Execute incremental training
        result = self.train_model(
            training_data=new_data,
            epochs=epochs,
            lr=incremental_lr
        )
        
        if result["status"] == "success":
            logger.info("增量训练完成 | Incremental training completed")
        
        return result
    
    def transfer_learn(self, target_language: str, training_data=None, epochs: int = 3, lr_scale: float = 0.3):
        """
        跨语言迁移学习
        (Cross-language transfer learning)
        
        参数 Parameters:
        target_language: 目标语言 (Target language)
        training_data: 训练数据 (Training data)
        epochs: 训练轮数 (Number of training epochs)
        lr_scale: 学习率缩放因子 (Learning rate scale factor)
        """
        if not self.config["enable_knowledge_transfer"]:
            logger.warning("知识迁移功能未启用 | Knowledge transfer feature is not enabled")
            return {
                "status": "error",
                "message": "知识迁移功能未启用 | Knowledge transfer feature is not enabled"
            }
        
        # 检查目标语言是否支持 | Check if target language is supported
        if target_language not in self.supported_languages:
            logger.warning(f"不支持的目标语言: {target_language} | Unsupported target language: {target_language}")
            return {
                "status": "error",
                "message": f"不支持的目标语言: {target_language} | Unsupported target language: {target_language}"
            }
        
        logger.info(f"开始向语言 '{target_language}' 的迁移学习 | Starting transfer learning to language '{target_language}'")
        
        # 如果没有提供训练数据，尝试加载目标语言特定数据 | If no training data provided, try to load target language specific data
        if training_data is None:
            training_data = self._load_language_specific_data(target_language)
        
        # 减小学习率进行迁移学习 | Reduce learning rate for transfer learning
        transfer_lr = self.config["learning_rate"] * lr_scale
        
        # 执行迁移学习 | Execute transfer learning
        result = self.train_model(
            training_data=training_data,
            epochs=epochs,
            lr=transfer_lr
        )
        
        if result["status"] == "success":
            logger.info(f"向语言 '{target_language}' 的迁移学习完成 | Transfer learning to language '{target_language}' completed")
        
        return result
    
    def _load_language_specific_data(self, language: str):
        """加载特定语言的数据 | Load language specific data"""
        # 尝试从数据路径加载特定语言的数据 | Try to load language specific data from data paths
        language_data = []
        
        for data_path in self.config["data_paths"]:
            path = Path(data_path)
            # 检查是否有语言特定的子目录 | Check if there's a language specific subdirectory
            lang_dir = path / language
            if lang_dir.is_dir():
                # 加载语言特定目录中的所有数据文件 | Load all data files in language specific directory
                for file_path in lang_dir.glob("**/*"):
                    if file_path.suffix in ('.json', '.csv'):
                        language_data.extend(self._load_data_from_file(str(file_path)))
        
        # 如果没有找到特定语言的数据，返回空列表 | Return empty list if no language specific data found
        return language_data

# 向后兼容的类名 | Backward compatible class names
MultilingualEmotionalLLM = UnifiedLanguageModel
EnhancedMultilingualEmotionalLLM = UnifiedLanguageModel