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
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import logging
from datetime import datetime
import requests
import psutil
import gc
from pathlib import Path

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
        self.supported_languages = ["zh", "en", "de", "ja", "ru"]
        
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
            }
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
            
        except Exception as e:
            logger.error(f"模型加载失败: {e} | Model loading failed: {e}")
            raise
    
    def _init_enhanced_features(self):
        """初始化增强功能 | Initialize enhanced features"""
        # 情感强度回归器 | Emotion intensity regressor
        self.intensity_regressor = nn.Linear(
            self.base_model.config.hidden_size, 1
        )
        
        # 多语言适配器 | Multilingual adapter
        self.language_adapter = nn.ModuleDict({
            lang: nn.Linear(self.base_model.config.hidden_size, 128)
            for lang in self.supported_languages
        })
        
        self.self_learning_data = []
        self.emotion_intensity_cache = {}
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        (Forward propagation)
        """
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
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
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            if self.is_enhanced:
                lm_logits, emotion_logits, intensity = self.forward(**inputs)
            else:
                lm_logits, emotion_logits = self.forward(**inputs)
        
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
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """获取模型状态信息 | Get model status information"""
        import psutil
        
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
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息 | Get model information"""
        return self.get_status()
    
    def train_model(self, dataset, epochs: int = 3, lr: float = 1e-5):
        """
        统一训练接口
        (Unified training interface)
        
        参数 Parameters:
        dataset: 训练数据集 (Training dataset)
        epochs: 训练轮数 (Number of training epochs)
        lr: 学习率 (Learning rate)
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion_lm = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        criterion_emotion = nn.CrossEntropyLoss()
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataset:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                emotion_labels = batch['emotion_labels']
                lm_labels = batch['lm_labels']
                
                optimizer.zero_grad()
                
                if self.is_enhanced:
                    lm_logits, emotion_logits, _ = self.forward(input_ids, attention_mask)
                else:
                    lm_logits, emotion_logits = self.forward(input_ids, attention_mask)
                
                lm_loss = criterion_lm(lm_logits.view(-1, self.tokenizer.vocab_size), 
                                      lm_labels.view(-1))
                emotion_loss = criterion_emotion(emotion_logits, emotion_labels)
                loss = lm_loss + emotion_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataset):.4f}")
            self.training_history.append({
                "epoch": epoch + 1,
                "loss": total_loss / len(dataset),
                "timestamp": datetime.now().isoformat()
            })
        
        logger.info("Training completed!")

# 向后兼容的类名 | Backward compatible class names
MultilingualEmotionalLLM = UnifiedLanguageModel
EnhancedMultilingualEmotionalLLM = UnifiedLanguageModel