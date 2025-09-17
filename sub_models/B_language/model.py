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
大语言模型实现 (Large Language Model Implementation)
具有多语言交互和情感推理能力
(Multilingual interaction and emotional reasoning capabilities)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultilingualEmotionalLLM(nn.Module):
    """
    多语言情感大语言模型
    (Multilingual Emotional Large Language Model)
    """
    def __init__(self, model_name="xlm-roberta-base"):
        """
        初始化多语言情感模型
        (Initialize multilingual emotional model)
        
        参数 Parameters:
        model_name: 预训练模型名称 (Pretrained model name)
        """
        super().__init__()
        # 加载多语言基础模型 (Load multilingual base model)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 情感分析层 (Emotional analysis layer)
        self.emotion_head = nn.Linear(self.base_model.config.hidden_size, 7)  # 7种基本情绪
        
        # 语言输出层 (Language output layer)
        self.lm_head = nn.Linear(self.base_model.config.hidden_size, self.tokenizer.vocab_size)

    def forward(self, input_ids, attention_mask):
        """
        前向传播
        (Forward propagation)
        
        参数 Parameters:
        input_ids: 输入token ID (Input token IDs)
        attention_mask: 注意力掩码 (Attention mask)
        """
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 情感预测 (Emotion prediction)
        emotion_logits = self.emotion_head(sequence_output[:, 0, :])
        
        # 语言建模 (Language modeling)
        lm_logits = self.lm_head(sequence_output)
        
        return lm_logits, emotion_logits

    def train_model(self, dataset, epochs=3, lr=1e-5):
        """
        训练模型
        (Train the model)
        
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
                
                # 前向传播
                lm_logits, emotion_logits = self.forward(input_ids, attention_mask)
                
                # 计算损失
                lm_loss = criterion_lm(lm_logits.view(-1, self.tokenizer.vocab_size), 
                                      lm_labels.view(-1))
                emotion_loss = criterion_emotion(emotion_logits, emotion_labels)
                loss = lm_loss + emotion_loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataset):.4f}")
        
        print("Training completed!")
                
    def predict(self, text, language='en'):
        """
        生成预测（带情感推理）
        (Generate predictions with emotional reasoning)

        参数 Parameters:
        text: 输入文本 (Input text)
        language: 语言代码 (Language code)
        """
        # 记录请求开始时间
        start_time = time.time()
        
        try:
            # 加载对应语言资源
            self._load_language_resources(language)
            
            # 记录最后处理的提示长度
            self.last_prompt_length = len(text.split())

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                lm_logits, emotion_logits = self.forward(**inputs)

            # 情感推理 - 增强版
            emotion_probs = torch.softmax(emotion_logits, dim=-1)
            emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
            
            # 获取情感概率分布和主要情感
            emotion_distribution = {emotions[i]: emotion_probs[0][i].item() for i in range(len(emotions))}
            primary_emotion_id = torch.argmax(emotion_probs).item()
            primary_emotion = emotions[primary_emotion_id]
            
            # 生成响应（考虑情感和语言）
            generated_ids = torch.argmax(lm_logits, dim=-1)
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # 情感增强响应 - 根据不同语言和情感生成不同的响应风格
            if primary_emotion == "joy":
                response = self._enhance_with_emotion(response, "joy", language)
            elif primary_emotion == "sadness":
                response = self._enhance_with_emotion(response, "sadness", language)
            elif primary_emotion == "anger":
                response = self._enhance_with_emotion(response, "anger", language)
            elif primary_emotion == "fear":
                response = self._enhance_with_emotion(response, "fear", language)
            elif primary_emotion == "surprise":
                response = self._enhance_with_emotion(response, "surprise", language)
            elif primary_emotion == "disgust":
                response = self._enhance_with_emotion(response, "disgust", language)
            
            # 计算处理时间
            process_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 更新统计信息
            self._update_stats(success=True, process_time=process_time, language=language)
            
            return {
                "response": response,
                "primary_emotion": primary_emotion,
                "emotion_distribution": emotion_distribution,
                "confidence": float(emotion_probs[0][primary_emotion_id]),
                "language": language,
                "processing_time_ms": process_time
            }
            
        except Exception as e:
            # 计算处理时间
            process_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 更新统计信息 - 标记为失败
            self._update_stats(success=False, process_time=process_time, language=language)
            
            import logging
            logging.error(f"Prediction error: {str(e)}")
            
            return {
                "response": "I'm sorry, I couldn't process your request at the moment.",
                "primary_emotion": "neutral",
                "emotion_distribution": {e: 0 for e in ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]},
                "confidence": 0.0,
                "language": language,
                "processing_time_ms": process_time,
                "error": str(e)
            }
            
    def _enhance_with_emotion(self, text, emotion, language):
        """\根据情感和语言增强文本响应"""
        # 为不同语言和情感定义增强模式
        enhancements = {
            'joy': {
                'en': ["😊 ", "Great! ", "Wonderful! "],
                'zh': ["😊 ", "太棒了！", "太好了！"],
                'ja': ["😊 ", "すばらしい！", "よかった！"],
                'de': ["😊 ", "Fantastisch! ", "Super! "],
                'fr': ["😊 ", "Génial! ", "Super! "]
            },
            'sadness': {
                'en': ["😢 ", "I'm sorry to hear that. ", "That's unfortunate. "],
                'zh': ["😢 ", "听到这个我很遗憾。", "真不幸。"],
                'ja': ["😢 ", "それは残念です。", "大変ですね。"],
                'de': ["😢 ", "Das tut mir leid. ", "Schade. "],
                'fr': ["😢 ", "Je suis désolé. ", "C'est dommage. "]
            },
            'anger': {
                'en': ["😠 ", "That's frustrating. ", "I understand your frustration. "],
                'zh': ["😠 ", "这确实令人沮丧。", "我理解你的感受。"],
                'ja': ["😠 ", "イラッとしますね。", "お気持ちはわかります。"],
                'de': ["😠 ", "Das ist frustrierend. ", "Ich verstehe Ihre Frustration. "],
                'fr': ["😠 ", "C'est frustrant. ", "Je comprends votre frustration. "]
            },
            'fear': {
                'en': ["😨 ", "I understand your concern. ", "Let's address this carefully. "],
                'zh': ["😨 ", "我理解你的担忧。", "让我们谨慎处理。"],
                'ja': ["😨 ", "心配はわかります。", "慎重に対処しましょう。"],
                'de': ["😨 ", "Ich verstehe Ihre Sorge. ", "Lassen Sie uns das sorgfältig angehen. "],
                'fr': ["😨 ", "Je comprends votre inquiétude. ", "Traitons cela avec précaution. "]
            },
            'surprise': {
                'en': ["😲 ", "Wow! ", "That's surprising! "],
                'zh': ["😲 ", "哇！", "真令人惊讶！"],
                'ja': ["😲 ", "わぁ！", "びっくりしました！"],
                'de': ["😲 ", "Wow! ", "Das ist überraschend! "],
                'fr': ["😲 ", "Wow! ", "C'est surprenant! "]
            },
            'disgust': {
                'en': ["😒 ", "That's unpleasant. ", "That's not ideal. "],
                'zh': ["😒 ", "这令人不愉快。", "这不太理想。"],
                'ja': ["😒 ", "気持ち悪いですね。", "理想的ではないですね。"],
                'de': ["😒 ", "Das ist unangenehm. ", "Das ist nicht ideal. "],
                'fr': ["😒 ", "C'est désagréable. ", "Ce n'est pas idéal. "]
            }
        }
        
        # 获取适合当前情感和语言的增强前缀
        if emotion in enhancements and language in enhancements[emotion]:
            prefixes = enhancements[emotion][language]
            prefix = random.choice(prefixes) if prefixes else ""
            return f"{prefix}{text}"
        
        return text

    def get_status(self):
        """
        获取模型状态信息
        Get model status information
        
        返回 Returns:
        状态字典包含模型健康状态、内存使用、性能指标等
        Status dictionary containing model health, memory usage, performance metrics, etc.
        """
        import psutil
        import torch
        import time
        
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
        """
        获取输入统计信息
        Get input statistics
        
        返回 Returns:
        输入统计字典包含处理量、成功率等
        Input statistics dictionary containing processing volume, success rate, etc.
        """
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

    # 初始化模型时添加统计跟踪属性
    def __post_init__(self):
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

# 模型保存和加载函数 (Model save/load functions)
def save_model(model, path):
    """保存模型到文件 (Save model to file)"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': model.tokenizer,
        'emotion_labels': model.emotion_labels
    }, path)

def load_model(path, model_name="xlm-roberta-base"):
    """从文件加载模型 (Load model from file)"""
    model = MultilingualEmotionalLLM(model_name)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.tokenizer = checkpoint['tokenizer']
    model.emotion_labels = checkpoint['emotion_labels']
    return model

# 新增：语言资源管理
def _load_language_resources(self, lang_code):
    """加载指定语言资源"""
    # 这里会连接到主系统的语言资源管理器
    # 实际实现需要与manager_model/language_resources.py集成
    print(f"Switched to {lang_code} language resources")
