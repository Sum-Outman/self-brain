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
import torch.nn.functional as F
import math

class LocalTokenizer:
    """本地字符级分词器"""
    def __init__(self):
        self.vocab = {}
        self.id_to_char = {}
        self.vocab_size = 0
        self.pad_token_id = 0
        self.build_vocab()
    
    def build_vocab(self):
        """构建字符词汇表"""
        # 基本字符集
        chars = ['<PAD>', '<UNK>']
        # 添加ASCII可打印字符
        for i in range(32, 127):
            chars.append(chr(i))
        # 添加常见中文字符
        common_chinese = '的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严'
        chars.extend(list(common_chinese))
        
        # 构建词汇表
        for idx, char in enumerate(chars):
            self.vocab[char] = idx
            self.id_to_char[idx] = char
        
        self.vocab_size = len(self.vocab)
    
    def encode(self, text, max_length=512):
        """编码文本为token ID"""
        tokens = []
        for char in text[:max_length]:
            tokens.append(self.vocab.get(char, self.vocab['<UNK>']))
        
        # 填充到最大长度
        if len(tokens) < max_length:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, token_ids):
        """解码token ID为文本"""
        text = ''
        for token_id in token_ids:
            if token_id in self.id_to_char and token_id != self.pad_token_id:
                text += self.id_to_char[token_id]
        return text
    
    def __len__(self):
        return self.vocab_size

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LocalMultilingualLLM(nn.Module):
    """
    本地多语言情感大语言模型
    (Local Multilingual Emotional Large Language Model)
    """
    def __init__(self, vocab_size=3000, d_model=256, nhead=8, num_layers=6, max_length=512):
        """
        初始化本地多语言情感模型
        (Initialize local multilingual emotional model)
        
        参数 Parameters:
        vocab_size: 词汇表大小 (Vocabulary size)
        d_model: 模型维度 (Model dimension)
        nhead: 注意力头数 (Number of attention heads)
        num_layers: Transformer层数 (Number of transformer layers)
        max_length: 最大序列长度 (Maximum sequence length)
        """
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 情感分析层 (7种基本情绪)
        self.emotion_head = nn.Linear(d_model, 7)
        
        # 语言输出层
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # 本地分词器
        self.tokenizer = LocalTokenizer()
        
        # 模型配置
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'max_length': max_length
        }
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        (Forward propagation)
        
        参数 Parameters:
        input_ids: 输入token ID (Input token IDs)
        attention_mask: 注意力掩码 (Attention mask)
        """
        # 词嵌入
        x = self.embedding(input_ids)
        
        # 位置编码
        x = self.pos_encoding(x.transpose(0, 1))
        
        # Transformer编码
        if attention_mask is None:
            # 创建默认的注意力掩码
            seq_len = input_ids.size(1)
            attention_mask = torch.ones(seq_len, seq_len, device=input_ids.device)
        
        sequence_output = self.transformer_encoder(x, attention_mask)
        sequence_output = sequence_output.transpose(0, 1)
        
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
