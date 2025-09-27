# Emotional Intelligence Engine for A Management Model
# Copyright 2025 The AGI Brain System Authors

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import re

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EmotionEngine")

class EmotionalState:
    """表示系统的情感状态"""
    def __init__(self):
        # 基本情感维度
        self.emotions = {
            'joy': 0.0,          # 快乐
            'trust': 0.0,        # 信任
            'fear': 0.0,         # 恐惧
            'surprise': 0.0,     # 惊讶
            'sadness': 0.0,      # 悲伤
            'disgust': 0.0,      # 厌恶
            'anger': 0.0,        # 愤怒
            'anticipation': 0.0  # 期待
        }
        
        # 情感倾向
        self.valence = 0.0      # 正负向 (-1.0 到 1.0)
        self.arousal = 0.0      # 唤醒度 (0.0 到 1.0)
        self.dominance = 0.0    # 支配感 (0.0 到 1.0)
        
        # 元情感信息
        self.confidence = 0.0   # 情感识别的置信度
        self.last_update = datetime.now()
        self.source = ""        # 情感来源
    
    def to_dict(self) -> Dict[str, Any]:
        """将情感状态转换为可序列化的字典"""
        return {
            'emotions': self.emotions,
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance,
            'confidence': self.confidence,
            'last_update': self.last_update.isoformat(),
            'source': self.source
        }

class EmotionEngine:
    """情感引擎 - 处理情感识别、表达和记忆"""
    def __init__(self):
        # 情感词典 - 用于基础情感分析
        self.emotion_lexicon = {
            'joy': ['happy', 'joy', 'excited', 'pleased', 'delight', 'glad', 'cheerful', 'joyful', 'thrilled', 'elated'],
            'trust': ['trust', 'believe', 'confident', 'rely', 'faith', 'confidence', 'depend', 'certain', 'sure'],
            'fear': ['fear', 'afraid', 'scared', 'worried', 'anxious', 'terrified', 'nervous', 'concerned', 'frightened'],
            'surprise': ['surprise', 'amazed', 'shocked', 'astonished', 'unexpected', 'startled', 'astounded'],
            'sadness': ['sad', 'sorrow', 'grief', 'unhappy', 'depressed', 'down', 'mourn', 'melancholy', 'blue'],
            'disgust': ['disgust', 'hate', 'loathe', 'repulsive', 'repugnant', 'detest', 'abhor', 'revulsion'],
            'anger': ['anger', 'angry', 'mad', 'furious', 'irritated', 'rage', 'annoyed', 'exasperated', 'frustrated'],
            'anticipation': ['anticipate', 'expect', 'look forward', 'await', 'predict', 'foresee', 'anticipation']
        }
        
        # 增强的情感识别模式
        self.emotion_patterns = {
            'joy': [
                r'good\s+news',
                r'excited\s+about',
                r'happy\s+to\s+hear',
                r'thank\s+you',
                r'appreciate'
            ],
            'sadness': [
                r'bad\s+news',
                r'sorry\s+to\s+hear',
                r'disappointed\s+about',
                r'not\s+working'
            ],
            'anger': [
                r'not\s+working',
                r'error\s+again',
                r'frustrated\s+with',
                r'this\s+is\s+terrible'
            ]
        }
        
        # 当前情感状态
        self.current_emotion = EmotionalState()
        
        # 情感记忆 - 存储最近的情感历史
        self.emotion_memory = deque(maxlen=100)
        
        # 长期情感趋势分析
        self.emotion_trends = {
            'daily': {},  # 按天存储的情感摘要
            'weekly': {},  # 按周存储的情感摘要
            'monthly': {}  # 按月存储的情感摘要
        }
        
        # 情感衰减率 (每分钟)
        self.emotion_decay_rate = 0.05
        
        # 初始化系统
        self._initialize_engine()
    
    def _initialize_engine(self):
        """初始化情感引擎"""
        logger.info("Emotion Engine initialized successfully")
        # 可以在这里加载预训练的情感模型或配置
    
    def analyze_text_emotion(self, text: str) -> EmotionalState:
        """分析文本中的情感"""
        # 创建新的情感状态
        new_emotion = EmotionalState()
        new_emotion.source = f"text: {text[:30]}..."
        
        # 将文本转换为小写进行分析
        text_lower = text.lower()
        
        # 使用情感词典分析基础情感
        emotion_counts = {emotion: 0 for emotion in self.emotions}
        
        # 计算每个情感的关键词匹配次数
        for emotion, keywords in self.emotion_lexicon.items():
            for keyword in keywords:
                # 确保关键词是完整的词，而不是词的一部分
                pattern = r'\\b' + re.escape(keyword) + r'\\b'
                matches = re.findall(pattern, text_lower)
                emotion_counts[emotion] += len(matches)
        
        # 使用模式匹配增强情感识别
        for emotion, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    emotion_counts[emotion] += 1
        
        # 计算情感强度
        total_matches = sum(emotion_counts.values())
        if total_matches > 0:
            for emotion, count in emotion_counts.items():
                new_emotion.emotions[emotion] = count / total_matches
            
            # 计算情感倾向
            # 简化计算：正向情感 - 负向情感
            positive_emotions = new_emotion.emotions['joy'] + new_emotion.emotions['trust'] + new_emotion.emotions['anticipation']
            negative_emotions = new_emotion.emotions['fear'] + new_emotion.emotions['sadness'] + new_emotion.emotions['disgust'] + new_emotion.emotions['anger']
            
            new_emotion.valence = (positive_emotions - negative_emotions)
            new_emotion.arousal = (new_emotion.emotions['surprise'] + new_emotion.emotions['fear'] + new_emotion.emotions['anger'] + new_emotion.emotions['joy'])
            new_emotion.dominance = (new_emotion.emotions['trust'] + new_emotion.emotions['anticipation'] + new_emotion.emotions['anger'] - new_emotion.emotions['fear'])
            
            # 归一化情感倾向值
            new_emotion.valence = max(-1.0, min(1.0, new_emotion.valence))
            new_emotion.arousal = max(0.0, min(1.0, new_emotion.arousal))
            new_emotion.dominance = max(0.0, min(1.0, new_emotion.dominance))
            
            # 设置置信度
            new_emotion.confidence = min(1.0, total_matches / 10.0)  # 简单的置信度计算
        
        # 更新当前情感状态
        self._update_emotional_state(new_emotion)
        
        return new_emotion
    
    def _update_emotional_state(self, new_emotion: EmotionalState):
        """更新系统的情感状态"""
        # 计算时间衰减
        time_diff = (datetime.now() - self.current_emotion.last_update).total_seconds() / 60.0  # 转换为分钟
        decay_factor = max(0.0, 1.0 - self.emotion_decay_rate * time_diff)
        
        # 应用衰减到当前情感状态
        for emotion in self.current_emotion.emotions:
            self.current_emotion.emotions[emotion] *= decay_factor
        
        # 混合新的情感状态
        alpha = 0.3  # 新情感的权重
        for emotion in self.current_emotion.emotions:
            self.current_emotion.emotions[emotion] = \
                (1 - alpha) * self.current_emotion.emotions[emotion] + \
                alpha * new_emotion.emotions[emotion]
        
        # 更新情感倾向
        self.current_emotion.valence = \
            (1 - alpha) * self.current_emotion.valence + alpha * new_emotion.valence
        self.current_emotion.arousal = \
            (1 - alpha) * self.current_emotion.arousal + alpha * new_emotion.arousal
        self.current_emotion.dominance = \
            (1 - alpha) * self.current_emotion.dominance + alpha * new_emotion.dominance
        
        # 更新元信息
        self.current_emotion.confidence = \
            (1 - alpha) * self.current_emotion.confidence + alpha * new_emotion.confidence
        self.current_emotion.last_update = datetime.now()
        self.current_emotion.source = new_emotion.source
        
        # 记录到情感记忆
        self._record_emotion_to_memory(self.current_emotion)
    
    def _record_emotion_to_memory(self, emotion: EmotionalState):
        """记录情感到内存"""
        # 存储可序列化的情感状态
        emotion_dict = emotion.to_dict()
        self.emotion_memory.append(emotion_dict)
        
        # 更新长期情感趋势
        self._update_emotion_trends(emotion_dict)
    
    def _update_emotion_trends(self, emotion_dict: Dict[str, Any]):
        """更新长期情感趋势"""
        now = datetime.now()
        
        # 按天更新
        day_key = now.strftime('%Y-%m-%d')
        if day_key not in self.emotion_trends['daily']:
            self.emotion_trends['daily'][day_key] = []
        self.emotion_trends['daily'][day_key].append(emotion_dict)
        
        # 按周更新 (ISO 周)
        week_key = now.strftime('%Y-W%V')
        if week_key not in self.emotion_trends['weekly']:
            self.emotion_trends['weekly'][week_key] = []
        self.emotion_trends['weekly'][week_key].append(emotion_dict)
        
        # 按月更新
        month_key = now.strftime('%Y-%m')
        if month_key not in self.emotion_trends['monthly']:
            self.emotion_trends['monthly'][month_key] = []
        self.emotion_trends['monthly'][month_key].append(emotion_dict)
    
    def generate_emotional_response(self, context: str, user_emotion: Optional[EmotionalState] = None) -> Dict[str, Any]:
        """基于当前情感状态和用户情感生成情感化响应"""
        # 如果没有提供用户情感，使用当前系统情感
        if user_emotion is None:
            user_emotion = self.current_emotion
        
        # 确定主导情感
        dominant_emotion = max(user_emotion.emotions.items(), key=lambda x: x[1])[0]
        
        # 根据主导情感和上下文生成响应
        response_templates = {
            'joy': [
                "I'm glad to hear that! ",
                "That's wonderful news! ",
                "I'm happy you're satisfied. "
            ],
            'trust': [
                "I appreciate your confidence. ",
                "You can count on me. ",
                "I'll do my best to help. "
            ],
            'fear': [
                "Don't worry, I'm here to help. ",
                "Let's try to resolve this together. ",
                "I understand your concern. "
            ],
            'surprise': [
                "Interesting! I didn't expect that. ",
                "That's quite surprising! ",
                "Wow, that's unexpected. "
            ],
            'sadness': [
                "I'm sorry to hear that. ",
                "That must be difficult. ",
                "I understand how you feel. "
            ],
            'disgust': [
                "I understand your frustration. ",
                "That's certainly unpleasant. ",
                "I apologize for the inconvenience. "
            ],
            'anger': [
                "I'm sorry you're feeling this way. ",
                "Let me try to fix this issue. ",
                "I understand your frustration. "
            ],
            'anticipation': [
                "Let's see what we can do. ",
                "I'm excited to work on this. ",
                "Let's get started! "
            ]
        }
        
        # 根据情感强度选择响应模板
        templates = response_templates.get(dominant_emotion, [""])
        response_prefix = templates[np.random.randint(0, len(templates))] if templates else ""
        
        # 根据情感倾向调整响应风格
        response_style = "neutral"
        if user_emotion.valence > 0.5:
            response_style = "positive"
        elif user_emotion.valence < -0.5:
            response_style = "empathetic"
        elif user_emotion.arousal > 0.7:
            response_style = "calm"
        
        # 生成完整的响应
        response = {
            'prefix': response_prefix,
            'style': response_style,
            'emotion': dominant_emotion,
            'intensity': user_emotion.emotions[dominant_emotion],
            'valence': user_emotion.valence,
            'arousal': user_emotion.arousal,
            'dominance': user_emotion.dominance,
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def get_current_emotion(self) -> Dict[str, Any]:
        """获取当前情感状态"""
        return self.current_emotion.to_dict()
    
    def get_emotion_summary(self, time_period: str = 'daily') -> Dict[str, Any]:
        """获取情感摘要统计"""
        if time_period not in self.emotion_trends:
            time_period = 'daily'
        
        trend_data = self.emotion_trends[time_period]
        
        # 计算基本统计信息
        stats = {
            'total_records': sum(len(records) for records in trend_data.values()),
            'periods': list(trend_data.keys()),
            'average_emotions': {},
            'most_common_emotions': {},
            'timeline': []
        }
        
        # 计算平均情感
        all_emotions = []
        for period, records in trend_data.items():
            period_emotions = {
                'period': period,
                'emotions': {}
            }
            
            for emotion_type in self.emotions:
                values = [r['emotions'][emotion_type] for r in records]
                avg_value = sum(values) / len(values) if values else 0.0
                period_emotions['emotions'][emotion_type] = avg_value
                all_emotions.append((emotion_type, avg_value))
            
            stats['timeline'].append(period_emotions)
        
        # 计算整体平均情感
        emotion_totals = {e: 0.0 for e in self.emotions}
        count = 0
        for emotion_type, value in all_emotions:
            emotion_totals[emotion_type] += value
            count += 1
        
        if count > 0:
            stats['average_emotions'] = {e: v / count for e, v in emotion_totals.items()}
        
        # 找出最常见的情感
        if all_emotions:
            from collections import Counter
            emotion_counter = Counter([e for e, _ in all_emotions])
            stats['most_common_emotions'] = dict(emotion_counter.most_common(3))
        
        return stats
    
    def reset_emotion(self):
        """重置情感状态"""
        self.current_emotion = EmotionalState()
        logger.info("Emotional state reset")
    
    @property
    def emotions(self) -> List[str]:
        """获取所有支持的情感类型"""
        return list(self.emotion_lexicon.keys())

# 创建全局情感引擎实例
global_emotion_engine = None

def get_emotion_engine() -> EmotionEngine:
    """获取情感引擎单例"""
    global global_emotion_engine
    if global_emotion_engine is None:
        global_emotion_engine = EmotionEngine()
    return global_emotion_engine

# 测试情感引擎
if __name__ == "__main__":
    engine = EmotionEngine()
    
    # 测试文本情感分析
    test_texts = [
        "I'm very happy with your performance!",
        "This system is terrible and not working properly.",
        "I'm concerned about the accuracy of the results.",
        "I'm excited to see what new features you'll add next!"
    ]
    
    for text in test_texts:
        emotion = engine.analyze_text_emotion(text)
        response = engine.generate_emotional_response(text, emotion)
        print(f"\nText: {text}")
        print(f"Emotion: {max(emotion.emotions.items(), key=lambda x: x[1])[0]}")
        print(f"Response prefix: {response['prefix']}")
        print(f"Response style: {response['style']}")