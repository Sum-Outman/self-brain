# 高级情感引擎 | Advanced Emotion Engine
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

import numpy as np
import torch
import torch.nn as nn
from transformers import pipeline
import logging
from datetime import datetime
from collections import deque
import json
import re
from typing import Dict, List, Any, Optional

class AdvancedEmotionEngine:
    """高级情感引擎 - 实现复杂的情感分析和表达 | Advanced Emotion Engine - Implements complex emotion analysis and expression"""
    
    def __init__(self, model_registry, language='zh'):
        self.model_registry = model_registry
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        # 8维情感模型: 喜悦、悲伤、愤怒、恐惧、信任、惊讶、期待、厌恶
        # 8-dimensional emotion model: joy, sadness, anger, fear, trust, surprise, anticipation, disgust
        self.emotional_state = {
            "joy": 0.5, "sadness": 0.1, "anger": 0.1, "fear": 0.1,
            "trust": 0.6, "surprise": 0.2, "anticipation": 0.4, "disgust": 0.1,
            "emotional_memory": [],  # 情感记忆 | Emotional memory
            "emotional_patterns": {},  # 情感模式识别 | Emotion pattern recognition
            "adaptive_responses": {},  # 自适应情感响应 | Adaptive emotional responses
            "current_primary_emotion": "neutral",
            "emotional_intensity": 0.5,
            "emotional_confidence": 0.8,
            "valence": 0.5,  # 愉悦度 (-1 to 1)
            "arousal": 0.5,  # 激活度 (0 to 1)
            "dominance": 0.5,  # 支配度 (0 to 1)
            "emotional_coherence": 0.7  # 情感一致性 | Emotional coherence
        }
        
        # 情感历史记录 | Emotion history records
        self.emotion_history = deque(maxlen=1000)
        
        # 情感模式数据库 | Emotion pattern database
        self.emotion_patterns_db = {}
        
        # 多语言情感标签 | Multilingual emotion labels
        self.emotion_labels = self._load_emotion_labels()
        
        # 情感表达模板 | Emotion expression templates
        self.expression_templates = self._load_expression_templates()
        
        # 自适应响应规则 | Adaptive response rules
        self.adaptive_rules = self._load_adaptive_rules()
        
        # 初始化情感分析模型 | Initialize emotion analysis models
        self._initialize_emotion_models()
        
        self.logger.info("高级情感引擎初始化完成 | Advanced emotion engine initialized")

    def _load_emotion_labels(self):
        """加载多语言情感标签 | Load multilingual emotion labels"""
        return {
            'zh': {
                'anger': '愤怒', 'disgust': '厌恶', 'fear': '恐惧', 
                'joy': '快乐', 'neutral': '中性', 'sadness': '悲伤', 
                'surprise': '惊讶', 'curious': '好奇', 'confident': '自信',
                'confused': '困惑', 'excited': '兴奋', 'frustrated': '沮丧',
                'grateful': '感激', 'hopeful': '希望', 'proud': '自豪',
                'relaxed': '放松', 'worried': '担忧'
            },
            'en': {
                'anger': 'anger', 'disgust': 'disgust', 'fear': 'fear',
                'joy': 'joy', 'neutral': 'neutral', 'sadness': 'sadness',
                'surprise': 'surprise', 'curious': 'curious', 'confident': 'confident',
                'confused': 'confused', 'excited': 'excited', 'frustrated': 'frustrated',
                'grateful': 'grateful', 'hopeful': 'hopeful', 'proud': 'proud',
                'relaxed': 'relaxed', 'worried': 'worried'
            }
        }

    def _load_expression_templates(self):
        """加载情感表达模板 | Load emotion expression templates"""
        return {
            'zh': {
                'anger': ["我很生气因为{}", "这让我感到愤怒：{}", "我对{}感到非常不满"],
                'joy': ["我很开心因为{}", "这让我感到快乐：{}", "我对{}感到非常高兴"],
                'sadness': ["我感到悲伤因为{}", "这让我难过：{}", "我对{}感到伤心"],
                'surprise': ["我很惊讶于{}", "这让我吃惊：{}", "{}让我感到意外"],
                'neutral': ["我注意到{}", "关于{}", "对于{}"],
                'curious': ["我对{}很好奇", "我想了解更多关于{}", "{}激起了我的兴趣"],
                'confident': ["我确信{}", "我对{}有信心", "我相信{}"],
                'fear': ["我害怕{}", "这让我恐惧：{}", "{}让我感到不安"],
                'trust': ["我相信{}", "我对{}有信心", "{}让我感到安心"],
                'anticipation': ["我期待{}", "我对{}充满期待", "{}让我兴奋不已"],
                'disgust': ["我对{}感到厌恶", "这让我恶心：{}", "{}让我反感"]
            },
            'en': {
                'anger': ["I'm angry because {}", "This makes me furious: {}", "I'm very dissatisfied with {}"],
                'joy': ["I'm happy because {}", "This brings me joy: {}", "I'm delighted about {}"],
                'sadness': ["I feel sad because {}", "This makes me sad: {}", "I'm heartbroken about {}"],
                'surprise': ["I'm surprised by {}", "This astonishes me: {}", "{} took me by surprise"],
                'neutral': ["I notice {}", "Regarding {}", "About {}"],
                'curious': ["I'm curious about {}", "I want to learn more about {}", "{} piques my interest"],
                'confident': ["I'm confident that {}", "I have faith in {}", "I believe in {}"],
                'fear': ["I'm afraid of {}", "This frightens me: {}", "{} makes me uneasy"],
                'trust': ["I trust {}", "I have confidence in {}", "{} makes me feel secure"],
                'anticipation': ["I anticipate {}", "I'm looking forward to {}", "{} excites me"],
                'disgust': ["I'm disgusted by {}", "This revolts me: {}", "{} repulses me"]
            }
        }

    def _load_adaptive_rules(self):
        """加载自适应响应规则 | Load adaptive response rules"""
        return {
            'zh': {
                'high_arousal': {
                    'joy': "采用更热情和兴奋的表达方式",
                    'anger': "控制情绪强度，避免过度反应",
                    'fear': "提供安抚和 reassurance",
                    'surprise': "表达惊讶但保持理性"
                },
                'low_arousal': {
                    'sadness': "给予更多共情和支持",
                    'neutral': "保持平静和理性的语调",
                    'disgust': "温和表达不满"
                },
                'high_dominance': {
                    'anger': " assertive but not aggressive",
                    'confidence': "强化自信表达",
                    'trust': "强调可靠性和确定性"
                },
                'low_dominance': {
                    'fear': "提供更多指导和支持",
                    'sadness': "表达理解和同情",
                    'surprise': "表现出好奇而非震惊"
                }
            },
            'en': {
                'high_arousal': {
                    'joy': "Use more enthusiastic and excited expressions",
                    'anger': "Control emotion intensity, avoid overreaction",
                    'fear': "Provide comfort and reassurance",
                    'surprise': "Express surprise but maintain rationality"
                },
                'low_arousal': {
                    'sadness': "Offer more empathy and support",
                    'neutral': "Maintain calm and rational tone",
                    'disgust': "Express dissatisfaction gently"
                },
                'high_dominance': {
                    'anger': "Assertive but not aggressive",
                    'confidence': "Strengthen confident expressions",
                    'trust': "Emphasize reliability and certainty"
                },
                'low_dominance': {
                    'fear': "Provide more guidance and support",
                    'sadness': "Express understanding and sympathy",
                    'surprise': "Show curiosity rather than shock"
                }
            }
        }

    def _initialize_emotion_models(self):
        """初始化情感分析模型 | Initialize emotion analysis models"""
        try:
            # 使用transformers的情感分析管道
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
            self.logger.info("情感分析模型加载成功")
        except Exception as e:
            self.logger.error(f"情感分析模型加载失败: {e}")
            self.emotion_analyzer = None

    def analyze_emotion(self, text, context=None):
        """分析文本情感 | Analyze text emotion"""
        try:
            if self.emotion_analyzer:
                # 使用模型分析情感
                results = self.emotion_analyzer(text)
                emotion_data = self._process_emotion_results(results, context)
                
                # 更新情感状态
                self._update_emotional_state(emotion_data)
                
                # 记录情感历史
                self.emotion_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'text': text,
                    'emotion': emotion_data,
                    'context': context
                })
                
                return emotion_data
            else:
                # 回退到简单的情感分析
                return self._fallback_emotion_analysis(text)
                
        except Exception as e:
            self.logger.error(f"情感分析错误: {e}")
            return self._get_neutral_emotion()

    def _process_emotion_results(self, results, context):
        """处理情感分析结果 | Process emotion analysis results"""
        # 提取主要情感
        primary_emotion = results[0][0]
        emotion_label = primary_emotion['label'].lower()
        confidence = primary_emotion['score']
        
        # 考虑上下文情感
        context_influence = self._apply_context_influence(context, emotion_label)
        
        # 计算情感强度
        intensity = self._calculate_emotion_intensity(confidence, context_influence)
        
        # 获取本地化情感标签
        localized_label = self.emotion_labels.get(self.language, {}).get(
            emotion_label, emotion_label
        )
        
        return {
            'type': emotion_label,
            'localized_type': localized_label,
            'confidence': confidence,
            'intensity': intensity,
            'secondary_emotions': [
                {
                    'type': result['label'].lower(),
                    'score': result['score'],
                    'localized_type': self.emotion_labels.get(self.language, {}).get(
                        result['label'].lower(), result['label'].lower()
                    )
                }
                for result in results[0][1:3]  # 取前3个次要情感
            ],
            'context_influence': context_influence,
            'timestamp': datetime.now().isoformat()
        }

    def _apply_context_influence(self, context, current_emotion):
        """应用上下文情感影响 | Apply context emotion influence"""
        if not context or not self.emotion_history:
            return 0.0
            
        # 分析最近的情感趋势
        recent_emotions = list(self.emotion_history)[-5:]  # 最近5个情感记录
        if not recent_emotions:
            return 0.0
            
        # 计算情感一致性
        same_emotion_count = sum(
            1 for e in recent_emotions 
            if e['emotion']['type'] == current_emotion
        )
        
        consistency = same_emotion_count / len(recent_emotions)
        return min(1.0, consistency * 0.3)  # 上下文影响最大30%

    def _calculate_emotion_intensity(self, confidence, context_influence):
        """计算情感强度 | Calculate emotion intensity"""
        # 基础强度基于置信度
        base_intensity = confidence
        
        # 应用上下文影响
        intensity = base_intensity * (1 + context_influence)
        
        # 确保在0-1范围内
        return min(1.0, max(0.1, intensity))

    def _fallback_emotion_analysis(self, text):
        """回退情感分析 | Fallback emotion analysis"""
        # 简单的基于关键词的情感分析
        text_lower = text.lower()
        
        emotion_keywords = {
            'joy': ['happy', 'joy', '高兴', '开心', '快乐', 'excited', '兴奋'],
            'sadness': ['sad', 'sadness', '悲伤', '难过', '伤心', 'depressed', '沮丧'],
            'anger': ['angry', 'anger', '愤怒', '生气', '恼火', 'furious', '愤怒'],
            'surprise': ['surprise', 'surprised', '惊讶', '吃惊', '意外', 'amazed', '震惊'],
            'fear': ['fear', 'afraid', '害怕', '恐惧', '担心', 'scared', '恐惧'],
            'neutral': ['ok', 'fine', '正常', '一般', 'ordinary', 'usual', '平常']
        }
        
        detected_emotion = 'neutral'
        confidence = 0.5
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_emotion = emotion
                    confidence = 0.7
                    break
        
        localized_label = self.emotion_labels.get(self.language, {}).get(
            detected_emotion, detected_emotion
        )
        
        return {
            'type': detected_emotion,
            'localized_type': localized_label,
            'confidence': confidence,
            'intensity': confidence,
            'secondary_emotions': [],
            'context_influence': 0.0,
            'timestamp': datetime.now().isoformat()
        }

    def _get_neutral_emotion(self):
        """获取中性情感 | Get neutral emotion"""
        localized_label = self.emotion_labels.get(self.language, {}).get('neutral', 'neutral')
        return {
            'type': 'neutral',
            'localized_type': localized_label,
            'confidence': 0.5,
            'intensity': 0.5,
            'secondary_emotions': [],
            'context_influence': 0.0,
            'timestamp': datetime.now().isoformat()
        }

    def generate_emotional_response(self, content, target_emotion=None):
        """生成带有情感的表达 | Generate emotional expression"""
        if target_emotion:
            emotion_type = target_emotion
            intensity = 0.7
        else:
            emotion_type = self.emotional_state['current_emotion']
            intensity = self.emotional_state['intensity']
        
        # 获取情感模板
        templates = self.expression_templates.get(self.language, {})
        emotion_templates = templates.get(emotion_type, templates.get('neutral', []))
        
        if not emotion_templates:
            return content
        
        # 选择模板并填充内容
        template = np.random.choice(emotion_templates)
        response = template.format(content)
        
        # 添加情感强度修饰
        response = self._add_emotion_intensity(response, intensity)
        
        return response

    def _add_emotion_intensity(self, text, intensity):
        """添加情感强度修饰 | Add emotion intensity modifiers"""
        intensity_modifiers = {
            'zh': {
                'high': ['非常', '极其', '特别', '十分'],
                'medium': ['很', '相当', '比较'],
                'low': ['有点', '稍微', '略微']
            },
            'en': {
                'high': ['very', 'extremely', 'exceptionally', 'incredibly'],
                'medium': ['quite', 'rather', 'fairly'],
                'low': ['a bit', 'slightly', 'somewhat']
            }
        }
        
        modifiers = intensity_modifiers.get(self.language, {})
        
        if intensity > 0.8:
            modifier = np.random.choice(modifiers.get('high', ['']))
        elif intensity > 0.5:
            modifier = np.random.choice(modifiers.get('medium', ['']))
        else:
            modifier = np.random.choice(modifiers.get('low', ['']))
        
        if modifier:
            return f"{modifier} {text}"
        return text

    def set_emotion_state(self, emotion_state):
        """设置情感状态 | Set emotion state"""
        self.emotional_state.update(emotion_state)
        self.logger.info(f"情感状态更新: {emotion_state}")

    def get_emotional_state(self):
        """获取当前情感状态 | Get current emotional state"""
        return self.emotional_state

    def get_detailed_state(self):
        """获取详细情感状态 | Get detailed emotional state"""
        return {
            'current_emotion': self.emotional_state['current_emotion'],
            'intensity': self.emotional_state['intensity'],
            'confidence': self.emotional_state['confidence'],
            'valence': self.emotional_state['valence'],
            'arousal': self.emotional_state['arousal'],
            'dominance': self.emotional_state['dominance'],
            'recent_history': list(self.emotion_history)[-5:] if self.emotion_history else [],
            'language': self.language
        }

    def set_language(self, language):
        """设置语言 | Set language"""
        if language in ['zh', 'en']:
            self.language = language
            self.logger.info(f"情感引擎语言设置为: {language}")
        else:
            self.logger.warning(f"不支持的语言: {language}")

    def _update_emotional_state(self, emotion_data):
        """更新情感状态 | Update emotional state"""
        emotion_type = emotion_data['type']
        intensity = emotion_data['intensity']
        
        # 更新8维情感值
        if emotion_type in self.emotional_state:
            self.emotional_state[emotion_type] = intensity
        
        # 更新当前主要情感
        self.emotional_state['current_primary_emotion'] = emotion_type
        self.emotional_state['emotional_intensity'] = intensity
        self.emotional_state['emotional_confidence'] = emotion_data['confidence']
        
        # 更新VAD维度（简化更新）
        # 基于情感类型调整valence, arousal, dominance
        vad_map = {
            'joy': (0.7, 0.6, 0.6),
            'sadness': (-0.6, 0.3, -0.4),
            'anger': (-0.5, 0.8, 0.7),
            'fear': (-0.7, 0.9, -0.5),
            'trust': (0.6, 0.4, 0.5),
            'surprise': (0.3, 0.7, 0.2),
            'anticipation': (0.5, 0.5, 0.4),
            'disgust': (-0.6, 0.5, -0.3),
            'neutral': (0.0, 0.3, 0.0)
        }
        
        if emotion_type in vad_map:
            v, a, d = vad_map[emotion_type]
            self.emotional_state['valence'] = v * intensity
            self.emotional_state['arousal'] = a * intensity
            self.emotional_state['dominance'] = d * intensity
        
        # 添加情感记忆
        self._add_to_emotional_memory(emotion_data)
        
        # 检测情感模式
        self._detect_emotional_patterns()
        
        self.logger.debug(f"情感状态已更新: {self.emotional_state}")

    def _add_to_emotional_memory(self, emotion_data):
        """添加情感记忆 | Add to emotional memory"""
        memory_entry = {
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion_data['type'],
            'intensity': emotion_data['intensity'],
            'context': emotion_data.get('context', ''),
            'triggers': self._extract_emotional_triggers(emotion_data)
        }
        self.emotional_state['emotional_memory'].append(memory_entry)
        # 保持内存大小
        if len(self.emotional_state['emotional_memory']) > 100:
            self.emotional_state['emotional_memory'].pop(0)

    def _extract_emotional_triggers(self, emotion_data):
        """提取情感触发器 | Extract emotional triggers"""
        # 简化版：从文本中提取关键词
        text = emotion_data.get('text', '')
        if text:
            # 使用简单的方法提取名词或动词作为触发器
            words = text.split()
            triggers = [word for word in words if len(word) > 3]  # 简单过滤
            return triggers[:5]  # 返回前5个
        return []

    def _detect_emotional_patterns(self):
        """检测情感模式 | Detect emotional patterns"""
        memory = self.emotional_state['emotional_memory']
        if len(memory) < 3:
            return  # 需要足够的数据
        
        # 分析最近的情感序列
        recent_emotions = [entry['emotion'] for entry in memory[-10:]]  # 最近10个
        
        # 查找模式，例如连续相同情感
        pattern_key = '->'.join(recent_emotions)
        if pattern_key in self.emotion_patterns_db:
            self.emotion_patterns_db[pattern_key]['count'] += 1
            self.emotion_patterns_db[pattern_key]['last_occurred'] = datetime.now().isoformat()
        else:
            self.emotion_patterns_db[pattern_key] = {
                'count': 1,
                'first_occurred': datetime.now().isoformat(),
                'last_occurred': datetime.now().isoformat()
            }
        
        # 更新情感模式状态
        self.emotional_state['emotional_patterns'] = self.emotion_patterns_db

    def export_emotional_data(self, file_path):
        """导出情感数据 | Export emotional data"""
        data = {
            'emotional_state': self.emotional_state,
            'emotion_history': list(self.emotion_history),
            'emotion_patterns_db': self.emotion_patterns_db
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"情感数据已导出到: {file_path}")

    def import_emotional_data(self, file_path):
        """导入情感数据 | Import emotional data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.emotional_state.update(data.get('emotional_state', {}))
            self.emotion_history = deque(data.get('emotion_history', []), maxlen=1000)
            self.emotion_patterns_db = data.get('emotion_patterns_db', {})
            self.logger.info(f"情感数据已从 {file_path} 导入")
        except Exception as e:
            self.logger.error(f"导入情感数据失败: {e}")

    def clear_emotional_memory(self):
        """清空情感记忆 | Clear emotional memory"""
        self.emotional_state['emotional_memory'] = []
        self.emotion_history.clear()
        self.emotion_patterns_db = {}
        self.logger.info("情感记忆已清空")

# 单例实例 | Singleton instance
_emotion_engine_instance = None

def get_emotion_engine(model_registry, language='zh'):
    """获取情感引擎实例 | Get emotion engine instance"""
    global _emotion_engine_instance
    if _emotion_engine_instance is None:
        _emotion_engine_instance = AdvancedEmotionEngine(model_registry, language)
    return _emotion_engine_instance
