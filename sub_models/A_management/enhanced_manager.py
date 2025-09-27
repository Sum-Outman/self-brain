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
增强的A_management模型，管理下属模型的情感分析功能
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
from datetime import datetime
from collections import deque
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('A_management')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ManagementModel(nn.Module):
    """
    增强的管理模型，具有下属模型情感分析功能管理能力
    """
    def __init__(self, config=None):
        """
        初始化管理模型
        
        参数:
            config: 配置字典，包含模型参数和下属模型信息
        """
        super().__init__()
        
        # 配置默认值
        default_config = {
            'hidden_dim': 512,
            'num_strategies': 10,
            'num_emotions': 7,
            'sub_model_config': {
                'B_language': {'has_emotion_analysis': True, 'emotion_types': ['main', 'sub']},
                'C_audio': {'has_emotion_analysis': True, 'emotion_types': ['main']},
                'D_image': {'has_emotion_analysis': True, 'emotion_types': ['main']},
                'E_video': {'has_emotion_analysis': True, 'emotion_types': ['main', 'intensity']},
                'F_spatial': {'has_emotion_analysis': False},
                'G_sensor': {'has_emotion_analysis': False},
                'I_knowledge': {'has_emotion_analysis': False},
                'J_motion': {'has_emotion_analysis': False},
                'K_programming': {'has_emotion_analysis': False}
            },
            'emotion_weights': {
                'B_language': 0.35,
                'C_audio': 0.25,
                'D_image': 0.20,
                'E_video': 0.20
            },
            'decision_threshold': 0.6
        }
        
        # 合并配置
        self.config = {**default_config, **(config or {})}
        
        # 主管理层
        self.manager_layer = nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'])
        
        # 策略输出层
        self.strategy_layer = nn.Linear(self.config['hidden_dim'], self.config['num_strategies'])
        
        # 情感分析输出层
        self.emotion_layer = nn.Linear(self.config['hidden_dim'], self.config['num_emotions'])
        
        # 激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
        # 模型状态记录
        self.sub_models = {}
        self.sub_model_status = {}
        self.emotion_history = deque(maxlen=100)
        self.decision_history = deque(maxlen=100)
        
        # 初始化模型状态
        self.initialize_model_state()
        
    def initialize_model_state(self):
        """
        初始化模型状态
        """
        for model_name in self.config['sub_model_config']:
            self.sub_model_status[model_name] = {
                'status': 'inactive',
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'performance': {'accuracy': 0.0, 'inference_time': 0.0},
                'emotion_stats': {}
            }
    
    def forward(self, input_features, sub_model_outputs=None):
        """
        前向传播，处理输入并整合下属模型的输出
        
        参数:
            input_features: 输入特征
            sub_model_outputs: 下属模型的输出字典
        
        返回:
            策略预测和情感预测
        """
        # 处理输入特征
        if isinstance(input_features, dict):
            # 处理字典类型输入
            feature_vector = self._process_dict_input(input_features)
        else:
            # 假设是张量类型输入
            feature_vector = input_features
        
        # 主管理层处理
        management_output = self.relu(self.manager_layer(feature_vector))
        
        # 生成策略预测
        strategy_logits = self.strategy_layer(management_output)
        strategy_probs = self.softmax(strategy_logits)
        
        # 生成情感预测
        emotion_logits = self.emotion_layer(management_output)
        emotion_probs = self.softmax(emotion_logits)
        
        # 如果有下属模型输出，整合情感分析结果
        if sub_model_outputs:
            integrated_emotions = self._integrate_sub_model_emotions(sub_model_outputs)
            final_emotion = self._combine_emotions(emotion_probs, integrated_emotions)
            return strategy_probs, final_emotion
        
        return strategy_probs, emotion_probs
    
    def _process_dict_input(self, input_dict):
        """
        处理字典类型的输入
        """
        # 提取并合并各种输入特征
        feature_list = []
        
        # 处理文本特征
        if 'text' in input_dict:
            text_feat = torch.tensor(input_dict['text']).to(device) if isinstance(input_dict['text'], list) else input_dict['text']
            if len(text_feat.shape) == 1:
                text_feat = text_feat.unsqueeze(0)
            feature_list.append(text_feat)
        
        # 处理其他类型的特征
        for key, value in input_dict.items():
            if key != 'text' and isinstance(value, (torch.Tensor, list, np.ndarray)):
                if isinstance(value, (list, np.ndarray)):
                    value = torch.tensor(value).to(device)
                if len(value.shape) == 1:
                    value = value.unsqueeze(0)
                feature_list.append(value)
        
        # 如果没有特征，创建默认特征
        if not feature_list:
            return torch.zeros(1, self.config['hidden_dim']).to(device)
        
        # 合并特征（这里简单拼接，实际可能需要更复杂的特征融合）
        max_length = max(f.shape[1] for f in feature_list)
        padded_features = []
        
        for feat in feature_list:
            if feat.shape[1] < max_length:
                padding = torch.zeros(feat.shape[0], max_length - feat.shape[1]).to(device)
                padded_feat = torch.cat([feat, padding], dim=1)
            else:
                padded_feat = feat[:, :max_length]
            padded_features.append(padded_feat)
        
        # 特征融合
        combined = torch.cat(padded_features, dim=1)
        
        # 调整到目标维度
        if combined.shape[1] != self.config['hidden_dim']:
            proj_layer = nn.Linear(combined.shape[1], self.config['hidden_dim']).to(device)
            combined = proj_layer(combined)
        
        return combined
    
    def _integrate_sub_model_emotions(self, sub_model_outputs):
        """
        整合下属模型的情感分析结果
        
        参数:
            sub_model_outputs: 下属模型的输出字典
        
        返回:
            整合后的情感结果
        """
        integrated = {}
        
        # 定义标准情绪列表（与各模型保持一致）
        standard_emotions = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        
        # 初始化整合情感字典
        for emotion in standard_emotions:
            integrated[emotion] = 0.0
        
        total_weight = 0.0
        
        # 遍历下属模型输出
        for model_name, output in sub_model_outputs.items():
            # 检查模型是否配置了情感分析功能
            if (model_name in self.config['sub_model_config'] and 
                self.config['sub_model_config'][model_name]['has_emotion_analysis']):
                
                # 获取该模型的权重
                weight = self.config['emotion_weights'].get(model_name, 0.1)
                total_weight += weight
                
                # 处理不同模型的情感输出格式
                if model_name == 'B_language' and 'emotion_distribution' in output:
                    # 处理语言模型的情感分布
                    for emotion, prob in output['emotion_distribution'].items():
                        if emotion in integrated:
                            integrated[emotion] += prob * weight
                
                elif 'emotion' in output:
                    # 处理简单的情感标签输出
                    emotion = output['emotion']
                    if emotion in integrated:
                        integrated[emotion] += 1.0 * weight
                
                elif 'emotion_probs' in output:
                    # 处理概率分布输出
                    probs = output['emotion_probs']
                    if isinstance(probs, dict):
                        for emotion, prob in probs.items():
                            if emotion in integrated:
                                integrated[emotion] += prob * weight
                    elif isinstance(probs, (list, np.ndarray, torch.Tensor)):
                        # 假设顺序与standard_emotions一致
                        for i, prob in enumerate(probs):
                            if i < len(standard_emotions):
                                emotion = standard_emotions[i]
                                integrated[emotion] += float(prob) * weight
        
        # 归一化整合结果
        if total_weight > 0:
            for emotion in integrated:
                integrated[emotion] /= total_weight
        
        # 更新下属模型状态
        self._update_sub_model_status(sub_model_outputs)
        
        return integrated
    
    def _combine_emotions(self, own_emotion, integrated_emotions):
        """
        结合自身情感分析结果和整合后的下属模型情感结果
        """
        # 转换自身情感结果为字典格式
        standard_emotions = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        own_emotion_dict = {}
        
        for i, emotion in enumerate(standard_emotions):
            if i < len(own_emotion):
                own_emotion_dict[emotion] = float(own_emotion[i])
            else:
                own_emotion_dict[emotion] = 0.0
        
        # 结合两种情感结果（这里使用简单平均，实际可能需要更复杂的结合策略）
        combined = {}
        for emotion in standard_emotions:
            combined[emotion] = (own_emotion_dict.get(emotion, 0.0) + integrated_emotions.get(emotion, 0.0)) / 2
        
        # 将结果转换回张量
        combined_tensor = torch.tensor([combined[emotion] for emotion in standard_emotions]).to(device)
        
        # 记录情感历史
        self.emotion_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'own_emotion': own_emotion_dict,
            'integrated_emotions': integrated_emotions,
            'combined_emotion': combined
        })
        
        return combined_tensor
    
    def _update_sub_model_status(self, sub_model_outputs):
        """
        更新下属模型的状态信息
        """
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for model_name, output in sub_model_outputs.items():
            if model_name in self.sub_model_status:
                # 更新基本状态
                self.sub_model_status[model_name]['status'] = 'active'
                self.sub_model_status[model_name]['last_update'] = current_time
                
                # 更新性能指标
                if 'accuracy' in output:
                    self.sub_model_status[model_name]['performance']['accuracy'] = float(output['accuracy'])
                if 'processing_time_ms' in output:
                    self.sub_model_status[model_name]['performance']['inference_time'] = float(output['processing_time_ms'])
                
                # 更新情感统计
                emotion_stats = self.sub_model_status[model_name]['emotion_stats']
                if 'emotion_distribution' in output:
                    for emotion, prob in output['emotion_distribution'].items():
                        if emotion not in emotion_stats:
                            emotion_stats[emotion] = []
                        emotion_stats[emotion].append(float(prob))
                        # 限制历史记录长度
                        if len(emotion_stats[emotion]) > 50:
                            emotion_stats[emotion] = emotion_stats[emotion][-50:]
    
    def register_sub_model(self, model_name, model_instance):
        """
        注册下属模型
        
        参数:
            model_name: 模型名称
            model_instance: 模型实例
        """
        self.sub_models[model_name] = model_instance
        logger.info(f"Sub-model {model_name} registered successfully.")
        
        # 更新模型状态
        if model_name in self.sub_model_status:
            self.sub_model_status[model_name]['status'] = 'active'
            self.sub_model_status[model_name]['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def unregister_sub_model(self, model_name):
        """
        注销下属模型
        
        参数:
            model_name: 模型名称
        """
        if model_name in self.sub_models:
            del self.sub_models[model_name]
            logger.info(f"Sub-model {model_name} unregistered successfully.")
            
            # 更新模型状态
            if model_name in self.sub_model_status:
                self.sub_model_status[model_name]['status'] = 'inactive'
                self.sub_model_status[model_name]['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            logger.warning(f"Sub-model {model_name} not found.")
    
    def process_task(self, task_input, use_sub_models=True):
        """
        处理任务，可选是否使用下属模型
        
        参数:
            task_input: 任务输入
            use_sub_models: 是否使用下属模型
        
        返回:
            处理结果
        """
        start_time = time.time()
        
        # 准备输入特征
        if isinstance(task_input, str):
            input_features = {'text': task_input}
        else:
            input_features = task_input
        
        # 初始化结果字典
        result = {
            'task_type': self._classify_task_type(input_features),
            'manager_decision': None,
            'manager_emotion': None,
            'sub_model_results': {},
            'integrated_result': None
        }
        
        # 收集下属模型的输出
        sub_model_outputs = {}
        if use_sub_models:
            sub_model_outputs = self._collect_sub_model_outputs(input_features)
            result['sub_model_results'] = sub_model_outputs
        
        # 前向传播获取管理模型的决策和情感
        with torch.no_grad():
            strategy_probs, emotion_probs = self.forward(input_features, sub_model_outputs if use_sub_models else None)
        
        # 转换为numpy数组便于处理
        if isinstance(strategy_probs, torch.Tensor):
            strategy_probs = strategy_probs.cpu().numpy()
        if isinstance(emotion_probs, torch.Tensor):
            emotion_probs = emotion_probs.cpu().numpy()
        
        # 获取最高概率的策略和情感
        best_strategy_idx = np.argmax(strategy_probs)
        
        # 如果是numpy数组，转换为列表
        if isinstance(strategy_probs, np.ndarray):
            strategy_probs = strategy_probs.tolist()
        if isinstance(emotion_probs, np.ndarray):
            emotion_probs = emotion_probs.tolist()
        
        # 记录决策和情感
        result['manager_decision'] = {
            'strategy_id': int(best_strategy_idx),
            'confidence': float(strategy_probs[best_strategy_idx]),
            'all_strategies': strategy_probs
        }
        
        # 构建情感字典
        standard_emotions = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        emotion_dict = {}
        for i, emotion in enumerate(standard_emotions):
            if i < len(emotion_probs):
                emotion_dict[emotion] = float(emotion_probs[i])
        
        result['manager_emotion'] = emotion_dict
        
        # 整合结果
        result['integrated_result'] = self._generate_integrated_result(result)
        
        # 记录决策历史
        self.decision_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'task_type': result['task_type'],
            'decision': result['manager_decision'],
            'emotion': result['manager_emotion'],
            'processing_time': (time.time() - start_time) * 1000  # 毫秒
        })
        
        return result
    
    def _classify_task_type(self, task_input):
        """
        分类任务类型
        """
        # 基于输入特征的简单任务类型分类
        if isinstance(task_input, dict):
            if 'text' in task_input and any(key in task_input for key in ['audio', 'image', 'video']):
                return 'multimodal'
            elif 'text' in task_input:
                return 'text'
            elif 'audio' in task_input:
                return 'audio'
            elif 'image' in task_input:
                return 'image'
            elif 'video' in task_input:
                return 'video'
            elif 'sensor' in task_input:
                return 'sensor'
            elif 'spatial' in task_input:
                return 'spatial'
        elif isinstance(task_input, str):
            return 'text'
        
        return 'unknown'
    
    def _collect_sub_model_outputs(self, task_input):
        """
        收集下属模型的输出
        """
        outputs = {}
        task_type = self._classify_task_type(task_input)
        
        # 根据任务类型选择合适的下属模型
        if task_type == 'text' or task_type == 'multimodal':
            if 'B_language' in self.sub_models:
                try:
                    # 提取文本部分
                    text_input = task_input['text'] if isinstance(task_input, dict) and 'text' in task_input else str(task_input)
                    lang_output = self.sub_models['B_language'].predict(text_input)
                    outputs['B_language'] = lang_output
                    logger.debug(f"B_language model output collected: {lang_output.get('primary_emotion', 'unknown')}")
                except Exception as e:
                    logger.error(f"Error collecting B_language output: {str(e)}")
        
        # 其他模型调用逻辑根据需要添加
        
        return outputs
    
    def _generate_integrated_result(self, partial_result):
        """
        生成整合结果
        """
        # 获取决策和情感
        decision = partial_result['manager_decision']
        emotion = partial_result['manager_emotion']
        
        # 基于决策置信度和情感状态生成整合结果
        confidence = decision['confidence']
        primary_emotion = max(emotion, key=emotion.get) if emotion else 'neutral'
        
        # 根据置信度阈值和情感类型调整结果
        if confidence >= self.config['decision_threshold']:
            result_type = 'high_confidence'
        else:
            result_type = 'low_confidence'
            # 低置信度时，可能需要更多信息或调用更多模型
        
        integrated = {
            'result_type': result_type,
            'confidence_score': confidence,
            'primary_emotion': primary_emotion,
            'emotion_intensity': emotion.get(primary_emotion, 0.0),
            'recommended_action': self._determine_recommended_action(decision, emotion)
        }
        
        return integrated
    
    def _determine_recommended_action(self, decision, emotion):
        """
        基于决策和情感确定推荐动作
        """
        strategy_id = decision['strategy_id']
        primary_emotion = max(emotion, key=emotion.get) if emotion else 'neutral'
        emotion_intensity = emotion.get(primary_emotion, 0.0)
        
        # 简单的动作推荐逻辑，实际应用中可以更复杂
        if emotion_intensity > 0.7:
            # 高情感强度时的动作推荐
            if primary_emotion in ['joy', 'surprise']:
                return {'type': 'positive_feedback', 'priority': 'high'}
            elif primary_emotion in ['anger', 'fear', 'sadness', 'disgust']:
                return {'type': 'emotion_regulation', 'priority': 'urgent'}
        
        # 基于策略的动作推荐
        if strategy_id % 3 == 0:
            return {'type': 'gather_more_info', 'priority': 'medium'}
        elif strategy_id % 3 == 1:
            return {'type': 'take_action', 'priority': 'high'}
        else:
            return {'type': 'monitor_situation', 'priority': 'low'}
    
    def adjust_response_based_on_emotion(self, response, emotion_context, user_emotion=None):
        """
        根据情感上下文调整响应
        
        参数:
            response: 原始响应
            emotion_context: 情感上下文信息
            user_emotion: 用户情感信息（可选）
        
        返回:
            调整后的响应
        """
        if not emotion_context:
            return response
        
        # 获取主要情感和强度
        primary_emotion = max(emotion_context, key=emotion_context.get) if emotion_context else 'neutral'
        emotion_intensity = emotion_context.get(primary_emotion, 0.0)
        
        # 根据不同情感和强度调整响应
        enhancements = {
            'joy': {
                'prefixes': ['😊 ', 'Great! ', 'Wonderful! '],
                'suffixes': [' 😊', '!', ' :D']
            },
            'sadness': {
                'prefixes': ['😢 ', 'I\'m sorry to hear that. ', 'That\'s unfortunate. '],
                'suffixes': [' 😢', '.', '...']
            },
            'anger': {
                'prefixes': ['😠 ', 'That\'s frustrating. ', 'I understand your frustration. '],
                'suffixes': [' 😠', '.', '!']
            },
            'fear': {
                'prefixes': ['😨 ', 'I understand your concern. ', 'Let\'s address this carefully. '],
                'suffixes': [' 😨', '.', '...']
            },
            'surprise': {
                'prefixes': ['😲 ', 'Wow! ', 'That\'s surprising! '],
                'suffixes': [' 😲', '!', '!!']
            },
            'disgust': {
                'prefixes': ['😒 ', 'That\'s unpleasant. ', 'That\'s not ideal. '],
                'suffixes': [' 😒', '.', '...']
            }
        }
        
        # 如果情感在增强字典中
        if primary_emotion in enhancements and emotion_intensity > 0.3:
            enh = enhancements[primary_emotion]
            
            # 根据情感强度选择增强程度
            if emotion_intensity > 0.7:
                # 高强度情感，使用前缀和后缀
                prefix = np.random.choice(enh['prefixes'])
                suffix = np.random.choice(enh['suffixes'])
                adjusted_response = f"{prefix}{response}{suffix}"
            elif emotion_intensity > 0.5:
                # 中等强度情感，只使用前缀
                prefix = np.random.choice(enh['prefixes'])
                adjusted_response = f"{prefix}{response}"
            else:
                # 低强度情感，只使用后缀
                suffix = np.random.choice(enh['suffixes'])
                adjusted_response = f"{response}{suffix}"
            
            return adjusted_response
        
        # 默认不调整
        return response
    
    def get_sub_model_status(self, model_name=None):
        """
        获取下属模型的状态信息
        
        参数:
            model_name: 可选，模型名称，如果不提供则返回所有模型的状态
        
        返回:
            模型状态信息
        """
        if model_name:
            return self.sub_model_status.get(model_name, {})
        else:
            return self.sub_model_status
    
    def get_system_status(self):
        """
        获取整个管理系统的状态信息
        """
        # 计算活跃模型数量
        active_models = sum(1 for status in self.sub_model_status.values() if status['status'] == 'active')
        total_models = len(self.sub_model_status)
        
        # 计算平均性能指标
        avg_accuracy = []
        avg_inference_time = []
        
        for status in self.sub_model_status.values():
            if status['performance']['accuracy'] > 0:
                avg_accuracy.append(status['performance']['accuracy'])
            if status['performance']['inference_time'] > 0:
                avg_inference_time.append(status['performance']['inference_time'])
        
        # 获取最近的情感趋势
        recent_emotions = []
        for emotion_record in list(self.emotion_history)[-10:]:  # 最近10条记录
            if 'combined_emotion' in emotion_record:
                recent_emotions.append(emotion_record['combined_emotion'])
        
        # 计算情感趋势
        emotion_trend = {}
        if recent_emotions:
            for emotion in recent_emotions[0]:
                values = [record[emotion] for record in recent_emotions]
                emotion_trend[emotion] = {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values) if values else 0,
                    'change': values[-1] - values[0] if len(values) > 1 else 0
                }
        
        status = {
            'system_status': 'healthy' if active_models > 0 else 'warning',
            'active_models': active_models,
            'total_models': total_models,
            'model_status': self.sub_model_status,
            'performance_metrics': {
                'avg_accuracy': np.mean(avg_accuracy) if avg_accuracy else 0,
                'avg_inference_time_ms': np.mean(avg_inference_time) if avg_inference_time else 0
            },
            'emotion_trend': emotion_trend,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return status
    
    def save_model(self, path):
        """
        保存模型到文件
        
        参数:
            path: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型状态和配置
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'sub_model_status': self.sub_model_status
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        从文件加载模型
        
        参数:
            path: 模型文件路径
        """
        checkpoint = torch.load(path, map_location=device)
        
        # 加载模型状态字典
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载配置
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        # 加载下属模型状态
        if 'sub_model_status' in checkpoint:
            self.sub_model_status = checkpoint['sub_model_status']
        
        logger.info(f"Model loaded from {path}")
        
        return self

# 工具函数
def create_management_model(config=None):
    """
    创建管理模型实例
    
    参数:
        config: 配置字典
    
    返回:
        ManagementModel实例
    """
    model = ManagementModel(config)
    
    return model

def generate_management_report(model, report_path):
    """
    生成管理报告
    
    参数:
        model: ManagementModel实例
        report_path: 报告保存路径
    """
    # 获取系统状态
    system_status = model.get_system_status()
    
    # 获取最近的决策历史
    recent_decisions = list(model.decision_history)[-50:]  # 最近50条决策
    
    # 准备报告数据
    report = {
        'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'system_status': system_status,
        'recent_decisions': recent_decisions,
        'model_config': model.config,
        'summary': {
            'active_models': system_status['active_models'],
            'avg_accuracy': system_status['performance_metrics']['avg_accuracy'],
            'dominant_emotion': max(system_status['emotion_trend'], key=lambda x: system_status['emotion_trend'][x]['current']) if system_status['emotion_trend'] else 'neutral'
        }
    }
    
    # 保存报告
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Management report generated and saved to {report_path}")
    
    return report