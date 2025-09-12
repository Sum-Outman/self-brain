#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实A管理模型训练系统 - 实时数据生成和训练
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import websockets
import threading
import queue
from collections import deque
import os
import sys

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_trainable_a_manager import RealTrainableAManager, ModelConfig, InteractiveAManager

class RealTimeDataGenerator:
    """实时数据生成器 - 模拟真实环境数据"""
    
    def __init__(self):
        self.running = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.emotion_history = deque(maxlen=1000)
        self.task_history = deque(maxlen=1000)
        
        # 模拟数据模板
        self.task_templates = {
            'text': [
                "分析这段文本的情感色彩",
                "将这段文字翻译成英文",
                "总结这篇文章的主要内容",
                "检测文本中的敏感信息",
                "生成一段关于AI的诗歌"
            ],
            'audio': [
                "识别这段语音的内容",
                "将语音转换为文字",
                "分析说话人的情感",
                "检测语音中的关键词",
                "生成语音回复"
            ],
            'image': [
                "识别图片中的物体",
                "检测人脸并分析表情",
                "提取图片中的文字",
                "判断图片的风格",
                "生成图片描述"
            ],
            'video': [
                "分析视频内容",
                "检测视频中的人物",
                "提取视频关键帧",
                "识别视频中的动作",
                "生成视频摘要"
            ],
            'spatial': [
                "计算物体的3D位置",
                "分析空间关系",
                "检测障碍物",
                "规划路径",
                "测量距离"
            ],
            'sensor': [
                "分析传感器数据",
                "检测异常读数",
                "预测设备状态",
                "校准传感器",
                "数据融合"
            ],
            'control': [
                "优化系统性能",
                "调整参数设置",
                "监控系统状态",
                "故障诊断",
                "资源分配"
            ],
            'motion': [
                "规划运动轨迹",
                "控制机械臂",
                "平衡控制",
                "步态规划",
                "碰撞避免"
            ],
            'knowledge': [
                "查询知识库",
                "推理因果关系",
                "验证事实",
                "知识图谱查询",
                "专家咨询"
            ],
            'programming': [
                "生成代码片段",
                "调试程序",
                "代码优化",
                "算法设计",
                "API文档"
            ]
        }
        
        # 情感标签
        self.emotion_labels = [
            'joy', 'sadness', 'anger', 'fear', 
            'surprise', 'disgust', 'trust', 'anticipation'
        ]
        
        # 模型名称
        self.model_names = [
            'A_language', 'B_audio', 'C_image', 'D_video', 'E_spatial',
            'F_sensor', 'G_computer', 'H_motion', 'I_knowledge', 
            'J_controller', 'K_programming'
        ]
    
    def start_generation(self):
        """开始实时数据生成"""
        self.running = True
        self.generation_thread = threading.Thread(target=self._generate_loop)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        logging.info("实时数据生成器已启动")
    
    def stop_generation(self):
        """停止数据生成"""
        self.running = False
        if hasattr(self, 'generation_thread'):
            self.generation_thread.join()
        logging.info("实时数据生成器已停止")
    
    def _generate_loop(self):
        """数据生成主循环"""
        while self.running:
            try:
                # 生成任务数据
                task_data = self._generate_task_data()
                
                # 生成情感数据
                emotion_data = self._generate_emotion_data()
                
                # 生成训练样本
                training_sample = self._generate_training_sample(task_data, emotion_data)
                
                # 放入队列
                if not self.data_queue.full():
                    self.data_queue.put(training_sample)
                
                # 记录历史
                self.task_history.append(task_data)
                self.emotion_history.append(emotion_data)
                
                time.sleep(0.1)  # 控制生成速度
                
            except Exception as e:
                logging.error(f"数据生成错误: {e}")
    
    def _generate_task_data(self) -> Dict:
        """生成任务数据"""
        task_type = np.random.choice(list(self.task_templates.keys()))
        description = np.random.choice(self.task_templates[task_type])
        
        # 生成复杂度评分
        complexity = np.random.beta(2, 5)  # 偏向低复杂度
        
        # 生成预期处理时间
        expected_time = np.random.exponential(2) + 0.5
        
        return {
            'type': task_type,
            'description': description,
            'complexity': float(complexity),
            'expected_time': float(expected_time),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_emotion_data(self) -> Dict:
        """生成情感数据"""
        # 基于时间变化的情感
        current_time = datetime.now()
        hour_factor = np.sin(2 * np.pi * current_time.hour / 24)
        
        # 基础情感分布
        base_emotions = np.array([0.3, 0.1, 0.05, 0.05, 0.2, 0.05, 0.15, 0.1])
        
        # 添加随机波动和时间影响
        noise = np.random.normal(0, 0.1, 8)
        time_influence = np.array([
            hour_factor * 0.1,  # joy
            -abs(hour_factor) * 0.05,  # sadness
            abs(hour_factor) * 0.03,  # anger
            0.02,  # fear
            0.05,  # surprise
            0.02,  # disgust
            0.03,  # trust
            0.04   # anticipation
        ])
        
        emotions = base_emotions + noise + time_influence
        emotions = np.clip(emotions, 0.01, 0.99)
        emotions = emotions / emotions.sum()
        
        return {
            'emotions': emotions.tolist(),
            'timestamp': current_time.isoformat(),
            'hour_factor': float(hour_factor)
        }
    
    def _generate_training_sample(self, task_data: Dict, emotion_data: Dict) -> Dict:
        """生成训练样本"""
        # 模拟输入特征
        input_features = torch.randn(50, 768)
        
        # 生成目标标签
        emotion_targets = torch.tensor(emotion_data['emotions'])
        
        # 根据任务类型选择目标模型
        task_type_idx = list(self.task_templates.keys()).index(task_data['type'])
        model_targets = torch.zeros(11)
        model_targets[task_type_idx % 11] = 0.8  # 主要模型
        model_targets[(task_type_idx + 1) % 11] = 0.2  # 辅助模型
        
        # 生成其他目标
        importance_target = torch.tensor([task_data['complexity']])
        confidence_target = torch.tensor([0.8 + 0.2 * np.random.random()])
        
        return {
            'input': input_features,
            'emotion_target': emotion_targets,
            'model_target': model_targets,
            'importance_target': importance_target,
            'confidence_target': confidence_target,
            'metadata': {
                'task': task_data,
                'emotion': emotion_data
            }
        }

class RealTimeTrainer:
    """实时训练器"""
    
    def __init__(self, model: RealTrainableAManager, data_generator: RealTimeDataGenerator):
        self.model = model
        self.data_generator = data_generator
        self.config = ModelConfig()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        # 损失函数
        self.criterion_emotion = nn.BCELoss()
        self.criterion_model = nn.KLDivLoss(reduction='batchmean')
        self.criterion_regression = nn.MSELoss()
        
        # 训练状态
        self.training = False
        self.metrics = {
            'total_loss': deque(maxlen=100),
            'emotion_loss': deque(maxlen=100),
            'model_loss': deque(maxlen=100),
            'confidence_loss': deque(maxlen=100)
        }
        
        # WebSocket服务器
        self.clients = set()
        self.websocket_thread = None
    
    def start_training(self):
        """开始实时训练"""
        self.training = True
        self.data_generator.start_generation()
        
        # 启动训练线程
        training_thread = threading.Thread(target=self._training_loop)
        training_thread.daemon = True
        training_thread.start()
        
        # 启动WebSocket服务器
        self.websocket_thread = threading.Thread(target=self._start_websocket_server)
        self.websocket_thread.daemon = True
        self.websocket_thread.start()
        
        logging.info("实时训练系统已启动")
    
    def stop_training(self):
        """停止训练"""
        self.training = False
        self.data_generator.stop_generation()
        logging.info("实时训练系统已停止")
    
    def _training_loop(self):
        """训练主循环"""
        batch_size = 16
        accumulation_steps = 4
        
        while self.training:
            try:
                # 收集批次数据
                batch_data = []
                for _ in range(batch_size):
                    if not self.data_generator.data_queue.empty():
                        sample = self.data_generator.data_queue.get()
                        batch_data.append(sample)
                
                if len(batch_data) == 0:
                    time.sleep(0.1)
                    continue
                
                # 准备训练数据
                inputs = torch.stack([sample['input'] for sample in batch_data])
                emotion_targets = torch.stack([sample['emotion_target'] for sample in batch_data])
                model_targets = torch.stack([sample['model_target'] for sample in batch_data])
                importance_targets = torch.stack([sample['importance_target'] for sample in batch_data])
                confidence_targets = torch.stack([sample['confidence_target'] for sample in batch_data])
                
                # 转移到设备
                inputs = inputs.to(self.config.device)
                emotion_targets = emotion_targets.to(self.config.device)
                model_targets = model_targets.to(self.config.device)
                importance_targets = importance_targets.to(self.config.device)
                confidence_targets = confidence_targets.to(self.config.device)
                
                # 前向传播
                self.model.train()
                outputs = self.model(inputs)
                
                # 计算损失
                emotion_loss = self.criterion_emotion(outputs['emotions'], emotion_targets)
                model_loss = self.criterion_model(
                    F.log_softmax(outputs['model_weights'], dim=-1),
                    model_targets
                )
                importance_loss = self.criterion_regression(
                    outputs['task_embedding'].mean(dim=1), 
                    importance_targets.squeeze()
                )
                confidence_loss = self.criterion_regression(outputs['confidence'], confidence_targets)
                
                # 总损失
                total_loss = (
                    emotion_loss * 0.3 +
                    model_loss * 0.4 +
                    importance_loss * 0.2 +
                    confidence_loss * 0.1
                )
                
                # 反向传播
                total_loss.backward()
                
                # 梯度累积
                if len(batch_data) >= batch_size:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # 记录指标
                self.metrics['total_loss'].append(total_loss.item())
                self.metrics['emotion_loss'].append(emotion_loss.item())
                self.metrics['model_loss'].append(model_loss.item())
                self.metrics['confidence_loss'].append(confidence_loss.item())
                
                # 定期保存模型
                if len(self.metrics['total_loss']) % 100 == 0:
                    self.model.save_model('a_manager_realtime_latest.pth')
                
                # 广播训练状态
                asyncio.create_task(self.broadcast_training_status())
                
            except Exception as e:
                logging.error(f"训练循环错误: {e}")
                time.sleep(1)
    
    async def broadcast_training_status(self):
        """广播训练状态"""
        if not self.clients:
            return
        
        status = {
            'type': 'training_status',
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'total_loss': np.mean(list(self.metrics['total_loss'])) if self.metrics['total_loss'] else 0,
                'emotion_loss': np.mean(list(self.metrics['emotion_loss'])) if self.metrics['emotion_loss'] else 0,
                'model_loss': np.mean(list(self.metrics['model_loss'])) if self.metrics['model_loss'] else 0,
                'confidence_loss': np.mean(list(self.metrics['confidence_loss'])) if self.metrics['confidence_loss'] else 0
            },
            'queue_size': self.data_generator.data_queue.qsize(),
            'training': self.training
        }
        
        message = json.dumps(status)
        await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True
        )
    
    async def handle_websocket_client(self, websocket, path):
        """处理WebSocket客户端"""
        self.clients.add(websocket)
        logging.info(f"客户端连接: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'get_status':
                    await self.broadcast_training_status()
                
                elif data['type'] == 'process_task':
                    # 处理任务请求
                    interactive = InteractiveAManager()
                    result = interactive.process_task(
                        data['task_type'], 
                        data['description']
                    )
                    await websocket.send(json.dumps({
                        'type': 'task_result',
                        'result': result
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logging.info(f"客户端断开: {websocket.remote_address}")
    
    def _start_websocket_server(self):
        """启动WebSocket服务器"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        start_server = websockets.serve(
            self.handle_websocket_client,
            'localhost',
            8766
        )
        
        loop.run_until_complete(start_server)
        logging.info("WebSocket服务器已启动: ws://localhost:8766")
        loop.run_forever()

class AManagerDashboard:
    """A管理模型训练仪表板"""
    
    def __init__(self):
        self.config = ModelConfig()
        self.model = RealTrainableAManager(self.config)
        self.data_generator = RealTimeDataGenerator()
        self.trainer = RealTimeTrainer(self.model, self.data_generator)
    
    def start(self):
        """启动完整系统"""
        print("🎯 启动A管理模型实时训练系统")
        print("=" * 50)
        
        # 检查设备
        print(f"设备: {self.config.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 启动训练
        self.trainer.start_training()
        
        print("\n✅ 系统已启动！")
        print("📊 访问仪表板: http://localhost:8766")
        print("🔧 训练状态: 实时更新中...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 停止系统...")
            self.trainer.stop_training()

if __name__ == "__main__":
    dashboard = AManagerDashboard()
    dashboard.start()