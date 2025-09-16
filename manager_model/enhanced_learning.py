import torch
import numpy as np
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Callable
import threading
from collections import deque
import json
import os
from abc import ABC, abstractmethod
import random
import copy
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_logger_level(level):
    logger.setLevel(level)

class LearningEnvironment(ABC):
    """强化学习环境抽象基类"""
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """重置环境到初始状态"""
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """执行动作并返回结果"""
        pass
    
    @abstractmethod
    def get_observation_space(self) -> Dict[str, Any]:
        """获取观测空间信息"""
        pass
    
    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        """获取动作空间信息"""
        pass

class RewardFunction(ABC):
    """奖励函数抽象基类"""
    
    @abstractmethod
    def calculate_reward(self, state: Dict[str, Any], action: Any, next_state: Dict[str, Any], done: bool) -> float:
        """计算奖励值"""
        pass

class ModelPerformanceTracker:
    """模型性能跟踪器"""
    
    def __init__(self, window_size: int = 100):
        self.metrics_history = {
            'loss': deque(maxlen=window_size),
            'accuracy': deque(maxlen=window_size),
            'precision': deque(maxlen=window_size),
            'recall': deque(maxlen=window_size),
            'f1_score': deque(maxlen=window_size),
            'response_time': deque(maxlen=window_size)
        }
        self.performance_improvement = 0.0
        self.baseline_metrics = None
        
    def add_metrics(self, metrics: Dict[str, float]):
        """添加一组性能指标"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # 更新性能提升指标
        self._update_performance_improvement()
    
    def get_current_metrics(self) -> Dict[str, float]:
        """获取当前性能指标"""
        return {
            key: np.mean(values) if values else 0.0 
            for key, values in self.metrics_history.items()
        }
    
    def _update_performance_improvement(self):
        """更新性能提升指标"""
        current = self.get_current_metrics()
        
        if self.baseline_metrics is None:
            self.baseline_metrics = current
            return
        
        # 计算性能提升（简化版）
        improvements = []
        for key, value in current.items():
            baseline_value = self.baseline_metrics[key]
            if baseline_value > 0:
                if key == 'loss' or key == 'response_time':
                    # 对于损失和响应时间，值越小越好
                    improvement = (baseline_value - value) / baseline_value
                else:
                    # 对于准确率等指标，值越大越好
                    improvement = (value - baseline_value) / baseline_value
                improvements.append(improvement)
        
        if improvements:
            self.performance_improvement = np.mean(improvements)
    
    def reset_baseline(self):
        """重置性能基线"""
        self.baseline_metrics = self.get_current_metrics()

class ReinforcementLearningAgent:
    """强化学习智能体"""
    
    def __init__(self, model, env: LearningEnvironment, reward_fn: RewardFunction, config: Dict = None):
        self.model = model
        self.env = env
        self.reward_fn = reward_fn
        
        # 默认配置
        self.config = {
            'gamma': 0.99,          # 折扣因子
            'epsilon': 1.0,         # 探索率
            'epsilon_min': 0.01,    # 最小探索率
            'epsilon_decay': 0.995, # 探索率衰减
            'learning_rate': 0.001, # 学习率
            'batch_size': 64,       # 批次大小
            'memory_size': 10000,   # 经验回放缓冲区大小
            'target_update': 10     # 目标网络更新频率
        }
        
        if config:
            self.config.update(config)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=self.config['memory_size'])
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        # 目标网络
        self.target_model = copy.deepcopy(self.model)
        
        # 训练状态
        self.training = False
        self.steps_done = 0
        self.episode_rewards = []
        self.performance_tracker = ModelPerformanceTracker()
        
    def select_action(self, state: Dict[str, Any]) -> Any:
        """选择动作"""
        # epsilon-贪婪策略
        if random.random() < self.config['epsilon']:
            # 探索：随机选择动作
            action_space = self.env.get_action_space()
            # 根据动作空间类型选择随机动作
            if isinstance(action_space, Dict) and 'discrete' in action_space:
                return random.randint(0, action_space['discrete'] - 1)
            elif isinstance(action_space, Dict) and 'continuous' in action_space:
                low, high = action_space['continuous']
                return np.random.uniform(low, high)
            else:
                return random.choice(action_space)
        else:
            # 利用：模型预测最佳动作
            with torch.no_grad():
                # 假设模型可以接受状态并返回动作概率分布
                state_tensor = torch.FloatTensor(self._flatten_state(state))
                action_probs = self.model(state_tensor)
                return torch.argmax(action_probs).item()
    
    def train_step(self, batch_size: int = None):
        """执行一步训练"""
        if len(self.memory) < (batch_size or self.config['batch_size']):
            return
        
        # 从记忆中随机采样批次
        batch = random.sample(self.memory, batch_size or self.config['batch_size'])
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states_tensor = torch.FloatTensor([self._flatten_state(s) for s in states])
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor([self._flatten_state(s) for s in next_states])
        dones_tensor = torch.FloatTensor(dones)
        
        # 计算当前Q值
        current_q = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_model(next_states_tensor).max(1)[0]
            target_q = rewards_tensor + (1 - dones_tensor) * self.config['gamma'] * next_q
        
        # 计算损失并更新模型
        loss = torch.nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新探索率
        self.config['epsilon'] = max(self.config['epsilon_min'], self.config['epsilon'] * self.config['epsilon_decay'])
        
        # 定期更新目标网络
        self.steps_done += 1
        if self.steps_done % self.config['target_update'] == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        return {'loss': loss.item()}
    
    def remember(self, state: Dict[str, Any], action: Any, reward: float, next_state: Dict[str, Any], done: bool):
        """将经验存储到记忆缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_episode(self, max_steps: int = 1000):
        """训练一个回合"""
        state = self.env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # 选择动作
            action = self.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            self.remember(state, action, reward, next_state, done)
            
            # 更新状态和奖励
            state = next_state
            total_reward += reward
            steps += 1
            
            # 执行训练步骤
            metrics = self.train_step()
            if metrics:
                self.performance_tracker.add_metrics(metrics)
        
        self.episode_rewards.append(total_reward)
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': self.config['epsilon'],
            'metrics': self.performance_tracker.get_current_metrics()
        }
    
    def _flatten_state(self, state: Dict[str, Any]) -> List[float]:
        """将状态字典展平为特征向量"""
        # 简化实现，实际应用中需要根据状态空间定义具体的特征提取方法
        features = []
        for key, value in state.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, (list, np.ndarray)):
                features.extend([float(x) for x in value[:10]])  # 限制长度以避免维度爆炸
        return features

class MetaLearner:
    """元学习器实现"""
    
    def __init__(self, base_model, config: Dict = None):
        self.base_model = base_model
        self.meta_model = copy.deepcopy(base_model)
        
        # 默认配置
        self.config = {
            'meta_lr': 0.001,      # 元学习率
            'inner_lr': 0.01,      # 内部学习率
            'k_shot': 5,           # K-shot学习
            'k_query': 5,          # 用于评估的样本数
            'num_tasks': 10,       # 每次元训练的任务数
            'meta_batch_size': 3   # 元批次大小
        }
        
        if config:
            self.config.update(config)
        
        # 元优化器
        self.meta_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=self.config['meta_lr'])
        
        # 性能跟踪
        self.performance_tracker = ModelPerformanceTracker()
        
    def adapt_to_task(self, task_data: Dict[str, Any], num_steps: int = 1) -> torch.nn.Module:
        """快速适应新任务"""
        # 创建模型副本用于适应
        adapted_model = copy.deepcopy(self.meta_model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.config['inner_lr'])
        
        # 在少量任务数据上进行快速适应
        for _ in range(num_steps):
            # 假设task_data包含'support_set'用于适应
            for x, y in task_data['support_set']:
                optimizer.zero_grad()
                output = adapted_model(x)
                loss = torch.nn.MSELoss()(output, y) if isinstance(y, torch.Tensor) else torch.nn.CrossEntropyLoss()(output, y)
                loss.backward()
                optimizer.step()
        
        return adapted_model
    
    def meta_train(self, tasks: List[Dict[str, Any]]):
        """元训练过程"""
        meta_losses = []
        
        # 对每个任务进行元学习
        for task in tasks[:self.config['meta_batch_size']]:
            # 快速适应任务
            adapted_model = self.adapt_to_task(task, num_steps=5)
            
            # 在查询集上计算损失
            query_loss = 0
            for x, y in task['query_set']:
                output = adapted_model(x)
                loss = torch.nn.MSELoss()(output, y) if isinstance(y, torch.Tensor) else torch.nn.CrossEntropyLoss()(output, y)
                query_loss += loss
            
            meta_losses.append(query_loss / len(task['query_set']))
        
        # 计算平均元损失并更新元模型
        self.meta_optimizer.zero_grad()
        avg_meta_loss = torch.stack(meta_losses).mean()
        avg_meta_loss.backward()
        self.meta_optimizer.step()
        
        # 更新性能跟踪
        self.performance_tracker.add_metrics({'loss': avg_meta_loss.item()})
        
        return {
            'meta_loss': avg_meta_loss.item(),
            'metrics': self.performance_tracker.get_current_metrics()
        }
    
    def generate_tasks(self, data_generator: Callable) -> List[Dict[str, Any]]:
        """生成元学习任务"""
        tasks = []
        
        for _ in range(self.config['num_tasks']):
            # 生成任务数据
            task_data = data_generator()
            
            # 分割支持集和查询集
            support_set = task_data[:self.config['k_shot']]
            query_set = task_data[self.config['k_shot']:self.config['k_shot']+self.config['k_query']]
            
            tasks.append({
                'support_set': support_set,
                'query_set': query_set
            })
        
        return tasks

class KnowledgeDistillation:
    """知识蒸馏实现"""
    
    def __init__(self, teacher_model, student_model, config: Dict = None):
        self.teacher_model = teacher_model
        self.student_model = student_model
        
        # 默认配置
        self.config = {
            'temperature': 1.0,    # 温度参数，控制软化程度
            'alpha': 0.7,          # 知识蒸馏损失权重
            'learning_rate': 0.001,# 学习率
            'batch_size': 32       # 批次大小
        }
        
        if config:
            self.config.update(config)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=self.config['learning_rate'])
        
        # 性能跟踪
        self.performance_tracker = ModelPerformanceTracker()
        
    def distill_knowledge(self, data_loader):
        """执行知识蒸馏"""
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # 设置教师模型为评估模式
        self.teacher_model.eval()
        # 设置学生模型为训练模式
        self.student_model.train()
        
        for batch in data_loader:
            inputs, labels = batch
            
            # 教师模型生成软标签
            with torch.no_grad():
                teacher_logits = self.teacher_model(inputs)
                teacher_probs = torch.nn.functional.softmax(teacher_logits / self.config['temperature'], dim=1)
            
            # 学生模型预测
            student_logits = self.student_model(inputs)
            student_probs = torch.nn.functional.softmax(student_logits / self.config['temperature'], dim=1)
            
            # 计算知识蒸馏损失（KL散度）
            distillation_loss = torch.nn.KLDivLoss(reduction='batchmean')(student_probs.log(), teacher_probs)
            
            # 计算硬标签损失
            hard_loss = torch.nn.CrossEntropyLoss()(student_logits, labels)
            
            # 总损失
            loss = self.config['alpha'] * distillation_loss + (1 - self.config['alpha']) * hard_loss
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(student_logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        # 计算指标
        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_samples
        
        # 更新性能跟踪
        self.performance_tracker.add_metrics({'loss': avg_loss, 'accuracy': accuracy})
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'metrics': self.performance_tracker.get_current_metrics()
        }

class OnlineLearner:
    """在线学习器实现"""
    
    def __init__(self, model, config: Dict = None):
        self.model = model
        
        # 默认配置
        self.config = {
            'learning_rate': 0.0001, # 较低的学习率以避免过拟合
            'batch_size': 1,         # 在线学习通常使用小批量或单样本
            'momentum': 0.9,         # 动量参数
            'weight_decay': 1e-4,    # 权重衰减
            'adaptation_threshold': 0.6, # 适应新数据的阈值
            'forgetting_protection': True, # 防止遗忘机制
            'buffer_size': 1000      # 用于存储重要样本的缓冲区大小
        }
        
        if config:
            self.config.update(config)
        
        # 优化器
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay']
        )
        
        # 重要样本缓冲区
        self.important_samples = deque(maxlen=self.config['buffer_size'])
        
        # 学习状态
        self.running = False
        self.learning_thread = None
        
        # 性能跟踪
        self.performance_tracker = ModelPerformanceTracker()
        
    def start_online_learning(self):
        """启动在线学习线程"""
        if not self.running:
            self.running = True
            self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
            self.learning_thread.start()
            logger.info("在线学习已启动")
    
    def stop_online_learning(self):
        """停止在线学习线程"""
        self.running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5.0)
            logger.info("在线学习已停止")
    
    def _learning_loop(self):
        """在线学习主循环"""
        while self.running:
            # 检查是否有新数据可用
            # 注意：实际应用中需要实现数据队列或流接口
            time.sleep(1.0)  # 避免CPU占用过高
    
    def learn_from_instance(self, x, y, confidence_threshold: float = 0.8):
        """从单个实例中学习"""
        # 确保数据是张量
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.Tensor(y)
        
        # 前向传播
        output = self.model(x)
        
        # 计算置信度和损失
        probs = torch.nn.functional.softmax(output, dim=1) if output.dim() > 1 else torch.sigmoid(output)
        confidence, predicted = torch.max(probs, 1) if output.dim() > 1 else (probs.item(), probs.round().item())
        
        # 如果预测置信度低，认为这是一个需要学习的新样本
        if confidence < self.config['adaptation_threshold']:
            # 计算损失
            loss = torch.nn.MSELoss()(output, y) if output.dim() == 1 else torch.nn.CrossEntropyLoss()(output, y)
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 存储重要样本以防止遗忘
            if self.config['forgetting_protection'] and confidence < confidence_threshold:
                self.important_samples.append((x.detach(), y.detach()))
            
            # 定期重放重要样本
            if self.config['forgetting_protection'] and len(self.important_samples) > 0 and random.random() < 0.1:
                self._replay_important_samples()
            
            # 更新性能跟踪
            self.performance_tracker.add_metrics({
                'loss': loss.item(),
                'confidence': confidence.item()
            })
            
            return {
                'success': True,
                'loss': loss.item(),
                'confidence': confidence.item(),
                'metrics': self.performance_tracker.get_current_metrics()
            }
        
        return {
            'success': False,
            'reason': 'Confidence above threshold',
            'confidence': confidence.item()
        }
    
    def _replay_important_samples(self):
        """重放重要样本以防止遗忘"""
        if not self.important_samples:
            return
        
        # 随机选择一些重要样本
        num_samples = min(5, len(self.important_samples))
        samples = random.sample(self.important_samples, num_samples)
        
        # 重放这些样本
        for x, y in samples:
            output = self.model(x)
            loss = torch.nn.MSELoss()(output, y) if output.dim() == 1 else torch.nn.CrossEntropyLoss()(output, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class TransferLearningManager:
    """迁移学习管理器"""
    
    def __init__(self, source_model, config: Dict = None):
        self.source_model = source_model
        
        # 默认配置
        self.config = {
            'freeze_base': True,    # 是否冻结基础层
            'learning_rate': 0.001, # 学习率
            'lr_scale': 0.1,        # 学习率缩放因子
            'batch_size': 32,       # 批次大小
            'epochs': 5,            # 训练轮数
            'early_stopping': True, # 早停机制
            'patience': 3           # 早停耐心值
        }
        
        if config:
            self.config.update(config)
        
        # 性能跟踪
        self.performance_tracker = ModelPerformanceTracker()
        
    def prepare_target_model(self, target_task: str, num_classes: int = None) -> torch.nn.Module:
        """准备目标任务模型"""
        # 创建源模型的副本
        target_model = copy.deepcopy(self.source_model)
        
        # 根据目标任务调整输出层
        # 注意：实际应用中需要根据具体模型结构实现
        if num_classes is not None:
            # 假设最后一层是全连接层
            # 这里简化实现，实际应该根据具体模型结构调整
            pass
        
        # 冻结基础层
        if self.config['freeze_base']:
            for param in target_model.parameters()[:-2]:  # 冻结除最后两层外的所有层
                param.requires_grad = False
        
        return target_model
    
    def transfer_learn(self, target_model, train_loader, val_loader=None):
        """执行迁移学习"""
        # 优化器只更新可训练的参数
        trainable_params = filter(lambda p: p.requires_grad, target_model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.config['learning_rate'])
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            target_model.train()
            train_loss = 0
            train_correct = 0
            train_samples = 0
            
            for batch in train_loader:
                inputs, labels = batch
                
                optimizer.zero_grad()
                outputs = target_model(inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
                train_samples += labels.size(0)
            
            # 验证阶段
            val_loss = 0
            val_correct = 0
            val_samples = 0
            
            if val_loader:
                target_model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, labels = batch
                        outputs = target_model(inputs)
                        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_samples += labels.size(0)
            
            # 计算指标
            train_acc = train_correct / train_samples
            avg_train_loss = train_loss / len(train_loader)
            
            metrics = {
                'loss': avg_train_loss,
                'accuracy': train_acc
            }
            
            if val_loader:
                val_acc = val_correct / val_samples
                avg_val_loss = val_loss / len(val_loader)
                metrics['val_loss'] = avg_val_loss
                metrics['val_accuracy'] = val_acc
                
                # 学习率调度
                scheduler.step(avg_val_loss)
                
                # 早停机制
                if self.config['early_stopping'] and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['patience']:
                        logger.info(f"早停机制触发，在第{epoch+1}轮停止训练")
                        break
            
            # 更新性能跟踪
            self.performance_tracker.add_metrics(metrics)
            
        return {
            'target_model': target_model,
            'metrics': self.performance_tracker.get_current_metrics()
        }

class EnhancedLearningSystem:
    """增强学习系统"""
    
    def __init__(self, models: Dict[str, Any] = None, config_path: str = None):
        # 模型注册表
        self.models = models or {}
        
        # 加载配置
        self.config = {
            'reinforcement_learning': {
                'enabled': True,
                'gamma': 0.99,
                'epsilon': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'learning_rate': 0.001
            },
            'meta_learning': {
                'enabled': True,
                'meta_lr': 0.001,
                'inner_lr': 0.01,
                'k_shot': 5
            },
            'knowledge_distillation': {
                'enabled': True,
                'temperature': 1.0,
                'alpha': 0.7,
                'learning_rate': 0.001
            },
            'online_learning': {
                'enabled': True,
                'learning_rate': 0.0001,
                'adaptation_threshold': 0.6,
                'forgetting_protection': True
            },
            'transfer_learning': {
                'enabled': True,
                'freeze_base': True,
                'learning_rate': 0.001,
                'epochs': 5
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            except Exception as e:
                logger.error(f"加载配置文件失败: {str(e)}")
        
        # 初始化各个学习组件
        self.rl_agents = {}
        self.meta_learners = {}
        self.kd_instances = {}
        self.online_learners = {}
        self.transfer_managers = {}
        
        # 环境和奖励函数注册表
        self.environments = {}
        self.reward_functions = {}
        
        # 全局性能跟踪
        self.global_performance = ModelPerformanceTracker()
        
        logger.info("增强学习系统初始化完成")
    
    def register_model(self, model_name: str, model):
        """注册模型"""
        self.models[model_name] = model
        logger.info(f"模型 '{model_name}' 已注册")
    
    def register_environment(self, env_name: str, env: LearningEnvironment):
        """注册强化学习环境"""
        self.environments[env_name] = env
        logger.info(f"环境 '{env_name}' 已注册")
    
    def register_reward_function(self, reward_name: str, reward_fn: RewardFunction):
        """注册奖励函数"""
        self.reward_functions[reward_name] = reward_fn
        logger.info(f"奖励函数 '{reward_name}' 已注册")
    
    def create_rl_agent(self, agent_name: str, model_name: str, env_name: str, reward_name: str, config: Dict = None):
        """创建强化学习智能体"""
        if model_name not in self.models:
            logger.error(f"模型 '{model_name}' 未注册")
            return None
        
        if env_name not in self.environments:
            logger.error(f"环境 '{env_name}' 未注册")
            return None
        
        if reward_name not in self.reward_functions:
            logger.error(f"奖励函数 '{reward_name}' 未注册")
            return None
        
        # 合并配置
        agent_config = self.config['reinforcement_learning'].copy()
        if config:
            agent_config.update(config)
        
        # 创建智能体
        agent = ReinforcementLearningAgent(
            self.models[model_name],
            self.environments[env_name],
            self.reward_functions[reward_name],
            agent_config
        )
        
        self.rl_agents[agent_name] = agent
        logger.info(f"强化学习智能体 '{agent_name}' 已创建")
        
        return agent
    
    def create_meta_learner(self, learner_name: str, model_name: str, config: Dict = None):
        """创建元学习器"""
        if model_name not in self.models:
            logger.error(f"模型 '{model_name}' 未注册")
            return None
        
        # 合并配置
        learner_config = self.config['meta_learning'].copy()
        if config:
            learner_config.update(config)
        
        # 创建元学习器
        meta_learner = MetaLearner(self.models[model_name], learner_config)
        
        self.meta_learners[learner_name] = meta_learner
        logger.info(f"元学习器 '{learner_name}' 已创建")
        
        return meta_learner
    
    def create_kd_instance(self, instance_name: str, teacher_model_name: str, student_model_name: str, config: Dict = None):
        """创建知识蒸馏实例"""
        if teacher_model_name not in self.models:
            logger.error(f"教师模型 '{teacher_model_name}' 未注册")
            return None
        
        if student_model_name not in self.models:
            logger.error(f"学生模型 '{student_model_name}' 未注册")
            return None
        
        # 合并配置
        kd_config = self.config['knowledge_distillation'].copy()
        if config:
            kd_config.update(config)
        
        # 创建知识蒸馏实例
        kd_instance = KnowledgeDistillation(
            self.models[teacher_model_name],
            self.models[student_model_name],
            kd_config
        )
        
        self.kd_instances[instance_name] = kd_instance
        logger.info(f"知识蒸馏实例 '{instance_name}' 已创建")
        
        return kd_instance
    
    def create_online_learner(self, learner_name: str, model_name: str, config: Dict = None):
        """创建在线学习器"""
        if model_name not in self.models:
            logger.error(f"模型 '{model_name}' 未注册")
            return None
        
        # 合并配置
        online_config = self.config['online_learning'].copy()
        if config:
            online_config.update(config)
        
        # 创建在线学习器
        online_learner = OnlineLearner(self.models[model_name], online_config)
        
        self.online_learners[learner_name] = online_learner
        logger.info(f"在线学习器 '{learner_name}' 已创建")
        
        return online_learner
    
    def create_transfer_manager(self, manager_name: str, source_model_name: str, config: Dict = None):
        """创建迁移学习管理器"""
        if source_model_name not in self.models:
            logger.error(f"源模型 '{source_model_name}' 未注册")
            return None
        
        # 合并配置
        transfer_config = self.config['transfer_learning'].copy()
        if config:
            transfer_config.update(config)
        
        # 创建迁移学习管理器
        transfer_manager = TransferLearningManager(self.models[source_model_name], transfer_config)
        
        self.transfer_managers[manager_name] = transfer_manager
        logger.info(f"迁移学习管理器 '{manager_name}' 已创建")
        
        return transfer_manager
    
    def enable_feature(self, feature_name: str):
        """启用特定学习功能"""
        if feature_name in self.config:
            self.config[feature_name]['enabled'] = True
            logger.info(f"功能 '{feature_name}' 已启用")
            
    @property
    def reinforcement_learning(self):
        """获取默认的强化学习智能体"""
        if 'default' in self.rl_agents:
            return self.rl_agents['default']
        # 如果默认智能体不存在，创建一个默认的
        if self.models and self.environments and self.reward_functions:
            # 使用第一个可用的模型、环境和奖励函数
            model_name = next(iter(self.models.keys()))
            env_name = next(iter(self.environments.keys()))
            reward_name = next(iter(self.reward_functions.keys()))
            return self.create_rl_agent('default', model_name, env_name, reward_name)
        logger.warning("没有可用的模型、环境或奖励函数来创建默认强化学习智能体")
        return None
        
    @property
    def meta_learning(self):
        """获取默认的元学习器"""
        if 'default' in self.meta_learners:
            return self.meta_learners['default']
        # 如果默认元学习器不存在，创建一个默认的
        if self.models:
            model_name = next(iter(self.models.keys()))
            return self.create_meta_learner('default', model_name)
        logger.warning("没有可用的模型来创建默认元学习器")
        return None
        
    @property
    def knowledge_distillation(self):
        """获取默认的知识蒸馏实例"""
        if 'default' in self.kd_instances:
            return self.kd_instances['default']
        # 如果默认知识蒸馏实例不存在，并且有至少两个模型，创建一个默认的
        if len(self.models) >= 2:
            model_names = list(self.models.keys())
            return self.create_kd_instance('default', model_names[0], model_names[1])
        logger.warning("没有足够的模型来创建默认知识蒸馏实例")
        return None
        
    @property
    def online_learning(self):
        """获取默认的在线学习器"""
        if 'default' in self.online_learners:
            return self.online_learners['default']
        # 如果默认在线学习器不存在，创建一个默认的
        if self.models:
            model_name = next(iter(self.models.keys()))
            return self.create_online_learner('default', model_name)
        logger.warning("没有可用的模型来创建默认在线学习器")
        return None
        
    @property
    def transfer_learning(self):
        """获取默认的迁移学习管理器"""
        if 'default' in self.transfer_managers:
            return self.transfer_managers['default']
        # 如果默认迁移学习管理器不存在，创建一个默认的
        if self.models:
            model_name = next(iter(self.models.keys()))
            return self.create_transfer_manager('default', model_name)
        logger.warning("没有可用的模型来创建默认迁移学习管理器")
        return None

    def disable_feature(self, feature_name: str):
        """禁用特定学习功能"""
        if feature_name in self.config:
            self.config[feature_name]['enabled'] = False
            logger.info(f"功能 '{feature_name}' 已禁用")
            return True
        
        logger.error(f"未知功能 '{feature_name}'")
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'config': self.config,
            'models': list(self.models.keys()),
            'environments': list(self.environments.keys()),
            'reward_functions': list(self.reward_functions.keys()),
            'rl_agents': list(self.rl_agents.keys()),
            'meta_learners': list(self.meta_learners.keys()),
            'kd_instances': list(self.kd_instances.keys()),
            'online_learners': list(self.online_learners.keys()),
            'transfer_managers': list(self.transfer_managers.keys()),
            'global_performance': self.global_performance.get_current_metrics(),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_config(self, config_path: str):
        """保存配置到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"配置已保存到 '{config_path}'")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")
            return False
    
    def load_config(self, config_path: str):
        """从文件加载配置"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                
                logger.info(f"配置已从 '{config_path}' 加载")
                return True
            else:
                logger.error(f"配置文件不存在: '{config_path}'")
                return False
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            return False

# 实现一些默认的环境和奖励函数

class DefaultSystemEnvironment(LearningEnvironment):
    """默认的系统环境实现"""
    
    def __init__(self, system_state_getter: Callable):
        self.system_state_getter = system_state_getter
        self.current_state = None
    
    def reset(self) -> Dict[str, Any]:
        """重置环境到初始状态"""
        self.current_state = self.system_state_getter()
        return self.current_state
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """执行动作并返回结果"""
        # 这里简化实现，实际应用中需要根据动作类型执行相应的操作
        # 并获取新的系统状态
        next_state = self.system_state_getter()
        
        # 假设环境总是活跃的
        done = False
        
        # 额外信息
        info = {'action': action}
        
        # 奖励将由奖励函数计算
        reward = 0
        
        self.current_state = next_state
        return next_state, reward, done, info
    
    def get_observation_space(self) -> Dict[str, Any]:
        """获取观测空间信息"""
        # 简化实现，返回离散的观测空间
        return {'discrete': 100}
    
    def get_action_space(self) -> Dict[str, Any]:
        """获取动作空间信息"""
        # 简化实现，返回离散的动作空间
        return {'discrete': 10}

class DefaultRewardFunction(RewardFunction):
    """默认的奖励函数实现"""
    
    def __init__(self, performance_metrics_getter: Callable):
        self.performance_metrics_getter = performance_metrics_getter
    
    def calculate_reward(self, state: Dict[str, Any], action: Any, next_state: Dict[str, Any], done: bool) -> float:
        """计算奖励值"""
        # 获取性能指标
        metrics = self.performance_metrics_getter()
        
        # 基于性能指标计算奖励
        # 这里简化实现，实际应用中需要根据具体任务定义奖励函数
        reward = 0
        
        # 例如，基于准确率的奖励
        if 'accuracy' in metrics:
            reward += metrics['accuracy'] * 10
        
        # 基于响应时间的惩罚
        if 'response_time' in metrics:
            reward -= metrics['response_time']
        
        # 基于损失的惩罚
        if 'loss' in metrics:
            reward -= metrics['loss'] * 5
        
        return reward

# 创建一个全局的增强学习系统实例
enhanced_learning_system = EnhancedLearningSystem()

# 导出主要组件供外部使用
def get_enhanced_learning_system():
    return enhanced_learning_system