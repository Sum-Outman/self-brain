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
知识库自主学习模块
实现各模型对知识库内容的自主学习功能
"""

import os
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('KnowledgeAutoLearning')

class LearningStatus(Enum):
    IDLE = "idle"
    LEARNING = "learning"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"

class LearningDomain(Enum):
    PHYSICS = "physics"
    MATHEMATICS = "mathematics"
    CHEMISTRY = "chemistry"
    MEDICINE = "medicine"
    LAW = "law"
    HISTORY = "history"
    SOCIOLOGY = "sociology"
    PSYCHOLOGY = "psychology"
    ECONOMICS = "economics"
    MECHANICAL_ENGINEERING = "mechanical_engineering"
    ELECTRICAL_ENGINEERING = "electrical_engineering"
    CIVIL_ENGINEERING = "civil_engineering"
    COMPUTER_SCIENCE = "computer_science"
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    ROBOTICS = "robotics"
    OTHER = "other"

class KnowledgeAutoLearning:
    """
    知识库自主学习系统
    允许各模型针对自己的功能特性对知识库内容进行自主学习
    """
    def __init__(self, knowledge_base, model_id: str, config: Optional[Dict] = None):
        """
        初始化自主学习系统
        
        参数:
            knowledge_base: 知识库实例
            model_id: 模型ID
            config: 学习配置
        """
        self.knowledge_base = knowledge_base
        self.model_id = model_id
        
        # 默认配置
        default_config = {
            'learning_rate': 0.01,  # 学习率
            'batch_size': 32,       # 批次大小
            'epochs': 10,           # 训练轮数
            'max_concurrent_domains': 3,  # 最大并发学习领域
            'learning_interval': 3600,    # 学习间隔(秒)
            'save_interval': 600,    # 保存间隔(秒)
            'progress_threshold': 0.01,   # 进度阈值
            'performance_threshold': 0.8,  # 性能阈值
            'max_retries': 3         # 最大重试次数
        }
        
        # 合并配置
        self.config = {**default_config, **(config or {})}
        
        # 学习状态
        self.status = LearningStatus.IDLE
        self.current_domains = []
        self.learning_progress = {}
        self.learning_history = []
        self.performance_metrics = {}
        
        # 学习线程
        self.learning_thread = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        # 学习回调
        self.progress_callback = None
        self.completion_callback = None
        
        # 模型特定学习配置
        self.model_specific_domains = self._get_model_specific_domains()
        
        logger.info(f"自主学习系统初始化成功 - 模型ID: {self.model_id}")
        
    def _get_model_specific_domains(self) -> List[LearningDomain]:
        """\获取模型特定的学习领域"""
        domain_mapping = {
            'A_management': [LearningDomain.COMPUTER_SCIENCE, LearningDomain.PSYCHOLOGY, LearningDomain.ARTIFICIAL_INTELLIGENCE],
            'B_language': [LearningDomain.LAW, LearningDomain.HISTORY, LearningDomain.SOCIOLOGY],
            'C_audio': [LearningDomain.COMPUTER_SCIENCE],
            'D_image': [LearningDomain.COMPUTER_SCIENCE, LearningDomain.ARTIFICIAL_INTELLIGENCE],
            'E_video': [LearningDomain.COMPUTER_SCIENCE, LearningDomain.ARTIFICIAL_INTELLIGENCE],
            'F_spatial': [LearningDomain.PHYSICS, LearningDomain.MATHEMATICS, LearningDomain.ROBOTICS],
            'G_sensor': [LearningDomain.PHYSICS, LearningDomain.MECHANICAL_ENGINEERING],
            'H_computer_control': [LearningDomain.COMPUTER_SCIENCE, LearningDomain.ELECTRICAL_ENGINEERING],
            'J_motion': [LearningDomain.PHYSICS, LearningDomain.MECHANICAL_ENGINEERING, LearningDomain.ROBOTICS],
            'K_programming': [LearningDomain.COMPUTER_SCIENCE, LearningDomain.ARTIFICIAL_INTELLIGENCE]
        }
        
        return domain_mapping.get(self.model_id, [LearningDomain.OTHER])
        
    def start_learning(self, domains: Optional[List[LearningDomain]] = None):
        """
        开始自主学习
        
        参数:
            domains: 学习领域列表，None表示使用模型特定领域
        """
        if self.status == LearningStatus.LEARNING:
            logger.warning("自主学习已经在进行中")
            return False
            
        # 设置学习领域
        if domains is None:
            domains = self.model_specific_domains
        
        # 检查领域数量
        if len(domains) > self.config['max_concurrent_domains']:
            domains = domains[:self.config['max_concurrent_domains']]
            logger.info(f"学习领域数量超过限制，已截取前{self.config['max_concurrent_domains']}个领域")
            
        # 更新状态
        self.status = LearningStatus.LEARNING
        self.current_domains = domains
        self.stop_event.clear()
        self.pause_event.clear()
        
        # 初始化进度
        for domain in domains:
            self.learning_progress[domain.value] = 0.0
            
        # 启动学习线程
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        
        logger.info(f"自主学习已启动 - 模型ID: {self.model_id}, 领域: {[d.value for d in domains]}")
        return True
        
    def pause_learning(self):
        """暂停自主学习"""
        if self.status == LearningStatus.LEARNING:
            self.status = LearningStatus.PAUSED
            self.pause_event.set()
            logger.info(f"自主学习已暂停 - 模型ID: {self.model_id}")
            return True
        return False
        
    def resume_learning(self):
        """恢复自主学习"""
        if self.status == LearningStatus.PAUSED:
            self.status = LearningStatus.LEARNING
            self.pause_event.clear()
            logger.info(f"自主学习已恢复 - 模型ID: {self.model_id}")
            return True
        return False
        
    def stop_learning(self):
        """停止自主学习"""
        if self.status in [LearningStatus.LEARNING, LearningStatus.PAUSED]:
            self.status = LearningStatus.IDLE
            self.stop_event.set()
            
            if self.learning_thread:
                self.learning_thread.join(timeout=5.0)
                
            logger.info(f"自主学习已停止 - 模型ID: {self.model_id}")
            return True
        return False
        
    def _learning_loop(self):
        """自主学习循环"""
        last_save_time = time.time()
        
        try:
            while not self.stop_event.is_set():
                # 检查是否暂停
                if self.pause_event.is_set():
                    time.sleep(1.0)
                    continue
                    
                # 学习逻辑
                for domain in self.current_domains:
                    if self.stop_event.is_set():
                        break
                        
                    try:
                        # 执行领域学习
                        progress_increment = self._learn_domain(domain)
                        
                        # 更新进度
                        current_progress = self.learning_progress.get(domain.value, 0.0)
                        new_progress = min(current_progress + progress_increment, 1.0)
                        self.learning_progress[domain.value] = new_progress
                        
                        # 通知进度
                        if self.progress_callback and progress_increment >= self.config['progress_threshold']:
                            self.progress_callback({
                                'model_id': self.model_id,
                                'domain': domain.value,
                                'progress': new_progress
                            })
                            
                        # 检查是否完成
                        if new_progress >= 1.0:
                            # 评估学习效果
                            performance = self._evaluate_learning(domain)
                            self.performance_metrics[domain.value] = performance
                            
                            # 记录历史
                            self._record_learning_history(domain, performance)
                            
                            logger.info(f"领域学习完成 - 模型ID: {self.model_id}, 领域: {domain.value}, 性能: {performance:.2f}")
                            
                    except Exception as e:
                        logger.error(f"领域学习出错 - 模型ID: {self.model_id}, 领域: {domain.value}, 错误: {str(e)}")
                        
                # 保存进度
                current_time = time.time()
                if current_time - last_save_time >= self.config['save_interval']:
                    self._save_learning_progress()
                    last_save_time = current_time
                    
                # 检查是否所有领域都已完成
                if all(progress >= 1.0 for progress in self.learning_progress.values()):
                    self.status = LearningStatus.COMPLETED
                    
                    # 保存最终结果
                    self._save_learning_progress()
                    
                    # 通知完成
                    if self.completion_callback:
                        self.completion_callback({
                            'model_id': self.model_id,
                            'status': 'completed',
                            'metrics': self.performance_metrics
                        })
                        
                    logger.info(f"所有领域学习完成 - 模型ID: {self.model_id}")
                    break
                    
                # 间隔等待
                time.sleep(self.config['learning_interval'])
                
        except Exception as e:
            self.status = LearningStatus.ERROR
            logger.error(f"自主学习循环出错 - 模型ID: {self.model_id}, 错误: {str(e)}")
            
    def _learn_domain(self, domain: LearningDomain) -> float:
        """
        学习特定领域的知识
        
        参数:
            domain: 学习领域
        
        返回:
            学习进度增量 (0.0-1.0)
        """
        logger.info(f"开始学习领域 - 模型ID: {self.model_id}, 领域: {domain.value}")
        
        # 1. 从知识库检索领域相关知识
        domain_knowledge = self._retrieve_domain_knowledge(domain)
        
        if not domain_knowledge:
            logger.warning(f"未找到领域相关知识 - 模型ID: {self.model_id}, 领域: {domain.value}")
            return 0.0
            
        # 2. 处理和理解知识
        processed_knowledge = self._process_knowledge(domain_knowledge)
        
        # 3. 应用学习到的知识
        self._apply_learned_knowledge(processed_knowledge)
        
        # 4. 返回学习进度增量
        # 这里是简化实现，实际应根据学习内容和难度计算进度
        progress_increment = 1.0 / self.config['epochs']
        
        return progress_increment
        
    def _retrieve_domain_knowledge(self, domain: LearningDomain) -> List[Dict[str, Any]]:
        """从知识库检索领域相关知识"""
        try:
            # 从知识库获取相关知识条目
            # 这里调用知识库API，实际实现应根据知识库的具体接口调整
            knowledge_items = self.knowledge_base.retrieve_knowledge_by_domain(domain.value)
            return knowledge_items
        except Exception as e:
            logger.error(f"检索领域知识出错 - 模型ID: {self.model_id}, 领域: {domain.value}, 错误: {str(e)}")
            return []
            
    def _process_knowledge(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理和理解知识"""
        processed_items = []
        
        for item in knowledge_items:
            try:
                # 处理知识条目
                processed_item = {
                    'id': item.get('id', ''),
                    'title': item.get('title', ''),
                    'content': item.get('content', ''),
                    'relevance': self._calculate_relevance(item),
                    'timestamp': datetime.now().isoformat()
                }
                
                processed_items.append(processed_item)
                
            except Exception as e:
                logger.error(f"处理知识条目出错 - 模型ID: {self.model_id}, 错误: {str(e)}")
                
        # 按相关性排序
        processed_items.sort(key=lambda x: x['relevance'], reverse=True)
        
        return processed_items
        
    def _calculate_relevance(self, knowledge_item: Dict[str, Any]) -> float:
        """计算知识条目的相关性"""
        # 简化实现，实际应根据模型类型和需求进行更复杂的相关性计算
        return 0.5 + 0.5 * min(1.0, len(knowledge_item.get('content', '')) / 1000)
        
    def _apply_learned_knowledge(self, processed_knowledge: List[Dict[str, Any]]):
        """应用学习到的知识"""
        # 这里是简化实现，实际应根据模型类型和需求实现具体的知识应用逻辑
        # 例如，更新模型参数、改进决策逻辑、优化性能等
        logger.info(f"应用学习到的知识 - 模型ID: {self.model_id}, 知识条目数: {len(processed_knowledge)}")
        
    def _evaluate_learning(self, domain: LearningDomain) -> float:
        """评估学习效果"""
        # 简化实现，实际应根据模型类型和需求实现具体的评估逻辑
        # 返回0.0-1.0之间的性能分数
        return 0.9 + 0.1 * (1.0 - self.learning_progress.get(domain.value, 0.0))
        
    def _record_learning_history(self, domain: LearningDomain, performance: float):
        """记录学习历史"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'domain': domain.value,
            'performance': performance,
            'duration': time.time() - self.learning_history[-1]['timestamp'] if self.learning_history else 0
        }
        
        self.learning_history.append(history_entry)
        
        # 限制历史记录数量
        if len(self.learning_history) > 1000:
            self.learning_history.pop(0)
            
    def _save_learning_progress(self):
        """保存学习进度"""
        try:
            progress_data = {
                'model_id': self.model_id,
                'status': self.status.value,
                'progress': self.learning_progress,
                'metrics': self.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存到文件或数据库
            # 这里简化实现，实际应根据系统架构调整
            progress_dir = os.path.join('learning_progress', self.model_id)
            os.makedirs(progress_dir, exist_ok=True)
            
            progress_file = os.path.join(progress_dir, f'progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"学习进度已保存 - 模型ID: {self.model_id}, 文件: {progress_file}")
            
        except Exception as e:
            logger.error(f"保存学习进度出错 - 模型ID: {self.model_id}, 错误: {str(e)}")
            
    def get_status(self) -> Dict[str, Any]:
        """获取自主学习状态"""
        return {
            'model_id': self.model_id,
            'status': self.status.value,
            'current_domains': [d.value for d in self.current_domains],
            'progress': self.learning_progress,
            'metrics': self.performance_metrics
        }
        
    def set_progress_callback(self, callback: Callable):
        """设置进度回调函数"""
        self.progress_callback = callback
        
    def set_completion_callback(self, callback: Callable):
        """设置完成回调函数"""
        self.completion_callback = callback
        
# 创建全局自主学习管理器
class AutoLearningManager:
    """\自主学习管理器，管理所有模型的自主学习过程"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AutoLearningManager, cls).__new__(cls)
            cls._instance.learning_systems = {}
            cls._instance.lock = threading.Lock()
        return cls._instance
        
    def register_model(self, model_id: str, knowledge_base, config: Optional[Dict] = None) -> KnowledgeAutoLearning:
        """
        注册模型到自主学习管理器
        
        参数:
            model_id: 模型ID
            knowledge_base: 知识库实例
            config: 学习配置
        
        返回:
            知识自主学习实例
        """
        with self.lock:
            if model_id not in self.learning_systems:
                self.learning_systems[model_id] = KnowledgeAutoLearning(knowledge_base, model_id, config)
                logger.info(f"模型已注册到自主学习管理器 - 模型ID: {model_id}")
            
            return self.learning_systems[model_id]
            
    def start_learning(self, model_id: str = None, domains: Optional[List[LearningDomain]] = None) -> bool:
        """
        启动模型的自主学习
        
        参数:
            model_id: 模型ID，None表示启动所有模型
            domains: 学习领域列表
        
        返回:
            是否成功启动
        """
        with self.lock:
            if model_id is None:
                # 启动所有模型
                results = []
                for mid, system in self.learning_systems.items():
                    result = system.start_learning(domains)
                    results.append(result)
                return all(results)
                
            elif model_id in self.learning_systems:
                # 启动特定模型
                return self.learning_systems[model_id].start_learning(domains)
                
            else:
                logger.error(f"模型未注册 - 模型ID: {model_id}")
                return False
                
    def stop_learning(self, model_id: str = None) -> bool:
        """
        停止模型的自主学习
        
        参数:
            model_id: 模型ID，None表示停止所有模型
        
        返回:
            是否成功停止
        """
        with self.lock:
            if model_id is None:
                # 停止所有模型
                results = []
                for mid, system in self.learning_systems.items():
                    result = system.stop_learning()
                    results.append(result)
                return all(results)
                
            elif model_id in self.learning_systems:
                # 停止特定模型
                return self.learning_systems[model_id].stop_learning()
                
            else:
                logger.error(f"模型未注册 - 模型ID: {model_id}")
                return False
                
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """\获取所有模型的自主学习状态"""
        statuses = {}
        with self.lock:
            for model_id, system in self.learning_systems.items():
                statuses[model_id] = system.get_status()
        return statuses
        
    def get_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """\获取特定模型的自主学习状态"""
        with self.lock:
            if model_id in self.learning_systems:
                return self.learning_systems[model_id].get_status()
            else:
                logger.error(f"模型未注册 - 模型ID: {model_id}")
                return None

# 创建全局实例
auto_learning_manager = AutoLearningManager()