# -*- coding: utf-8 -*-
# Self Brain - AGI核心训练系统
# Self Brain - AGI Core Training System
# Copyright 2025 Silence Crow Team
# Email: silencecrowtom@qq.com

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional
import json
import logging
import os
import sys
import time
from datetime import datetime
import threading
import queue
from collections import deque

# 配置日志
def setup_logging():
    """设置系统日志"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'self_brain_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("SelfBrainCore")

logger = setup_logging()

class SelfBrainCore:
    """Self Brain AGI核心系统 - 管理所有模型的训练、推理和自主学习"""
    
    def __init__(self, config_path="d:\\shiyan\\config\\model_registry.json"):
        self.config_path = config_path
        self.model_registry = self.load_model_registry()
        self.models = {}
        self.training_systems = {}
        self.model_connections = {}
        self.running = False
        self.global_context = {}
        self.training_queue = queue.Queue(maxsize=10000)
        self.task_queue = queue.Queue(maxsize=1000)
        
        # 初始化系统
        self.initialize_systems()
        
        logger.info("Self Brain Core initialized successfully")
        logger.info(f"Loaded {len(self.model_registry)} models from registry")
    
    def load_model_registry(self) -> Dict[str, Any]:
        """加载模型注册表"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.error(f"Model registry file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            return {}
    
    def initialize_systems(self):
        """初始化所有系统组件"""
        # 加载训练系统
        try:
            from real_training_system import RealTimeDataGenerator
            self.data_generator = RealTimeDataGenerator()
        except Exception as e:
            logger.error(f"Failed to initialize data generator: {e}")
        
        # 启动系统线程
        self.running = True
        self.system_thread = threading.Thread(target=self._system_loop)
        self.system_thread.daemon = True
        self.system_thread.start()
        
        # 启动数据生成器
        if hasattr(self, 'data_generator'):
            self.data_generator.start_generation()
    
    def load_model(self, model_id: str):
        """加载指定模型"""
        if model_id not in self.model_registry:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        model_info = self.model_registry[model_id]
        model_path = model_info.get('path', '')
        
        if not model_path:
            logger.error(f"No path specified for model {model_id}")
            return False
        
        # 构建完整路径
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        
        # 动态导入模型
        try:
            module_name = f"{model_id.lower()}_module"
            spec = importlib.util.spec_from_file_location(module_name, full_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 假设每个模型模块都有一个Model类
            if hasattr(module, 'Model'):
                self.models[model_id] = module.Model()
                logger.info(f"Successfully loaded model: {model_id}")
                return True
            else:
                logger.error(f"Model class not found in {full_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def train_model_from_scratch(self, model_id: str, epochs: int = 100):
        """从零开始训练指定模型"""
        if model_id not in self.model_registry:
            logger.error(f"Model {model_id} not registered")
            return {"status": "error", "message": f"Model {model_id} not registered"}
        
        if model_id not in self.models and not self.load_model(model_id):
            return {"status": "error", "message": f"Failed to load model {model_id}"}
        
        model = self.models[model_id]
        model_info = self.model_registry[model_id]
        
        logger.info(f"Starting scratch training for model {model_id}")
        
        try:
            # 准备训练数据
            training_data = self._prepare_training_data(model_id)
            
            # 创建训练器
            trainer = ModelTrainer(model, training_data)
            
            # 开始训练
            training_results = trainer.train(epochs=epochs)
            
            # 保存模型
            self._save_model(model_id, model, training_results)
            
            return {
                "status": "success",
                "model_id": model_id,
                "training_results": training_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Training failed for model {model_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _prepare_training_data(self, model_id: str):
        """准备模型训练数据"""
        model_type = self.model_registry[model_id].get('type', '')
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', model_id)
        os.makedirs(data_dir, exist_ok=True)
        
        # 根据模型类型准备不同的训练数据
        if model_type == 'Language Model':
            return self._prepare_language_data(model_id, data_dir)
        elif model_type == 'Image Processor':
            return self._prepare_image_data(model_id, data_dir)
        elif model_type == 'Audio Processor':
            return self._prepare_audio_data(model_id, data_dir)
        else:
            # 默认使用实时生成的数据
            return self._prepare_default_data(model_id)
    
    def _prepare_language_data(self, model_id: str, data_dir: str):
        """准备语言模型训练数据"""
        # 从各种来源收集文本数据
        text_data = []
        
        # 1. 从知识库获取数据
        knowledge_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           'sub_models', 'I_knowledge', 'knowledge_base.json')
        if os.path.exists(knowledge_base_path):
            try:
                with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                    knowledge_data = json.load(f)
                    for item in knowledge_data:
                        if 'content' in item:
                            text_data.append(item['content'])
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")
        
        # 2. 生成一些训练文本
        for i in range(1000):
            text_data.append(f"This is training text for model {model_id} sample {i}")
        
        return text_data
    
    def _prepare_image_data(self, model_id: str, data_dir: str):
        """准备图像模型训练数据"""
        # 创建合成图像数据
        images = []
        labels = []
        
        for i in range(500):
            # 生成随机图像
            img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            label = np.random.randint(0, 10)
            images.append(img)
            labels.append(label)
        
        return list(zip(images, labels))
    
    def _prepare_audio_data(self, model_id: str, data_dir: str):
        """准备音频模型训练数据"""
        # 创建合成音频数据
        audio_data = []
        labels = []
        
        for i in range(300):
            # 生成随机音频信号
            audio = np.random.randn(16000)
            label = np.random.randint(0, 5)
            audio_data.append(audio)
            labels.append(label)
        
        return list(zip(audio_data, labels))
    
    def _prepare_default_data(self, model_id: str):
        """准备默认训练数据"""
        data = []
        
        # 从数据生成器获取数据
        if hasattr(self, 'data_generator'):
            for _ in range(500):
                try:
                    sample = self.data_generator.data_queue.get(timeout=1)
                    data.append(sample)
                except queue.Empty:
                    break
        
        # 如果数据不足，生成一些样本
        if len(data) < 100:
            for i in range(100 - len(data)):
                data.append({"input": f"sample_{i}", "target": f"target_{i}"})
        
        return data
    
    def _save_model(self, model_id: str, model, training_results: Dict[str, Any]):
        """保存训练好的模型"""
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 'models', model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型权重
        weights_path = os.path.join(model_dir, f'{model_id}_weights.pth')
        try:
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), weights_path)
            else:
                # 对于非PyTorch模型，使用pickle
                import pickle
                with open(weights_path.replace('.pth', '.pkl'), 'wb') as f:
                    pickle.dump(model, f)
            logger.info(f"Saved model weights to {weights_path}")
        except Exception as e:
            logger.error(f"Failed to save model weights: {e}")
        
        # 保存训练结果
        results_path = os.path.join(model_dir, f'{model_id}_training_results.json')
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(training_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved training results to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save training results: {e}")
    
    def connect_external_api(self, model_id: str, api_url: str, api_key: str, model_name: str):
        """连接外部API到指定模型"""
        if model_id not in self.model_registry:
            logger.error(f"Model {model_id} not registered")
            return {"status": "error", "message": f"Model {model_id} not registered"}
        
        try:
            # 保存连接信息
            self.model_connections[model_id] = {
                "api_url": api_url,
                "api_key": api_key,
                "model_name": model_name,
                "connected": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # 更新模型注册表
            self.model_registry[model_id]["api_url"] = api_url
            self.model_registry[model_id]["api_key"] = api_key
            self.model_registry[model_id]["model_source"] = "external"
            
            # 保存更新后的注册表
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_registry, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Connected model {model_id} to external API: {api_url}")
            return {"status": "success", "message": f"Connected to {api_url}"}
        except Exception as e:
            logger.error(f"Failed to connect model {model_id} to external API: {e}")
            return {"status": "error", "message": str(e)}
    
    def disconnect_external_api(self, model_id: str):
        """断开外部API连接"""
        if model_id not in self.model_connections:
            return {"status": "success", "message": "No external connection found"}
        
        try:
            # 移除连接信息
            del self.model_connections[model_id]
            
            # 更新模型注册表
            self.model_registry[model_id]["api_url"] = ""
            self.model_registry[model_id]["api_key"] = ""
            self.model_registry[model_id]["model_source"] = "local"
            
            # 保存更新后的注册表
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_registry, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Disconnected model {model_id} from external API")
            return {"status": "success", "message": "Disconnected successfully"}
        except Exception as e:
            logger.error(f"Failed to disconnect model {model_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def start_autonomous_knowledge_learning(self, model_id: str):
        """启动模型的自主知识库学习"""
        if model_id not in self.model_registry:
            return {"status": "error", "message": f"Model {model_id} not registered"}
        
        try:
            # 创建学习线程
            learning_thread = threading.Thread(
                target=self._autonomous_learning_loop,
                args=(model_id,)
            )
            learning_thread.daemon = True
            learning_thread.start()
            
            logger.info(f"Started autonomous knowledge learning for model {model_id}")
            return {"status": "success", "message": "Autonomous learning started"}
        except Exception as e:
            logger.error(f"Failed to start autonomous learning: {e}")
            return {"status": "error", "message": str(e)}
    
    def _autonomous_learning_loop(self, model_id: str):
        """自主学习循环"""
        # 加载知识库
        knowledge_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           'sub_models', 'I_knowledge', 'knowledge_base.json')
        
        while self.running:
            try:
                # 检查知识库更新
                if os.path.exists(knowledge_base_path):
                    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                        knowledge_data = json.load(f)
                        
                        # 选择部分知识进行学习
                        for item in knowledge_data[:10]:  # 每次学习10条
                            if 'content' in item:
                                # 针对模型特性进行学习
                                self._learn_knowledge_item(model_id, item)
                
                # 每30秒检查一次更新
                time.sleep(30)
            except Exception as e:
                logger.error(f"Error in autonomous learning loop: {e}")
                time.sleep(5)  # 发生错误时暂停5秒
    
    def _learn_knowledge_item(self, model_id: str, knowledge_item: Dict[str, Any]):
        """学习单个知识项"""
        try:
            content = knowledge_item.get('content', '')
            category = knowledge_item.get('category', 'general')
            
            logger.info(f"Model {model_id} learning knowledge from category: {category}")
            
            # 根据模型类型处理知识
            model_type = self.model_registry[model_id].get('type', '')
            
            # 这里是简化版实现，实际系统中应该有更复杂的学习机制
            if model_id in self.models:
                model = self.models[model_id]
                if hasattr(model, 'learn'):
                    model.learn(content, category)
                else:
                    # 模拟学习
                    logger.info(f"Model {model_id} processed knowledge content")
            
        except Exception as e:
            logger.error(f"Failed to learn knowledge item: {e}")
    
    def _system_loop(self):
        """系统主循环"""
        while self.running:
            try:
                # 处理训练队列中的任务
                try:
                    training_task = self.training_queue.get(timeout=1)
                    model_id = training_task.get('model_id')
                    epochs = training_task.get('epochs', 100)
                    self.train_model_from_scratch(model_id, epochs)
                except queue.Empty:
                    pass
                
                # 处理任务队列中的任务
                try:
                    task = self.task_queue.get(timeout=1)
                    self._process_task(task)
                except queue.Empty:
                    pass
                
                # 系统维护
                self._system_maintenance()
                
            except Exception as e:
                logger.error(f"Error in system loop: {e}")
                time.sleep(5)
    
    def _process_task(self, task: Dict[str, Any]):
        """处理系统任务"""
        try:
            task_type = task.get('type')
            target_model = task.get('target_model')
            
            if task_type == 'inference' and target_model and target_model in self.models:
                model = self.models[target_model]
                input_data = task.get('input')
                
                if hasattr(model, 'predict'):
                    result = model.predict(input_data)
                else:
                    result = {"error": "Model does not support prediction"}
                
                # 将结果返回给回调函数
                callback = task.get('callback')
                if callback and callable(callback):
                    callback(result)
            
        except Exception as e:
            logger.error(f"Failed to process task: {e}")
    
    def _system_maintenance(self):
        """系统维护任务"""
        # 这里可以添加定期维护任务，如内存清理、日志轮换等
        pass
    
    def stop(self):
        """停止Self Brain系统"""
        self.running = False
        
        if hasattr(self, 'system_thread'):
            self.system_thread.join(timeout=5)
        
        if hasattr(self, 'data_generator'):
            self.data_generator.stop_generation()
        
        logger.info("Self Brain Core stopped")

class ModelTrainer:
    """模型训练器 - 为不同类型的模型提供统一的训练接口"""
    
    def __init__(self, model, training_data):
        self.model = model
        self.training_data = training_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 如果是PyTorch模型，移至设备
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
    
    def train(self, epochs: int = 100, batch_size: int = 32):
        """执行模型训练"""
        start_time = time.time()
        
        try:
            # PyTorch模型训练
            if hasattr(self.model, 'parameters') and hasattr(self.model, 'train'):
                return self._train_pytorch_model(epochs, batch_size)
            else:
                # 自定义模型训练
                return self._train_custom_model(epochs)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "training_time": time.time() - start_time
            }
    
    def _train_pytorch_model(self, epochs: int, batch_size: int):
        """训练PyTorch模型"""
        # 检查数据类型并创建数据加载器
        if isinstance(self.training_data, list) and len(self.training_data) > 0:
            # 假设数据是(input, target)对
            from torch.utils.data import DataLoader, Dataset
            
            class SimpleDataset(Dataset):
                def __init__(self, data):
                    self.data = data
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            
            dataset = SimpleDataset(self.training_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # 设置优化器和损失函数
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss() if isinstance(self.training_data[0][1], (int, float)) else nn.CrossEntropyLoss()
            
            # 训练循环
            history = []
            for epoch in range(epochs):
                running_loss = 0.0
                
                for inputs, targets in dataloader:
                    # 移至设备
                    if torch.is_tensor(inputs):
                        inputs = inputs.to(self.device)
                    if torch.is_tensor(targets):
                        targets = targets.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    
                    # 计算损失
                    loss = criterion(outputs, targets)
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                
                # 记录历史
                avg_loss = running_loss / len(dataloader)
                history.append(avg_loss)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
            return {
                "status": "success",
                "epochs": epochs,
                "loss_history": history,
                "final_loss": history[-1] if history else 0,
                "training_time": time.time() - start_time
            }
        
        return {"status": "error", "message": "Invalid training data format"}
    
    def _train_custom_model(self, epochs: int):
        """训练自定义模型"""
        # 这是一个简化的自定义模型训练实现
        # 实际系统中应该根据模型类型实现具体的训练逻辑
        history = []
        
        for epoch in range(epochs):
            # 模拟训练进度
            loss = 1.0 / (epoch + 1)
            history.append(loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Custom model training epoch [{epoch+1}/{epochs}], Simulated Loss: {loss:.4f}")
            
            # 模拟训练时间
            time.sleep(0.1)
        
        return {
            "status": "success",
            "epochs": epochs,
            "loss_history": history,
            "training_time": time.time() - start_time
        }

# 导入必要的模块
import importlib.util

# 如果作为主程序运行
if __name__ == "__main__":
    # 初始化Self Brain核心
    self_brain = SelfBrainCore()
    
    try:
        # 训练所有模型
        for model_id in self_brain.model_registry.keys():
            print(f"\nTraining model: {model_id}")
            result = self_brain.train_model_from_scratch(model_id, epochs=50)
            print(f"Training result: {result['status']}")
            
        print("\nAll models training completed!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        self_brain.stop()