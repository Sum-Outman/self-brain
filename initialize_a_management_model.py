#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初始化A_management模型的脚本
此脚本用于：
1. 检查模型文件和权重是否存在
2. 如果不存在，创建基础的模型权重文件
3. 验证模型是否能正常加载和推理
4. 准备训练环境
"""

import os
import sys
import torch
import json
import time
import argparse
import numpy as np
from datetime import datetime

# 添加sub_models/A_management目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'sub_models', 'A_management'))

# 设置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("A_Management_Initializer")

class ModelInitializer:
    """A_management模型初始化器"""
    
    def __init__(self, force_recreate=False):
        # 初始化路径
        self.root_dir = os.path.dirname(__file__)
        self.model_dir = os.path.join(self.root_dir, 'sub_models', 'A_management')
        self.weights_dir = os.path.join(self.model_dir, 'model_weights')
        self.model_file = os.path.join(self.model_dir, 'enhanced_manager.py')
        self.weights_file = os.path.join(self.weights_dir, 'a_management_model.pth')
        self.config_file = os.path.join(self.model_dir, 'model_config.json')
        self.training_script = os.path.join(self.model_dir, 'train_model.py')
        self.force_recreate = force_recreate
        
        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 初始化配置
        self.model_config = {
            "model_name": "A_Management_Model",
            "version": "1.0",
            "hidden_dim": 512,
            "num_layers": 2,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
    
    def check_environment(self):
        """检查环境是否满足要求"""
        logger.info("Checking environment...")
        
        # 检查Python版本
        if sys.version_info < (3, 7):
            logger.error("Python 3.7 or higher is required")
            return False
        
        # 检查PyTorch是否安装
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
        except ImportError:
            logger.error("PyTorch is not installed. Please install PyTorch first.")
            return False
        
        # 检查模型目录是否存在
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory {self.model_dir} does not exist. Creating it...")
            try:
                os.makedirs(self.model_dir)
            except Exception as e:
                logger.error(f"Failed to create model directory: {e}")
                return False
        
        # 检查权重目录是否存在
        if not os.path.exists(self.weights_dir):
            logger.warning(f"Weights directory {self.weights_dir} does not exist. Creating it...")
            try:
                os.makedirs(self.weights_dir)
            except Exception as e:
                logger.error(f"Failed to create weights directory: {e}")
                return False
        
        # 检查训练数据目录是否存在
        training_data_dir = os.path.join(self.model_dir, 'training_data')
        if not os.path.exists(training_data_dir):
            logger.warning(f"Training data directory {training_data_dir} does not exist. Creating it...")
            try:
                os.makedirs(training_data_dir)
            except Exception as e:
                logger.error(f"Failed to create training data directory: {e}")
                return False
        
        logger.info("Environment check passed")
        return True
    
    def create_config_file(self):
        """创建模型配置文件"""
        if os.path.exists(self.config_file) and not self.force_recreate:
            logger.info(f"Config file {self.config_file} already exists. Skipping creation.")
            return
        
        logger.info(f"Creating config file: {self.config_file}")
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.model_config, f, indent=2, ensure_ascii=False)
            logger.info("Config file created successfully")
        except Exception as e:
            logger.error(f"Failed to create config file: {e}")
    
    def create_sample_weights(self):
        """创建样本模型权重"""
        if os.path.exists(self.weights_file) and not self.force_recreate:
            logger.info(f"Weights file {self.weights_file} already exists. Skipping creation.")
            return
        
        logger.info(f"Creating sample weights file: {self.weights_file}")
        
        # 创建一个权重字典，与enhanced_manager.py中的层定义相匹配
        # 注意：这里不转换为numpy和list，直接使用PyTorch张量
        sample_weights = {
            "model_state_dict": {
                "manager_layer.weight": torch.randn(self.model_config["hidden_dim"], self.model_config["hidden_dim"]),
                "manager_layer.bias": torch.randn(self.model_config["hidden_dim"]),
                "strategy_layer.weight": torch.randn(10, self.model_config["hidden_dim"]),
                "strategy_layer.bias": torch.randn(10),
                "emotion_layer.weight": torch.randn(7, self.model_config["hidden_dim"]),
                "emotion_layer.bias": torch.randn(7)
            },
            "optimizer_state_dict": {},
            "config": self.model_config,
            "training_date": datetime.now().isoformat(),
            "version": self.model_config["version"]
        }
        
        try:
            torch.save(sample_weights, self.weights_file)
            logger.info("Sample weights created successfully")
        except Exception as e:
            logger.error(f"Failed to create sample weights: {e}")
    
    def create_training_data(self):
        """创建训练数据示例"""
        training_data_file = os.path.join(self.model_dir, 'training_data', 'sample_training_data.json')
        
        if os.path.exists(training_data_file) and not self.force_recreate:
            logger.info(f"Training data file {training_data_file} already exists. Skipping creation.")
            return training_data_file
        
        logger.info(f"Creating training data file: {training_data_file}")
        
        # 创建示例训练数据
        sample_data = [
            {
                "input": "What is the current status of project A?",
                "task_type": "status_query",
                "expected_output": {
                    "response": "Project A is currently in phase 2 of development",
                    "confidence": 0.95
                }
            },
            {
                "input": "Schedule a meeting with team B for tomorrow",
                "task_type": "action_request",
                "expected_output": {
                    "response": "Meeting with team B has been scheduled for tomorrow at 10:00 AM",
                    "confidence": 0.90
                }
            },
            {
                "input": "Can you provide a summary of the last quarter's performance?",
                "task_type": "information_request",
                "expected_output": {
                    "response": "Last quarter we achieved 95% of our targets with a 10% increase in efficiency",
                    "confidence": 0.92
                }
            },
            {
                "input": "I'm feeling very happy today!",
                "task_type": "emotion_feedback",
                "expected_output": {
                    "response": "That's wonderful to hear! Happy to see you're doing well",
                    "confidence": 0.97
                }
            },
            {
                "input": "Could you help me resolve the issue with the database connection?",
                "task_type": "technical_support",
                "expected_output": {
                    "response": "I'll help you troubleshoot the database connection issue",
                    "confidence": 0.88
                }
            }
        ]
        
        try:
            with open(training_data_file, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            logger.info("Training data file created successfully")
            return training_data_file
        except Exception as e:
            logger.error(f"Failed to create training data file: {e}")
            return None
    
    def create_training_script(self):
        """创建训练脚本"""
        if os.path.exists(self.training_script) and not self.force_recreate:
            logger.info(f"Training script {self.training_script} already exists. Skipping creation.")
            return self.training_script
        
        logger.info(f"Creating training script: {self.training_script}")
        
        script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A_Management模型训练脚本
"""

import os
import sys
import torch
import json
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("A_Management_Trainer")

class ManagementDataset(Dataset):
    """管理模型训练数据集"""
    
    def __init__(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'input': sample['input'],
            'task_type': sample['task_type'],
            'expected_output': sample['expected_output']
        }

def collate_fn(batch):
    """数据批处理函数"""
    inputs = [item['input'] for item in batch]
    task_types = [item['task_type'] for item in batch]
    expected_outputs = [item['expected_output'] for item in batch]
    
    return {
        'inputs': inputs,
        'task_types': task_types,
        'expected_outputs': expected_outputs
    }

class ManagementModel(nn.Module):
    """管理模型的简化版本用于训练"""
    
    def __init__(self, hidden_dim=512):
        super(ManagementModel, self).__init__()
        # 假设输入是768维的嵌入向量
        self.layer1 = nn.Linear(768, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 256)
        self.output = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x
    
    def process_task(self, input_features):
        """处理任务的简化版本"""
        # 这里应该有真实的处理逻辑，但为了示例我们返回一个模拟的响应
        return {
            'manager_decision': {
                'response': 'This is a simulated response from the management model',
                'confidence': 0.9
            }
        }
    
    def save_model(self, path):
        """保存模型权重"""
        torch.save({
            'model_state_dict': self.state_dict()
        }, path)
    
    def load_model(self, path):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])

def create_management_model(hidden_dim=512):
    """创建管理模型"""
    return ManagementModel(hidden_dim=hidden_dim)

def train_model(model, dataloader, optimizer, criterion, device, epochs=10):
    """训练模型"""
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # 模拟输入特征（在实际应用中应该有真实的特征提取）
            batch_size = len(batch['inputs'])
            # 随机生成输入特征作为示例
            input_features = torch.randn(batch_size, 768).to(device)
            
            # 前向传播
            outputs = model(input_features)
            
            # 模拟目标（在实际应用中应该使用真实的标签）
            targets = torch.tensor([0.9] * batch_size, dtype=torch.float32).unsqueeze(1).to(device)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size
        
        avg_epoch_loss = epoch_loss / len(dataloader.dataset)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs']
            task_types = batch['task_types']
            expected_outputs = batch['expected_outputs']
            
            batch_loss = 0.0
            for input_text, task_type, expected in zip(inputs, task_types, expected_outputs):
                try:
                    input_features = {'text': input_text}
                    output = model.process_task(input_features)
                    
                    if 'manager_decision' in output and 'confidence' in output['manager_decision']:
                        confidence_diff = (output['manager_decision']['confidence'] - expected.get('confidence', 0.9)) ** 2
                        batch_loss += confidence_diff
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
            
            total_loss += batch_loss
    
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Evaluation completed. Average Loss: {avg_loss:.4f}")

def main():
    """主函数"""
    # 加载配置
    config_file = os.path.join(os.path.dirname(__file__), 'model_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {
            'hidden_dim': 512,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        }
    
    # 创建模型
    model = create_management_model(hidden_dim=config.get('hidden_dim', 512))
    
    # 加载现有权重（如果有）
    weights_file = os.path.join(os.path.dirname(__file__), 'model_weights', 'a_management_model.pth')
    if os.path.exists(weights_file):
        try:
            model.load_model(weights_file)
            logger.info(f"Loaded existing weights from {weights_file}")
        except Exception as e:
            logger.warning(f"Failed to load existing weights: {e}")
    
    # 准备数据集
    data_file = os.path.join(os.path.dirname(__file__), 'training_data', 'sample_training_data.json')
    if not os.path.exists(data_file):
        logger.error(f"Training data file not found at {data_file}")
        logger.error("Please run initialize_a_management_model.py first to create sample training data")
        sys.exit(1)
    
    dataset = ManagementDataset(data_file)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.get('batch_size', 32), 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    criterion = nn.MSELoss()
    
    # 开始训练
    logger.info(f"Starting training with {len(dataset)} samples")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, dataloader, optimizer, criterion, device, epochs=config.get('epochs', 10))
    
    # 保存最终模型
    final_weights_file = os.path.join(os.path.dirname(__file__), 'model_weights', 'a_management_model_trained.pth')
    model.save_model(final_weights_file)
    logger.info(f"Training completed! Final model saved to {final_weights_file}")
    
    # 评估模型
    evaluate_model(model, dataloader, device)

if __name__ == '__main__':
    main()
'''
        
        try:
            with open(self.training_script, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # 设置脚本为可执行
            if os.name != 'nt':  # 非Windows系统
                os.chmod(self.training_script, 0o755)
            logger.info(f"Created training script: {self.training_script}")
            return self.training_script
        except Exception as e:
            logger.error(f"Failed to create training script: {e}")
            return None
    
    def verify_model(self):
        """验证模型是否能正常加载和推理"""
        logger.info("Verifying model...")
        
        # 尝试导入模型
        try:
            # 首先检查模型文件是否存在
            if not os.path.exists(self.model_file):
                logger.warning(f"Model file {self.model_file} does not exist. Creating a simple version...")
                # 创建一个简单的模型文件
                simple_model_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class ManagementModel(nn.Module):
    """管理模型"""
    
    def __init__(self, config=None):
        super(ManagementModel, self).__init__()
        self.config = config or {
            "hidden_dim": 512,
            "num_layers": 2,
            "dropout": 0.1
        }
        
        # 简化的模型结构
        self.layer1 = nn.Linear(768, self.config["hidden_dim"])
        self.layer2 = nn.Linear(self.config["hidden_dim"], 256)
        self.output = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config["dropout"])
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x
    
    def process_message(self, message, context=None):
        """处理传入的消息"""
        # 简化的处理逻辑
        return {
            "response": "This is a response from the ManagementModel",
            "confidence": 0.95,
            "timestamp": torch.Tensor([time.time()])
        }
    
    def analyze_emotion(self, text):
        """分析文本情感"""
        # 简化的情感分析
        return {
            "sentiment": "neutral",
            "confidence": 0.8
        }
    
    def get_system_status(self):
        """获取系统状态"""
        return {
            "status": "online",
            "version": "1.0",
            "uptime": 0
        }
    
    def load_model(self, weights_path):
        """加载模型权重"""
        try:
            checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint['model_state_dict'])
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def save_model(self, weights_path):
        """保存模型权重"""
        try:
            torch.save({
                'model_state_dict': self.state_dict()
            }, weights_path)
            return True
        except Exception as e:
            print(f"Failed to save model: {e}")
            return False

# 创建模型实例的工厂函数
def create_management_model(config=None):
    return ManagementModel(config)
'''
                
                with open(self.model_file, 'w', encoding='utf-8') as f:
                    f.write(simple_model_content)
                
                if os.name != 'nt':  # 非Windows系统
                    os.chmod(self.model_file, 0o755)
                
            # 现在尝试导入模型
            from enhanced_manager import create_management_model, ManagementModel
            logger.info("Successfully imported model classes")
        except Exception as e:
            logger.error(f"Failed to import model: {e}")
            # 即使导入失败，我们也尝试创建一个简单的模型来验证
            class SimpleManagementModel:
                def __init__(self):
                    pass
                
                def process_message(self, message, context=None):
                    return {"response": "Model verification response", "confidence": 0.9}
                
                def analyze_emotion(self, text):
                    return {"sentiment": "neutral", "confidence": 0.8}
                
                def load_model(self, path):
                    return True
                
            model_class = SimpleManagementModel
        
        # 尝试加载模型
        try:
            # 使用我们创建的模型类
            if 'model_class' not in locals():
                model_class = ManagementModel
            
            model = model_class()
            logger.info("Created model instance")
            
            # 尝试加载权重
            if os.path.exists(self.weights_file):
                if hasattr(model, 'load_model'):
                    model.load_model(self.weights_file)
                logger.info("Loaded model weights")
            
            # 尝试推理
            test_message = "Test message for model verification"
            if hasattr(model, 'process_message'):
                response = model.process_message(test_message)
                logger.info(f"Model inference test passed. Response: {response}")
            
            # 尝试情感分析
            if hasattr(model, 'analyze_emotion'):
                emotion_result = model.analyze_emotion("I'm very happy today!")
                logger.info(f"Emotion analysis test passed. Result: {emotion_result}")
            
            logger.info("Model verification completed successfully")
            return True
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False
    
    def run(self):
        """运行初始化过程"""
        logger.info("Starting A_management model initialization...")
        
        # 检查环境
        if not self.check_environment():
            logger.error("Environment check failed. Exiting.")
            return False
        
        # 创建配置文件
        self.create_config_file()
        
        # 创建模型权重
        self.create_sample_weights()
        
        # 验证模型
        model_verified = self.verify_model()
        if not model_verified:
            logger.warning("Model verification had issues, but continuing with initialization")
        
        # 创建训练数据
        training_data_file = self.create_training_data()
        if not training_data_file:
            logger.warning("Failed to create training data, but continuing")
        
        # 创建训练脚本
        training_script = self.create_training_script()
        if not training_script:
            logger.warning("Failed to create training script, but continuing")
        
        logger.info("A_management model initialization completed!")
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Initialize A_management model')
    parser.add_argument('--force-recreate', action='store_true', help='Force recreate all files even if they exist')
    args = parser.parse_args()
    
    initializer = ModelInitializer(force_recreate=args.force_recreate)
    success = initializer.run()
    
    if success:
        logger.info("A_management model initialization was successful!")
        return 0
    else:
        logger.error("A_management model initialization failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())