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

# D_image模型训练器
# D_image Model Trainer

import logging
import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("D_image_Trainer")

class ImageDataset(Dataset):
    """图像数据集 | Image dataset"""
    
    def __init__(self, data_path: str, image_size: int = 224):
        self.data_path = data_path
        self.image_size = image_size
        self.samples = self._load_data()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_data(self):
        """加载训练数据 | Load training data"""
        samples = []
        data_file = os.path.join(self.data_path, "training_data.json")
        
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    samples = data.get("samples", [])
            except Exception as e:
                logger.error(f"加载训练数据错误: {str(e)}")
        
        # 如果没有数据，创建一些示例数据信息
        if not samples:
            samples = [
                {"image_path": "data/images/cat_01.jpg", "label": "cat", "category": "animal"},
                {"image_path": "data/images/dog_01.jpg", "label": "dog", "category": "animal"},
                {"image_path": "data/images/car_01.jpg", "label": "car", "category": "vehicle"},
                {"image_path": "data/images/tree_01.jpg", "label": "tree", "category": "plant"},
                {"image_path": "data/images/person_01.jpg", "label": "person", "category": "human"}
            ]
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample.get("image_path", "")
        label = sample.get("label", "")
        
        try:
            # 加载图像文件（如果存在）
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                # 生成模拟图像数据
                image = self._generate_dummy_image()
            
            # 应用变换
            image_tensor = self.transform(image)
            
            # 标签编码
            label_map = {"cat": 0, "dog": 1, "car": 2, "tree": 3, "person": 4}
            label_tensor = torch.tensor(label_map.get(label, 0), dtype=torch.long)
            
            return image_tensor, label_tensor
            
        except Exception as e:
            logger.error(f"处理图像数据错误: {str(e)}")
            # 返回空数据
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32), torch.tensor(0, dtype=torch.long)
    
    def _generate_dummy_image(self):
        """生成虚拟图像数据 | Generate dummy image data"""
        # 创建随机RGB图像
        img_array = np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
        return Image.fromarray(img_array)

class SimpleImageModel(nn.Module):
    """简单的图像模型 | Simple image model"""
    
    def __init__(self, num_classes: int = 5):
        super(SimpleImageModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224)
        x = self.pool(self.relu(self.conv1(x)))  # (batch_size, 32, 112, 112)
        x = self.pool(self.relu(self.conv2(x)))  # (batch_size, 64, 56, 56)
        x = self.pool(self.relu(self.conv3(x)))  # (batch_size, 128, 28, 28)
        x = x.view(-1, 128 * 28 * 28)  # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ModelTrainer:
    def __init__(self, model_path: str = None):
        """初始化图像模型训练器 | Initialize image model trainer
        参数:
            model_path: 模型保存路径 | Model save path
        """
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or "models/d_image_model.pth"
        self.training_history = []
        
        # 创建模型目录
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
    def initialize_model(self, num_classes: int = 5):
        """初始化模型 | Initialize model"""
        self.model = SimpleImageModel(num_classes)
        self.model.to(self.device)
        logger.info(f"模型初始化完成，设备: {self.device}")
        
    def load_model(self, model_path: str = None):
        """加载预训练模型 | Load pretrained model"""
        path = model_path or self.model_path
        if os.path.exists(path):
            try:
                self.model = torch.load(path, map_location=self.device)
                self.model.to(self.device)
                logger.info(f"模型加载成功: {path}")
                return True
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
        return False
    
    def save_model(self, model_path: str = None):
        """保存模型 | Save model"""
        path = model_path or self.model_path
        try:
            torch.save(self.model, path)
            logger.info(f"模型保存成功: {path}")
            return True
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
            return False
    
    def train(self, epochs: int, batch_size: int, learning_rate: float, 
             data_path: str = None, callback: callable = None) -> Dict:
        """训练图像模型 | Train image model
        参数:
            epochs: 训练轮数 | Number of epochs
            batch_size: 批量大小 | Batch size
            learning_rate: 学习率 | Learning rate
            data_path: 数据路径 | Data path
            callback: 回调函数 | Callback function
        返回:
            训练结果字典 | Training result dictionary
        """
        logger.info("开始训练图像模型 | Starting image model training")
        logger.info(f"配置: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
        
        # 初始化模型（如果尚未初始化）
        if self.model is None:
            self.initialize_model()
        
        # 准备数据
        dataset = ImageDataset(data_path or "data/d_image")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            epoch_samples = 0
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                acc = (predicted == labels).float().mean()
                
                # 更新统计
                batch_size = images.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_acc += acc.item() * batch_size
                epoch_samples += batch_size
                
                # 调用回调函数（用于进度更新）
                if callback:
                    progress = (epoch * len(dataloader) + batch_idx + 1) / (epochs * len(dataloader)) * 100
                    callback(progress, epoch + 1, {
                        'loss': loss.item(),
                        'accuracy': acc.item(),
                        'batch': batch_idx + 1
                    })
            
            # 计算epoch统计
            avg_loss = epoch_loss / epoch_samples
            avg_acc = epoch_acc / epoch_samples
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
            
            # 保存训练历史
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': avg_acc,
                'samples': epoch_samples
            })
            
            # 更新总统计
            total_loss += epoch_loss
            total_acc += epoch_acc
            total_samples += epoch_samples
        
        # 计算总统计
        final_loss = total_loss / total_samples
        final_acc = total_acc / total_samples
        
        # 保存模型
        self.save_model()
        
        logger.info("图像模型训练完成 | Image model training completed")
        
        return {
            "loss": final_loss,
            "accuracy": final_acc,
            "epochs": epochs,
            "samples": total_samples,
            "history": self.training_history
        }
    
    def evaluate(self, data_path: str = None) -> Dict:
        """评估模型 | Evaluate model
        参数:
            data_path: 数据路径 | Data path
        返回:
            评估结果字典 | Evaluation result dictionary
        """
        if self.model is None:
            logger.error("模型未初始化 | Model not initialized")
            return {"error": "Model not initialized"}
        
        # 准备数据
        dataset = ImageDataset(data_path or "data/d_image")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 评估循环
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                acc = (predicted == labels).float().mean()
                
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_acc += acc.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        
        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
            "samples": total_samples
        }
    
    def predict(self, image_path: str) -> Dict:
        """预测图像 | Predict image
        参数:
            image_path: 图像文件路径 | Image file path
        返回:
            预测结果字典 | Prediction result dictionary
        """
        if self.model is None:
            logger.error("模型未初始化 | Model not initialized")
            return {"error": "Model not initialized"}
        
        self.model.eval()
        
        try:
            # 加载图像文件
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                # 生成模拟图像
                image = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
            
            # 应用变换
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # 标签映射
            label_map = {0: "cat", 1: "dog", 2: "car", 3: "tree", 4: "person"}
            predicted_label = label_map.get(predicted.item(), "unknown")
            
            return {
                "image_path": image_path,
                "predicted_label": predicted_label,
                "confidence": confidence.item(),
                "probabilities": probabilities.cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"图像预测错误: {str(e)}")
            return {"error": str(e)}

    def generate_image(self, text: str = None, emotion: str = "neutral", style: str = "realistic") -> str:
        """生成图像 | Generate image
        参数:
            text: 文本描述（可选） | Text description (optional)
            emotion: 情感状态 | Emotion state
            style: 图像风格 | Image style
        返回:
            生成的图像文件路径 | Generated image file path
        """
        try:
            # 创建输出目录
            output_dir = "generated_images"
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成图像文件路径
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"generated_{emotion}_{style}_{timestamp}.jpg")
            
            # 模拟图像生成（在实际应用中应使用真实的图像生成模型）
            image_size = 224
            img_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            
            if emotion == "happy":
                # 快乐的图像 - 明亮的颜色
                img_array = np.random.randint(150, 256, (image_size, image_size, 3), dtype=np.uint8)
            elif emotion == "sad":
                # 悲伤的图像 - 暗淡的颜色
                img_array = np.random.randint(0, 100, (image_size, image_size, 3), dtype=np.uint8)
            elif emotion == "angry":
                # 愤怒的图像 - 红色调
                img_array = np.random.randint(100, 200, (image_size, image_size, 3), dtype=np.uint8)
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 50, 0, 255)  # 增加红色
            else:
                # 中性的图像
                img_array = np.random.randint(50, 200, (image_size, image_size, 3), dtype=np.uint8)
            
            # 根据风格调整图像
            if style == "abstract":
                # 抽象风格 - 添加几何图案
                center = image_size // 2
                radius = image_size // 4
                y, x = np.ogrid[:image_size, :image_size]
                mask = (x - center)**2 + (y - center)**2 <= radius**2
                img_array[mask] = np.random.randint(0, 256, 3)
            
            # 创建图像并保存
            image = Image.fromarray(img_array)
            image.save(output_path)
            
            logger.info(f"图像生成成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"图像生成错误: {str(e)}")
            return ""

    def enhance_image(self, image_path: str, quality: float = 1.0) -> str:
        """增强图像质量 | Enhance image quality
        参数:
            image_path: 图像文件路径 | Image file path
            quality: 质量增强因子 | Quality enhancement factor
        返回:
            增强后的图像文件路径 | Enhanced image file path
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"图像文件不存在: {image_path}")
                return ""
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 创建输出目录
            output_dir = "enhanced_images"
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成输出文件路径
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"enhanced_{timestamp}.jpg")
            
            # 简单的图像增强（在实际应用中应使用更复杂的算法）
            if quality > 1.0:
                # 增加对比度
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(quality)
                
                # 增加锐度
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(min(quality, 2.0))
            
            # 保存增强后的图像
            image.save(output_path, quality=95)
            
            logger.info(f"图像增强成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"图像增强错误: {str(e)}")
            return ""

# 全局训练器实例
global_trainer = ModelTrainer()

def get_trainer():
    """获取全局训练器实例 | Get global trainer instance"""
    return global_trainer

# 测试代码
if __name__ == '__main__':
    # 测试训练器
    trainer = ModelTrainer()
    
    # 训练模型
    result = trainer.train(
        epochs=3,
        batch_size=8,
        learning_rate=0.001
    )
    
    print(f"训练结果: {result}")
    
    # 测试图像生成
    image_path = trainer.generate_image(emotion="happy", style="abstract")
    if image_path:
        print(f"生成的图像: {image_path}")
    
    # 测试预测（使用生成的图像）
    if image_path:
        prediction = trainer.predict(image_path)
        print(f"预测结果: {prediction}")
    
    # 测试图像增强
    if image_path:
        enhanced_path = trainer.enhance_image(image_path, quality=1.5)
        if enhanced_path:
            print(f"增强后的图像: {enhanced_path}")
