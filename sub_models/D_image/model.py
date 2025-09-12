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

# 图片视觉处理模型定义
# Image Visual Processing Model Definition

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import cv2
import requests
import json
import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torchvision.transforms.functional as F

class ImageModel(nn.Module):
    def __init__(self, num_classes=1000, config_path="config/image_config.json"):
        """初始化图片视觉处理模型 | Initialize image visual processing model"""
        super(ImageModel, self).__init__()
        # 使用预训练的ResNet作为基础模型
        # Use pre-trained ResNet as base model
        self.base_model = models.resnet50(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
        # 加载配置 | Load configuration
        self.config = self.load_config(config_path)
        self.model_type = self.config.get("model_type", "local")
        self.external_api_config = self.config.get("external_api", {})
        
        # 初始化图像生成模型 | Initialize image generation model
        self.generation_model = None
        if self.model_type == "local":
            try:
                # 加载Stable Diffusion模型用于图像生成 | Load Stable Diffusion model for image generation
                model_id = "stabilityai/stable-diffusion-2-1"
                scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
                self.generation_model = StableDiffusionPipeline.from_pretrained(
                    model_id, scheduler=scheduler, torch_dtype=torch.float16
                )
                if torch.cuda.is_available():
                    self.generation_model = self.generation_model.to("cuda")
                print("图像生成模型加载成功 | Image generation model loaded successfully")
            except Exception as e:
                print(f"图像生成模型加载失败: {e} | Image generation model loading failed: {e}")
        
        # 情感到视觉风格的映射 | Emotion to visual style mapping
        self.emotion_styles = {
            "happy": "bright, vibrant, colorful, joyful, sunny, cheerful",
            "sad": "dark, gloomy, monochromatic, melancholic, rainy, somber",
            "angry": "intense, fiery, red tones, dramatic, chaotic, stormy",
            "fear": "dark, shadowy, mysterious, eerie, suspenseful, foggy",
            "surprise": "dynamic, unexpected, contrasting, explosive, sparkling",
            "neutral": "balanced, clear, realistic, natural, peaceful, calm",
            "excited": "energetic, vibrant, dynamic, colorful, sparkling, lively"
        }
        
    def load_config(self, config_path):
        """加载配置文件 | Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {
                "model_type": "local",
                "default_style": "realistic"
            }
        
    def forward(self, x):
        """前向传播 | Forward pass"""
        return self.base_model(x)
    
    def recognize_image(self, image_path):
        """识别图片内容 | Recognize image content"""
        try:
            # 加载并预处理图像
            # Load and preprocess image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            # 使用模型进行预测
            # Use model for prediction
            with torch.no_grad():
                outputs = self.base_model(image_tensor)
                _, predicted = torch.max(outputs, 1)
            
            # 获取类别名称 | Get class names
            class_names = self.get_imagenet_classes()
            predicted_class_name = class_names[predicted.item()] if predicted.item() < len(class_names) else "unknown"
            
            # 返回识别结果
            # Return recognition result
            return {
                'status': 'success',
                'predicted_class': predicted.item(),
                'predicted_class_name': predicted_class_name,
                'confidence': torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_imagenet_classes(self):
        """获取ImageNet类别名称 | Get ImageNet class names"""
        try:
            # 尝试加载本地类别文件 | Try to load local class file
            with open("config/imagenet_classes.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            # 默认ImageNet类别 | Default ImageNet classes
            return ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark"]  # 简化的列表
    
    def modify_image(self, image_path, modifications):
        """修改图片内容 | Modify image content"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # 应用修改 | Apply modifications
            for mod in modifications:
                operation = mod.get("operation")
                params = mod.get("parameters", {})
                
                if operation == "crop":
                    # 裁剪图片 | Crop image
                    left = params.get("left", 0)
                    top = params.get("top", 0)
                    right = params.get("right", image.width)
                    bottom = params.get("bottom", image.height)
                    image = image.crop((left, top, right, bottom))
                    
                elif operation == "rotate":
                    # 旋转图片 | Rotate image
                    angle = params.get("angle", 0)
                    image = image.rotate(angle, expand=True)
                    
                elif operation == "flip":
                    # 翻转图片 | Flip image
                    direction = params.get("direction", "horizontal")
                    if direction == "horizontal":
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    else:
                        image = image.transpose(Image.FLIP_TOP_BOTTOM)
                        
                elif operation == "filter":
                    # 应用滤镜 | Apply filter
                    filter_type = params.get("type", "blur")
                    if filter_type == "blur":
                        radius = params.get("radius", 2)
                        image = image.filter(ImageFilter.GaussianBlur(radius))
                    elif filter_type == "sharpen":
                        image = image.filter(ImageFilter.SHARPEN)
                    elif filter_type == "edge_enhance":
                        image = image.filter(ImageFilter.EDGE_ENHANCE)
                    elif filter_type == "contour":
                        image = image.filter(ImageFilter.CONTOUR)
                    elif filter_type == "emboss":
                        image = image.filter(ImageFilter.EMBOSS)
                        
                elif operation == "adjust":
                    # 调整图像参数 | Adjust image parameters
                    brightness = params.get("brightness", 1.0)
                    contrast = params.get("contrast", 1.0)
                    saturation = params.get("saturation", 1.0)
                    sharpness = params.get("sharpness", 1.0)
                    
                    if brightness != 1.0:
                        image = ImageEnhance.Brightness(image).enhance(brightness)
                    if contrast != 1.0:
                        image = ImageEnhance.Contrast(image).enhance(contrast)
                    if saturation != 1.0:
                        image = ImageEnhance.Color(image).enhance(saturation)
                    if sharpness != 1.0:
                        image = ImageEnhance.Sharpness(image).enhance(sharpness)
                        
                elif operation == "draw":
                    # 绘制图形 | Draw shapes
                    draw = ImageDraw.Draw(image)
                    shape = params.get("shape", "rectangle")
                    color = params.get("color", "#FF0000")
                    width = params.get("width", 2)
                    
                    if shape == "rectangle":
                        x1, y1, x2, y2 = params.get("coordinates", [10, 10, 100, 100])
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                    elif shape == "circle":
                        x, y, radius = params.get("coordinates", [50, 50, 30])
                        draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, width=width)
                    elif shape == "line":
                        x1, y1, x2, y2 = params.get("coordinates", [10, 10, 100, 100])
                        draw.line([x1, y1, x2, y2], fill=color, width=width)
                    elif shape == "text":
                        text = params.get("text", "Hello")
                        x, y = params.get("coordinates", [10, 10])
                        font_size = params.get("font_size", 20)
                        try:
                            font = ImageFont.truetype("arial.ttf", font_size)
                        except:
                            font = ImageFont.load_default()
                        draw.text((x, y), text, fill=color, font=font)
            
            # 保存修改后的图片 | Save modified image
            output_path = "output/modified_image.jpg"
            os.makedirs("output", exist_ok=True)
            image.save(output_path, quality=95)
            
            return {
                'status': 'success',
                'message': 'Image modified successfully',
                'output_path': output_path
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def adjust_image(self, image_path, size=None, quality=None, clarity=None):
        """调整图片清晰度和大小 | Adjust image clarity and size"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # 调整大小 | Adjust size
            if size:
                width, height = size
                image = image.resize((width, height), Image.LANCZOS)
            
            # 调整清晰度 | Adjust clarity
            if clarity is not None:
                if clarity > 0:
                    # 锐化图像 | Sharpen image
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(1.0 + clarity * 0.5)
                else:
                    # 模糊图像 | Blur image
                    blur_radius = abs(clarity) * 2
                    image = image.filter(ImageFilter.GaussianBlur(blur_radius))
            
            # 保存调整后的图片 | Save adjusted image
            output_path = "output/adjusted_image.jpg"
            os.makedirs("output", exist_ok=True)
            
            save_quality = quality if quality else 95
            image.save(output_path, quality=save_quality)
            
            return {
                'status': 'success',
                'message': 'Image adjusted successfully',
                'output_path': output_path,
                'new_size': image.size,
                'quality': save_quality
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def generate_image(self, prompt, emotion="neutral", size=(512, 512)):
        """根据语义和情感生成图片 | Generate image based on semantics and emotion"""
        try:
            # 根据情感调整提示词 | Adjust prompt based on emotion
            emotion_style = self.emotion_styles.get(emotion, self.emotion_styles["neutral"])
            enhanced_prompt = f"{prompt}, {emotion_style}, high quality, detailed, 4k"
            
            if self.model_type == "external":
                # 使用外部API生成图像 | Use external API for image generation
                return self._call_external_api("generate", {
                    "prompt": enhanced_prompt,
                    "size": size,
                    "emotion": emotion
                })
            
            # 使用本地模型生成图像 | Use local model for image generation
            if self.generation_model is None:
                return {'status': 'error', 'message': 'Image generation model not available'}
            
            # 生成图像 | Generate image
            with torch.no_grad():
                if torch.cuda.is_available():
                    image = self.generation_model(enhanced_prompt, height=size[1], width=size[0]).images[0]
                else:
                    image = self.generation_model(enhanced_prompt, height=size[1], width=size[0]).images[0]
            
            # 保存生成的图片 | Save generated image
            output_path = "output/generated_image.png"
            os.makedirs("output", exist_ok=True)
            image.save(output_path)
            
            return {
                'status': 'success',
                'message': 'Image generated successfully',
                'output_path': output_path,
                'prompt': prompt,
                'emotion': emotion,
                'size': size
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _call_external_api(self, operation, data):
        """调用外部API | Call external API"""
        try:
            api_endpoint = self.external_api_config.get("endpoint")
            api_key = self.external_api_config.get("api_key")
            
            if not api_endpoint or not api_key:
                return {'status': 'error', 'message': 'API configuration missing'}
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{api_endpoint}/{operation}",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'status': 'error', 'message': f'API error: {response.status_code}'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def validate_api_connection(self):
        """验证外部API连接 | Validate external API connection"""
        api_endpoint = self.external_api_config.get("endpoint")
        api_key = self.external_api_config.get("api_key")
        
        if not api_endpoint or not api_key:
            return False
            
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{api_endpoint}/status", headers=headers, timeout=5)
            return response.status_code == 200
        except:
            return False

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
        
        # 获取内存使用情况
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # 获取GPU内存使用情况（如果可用）
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
        return {
            "status": "active",
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "gpu_memory_mb": gpu_memory,
            "parameters_count": sum(p.numel() for p in self.parameters()),
            "model_type": self.model_type,
            "generation_model_available": self.generation_model is not None,
            "last_activity": "2025-08-25 10:00:00",  # 应记录实际最后活动时间
            "performance": {
                "processing_speed": "待测量",
                "recognition_accuracy": "待测量"
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
        # 这里应该从实际使用中收集统计数据
        # 暂时返回模拟数据
        return {
            "total_requests": 180,
            "successful_requests": 165,
            "failed_requests": 15,
            "average_response_time_ms": 200,
            "last_hour_requests": 30,
            "processing_types": {
                "image_recognition": 80,
                "image_modification": 45,
                "image_adjustment": 35,
                "image_generation": 20
            },
            "external_api_connected": self.validate_api_connection()
        }

if __name__ == '__main__':
    # 测试模型
    # Test model
    model = ImageModel()
    print("图片视觉处理模型初始化成功 | Image visual processing model initialized successfully")
    
    # 测试图像识别 | Test image recognition
    print("测试图像识别: ", model.recognize_image("test_image.jpg"))
    
    # 测试图像修改 | Test image modification
    modifications = [
        {"operation": "crop", "parameters": {"left": 10, "top": 10, "right": 200, "bottom": 200}},
        {"operation": "adjust", "parameters": {"brightness": 1.2, "contrast": 1.1}}
    ]
    print("测试图像修改: ", model.modify_image("test_image.jpg", modifications))
    
    # 测试图像调整 | Test image adjustment
    print("测试图像调整: ", model.adjust_image("test_image.jpg", size=(300, 300), quality=90, clarity=0.5))
    
    # 测试图像生成 | Test image generation
    print("测试图像生成: ", model.generate_image("a beautiful sunset", "happy", (512, 512)))
    
    # 测试新添加的方法
    print("模型状态: ", model.get_status())
    print("输入统计: ", model.get_input_stats())
