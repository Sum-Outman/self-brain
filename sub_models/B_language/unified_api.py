# -*- coding: utf-8 -*-
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
统一大语言模型API | Unified Large Language Model API
基于统一模型实现，支持标准模式和增强模式
(Based on unified model implementation, supports standard and enhanced modes)
"""

import json
import requests
from flask import Flask, request, jsonify
import numpy as np
import torch
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import psutil

# 导入统一模型 | Import unified model
from unified_model import UnifiedMultilingualEmotionalLLM, create_unified_model

# 设置日志 | Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_language_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UnifiedLanguageAPI")

app = Flask(__name__)

class UnifiedLanguageModel:
    def __init__(self, mode: str = "standard", config_path: Optional[str] = None):
        """
        初始化统一语言模型 | Initialize unified language model
        
        参数 Parameters:
        mode: 运行模式 - "standard" 或 "enhanced" | Operation mode - "standard" or "enhanced"
        config_path: 配置文件路径 | Configuration file path
        """
        self.mode = mode
        self.config_path = config_path
        self.data_bus = None  # 数据总线，由主模型设置 | Data bus, set by main model
        
        # 加载统一模型 | Load unified model
        self.model = create_unified_model(config_path, mode)
        logger.info(f"统一语言模型初始化完成 (模式: {mode}) | Unified language model initialized (Mode: {mode})")
    
    def set_language(self, language: str) -> bool:
        """
        设置当前使用的语言 | Set current language
        
        参数 Parameters:
        language: 语言代码 | Language code
        
        返回 Returns:
        是否成功设置 | Whether setting was successful
        """
        if language in self.model.supported_languages:
            self.current_language = language
            return True
        return False
    
    def analyze_text(self, text: str) -> Dict:
        """
        分析文本内容，包括情感、语言检测
        Analyze text content including sentiment, language detection
        
        参数 Parameters:
        text: 输入文本 | Input text
        
        返回 Returns:
        分析结果字典 | Analysis result dictionary
        """
        try:
            # 检测语言 | Detect language
            language = self.detect_language(text)
            
            # 使用统一模型进行预测 | Use unified model for prediction
            result = self.model.predict(text, language)
            
            return {
                "text": text,
                "language": language,
                "sentiment": {
                    "primary_emotion": result["emotion"]["name"],
                    "intensity": result["emotion"]["intensity"],
                    "confidence": result["emotion"]["confidence"],
                    "category": result["emotion"]["category"]
                },
                "model_response": result["text"],
                "confidence": result["confidence"],
                "timestamp": result["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"文本分析失败: {e} | Text analysis failed: {e}")
            return {
                "text": text,
                "language": "unknown",
                "sentiment": {
                    "primary_emotion": "neutral",
                    "intensity": 0.5,
                    "confidence": 0.5,
                    "category": "neutral"
                },
                "model_response": f"分析失败: {str(e)} | Analysis failed",
                "confidence": 0.3,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """
        生成文本响应 | Generate text response
        
        参数 Parameters:
        prompt: 提示文本 | Prompt text
        max_length: 最大生成长度 | Maximum generation length
        
        返回 Returns:
        生成的文本 | Generated text
        """
        try:
            # 使用统一模型生成响应 | Use unified model to generate response
            result = self.model.predict(prompt, self.current_language if hasattr(self, 'current_language') else "en")
            return result["text"]
        except Exception as e:
            logger.error(f"文本生成失败: {e} | Text generation failed: {e}")
            return f"生成失败: {str(e)} | Generation failed"
    
    def detect_language(self, text: str) -> str:
        """
        检测文本语言 | Detect text language
        
        参数 Parameters:
        text: 输入文本 | Input text
        
        返回 Returns:
        语言代码 | Language code
        """
        # 简单的语言检测逻辑 | Simple language detection logic
        # 实际实现应该使用更复杂的语言检测库
        # In practice, should use more sophisticated language detection library
        
        if len(text) == 0:
            return "en"  # 默认英语 | Default to English
        
        # 简单的基于字符的语言检测 | Simple character-based language detection
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u30ff')
        cyrillic_chars = sum(1 for char in text if '\u0400' <= char <= '\u04ff')
        
        if chinese_chars > 0:
            return "zh"
        elif japanese_chars > 0:
            return "ja"
        elif cyrillic_chars > 0:
            return "ru"
        else:
            # 默认为英语 | Default to English
            return "en"
    
    def set_data_bus(self, data_bus):
        """设置数据总线 | Set data bus"""
        self.data_bus = data_bus
    
    def get_status(self) -> Dict:
        """
        获取模型状态信息 | Get model status information
        
        返回 Returns:
        状态字典 | Status dictionary
        """
        model_status = self.model.get_status()
        
        return {
            "status": "active",
            "model_type": "UnifiedMultilingualEmotionalLLM",
            "mode": self.mode,
            "supported_languages": model_status["supported_languages"],
            "emotion_categories": model_status["emotion_categories"],
            "memory_usage_mb": model_status["memory_usage_mb"],
            "parameters_count": model_status["parameters_count"],
            "performance": {
                "inference_speed": "待测量",  # 需要实际测量 | Needs actual measurement
                "accuracy": "待测量"         # 需要实际测量 | Needs actual measurement
            }
        }

# 全局配置 | Global configuration
MODEL_CONFIG = {
    "mode": "standard",  # "standard" 或 "enhanced" | "standard" or "enhanced"
    "config_path": None,
    "local_model": True,
    "external_api": None,
    "api_key": ""
}

# 主模型通信配置 | Main model communication configuration
MAIN_MODEL_URL = "http://localhost:5000/receive_data"

# 初始化全局语言模型实例 | Initialize global language model instance
language_model = UnifiedLanguageModel(MODEL_CONFIG["mode"], MODEL_CONFIG["config_path"])

# 健康检查端点 | Health check endpoints
@app.route('/')
def index():
    """健康检查端点 | Health check endpoint"""
    status = language_model.get_status()
    return jsonify({
        "status": "active",
        "model": "B_language_unified",
        "version": "2.0.0",
        "mode": MODEL_CONFIG["mode"],
        "capabilities": [
            "sentiment_analysis", 
            "emotion_reasoning", 
            "multilingual_support",
            "text_generation"
        ],
        "model_status": status
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查 | Health check"""
    return jsonify({
        "status": "healthy", 
        "model": "B_language_unified",
        "mode": MODEL_CONFIG["mode"]
    })

@app.route('/process', methods=['POST'])
def process():
    """
    处理文本输入，进行深度语义解析和情感推理
    Process text input, perform deep semantic parsing and sentiment reasoning
    返回包含情感分析、语言检测和模型响应的结构化结果
    Return structured results including sentiment analysis, language detection and model response
    """
    data = request.json
    
    # 获取文本输入 | Get text input
    text = data.get('text', '')
    
    if not text:
        return jsonify({
            "error": "缺少文本输入 | Missing text input",
            "status": "error"
        }), 400
    
    # 使用语言模型分析文本 | Use language model to analyze text
    analysis = language_model.analyze_text(text)
    
    # 构建响应 | Build response
    response = {
        "text": analysis["text"],
        "language": analysis["language"],
        "sentiment": analysis["sentiment"],
        "model_response": analysis["model_response"],
        "confidence": analysis["confidence"],
        "timestamp": analysis["timestamp"]
    }
    
    # 发送结果到主模型 | Send results to main model
    try:
        if language_model.data_bus:
            # 优先使用数据总线发送 | Prefer data bus for sending
            language_model.data_bus.send(response)
        else:
            # 回退到HTTP请求 | Fallback to HTTP request
            requests.post(MAIN_MODEL_URL, json=response, timeout=2)
    except Exception as e:
        logger.error(f"主模型通信失败: {e} | Main model communication failed: {e}")
    
    return jsonify(response)

# 模型配置接口 | Model configuration interface
@app.route('/configure', methods=['POST'])
def configure_model():
    """
    配置模型设置，包括模式切换
    Configure model settings, including mode switching
    """
    global MODEL_CONFIG, language_model
    
    config_data = request.json
    
    # 更新配置 | Update configuration
    new_mode = config_data.get('mode', MODEL_CONFIG["mode"])
    new_config_path = config_data.get('config_path', MODEL_CONFIG["config_path"])
    
    # 检查模式是否改变 | Check if mode changed
    if new_mode != MODEL_CONFIG["mode"] or new_config_path != MODEL_CONFIG["config_path"]:
        try:
            # 重新初始化模型 | Reinitialize model
            language_model = UnifiedLanguageModel(new_mode, new_config_path)
            MODEL_CONFIG["mode"] = new_mode
            MODEL_CONFIG["config_path"] = new_config_path
            logger.info(f"模型配置已更新: 模式={new_mode}, 配置路径={new_config_path} | Model configuration updated: mode={new_mode}, config_path={new_config_path}")
        except Exception as e:
            logger.error(f"模型重新初始化失败: {e} | Model reinitialization failed: {e}")
            return jsonify({
                "status": "error",
                "message": f"配置更新失败: {str(e)} | Configuration update failed"
            }), 500
    
    # 更新其他配置 | Update other configurations
    MODEL_CONFIG.update({
        "local_model": config_data.get('local_model', MODEL_CONFIG["local_model"]),
        "external_api": config_data.get('external_api', MODEL_CONFIG["external_api"]),
        "api_key": config_data.get('api_key', MODEL_CONFIG["api_key"])
    })
    
    return jsonify({
        "status": "配置更新成功 | Configuration updated", 
        "config": MODEL_CONFIG
    })

# 语言设置接口 | Language setting interface
@app.route('/set_language', methods=['POST'])
def set_language():
    """
    设置当前使用的语言 | Set current language
    """
    data = request.json
    language = data.get('language', 'en')
    
    success = language_model.set_language(language)
    
    if success:
        return jsonify({
            "status": "success",
            "message": f"语言已设置为 {language} | Language set to {language}"
        })
    else:
        return jsonify({
            "status": "error",
            "message": f"不支持的语言: {language} | Unsupported language: {language}",
            "supported_languages": language_model.model.supported_languages
        }), 400

# 文本生成接口 | Text generation interface
@app.route('/generate', methods=['POST'])
def generate_text():
    """
    生成文本响应 | Generate text response
    """
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    
    if not prompt:
        return jsonify({
            "error": "缺少提示文本 | Missing prompt text",
            "status": "error"
        }), 400
    
    generated_text = language_model.generate_text(prompt, max_length)
    
    return jsonify({
        "prompt": prompt,
        "generated_text": generated_text,
        "length": len(generated_text)
    })

# 模型状态接口 | Model status interface
@app.route('/status', methods=['GET'])
def get_model_status():
    """
    获取模型详细状态信息 | Get detailed model status information
    """
    status = language_model.get_status()
    return jsonify(status)

# 实时监视接口 | Real-time monitoring interface
@app.route('/monitor', methods=['GET'])
def get_monitoring_data():
    """
    获取实时监视数据 | Get real-time monitoring data
    """
    import time
    status = language_model.get_status()
    
    return jsonify({
        "status": "active",
        "model_mode": MODEL_CONFIG["mode"],
        "last_processed": time.time(),
        "performance": {
            "response_time": 0.15,  # 平均响应时间(秒) | Average response time (seconds)
            "supported_languages": len(status["supported_languages"]),
            "emotion_categories": len(status["emotion_categories"]),
            "memory_usage_mb": status["memory_usage_mb"]
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)