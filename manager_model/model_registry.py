# -*- coding: utf-8 -*-
# 模型注册表 - 管理所有子模型的配置和状态
# Model Registry - Manage configuration and status of all submodels
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License")
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

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import requests
from datetime import datetime
import threading
import time
import abc

class BaseModel(abc.ABC):
    """所有模型的基类 | Base class for all models
    提供模型的基本接口和通用功能 | Provides basic interface and common functionality for models
    """
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        """初始化模型 | Initialize model
        
        参数 Parameters:
        model_id: 模型唯一标识符 | Model unique identifier
        config: 模型配置字典 | Model configuration dictionary
        """
        self.model_id = model_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{model_id}")
        
    @abc.abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """处理输入数据 - 抽象方法 | Process input data - abstract method
        
        参数 Parameters:
        input_data: 输入数据 | Input data
        
        返回 Returns:
        处理结果字典 | Processing result dictionary
        """
        pass
    
    def train(self, training_data: Any) -> Dict[str, Any]:
        """训练模型 | Train model
        
        参数 Parameters:
        training_data: 训练数据 | Training data
        
        返回 Returns:
        训练结果字典 | Training result dictionary
        """
        return {"status": "not_implemented", "message": "Training not implemented for this model"}
    
    def evaluate(self, evaluation_data: Any) -> Dict[str, Any]:
        """评估模型性能 | Evaluate model performance
        
        参数 Parameters:
        evaluation_data: 评估数据 | Evaluation data
        
        返回 Returns:
        评估结果字典 | Evaluation result dictionary
        """
        return {"status": "not_implemented", "message": "Evaluation not implemented for this model"}
    
    def receive_data(self, data: Any) -> None:
        """接收广播数据 | Receive broadcast data
        
        参数 Parameters:
        data: 广播数据 | Broadcast data
        """
        # 默认实现：记录接收到的数据 | Default implementation: log received data
        self.logger.debug(f"Received broadcast data: {data}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取模型状态 | Get model status
        
        返回 Returns:
        状态信息字典 | Status information dictionary
        """
        return {
            "model_id": self.model_id,
            "status": "active",
            "config": self.config,
            "capabilities": self.config.get("capabilities", [])
        }

class ModelRegistry:
    """模型注册表 - 管理所有子模型的配置、状态和连接
    Model Registry - Manage configuration, status and connections of all submodels
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化模型注册表 | Initialize model registry
        
        参数 Parameters:
        config_path: 配置文件路径 | Configuration file path
        """
        self.logger = logging.getLogger(__name__)
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.model_ports: Dict[str, int] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.last_checked: Dict[str, datetime] = {}
        
        # 默认模型配置 | Default model configuration
        self.default_config = {
            "B_language": {
                "name": "大语言模型",
                "description": "多语言交互和情感推理",
                "type": "local",
                "enabled": True,
                "port": 8001,
                "api_url": "",
                "api_key": "",
                "local_model_path": "sub_models/B_language",
                "capabilities": ["text_generation", "emotion_analysis", "multilingual"],
                "health_endpoint": "/health",
                "check_interval": 30
            },
            "C_audio": {
                "name": "音频处理模型",
                "description": "语音识别、音乐合成和音频处理",
                "type": "local",
                "enabled": True,
                "port": 8002,
                "api_url": "",
                "api_key": "",
                "local_model_path": "sub_models/C_audio",
                "capabilities": ["speech_recognition", "music_generation", "audio_effects"],
                "health_endpoint": "/health",
                "check_interval": 30
            },
            "D_image": {
                "name": "图片视觉处理模型",
                "description": "图像识别、编辑和生成",
                "type": "local",
                "enabled": True,
                "port": 8003,
                "api_url": "",
                "api_key": "",
                "local_model_path": "sub_models/D_image",
                "capabilities": ["image_recognition", "image_editing", "image_generation"],
                "health_endpoint": "/health",
                "check_interval": 30
            },
            "E_video": {
                "name": "视频流视觉处理模型",
                "description": "视频分析、编辑和生成",
                "type": "local",
                "enabled": True,
                "port": 8004,
                "api_url": "",
                "api_key": "",
                "local_model_path": "sub_models/E_video",
                "capabilities": ["video_analysis", "video_editing", "video_generation"],
                "health_endpoint": "/health",
                "check_interval": 30
            },
            "F_spatial": {
                "name": "双目空间定位感知模型",
                "description": "3D空间建模和定位",
                "type": "local",
                "enabled": True,
                "port": 8005,
                "api_url": "",
                "api_key": "",
                "local_model_path": "sub_models/F_spatial",
                "capabilities": ["3d_mapping", "spatial_localization", "motion_prediction"],
                "health_endpoint": "/health",
                "check_interval": 30
            },
            "G_sensor": {
                "name": "传感器感知模型",
                "description": "多传感器数据采集和处理",
                "type": "local",
                "enabled": True,
                "port": 8006,
                "api_url": "",
                "api_key": "",
                "local_model_path": "sub_models/G_sensor",
                "capabilities": ["sensor_data", "environment_monitoring", "real_time_processing"],
                "health_endpoint": "/health",
                "check_interval": 10  # 更频繁的检查
            },
            "H_computer_control": {
                "name": "计算机控制模型",
                "description": "系统命令执行和控制",
                "type": "local",
                "enabled": True,
                "port": 8007,
                "api_url": "",
                "api_key": "",
                "local_model_path": "sub_models/H_computer_control",
                "capabilities": ["system_control", "batch_processing", "multi_platform"],
                "health_endpoint": "/health",
                "check_interval": 30
            },
            "I_knowledge": {
                "name": "知识库专家模型",
                "description": "多领域知识检索和推理",
                "type": "local",
                "enabled": True,
                "port": 8008,
                "api_url": "",
                "api_key": "",
                "local_model_path": "sub_models/I_knowledge",
                "capabilities": ["knowledge_retrieval", "expert_system", "teaching_assistance"],
                "health_endpoint": "/health",
                "check_interval": 30
            },
            "J_motion": {
                "name": "运动和执行器控制模型",
                "description": "多端口设备控制",
                "type": "local",
                "enabled": True,
                "port": 8009,
                "api_url": "",
                "api_key": "",
                "local_model_path": "sub_models/J_motion",
                "capabilities": ["motion_control", "actuator_management", "multi_protocol"],
                "health_endpoint": "/health",
                "check_interval": 30
            },
            "K_programming": {
                "name": "编程模型",
                "description": "代码生成和系统优化",
                "type": "local",
                "enabled": True,
                "port": 8010,
                "api_url": "",
                "api_key": "",
                "local_model_path": "sub_models/K_programming",
                "capabilities": ["code_generation", "system_optimization", "debugging"],
                "health_endpoint": "/health",
                "check_interval": 30
            }
        }
        
        # 加载配置 | Load configuration
        self.config_path = config_path or "config/model_registry.json"
        self._load_config()
        
        # 启动健康检查线程 | Start health check thread
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        self.logger.info("模型注册表初始化完成 | Model registry initialized")

    def _load_config(self):
        """加载配置文件 | Load configuration file"""
        try:
            config_dir = os.path.dirname(self.config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # 更新默认配置 | Update default configuration
                for model_name, config in user_config.items():
                    if model_name in self.default_config:
                        self.default_config[model_name].update(config)
                    else:
                        self.default_config[model_name] = config
                
                self.logger.info(f"配置文件加载成功: {self.config_path}")
            else:
                # 创建默认配置文件 | Create default configuration file
                self._save_config()
                self.logger.info(f"创建默认配置文件: {self.config_path}")
                
        except Exception as e:
            self.logger.error(f"配置文件加载失败: {e}")
    
    def _save_config(self):
        """保存配置文件 | Save configuration file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.default_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"配置文件保存失败: {e}")

    def load_registry(self):
        """加载模型注册表 | Load model registry"""
        self.registry = self.default_config.copy()
        
        # 初始化端口映射 | Initialize port mapping
        for model_name, config in self.registry.items():
            if config['enabled']:
                self.model_ports[model_name] = config['port']
                self.health_status[model_name] = {
                    'status': 'unknown',
                    'last_check': datetime.now(),
                    'response_time': 0,
                    'error_count': 0
                }
        
        self.logger.info(f"注册表加载完成，共{len(self.registry)}个模型 | Registry loaded with {len(self.registry)} models")

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息 | Get model information"""
        return self.registry.get(model_name)

    def get_model_ports(self) -> Dict[str, int]:
        """获取模型端口映射 | Get model port mapping"""
        return self.model_ports

    def update_model_config(self, model_name: str, config: Dict[str, Any]) -> bool:
        """更新模型配置 | Update model configuration"""
        if model_name not in self.registry:
            self.logger.warning(f"未知模型: {model_name}")
            return False
        
        self.registry[model_name].update(config)
        
        # 更新端口映射 | Update port mapping
        if 'port' in config:
            self.model_ports[model_name] = config['port']
        
        # 保存配置 | Save configuration
        self._save_config()
        
        self.logger.info(f"模型配置更新: {model_name}")
        return True

    def enable_model(self, model_name: str) -> bool:
        """启用模型 | Enable model"""
        if model_name not in self.registry:
            self.logger.warning(f"未知模型: {model_name}")
            return False
        
        self.registry[model_name]['enabled'] = True
        self.model_ports[model_name] = self.registry[model_name]['port']
        self._save_config()
        
        self.logger.info(f"模型已启用: {model_name}")
        return True

    def disable_model(self, model_name: str) -> bool:
        """禁用模型 | Disable model"""
        if model_name not in self.registry:
            self.logger.warning(f"未知模型: {model_name}")
            return False
        
        self.registry[model_name]['enabled'] = False
        if model_name in self.model_ports:
            del self.model_ports[model_name]
        self._save_config()
        
        self.logger.info(f"模型已禁用: {model_name}")
        return True

    def switch_to_external(self, model_name: str, api_url: str, api_key: str = "") -> bool:
        """切换到外部API模型 | Switch to external API model"""
        if model_name not in self.registry:
            self.logger.warning(f"未知模型: {model_name}")
            return False
        
        self.registry[model_name].update({
            'type': 'external',
            'api_url': api_url,
            'api_key': api_key
        })
        
        # 从端口映射中移除 | Remove from port mapping
        if model_name in self.model_ports:
            del self.model_ports[model_name]
        
        self._save_config()
        self.logger.info(f"模型切换到外部API: {model_name}")
        return True

    def switch_to_local(self, model_name: str) -> bool:
        """切换到本地模型 | Switch to local model"""
        if model_name not in self.registry:
            self.logger.warning(f"未知模型: {model_name}")
            return False
        
        self.registry[model_name].update({
            'type': 'local',
            'api_url': '',
            'api_key': ''
        })
        
        # 添加到端口映射 | Add to port mapping
        self.model_ports[model_name] = self.registry[model_name]['port']
        
        self._save_config()
        self.logger.info(f"模型切换到本地: {model_name}")
        return True

    def test_connection(self, model_name: str) -> Dict[str, Any]:
        """测试模型连接 | Test model connection"""
        if model_name not in self.registry:
            return {'status': 'error', 'message': f'未知模型: {model_name}'}
        
        model_config = self.registry[model_name]
        
        if model_config['type'] == 'external':
            # 测试外部API连接 | Test external API connection
            return self._test_external_connection(model_config)
        else:
            # 测试本地模型连接 | Test local model connection
            return self._test_local_connection(model_config)

    def validate_external_api_config(self, api_url: str, api_key: str = "") -> Dict[str, Any]:
        """验证外部API配置 | Validate external API configuration"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}" if api_key else "",
                "Content-Type": "application/json"
            }
            
            # 尝试连接API
            response = requests.get(
                f"{api_url.rstrip('/')}/health",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    'status': 'success',
                    'message': 'API连接验证成功',
                    'response_time': response.elapsed.total_seconds(),
                    'api_info': response.json().get('info', {})
                }
            else:
                return {
                    'status': 'error',
                    'message': f'API返回错误状态码: {response.status_code}',
                    'response_time': response.elapsed.total_seconds()
                }
                
        except requests.exceptions.Timeout:
            return {
                'status': 'error',
                'message': 'API连接超时，请检查网络或URL'
            }
        except requests.exceptions.ConnectionError:
            return {
                'status': 'error',
                'message': '无法连接到API，请检查URL是否正确'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'连接失败: {str(e)}'
            }

    def _test_external_connection(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """测试外部API连接 | Test external API connection"""
        try:
            headers = {
                "Authorization": f"Bearer {model_config.get('api_key', '')}",
                "Content-Type": "application/json"
            }
            
            # 简单的健康检查请求 | Simple health check request
            response = requests.get(
                f"{model_config['api_url']}/health",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                return {
                    'status': 'success',
                    'message': '外部API连接成功',
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'status': 'error',
                    'message': f'API返回错误: {response.status_code}',
                    'response_time': response.elapsed.total_seconds()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'连接失败: {str(e)}'
            }

    def _test_local_connection(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """测试本地模型连接 | Test local model connection"""
        try:
            response = requests.get(
                f"http://localhost:{model_config['port']}/health",
                timeout=3
            )
            
            if response.status_code == 200:
                return {
                    'status': 'success',
                    'message': '本地模型连接成功',
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'status': 'error',
                    'message': f'模型返回错误: {response.status_code}',
                    'response_time': response.elapsed.total_seconds()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'连接失败: {str(e)}'
            }

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型健康状态 | Get health status of all models"""
        return self.health_status

    def _health_check_loop(self):
        """健康检查循环 | Health check loop"""
        while True:
            try:
                for model_name, config in self.registry.items():
                    if not config['enabled']:
                        continue
                    
                    # 检查是否需要健康检查 | Check if health check is needed
                    last_check = self.last_checked.get(model_name)
                    check_interval = config.get('check_interval', 30)
                    
                    if (last_check is None or 
                        (datetime.now() - last_check).total_seconds() >= check_interval):
                        
                        result = self.test_connection(model_name)
                        
                        # 更新健康状态 | Update health status
                        self.health_status[model_name] = {
                            'status': result['status'],
                            'last_check': datetime.now(),
                            'response_time': result.get('response_time', 0),
                            'message': result.get('message', '')
                        }
                        
                        self.last_checked[model_name] = datetime.now()
                        
                        if result['status'] == 'success':
                            self.logger.debug(f"健康检查成功: {model_name}")
                        else:
                            self.logger.warning(f"健康检查失败: {model_name} - {result.get('message', '')}")
                
                # 休眠一段时间 | Sleep for a while
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"健康检查循环错误: {e}")
                time.sleep(30)

    def get_all_models(self) -> List[str]:
        """获取所有模型名称 | Get all model names"""
        return list(self.registry.keys())

    def get_enabled_models(self) -> List[str]:
        """获取已启用的模型 | Get enabled models"""
        return [name for name, config in self.registry.items() if config['enabled']]

    def get_model_capabilities(self, model_name: str) -> List[str]:
        """获取模型能力 | Get model capabilities"""
        if model_name in self.registry:
            return self.registry[model_name].get('capabilities', [])
        return []

    def find_models_by_capability(self, capability: str) -> List[str]:
        """根据能力查找模型 | Find models by capability"""
        return [name for name, config in self.registry.items() 
                if capability in config.get('capabilities', []) and config['enabled']]

    def get_registry_summary(self) -> Dict[str, Any]:
        """获取注册表摘要 | Get registry summary"""
        enabled_count = len(self.get_enabled_models())
        external_count = len([config for config in self.registry.values() 
                             if config.get('type') == 'external' and config['enabled']])
        local_count = enabled_count - external_count
        
        # 计算健康状态统计 | Calculate health status statistics
        healthy_count = sum(1 for status in self.health_status.values() 
                           if status.get('status') == 'success')
        
        return {
            'total_models': len(self.registry),
            'enabled_models': enabled_count,
            'disabled_models': len(self.registry) - enabled_count,
            'local_models': local_count,
            'external_models': external_count,
            'healthy_models': healthy_count,
            'unhealthy_models': enabled_count - healthy_count,
            'last_update': datetime.now().isoformat()
        }

# 单例模式 | Singleton pattern
_model_registry_instance = None

def get_model_registry(config_path: Optional[str] = None) -> ModelRegistry:
    """获取模型注册表实例 | Get model registry instance"""
    global _model_registry_instance
    if _model_registry_instance is None:
        _model_registry_instance = ModelRegistry(config_path)
    return _model_registry_instance

def get_model_endpoint(model_name: str) -> Optional[Dict[str, Any]]:
    """获取模型端点信息 | Get model endpoint information"""
    registry = get_model_registry()
    return registry.get_model_info(model_name)

def get_all_models() -> List[str]:
    """获取所有模型名称 | Get all model names"""
    registry = get_model_registry()
    return registry.get_all_models()

if __name__ == '__main__':
    # 测试代码 | Test code
    registry = ModelRegistry()
    registry.load_registry()
    
    print("模型注册表摘要 | Model Registry Summary:")
    summary = registry.get_registry_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    print("\n所有模型 | All Models:")
    for model_name in registry.get_all_models():
        info = registry.get_model_info(model_name)
        print(f"{model_name}: {info['name']} - {info['description']}")
