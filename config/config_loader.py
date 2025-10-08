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

# 配置加载器 - Configuration Loader
# 负责系统配置的加载、验证和管理 - Responsible for loading, validating and managing system configuration

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

class ConfigLoader:
    """系统配置加载器类 - System configuration loader class"""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        初始化配置加载器 - Initialize configuration loader
        
        Args:
            config_path (str): 配置文件路径 | Configuration file path
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> bool:
        """
        加载配置文件 - Load configuration file
        
        Returns:
            bool: 是否成功加载 | Whether successfully loaded
        """
        try:
            if not os.path.exists(self.config_path):
                self.logger.error(f"配置文件不存在: {self.config_path} | Configuration file not found")
                return False
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                
            if self._validate_config():
                self.logger.info("配置文件加载成功 | Configuration file loaded successfully")
                return True
            else:
                self.logger.error("配置文件验证失败 | Configuration validation failed")
                return False
                
        except yaml.YAMLError as e:
            self.logger.error(f"YAML解析错误: {e} | YAML parsing error")
            return False
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e} | Failed to load configuration file")
            return False
    
    def _validate_config(self) -> bool:
        """
        验证配置完整性 - Validate configuration integrity
        
        Returns:
            bool: 配置是否有效 | Whether configuration is valid
        """
        required_sections = ['ports', 'models', 'training', 'performance']
        
        for section in required_sections:
            if section not in self.config:
                self.logger.error(f"缺少必需配置节: {section} | Missing required configuration section")
                return False
        
        # 验证端口配置 - Validate port configuration
        ports = self.config.get('ports', {})
        required_ports = ['manager', 'language', 'audio', 'image', 'video', 
                         'spatial', 'sensor', 'computer_control', 'knowledge', 
                         'motion', 'programming', 'web_frontend']
        
        for port_name in required_ports:
            if port_name not in ports:
                self.logger.error(f"缺少端口配置: {port_name} | Missing port configuration")
                return False
            if not isinstance(ports[port_name], int) or ports[port_name] <= 0:
                self.logger.error(f"无效的端口号: {port_name} = {ports[port_name]} | Invalid port number")
                return False
        
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值 - Get configuration value
        
        Args:
            key (str): 配置键，支持点分隔符（如 'ports.manager'） | Configuration key, supports dot notation
            default (Any): 默认值 | Default value
            
        Returns:
            Any: 配置值 | Configuration value
        """
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_port(self, model_name: str) -> Optional[int]:
        """
        获取模型端口号 - Get model port number
        
        Args:
            model_name (str): 模型名称 | Model name
            
        Returns:
            Optional[int]: 端口号，如果不存在返回None | Port number, None if not exists
        """
        port_key = f"ports.{model_name.lower()}"
        return self.get(port_key)
    
    def is_model_enabled(self, model_name: str) -> bool:
        """
        检查模型是否启用 - Check if model is enabled
        
        Args:
            model_name (str): 模型名称 | Model name
            
        Returns:
            bool: 是否启用 | Whether enabled
        """
        enabled_key = f"models.local_models.{model_name.upper()}"
        return self.get(enabled_key, False)
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        更新配置 - Update configuration
        
        Args:
            updates (Dict[str, Any]): 要更新的配置键值对 | Configuration key-value pairs to update
            
        Returns:
            bool: 是否成功更新 | Whether successfully updated
        """
        try:
            # 深度更新配置 - Deep update configuration
            for key, value in updates.items():
                keys = key.split('.')
                current = self.config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            
            # 保存更新后的配置 - Save updated configuration
            return self.save_config()
            
        except Exception as e:
            self.logger.error(f"更新配置失败: {e} | Failed to update configuration")
            return False
    
    def save_config(self) -> bool:
        """
        保存配置到文件 - Save configuration to file
        
        Returns:
            bool: 是否成功保存 | Whether successfully saved
        """
        try:
            # 确保配置目录存在 - Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info("配置文件保存成功 | Configuration file saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e} | Failed to save configuration file")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        获取配置字典 - Get configuration dictionary
        
        Returns:
            Dict[str, Any]: 配置字典 | Configuration dictionary
        """
        return self.config.copy()
    
    def to_json(self) -> str:
        """
        获取JSON格式配置 - Get configuration in JSON format
        
        Returns:
            str: JSON格式的配置 | Configuration in JSON format
        """
        return json.dumps(self.config, ensure_ascii=False, indent=2)
    
    def reload(self) -> bool:
        """
        重新加载配置 - Reload configuration
        
        Returns:
            bool: 是否成功重新加载 | Whether successfully reloaded
        """
        return self.load_config()

# 全局配置实例 - Global configuration instance
_config_loader = None

def get_config_loader(config_path: str = "config/system_config.yaml") -> ConfigLoader:
    """
    获取配置加载器实例 - Get configuration loader instance
    
    Args:
        config_path (str): 配置文件路径 | Configuration file path
        
    Returns:
        ConfigLoader: 配置加载器实例 | Configuration loader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
        _config_loader.load_config()
    return _config_loader

def get_config(key: str, default: Any = None) -> Any:
    """
    快速获取配置值 - Quick get configuration value
    
    Args:
        key (str): 配置键 | Configuration key
        default (Any): 默认值 | Default value
        
    Returns:
        Any: 配置值 | Configuration value
    """
    return get_config_loader().get(key, default)

def get_model_port(model_name: str) -> Optional[int]:
    """
    快速获取模型端口 - Quick get model port
    
    Args:
        model_name (str): 模型名称 | Model name
        
    Returns:
        Optional[int]: 端口号 | Port number
    """
    return get_config_loader().get_port(model_name)

# 测试代码 - Test code
if __name__ == '__main__':
    # 配置日志 - Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试配置加载 - Test configuration loading
    loader = ConfigLoader()
    if loader.load_config():
        print("配置加载测试成功 | Configuration loading test successful")
        
        # 测试获取配置 - Test getting configuration
        manager_port = loader.get_port('manager')
        print(f"管理模型端口: {manager_port} | Manager model port: {manager_port}")
        
        # 测试模型启用状态 - Test model enabled status
        is_audio_enabled = loader.is_model_enabled('C_audio')
        print(f"音频模型启用状态: {is_audio_enabled} | Audio model enabled: {is_audio_enabled}")
        
        # 输出JSON配置 - Output JSON configuration
        print("\n当前配置: | Current configuration:")
        print(loader.to_json())
    else:
        print("配置加载测试失败 | Configuration loading test failed")