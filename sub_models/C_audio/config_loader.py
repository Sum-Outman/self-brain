# -*- coding: utf-8 -*-
# Copyright 2025 The AGI Brain System Authors
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
配置文件加载器 | Configuration Loader
用于加载和管理音频处理模型的配置设置
Used to load and manage configuration settings for audio processing models
"""

import yaml
import os
from typing import Dict, Any, Optional
import logging
from pathlib import Path

class AudioConfigLoader:
    """
    音频配置加载器类 | Audio Configuration Loader Class
    负责加载、验证和管理音频处理模型的配置
    Responsible for loading, validating and managing audio processing model configurations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器 | Initialize configuration loader
        
        参数 Parameters:
        config_path: 配置文件路径 | Configuration file path
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "sub_models/C_audio/config.yaml"
        self.config = {}
        self.default_config = self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置 | Get default configuration
        
        返回 Returns:
        默认配置字典 | Default configuration dictionary
        """
        return {
            "model": {
                "name": "enhanced_audio_processing_model",
                "version": "1.0.0",
                "description": "增强型音频处理模型 | Enhanced audio processing model",
                "author": "AGI Brain System Team",
                "license": "Apache-2.0"
            },
            "audio": {
                "sample_rate": 16000,
                "frame_length": 1024,
                "hop_length": 256,
                "n_mels": 128,
                "n_fft": 2048,
                "max_audio_length": 30,
                "supported_formats": ["wav", "mp3", "flac", "ogg", "m4a"]
            },
            "realtime": {
                "enabled": True,
                "buffer_size": 10,
                "processing_interval_ms": 100,
                "max_latency_ms": 500,
                "auto_start": False
            },
            "speech_recognition": {
                "model": "facebook/wav2vec2-large-960h-lv60-self",
                "confidence_threshold": 0.7,
                "max_alternatives": 3,
                "language_detection": True,
                "supported_languages": ["zh", "en", "ja", "ko", "fr", "de", "es", "ru"]
            },
            "emotion_analysis": {
                "model": "superb/hubert-large-superb-er",
                "emotion_categories": ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"],
                "intensity_threshold": 0.3
            },
            "speech_synthesis": {
                "enabled": True,
                "default_voice": "default",
                "voice_types": ["default", "child", "elderly", "male", "female", "robot", "alien"],
                "emotion_types": ["neutral", "happy", "sad", "angry", "excited", "calm"],
                "sampling_rate": 22050
            },
            "music_recognition": {
                "enabled": True,
                "feature_extraction": True,
                "genres": ["pop", "rock", "jazz", "classical", "electronic", "hiphop", "country", "blues"],
                "bpm_range": {"min": 60, "max": 200}
            },
            "noise_analysis": {
                "enabled": True,
                "noise_types": ["background", "white", "pink", "brown", "impulse", "periodic"],
                "intensity_levels": ["low", "medium", "high"],
                "recommendation_enabled": True
            },
            "audio_effects": {
                "enabled": True,
                "supported_effects": ["echo", "reverb", "pitch_shift", "robot_voice", "chorus", 
                                    "alien_voice", "phaser", "flanger", "distortion", "compressor"],
                "default_parameters": {
                    "echo": {"delay": 0.5, "decay": 0.5},
                    "reverb": {"room_size": 0.7},
                    "pitch_shift": {"semitones": 2},
                    "robot_voice": {"intensity": 0.8}
                }
            },
            "external_apis": {
                "use_external_api": False,
                "apis": {
                    "google_speech": {
                        "enabled": False,
                        "name": "Google Speech-to-Text",
                        "url": "https://speech.googleapis.com/v1/speech:recognize",
                        "auth_method": "api_key",
                        "parameters": {
                            "languageCode": "zh-CN",
                            "model": "default",
                            "enableAutomaticPunctuation": True
                        }
                    },
                    "openai_whisper": {
                        "enabled": False,
                        "name": "OpenAI Whisper",
                        "url": "https://api.openai.com/v1/audio/transcriptions",
                        "auth_method": "bearer_token",
                        "parameters": {
                            "model": "whisper-1",
                            "response_format": "json"
                        }
                    }
                }
            },
            "performance": {
                "monitoring_enabled": True,
                "metrics": ["processing_time", "memory_usage", "recognition_accuracy", 
                           "synthesis_quality", "real_time_latency"],
                "sampling_interval": 60,
                "retention_period": 86400
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/audio_model.log",
                "max_file_size": 10485760,
                "backup_count": 5,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "output": {
                "directory": "output",
                "file_naming": "timestamp",
                "formats": {
                    "audio": ["wav", "mp3"],
                    "text": ["json", "txt"],
                    "visualization": ["png", "svg"]
                },
                "quality_presets": {
                    "low": {"bitrate": 64, "sample_rate": 16000},
                    "medium": {"bitrate": 128, "sample_rate": 22050},
                    "high": {"bitrate": 256, "sample_rate": 44100}
                }
            },
            "resources": {
                "max_memory_mb": 2048,
                "max_processing_threads": 4,
                "gpu_acceleration": True,
                "batch_processing": True,
                "batch_size": 8
            },
            "training": {
                "enabled": False,
                "datasets": {
                    "speech_recognition": {
                        "path": "data/speech",
                        "formats": ["wav", "txt"]
                    },
                    "emotion_analysis": {
                        "path": "data/emotion",
                        "formats": ["wav", "json"]
                    }
                },
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 16,
                    "epochs": 10,
                    "validation_split": 0.2
                },
                "augmentation": {
                    "enabled": True,
                    "techniques": ["noise_injection", "time_stretch", "pitch_shift", "speed_change"]
                }
            },
            "integration": {
                "main_system": {
                    "enabled": True,
                    "communication_protocol": "http",
                    "host": "localhost",
                    "port": 8000,
                    "endpoint": "/api/audio/process",
                    "timeout": 30
                },
                "data_bus": {
                    "enabled": True,
                    "topics": ["audio_input", "audio_output", "processing_status"],
                    "serialization_format": "json"
                },
                "web_interface": {
                    "enabled": True,
                    "port": 3000,
                    "realtime_updates": True
                }
            },
            "security": {
                "encryption": {
                    "enabled": False,
                    "algorithm": "AES-256",
                    "key_rotation": True,
                    "rotation_interval": 86400
                },
                "authentication": {
                    "enabled": False,
                    "method": "jwt",
                    "token_expiry": 3600
                },
                "data_privacy": {
                    "enabled": True,
                    "anonymization": False,
                    "retention_policy": "30d"
                }
            },
            "localization": {
                "default_language": "zh",
                "supported_languages": ["zh", "en"],
                "auto_detect": True,
                "fallback_language": "en"
            },
            "debug": {
                "enabled": False,
                "features": ["detailed_logging", "performance_profiling", 
                            "memory_profiling", "input_validation"],
                "log_level": "DEBUG"
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件 | Load configuration file
        
        返回 Returns:
        配置字典 | Configuration dictionary
        """
        try:
            # 检查配置文件是否存在
            if not os.path.exists(self.config_path):
                self.logger.warning(f"配置文件不存在，使用默认配置: {self.config_path} | Config file does not exist, using default config: {self.config_path}")
                self.config = self.default_config
                return self.config
            
            # 读取配置文件
            with open(self.config_path, 'r', encoding='utf-8') as file:
                loaded_config = yaml.safe_load(file)
            
            # 合并配置（使用默认配置作为基础，用文件配置覆盖）
            self.config = self._deep_merge(self.default_config, loaded_config)
            
            # 验证配置
            self._validate_config()
            
            self.logger.info(f"配置文件加载成功: {self.config_path} | Config file loaded successfully: {self.config_path}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"配置文件加载失败: {e} | Config file loading failed: {e}")
            self.config = self.default_config
            return self.config
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典 | Deep merge two dictionaries
        
        参数 Parameters:
        base: 基础字典 | Base dictionary
        update: 更新字典 | Update dictionary
        
        返回 Returns:
        合并后的字典 | Merged dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self) -> bool:
        """
        验证配置有效性 | Validate configuration validity
        
        返回 Returns:
        是否有效 | Whether valid
        """
        try:
            # 验证基本配置
            required_sections = ["model", "audio", "speech_recognition", "emotion_analysis"]
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"缺少必要配置节: {section} | Missing required config section: {section}")
            
            # 验证音频配置
            audio_config = self.config["audio"]
            if not isinstance(audio_config["sample_rate"], int) or audio_config["sample_rate"] <= 0:
                raise ValueError(f"无效的采样率: {audio_config['sample_rate']} | Invalid sample rate: {audio_config['sample_rate']}")
            
            # 验证实时处理配置
            realtime_config = self.config["realtime"]
            if not isinstance(realtime_config["buffer_size"], int) or realtime_config["buffer_size"] <= 0:
                raise ValueError(f"无效的缓冲区大小: {realtime_config['buffer_size']} | Invalid buffer size: {realtime_config['buffer_size']}")
            
            self.logger.info("配置验证通过 | Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {e} | Configuration validation failed: {e}")
            return False
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值 | Get configuration value
        
        参数 Parameters:
        key_path: 键路径（使用点号分隔）| Key path (using dot notation)
        default: 默认值 | Default value
        
        返回 Returns:
        配置值 | Configuration value
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception as e:
            self.logger.warning(f"获取配置值失败: {key_path}, 使用默认值: {default} | Failed to get config value: {key_path}, using default: {default}")
            return default
    
    def update_config_value(self, key_path: str, value: Any) -> bool:
        """
        更新配置值 | Update configuration value
        
        参数 Parameters:
        key_path: 键路径（使用点号分隔）| Key path (using dot notation)
        value: 新值 | New value
        
        返回 Returns:
        是否成功 | Whether successful
        """
        try:
            keys = key_path.split('.')
            config_ref = self.config
            
            # 遍历到最后一个键
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]
            
            # 设置值
            config_ref[keys[-1]] = value
            return True
            
        except Exception as e:
            self.logger.error(f"更新配置值失败: {key_path} | Failed to update config value: {key_path}")
            return False
    
    def save_config(self, file_path: Optional[str] = None) -> bool:
        """
        保存配置到文件 | Save configuration to file
        
        参数 Parameters:
        file_path: 文件路径 | File path
        
        返回 Returns:
        是否成功 | Whether successful
        """
        try:
            save_path = file_path or self.config_path
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存配置
            with open(save_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, allow_unicode=True, default_flow_style=False)
            
            self.logger.info(f"配置已保存到: {save_path} | Configuration saved to: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存配置失败: {e} | Failed to save configuration: {e}")
            return False
    
    def create_backup(self, backup_path: Optional[str] = None) -> bool:
        """
        创建配置备份 | Create configuration backup
        
        参数 Parameters:
        backup_path: 备份路径 | Backup path
        
        返回 Returns:
        是否成功 | Whether successful
        """
        try:
            if backup_path is None:
                # 生成默认备份路径
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = os.path.join(os.path.dirname(self.config_path), "backups")
                backup_path = os.path.join(backup_dir, f"config_backup_{timestamp}.yaml")
            
            # 确保备份目录存在
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # 保存备份
            with open(backup_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, allow_unicode=True, default_flow_style=False)
            
            self.logger.info(f"配置备份已创建: {backup_path} | Configuration backup created: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建配置备份失败: {e} | Failed to create configuration backup: {e}")
            return False
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """
        从备份恢复配置 | Restore configuration from backup
        
        参数 Parameters:
        backup_path: 备份路径 | Backup path
        
        返回 Returns:
        是否成功 | Whether successful
        """
        try:
            if not os.path.exists(backup_path):
                self.logger.error(f"备份文件不存在: {backup_path} | Backup file does not exist: {backup_path}")
                return False
            
            # 读取备份配置
            with open(backup_path, 'r', encoding='utf-8') as file:
                backup_config = yaml.safe_load(file)
            
            # 更新当前配置
            self.config = self._deep_merge(self.default_config, backup_config)
            
            # 保存到当前配置文件
            self.save_config()
            
            self.logger.info(f"配置已从备份恢复: {backup_path} | Configuration restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"从备份恢复配置失败: {e} | Failed to restore configuration from backup: {e}")
            return False

# 工具函数 | Utility functions
def load_audio_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载音频配置 | Load audio configuration
    
    参数 Parameters:
    config_path: 配置文件路径 | Configuration file path
    
    返回 Returns:
    配置字典 | Configuration dictionary
    """
    loader = AudioConfigLoader(config_path)
    return loader.load_config()

def get_audio_config_value(key_path: str, default: Any = None, 
                          config_path: Optional[str] = None) -> Any:
    """
    获取音频配置值 | Get audio configuration value
    
    参数 Parameters:
    key_path: 键路径 | Key path
    default: 默认值 | Default value
    config_path: 配置文件路径 | Configuration file path
    
    返回 Returns:
    配置值 | Configuration value
    """
    loader = AudioConfigLoader(config_path)
    loader.load_config()
    return loader.get_config_value(key_path, default)

def update_audio_config_value(key_path: str, value: Any, 
                             config_path: Optional[str] = None) -> bool:
    """
    更新音频配置值 | Update audio configuration value
    
    参数 Parameters:
    key_path: 键路径 | Key path
    value: 新值 | New value
    config_path: 配置文件路径 | Configuration file path
    
    返回 Returns:
    是否成功 | Whether successful
    """
    loader = AudioConfigLoader(config_path)
    loader.load_config()
    success = loader.update_config_value(key_path, value)
    if success:
        loader.save_config()
    return success

# 示例用法 | Example usage
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置加载器
    config_loader = AudioConfigLoader("sub_models/C_audio/config.yaml")
    
    # 加载配置
    config = config_loader.load_config()
    
    # 获取配置值
    sample_rate = config_loader.get_config_value("audio.sample_rate")
    print(f"采样率: {sample_rate} | Sample rate: {sample_rate}")
    
    # 更新配置值
    config_loader.update_config_value("audio.sample_rate", 44100)
    
    # 保存配置
    config_loader.save_config()
    
    # 创建备份
    config_loader.create_backup()
