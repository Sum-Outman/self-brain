# -*- coding: utf-8 -*-
"""
Training Configuration Manager
This module provides functionality to manage training configurations for different models.
"""

import logging
import json
import os
import copy
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import threading
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TrainingConfigManager')

class TrainingConfigManager:
    """Class for managing training configurations"""
    
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TrainingConfigManager, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the configuration manager"""
        # Base directory for configuration files
        self.base_config_dir = "d:/shiyan/web_interface/training_manager/configs"
        
        # Ensure the configuration directory exists
        if not os.path.exists(self.base_config_dir):
            try:
                os.makedirs(self.base_config_dir)
                logger.info(f"Created configuration directory: {self.base_config_dir}")
            except Exception as e:
                logger.error(f"Failed to create configuration directory: {str(e)}")
        
        # Dictionary to store configurations in memory
        self.configs = {}
        
        # Default configurations for different models
        self.default_configs = {
            'A': self._get_default_A_config(),
            'B': self._get_default_B_config(),
            'C': self._get_default_C_config(),
            'D': self._get_default_D_config(),
            'E': self._get_default_E_config(),
            'F': self._get_default_F_config(),
            'G': self._get_default_G_config(),
            'H': self._get_default_H_config(),
            'I': self._get_default_I_config(),
            'J': self._get_default_J_config(),
            'K': self._get_default_K_config()
        }
        
        # Load existing configurations
        self._load_all_configs()
    
    def _get_default_A_config(self) -> Dict[str, Any]:
        """Get default configuration for model A (Management Model)"""
        return {
            'model_id': 'A',
            'model_name': 'Management Model',
            'description': 'Main management AI model that coordinates other models',
            'architecture': {
                'type': 'transformer',
                'num_layers': 6,
                'hidden_size': 768,
                'num_heads': 12,
                'feedforward_size': 3072,
                'dropout_rate': 0.1
            },
            'training': {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.0001,
                'weight_decay': 0.01,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'warmup_steps': 1000,
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'monitor': 'val_loss'
                },
                'loss_function': 'cross_entropy',
                'metrics': ['accuracy', 'f1_score'],
                'gradient_clipping': 1.0,
                'mixed_precision': True
            },
            'data': {
                'train_data_path': 'data/train/A/',
                'val_data_path': 'data/val/A/',
                'test_data_path': 'data/test/A/',
                'from_scratch': True,
                'input_size': 512,
                'preprocessing': {
                    'normalize': True,
                    'shuffle': True,
                    'augment': False
                },
                'data_loader': {
                    'num_workers': 4,
                    'pin_memory': True
                }
            },
            'device': {
                'use_gpu': True,
                'device_ids': [0],
                'dtype': 'float32'
            },
            'checkpointing': {
                'enabled': True,
                'save_dir': 'checkpoints/A/',
                'save_frequency': 1,
                'keep_best_only': True,
                'save_weights_only': False
            },
            'logging': {
                'enabled': True,
                'log_dir': 'logs/A/',
                'log_frequency': 10,
                'use_tensorboard': True
            },
            'knowledge_integration': {
                'enabled': True,
                'knowledge_model_id': 'J',
                'integration_strategy': 'attention_based'
            },
            'external_api': {
                'enabled': False,
                'provider': 'openai',
                'api_key': '',
                'api_base': '',
                'model_name': 'gpt-4'
            },
            'emotion_analysis': {
                'enabled': True,
                'emotion_categories': ['happy', 'sad', 'angry', 'fearful', 'surprised', 'neutral']
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_default_B_config(self) -> Dict[str, Any]:
        """Get default configuration for model B (Language Model)"""
        return {
            'model_id': 'B',
            'model_name': 'Language Model',
            'description': 'Large language model with multilingual capabilities and emotional reasoning',
            'architecture': {
                'type': 'gpt_like',
                'num_layers': 12,
                'hidden_size': 768,
                'num_heads': 12,
                'feedforward_size': 3072,
                'vocab_size': 50257,
                'max_position_embeddings': 1024,
                'dropout_rate': 0.1
            },
            'training': {
                'batch_size': 16,
                'epochs': 50,
                'learning_rate': 0.00005,
                'weight_decay': 0.01,
                'optimizer': 'adamw',
                'scheduler': 'cosine_with_restarts',
                'warmup_steps': 2000,
                'early_stopping': {
                    'enabled': True,
                    'patience': 8,
                    'monitor': 'val_perplexity'
                },
                'loss_function': 'cross_entropy',
                'metrics': ['perplexity', 'accuracy'],
                'gradient_clipping': 1.0,
                'mixed_precision': True
            },
            'data': {
                'train_data_path': 'data/train/B/',
                'val_data_path': 'data/val/B/',
                'test_data_path': 'data/test/B/',
                'from_scratch': True,
                'preprocessing': {
                    'lowercase': False,
                    'remove_punctuation': False,
                    'tokenize': True,
                    'max_length': 1024
                },
                'data_loader': {
                    'num_workers': 8,
                    'pin_memory': True
                }
            },
            'device': {
                'use_gpu': True,
                'device_ids': [0],
                'dtype': 'float32'
            },
            'checkpointing': {
                'enabled': True,
                'save_dir': 'checkpoints/B/',
                'save_frequency': 1,
                'keep_best_only': True,
                'save_weights_only': False
            },
            'logging': {
                'enabled': True,
                'log_dir': 'logs/B/',
                'log_frequency': 50,
                'use_tensorboard': True
            },
            'knowledge_integration': {
                'enabled': True,
                'knowledge_model_id': 'J',
                'integration_strategy': 'context_enrichment'
            },
            'external_api': {
                'enabled': False,
                'provider': 'openai',
                'api_key': '',
                'api_base': '',
                'model_name': 'gpt-3.5-turbo'
            },
            'multilingual': {
                'enabled': True,
                'supported_languages': ['en', 'zh', 'es', 'fr', 'de', 'ja', 'ko', 'ru']
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_default_C_config(self) -> Dict[str, Any]:
        """Get default configuration for model C (Audio Processing Model)"""
        return {
            'model_id': 'C',
            'model_name': 'Audio Processing Model',
            'description': 'Model for audio recognition, synthesis, and processing',
            'architecture': {
                'type': 'cnn_rnn',
                'cnn_layers': 4,
                'rnn_layers': 2,
                'hidden_size': 256,
                'dropout_rate': 0.2,
                'num_classes': 30
            },
            'training': {
                'batch_size': 64,
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'optimizer': 'adam',
                'scheduler': 'step',
                'step_size': 20,
                'gamma': 0.5,
                'early_stopping': {
                    'enabled': True,
                    'patience': 15,
                    'monitor': 'val_accuracy'
                },
                'loss_function': 'cross_entropy',
                'metrics': ['accuracy', 'precision', 'recall'],
                'gradient_clipping': 2.0
            },
            'data': {
                'train_data_path': 'data/train/C/',
                'val_data_path': 'data/val/C/',
                'test_data_path': 'data/test/C/',
                'from_scratch': True,
                'sample_rate': 16000,
                'n_mfcc': 13,
                'max_duration': 10.0,  # seconds
                'preprocessing': {
                    'normalize_volume': True,
                    'noise_reduction': True,
                    'augment': True
                },
                'data_loader': {
                    'num_workers': 4,
                    'pin_memory': True
                }
            },
            'device': {
                'use_gpu': True,
                'device_ids': [0],
                'dtype': 'float32'
            },
            'checkpointing': {
                'enabled': True,
                'save_dir': 'checkpoints/C/',
                'save_frequency': 5,
                'keep_best_only': True,
                'save_weights_only': True
            },
            'logging': {
                'enabled': True,
                'log_dir': 'logs/C/',
                'log_frequency': 10,
                'use_tensorboard': True
            },
            'knowledge_integration': {
                'enabled': True,
                'knowledge_model_id': 'J',
                'integration_strategy': 'feature_enhancement'
            },
            'external_api': {
                'enabled': False,
                'provider': 'google',
                'api_key': '',
                'api_base': '',
                'model_name': 'cloud_speech_to_text'
            },
            'audio_processing': {
                'speech_recognition': {
                    'enabled': True,
                    'languages': ['en-US', 'zh-CN']
                },
                'speech_synthesis': {
                    'enabled': True,
                    'voices': ['male', 'female']
                },
                'music_recognition': {
                    'enabled': True
                },
                'noise_reduction': {
                    'enabled': True,
                    'strength': 0.7
                }
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_default_D_config(self) -> Dict[str, Any]:
        """Get default configuration for model D (Image Processing Model)"""
        return {
            'model_id': 'D',
            'model_name': 'Image Processing Model',
            'description': 'Model for image recognition, modification, and generation',
            'architecture': {
                'type': 'resnet',
                'version': 'resnet50',
                'num_classes': 1000,
                'pretrained': False,
                'dropout_rate': 0.2
            },
            'training': {
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'optimizer': 'sgd',
                'momentum': 0.9,
                'scheduler': 'cosine',
                'warmup_steps': 1000,
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'monitor': 'val_accuracy'
                },
                'loss_function': 'cross_entropy',
                'metrics': ['accuracy', 'precision', 'recall'],
                'gradient_clipping': 5.0
            },
            'data': {
                'train_data_path': 'data/train/D/',
                'val_data_path': 'data/val/D/',
                'test_data_path': 'data/test/D/',
                'from_scratch': True,
                'image_size': (224, 224),
                'preprocessing': {
                    'normalize': True,
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'augment': True,
                    'rotation_range': 30,
                    'horizontal_flip': True,
                    'vertical_flip': False
                },
                'data_loader': {
                    'num_workers': 8,
                    'pin_memory': True
                }
            },
            'device': {
                'use_gpu': True,
                'device_ids': [0],
                'dtype': 'float32'
            },
            'checkpointing': {
                'enabled': True,
                'save_dir': 'checkpoints/D/',
                'save_frequency': 5,
                'keep_best_only': True,
                'save_weights_only': True
            },
            'logging': {
                'enabled': True,
                'log_dir': 'logs/D/',
                'log_frequency': 20,
                'use_tensorboard': True
            },
            'knowledge_integration': {
                'enabled': True,
                'knowledge_model_id': 'J',
                'integration_strategy': 'semantic_guidance'
            },
            'external_api': {
                'enabled': False,
                'provider': 'openai',
                'api_key': '',
                'api_base': '',
                'model_name': 'dall-e-2'
            },
            'image_processing': {
                'object_detection': {
                    'enabled': True,
                    'max_objects': 20
                },
                'image_generation': {
                    'enabled': True,
                    'resolution': '1024x1024'
                },
                'image_enhancement': {
                    'enabled': True,
                    'max_upscale': 4
                }
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_default_E_config(self) -> Dict[str, Any]:
        """Get default configuration for model E (Video Processing Model)"""
        return {
            'model_id': 'E',
            'model_name': 'Video Processing Model',
            'description': 'Model for video content recognition, editing, and generation',
            'architecture': {
                'type': '3d_cnn',
                'num_classes': 200,
                'dropout_rate': 0.3,
                'pretrained': False
            },
            'training': {
                'batch_size': 8,
                'epochs': 30,
                'learning_rate': 0.0001,
                'weight_decay': 0.0001,
                'optimizer': 'adam',
                'scheduler': 'cosine',
                'early_stopping': {
                    'enabled': True,
                    'patience': 8,
                    'monitor': 'val_accuracy'
                },
                'loss_function': 'cross_entropy',
                'metrics': ['accuracy', 'precision', 'recall'],
                'gradient_clipping': 5.0
            },
            'data': {
                'train_data_path': 'data/train/E/',
                'val_data_path': 'data/val/E/',
                'test_data_path': 'data/test/E/',
                'from_scratch': True,
                'frame_size': (224, 224),
                'num_frames': 16,
                'fps': 30,
                'preprocessing': {
                    'normalize': True,
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'augment': True
                },
                'data_loader': {
                    'num_workers': 4,
                    'pin_memory': True
                }
            },
            'device': {
                'use_gpu': True,
                'device_ids': [0],
                'dtype': 'float32'
            },
            'checkpointing': {
                'enabled': True,
                'save_dir': 'checkpoints/E/',
                'save_frequency': 3,
                'keep_best_only': True,
                'save_weights_only': True
            },
            'logging': {
                'enabled': True,
                'log_dir': 'logs/E/',
                'log_frequency': 10,
                'use_tensorboard': True
            },
            'knowledge_integration': {
                'enabled': True,
                'knowledge_model_id': 'J',
                'integration_strategy': 'temporal_semantic_guidance'
            },
            'external_api': {
                'enabled': False,
                'provider': 'google',
                'api_key': '',
                'api_base': '',
                'model_name': 'cloud_video_intelligence'
            },
            'video_processing': {
                'action_recognition': {
                    'enabled': True
                },
                'video_editing': {
                    'enabled': True,
                    'supported_formats': ['mp4', 'avi', 'mov']
                },
                'video_generation': {
                    'enabled': True,
                    'max_duration': 60
                }
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_default_F_config(self) -> Dict[str, Any]:
        """Get default configuration for model F (Binocular Spatial Model)"""
        return {
            'model_id': 'F',
            'model_name': 'Binocular Spatial Model',
            'description': 'Model for spatial recognition, 3D modeling, and depth perception',
            'architecture': {
                'type': 'stereo_cnn',
                'disparity_range': 64,
                'dropout_rate': 0.2
            },
            'training': {
                'batch_size': 16,
                'epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'optimizer': 'adam',
                'scheduler': 'step',
                'step_size': 15,
                'gamma': 0.5,
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'monitor': 'val_loss'
                },
                'loss_function': 'l1_loss',
                'metrics': ['epe', 'd1'],
                'gradient_clipping': 2.0
            },
            'data': {
                'train_data_path': 'data/train/F/',
                'val_data_path': 'data/val/F/',
                'test_data_path': 'data/test/F/',
                'from_scratch': True,
                'image_size': (512, 512),
                'preprocessing': {
                    'normalize': True,
                    'augment': True
                },
                'data_loader': {
                    'num_workers': 4,
                    'pin_memory': True
                }
            },
            'device': {
                'use_gpu': True,
                'device_ids': [0],
                'dtype': 'float32'
            },
            'checkpointing': {
                'enabled': True,
                'save_dir': 'checkpoints/F/',
                'save_frequency': 5,
                'keep_best_only': True,
                'save_weights_only': True
            },
            'logging': {
                'enabled': True,
                'log_dir': 'logs/F/',
                'log_frequency': 10,
                'use_tensorboard': True
            },
            'knowledge_integration': {
                'enabled': True,
                'knowledge_model_id': 'J',
                'integration_strategy': 'geometric_knowledge_injection'
            },
            'external_api': {
                'enabled': False,
                'provider': 'microsoft',
                'api_key': '',
                'api_base': '',
                'model_name': 'azure_spatial_anchors'
            },
            'spatial_processing': {
                'depth_estimation': {
                    'enabled': True,
                    'max_depth': 10.0  # meters
                },
                '3d_reconstruction': {
                    'enabled': True,
                    'quality': 'medium'
                },
                'object_tracking': {
                    'enabled': True,
                    'max_objects': 10
                }
            },
            'camera_config': {
                'left_camera_id': 0,
                'right_camera_id': 1,
                'calibration_file': 'config/camera_calibration.json'
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_default_G_config(self) -> Dict[str, Any]:
        """Get default configuration for model G (Sensor Model)"""
        return {
            'model_id': 'G',
            'model_name': 'Sensor Model',
            'description': 'Model for sensor data processing and analysis',
            'architecture': {
                'type': 'lstm',
                'num_layers': 3,
                'hidden_size': 128,
                'dropout_rate': 0.2
            },
            'training': {
                'batch_size': 64,
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'optimizer': 'adam',
                'scheduler': 'step',
                'step_size': 20,
                'gamma': 0.5,
                'early_stopping': {
                    'enabled': True,
                    'patience': 15,
                    'monitor': 'val_loss'
                },
                'loss_function': 'mse_loss',
                'metrics': ['mae', 'rmse'],
                'gradient_clipping': 1.0
            },
            'data': {
                'train_data_path': 'data/train/G/',
                'val_data_path': 'data/val/G/',
                'test_data_path': 'data/test/G/',
                'from_scratch': True,
                'sample_rate': 100,
                'window_size': 100,
                'preprocessing': {
                    'normalize': True,
                    'impute_missing': True,
                    'outlier_removal': True
                },
                'data_loader': {
                    'num_workers': 2,
                    'pin_memory': True
                }
            },
            'device': {
                'use_gpu': True,
                'device_ids': [0],
                'dtype': 'float32'
            },
            'checkpointing': {
                'enabled': True,
                'save_dir': 'checkpoints/G/',
                'save_frequency': 10,
                'keep_best_only': True,
                'save_weights_only': True
            },
            'logging': {
                'enabled': True,
                'log_dir': 'logs/G/',
                'log_frequency': 50,
                'use_tensorboard': True
            },
            'knowledge_integration': {
                'enabled': True,
                'knowledge_model_id': 'J',
                'integration_strategy': 'physical_model_integration'
            },
            'external_api': {
                'enabled': False,
                'provider': 'aws',
                'api_key': '',
                'api_base': '',
                'model_name': 'iot_analytics'
            },
            'sensors': {
                'temperature': {
                    'enabled': True,
                    'units': 'celsius',
                    'range': [-40, 125]
                },
                'humidity': {
                    'enabled': True,
                    'units': 'percentage',
                    'range': [0, 100]
                },
                'accelerometer': {
                    'enabled': True,
                    'units': 'g',
                    'range': [-16, 16]
                },
                'gyroscope': {
                    'enabled': True,
                    'units': 'dps',
                    'range': [-2000, 2000]
                },
                'pressure': {
                    'enabled': True,
                    'units': 'hpa',
                    'range': [300, 1100]
                },
                'light': {
                    'enabled': True,
                    'units': 'lux',
                    'range': [0, 100000]
                },
                'distance': {
                    'enabled': True,
                    'units': 'cm',
                    'range': [0, 1000]
                }
            },
            'serial_config': {
                'port': 'COM3',
                'baud_rate': 115200,
                'timeout': 0.1
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_default_H_config(self) -> Dict[str, Any]:
        """Get default configuration for model H (Computer Control Model)"""
        return {
            'model_id': 'H',
            'model_name': 'Computer Control Model',
            'description': 'Model for controlling computer operations and systems',
            'architecture': {
                'type': 'transformer',
                'num_layers': 4,
                'hidden_size': 512,
                'num_heads': 8,
                'dropout_rate': 0.1
            },
            'training': {
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.0001,
                'weight_decay': 0.0001,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'monitor': 'val_accuracy'
                },
                'loss_function': 'cross_entropy',
                'metrics': ['accuracy'],
                'gradient_clipping': 1.0
            },
            'data': {
                'train_data_path': 'data/train/H/',
                'val_data_path': 'data/val/H/',
                'test_data_path': 'data/test/H/',
                'from_scratch': True,
                'preprocessing': {
                    'normalize': True,
                    'shuffle': True
                },
                'data_loader': {
                    'num_workers': 4,
                    'pin_memory': True
                }
            },
            'device': {
                'use_gpu': True,
                'device_ids': [0],
                'dtype': 'float32'
            },
            'checkpointing': {
                'enabled': True,
                'save_dir': 'checkpoints/H/',
                'save_frequency': 5,
                'keep_best_only': True,
                'save_weights_only': True
            },
            'logging': {
                'enabled': True,
                'log_dir': 'logs/H/',
                'log_frequency': 20,
                'use_tensorboard': True
            },
            'knowledge_integration': {
                'enabled': True,
                'knowledge_model_id': 'J',
                'integration_strategy': 'command_knowledge_injection'
            },
            'external_api': {
                'enabled': False,
                'provider': 'none',
                'api_key': '',
                'api_base': '',
                'model_name': ''
            },
            'system_compatibility': {
                'windows': {
                    'enabled': True,
                    'version': '10'
                },
                'linux': {
                    'enabled': True,
                    'distro': 'ubuntu'
                },
                'macos': {
                    'enabled': True,
                    'version': '12.0'
                }
            },
            'security': {
                'enabled': True,
                'sandboxing': True,
                'allowed_commands': ['file_operations', 'system_info', 'process_management', 'network_basic']
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_default_I_config(self) -> Dict[str, Any]:
        """Get default configuration for model I (Motion Control Model)"""
        return {
            'model_id': 'I',
            'model_name': 'Motion Control Model',
            'description': 'Model for controlling motion and actuators',
            'architecture': {
                'type': 'rnn',
                'num_layers': 3,
                'hidden_size': 256,
                'dropout_rate': 0.2
            },
            'training': {
                'batch_size': 64,
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'optimizer': 'adam',
                'scheduler': 'step',
                'step_size': 20,
                'gamma': 0.5,
                'early_stopping': {
                    'enabled': True,
                    'patience': 15,
                    'monitor': 'val_loss'
                },
                'loss_function': 'mse_loss',
                'metrics': ['mae', 'rmse'],
                'gradient_clipping': 2.0
            },
            'data': {
                'train_data_path': 'data/train/I/',
                'val_data_path': 'data/val/I/',
                'test_data_path': 'data/test/I/',
                'from_scratch': True,
                'preprocessing': {
                    'normalize': True,
                    'shuffle': True
                },
                'data_loader': {
                    'num_workers': 4,
                    'pin_memory': True
                }
            },
            'device': {
                'use_gpu': True,
                'device_ids': [0],
                'dtype': 'float32'
            },
            'checkpointing': {
                'enabled': True,
                'save_dir': 'checkpoints/I/',
                'save_frequency': 10,
                'keep_best_only': True,
                'save_weights_only': True
            },
            'logging': {
                'enabled': True,
                'log_dir': 'logs/I/',
                'log_frequency': 50,
                'use_tensorboard': True
            },
            'knowledge_integration': {
                'enabled': True,
                'knowledge_model_id': 'J',
                'integration_strategy': 'kinematic_model_integration'
            },
            'external_api': {
                'enabled': False,
                'provider': 'none',
                'api_key': '',
                'api_base': '',
                'model_name': ''
            },
            'actuator_control': {
                'serial_ports': {
                    'COM1': {
                        'enabled': True,
                        'baud_rate': 9600,
                        'timeout': 0.1
                    },
                    'COM2': {
                        'enabled': False,
                        'baud_rate': 115200,
                        'timeout': 0.1
                    }
                },
                'pwm_channels': {
                    'channel_1': {
                        'enabled': True,
                        'min_value': 0,
                        'max_value': 100
                    },
                    'channel_2': {
                        'enabled': True,
                        'min_value': 0,
                        'max_value': 100
                    }
                },
                'digital_outputs': {
                    'output_1': {
                        'enabled': True
                    },
                    'output_2': {
                        'enabled': True
                    }
                }
            },
            'motion_planning': {
                'enabled': True,
                'max_velocity': 1.0,
                'max_acceleration': 0.5
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_default_J_config(self) -> Dict[str, Any]:
        """Get default configuration for model J (Knowledge Base Model)"""
        return {
            'model_id': 'J',
            'model_name': 'Knowledge Base Model',
            'description': 'Expert knowledge model with comprehensive domain knowledge',
            'architecture': {
                'type': 'knowledge_graph',
                'embedding_dim': 128,
                'dropout_rate': 0.1
            },
            'training': {
                'batch_size': 128,
                'epochs': 200,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'optimizer': 'adam',
                'scheduler': 'cosine',
                'early_stopping': {
                    'enabled': True,
                    'patience': 20,
                    'monitor': 'val_loss'
                },
                'loss_function': 'contrastive_loss',
                'metrics': ['accuracy', 'f1_score'],
                'gradient_clipping': 1.0
            },
            'data': {
                'train_data_path': 'data/train/J/',
                'val_data_path': 'data/val/J/',
                'test_data_path': 'data/test/J/',
                'from_scratch': True,
                'preprocessing': {
                    'normalize': True,
                    'shuffle': True
                },
                'data_loader': {
                    'num_workers': 8,
                    'pin_memory': True
                }
            },
            'device': {
                'use_gpu': True,
                'device_ids': [0],
                'dtype': 'float32'
            },
            'checkpointing': {
                'enabled': True,
                'save_dir': 'checkpoints/J/',
                'save_frequency': 10,
                'keep_best_only': True,
                'save_weights_only': True
            },
            'logging': {
                'enabled': True,
                'log_dir': 'logs/J/',
                'log_frequency': 100,
                'use_tensorboard': True
            },
            'knowledge_integration': {
                'enabled': True,
                'self_learning': {
                    'enabled': True,
                    'learning_rate': 0.0001,
                    'update_frequency': 100
                }
            },
            'external_api': {
                'enabled': False,
                'provider': 'wolfram_alpha',
                'api_key': '',
                'api_base': '',
                'model_name': 'knowledge_engine'
            },
            'knowledge_domains': {
                'physics': {
                    'enabled': True,
                    'subdomains': ['classical', 'quantum', 'relativity']
                },
                'mathematics': {
                    'enabled': True,
                    'subdomains': ['algebra', 'calculus', 'geometry', 'statistics']
                },
                'chemistry': {
                    'enabled': True,
                    'subdomains': ['organic', 'inorganic', 'physical']
                },
                'biology': {
                    'enabled': True,
                    'subdomains': ['molecular', 'cellular', 'ecology']
                },
                'medicine': {
                    'enabled': True,
                    'subdomains': ['anatomy', 'physiology', 'pharmacology']
                },
                'engineering': {
                    'enabled': True,
                    'subdomains': ['mechanical', 'electrical', 'civil', 'computer']
                },
                'computer_science': {
                    'enabled': True,
                    'subdomains': ['algorithms', 'data_structures', 'ai']
                },
                'economics': {
                    'enabled': True,
                    'subdomains': ['microeconomics', 'macroeconomics']
                },
                'history': {
                    'enabled': True,
                    'subdomains': ['ancient', 'medieval', 'modern']
                },
                'psychology': {
                    'enabled': True,
                    'subdomains': ['cognitive', 'developmental', 'clinical']
                }
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_default_K_config(self) -> Dict[str, Any]:
        """Get default configuration for model K (Programming Model)"""
        return {
            'model_id': 'K',
            'model_name': 'Programming Model',
            'description': 'Model for code generation, completion, and debugging',
            'architecture': {
                'type': 'code_transformer',
                'num_layers': 8,
                'hidden_size': 768,
                'num_heads': 12,
                'feedforward_size': 3072,
                'vocab_size': 30000,
                'dropout_rate': 0.1
            },
            'training': {
                'batch_size': 16,
                'epochs': 50,
                'learning_rate': 0.00005,
                'weight_decay': 0.01,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'monitor': 'val_loss'
                },
                'loss_function': 'cross_entropy',
                'metrics': ['accuracy', 'perplexity'],
                'gradient_clipping': 1.0,
                'mixed_precision': True
            },
            'data': {
                'train_data_path': 'data/train/K/',
                'val_data_path': 'data/val/K/',
                'test_data_path': 'data/test/K/',
                'from_scratch': True,
                'max_length': 1024,
                'preprocessing': {
                    'tokenize': True,
                    'shuffle': True
                },
                'data_loader': {
                    'num_workers': 8,
                    'pin_memory': True
                }
            },
            'device': {
                'use_gpu': True,
                'device_ids': [0],
                'dtype': 'float32'
            },
            'checkpointing': {
                'enabled': True,
                'save_dir': 'checkpoints/K/',
                'save_frequency': 5,
                'keep_best_only': True,
                'save_weights_only': True
            },
            'logging': {
                'enabled': True,
                'log_dir': 'logs/K/',
                'log_frequency': 50,
                'use_tensorboard': True
            },
            'knowledge_integration': {
                'enabled': True,
                'knowledge_model_id': 'J',
                'integration_strategy': 'code_knowledge_enrichment'
            },
            'external_api': {
                'enabled': False,
                'provider': 'openai',
                'api_key': '',
                'api_base': '',
                'model_name': 'code-davinci-002'
            },
            'programming_languages': {
                'python': {
                    'enabled': True,
                    'version': '3.9'
                },
                'javascript': {
                    'enabled': True,
                    'version': 'es6'
                },
                'java': {
                    'enabled': True,
                    'version': '11'
                },
                'csharp': {
                    'enabled': True,
                    'version': '9.0'
                },
                'cpp': {
                    'enabled': True,
                    'version': 'c++17'
                },
                'go': {
                    'enabled': True,
                    'version': '1.17'
                },
                'rust': {
                    'enabled': True,
                    'version': '1.56'
                },
                'sql': {
                    'enabled': True,
                    'dialects': ['mysql', 'postgresql', 'sqlite']
                }
            },
            'code_tools': {
                'linting': {
                    'enabled': True
                },
                'testing': {
                    'enabled': True
                },
                'debugging': {
                    'enabled': True
                },
                'refactoring': {
                    'enabled': True
                }
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _load_all_configs(self) -> None:
        """Load all configurations from disk"""
        try:
            # Check if the configuration directory exists
            if not os.path.exists(self.base_config_dir):
                logger.warning(f"Configuration directory {self.base_config_dir} does not exist")
                return
            
            # Get all model IDs from default configs
            model_ids = list(self.default_configs.keys())
            
            # Load configuration for each model
            for model_id in model_ids:
                config_path = os.path.join(self.base_config_dir, f"{model_id}_config.json")
                
                # If configuration file exists, load it
                if os.path.isfile(config_path):
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            self.configs[model_id] = config
                            logger.info(f"Loaded configuration for model {model_id} from {config_path}")
                    except Exception as e:
                        logger.error(f"Failed to load configuration for model {model_id}: {str(e)}")
                        # Use default configuration if loading fails
                        self.configs[model_id] = copy.deepcopy(self.default_configs[model_id])
                else:
                    # Use default configuration if file doesn't exist
                    self.configs[model_id] = copy.deepcopy(self.default_configs[model_id])
                    logger.info(f"Using default configuration for model {model_id}")
                    # Save the default configuration to disk
                    self.save_config(model_id)
        except Exception as e:
            logger.error(f"Error loading all configurations: {str(e)}")
    
    def get_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.configs and model_id not in self.default_configs:
                logger.error(f"Invalid model ID: {model_id}")
                return None
            
            # If configuration doesn't exist, create it from default
            if model_id not in self.configs:
                self.configs[model_id] = copy.deepcopy(self.default_configs[model_id])
                self.save_config(model_id)
                
            return copy.deepcopy(self.configs[model_id])
    
    def update_config(self, model_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update configuration for a specific model"""
        with self._lock:
            try:
                # Check if model ID is valid
                if model_id not in self.configs and model_id not in self.default_configs:
                    logger.error(f"Invalid model ID: {model_id}")
                    return False
                
                # If configuration doesn't exist, create it from default
                if model_id not in self.configs:
                    self.configs[model_id] = copy.deepcopy(self.default_configs[model_id])
                
                # Apply updates recursively
                self._update_nested_dict(self.configs[model_id], config_updates)
                
                # Update last_updated timestamp
                self.configs[model_id]['last_updated'] = datetime.now().isoformat()
                
                # Save the updated configuration to disk
                return self.save_config(model_id)
            except Exception as e:
                logger.error(f"Failed to update configuration for model {model_id}: {str(e)}")
                return False
    
    def _update_nested_dict(self, original: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Update a nested dictionary with new values"""
        for key, value in updates.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                # If both are dictionaries, update recursively
                self._update_nested_dict(original[key], value)
            else:
                # Otherwise, replace the value
                original[key] = value
    
    def save_config(self, model_id: str) -> bool:
        """Save configuration for a specific model to disk"""
        try:
            # Check if model ID is valid
            if model_id not in self.configs:
                logger.error(f"No configuration found for model {model_id}")
                return False
            
            # Ensure the configuration directory exists
            if not os.path.exists(self.base_config_dir):
                os.makedirs(self.base_config_dir)
            
            # Save configuration to file
            config_path = os.path.join(self.base_config_dir, f"{model_id}_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.configs[model_id], f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved configuration for model {model_id} to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration for model {model_id}: {str(e)}")
            return False
    
    def reset_config(self, model_id: str) -> bool:
        """Reset configuration for a specific model to default"""
        with self._lock:
            try:
                # Check if model ID is valid
                if model_id not in self.default_configs:
                    logger.error(f"Invalid model ID: {model_id}")
                    return False
                
                # Reset configuration to default
                self.configs[model_id] = copy.deepcopy(self.default_configs[model_id])
                
                # Save the reset configuration to disk
                return self.save_config(model_id)
            except Exception as e:
                logger.error(f"Failed to reset configuration for model {model_id}: {str(e)}")
                return False
    
    def export_config(self, model_id: str, export_path: str) -> bool:
        """Export configuration for a specific model to a file"""
        try:
            # Get configuration
            config = self.get_config(model_id)
            if config is None:
                return False
            
            # Ensure the export directory exists
            export_dir = os.path.dirname(export_path)
            if export_dir and not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # Export configuration to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Exported configuration for model {model_id} to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export configuration for model {model_id}: {str(e)}")
            return False
    
    def import_config(self, model_id: str, import_path: str) -> bool:
        """Import configuration for a specific model from a file"""
        try:
            # Check if import file exists
            if not os.path.isfile(import_path):
                logger.error(f"Import file does not exist: {import_path}")
                return False
            
            # Load configuration from file
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            # Validate imported configuration
            if not isinstance(imported_config, dict) or 'model_id' not in imported_config:
                logger.error(f"Invalid configuration format in {import_path}")
                return False
            
            # Ensure model ID matches
            if imported_config['model_id'] != model_id:
                logger.warning(f"Model ID in configuration ({imported_config['model_id']}) does not match target model ID ({model_id})")
                # Update model ID to match target
                imported_config['model_id'] = model_id
            
            # Update the configuration
            with self._lock:
                self.configs[model_id] = imported_config
                # Update last_updated timestamp
                self.configs[model_id]['last_updated'] = datetime.now().isoformat()
                # Save the imported configuration to disk
                return self.save_config(model_id)
        except Exception as e:
            logger.error(f"Failed to import configuration for model {model_id}: {str(e)}")
            return False
    
    def validate_config(self, model_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration for a specific model"""
        try:
            # Check if model ID is valid
            if model_id not in self.default_configs:
                return {
                    'is_valid': False,
                    'message': f"Invalid model ID: {model_id}"
                }
            
            # Get default configuration for comparison
            default_config = self.default_configs[model_id]
            
            # Check required fields
            required_fields = ['model_id', 'model_name', 'training', 'data', 'device']
            for field in required_fields:
                if field not in config:
                    return {
                        'is_valid': False,
                        'message': f"Required field '{field}' is missing"
                    }
            
            # Check model ID matches
            if config['model_id'] != model_id:
                return {
                    'is_valid': False,
                    'message': f"Model ID in configuration ({config['model_id']}) does not match target model ID ({model_id})"
                }
            
            # Check training configuration
            if 'batch_size' not in config['training'] or config['training']['batch_size'] <= 0:
                return {
                    'is_valid': False,
                    'message': "Invalid batch size in training configuration"
                }
            
            if 'epochs' not in config['training'] or config['training']['epochs'] <= 0:
                return {
                    'is_valid': False,
                    'message': "Invalid number of epochs in training configuration"
                }
            
            # Check device configuration
            if 'use_gpu' not in config['device']:
                return {
                    'is_valid': False,
                    'message': "'use_gpu' field is missing in device configuration"
                }
            
            # Configuration is valid
            return {
                'is_valid': True,
                'message': "Configuration is valid"
            }
        except Exception as e:
            logger.error(f"Failed to validate configuration for model {model_id}: {str(e)}")
            return {
                'is_valid': False,
                'message': str(e)
            }
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all configurations"""
        with self._lock:
            # Ensure all configurations are loaded
            for model_id in self.default_configs.keys():
                if model_id not in self.configs:
                    self.configs[model_id] = copy.deepcopy(self.default_configs[model_id])
                    self.save_config(model_id)
            
            return copy.deepcopy(self.configs)
    
    def get_model_list(self) -> List[Dict[str, Any]]:
        """Get list of all models with basic information"""
        model_list = []
        
        for model_id, config in self.get_all_configs().items():
            model_info = {
                'model_id': model_id,
                'model_name': config.get('model_name', 'Unknown'),
                'description': config.get('description', ''),
                'last_updated': config.get('last_updated', ''),
                'external_api_enabled': config.get('external_api', {}).get('enabled', False)
            }
            model_list.append(model_info)
        
        return model_list
    
    def search_configs(self, search_term: str) -> List[Dict[str, Any]]:
        """Search configurations for a specific term"""
        results = []
        search_term_lower = search_term.lower()
        
        for model_id, config in self.get_all_configs().items():
            # Convert configuration to string and search
            config_str = json.dumps(config).lower()
            if search_term_lower in config_str:
                # Add basic model information to results
                model_info = {
                    'model_id': model_id,
                    'model_name': config.get('model_name', 'Unknown'),
                    'description': config.get('description', '')
                }
                results.append(model_info)
        
        return results
    
    def set_external_api_status(self, model_id: str, enabled: bool) -> bool:
        """Enable or disable external API for a specific model"""
        return self.update_config(model_id, {'external_api': {'enabled': enabled}})
    
    def update_external_api_config(self, model_id: str, api_config: Dict[str, Any]) -> bool:
        """Update external API configuration for a specific model"""
        return self.update_config(model_id, {'external_api': api_config})
    
    def get_external_api_status(self, model_id: str) -> Dict[str, Any]:
        """Get external API status for a specific model"""
        config = self.get_config(model_id)
        if config is None:
            return {
                'enabled': False,
                'provider': '',
                'api_key': '',
                'api_base': '',
                'model_name': ''
            }
        
        external_api = config.get('external_api', {})
        return {
            'enabled': external_api.get('enabled', False),
            'provider': external_api.get('provider', ''),
            'api_key': external_api.get('api_key', ''),
            'api_base': external_api.get('api_base', ''),
            'model_name': external_api.get('model_name', '')
        }

# Initialize the configuration manager
def get_config_manager() -> TrainingConfigManager:
    """Get the singleton instance of TrainingConfigManager"""
    return TrainingConfigManager()