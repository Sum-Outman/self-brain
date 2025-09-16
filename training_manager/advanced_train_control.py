# -*- coding: utf-8 -*-
# Advanced Training Control Panel - Manage training of all AGI system models
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License")

import json
import logging
import time
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Union
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import psutil
import gc
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import gc
import re
import gettext
import locale
import random
import os
import yaml
from config.config_loader import get_config, get_config_loader  # Import config loader functions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_control.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrainingControl")

class TrainingMode(Enum):
    """Training mode enumeration"""
    INDIVIDUAL = "individual"  # Individual training
    JOINT = "joint"            # Joint training
    TRANSFER = "transfer"      # Transfer learning
    FINE_TUNE = "fine_tune"    # Fine-tuning
    PRETRAINING = "pretraining"  # Pretraining

class TrainingStatus(Enum):
    """Training status enumeration"""
    IDLE = "idle"              # Idle
    PREPARING = "preparing"    # Preparing
    TRAINING = "training"      # Training
    VALIDATING = "validating"  # Validating
    COMPLETED = "completed"    # Completed
    FAILED = "failed"          # Failed
    PAUSED = "paused"          # Paused

class LanguageManager:
    """Language Manager"""
    
    def __init__(self, language: str = "en_US"):
        self.language = language
        self.translations = self._load_translations()
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translations"""
        return {
            "en_US": {
                "training_started": "Training started",
                "training_completed": "Training completed",
                "training_failed": "Training failed",
                "training_paused": "Training paused",
                "training_resumed": "Training resumed",
                "queue_cleared": "Training queue cleared",
                "resource_insufficient": "Insufficient resources",
                "model_not_found": "Model not found",
                "invalid_request": "Invalid request",
                "preparing_data": "Preparing data",
                "loading_models": "Loading models",
                "configuring_training": "Configuring training parameters",
                "checking_resources": "Checking resources",
                "starting_training": "Starting training",
                "training_in_progress": "Training in progress",
                "validation_in_progress": "Validation in progress",
                "saving_checkpoint": "Saving checkpoint",
                "generating_report": "Generating report",
                "shutting_down": "Shutting down"
            }
        }
    
    def set_language(self, language: str):
        """Set language"""
        if language in self.translations:
            self.language = language
        else:
            self.language = "en_US"
    
    def get_text(self, key: str) -> str:
        """Get translated text"""
        return self.translations.get(self.language, {}).get(key, key)
    
    def get_available_languages(self) -> List[str]:
        """Get available languages list"""
        return list(self.translations.keys())

class AdvancedTrainingController:
    """
    Advanced Training Control Panel - Enhanced Version
    Manage individual, joint training and intelligent collaboration of all models
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Advanced Training Controller - Enhanced Version
        
        Parameters:
        config_path: Configuration file path
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load system config from config/system_config.yaml
        self.system_config = get_config_loader(config_path="config/system_config.yaml").to_dict()
        
        # Training status
        self.training_status = {
            "current_mode": TrainingMode.INDIVIDUAL.value,
            "current_status": TrainingStatus.IDLE.value,
            "active_models": [],
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": 0,
            "start_time": None,
            "end_time": None,
            "metrics": {},
            "collaboration_level": "basic",  # Collaboration level: basic, intermediate, advanced
            "knowledge_assist_enabled": False,
            "knowledge_model_id": None
        }
        
        # Training history
        self.training_history = deque(maxlen=1000)
        
        # Performance metrics
        self.performance_metrics = {
            "total_trainings": 0,
            "successful_trainings": 0,
            "failed_trainings": 0,
            "average_training_time": 0.0,
            "model_performance": defaultdict(dict),
            "collaboration_efficiency": 1.0,  # Collaboration efficiency metric
            "knowledge_utilization_rate": 0.0  # Knowledge base utilization rate
        }
        
        # Collaboration statistics
        self.collaboration_stats = {
            "total_collaborations": 0,
            "successful_collaborations": 0,
            "model_interactions": defaultdict(int),
            "knowledge_sharing_events": 0
        }
        
        # Training queue
        self.training_queue = deque()
        self.is_training = False
        self.training_thread = None
        
        # Model registry - Enhanced
        self.model_registry = self._initialize_enhanced_model_registry()
        
        # Knowledge base integration
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Collaboration engine
        self.collaboration_engine = self._initialize_collaboration_engine()
        
        # Language manager
        self.language_manager = LanguageManager()
        
        # Real-time data bus
        self.data_bus = self._initialize_data_bus()
        
        logger.info("Advanced Training Controller Enhanced initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration file"""
        # Load external model registry configuration for Chinese model types
        external_model_config = self._load_model_registry_config()
        
        default_config = {
            "system": {
                "name": "Advanced_Training_Controller",
                "version": "2.0.0",
                "description": "Advanced Training Control Panel - Manage AGI system model training",
                "license": "Apache-2.0"
            },
            "training": {
                "modes": {
                    "individual": {
                        "enabled": True,
                        "max_concurrent": 3,
                        "default_epochs": 100,
                        "batch_size": 32,
                        "learning_rate": 0.001
                    },
                    "joint": {
                        "enabled": True,
                        "max_concurrent": 1,
                        "default_epochs": 50,
                        "batch_size": 16,
                        "learning_rate": 0.0005
                    },
                    "transfer": {
                        "enabled": True,
                        "max_concurrent": 2,
                        "default_epochs": 30,
                        "batch_size": 24,
                        "learning_rate": 0.0001
                    },
                    "fine_tune": {
                        "enabled": True,
                        "max_concurrent": 2,
                        "default_epochs": 20,
                        "batch_size": 16,
                        "learning_rate": 0.00005
                    }
                },
                "scheduling": {
                    "auto_schedule": True,
                    "priority_weights": {
                        "critical": 3.0,
                        "high": 2.0,
                        "medium": 1.5,
                        "low": 1.0
                    },
                    "resource_aware": True,
                    "min_memory_mb": 1024,
                    "min_cpu_percent": 20
                },
                "optimization": {
                    "early_stopping": True,
                    "patience": 10,
                    "learning_rate_scheduling": True,
                    "gradient_clipping": True,
                    "regularization": True
                },
                "monitoring": {
                    "real_time_metrics": True,
                    "checkpoint_frequency": 5,
                    "validation_frequency": 2,
                    "performance_logging": True
                }
            },
            "models": {}
        }
        
        # Use external model configuration with Chinese model types
        if external_model_config:
            for model_id, model_info in external_model_config.items():
                # Handle both string and dict types for model type
                type_info = model_info.get("type", {})
                if isinstance(type_info, dict):
                    model_type = type_info.get("zh", "Unknown")
                else:
                    model_type = str(type_info)
                
                default_config["models"][model_id] = {
                    "trainable": True,
                    "model_type": model_type,
                    "model_source": model_info.get("model_source", "local"),
                    "api_config": {
                        "api_url": model_info.get("api_url", ""),
                        "api_key": model_info.get("api_key", "")
                    },
                    "default_mode": TrainingMode.INDIVIDUAL.value,
                    "data_requirements": [],
                    "performance_metrics": ["accuracy"]
                }
        else:
            # Fallback to hardcoded English model types if external config not found
            default_config["models"] = {
                "A_management": {
                    "trainable": True,
                    "model_type": "Manager Model",
                    "model_source": "local",
                    "api_config": {},
                    "default_mode": TrainingMode.JOINT.value,
                    "data_requirements": ["system_logs", "performance_data", "user_feedback", "model_outputs"],
                    "performance_metrics": ["management_efficiency", "resource_optimization", "model_coordination", "decision_accuracy", "system_stability"]
                },
                "B_language": {
                    "trainable": True,
                    "model_type": "Language Model",
                    "model_source": "local",
                    "api_config": {},
                    "default_mode": TrainingMode.INDIVIDUAL.value,
                    "data_requirements": ["text_data"],
                    "performance_metrics": ["accuracy", "perplexity", "bleu_score"]
                },
                "C_audio": {
                    "trainable": True,
                    "model_type": "Audio Processor",
                    "model_source": "local",
                    "api_config": {},
                    "default_mode": TrainingMode.INDIVIDUAL.value,
                    "data_requirements": ["audio_data", "transcripts"],
                    "performance_metrics": ["mse", "snr", "wer"]
                },
                "D_image": {
                    "trainable": True,
                    "model_type": "Image Processor",
                    "model_source": "local",
                    "api_config": {},
                    "default_mode": TrainingMode.INDIVIDUAL.value,
                    "data_requirements": ["image_data", "labels"],
                    "performance_metrics": ["accuracy", "precision", "recall", "f1_score"]
                },
                "E_video": {
                    "trainable": True,
                    "model_type": "Video Processor",
                    "model_source": "local",
                    "api_config": {},
                    "default_mode": TrainingMode.JOINT.value,
                    "data_requirements": ["video_data", "annotations"],
                    "performance_metrics": ["mse", "ssim", "psnr"]
                },
                "F_spatial": {
                    "trainable": True,
                    "model_type": "Spatial Locator",
                    "model_source": "local",
                    "api_config": {},
                    "default_mode": TrainingMode.INDIVIDUAL.value,
                    "data_requirements": ["point_clouds", "depth_maps"],
                    "performance_metrics": ["accuracy", "iou", "rmse"]
                },
                "G_sensor": {
                    "trainable": True,
                    "model_type": "Sensor Processor",
                    "model_source": "local",
                    "api_config": {},
                    "default_mode": TrainingMode.INDIVIDUAL.value,
                    "data_requirements": ["sensor_readings", "labels"],
                    "performance_metrics": ["mse", "mae", "r2_score"]
                },
                "H_computer_control": {
                    "trainable": True,
                    "model_type": "Computer Controller",
                    "model_source": "local",
                    "api_config": {},
                    "default_mode": TrainingMode.TRANSFER.value,
                    "data_requirements": ["command_sequences", "responses"],
                    "performance_metrics": ["success_rate", "response_time"]
                },
                "I_knowledge": {
                    "trainable": True,
                    "model_type": "Knowledge Expert Model",
                    "model_source": "local",
                    "api_config": {},
                    "default_mode": TrainingMode.TRANSFER.value,
                    "data_requirements": ["knowledge_graphs", "text_data", "structured_data"],
                    "performance_metrics": ["knowledge_accuracy", "query_response_time", "coverage"]
                },
                "J_motion": {
                    "trainable": True,
                    "model_type": "Motion Controller",
                    "model_source": "local",
                    "api_config": {},
                    "default_mode": TrainingMode.INDIVIDUAL.value,
                    "data_requirements": ["motion_data", "sensor_readings"],
                    "performance_metrics": ["control_accuracy", "response_time", "stability"]
                },
                "K_programming": {
                    "trainable": True,
                    "model_type": "Programming Model",
                    "model_source": "local",
                    "api_config": {},
                    "default_mode": TrainingMode.INDIVIDUAL.value,
                    "data_requirements": ["code_snippets", "documentation", "test_cases"],
                    "performance_metrics": ["correctness", "efficiency", "readability"]
                }
            }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Config file loaded successfully: {config_path}")
            except Exception as e:
                logger.error(f"Config file loading failed: {e}")
        
        return default_config
    
    def _load_model_registry_config(self) -> Dict[str, Any]:
        """Load model registry configuration from external file"""
        registry_config_path = Path("config/model_registry.json")
        if registry_config_path.exists():
            try:
                with open(registry_config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load model registry config: {e}")
        return {}
    
    def _initialize_enhanced_model_registry(self) -> Dict[str, Any]:
        """Initialize enhanced model registry"""
        registry = {}
        
        for model_name, model_config in self.config["models"].items():
            registry[model_name] = {
                "config": model_config,
                "training_history": [],
                "best_performance": {},
                "current_status": "not_loaded",
                "last_trained": None,
                "total_training_time": 0,
                "training_sessions": 0,
                "collaboration_score": 0.0,  # Collaboration capability score
                "knowledge_utilization": 0.0,  # Knowledge base utilization rate
                "interaction_count": 0,  # Interaction count with other models
                "performance_trend": "stable",  # Performance trend
                "compatibility": self._get_model_compatibility(model_name),  # Compatibility information
                "resource_profile": self._create_resource_profile(model_name)  # Resource usage profile
            }
        
        return registry
    
    def _get_model_compatibility(self, model_name: str) -> Dict[str, Any]:
        """Get model compatibility information"""
        model_config = self.config["models"][model_name]
        
        return {
            "data_types": model_config["data_requirements"],
            "preferred_mode": model_config["default_mode"],
            "compatible_models": self._find_compatible_models(model_name),
            "interaction_patterns": ["data_sharing", "parameter_exchange", "knowledge_transfer"],
            "api_compatibility": ["REST", "gRPC", "WebSocket"]
        }
    
    def _find_compatible_models(self, model_name: str) -> List[str]:
        """Find compatible models"""
        compatible_models = []
        current_model_config = self.config["models"][model_name]
        
        for other_model, other_config in self.config["models"].items():
            if other_model != model_name:
                # Check data requirement compatibility
                common_data = set(current_model_config["data_requirements"]) & set(other_config["data_requirements"])
                if common_data:
                    compatible_models.append(other_model)
        
        return compatible_models
    
    def _create_resource_profile(self, model_name: str) -> Dict[str, Any]:
        """Create resource usage profile"""
        # Estimate resource requirements based on model type
        # Reduced for development environment
        resource_profiles = {
            "A_management": {"memory_mb": 32, "cpu_percent": 1, "gpu_memory_mb": 0, "disk_space_mb": 1500},
            "B_language": {"memory_mb": 32, "cpu_percent": 1, "gpu_memory_mb": 0, "disk_space_mb": 1000},
            "C_audio": {"memory_mb": 16, "cpu_percent": 1, "gpu_memory_mb": 0, "disk_space_mb": 800},
            "D_vision": {"memory_mb": 16, "cpu_percent": 1, "gpu_memory_mb": 0, "disk_space_mb": 800},
            "E_multimodal": {"memory_mb": 32, "cpu_percent": 1, "gpu_memory_mb": 0, "disk_space_mb": 1200},
            "F_reasoning": {"memory_mb": 32, "cpu_percent": 1, "gpu_memory_mb": 0, "disk_space_mb": 1500},
            "G_creative": {"memory_mb": 16, "cpu_percent": 1, "gpu_memory_mb": 0, "disk_space_mb": 800},
            "H_technical": {"memory_mb": 16, "cpu_percent": 1, "gpu_memory_mb": 0, "disk_space_mb": 800},
            "I_social": {"memory_mb": 16, "cpu_percent": 1, "gpu_memory_mb": 0, "disk_space_mb": 800},
            "J_emotional": {"memory_mb": 16, "cpu_percent": 1, "gpu_memory_mb": 0, "disk_space_mb": 800},
            "K_programming": {"memory_mb": 32, "cpu_percent": 1, "gpu_memory_mb": 0, "disk_space_mb": 1200}
        }
        
        return resource_profiles.get(model_name, {"memory_mb": 256, "cpu_percent": 10, "gpu_memory_mb": 128, "disk_space_mb": 500})
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize knowledge base integration"""
        return {
            "enabled": True,
            "knowledge_model_id": "I_knowledge",
            "sync_frequency": 300,  # 5 minutes
            "last_sync": None,
            "knowledge_cache": {},
            "recommendations": deque(maxlen=100)
        }
    
    def _initialize_collaboration_engine(self) -> Dict[str, Any]:
        """Initialize collaboration engine"""
        return {
            "enabled": True,
            "collaboration_threshold": 0.7,
            "interaction_matrix": defaultdict(dict),
            "collaboration_history": deque(maxlen=500),
            "recommendation_engine": None
        }
    
    def _initialize_data_bus(self) -> Dict[str, Any]:
        """Initialize real-time data bus"""
        return {
            "enabled": True,
            "subscribers": defaultdict(list),
            "data_cache": deque(maxlen=1000),
            "event_queue": deque(maxlen=500)
        }
    
    def _ensure_data_directories(self):
        """Ensure training data directories exist based on system configuration"""
        try:
            # Create main data directory if it doesn't exist
            main_data_path = self.system_config.get('training', {}).get('data_paths', {}).get('main', 'data/training')
            if main_data_path:
                os.makedirs(main_data_path, exist_ok=True)
                self._add_training_log(f"Main data directory ensured: {main_data_path}")
            
            # Create model-specific data directories
            for model_name in self.system_config.get('training', {}).get('data_paths', {}):
                if model_name != 'main':
                    model_data_path = self.system_config['training']['data_paths'][model_name]
                    if model_data_path:
                        os.makedirs(model_data_path, exist_ok=True)
                        self._add_training_log(f"Model data directory ensured: {model_data_path}")
        except Exception as e:
            self._add_training_log(f"Failed to create data directories: {str(e)}", is_error=True)
            logger.error(f"Failed to create data directories: {str(e)}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status - Enhanced version"""
        try:
            # Get CPU usage with safe interval
            try:
                cpu_percent = psutil.cpu_percent(interval=0.5)  # Reduced interval
            except Exception:
                cpu_percent = 0
            
            # Get memory usage
            try:
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                available_memory_mb = memory.available / (1024 * 1024)
            except Exception:
                memory_percent = 0
                available_memory_mb = 0
            
            # Get disk usage - Windows compatible
            try:
                # Try multiple disk detection methods
                disk = None
                for path in ['C:', 'D:', 'E:', 'F:', 'G:', 'H:', '/', '.']:
                    try:
                        disk = psutil.disk_usage(path)
                        break
                    except (OSError, ValueError):
                        continue
                
                if disk is None:
                    disk_percent = 0
                    available_disk_gb = 0
                else:
                    disk_percent = disk.percent
                    available_disk_gb = disk.free / (1024 * 1024 * 1024)
            except Exception:
                disk_percent = 0
                available_disk_gb = 0
            
            # Get GPU information
            gpu_info = self._get_gpu_info()
            
            # Get system load - Windows safe
            load_avg = (0, 0, 0)  # Default for Windows
            
            # Get running processes - simplified and safe
            processes = []
            try:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        pinfo = proc.info
                        if pinfo:
                            processes.append({
                                'pid': pinfo.get('pid', 0),
                                'name': str(pinfo.get('name', 'unknown'))
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                    except Exception:
                        continue
            except Exception:
                pass
            
            # Limit processes
            processes = processes[:10]
            
            health_status = {
                "status": "healthy",
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent,
                "available_memory_mb": available_memory_mb,
                "disk_usage": disk_percent,
                "available_disk_gb": available_disk_gb,
                "gpu_info": gpu_info,
                "load_average": load_avg,
                "processes": processes,
                "timestamp": datetime.now().isoformat()
            }
            
            # Determine overall health
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
                health_status["status"] = "critical"
            elif cpu_percent > 80 or memory_percent > 80 or disk_percent > 90:
                health_status["status"] = "warning"
            
            return health_status
            
        except Exception as e:
            # Use safe string representation
            error_msg = str(e).encode('utf-8', 'ignore').decode('utf-8')
            logger.error(f"Failed to get system health: {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information - Enhanced detection"""
        try:
            # Method 1: Use GPUtil library
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Get first GPU
                    gpu_info = {
                        "available": True,
                        "count": len(gpus),
                        "gpu_model": gpu.name,
                        "gpu_usage_percent": gpu.load * 100,
                        "gpu_memory_mb": gpu.memoryTotal,
                        "gpu_memory_used_mb": gpu.memoryUsed,
                        "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "temperature": gpu.temperature
                    }
                    logger.info(f"GPU detected: {gpu.name} ({gpu.load * 100}%)")
                    return gpu_info
                else:
                    logger.info("No physical GPU detected, using configured GPU info")
                    return {"available": False, "count": 0, "gpu_model": "CPU", "gpu_usage_percent": 0}
            except ImportError:
                logger.warning("GPUtil library not available, using configured GPU info")
                return {"available": False, "count": 0, "gpu_model": "CPU", "gpu_usage_percent": 0}
                
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}, using configured GPU info")
            return {"available": False, "count": 0, "gpu_model": "CPU", "gpu_usage_percent": 0}
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics - Enhanced version"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Get GPU metrics
            gpu_info = self._get_gpu_info()
            
            # Get training-specific metrics
            training_metrics = {
                "active_models": len(self.training_status["active_models"]),
                "current_mode": self.training_status["current_mode"],
                "training_status": self.training_status["current_status"],
                "progress": self.training_status["progress"],
                "queue_length": len(self.training_queue),
                "total_trainings": self.performance_metrics["total_trainings"],
                "successful_trainings": self.performance_metrics["successful_trainings"],
                "failed_trainings": self.performance_metrics["failed_trainings"]
            }
            
            # Add per-model metrics
            model_metrics = {}
            for model_name, registry_data in self.model_registry.items():
                model_metrics[model_name] = {
                    "training_sessions": registry_data["training_sessions"],
                    "total_training_time": registry_data["total_training_time"],
                    "last_trained": registry_data["last_trained"],
                    "collaboration_score": registry_data["collaboration_score"],
                    "performance_trend": registry_data["performance_trend"]
                }
            
            return {
                "system": {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "available_memory_mb": memory.available / (1024 * 1024),
                    "gpu_info": gpu_info
                },
                "training": training_metrics,
                "models": model_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Use safe logging approach
            safe_error_msg = str(e).replace('%', '%%')
            logger.error("Failed to get real-time metrics: " + safe_error_msg)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get training status - Enhanced version"""
        return {
            "current_mode": self.training_status["current_mode"],
            "current_status": self.training_status["current_status"],
            "active_models": self.training_status["active_models"],
            "progress": self.training_status["progress"],
            "current_epoch": self.training_status["current_epoch"],
            "total_epochs": self.training_status["total_epochs"],
            "start_time": self.training_status["start_time"],
            "end_time": self.training_status["end_time"],
            "metrics": self.training_status["metrics"],
            "queue_length": len(self.training_queue),
            "collaboration_level": self.training_status["collaboration_level"],
            "knowledge_assist_enabled": self.training_status["knowledge_assist_enabled"],
            "timestamp": datetime.now().isoformat()
        }
    
    def start_training(self, model_names: List[str], mode: TrainingMode, 
                      config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start training - Enhanced version"""
        try:
            # Validate model names
            invalid_models = [m for m in model_names if m not in self.model_registry]
            if invalid_models:
                return {
                    "status": "error",
                    "message": f"Invalid model names: {invalid_models}"
                }
            
            # Check resource availability
            resource_check = self._check_resource_availability(model_names, mode)
            if not resource_check["available"]:
                return {
                    "status": "error",
                    "message": f"Insufficient resources: {resource_check['message']}"
                }
            
            # Create training configuration
            training_config = config or {}
            training_config.update({
                "model_names": model_names,
                "mode": mode,
                "start_time": datetime.now().isoformat()
            })
            
            # Generate training ID
            training_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            # Update training status
            self.training_status.update({
                "current_mode": mode.value,
                "current_status": TrainingStatus.PREPARING.value,
                "active_models": model_names,
                "progress": 0.0,
                "current_epoch": 0,
                "total_epochs": training_config.get("epochs", 50),
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "metrics": {},
                "task_name": training_config.get("task_name", f"Training_{training_id[:15]}")
            })
            
            # Initialize training logs for this session
            self.training_logs = []
            task_name = self.training_status["task_name"]
            self._add_training_log(f"Starting training session: {task_name}")
            self._add_training_log(f"Configuration: Mode={mode.value}, Models={', '.join(model_names)}")
            self._add_training_log(f"Total epochs: {training_config.get('epochs', 50)}")
            
            # Start training in background thread
            self.training_thread = threading.Thread(
                target=self._execute_training,
                args=(training_id, model_names, mode, training_config)
            )
            self.training_thread.daemon = True
            self.training_thread.start()
            
            logger.info(f"Training started: {training_id}, mode: {mode.value}, models: {model_names}, task: {task_name}")
            
            return {
                "status": "success",
                "training_id": training_id,
                "message": f"Training started successfully with {len(model_names)} models",
                "task_name": task_name
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Training start failed: {error_msg}")
            return {
                "status": "error",
                "message": f"Training start failed: {error_msg}"
            }
    
    def _add_training_log(self, message: str) -> None:
        """Add a log entry to the training logs"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "message": message
        }
        self.training_logs.append(log_entry)
        
        # Keep logs to a reasonable size (e.g., 1000 entries)
        if len(self.training_logs) > 1000:
            self.training_logs.pop(0)
        
        # Emit log via WebSocket if available
        if hasattr(self, 'socketio') and self.socketio:
            self.socketio.emit('training_log', {
                'message': message,
                'timestamp': timestamp
            }, namespace='/ws/training')
        
        # Also log to system logger
        logger.info(f"[Training] {message}")
    
    def _check_resource_availability(self, model_names: List[str], mode: TrainingMode) -> Dict[str, Any]:
        """Check resource availability"""
        # Estimate total resource requirements
        total_memory_mb = 0
        total_cpu_percent = 0
        
        for model_name in model_names:
            profile = self.model_registry[model_name]["resource_profile"]
            total_memory_mb += profile["memory_mb"]
            total_cpu_percent += profile["cpu_percent"]
        
        # Consider training configuration impact
        if mode == TrainingMode.JOINT:
            total_memory_mb *= 1.05  # Joint training requires more memory (minimal for dev)
            total_cpu_percent *= 1.02
        
        # Check system resources
        memory = psutil.virtual_memory()
        available_memory_mb = memory.available / (1024 * 1024)
        
        cpu_percent = psutil.cpu_percent(interval=1)
        available_cpu = 100 - cpu_percent
        
        issues = []
        
        if available_memory_mb < total_memory_mb:
            issues.append(f"Insufficient memory: need {total_memory_mb:.1f}MB, available {available_memory_mb:.1f}MB")
        
        if available_cpu < total_cpu_percent:
            issues.append(f"Insufficient CPU: need {total_cpu_percent:.1f}%, available {available_cpu:.1f}%")
        
        return {
            "available": len(issues) == 0,
            "message": "; ".join(issues) if issues else "Resources sufficient"
        }
    
    def _execute_training(self, training_id: str, model_names: List[str], 
                         mode: TrainingMode, config: Dict[str, Any]):
        """Execute training"""
        try:
            # Update status to training
            self.training_status["current_status"] = TrainingStatus.TRAINING.value
            self.is_training = True
            
            self._add_training_log(f"Training started for {', '.join(model_names)}")
            self._add_training_log(f"Training mode: {mode.value}")
            
            # Execute different training strategies based on mode
            if mode == TrainingMode.INDIVIDUAL:
                result = self._execute_individual_training(training_id, model_names, config)
            elif mode == TrainingMode.JOINT:
                result = self._execute_joint_training(training_id, model_names, config)
            elif mode == TrainingMode.TRANSFER:
                result = self._execute_transfer_training(training_id, model_names, config)
            elif mode == TrainingMode.FINE_TUNE:
                result = self._execute_fine_tune_training(training_id, model_names, config)
            elif mode == TrainingMode.PRETRAINING:
                result = self._execute_pretraining(training_id, model_names, config)
            else:
                result = {"status": "failed", "message": "Unknown training mode"}
            
            # Record training result
            self.performance_metrics["total_trainings"] += 1
            if result["status"] == "success":
                self.performance_metrics["successful_trainings"] += 1
                self._add_training_log(f"Training completed successfully for {', '.join(model_names)}")
                if result.get("final_accuracy"):
                    self._add_training_log(f"Final accuracy: {result['final_accuracy']:.2%}")
            else:
                self.performance_metrics["failed_trainings"] += 1
                self._add_training_log(f"Training failed: {result.get('message', 'Unknown error')}", is_error=True)
            
            # Update training status
            self.training_status["current_status"] = (
                TrainingStatus.COMPLETED.value if result["status"] == "success" 
                else TrainingStatus.FAILED.value
            )
            self.training_status["end_time"] = datetime.now().isoformat()
            self.is_training = False
            
            # Update average training time
            if result.get("duration"):
                total_time = self.performance_metrics["average_training_time"] * (self.performance_metrics["total_trainings"] - 1)
                total_time += result["duration"]
                self.performance_metrics["average_training_time"] = total_time / self.performance_metrics["total_trainings"]
                self._add_training_log(f"Training duration: {result['duration']} seconds")
            
            # Log completion
            status_text = "completed successfully" if result["status"] == "success" else "failed"
            self._add_training_log(f"Training session {status_text}")
            
            logger.info(f"Training completed: {training_id}, status: {result['status']}")
            
        except Exception as e:
            error_msg = str(e)
            self._add_training_log(f"Exception occurred: {error_msg}", is_error=True)
            logger.error(f"Training execution failed {training_id}: {error_msg}")
            
            # Update training status to failed
            self.training_status["current_status"] = TrainingStatus.FAILED.value
            self.training_status["end_time"] = datetime.now().isoformat()
            self.is_training = False
            self.performance_metrics["failed_trainings"] += 1
    
    def _execute_individual_training(self, training_id: str, model_names: List[str], 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual training"""
        try:
            if len(model_names) != 1:
                return {"status": "error", "message": "Individual training requires exactly one model"}
            
            model_name = model_names[0]
            
            # Check if training was cancelled
            if self.training_status.get("cancel_requested", False):
                return {"status": "cancelled", "message": "Training cancelled"}
            
            # Update currently training model
            self.training_status["active_models"] = [model_name]
            
            # Simulate training process
            epochs = config.get("epochs", 50)
            model_result = self._simulate_model_training(model_name, epochs, config)
            
            if model_result["status"] != "success":
                return {
                    "status": "failed", 
                    "message": f"Model {model_name} training failed: {model_result['message']}"
                }
            
            # Update progress
            self.training_status["progress"] = 100.0
            self.training_status["current_epoch"] = epochs
            
            # Update model registry
            self.model_registry[model_name]["training_sessions"] += 1
            self.model_registry[model_name]["last_trained"] = datetime.now().isoformat()
            
            return {
                "status": "success",
                "message": "Individual training completed",
                "duration": model_result.get("duration", 0)
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _execute_joint_training(self, training_id: str, model_names: List[str], 
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute joint training"""
        try:
            # Use enhanced joint training coordinator
            from .enhanced_joint_trainer import EnhancedJointTrainer
            
            joint_trainer = EnhancedJointTrainer()
            
            # Prepare joint training request
            training_request = {
                "models": model_names,
                "mode": "collaborative",  # Default to collaborative mode
                "epochs": config.get("epochs", 50),
                "batch_size": config.get("batch_size", 16),
                "learning_rate": config.get("learning_rate", 0.0005),
                "knowledge_assisted": self.training_status["knowledge_assist_enabled"]
            }
            
            # Start joint training
            result = joint_trainer.start_joint_training(training_request)
            
            if result["status"] != "success":
                return {
                    "status": "failed",
                    "message": f"Joint training failed to start: {result.get('message', 'Unknown error')}"
                }
            
            # Monitor joint training progress
            training_id = result["training_id"]
            final_result = joint_trainer.monitor_joint_training_progress(training_id)
            
            return final_result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Joint training execution failed: {error_msg}")
            return {
                "status": "failed",
                "message": f"Joint training execution failed: {error_msg}"
            }
    
    def _execute_transfer_training(self, training_id: str, model_names: List[str], 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute transfer learning training"""
        # Simplified implementation: use individual training to simulate transfer learning
        return self._execute_individual_training(training_id, model_names, config)
    
    def _execute_fine_tune_training(self, training_id: str, model_names: List[str], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fine-tuning training"""
        # Simplified implementation: use individual training to simulate fine-tuning
        return self._execute_individual_training(training_id, model_names, config)
    
    def _execute_pretraining(self, training_id: str, model_names: List[str], 
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pretraining"""
        # Pretraining usually requires longer training time and larger data
        pretraining_config = config.copy()
        
        # Set pretraining-specific configuration
        pretraining_config["epochs"] = max(config.get("epochs", 100), 50)  # At least 50 epochs
        pretraining_config["learning_rate"] = min(config.get("learning_rate", 0.001), 0.0001)  # Smaller learning rate
        pretraining_config["batch_size"] = max(config.get("batch_size", 32), 64)  # Larger batch size
        
        logger.info(f"Starting pretraining mode for models: {model_names}")
        
        # Use individual training mode for pretraining
        return self._execute_individual_training(training_id, model_names, pretraining_config)
    
    def _load_training_data(self, model_name: str):
        """Load training data from configured paths"""
        try:
            # Get model-specific data path from system config
            model_data_path = self.system_config.get('training', {}).get('data_paths', {}).get(model_name.lower(), '')
            
            # If no specific path, use main data path
            if not model_data_path:
                model_data_path = self.system_config.get('training', {}).get('data_paths', {}).get('main', 'data/training')
                self._add_training_log(f"Using main data path for {model_name}: {model_data_path}")
            else:
                self._add_training_log(f"Using model-specific data path for {model_name}: {model_data_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(model_data_path, exist_ok=True)
            
            # Check if there are any data files
            data_files = []
            for ext in ['json', 'csv', 'txt', 'parquet', 'npy', 'npz']:
                data_files.extend(Path(model_data_path).glob(f'*.{ext}'))
            
            if not data_files:
                # No data files found, create sample data
                sample_data = self._generate_sample_data(model_name)
                sample_file = os.path.join(model_data_path, 'sample_training_data.json')
                with open(sample_file, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
                self._add_training_log(f"No data files found, created sample data: {sample_file}")
                return sample_data
            
            # Load first found data file
            data_file = data_files[0]
            self._add_training_log(f"Loading training data from: {data_file}")
            
            # Load based on file extension
            if data_file.suffix == '.json':
                with open(data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif data_file.suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(data_file)
                return df.to_dict('records')
            elif data_file.suffix in ['.npy', '.npz']:
                import numpy as np
                if data_file.suffix == '.npy':
                    data = np.load(str(data_file))
                    return data.tolist() if isinstance(data, np.ndarray) else data.item()
                else:
                    with np.load(str(data_file)) as data:
                        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}
            else:
                # Default to text file
                with open(data_file, 'r', encoding='utf-8') as f:
                    return f.readlines()
        except Exception as e:
            self._add_training_log(f"Failed to load training data: {str(e)}", is_error=True)
            logger.error(f"Failed to load training data: {str(e)}")
            # Return sample data as fallback
            return self._generate_sample_data(model_name)
    
    def _generate_sample_data(self, model_name: str) -> List[Dict[str, Any]]:
        """Generate sample training data for a model"""
        sample_sizes = {
            'A_management': 100,
            'B_language': 200,
            'C_audio': 150,
            'D_image': 150,
            'E_video': 100,
            'F_spatial': 100,
            'G_sensor': 200,
            'H_computer_control': 100,
            'I_knowledge': 200,
            'J_motion': 150,
            'K_programming': 150
        }
        
        sample_size = sample_sizes.get(model_name, 100)
        sample_data = []
        
        # Generate model-specific sample data
        if model_name == 'B_language':
            texts = [
                "This is a sample text for language model training",
                "Another example of text data",
                "Language models can process and generate human language",
                "Training data is essential for machine learning models",
                "The quick brown fox jumps over the lazy dog"
            ]
            for i in range(sample_size):
                sample_data.append({
                    'id': i,
                    'text': random.choice(texts) + f" (example {i})",
                    'label': random.choice([0, 1, 2]),
                    'metadata': {
                        'source': 'sample',
                        'timestamp': datetime.now().isoformat()
                    }
                })
        else:
            # Generic sample data for other models
            for i in range(sample_size):
                sample_data.append({
                    'id': i,
                    'input': f"Sample input {i}",
                    'output': f"Sample output {i}",
                    'label': random.choice([0, 1]) if random.random() > 0.3 else 0,
                    'metadata': {
                        'model': model_name,
                        'source': 'sample',
                        'timestamp': datetime.now().isoformat()
                    }
                })
        
        return sample_data
    
    def _simulate_model_training(self, model_name: str, epochs: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate model training process using system configuration"""
        try:
            # Ensure data directories exist
            self._ensure_data_directories()
            
            # Load training parameters from system configuration
            system_training_config = self.system_config.get('training', {})
            
            # Override with config parameters if provided
            epochs = config.get('epochs', system_training_config.get('epochs', 50))
            batch_size = config.get('batch_size', system_training_config.get('batch_size', 32))
            learning_rate = config.get('learning_rate', system_training_config.get('learning_rate', 0.001))
            validation_split = config.get('validation_split', system_training_config.get('validation_split', 0.2))
            
            # Set base duration based on model complexity
            base_duration = 60  # 60 seconds base
            model_type = self.config["models"][model_name]["model_type"]
            
            # Adjust training parameters based on training type
            training_type = "self-supervised"
            if config.get("mode") == "pretraining":
                epochs = max(epochs, 50)  # Pretraining at least 50 epochs
                learning_rate = min(learning_rate, 0.0001)  # Pretraining uses smaller learning rate
                training_type = "pretraining"
                self._add_training_log(f"Starting {training_type} for {model_name}")
            elif config.get("mode") == "transfer":
                training_type = "supervised"
                self._add_training_log(f"Starting {training_type} training for {model_name}")
            elif config.get("mode") == "fine_tune":
                training_type = "semi-supervised"
                self._add_training_log(f"Starting {training_type} training for {model_name}")
            else:
                self._add_training_log(f"Starting {training_type} training for {model_name}")
            
            # Load training data
            training_data = self._load_training_data(model_name)
            data_size = len(training_data) if isinstance(training_data, list) else 100
            
            # Split data into training and validation sets
            train_size = int(data_size * (1 - validation_split))
            
            self._add_training_log(f"Configuration: {epochs} epochs, {batch_size} batch size, {learning_rate} learning rate")
            self._add_training_log(f"Data size: {data_size} samples, Training: {train_size} samples, Validation: {data_size - train_size} samples")
            
            # Check if training was cancelled
            if self.training_status.get("cancel_requested", False):
                self._add_training_log(f"Training for {model_name} cancelled")
                return {"status": "cancelled", "message": "Training cancelled"}

            # Split data into training and validation sets
            if isinstance(training_data, list) and len(training_data) > 0:
                random.shuffle(training_data)
                train_data = training_data[:train_size]
                val_data = training_data[train_size:]
                self._add_training_log(f"Data split complete: {len(train_data)} training samples, {len(val_data)} validation samples")
            else:
                train_data = []
                val_data = []
                self._add_training_log("Warning: No valid data to split")

            # Initialize training metrics
            metrics = {
                "train_accuracy": 0.5 + random.random() * 0.4,  # Random accuracy between 0.5 and 0.9
                "train_loss": 1.0 - random.random() * 0.7,      # Random loss between 0.3 and 1.0
                "val_accuracy": 0.5 + random.random() * 0.3,    # Validation accuracy slightly lower
                "val_loss": 1.0 - random.random() * 0.6,        # Validation loss slightly higher
                "precision": 0.5 + random.random() * 0.4,
                "recall": 0.5 + random.random() * 0.4,
                "f1": 0.5 + random.random() * 0.4,
                "training_time": base_duration * epochs / 100,
                "samples_processed": len(train_data) * epochs if train_data else random.randint(1000, 10000)
            }

            # Simulate epoch-wise training with validation
            best_val_accuracy = metrics["val_accuracy"]
            patience = 5  # Early stopping patience
            no_improvement_count = 0

            for epoch in range(epochs):
                # Update epoch progress
                current_epoch = epoch + 1
                self.training_status["current_epoch"] = current_epoch
                self.training_status["progress"] = (current_epoch / epochs) * 100

                # Simulate training on batches
                total_batches = max(1, (len(train_data) + batch_size - 1) // batch_size) if train_data else 1
                
                for batch_idx in range(total_batches):
                    # Update batch progress
                    batch_progress = ((batch_idx + 1) / total_batches) * 100
                    self.training_status["batch_progress"] = batch_progress
                    
                    # Simulate batch processing time
                    time.sleep(0.05)  # Shorter sleep for batches

                # Simulate validation
                current_val_accuracy = metrics["val_accuracy"] + (random.random() - 0.45) * 0.02
                current_val_loss = metrics["val_loss"] - (random.random() - 0.5) * 0.05
                
                metrics["val_accuracy"] = current_val_accuracy
                metrics["val_loss"] = current_val_loss

                # Update metrics with learning rate decay
                if current_epoch % 10 == 0:
                    learning_rate *= 0.9  # Decay learning rate
                    self._add_training_log(f"Learning rate decayed to: {learning_rate:.6f}")

                # Update training metrics (simulate improvement)
                metrics["train_accuracy"] = min(metrics["train_accuracy"] + random.random() * 0.02, 0.95)
                metrics["train_loss"] = max(metrics["train_loss"] - random.random() * 0.03, 0.1)

                # Update metrics dictionary
                self.training_status["metrics"] = {
                    "train_loss": metrics["train_loss"],
                    "train_accuracy": metrics["train_accuracy"],
                    "val_loss": metrics["val_loss"],
                    "val_accuracy": metrics["val_accuracy"],
                    "learning_rate": learning_rate
                }

                # Add epoch log with validation metrics
                if current_epoch % 5 == 0 or current_epoch == epochs:
                    epoch_log = f"Epoch {current_epoch}/{epochs} - "
                    epoch_log += f"Train: Accuracy: {metrics['train_accuracy']:.4f}, Loss: {metrics['train_loss']:.4f} - "
                    epoch_log += f"Validation: Accuracy: {metrics['val_accuracy']:.4f}, Loss: {metrics['val_loss']:.4f}"
                    self._add_training_log(epoch_log)

                # Check for early stopping
                if current_val_accuracy > best_val_accuracy:
                    best_val_accuracy = current_val_accuracy
                    no_improvement_count = 0
                    if current_epoch % 5 == 0:
                        self._add_training_log(f"New best validation accuracy: {best_val_accuracy:.4f}")
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience and current_epoch >= 10:
                        self._add_training_log(f"Early stopping triggered after {patience} epochs without improvement")
                        break

                # Simulate training time
                time.sleep(0.1)  # Sleep for simulation

                # Check if training was cancelled
                if self.training_status.get("cancel_requested", False):
                    self._add_training_log(f"Training for {model_name} cancelled")
                    return {"status": "cancelled", "message": "Training cancelled"}

            # Simulate training duration
            duration = base_duration + (current_epoch * 2) + random.randint(10, 30)

            # Update model registry with training results
            model_key = model_name.lower()
            if model_key in self.model_registry:
                self.model_registry[model_key].update({
                    "last_trained": datetime.now().isoformat(),
                    "training_metrics": metrics,
                    "best_val_accuracy": best_val_accuracy,
                    "epochs_trained": current_epoch
                })

            # Log completion
            self._add_training_log(f"Training for {model_name} completed - Best validation accuracy: {best_val_accuracy:.4f}")

            return {
                "status": "success",
                "duration": duration,
                "metrics": metrics,
                "best_val_accuracy": best_val_accuracy,
                "epochs_completed": current_epoch
            }

        except Exception as e:
            self._add_training_log(f"Error in training {model_name}: {str(e)}", is_error=True)
            logger.error(f"Training failed for {model_name}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def pause_training(self) -> Dict[str, Any]:
        """Pause current training"""
        if self.is_training:
            self.training_status["current_status"] = TrainingStatus.PAUSED.value
            return {"status": "success", "message": "Training paused"}
        return {"status": "error", "message": "No active training to pause"}
    
    def resume_training(self) -> Dict[str, Any]:
        """Resume paused training"""
        if self.training_status["current_status"] == TrainingStatus.PAUSED.value:
            self.training_status["current_status"] = TrainingStatus.TRAINING.value
            return {"status": "success", "message": "Training resumed"}
        return {"status": "error", "message": "No paused training to resume"}
    
    def stop_training(self) -> Dict[str, Any]:
        """Stop current training"""
        if self.is_training:
            self.training_status["cancel_requested"] = True
            return {"status": "success", "message": "Training stop requested"}
        return {"status": "error", "message": "No active training to stop"}
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return list(self.training_history)
    
    def get_model_registry(self) -> Dict[str, Any]:
        """Get model registry"""
        return self.model_registry

    def get_model_configuration(self, model_id: str) -> Dict[str, Any]:
        """Get specific model configuration"""
        try:
            if model_id in self.model_registry:
                return {
                    "status": "success",
                    "model_id": model_id,
                    "config": self.model_registry[model_id]
                }
            else:
                return {
                    "status": "error",
                    "message": f"Model '{model_id}' not found in registry"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get model configuration: {str(e)}"
            }

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics"""
        return dict(self.collaboration_stats)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data - combines system health, metrics, and training status"""
        try:
            health_data = self.get_system_health()
            metrics_data = self.get_real_time_metrics()
            training_status = self.get_training_status()
            
            return {
                'status': 'success',
                'data': {
                    'health': health_data,
                    'metrics': metrics_data,
                    'training': training_status
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'data': {
                    'health': {'status': 'healthy', 'components': []},
                    'metrics': {'cpu_usage': 0, 'memory_usage': 0, 'gpu_usage': 0},
                    'training': {'status': 'idle', 'sessions': []}
                }
            }

    def update_collaboration_level(self, level: str) -> Dict[str, Any]:
        """Update collaboration level"""
        valid_levels = ["basic", "intermediate", "advanced"]
        if level not in valid_levels:
            return {"status": "error", "message": f"Invalid collaboration level. Must be one of {valid_levels}"}
        
        self.training_status["collaboration_level"] = level
        return {"status": "success", "message": f"Collaboration level updated to {level}"}
    
    def enable_knowledge_assist(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable knowledge base assistance"""
        self.training_status["knowledge_assist_enabled"] = enabled
        status = "enabled" if enabled else "disabled"
        return {"status": "success", "message": f"Knowledge base assistance {status}"}
    
    def clear_training_queue(self) -> Dict[str, Any]:
        """Clear training queue"""
        queue_length = len(self.training_queue)
        self.training_queue.clear()
        return {
            "status": "success",
            "message": f"Training queue cleared ({queue_length} items removed)"
        }
    
    def shutdown(self) -> Dict[str, Any]:
        """Shutdown the training controller"""
        try:
            # Stop any active training
            if self.is_training:
                self.stop_training()
                if self.training_thread and self.training_thread.is_alive():
                    self.training_thread.join(timeout=10)
            
            # Clear queue
            self.clear_training_queue()
            
            # Update status
            self.training_status["current_status"] = TrainingStatus.IDLE.value
            
            logger.info("Advanced Training Controller shut down successfully")
            
            return {
                "status": "success",
                "message": "Training controller shut down successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Shutdown failed: {str(e)}"
            }

    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Delete a model from the registry"""
        try:
            if model_id not in self.model_registry:
                return {
                    "status": "error",
                    "message": f"Model '{model_id}' not found in registry"
                }
            
            # Remove from registry
            del self.model_registry[model_id]
            
            # Remove from training history if exists
            self.training_history = [h for h in self.training_history if h.get('model_id') != model_id]
            
            # Remove from training queue if exists
            self.training_queue = [q for q in self.training_queue if q.get('model_id') != model_id]
            
            logger.info(f"Model '{model_id}' deleted successfully")
            
            return {
                "status": "success",
                "message": f"Model '{model_id}' deleted successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to delete model: {str(e)}"
            }

# Global instance
_training_controller = None

def get_training_controller() -> AdvancedTrainingController:
    """Get global training controller instance"""
    global _training_controller
    if _training_controller is None:
        _training_controller = AdvancedTrainingController()
    return _training_controller

# API endpoints for external access
def start_training_api(model_names: List[str], mode: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """API endpoint to start training"""
    controller = get_training_controller()
    try:
        training_mode = TrainingMode(mode)
        return controller.start_training(model_names, training_mode, config)
    except ValueError:
        return {"status": "error", "message": f"Invalid training mode: {mode}"}

def get_training_status_api() -> Dict[str, Any]:
    """API endpoint to get training status"""
    controller = get_training_controller()
    return controller.get_training_status()

def get_system_health_api() -> Dict[str, Any]:
    """API endpoint to get system health"""
    controller = get_training_controller()
    return controller.get_system_health()

def get_real_time_metrics_api() -> Dict[str, Any]:
    """API endpoint to get real-time metrics"""
    controller = get_training_controller()
    return controller.get_real_time_metrics()

def pause_training_api() -> Dict[str, Any]:
    """API endpoint to pause training"""
    controller = get_training_controller()
    return controller.pause_training()

def resume_training_api() -> Dict[str, Any]:
    """API endpoint to resume training"""
    controller = get_training_controller()
    return controller.resume_training()

def stop_training_api() -> Dict[str, Any]:
    """API endpoint to stop training"""
    controller = get_training_controller()
    return controller.stop_training()

if __name__ == "__main__":
    # Test the training controller
    controller = get_training_controller()
    
    # Test system health
    health = controller.get_system_health()
    print("System Health:", json.dumps(health, indent=2))
    
    # Test training
    print("Starting test training...")
    result = controller.start_training(["B_language"], TrainingMode.INDIVIDUAL, {"epochs": 5})
    print("Training result:", json.dumps(result, indent=2))
    
    # Wait a bit and check status
    time.sleep(2)
    status = controller.get_training_status()
    print("Training status:", json.dumps(status, indent=2))
