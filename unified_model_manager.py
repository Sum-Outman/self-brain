#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified Model Manager
This module provides a unified interface for managing all models in the Self Brain AGI system.
"""

import os
import sys
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Union
import requests
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UnifiedModelManager")

class ModelManager:
    """Unified model manager for the Self Brain AGI system
    
    Manages all models, handles model loading, configuration, and provides a unified interface
    for interacting with different models.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the model manager
        
        Args:
            config_path: Path to the model configuration file
        """
        # Load configuration
        self.config_path = config_path
        self.config = self._load_config()
        
        # Dictionary to store model instances
        self.models: Dict[str, Any] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Dictionary to track model status
        self.model_status: Dict[str, Dict[str, Any]] = {}
        
        # Initialize models
        self._initialize_models()
        
        logger.info("ModelManager initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from file
        
        Returns:
            Dictionary with model configuration
        """
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Loaded model configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Failed to load model configuration: {str(e)}")
                return self._get_default_config()
        else:
            logger.warning("Model configuration file not found, using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration
        
        Returns:
            Dictionary with default model configuration
        """
        default_config = {
            "A_management": {
                "mode": "local",
                "path": "sub_models/A_management",
                "config": {}
            },
            "B_language": {
                "mode": "local",
                "path": "sub_models/B_language",
                "config": {}
            },
            "C_audio": {
                "mode": "local",
                "path": "sub_models/C_audio",
                "config": {}
            },
            "D_image": {
                "mode": "local",
                "path": "sub_models/D_image",
                "config": {}
            },
            "E_video": {
                "mode": "local",
                "path": "sub_models/E_video",
                "config": {}
            },
            "F_spatial": {
                "mode": "local",
                "path": "sub_models/F_spatial",
                "config": {}
            },
            "G_sensor": {
                "mode": "local",
                "path": "sub_models/G_sensor",
                "config": {}
            },
            "H_computer_control": {
                "mode": "local",
                "path": "sub_models/H_computer_control",
                "config": {}
            },
            "I_knowledge": {
                "mode": "local",
                "path": "sub_models/I_knowledge",
                "config": {}
            },
            "J_motion": {
                "mode": "local",
                "path": "sub_models/J_motion",
                "config": {}
            },
            "K_programming": {
                "mode": "local",
                "path": "sub_models/K_programming",
                "config": {}
            }
        }
        
        # If config path was provided but file doesn't exist, save default config
        if self.config_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                logger.info(f"Created default model configuration at {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to save default model configuration: {str(e)}")
        
        return default_config
    
    def _initialize_models(self) -> None:
        """Initialize all models based on configuration"""
        with self._lock:
            for model_name, model_config in self.config.items():
                try:
                    self._initialize_model(model_name, model_config)
                except Exception as e:
                    logger.error(f"Failed to initialize model {model_name}: {str(e)}")
                    self.model_status[model_name] = {
                        "status": "error",
                        "message": str(e),
                        "last_updated": datetime.now().isoformat()
                    }
    
    def _initialize_model(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """Initialize a specific model
        
        Args:
            model_name: Name of the model
            model_config: Configuration for the model
        """
        mode = model_config.get("mode", "local")
        
        if mode == "local":
            self._initialize_local_model(model_name, model_config)
        elif mode == "external":
            self._initialize_external_model(model_name, model_config)
        else:
            raise ValueError(f"Unsupported model mode: {mode}")
    
    def _initialize_local_model(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """Initialize a local model
        
        Args:
            model_name: Name of the model
            model_config: Configuration for the model
        """
        try:
            # Get model path
            model_path = model_config.get("path", f"sub_models/{model_name}")
            
            # Make path absolute
            if not os.path.isabs(model_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(base_dir, model_path)
            
            # Check if model directory exists
            if not os.path.exists(model_path):
                logger.warning(f"Model directory not found: {model_path}")
                return
            
            # Add model directory to path
            sys.path.append(model_path)
            
            # Try to import the model
            try:
                # Import the model module
                model_module_name = model_name.lower()
                model_module = __import__(f"{model_name}.model", fromlist=[model_module_name])
                
                # Get the model class
                model_class = getattr(model_module, f"{model_name}Model", None)
                
                if model_class is None:
                    # Fallback to generic import
                    model_class = getattr(model_module, "Model", None)
                
                if model_class is not None:
                    # Initialize the model
                    self.models[model_name] = model_class(model_config.get("config", {}))
                    logger.info(f"Initialized local model: {model_name}")
                    self.model_status[model_name] = {
                        "status": "initialized",
                        "mode": "local",
                        "path": model_path,
                        "last_updated": datetime.now().isoformat()
                    }
                else:
                    logger.warning(f"Could not find model class for {model_name}")
                    # Create a fallback model
                    self._create_fallback_model(model_name, model_config)
            except Exception as e:
                logger.error(f"Failed to import model {model_name}: {str(e)}")
                # Create a fallback model
                self._create_fallback_model(model_name, model_config)
        except Exception as e:
            logger.error(f"Error initializing local model {model_name}: {str(e)}")
            raise
    
    def _initialize_external_model(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """Initialize an external API model
        
        Args:
            model_name: Name of the model
            model_config: Configuration for the model
        """
        try:
            # Create a wrapper for the external API model
            class ExternalModelWrapper:
                def __init__(self, config):
                    self.config = config
                    self.model_name = model_name
                    self.api_url = config.get("api_url", "")
                    self.api_key = config.get("api_key", "")
                    self.external_model_name = config.get("external_model_name", "")
                    logger.info(f"Initialized external model wrapper for {model_name}")
                
                def predict(self, data):
                    # In a real implementation, this would make an API call to the external model
                    return {
                        "status": "success",
                        "model": self.external_model_name,
                        "message": "External model prediction would be returned here",
                        "simulated": True,
                        "input_received": data
                    }
                
                def train(self, **kwargs):
                    # In a real implementation, this would make an API call to the external model for training
                    return {
                        "status": "success",
                        "model": self.external_model_name,
                        "message": "External model training would be initiated here",
                        "simulated": True,
                        "training_params": kwargs
                    }
                
                def evaluate(self, data):
                    # In a real implementation, this would make an API call to evaluate the external model
                    return {
                        "status": "success",
                        "model": self.external_model_name,
                        "message": "External model evaluation would be returned here",
                        "simulated": True,
                        "evaluation_data": data
                    }
            
            # Initialize the external model wrapper
            self.models[model_name] = ExternalModelWrapper(model_config.get("config", {}))
            logger.info(f"Initialized external model wrapper: {model_name}")
            self.model_status[model_name] = {
                "status": "initialized",
                "mode": "external",
                "external_model_name": model_config.get("config", {}).get("external_model_name", "unknown"),
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error initializing external model {model_name}: {str(e)}")
            raise
    
    def _create_fallback_model(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """Create a fallback model when the actual model can't be loaded
        
        Args:
            model_name: Name of the model
            model_config: Configuration for the model
        """
        class FallbackModel:
            def __init__(self, config):
                self.config = config
                self.model_name = model_name
            
            def predict(self, data):
                return {
                    "status": "error",
                    "model": self.model_name,
                    "message": f"Model {self.model_name} is not available. Please check the logs.",
                    "fallback": True,
                    "input_received": data
                }
            
            def train(self, **kwargs):
                return {
                    "status": "error",
                    "model": self.model_name,
                    "message": f"Model {self.model_name} training is not available",
                    "fallback": True
                }
            
            def evaluate(self, data):
                return {
                    "status": "error",
                    "model": self.model_name,
                    "message": f"Model {self.model_name} evaluation is not available",
                    "fallback": True
                }
        
        # Initialize fallback model
        self.models[model_name] = FallbackModel(model_config.get("config", {}))
        self.model_status[model_name] = {
            "status": "fallback",
            "message": "Using fallback model implementation",
            "last_updated": datetime.now().isoformat()
        }
        logger.warning(f"Using fallback model for {model_name}")
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model by name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance or None if not found
        """
        with self._lock:
            return self.models.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models
        
        Returns:
            List of model names
        """
        with self._lock:
            return list(self.models.keys())
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models
        
        Returns:
            Dictionary with model status information
        """
        with self._lock:
            return self.model_status.copy()
    
    def reload_model(self, model_name: str) -> bool:
        """Reload a specific model
        
        Args:
            model_name: Name of the model to reload
            
        Returns:
            True if reload was successful, False otherwise
        """
        with self._lock:
            if model_name in self.config:
                try:
                    self._initialize_model(model_name, self.config[model_name])
                    logger.info(f"Reloaded model: {model_name}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to reload model {model_name}: {str(e)}")
                    return False
            else:
                logger.warning(f"Model not found in config: {model_name}")
                return False
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update model configuration
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            with self._lock:
                self.config = new_config
                # Save updated config if path is set
                if self.config_path:
                    with open(self.config_path, 'w', encoding='utf-8') as f:
                        json.dump(new_config, f, indent=2, ensure_ascii=False)
                # Reinitialize models
                self._initialize_models()
            logger.info("Updated model configuration successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to update model configuration: {str(e)}")
            return False

# Singleton instance of ModelManager
_model_manager_instance = None
_model_manager_lock = threading.Lock()

def get_model_manager(config_path: Optional[str] = None) -> ModelManager:
    """Get the singleton instance of ModelManager
    
    Args:
        config_path: Path to the model configuration file
        
    Returns:
        The singleton ModelManager instance
    """
    global _model_manager_instance
    with _model_manager_lock:
        if _model_manager_instance is None:
            _model_manager_instance = ModelManager(config_path)
    return _model_manager_instance

# Example usage
if __name__ == "__main__":
    # Create a model manager instance
    manager = get_model_manager()
    
    # List available models
    print("Available models:", manager.get_available_models())
    
    # Get model status
    print("Model status:", manager.get_model_status())