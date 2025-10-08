#!/usr/bin/env python
# Self Brain AGI System - Training Manager
# Copyright 2025 AGI System Team

"""
Training Manager Module

This module provides the core functionality for training, evaluating, and managing
all models in the Self Brain AGI system.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SelfBrainTrainingManager")

# Import submodules
try:
    # Try to import existing model architectures and config manager
    from .model_architectures import (
        BaseModel,
        ModelA,
        ModelB,
        ModelC,
        ModelD,
        ModelE,
        ModelF,
        ModelG,
        ModelH,
        ModelI,
        ModelJ,
        ModelK,
        create_model,
        get_model_info,
        list_available_models
    )
    
    from .training_config_manager import TrainingConfigManager
    has_existing_components = True
except ImportError:
    logger.warning("Existing model architectures not found, initializing with basic components")
    has_existing_components = False

# Import new training components
from .model_trainer import (
    ModelTrainer, 
    AManagementModelTrainer, 
    BLanguageModelTrainer,
    CAudioModelTrainer,
    DImageModelTrainer,
    EVideoModelTrainer,
    FSpaceModelTrainer,
    GSensorModelTrainer,
    HComputerControlModelTrainer,
    IMotionControlModelTrainer,
    JKnowledgeModelTrainer,
    KProgrammingModelTrainer,
    ModelTrainingManager,
    training_manager
)
from .data_manager import DataManager, data_manager

# Define package version and information
__version__ = '1.0.0'
__author__ = 'silencecrowtom@qq.com'
__description__ = 'Self Brain - AI Training Manager'
__package__ = 'training_manager'

# Global training configuration
TRAINING_CONFIG = {
    "max_workers": 5,
    "default_epochs": 100,
    "default_batch_size": 32,
    "default_learning_rate": 0.001,
    "model_registry_path": "config/model_registry.json",
    "training_history_dir": "data/training_histories",
    "model_save_path": "./models",
    "training_data_path": "./training_data",
    "checkpoint_interval": 5,
    "log_interval": 1,
    "api_port": 5001
}

# Ensure training history directory exists
training_history_dir = TRAINING_CONFIG["training_history_dir"]
os.makedirs(training_history_dir, exist_ok=True)

# Define exports based on available components
exports = [
    "ModelTrainer", 
    "AManagementModelTrainer", 
    "BLanguageModelTrainer",
    "CAudioModelTrainer",
    "DImageModelTrainer",
    "EVideoModelTrainer",
    "FSpaceModelTrainer",
    "GSensorModelTrainer",
    "HComputerControlModelTrainer",
    "IMotionControlModelTrainer",
    "JKnowledgeModelTrainer",
    "KProgrammingModelTrainer",
    "ModelTrainingManager", 
    "training_manager",
    "DataManager", 
    "data_manager",
    "train_model",
    "stop_training",
    "get_training_status",
    "import_training_data",
    "generate_sample_data",
    "get_data_statistics",
    "clear_model_data",
    "get_available_models",
    "initialize_training_system"
]

# Add existing components to exports if available
if has_existing_components:
    exports.extend([
        "BaseModel",
        "ModelA", "ModelB", "ModelC", "ModelD", "ModelE",
        "ModelF", "ModelG", "ModelH", "ModelI", "ModelJ", "ModelK",
        "create_model",
        "get_model_info",
        "list_available_models",
        "TrainingConfigManager"
    ])

__all__ = exports

# Load model registry
def load_model_registry():
    """Load the model registry"""
    registry_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), TRAINING_CONFIG["model_registry_path"])
    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load model registry: {str(e)}")
        return {}

# Save training history
def save_training_history(model_id, history):
    """Save training history for a model"""
    history_file = os.path.join(training_history_dir, f"{model_id}_history.json")
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        logger.info(f"Training history saved for model {model_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save training history for model {model_id}: {str(e)}")
        return False

# Get training history
def get_training_history(model_id):
    """Get training history for a model"""
    history_file = os.path.join(training_history_dir, f"{model_id}_history.json")
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load training history for model {model_id}: {str(e)}")
        return {}

# API Functions
def train_model(model_id, hyperparameters=None):
    """
    Train a specific model
    
    Args:
        model_id (str): The ID of the model to train
        hyperparameters (dict): Optional hyperparameters to override default settings
        
    Returns:
        tuple: (success, message)
    """
    try:
        # Update hyperparameters if provided
        if hyperparameters and model_id in training_manager.trainers:
            trainer = training_manager.trainers[model_id]
            for key, value in hyperparameters.items():
                if key in trainer.config.get('hyperparameters', {}):
                    trainer.config['hyperparameters'][key] = value
            trainer.save_config()
        
        # Start training
        success, message = training_manager.start_training(model_id)
        
        # If training started successfully, create a basic history entry
        if success:
            history = {
                "model_id": model_id,
                "start_time": datetime.now().isoformat(),
                "status": "training",
                "hyperparameters": hyperparameters or {},
                "history": []
            }
            save_training_history(model_id, history)
        
        return success, message
        
    except Exception as e:
        logger.error(f"Error in train_model for {model_id}: {str(e)}")
        return False, str(e)

def stop_training(model_id):
    """
    Stop training for a specific model
    
    Args:
        model_id (str): The ID of the model to stop training
        
    Returns:
        tuple: (success, message)
    """
    try:
        success, message = training_manager.stop_training(model_id)
        
        # Update training history
        if success:
            history = get_training_history(model_id)
            if history:
                history["end_time"] = datetime.now().isoformat()
                history["status"] = "stopped"
                save_training_history(model_id, history)
        
        return success, message
        
    except Exception as e:
        logger.error(f"Error in stop_training for {model_id}: {str(e)}")
        return False, str(e)

def get_training_status(model_id=None):
    """
    Get training status for a specific model or all models
    
    Args:
        model_id (str): Optional model ID
        
    Returns:
        dict: Status information
    """
    try:
        status = training_manager.get_training_status(model_id)
        
        # If getting status for a specific model, include training history
        if model_id:
            history = get_training_history(model_id)
            status["training_history"] = history
        
        return status
        
    except Exception as e:
        logger.error(f"Error in get_training_status: {str(e)}")
        return {"error": str(e)}

def import_training_data(model_id, file_path, split_ratio=(0.7, 0.2, 0.1)):
    """
    Import training data for a model
    
    Args:
        model_id (str): The ID of the model
        file_path (str): Path to the data file
        split_ratio (tuple): Train/validation/test split ratio
        
    Returns:
        tuple: (success, result)
    """
    try:
        return data_manager.import_data(model_id, file_path, split_ratio)
        
    except Exception as e:
        logger.error(f"Error in import_training_data for {model_id}: {str(e)}")
        return False, str(e)

def import_batch_training_data(model_id, directory_path, split_ratio=(0.7, 0.2, 0.1)):
    """
    Import multiple training data files for a model
    
    Args:
        model_id (str): The ID of the model
        directory_path (str): Path to the directory containing data files
        split_ratio (tuple): Train/validation/test split ratio
        
    Returns:
        tuple: (success, result)
    """
    try:
        return data_manager.import_batch_data(model_id, directory_path, split_ratio)
        
    except Exception as e:
        logger.error(f"Error in import_batch_training_data for {model_id}: {str(e)}")
        return False, str(e)

def generate_sample_data(model_id, sample_size=100):
    """
    Generate sample training data for a model
    
    Args:
        model_id (str): The ID of the model
        sample_size (int): Number of samples to generate
        
    Returns:
        tuple: (success, result)
    """
    try:
        return data_manager.generate_sample_data(model_id, sample_size)
        
    except Exception as e:
        logger.error(f"Error in generate_sample_data for {model_id}: {str(e)}")
        return False, str(e)

def get_data_statistics(model_id):
    """
    Get statistics about the training data for a model
    
    Args:
        model_id (str): The ID of the model
        
    Returns:
        dict: Data statistics
    """
    try:
        return data_manager.get_data_statistics(model_id)
        
    except Exception as e:
        logger.error(f"Error in get_data_statistics for {model_id}: {str(e)}")
        return {"error": str(e)}

def clear_model_data(model_id, include_raw=False):
    """
    Clear all training data for a model
    
    Args:
        model_id (str): The ID of the model
        include_raw (bool): Whether to include raw data
        
    Returns:
        tuple: (success, message)
    """
    try:
        return data_manager.clear_model_data(model_id, include_raw)
        
    except Exception as e:
        logger.error(f"Error in clear_model_data for {model_id}: {str(e)}")
        return False, str(e)

def get_available_models():
    """
    Get a list of all available models that can be trained
    
    Returns:
        list: List of model IDs
    """
    try:
        return list(training_manager.trainers.keys())
        
    except Exception as e:
        logger.error(f"Error in get_available_models: {str(e)}")
        return []

def initialize_training_system():
    """
    Initialize the training system
    
    Returns:
        bool: Success status
    """
    try:
        # Ensure all model trainers are initialized
        model_registry = load_model_registry()
        
        # Log the number of available models
        logger.info(f"Initializing training system with {len(model_registry)} registered models")
        logger.info(f"Available model trainers: {list(training_manager.trainers.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in initialize_training_system: {str(e)}")
        return False

# Initialize the training system when the module is loaded
initialize_training_system()