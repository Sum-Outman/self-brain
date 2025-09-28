# Self Brain Training Manager Package
"""
Training Manager for the Self Brain AI system.
Provides model architectures and training configuration management.

Author: silencecrowtom@qq.com
"""

# Import main components from the package
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

# Define package version
__version__ = '1.0.0'

# Define package information
__author__ = 'silencecrowtom@qq.com'
__description__ = 'Self Brain - AI Training Manager'
__package__ = 'training_manager'

# Make important classes and functions available at package level
__all__ = [
    'BaseModel',
    'ModelA', 'ModelB', 'ModelC', 'ModelD', 'ModelE',
    'ModelF', 'ModelG', 'ModelH', 'ModelI', 'ModelJ', 'ModelK',
    'create_model',
    'get_model_info',
    'list_available_models',
    'TrainingConfigManager'
]