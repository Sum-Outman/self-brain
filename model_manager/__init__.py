# Self Brain Model Manager
# Author: silencecrowtom@qq.com
# This module provides the core functionality for managing all AI models in the system

# Import the model registry
from .model_registry import (
    ModelRegistry,
    global_model_registry,
    get_model_info,
    get_all_models_info,
    update_model_status_api,
    update_model_api_config_api,
    set_model_provider_api,
    validate_model_api_connection
)

# Version information
__version__ = '1.0.0'

# Define what is exported when importing * from this package
__all__ = [
    'ModelRegistry',
    'global_model_registry',
    'get_model_info',
    'get_all_models_info',
    'update_model_status_api',
    'update_model_api_config_api',
    'set_model_provider_api',
    'validate_model_api_connection'
]