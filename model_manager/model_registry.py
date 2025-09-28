# Self Brain Model Registry
# Author: silencecrowtom@qq.com
# This module manages the registration and configuration of all AI models in the system

import os
import json
import logging
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SelfBrainModelRegistry')

class ModelRegistry:
    """Registry for managing all AI models in the Self Brain system"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelRegistry, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the model registry"""
        self.registry_path = os.path.join('d:\\shiyan\\config', 'model_registry.json')
        self.models = {}
        self._load_registry()
        self._sync_with_config()
    
    def _load_registry(self):
        """Load the model registry from the configuration file"""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                    self.models = registry_data.get('models', {})
                    logger.info(f"Loaded {len(self.models)} models from registry")
            else:
                # Create default registry if it doesn't exist
                self._create_default_registry()
        except Exception as e:
            logger.error(f"Failed to load model registry: {str(e)}")
            # Fallback to default models
            self._create_default_registry()
    
    def _create_default_registry(self):
        """Create a default model registry"""
        # Define the 11 core models based on system requirements
        self.models = {
            'A_management': {
                'name': 'Management Model',
                'type': 'management',
                'description': 'Main interactive AI model that manages all other models',
                'local_supported': True,
                'api_supported': True,
                'current_provider': 'local',  # local or api
                'api_config': {
                    'provider': '',
                    'api_key': '',
                    'base_url': '',
                    'model_name': ''
                },
                'status': 'idle',  # idle, training, running, error
                'port': 5000,
                'last_updated': datetime.now().isoformat()
            },
            'B_language': {
                'name': 'Language Model',
                'type': 'language',
                'description': 'Large language model with multilingual capabilities and emotional reasoning',
                'local_supported': True,
                'api_supported': True,
                'current_provider': 'local',
                'api_config': {
                    'provider': '',
                    'api_key': '',
                    'base_url': '',
                    'model_name': ''
                },
                'status': 'idle',
                'port': 5001,
                'last_updated': datetime.now().isoformat()
            },
            'C_audio': {
                'name': 'Audio Processing Model',
                'type': 'audio',
                'description': 'Model for speech recognition, tone analysis, and audio synthesis',
                'local_supported': True,
                'api_supported': True,
                'current_provider': 'local',
                'api_config': {
                    'provider': '',
                    'api_key': '',
                    'base_url': '',
                    'model_name': ''
                },
                'status': 'idle',
                'port': 5002,
                'last_updated': datetime.now().isoformat()
            },
            'D_image': {
                'name': 'Image Processing Model',
                'type': 'vision',
                'description': 'Model for image recognition, modification, and generation',
                'local_supported': True,
                'api_supported': True,
                'current_provider': 'local',
                'api_config': {
                    'provider': '',
                    'api_key': '',
                    'base_url': '',
                    'model_name': ''
                },
                'status': 'idle',
                'port': 5003,
                'last_updated': datetime.now().isoformat()
            },
            'E_video': {
                'name': 'Video Processing Model',
                'type': 'vision',
                'description': 'Model for video content recognition, editing, and generation',
                'local_supported': True,
                'api_supported': True,
                'current_provider': 'local',
                'api_config': {
                    'provider': '',
                    'api_key': '',
                    'base_url': '',
                    'model_name': ''
                },
                'status': 'idle',
                'port': 5004,
                'last_updated': datetime.now().isoformat()
            },
            'F_spatial': {
                'name': 'Spatial Perception Model',
                'type': 'vision',
                'description': 'Binocular spatial perception model for 3D modeling and object tracking',
                'local_supported': True,
                'api_supported': False,
                'current_provider': 'local',
                'api_config': {},
                'status': 'idle',
                'port': 5005,
                'last_updated': datetime.now().isoformat()
            },
            'G_sensor': {
                'name': 'Sensor Processing Model',
                'type': 'sensor',
                'description': 'Model for processing data from various sensors',
                'local_supported': True,
                'api_supported': False,
                'current_provider': 'local',
                'api_config': {},
                'status': 'idle',
                'port': 5006,
                'last_updated': datetime.now().isoformat()
            },
            'H_computer': {
                'name': 'Computer Control Model',
                'type': 'control',
                'description': 'Model for controlling computer operations and system compatibility',
                'local_supported': True,
                'api_supported': False,
                'current_provider': 'local',
                'api_config': {},
                'status': 'idle',
                'port': 5007,
                'last_updated': datetime.now().isoformat()
            },
            'I_motion': {
                'name': 'Motion Control Model',
                'type': 'control',
                'description': 'Model for controlling external devices and actuators',
                'local_supported': True,
                'api_supported': False,
                'current_provider': 'local',
                'api_config': {},
                'status': 'idle',
                'port': 5008,
                'last_updated': datetime.now().isoformat()
            },
            'J_knowledge': {
                'name': 'Knowledge Base Model',
                'type': 'knowledge',
                'description': 'Core knowledge expert model with comprehensive domain knowledge',
                'local_supported': True,
                'api_supported': True,
                'current_provider': 'local',
                'api_config': {
                    'provider': '',
                    'api_key': '',
                    'base_url': '',
                    'model_name': ''
                },
                'status': 'idle',
                'port': 5009,
                'last_updated': datetime.now().isoformat()
            },
            'K_programming': {
                'name': 'Programming Model',
                'type': 'coding',
                'description': 'Model for code generation, debugging, and system self-improvement',
                'local_supported': True,
                'api_supported': True,
                'current_provider': 'local',
                'api_config': {
                    'provider': '',
                    'api_key': '',
                    'base_url': '',
                    'model_name': ''
                },
                'status': 'idle',
                'port': 5010,
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # Save the default registry
        self._save_registry()
    
    def _save_registry(self):
        """Save the model registry to the configuration file"""
        try:
            # Ensure the config directory exists
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            
            registry_data = {
                'version': '1.0.0',
                'last_updated': datetime.now().isoformat(),
                'models': self.models
            }
            
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved model registry with {len(self.models)} models")
        except Exception as e:
            logger.error(f"Failed to save model registry: {str(e)}")
    
    def _sync_with_config(self):
        """Sync model registry with system configuration"""
        try:
            config_path = os.path.join('d:\\shiyan\\config', 'system_config.yaml')
            if os.path.exists(config_path):
                # In a real implementation, we would read the YAML file
                # and sync the port configurations
                pass
        except Exception as e:
            logger.error(f"Failed to sync with system config: {str(e)}")
    
    def get_model(self, model_id):
        """Get a model by its ID"""
        return self.models.get(model_id)
    
    def get_all_models(self):
        """Get all models in the registry"""
        return self.models
    
    def update_model_status(self, model_id, status):
        """Update the status of a model"""
        if model_id in self.models:
            self.models[model_id]['status'] = status
            self.models[model_id]['last_updated'] = datetime.now().isoformat()
            self._save_registry()
            logger.info(f"Updated status of model {model_id} to {status}")
            return True
        return False
    
    def update_model_api_config(self, model_id, api_config):
        """Update the API configuration of a model"""
        if model_id in self.models and self.models[model_id]['api_supported']:
            self.models[model_id]['api_config'] = api_config
            self.models[model_id]['last_updated'] = datetime.now().isoformat()
            self._save_registry()
            logger.info(f"Updated API configuration for model {model_id}")
            return True
        return False
    
    def set_model_provider(self, model_id, provider):
        """Set the provider for a model (local or api)"""
        if model_id in self.models:
            if provider == 'api' and not self.models[model_id]['api_supported']:
                logger.error(f"Model {model_id} does not support API providers")
                return False
            
            self.models[model_id]['current_provider'] = provider
            self.models[model_id]['last_updated'] = datetime.now().isoformat()
            self._save_registry()
            logger.info(f"Set provider for model {model_id} to {provider}")
            return True
        return False
    
    def validate_api_connection(self, model_id):
        """Validate the API connection for a model"""
        # In a real implementation, this would test the API connection
        # For now, we'll just simulate a check
        if model_id in self.models and self.models[model_id]['current_provider'] == 'api':
            api_config = self.models[model_id]['api_config']
            # Check if all required fields are filled
            if all([api_config.get('api_key'), api_config.get('base_url'), api_config.get('model_name')]):
                logger.info(f"API connection validation passed for model {model_id}")
                return True
        logger.error(f"API connection validation failed for model {model_id}")
        return False

# Create a global instance of the model registry
global_model_registry = ModelRegistry()

# Helper functions for the API
def get_model_info(model_id):
    """Get information about a specific model"""
    return global_model_registry.get_model(model_id)

def get_all_models_info():
    """Get information about all models"""
    return global_model_registry.get_all_models()

def update_model_status_api(model_id, status):
    """Update the status of a model via API"""
    return global_model_registry.update_model_status(model_id, status)

def update_model_api_config_api(model_id, api_config):
    """Update the API configuration of a model via API"""
    return global_model_registry.update_model_api_config(model_id, api_config)

def set_model_provider_api(model_id, provider):
    """Set the provider for a model via API"""
    return global_model_registry.set_model_provider(model_id, provider)

def validate_model_api_connection(model_id):
    """Validate the API connection for a model via API"""
    return global_model_registry.validate_api_connection(model_id)