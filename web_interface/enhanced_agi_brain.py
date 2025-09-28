import logging
import os
import json
from datetime import datetime
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('enhanced_agi_brain')

class AGIBrainCore:
    """Core AGI Brain module that manages all AI models and their interactions"""
    
    def __init__(self):
        self.models = {}
        self.model_registry = {}
        self.model_configs = {}
        self.is_initialized = False
        self.lock = threading.RLock()
        
        # Initialize core components
        self._initialize_core()
    
    def _initialize_core(self):
        """Initialize core components of the AGI Brain"""
        try:
            # Load model registry
            self._load_model_registry()
            
            # Initialize models
            self._initialize_models()
            
            self.is_initialized = True
            logger.info("核心组件初始化完成")
        except Exception as e:
            logger.error(f"Core initialization failed: {e}")
            self.is_initialized = False
    
    def _load_model_registry(self):
        """Load model registry from configuration file"""
        try:
            # Try to load from different possible locations
            registry_paths = [
                'D:\config\model_registry.json',  # Path from logs
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'model_registry.json'),
                os.path.join(os.path.dirname(__file__), 'config', 'model_registry.json'),
                'model_registry.json'
            ]
            
            registry_loaded = False
            for registry_path in registry_paths:
                try:
                    if os.path.exists(registry_path):
                        with open(registry_path, 'r', encoding='utf-8') as f:
                            self.model_registry = json.load(f)
                            registry_loaded = True
                            logger.info(f"Model registry loaded from {registry_path}")
                            break
                except Exception as e:
                    logger.warning(f"Failed to load model registry from {registry_path}: {e}")
            
            if not registry_loaded:
                logger.warning("模型注册表文件不存在: D:\config\model_registry.json")
                # Create a default model registry
                self._create_default_model_registry()
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            # Create a default model registry even if there's an error
            self._create_default_model_registry()
    
    def _create_default_model_registry(self):
        """Create a default model registry"""
        self.model_registry = {
            'version': '1.0.0',
            'models': [
                {
                    'id': 'A_management',
                    'name': 'Management Model',
                    'type': 'management',
                    'description': 'Main model that manages all other models',
                    'module_path': 'sub_models.A_management',
                    'class_name': 'ManagementModel',
                    'is_active': True
                },
                {
                    'id': 'B_language',
                    'name': 'Language Model',
                    'type': 'language',
                    'description': 'Natural language processing model',
                    'module_path': 'sub_models.B_language',
                    'class_name': 'LanguageModel',
                    'is_active': True
                },
                {
                    'id': 'C_audio',
                    'name': 'Audio Model',
                    'type': 'audio',
                    'description': 'Audio processing and generation model',
                    'module_path': 'sub_models.C_audio',
                    'class_name': 'AudioModel',
                    'is_active': True
                },
                {
                    'id': 'D_image',
                    'name': 'Image Model',
                    'type': 'image',
                    'description': 'Image processing and generation model',
                    'module_path': 'sub_models.D_image',
                    'class_name': 'ImageModel',
                    'is_active': True
                },
                {
                    'id': 'E_video',
                    'name': 'Video Model',
                    'type': 'video',
                    'description': 'Video processing and generation model',
                    'module_path': 'sub_models.E_video',
                    'class_name': 'VideoModel',
                    'is_active': True
                },
                {
                    'id': 'F_spatial',
                    'name': 'Spatial Model',
                    'type': 'spatial',
                    'description': 'Spatial awareness and 3D modeling model',
                    'module_path': 'sub_models.F_spatial',
                    'class_name': 'SpatialModel',
                    'is_active': True
                },
                {
                    'id': 'G_sensor',
                    'name': 'Sensor Model',
                    'type': 'sensor',
                    'description': 'Sensor data processing model',
                    'module_path': 'sub_models.G_sensor',
                    'class_name': 'SensorModel',
                    'is_active': True
                },
                {
                    'id': 'H_computer',
                    'name': 'Computer Control Model',
                    'type': 'computer',
                    'description': 'Computer system control model',
                    'module_path': 'sub_models.H_computer',
                    'class_name': 'ComputerControlModel',
                    'is_active': True
                },
                {
                    'id': 'I_motion',
                    'name': 'Motion Model',
                    'type': 'motion',
                    'description': 'Motion and actuator control model',
                    'module_path': 'sub_models.I_motion',
                    'class_name': 'MotionModel',
                    'is_active': True
                },
                {
                    'id': 'J_knowledge',
                    'name': 'Knowledge Model',
                    'type': 'knowledge',
                    'description': 'Knowledge base and expert system model',
                    'module_path': 'sub_models.J_knowledge',
                    'class_name': 'KnowledgeModel',
                    'is_active': True
                },
                {
                    'id': 'K_programming',
                    'name': 'Programming Model',
                    'type': 'programming',
                    'description': 'Code generation and programming assistance model',
                    'module_path': 'sub_models.K_programming',
                    'class_name': 'ProgrammingModel',
                    'is_active': True
                }
            ]
        }
        
        # Save the default registry to a local file for future use
        try:
            config_dir = os.path.join(os.path.dirname(__file__), 'config')
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            registry_path = os.path.join(config_dir, 'model_registry.json')
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_registry, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Default model registry saved to {registry_path}")
        except Exception as e:
            logger.warning(f"Failed to save default model registry: {e}")
    
    def _initialize_models(self):
        """Initialize all models based on the registry"""
        for model_info in self.model_registry.get('models', []):
            try:
                if model_info.get('is_active', False):
                    logger.info(f"Initializing model: {model_info['name']} ({model_info['id']})")
                    
                    # In a real implementation, this would dynamically import and initialize the model
                    # For now, we'll just create a placeholder
                    model_placeholder = {
                        'id': model_info['id'],
                        'name': model_info['name'],
                        'type': model_info['type'],
                        'description': model_info['description'],
                        'is_initialized': True,
                        'status': 'ready',
                        'last_update': datetime.now().isoformat()
                    }
                    
                    self.models[model_info['id']] = model_placeholder
            except Exception as e:
                logger.error(f"Failed to initialize model {model_info.get('name', 'Unknown')}: {e}")
    
    def get_model(self, model_id):
        """Get a specific model by ID"""
        with self.lock:
            return self.models.get(model_id)
    
    def get_all_models(self):
        """Get all initialized models"""
        with self.lock:
            return list(self.models.values())
    
    def get_model_registry(self):
        """Get the complete model registry"""
        with self.lock:
            return self.model_registry.copy()
    
    def update_model_status(self, model_id, status, **kwargs):
        """Update the status of a model"""
        with self.lock:
            if model_id in self.models:
                self.models[model_id]['status'] = status
                self.models[model_id]['last_update'] = datetime.now().isoformat()
                
                # Update additional fields
                for key, value in kwargs.items():
                    self.models[model_id][key] = value
                
                logger.info(f"Updated model {model_id} status to {status}")
                return True
            
            logger.warning(f"Model {model_id} not found")
            return False
    
    def process_request(self, model_id, request_data):
        """Process a request for a specific model"""
        with self.lock:
            model = self.get_model(model_id)
            if not model or model.get('status') != 'ready':
                logger.error(f"Model {model_id} not available or not ready")
                return {
                    'status': 'error',
                    'message': f'Model {model_id} not available or not ready'
                }
            
            try:
                # In a real implementation, this would forward the request to the actual model
                # For now, we'll just return a placeholder response
                logger.info(f"Processing request for model {model_id}")
                
                # Update model last used timestamp
                model['last_used'] = datetime.now().isoformat()
                
                # Return a placeholder response based on model type
                response = {
                    'status': 'success',
                    'model_id': model_id,
                    'model_name': model['name'],
                    'timestamp': datetime.now().isoformat(),
                    'request_received': request_data
                }
                
                # Add model-specific placeholder data
                if model['type'] == 'management':
                    response['data'] = {"message": "Management model processing your request"}
                elif model['type'] == 'language':
                    response['data'] = {"text": "This is a placeholder response from the language model"}
                elif model['type'] == 'audio':
                    response['data'] = {"audio_url": "placeholder_audio.wav"}
                elif model['type'] == 'image':
                    response['data'] = {"image_url": "placeholder_image.jpg"}
                elif model['type'] == 'video':
                    response['data'] = {"video_url": "placeholder_video.mp4"}
                elif model['type'] == 'spatial':
                    response['data'] = {"space_coordinates": [0.0, 0.0, 0.0]}
                elif model['type'] == 'sensor':
                    response['data'] = {"sensor_readings": {"temperature": 25.0, "humidity": 45.0}}
                elif model['type'] == 'computer':
                    response['data'] = {"command_executed": True}
                elif model['type'] == 'motion':
                    response['data'] = {"motion_command": "move_forward"}
                elif model['type'] == 'knowledge':
                    response['data'] = {"knowledge_facts": ["Placeholder fact 1", "Placeholder fact 2"]}
                elif model['type'] == 'programming':
                    response['data'] = {"code_snippet": "print('Hello, World!')"}
                else:
                    response['data'] = {"result": "Processed by generic model"}
                
                return response
            except Exception as e:
                logger.error(f"Error processing request for model {model_id}: {e}")
                return {
                    'status': 'error',
                    'message': str(e),
                    'model_id': model_id
                }
    
    def shutdown(self):
        """Shutdown the AGI Brain core"""
        with self.lock:
            logger.info("Shutting down AGI Brain core")
            self.is_initialized = False
            
            # Clean up resources
            for model_id, model in self.models.items():
                try:
                    logger.info(f"Cleaning up model: {model_id}")
                    # In a real implementation, this would properly shut down the model
                except Exception as e:
                    logger.error(f"Error cleaning up model {model_id}: {e}")
            
            self.models.clear()
            logger.info("AGI Brain core shutdown complete")