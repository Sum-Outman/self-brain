import logging
import time
import os
import json
from threading import Lock, Thread
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelStatus:
    """Model status representation with detailed information"""
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.active = True  # Default to active state
        self.training = False
        self.training_progress = 0.0
        self.last_active_time = time.time()
        self.status_message = "Model initialized"
        self.error_count = 0
        self.performance_metrics = {}
        self.api_connected = False
        self.api_status = ""
        
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'model_id': self.model_id,
            'active': self.active,
            'training': self.training,
            'training_progress': self.training_progress,
            'last_active_time': self.last_active_time,
            'status_message': self.status_message,
            'error_count': self.error_count,
            'performance_metrics': self.performance_metrics,
            'api_connected': self.api_connected,
            'api_status': self.api_status
        }

class TrainingController:
    """Enhanced training controller for managing all models"""
    def __init__(self):
        self._lock = Lock()
        self._model_statuses: Dict[str, ModelStatus] = {}
        self._model_registry = {}
        self._system_health = {
            'status': 'running',
            'last_health_check': time.time(),
            'error_count': 0
        }
        self._training_sessions = {}
        self._initialized = False
        self._registry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'config', 'model_registry.json'
        )
        
        # Initialize the controller
        self._initialize()
        
        # Start background monitoring thread
        self._monitor_thread = Thread(target=self._monitor_system, daemon=True)
        self._monitor_thread.start()
    
    def _initialize(self):
        """Initialize the training controller"""
        try:
            # Load model registry
            if os.path.exists(self._registry_path):
                with open(self._registry_path, 'r', encoding='utf-8') as f:
                    self._model_registry = json.load(f)
                
                # Create ModelStatus for each model in the registry and mark them as active
                with self._lock:
                    for model_id in self._model_registry:
                        self._model_statuses[model_id] = ModelStatus(model_id)
                        self._model_statuses[model_id].active = True
                        self._model_statuses[model_id].status_message = "Model registered and active"
                        
                logger.info(f"Successfully loaded {len(self._model_registry)} models from registry")
            else:
                logger.warning(f"Model registry file not found at {self._registry_path}")
            
            self._initialized = True
            self._system_health['status'] = 'running'
            logger.info("Training controller initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize training controller: {str(e)}")
            self._system_health['status'] = 'error'
            self._system_health['error_count'] += 1
    
    def _monitor_system(self):
        """Monitor system health and model statuses in background"""
        while True:
            try:
                with self._lock:
                    # Update last active time for all models
                    for model_status in self._model_statuses.values():
                        if model_status.active:
                            model_status.last_active_time = time.time()
                    
                    # Update system health check time
                    self._system_health['last_health_check'] = time.time()
                
            except Exception as e:
                logger.error(f"System monitoring error: {str(e)}")
                self._system_health['error_count'] += 1
            
            # Sleep for 10 seconds before next check
            time.sleep(10)
    
    def get_model_registry(self) -> Dict:
        """Get the complete model registry"""
        return self._model_registry.copy()
    
    def get_training_history(self) -> List:
        """Get training history"""
        return []  # Placeholder for future implementation
    
    def get_system_health(self) -> Dict:
        """Get current system health status"""
        return self._system_health.copy()
    
    def start_training(self, *args, **kwargs) -> Dict:
        """Start training for a model"""
        try:
            model_id = kwargs.get('model_id')
            if model_id and model_id in self._model_statuses:
                with self._lock:
                    self._model_statuses[model_id].training = True
                    self._model_statuses[model_id].training_progress = 0.0
                    self._model_statuses[model_id].status_message = "Training started"
                logger.info(f"Started training for model: {model_id}")
                return {'status': 'success', 'message': f'Training started for model {model_id}'}
            return {'status': 'error', 'message': 'Invalid model ID'}
        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def stop_training(self, *args, **kwargs) -> bool:
        """Stop training for a model"""
        try:
            model_id = kwargs.get('model_id')
            if model_id and model_id in self._model_statuses:
                with self._lock:
                    self._model_statuses[model_id].training = False
                    self._model_statuses[model_id].status_message = "Training stopped"
                logger.info(f"Stopped training for model: {model_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop training: {str(e)}")
            return False
    
    def load_model(self, model_id: str, config: Dict = None) -> bool:
        """Load a model with configuration"""
        try:
            with self._lock:
                if model_id not in self._model_statuses:
                    self._model_statuses[model_id] = ModelStatus(model_id)
                self._model_statuses[model_id].active = True
                self._model_statuses[model_id].status_message = "Model loaded successfully"
                
                # Update registry if needed
                if model_id not in self._model_registry and config:
                    self._model_registry[model_id] = config
            
            logger.info(f"Model loaded: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return False
    
    def update_model_configuration(self, model_id: str, config: Dict) -> bool:
        """Update model configuration"""
        try:
            with self._lock:
                if model_id in self._model_registry:
                    self._model_registry[model_id].update(config)
            logger.info(f"Updated configuration for model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update model configuration: {str(e)}")
            return False
    
    def get_model_configuration(self, model_id: str) -> Dict:
        """Get model configuration"""
        return self._model_registry.get(model_id, {})
    
    def start_model_service(self, model_id: str) -> bool:
        """Start model service"""
        try:
            with self._lock:
                if model_id in self._model_statuses:
                    self._model_statuses[model_id].active = True
                    self._model_statuses[model_id].status_message = "Model service started"
            logger.info(f"Started service for model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start model service: {str(e)}")
            return False
    
    def stop_model_service(self, model_id: str) -> bool:
        """Stop model service"""
        try:
            with self._lock:
                if model_id in self._model_statuses:
                    self._model_statuses[model_id].active = False
                    self._model_statuses[model_id].status_message = "Model service stopped"
            logger.info(f"Stopped service for model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop model service: {str(e)}")
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model"""
        try:
            with self._lock:
                if model_id in self._model_statuses:
                    del self._model_statuses[model_id]
                if model_id in self._model_registry:
                    del self._model_registry[model_id]
            logger.info(f"Deleted model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model: {str(e)}")
            return False
    
    def get_training_modes(self) -> List[str]:
        """Get available training modes"""
        return ['individual', 'joint', 'knowledge_transfer', 'reinforcement']
    
    def get_performance_analytics(self) -> Dict:
        """Get performance analytics"""
        analytics = {}
        with self._lock:
            for model_id, status in self._model_statuses.items():
                analytics[model_id] = {
                    'active': status.active,
                    'training': status.training,
                    'training_progress': status.training_progress,
                    'status_message': status.status_message,
                    'last_active_time': status.last_active_time
                }
        return analytics
    
    def get_knowledge_base_status(self) -> Dict:
        """Get knowledge base status"""
        return {'status': 'available', 'size': 'N/A', 'last_update': time.time()}
    
    def update_knowledge_base(self, *args, **kwargs) -> bool:
        """Update knowledge base"""
        logger.info("Knowledge base update requested")
        return True

# Singleton instance of TrainingController
training_controller_instance = None

def get_training_controller() -> TrainingController:
    """Get singleton instance of TrainingController"""
    global training_controller_instance
    if training_controller_instance is None:
        training_controller_instance = TrainingController()
    return training_controller_instance