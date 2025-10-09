import logging
<<<<<<< HEAD
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
=======
import threading
import queue
from datetime import datetime
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TrainingController')

class TrainingController:
    """Controller for managing all training processes across different models"""
    
    def __init__(self):
        self.training_queues = {}
        self.active_sessions = {}
        self.completed_sessions = {}
        self.failed_sessions = {}
        self.lock = threading.RLock()
        self.is_running = False
        self.thread = None
        self.model_configs = {}
        self.global_training_config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'use_gpu': True,
            'early_stopping': True,
            'validation_split': 0.2
        }
    
    def start(self):
        """Start the training controller"""
        with self.lock:
            if self.is_running:
                return
            
            self.is_running = True
            self.thread = threading.Thread(target=self._process_training_queues, daemon=True)
            self.thread.start()
            logger.info("TrainingController started")
    
    def stop(self):
        """Stop the training controller"""
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
            if self.thread:
                self.thread.join(2.0)
            logger.info("TrainingController stopped")
    
    def _process_training_queues(self):
        """Process training jobs from all queues"""
        while self.is_running:
            # In a real implementation, this would process training jobs from all queues
            # and manage the training sessions
            time.sleep(1.0)
    
    def add_training_job(self, model_id, job_config):
        """Add a training job to the queue"""
        with self.lock:
            if model_id not in self.training_queues:
                self.training_queues[model_id] = queue.Queue()
            
            # Generate a unique session ID
            session_id = f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            job_config['session_id'] = session_id
            job_config['start_time'] = datetime.now().isoformat()
            job_config['status'] = 'queued'
            
            # Add to queue
            self.training_queues[model_id].put(job_config)
            
            logger.info(f"Added training job {session_id} for model {model_id}")
            
            return session_id
    
    def get_training_status(self, session_id=None):
        """Get the status of a training session or all sessions"""
        with self.lock:
            if session_id:
                # Check all session dictionaries
                if session_id in self.active_sessions:
                    return self.active_sessions[session_id]
                elif session_id in self.completed_sessions:
                    return self.completed_sessions[session_id]
                elif session_id in self.failed_sessions:
                    return self.failed_sessions[session_id]
                else:
                    # Check if it's still in the queue
                    for model_id, queue in self.training_queues.items():
                        if any(job['session_id'] == session_id for job in list(queue.queue)):
                            return {'session_id': session_id, 'status': 'queued'}
                    
                    return {'session_id': session_id, 'status': 'not_found'}
            else:
                # Return status of all sessions
                return {
                    'active': list(self.active_sessions.values()),
                    'completed': list(self.completed_sessions.values()),
                    'failed': list(self.failed_sessions.values()),
                    'queued': {
                        model_id: list(queue.queue) for model_id, queue in self.training_queues.items()
                    }
                }
    
    def cancel_training(self, session_id):
        """Cancel a training session"""
        with self.lock:
            # Check if it's in active sessions
            if session_id in self.active_sessions:
                # In a real implementation, this would send a cancel signal to the training process
                self.active_sessions[session_id]['status'] = 'cancelled'
                self.active_sessions[session_id]['end_time'] = datetime.now().isoformat()
                
                # Move to failed sessions
                self.failed_sessions[session_id] = self.active_sessions.pop(session_id)
                
                logger.info(f"Cancelled training session {session_id}")
                return True
            
            # Check if it's in the queue
            for model_id, queue in self.training_queues.items():
                queue_items = list(queue.queue)
                for i, item in enumerate(queue_items):
                    if item['session_id'] == session_id:
                        # Remove from queue
                        queue_items.pop(i)
                        # Recreate the queue
                        self.training_queues[model_id] = queue.Queue()
                        for item in queue_items:
                            self.training_queues[model_id].put(item)
                        
                        logger.info(f"Removed training job {session_id} from queue")
                        return True
            
            logger.warning(f"Training session {session_id} not found for cancellation")
            return False
    
    def update_model_config(self, model_id, config):
        """Update configuration for a specific model"""
        with self.lock:
            self.model_configs[model_id] = config
            logger.info(f"Updated configuration for model {model_id}")
    
    def get_model_config(self, model_id):
        """Get configuration for a specific model"""
        with self.lock:
            return self.model_configs.get(model_id, {})
    
    def update_global_config(self, config):
        """Update global training configuration"""
        with self.lock:
            self.global_training_config.update(config)
            logger.info("Updated global training configuration")
    
    def get_global_config(self):
        """Get global training configuration"""
        with self.lock:
            return self.global_training_config.copy()

# Global instance cache
_training_controller_instance = None

from flask import Blueprint, jsonify, request

# Create blueprint
training_control_bp = Blueprint('training_control', __name__)

@training_control_bp.route('/api/training/jobs', methods=['POST'])
def add_training_job():
    """Add a training job"""
    data = request.json
    model_id = data.get('model_id')
    job_config = data.get('config', {})
    
    if not model_id:
        return jsonify({'status': 'error', 'message': 'Missing model_id parameter'}), 400
    
    controller = get_training_controller()
    session_id = controller.add_training_job(model_id, job_config)
    
    return jsonify({
        'status': 'success',
        'session_id': session_id,
        'message': f'Training job added for model {model_id}'
    })

@training_control_bp.route('/api/training/status', methods=['GET'])
def get_training_status_route():
    """Get training status"""
    session_id = request.args.get('session_id')
    
    controller = get_training_controller()
    status = controller.get_training_status(session_id)
    
    return jsonify({
        'status': 'success',
        'result': status
    })

@training_control_bp.route('/api/training/cancel', methods=['POST'])
def cancel_training_route():
    """Cancel a training session"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'status': 'error', 'message': 'Missing session_id parameter'}), 400
    
    controller = get_training_controller()
    success = controller.cancel_training(session_id)
    
    if success:
        return jsonify({'status': 'success', 'message': f'Training session {session_id} cancelled'})
    else:
        return jsonify({'status': 'error', 'message': f'Training session {session_id} not found'}), 404

@training_control_bp.route('/api/training/config', methods=['GET', 'POST'])
def training_config():
    """Get or update training configuration"""
    controller = get_training_controller()
    
    if request.method == 'GET':
        model_id = request.args.get('model_id')
        if model_id:
            config = controller.get_model_config(model_id)
        else:
            config = controller.get_global_config()
        
        return jsonify({
            'status': 'success',
            'config': config
        })
    
    elif request.method == 'POST':
        data = request.json
        model_id = data.get('model_id')
        config = data.get('config', {})
        
        if model_id:
            controller.update_model_config(model_id, config)
            message = f'Configuration updated for model {model_id}'
        else:
            controller.update_global_config(config)
            message = 'Global configuration updated'
        
        return jsonify({
            'status': 'success',
            'message': message
        })

def get_training_controller():
    """Get a singleton instance of TrainingController"""
    global _training_controller_instance
    if _training_controller_instance is None:
        _training_controller_instance = TrainingController()
        _training_controller_instance.start()
        logger.info("TrainingController initialized")
    return _training_controller_instance

# Import time at the top of the file if not already present
import time
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
