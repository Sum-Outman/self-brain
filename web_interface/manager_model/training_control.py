import logging
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