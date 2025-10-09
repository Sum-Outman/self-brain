# -*- coding: utf-8 -*-
"""
Advanced Training Control Module
This module provides centralized control for all model training operations in the Self Brain system.
"""

import logging
import json
import os
import time
import threading
import queue
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AdvancedTrainControl')

class TrainingMode(Enum):
    """Training modes enumeration"""
    INDIVIDUAL = 'individual'
    JOINT = 'joint'
    TRANSFER = 'transfer'
    FINE_TUNE = 'fine_tune'
    PRETRAINING = 'pretraining'

class TrainingController:
    """Central controller for managing model training operations"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TrainingController, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the training controller"""
        # Training session management
        self.training_sessions: Dict[str, Dict] = {}
        self.active_training: Optional[str] = None
        self.training_queue = queue.Queue()
        
        # Model registry - keeps track of all models and their training status
        self.model_registry: Dict[str, Dict] = {
            'A': {'name': 'Management Model', 'type': 'management', 'status': 'not_loaded'},
            'B': {'name': 'Language Model', 'type': 'language', 'status': 'not_loaded'},
            'C': {'name': 'Audio Processing Model', 'type': 'audio', 'status': 'not_loaded'},
            'D': {'name': 'Image Processing Model', 'type': 'image', 'status': 'not_loaded'},
            'E': {'name': 'Video Processing Model', 'type': 'video', 'status': 'not_loaded'},
            'F': {'name': 'Spatial Perception Model', 'type': 'spatial', 'status': 'not_loaded'},
            'G': {'name': 'Sensor Perception Model', 'type': 'sensor', 'status': 'not_loaded'},
            'H': {'name': 'Computer Control Model', 'type': 'computer', 'status': 'not_loaded'},
            'I': {'name': 'Motion Control Model', 'type': 'motion', 'status': 'not_loaded'},
            'J': {'name': 'Knowledge Base Model', 'type': 'knowledge', 'status': 'not_loaded'},
            'K': {'name': 'Programming Model', 'type': 'programming', 'status': 'not_loaded'}
        }
        
        # System health monitoring
        self.system_health = {
            'performance': {'total_trainings': 0, 'successful_trainings': 0, 'failed_trainings': 0},
            'training_controller': {'active_models': 0, 'queue_length': 0},
            'knowledge_base': {'connected': False, 'total_knowledge_items': 0, 'last_update': None},
            'collaboration': {'total_collaborations': 0, 'knowledge_sharing_events': 0}
        }
        
        # Thread management
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.queue_processor_thread: Optional[threading.Thread] = None
        self.is_queue_processing = False
        
        # Start queue processor
        self.start_queue_processor()
        
        # Configuration paths
        self.config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '../config'
        )
        self.training_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '../../training_data'
        )
        
        # Ensure directories exist
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        logger.info("Advanced Training Controller initialized successfully")
    
    def start_queue_processor(self):
        """Start the training queue processor thread"""
        if not self.is_queue_processing:
            self.is_queue_processing = True
            self.queue_processor_thread = threading.Thread(target=self._process_training_queue)
            self.queue_processor_thread.daemon = True
            self.queue_processor_thread.start()
            logger.info("Training queue processor started")
    
    def _process_training_queue(self):
        """Process training tasks from the queue"""
        while self.is_queue_processing:
            try:
                # Wait for a training task
                task = self.training_queue.get(timeout=1)
                
                # If not already training, start the new task
                if not self.is_training:
                    self._start_training(task)
                else:
                    # Otherwise, requeue the task
                    self.training_queue.put(task)
                    logger.info(f"Training already in progress, requeuing task: {task['training_id']}")
                
                # Mark the task as done
                self.training_queue.task_done()
            except queue.Empty:
                # Queue is empty, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error processing training queue: {str(e)}")
                time.sleep(1)
    
    def start_training(self, model_id: str, training_config: Dict[str, Any] = None) -> str:
        """Start training for a specific model"""
        # Validate model ID
        if model_id not in self.model_registry:
            raise ValueError(f"Invalid model ID: {model_id}")
        
        # Generate training ID
        training_id = f"training_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create training configuration
        config = training_config or {
            'model_id': model_id,
            'training_type': 'individual',
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'data_path': os.path.join(self.training_data_dir, model_id.lower())
        }
        
        # Create training session
        training_session = {
            'training_id': training_id,
            'model_id': model_id,
            'config': config,
            'status': 'queued',
            'start_time': None,
            'end_time': None,
            'progress': 0,
            'metrics': {}
        }
        
        # Store training session
        self.training_sessions[training_id] = training_session
        
        # Add to training queue
        self.training_queue.put(training_session)
        
        logger.info(f"Training session created: {training_id} for model {model_id}")
        
        return training_id
    
    def start_joint_training(self, model_ids: List[str], training_config: Dict[str, Any] = None) -> str:
        """Start joint training for multiple models"""
        # Validate model IDs
        for model_id in model_ids:
            if model_id not in self.model_registry:
                raise ValueError(f"Invalid model ID: {model_id}")
        
        # Generate training ID
        training_id = f"training_joint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create training configuration
        config = training_config or {
            'model_ids': model_ids,
            'training_type': 'joint',
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'data_path': os.path.join(self.training_data_dir, 'joint')
        }
        
        # Create training session
        training_session = {
            'training_id': training_id,
            'model_ids': model_ids,
            'config': config,
            'status': 'queued',
            'start_time': None,
            'end_time': None,
            'progress': 0,
            'metrics': {}
        }
        
        # Store training session
        self.training_sessions[training_id] = training_session
        
        # Add to training queue
        self.training_queue.put(training_session)
        
        logger.info(f"Joint training session created: {training_id} for models {', '.join(model_ids)}")
        
        return training_id
    
    def _start_training(self, training_session: Dict[str, Any]):
        """Start the actual training process in a separate thread"""
        # Update session status
        self.is_training = True
        self.active_training = training_session['training_id']
        training_session['status'] = 'training'
        training_session['start_time'] = datetime.now().isoformat()
        
        # Start training thread
        self.training_thread = threading.Thread(
            target=self._training_thread_func,
            args=(training_session,)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
        
        logger.info(f"Training started: {training_session['training_id']}")
    
    def _training_thread_func(self, training_session: Dict[str, Any]):
        """Training thread function"""
        try:
            # Extract session information
            training_id = training_session['training_id']
            training_type = training_session['config']['training_type']
            
            if training_type == 'individual':
                model_id = training_session['model_id']
                self._train_individual_model(training_session)
            else:
                model_ids = training_session['model_ids']
                self._train_joint_models(training_session)
            
            # Update system health stats
            self.system_health['performance']['total_trainings'] += 1
            self.system_health['performance']['successful_trainings'] += 1
            
            # Update session status
            training_session['status'] = 'completed'
            training_session['end_time'] = datetime.now().isoformat()
            training_session['progress'] = 100
            
            logger.info(f"Training completed successfully: {training_id}")
            
        except Exception as e:
            # Update system health stats
            self.system_health['performance']['total_trainings'] += 1
            self.system_health['performance']['failed_trainings'] += 1
            
            # Update session status
            training_session['status'] = 'failed'
            training_session['end_time'] = datetime.now().isoformat()
            training_session['error'] = str(e)
            
            logger.error(f"Training failed: {training_id}, Error: {str(e)}")
        finally:
            # Mark training as complete
            self.is_training = False
            self.active_training = None
    
    def _train_individual_model(self, training_session: Dict[str, Any]):
        """Train a single model"""
        model_id = training_session['model_id']
        config = training_session['config']
        epochs = config.get('epochs', 10)
        
        # Update model status
        self.model_registry[model_id]['status'] = 'training'
        self.system_health['training_controller']['active_models'] += 1
        
        try:
            # Simulate training process
            for epoch in range(epochs):
                # Update progress
                progress = int(((epoch + 1) / epochs) * 100)
                training_session['progress'] = progress
                
                # Simulate training metrics
                training_session['metrics'] = {
                    'epoch': epoch + 1,
                    'loss': 0.5 * (1 - progress/100),
                    'accuracy': 0.5 + (progress/200),
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Model {model_id} training epoch {epoch+1}/{epochs}, Progress: {progress}%")
                
                # Simulate training time
                time.sleep(2)
            
            # Save training results
            self._save_training_results(training_session)
            
        finally:
            # Update model status
            self.model_registry[model_id]['status'] = 'trained'
            self.model_registry[model_id]['last_trained'] = datetime.now().isoformat()
            self.system_health['training_controller']['active_models'] -= 1
    
    def _train_joint_models(self, training_session: Dict[str, Any]):
        """Train multiple models jointly"""
        model_ids = training_session['model_ids']
        config = training_session['config']
        epochs = config.get('epochs', 10)
        
        # Update models status
        for model_id in model_ids:
            self.model_registry[model_id]['status'] = 'training'
            self.system_health['training_controller']['active_models'] += 1
        
        try:
            # Simulate joint training process
            for epoch in range(epochs):
                # Update progress
                progress = int(((epoch + 1) / epochs) * 100)
                training_session['progress'] = progress
                
                # Simulate training metrics for each model
                model_metrics = {}
                for model_id in model_ids:
                    model_metrics[model_id] = {
                        'epoch': epoch + 1,
                        'loss': 0.5 * (1 - progress/100),
                        'accuracy': 0.5 + (progress/200),
                        'timestamp': datetime.now().isoformat()
                    }
                
                training_session['metrics'] = model_metrics
                
                logger.info(f"Joint training epoch {epoch+1}/{epochs}, Progress: {progress}%")
                
                # Simulate training time
                time.sleep(3)
            
            # Save training results
            self._save_training_results(training_session)
            
        finally:
            # Update models status
            for model_id in model_ids:
                self.model_registry[model_id]['status'] = 'trained'
                self.model_registry[model_id]['last_trained'] = datetime.now().isoformat()
                self.system_health['training_controller']['active_models'] -= 1
    
    def _save_training_results(self, training_session: Dict[str, Any]):
        """Save training results to a file"""
        try:
            # Create results directory
            results_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                '../../training_results'
            )
            os.makedirs(results_dir, exist_ok=True)
            
            # Save results to JSON file
            results_file = os.path.join(results_dir, f"{training_session['training_id']}_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(training_session, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Training results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save training results: {str(e)}")
    
    def stop_training(self, training_id: str = None) -> bool:
        """Stop a specific training session or the active one"""
        # If no training ID is provided, stop the active training
        if training_id is None:
            training_id = self.active_training
        
        if training_id and training_id in self.training_sessions:
            session = self.training_sessions[training_id]
            
            # Update session status
            session['status'] = 'stopped'
            session['end_time'] = datetime.now().isoformat()
            
            logger.info(f"Training stopped: {training_id}")
            
            return True
        
        logger.warning(f"No active training session found with ID: {training_id}")
        return False
    
    def get_training_status(self, training_id: str = None) -> Dict[str, Any]:
        """Get the status of a specific training session or the active one"""
        # If no training ID is provided, get the active training status
        if training_id is None:
            training_id = self.active_training
        
        if training_id and training_id in self.training_sessions:
            return {
                'training_id': training_id,
                'overall_status': self.training_sessions[training_id]['status'],
                'training_mode': self.training_sessions[training_id]['config'].get('training_type', 'individual'),
                'progress': self.training_sessions[training_id]['progress'],
                'time_info': {
                    'start_time': self.training_sessions[training_id]['start_time'],
                    'end_time': self.training_sessions[training_id]['end_time']
                },
                'active_models': self.training_sessions[training_id].get('model_ids', [self.training_sessions[training_id].get('model_id')])
            }
        
        # Return empty status if no session found
        return {
            'training_id': 'none',
            'overall_status': 'idle',
            'training_mode': 'none',
            'progress': 0,
            'time_info': {
                'start_time': None,
                'end_time': None
            },
            'active_models': []
        }
    
    def get_model_registry(self) -> Dict[str, Dict]:
        """Get the complete model registry"""
        return self.model_registry.copy()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get the current system health status"""
        # Update queue length
        self.system_health['training_controller']['queue_length'] = self.training_queue.qsize()
        
        return self.system_health.copy()
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time training metrics"""
        if self.active_training and self.active_training in self.training_sessions:
            session = self.training_sessions[self.active_training]
            return {
                'training_id': self.active_training,
                'progress': session['progress'],
                'metrics': session['metrics'],
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'training_id': 'none',
            'progress': 0,
            'metrics': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Shut down the training controller"""
        # Stop queue processing
        self.is_queue_processing = False
        
        # Stop active training
        if self.active_training:
            self.stop_training(self.active_training)
        
        # Wait for threads to finish
        if self.queue_processor_thread and self.queue_processor_thread.is_alive():
            self.queue_processor_thread.join(timeout=5.0)
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)
        
        logger.info("Advanced Training Controller shut down")

# Singleton access function
def get_training_controller() -> TrainingController:
    """Get the singleton instance of the TrainingController"""
    return TrainingController()

# Initialize the training controller when the module is loaded
training_controller = get_training_controller()
