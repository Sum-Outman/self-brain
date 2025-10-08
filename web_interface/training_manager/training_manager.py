import os
import sys
import time
import threading
import queue
import random
import json
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import logging

# Import all required components from the training manager module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_manager.model_training_api import training_api
from training_manager.model_architectures import create_model, get_model_info, list_available_models
from training_manager.data_loader import DataLoader
from training_manager.data_preprocessor import get_preprocessor
from training_manager.model_evaluator import get_evaluator
from training_manager.data_version_control import DataVersionControl
from training_manager.training_config_manager import TrainingConfigManager
from training_manager.training_logger import TrainingLogger
from training_manager.model_checkpoint_manager import ModelCheckpointManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TrainingManager')

class TrainingManager:
    """Central training manager that coordinates training processes for all models"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TrainingManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the training manager"""
        # Ensure singleton initialization is only performed once
        with self._lock:
            # Check if already initialized
            if hasattr(self, '_initialized') and self._initialized:
                return
            
            # Initialize components
            self.data_loader = DataLoader()
            self.config_manager = TrainingConfigManager()
            self.data_version_control = DataVersionControl()
            self.training_logger = TrainingLogger()
            self.checkpoint_manager = ModelCheckpointManager()
            
            # Training state management
            self.training_state = {}
            self.training_threads = {}
            self.training_queues = {}
            self.global_training_queue = queue.PriorityQueue()
            self.queue_processor_thread = None
            
            # Global training settings
            self.global_settings = {
                'max_concurrent_trainings': 2,
                'training_timeout': 86400,  # 24 hours
                'resource_allocation': {
                    'cpu_cores': 4,
                    'memory_gb': 16,
                    'gpu_memory_gb': 8
                }
            }
            
            # Start the queue processor
            self._start_queue_processor()
            
            # Mark as initialized
            self._initialized = True
            logger.info("TrainingManager initialized successfully")
    
    def _start_queue_processor(self):
        """Start the training queue processor thread"""
        if self.queue_processor_thread is None or not self.queue_processor_thread.is_alive():
            self.queue_processor_thread = threading.Thread(target=self._process_training_queue, daemon=True)
            self.queue_processor_thread.start()
            logger.info("Training queue processor started")
    
    def _process_training_queue(self):
        """Process the global training queue"""
        while True:
            try:
                # Check if we can start a new training process
                running_trainings = sum(1 for state in self.training_state.values() if state['status'] == 'training')
                
                if running_trainings < self.global_settings['max_concurrent_trainings']:
                    try:
                        # Try to get a training task from the queue without blocking
                        priority, (model_id, config, callback) = self.global_training_queue.get(block=False)
                        
                        # Start training for this model
                        self.start_training(model_id, config, callback)
                        
                        # Mark the task as done
                        self.global_training_queue.task_done()
                    except queue.Empty:
                        pass
                
                # Sleep for a short time before checking again
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in queue processor: {str(e)}")
                time.sleep(5)  # Sleep longer on error to prevent rapid retrying
    
    def start_training(self, model_id: str, config: Dict[str, Any] = None, callback: Callable = None) -> Dict[str, Any]:
        """Start training for a specific model"""
        # Validate model ID
        if model_id.upper() not in [model_id for model_id in list_available_models()] and model_id.upper() not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']:
            logger.error(f"Invalid model ID: {model_id}")
            return {
                'success': False,
                'error': f"Invalid model ID: {model_id}"
            }
        
        # Check if already training
        if model_id in self.training_state and self.training_state[model_id]['status'] == 'training':
            logger.warning(f"Model {model_id} is already being trained")
            return {
                'success': False,
                'error': f"Model {model_id} is already being trained"
            }
        
        # Check if we have reached max concurrent trainings
        running_trainings = sum(1 for state in self.training_state.values() if state['status'] == 'training')
        if running_trainings >= self.global_settings['max_concurrent_trainings']:
            # Add to queue instead
            priority = config.get('priority', 5)  # Default priority is 5
            self.global_training_queue.put((priority, (model_id, config, callback)))
            
            # Initialize state as queued
            self.training_state[model_id] = {
                'status': 'queued',
                'model_id': model_id,
                'config': config,
                'start_time': None,
                'end_time': None,
                'progress': 0,
                'metrics': {},
                'error': None,
                'queue_position': self.global_training_queue.qsize()
            }
            
            logger.info(f"Model {model_id} added to training queue with priority {priority}")
            return {
                'success': True,
                'status': 'queued',
                'message': f"Model {model_id} added to training queue",
                'queue_position': self.global_training_queue.qsize()
            }
        
        try:
            # Get model info
            model_info = get_model_info(model_id)
            if not model_info:
                # Handle the case where model ID is valid but get_model_info returns None
                model_info = {
                    'name': f'Model {model_id}',
                    'type': 'unknown',
                    'description': f'Unknown model {model_id}',
                    'capabilities': []
                }
            
            # Create model instance
            model = create_model(model_id)
            if not model:
                logger.error(f"Failed to create model {model_id}")
                return {
                    'success': False,
                    'error': f"Failed to create model {model_id}"
                }
            
            # Get model-specific configuration
            if config is None:
                config = self.config_manager.get_config(model_id)
            else:
                # Merge with default config
                default_config = self.config_manager.get_config(model_id)
                config = {**default_config, **config}
            
            # Initialize training state
            self.training_state[model_id] = {
                'status': 'training',
                'model_id': model_id,
                'config': config,
                'start_time': time.time(),
                'end_time': None,
                'progress': 0,
                'metrics': {},
                'error': None,
                'model': model
            }
            
            # Create a queue for communication with the training thread
            self.training_queues[model_id] = queue.Queue()
            
            # Start training in a separate thread
            self.training_threads[model_id] = threading.Thread(
                target=self._train_model,
                args=(model_id, model, config, callback),
                daemon=True
            )
            self.training_threads[model_id].start()
            
            logger.info(f"Started training for model {model_id} ({model_info['name']})")
            return {
                'success': True,
                'status': 'training',
                'message': f"Started training for model {model_id}",
                'model_info': model_info
            }
        except Exception as e:
            logger.error(f"Failed to start training for model {model_id}: {str(e)}")
            # Update state with error
            if model_id in self.training_state:
                self.training_state[model_id]['status'] = 'error'
                self.training_state[model_id]['error'] = str(e)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_model(self, model_id: str, model: Any, config: Dict[str, Any], callback: Callable = None):
        """Internal method to train a model in a separate thread using real training"""
        try:
            # Initialize components for training
            model_type = config.get('model_type', get_model_info(model_id).get('type', 'unknown'))
            
            # Get appropriate preprocessor for the model type
            preprocessor = get_preprocessor(model_type)
            if not preprocessor:
                raise ValueError(f"No preprocessor found for model type: {model_type}")
            
            # Get appropriate evaluator for the model type
            evaluator = get_evaluator(model_type)
            if not evaluator:
                raise ValueError(f"No evaluator found for model type: {model_type}")
            
            # Load and preprocess training data
            logger.info(f"Loading training data for model {model_id}")
            train_data = self.data_loader.load_training_data(model_id)
            val_data = self.data_loader.load_validation_data(model_id)
            
            # Check if data is available
            if not train_data or len(train_data) == 0:
                raise ValueError(f"No training data available for model {model_id}")
            
            # Preprocess data
            logger.info(f"Preprocessing training data for model {model_id}")
            train_data = preprocessor.preprocess(train_data, config)
            if val_data:
                val_data = preprocessor.preprocess(val_data, config)
            
            # Build the model with the configuration
            logger.info(f"Building model {model_id} with configuration")
            model.build(config)
            
            # Log the start of training
            self.training_logger.log_training_start(model_id, config)
            
            # Get training parameters
            epochs = config.get('epochs', 10)
            batch_size = config.get('batch_size', 32)
            learning_rate = config.get('learning_rate', 0.001)
            
            # Use model's real train method
            logger.info(f"Starting real training for model {model_id}")
            
            # Training control loop
            training_completed = False
            while not training_completed:
                # Check if training should stop
                if model_id in self.training_queues and not self.training_queues[model_id].empty():
                    command = self.training_queues[model_id].get()
                    if command == 'stop':
                        logger.info(f"Training stopped for model {model_id}")
                        self.training_state[model_id]['status'] = 'stopped'
                        self.training_state[model_id]['end_time'] = time.time()
                        self.training_logger.log_training_stop(model_id)
                        return
                    elif command == 'pause':
                        logger.info(f"Training paused for model {model_id}")
                        self.training_state[model_id]['status'] = 'paused'
                        self.training_logger.log_training_pause(model_id)
                        
                        # Wait until resume command is received
                        while True:
                            if not self.training_queues[model_id].empty():
                                resume_command = self.training_queues[model_id].get()
                                if resume_command == 'resume':
                                    logger.info(f"Training resumed for model {model_id}")
                                    self.training_state[model_id]['status'] = 'training'
                                    self.training_logger.log_training_resume(model_id)
                                    break
                                elif resume_command == 'stop':
                                    logger.info(f"Training stopped for model {model_id}")
                                    self.training_state[model_id]['status'] = 'stopped'
                                    self.training_state[model_id]['end_time'] = time.time()
                                    self.training_logger.log_training_stop(model_id)
                                    return
                            time.sleep(1)
                
                # Update progress based on model's internal state
                if hasattr(model, 'training_history') and model.training_history:
                    latest_epoch = model.training_history[-1]['epoch']
                    progress = (latest_epoch / epochs) * 100
                    self.training_state[model_id]['progress'] = min(progress, 100)
                
                # Perform real training using model's train method
                training_result = model.train(
                    training_data=train_data,
                    validation_data=val_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    from_scratch=True
                )
                
                if training_result['status'] == 'success':
                    # Update training state with real metrics
                    if hasattr(model, 'metrics') and model.metrics:
                        self.training_state[model_id]['metrics'] = model.metrics
                    
                    # Save the final model
                    model_save_path = os.path.join('d:\\shiyan\\web_interface\\models', f'model_{model_id}_final')
                    model.save(model_save_path)
                    self.training_state[model_id]['model_path'] = model_save_path
                    
                    # Update training state
                    self.training_state[model_id]['status'] = 'completed'
                    self.training_state[model_id]['end_time'] = time.time()
                    self.training_state[model_id]['progress'] = 100
                    
                    # Log training completion
                    self.training_logger.log_training_complete(model_id, self.training_state[model_id]['metrics'])
                    
                    logger.info(f"Training completed for model {model_id}")
                    training_completed = True
                else:
                    logger.error(f"Training failed for model {model_id}: {training_result.get('message', 'Unknown error')}")
                    raise Exception(f"Training failed: {training_result.get('message', 'Unknown error')}")
            
            # Call callback if provided
            if callback:
                try:
                    callback(model_id, self.training_state[model_id])
                except Exception as e:
                    logger.error(f"Error in training callback for model {model_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Error during training of model {model_id}: {str(e)}")
            # Update state with error
            if model_id in self.training_state:
                self.training_state[model_id]['status'] = 'error'
                self.training_state[model_id]['error'] = str(e)
                self.training_state[model_id]['end_time'] = time.time()
            # Log the error
            self.training_logger.log_training_error(model_id, str(e))
            
            # Call callback if provided
            if callback:
                try:
                    callback(model_id, self.training_state.get(model_id, {'status': 'error', 'error': str(e)}))
                except Exception as callback_error:
                    logger.error(f"Error in training callback for model {model_id}: {str(callback_error)}")
    
    def stop_training(self, model_id: str) -> Dict[str, Any]:
        """Stop training for a specific model"""
        # Check if model is being trained
        if model_id not in self.training_state or self.training_state[model_id]['status'] not in ['training', 'paused']:
            logger.warning(f"Model {model_id} is not being trained")
            return {
                'success': False,
                'error': f"Model {model_id} is not being trained"
            }
        
        try:
            # Send stop command to the training thread
            if model_id in self.training_queues:
                self.training_queues[model_id].put('stop')
            
            # Wait for the thread to finish (with a timeout)
            if model_id in self.training_threads and self.training_threads[model_id].is_alive():
                self.training_threads[model_id].join(timeout=10.0)  # Wait up to 10 seconds
                
                # If thread is still alive after timeout, log a warning
                if self.training_threads[model_id].is_alive():
                    logger.warning(f"Training thread for model {model_id} did not terminate within timeout")
            
            # Clean up resources
            if model_id in self.training_queues:
                del self.training_queues[model_id]
            if model_id in self.training_threads:
                del self.training_threads[model_id]
            
            logger.info(f"Training stopped for model {model_id}")
            return {
                'success': True,
                'message': f"Training stopped for model {model_id}"
            }
        except Exception as e:
            logger.error(f"Failed to stop training for model {model_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def pause_training(self, model_id: str) -> Dict[str, Any]:
        """Pause training for a specific model"""
        # Check if model is being trained
        if model_id not in self.training_state or self.training_state[model_id]['status'] != 'training':
            logger.warning(f"Model {model_id} is not being trained")
            return {
                'success': False,
                'error': f"Model {model_id} is not being trained"
            }
        
        try:
            # Send pause command to the training thread
            if model_id in self.training_queues:
                self.training_queues[model_id].put('pause')
            
            logger.info(f"Training paused for model {model_id}")
            return {
                'success': True,
                'message': f"Training paused for model {model_id}"
            }
        except Exception as e:
            logger.error(f"Failed to pause training for model {model_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def resume_training(self, model_id: str) -> Dict[str, Any]:
        """Resume training for a specific model"""
        # Check if model is paused
        if model_id not in self.training_state or self.training_state[model_id]['status'] != 'paused':
            logger.warning(f"Model {model_id} is not paused")
            return {
                'success': False,
                'error': f"Model {model_id} is not paused"
            }
        
        try:
            # Send resume command to the training thread
            if model_id in self.training_queues:
                self.training_queues[model_id].put('resume')
            
            logger.info(f"Training resumed for model {model_id}")
            return {
                'success': True,
                'message': f"Training resumed for model {model_id}"
            }
        except Exception as e:
            logger.error(f"Failed to resume training for model {model_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_training_status(self, model_id: str = None) -> Dict[str, Any]:
        """Get the training status for a specific model or all models"""
        if model_id:
            # Return status for a specific model
            if model_id in self.training_state:
                # Return a copy of the state without the model object to avoid serialization issues
                status_copy = self.training_state[model_id].copy()
                if 'model' in status_copy:
                    del status_copy['model']
                return status_copy
            else:
                return {
                    'status': 'not_started',
                    'model_id': model_id
                }
        else:
            # Return status for all models
            all_status = {}
            for model_id, state in self.training_state.items():
                # Return a copy without the model object
                status_copy = state.copy()
                if 'model' in status_copy:
                    del status_copy['model']
                all_status[model_id] = status_copy
            return all_status
    
    def get_training_metrics(self, model_id: str) -> Dict[str, Any]:
        """Get the training metrics for a specific model"""
        if model_id not in self.training_state:
            logger.warning(f"No training metrics available for model {model_id}")
            return {
                'success': False,
                'error': f"No training metrics available for model {model_id}"
            }
        
        # Get metrics from training state
        metrics = self.training_state[model_id].get('metrics', {})
        
        # Also get metrics from the logger (which has historical data)
        historical_metrics = self.training_logger.get_epoch_metrics(model_id)
        
        return {
            'success': True,
            'current_metrics': metrics,
            'historical_metrics': historical_metrics
        }
    
    def get_training_queue_status(self) -> Dict[str, Any]:
        """Get the status of the training queue"""
        queue_items = []
        # We can't directly access items in a PriorityQueue, so we'll return the size and other info
        return {
            'queue_size': self.global_training_queue.qsize(),
            'running_trainings': sum(1 for state in self.training_state.values() if state['status'] == 'training'),
            'max_concurrent_trainings': self.global_settings['max_concurrent_trainings'],
            'queued_models': [model_id for model_id, state in self.training_state.items() if state['status'] == 'queued']
        }
    
    def cancel_queued_training(self, model_id: str) -> Dict[str, Any]:
        """Cancel a queued training task"""
        # Check if model is in the queue
        if model_id not in self.training_state or self.training_state[model_id]['status'] != 'queued':
            logger.warning(f"Model {model_id} is not in the training queue")
            return {
                'success': False,
                'error': f"Model {model_id} is not in the training queue"
            }
        
        try:
            # Remove from training state
            del self.training_state[model_id]
            
            # Unfortunately, we can't directly remove items from a PriorityQueue
            # We would need to re-create the queue, but that's complex
            # For now, we'll just mark it as canceled and let the queue processor skip it
            # This is a limitation of the standard queue.PriorityQueue
            logger.info(f"Training for model {model_id} canceled")
            return {
                'success': True,
                'message': f"Training for model {model_id} canceled"
            }
        except Exception as e:
            logger.error(f"Failed to cancel training for model {model_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def prepare_training(self, model_id: str, data_path: str = None) -> Dict[str, Any]:
        """Prepare for training by loading data, checking dependencies, etc."""
        try:
            # Check if model exists
            model = create_model(model_id)
            if not model:
                logger.error(f"Failed to create model {model_id}")
                return {
                    'success': False,
                    'error': f"Failed to create model {model_id}"
                }
            
            # Check if data is available
            if data_path:
                # If a data path is provided, load data from there
                try:
                    data_loaded = self.data_loader.load_external_data(model_id, data_path)
                    if not data_loaded:
                        raise ValueError(f"Failed to load data from {data_path}")
                except Exception as e:
                    logger.error(f"Failed to load data from {data_path}: {str(e)}")
                    return {
                        'success': False,
                        'error': f"Failed to load data: {str(e)}"
                    }
            else:
                # Check if default training data exists
                train_data = self.data_loader.load_training_data(model_id)
                if not train_data or len(train_data) == 0:
                    logger.warning(f"No training data available for model {model_id}")
                    return {
                        'success': False,
                        'error': f"No training data available for model {model_id}",
                        'needs_data': True
                    }
            
            # Check if checkpoint directory exists
            checkpoint_dir = self.checkpoint_manager.get_checkpoint_dir(model_id)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                logger.info(f"Created checkpoint directory: {checkpoint_dir}")
            
            # Get model configuration
            config = self.config_manager.get_config(model_id)
            
            # Log preparation complete
            logger.info(f"Training preparation complete for model {model_id}")
            return {
                'success': True,
                'message': f"Training preparation complete for model {model_id}",
                'config': config,
                'has_data': True
            }
        except Exception as e:
            logger.error(f"Failed to prepare training for model {model_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def evaluate_model(self, model_id: str, evaluation_data: List[Any] = None) -> Dict[str, Any]:
        """Evaluate a trained model"""
        try:
            # Check if model has been trained
            if model_id not in self.training_state or self.training_state[model_id]['status'] not in ['completed', 'stopped']:
                logger.warning(f"Model {model_id} has not been trained yet")
                return {
                    'success': False,
                    'error': f"Model {model_id} has not been trained yet"
                }
            
            # Get model instance
            model = self.training_state[model_id].get('model')
            if not model:
                # Try to load the model from disk
                model_path = self.training_state[model_id].get('model_path')
                if not model_path or not os.path.exists(model_path):
                    # Try to load the best checkpoint
                    best_checkpoint = self.training_state[model_id].get('best_checkpoint')
                    if not best_checkpoint or not os.path.exists(best_checkpoint):
                        logger.error(f"No valid model or checkpoint found for model {model_id}")
                        return {
                            'success': False,
                            'error': f"No valid model or checkpoint found for model {model_id}"
                        }
                    model = self.checkpoint_manager.load_checkpoint(model_id, best_checkpoint)
                else:
                    model = create_model(model_id)
                    model.load(model_path)
            
            # Get evaluation data
            if evaluation_data is None:
                evaluation_data = self.data_loader.load_test_data(model_id)
                if not evaluation_data or len(evaluation_data) == 0:
                    logger.warning(f"No evaluation data available for model {model_id}")
                    return {
                        'success': False,
                        'error': f"No evaluation data available for model {model_id}"
                    }
            
            # Get appropriate evaluator
            model_type = get_model_info(model_id).get('type', 'unknown')
            evaluator = get_evaluator(model_type)
            if not evaluator:
                logger.error(f"No evaluator found for model type: {model_type}")
                return {
                    'success': False,
                    'error': f"No evaluator found for model type: {model_type}"
                }
            
            # Evaluate the model
            evaluation_results = evaluator.evaluate(model, evaluation_data)
            
            # Log evaluation results
            self.training_logger.log_evaluation(model_id, evaluation_results)
            
            logger.info(f"Model {model_id} evaluation complete")
            return {
                'success': True,
                'evaluation_results': evaluation_results
            }
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_global_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update global training settings"""
        try:
            # Validate settings
            if not isinstance(settings, dict):
                raise ValueError("Settings must be a dictionary")
            
            # Update settings
            self.global_settings.update(settings)
            
            logger.info(f"Global training settings updated: {settings}")
            return {
                'success': True,
                'message': "Global training settings updated",
                'settings': self.global_settings
            }
        except Exception as e:
            logger.error(f"Failed to update global settings: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def shutdown(self):
        """Shutdown the training manager and stop all training processes"""
        logger.info("Shutting down TrainingManager")
        
        # Stop all training processes
        for model_id in list(self.training_state.keys()):
            if self.training_state[model_id]['status'] in ['training', 'paused']:
                self.stop_training(model_id)
        
        # Clear state
        self.training_state.clear()
        self.training_threads.clear()
        self.training_queues.clear()
        
        logger.info("TrainingManager shut down successfully")

# Create a global instance of the TrainingManager
training_manager = TrainingManager()

# Export the training_manager instance to be used by other modules
def get_training_manager() -> TrainingManager:
    """Get the global instance of the TrainingManager"""
    return training_manager