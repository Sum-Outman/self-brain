# Copyright 2025 The AI Management System Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Training Scheduler

import threading
import time
import queue
import os
import json
import logging
import importlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import traceback
import numpy as np
import psutil
from manager_model.language_resources import get_string
from manager_model.model_registry import ModelRegistry
from manager_model.data_broker import DataBroker
from manager_model.data_bus import DataBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrainScheduler")

# Knowledge base model interface
KNOWLEDGE_MODEL_ID = "I_knowledge"

class TrainingPriority:
    """Training task priority enumeration"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 5
    LOW = 8
    BACKGROUND = 10

class TrainingTask:
    def __init__(self, model_ids: List[str], training_type: str, config: Dict, 
                 lang: str = "zh", priority=TrainingPriority.MEDIUM, 
                 scheduled_time: Optional[datetime] = None):
        """Initialize training task
        Args:
            model_ids: List of model IDs
            training_type: Training type ('single' or 'joint')
            config: Training configuration
            lang: Language code (zh/en/de/ja/ru)
            priority: Task priority (0-10, 0 is highest)
            scheduled_time: Scheduled start time (optional)
        """
        self.id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{training_type}"
        self.model_ids = model_ids
        self.training_type = training_type
        self.config = config
        self.lang = lang
        self.priority = priority
        self.scheduled_time = scheduled_time or datetime.now()
        self.created_at = datetime.now()
        self.status = "pending"
        self.start_time = None
        self.end_time = None
        self.progress = 0
        self.logs = []
        self.epochs = config.get('epochs', 10)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.current_epoch = 0
        self.metrics = {}
        self.knowledge_assisted = config.get('knowledge_assisted', False)
        self.efficiency_metrics = {
            "start_time": None,
            "last_epoch_time": None,
            "epoch_times": [],
            "resource_usage": []
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return {
            'task_id': self.id,
            'model_ids': self.model_ids,
            'training_type': self.training_type,
            'config': self.config,
            'lang': self.lang,
            'priority': self.priority,
            'scheduled_time': self.scheduled_time.isoformat() if self.scheduled_time else None,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'progress': self.progress,
            'current_epoch': self.current_epoch,
            'metrics': self.metrics,
            'knowledge_assisted': self.knowledge_assisted
        }
        
    def start(self):
        """Start training task"""
        self.status = "running"
        self.start_time = datetime.now()
        self.efficiency_metrics["start_time"] = self.start_time
        self.log(get_string("training_task_started", self.lang).format(task_id=self.id))
        self.log(get_string("training_type_models", self.lang).format(
            training_type=self.training_type, 
            models=", ".join(self.model_ids)
        ))
        self.log(get_string("training_config", self.lang).format(config=json.dumps(self.config, indent=2)))
        
        # Initialize knowledge assistance
        if self.knowledge_assisted:
            self._init_knowledge_assistance()
        
    def update_progress(self, progress: int, epoch: int = None, metrics: Dict = None):
        """Update training progress
        Args:
            progress: Progress percentage
            epoch: Current epoch
            metrics: Training metrics
        """
        self.progress = progress
        if epoch is not None:
            self.current_epoch = epoch
        if metrics:
            self.metrics.update(metrics)
        
        progress_msg = f"Progress update: {progress}%"
        if epoch is not None:
            progress_msg += f", Epoch: {epoch}/{self.epochs}"
        if metrics:
            progress_msg += f", Metrics: {metrics}"
        
        self.log(progress_msg)
        
    def complete(self, final_metrics: Dict = None):
        """Complete training task
        Args:
            final_metrics: Final metrics
        """
        self.status = "completed"
        self.end_time = datetime.now()
        self.progress = 100
        if final_metrics:
            self.metrics.update(final_metrics)
        
        duration = (self.end_time - self.start_time).total_seconds()
        self.log(f"Training task {self.id} completed")
        self.log(f"Total duration: {duration:.2f} seconds")
        self.log(f"Final metrics: {self.metrics}")
        
    def fail(self, reason: str):
        """Mark task as failed
        Args:
            reason: Failure reason
        """
        self.status = "failed"
        self.end_time = datetime.now()
        self.log(f"Training task failed: {reason}")
        
    def log(self, message: str, lang: str = None):
        """Add log entry
        Args:
            message: Log message
            lang: Language code (overrides task language)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        logger.info(log_entry)
        
    def log_multilingual(self, key: str, **kwargs):
        """Add multilingual log entry
        Args:
            key: Language resource key
            kwargs: Format arguments
        """
        message = get_string(key, self.lang).format(**kwargs)
        self.log(message)
        
    def _init_knowledge_assistance(self):
        """Initialize knowledge-assisted training"""
        try:
            # Get knowledge model
            model_registry = ModelRegistry()
            knowledge_model = model_registry.get_model(KNOWLEDGE_MODEL_ID)
            
            if knowledge_model and knowledge_model["active"]:
                # Get training advice
                advice = knowledge_model["instance"].get_training_advice(
                    model_ids=self.model_ids,
                    training_type=self.training_type,
                    config=self.config
                )
                
                if advice:
                    self.log_multilingual("knowledge_advice_received", advice=advice)
                    # Apply knowledge base suggestions
                    if "learning_rate" in advice:
                        self.learning_rate = advice["learning_rate"]
                    if "batch_size" in advice:
                        self.batch_size = advice["batch_size"]
                    # Other optimization parameters...
            else:
                self.log_multilingual("knowledge_model_inactive")
        except Exception as e:
            self.log_multilingual("knowledge_init_failed", error=str(e))

class TrainScheduler:
    def __init__(self, data_bus: Optional[DataBus] = None, max_concurrent_tasks: int = 2):
        """Initialize training scheduler
        Args:
            data_bus: Data bus instance for communication
            max_concurrent_tasks: Maximum number of concurrent training tasks
        """
        # Basic attributes
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []
        self.entry_count = 0
        self.training_lock = threading.Lock()
        
        # Enhanced features
        self.data_bus = data_bus
        self.max_concurrent_tasks = max_concurrent_tasks
        self.resource_limits = {
            'cpu_usage': 80.0,  # CPU usage limit percentage
            'memory_usage': 85.0  # Memory usage limit percentage
        }
        
        # Critical: Set running flag before starting worker thread
        self.running = True
        self.progress_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        
        # Now start the worker thread after all attributes are initialized
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
    def start_training(self, model_ids: List[str], training_type: str, 
                      epochs: int = 10, batch_size: int = 32, 
                      learning_rate: float = 0.001, knowledge_assisted: bool = False) -> str:
        """Start training
        Args:
            model_ids: List of model IDs
            training_type: Training type ('single' or 'joint')
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            knowledge_assisted: Whether to use knowledge assistance
        Returns:
            Task ID
        """
        config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'knowledge_assisted': knowledge_assisted
        }
        
        task = TrainingTask(model_ids, training_type, config)
        return self.add_task(task)
    
    def add_task(self, task: TrainingTask) -> str:
        """Add a training task to the queue
        Args:
            task: Training task object
        Returns:
            Task ID
        """
        self.entry_count += 1
        self.task_queue.put((task.priority, self.entry_count, task))
        
        # Notify status update via data bus if available
        self._notify_status_update(task)
        
        logger.info(f"Task added to queue: {task.id}, Models: {task.model_ids}, Priority: {task.priority}")
        return task.id
    
    def schedule_training(self, model_ids: List[str], training_type: str, scheduled_time: datetime, 
                         epochs: int = 10, batch_size: int = 32, 
                         learning_rate: float = 0.001, knowledge_assisted: bool = False, 
                         priority: int = TrainingPriority.MEDIUM) -> str:
        """Schedule training for a specific time
        Args:
            model_ids: List of model IDs
            training_type: Training type ('single' or 'joint')
            scheduled_time: Scheduled start time
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            knowledge_assisted: Whether to use knowledge assistance
            priority: Task priority
        Returns:
            Task ID
        """
        config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'knowledge_assisted': knowledge_assisted
        }
        
        task = TrainingTask(model_ids, training_type, config, scheduled_time=scheduled_time, priority=priority)
        return self.add_task(task)
        
    def _worker(self):
        """Worker thread to process tasks"""
        while self.running:
            try:
                # Check if we can start a new task based on resource availability and concurrency limit
                if len(self.active_tasks) < self.max_concurrent_tasks and self._check_resource_availability():
                    try:
                        # Try to get a task without blocking
                        priority, count, task = self.task_queue.get_nowait()
                        if task is None:
                            break
                             
                        # Check if task is scheduled for now or later
                        if task.scheduled_time <= datetime.now():
                            with self.training_lock:
                                self.active_tasks[task.id] = task
                            task.start()
                        else:
                            # Put task back to queue if not time yet
                            self.task_queue.put((priority, count, task))
                            time.sleep(1)  # Sleep briefly before next check
                            continue
                    except queue.Empty:
                        # No tasks in queue, wait briefly
                        time.sleep(1)
                        continue
                else:
                    # Resource limit reached or max concurrent tasks running
                    time.sleep(1)
                    continue
                
                try:
                    # Execute actual training
                    if task.training_type == 'single':
                        self._execute_single_training(task)
                    elif task.training_type == 'joint':
                        self._execute_joint_training(task)
                    else:
                        raise ValueError(f"Unsupported training type: {task.training_type}")
                        
                    # Only mark as completed if not stopped
                    if task.status != "stopped":
                        task.complete()
                    
                except Exception as e:
                    error_msg = f"Training error: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    task.fail(error_msg)
                    self.failed_tasks.append(task)
                    # Record detailed error information to system log
                    logger.critical(f"Training task {task.id} critical error: {traceback.format_exc()}")
                    
                finally:
                    self.completed_tasks.append(task)
                    if task.id in self.active_tasks:
                        del self.active_tasks[task.id]
                    self.task_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Worker thread error: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(5)  # Prevent infinite loop
    
    def _check_resource_availability(self) -> bool:
        """Check if system resources are available for training
        Returns:
            True if resources are available, False otherwise
        """
        try:
            # Get CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            # Get memory usage
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            
            logger.debug(f"System resource usage - CPU: {cpu_usage}%, Memory: {memory_usage}%")
            
            # Check if below thresholds
            if cpu_usage < self.resource_limits['cpu_usage'] and memory_usage < self.resource_limits['memory_usage']:
                return True
            
            # For critical priority tasks, we might allow higher resource usage
            # Peek at the next task's priority
            if not self.task_queue.empty():
                next_priority = self.task_queue.queue[0][0]  # First element is priority
                if next_priority <= TrainingPriority.HIGH:
                    # Allow slightly higher resource usage for high priority tasks
                    return cpu_usage < 90.0 and memory_usage < 90.0
            
            return False
            
        except Exception as e:
            logger.error(f"Resource check error: {str(e)}")
            # In case of error, be conservative and return False
            return False
    
    def _notify_status_update(self, task: TrainingTask):
        """Notify status update via data bus and callbacks
        Args:
            task: Training task
        """
        # Publish status update via data bus if available
        if self.data_bus:
            try:
                self.data_bus.publish_message(
                    'training_status',
                    task.to_dict()
                )
            except Exception as e:
                logger.error(f"Failed to publish status update: {e}")
        
        # Call status callbacks
        for callback in self.status_callbacks:
            try:
                callback(task.to_dict())
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    def _notify_progress_update(self, task: TrainingTask):
        """Notify progress update via data bus and callbacks
        Args:
            task: Training task
        """
        # Publish progress update via data bus if available
        if self.data_bus:
            try:
                self.data_bus.publish_message(
                    'training_progress',
                    {
                        'task_id': task.id,
                        'progress': task.progress,
                        'current_epoch': task.current_epoch,
                        'total_epochs': task.epochs
                    }
                )
            except Exception as e:
                logger.error(f"Failed to publish progress update: {e}")
        
        # Call progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback({
                    'task_id': task.id,
                    'progress': task.progress,
                    'current_epoch': task.current_epoch,
                    'total_epochs': task.epochs
                })
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def _execute_single_training(self, task: TrainingTask):
        """Execute single model training with real training logic
        Args:
            task: Training task
        """
        if len(task.model_ids) != 1:
            raise ValueError("Single model training can only have one model ID")
            
        model_id = task.model_ids[0]
        
        try:
            # Import the advanced training controller
            from training_manager.advanced_train_control import AdvancedTrainingController
            
            # Create training controller instance
            controller = AdvancedTrainingController()
            
            # Prepare training configuration
            config = {
                'epochs': task.epochs,
                'batch_size': task.batch_size,
                'learning_rate': task.learning_rate,
                'knowledge_assisted': task.knowledge_assisted
            }
            
            task.log(f"Starting real training for model {model_id}")
            task.log(f"Configuration: {config}")
            
            # Execute real training
            result = controller._execute_real_model_training(model_id, task.epochs, config)
            
            if result["status"] == "success":
                # Update task with real training results
                task.update_progress(100, task.epochs, {
                    'loss': result.get('final_metrics', {}).get('loss', 0),
                    'accuracy': result.get('final_metrics', {}).get('accuracy', 0),
                    'best_val_accuracy': result.get('best_val_accuracy', 0),
                    'duration': result.get('duration', 0),
                    'epochs_completed': result.get('epochs_completed', task.epochs)
                })
                task.log(f"Training completed successfully: {result.get('message', '')}")
            elif result["status"] == "cancelled":
                task.log("Training was cancelled")
                task.status = "stopped"
            else:
                task.fail(f"Training failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            error_msg = f"Real training execution failed: {str(e)}"
            task.fail(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
    
    def _execute_joint_training(self, task: TrainingTask):
        """Execute joint training with real training logic
        Args:
            task: Training task
        """
        if len(task.model_ids) < 2:
            raise ValueError(get_string("joint_training_min_models", task.lang))
            
        try:
            # Import the advanced training controller
            from training_manager.advanced_train_control import AdvancedTrainingController
            
            # Create training controller instance
            controller = AdvancedTrainingController()
            
            # Prepare training configuration
            config = {
                'epochs': task.epochs,
                'batch_size': task.batch_size,
                'learning_rate': task.learning_rate,
                'knowledge_assisted': task.knowledge_assisted
            }
            
            task.log(f"Starting real joint training for models: {', '.join(task.model_ids)}")
            task.log(f"Configuration: {config}")
            
            # Execute real joint training
            result = controller._execute_joint_training(
                "joint_training_" + task.id,
                task.model_ids,
                controller.TrainingMode.JOINT,
                config
            )
            
            if result["status"] == "success":
                # Update task with real training results
                task.update_progress(100, task.epochs, {
                    'joint_loss': result.get('final_metrics', {}).get('loss', 0),
                    'joint_accuracy': result.get('final_metrics', {}).get('accuracy', 0),
                    'best_val_accuracy': result.get('best_val_accuracy', 0),
                    'duration': result.get('duration', 0),
                    'epochs_completed': result.get('epochs_completed', task.epochs),
                    'models_trained': len(task.model_ids)
                })
                task.log(f"Joint training completed successfully: {result.get('message', '')}")
            elif result["status"] == "cancelled":
                task.log("Joint training was cancelled")
                task.status = "stopped"
            else:
                task.fail(f"Joint training failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            error_msg = f"Real joint training execution failed: {str(e)}"
            task.fail(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
        
    def _get_epoch_advice(self, task: TrainingTask, epoch: int):
        """Get epoch advice from knowledge base"""
        try:
            model_registry = ModelRegistry()
            knowledge_model = model_registry.get_model(KNOWLEDGE_MODEL_ID)
            
            if knowledge_model and knowledge_model["active"]:
                advice = knowledge_model["instance"].get_epoch_advice(
                    model_ids=task.model_ids,
                    current_epoch=epoch,
                    total_epochs=task.epochs,
                    metrics=task.metrics
                )
                
                if advice:
                    task.log_multilingual("knowledge_epoch_advice", epoch=epoch+1, advice=advice)
                    # Apply suggestions to training parameters
                    if "learning_rate_adjustment" in advice:
                        task.learning_rate *= advice["learning_rate_adjustment"]
                    if "batch_size_adjustment" in advice:
                        task.batch_size = max(1, int(task.batch_size * advice["batch_size_adjustment"]))
        except Exception as e:
            task.log_multilingual("knowledge_advice_failed", error=str(e))
            
    def _adjust_training_strategy(self, task: TrainingTask, epoch: int):
        """Adjust training strategy based on efficiency metrics"""
        if len(task.efficiency_metrics["epoch_times"]) < 2:
            return
            
        # Calculate average epoch time
        avg_epoch_time = np.mean(task.efficiency_metrics["epoch_times"])
        
        # If epoch time increases by more than 20%, adjust batch size
        if task.efficiency_metrics["epoch_times"][-1] > avg_epoch_time * 1.2:
            new_batch_size = max(8, int(task.batch_size * 0.9))
            if new_batch_size != task.batch_size:
                task.log_multilingual("batch_size_adjusted", 
                                    old_size=task.batch_size, 
                                    new_size=new_batch_size,
                                    reason=get_string("high_epoch_time", task.lang))
                task.batch_size = new_batch_size
                
        # If accuracy improvement is slow, adjust learning rate
        if "joint_accuracy" in task.metrics and len(task.metrics["joint_accuracy"]) > 3:
            last_three = task.metrics["joint_accuracy"][-3:]
            improvement = last_three[-1] - last_three[0]
            if improvement < 0.01:  # Improvement less than 1%
                new_lr = task.learning_rate * 1.1
                task.log_multilingual("learning_rate_adjusted", 
                                    old_rate=task.learning_rate, 
                                    new_rate=new_lr,
                                    reason=get_string("low_accuracy_improvement", task.lang))
                task.learning_rate = new_lr
    
    def _get_model_trainer(self, model_id: str):
        """Get model trainer
        Args:
            model_id: Model ID
        Returns:
            Trainer instance or None
        """
        try:
            # Determine model type based on model ID
            model_type = model_id.split('_')[0]
            
            # Dynamically import training module
            module_path = f"sub_models.{model_id}.trainer"
            try:
                trainer_module = importlib.import_module(module_path)
                if hasattr(trainer_module, 'ModelTrainer'):
                    return trainer_module.ModelTrainer()
            except ImportError:
                # If specific trainer not found, try generic trainer
                pass
            
            # Try generic training interface
            generic_module_path = f"sub_models.{model_id}.training"
            try:
                generic_module = importlib.import_module(generic_module_path)
                if hasattr(generic_module, 'train_model'):
                    return generic_module
            except ImportError:
                pass
                
            # Finally try model's own training method
            model_module_path = f"sub_models.{model_id}.app"
            try:
                model_module = importlib.import_module(model_module_path)
                if hasattr(model_module, 'train'):
                    return model_module
            except ImportError:
                pass
                
            logger.warning(f"Could not find trainer for model {model_id}")
            # Record detailed error information
            logger.debug(f"Attempted import paths: {module_path}, {generic_module_path}, {model_module_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting model trainer: {str(e)}")
            return None
    
    def pause_training(self, task_id: str = None):
        """Pause training
        Args:
            task_id: Task ID (optional, all if not provided)
        """
        if task_id:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].status = "paused"
                self.active_tasks[task_id].log("Training paused")
                return True
            return False
        else:
            # Pause all active tasks
            for task in self.active_tasks.values():
                task.status = "paused"
                task.log("Training paused")
            return True
    
    def resume_training(self, task_id: str = None):
        """Resume training
        Args:
            task_id: Task ID (optional, all if not provided)
        """
        if task_id:
            if task_id in self.active_tasks and self.active_tasks[task_id].status == "paused":
                self.active_tasks[task_id].status = "running"
                self.active_tasks[task_id].log("Training resumed")
                return True
            return False
        else:
            # Resume all paused tasks
            for task in self.active_tasks.values():
                if task.status == "paused":
                    task.status = "running"
                    task.log("Training resumed")
            return True
    
    def stop_training(self, task_id: str = None):
        """Stop training
        Args:
            task_id: Task ID (optional, all if not provided)
        """
        if task_id:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].status = "stopped"
                self.active_tasks[task_id].log("Training stopped")
                return True
            return False
        else:
            # Stop all active tasks
            for task in self.active_tasks.values():
                task.status = "stopped"
                task.log("Training stopped")
            return True
    
    def get_training_status(self, task_id: str = None) -> Dict:
        """Get training status
        Args:
            task_id: Task ID (optional, all if not provided)
        Returns:
            Training status dictionary
        """
        if task_id:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                return self._format_task_status(task)
            # Check completed tasks
            for task in self.completed_tasks:
                if task.id == task_id:
                    return self._format_task_status(task)
            # Check failed tasks
            for task in self.failed_tasks:
                if task.id == task_id:
                    return self._format_task_status(task)
            return {'error': get_string("task_not_found", "en").format(task_id=task_id)}
        
        # Return all task statuses
        all_status = {}
        
        # Active tasks
        for task_id, task in self.active_tasks.items():
            all_status[task_id] = self._format_task_status(task)
        
        # Completed tasks
        for task in self.completed_tasks:
            all_status[task.id] = self._format_task_status(task)
        
        # Failed tasks
        for task in self.failed_tasks:
            all_status[task.id] = self._format_task_status(task)
        
        return all_status
        
    def _format_task_status(self, task: TrainingTask) -> Dict:
        """Format task status"""
        status_dict = {
            'task_id': task.id,
            'status': task.status,
            'progress': task.progress,
            'current_epoch': task.current_epoch,
            'total_epochs': task.epochs,
            'start_time': task.start_time.isoformat() if task.start_time else None,
            'metrics': task.metrics,
            'efficiency': self._calculate_efficiency_metrics(task),
            'knowledge_assisted': task.knowledge_assisted
        }
        
        if task.status in ["completed", "failed", "stopped"]:
            status_dict['end_time'] = task.end_time.isoformat() if task.end_time else None
            
        if task.status == "failed" and task.logs:
            status_dict['error'] = task.logs[-1]
            
        return status_dict
        
    def _calculate_efficiency_metrics(self, task: TrainingTask) -> Dict:
        """Calculate training efficiency metrics"""
        if not task.efficiency_metrics["epoch_times"]:
            return {}
            
        metrics = {
            "avg_epoch_time": np.mean(task.efficiency_metrics["epoch_times"]),
            "min_epoch_time": np.min(task.efficiency_metrics["epoch_times"]),
            "max_epoch_time": np.max(task.efficiency_metrics["epoch_times"]),
            "total_training_time": (datetime.now() - task.efficiency_metrics["start_time"]).total_seconds() 
                                   if task.status == "running" else 
                                   (task.end_time - task.start_time).total_seconds(),
            "epoch_times": task.efficiency_metrics["epoch_times"]
        }
        
        # Calculate resource usage efficiency
        if task.metrics and "resource_usage" in task.metrics:
            metrics["avg_cpu_usage"] = np.mean([r["cpu"] for r in task.metrics["resource_usage"]])
            metrics["avg_memory_usage"] = np.mean([r["memory"] for r in task.metrics["resource_usage"]])
            
        return metrics
    
    def get_training_progress(self, task_id: str = None) -> Dict:
        """Get training progress
        Args:
            task_id: Task ID (optional, all if not provided)
        Returns:
            Training progress dictionary
        """
        status = self.get_training_status(task_id)
        if 'error' in status:
            return status
        
        if task_id:
            return {
                'task_id': task_id,
                'progress': status.get('progress', 0),
                'status': status.get('status', 'unknown'),
                'current_epoch': status.get('current_epoch', 0),
                'total_epochs': status.get('total_epochs', 0)
            }
        
        # Progress of all tasks
        progress = {}
        for task_id, task_status in status.items():
            progress[task_id] = {
                'progress': task_status.get('progress', 0),
                'status': task_status.get('status', 'unknown'),
                'current_epoch': task_status.get('current_epoch', 0),
                'total_epochs': task_status.get('total_epochs', 0)
            }
        return progress
    
    def get_task_logs(self, task_id: str) -> List[str]:
        """Get task logs
        Args:
            task_id: Task ID
        Returns:
            List of logs
        """
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].logs
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.id == task_id:
                return task.logs
        
        # Check failed tasks
        for task in self.failed_tasks:
            if task.id == task_id:
                return task.logs
        
        return []
    
    def shutdown(self):
        """Shutdown scheduler"""
        self.running = False
        
        # Stop all tasks
        self.stop_training()
        
        # Put special task to terminate worker thread
        self.task_queue.put((0, 0, None))
        
        # Wait for worker thread to end
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
    
    def register_progress_callback(self, callback: Callable):
        """Register a progress callback function
        Args:
            callback: Callback function to be called on progress updates
        """
        if callback not in self.progress_callbacks:
            self.progress_callbacks.append(callback)
    
    def register_status_callback(self, callback: Callable):
        """Register a status callback function
        Args:
            callback: Callback function to be called on status updates
        """
        if callback not in self.status_callbacks:
            self.status_callbacks.append(callback)
    
    def set_resource_limits(self, cpu_usage: float = None, memory_usage: float = None):
        """Set resource usage limits
        Args:
            cpu_usage: CPU usage limit percentage (0-100)
            memory_usage: Memory usage limit percentage (0-100)
        """
        with self.training_lock:
            if cpu_usage is not None:
                self.resource_limits['cpu_usage'] = max(0.0, min(100.0, cpu_usage))
            if memory_usage is not None:
                self.resource_limits['memory_usage'] = max(0.0, min(100.0, memory_usage))
            
        logger.info(f"Updated resource limits: CPU={self.resource_limits['cpu_usage']}%, Memory={self.resource_limits['memory_usage']}%")
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics
        Returns:
            Dictionary with scheduler statistics
        """
        with self.training_lock:
            return {
                'running': self.running,
                'queued_tasks': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'resource_limits': self.resource_limits,
                'max_concurrent_tasks': self.max_concurrent_tasks
            }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a training task
        Args:
            task_id: Task ID
        Returns:
            True if task was found and cancelled, False otherwise
        """
        with self.training_lock:
            # Check if it's an active task
            if task_id in self.active_tasks:
                self.active_tasks[task_id].status = "stopped"
                self.active_tasks[task_id].log("Training task cancelled")
                return True
            
            # Check if it's in the queue - this requires rebuilding the queue
            temp_queue = queue.PriorityQueue()
            task_found = False
            
            while not self.task_queue.empty():
                current_task = self.task_queue.get()[2]  # Extract the task object
                if current_task.id == task_id:
                    current_task.status = "cancelled"
                    current_task.log("Training task cancelled before starting")
                    self._notify_status_update(current_task)
                    task_found = True
                    logger.info(f"Task cancelled: {task_id}")
                else:
                    self.entry_count += 1
                    temp_queue.put((current_task.priority, self.entry_count, current_task))
            
            # Replace the original queue
            self.task_queue = temp_queue
            
            return task_found

# Global scheduler instance
training_scheduler = None

def get_training_scheduler(data_bus: Optional[DataBus] = None) -> TrainScheduler:
    """Get global training scheduler instance
    Args:
        data_bus: Data bus instance for communication
    Returns:
        TrainScheduler instance
    """
    global training_scheduler
    if training_scheduler is None:
        training_scheduler = TrainScheduler(data_bus)
    return training_scheduler
    

# Test code
if __name__ == '__main__':
    # Test scheduler with enhanced features
    from manager_model.data_bus import DataBus
    
    # Create data bus instance
    data_bus = DataBus()
    
    # Get scheduler instance
    scheduler = get_training_scheduler(data_bus)
    
    # Register progress callback
    def progress_callback(progress_data):
        print(f"Progress update: Task ID={progress_data['task_id']}, Progress={progress_data['progress']}%, "
              f"Epoch={progress_data['current_epoch']}/{progress_data['total_epochs']}")
    
    scheduler.register_progress_callback(progress_callback)
    
    # Register status callback
    def status_callback(status_data):
        print(f"Status update: Task ID={status_data['task_id']}, Status={status_data['status']}, "
              f"Models={status_data['model_ids']}")
    
    scheduler.register_status_callback(status_callback)
    
    # Set resource limits
    scheduler.set_resource_limits(cpu_usage=75.0, memory_usage=80.0)
    
    print("Adding test tasks...")
    
    # High priority task
    task_id1 = scheduler.start_training(
        model_ids=["B_language", "I_knowledge"],
        training_type="joint",
        epochs=5,
        batch_size=16,
        learning_rate=0.001,
        knowledge_assisted=True
    )
    
    # Scheduled task (will start after 10 seconds)
    future_time = datetime.now() + timedelta(seconds=10)
    task_id2 = scheduler.schedule_training(
        model_ids=["D_image"],
        training_type="single",
        scheduled_time=future_time,
        epochs=3,
        batch_size=8,
        learning_rate=0.0005,
        priority=TrainingPriority.LOW
    )
    
    # Background priority task
    task_id3 = scheduler.start_training(
        model_ids=["K_programming"],
        training_type="single",
        epochs=4,
        batch_size=16,
        learning_rate=0.001,
        priority=TrainingPriority.BACKGROUND
    )
    
    print(f"Added tasks: {task_id1}, {task_id2}, {task_id3}")
    
    # Monitor task progress and scheduler stats
    try:
        for _ in range(60):  # Run for 60 seconds
            time.sleep(1)
            # Print scheduler stats
            stats = scheduler.get_scheduler_stats()
            print(f"\rScheduler stats: Active tasks={stats['active_tasks']}, "
                  f"Queued tasks={stats['queued_tasks']}, "
                  f"Completed tasks={stats['completed_tasks']}", end="")
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        scheduler.stop_training()
    
    finally:
        # Shutdown scheduler
        scheduler.shutdown()
        print("\nTraining scheduler shut down")
        
        # Print final statistics
        stats = scheduler.get_scheduler_stats()
        print(f"Final statistics: Active={stats['active_tasks']}, "
              f"Completed={stats['completed_tasks']}, Failed={stats['failed_tasks']}")
