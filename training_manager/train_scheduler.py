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
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import traceback
import numpy as np
from manager_model.language_resources import get_string
from manager_model.model_registry import ModelRegistry
from manager_model.data_broker import DataBroker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrainScheduler")

# Knowledge base model interface
KNOWLEDGE_MODEL_ID = "I_knowledge"

class TrainingTask:
    def __init__(self, model_ids: List[str], training_type: str, config: Dict, lang: str = "zh", priority=5):
        """Initialize training task
        Args:
            model_ids: List of model IDs
            training_type: Training type ('single' or 'joint')
            config: Training configuration
            lang: Language code (zh/en/de/ja/ru)
            priority: Task priority (1-10, 1 is highest)
        """
        self.id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}_{training_type}"
        self.model_ids = model_ids
        self.training_type = training_type
        self.config = config
        self.lang = lang
        self.priority = priority
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
    def __init__(self):
        """Initialize training scheduler"""
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.entry_count = 0
        self.training_lock = threading.Lock()
        
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
        self.entry_count += 1
        self.task_queue.put((task.priority, self.entry_count, task))
        return task.id
        
    def _worker(self):
        """Worker thread to process tasks"""
        while True:
            try:
                priority, count, task = self.task_queue.get()
                if task is None:
                    break
                    
                self.active_tasks[task.id] = task
                task.start()
                
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
                time.sleep(1)  # Prevent infinite loop
    
    def _execute_single_training(self, task: TrainingTask):
        """Execute single model training
        Args:
            task: Training task
        """
        if len(task.model_ids) != 1:
            raise ValueError("Single model training can only have one model ID")
            
        model_id = task.model_ids[0]
        
        # Get model trainer
        trainer = self._get_model_trainer(model_id)
        if not trainer:
            raise ValueError(f"Could not find trainer for model {model_id}")
        
        # Execute training
        for epoch in range(task.epochs):
            if task.status == "stopped":
                break
                
            # Simulate training steps
            for batch in range(0, 100, task.batch_size):
                if task.status == "stopped":
                    break
                    
                # Calculate progress
                epoch_progress = (epoch / task.epochs) * 100
                batch_progress = (batch / 100) * (100 / task.epochs)
                progress = min(epoch_progress + batch_progress, 100)
                
                # Update progress
                metrics = {
                    'loss': 0.1 * (1 - epoch/task.epochs),
                    'accuracy': 0.8 * (epoch/task.epochs),
                    'batch': batch
                }
                task.update_progress(int(progress), epoch + 1, metrics)
                
                # Simulate training time
                time.sleep(0.01 * task.batch_size / 32)
                
                # Check if stopped
                if task.status == "stopped":
                    task.log("Training task manually stopped")
                    break
    
    def _execute_joint_training(self, task: TrainingTask):
        """Execute joint training
        Args:
            task: Training task
        """
        if len(task.model_ids) < 2:
            raise ValueError(get_string("joint_training_min_models", task.lang))
            
        # Get trainers for all models
        trainers = {}
        for model_id in task.model_ids:
            trainer = self._get_model_trainer(model_id)
            if not trainer:
                raise ValueError(get_string("model_trainer_not_found", task.lang).format(model_id=model_id))
            trainers[model_id] = trainer
        
        # Create data sharing bus
        data_broker = DataBroker()
        task.log_multilingual("data_broker_created")
        
        # Execute joint training
        epoch_start_time = datetime.now()
        for epoch in range(task.epochs):
            if task.status == "stopped":
                break
                
            # Record epoch start time
            epoch_start = time.time()
            
            # Knowledge base assistance: get epoch advice
            if task.knowledge_assisted:
                self._get_epoch_advice(task, epoch)
                
            # Execute joint training steps
            for batch in range(0, 100, task.batch_size):
                if task.status == "stopped":
                    break
                    
                # Calculate progress
                epoch_progress = (epoch / task.epochs) * 100
                batch_progress = (batch / 100) * (100 / task.epochs)
                progress = min(epoch_progress + batch_progress, 100)
                
                # Inter-model data sharing
                shared_data = {}
                for model_id, trainer in trainers.items():
                    # Get model output as shared data
                    model_output = trainer.get_intermediate_output()
                    shared_data[model_id] = model_output
                    data_broker.publish(f"{model_id}_output", model_output)
                
                # Update progress
                metrics = {
                    'joint_loss': 0.15 * (1 - epoch/task.epochs),
                    'joint_accuracy': 0.75 * (epoch/task.epochs),
                    'batch': batch,
                    'models': task.model_ids,
                    'shared_data_size': len(json.dumps(shared_data))
                }
                task.update_progress(int(progress), epoch + 1, metrics)
                
                # Simulate training time
                time.sleep(0.015 * task.batch_size / 32)
                
                # Check if stopped
                if task.status == "stopped":
                    task.log_multilingual("training_manually_stopped")
                    break
            
            # Record epoch time
            epoch_time = time.time() - epoch_start
            task.efficiency_metrics["epoch_times"].append(epoch_time)
            task.efficiency_metrics["last_epoch_time"] = datetime.now()
            
            # Dynamic adjustment strategy
            self._adjust_training_strategy(task, epoch)
            
        # Training completed, clean up data bus
        data_broker.cleanup()
        task.log_multilingual("data_broker_cleaned")
        
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
        # Stop all tasks
        self.stop_training()
        
        # Put special task to terminate worker thread
        self.task_queue.put((0, 0, None))
        
        # Wait for worker thread to end
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

    # Test code
if __name__ == '__main__':
    # Test scheduler
    scheduler = TrainScheduler()
    
    # Test single model training
    single_task_id = scheduler.start_training(
        model_ids=["B_language"],
        training_type="single",
        epochs=5,
        batch_size=16,
        learning_rate=0.001
    )
    print(f"Single model training task scheduled: {single_task_id}")
    
    # Test joint training
    joint_task_id = scheduler.start_training(
        model_ids=["B_language", "C_audio"],
        training_type="joint",
        epochs=3,
        batch_size=8,
        learning_rate=0.0005
    )
    print(f"Joint training task scheduled: {joint_task_id}")
    
    # Monitor task progress
    try:
        while True:
            time.sleep(2)
            status = scheduler.get_training_status()
            print("\nCurrent training status:")
            for task_id, task_status in status.items():
                print(f"  {task_id}: {task_status['status']}, Progress: {task_status.get('progress', 0)}%")
            
            # Check if all tasks are completed or failed
            all_done = all(task_status['status'] in ['completed', 'failed', 'stopped'] 
                          for task_status in status.values())
            if all_done:
                break
                
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        scheduler.stop_training()
    
    scheduler.shutdown()
    print("Test completed")
