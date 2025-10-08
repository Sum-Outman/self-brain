# Training Dashboard - Web interface for managing training tasks

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import psutil
import importlib

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from training_manager.train_scheduler import get_training_scheduler, TrainingPriority, TrainingTask
from manager_model.data_bus import DataBus
from manager_model.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrainingDashboard")

# Initialize Flask application
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'training-dashboard-secret-key'

# Enable CORS
CORS(app)

# Initialize data bus and training scheduler
data_bus = DataBus()
training_scheduler = get_training_scheduler(data_bus)

# Initialize model registry with real implementation
model_registry = ModelRegistry()

# Global variables for storing training history
TRAINING_HISTORY = []

@app.route('/')
def dashboard():
    """Main dashboard page"""
    # Get scheduler statistics
    scheduler_stats = training_scheduler.get_scheduler_stats()
    
    # Get active tasks
    active_tasks = []
    for task_id, task in training_scheduler.active_tasks.items():
        active_tasks.append({
            'id': task.id,
            'model_ids': task.model_ids,
            'status': task.status,
            'progress': task.progress,
            'current_epoch': task.current_epoch,
            'total_epochs': task.epochs,
            'start_time': task.start_time.isoformat() if task.start_time else None
        })
    
    # Get queued tasks (approximation since we can't directly access the queue)
    queued_count = scheduler_stats['queued_tasks']
    
    # Get resource usage
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    
    # Get available models
    available_models = model_registry.get_all_models()
    
    return render_template('training_dashboard.html', 
                          scheduler_stats=scheduler_stats,
                          active_tasks=active_tasks,
                          queued_count=queued_count,
                          cpu_usage=cpu_usage,
                          memory_usage=memory_usage,
                          available_models=available_models,
                          training_history=TRAINING_HISTORY[-50:])

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks (active and queued)"""
    # Get active tasks
    active_tasks = []
    for task_id, task in training_scheduler.active_tasks.items():
        active_tasks.append({
            'id': task.id,
            'model_ids': task.model_ids,
            'status': task.status,
            'progress': task.progress,
            'current_epoch': task.current_epoch,
            'total_epochs': task.epochs,
            'training_type': task.training_type,
            'priority': task.priority,
            'start_time': task.start_time.isoformat() if task.start_time else None,
            'created_at': task.created_at.isoformat() if hasattr(task, 'created_at') else None
        })
    
    # Get scheduler statistics
    scheduler_stats = training_scheduler.get_scheduler_stats()
    
    # Get resource usage
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    
    return jsonify({
        'active_tasks': active_tasks,
        'queued_count': scheduler_stats['queued_tasks'],
        'completed_count': scheduler_stats['completed_tasks'],
        'failed_count': scheduler_stats['failed_tasks'],
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage
    })

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task(task_id):
    """Get details of a specific task"""
    # Check active tasks first
    if task_id in training_scheduler.active_tasks:
        task = training_scheduler.active_tasks[task_id]
        return jsonify({
            'id': task.id,
            'model_ids': task.model_ids,
            'status': task.status,
            'progress': task.progress,
            'current_epoch': task.current_epoch,
            'total_epochs': task.epochs,
            'training_type': task.training_type,
            'priority': task.priority,
            'config': task.config,
            'metrics': task.metrics,
            'start_time': task.start_time.isoformat() if task.start_time else None,
            'created_at': task.created_at.isoformat() if hasattr(task, 'created_at') else None,
            'logs': task.logs
        })
    
    # If task not found in active tasks, search in completed tasks
    for task in training_scheduler.completed_tasks + training_scheduler.failed_tasks:
        if task.id == task_id:
            return jsonify({
                'id': task.id,
                'model_ids': task.model_ids,
                'status': task.status,
                'progress': task.progress,
                'current_epoch': task.current_epoch,
                'total_epochs': task.epochs,
                'training_type': task.training_type,
                'priority': task.priority,
                'config': task.config,
                'metrics': task.metrics,
                'start_time': task.start_time.isoformat() if task.start_time else None,
                'created_at': task.created_at.isoformat(),
                'logs': task.logs
            })
    
    # Task not found
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_alt(task_id):
    """Alternative endpoint for getting task details"""
    return get_task(task_id)

@app.route('/api/tasks/create', methods=['POST'])
def create_task():
    """Create a new training task"""
    try:
        data = request.json
        
        # Validate required fields
        if not all(key in data for key in ['models', 'training-type']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Extract and validate parameters
        model_ids = data.get('models', [])
        if not isinstance(model_ids, list) or len(model_ids) == 0:
            return jsonify({'error': 'Invalid model IDs'}), 400
        
        training_type = data.get('training-type')
        if training_type not in ['single', 'joint']:
            return jsonify({'error': 'Invalid training type'}), 400
        
        # Extract optional parameters
        epochs = data.get('epochs', 10)
        batch_size = data.get('batch-size', 32)
        learning_rate = data.get('learning-rate', 0.001)
        knowledge_assisted = data.get('knowledge-assisted', False)
        priority = data.get('priority', TrainingPriority.MEDIUM)
        
        # Create task configuration
        config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'knowledge_assisted': knowledge_assisted
        }
        
        # Create training task
        task_id = training_scheduler.start_training(
            model_ids=model_ids,
            training_type=training_type,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            knowledge_assisted=knowledge_assisted
        )
        
        # Log the task creation
        logger.info(f"Created new training task: {task_id}, Models: {model_ids}")
        
        # Add to training history
        TRAINING_HISTORY.insert(0, {
            'task_id': task_id,
            'status': 'Started',
            'timestamp': datetime.now().isoformat(),
            'models': model_ids
        })
        
        return jsonify({'task_id': task_id, 'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/task/<task_id>/stop', methods=['POST'])
def stop_task(task_id):
    """Stop a training task"""
    try:
        success = training_scheduler.cancel_task(task_id)
        if success:
            logger.info(f"Stopped training task: {task_id}")
            return jsonify({'status': 'success', 'message': 'Task stopped'})
        else:
                # Task not found or couldn't be stopped
                logger.warning(f"Failed to stop training task: {task_id}")
                return jsonify({'status': 'error', 'message': 'Failed to stop task'}), 404
    except Exception as e:
        logger.error(f"Error stopping task: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/<task_id>/stop', methods=['POST'])
def stop_task_alt(task_id):
    """Alternative endpoint for stopping a task"""
    return stop_task(task_id)

@app.route('/api/settings/resource-limits', methods=['POST'])
def set_resource_limits_alt():
    """Alternative endpoint for setting resource limits"""
    try:
        data = request.json
        cpu_limit = data.get('cpu_limit')
        memory_limit = data.get('memory_limit')
        
        # In a real implementation, we would set these limits
        logger.info(f"Setting resource limits: CPU={cpu_limit}%, Memory={memory_limit}%")
        
        return jsonify({'status': 'success', 'message': 'Resource limits updated'})
    except Exception as e:
        logger.error(f"Error setting resource limits: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/resources', methods=['GET'])
def get_resources():
    """Get current system resources"""
    try:
        # Get CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        memory_available = memory_info.available / (1024 * 1024 * 1024)  # Convert to GB
        
        # Get disk usage
        disk_info = psutil.disk_usage('/')
        disk_usage = disk_info.percent
        
        # Get scheduler resource limits
        scheduler_stats = training_scheduler.get_scheduler_stats()
        resource_limits = scheduler_stats.get('resource_limits', {})
        
        return jsonify({
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'memory_available_gb': round(memory_available, 2),
            'disk_usage': disk_usage,
            'resource_limits': resource_limits
        })
    except Exception as e:
        logger.error(f"Error getting resources: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/resources/limits', methods=['POST'])
def set_resource_limits():
    """Set resource limits"""
    try:
        data = request.json
        cpu_limit = data.get('cpu_limit')
        memory_limit = data.get('memory_limit')
        
        training_scheduler.set_resource_limits(
            cpu_usage=cpu_limit if cpu_limit is not None else None,
            memory_usage=memory_limit if memory_limit is not None else None
        )
        
        return jsonify({'status': 'success', 'message': 'Resource limits updated'})
    except Exception as e:
        logger.error(f"Error setting resource limits: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get available models"""
    try:
        available_models = model_registry.get_all_models()
        return jsonify({'models': available_models})
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics"""
    try:
        scheduler_stats = training_scheduler.get_scheduler_stats()
        
        # Get system resources
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # Get priority distribution
        # Note: We can't directly count priorities in the queue, so this is an approximation
        priority_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'background': 0
        }
        
        # Count priorities in active tasks
        for task in training_scheduler.active_tasks.values():
            if task.priority == TrainingPriority.CRITICAL:
                priority_counts['critical'] += 1
            elif task.priority == TrainingPriority.HIGH:
                priority_counts['high'] += 1
            elif task.priority == TrainingPriority.MEDIUM:
                priority_counts['medium'] += 1
            elif task.priority == TrainingPriority.LOW:
                priority_counts['low'] += 1
            elif task.priority == TrainingPriority.BACKGROUND:
                priority_counts['background'] += 1
        
        return jsonify({
            'scheduler_stats': scheduler_stats,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'priority_distribution': priority_counts
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Register progress and status callbacks
def progress_callback(progress_data):
    """Callback for progress updates"""
    # This could be used to broadcast progress updates via WebSockets
    logger.debug(f"Progress update: {progress_data}")

def status_callback(status_data):
    """Callback for status updates"""
    # Add to training history
    global TRAINING_HISTORY
    TRAINING_HISTORY.append({
        'task_id': status_data.get('task_id'),
        'status': status_data.get('status'),
        'timestamp': datetime.now().isoformat(),
        'models': status_data.get('model_ids')
    })
    # Keep history to a reasonable size
    if len(TRAINING_HISTORY) > 1000:
        TRAINING_HISTORY = TRAINING_HISTORY[-1000:]
    
    logger.info(f"Status update: Task {status_data.get('task_id')} is now {status_data.get('status')}")

training_scheduler.register_progress_callback(progress_callback)
training_scheduler.register_status_callback(status_callback)

if __name__ == '__main__':
    # Create template and static folders if they don't exist
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        # Create css and js subdirectories
        os.makedirs(os.path.join(static_dir, 'css'))
        os.makedirs(os.path.join(static_dir, 'js'))
    
    # Start the Flask application
    logger.info("Starting Training Dashboard on http://localhost:5012")
    app.run(host='0.0.0.0', port=5012, debug=True)