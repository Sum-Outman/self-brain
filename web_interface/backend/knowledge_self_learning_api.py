# Copyright 2025 AGI System Team
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

# Knowledge Base Self-Learning API Implementation

from flask import Blueprint, request, jsonify
import logging
import threading
import time
from datetime import datetime
import json
from pathlib import Path
import os
import sys

# Import DataBus for real communication with models
from data_bus import get_data_bus, DataBus

# Initialize DataBus instance for communication
data_bus = get_data_bus()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
knowledge_self_learning_bp = Blueprint('knowledge_self_learning', __name__)

# Global state for self-learning
self_learning_state = {
    'active': False,
    'model': 'all',
    'start_time': None,
    'end_time': None,
    'status': 'idle',  # idle, starting, running, stopping, completed, error
    'progress': 0,
    'logs': [],
    'learning_thread': None
}

# Model-specific learning configurations
model_learning_configs = {
    'A': {'domains': ['management', 'psychology'], 'learning_rate': 0.01},
    'B': {'domains': ['linguistics', 'literature'], 'learning_rate': 0.02},
    'C': {'domains': ['acoustics', 'music'], 'learning_rate': 0.015},
    'D': {'domains': ['computer_vision', 'art'], 'learning_rate': 0.02},
    'E': {'domains': ['video_analysis', 'cinematography'], 'learning_rate': 0.015},
    'F': {'domains': ['physics', 'geometry'], 'learning_rate': 0.01},
    'G': {'domains': ['sensor_technology', 'signal_processing'], 'learning_rate': 0.015},
    'H': {'domains': ['computer_science', 'operating_systems'], 'learning_rate': 0.02},
    'I': {'domains': ['kinematics', 'control_systems'], 'learning_rate': 0.01},
    'J': {'domains': ['all'], 'learning_rate': 0.01},
    'K': {'domains': ['programming', 'software_engineering'], 'learning_rate': 0.02}
}

@knowledge_self_learning_bp.route('/api/knowledge/self_learning/start', methods=['POST'])
def start_self_learning():
    """
    Start self-learning process for selected model(s)
    """
    global self_learning_state
    
    try:
        data = request.json
        model = data.get('model', 'all')
        
        # Validate model
        valid_models = list(model_learning_configs.keys()) + ['all']
        if model not in valid_models:
            return jsonify({'status': 'error', 'message': f'Invalid model. Valid options: {valid_models}'}), 400
        
        # Check if self-learning is already active
        if self_learning_state['active']:
            return jsonify({'status': 'error', 'message': 'Self-learning is already running'}), 400
        
        # Reset state
        self_learning_state = {
            'active': True,
            'model': model,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'starting',
            'progress': 0,
            'logs': [f'Starting self-learning for model(s): {model} at {self_learning_state["start_time"]}'],
            'learning_thread': None
        }
        
        logger.info(f'Starting self-learning for model: {model}')
        
        # Start learning in a separate thread
        learning_thread = threading.Thread(target=run_self_learning, args=(model,))
        learning_thread.daemon = True
        learning_thread.start()
        
        self_learning_state['learning_thread'] = learning_thread
        
        return jsonify({
            'status': 'success', 
            'message': 'Self-learning process started',
            'data': {
                'model': model,
                'start_time': self_learning_state['start_time']
            }
        })
    
    except Exception as e:
        logger.error(f'Error starting self-learning: {str(e)}')
        self_learning_state['status'] = 'error'
        self_learning_state['logs'].append(f'Error: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@knowledge_self_learning_bp.route('/api/knowledge/self_learning/stop', methods=['POST'])
def stop_self_learning():
    """
    Stop ongoing self-learning process
    """
    global self_learning_state
    
    try:
        # Check if self-learning is active
        if not self_learning_state['active']:
            return jsonify({'status': 'error', 'message': 'No self-learning process is active'}), 400
        
        logger.info('Stopping self-learning process')
        self_learning_state['status'] = 'stopping'
        self_learning_state['logs'].append(f'Stopping self-learning at {datetime.now().isoformat()}')
        
        # Send stop command to all models via data bus
        model = self_learning_state['model']
        models_to_stop = [model] if model != 'all' else list(model_learning_configs.keys())
        
        for current_model in models_to_stop:
            data_bus.publish('self_learning_control', {
                'action': 'stop',
                'model_id': current_model
            })
            self_learning_state['logs'].append(f'Sent stop command to model {current_model}')
        
        # Set flag to signal threads to stop
        self_learning_state['active'] = False
        
        # Wait for thread to finish
        if self_learning_state['learning_thread'] and self_learning_state['learning_thread'].is_alive():
            # Wait for a short time to allow the thread to stop gracefully
            self_learning_state['learning_thread'].join(timeout=5.0)
        
        self_learning_state['end_time'] = datetime.now().isoformat()
        self_learning_state['status'] = 'stopped'
        
        return jsonify({
            'status': 'success', 
            'message': 'Self-learning process stopped',
            'data': {
                'start_time': self_learning_state['start_time'],
                'end_time': self_learning_state['end_time'],
                'progress': self_learning_state['progress']
            }
        })
    
    except Exception as e:
        logger.error(f'Error stopping self-learning: {str(e)}')
        self_learning_state['status'] = 'error'
        self_learning_state['logs'].append(f'Error during stop: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@knowledge_self_learning_bp.route('/api/knowledge/self_learning/status', methods=['GET'])
def get_self_learning_status():
    """
    Get current status of self-learning process
    """
    global self_learning_state
    
    try:
        # Prepare status response
        status_response = {
            'active': self_learning_state['active'],
            'model': self_learning_state['model'],
            'start_time': self_learning_state['start_time'],
            'end_time': self_learning_state['end_time'],
            'status': self_learning_state['status'],
            'progress': self_learning_state['progress'],
            'log_count': len(self_learning_state['logs'])
        }
        
        return jsonify({'status': 'success', 'data': status_response})
    
    except Exception as e:
        logger.error(f'Error getting self-learning status: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@knowledge_self_learning_bp.route('/api/knowledge/self_learning/progress', methods=['GET'])
def get_self_learning_progress():
    """
    Get detailed progress and logs of self-learning process
    """
    global self_learning_state
    
    try:
        # Get parameters for log pagination
        start = int(request.args.get('start', 0))
        limit = int(request.args.get('limit', 50))
        
        # Slice logs based on pagination
        paginated_logs = self_learning_state['logs'][start:start + limit]
        
        # Calculate duration if available
        duration = None
        if self_learning_state['start_time']:
            start_time = datetime.fromisoformat(self_learning_state['start_time'])
            if self_learning_state['end_time']:
                end_time = datetime.fromisoformat(self_learning_state['end_time'])
                duration = str(end_time - start_time)
            else:
                duration = str(datetime.now() - start_time)
        
        progress_response = {
            'active': self_learning_state['active'],
            'model': self_learning_state['model'],
            'status': self_learning_state['status'],
            'progress': self_learning_state['progress'],
            'duration': duration,
            'logs': paginated_logs,
            'total_logs': len(self_learning_state['logs'])
        }
        
        return jsonify({'status': 'success', 'data': progress_response})
    
    except Exception as e:
        logger.error(f'Error getting self-learning progress: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

def run_self_learning(model):
    """
    Run the self-learning process in a separate thread by communicating with the actual models
    """
    global self_learning_state
    
    try:
        logger.info(f'Starting self-learning thread for model: {model}')
        self_learning_state['status'] = 'running'
        self_learning_state['logs'].append('Self-learning process is running')
        
        # Determine which models to process
        models_to_process = [model] if model != 'all' else list(model_learning_configs.keys())
        total_models = len(models_to_process)
        
        # Start progress monitoring thread
        progress_monitor = threading.Thread(target=monitor_progress, args=(models_to_process,))
        progress_monitor.daemon = True
        progress_monitor.start()
        
        # Process each model
        for idx, current_model in enumerate(models_to_process, 1):
            if not self_learning_state['active']:  # Check if we should stop
                self_learning_state['logs'].append(f'Self-learning interrupted during processing of model {current_model}')
                break
            
            # Get model configuration
            config = model_learning_configs.get(current_model, {'domains': ['all'], 'learning_rate': 0.01})
            
            self_learning_state['logs'].append(f'Starting self-learning for model {current_model} with domains: {config["domains"]}')
            
            # Send start self-learning command to the model via data bus
            data_bus.publish('self_learning_control', {
                'action': 'start',
                'model_id': current_model,
                'config': config
            })
            
            # Update progress
            self_learning_state['progress'] = int((idx / total_models) * 100)
        
        # Wait for a short time to ensure commands are processed
        time.sleep(2)
        
        # Continue running until stopped
        while self_learning_state['active']:
            time.sleep(1)
        
        # Complete the learning process
        if self_learning_state['status'] != 'stopped' and self_learning_state['status'] != 'error':
            self_learning_state['status'] = 'completed'
            self_learning_state['end_time'] = datetime.now().isoformat()
            self_learning_state['logs'].append(f'Self-learning completed at {self_learning_state["end_time"]}')
            logger.info('Self-learning process completed successfully')
        
    except Exception as e:
        logger.error(f'Error during self-learning process: {str(e)}')
        self_learning_state['status'] = 'error'
        self_learning_state['logs'].append(f'Critical error: {str(e)}')
    finally:
        # Ensure the state is properly updated
        if self_learning_state['status'] not in ['stopped', 'error']:
            self_learning_state['active'] = False


def monitor_progress(models_to_process):
    """
    Monitor the progress of self-learning for all models
    """
    # Subscribe to progress updates
    progress_updates = {model: 0 for model in models_to_process}
    
    # Define callback for progress updates
    def progress_callback(data):
        model_id = data.get('model_id', '')
        progress = data.get('progress', 0)
        if model_id in progress_updates:
            progress_updates[model_id] = progress
            avg_progress = sum(progress_updates.values()) / len(progress_updates)
            self_learning_state['progress'] = int(avg_progress)
            self_learning_state['logs'].append(f'Model {model_id} self-learning progress: {progress}%')
    
    # Subscribe to self-learning progress updates
    data_bus.subscribe('self_learning_progress', progress_callback)
    
    # Continue monitoring until self-learning is stopped
    while self_learning_state['active']:
        time.sleep(5)
    
    # Unsubscribe when done
    data_bus.unsubscribe('self_learning_progress', progress_callback)

@knowledge_self_learning_bp.route('/api/knowledge/self_learning/models', methods=['GET'])
def get_available_models():
    """
    Get list of available models for self-learning
    """
    try:
        # Return model information
        models_info = {}
        for model_key, config in model_learning_configs.items():
            models_info[model_key] = {
                'domains': config['domains'],
                'learning_rate': config['learning_rate']
            }
        
        return jsonify({
            'status': 'success',
            'data': {
                'models': models_info
            }
        })
    
    except Exception as e:
        logger.error(f'Error getting available models: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500