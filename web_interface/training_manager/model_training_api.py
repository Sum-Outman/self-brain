# -*- coding: utf-8 -*-
"""
Model Training API Module
This module provides REST API endpoints for controlling model training operations.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Any, List, Optional
import threading
from flask import Blueprint, request, jsonify, current_app, make_response

# Import required components from the training manager module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_manager.training_manager import get_training_manager
from training_manager.model_architectures import list_available_models, get_model_info
from training_manager.data_version_control import DataVersionControl
from training_manager.training_config_manager import TrainingConfigManager
from training_manager.training_logger import TrainingLogger
from training_manager.model_checkpoint_manager import ModelCheckpointManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ModelTrainingAPI')

# Get instances of required components
training_manager = get_training_manager()
data_version_control = DataVersionControl()
config_manager = TrainingConfigManager()
training_logger = TrainingLogger()
checkpoint_manager = ModelCheckpointManager()

# Create Blueprint for model training API
model_training_api = Blueprint('model_training_api', __name__)

@model_training_api.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training for a specific model or multiple models"""
    try:
        # Get request data
        data = request.json
        
        # Validate request data
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Check if it's joint training
        if 'model_ids' in data and isinstance(data['model_ids'], list):
            # Joint training - we'll handle this by starting each model separately with the same config
            model_ids = data['model_ids']
            config = data.get('config', {})
            
            results = []
            for model_id in model_ids:
                # Start individual training for each model
                result = training_manager.start_training(model_id, config)
                if result.get('success'):
                    results.append({
                        'model_id': model_id,
                        'status': 'success',
                        'message': result.get('message', f'Training started for model {model_id}')
                    })
                else:
                    results.append({
                        'model_id': model_id,
                        'status': 'error',
                        'error': result.get('error', f'Failed to start training for model {model_id}')
                    })
            
            # Check if all models were successfully started
            all_success = all(r['status'] == 'success' for r in results)
            
            return jsonify({
                'status': 'success' if all_success else 'partial',
                'results': results,
                'message': f'Training started for models {', '.join(model_ids)}',
                'training_type': 'joint'
            }), 200 if all_success else 207
        elif 'model_id' in data:
            # Individual training
            model_id = data['model_id']
            config = data.get('config', {})
            
            # Start individual training
            result = training_manager.start_training(model_id, config)
            
            if result.get('success'):
                return jsonify({
                    'status': 'success',
                    'training_id': model_id,
                    'message': result.get('message', f'Training started for model {model_id}'),
                    'training_type': 'individual'
                }), 200
            else:
                return jsonify({
                    'status': 'error',
                    'message': result.get('error', f'Failed to start training for model {model_id}')
                }), 400
        else:
            return jsonify({
                'status': 'error',
                'message': 'Either model_id or model_ids must be provided'
            }), 400
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to start training: {str(e)}'
        }), 500

@model_training_api.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop training for a specific model"""
    try:
        # Get request data
        data = request.json
        
        # Validate request data
        if not data or 'training_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Training ID is required'
            }), 400
        
        training_id = data['training_id']
        
        # Stop training
        result = training_manager.stop_training(training_id)
        
        if result.get('success'):
            return jsonify({
                'status': 'success',
                'message': result.get('message', f'Training {training_id} stopped')
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('error', f'Failed to stop training {training_id}')
            }), 400
    except Exception as e:
        logger.error(f"Error stopping training: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to stop training: {str(e)}'
        }), 500

@model_training_api.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get the status of training sessions"""
    try:
        # Log the API request for debugging
        logger.info(f"API request received: /api/training/status from {request.remote_addr}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        # Get parameters
        training_id = request.args.get('training_id')
        
        if training_id:
            # Get status for specific training
            status = training_manager.get_training_status(training_id)
            
            if status:
                return jsonify({
                    'status': 'success',
                    'data': status
                }), 200
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Training {training_id} not found'
                }), 404
        else:
            # Get all training statuses
            statuses = training_manager.get_all_training_statuses()
            
            return jsonify({
                'status': 'success',
                'data': statuses
            }), 200
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get training status: {str(e)}'
        }), 500

@model_training_api.route('/api/training/metrics', methods=['GET'])
def get_training_metrics():
    """Get training metrics for a specific model"""
    try:
        # Get request parameters
        training_id = request.args.get('training_id')
        model_id = request.args.get('model_id')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        metrics = request.args.get('metrics')
        
        # Convert time parameters to integers if provided
        if start_time:
            try:
                start_time = int(start_time)
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid start_time format'
                }), 400
        
        if end_time:
            try:
                end_time = int(end_time)
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid end_time format'
                }), 400
        
        # Parse metrics list if provided
        metrics_list = None
        if metrics:
            metrics_list = metrics.split(',')
        
        if training_id:
            # Get metrics for specific training session
            metrics_data = training_manager.get_training_metrics(training_id)
        elif model_id:
            # Get metrics for specific model
            metrics_data = training_manager.get_model_metrics(model_id, start_time, end_time, metrics_list)
        else:
            # Get all metrics
            metrics_data = training_manager.get_all_metrics()
        
        return jsonify({
            'status': 'success',
            'data': metrics_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting training metrics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get training metrics: {str(e)}'
        }), 500

@model_training_api.route('/api/models/registry', methods=['GET'])
def model_registry():
    """Get the model registry with available models"""
    try:
        # Get available models
        models = list_available_models()
        
        # Get detailed information for each model
        registry = []
        for model_id in models:
            model_info = get_model_info(model_id)
            if model_info:
                registry.append(model_info)
        
        return jsonify({
            'status': 'success',
            'data': registry
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model registry: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get model registry: {str(e)}'
        }), 500

@model_training_api.route('/api/training/system_health', methods=['GET'])
def system_health():
    """Get system health and resource usage"""
    try:
        # Get system health
        health = training_manager.get_system_health()
        
        return jsonify({
            'status': 'success',
            'data': health
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get system health: {str(e)}'
        }), 500

@model_training_api.route('/api/training/queue', methods=['GET', 'POST'])
def get_training_queue():
    """Get or modify training queue"""
    try:
        if request.method == 'GET':
            # Get training queue
            queue = training_manager.get_training_queue()
            
            return jsonify({
                'status': 'success',
                'data': queue
            }), 200
        elif request.method == 'POST':
            # Modify training queue
            data = request.json
            
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'No data provided'
                }), 400
            
            if 'action' not in data:
                return jsonify({
                    'status': 'error',
                    'message': 'Action is required'
                }), 400
            
            if data['action'] == 'reorder' and 'new_order' in data:
                # Reorder queue
                result = training_manager.reorder_training_queue(data['new_order'])
                
                if result.get('success'):
                    return jsonify({
                        'status': 'success',
                        'message': result.get('message', 'Training queue reordered')
                    }), 200
                else:
                    return jsonify({
                        'status': 'error',
                        'message': result.get('error', 'Failed to reorder training queue')
                    }), 400
            elif data['action'] == 'clear':
                # Clear queue
                result = training_manager.clear_training_queue()
                
                if result.get('success'):
                    return jsonify({
                        'status': 'success',
                        'message': result.get('message', 'Training queue cleared')
                    }), 200
                else:
                    return jsonify({
                        'status': 'error',
                        'message': result.get('error', 'Failed to clear training queue')
                    }), 400
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid action'
                }), 400
                
    except Exception as e:
        logger.error(f"Error managing training queue: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to manage training queue: {str(e)}'
        }), 500

@model_training_api.route('/api/models/<model_id>/status', methods=['GET'])
def get_model_status(model_id: str):
    """Get the status of a specific model"""
    try:
        # Get model status
        status = training_manager.get_model_status(model_id)
        
        if status:
            return jsonify({
                'status': 'success',
                'data': status
            }), 200
        else:
            # Try to get model info if no status is available
            model_info = get_model_info(model_id)
            if model_info:
                # Return basic model info with status as 'idle'
                return jsonify({
                    'status': 'success',
                    'data': {
                        **model_info,
                        'status': 'idle',
                        'last_training_time': None,
                        'training_progress': 0
                    }
                }), 200
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Model {model_id} not found'
                }), 404
                
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get model status: {str(e)}'
        }), 500

@model_training_api.route('/api/training/sessions', methods=['GET'])
def get_all_training_sessions():
    """Get all training sessions"""
    try:
        # Get all training sessions from training manager
        sessions = training_manager.get_all_training_sessions()
        
        return jsonify({
            'status': 'success',
            'data': sessions
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting training sessions: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get training sessions: {str(e)}'
        }), 500

@model_training_api.route('/api/training/prepare', methods=['POST'])
def prepare_training():
    """Prepare a model for training by loading necessary resources"""
    try:
        # Get request data
        data = request.json
        
        # Validate request data
        if not data or 'model_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Model ID must be provided'
            }), 400
        
        model_id = data['model_id']
        
        # Use training manager to prepare the model
        result = training_manager.prepare_model_for_training(model_id)
        
        if result.get('success'):
            return jsonify({
                'status': 'success',
                'message': result.get('message', f'Model {model_id} preparation started')
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('error', f'Failed to prepare model {model_id}')
            }), 400
        
    except Exception as e:
        logger.error(f"Error preparing model for training: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to prepare model: {str(e)}'
        }), 500

@model_training_api.route('/api/training/cancel', methods=['POST'])
def cancel_training():
    """Cancel a queued training session"""
    try:
        # Get request data
        data = request.json
        
        # Validate request data
        if not data or 'training_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Training ID must be provided'
            }), 400
        
        training_id = data['training_id']
        
        # Use training manager to cancel the training session
        result = training_manager.cancel_training(training_id)
        
        if result.get('success'):
            return jsonify({
                'status': 'success',
                'message': result.get('message', f'Training session {training_id} cancelled')
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('error', f'Failed to cancel training session {training_id}')
            }), 400
        
    except Exception as e:
        logger.error(f"Error cancelling training: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to cancel training: {str(e)}'
        }), 500

# Initialize the training API blueprint when the module is loaded
def initialize_training_api(app):
    """Initialize the training API with the Flask app"""
    if app is not None:
        app.register_blueprint(model_training_api)
        logger.info("Model Training API initialized successfully")
    else:
        logger.error("Failed to initialize Model Training API: No app provided")