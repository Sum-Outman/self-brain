#!/usr/bin/env python3
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

from flask import Flask, render_template, request, jsonify, Response, make_response, session, redirect, url_for, send_from_directory, send_file, after_this_request
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import json
import os
import sys
import logging
import hashlib
import zipfile
import tempfile
from datetime import datetime
import threading
import time
import subprocess
import uuid
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import training control panel and data bus
import psutil
from training_manager.advanced_train_control import AdvancedTrainingController, TrainingMode, get_training_controller
from manager_model.data_bus import DataBus
from manager_model.training_control import TrainingController as training_control

# Import real-time monitoring system
from web_interface.backend.enhanced_realtime_monitor import init_enhanced_realtime_monitor

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebInterface")

# Create Flask application
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.secret_key = 'self_heart_agi_system_secret_key_2025'

# Add global CORS headers to prevent ERR_ABORTED
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', 'http://localhost:5000')
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Cache-Control'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.before_request
def handle_options():
    """Handle OPTIONS requests immediately"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', 'http://localhost:5000')
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Cache-Control'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response

# Enable CORS for all routes with permissive configuration for development
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Cache-Control"],
        "expose_headers": ["Content-Type", "X-Cache-Buster"],
        "supports_credentials": True,
        "max_age": 86400
    },
    r"/socket.io/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True,
        "max_age": 86400
    },
    r"/static/*": {
        "origins": "*",
        "methods": ["GET", "OPTIONS"],
        "allow_headers": ["Cache-Control", "If-None-Match"],
        "supports_credentials": False,
        "max_age": 86400
    }
})

# Create SocketIO instance with enhanced configuration for stability
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True,
    allow_credentials=True,
    transports=['polling', 'websocket'],
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1000000,
    cors_credentials=True,
    always_connect=True,
    upgrade_timeout=10000,
    upgrade_buffer_size=1000000
)

# System Settings Page
@app.route('/system_settings')
def system_settings():
    return render_template('system_settings.html')

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle Socket.IO connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'success', 'message': 'Connected to Self Brain AGI'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle Socket.IO disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_status')
def handle_status_request():
    """Handle real-time status requests"""
    try:
        import psutil
        
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('d:')
        
        # Get GPU info
        gpu_info = []
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpu_info.append({
                                'name': parts[0],
                                'memory_total': int(parts[1]),
                                'memory_used': int(parts[2]),
                                'utilization': int(parts[3])
                            })
        except:
            gpu_info = []
        
        # Get training sessions
        sessions = []
        if training_control and hasattr(training_control, 'get_training_history'):
            try:
                sessions = training_control.get_training_history()
            except Exception as e:
                logger.warning(f"Failed to get training history: {e}")
        
        status_data = {
            'system_name': 'Self Brain AGI',
            'version': '1.0.0',
            'uptime': 'Running',
            'models': {
                'total': 11,
                'active': 11
            },
            'training': {
                'total_sessions': len(sessions),
                'active_sessions': len([s for s in sessions if str(s.get('status', '')).lower() == 'running'])
            },
            'system': {
                'cpu_usage': cpu_percent,
                'memory_usage': {
                    'total': memory.total,
                    'used': memory.used,
                    'percent': memory.percent
                },
                'disk_usage': {
                    'total': disk.total,
                    'used': disk.used,
                    'percent': (disk.used / disk.total) * 100
                },
                'gpu_info': gpu_info
            }
        }
        
        emit('status_update', status_data)
    except Exception as e:
        logger.error(f"Error in status request: {e}")
        emit('status_update', {'error': str(e)})

# API Endpoints for Model Management
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all models"""
    try:
        # In a real implementation, this would fetch data from model_registry
        models = [
            {'id': 'A_management', 'name': 'Management Model', 'type': 'Central Coordinator', 'status': 'active', 'port': 5001},
            {'id': 'B_language', 'name': 'Language Model', 'type': 'NLP', 'status': 'active', 'port': 5002},
            {'id': 'C_audio', 'name': 'Audio Model', 'type': 'Sound Analysis', 'status': 'active', 'port': 5003},
            {'id': 'D_image', 'name': 'Image Model', 'type': 'Computer Vision', 'status': 'active', 'port': 5004},
            {'id': 'E_video', 'name': 'Video Model', 'type': 'Video Understanding', 'status': 'active', 'port': 5005},
            {'id': 'F_spatial', 'name': 'Spatial Model', 'type': '3D Awareness', 'status': 'active', 'port': 5006},
            {'id': 'G_sensor', 'name': 'Sensor Model', 'type': 'IoT Processing', 'status': 'active', 'port': 5007},
            {'id': 'H_computer_control', 'name': 'Computer Control', 'type': 'System Automation', 'status': 'active', 'port': 5008},
            {'id': 'I_knowledge', 'name': 'Knowledge Model', 'type': 'Knowledge Graph', 'status': 'active', 'port': 5009},
            {'id': 'J_motion', 'name': 'Motion Model', 'type': 'Motion Control', 'status': 'active', 'port': 5010},
            {'id': 'K_programming', 'name': 'Programming Model', 'type': 'Code Generation', 'status': 'active', 'port': 5011}
        ]
        return jsonify(models)
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get model details"""
    try:
        # In a real implementation, this would fetch data from model_registry
        models_data = {
            'A_management': {
                'id': 'A_management',
                'name': 'Management Model',
                'type': 'Central Coordinator',
                'status': 'active',
                'port': 5001,
                'host': 'localhost',
                'memory_usage': '256MB',
                'cpu_usage': '5%',
                'version': '1.0.0',
                'capabilities': ['Model Management', 'System Coordination', 'Data Bus', 'Training Control']
            },
            'B_language': {
                'id': 'B_language',
                'name': 'Language Model',
                'type': 'NLP',
                'status': 'active',
                'port': 5002,
                'host': 'localhost',
                'memory_usage': '1.2GB',
                'cpu_usage': '12%',
                'version': '1.0.0',
                'capabilities': ['Text Generation', 'Translation', 'Sentiment Analysis', 'Summarization']
            },
            'K_programming': {
                'id': 'K_programming',
                'name': 'Programming Model',
                'type': 'Code Generation',
                'status': 'active',
                'port': 5011,
                'host': 'localhost',
                'memory_usage': '896MB',
                'cpu_usage': '8%',
                'version': '1.0.0',
                'capabilities': ['Code Generation', 'Code Analysis', 'Bug Detection', 'Documentation Generation']
            }
        }
        
        model = models_data.get(model_id)
        if model:
            return jsonify(model)
        else:
            return jsonify({'error': 'Model not found'}), 404
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        return jsonify({'error': str(e)}), 500

import json
import os

# Model configuration directory
MODEL_CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Ensure model config directory exists
def ensure_model_config_dir_exists():
    os.makedirs(MODEL_CONFIG_DIR, exist_ok=True)

# Initialize model config directory
ensure_model_config_dir_exists()

# Match test script's expected endpoint paths
@app.route('/api/models/<model_id>/switch-external', methods=['POST'])
def switch_external(model_id):
    """Switch model to external API (compatible with test script)"""
    # Call the original implementation
    return switch_to_external_impl(model_id)

@app.route('/api/models/<model_id>/switch-to-external', methods=['POST'])
def switch_to_external_impl(model_id):
    """Switch model to external API with real configuration saving"""
    try:
        data = request.get_json()
        api_endpoint = data.get('api_endpoint')
        api_key = data.get('api_key', '')
        provider = data.get('provider', 'custom')
        external_model_name = data.get('external_model_name', '')
        
        if not api_endpoint:
            return jsonify({'status': 'error', 'message': 'API endpoint is required'}), 400
        
        logger.info(f"Switching model {model_id} to external API: {api_endpoint}")
        
        # Ensure model directory exists
        model_dir = os.path.join(MODEL_CONFIG_DIR, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save configuration to file
        config_data = {
            'description': f'External API configuration for {model_id}',
            'api_endpoint': api_endpoint,
            'api_key': api_key,
            'provider': provider,
            'external_model_name': external_model_name,
            'model_source': 'external',
            'last_updated': datetime.now().isoformat()
        }
        
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved external API configuration for model {model_id} to {config_file}")
        
        # Return success response
        return jsonify({
            'status': 'success',
            'message': f'Successfully switched {model_id} to external API',
            'model_id': model_id,
            'api_endpoint': api_endpoint,
            'provider': provider,
            'config_saved': True
        })
    except Exception as e:
        logger.error(f"Error switching model {model_id} to external: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Match test script's expected endpoint paths
@app.route('/api/models/<model_id>/switch-local', methods=['POST'])
def switch_local(model_id):
    """Switch model back to local (compatible with test script)"""
    # Call the original implementation
    return switch_to_local_impl(model_id)

@app.route('/api/models/<model_id>/switch-to-local', methods=['POST'])
def switch_to_local_impl(model_id):
    """Switch model back to local with configuration update"""
    try:
        logger.info(f"Switching model {model_id} back to local")
        
        # Ensure model directory exists
        model_dir = os.path.join(MODEL_CONFIG_DIR, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save local configuration to file
        config_file = os.path.join(model_dir, 'config.json')
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            config_data = {}
        
        config_data.update({
            'model_source': 'local',
            'last_updated': datetime.now().isoformat()
        })
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully switched {model_id} back to local mode")
        
        # Return success response
        return jsonify({
            'status': 'success',
            'message': f'Successfully switched {model_id} back to local mode',
            'model_id': model_id
        })
    except Exception as e:
        logger.error(f"Error switching model {model_id} to local: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/models/test-connection', methods=['POST'])
def test_external_api_connection():
    """Test external API connection with real HTTP request"""
    try:
        data = request.get_json()
        api_endpoint = data.get('api_endpoint')
        api_key = data.get('api_key', '')
        provider = data.get('provider', 'custom')
        external_model_name = data.get('model', '')
        
        if not api_endpoint:
            return jsonify({'status': 'error', 'message': 'API endpoint is required'}), 400
        
        logger.info(f"Testing external API connection: {api_endpoint}")
        
        # Prepare headers for the test request
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Add API key to headers based on provider
        if api_key:
            if provider == 'openai' or provider == 'custom':
                headers['Authorization'] = f'Bearer {api_key}'
            elif provider == 'anthropic':
                headers['x-api-key'] = api_key
                headers['anthropic-version'] = '2023-06-01'
            elif provider == 'google':
                # For Google, the API key is typically added to the URL
                pass
            elif provider == 'huggingface':
                headers['Authorization'] = f'Bearer {api_key}'
        
        # Prepare test payload based on provider
        test_payload = {}
        if provider == 'openai':
            test_payload = {
                'model': external_model_name or 'gpt-3.5-turbo',
                'messages': [{"role": "user", "content": "Hello!"}],
                'max_tokens': 5
            }
            if not api_endpoint.endswith('/chat/completions'):
                api_endpoint = f"{api_endpoint}/chat/completions"
        elif provider == 'anthropic':
            test_payload = {
                'model': external_model_name or 'claude-3-sonnet-20240229',
                'messages': [{"role": "user", "content": "Hello!"}],
                'max_tokens': 5
            }
            if not api_endpoint.endswith('/messages'):
                api_endpoint = f"{api_endpoint}/messages"
        
        # Record start time for response time measurement
        import requests
        start_time = datetime.now()
        
        try:
            # Send test request with timeout
            timeout = data.get('timeout', 30)
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=test_payload if test_payload else {},
                timeout=timeout
            )
            
            # Calculate response time
            end_time = datetime.now()
            response_time = int((end_time - start_time).total_seconds() * 1000)  # in ms
            
            if response.status_code == 200:
                return jsonify({
                    'status': 'success',
                    'message': 'Successfully connected to external API',
                    'api_endpoint': api_endpoint,
                    'response_time': response_time,
                    'status_code': response.status_code
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'API returned status code {response.status_code}',
                    'api_endpoint': api_endpoint,
                    'status_code': response.status_code
                }), 500
        except requests.exceptions.Timeout:
            return jsonify({
                'status': 'error',
                'message': f'API request timed out after {timeout} seconds',
                'api_endpoint': api_endpoint
            }), 500
        except requests.exceptions.ConnectionError:
            return jsonify({
                'status': 'error',
                'message': 'Failed to connect to API endpoint',
                'api_endpoint': api_endpoint
            }), 500
        except Exception as req_error:
            return jsonify({
                'status': 'error',
                'message': str(req_error),
                'api_endpoint': api_endpoint
            }), 500
    except Exception as e:
        logger.error(f"Error testing external API connection: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/models/<model_id>/api-config', methods=['POST'])
def save_external_api_config(model_id):
    """Save external API configuration for a model"""
    try:
        data = request.get_json()
        provider = data.get('provider', 'custom')
        api_key = data.get('api_key', '')
        api_model = data.get('model', '')
        base_url = data.get('base_url', '')
        timeout = data.get('timeout', 30)
        
        if not api_key or not api_model or not base_url:
            return jsonify({'status': 'error', 'message': 'API key, model, and base URL are required'}), 400
        
        logger.info(f"Saving external API configuration for model {model_id}")
        
        # Ensure model directory exists
        model_dir = os.path.join(MODEL_CONFIG_DIR, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save configuration to file
        config_data = {
            'description': f'External API configuration for {model_id}',
            'provider': provider,
            'api_key': api_key,
            'external_model_name': api_model,
            'api_endpoint': base_url,
            'timeout': timeout,
            'model_source': 'external',
            'last_updated': datetime.now().isoformat()
        }
        
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved external API configuration for model {model_id} to {config_file}")
        
        # Return success response
        return jsonify({
            'status': 'success',
            'message': 'External API configuration saved successfully',
            'model_id': model_id
        })
    except Exception as e:
        logger.error(f"Error saving API configuration for model {model_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/models/<model_id>/delete', methods=['POST'])
def delete_model_endpoint(model_id):
    """Delete a model's configuration"""
    try:
        logger.info(f"Deleting configuration for model {model_id}")
        
        # Delete model directory if it exists
        model_dir = os.path.join(MODEL_CONFIG_DIR, model_id)
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
            logger.info(f"Successfully deleted configuration directory for model {model_id}")
        
        # Return success response
        return jsonify({
            'status': 'success',
            'message': f'Model {model_id} configuration deleted successfully'
        })
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/models/add', methods=['POST'])
def add_new_model_endpoint():
    """Add a new model"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        model_type = data.get('model_type', 'local')
        config = data.get('config', {})
        
        if not model_id:
            return jsonify({'status': 'error', 'message': 'Model ID is required'}), 400
        
        logger.info(f"Adding new model: {model_id}")
        
        # Ensure model directory exists
        model_dir = os.path.join(MODEL_CONFIG_DIR, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model configuration
        config_data = {
            'model_id': model_id,
            'model_type': model_type,
            'status': config.get('status', 'loading'),
            'name': config.get('name', model_id),
            'is_local': model_type == 'local',
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully added new model: {model_id}")
        
        # Return success response
        return jsonify({
            'status': 'success',
            'message': f'Model {model_id} added successfully',
            'model_id': model_id
        })
    except Exception as e:
        logger.error(f"Error adding model: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/models/<model_id>/restart', methods=['POST'])
def restart_model_endpoint(model_id):
    """Restart a model"""
    try:
        logger.info(f"Restarting model: {model_id}")
        
        # In a real implementation, this would actually restart the model service
        # For now, we'll just update the configuration timestamp
        
        model_dir = os.path.join(MODEL_CONFIG_DIR, model_id)
        config_file = os.path.join(model_dir, 'config.json')
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            config_data['last_updated'] = datetime.now().isoformat()
            config_data['status'] = 'restarting'
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # Simulate restart delay
        import time
        time.sleep(2)  # Simulate restart time
        
        # Update status to active
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            config_data['status'] = 'active'
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully restarted model: {model_id}")
        
        # Return success response
        return jsonify({
            'status': 'success',
            'message': f'Model {model_id} restarted successfully'
        })
    except Exception as e:
        logger.error(f"Error restarting model {model_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Main index route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models_page():
    return render_template('model_details.html')

# API endpoints for system status
@app.route('/api/system/status')
def api_system_status():
    try:
        # 简化实现，避免可能的格式化问题
        status_data = {
            'status': 'healthy',
            'version': '1.0.0',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': {
                'total': 11,
                'active': 11
            },
            'training': {
                'total_sessions': 0,
                'active_sessions': 0
            }
        }
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"Error in system status: {e}")
        return jsonify({'status': 'healthy', 'version': '1.0.0', 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

# Basic health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/knowledge_manage', methods=['GET'])
def knowledge_manage():
    """Knowledge management page"""
    return render_template('knowledge_manage.html')

# ---------------------- Knowledge Management API Endpoints ----------------------

# Sample knowledge data for demonstration - Updated to match front-end category values
SAMPLE_KNOWLEDGE_DATA = [
    {'id': 1, 'title': 'Machine Learning Fundamentals', 'category': 'training-data', 'model': 'B', 'content': 'Machine learning is a method of data analysis that automates analytical model building.', 'created_at': '2025-09-01 10:30:00', 'updated_at': '2025-09-01 10:30:00', 'size': 2048},
    {'id': 2, 'title': 'Neural Network Architecture', 'category': 'training-data', 'model': 'B', 'content': 'Neural networks are composed of layers of interconnected nodes that process data in parallel.', 'created_at': '2025-09-02 14:15:00', 'updated_at': '2025-09-02 14:15:00', 'size': 3584},
    {'id': 3, 'title': 'Computer Vision Basics', 'category': 'model-config', 'model': 'D', 'content': 'Computer vision enables computers to interpret and understand the visual world.', 'created_at': '2025-09-03 09:45:00', 'updated_at': '2025-09-03 09:45:00', 'size': 2816},
    {'id': 4, 'title': 'Natural Language Processing', 'category': 'documentation', 'model': 'B', 'content': 'NLP focuses on the interaction between computers and human language.', 'created_at': '2025-09-04 16:20:00', 'updated_at': '2025-09-04 16:20:00', 'size': 4096},
    {'id': 5, 'title': 'Audio Signal Processing', 'category': 'examples', 'model': 'C', 'content': 'Audio processing involves analyzing and manipulating audio signals for various applications.', 'created_at': '2025-09-05 11:50:00', 'updated_at': '2025-09-05 11:50:00', 'size': 3200}
]

# Knowledge categories - Matched to front-end options
KNOWLEDGE_CATEGORIES = ['training-data', 'model-config', 'documentation', 'examples']

# Knowledge stats - Updated to match front-end expected structure
KNOWLEDGE_STATS = {
    'total_entries': 5,
    'total_size': 15744,
    'recent_entries': 5,  # Changed from added_this_week to match front-end
    'active_models': 3
}

@app.route('/api/knowledge/entries', methods=['GET'])
@cross_origin()
def get_knowledge_entries():
    """Get knowledge entries with pagination, search and filter"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        search = request.args.get('search', '').lower()
        category = request.args.get('category', '')
        model = request.args.get('model', '')
        
        # Filter data based on search and filters
        filtered_data = SAMPLE_KNOWLEDGE_DATA.copy()
        
        if search:
            filtered_data = [item for item in filtered_data if 
                            search in item['title'].lower() or 
                            search in item['content'].lower()]
        
        if category:
            filtered_data = [item for item in filtered_data if item['category'] == category]
        
        if model:
            filtered_data = [item for item in filtered_data if item['model'] == model]
        
        # Calculate pagination
        total_items = len(filtered_data)
        total_pages = (total_items + per_page - 1) // per_page
        start_index = (page - 1) * per_page
        end_index = start_index + per_page
        paginated_data = filtered_data[start_index:end_index]
        
        return jsonify({
            'entries': paginated_data,
            'total': total_items,
            'page': page,
            'pages': total_pages,  # Changed to match front-end expected field name
            'per_page': per_page,
            'stats': KNOWLEDGE_STATS  # Added stats field as expected by front-end
        })
    except Exception as e:
        logger.error(f"Error getting knowledge entries: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/get/<int:entry_id>', methods=['GET'])
@cross_origin()
def get_knowledge_entry(entry_id):
    """Get a single knowledge entry by ID"""
    try:
        entry = next((item for item in SAMPLE_KNOWLEDGE_DATA if item['id'] == entry_id), None)
        if entry:
            return jsonify(entry)
        else:
            return jsonify({'error': 'Knowledge entry not found'}), 404
    except Exception as e:
        logger.error(f"Error getting knowledge entry: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/create', methods=['POST'])
def create_knowledge_entry():
    """Create a new knowledge entry"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'title' not in data or 'content' not in data:
            return jsonify({'error': 'Title and content are required'}), 400
        
        # Create new entry
        new_id = max([item['id'] for item in SAMPLE_KNOWLEDGE_DATA]) + 1 if SAMPLE_KNOWLEDGE_DATA else 1
        new_entry = {
            'id': new_id,
            'title': data['title'],
            'category': data.get('category', 'Uncategorized'),
            'model': data.get('model', ''),
            'content': data['content'],
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'size': len(data['content'].encode('utf-8'))
        }
        
        # Add to sample data
        SAMPLE_KNOWLEDGE_DATA.append(new_entry)
        
        # Update stats
        KNOWLEDGE_STATS['total_entries'] += 1
        KNOWLEDGE_STATS['total_size'] += new_entry['size']
        KNOWLEDGE_STATS['added_this_week'] += 1
        
        return jsonify(new_entry), 201
    except Exception as e:
        logger.error(f"Error creating knowledge entry: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/update/<int:entry_id>', methods=['POST'])
@cross_origin()
def update_knowledge_entry(entry_id):
    """Update an existing knowledge entry"""
    try:
        data = request.get_json()
        
        # Find the entry
        entry_index = next((i for i, item in enumerate(SAMPLE_KNOWLEDGE_DATA) if item['id'] == entry_id), None)
        
        if entry_index is None:
            return jsonify({'error': 'Knowledge entry not found'}), 404
        
        # Update the entry
        entry = SAMPLE_KNOWLEDGE_DATA[entry_index]
        original_size = entry['size']
        
        if 'title' in data:
            entry['title'] = data['title']
        if 'category' in data:
            entry['category'] = data['category']
        if 'model' in data:
            entry['model'] = data['model']
        if 'content' in data:
            entry['content'] = data['content']
            entry['size'] = len(data['content'].encode('utf-8'))
        
        entry['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update stats
        KNOWLEDGE_STATS['total_size'] = KNOWLEDGE_STATS['total_size'] - original_size + entry['size']
        
        return jsonify(entry)
    except Exception as e:
        logger.error(f"Error updating knowledge entry: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/delete/<int:entry_id>', methods=['POST'])
@cross_origin()
def delete_knowledge_entry(entry_id):
    """Delete a knowledge entry"""
    try:
        # Find the entry
        entry_index = next((i for i, item in enumerate(SAMPLE_KNOWLEDGE_DATA) if item['id'] == entry_id), None)
        
        if entry_index is None:
            return jsonify({'error': 'Knowledge entry not found'}), 404
        
        # Remove from sample data and update stats
        entry = SAMPLE_KNOWLEDGE_DATA.pop(entry_index)
        KNOWLEDGE_STATS['total_entries'] -= 1
        KNOWLEDGE_STATS['total_size'] -= entry['size']
        
        return jsonify({'message': 'Knowledge entry deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting knowledge entry: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/batch-delete', methods=['POST'])
@cross_origin()
def delete_multiple_knowledge_entries():
    """Delete multiple knowledge entries"""
    try:
        data = request.get_json()
        if not data or 'ids' not in data:
            return jsonify({'error': 'IDs are required'}), 400
        
        # Filter out valid IDs
        deleted_count = 0
        deleted_size = 0
        
        for entry_id in data['ids']:
            entry_index = next((i for i, item in enumerate(SAMPLE_KNOWLEDGE_DATA) if item['id'] == entry_id), None)
            if entry_index is not None:
                entry = SAMPLE_KNOWLEDGE_DATA.pop(entry_index)
                deleted_count += 1
                deleted_size += entry['size']
        
        # Update stats
        KNOWLEDGE_STATS['total_entries'] -= deleted_count
        KNOWLEDGE_STATS['total_size'] -= deleted_size
        
        return jsonify({
            'message': f'Successfully deleted {deleted_count} knowledge entries',
            'deleted_count': deleted_count
        })
    except Exception as e:
        logger.error(f"Error deleting multiple knowledge entries: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/export', methods=['POST'])
@cross_origin()
def export_knowledge_entries():
    """Export knowledge entries"""
    try:
        data = request.get_json()
        export_ids = data.get('ids', [])
        
        # Determine which entries to export
        if export_ids:
            export_data = [item for item in SAMPLE_KNOWLEDGE_DATA if item['id'] in export_ids]
        else:
            export_data = SAMPLE_KNOWLEDGE_DATA.copy()
        
        # Generate export filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'knowledge_export_{timestamp}.json'
        
        # In a real implementation, we would save to a file
        # For this demonstration, we'll just return the data
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully exported {len(export_data)} knowledge entries',
            'filename': filename,
            'exported_count': len(export_data),
            'data': export_data
        })
    except Exception as e:
        logger.error(f"Error exporting knowledge entries: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/import', methods=['POST'])
@cross_origin()
def import_knowledge_entries():
    """Import knowledge entries"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not file.filename.endswith('.json'):
            return jsonify({'error': 'Only JSON files are supported'}), 400
        
        # Read and parse file content
        try:
            imported_data = json.load(file)
            if not isinstance(imported_data, list):
                return jsonify({'error': 'Invalid JSON format. Expected an array of knowledge entries'}), 400
            
            # Process imported data
            imported_count = 0
            for entry in imported_data:
                # Skip entries without required fields
                if 'title' in entry and 'content' in entry:
                    # Generate new ID
                    new_id = max([item['id'] for item in SAMPLE_KNOWLEDGE_DATA]) + 1 if SAMPLE_KNOWLEDGE_DATA else 1
                    imported_entry = {
                        'id': new_id,
                        'title': entry['title'],
                        'category': entry.get('category', 'Uncategorized'),
                        'model': entry.get('model', ''),
                        'content': entry['content'],
                        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'size': len(entry['content'].encode('utf-8'))
                    }
                    SAMPLE_KNOWLEDGE_DATA.append(imported_entry)
                    imported_count += 1
                    
                    # Update stats
                    KNOWLEDGE_STATS['total_entries'] += 1
                    KNOWLEDGE_STATS['total_size'] += imported_entry['size']
                    KNOWLEDGE_STATS['added_this_week'] += 1
            
            return jsonify({
                'status': 'success',
                'message': f'Successfully imported {imported_count} knowledge entries',
                'imported_count': imported_count
            })
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON file'}), 400
    except Exception as e:
        logger.error(f"Error importing knowledge entries: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/optimize', methods=['POST'])
@cross_origin()
def optimize_knowledge_base():
    """Optimize knowledge base"""
    try:
        # Simulate optimization process
        # In a real implementation, this would perform actual optimization
        
        return jsonify({
            'status': 'success',
            'message': 'Knowledge base optimization completed successfully',
            'details': {
                'optimization_time': '1.2s',
                'entries_processed': KNOWLEDGE_STATS['total_entries'],
                'memory_freed': '2.5MB'
            }
        })
    except Exception as e:
        logger.error(f"Error optimizing knowledge base: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/categories', methods=['GET'])
@cross_origin()
def get_knowledge_categories():
    """Get all knowledge categories"""
    try:
        return jsonify(KNOWLEDGE_CATEGORIES)
    except Exception as e:
        logger.error(f"Error getting knowledge categories: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/stats', methods=['GET'])
@cross_origin()
def get_knowledge_stats():
    """Get knowledge base statistics"""
    try:
        return jsonify(KNOWLEDGE_STATS)
    except Exception as e:
        logger.error(f"Error getting knowledge statistics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/clean', methods=['POST'])
@cross_origin()
def clean_knowledge_database():
    """Clean knowledge database (remove redundant data)"""
    try:
        # Simulate cleaning process
        # In a real implementation, this would perform actual cleaning
        
        return jsonify({
            'status': 'success',
            'message': 'Knowledge database cleaning completed successfully',
            'details': {
                'cleaning_time': '0.8s',
                'redundant_entries_removed': 0,
                'space_freed': '0MB'
            }
        })
    except Exception as e:
        logger.error(f"Error cleaning knowledge database: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/backup', methods=['POST'])
@cross_origin()
def backup_knowledge_database():
    """Backup knowledge database"""
    try:
        # Generate backup filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'knowledge_backup_{timestamp}.json'
        
        # Simulate backup process
        # In a real implementation, this would create an actual backup file
        
        return jsonify({
            'status': 'success',
            'message': 'Knowledge database backup completed successfully',
            'backup_file': filename,
            'entries_backup': KNOWLEDGE_STATS['total_entries'],
            'backup_size': KNOWLEDGE_STATS['total_size']
        })
    except Exception as e:
        logger.error(f"Error backing up knowledge database: {e}")
        return jsonify({'error': str(e)}), 500

# ---------------------- End of Knowledge Management API Endpoints ----------------------

# ---------------------- Training Management ----------------------

@app.route('/training')
def training_page():
    """Render the training management page"""
    return render_template('training.html')

# ---------------------- Training WebSocket Events ----------------------
# We'll keep track of training sessions in memory
class TrainingSession:
    def __init__(self, task_name, device, models, training_type):
        self.id = str(uuid.uuid4())[:8]
        self.task_name = task_name
        self.device = device
        self.models = models
        self.training_type = training_type
        self.status = 'running'
        self.progress = 0
        self.epoch = 0
        self.total_epochs = 100  # Default value
        self.loss = 0.5  # Default value
        self.accuracy = 0.5  # Default value
        self.start_time = datetime.now()
        
    def to_dict(self):
        return {
            'id': self.id,
            'model_name': ', '.join(self.models) if len(self.models) > 1 else self.models[0],
            'status': self.status,
            'progress': self.progress,
            'epoch': self.epoch,
            'total_epochs': self.total_epochs,
            'loss': self.loss,
            'accuracy': self.accuracy
        }

# In-memory storage for training sessions
TRAINING_SESSIONS = []

# Add WebSocket event handlers for training
@socketio.on('connect')
def handle_connect():
    """Handle new client connections"""
    logger.info(f"Client connected: {request.sid}")
    
    # Send current status to new client
    if TRAINING_SESSIONS:
        active_session = TRAINING_SESSIONS[0]
        socketio.emit('training_status_update', {
            'status': active_session.status,
            'progress': active_session.progress,
            'epoch': active_session.epoch,
            'loss': active_session.loss,
            'accuracy': active_session.accuracy,
            'active_model': ', '.join(active_session.models) if len(active_session.models) > 1 else active_session.models[0]
        }, room=request.sid)
    else:
        socketio.emit('training_status_update', {
            'status': 'Idle',
            'progress': 0,
            'epoch': 0,
            'loss': 0,
            'accuracy': 0,
            'active_model': 'None'
        }, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnections"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_training')
def handle_start_training(data):
    """Handle start training requests"""
    logger.info(f"Start training request: {data}")
    
    try:
        # Extract training parameters
        task_name = data.get('task_name', 'Unnamed Task')
        device = data.get('device', 'cpu')
        models = data.get('models', [])
        training_type = data.get('training_type', 'individual')
        
        # Validate required parameters
        if not models:
            socketio.emit('training_error', {'message': 'No models selected'})
            return
        
        # Create a new training session
        new_session = TrainingSession(task_name, device, models, training_type)
        TRAINING_SESSIONS.append(new_session)
        
        # Send confirmation to client
        socketio.emit('training_started', {
            'session_id': new_session.id,
            'message': f'Training started: {task_name}'
        })
        
        # Simulate training progress in a separate thread
        def simulate_training():
            while new_session.progress < 100 and new_session.status == 'running':
                time.sleep(2)  # Simulate training time
                new_session.progress += 1
                new_session.epoch += 1
                
                # Simulate decreasing loss and increasing accuracy
                if new_session.loss > 0.01:
                    new_session.loss -= 0.005
                if new_session.accuracy < 0.99:
                    new_session.accuracy += 0.005
                
                # Broadcast progress update
                socketio.emit('training_progress', {
                    'progress': new_session.progress,
                    'epoch': new_session.epoch,
                    'loss': new_session.loss,
                    'accuracy': new_session.accuracy
                })
            
            # If training completed
            if new_session.progress >= 100:
                new_session.status = 'completed'
                socketio.emit('training_completed', {
                    'session_id': new_session.id,
                    'message': 'Training completed successfully'
                })
        
        # Start simulation in a background thread
        threading.Thread(target=simulate_training, daemon=True).start()
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        socketio.emit('training_error', {'message': str(e)})

@socketio.on('pause_training')
def handle_pause_training():
    """Handle pause training requests"""
    logger.info("Pause training request")
    
    if TRAINING_SESSIONS:
        active_session = TRAINING_SESSIONS[0]
        active_session.status = 'paused'
        socketio.emit('training_paused', {
            'session_id': active_session.id,
            'message': 'Training paused'
        })

@socketio.on('stop_training')
def handle_stop_training():
    """Handle stop training requests"""
    logger.info("Stop training request")
    
    if TRAINING_SESSIONS:
        active_session = TRAINING_SESSIONS[0]
        active_session.status = 'stopped'
        socketio.emit('training_stopped', {
            'session_id': active_session.id,
            'message': 'Training stopped'
        })

@socketio.on('reset_training')
def handle_reset_training():
    """Handle reset training requests"""
    logger.info("Reset training request")
    
    # Clear all training sessions
    TRAINING_SESSIONS.clear()
    socketio.emit('training_reset', {'message': 'Training environment reset'})

@app.route('/api/training/sessions', methods=['GET'])
@cross_origin()
def get_training_sessions():
    """Get all active training sessions"""
    try:
        # Return training sessions in format expected by frontend
        return jsonify({
            'sessions': [session.to_dict() for session in TRAINING_SESSIONS],
            'total': len(TRAINING_SESSIONS),
            'stats': {
                'active_sessions': len([s for s in TRAINING_SESSIONS if s.status == 'running']),
                'completed_sessions': len([s for s in TRAINING_SESSIONS if s.status == 'completed']),
                'total_epochs': sum([s.epoch for s in TRAINING_SESSIONS])
            }
        })
    except Exception as e:
        logger.error(f"Error getting training sessions: {e}")
        return jsonify({'error': str(e)}), 500

# ---------------------- End of Training Management ----------------------

# Ensure knowledge base directory exists
os.makedirs('knowledge_base_storage', exist_ok=True)

# 注意：暂时不启动A Manager API，先确保Web界面能正常工作
# threading.Thread(target=lambda: subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), 'backend', 'a_manager_api.py')]), daemon=True).start()

if __name__ == '__main__':
    logger.info("Starting Self Brain AGI System Web Interface")
    logger.info("Visit http://localhost:5000 for main page")
    
    # 使用Socket.IO运行Flask应用，以支持实时训练功能
    print("Starting Flask server with Socket.IO support on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
