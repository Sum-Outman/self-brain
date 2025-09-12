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

# Dashboard Backend API

import json
import random
import requests
try:
    import psutil
except ImportError:
    psutil = None  # Set to None if psutil is not available
from flask import Blueprint, jsonify, request, g, current_app
import os
import time
from datetime import datetime

# Create blueprint
dashboard_bp = Blueprint('dashboard', __name__)

# Store training metrics data
training_metrics_data = {
    'last_updated': None,
    'active_trainings': [],
    'completed_trainings': []
}

# Multilingual error messages
ERROR_MESSAGES = {
    'en': {
        'invalid_format': 'Invalid data format: expected object with active_trainings and completed_trainings',
        'missing_fields': 'Missing required fields: {fields}',
        'invalid_list': 'Invalid data format: expected list',
        'invalid_training_item': 'Invalid training item at index {index}: missing required fields',
        'monitoring_not_implemented': 'Monitoring service connection endpoint ready for implementation',
        'invalid_api_key': 'Invalid API key',
        'rate_limit': 'Too many requests, please try again later',
        'api_key_rotated': 'API key successfully rotated',
        'invalid_rotation_request': 'Invalid API key rotation request',
        'monitoring_connected': 'Successfully connected to monitoring service',
        'monitoring_disconnected': 'Monitoring service disconnected',
        'monitoring_connection_failed': 'Failed to connect to monitoring service: {error}',
        'invalid_monitoring_config': 'Invalid monitoring configuration'
    }
}

from collections import deque, defaultdict
import logging

# Configure logging
logger = logging.getLogger('dashboard')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/dashboard.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Rate limit tracking - using a more efficient data structure
rate_limit_data = defaultdict(lambda: deque(maxlen=30))

def get_error_message(key):
    """Get error message for current locale"""
    lang = getattr(g, 'lang', 'en')  # Default to English
    return ERROR_MESSAGES.get(lang, ERROR_MESSAGES['en']).get(key, key)

def load_metrics_data():
    """Load training metrics data from file"""
    global training_metrics_data
    try:
        data_file = os.path.join(current_app.config['DATA_DIR'], 'training_metrics.json')
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                training_metrics_data = json.load(f)
    except Exception as e:
        current_app.logger.error(f"Error loading metrics data: {str(e)}")

def save_metrics_data():
    """Save training metrics data to file"""
    try:
        data_file = os.path.join(current_app.config['DATA_DIR'], 'training_metrics.json')
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(training_metrics_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        current_app.logger.error(f"Error saving metrics data: {str(e)}")

def check_api_key():
    """Verify API key"""
    api_key = request.headers.get('X-API-KEY')
    if not api_key or api_key != current_app.config['API_KEY']:
        return jsonify({
            'status': 'error',
            'message': get_error_message('invalid_api_key')
        }), 401
    return None

def check_rate_limit(ip, endpoint):
    """Check rate limit - using a more efficient implementation"""
    key = f"{ip}:{endpoint}"
    now = time.time()
    
    # Clear records older than 60 seconds
    while rate_limit_data[key] and now - rate_limit_data[key][0] > 60:
        rate_limit_data[key].popleft()
    
    # Add current request timestamp
    rate_limit_data[key].append(now)
    
    # Check request frequency
    if len(rate_limit_data[key]) > 30:  # 30 requests per minute
        logger.warning(f"Rate limit exceeded for {key}")
        return jsonify({
            'status': 'error',
            'message': get_error_message('rate_limit')
        }), 429
    return None

def check_monitoring_connection(endpoint):
    """Check monitoring service connection"""
    try:
        # Try to connect to monitoring service
        response = requests.get(f"{endpoint}/status", timeout=3)
        if response.status_code == 200:
            return True, None
        return False, get_error_message('monitoring_connection_failed').format(error=f"HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        return False, get_error_message('monitoring_connection_failed').format(error=str(e))
    except Exception as e:
        return False, get_error_message('monitoring_connection_failed').format(error=str(e))

@dashboard_bp.route('/api/update_training_metrics', methods=['POST'])
def update_training_metrics():
    """Update training metrics data"""
    global training_metrics_data
    
    # Security verification
    error = check_api_key()
    if error:
        return error
    
    # Rate limiting
    error = check_rate_limit(request.remote_addr, 'update_training_metrics')
    if error:
        return error
    
    try:
        data = request.get_json()
        
        # Create data directory if it doesn't exist
        if not os.path.exists(current_app.config['DATA_DIR']):
            os.makedirs(current_app.config['DATA_DIR'])
        
        # Load existing data
        load_metrics_data()
        
        # Validate data structure
        if not isinstance(data, dict):
            return jsonify({
                'status': 'error', 
                'message': get_error_message('invalid_format')
            }), 400
        
        # Validate required fields
        required_fields = ['active_trainings', 'completed_trainings']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': get_error_message('missing_fields').format(fields=", ".join(missing_fields))
            }), 400
        
        # Validate data structure
        if not isinstance(data['active_trainings'], list) or not isinstance(data['completed_trainings'], list):
            return jsonify({
                'status': 'error',
                'message': get_error_message('invalid_list')
            }), 400
        
        # Validate training item format
        for i, item in enumerate(data['active_trainings']):
            if not isinstance(item, dict) or 'model_name' not in item or 'progress' not in item:
                return jsonify({
                    'status': 'error',
                    'message': get_error_message('invalid_training_item').format(index=i)
                }), 400
        
        for i, item in enumerate(data['completed_trainings']):
            if not isinstance(item, dict) or 'model_name' not in item or 'result' not in item:
                return jsonify({
                    'status': 'error',
                    'message': get_error_message('invalid_training_item').format(index=i)
                }), 400
        
        # Update data
        training_metrics_data = {
            'last_updated': datetime.now().isoformat(),
            'active_trainings': data['active_trainings'],
            'completed_trainings': data['completed_trainings']
        }
        
        # Save data
        save_metrics_data()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@dashboard_bp.route('/api/get_training_metrics', methods=['GET'])
def get_training_metrics():
    """Get training metrics data"""
    # Rate limiting
    if error := check_rate_limit(request.remote_addr, 'get_training_metrics'):
        return error
    
    # Load latest data
    load_metrics_data()
    
    return jsonify({
        'status': 'success',
        'data': training_metrics_data
    })

@dashboard_bp.route('/api/update_completed_trainings', methods=['POST'])
def update_completed_trainings():
    """Update completed trainings data"""
    global training_metrics_data
    
    # Security verification
    if error := check_api_key():
        return error
    
    # Rate limiting
    if error := check_rate_limit(request.remote_addr, 'update_completed_trainings'):
        return error
    
    try:
        data = request.get_json()
        
        # Load existing data
        load_metrics_data()
        
        if not isinstance(data, list):
            return jsonify({
                'status': 'error', 
                'message': get_error_message('invalid_list')
            }), 400
        
        # Validate training item format
        for i, item in enumerate(data):
            if not isinstance(item, dict) or 'model_name' not in item or 'result' not in item:
                return jsonify({
                    'status': 'error',
                    'message': get_error_message('invalid_training_item').format(index=i)
                }), 400
        
        training_metrics_data['completed_trainings'] = data
        training_metrics_data['last_updated'] = datetime.now().isoformat()
        
        # Save data
        save_metrics_data()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@dashboard_bp.route('/api/system_status', methods=['GET'])
def get_system_status():
    """Get system status"""
    # Rate limiting
    if error := check_rate_limit(request.remote_addr, 'get_system_status'):
        return error
    
    # Load monitoring configuration
    monitoring_config = current_app.config.get('MONITORING', {})
    if monitoring_config.get('enabled'):
        endpoint = monitoring_config.get('endpoint', '')
        if not endpoint:
            return jsonify({
                'status': 'error',
                'message': get_error_message('invalid_monitoring_config')
            }), 400
        
        # Check monitoring service connection
        connected, error_msg = check_monitoring_connection(endpoint)
        if not connected:
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 500
        
        try:
            # Get real data from monitoring service
            data_url = f"{endpoint}/data"
            response = requests.get(data_url, timeout=5)
            if response.status_code == 200:
                return jsonify({
                    'status': 'success',
                    'data': response.json(),
                    'source': 'monitoring_service'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': get_error_message('monitoring_connection_failed').format(
                        error=f"HTTP {response.status_code}"
                    )
                }), 500
        except requests.exceptions.RequestException as e:
            return jsonify({
                'status': 'error',
                'message': get_error_message('monitoring_connection_failed').format(error=str(e))
            }), 500
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': get_error_message('monitoring_connection_failed').format(error=str(e))
            }), 500
    else:
        # Monitoring service not enabled, try to use psutil to get real data (if available), otherwise use simulated data
        if psutil is not None:  # Check if psutil is available
            try:
                # Use psutil to get system status
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
                disk_usage = psutil.disk_usage('/').percent
                # Get current network IO counters and timestamp - safely handle Windows format character errors
                current_net_io = None
                current_time = time.time()
                try:
                    current_net_io = psutil.net_io_counters()
                except Exception as net_error:
                    # Format character error that may occur on Windows, use default values
                    current_app.logger.warning(f"Network IO monitoring limited: {net_error}")
                    # Create default network IO counter object
                    class DefaultNetIO:
                        bytes_sent = 0
                        bytes_recv = 0
                    current_net_io = DefaultNetIO()
                
                # Initialize network IO speed calculation
                upload_speed = 0.0
                download_speed = 0.0
                
                # If previous record exists, calculate speed
                if hasattr(g, 'last_net_io') and hasattr(g, 'last_net_io_time'):
                    time_delta = current_time - g.last_net_io_time
                    if time_delta > 0.5:  # Ensure sufficient time interval
                        upload_speed = (current_net_io.bytes_sent - g.last_net_io.bytes_sent) / (1024*1024) / time_delta
                        download_speed = (current_net_io.bytes_recv - g.last_net_io.bytes_recv) / (1024*1024) / time_delta
                    else:
                        # Time interval too short, use previous values
                        upload_speed = getattr(g, 'last_upload_speed', 0.0)
                        download_speed = getattr(g, 'last_download_speed', 0.0)
                else:
                    # First request, use default values
                    upload_speed = 0.0
                    download_speed = 0.0
                
                # Store current values for next use
                g.last_net_io = current_net_io
                g.last_net_io_time = current_time
                g.last_upload_speed = upload_speed
                g.last_download_speed = download_speed
                
                network_io = f"Upload: {upload_speed:.1f} MB/s, Download: {download_speed:.1f} MB/s"
                current_app.logger.info(f"Network IO: {network_io}")
                
                network_io = f"Upload: {upload_speed:.1f} MB/s, Download: {download_speed:.1f} MB/s"
                active_processes = len(psutil.pids())

                # Try to get GPU usage and model
                gpu_usage = 0
                gpu_model = "Unknown"
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage = gpus[0].load * 100  # Get first GPU usage rate
                        gpu_model = gpus[0].name  # Get first GPU model
                except ImportError:
                    # Use simulated data when GPUtil is not available
                    gpu_usage = round(random.uniform(5, 95), 1)
                    gpu_model = "NVIDIA GeForce RTX 4090" if random.random() > 0.5 else "AMD Radeon RX 7900 XTX"
                except Exception as e:
                    current_app.logger.warning(f"Error getting GPU info: {str(e)}")
                    # Use simulated data when error occurs
                    gpu_usage = round(random.uniform(5, 95), 1)
                    gpu_model = "NVIDIA GeForce RTX 4090" if random.random() > 0.5 else "AMD Radeon RX 7900 XTX"

                return jsonify({
                    'status': 'success',
                    'data': {
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage,
                        'gpu_usage': gpu_usage,
                        'gpu_model': gpu_model,  # Add GPU model
                        'network_io': network_io,
                        'disk_usage': disk_usage,
                        'disk_path': '/',
                        'active_processes': active_processes,
                        'last_updated': datetime.now().isoformat()
                    },
                    'source': 'local'
                })
            except Exception as e:
                current_app.logger.error(f"Error getting system status with psutil: {str(e)}")
                # Fall back to simulated data when error occurs
                pass

        # If psutil is not available or error, return simulated data
        # Generate simulated data
        cpu_usage = round(random.uniform(10, 80), 1)
        memory_usage = round(random.uniform(30, 90), 1)
        gpu_usage = round(random.uniform(5, 95), 1)
        upload_speed = round(random.uniform(10, 100), 1)
        download_speed = round(random.uniform(10, 100), 1)
        network_io = f"Upload: {upload_speed:.1f} MB/s, Download: {download_speed:.1f} MB/s"
        disk_usage = round(random.uniform(20, 85), 1)
        active_processes = random.randint(50, 200)
        
        # Add GPU model simulation information
        gpu_model = "NVIDIA GeForce RTX 4090" if random.random() > 0.5 else "AMD Radeon RX 7900 XTX"
        
        return jsonify({
            'status': 'success',
            'data': {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'gpu_usage': gpu_usage,
                'gpu_model': gpu_model,  # Add GPU model information
                'network_io': network_io,
                'disk_usage': disk_usage,
                'disk_path': '/',
                'active_processes': active_processes,
                'last_updated': datetime.now().isoformat()
            },
            'source': 'simulated'
        })

@dashboard_bp.route('/api/rotate_api_key', methods=['POST'])
def rotate_api_key():
    """Rotate API key"""
    # Security verification
    if error := check_api_key():
        return error
    
    # Rate limiting
    if error := check_rate_limit(request.remote_addr, 'rotate_api_key'):
        return error
    
    try:
        # Generate new API key
        new_key = os.urandom(24).hex()
        
        # Update configuration
        current_app.config['API_KEY'] = new_key
        
        # Save to configuration file
        config_path = os.path.join('config', 'dashboard_config.json')
        config_data = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        
        config_data['API_KEY'] = new_key
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        logger.info("API key rotated successfully")
        
        return jsonify({
            'status': 'success',
            'message': get_error_message('api_key_rotated'),
            'new_api_key': new_key
        })
    except Exception as e:
        logger.error(f"Error rotating API key: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@dashboard_bp.route('/api/connect_monitoring', methods=['POST'])
def connect_monitoring_service():
    """Connect to system monitoring service"""
    # TODO: Implement actual integration with system monitoring service
    return jsonify({
        'status': 'success',
        'message': get_error_message('monitoring_not_implemented')
    })

@dashboard_bp.before_app_first_request
def load_configuration():
    """Load configuration on first app request"""
    # Default configuration
    current_app.config.setdefault('API_KEY', 'default_secret_key')
    current_app.config.setdefault('DATA_DIR', 'data')
    current_app.config.setdefault('MONITORING', {
        'enabled': False,
        'endpoint': 'http://localhost:5000/monitoring'
    })
    
    # Try to load from config file
    try:
        config_path = os.path.join('config', 'dashboard_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                current_app.config.update(json.load(f))
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
    
    # Ensure log directory exists
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

@dashboard_bp.before_request
def set_language():
    """Set request language environment"""
    # Get language setting from request header, default to English
    g.lang = request.headers.get('Accept-Language', 'en')[:2].lower()
    if g.lang not in ['en', 'zh', 'de', 'ja', 'ru']:
        g.lang = 'en'
        
    # Ensure configuration is loaded
    if not hasattr(current_app, 'config'):
        load_configuration()
            
    # Ensure log directory exists (double check)
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
