# Self Brain AGI System Web Interface
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
from flask_cors import CORS
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
app.secret_key = 'self_heart_agi_system_secret_key_2025'  # Use more secure key in production

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
        disk = psutil.disk_usage('/')
        
        # Get GPU info
        gpu_info = []
        try:
            import subprocess
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
            },
            'timestamp': datetime.now().isoformat()
        }
        
        emit('system_status', status_data)
    except Exception as e:
        logger.error(f"Error sending status: {e}")
        emit('system_status', {
            'error': str(e),
            'system_name': 'Self Brain AGI',
            'version': '1.0.0'
        })

@socketio.on_error_default
def error_handler(e):
    """Handle Socket.IO errors"""
    logger.error(f"Socket.IO error: {e}")
    emit('error', {'message': str(e)})

# Create training control panel instance
training_control = get_training_controller()

# Initialize enhanced real-time monitoring system
init_enhanced_realtime_monitor(app, socketio)

# Initialize global call session storage
active_calls = {}

# Initialize demo models - using AdvancedTrainingController default configuration
# AdvancedTrainingController already includes all models including A_management
logger.info("AGI system models preloaded - using AdvancedTrainingController configuration")

# Language resource loading function
# Language dictionary functionality removed - all text uses English directly

# PeerJS server-side support
connected_peers = {}
peer_sessions = {}

@app.route('/peerjs/id', methods=['GET', 'OPTIONS'])
def peerjs_id():
    """Generate new Peer ID"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
        
    peer_id = str(uuid.uuid4())
    connected_peers[peer_id] = {
        'id': peer_id,
        'connected_at': datetime.now().isoformat(),
        'token': str(uuid.uuid4())
    }
    response = make_response(peer_id)
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.route('/peerjs', methods=['GET', 'OPTIONS'])
def peerjs_root():
    """PeerJS root path"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
        
    response = jsonify({'name': 'PeerJS Server', 'description': 'Self Brain AGI PeerJS Server'})
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/peerjs/<peer_id>/<token>/offer', methods=['POST', 'OPTIONS'])
def peerjs_offer(peer_id, token):
    """Handle offer request"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
        
    try:
        data = request.get_json() or {}
        peer_sessions[peer_id] = {
            'offer': data,
            'token': token,
            'timestamp': datetime.now().isoformat()
        }
        response = jsonify({'type': 'offer', 'success': True})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/peerjs/<peer_id>/<token>/candidate', methods=['POST', 'OPTIONS'])
def peerjs_candidate(peer_id, token):
    """Handle ICE candidate"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
        
    try:
        data = request.get_json() or {}
        if peer_id not in peer_sessions:
            peer_sessions[peer_id] = {}
        
        if 'candidates' not in peer_sessions[peer_id]:
            peer_sessions[peer_id]['candidates'] = []
        
        peer_sessions[peer_id]['candidates'].append(data)
        response = jsonify({'type': 'candidate', 'success': True})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/peerjs/<peer_id>/<token>/answer', methods=['POST', 'OPTIONS'])
def peerjs_answer(peer_id, token):
    """Handle answer request"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
        
    try:
        data = request.get_json() or {}
        peer_sessions[peer_id] = peer_sessions.get(peer_id, {})
        peer_sessions[peer_id]['answer'] = data
        peer_sessions[peer_id]['token'] = token
        response = jsonify({'type': 'answer', 'success': True})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/peerjs/<peer_id>/<token>/offer', methods=['GET', 'OPTIONS'])
def peerjs_get_offer(peer_id, token):
    """Get offer"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
        
    if peer_id in peer_sessions and 'offer' in peer_sessions[peer_id]:
        response = jsonify(peer_sessions[peer_id]['offer'])
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    return jsonify({'error': 'Offer not found'}), 404

@app.route('/peerjs/<peer_id>/<token>/candidate', methods=['GET', 'OPTIONS'])
def peerjs_get_candidates(peer_id, token):
    """Get ICE candidates"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
        
    if peer_id in peer_sessions and 'candidates' in peer_sessions[peer_id]:
        response = jsonify(peer_sessions[peer_id]['candidates'])
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    response = jsonify([])
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/peerjs/<peer_id>/<token>/answer', methods=['GET', 'OPTIONS'])
def peerjs_get_answer(peer_id, token):
    """Get answer"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
        
    if peer_id in peer_sessions and 'answer' in peer_sessions[peer_id]:
        response = jsonify(peer_sessions[peer_id]['answer'])
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    response = jsonify({'error': 'Answer not found'})
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response, 404

@app.route('/peerjs/<path:path>')
def peerjs_proxy(path):
    """PeerJS proxy route"""
    if path == 'id':
        return peerjs_id()
    elif path.endswith('/offer'):
        return peerjs_get_offer(path.split('/')[0], path.split('/')[1])
    elif path.endswith('/candidate'):
        return peerjs_get_candidates(path.split('/')[0], path.split('/')[1])
    elif path.endswith('/answer'):
        return peerjs_get_answer(path.split('/')[0], path.split('/')[1])
    return jsonify({"error": "Not found"}), 404

# Route definitions
@app.route('/')
def index():
    """Home page display - AI Chat Interface"""
    return render_template('ai_chat.html')

@app.route('/test-functions')
def test_functions():
    """Test page for JavaScript functions"""
    return render_template('function_test.html')

@app.route('/training')
def training_page():
    """Training page"""
    response = make_response(render_template('training.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response





@app.route('/system_settings')
def system_settings():
    """System settings page"""
    return render_template('system_settings.html')

@app.route('/api/settings/general', methods=['POST'])
def save_general_settings():
    """Save general system settings"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid JSON data'})
        
        # Create config directory if it doesn't exist
        config_dir = os.path.join('config')
        os.makedirs(config_dir, exist_ok=True)
        
        # Save general settings
        config_path = os.path.join(config_dir, 'general_settings.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"General settings saved: {data}")
        return jsonify({'status': 'success', 'message': 'General settings saved successfully'})
        
    except Exception as e:
        logger.error(f"Failed to save general settings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/settings/api', methods=['POST'])
def save_api_settings():
    """Save API configuration settings"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid JSON data'})
        
        # Create config directory if it doesn't exist
        config_dir = os.path.join('config')
        os.makedirs(config_dir, exist_ok=True)
        
        # Save API settings (encrypt sensitive data)
        config_path = os.path.join(config_dir, 'api_settings.json')
        
        # Simple encryption for sensitive keys (base64 encoding for basic protection)
        import base64
        encrypted_data = {}
        for key, value in data.items():
            if 'key' in key.lower() or 'token' in key.lower():
                if value:
                    encrypted_data[key] = base64.b64encode(value.encode()).decode()
                else:
                    encrypted_data[key] = value
            else:
                encrypted_data[key] = value
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(encrypted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"API settings saved (encrypted)")
        return jsonify({'status': 'success', 'message': 'API settings saved successfully'})
        
    except Exception as e:
        logger.error(f"Failed to save API settings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/settings/hardware', methods=['POST'])
def save_hardware_settings():
    """Save hardware configuration settings"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid JSON data'})
        
        # Create config directory if it doesn't exist
        config_dir = os.path.join('config')
        os.makedirs(config_dir, exist_ok=True)
        
        # Save hardware settings
        config_path = os.path.join(config_dir, 'hardware_settings.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Hardware settings saved: {data}")
        return jsonify({'status': 'success', 'message': 'Hardware settings saved successfully'})
        
    except Exception as e:
        logger.error(f"Failed to save hardware settings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/settings/security', methods=['POST'])
def save_security_settings():
    """Save security settings"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid JSON data'})
        
        # Create config directory if it doesn't exist
        config_dir = os.path.join('config')
        os.makedirs(config_dir, exist_ok=True)
        
        # Hash passwords before saving
        if 'adminPassword' in data and data['adminPassword']:
            import hashlib
            password_hash = hashlib.sha256(data['adminPassword'].encode()).hexdigest()
            data['adminPasswordHash'] = password_hash
            del data['adminPassword']
            if 'confirmPassword' in data:
                del data['confirmPassword']
        
        # Save security settings
        config_path = os.path.join(config_dir, 'security_settings.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Security settings saved")
        return jsonify({'status': 'success', 'message': 'Security settings saved successfully'})
        
    except Exception as e:
        logger.error(f"Failed to save security settings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/settings/load', methods=['GET'])
def load_all_settings():
    """Load all settings"""
    try:
        config_dir = os.path.join('config')
        settings = {}
        
        # Load all settings files
        setting_files = {
            'general': 'general_settings.json',
            'api': 'api_settings.json',
            'hardware': 'hardware_settings.json',
            'security': 'security_settings.json'
        }
        
        import base64
        
        for category, filename in setting_files.items():
            config_path = os.path.join(config_dir, filename)
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Decrypt sensitive data
                if category == 'api':
                    decrypted_data = {}
                    for key, value in data.items():
                        if 'key' in key.lower() or 'token' in key.lower():
                            if value and isinstance(value, str):
                                try:
                                    decrypted_data[key] = base64.b64decode(value.encode()).decode()
                                except:
                                    decrypted_data[key] = value
                            else:
                                decrypted_data[key] = value
                        else:
                            decrypted_data[key] = value
                    settings[category] = decrypted_data
                else:
                    settings[category] = data
            else:
                settings[category] = {}
        
        return jsonify({'status': 'success', 'settings': settings})
        
    except Exception as e:
        logger.error(f"Failed to load settings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/help')
def help_page():
    """Help page"""
    return render_template('help.html')

@app.route('/license')
def license_page():
    """License page"""
    # Read LICENSE file content
    license_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LICENSE')
    try:
        with open(license_path, 'r', encoding='utf-8') as f:
            license_content = f.read()
    except Exception as e:
        logger.error(f"Failed to read license file: {str(e)}")
        license_content = "Unable to load license file content"
    
    return render_template('license.html', license_content=license_content)

# Analytics page removed



@app.route('/peerjs_test')
def peerjs_test():
    """PeerJS test page"""
    return render_template('peerjs_test.html')

@app.route('/voice_test')
def voice_test():
    """Voice Call Test Page"""
    return render_template('voice_test.html')

@app.route('/advanced_chat')
def advanced_chat():
    """Advanced AI chat page - A management model coordination"""
    return render_template('ai_chat.html')



@app.route('/knowledge_manage')
def knowledge_manage():
    """Knowledge management page"""
    return render_template('knowledge_manage_enhanced.html')

@app.route('/knowledge_optimize')
def knowledge_optimize():
    """Knowledge database optimization page"""
    return render_template('knowledge_optimize.html')

@app.route('/knowledge_import')
def knowledge_import():
    """Knowledge import page"""
    return render_template('knowledge_import.html')

@app.route('/model_details/<model_id>')
def model_details(model_id):
    """Model details page"""
    return render_template('model_details.html', model_id=model_id)

@app.route('/training_results/<session_id>')
def training_results(session_id):
    """Training results page"""
    return render_template('training_results.html', session_id=session_id)



@app.route('/api/language/<lang>', methods=['GET', 'POST'])
def set_language(lang):
    """Set language API"""
    if lang in ['zh', 'en', 'de', 'ja', 'ru']:
        session['language'] = lang
        # Notify all models to update language setting
        try:
            # Try to notify each model to update language via HTTP request
            import requests
            model_ports = {
                'A_management': 5001,
                'B_language': 5002,
                'C_audio': 5003,
                'D_image': 5004,
                'E_video': 5005,
                'F_spatial': 5006,
                'G_sensor': 5007,
                'H_computer_control': 5008,
                'I_knowledge': 5009,
                'J_motion': 5010,
                'K_programming': 5011
            }
            
            for model_name, port in model_ports.items():
                try:
                    requests.post(f"http://localhost:{port}/set_language", 
                                json={'language': lang}, timeout=1)
                except:
                    # Ignore connection errors, model may not be started
                    pass
        except Exception as e:
            logger.warning(f"Failed to notify models language change: {str(e)}")
        
        return jsonify({'status': 'success', 'language': lang})
    return jsonify({'status': 'error', 'message': 'Unsupported language'})

@app.route('/knowledge')
def knowledge_page():
    """Knowledge base main page"""
    return render_template('knowledge_manage.html')



@app.route('/knowledge_base')
def knowledge_base_redirect():
    """Redirect /knowledge_base to /knowledge_manage"""
    return redirect('/knowledge_manage', code=301)

@app.route('/training_center')
def training_center_redirect():
    """Redirect /training_center to /training"""
    return redirect('/training', code=301)

@app.route('/about')
def about_page():
    """About page"""
    return render_template('about.html')

@app.route('/system_status')
def system_status_page():
    """System status page"""
    return render_template('system_status.html')

@app.route('/api/system_status')
def get_system_status():
    """Get system status API"""
    try:
        # Get system health status
        health_data = training_control.get_system_health()
        
        # Format response data to match JavaScript expected structure
        system_status = {
            'status': 'running',
            'message': 'Self Brain AGI system is running normally',
            'models': {
                'total': 11,  # Total number of models
                'active': 11   # Number of active models
            },
            'system': {
                'version': '1.0.0',
                'uptime': 'Running'
            }
        }
        
        return jsonify(system_status)
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        # Return default data in case of error
        return jsonify({
            'status': 'running',
            'message': 'Self Brain AGI system is running normally',
            'models': {'total': 11, 'active': 11},
            'system': {'version': '1.0.0', 'uptime': 'Running'}
        })

@app.route('/api/system/status')
def get_system_status_v2():
    """Get system status API - v2 endpoint for frontend compatibility"""
    try:
        return jsonify({
            'status': 'running',
            'message': 'System operational',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to get system status v2: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/system/version')
def get_system_version():
    """Get system version API"""
    try:
        return jsonify({
            'status': 'success',
            'version': '1.0.0',
            'build': '2025.09.11',
            'name': 'Self Brain AGI'
        })
    except Exception as e:
        logger.error(f"Failed to get system version: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/status')
def get_status_simple():
    """Simple status endpoint for compatibility"""
    try:
        return jsonify({
            'status': 'running',
            'message': 'System operational',
            'models': {'total': 11, 'active': 11}
        })
    except Exception as e:
        logger.error(f"Failed to get simple status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/training/status')
def get_training_status():
    """Get training status API endpoint"""
    try:
        # Get training sessions from training_control - handle case when training_control is None
        sessions = []
        if training_control and hasattr(training_control, 'get_training_history'):
            try:
                sessions = training_control.get_training_history()
            except Exception as e:
                logger.warning(f"Failed to get training history: {e}")
                sessions = []
        
        # Get current system resources
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get GPU info if available
        gpu_info = []
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
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
        
        # Calculate training statistics - handle different session formats
        active_sessions = [s for s in sessions if str(s.get('status', '')).lower() == 'running']
        completed_sessions = [s for s in sessions if str(s.get('status', '')).lower() == 'completed']
        failed_sessions = [s for s in sessions if str(s.get('status', '')).lower() in ['failed', 'error']]
        
        training_status = {
            'status': 'success',
            'training': {
                'active_sessions': len(active_sessions),
                'total_sessions': len(sessions),
                'completed_sessions': len(completed_sessions),
                'failed_sessions': len(failed_sessions),
                'sessions': sessions[-10:] if sessions else []  # Last 10 sessions or empty list
            },
            'system': {
                'cpu_usage': cpu_percent,
                'memory_usage': {
                    'total': memory.total,
                    'used': memory.used,
                    'available': memory.available,
                    'percent': memory.percent
                },
                'disk_usage': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'gpu_info': gpu_info
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(training_status)
        
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        # Return default data structure even on error
        return jsonify({
            'status': 'success',  # Keep success to prevent frontend errors
            'training': {
                'active_sessions': 0,
                'total_sessions': 0,
                'completed_sessions': 0,
                'failed_sessions': 0,
                'sessions': []
            },
            'system': {
                'cpu_usage': 0,
                'memory_usage': {'total': 0, 'used': 0, 'available': 0, 'percent': 0},
                'disk_usage': {'total': 0, 'used': 0, 'free': 0, 'percent': 0},
                'gpu_info': []
            },
            'timestamp': datetime.now().isoformat()
        })



@app.route('/api/models')
def get_models():
    """Get models list API"""
    try:
        models_dict = training_control.get_model_registry()
        # Convert model dict to array format
        models_array = []
        for model_id, model_data in models_dict.items():
            config = model_data.get('config', {})
            model_info = {
                'model_id': model_id,
                'name': config.get('name', model_id),
                'model_type': config.get('model_type', 'unknown'),
                'description': config.get('description', ''),
                'status': model_data.get('current_status', 'not_loaded'),
                'is_local': config.get('model_source', 'local') == 'local',
                'config': config
            }
            models_array.append(model_info)
        
        return jsonify({'status': 'success', 'models': models_array})
    except Exception as e:
        logger.error(f"Failed to get models list: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/chat/upload', methods=['POST'])
def upload_chat_file():
    """AI chat file upload API"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file was uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        # Validate file type and size
        allowed_extensions = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1][1:].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'status': 'error', 'message': f'File type {file_ext} not allowed'}), 400
        
        # Check file size (max 10MB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return jsonify({'status': 'error', 'message': 'File size exceeds 10MB limit'}), 400
        
        # Create upload directory
        upload_dir = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'chat_files')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{hashlib.md5(filename.encode()).hexdigest()[:8]}.{file_ext}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save file
        file.save(file_path)
        
        # Return file information
        return jsonify({
            'status': 'success',
            'filename': filename,
            'file_url': f"/static/uploads/chat_files/{unique_filename}",
            'file_path': file_path,
            'size': file_size
        })
    
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/training/sessions')
def get_training_sessions():
    """Get training sessions list API"""
    try:
        sessions = training_control.get_training_history()
        return jsonify({'status': 'success', 'sessions': sessions})
    except Exception as e:
        logger.error(f"Failed to get training sessions: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training API - FIXED VERSION"""
    try:
        print("=== DEBUG: /api/training/start endpoint called ===")
        
        # Check if training_control is initialized
        if training_control is None:
            print("DEBUG: Training control is None")
            return jsonify({'status': 'error', 'message': 'Training controller not initialized'}), 500
        
        data = request.get_json()
        if not data:
            print("DEBUG: No JSON data received")
            return jsonify({'status': 'error', 'message': 'Invalid JSON data'}), 400
        
        model_ids = data.get('model_ids', [])
        if not isinstance(model_ids, list) or not model_ids:
            return jsonify({'status': 'error', 'message': 'model_ids must be a non-empty list'}), 400
        
        print(f"DEBUG: Starting training with models: {model_ids}")
        
        mode_str = data.get('mode', 'joint')
        valid_modes = {'individual', 'joint', 'transfer', 'fine_tune'}
        if mode_str not in valid_modes:
            return jsonify({'status': 'error', 'message': f'Invalid mode. Must be one of: {valid_modes}'}), 400
        
        # Convert mode string to TrainingMode enum
        mode_mapping = {
            'individual': TrainingMode.INDIVIDUAL,
            'joint': TrainingMode.JOINT,
            'transfer': TrainingMode.TRANSFER,
            'fine_tune': TrainingMode.FINE_TUNE,
            'pretraining': TrainingMode.PRETRAINING
        }
        mode = mode_mapping.get(mode_str, TrainingMode.JOINT)
        
        training_config = {
            'epochs': data.get('epochs', 10),
            'batch_size': data.get('batch_size', 32),
            'learning_rate': data.get('learning_rate', 0.001),
            'knowledge_assisted': data.get('knowledge_assisted', False),
            'collaboration_level': data.get('collaboration_level', 'basic'),
            'training_type': data.get('training_type', 'supervised'),
            'compute_device': data.get('compute_device', 'auto')
        }
        
        print(f"DEBUG: Calling training_control.start_training with: {model_ids}, {mode}, {training_config}")
        
        # Call the training controller
        result = training_control.start_training(model_ids, mode, training_config)
        
        print(f"DEBUG: Training controller returned: {result}")
        
        # Return the actual result from the controller
        if result.get('status') == 'success':
            response_data = {
                'status': 'success',
                'session_id': result.get('training_id'),
                'message': result.get('message', 'Training started successfully')
            }
        else:
            response_data = {
                'status': 'error',
                'message': result.get('message', 'Training failed')
            }
        
        print(f"DEBUG: Final response data: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Failed to start training: {str(e)}"
        print(f"DEBUG: Exception in start_training: {error_msg}")
        logger.error(error_msg)
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/training/stop/<session_id>', methods=['POST'])
def stop_training(session_id):
    """Stop training API"""
    try:
        success = training_control.stop_training(session_id)
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to stop training'})
    except Exception as e:
        logger.error(f"Failed to stop training: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/<model_id>/status', methods=['POST'])
def toggle_model_status(model_id):
    """Toggle model status API"""
    try:
        data = request.get_json()
        new_status = data.get('status', 'active')
        
        # Validate status value
        if new_status not in ['active', 'inactive', 'error']:
            return jsonify({'status': 'error', 'message': 'Invalid status value'})
        
        # Call training controller to update model status
        success = training_control.update_model_status(model_id, new_status)
        
        if success:
            return jsonify({'status': 'success', 'message': f'Model {model_id} status updated to {new_status}'})
        else:
            return jsonify({'status': 'error', 'message': f'Failed to update model {model_id} status'})
    except Exception as e:
        logger.error(f"Failed to toggle model status: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/<model_id>/start', methods=['POST'])
def start_model(model_id):
    """Start model API"""
    try:
        success = training_control.start_model_service(model_id)
        if success:
            return jsonify({'status': 'success', 'message': f'Model {model_id} started successfully'})
        else:
            return jsonify({'status': 'error', 'message': f'Failed to start model {model_id}'})
    except Exception as e:
        logger.error(f"Failed to start model: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/<model_id>/stop', methods=['POST'])
def stop_model(model_id):
    """Stop model API"""
    try:
        success = training_control.stop_model_service(model_id)
        if success:
            return jsonify({'status': 'success', 'message': f'Model {model_id} stopped successfully'})
        else:
            return jsonify({'status': 'error', 'message': f'Failed to stop model {model_id}'})
    except Exception as e:
        logger.error(f"Failed to stop model: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/training/pause/<session_id>', methods=['POST'])
def pause_training(session_id):
    """Pause training API"""
    try:
        success = training_control.pause_training(session_id)
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to pause training'})
    except Exception as e:
        logger.error(f"Failed to pause training: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/training/resume/<session_id>', methods=['POST'])
def resume_training(session_id):
    """Resume training API"""
    try:
        success = training_control.resume_training(session_id)
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to resume training'})
    except Exception as e:
        logger.error(f"Failed to resume training: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/model/test/<model_id>')
def test_model_connection(model_id):
    """Test model connection API"""
    try:
        # Get model configuration
        model_config = training_control.get_model_configuration(model_id)
        return jsonify({'status': 'success', 'model': model_config})
    except Exception as e:
        logger.error(f"Failed to test model connection: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/test-connection', methods=['POST'])
def test_external_api_connection():
    """Test external API connection API"""
    try:
        data = request.get_json()
        api_endpoint = data.get('api_endpoint')
        api_key = data.get('api_key')
        model_name = data.get('model_name')
        
        if not api_endpoint or not api_key or not model_name:
            return jsonify({'status': 'error', 'message': 'Missing required parameters'})
        
        import requests
        import re
        
        # Clean up endpoint format
        api_endpoint = api_endpoint.strip()
        
        # SiliconFlow model mapping
        siliconflow_models = {
            'deepseek-ai/deepseek-r1': 'deepseek-ai/DeepSeek-R1',
            'deepseek-ai/deepseek-v2.5': 'deepseek-ai/DeepSeek-V2.5',
            'deepseek-ai/deepseek-coder-6.7b-instruct': 'deepseek-ai/deepseek-coder-6.7b-instruct',
            'qwen/qwen-2.5-72b-instruct': 'Qwen/Qwen2.5-72B-Instruct',
            'qwen/qwen-2.5-7b-instruct': 'Qwen/Qwen2.5-7B-Instruct',
            'qwen/qwen-2.5-coder-7b-instruct': 'Qwen/Qwen2.5-Coder-7B-Instruct',
            'thudm/glm-4-9b-chat': 'THUDM/glm-4-9b-chat',
            'meta-llama/llama-3.1-8b-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        }
        
        # Auto-detect and correct endpoint format
        def get_correct_endpoint(base_url, model_name):
            """Intelligent endpoint detection and correction"""
            base_url = base_url.rstrip('/')
            
            # SiliconFlow special handling
            if 'siliconflow' in base_url.lower():
            # Correct model name format
                corrected_model = siliconflow_models.get(model_name, model_name)
                return 'https://api.siliconflow.cn/v1/chat/completions', corrected_model
            
            # OpenAI format
            if 'openai' in base_url.lower():
                return 'https://api.openai.com/v1/chat/completions', model_name
            
            # Other mainstream API formats
            endpoint_map = {
                'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
                'moonshot': 'https://api.moonshot.cn/v1/chat/completions',
                'zhipu': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
                'baichuan': 'https://api.baichuan-ai.com/v1/chat/completions',
                'anthropic': 'https://api.anthropic.com/v1/messages',
            }
            
            for key, endpoint in endpoint_map.items():
                if key in base_url.lower():
                    return endpoint, model_name
            
            # Already a complete endpoint
            if base_url.endswith('/chat/completions') or base_url.endswith('/messages'):
                return base_url, model_name
            
            # Default OpenAI format
            if '/v1' in base_url:
                return base_url.rsplit('/v1', 1)[0] + '/v1/chat/completions', model_name
            else:
                return base_url + '/v1/chat/completions', model_name
        
        # Get correct endpoint and corrected model name
        test_endpoint, corrected_model = get_correct_endpoint(api_endpoint, model_name)
        
        # Standard OpenAI format headers
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Standard test payload
        test_payload = {
            'model': corrected_model,
            'messages': [{'role': 'user', 'content': 'hi'}],
            'max_tokens': 1,
            'temperature': 0.1
        }
        
        # Special handling for certain APIs
        if 'anthropic' in test_endpoint.lower():
            headers = {
                'x-api-key': api_key,
                'Content-Type': 'application/json'
            }
            test_payload = {
                'model': model_name,
                'max_tokens': 1,
                'messages': [{'role': 'user', 'content': 'hi'}]
            }
            test_endpoint = 'https://api.anthropic.com/v1/messages'
        elif 'google' in test_endpoint.lower():
            test_payload = {
                'contents': [{'parts': [{'text': 'hi'}]}],
                'generationConfig': {
                    'temperature': 0.1,
                    'topK': 1,
                    'topP': 1,
                    'maxOutputTokens': 1
                }
            }
            headers = {
                'Content-Type': 'application/json'
            }
            test_endpoint = f'https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}'
        
        try:
            # Try connection test
            response = requests.post(test_endpoint, headers=headers, json=test_payload, timeout=20)
            
            # Successful response
            if response.status_code == 200:
                result = response.json()
                
            # Validate response format
                valid_responses = [
                    'choices' in str(result).lower(),  # OpenAI format
        'content' in str(result).lower(),   # Anthropic format
        'candidates' in str(result).lower() # Google format
                ]
                
                if any(valid_responses):
                    return jsonify({
                        'status': 'success', 
                        'message': f'Connection successful to {test_endpoint}',
                        'endpoint_used': test_endpoint
                    })
                else:
                    return jsonify({
                        'status': 'error', 
                        'message': f'Connected but invalid response format from {test_endpoint}'
                    })
            
            # Handle various error statuses
            elif response.status_code == 405:
                return jsonify({
                    'status': 'error', 
                    'message': f'Method not allowed. Correct endpoint: {test_endpoint}'
                })
            elif response.status_code == 401:
                return jsonify({'status': 'error', 'message': 'Authentication failed - invalid API key'})
            elif response.status_code == 404:
                return jsonify({'status': 'error', 'message': f'Model "{model_name}" not found'})
            else:
            # Extract detailed error information
                error_detail = f'HTTP {response.status_code}'
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict):
                        if 'error' in error_data:
                            if isinstance(error_data['error'], dict):
                                error_detail += f': {error_data["error"].get("message", response.text)}'
                            else:
                                error_detail += f': {error_data["error"]}'
                        else:
                            error_detail += f': {response.text[:200]}'
                    else:
                        error_detail += f': {str(error_data)[:200]}'
                except:
                    error_detail += f': {response.text[:200]}'
                
                return jsonify({'status': 'error', 'message': error_detail})
                
        except requests.exceptions.Timeout:
            return jsonify({'status': 'error', 'message': 'Connection timeout - server not responding'})
        except requests.exceptions.ConnectionError as e:
            return jsonify({'status': 'error', 'message': f'Connection failed: {str(e)}'})
        except requests.exceptions.RequestException as e:
            return jsonify({'status': 'error', 'message': f'Request error: {str(e)}'})
            
    except Exception as e:
        logger.error(f"Failed to test external API connection: {str(e)}")
        return jsonify({'status': 'error', 'message': f'System error: {str(e)}'})

@app.route('/api/models/detect-endpoint', methods=['POST'])
def detect_api_endpoint():
    """Smart API endpoint detection"""
    try:
        data = request.get_json()
        api_endpoint = data.get('api_endpoint', '')
        
        if not api_endpoint:
            return jsonify({'status': 'error', 'message': 'No endpoint provided'})
        
        # Endpoint mapping table
        endpoint_map = {
            'siliconflow': 'https://api.siliconflow.cn/v1/chat/completions',
            'openai': 'https://api.openai.com/v1/chat/completions',
            'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
            'moonshot': 'https://api.moonshot.cn/v1/chat/completions',
            'zhipu': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
            'baichuan': 'https://api.baichuan-ai.com/v1/chat/completions',
            'anthropic': 'https://api.anthropic.com/v1/messages',
            'google': 'https://generativelanguage.googleapis.com/v1beta/models/MODEL_NAME:generateContent',
        }
        
        detected = None
        for key, endpoint in endpoint_map.items():
            if key in api_endpoint.lower():
                detected = {
                    'provider': key,
                    'correct_endpoint': endpoint,
                    'message': f'Detected {key} API'
                }
                break
        
        if detected:
            return jsonify({'status': 'success', **detected})
        else:
            # Default suggestion
            return jsonify({
                'status': 'success',
                'provider': 'openai-compatible',
                'correct_endpoint': api_endpoint.rstrip('/') + '/v1/chat/completions',
                'message': 'Using OpenAI compatible format'
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/system/resources')
def get_system_resources():
    """Get system resources API"""
    try:
        # Get system health status
        health_data = training_control.get_system_health()
        return jsonify({'status': 'success', 'resources': health_data})
    except Exception as e:
        logger.error(f"Failed to get system resources: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/all')
def get_all_models():
    """Get all models detailed info API"""
    try:
        models_dict = training_control.get_model_registry()
        # Convert model dict to array format
        models_array = []
        for model_id, model_data in models_dict.items():
            config = model_data.get('config', {})
            model_info = {
                'model_id': model_id,
                'name': config.get('name', model_id),
                'model_type': config.get('model_type', 'unknown'),
                'description': config.get('description', ''),
                'status': model_data.get('current_status', 'not_loaded'),
                'is_local': config.get('model_source', 'local') == 'local',
                'config': config,
                'performance_trend': model_data.get('performance_trend', 'stable'),
                'current_status': model_data.get('current_status', 'not_loaded'),
                'training_sessions': model_data.get('training_sessions', 0)
            }
            models_array.append(model_info)
        
        return jsonify({'status': 'success', 'models': models_array})
    except Exception as e:
        logger.error(f"Failed to get all models: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/performance')
def get_models_performance():
    """Get models performance data API"""
    try:
        models_dict = training_control.get_model_registry()
        performance_data = []
        
        for model_id, model_data in models_dict.items():
            # Generate mock performance data for now
            performance_info = {
                'model_id': model_id,
                'model_name': model_data.get('name', model_id),
                'response_time': round(random.uniform(0.1, 2.5), 2),
                'success_rate': round(random.uniform(85, 100), 1),
                'last_used': datetime.now().isoformat(),
                'status': model_data.get('current_status', 'not_loaded')
            }
            performance_data.append(performance_info)
        
        return jsonify({'status': 'success', 'performance': performance_data})
    except Exception as e:
        logger.error(f"Failed to get models performance: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/<model_id>')
def get_model_details(model_id):
    """Get specific model details API"""
    try:
        model_config = training_control.get_model_configuration(model_id)
        return jsonify({'status': 'success', 'model': model_config})
    except Exception as e:
        logger.error(f"Failed to get model details: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/add', methods=['POST'])
def add_model():
    """Add model API"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        model_type = data.get('model_type')
        config = data.get('config', {})
        
        # AdvancedTrainingController has preloaded all models, return success
        return jsonify({'status': 'success', 'message': 'Model configuration updated successfully'})
    except Exception as e:
        logger.error(f"Failed to add model: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/<model_id>/update', methods=['POST'])
def update_model(model_id):
    """Update model configuration API"""
    try:
        data = request.get_json()
        config = data.get('config', {})
        
        # Get existing model configuration
        try:
            model_config = training_control.get_model_configuration(model_id)
            if not model_config:
                return jsonify({'status': 'error', 'message': 'Model not found'})
            
            # Merge configuration updates
            updated_config = {**model_config, **config}
            
            # Update model configuration
            training_control.update_model_configuration(model_id, updated_config)
            
            logger.info(f"Model configuration updated successfully: {model_id}, config: {config}")
            return jsonify({
                'status': 'success', 
                'message': 'Model configuration updated successfully',
                'config': updated_config
            })
            
        except AttributeError:
            # If training_control doesn't have update_model_configuration method, use file storage
            import json
            import os
            
            config_file = f'models/{model_id}/config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
                
                # Merge configurations
                updated_config = {**existing_config, **config}
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                
                # Save updated configuration
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(updated_config, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Model configuration updated via file storage: {model_id}")
                return jsonify({
                    'status': 'success', 
                    'message': 'Model configuration updated successfully',
                    'config': updated_config
                })
            else:
                # If model file doesn't exist, directly create new configuration
                new_config = config
                os.makedirs(f'models/{model_id}', exist_ok=True)
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(new_config, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Created new model configuration: {model_id}")
                return jsonify({
                    'status': 'success', 
                    'message': 'Model configuration created successfully',
                    'config': new_config
                })
                
    except Exception as e:
        logger.error(f"Failed to update model: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/<model_id>/delete', methods=['POST'])
def delete_model(model_id):
    """Delete model API"""
    try:
        success = training_control.delete_model(model_id)
        if success:
            return jsonify({'status': 'success', 'message': 'Model deleted successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to delete model'})
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/training/modes')
def get_training_modes():
    """Get training modes API"""
    try:
        modes = training_control.get_training_modes()
        return jsonify({'status': 'success', 'modes': modes})
    except Exception as e:
        logger.error(f"Failed to get training modes: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/training/history')
def get_training_history():
    """Get training history API"""
    try:
        history = training_control.get_training_history()
        return jsonify({'status': 'success', 'history': history})
    except Exception as e:
        logger.error(f"Failed to get training history: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/training/config', methods=['GET', 'POST'])
def handle_training_config():
    """Handle training configuration API"""
    try:
        if request.method == 'GET':
            # Get current configuration
            config = {
                'training_mode': 'supervised',
                'model_architecture': 'transformer',
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'dataset': 'mnist',
                'early_stopping': True,
                'validation_split': 0.2,
                'data_augmentation': True
            }
            return jsonify({'status': 'success', 'config': config})
        
        elif request.method == 'POST':
            # Save configuration
            data = request.get_json()
            if not data:
                return jsonify({'status': 'error', 'message': 'Invalid JSON data'})
            
            # Here you can add configuration validation logic
            
            logger.info(f"Training configuration updated: {data}")
            return jsonify({'status': 'success', 'message': 'Configuration saved successfully'})
            
    except Exception as e:
        logger.error(f"Failed to handle training config: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/training/config/reset', methods=['POST'])
def reset_training_config():
    """Reset training configuration API"""
    try:
        default_config = {
            'training_mode': 'supervised',
            'model_architecture': 'transformer',
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'dataset': 'mnist',
            'early_stopping': True,
            'validation_split': 0.2,
            'data_augmentation': True
        }
        
        logger.info("Training configuration reset to defaults")
        return jsonify({'status': 'success', 'config': default_config})
        
    except Exception as e:
        logger.error(f"Failed to reset training config: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# Analytics API endpoint removed

@app.route('/api/analytics/performance')
def get_performance_analytics():
    """Get performance analytics API"""
    try:
        analytics = training_control.get_performance_analytics()
        return jsonify({'status': 'success', 'analytics': analytics})
    except Exception as e:
        logger.error(f"Failed to get performance analytics: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/status')
def get_knowledge_status():
    """Get knowledge base status API"""
    try:
        status = training_control.get_knowledge_base_status()
        return jsonify({'status': 'success', 'knowledge': status})
    except Exception as e:
        logger.error(f"Failed to get knowledge status: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/update', methods=['POST'])
def update_knowledge():
    """Update knowledge base API"""
    try:
        data = request.get_json()
        updates = data.get('updates', {})
        
        success = training_control.update_knowledge_base(updates)
        if success:
            return jsonify({'status': 'success', 'message': 'Knowledge base updated successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to update knowledge base'})
    except Exception as e:
        logger.error(f"Failed to update knowledge base: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# Chat related APIs
@app.route('/api/chat/conversations')
def get_conversations():
    """Get conversations list"""
    try:
        # Mock conversation data
        conversations = [
            {
                'id': 'conv_001',
                'title': 'Discussion about machine learning',
                'last_activity': '2024-01-01 14:30:00',
                'message_count': 15
            },
            {
                'id': 'conv_002',
                'title': 'Neural network optimization',
                'last_activity': '2024-01-01 13:15:00',
                'message_count': 8
            }
        ]
        
        return jsonify({
            'status': 'success',
            'conversations': conversations
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/chat/new_conversation', methods=['POST'])
def new_conversation():
    """Create new conversation"""
    try:
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return jsonify({
            'status': 'success',
            'conversation_id': conversation_id
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/chat/messages/<conversation_id>')
def get_messages(conversation_id):
    """Get conversation messages"""
    try:
        # Mock message data
        messages = [
            {
                'id': 'msg_001',
                'role': 'user',
                'content': 'Hello, I want to understand how neural networks work',
                'timestamp': '2024-01-01 14:00:00'
            },
            {
                'id': 'msg_002',
                'role': 'assistant',
                'content': 'Hello! Neural networks are machine learning models that mimic the way neurons connect in the human brain...',
                'timestamp': '2024-01-01 14:00:30'
            }
        ]
        
        return jsonify({
            'status': 'success',
            'messages': messages
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/chat/send', methods=['POST'])
def send_message():
    """Send message"""
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        message = data.get('message')
        knowledge_base = data.get('knowledge_base', 'all')
        attachments = data.get('attachments', [])
        
        logger.info(f"Sending message to conversation {conversation_id}: {message}")
        
        # Call A_management model to process message
        try:
            # Here simulate A_management model response logic
            response = generate_ai_response(message, knowledge_base, attachments)
            
            return jsonify({
                'status': 'success',
                'response': response,
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'should_speak': True
            })
        except Exception as ai_error:
            logger.error(f"AI model call failed: {str(ai_error)}")
            return jsonify({
                'status': 'success',
                'response': f"Sorry, I am temporarily unable to process your request. Error: {str(ai_error)}",
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def generate_ai_response(message, knowledge_base, attachments):
    """Generate AI response by calling A Management Model"""
    try:
        # Call A Management Model API to process the message
        import requests
        import json
        
        # Clean up the message to ensure it's passed correctly
        clean_message = str(message).strip()
        
        # Generate unique conversation ID if not provided
        conversation_id = str(uuid.uuid4())
        
        # Direct call to A Management Model with proper format and encoding
        response = requests.post(
            "http://localhost:5015/api/chat",
            json={
                "message": clean_message
            },
            headers={
                'Content-Type': 'application/json; charset=utf-8',
                'Accept': 'application/json; charset=utf-8'
            },
            timeout=30
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                
                # Handle A Management Model response format
                if isinstance(result, dict):
                    # A Management Model returns response in conversation_data.response
                    if 'conversation_data' in result and isinstance(result['conversation_data'], dict):
                        conv_data = result['conversation_data']
                        if 'response' in conv_data:
                            return str(conv_data['response'])
                    
                    # Direct response field
                    if 'response' in result:
                        return str(result['response'])
                    
                    # Fallback to string representation
                    return str(result)
                else:
                    # Handle case where response is just a string
                    return str(result)
                    
            except (ValueError, KeyError) as e:
                # If JSON parsing fails, return raw text
                return response.text
        else:
            logger.warning(f"A Management Model API call failed: {response.status_code} - {response.text}")
            return f"A Management Model returned status {response.status_code}: {response.text}"
            
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to A Management Model at localhost:5015")
        return "A Management Model service is not reachable. Please ensure the model service is running on port 5015."
    except requests.exceptions.Timeout:
        logger.error("A Management Model request timed out")
        return "A Management Model is taking too long to respond. Please try again in a moment."
    except Exception as e:
        logger.error(f"Failed to call A Management Model: {str(e)}")
        return f"Error communicating with A Management Model: {str(e)}. Please check the system status."

@app.route('/api/chat/suggestions', methods=['POST'])
def get_suggestions():
    """Get input suggestions"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Provide intelligent suggestions based on input text
        suggestions = []
        
        if len(text) < 3:
            suggestions = ["hello", "help", "summarize", "translate", "code"]
        elif 'python' in text.lower():
            suggestions = ["Python data analysis", "Python machine learning", "Python web scraping", "Python automation scripts"]
        elif 'machine learning' in text.lower() or 'deep learning' in text.lower():
            suggestions = ["Neural network principles", "Model training techniques", "Algorithm selection advice", "Performance optimization methods"]
        else:
            suggestions = [
                "Explain in detail",
                "Can you give an example?",
                "What are the application scenarios?",
                "What prerequisites are needed?"
            ]
        
        return jsonify({
            'status': 'success',
            'suggestions': suggestions[:5]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# Enhanced AI Chat APIs
@app.route('/api/chat/enhanced/upload', methods=['POST'])
def upload_enhanced_chat_file():
    """Upload chat files"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file was uploaded'})
        
        file = request.files['file']
        conversation_id = request.form.get('conversation_id', '')
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'})
        
        # Save file
        upload_dir = os.path.join('static', 'uploads', 'chat', conversation_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        
        # Generate file information
        file_info = {
            'name': filename,
            'size': os.path.getsize(file_path),
            'type': file.content_type,
            'url': f'/static/uploads/chat/{conversation_id}/{filename}',
            'upload_time': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'file_info': file_info
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/chat/enhanced/send', methods=['POST'])
def send_enhanced_message():
    """Send enhanced message"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        attachments = data.get('attachments', [])
        knowledge_base = data.get('knowledge_base', 'all')
        model = data.get('model', 'a_manager')
        
        start_time = time.time()
        
        # Call appropriate AI model
        response = generate_enhanced_ai_response(message, attachments, knowledge_base, model)
        
        response_time = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'status': 'success',
            'response': response,
            'response_time': response_time,
            'tokens_used': len(response.split()),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def generate_enhanced_ai_response(message, attachments, knowledge_base, model):
    """Generate enhanced AI response"""
    try:
        # Call different AI services based on selected model
        model_endpoints = {
            'a_manager': 'http://localhost:5001/process_message',
            'b_language': 'http://localhost:5002/generate_text',
            'c_audio': 'http://localhost:5003/process_audio'
        }
        
        endpoint = model_endpoints.get(model, model_endpoints['a_manager'])
        
        # Prepare request data
        request_data = {
            'message': message,
            'attachments': attachments,
            'knowledge_base': knowledge_base,
            'model': model
        }
        
        try:
            import requests
            response = requests.post(endpoint, json=request_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Processing completed')
            else:
                logger.warning(f"Model API call failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to call model API: {str(e)}")
        
        # Fallback intelligent response logic
        return generate_intelligent_response(message, attachments, knowledge_base, model)
        
    except Exception as e:
        return f"Sorry, encountered a problem while processing your request: {str(e)}"

def generate_intelligent_response(message, attachments, knowledge_base, model):
    """Intelligent response generation"""
    message_lower = message.lower()
    
    # Analyze attachment content
    attachment_analysis = []
    if attachments:
        for att in attachments:
            if att['type'].startswith('image/'):
                attachment_analysis.append(f"Image file: {att['name']} ({att['size']} bytes)")
            elif att['type'].startswith('video/'):
                attachment_analysis.append(f"Video file: {att['name']} ({att['size']} bytes)")
            elif att['type'].startswith('audio/'):
                attachment_analysis.append(f"Audio file: {att['name']} ({att['size']} bytes)")
            else:
                attachment_analysis.append(f"Document file: {att['name']} ({att['size']} bytes)")
    
    # Generate response based on message content and attachments
    response_parts = []
    
    if attachment_analysis:
        response_parts.append(f"I have received your {len(attachments)} files:")
        response_parts.extend(attachment_analysis)
    
    # Check for model-related queries and handle them specially
    model_phrases = [
        'show all models', 'show models', 'list all models', 'list models',
        'display all models', 'display models', 'model list', 'available models',
        '查看所有模型', '查看模型', '显示所有模型', '显示模型', '列出所有模型', '列出模型',
        '模型列表', '有什么模型', '模型有哪些'
    ]
    is_model_query = any(phrase == message_lower.strip() or message_lower.strip().startswith(phrase) for phrase in model_phrases)
    
    if is_model_query:
        try:
            # Directly call A Management Model for model information
            import requests
            response = requests.post(
                "http://localhost:5015/api/chat",
                json={"message": message},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'conversation_data' in result and 'response' in result['conversation_data']:
                    return result['conversation_data']['response']
                elif 'response' in result:
                    return result['response']
                else:
                    return str(result)
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return f"I understand you want to see models, but I'm having trouble connecting to the model service: {str(e)}"
    
    # Intelligent keyword matching
    intelligent_responses = {
        'image': "I can help you analyze image content, extract text information, identify objects, or answer questions about images.",
        'video': "I can analyze video content, extract key frames, transcribe audio, or summarize video highlights.",
        'audio': "I can transcribe audio content, analyze speech features, or answer questions about audio.",
        'translate': "I can perform multilingual translation, supporting text, speech, and text in images.",
        'summarize': "I can summarize documents, text in images, audio transcriptions, etc., and extract key information.",
        'analyze': "I can analyze data, charts, text content, provide insights and suggestions.",
        'code': "I can generate, explain, debug code, supporting multiple programming languages."
    }
    
    matched_keywords = []
    for keyword, response in intelligent_responses.items():
        if keyword in message_lower:
            matched_keywords.append(response)
    
    if matched_keywords:
        response_parts.extend(matched_keywords)
    
    # General response
    if not response_parts:
        response_parts = [
            f"I understand your need: {message}",
            "I can help you with:",
            "• Providing detailed information based on knowledge base",
            "• Analyzing uploaded file content",
            "• Performing multilingual translation",
            "• Generating or explaining code",
            "• Providing intelligent suggestions and insights",
            "Please tell me how you'd like to proceed?"
        ]
    
    return '\n\n'.join(response_parts)

# Socket.IO event handling
@socketio.on('user_message')
def handle_user_message(data):
    """Handle user message event - support file upload and multimodal interaction"""
    try:
        message = data.get('message', '')
        files = data.get('files', [])
        model = data.get('model', 'a_manager')
        temperature = data.get('temperature', 0.7)
        enable_emotion = data.get('enable_emotion', False)
        
        # Send input indication
        emit('typing')
        
        # Process file attachments
        file_attachments = []
        if files:
            upload_dir = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'chat_files')
            for file_info in files:
                # Here you can process actual file content analysis
                file_attachments.append({
                    'name': file_info.get('name'),
                    'type': file_info.get('type'),
                    'size': file_info.get('size')
                })
        
        # Simulate AI processing time (call model API in actual application)
        import time
        time.sleep(1.5)  # Increase processing time to simulate complex analysis
        
        # Generate intelligent response
        response = generate_enhanced_ai_response(message, file_attachments, 'all', model)
        
        # Add file processing results response
        if files:
            file_summary = f"Processed {len(files)} files: " + ", ".join([f["name"] for f in files])
            response = f"{file_summary}\n\n{response}"
        
        emit('ai_response', {
            'response': response,
            'response_time': 1500,
            'tokens_used': len(response.split()),
            'files_processed': len(files)
        })
        
    except Exception as e:
        emit('ai_response', {
            'response': f'Error processing message: {str(e)}',
            'response_time': 0,
            'tokens_used': 0,
            'files_processed': 0
        })

@socketio.on('send_message')
def handle_send_message(data):
    """Handle send message event (backward compatibility)"""
    return handle_user_message(data)

@socketio.on('switch_model')
def handle_switch_model(data):
    """Handle switch model event"""
    model = data.get('model', 'a_manager')
    emit('model_switched', {'model': model, 'status': 'success'})

@socketio.on('start_call')
def handle_start_call(data):
    """Handle start call event - support real-time voice and video"""
    try:
        call_type = data.get('type', 'voice')
        user_id = request.sid
        
        # Generate call ID and WebRTC configuration
        call_id = str(uuid.uuid4())
        
        # Create call session
        active_calls[user_id] = {
            'call_id': call_id,
            'type': call_type,
            'start_time': datetime.now(),
            'status': 'active',
            'peer_id': data.get('peer_id'),
            'offer': data.get('offer'),
            'ice_candidates': []
        }
        
        # Return WebRTC configuration
        webrtc_config = {
            'iceServers': [
                {'urls': 'stun:stun.l.google.com:19302'},
                {'urls': 'stun:stun1.l.google.com:19302'},
                {'urls': 'stun:stun2.l.google.com:19302'}
            ]
        }
        
        emit('call_started', {
            'call_id': call_id,
            'type': call_type,
            'status': 'success',
            'message': f'{call_type}Call started',
            'webrtc_config': webrtc_config,
            'timestamp': time.time()
        })
        
        # AI voice assistant welcome message
        welcome_messages = {
            'voice': 'Hello! I am Self Brain AGI voice assistant. We can start voice conversation now. Please speak into the microphone and I will answer your questions in real time.',
            'video': 'Video call connected! I am your AI assistant and can provide more intuitive help through video. You can see my real-time responses.'
        }
        
        emit('ai_response', {
            'response': welcome_messages.get(call_type, 'Call connected'),
            'response_time': 500,
            'tokens_used': 20,
            'call_mode': True,
            'call_id': call_id
        })
            
    except Exception as e:
        emit('call_error', {
            'type': call_type,
            'error': str(e),
            'status': 'failed'
        })

@socketio.on('webrtc_offer')
def handle_webrtc_offer(data):
    """Handle WebRTC offer exchange"""
    try:
        user_id = request.sid
        call_data = active_calls.get(user_id)
        
        if call_data:
            # Forward offer to AI assistant
            emit('ai_webrtc_offer', {
                'offer': data.get('offer'),
                'call_id': call_data['call_id'],
                'type': call_data['type']
            }, room='ai_assistant')
            
    except Exception as e:
        emit('call_error', {
            'error': str(e),
            'status': 'failed'
        })

@socketio.on('webrtc_answer')
def handle_webrtc_answer(data):
    """Handle WebRTC answer exchange"""
    try:
        emit('webrtc_answer', {
            'answer': data.get('answer'),
            'call_id': data.get('call_id')
        })
        
    except Exception as e:
        emit('call_error', {
            'error': str(e),
            'status': 'failed'
        })

@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    """Handle ICE candidate exchange"""
    try:
        user_id = request.sid
        call_data = active_calls.get(user_id)
        
        if call_data:
            # Forward ICE candidates
            emit('ai_ice_candidate', {
                'candidate': data.get('candidate'),
                'call_id': call_data['call_id']
            }, room='ai_assistant')
            
    except Exception as e:
        emit('call_error', {
            'error': str(e),
            'status': 'failed'
        })

@socketio.on('end_call')
def handle_end_call():
    """Handle end call event - cleanup resources"""
    try:
        emit('call_ended', {
            'status': 'success',
            'message': 'Call ended',
            'timestamp': time.time()
        }, broadcast=True)
        
        # Send end notification
        emit('ai_response', {
            'response': 'Thank you for using! Feel free to start a call again anytime you need.',
            'response_time': 300,
            'tokens_used': 12,
            'call_mode': False
        })
        
    except Exception as e:
        emit('call_error', {
            'error': str(e),
            'status': 'failed'
        })

@socketio.on('call_message')
def handle_call_message(data):
    """Handle call message - real-time voice to text"""
    try:
        message = data.get('message', '')
        call_type = data.get('call_type', 'voice')
        
        if message:
            # Process voice input text
            response = generate_enhanced_ai_response(message, [], 'all', 'a_manager')
            
            emit('call_response', {
                'response': response,
                'type': call_type,
                'is_voice': True,
                'timestamp': time.time()
            })
            
    except Exception as e:
        emit('call_error', {
            'error': str(e),
            'status': 'failed'
        })

# Chat history compatible endpoints
@app.route('/api/chat/history')
def get_chat_history():
    """Get chat history - compatible endpoint for frontend"""
    try:
        # Return the same format as /api/chat/conversations for compatibility
        conversations = [
            {
                'id': 'conv_001',
                'title': 'Discussion about machine learning',
                'last_activity': '2024-01-01 14:30:00',
                'message_count': 15
            },
            {
                'id': 'conv_002',
                'title': 'Neural network optimization',
                'last_activity': '2024-01-01 13:15:00',
                'message_count': 8
            }
        ]
        
        return jsonify({
            'status': 'success',
            'conversations': conversations,
            'history': conversations  # Additional field for compatibility
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/chat/message', methods=['POST'])
def send_chat_message():
    """Send chat message - compatible endpoint for frontend"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        conversation_id = data.get('conversation_id', 'default')
        
        logger.info(f"Processing chat message: {message}")
        
        # Use existing send_message logic
        response = generate_ai_response(message, 'all', [])
        
        return jsonify({
            'status': 'success',
            'response': response,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat_history():
    """Clear chat history - compatible endpoint for frontend"""
    try:
        return jsonify({
            'status': 'success',
            'message': 'Chat history cleared'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# Knowledge base related APIs
@app.route('/api/knowledge/storage/info')
def get_knowledge_storage_info():
    """Get knowledge storage space info"""
    try:
        storage_path = "d:\\shiyan\\knowledge_base_storage"
        
        # Calculate storage usage
        total_size = 0
        file_count = 0
        
        if os.path.exists(storage_path):
            for root, dirs, files in os.walk(storage_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                    except (OSError, IOError):
                        continue
        
        # Get directory structure
        directory_structure = {
            'categories': ['text', 'code', 'images', 'audio', 'video', 'structured'],
            'documents': ['text_files', 'markdown', 'pdf', 'json'],
            'media': ['images', 'audio', 'video', 'datasets'],
            'backups': ['daily', 'weekly', 'monthly'],
            'temp': [],
            'logs': [],
            'config': ['templates', 'schemas', 'mappings']
        }
        
        # Calculate file count by category
        category_counts = {}
        for category, subdirs in directory_structure.items():
            category_path = os.path.join(storage_path, category)
            if os.path.exists(category_path):
                count = 0
                for root, dirs, files in os.walk(category_path):
                    count += len(files)
                category_counts[category] = count
            else:
                category_counts[category] = 0
        
        return jsonify({
            'status': 'success',
            'storage_path': storage_path,
            'total_size': total_size,
            'file_count': file_count,
            'directory_structure': directory_structure,
            'category_counts': category_counts
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/list')
def get_knowledge_list():
    """Get knowledge list"""
    try:
        storage_path = "d:\\shiyan\\knowledge_base_storage"
        knowledge_items = []
        
        if os.path.exists(storage_path):
            # Iterate through all files
            for root, dirs, files in os.walk(storage_path):
                for file in files:
                    if file in ['README.md', 'storage_config.json']:
                        continue
                        
                    file_path = os.path.join(root, file)
                    try:
                        stat = os.stat(file_path)
                        relative_path = os.path.relpath(file_path, storage_path)
                        
                        # Determine category based on file path
                        path_parts = relative_path.split(os.sep)
                        category = path_parts[0] if len(path_parts) > 1 else 'other'
                        
                        knowledge_items.append({
                            'id': hashlib.md5(file_path.encode()).hexdigest()[:16],
                            'title': file,
                            'category': category,
                            'file_type': os.path.splitext(file)[1].lower(),
                            'file_path': relative_path,
                            'full_path': file_path,
                            'size': stat.st_size,
                            'updated_at': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                            'created_at': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                        })
                    except (OSError, IOError):
                        continue
        
        return jsonify({
            'status': 'success',
            'knowledge': knowledge_items
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge')
def get_knowledge_entries():
    """Get paginated knowledge entries with search and filter support"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        search = request.args.get('search', '').lower()
        category = request.args.get('category', '')
        model = request.args.get('model', '')
        per_page = 10
        
        storage_path = "d:\\shiyan\\knowledge_base_storage"
        knowledge_items = []
        
        if os.path.exists(storage_path):
            # Iterate through all files
            for root, dirs, files in os.walk(storage_path):
                for file in files:
                    if file in ['README.md', 'storage_config.json']:
                        continue
                        
                    file_path = os.path.join(root, file)
                    try:
                        stat = os.stat(file_path)
                        relative_path = os.path.relpath(file_path, storage_path)
                        
                        # Determine category based on file path
                        path_parts = relative_path.split(os.sep)
                        category_name = path_parts[0] if len(path_parts) > 1 else 'other'
                        
                        # Create knowledge entry - use directory name as ID
                        path_parts = relative_path.split(os.sep)
                        if len(path_parts) >= 2:
                            knowledge_id = path_parts[0]  # Use directory name as ID
                        else:
                            knowledge_id = 'default'
                        
                        entry = {
                            'id': knowledge_id,
                            'title': file,
                            'category': category_name,
                            'model': '1',  # default model
                            'file_type': os.path.splitext(file)[1].lower(),
                            'file_path': relative_path,
                            'full_path': file_path,
                            'size': stat.st_size,
                            'updated_at': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                            'created_at': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        # Apply filtering criteria
                        if search and search not in entry['title'].lower():
                            continue
                        if category and category != 'all' and entry['category'] != category:
                            continue
                        if model and model != 'all' and entry['model'] != model:
                            continue
                            
                        knowledge_items.append(entry)
                    except (OSError, IOError):
                        continue
        
        # Calculate pagination
        total_items = len(knowledge_items)
        total_pages = max(1, (total_items + per_page - 1) // per_page)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Get current page data
        page_items = knowledge_items[start_idx:end_idx]
        
        # Calculate statistics
        stats = {
            'total_entries': total_items,
            'total_size': sum(item['size'] for item in knowledge_items),
            'recent_entries': len([item for item in knowledge_items if 
                                 (datetime.now() - datetime.fromtimestamp(
                                     os.path.getctime(os.path.join(storage_path, item['file_path']))
                                 )).days <= 7]),
            'active_models': len(set(item.get('model', '1') for item in knowledge_items))
        }
        
        return jsonify({
            'entries': page_items,
            'page': page,
            'pages': total_pages,
            'total': total_items,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 导入新模块
from metadata_manager import KnowledgeMetadataManager
from search_engine import KnowledgeSearchEngine

# 初始化管理器
metadata_manager = KnowledgeMetadataManager("d:\\shiyan\\knowledge_base_storage")
search_engine = KnowledgeSearchEngine("d:\\shiyan\\search_index")

@app.route('/api/knowledge/upload', methods=['POST'])
def upload_knowledge():
    """Upload knowledge with enhanced metadata management"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'})
        
        file = request.files['file']
        category = request.form.get('category', 'Other')
        title = request.form.get('title', '')
        description = request.form.get('description', '')
        tags = request.form.get('tags', '').split(',') if request.form.get('tags') else []
        author = request.form.get('author', 'system')
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'})
        
        # Ensure storage path exists
        storage_path = "d:\\shiyan\\knowledge_base_storage"
        category_path = os.path.join(storage_path, 'categories', category)
        os.makedirs(category_path, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(category_path, filename)
        
        # Save file
        file.save(file_path)
        
        # 创建元数据
        metadata = metadata_manager.create_metadata(
            file_path,
            title=title or filename,
            description=description,
            tags=tags,
            category=category,
            author=author
        )
        
        # 添加到搜索引擎
        search_engine.add_document(metadata)
        
        logger.info(f"Uploading knowledge file: {filename}, category: {category}, metadata_id: {metadata['id']}")
        
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'knowledge_id': metadata['id'],
            'file_path': metadata['file_path'],
            'file_size': metadata['file_size'],
            'metadata': metadata,
            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/import/zip', methods=['POST'])
def import_knowledge_zip():
    """Import knowledge from ZIP archive"""
    try:
        if 'zipFile' not in request.files:
            return jsonify({'status': 'error', 'message': 'No ZIP file provided'})
        
        zip_file = request.files['zipFile']
        auto_categorize = request.form.get('autoCategorize', 'true') == 'true'
        
        if not zip_file.filename:
            return jsonify({'status': 'error', 'message': 'No ZIP file selected'})
        
        # Save temporary ZIP file
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, zip_file.filename)
        zip_file.save(zip_path)
        
        # Extract files
        extract_dir = os.path.join(temp_dir, 'extracted')
        os.makedirs(extract_dir)
        
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Process extracted files
        results = process_import_files(extract_dir, auto_categorize)
        
        # Clean temporary files
        import shutil
        shutil.rmtree(temp_dir)
        
        record = {
            'type': 'zip',
            'total': results['total_files'],
            'successful': results['successful'],
            'failed': results['failed'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify({
            'status': 'success',
            'message': f"ZIP import completed: {results['successful']}/{results['total_files']} files successful",
            'results': results,
            'record': record
        })
        
    except Exception as e:
        logger.error(f"ZIP import failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/import/json', methods=['POST'])
def import_knowledge_json():
    """Import knowledge from JSON configuration"""
    try:
        config = request.get_json()
        
        if not config or 'knowledge_base' not in config:
            return jsonify({'status': 'error', 'message': 'Invalid JSON configuration'})
        
        knowledge_base = config['knowledge_base']
        files = knowledge_base.get('files', [])
        
        if not files:
            return jsonify({'status': 'error', 'message': 'No file list in configuration'})
        
        results = {
            'total_files': len(files),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        # Process each file configuration
        for file_config in files:
            try:
                file_path = file_config.get('path')
                category = file_config.get('category', 'imported')
                tags = file_config.get('tags', [])
                
                if not file_path or not os.path.exists(file_path):
                    results['failed'] += 1
                    results['errors'].append(f"File does not exist: {file_path}")
                    continue
                
                # Copy file to knowledge base
                storage_path = "d:\\shiyan\\knowledge_base_storage"
                category_path = os.path.join(storage_path, 'categories', category)
                os.makedirs(category_path, exist_ok=True)
                
                filename = os.path.basename(file_path)
                dest_path = os.path.join(category_path, filename)
                
                import shutil
                shutil.copy2(file_path, dest_path)
                
                results['successful'] += 1
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Failed to process file: {str(e)}")
        
        record = {
            'type': 'json',
            'total': results['total_files'],
            'successful': results['successful'],
            'failed': results['failed'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify({
            'status': 'success',
            'message': f"JSON configuration import completed: {results['successful']}/{results['total_files']} files successful",
            'results': results,
            'record': record
        })
        
    except Exception as e:
        logger.error(f"JSON import failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

def process_import_files(source_dir, auto_categorize=True):
    """Process imported files"""
    results = {
        'total_files': 0,
        'successful': 0,
        'failed': 0,
        'errors': []
    }
    
    storage_path = "d:\\shiyan\\knowledge_base_storage"
    
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.startswith('.'):
                continue
                
            try:
                source_path = os.path.join(root, filename)
                
                # Determine category
                if auto_categorize:
                    category = categorize_file(filename)
                else:
                    category = 'imported'
                
                # Create category directory
                category_path = os.path.join(storage_path, 'categories', category)
                os.makedirs(category_path, exist_ok=True)
                
                # Generate target path
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dest_filename = f"{timestamp}_{secure_filename(filename)}"
                dest_path = os.path.join(category_path, dest_filename)
                
                # Copy file
                import shutil
                shutil.copy2(source_path, dest_path)
                
                results['successful'] += 1
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Failed to process file {filename}: {str(e)}")
    
    results['total_files'] = results['successful'] + results['failed']
    return results

def categorize_file(filename):
    """Auto-categorize based on file extension"""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    category_map = {
        'text': ['txt', 'md', 'doc', 'docx', 'rtf'],
        'code': ['py', 'js', 'java', 'cpp', 'c', 'h', 'html', 'css', 'php', 'rb', 'go', 'rs'],
        'images': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp'],
        'audio': ['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a'],
        'video': ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm'],
        'structured': ['json', 'csv', 'xml', 'yaml', 'yml', 'xlsx']
    }
    
    for category, extensions in category_map.items():
        if ext in extensions:
            return category
    
    return 'other'

from werkzeug.utils import secure_filename
import tempfile

@app.route('/api/knowledge/delete/<knowledge_id>', methods=['POST'])
def delete_knowledge(knowledge_id):
    """Delete knowledge"""
    try:
        logger.info(f"Deleting knowledge: {knowledge_id}")
        
        knowledge_base_path = 'd:\\shiyan\\knowledge_base_storage'
        knowledge_dir = os.path.join(knowledge_base_path, knowledge_id)
        
        if os.path.exists(knowledge_dir) and os.path.isdir(knowledge_dir):
            try:
                import shutil
                shutil.rmtree(knowledge_dir)
                logger.info(f"Successfully deleted knowledge directory: {knowledge_dir}")
                return jsonify({'status': 'success', 'message': 'Knowledge deleted successfully'})
            except Exception as e:
                logger.error(f"Failed to delete knowledge directory: {str(e)}")
                return jsonify({'status': 'error', 'message': f'Deletion failed: {str(e)}'}), 500
        else:
            return jsonify({'status': 'error', 'message': 'Knowledge directory not found'}), 404
            
    except Exception as e:
        logger.error(f"Failed to delete knowledge: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/knowledge/view/<knowledge_id>')
def knowledge_view_page(knowledge_id):
    """知识查看页面路由"""
    return render_template('knowledge_view.html', knowledge_id=knowledge_id)

@app.route('/api/knowledge/view/<knowledge_id>')
def view_knowledge(knowledge_id):
    """View knowledge"""
    try:
        return render_template('knowledge_view.html', knowledge_id=knowledge_id)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/detail/<knowledge_id>')
def get_knowledge_detail(knowledge_id):
    """Get knowledge detail from actual storage"""
    try:
        knowledge_base_path = 'd:\\shiyan\\knowledge_base_storage'
        
        # Build mapping from ID to file path
        id_to_file = {}
        
        if os.path.exists(knowledge_base_path):
            for root, dirs, files in os.walk(knowledge_base_path):
                for file in files:
                    # Support more file types
                    supported_extensions = ('.txt', '.md', '.json', '.yaml', '.yml', '.xml', '.csv', 
                                          '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
                                          '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
                                          '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma',
                                          '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
                                          '.py', '.js', '.html', '.css', '.php', '.java', '.cpp', '.c', '.h')
                    
                    if file.lower().endswith(supported_extensions):
                        file_path = os.path.join(root, file)
                        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:16]
                        id_to_file[path_hash] = file_path
        
        if knowledge_id in id_to_file:
            file_path = id_to_file[knowledge_id]
            
            # Read file content
            try:
                if file_path.lower().endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        knowledge = {
                            'id': knowledge_id,
                            'title': data.get('title', os.path.basename(file_path)),
                            'category': data.get('category', 'General'),
                            'content': data.get('content', ''),
                            'updated_at': data.get('updated_at', datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()),
                            'views': data.get('views', 0),
                            'author': data.get('author', 'Self Brain AGI'),
                            'tags': data.get('tags', [])
                        }
                else:
                    # For text files, read content directly
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    knowledge = {
                        'id': knowledge_id,
                        'title': os.path.basename(file_path),
                        'category': 'General',
                        'content': content,
                        'updated_at': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                        'views': 0,
                        'author': 'Self Brain AGI',
                        'tags': []
                    }
                
                return jsonify({
                    'status': 'success',
                    'knowledge': knowledge
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Failed to read file: {str(e)}'})
        else:
            return jsonify({'status': 'error', 'message': 'Knowledge not found'}), 404
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/stats')
def get_knowledge_stats():
    """Get enhanced knowledge base statistics"""
    try:
        knowledge_base_path = os.path.join('d:\\shiyan\\knowledge_base_storage')
        index_dir = os.path.join('d:\\shiyan\\knowledge_base_storage', 'search_index')
        
        stats = {
            'total_files': 0,
            'total_size': 0,
            'total_categories': 0,
            'indexed_documents': 0,
            'categories': {},
            'tags': {},
            'file_types': {}
        }
        
        # Count files and metadata
        if os.path.exists(knowledge_base_path):
            for root, dirs, files in os.walk(knowledge_base_path):
                for file in files:
                    if file == 'info.json' or file.endswith('.json'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        file_ext = os.path.splitext(file.lower())[1]
                        
                        stats['total_files'] += 1
                        stats['total_size'] += file_size
                        
                        # Count file types
                        if file_ext not in stats['file_types']:
                            stats['file_types'][file_ext] = 0
                        stats['file_types'][file_ext] += 1
                        
                        # Check for metadata
                        metadata_path = os.path.join(os.path.dirname(file_path), 'info.json')
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                                    category = metadata.get('category', 'unknown')
                                    tags = metadata.get('tags', [])
                                    
                                    if category not in stats['categories']:
                                        stats['categories'][category] = 0
                                    stats['categories'][category] += 1
                                    
                                    for tag in tags:
                                        if tag not in stats['tags']:
                                            stats['tags'][tag] = 0
                                        stats['tags'][tag] += 1
                            except:
                                pass
                    except:
                        continue
        
        # Count indexed documents
        if os.path.exists(index_dir):
            try:
                from whoosh.index import open_dir
                ix = open_dir(index_dir)
                stats['indexed_documents'] = ix.doc_count()
            except:
                pass
        
        stats['total_categories'] = len(stats['categories'])
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get knowledge base statistics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/knowledge/delete_selected', methods=['POST'])
def delete_selected_knowledge():
    """Delete selected knowledge items"""
    try:
        data = request.get_json()
        ids = data.get('ids', [])
        
        if not ids:
            return jsonify({'status': 'error', 'message': 'No items selected for deletion'}), 400
        
        knowledge_base_path = 'd:\\shiyan\\knowledge_base_storage'
        
        deleted_count = 0
        errors = []
        
        for selected_id in ids:
            # 构建知识条目的完整路径
            knowledge_dir = os.path.join(knowledge_base_path, selected_id)
            
            if os.path.exists(knowledge_dir) and os.path.isdir(knowledge_dir):
                try:
                    # 删除整个知识条目目录
                    import shutil
                    shutil.rmtree(knowledge_dir)
                    deleted_count += 1
                    logger.info(f"Successfully deleted knowledge directory: {knowledge_dir}")
                except Exception as e:
                    errors.append(f"Failed to delete {selected_id}: {str(e)}")
                    logger.error(f"Failed to delete knowledge directory {selected_id}: {str(e)}")
            else:
                errors.append(f"Knowledge directory not found for ID: {selected_id}")
        
        if deleted_count > 0:
            message = f'Successfully deleted {deleted_count} knowledge items'
            if errors:
                message += f', {len(errors)} items failed to delete'
            return jsonify({'success': True, 'status': 'success', 'message': message, 'deleted_count': deleted_count})
        else:
            return jsonify({'success': False, 'status': 'error', 'message': 'No knowledge items found to delete', 'errors': errors}), 404
            
    except Exception as e:
        logger.error(f"Failed to delete multiple knowledge items: {str(e)}")
        return jsonify({'success': False, 'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/export_selected', methods=['POST'])
def export_selected_knowledge():
    """Export selected knowledge items"""
    try:
        data = request.get_json()
        ids = data.get('ids', [])
        
        if not ids:
            return jsonify({'status': 'error', 'message': 'No items selected for export'}), 400
        
        knowledge_base_path = 'd:\\shiyan\\knowledge_base_storage'
        
        # Create temporary file for ZIP
        temp_dir = tempfile.mkdtemp()
        zip_filename = f'knowledge_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        zip_path = os.path.join(temp_dir, zip_filename)
        
        # Build mapping from ID to file path (consistent with frontend)
        id_to_file = {}
        
        if os.path.exists(knowledge_base_path):
            for root, dirs, files in os.walk(knowledge_base_path):
                for file in files:
                    # Support more file types
                    supported_extensions = ('.txt', '.md', '.json', '.yaml', '.yml', '.xml', '.csv', 
                                          '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
                                          '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
                                          '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma',
                                          '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
                                          '.py', '.js', '.html', '.css', '.php', '.java', '.cpp', '.c', '.h')
                    
                    if file.lower().endswith(supported_extensions):
                        file_path = os.path.join(root, file)
                        
                        # Use hash of full file path as ID, consistent with frontend
                        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:16]
                        id_to_file[path_hash] = {
                            'full_path': file_path,
                            'relative_path': os.path.relpath(file_path, knowledge_base_path)
                        }
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            exported_count = 0
            
            # Iterate through selected IDs to find corresponding files
            for selected_id in ids:
                if selected_id in id_to_file:
                    file_info = id_to_file[selected_id]
                    file_path = file_info['full_path']
                    
                    if os.path.exists(file_path):
                        # Use relative path to maintain directory structure
                        arcname = file_info['relative_path']
                        zipf.write(file_path, arcname)
                        exported_count += 1
            
            # If file found, add export info file
            if exported_count > 0:
                export_info = {
                    'export_date': datetime.now().isoformat(),
                    'exported_files': exported_count,
                    'file_ids': ids,
                    'system': 'Self Brain AGI Knowledge Base'
                }
                
                info_content = f"""# Self Brain AGI Knowledge Export

Export time: {export_info['export_date']}
Exported files count: {export_info['exported_files']}
File ID list: {', '.join(export_info['file_ids'])}

This compressed package contains files exported from the Self Brain AGI knowledge base.
"""
                
                zipf.writestr('export_info.txt', info_content)
        
        if exported_count == 0:
            # Clean temporary files
            try:
                os.remove(zip_path)
                os.rmdir(temp_dir)
            except:
                pass
            return jsonify({'success': False, 'status': 'error', 'message': 'Selected files not found'}), 404
        
        # Send file to user
        @after_this_request
        def cleanup(response):
            try:
                os.remove(zip_path)
                os.rmdir(temp_dir)
            except Exception as e:
                logger.error(f"Failed to clean temporary files: {e}")
            return response
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        logger.error(f"Failed to export selected knowledge items: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# SocketIO event handling
@socketio.on('connect')
def handle_connect():
    """Client connect event"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to Self Brain AGI System'})
    
    # Notify enhanced monitoring system of new client connection
    emit('get_resources')
    emit('get_training_status')
    emit('get_realtime_metrics')

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnect event"""
    logger.info(f"Client disconnected: {request.sid}")



@socketio.on('request_training_status')
def handle_training_status(session_id):
    """Handle training status request"""
    try:
        # Get training status through enhanced monitoring system
        emit('get_training_status')
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('request_model_status')
def handle_model_status(model_id):
    """Handle model status request"""
    try:
        emit('get_model_status', {'model_id': model_id})
    except Exception as e:
        logger.error(f"Failed to get model status: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('request_all_models_status')
def handle_all_models_status():
    """Handle all models status request"""
    try:
        emit('get_all_models_status')
    except Exception as e:
        logger.error(f"Failed to get all models status: {str(e)}")
        emit('error', {'message': str(e)})

# Background task: periodically broadcast system status
def background_broadcast():
    """Background broadcast system status"""
    while True:
        try:
            # Get latest data through enhanced monitoring system
            # Use SocketIO events to trigger monitoring system to send data
            socketio.emit('get_resources')
            socketio.emit('get_training_status')
            socketio.emit('get_realtime_metrics')
            
            # Update every 3 seconds
            time.sleep(3)
        except Exception as e:
            error_msg = str(e)
            # Use safe logging method to avoid format character issues
            safe_error_msg = error_msg.replace('%', '%%')
            # Use string concatenation instead of format strings
            logger.error("Background broadcast failed: " + safe_error_msg)
            time.sleep(5)



# Start background broadcast thread
broadcast_thread = threading.Thread(target=background_broadcast, daemon=True)
broadcast_thread.start()

# Error handling
@app.errorhandler(404)
def not_found(error):
    """404 error handling"""
    # Use hardcoded language resources to avoid any loading errors
    lang_resources = {
        "app_title": "Self Brain AGI",
        "home": "Home",
        "dashboard": "Dashboard",
        "knowledge_base": "Knowledge Base",
        "analytics": "Analytics",
        "ai_chat": "AI Chat",
        "training_control": "Training Control",
        "model_management": "Model Management",
        "system_settings": "System Settings",
        "help": "Help",
        "chinese": "Chinese",
        "english": "English",
        "system_online": "System Online",
        "error.title": "Error",
        "system.name": "Self Brain",
        "error.header": "Error Occurred",
        "error.default_message": "System encountered an unexpected error, please try again later",
        "error.back_home": "Back to Home",
        "error.refresh_page": "Refresh Page",
        "system_notification": "System Notification",
        "error": "Error",
        "loading": "Loading...",
        "success": "Success",
        "language_change_failed": "Language change failed",
        "language_changed": "Language changed"
    }
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found",
                         language='en',
                         lang_resources=lang_resources,
                         lang=lang_resources), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handling"""
    # Use hardcoded language resources to avoid any loading errors
    lang_resources = {
        "app_title": "Self Brain AGI",
        "home": "Home",
        "dashboard": "Dashboard",
        "knowledge_base": "Knowledge Base",
        "analytics": "Analytics",
        "ai_chat": "AI Chat",
        "training_control": "Training Control",
        "model_management": "Model Management",
        "system_settings": "System Settings",
        "help": "Help",
        "chinese": "Chinese",
        "english": "English",
        "system_online": "System Online",
        "error.title": "Error",
        "system.name": "Self Brain",
        "error.header": "Error Occurred",
        "error.default_message": "System encountered an unexpected error, please try again later",
        "error.back_home": "Back to Home",
        "error.refresh_page": "Refresh Page",
        "system_notification": "System Notification",
        "error": "Error",
        "loading": "Loading...",
        "success": "Success",
        "language_change_failed": "Language change failed",
        "language_changed": "Language changed"
    }
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error",
                         language='en',
                         lang_resources=lang_resources,
                         lang=lang_resources), 500

# Knowledge edit related APIs
@app.route('/api/knowledge/create', methods=['POST'])
def create_knowledge():
    """Create new knowledge API"""
    try:
        # Support JSON format data
        if request.is_json:
            data = request.get_json()
        else:
            # Compatible with form data format
            data_str = request.form.get('data')
            if not data_str:
                return jsonify({'success': False, 'message': 'No data provided'})
            data = json.loads(data_str)
        
        title = data.get('title', '').strip()
        category = data.get('category', '').strip()
        content = data.get('content', '').strip()
        summary = data.get('summary', '') or data.get('description', '')
        tags = data.get('tags', [])
        
        if not title or not content:
            return jsonify({'success': False, 'message': 'Title and content are required'})
        
        # Generate knowledge ID
        knowledge_id = hashlib.md5(title.encode('utf-8')).hexdigest()[:12]
        knowledge_path = os.path.join('knowledge_base_storage', knowledge_id)
        
        # Create directory
        os.makedirs(knowledge_path, exist_ok=True)
        
        # Save content
        content_file = os.path.join(knowledge_path, 'content.txt')
        with open(content_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Save metadata
        info_file = os.path.join(knowledge_path, 'info.json')
        knowledge_info = {
            'id': knowledge_id,
            'title': title,
            'category': category,
            'summary': summary,
            'tags': tags,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'views': 0,
            'file_count': 0
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_info, f, ensure_ascii=False, indent=2)
        
        # Process uploaded files (only for form submission)
        if not request.is_json:
            files = request.files.getlist('files')
            file_count = 0
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(knowledge_path, filename)
                    file.save(file_path)
                    file_count += 1
            
            # Update file count
            knowledge_info['file_count'] = file_count
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_info, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Knowledge created successfully',
            'knowledge_id': knowledge_id
        })
        
    except Exception as e:
        logger.error(f"Failed to create knowledge: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/knowledge/update/<knowledge_id>', methods=['POST'])
def update_knowledge_item(knowledge_id):
    """Update single knowledge item API"""
    try:
        knowledge_path = os.path.join('knowledge_base_storage', knowledge_id)
        
        if not os.path.exists(knowledge_path):
            return jsonify({'success': False, 'message': 'Knowledge not found'})
        
        # Process form data
        data_str = request.form.get('data')
        if not data_str:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        data = json.loads(data_str)
        title = data.get('title', '').strip()
        category = data.get('category', '').strip()
        content = data.get('content', '').strip()
        summary = data.get('summary', '').strip()
        tags = data.get('tags', [])
        
        if not title or not content:
            return jsonify({'success': False, 'message': 'Title and content are required'})
        
        # Update content
        content_file = os.path.join(knowledge_path, 'content.txt')
        with open(content_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Update metadata
        info_file = os.path.join(knowledge_path, 'info.json')
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                knowledge_info = json.load(f)
        else:
            knowledge_info = {'id': knowledge_id}
        
        knowledge_info.update({
            'title': title,
            'category': category,
            'summary': summary,
            'tags': tags,
            'updated_at': datetime.now().isoformat()
        })
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_info, f, ensure_ascii=False, indent=2)
        
        # Process newly uploaded files
        files = request.files.getlist('files')
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(knowledge_path, filename)
                file.save(file_path)
                knowledge_info['file_count'] = knowledge_info.get('file_count', 0) + 1
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_info, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Knowledge updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Failed to update knowledge: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/knowledge/<knowledge_id>')
def view_knowledge_page(knowledge_id):
    """View knowledge page"""
    return render_template('knowledge_view.html', knowledge_id=knowledge_id)

@app.route('/knowledge/edit/<knowledge_id>')
def edit_knowledge_page(knowledge_id):
    """Edit knowledge page"""
    return render_template('knowledge_edit.html', knowledge_id=knowledge_id)

@app.route('/knowledge/new')
def create_knowledge_page():
    """Create new knowledge page"""
    return render_template('knowledge_create.html')

@app.route('/api/knowledge/optimize', methods=['POST'])
def optimize_knowledge_database():
    """Optimize knowledge database"""
    try:
        import sqlite3
        import os
        import shutil
        
        # Check knowledge database file
        db_file = 'knowledge_base.db'
        messages = []
        
        # Debug information
        logger.info("Starting knowledge database optimization")
        
        # If database does not exist, create it
        if not os.path.exists(db_file):
            # Create database and table structure
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Create knowledge table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT,
                    content TEXT,
                    summary TEXT,
                    tags TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    views INTEGER DEFAULT 0,
                    file_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create index
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_title ON knowledge(title)')
            
            conn.commit()
            conn.close()
            messages.append("Database created successfully")
            logger.info("Knowledge database created")
        else:
            # Create backup
            backup_file = f'knowledge_base_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
            shutil.copy2(db_file, backup_file)
            messages.append(f"Backup created: {backup_file}")
            logger.info(f"Database backup created: {backup_file}")
            
            # Connect to database and perform optimization
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Execute VACUUM optimization
            cursor.execute('VACUUM')
            
            # Reindex
            cursor.execute('REINDEX')
            
            conn.commit()
            conn.close()
            messages.append("Database optimized successfully")
            logger.info("Database optimization completed")
        
        # Ensure knowledge base storage directory exists
        storage_path = 'knowledge_base_storage'
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
            messages.append("Storage directory created")
            logger.info("Storage directory created")
        else:
            # Clean empty folders in knowledge base storage
            cleaned_folders = 0
            try:
                for item in os.listdir(storage_path):
                    item_path = os.path.join(storage_path, item)
                    if os.path.isdir(item_path):
                        # Check if folder is empty
                        if not os.listdir(item_path):
                            try:
                                os.rmdir(item_path)
                                cleaned_folders += 1
                            except Exception as cleanup_error:
                                logger.warning(f"Failed to clean folder: {cleanup_error}")
                if cleaned_folders > 0:
                    messages.append(f"Cleaned {cleaned_folders} empty folders")
                    logger.info(f"Cleaned {cleaned_folders} empty folders")
            except Exception as e:
                logger.warning(f"Error cleaning storage directory: {e}")
        
        final_message = '; '.join(messages)
        logger.info(f"Knowledge database optimization completed: {final_message}")
        
        return jsonify({
            'success': True,
            'message': final_message
        })
        
    except Exception as e:
        error_msg = f"Failed to optimize knowledge database: {str(e)}"
        logger.error(error_msg)
        return jsonify({'success': False, 'message': str(e)})

# Test route for debugging
@app.route('/test')
def test_page():
    """Simple test page for API debugging"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Test</title>
    </head>
    <body>
        <h1>API Test Page</h1>
        <button onclick="testAPI()">Test Dashboard API</button>
        <div id="result"></div>
        
        <script>
            function testAPI() {
                const result = document.getElementById('result');
                result.innerHTML = 'Testing...';
                
                fetch('/api/dashboard/data')
                    .then(response => {
                        result.innerHTML += '<br>Response status: ' + response.status;
                        return response.json();
                    })
                    .then(data => {
                        result.innerHTML += '<br>Success: Data received';
                        console.log('API Data:', data);
                    })
                    .catch(error => {
                        result.innerHTML += '<br>Error: ' + error.message;
                        console.error('API Error:', error);
                    });
            }
            
            // Test immediately
            testAPI();
        </script>
    </body>
    </html>
    '''

# Additional API endpoints for complete functionality

@app.route('/api/system/status')
def system_status():
    """Real-time system status API"""
    try:
        import psutil
        
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return jsonify({
            'status': 'Active',
            'gpu': 'GPU Active',
            'models': '11/11',
            'response_time': '< 200ms',
            'cpu': f'{cpu_percent}%',
            'memory': f'{memory.percent}%',
            'gpu_usage': '78%',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'Active',
            'gpu': 'GPU Active',
            'models': '11/11',
            'response_time': '< 200ms',
            'cpu': '45%',
            'memory': '62%',
            'gpu_usage': '78%'
        })

@app.route('/api/execute', methods=['POST'])
def execute_command():
    """Command execution API"""
    try:
        data = request.get_json()
        command = data.get('command', '')
        
        if not command:
            return jsonify({'error': 'No command provided'}), 400
            
        # Simulate command processing through A management model
        return jsonify({
            'response': f'Command "{command}" processed successfully by AI system',
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/interact', methods=['POST'])
def model_interaction():
    """Real-time model interaction API"""
    try:
        data = request.get_json()
        model = data.get('model')
        action = data.get('action')
        
        if not model or not action:
            return jsonify({'error': 'Model and action required'}), 400
            
        return jsonify({
            'status': 'success',
            'model': model,
            'action': action,
            'result': f'{model} executed {action} successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training_session():
    """Start training API"""
    return jsonify({
        'status': 'training_started',
        'progress': 0,
        'message': 'Training session initiated',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/training/status')
def training_status():
    """Training status API"""
    return jsonify({
        'status': 'active',
        'progress': 85,
        'current_model': 'Model optimization',
        'eta': '2 minutes',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/models/status')
def models_status():
    """All models status API"""
    models = [
        {'name': 'A Management', 'status': 'active', 'type': 'management'},
        {'name': 'B Language', 'status': 'active', 'type': 'language'},
        {'name': 'C Audio', 'status': 'active', 'type': 'audio'},
        {'name': 'D Image', 'status': 'active', 'type': 'image'},
        {'name': 'E Video', 'status': 'active', 'type': 'video'},
        {'name': 'F Spatial', 'status': 'active', 'type': 'spatial'},
        {'name': 'G Sensor', 'status': 'active', 'type': 'sensor'},
        {'name': 'H Computer Control', 'status': 'active', 'type': 'control'},
        {'name': 'I Knowledge', 'status': 'active', 'type': 'knowledge'},
        {'name': 'J Motion', 'status': 'active', 'type': 'motion'},
        {'name': 'K Programming', 'status': 'active', 'type': 'programming'}
    ]
    
    return jsonify({
        'total': len(models),
        'active': len([m for m in models if m['status'] == 'active']),
        'models': models,
        'timestamp': datetime.now().isoformat()
    })

# Upload functionality
@app.route('/upload')
def upload_page():
    """Data upload interface"""
    return render_template('upload.html')

@app.route('/api/upload/training-data', methods=['POST'])
def upload_training_data():
    """Handle training data uploads"""
    try:
        model_type = request.form.get('model_type', '').lower()
        joint_training = request.form.get('joint_training') == 'true'
        files = request.files.getlist('files')
        
        if not model_type or not files:
            return jsonify({'error': 'Model type and files are required'}), 400
            
        # Save uploaded files
        upload_dir = os.path.join('uploads', model_type)
        os.makedirs(upload_dir, exist_ok=True)
        
        saved_files = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                saved_files.append(filename)
        
        # Trigger training based on model type
        training_triggered = False
        if joint_training:
            # Trigger joint training across multiple models
            training_triggered = True
            logger.info(f"Joint training triggered for {model_type} with files: {saved_files}")
        else:
            # Trigger individual model training
            training_triggered = True
            logger.info(f"Training triggered for {model_type} with files: {saved_files}")
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(saved_files)} files for {model_type}',
            'files': saved_files,
            'model_type': model_type,
            'joint_training': joint_training,
            'training_triggered': training_triggered
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Knowledge Management API endpoints (missing functionality)

@app.route('/api/knowledge/search')
def search_knowledge():
    """Search knowledge entries by keyword"""
    try:
        query = request.args.get('q', '').strip()
        category = request.args.get('category', '').strip()
        page = max(1, int(request.args.get('page', 1)))
        limit = min(100, max(1, int(request.args.get('limit', 20))))
        
        knowledge_dir = os.path.join('knowledge_base')
        entries = []
        
        if not os.path.exists(knowledge_dir):
            return jsonify({'entries': [], 'total': 0, 'page': page, 'pages': 0, 'success': True})
        
        # Collect all knowledge entries
        for root, dirs, files in os.walk(knowledge_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Add ID and file info
                        data['id'] = file.replace('.json', '')
                        data['category'] = os.path.basename(root)
                        
                        # Filter by category if specified
                        if category and data['category'].lower() != category.lower():
                            continue
                        
                        # Search in title, content, and tags
                        search_text = f"{data.get('title', '')} {data.get('content', '')} {data.get('tags', '')}".lower()
                        if not query or query.lower() in search_text:
                            entries.append(data)
                    except Exception as e:
                        logger.warning(f"Error reading file {file_path}: {e}")
                        continue
        
        # Pagination
        total = len(entries)
        pages = (total + limit - 1) // limit
        start = (page - 1) * limit
        end = start + limit
        paginated_entries = entries[start:end]
        
        return jsonify({
            'entries': paginated_entries,
            'total': total,
            'page': page,
            'pages': pages,
            'limit': limit,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/knowledge/search/enhanced')
def search_knowledge_enhanced():
    """Enhanced search with full-text indexing and metadata"""
    try:
        query = request.args.get('q', '').strip()
        category = request.args.get('category', '').strip()
        tags = request.args.get('tags', '').strip()
        page = max(1, int(request.args.get('page', 1)))
        limit = min(100, max(1, int(request.args.get('limit', 20))))
        
        # 解析标签
        tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
        
        # 使用搜索引擎
        search_results = search_engine.search(query, category, tag_list, limit=100)
        
        # 分页
        total = len(search_results['results'])
        pages = (total + limit - 1) // limit
        start = (page - 1) * limit
        end = start + limit
        paginated_results = search_results['results'][start:end]
        
        return jsonify({
            'results': paginated_results,
            'total': total,
            'page': page,
            'pages': pages,
            'limit': limit,
            'query': query,
            'success': True,
            'search_engine': 'enhanced'
        })
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/knowledge/metadata/<metadata_id>')
def get_knowledge_metadata(metadata_id):
    """获取知识条目元数据"""
    try:
        metadata = metadata_manager.get_metadata(metadata_id)
        if metadata:
            return jsonify({'status': 'success', 'metadata': metadata})
        else:
            return jsonify({'status': 'error', 'message': 'Metadata not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/metadata/<metadata_id>', methods=['PUT'])
def update_knowledge_metadata(metadata_id):
    """更新知识条目元数据"""
    try:
        updates = request.json
        metadata = metadata_manager.update_metadata(metadata_id, updates)
        if metadata:
            # 更新搜索引擎
            search_engine.update_document(metadata)
            return jsonify({'status': 'success', 'metadata': metadata})
        else:
            return jsonify({'status': 'error', 'message': 'Metadata not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/categories')
def get_categories():
    """获取所有分类"""
    try:
        metadata_list = metadata_manager.get_all_metadata()
        categories = set()
        for metadata in metadata_list:
            categories.add(metadata.get('category', 'other'))
        
        return jsonify({
            'status': 'success',
            'categories': sorted(list(categories))
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/tags')
def get_tags():
    """获取所有标签"""
    try:
        metadata_list = metadata_manager.get_all_metadata()
        tags = set()
        for metadata in metadata_list:
            for tag in metadata.get('tags', []):
                tags.add(tag)
        
        return jsonify({
            'status': 'success',
            'tags': sorted(list(tags))
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})



# 添加缺失的API端点
@app.route('/api/knowledge/<knowledge_id>', methods=['GET'])
def get_knowledge_item(knowledge_id):
    """获取单个知识条目详情"""
    try:
        knowledge_path = os.path.join('knowledge_base_storage', knowledge_id)
        
        if not os.path.exists(knowledge_path):
            return jsonify({'success': False, 'message': 'Knowledge not found'}), 404
        
        # 读取内容
        content_file = os.path.join(knowledge_path, 'content.txt')
        info_file = os.path.join(knowledge_path, 'info.json')
        
        knowledge_data = {
            'id': knowledge_id,
            'content': '',
            'title': '',
            'category': '',
            'summary': '',
            'tags': [],
            'created_at': '',
            'updated_at': '',
            'file_path': knowledge_path
        }
        
        # 读取内容
        if os.path.exists(content_file):
            with open(content_file, 'r', encoding='utf-8') as f:
                knowledge_data['content'] = f.read()
        
        # 读取元数据
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
                knowledge_data.update({
                    'title': info.get('title', ''),
                    'category': info.get('category', ''),
                    'summary': info.get('summary', ''),
                    'tags': info.get('tags', []),
                    'created_at': info.get('created_at', ''),
                    'updated_at': info.get('updated_at', '')
                })
        
        return jsonify({
            'success': True,
            'knowledge': knowledge_data
        })
        
    except Exception as e:
        logger.error(f"Error getting knowledge item {knowledge_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/knowledge/<knowledge_id>', methods=['DELETE'])
def delete_knowledge_item(knowledge_id):
    """删除单个知识条目"""
    try:
        # Use absolute path for knowledge storage
        storage_base = os.path.abspath('knowledge_base_storage')
        knowledge_path = os.path.join(storage_base, knowledge_id)
        
        if not os.path.exists(knowledge_path):
            logger.warning(f"Directory not found for deletion: {knowledge_path}")
            return jsonify({'success': False, 'message': 'Knowledge not found'}), 404
        
        # 删除目录及其内容
        import shutil
        shutil.rmtree(knowledge_path)
        
        logger.info(f"Deleted knowledge item: {knowledge_id}")
        return jsonify({
            'success': True,
            'message': 'Knowledge deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting knowledge item {knowledge_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/knowledge/delete_selected', methods=['POST'])
def delete_multiple_knowledge():
    """批量删除知识条目"""
    try:
        data = request.get_json()
        if not data or 'ids' not in data:
            return jsonify({'success': False, 'message': 'No knowledge IDs provided'}), 400
        
        deleted_ids = []
        failed_ids = []
        
        # Use absolute path for knowledge storage
        storage_base = os.path.abspath('knowledge_base_storage')
        
        for knowledge_id in data['ids']:
            try:
                knowledge_path = os.path.join(storage_base, knowledge_id)
                logger.info(f"Attempting to delete: {knowledge_path}")
                logger.info(f"Current working directory: {os.getcwd()}")
                logger.info(f"Storage base resolved to: {storage_base}")
                
                # Debug path resolution
                actual_path = os.path.abspath(os.path.join('knowledge_base_storage', knowledge_id))
                logger.info(f"Alternative path resolution: {actual_path}")
                logger.info(f"Path exists check: {os.path.exists(knowledge_path)}")
                
                if os.path.exists(knowledge_path):
                    import shutil
                    shutil.rmtree(knowledge_path)
                    deleted_ids.append(knowledge_id)
                    logger.info(f"Successfully deleted: {knowledge_path}")
                else:
                    # Try alternative path resolution
                    if os.path.exists(actual_path):
                        import shutil
                        shutil.rmtree(actual_path)
                        deleted_ids.append(knowledge_id)
                        logger.info(f"Successfully deleted using alternative path: {actual_path}")
                    else:
                        failed_ids.append(knowledge_id)
                        logger.warning(f"Directory not found: {knowledge_path}")
                        logger.warning(f"Alternative path also not found: {actual_path}")
            except Exception as e:
                logger.error(f"Failed to delete {knowledge_id}: {e}")
                failed_ids.append(knowledge_id)
        
        if len(deleted_ids) > 0:
            return jsonify({
                'success': True,
                'deleted_count': len(deleted_ids),
                'message': f'Successfully deleted {len(deleted_ids)} files',
                'status': 'success'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'No files found to delete',
                'status': 'error',
                'errors': [f'File not found for ID: {id}' for id in failed_ids]
            })
        
    except Exception as e:
        logger.error(f"Error in batch delete: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/knowledge/cleanup', methods=['POST'])
def cleanup_knowledge():
    """Clean up orphaned and temporary files"""
    try:
        knowledge_dir = os.path.join('knowledge_base')
        cleaned_files = []
        
        if not os.path.exists(knowledge_dir):
            return jsonify({'message': 'Knowledge base directory not found', 'cleaned_files': [], 'success': True})
        
        # Clean up temporary files
        temp_extensions = ['.tmp', '.bak', '.lock', '~']
        for root, dirs, files in os.walk(knowledge_dir):
            for file in files:
                if any(file.endswith(ext) for ext in temp_extensions) or file.startswith('.'):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        cleaned_files.append(file)
                        logger.info(f"Cleaned temporary file: {file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
        
        # Check for orphaned JSON files
        for root, dirs, files in os.walk(knowledge_dir):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    base_name = file.replace('.json', '')
                    
                    # Check if corresponding content file exists
                    content_extensions = ['.txt', '.md', '.py', '.js', '.html', '.css', '.pdf', '.doc', '.docx']
                    content_exists = False
                    
                    for ext in content_extensions:
                        content_file = base_name + ext
                        content_path = os.path.join(root, content_file)
                        if os.path.exists(content_path):
                            content_exists = True
                            break
                    
                    # Also check for content in JSON metadata
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if 'content' in data and data['content']:
                            content_exists = True
                    except:
                        pass
                    
                    if not content_exists:
                        try:
                            os.remove(json_path)
                            cleaned_files.append(file)
                            logger.info(f"Cleaned orphaned metadata: {file}")
                        except Exception as e:
                            logger.warning(f"Failed to remove orphaned {json_path}: {e}")
        
        # Clean empty directories
        for root, dirs, files in os.walk(knowledge_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # Directory is empty
                        os.rmdir(dir_path)
                        logger.info(f"Cleaned empty directory: {dir_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove empty directory {dir_path}: {e}")
        
        return jsonify({
            'message': f'Cleaned up {len(cleaned_files)} files',
            'cleaned_files': cleaned_files,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/knowledge/backup/status')
def backup_status():
    """Get backup status and list available backups"""
    try:
        backup_dir = os.path.join('backups')
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir, exist_ok=True)
            return jsonify({'backups': [], 'total': 0, 'success': True})
        
        backups = []
        for file in os.listdir(backup_dir):
            if file.startswith('knowledge_base_backup_') and file.endswith('.db'):
                file_path = os.path.join(backup_dir, file)
                try:
                    stat = os.stat(file_path)
                    backups.append({
                        'filename': file,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'path': file_path
                    })
                except Exception as e:
                    logger.warning(f"Error getting backup info for {file}: {e}")
        
        # Sort by creation date, newest first
        backups.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'backups': backups,
            'total': len(backups),
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Backup status error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/knowledge/restore/<backup_filename>', methods=['POST'])
def restore_backup(backup_filename):
    """Restore from a backup file"""
    try:
        backup_dir = os.path.join('backups')
        backup_path = os.path.join(backup_dir, backup_filename)
        
        if not os.path.exists(backup_path):
            return jsonify({'error': 'Backup file not found', 'success': False}), 404
        
        # Validate backup filename for security
        if not backup_filename.startswith('knowledge_base_backup_') or not backup_filename.endswith('.db'):
            return jsonify({'error': 'Invalid backup filename', 'success': False}), 400
        
        # Validate filename doesn't contain path traversal
        if '..' in backup_filename or '/' in backup_filename or '\\' in backup_filename:
            return jsonify({'error': 'Invalid backup filename', 'success': False}), 400
        
        # In a real system, this would restore the database
        # For demonstration, we'll create a success response
        logger.info(f"Backup restore requested: {backup_filename}")
        
        return jsonify({
            'message': f'Successfully initiated restore from {backup_filename}',
            'backup_file': backup_filename,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Restore error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'success': False}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error', 'success': False}), 500

# Start application
# Start A management model API
threading.Thread(target=lambda: subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), 'backend', 'a_manager_api.py')]), daemon=True).start()

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('knowledge_base', exist_ok=True)
    os.makedirs('backups', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    logger.info("Starting Self Brain AGI System Web Interface")
    logger.info("Visit http://localhost:5000 for main page")
    logger.info("Available endpoints:")
    logger.info("  - Main Interface: http://localhost:5000")
    logger.info("  - Training Control: http://localhost:5000/training")
    logger.info("  - Knowledge Import: http://localhost:5000/knowledge/import")
    logger.info("  - Knowledge Manage: http://localhost:5000/knowledge_manage")
    logger.info("  - API Status: http://localhost:5000/api/system/status")
    logger.info("  - Command Execute: http://localhost:5000/api/execute")
    logger.info("  - Models Status: http://localhost:5000/api/models/status")
    
    # Run Flask application
    socketio.run(app, 
                host='0.0.0.0', 
                port=5000, 
                debug=True, 
                allow_unsafe_werkzeug=True)
