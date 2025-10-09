#!/usr/bin/env python
# Self Brain AGI System - Web Interface Startup Script
# Copyright 2025 AGI System Team

import os
import sys
import logging
import threading
import time

# Add project root to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SelfBrainWebInterface")

# Check if required dependencies are installed
try:
    from flask import Flask, render_template, request, jsonify, make_response
    from flask_socketio import SocketIO, emit
    from flask_cors import CORS
    import psutil
    import json
    import yaml
    import uuid
    from datetime import datetime

except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.info("Please install dependencies using: pip install -r requirements.txt")
    sys.exit(1)

# Initialize the Flask application
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.secret_key = 'self_heart_agi_system_secret_key_2025'  # In production, use a secure key

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Cache-Control'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

@app.before_request
def handle_options():
    """Handle OPTIONS requests immediately"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
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
        "supports_credentials": True
    },
    r"/socket.io/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True
    },
    r"/static/*": {
        "origins": "*",
        "methods": ["GET", "OPTIONS"],
        "allow_headers": ["Cache-Control"],
        "supports_credentials": False
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
    always_connect=True
)

# Initialize core components with error handling
training_controller = None
camera_manager = None
device_manager = None
emotion_engine = None
model_api_manager = None
model_registry = {}

# Try to initialize Training Controller
try:
    from training_manager.advanced_train_control import TrainingController
    training_controller = TrainingController()
    logger.info("Training Controller initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Training Controller: {e}")
    # Create a mock TrainingController to prevent application failure
    class MockTrainingController:
        def __init__(self):
            self.training_sessions = []
        def start_training(self, *args, **kwargs):
            logger.info("Mock: Training started")
        def stop_training(self, *args, **kwargs):
            logger.info("Mock: Training stopped")
        def get_training_history(self):
            return []
    training_controller = MockTrainingController()

# Try to initialize Camera Manager
try:
    from camera_manager import get_camera_manager
    camera_manager = get_camera_manager()
    logger.info("Camera Manager initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Camera Manager: {e}")
    # Create a mock CameraManager to prevent application failure
    class MockCameraManager:
        def __init__(self):
            self.cameras = {}
        def get_available_cameras(self):
            return [{"id": "0", "name": "Mock Camera 1"}, {"id": "1", "name": "Mock Camera 2"}]
        def get_camera_feed(self, camera_id):
            return None
    camera_manager = MockCameraManager()

# Try to initialize Device Manager
try:
    from unified_device_communication import get_device_manager, device_bp
    app.register_blueprint(device_bp)
    device_manager = get_device_manager()
    if device_manager and camera_manager:
        device_manager.set_camera_manager(camera_manager)
        device_manager.start()
    logger.info("Device Manager initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Device Manager: {e}")

# Try to initialize Emotion Engine
try:
    from manager_model.emotion_engine import get_emotion_engine
    emotion_engine = get_emotion_engine()
    logger.info("Emotion Engine initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Emotion Engine: {e}")

# Try to initialize Model API Manager
try:
    from web_interface.backend.model_api_manager import get_model_api_manager
    model_api_manager = get_model_api_manager()
    logger.info("Model API Manager initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Model API Manager: {e}")

# Try to initialize Enhanced Realtime Monitor
try:
    from web_interface.backend.enhanced_realtime_monitor import init_enhanced_realtime_monitor
    init_enhanced_realtime_monitor(app, socketio)
    logger.info("Enhanced Realtime Monitor initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Enhanced Realtime Monitor: {e}")

# Try to initialize Knowledge Self Learning API
try:
    from web_interface.backend.knowledge_self_learning_api import knowledge_self_learning_bp
    app.register_blueprint(knowledge_self_learning_bp)
    logger.info("Knowledge Self Learning API initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Knowledge Self Learning API: {e}")

# Add root-level camera API endpoint
@app.route('/api/camera', methods=['GET', 'POST'])
def camera_root():
    """Root-level camera API endpoint that provides all camera functionality"""
    if request.method == 'GET':
        try:
            # Get available cameras
            available_cameras = []
            if camera_manager and hasattr(camera_manager, 'get_available_cameras'):
                available_cameras = camera_manager.get_available_cameras()
            else:
                # Default mock cameras if camera manager is not available
                available_cameras = [
                    {"id": "0", "name": "Integrated Camera", "width": 1280, "height": 720},
                    {"id": "1", "name": "External Camera", "width": 640, "height": 480}
                ]
            
            # Get active camera IDs
            active_camera_ids = []
            if camera_manager and hasattr(camera_manager, 'get_active_camera_ids'):
                active_camera_ids = camera_manager.get_active_camera_ids()
            
            # Return camera information
            return jsonify({
                "status": "success",
                "available_cameras": available_cameras,
                "active_camera_count": len(active_camera_ids),
                "active_camera_ids": active_camera_ids,
                "api_version": "1.0"
            })
        except Exception as e:
            logger.error(f"Error in GET /api/camera: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    elif request.method == 'POST':
        try:
            data = request.get_json() or {}
            operation = data.get('operation', 'start')
            camera_id = int(data.get('camera_id', 0))
            
            if operation.lower() == 'start':
                # Prepare camera parameters
                params = {}
                resolution = data.get('resolution', '1280x720')
                if resolution:
                    try:
                        width, height = map(int, resolution.split('x'))
                        params['width'] = width
                        params['height'] = height
                    except ValueError:
                        logger.warning(f"Invalid resolution format: {resolution}")
                
                # Start camera
                success = False
                if camera_manager and hasattr(camera_manager, 'start_camera'):
                    success = camera_manager.start_camera(camera_id, params)
                else:
                    # Mock success if camera manager is not available
                    success = True
                
                if success:
                    return jsonify({"status": "success", "message": f"Camera {camera_id} started"})
                else:
                    return jsonify({"status": "error", "message": f"Failed to start camera {camera_id}"}), 400
            
            elif operation.lower() == 'stop':
                # Stop camera
                success = False
                if camera_manager and hasattr(camera_manager, 'stop_camera'):
                    success = camera_manager.stop_camera(camera_id)
                else:
                    # Mock success if camera manager is not available
                    success = True
                
                if success:
                    return jsonify({"status": "success", "message": f"Camera {camera_id} stopped"})
                else:
                    return jsonify({"status": "error", "message": f"Failed to stop camera {camera_id}"}), 400
            
            else:
                return jsonify({"status": "error", "message": f"Unsupported operation: {operation}"}), 400
                
        except Exception as e:
            logger.error(f"Error in POST /api/camera: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

# Load model registry
try:
    registry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'model_registry.json')
    if os.path.exists(registry_path):
        with open(registry_path, 'r', encoding='utf-8') as f:
            model_registry = json.load(f)
    else:
        # Create default model registry if it doesn't exist
        model_registry = {
            "A_management": {"name": "Management Model", "type": "core", "active": True},
            "B_language": {"name": "Language Model", "type": "processing", "active": True},
            "C_audio": {"name": "Audio Model", "type": "processing", "active": True},
            "D_image": {"name": "Image Model", "type": "processing", "active": True},
            "E_video": {"name": "Video Model", "type": "processing", "active": True},
            "F_spatial": {"name": "Spatial Model", "type": "processing", "active": True},
            "G_sensor": {"name": "Sensor Model", "type": "input", "active": True},
            "H_computer": {"name": "Computer Control Model", "type": "output", "active": True},
            "I_actuator": {"name": "Actuator Model", "type": "output", "active": True},
            "J_knowledge": {"name": "Knowledge Model", "type": "core", "active": True},
            "K_programming": {"name": "Programming Model", "type": "core", "active": True}
        }
        # Ensure config directory exists
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(model_registry, f, indent=4, ensure_ascii=False)
    logger.info("Model registry loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model registry: {e}")
    model_registry = {}

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
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/') if os.name != 'nt' else psutil.disk_usage('C:')
        
        # Get GPU info (simplified for compatibility)
        gpu_info = []
        try:
            import subprocess
            if os.name == 'nt':  # Windows
                result = subprocess.run(['powershell', 'Get-WmiObject -Class Win32_VideoController | Select-Object Name,AdapterRAM'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip() and not line.strip().startswith('-'):
                            gpu_info.append({"name": line.strip(), "memory_total": 0, "memory_used": 0, "utilization": 0})
            else:  # Linux/Mac
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\n'):
                            if line:
                                parts = line.split(', ')
                                if len(parts) >= 4:
                                    gpu_info.append({
                                        'name': parts[0],
                                        'memory_total': int(parts[1]) if parts[1].isdigit() else 0,
                                        'memory_used': int(parts[2]) if parts[2].isdigit() else 0,
                                        'utilization': int(parts[3]) if parts[3].isdigit() else 0
                                    })
                except:
                    pass
        except:
            gpu_info = []
        
        # Get training sessions
        sessions = []
        if training_controller and hasattr(training_controller, 'get_training_history'):
            try:
                sessions = training_controller.get_training_history()
            except Exception as e:
                logger.warning(f"Failed to get training history: {e}")
        
        status_data = {
            'system_name': 'Self Brain AGI',
            'version': '1.0.0',
            'uptime': 'Running',
            'models': {
                'total': len(model_registry),
                'active': sum(1 for m in model_registry.values() if m.get('active', False))
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
    try:
        emit('error', {'message': str(e)})
    except:
        pass

# Define routes for the application
@app.route('/')
def index():
    """Home page display with English version"""
    return render_template('index_en.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('user_id', 'default_user')
        
        # Process chat message (simplified for this example)
        # In a real implementation, this would interact with the A_management model
        response = {
            'response': f"I received your message: {message}",
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"Chat message from {user_id}: {message}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error handling chat: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        config = data.get('config', {})
        
        if training_controller and hasattr(training_controller, 'start_training'):
            result = training_controller.start_training(model_id, config)
            logger.info(f"Started training for model: {model_id}")
            return jsonify({'success': True, 'result': result})
        else:
            logger.warning("Training controller not available")
            return jsonify({'success': False, 'error': 'Training controller not available'})
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop model training"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if training_controller and hasattr(training_controller, 'stop_training'):
            result = training_controller.stop_training(session_id)
            logger.info(f"Stopped training session: {session_id}")
            return jsonify({'success': True, 'result': result})
        else:
            logger.warning("Training controller not available")
            return jsonify({'success': False, 'error': 'Training controller not available'})
    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/camera/list', methods=['GET'])
def list_cameras():
    """List available cameras"""
    try:
        if camera_manager and hasattr(camera_manager, 'get_available_cameras'):
            cameras = camera_manager.get_available_cameras()
        else:
            # Default mock cameras if camera manager is not available
            cameras = [{"id": "0", "name": "Integrated Camera"}, {"id": "1", "name": "External Camera"}]
        
        logger.info(f"Available cameras: {len(cameras)}")
        return jsonify({'cameras': cameras, 'success': True})
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save_api_config', methods=['POST'])
def save_api_config():
    """Save API configuration for a model"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        api_url = data.get('api_url')
        api_key = data.get('api_key')
        api_model_name = data.get('api_model_name')
        api_type = data.get('api_type')
        replace_local = data.get('replace_local', False)
        
        if not model_id:
            return jsonify({'success': False, 'error': 'Invalid model ID'})
        
        # Update model registry
        if model_id not in model_registry:
            model_registry[model_id] = {"name": model_id, "type": "external", "active": True}
        
        model_registry[model_id]['api_url'] = api_url
        model_registry[model_id]['api_key'] = api_key
        model_registry[model_id]['api_model_name'] = api_model_name
        model_registry[model_id]['api_type'] = api_type
        model_registry[model_id]['model_source'] = 'external' if replace_local and api_url else 'local'
        
        # Save updated registry
        registry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'model_registry.json')
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(model_registry, f, indent=4, ensure_ascii=False)
        
        logger.info(f"API configuration saved for model: {model_id}")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error saving API config: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/system_settings')
def system_settings():
    """System settings page"""
    try:
        # Load system config
        system_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'system_config.yaml')
        system_config = {}
        
        if os.path.exists(system_config_path):
            with open(system_config_path, 'r', encoding='utf-8') as f:
                system_config = yaml.safe_load(f) or {}
        
        # Add 'active' field to model registry if not present
        for model_id in model_registry:
            if 'active' not in model_registry[model_id]:
                model_registry[model_id]['active'] = True
        
        return render_template('system_settings.html', 
                             model_registry=model_registry, 
                             model_registry_json=json.dumps(model_registry),
                             system_config=system_config)
    except Exception as e:
        logger.error(f"Error loading system settings: {str(e)}")
        return render_template('system_settings.html', 
                             model_registry={}, 
                             model_registry_json='{}',
                             system_config={})

# Add additional routes needed for the web interface
@app.route('/training')
def training_page():
    """Training page"""
    response = make_response(render_template('training.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/camera_management')
def camera_management():
    """Camera management page"""
    return render_template('camera_management.html')

@app.route('/device_communication')
def device_communication_page():
    """Device communication page"""
    return render_template('device_communication.html')

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
        
        logger.info(f"General settings saved")
        return jsonify({'status': 'success', 'message': 'General settings saved successfully'})
        
    except Exception as e:
        logger.error(f"Failed to save general settings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

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
    return response

# Start periodic system status updates
thread_running = False
def periodic_status_update():
    """Periodically send system status updates to all connected clients"""
    global thread_running
    thread_running = True
    
    while thread_running:
        try:
            # Force a status update
            with app.app_context():
                # Get system information
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/') if os.name != 'nt' else psutil.disk_usage('C:')
                
                # Simplified GPU info
                gpu_info = []
                
                status_data = {
                    'system_name': 'Self Brain AGI',
                    'version': '1.0.0',
                    'system': {
                        'cpu_usage': cpu_percent,
                        'memory_usage': {
                            'percent': memory.percent
                        },
                        'disk_usage': {
                            'percent': (disk.used / disk.total) * 100
                        },
                        'gpu_info': gpu_info
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                socketio.emit('system_status_update', status_data)
        except Exception as e:
            logger.error(f"Error in periodic status update: {e}")
        
        # Sleep for 5 seconds before next update
        time.sleep(5)

# Cleanup function for graceful shutdown
def cleanup():
    """Cleanup resources on shutdown"""
    global thread_running
    thread_running = False
    
    if device_manager and hasattr(device_manager, 'stop'):
        try:
            device_manager.stop()
            logger.info("Device Manager stopped")
        except Exception as e:
            logger.error(f"Error stopping Device Manager: {e}")
    
    logger.info("Self Brain AGI Web Interface shutdown complete")

# Main entry point
if __name__ == '__main__':
    try:
        # Start periodic status update thread
        status_thread = threading.Thread(target=periodic_status_update, daemon=True)
        status_thread.start()
        logger.info("Periodic status update thread started")
        
        # Check if we're in a development environment
        is_development = os.environ.get('FLASK_ENV', 'development') == 'development'
        
        # Run the Flask application
        logger.info("Starting Self Brain AGI Web Interface")
        logger.info("Access the interface at http://localhost:8080")
        
        # Use SocketIO's run method for better WebSocket support
        socketio.run(
            app,
            host='0.0.0.0',  # Listen on all interfaces
            port=8080,       # Default port for the web interface
            debug=is_development,
            use_reloader=False  # Disable reloader to avoid issues with threads
        )
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        cleanup()