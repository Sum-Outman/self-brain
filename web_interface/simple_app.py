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

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebInterface")

# Create Flask application
app = Flask(__name__,
           template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.secret_key = 'self_heart_agi_system_secret_key_2025'

# Add global CORS headers
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Cache-Control'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

app.after_request(add_cors_headers)

# Create SocketIO instance with optimized configuration
logger.info("Initializing SocketIO...")
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
    always_connect=True
)
logger.info("SocketIO initialized successfully")

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
    logger.info("Status request received")
    emit('status_update', {
        'status': 'running',
        'message': 'System operational',
        'models': {'total': 11, 'active': 11},
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('start_training')
def handle_start_training(data):
    """Handle start training request"""
    try:
        logger.info(f"Start training request: {data}")
        task_name = data.get('task_name', 'Default Task')
        device = data.get('device', 'cpu')
        models = data.get('models', [])
        training_type = data.get('training_type', 'standard')
        
        logger.info(f"Starting training: task_name={task_name}, device={device}, models={models}, training_type={training_type}")
        
        # Acknowledge the start training request
        emit('training_start_ack', {
            'status': 'success',
            'message': f'Training task "{task_name}" started successfully on {device}',
            'task_name': task_name,
            'device': device
        })
        
        # Simulate training progress (for testing)
        def simulate_training():
            for i in range(1, 101):
                socketio.sleep(0.5)
                socketio.emit('training_log_update', {
                    'message': f'Epoch {i}/100 - Loss: {0.5 * (1 - i/100):.4f} - Accuracy: {i:.1f}%'
                })
            socketio.emit('training_stop_ack', {
                'status': 'success',
                'message': 'Training completed successfully'
            })
        
        socketio.start_background_task(simulate_training)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to start training: {error_msg}")
        emit('error', {'message': f'Failed to start training: {error_msg}'})

@socketio.on('pause_training')
def handle_pause_training():
    """Handle pause training request"""
    try:
        logger.info("Pause training request received")
        emit('training_pause_ack', {'status': 'success', 'message': 'Training paused'})
    except Exception as e:
        logger.error(f"Failed to pause training: {str(e)}")
        emit('error', {'message': f'Failed to pause training: {str(e)}'})

@socketio.on('resume_training')
def handle_resume_training():
    """Handle resume training request"""
    try:
        logger.info("Resume training request received")
        emit('training_resume_ack', {'status': 'success', 'message': 'Training resumed'})
    except Exception as e:
        logger.error(f"Failed to resume training: {str(e)}")
        emit('error', {'message': f'Failed to resume training: {str(e)}'})

@socketio.on('stop_training')
def handle_stop_training():
    """Handle stop training request"""
    try:
        logger.info("Stop training request received")
        emit('training_stop_ack', {'status': 'success', 'message': 'Training stopping...'})
    except Exception as e:
        logger.error(f"Failed to stop training: {str(e)}")
        emit('error', {'message': f'Failed to stop training: {str(e)}'})

@socketio.on('reset_training')
def handle_reset_training():
    """Handle reset training request"""
    try:
        logger.info("Reset training request received")
        emit('training_reset_ack', {'status': 'success', 'message': 'Training reset'})
    except Exception as e:
        logger.error(f"Failed to reset training: {str(e)}")
        emit('error', {'message': f'Failed to reset training: {str(e)}'})

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('ai_chat.html')

@app.route('/training')
def training_page():
    """Training page"""
    logger.info("Training page accessed")
    response = render_template('training.html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/training/status')
def get_training_status():
    """Get training status API endpoint"""
    logger.info("Training status API accessed")
    return jsonify({
        'status': 'success',
        'training': {
            'active_sessions': 0,
            'total_sessions': 0,
            'completed_sessions': 0,
            'failed_sessions': 0,
            'sessions': []
        },
        'system': {
            'cpu_usage': 0,
            'memory_usage': {
                'total': 0,
                'used': 0,
                'available': 0,
                'percent': 0
            },
            'disk_usage': {
                'total': 0,
                'used': 0,
                'free': 0,
                'percent': 0
            },
            'gpu_info': []
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/training/sessions')
def get_training_sessions():
    """Get training sessions API endpoint"""
    logger.info("Training sessions API accessed")
    return jsonify({
        'sessions': [],
        'active_sessions': 0,
        'total_epochs': 0,
        'avg_accuracy': 0,
        'training_time': '0h'
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('knowledge_base', exist_ok=True)
    os.makedirs('backups', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    logger.info("Starting Self Brain AGI System - Simple Version")
    logger.info("Visit http://localhost:8080 for main page")
    logger.info("Available endpoints:")
    logger.info("  - Main Interface: http://localhost:8080")
    logger.info("  - Training Control: http://localhost:8080/training")
    
    # Run Flask application with Socket.IO
    socketio.run(app, 
                host='0.0.0.0', 
                port=8080,
                debug=True, 
                allow_unsafe_werkzeug=True,
                use_reloader=False)