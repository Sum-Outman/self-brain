# Self Brain AGI System - Simple Web Interface
# This is a simplified version of the web interface that can run without external dependencies

from flask import Flask, render_template, request, jsonify, Response, make_response, session
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime
import threading
import random
import json
import uuid
import time
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimpleWebInterface")

# Create Flask application
app = Flask(__name__, static_folder='web_interface/static', template_folder='web_interface/templates')
app.config['SECRET_KEY'] = 'simple_secret_key'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

# Initialize SocketIO and CORS
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Simulated model registry
model_registry = {
    'A_management': {'name': 'Management Model', 'status': 'running', 'type': 'main', 'port': 5001},
    'B_language': {'name': 'Language Model', 'status': 'running', 'type': 'language', 'port': 5002},
    'C_audio': {'name': 'Audio Model', 'status': 'running', 'type': 'audio', 'port': 5003},
    'D_image': {'name': 'Image Model', 'status': 'running', 'type': 'vision', 'port': 5004},
    'E_video': {'name': 'Video Model', 'status': 'running', 'type': 'vision', 'port': 5005},
    'F_spatial': {'name': 'Spatial Model', 'status': 'running', 'type': 'perception', 'port': 5006},
    'G_sensor': {'name': 'Sensor Model', 'status': 'running', 'type': 'perception', 'port': 5007},
    'H_computer': {'name': 'Computer Control Model', 'status': 'running', 'type': 'control', 'port': 5008},
    'I_motion': {'name': 'Motion Control Model', 'status': 'running', 'type': 'control', 'port': 5009},
    'J_knowledge': {'name': 'Knowledge Model', 'status': 'running', 'type': 'knowledge', 'port': 5010},
    'K_programming': {'name': 'Programming Model', 'status': 'running', 'type': 'utility', 'port': 5011}
}

# Simulated training control
class SimulatedTrainingControl:
    def __init__(self):
        self.training_sessions = []
        self._create_sample_sessions()
    
    def _create_sample_sessions(self):
        self.training_sessions = [
            {
                'id': str(uuid.uuid4()),
                'model_id': 'B_language',
                'model_name': 'Language Model',
                'status': 'completed',
                'accuracy': 0.92,
                'loss': 0.15,
                'start_time': (datetime.now() - timedelta(days=1)).isoformat(),
                'end_time': datetime.now().isoformat()
            },
            {
                'id': str(uuid.uuid4()),
                'model_id': 'D_image',
                'model_name': 'Image Model',
                'status': 'running',
                'accuracy': 0.78,
                'loss': 0.32,
                'start_time': (datetime.now() - timedelta(hours=2)).isoformat(),
                'end_time': None
            }
        ]

training_control = SimulatedTrainingControl()

# Simple device manager that doesn't depend on psutil or external libraries
class SimpleDeviceManager:
    def __init__(self):
        self.connected_devices = {}
        self.sensors = self._initialize_sensors()
        logger.info("SimpleDeviceManager initialized")
    
    def _initialize_sensors(self):
        """Initialize simulated sensors"""
        return [
            {
                'id': 'cpu_temperature',
                'name': 'CPU Temperature',
                'type': 'temperature',
                'status': 'available'
            },
            {
                'id': 'memory_usage',
                'name': 'Memory Usage',
                'type': 'memory',
                'status': 'available'
            },
            {
                'id': 'disk_usage',
                'name': 'Disk Usage',
                'type': 'storage',
                'status': 'available'
            }
        ]
    
    def list_devices(self):
        """List simulated devices"""
        devices = []
        
        # Simulate camera devices
        devices.append({
            'id': 'camera_0',
            'name': 'Integrated Webcam',
            'type': 'camera',
            'status': 'available'
        })
        devices.append({
            'id': 'camera_1',
            'name': 'External Webcam',
            'type': 'camera',
            'status': 'available'
        })
        
        # Simulate sensor devices
        devices.append({
            'id': 'sensor_0',
            'name': 'Temperature Sensor',
            'type': 'sensor',
            'status': 'available'
        })
        
        return devices
    
    def get_device_info(self, device_id):
        """Get simulated device information"""
        return {
            'id': device_id,
            'name': f'Simulated Device {device_id}',
            'type': 'generic_device',
            'status': 'available',
            'connected': device_id in self.connected_devices
        }
    
    def connect_device(self, device_id, params):
        """Simulate device connection"""
        try:
            self.connected_devices[device_id] = {
                'params': params,
                'connected_at': datetime.now().isoformat(),
                'status': 'connected'
            }
            return {'status': 'success', 'message': f'Device {device_id} connected'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def disconnect_device(self, device_id):
        """Simulate device disconnection"""
        if device_id in self.connected_devices:
            del self.connected_devices[device_id]
            return {'status': 'success', 'message': f'Device {device_id} disconnected'}
        return {'status': 'error', 'message': 'Device not found'}
    
    def send_command(self, device_id, command):
        """Simulate sending command to device"""
        if device_id in self.connected_devices:
            return {
                'status': 'success',
                'message': f'Command "{command}" sent to device {device_id}',
                'response': f'Command executed: {command}'
            }
        return {'status': 'error', 'message': 'Device not connected'}
    
    def list_sensors(self):
        """List simulated sensors"""
        return self.sensors
    
    def get_sensor_data(self, sensor_id=None):
        """Get simulated sensor data"""
        # Simulate sensor data
        if sensor_id == 'cpu_temperature':
            return {
                'sensor_id': sensor_id,
                'value': 45.0 + random.uniform(-5, 5),
                'unit': '°C',
                'timestamp': datetime.now().isoformat()
            }
        elif sensor_id == 'memory_usage':
            return {
                'sensor_id': sensor_id,
                'value': 60.0 + random.uniform(-10, 10),
                'unit': '%',
                'timestamp': datetime.now().isoformat()
            }
        elif sensor_id == 'disk_usage':
            return {
                'sensor_id': sensor_id,
                'value': 40.0 + random.uniform(-5, 5),
                'unit': '%',
                'timestamp': datetime.now().isoformat()
            }
        elif sensor_id is None:
            # Return all sensor data
            all_data = {}
            for sensor in self.sensors:
                all_data[sensor['id']] = self.get_sensor_data(sensor['id'])
            return all_data
        
        return {'status': 'error', 'message': 'Sensor not found'}

# Initialize device manager
device_manager = SimpleDeviceManager()

# Routes
@app.route('/')
def index():
    """Main interface route - Serve the simple HTML file"""
    with open('simple_index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/training')
def training():
    """Training control panel route"""
    with open('simple_index.html', 'r', encoding='utf-8') as f:
        content = f.read()
        # Modify the title for the training page
        return content.replace('<title>Self Brain AGI System</title>', '<title>Self Brain AGI System - Training Panel</title>')

@app.route('/device_communication')
def device_communication():
    """Device communication route"""
    with open('simple_index.html', 'r', encoding='utf-8') as f:
        content = f.read()
        # Modify the title for the device communication page
        return content.replace('<title>Self Brain AGI System</title>', '<title>Self Brain AGI System - Device Communication</title>')

# API endpoints
@app.route('/api/system/status')
def api_system_status():
    """System status API endpoint"""
    try:
        status_data = {
            'system_name': 'Self Brain AGI',
            'version': '1.0.0',
            'status': 'running',
            'models': model_registry,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/models/status')
def api_models_status():
    """Models status API endpoint"""
    try:
        status = {}
        for model_id, model_info in model_registry.items():
            status[model_id] = {
                'name': model_info['name'],
                'status': model_info['status'],
                'type': model_info['type'],
                'last_update': datetime.now().isoformat()
            }
        return jsonify({'status': 'success', 'models': status})
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/training/sessions')
def api_training_sessions():
    """Training sessions API endpoint"""
    try:
        sessions = training_control.training_sessions
        return jsonify({'status': 'success', 'sessions': sessions})
    except Exception as e:
        logger.error(f"Error getting training sessions: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/devices')
def api_get_devices():
    """Get all devices"""
    try:
        devices = device_manager.list_devices()
        return jsonify({'status': 'success', 'devices': devices})
    except Exception as e:
        logger.error(f"Error getting devices: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/devices/<device_id>')
def api_get_device(device_id):
    """Get device by ID"""
    try:
        device_info = device_manager.get_device_info(device_id)
        return jsonify({'status': 'success', 'device': device_info})
    except Exception as e:
        logger.error(f"Error getting device: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/devices/<device_id>/connect', methods=['POST'])
def api_connect_device(device_id):
    """Connect to a device"""
    try:
        params = request.json or {}
        result = device_manager.connect_device(device_id, params)
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        logger.error(f"Error connecting device: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/devices/<device_id>/disconnect', methods=['POST'])
def api_disconnect_device(device_id):
    """Disconnect from a device"""
    try:
        result = device_manager.disconnect_device(device_id)
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        logger.error(f"Error disconnecting device: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/devices/<device_id>/command', methods=['POST'])
def api_send_command(device_id):
    """Send command to device"""
    try:
        data = request.json
        if not data or 'command' not in data:
            return jsonify({'status': 'error', 'message': 'Command is required'}), 400
        
        result = device_manager.send_command(device_id, data['command'])
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        logger.error(f"Error sending command: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensors')
def api_get_sensors():
    """Get all sensors"""
    try:
        sensors = device_manager.list_sensors()
        return jsonify({'status': 'success', 'sensors': sensors})
    except Exception as e:
        logger.error(f"Error getting sensors: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensors/<sensor_id>/data')
def api_get_sensor_data(sensor_id):
    """Get sensor data"""
    try:
        data = device_manager.get_sensor_data(sensor_id)
        if 'status' in data and data['status'] == 'error':
            return jsonify(data), 400
        return jsonify({'status': 'success', 'data': data})
    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sensors/data')
def api_get_all_sensor_data():
    """Get all sensor data"""
    try:
        data = device_manager.get_sensor_data()
        return jsonify({
            'status': 'success',
            'sensors': data
        })
    except Exception as e:
        logger.error(f"Error getting sensor data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    # Send initial system status
    status = api_system_status().json
    emit('system_status_update', status)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('send_message')
def handle_message(data):
    """Handle incoming messages"""
    logger.info(f'Received message: {data}')
    
    # Simulate response from the management model
    response = {
        'sender': 'management_model',
        'content': f'Thank you for your message: {data.get("content", "")}',
        'timestamp': datetime.now().isoformat()
    }
    
    emit('receive_message', response)

@socketio.on('request_training_status')
def handle_training_status_request():
    """Handle training status request"""
    try:
        sessions = training_control.training_sessions
        emit('training_status_update', {'sessions': sessions})
    except Exception as e:
        logger.error(f"Error getting training status: {e}")

# Periodic update thread
def periodic_updates():
    """Send periodic updates to clients"""
    while True:
        try:
            with app.app_context():
                # Update system status
                status = api_system_status().json
                socketio.emit('system_status_update', status)
                
            # Update sensor data (doesn't need app context)
            sensor_data = device_manager.get_sensor_data()
            socketio.emit('sensor_data_update', sensor_data)
            
        except Exception as e:
            logger.error(f"Error in periodic updates: {e}")
        
        # Sleep for 5 seconds before next update
        time.sleep(5)

# Main function
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('knowledge_base', exist_ok=True)
    os.makedirs('backups', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Default port
    port = 5000  
    
    logger.info("Starting Self Brain AGI System - Simple Web Interface")
    logger.info(f"Visit http://localhost:{port} for main page")
    logger.info("Available endpoints:")
    logger.info(f"  - Main Interface: http://localhost:{port}")
    logger.info(f"  - Training Control: http://localhost:{port}/training")
    logger.info(f"  - Device Communication: http://localhost:{port}/device_communication")
    
    logger.info(f"Starting Web Interface on port {port}")
    
    # Start periodic update thread will be handled by SocketIO events
    # We don't start it here to avoid application context issues
    
    try:
        # Run Flask application
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=port, 
                    debug=True, 
                    use_reloader=False)
    finally:
        logger.info("Self Brain AGI System shutdown complete")