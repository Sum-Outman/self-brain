

import asyncio
import psutil
import time
import threading
from datetime import datetime
import json
from flask import Blueprint, jsonify, request
from flask_socketio import SocketIO, emit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
realtime_bp = Blueprint('realtime', __name__)

# Global variables to store monitoring data
monitor_data = {
    'resources': {
        'cpu_usage': 0,
        'memory_usage': 0,
        'disk_usage': 0,
        'network_io': {'sent': 0, 'recv': 0}
    },
    'training_status': {
        'sessions': {
            'total': 0,
            'active': 0,
            'completed': 0,
            'failed': 0
        },
        'active_sessions': []
    }
}

# Monitor thread control
monitor_thread = None
monitor_running = False

def get_system_resources():
    """Get system resource usage"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage (using system disk)
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Network IO - Safely handle Windows format character errors
        network_data = {
            'sent': 0,
            'recv': 0
        }
        
        try:
            net_io = psutil.net_io_counters()
            network_data = {
                'sent': net_io.bytes_sent,
                'recv': net_io.bytes_recv
            }
        except Exception as net_error:
            # Format character errors that may occur on Windows, use default values
            logger.warning(f"Network IO monitoring limited: {net_error}")
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory_percent,
            'disk_usage': disk_percent,
            'network_io': network_data,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get system resources: {e}")
        return {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'network_io': {'sent': 0, 'recv': 0},
            'timestamp': datetime.now().isoformat()
        }

def get_training_status():
    """Get training status information"""
    # Need to get actual data from training manager here
    # Temporarily return simulated data
    return {
        'sessions': {
            'total': 15,
            'active': 3,
            'completed': 10,
            'failed': 2
        },
        'active_sessions': [
            {
                'id': 'session_001',
                'mode': 'joint_training',
                'progress': 45,
                'status': 'training',
                'start_time': '2024-01-15 10:30:00',
                'models': ['A', 'B', 'C']
            },
            {
                'id': 'session_002',
                'mode': 'individual_training',
                'progress': 78,
                'status': 'validating',
                'start_time': '2024-01-15 11:15:00',
                'models': ['D']
            },
            {
                'id': 'session_003',
                'mode': 'fine_tuning',
                'progress': 22,
                'status': 'training',
                'start_time': '2024-01-15 12:00:00',
                'models': ['E', 'F']
            }
        ],
        'timestamp': datetime.now().isoformat()
    }

def monitor_loop(socketio):
    """Monitoring loop, periodically sends updates"""
    global monitor_running
    
    # Initialize network IO counter (safe handling of Windows format character errors)
    last_net_io = None
    try:
        last_net_io = psutil.net_io_counters()
    except Exception as net_error:
        logger.warning(f"Network IO monitoring initialization limited: {net_error}")
        # Create default network IO counter object
        class DefaultNetIO:
            bytes_sent = 0
            bytes_recv = 0
        last_net_io = DefaultNetIO()
    
    while monitor_running:
        try:
            # Get system resources
            resources = get_system_resources()
            
            # Calculate network speed (KB/s) - Safe handling of Windows format character errors
            current_net_io = None
            try:
                current_net_io = psutil.net_io_counters()
                time_diff = 1.0  # 1 second interval
                
                net_sent_speed = (current_net_io.bytes_sent - last_net_io.bytes_sent) / time_diff / 1024
                net_recv_speed = (current_net_io.bytes_recv - last_net_io.bytes_recv) / time_diff / 1024
                
                resources['network_io']['sent_speed'] = net_sent_speed
                resources['network_io']['recv_speed'] = net_recv_speed
                
                last_net_io = current_net_io
            except Exception as net_error:
                # Format character errors that may occur on Windows, use default values
                logger.warning(f"Network speed calculation limited: {net_error}")
                resources['network_io']['sent_speed'] = 0
                resources['network_io']['recv_speed'] = 0
            
            # Get training status
            training_status = get_training_status()
            
            # Update global data
            monitor_data['resources'] = resources
            monitor_data['training_status'] = training_status
            
            # Send updates via SocketIO
            socketio.emit('resource_update', {
                'resources': resources,
                'type': 'realtime'
            })
            
            socketio.emit('training_status', {
                'dashboard': training_status,
                'sessions': training_status['active_sessions'],
                'type': 'realtime'
            })
            
            # Log information
            logger.info(f"Resource monitoring: CPU={resources['cpu_usage']}%, "
                       f"Memory={resources['memory_usage']}%, "
                       f"Disk={resources['disk_usage']}%")
            
            # Wait 1 second
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            time.sleep(5)

def start_monitoring(socketio):
    """Start monitoring thread"""
    global monitor_thread, monitor_running
    
    if monitor_thread and monitor_thread.is_alive():
        return
    
    monitor_running = True
    monitor_thread = threading.Thread(
        target=monitor_loop,
        args=(socketio,),
        daemon=True
    )
    monitor_thread.start()
    logger.info("Real-time monitoring started")

def stop_monitoring():
    """Stop monitoring thread"""
    global monitor_running
    monitor_running = False
    logger.info("Real-time monitoring stopped")

# REST API routes
@realtime_bp.route('/api/monitor/resources', methods=['GET'])
def get_resources():
    """Get current system resource status"""
    return jsonify({
        'success': True,
        'data': monitor_data['resources'],
        'timestamp': datetime.now().isoformat()
    })

@realtime_bp.route('/api/monitor/training', methods=['GET'])
def get_training():
    """Get current training status"""
    return jsonify({
        'success': True,
        'data': monitor_data['training_status'],
        'timestamp': datetime.now().isoformat()
    })

@realtime_bp.route('/api/monitor/status', methods=['GET'])
def get_monitor_status():
    """Get monitoring system status"""
    return jsonify({
        'success': True,
        'monitoring': monitor_running,
        'thread_alive': monitor_thread.is_alive() if monitor_thread else False,
        'timestamp': datetime.now().isoformat()
    })

# SocketIO event handling
def setup_socketio_events(socketio):
    """Set up SocketIO event handling"""
    
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('monitor_status', {
            'monitoring': monitor_running,
            'message': 'Real-time monitoring service connected'
        })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('get_resources')
    def handle_get_resources():
        emit('resource_update', {
            'resources': monitor_data['resources'],
            'type': 'request'
        })
    
    @socketio.on('get_training_status')
    def handle_get_training_status():
        emit('training_status', {
            'dashboard': monitor_data['training_status'],
            'sessions': monitor_data['training_status']['active_sessions'],
            'type': 'request'
        })
    
    @socketio.on('start_monitoring')
    def handle_start_monitoring():
        start_monitoring(socketio)
        emit('monitor_status', {
            'monitoring': True,
            'message': 'Real-time monitoring started'
        })
    
    @socketio.on('stop_monitoring')
    def handle_stop_monitoring():
        stop_monitoring()
        emit('monitor_status', {
            'monitoring': False,
            'message': 'Real-time monitoring stopped'
        })
    
    @socketio.on('stop_session')
    def handle_stop_session(data):
        session_id = data.get('session_id')
        logger.info(f"Request to stop session: {session_id}")
        # Should call training manager's stop function here
        emit('session_stopped', {
            'session_id': session_id,
            'success': True,
            'message': f'Session {session_id} stopped'
        })

# Initialization function
def init_realtime_monitor(app, socketio):
    """Initialize real-time monitoring system"""
    # Register blueprint
    app.register_blueprint(realtime_bp)
    
    # Set up SocketIO events
    setup_socketio_events(socketio)
    
    # Start monitoring
    start_monitoring(socketio)
    
    logger.info("Real-time monitoring system initialization completed")

# Health check
@realtime_bp.route('/api/monitor/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'realtime_monitor',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })
