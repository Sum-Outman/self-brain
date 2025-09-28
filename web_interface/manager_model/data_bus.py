import logging
import threading
import queue
from datetime import datetime
import json
from flask import Blueprint, request, jsonify
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DataBus')

# Create blueprint
data_bus_bp = Blueprint('data_bus', __name__)

class DataBus:
    """Central communication hub for all models and components"""
    
    def __init__(self):
        self.message_queues = {}
        self.callbacks = {}
        self.lock = threading.RLock()
        self.is_running = False
        self.thread = None
        self.message_history = []
        self.max_history = 1000
    
    def start(self):
        """Start the data bus"""
        with self.lock:
            if self.is_running:
                return
            
            self.is_running = True
            self.thread = threading.Thread(target=self._process_messages, daemon=True)
            self.thread.start()
            logger.info("DataBus started")
    
    def stop(self):
        """Stop the data bus"""
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
            if self.thread:
                self.thread.join(2.0)
            logger.info("DataBus stopped")
    
    def _process_messages(self):
        """Process messages from all queues"""
        while self.is_running:
            # In a real implementation, this would process messages from all queues
            # and execute callbacks as needed
            time.sleep(0.1)
    
    def subscribe(self, topic, callback=None):
        """Subscribe to a topic"""
        with self.lock:
            if topic not in self.message_queues:
                self.message_queues[topic] = queue.Queue()
                self.callbacks[topic] = []
            
            if callback and callback not in self.callbacks[topic]:
                self.callbacks[topic].append(callback)
                logger.info(f"Subscribed to topic: {topic}")
            
            return self.message_queues[topic]
    
    def publish(self, topic, message):
        """Publish a message to a topic"""
        with self.lock:
            # Add timestamp to message
            if isinstance(message, dict):
                message['timestamp'] = datetime.now().isoformat()
            
            # Store in history
            self.message_history.append((topic, message))
            if len(self.message_history) > self.max_history:
                self.message_history.pop(0)
            
            # Publish to subscribers
            if topic in self.message_queues:
                self.message_queues[topic].put(message)
                
                # Execute callbacks
                for callback in self.callbacks.get(topic, []):
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in callback for topic {topic}: {e}")
            
            logger.debug(f"Published message to topic: {topic}")
    
    def get_history(self, topic=None, limit=100):
        """Get message history"""
        with self.lock:
            if topic:
                history = [(t, m) for t, m in self.message_history if t == topic]
            else:
                history = self.message_history
            
            return history[-limit:]
    
    def unsubscribe(self, topic, callback):
        """Unsubscribe from a topic"""
        with self.lock:
            if topic in self.callbacks and callback in self.callbacks[topic]:
                self.callbacks[topic].remove(callback)
                logger.info(f"Unsubscribed from topic: {topic}")

# Global instance cache
_data_bus_instance = None

@data_bus_bp.route('/api/data_bus/publish', methods=['POST'])
def publish_message():
    """Publish a message to the data bus"""
    data = request.json
    topic = data.get('topic')
    message = data.get('message')
    
    if not topic or message is None:
        return jsonify({'status': 'error', 'message': 'Missing topic or message'}), 400
    
    data_bus = get_data_bus()
    data_bus.publish(topic, message)
    
    return jsonify({'status': 'success', 'message': 'Message published'})

@data_bus_bp.route('/api/data_bus/history', methods=['GET'])
def get_message_history():
    """Get message history"""
    topic = request.args.get('topic')
    limit = int(request.args.get('limit', 100))
    
    data_bus = get_data_bus()
    history = data_bus.get_history(topic, limit)
    
    return jsonify({
        'status': 'success',
        'history': history
    })

@data_bus_bp.route('/api/data_bus/channels', methods=['GET'])
def get_available_channels():
    """Get list of available channels"""
    data_bus = get_data_bus()
    return jsonify({
        'status': 'success',
        'channels': list(data_bus.message_queues.keys())
    })

def get_data_bus():
    """Get a singleton instance of DataBus"""
    global _data_bus_instance
    if _data_bus_instance is None:
        _data_bus_instance = DataBus()
        _data_bus_instance.start()
        logger.info("DataBus initialized")
    return _data_bus_instance