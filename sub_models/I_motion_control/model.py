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

# Motion Control Model Implementation

import torch
import torch.nn as nn
import numpy as np
import logging
import threading
import time
import json
import os
import serial
import socket
from typing import Dict, List, Any, Optional, Tuple
from ..unified_model_manager import get_model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("I_motion_control_Model")

class MotionControlModel(nn.Module):
    """Motion Control Model for controlling external devices"""
    
    def __init__(self, input_features: int = 8, sequence_length: int = 10, num_control_types: int = 12):
        """Initialize motion control model
        
        Args:
            input_features: Number of input features
            sequence_length: Length of input sequence
            num_control_types: Number of control types
        """
        super(MotionControlModel, self).__init__()
        
        # Model architecture
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.num_control_types = num_control_types
        
        # LSTM layers for sequence processing
        self.lstm1 = nn.LSTM(input_size=input_features, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        
        # Output layers
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_control_types)
        
        # Model management
        self.model_manager = get_model_manager()
        self.data_bus = self._get_data_bus()
        self.is_initialized = False
        self.is_self_learning = False
        self.self_learning_thread = None
        self.servers = {}
        self.connections = {}
        
        # Communication interfaces
        self.serial_ports = {}
        self.tcp_sockets = {}
        
        # External device configuration
        self.device_configs = {}
        self.initialized_devices = set()
        
        # Initialize model
        self.initialize()
    
    def _get_data_bus(self):
        """Get or create a simple data bus for inter-model communication"""
        # Simple implementation of a data bus for communication
        class SimpleDataBus:
            def __init__(self):
                self.messages = []
                self.lock = threading.Lock()
                self.callbacks = {}
                
            def send(self, sender: str, message_type: str, data: Any):
                """Send a message to the data bus"""
                with self.lock:
                    message = {
                        'sender': sender,
                        'type': message_type,
                        'data': data,
                        'timestamp': time.time()
                    }
                    self.messages.append(message)
                    
                    # Trigger callbacks if registered
                    if message_type in self.callbacks:
                        for callback in self.callbacks[message_type]:
                            try:
                                callback(message)
                            except Exception as e:
                                logger.error(f"Error in data bus callback: {str(e)}")
                
            def register_callback(self, message_type: str, callback):
                """Register a callback for a specific message type"""
                with self.lock:
                    if message_type not in self.callbacks:
                        self.callbacks[message_type] = []
                    self.callbacks[message_type].append(callback)
                
            def get_messages(self, message_type: Optional[str] = None, max_messages: int = 100):
                """Get messages from the data bus"""
                with self.lock:
                    if message_type:
                        filtered_messages = [msg for msg in self.messages if msg['type'] == message_type]
                        return filtered_messages[-max_messages:]
                    return self.messages[-max_messages:]
        
        return SimpleDataBus()
    
    def initialize(self):
        """Initialize the motion control model"""
        try:
            # Register model with the manager
            if self.model_manager:
                self.model_manager._models['motion_control'] = self
                logger.info("Motion control model registered with UnifiedModelManager")
            
            # Load device configurations
            self._load_device_configs()
            
            # Initialize communication interfaces
            self._init_communication_interfaces()
            
            # Register message handlers
            self._register_message_handlers()
            
            self.is_initialized = True
            logger.info("Motion control model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing motion control model: {str(e)}")
            self.is_initialized = False
    
    def _load_device_configs(self):
        """Load device configurations from config file"""
        config_path = "config/motion_control_devices.json"
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.device_configs = json.load(f)
            else:
                # Create default device configurations
                self.device_configs = {
                    "serial_devices": [
                        {
                            "id": "arduino_1",
                            "port": "COM3",
                            "baudrate": 9600,
                            "timeout": 0.1
                        }
                    ],
                    "tcp_devices": [
                        {
                            "id": "robot_arm_1",
                            "host": "192.168.1.100",
                            "port": 5000
                        }
                    ],
                    "device_types": {
                        "arduino": "microcontroller",
                        "robot_arm": "robotic_manipulator",
                        "servo_controller": "actuator_controller",
                        "motor_driver": "motion_controller"
                    }
                }
                # Save default configs
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.device_configs, f, indent=2)
                logger.info(f"Default device configurations saved to {config_path}")
        except Exception as e:
            logger.error(f"Error loading device configurations: {str(e)}")
    
    def _init_communication_interfaces(self):
        """Initialize communication interfaces"""
        # Initialize serial ports
        for device in self.device_configs.get("serial_devices", []):
            try:
                ser = serial.Serial(
                    port=device["port"],
                    baudrate=device["baudrate"],
                    timeout=device["timeout"]
                )
                if ser.is_open:
                    self.serial_ports[device["id"]] = ser
                    self.initialized_devices.add(device["id"])
                    logger.info(f"Serial device {device['id']} initialized on port {device['port']}")
            except Exception as e:
                logger.error(f"Failed to initialize serial device {device['id']}: {str(e)}")
        
        # Initialize TCP connections
        for device in self.device_configs.get("tcp_devices", []):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                # Just create the socket, actual connection will be made when needed
                self.tcp_sockets[device["id"]] = {
                    "socket": sock,
                    "host": device["host"],
                    "port": device["port"],
                    "connected": False
                }
                logger.info(f"TCP socket for device {device['id']} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize TCP socket for device {device['id']}: {str(e)}")
    
    def _register_message_handlers(self):
        """Register message handlers for inter-model communication"""
        # Register handlers for different message types
        self.data_bus.register_callback("motion_command", self._handle_motion_command)
        self.data_bus.register_callback("device_status_request", self._handle_device_status_request)
        self.data_bus.register_callback("self_learning_command", self._handle_self_learning_command)
        
        # Register with knowledge base for self-learning
        if self.model_manager and 'knowledge' in self.model_manager._models:
            knowledge_model = self.model_manager._models['knowledge']
            self.data_bus.register_callback("knowledge_update", self._handle_knowledge_update)
            logger.info("Registered with knowledge base for self-learning")
    
    def _handle_motion_command(self, message: Dict):
        """Handle motion command messages"""
        try:
            command_data = message['data']
            device_id = command_data.get('device_id')
            command_type = command_data.get('command_type')
            parameters = command_data.get('parameters', {})
            
            if device_id and command_type:
                result = self.send_command(device_id, command_type, parameters)
                # Send response back
                self.data_bus.send(
                    sender="motion_control",
                    message_type="motion_command_response",
                    data={"request_id": message.get("request_id"), "result": result}
                )
        except Exception as e:
            logger.error(f"Error handling motion command: {str(e)}")
    
    def _handle_device_status_request(self, message: Dict):
        """Handle device status requests"""
        try:
            device_id = message['data'].get('device_id')
            status = self.get_device_status(device_id)
            self.data_bus.send(
                sender="motion_control",
                message_type="device_status_response",
                data={"request_id": message.get("request_id"), "status": status}
            )
        except Exception as e:
            logger.error(f"Error handling device status request: {str(e)}")
    
    def _handle_self_learning_command(self, message: Dict):
        """Handle self-learning commands"""
        try:
            command = message['data'].get('command')
            if command == "start":
                self.start_self_learning()
            elif command == "stop":
                self.stop_self_learning()
            elif command == "status":
                status = self.get_self_learning_status()
                self.data_bus.send(
                    sender="motion_control",
                    message_type="self_learning_status",
                    data={"status": status}
                )
        except Exception as e:
            logger.error(f"Error handling self-learning command: {str(e)}")
    
    def _handle_knowledge_update(self, message: Dict):
        """Handle knowledge updates from the knowledge base"""
        try:
            # This method is called when the knowledge base is updated
            # We can use this to trigger self-learning or update our internal models
            logger.info("Received knowledge update, considering for self-learning")
            # If self-learning is active, we might want to incorporate this new knowledge
            if self.is_self_learning:
                # TODO: Implement logic to incorporate new knowledge into self-learning
                pass
        except Exception as e:
            logger.error(f"Error handling knowledge update: {str(e)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_features]
            
        Returns:
            Output tensor of shape [batch_size, sequence_length, num_control_types]
        """
        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # Attention mechanism
        attn_out, _ = self.attention(out, out, out)
        out = out + attn_out  # Residual connection
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
    
    def send_command(self, device_id: str, command_type: str, parameters: Dict) -> Dict:
        """Send command to a motion control device
        
        Args:
            device_id: ID of the device
            command_type: Type of command
            parameters: Command parameters
            
        Returns:
            Dictionary with command execution result
        """
        try:
            # Check if device is initialized
            if device_id not in self.initialized_devices:
                # Try to initialize the device
                self._initialize_device(device_id)
                if device_id not in self.initialized_devices:
                    return {"success": False, "error": f"Device {device_id} not initialized"}
            
            # Format command
            command = self._format_command(device_id, command_type, parameters)
            
            # Send command based on device type
            if device_id in self.serial_ports:
                result = self._send_serial_command(device_id, command)
            elif device_id in self.tcp_sockets:
                result = self._send_tcp_command(device_id, command)
            else:
                result = {"success": False, "error": f"Unknown device {device_id}"}
            
            return result
        except Exception as e:
            logger.error(f"Error sending command to device {device_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _format_command(self, device_id: str, command_type: str, parameters: Dict) -> str:
        """Format command for specific device"""
        # This is a simplified implementation, actual formatting would depend on device protocol
        command_dict = {
            "cmd": command_type,
            "params": parameters,
            "timestamp": time.time()
        }
        return json.dumps(command_dict) + "\n"
    
    def _send_serial_command(self, device_id: str, command: str) -> Dict:
        """Send command over serial port"""
        try:
            ser = self.serial_ports[device_id]
            if ser.is_open:
                ser.write(command.encode())
                response = ser.readline().decode().strip()
                if response:
                    return {"success": True, "response": response}
                return {"success": True, "response": "Command sent"}
            else:
                return {"success": False, "error": "Serial port not open"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _send_tcp_command(self, device_id: str, command: str) -> Dict:
        """Send command over TCP socket"""
        try:
            tcp_info = self.tcp_sockets[device_id]
            sock = tcp_info["socket"]
            
            # Connect if not already connected
            if not tcp_info["connected"]:
                sock.connect((tcp_info["host"], tcp_info["port"]))
                tcp_info["connected"] = True
            
            # Send command
            sock.sendall(command.encode())
            
            # Receive response (with timeout)
            sock.settimeout(2.0)
            response = sock.recv(1024).decode().strip()
            
            return {"success": True, "response": response}
        except Exception as e:
            # If connection failed, mark as disconnected
            if device_id in self.tcp_sockets:
                self.tcp_sockets[device_id]["connected"] = False
            return {"success": False, "error": str(e)}
    
    def get_device_status(self, device_id: Optional[str] = None) -> Dict:
        """Get status of devices
        
        Args:
            device_id: Optional device ID to get status for specific device
            
        Returns:
            Dictionary with device status information
        """
        status = {}
        
        if device_id:
            # Get status for specific device
            if device_id in self.serial_ports:
                ser = self.serial_ports[device_id]
                status[device_id] = {
                    "type": "serial",
                    "connected": ser.is_open,
                    "port": ser.port,
                    "baudrate": ser.baudrate
                }
            elif device_id in self.tcp_sockets:
                tcp_info = self.tcp_sockets[device_id]
                status[device_id] = {
                    "type": "tcp",
                    "connected": tcp_info["connected"],
                    "host": tcp_info["host"],
                    "port": tcp_info["port"]
                }
            else:
                status[device_id] = {"error": "Unknown device"}
        else:
            # Get status for all devices
            for dev_id, ser in self.serial_ports.items():
                status[dev_id] = {
                    "type": "serial",
                    "connected": ser.is_open,
                    "port": ser.port,
                    "baudrate": ser.baudrate
                }
            
            for dev_id, tcp_info in self.tcp_sockets.items():
                status[dev_id] = {
                    "type": "tcp",
                    "connected": tcp_info["connected"],
                    "host": tcp_info["host"],
                    "port": tcp_info["port"]
                }
        
        return status
    
    def _initialize_device(self, device_id: str) -> bool:
        """Initialize a specific device"""
        try:
            # Check if device is in configs
            for device in self.device_configs.get("serial_devices", []):
                if device["id"] == device_id:
                    # Try to initialize serial device
                    ser = serial.Serial(
                        port=device["port"],
                        baudrate=device["baudrate"],
                        timeout=device["timeout"]
                    )
                    if ser.is_open:
                        self.serial_ports[device_id] = ser
                        self.initialized_devices.add(device_id)
                        logger.info(f"Serial device {device_id} initialized on port {device['port']}")
                        return True
                    break
            
            # Check TCP devices
            for device in self.device_configs.get("tcp_devices", []):
                if device["id"] == device_id:
                    # TCP devices are initialized on first use
                    # Just mark as initialized here
                    self.initialized_devices.add(device_id)
                    logger.info(f"TCP device {device_id} marked as initialized")
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"Error initializing device {device_id}: {str(e)}")
            return False
    
    def start_self_learning(self):
        """Start self-learning process"""
        if not self.is_self_learning:
            self.is_self_learning = True
            self.self_learning_thread = threading.Thread(target=self.self_learning_loop)
            self.self_learning_thread.daemon = True
            self.self_learning_thread.start()
            logger.info("Motion control model self-learning started")
        else:
            logger.info("Self-learning is already running")
    
    def stop_self_learning(self):
        """Stop self-learning process"""
        if self.is_self_learning:
            self.is_self_learning = False
            if self.self_learning_thread and self.self_learning_thread.is_alive():
                self.self_learning_thread.join(timeout=5.0)
            logger.info("Motion control model self-learning stopped")
        else:
            logger.info("Self-learning is not running")
    
    def get_self_learning_status(self) -> Dict:
        """Get status of self-learning process"""
        return {
            "running": self.is_self_learning,
            "thread_alive": self.self_learning_thread.is_alive() if self.self_learning_thread else False
        }
    
    def self_learning_loop(self):
        """Main loop for self-learning process"""
        logger.info("Starting motion control self-learning loop")
        
        while self.is_self_learning:
            try:
                # Get knowledge from knowledge base model
                if self.model_manager and 'knowledge' in self.model_manager._models:
                    knowledge_model = self.model_manager._models['knowledge']
                    
                    # Query relevant knowledge for motion control
                    motion_knowledge = knowledge_model.query_knowledge(
                        query="motion control techniques",
                        domain="robotics",
                        top_k=5
                    )
                    
                    # Process the acquired knowledge
                    if motion_knowledge and "results" in motion_knowledge:
                        for item in motion_knowledge["results"]:
                            # Learn from the knowledge item
                            self._learn_from_knowledge(item)
                
                # Also learn from device feedback and performance
                self._learn_from_device_feedback()
                
                # Sleep for a while before next iteration
                time.sleep(30)  # Sleep for 30 seconds
                
            except Exception as e:
                logger.error(f"Error in self-learning loop: {str(e)}")
                time.sleep(5)  # Sleep shorter time if there's an error
    
    def _learn_from_knowledge(self, knowledge_item: Dict):
        """Learn from a knowledge item acquired from the knowledge base"""
        try:
            # Extract relevant information from the knowledge item
            title = knowledge_item.get("title", "")
            content = knowledge_item.get("content", "")
            confidence = knowledge_item.get("confidence", 0.0)
            
            # If confidence is high enough, process the knowledge
            if confidence > 0.7:
                logger.info(f"Learning from knowledge: {title}")
                
                # This is a placeholder for actual learning logic
                # In a real implementation, this would update model weights or adjust control parameters
                # based on the new knowledge
                
                # For now, we'll just log that we're learning
                # TODO: Implement actual learning logic
                
                # Send a message that we've learned something
                self.data_bus.send(
                    sender="motion_control",
                    message_type="learning_update",
                    data={
                        "knowledge_title": title,
                        "action": "acquired_knowledge"
                    }
                )
        except Exception as e:
            logger.error(f"Error learning from knowledge: {str(e)}")
    
    def _learn_from_device_feedback(self):
        """Learn from feedback received from devices"""
        try:
            # Get status of all devices
            devices_status = self.get_device_status()
            
            # Analyze device performance
            for device_id, status in devices_status.items():
                # This is a placeholder for actual learning from device feedback
                # In a real implementation, this would analyze device response times,
                # error rates, and other performance metrics to improve control algorithms
                
                # For now, we'll just log that we're analyzing device feedback
                # TODO: Implement actual device feedback learning logic
                pass
        except Exception as e:
            logger.error(f"Error learning from device feedback: {str(e)}")
    
    def save_weights(self, path: str = "models/motion_control_model_weights.pth"):
        """Save model weights"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.state_dict(), path)
            logger.info(f"Motion control model weights saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model weights: {str(e)}")
    
    def load_weights(self, path: str = "models/motion_control_model_weights.pth"):
        """Load model weights"""
        try:
            if os.path.exists(path):
                self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                logger.info(f"Motion control model weights loaded from {path}")
            else:
                logger.warning(f"Model weights file not found: {path}")
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
    
    def close(self):
        """Close all resources"""
        try:
            # Stop self-learning
            self.stop_self_learning()
            
            # Close serial ports
            for device_id, ser in self.serial_ports.items():
                if ser.is_open:
                    ser.close()
                    logger.info(f"Serial port for device {device_id} closed")
            
            # Close TCP sockets
            for device_id, tcp_info in self.tcp_sockets.items():
                if tcp_info["socket"]:
                    tcp_info["socket"].close()
                    logger.info(f"TCP socket for device {device_id} closed")
            
            logger.info("Motion control model resources closed")
        except Exception as e:
            logger.error(f"Error closing model resources: {str(e)}")

# Global instance for singleton pattern
global_motion_model = None


def get_motion_model():
    """Get global motion control model instance"""
    global global_motion_model
    if global_motion_model is None:
        global_motion_model = MotionControlModel()
    return global_motion_model

# Test code if run directly
if __name__ == "__main__":
    try:
        # Initialize model
        model = MotionControlModel()
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, model.sequence_length, model.input_features)
        output = model(dummy_input)
        print(f"Forward pass test successful. Output shape: {output.shape}")
        
        # Test device status
        status = model.get_device_status()
        print(f"Device status: {json.dumps(status, indent=2)}")
        
        # Test self-learning control
        print("Testing self-learning control...")
        model.start_self_learning()
        time.sleep(2)
        sl_status = model.get_self_learning_status()
        print(f"Self-learning status: {sl_status}")
        model.stop_self_learning()
        
        print("Motion control model test completed")
    except Exception as e:
        print(f"Error during model test: {str(e)}")