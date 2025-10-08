#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Model (A Management Model) Implementation
This is the core management model that coordinates all other models and handles human interaction
"""

import logging
import json
import time
import threading
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict

# Import model registry, data bus, and emotion components
from manager_model.model_registry import ModelRegistry
from manager_model.data_bus import DataBus, get_data_bus
from manager_model.emotion_engine_fixed import EmotionalState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MainModel')

class ManagementModel(nn.Module):
    """Main management model implementation with emotion capabilities"""
    
    def __init__(self, submodel_registry=None):
        super().__init__()
        
        # Initialize core components
        self.model_registry = submodel_registry  # 使用外部传入的模型注册表
        self.data_bus = get_data_bus()
        self.emotion_engine = None  # 将在app.py中初始化
        
        # Model configuration
        self.model_id = 'A'
        self.name = 'Management Model'
        self.description = 'Main control model that coordinates all other models and handles human interaction'
        
        # Neural network architecture
        self.hidden_size = 512
        self.num_layers = 2
        self.attention_heads = 8
        
        # Input features count (based on all possible inputs)
        self.input_features = 1024
        
        # Initialize network layers
        self.initialize_network()
        
        # Training configuration
        self.lr = 0.001
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        # Memory and context
        self.context_window = deque(maxlen=100)
        self.session_memory = {}
        
        # Model status
        self.is_initialized = False
        self.is_training = False
        self.training_progress = 0
        self.performance_metrics = defaultdict(float)
        
        # Start initialization
        self.initialize()
    
    def initialize_network(self):
        """Initialize the neural network architecture"""
        # Input embedding layer
        self.input_embedding = nn.Linear(self.input_features, self.hidden_size)
        
        # Main processing layers
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.attention_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.1
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output layers
        self.output_layer = nn.Linear(self.hidden_size, self.input_features)
        
        # Emotion processing branch
        self.emotion_layer = nn.Linear(self.hidden_size, 256)
        self.emotion_output = nn.Linear(256, 8)  # 8 basic emotions
        
        # Task routing branch
        self.routing_layer = nn.Linear(self.hidden_size, 128)
        self.routing_output = nn.Linear(128, 11)  # 11 models (A-K)
    
    def initialize(self):
        """Initialize the model and connect to all sub-models"""
        try:
            logger.info(f"Initializing {self.name}...")
            
            # Load pre-trained weights if available
            self.load_weights()
            
            # Connect to all sub-models
            self.connect_sub_models()
            
            # Register message handlers
            self.register_handlers()
            
            self.is_initialized = True
            logger.info(f"{self.name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {str(e)}")
            self.is_initialized = False
    
    def connect_sub_models(self):
        """Connect to all registered sub-models"""
        # Get all registered models except self
        all_models = self.model_registry.get_all_models()
        
        for model_id, model_config in all_models.items():
            if model_id != self.model_id:
                try:
                    # Connect to the model through the data bus
                    self.data_bus.subscribe(f"model_{model_id}_output", self.handle_model_output)
                    logger.info(f"Connected to model {model_id}: {model_config.get('name', 'Unknown')}")
                except Exception as e:
                    logger.error(f"Failed to connect to model {model_id}: {str(e)}")
    
    def register_handlers(self):
        """Register message handlers for incoming data"""
        # Register handler for user input
        self.data_bus.subscribe("user_input", self.handle_user_input)
        
        # Register handler for system status
        self.data_bus.subscribe("system_status", self.handle_system_status)
    
    def handle_user_input(self, data):
        """Handle incoming user input"""
        try:
            # Extract user message
            message = data.get('message', '')
            session_id = data.get('session_id', 'default')
            timestamp = data.get('timestamp', datetime.now().isoformat())
            
            if not message:
                return
            
            # Log the input
            logger.info(f"Received user input: {message[:50]}... (session: {session_id})")
            
            # Store in context window
            self.context_window.append({
                'type': 'user',
                'content': message,
                'session_id': session_id,
                'timestamp': timestamp
            })
            
            # Process the input and generate response
            response = self.process_user_input(message, session_id)
            
            # Send response back to the user
            self.data_bus.publish("model_output", {
                'model_id': self.model_id,
                'session_id': session_id,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Store response in context window
            self.context_window.append({
                'type': 'model',
                'content': response,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling user input: {str(e)}")
            # Send error response
            self.data_bus.publish("model_output", {
                'model_id': self.model_id,
                'session_id': data.get('session_id', 'default'),
                'response': "I'm sorry, I encountered an error processing your request.",
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
    
    def process_user_input(self, message, session_id):
        """Process user input and generate appropriate response"""
        try:
            # Analyze emotion in the user's message
            emotion = self.emotion_engine.analyze_text(message)
            
            # Determine which model should handle this request
            target_model_id = self.determine_target_model(message, emotion)
            
            # Create a request for the target model
            request_data = {
                'request_id': f"req_{int(time.time() * 1000)}",
                'source': self.model_id,
                'session_id': session_id,
                'message': message,
                'context': list(self.context_window)[-5:],  # Last 5 messages as context
                'emotion': emotion,
                'timestamp': datetime.now().isoformat()
            }
            
            # If the target is this model (self), process locally
            if target_model_id == self.model_id:
                return self.process_local_request(request_data)
            
            # Otherwise, forward to the appropriate model
            response = self.forward_to_model(target_model_id, request_data)
            
            # Enhance response with emotional context
            enhanced_response = self.enhance_with_emotion(response, emotion)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return f"I'm sorry, I couldn't process your request properly. Error: {str(e)}"
    
    def determine_target_model(self, message, emotion):
        """Determine which model should handle the request based on content and emotion"""
        # Convert message to tensor for processing
        message_tensor = torch.tensor([ord(c) for c in message[:100]]).float().unsqueeze(0)
        
        # Pad to input features length
        if len(message_tensor[0]) < self.input_features:
            padding = torch.zeros((1, self.input_features - len(message_tensor[0])))
            message_tensor = torch.cat([message_tensor, padding], dim=1)
        else:
            message_tensor = message_tensor[:, :self.input_features]
        
        # Forward pass through the network
        with torch.no_grad():
            embedded = self.input_embedding(message_tensor)
            output = self.transformer_encoder(embedded.unsqueeze(0))
            routing_logits = self.routing_output(self.routing_layer(output.squeeze(0)))
        
        # Get the highest scoring model
        model_indices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        _, predicted_idx = torch.max(routing_logits, dim=1)
        
        # Default to model J (knowledge base) if uncertain
        confidence = torch.softmax(routing_logits, dim=1).max().item()
        if confidence < 0.3:
            target_model = 'J'
        else:
            target_idx = predicted_idx.item()
            if 0 <= target_idx < len(model_indices):
                target_model = model_indices[target_idx]
            else:
                target_model = 'J'
        
        logger.info(f"Routing request to model {target_model} with confidence {confidence:.2f}")
        return target_model
    
    def forward_to_model(self, model_id, request_data):
        """Forward request to the specified model and wait for response"""
        try:
            # Create a response event
            response_event = threading.Event()
            response_data = {}
            
            # Define a callback for the response
            def response_callback(data):
                nonlocal response_data
                response_data = data
                response_event.set()
            
            # Subscribe to the model's response channel
            response_channel = f"model_{model_id}_response_{request_data['request_id']}"
            self.data_bus.subscribe(response_channel, response_callback)
            
            # Publish the request to the model
            self.data_bus.publish(f"model_{model_id}_request", request_data)
            
            # Wait for response or timeout
            timeout = 30.0  # 30 seconds timeout
            if response_event.wait(timeout=timeout):
                # Unsubscribe from the response channel
                self.data_bus.unsubscribe(response_channel, response_callback)
                
                # Extract the response content
                return response_data.get('response', 'I received no response from the processing model.')
            else:
                # Timeout occurred
                logger.warning(f"Timeout waiting for response from model {model_id}")
                self.data_bus.unsubscribe(response_channel, response_callback)
                return f"I'm sorry, but the {model_id} model is taking too long to respond. Please try again later."
                
        except Exception as e:
            logger.error(f"Error forwarding to model {model_id}: {str(e)}")
            return f"I encountered an error while processing your request with the {model_id} model."
    
    def process_local_request(self, request_data):
        """Process requests that should be handled locally"""
        message = request_data.get('message', '').lower()
        
        # Handle basic system commands
        if any(cmd in message for cmd in ['system status', 'status', 'how are you']):
            return self.get_system_status_summary()
        elif any(cmd in message for cmd in ['help', 'what can you do']):
            return self.get_help_message()
        elif any(cmd in message for cmd in ['thank', 'thanks']):
            return "You're welcome! I'm here to help with any questions or tasks you have."
        
        # Default response for management model
        return "I'm processing your request as the main management model. How can I assist you further?"
    
    def enhance_with_emotion(self, response, emotion):
        """Enhance the response with appropriate emotional context"""
        if not hasattr(self, 'emotion_engine') or self.emotion_engine is None:
            return response
        
        try:
            # 如果传入的是情感对象，直接使用
            if isinstance(emotion, EmotionalState):
                emotional_response = self.emotion_engine.generate_emotional_response("", emotion)
            # 如果传入的是情感字符串（从app.py传入的primary emotion），创建一个简单的情感状态
            elif isinstance(emotion, str):
                emotional_state = EmotionalState()
                if emotion in emotional_state.emotions:
                    emotional_state.emotions[emotion] = 1.0
                emotional_response = self.emotion_engine.generate_emotional_response("", emotional_state)
            else:
                return response
            
            # 应用情感修饰符到响应
            if emotional_response and 'prefix' in emotional_response:
                enhanced = f"{emotional_response['prefix']} {response}"
                return enhanced.strip()
            
        except Exception as e:
            logger.error(f"Error enhancing response with emotion: {str(e)}")
            
        return response
    
    def handle_model_output(self, data):
        """Handle output from sub-models"""
        # This method can be expanded to process and react to sub-model outputs
        logger.debug(f"Received output from sub-model: {data}")
    
    def handle_system_status(self, data):
        """Handle system status updates"""
        # Update internal system status
        self.performance_metrics.update(data.get('metrics', {}))
    
    def get_system_status_summary(self):
        """Generate a summary of the current system status"""
        # Count active models from registry
        if self.model_registry:
            active_count = sum(1 for model in self.model_registry.values() if model.get('status') == 'active')
            total_count = len(self.model_registry)
        else:
            active_count = 0
            total_count = 0
        
        # Get emotion status if emotion engine is available
        if self.emotion_engine:
            try:
                emotion_status = self.emotion_engine.get_current_emotion()
                emotion_text = f"{emotion_status.get('primary', 'neutral')}"
            except:
                emotion_text = "neutral"
        else:
            emotion_text = "neutral"
        
        # Build summary
        summary = f"System Status:\n"
        summary += f"- Models: {active_count}/{total_count} active\n"
        summary += f"- Status: {'active' if self.is_initialized else 'initializing'}\n"
        summary += f"- My emotional state: {emotion_text}"
        
        return summary
    
    def get_help_message(self):
        """Generate a help message explaining the system's capabilities"""
        help_msg = "I am Self Brain, an advanced AI management system. I can help you with:\n"
        help_msg += "- Answering questions through my knowledge base\n"
        help_msg += "- Processing and generating text, images, and audio\n"
        help_msg += "- Understanding and responding to your emotions\n"
        help_msg += "- Managing and controlling external devices\n"
        help_msg += "- Providing insights and analysis on various topics\n"
        help_msg += "Just ask me anything, and I'll do my best to assist you!"
        
        return help_msg
    
    def train(self, training_data, epochs=10, batch_size=32, learning_rate=0.001):
        """Train the management model with the provided data"""
        try:
            self.is_training = True
            self.training_progress = 0
            self.lr = learning_rate
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
            
            # Convert training data to tensors
            inputs, targets = self.prepare_training_data(training_data)
            
            # Training loop
            for epoch in range(epochs):
                # Shuffle data
                permutation = torch.randperm(inputs.size()[0])
                inputs = inputs[permutation]
                targets = targets[permutation]
                
                # Mini-batch training
                for i in range(0, inputs.size()[0], batch_size):
                    batch_inputs = inputs[i:i+batch_size]
                    batch_targets = targets[i:i+batch_size]
                    
                    # Forward pass
                    outputs = self(batch_inputs)
                    loss = self.criterion(outputs, batch_targets)
                    
                    # Backward pass and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Update progress
                self.training_progress = int(((epoch + 1) / epochs) * 100)
                
                # Log progress
                if (epoch + 1) % 1 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
            # Save trained weights
            self.save_weights()
            
            self.is_training = False
            return {
                'status': 'success',
                'message': f'Training completed successfully in {epochs} epochs',
                'loss': loss.item()
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.is_training = False
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def prepare_training_data(self, training_data):
        """Prepare training data for the model"""
        inputs = []
        targets = []
        
        for item in training_data:
            # Process input message
            input_msg = item.get('input', '')
            input_tensor = torch.tensor([ord(c) for c in input_msg[:100]]).float()
            
            # Pad to input features length
            if len(input_tensor) < self.input_features:
                padding = torch.zeros(self.input_features - len(input_tensor))
                input_tensor = torch.cat([input_tensor, padding])
            else:
                input_tensor = input_tensor[:self.input_features]
            
            # Process target (expected output)
            target_msg = item.get('target', '')
            target_tensor = torch.tensor([ord(c) for c in target_msg[:100]]).float()
            
            # Pad to input features length
            if len(target_tensor) < self.input_features:
                padding = torch.zeros(self.input_features - len(target_tensor))
                target_tensor = torch.cat([target_tensor, padding])
            else:
                target_tensor = target_tensor[:self.input_features]
            
            inputs.append(input_tensor)
            targets.append(target_tensor)
        
        # Convert to tensors
        inputs_tensor = torch.stack(inputs)
        targets_tensor = torch.stack(targets)
        
        return inputs_tensor, targets_tensor
    
    def forward(self, x):
        """Forward pass of the neural network"""
        embedded = self.input_embedding(x)
        output = self.transformer_encoder(embedded.unsqueeze(0) if len(x.shape) == 2 else embedded)
        return self.output_layer(output.squeeze(0) if len(output.shape) == 3 else output)
    
    def save_weights(self, path=None):
        """Save the model weights"""
        try:
            if path is None:
                path = f'./models/management_model_weights.pth'
            
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            torch.save(self.state_dict(), path)
            logger.info(f"Model weights saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model weights: {str(e)}")
            return False
    
    def load_weights(self, path=None):
        """Load model weights from file"""
        try:
            if path is None:
                path = f'./models/management_model_weights.pth'
            
            import os
            if os.path.exists(path):
                self.load_state_dict(torch.load(path))
                logger.info(f"Model weights loaded from {path}")
                return True
            else:
                logger.info(f"No pre-trained weights found at {path}, using randomly initialized weights")
                return False
        except Exception as e:
            logger.error(f"Failed to load model weights: {str(e)}")
            return False
    
    def evaluate(self, test_data):
        """Evaluate the model on test data"""
        try:
            self.eval()
            
            inputs, targets = self.prepare_training_data(test_data)
            
            with torch.no_grad():
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
            
            # Calculate additional metrics
            mse = loss.item()
            rmse = np.sqrt(mse)
            
            self.train()  # Return to training mode
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'samples': len(test_data)
            }
            
            logger.info(f"Model evaluation: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

# Singleton instance of the management model
global_management_model = None

def get_management_model(submodel_registry=None):
    """Get the singleton instance of the management model"""
    global global_management_model
    if global_management_model is None:
        global_management_model = ManagementModel(submodel_registry)
    return global_management_model

# Initialize the model when this module is loaded
if __name__ == "__main__":
    # For testing purposes
    main_model = ManagementModel()
    print("Management Model initialized successfully")
    
    # Example interaction
    response = main_model.process_user_input("Hello, how are you?", "test_session")
    print(f"Response: {response}")