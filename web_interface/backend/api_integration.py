#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Integration Module
This module handles the integration of external AI services with the Self Brain system
"""

import logging
import json
import os
from datetime import datetime
import threading
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('APIIntegration')

# Import the external API configuration manager
from .external_api_config import get_external_api_config

class APIIntegrationManager:
    """Manages the integration of external AI services with the Self Brain system"""
    
    def __init__(self):
        # Get the external API config manager
        self.external_api_config = get_external_api_config()
        
        # Model registry to keep track of all available models
        self.model_registry = {
            'A': {'type': 'management', 'name': 'Management Model'},
            'B': {'type': 'language', 'name': 'Language Model'},
            'C': {'type': 'audio', 'name': 'Audio Processing Model'},
            'D': {'type': 'image', 'name': 'Image Processing Model'},
            'E': {'type': 'video', 'name': 'Video Processing Model'},
            'F': {'type': 'spatial', 'name': 'Spatial Perception Model'},
            'G': {'type': 'sensor', 'name': 'Sensor Perception Model'},
            'H': {'type': 'computer', 'name': 'Computer Control Model'},
            'I': {'type': 'motion', 'name': 'Motion Control Model'},
            'J': {'type': 'knowledge', 'name': 'Knowledge Base Model'},
            'K': {'type': 'programming', 'name': 'Programming Model'}
        }
        
        # Local model instances (will be populated by the main application)
        self.local_models = {}
        
        # Request queue for API calls
        self.request_queue = []
        
        # Request processing thread
        self.processing_thread = None
        self.is_processing = False
        
        # Initialize the request processor
        self.start_request_processor()
    
    def register_local_model(self, model_id: str, model_instance: Any) -> bool:
        """Register a local model instance"""
        if model_id in self.model_registry:
            self.local_models[model_id] = model_instance
            logger.info(f"Registered local model: {model_id} - {self.model_registry[model_id]['name']}")
            return True
        else:
            logger.error(f"Failed to register local model: Invalid model ID '{model_id}'")
            return False
    
    def get_model_instance(self, model_id: str) -> Any:
        """Get the appropriate model instance (local or external) based on configuration"""
        # Check if the model is configured to use external API
        if self.external_api_config.is_model_using_external_api(model_id):
            # Return the external API handler
            return ExternalAPIHandler(model_id, self.external_api_config)
        else:
            # Return the local model if available
            return self.local_models.get(model_id)
    
    def process_request(self, model_id: str, request_data: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Process a request using the appropriate model"""
        try:
            # Validate model ID
            if model_id not in self.model_registry:
                logger.error(f"Invalid model ID: {model_id}")
                return None, f"Invalid model ID: {model_id}"
            
            # Get the model instance
            model = self.get_model_instance(model_id)
            
            if model is None:
                logger.error(f"No model instance available for {model_id}")
                return None, f"No model instance available for {model_id}"
            
            # Process the request based on model type
            if isinstance(model, ExternalAPIHandler):
                # Handle external API request
                return model.process_request(request_data)
            else:
                # Handle local model request
                return self._process_local_model_request(model_id, model, request_data)
            
        except Exception as e:
            logger.error(f"Error processing request for model {model_id}: {str(e)}")
            return None, str(e)
    
    def _process_local_model_request(self, model_id: str, model: Any, request_data: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Process a request using a local model"""
        try:
            model_type = self.model_registry[model_id]['type']
            
            # Process request based on model type
            if model_type == 'management':
                # Management model handling
                if hasattr(model, 'process_input'):
                    return model.process_input(request_data.get('input', '')), None
                else:
                    return None, "Management model missing process_input method"
                    
            elif model_type == 'language':
                # Language model handling
                if hasattr(model, 'generate_response'):
                    return model.generate_response(request_data.get('prompt', '')), None
                else:
                    return None, "Language model missing generate_response method"
                    
            elif model_type == 'audio':
                # Audio model handling
                if hasattr(model, 'process_audio'):
                    return model.process_audio(request_data), None
                else:
                    return None, "Audio model missing process_audio method"
                    
            elif model_type == 'image':
                # Image model handling
                if hasattr(model, 'process_image'):
                    return model.process_image(request_data), None
                else:
                    return None, "Image model missing process_image method"
                    
            elif model_type == 'video':
                # Video model handling
                if hasattr(model, 'process_video'):
                    return model.process_video(request_data), None
                else:
                    return None, "Video model missing process_video method"
                    
            elif model_type == 'spatial':
                # Spatial model handling
                if hasattr(model, 'process_stereo_data'):
                    return model.process_stereo_data(request_data), None
                else:
                    return None, "Spatial model missing process_stereo_data method"
                    
            elif model_type == 'sensor':
                # Sensor model handling
                if hasattr(model, 'read_sensors'):
                    return model.read_sensors(request_data.get('sensors', [])), None
                else:
                    return None, "Sensor model missing read_sensors method"
                    
            elif model_type == 'computer':
                # Computer control model handling
                if hasattr(model, 'execute_command'):
                    return model.execute_command(request_data.get('command', '')), None
                else:
                    return None, "Computer control model missing execute_command method"
                    
            elif model_type == 'motion':
                # Motion control model handling
                if hasattr(model, 'control_actuators'):
                    return model.control_actuators(request_data), None
                else:
                    return None, "Motion control model missing control_actuators method"
                    
            elif model_type == 'knowledge':
                # Knowledge model handling
                if hasattr(model, 'query_knowledge'):
                    return model.query_knowledge(request_data.get('query', '')), None
                else:
                    return None, "Knowledge model missing query_knowledge method"
                    
            elif model_type == 'programming':
                # Programming model handling
                if hasattr(model, 'generate_code'):
                    return model.generate_code(request_data.get('prompt', '')), None
                else:
                    return None, "Programming model missing generate_code method"
                    
            else:
                return None, f"Unsupported model type: {model_type}"
                
        except Exception as e:
            logger.error(f"Error processing local model request: {str(e)}")
            return None, str(e)
    
    def start_request_processor(self):
        """Start the background request processor"""
        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._request_processor_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Started API request processor")
    
    def stop_request_processor(self):
        """Stop the background request processor"""
        if self.is_processing:
            self.is_processing = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            logger.info("Stopped API request processor")
    
    def _request_processor_loop(self):
        """Background loop for processing requests"""
        import time
        
        while self.is_processing:
            try:
                if self.request_queue:
                    # Get the next request
                    request_id, model_id, request_data, callback = self.request_queue.pop(0)
                    
                    # Process the request
                    result, error = self.process_request(model_id, request_data)
                    
                    # Call the callback with the result
                    if callback:
                        callback(request_id, result, error)
                else:
                    # Sleep for a short time if the queue is empty
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in request processor loop: {str(e)}")
                # Sleep for a short time before retrying
                time.sleep(0.5)
    
    def queue_request(self, model_id: str, request_data: Dict[str, Any], callback=None) -> str:
        """Queue a request for background processing"""
        # Generate a unique request ID
        request_id = f"req_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{os.urandom(4).hex()}"
        
        # Add the request to the queue
        self.request_queue.append((request_id, model_id, request_data, callback))
        
        logger.info(f"Queued request {request_id} for model {model_id}")
        return request_id
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get the status of a specific model"""
        status = {
            'model_id': model_id,
            'model_name': self.model_registry.get(model_id, {}).get('name', 'Unknown'),
            'model_type': self.model_registry.get(model_id, {}).get('type', 'unknown'),
            'is_local': model_id in self.local_models,
            'is_external': self.external_api_config.is_model_using_external_api(model_id),
            'status_time': datetime.now().isoformat()
        }
        
        # Add local model status if available
        if model_id in self.local_models:
            model = self.local_models[model_id]
            if hasattr(model, 'get_status'):
                try:
                    local_status = model.get_status()
                    status['local_status'] = local_status
                except Exception as e:
                    status['local_status'] = {'error': str(e)}
        
        # Add external API status if configured
        if self.external_api_config.is_model_using_external_api(model_id):
            api_status = self.external_api_config.get_connection_status(model_id)
            status['external_status'] = api_status
            
            # Add provider and model information
            model_config = self.external_api_config.get_model_external_config(model_id)
            status['external_config'] = model_config
        
        return status
    
    def get_all_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all models"""
        statuses = {}
        for model_id in self.model_registry:
            statuses[model_id] = self.get_model_status(model_id)
        return statuses
    
    def configure_external_api(self, model_id: str, provider_name: str, model_name: str, api_key: str = None) -> bool:
        """Configure external API for a model"""
        return self.external_api_config.configure_model_external_api(model_id, provider_name, model_name, api_key)
    
    def enable_external_api(self, model_id: str) -> bool:
        """Enable external API for a model"""
        return self.external_api_config.enable_model_external_api(model_id)
    
    def disable_external_api(self, model_id: str) -> bool:
        """Disable external API for a model"""
        return self.external_api_config.disable_model_external_api(model_id)
    
    def test_model_connection(self, model_id: str) -> Dict[str, Any]:
        """Test the connection to an external API model"""
        if self.external_api_config.is_model_using_external_api(model_id):
            success = self.external_api_config.test_connection(model_id)
            status = self.external_api_config.get_connection_status(model_id)
            return {
                'success': success,
                'status': status
            }
        else:
            return {
                'success': False,
                'status': {'message': 'Model is not configured to use external API'}
            }
    
    def get_supported_providers(self) -> List[str]:
        """Get a list of supported API providers"""
        return self.external_api_config.get_supported_providers()
    
    def get_provider_models(self, provider_name: str) -> List[str]:
        """Get a list of available models for a specific provider"""
        return self.external_api_config.get_supported_models(provider_name)
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_request_processor()
        
        # Cleanup external API config
        if hasattr(self.external_api_config, 'cleanup'):
            self.external_api_config.cleanup()

class ExternalAPIHandler:
    """Handler for external API requests"""
    
    def __init__(self, model_id: str, api_config):
        self.model_id = model_id
        self.api_config = api_config
        
        # Get model configuration
        self.model_config = self.api_config.get_model_external_config(model_id)
        
        # Get model type
        self.model_type = self.api_config.get_model_type(model_id)
        
        logger.info(f"Initialized ExternalAPIHandler for model {model_id} (type: {self.model_type})")
    
    def process_request(self, request_data: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Process a request using the external API"""
        try:
            # Validate that the model is enabled for external API
            if not self.api_config.is_model_using_external_api(self.model_id):
                return None, "Model is not configured to use external API"
            
            # Format the request based on model type
            formatted_request = self._format_request_based_on_model_type(request_data)
            
            if not formatted_request:
                return None, "Failed to format request based on model type"
            
            # Make the API call
            result, error = self.api_config.call_external_api(self.model_id, formatted_request)
            
            if error:
                return None, error
            
            # Process the result based on model type
            processed_result = self._process_result_based_on_model_type(result)
            
            return processed_result, None
            
        except Exception as e:
            logger.error(f"Error processing external API request: {str(e)}")
            return None, str(e)
    
    def _format_request_based_on_model_type(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format the request data based on the model type"""
        try:
            if self.model_type == 'management':
                # Management model request format
                return {
                    'messages': [
                        {'role': 'system', 'content': 'You are a management AI assistant.'},
                        {'role': 'user', 'content': request_data.get('input', '')}
                    ],
                    'max_tokens': request_data.get('max_tokens', 2000),
                    'temperature': request_data.get('temperature', 0.7)
                }
                
            elif self.model_type == 'language':
                # Language model request format
                return {
                    'messages': request_data.get('messages', [
                        {'role': 'user', 'content': request_data.get('prompt', '')}
                    ]),
                    'max_tokens': request_data.get('max_tokens', 1000),
                    'temperature': request_data.get('temperature', 0.7)
                }
                
            elif self.model_type == 'knowledge':
                # Knowledge model request format
                return {
                    'messages': [
                        {'role': 'system', 'content': 'You are a knowledgeable expert assistant.'},
                        {'role': 'user', 'content': request_data.get('query', '')}
                    ],
                    'max_tokens': request_data.get('max_tokens', 2000),
                    'temperature': request_data.get('temperature', 0.5)
                }
                
            elif self.model_type == 'programming':
                # Programming model request format
                return {
                    'messages': [
                        {'role': 'system', 'content': 'You are a programming assistant.'},
                        {'role': 'user', 'content': request_data.get('prompt', '')}
                    ],
                    'max_tokens': request_data.get('max_tokens', 3000),
                    'temperature': request_data.get('temperature', 0.6)
                }
                
            # For other model types, we'll use a default format for now
            # These would need more specific handling based on the actual API capabilities
            return {
                'messages': [
                    {'role': 'user', 'content': request_data.get('input', '')}
                ],
                'max_tokens': request_data.get('max_tokens', 1000),
                'temperature': request_data.get('temperature', 0.7)
            }
            
        except Exception as e:
            logger.error(f"Error formatting request: {str(e)}")
            return None
    
    def _process_result_based_on_model_type(self, result: Dict[str, Any]) -> Any:
        """Process the API result based on the model type"""
        try:
            # Extract the content from the result
            content = result.get('content', '')
            
            # Process based on model type
            if self.model_type == 'management':
                # Management model result processing
                return {
                    'response': content,
                    'model_info': {
                        'provider': result.get('provider'),
                        'model': result.get('model')
                    },
                    'usage': result.get('usage')
                }
                
            elif self.model_type == 'language':
                # Language model result processing
                return {
                    'response': content,
                    'model_info': {
                        'provider': result.get('provider'),
                        'model': result.get('model')
                    },
                    'usage': result.get('usage')
                }
                
            elif self.model_type == 'knowledge':
                # Knowledge model result processing
                return {
                    'answer': content,
                    'model_info': {
                        'provider': result.get('provider'),
                        'model': result.get('model')
                    },
                    'usage': result.get('usage')
                }
                
            elif self.model_type == 'programming':
                # Programming model result processing
                return {
                    'code': content,
                    'model_info': {
                        'provider': result.get('provider'),
                        'model': result.get('model')
                    },
                    'usage': result.get('usage')
                }
                
            # Default result processing for other model types
            return {
                'result': content,
                'model_info': {
                    'provider': result.get('provider'),
                    'model': result.get('model')
                },
                'usage': result.get('usage')
            }
            
        except Exception as e:
            logger.error(f"Error processing result: {str(e)}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the external API handler"""
        connection_status = self.api_config.get_connection_status(self.model_id)
        
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'connection_status': connection_status,
            'config': self.model_config,
            'last_checked': datetime.now().isoformat()
        }

# Create a singleton instance of the API integration manager
global_api_integration_manager = None

def get_api_integration_manager():
    """Get the singleton instance of the API integration manager"""
    global global_api_integration_manager
    if global_api_integration_manager is None:
        global_api_integration_manager = APIIntegrationManager()
    return global_api_integration_manager

# Initialize the integration manager when this module is loaded
if __name__ == "__main__":
    # For testing purposes
    api_manager = APIIntegrationManager()
    print("API Integration Manager initialized successfully")
    print(f"Supported providers: {api_manager.get_supported_providers()}")