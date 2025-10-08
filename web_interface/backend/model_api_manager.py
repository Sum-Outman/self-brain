#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model API Manager - Handles external API connections and configurations for individual models
Provides functionality to switch between local and external API implementations
and manage API connections for different providers.
"""

import json
import logging
import os
import requests
timeout = 30  # Default timeout for API requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAPIManager:
    """Manager for handling model API configurations and connections"""
    
    def __init__(self, training_control):
        """Initialize the API manager with training control"""
        self.training_control = training_control
        
        # Predefined API provider configurations
        self.provider_configs = {
            'openai': {
                'base_endpoint': 'https://api.openai.com/v1',
                'completion_endpoint': '/chat/completions',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer {api_key}'
                },
                'request_format': {
                    'model': '{model_name}',
                    'messages': '{messages}',
                    'temperature': 0.7,
                    'max_tokens': 1000
                },
                'response_parser': self._parse_openai_response
            },
            'anthropic': {
                'base_endpoint': 'https://api.anthropic.com/v1',
                'completion_endpoint': '/messages',
                'headers': {
                    'Content-Type': 'application/json',
                    'x-api-key': '{api_key}',
                    'anthropic-version': '2023-06-01'
                },
                'request_format': {
                    'model': '{model_name}',
                    'messages': '{messages}',
                    'temperature': 0.7,
                    'max_tokens': 1000
                },
                'response_parser': self._parse_anthropic_response
            },
            'google': {
                'base_endpoint': 'https://generativelanguage.googleapis.com/v1beta',
                'completion_endpoint': '/models/{model_name}:generateContent',
                'headers': {
                    'Content-Type': 'application/json',
                },
                'request_format': {
                    'contents': [
                        {
                            'parts': [
                                {'text': '{prompt}'}
                            ]
                        }
                    ],
                    'generationConfig': {
                        'temperature': 0.7,
                        'topK': 1,
                        'topP': 1,
                        'maxOutputTokens': 1000
                    }
                },
                'response_parser': self._parse_google_response
            },
            'siliconflow': {
                'base_endpoint': 'https://api.siliconflow.cn/v1',
                'completion_endpoint': '/chat/completions',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer {api_key}'
                },
                'request_format': {
                    'model': '{model_name}',
                    'messages': '{messages}',
                    'temperature': 0.7,
                    'max_tokens': 1000
                },
                'response_parser': self._parse_openai_response
            },
            'openrouter': {
                'base_endpoint': 'https://openrouter.ai/api/v1',
                'completion_endpoint': '/chat/completions',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer {api_key}'
                },
                'request_format': {
                    'model': '{model_name}',
                    'messages': '{messages}',
                    'temperature': 0.7,
                    'max_tokens': 1000
                },
                'response_parser': self._parse_openai_response
            }
        }
        
        # Model name mapping for specific providers
        self.model_name_mappings = {
            'siliconflow': {
                'deepseek-ai/deepseek-r1': 'deepseek-ai/DeepSeek-R1',
                'deepseek-ai/deepseek-v2.5': 'deepseek-ai/DeepSeek-V2.5',
                'deepseek-ai/deepseek-coder-6.7b-instruct': 'deepseek-ai/deepseek-coder-6.7b-instruct',
                'qwen/qwen-2.5-72b-instruct': 'Qwen/Qwen2.5-72B-Instruct',
                'qwen/qwen-2.5-7b-instruct': 'Qwen/Qwen2.5-7B-Instruct',
                'qwen/qwen-2.5-coder-7b-instruct': 'Qwen/Qwen2.5-Coder-7B-Instruct',
                'thudm/glm-4-9b-chat': 'THUDM/glm-4-9b-chat',
                'meta-llama/llama-3.1-8b-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            }
        }
    
    def get_all_models(self):
        """Get all models from the registry"""
        try:
            models_dict = self.training_control.get_model_registry()
            models_array = []
            
            for model_id, model_data in models_dict.items():
                config = model_data.get('config', {})
                model_info = {
                    'model_id': model_id,
                    'name': config.get('name', model_id),
                    'model_type': config.get('model_type', 'unknown'),
                    'description': config.get('description', ''),
                    'status': model_data.get('current_status', 'not_loaded'),
                    'model_source': config.get('model_source', 'local'),
                    'external_api': config.get('external_api')
                }
                models_array.append(model_info)
                
            return {'status': 'success', 'models': models_array}
        except Exception as e:
            logger.error(f"Failed to get models list: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_model_details(self, model_id):
        """Get detailed information for a specific model"""
        try:
            model_config = self.training_control.get_model_configuration(model_id)
            return {'status': 'success', 'model': model_config}
        except Exception as e:
            logger.error(f"Failed to get model details: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def test_api_connection(self, model_id, api_config):
        """Test connection to external API with the provided configuration"""
        try:
            # Extract API configuration
            api_provider = api_config.get('provider', 'openai').lower()
            api_endpoint = api_config.get('api_endpoint')
            api_key = api_config.get('api_key')
            api_model = api_config.get('api_model')
            timeout_value = api_config.get('timeout', timeout)
            
            if not api_endpoint or not api_key or not api_model:
                return {'status': 'error', 'message': 'Missing required parameters'}
            
            # Clean up endpoint format
            api_endpoint = api_endpoint.strip()
            
            # Get correct endpoint and handle model name mapping
            test_endpoint, corrected_model = self._get_correct_endpoint(api_endpoint, api_model, api_provider)
            
            # Prepare headers and payload
            headers, payload = self._prepare_request_payload(api_provider, api_key, corrected_model)
            
            # Make test request
            response = requests.post(test_endpoint, headers=headers, json=payload, timeout=timeout_value)
            
            if response.status_code == 200:
                result = response.json()
                provider_config = self.provider_configs.get(api_provider)
                if provider_config:
                    parsed_response = provider_config['response_parser'](result)
                else:
                    parsed_response = self._parse_default_response(result)
                
                return {
                    'status': 'success',
                    'response': parsed_response,
                    'message': 'Connection successful'
                }
            else:
                error_msg = f"External API returned status {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        if isinstance(error_data['error'], dict):
                            error_msg += f": {error_data['error'].get('message', 'Unknown error')}"
                        else:
                            error_msg += f": {error_data['error']}"
                except:
                    error_msg += f": {response.text[:200]}..."
                
                return {'status': 'error', 'message': error_msg}
        except requests.exceptions.Timeout:
            error_msg = f"External API request timed out after {timeout_value} seconds"
            logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
        except requests.exceptions.ConnectionError:
            error_msg = f"Failed to connect to external API at {api_endpoint}"
            logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
        except Exception as e:
            logger.error(f"Error testing external API: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def switch_model_to_external(self, model_id, api_config):
        """Switch a model to use external API"""
        try:
            # Validate required API config parameters
            if not api_config.get('api_endpoint') or not api_config.get('api_key') or not api_config.get('api_model'):
                return {'status': 'error', 'message': 'Missing required API configuration parameters'}
            
            # Get existing model configuration
            model_config = self.training_control.get_model_configuration(model_id)
            if not model_config:
                return {'status': 'error', 'message': 'Model not found'}
            
            # Update model configuration to use external API
            updated_config = {
                **model_config,
                'model_source': 'external',
                'external_api': api_config
            }
            
            # Update model configuration
            success = self.training_control.update_model_configuration(model_id, updated_config)
            
            if success:
                # Restart the model service with new configuration
                self.training_control.stop_model_service(model_id)
                self.training_control.start_model_service(model_id)
                
                logger.info(f"Model {model_id} successfully switched to external API")
                return {
                    'status': 'success', 
                    'message': 'Model successfully switched to external API',
                    'config': updated_config
                }
            else:
                return {'status': 'error', 'message': 'Failed to update model configuration'}
        except Exception as e:
            logger.error(f"Failed to switch model to external API: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def switch_model_to_local(self, model_id):
        """Switch a model back to local implementation"""
        try:
            # Get existing model configuration
            model_config = self.training_control.get_model_configuration(model_id)
            if not model_config:
                return {'status': 'error', 'message': 'Model not found'}
            
            # Update model configuration to use local implementation
            updated_config = {**model_config, 'model_source': 'local'}
            if 'external_api' in updated_config:
                del updated_config['external_api']
            
            # Update model configuration
            success = self.training_control.update_model_configuration(model_id, updated_config)
            
            if success:
                # Restart the model service with new configuration
                self.training_control.stop_model_service(model_id)
                self.training_control.start_model_service(model_id)
                
                logger.info(f"Model {model_id} successfully switched back to local implementation")
                return {
                    'status': 'success', 
                    'message': 'Model successfully switched back to local implementation',
                    'config': updated_config
                }
            else:
                return {'status': 'error', 'message': 'Failed to update model configuration'}
        except Exception as e:
            logger.error(f"Failed to switch model to local: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_correct_endpoint(self, base_url, model_name, provider):
        """Intelligent endpoint detection and correction"""
        base_url = base_url.rstrip('/')
        
        # Handle model name mapping for specific providers
        if provider in self.model_name_mappings:
            model_name = self.model_name_mappings[provider].get(model_name, model_name)
        
        # Check if provider configuration exists
        if provider in self.provider_configs:
            provider_config = self.provider_configs[provider]
            # Use predefined base endpoint if not provided
            if not base_url or base_url == provider_config['base_endpoint']:
                endpoint = provider_config['base_endpoint'] + provider_config['completion_endpoint']
                return endpoint, model_name
        
        # Already a complete endpoint
        if base_url.endswith('/chat/completions') or base_url.endswith('/messages'):
            return base_url, model_name
        
        # Default OpenAI format
        if '/v1' in base_url:
            return base_url.rsplit('/v1', 1)[0] + '/v1/chat/completions', model_name
        else:
            return base_url + '/v1/chat/completions', model_name
    
    def _prepare_request_payload(self, provider, api_key, model_name):
        """Prepare request headers and payload based on provider"""
        # Get provider configuration or default to OpenAI format
        provider_config = self.provider_configs.get(provider, self.provider_configs['openai'])
        
        # Prepare headers
        headers = {}
        for key, value in provider_config['headers'].items():
            headers[key] = value.format(api_key=api_key) if '{api_key}' in value else value
        
        # Prepare payload
        test_messages = [{'role': 'user', 'content': 'hi'}]
        
        # Handle Google's special format
        if provider == 'google':
            payload = {
                'contents': [{'parts': [{'text': 'hi'}]}],
                'generationConfig': {
                    'temperature': 0.1,
                    'topK': 1,
                    'topP': 1,
                    'maxOutputTokens': 1
                }
            }
            return headers, payload
        
        # Standard payload for other providers
        payload = {
            'model': model_name,
            'messages': test_messages,
            'max_tokens': 1,
            'temperature': 0.1
        }
        
        return headers, payload
    
    # Response parsers for different providers
    def _parse_openai_response(self, response_data):
        """Parse response from OpenAI compatible APIs"""
        if 'choices' in response_data and response_data['choices'] and 'message' in response_data['choices'][0]:
            return response_data['choices'][0]['message']['content']
        return json.dumps(response_data)
    
    def _parse_anthropic_response(self, response_data):
        """Parse response from Anthropic API"""
        if 'content' in response_data and response_data['content'] and 'text' in response_data['content'][0]:
            return response_data['content'][0]['text']
        return json.dumps(response_data)
    
    def _parse_google_response(self, response_data):
        """Parse response from Google API"""
        if 'candidates' in response_data and response_data['candidates'] and 'content' in response_data['candidates'][0]:
            content = response_data['candidates'][0]['content']
            if 'parts' in content and content['parts'] and 'text' in content['parts'][0]:
                return content['parts'][0]['text']
        return json.dumps(response_data)
    
    def _parse_default_response(self, response_data):
        """Default response parser"""
        try:
            return json.dumps(response_data)
        except:
            return str(response_data)

# Helper function to get the API manager instance
def get_model_api_manager(training_control=None):
    """Get or create the ModelAPIManager instance"""
    if not hasattr(get_model_api_manager, '_instance'):
        if training_control is None:
            # Import training control if not provided
            from manager_model.training_control import TrainingController
            training_control = TrainingController()
        get_model_api_manager._instance = ModelAPIManager(training_control)
    return get_model_api_manager._instance