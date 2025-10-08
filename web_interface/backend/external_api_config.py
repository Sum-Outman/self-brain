#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External API Configuration Module
This module handles configuration and connection to external AI service providers
"""

import logging
import json
import os
import requests
import time
from datetime import datetime
from collections import defaultdict
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ExternalAPIConfig')

class ExternalAPIConfig:
    """Manages configuration and connection to external AI service providers"""
    
    def __init__(self):
        # API providers configuration
        self.providers = {
            'openai': {
                'base_url': 'https://api.openai.com/v1',
                'default_model': 'gpt-3.5-turbo',
                'supported_models': [
                    'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 
                    'gpt-4', 'gpt-4-32k', 'gpt-4-turbo'
                ],
                'auth_type': 'bearer',
                'models_endpoint': '/models',
                'chat_endpoint': '/chat/completions',
                'embeddings_endpoint': '/embeddings',
                'image_endpoint': '/images/generations'
            },
            'anthropic': {
                'base_url': 'https://api.anthropic.com/v1',
                'default_model': 'claude-3-sonnet-20240229',
                'supported_models': [
                    'claude-3-opus-20240229', 
                    'claude-3-sonnet-20240229',
                    'claude-3-haiku-20240307'
                ],
                'auth_type': 'bearer',
                'api_key_header': 'x-api-key',
                'chat_endpoint': '/messages'
            },
            'google': {
                'base_url': 'https://generativelanguage.googleapis.com/v1beta',
                'default_model': 'gemini-pro',
                'supported_models': [
                    'gemini-pro', 'gemini-pro-vision'
                ],
                'auth_type': 'query_param',
                'api_key_param': 'key',
                'chat_endpoint': '/models/{model}:generateContent',
                'embeddings_endpoint': '/models/{model}:embedContent'
            },
            'groq': {
                'base_url': 'https://api.groq.com/openai/v1',
                'default_model': 'mixtral-8x7b-32768',
                'supported_models': [
                    'mixtral-8x7b-32768', 
                    'llama2-70b-4096',
                    'gemma-7b-it'
                ],
                'auth_type': 'bearer',
                'chat_endpoint': '/chat/completions',
                'models_endpoint': '/models'
            },
            'mistralai': {
                'base_url': 'https://api.mistral.ai/v1',
                'default_model': 'mistral-medium-latest',
                'supported_models': [
                    'mistral-tiny-latest', 
                    'mistral-small-latest',
                    'mistral-medium-latest'
                ],
                'auth_type': 'bearer',
                'chat_endpoint': '/chat/completions',
                'models_endpoint': '/models'
            }
        }
        
        # Model type mapping
        self.model_type_mapping = {
            'A': 'management',
            'B': 'language',
            'C': 'audio',
            'D': 'image',
            'E': 'video',
            'F': 'spatial',
            'G': 'sensor',
            'H': 'computer',
            'I': 'motion',
            'J': 'knowledge',
            'K': 'programming'
        }
        
        # API key storage
        self.api_keys = {}
        
        # Model-specific external API configurations
        self.model_external_configs = defaultdict(dict)
        
        # Connection status tracking
        self.connection_status = {}
        
        # Global settings from config file
        self.global_settings = {
            'timeout': 30,
            'retry_count': 3,
            'proxy': ''
        }
        
        # Configuration file path - now using the new api_settings.json
        self.config_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'config', 'api_settings.json'
        )
        
        # Legacy external config file path (for migration)
        self.legacy_config_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'config', 'external_api_config.json'
        )
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_file_path), exist_ok=True)
        
        # Load existing configuration
        self.load_config()
        
        # Test thread for connection status monitoring
        self.monitoring_thread = None
        self.is_monitoring = False
        self.start_monitoring()
    
    def load_config(self):
        """Load external API configuration from the new api_settings.json file"""
        try:
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Load global settings
                if 'global_settings' in config_data:
                    self.global_settings.update(config_data['global_settings'])
                
                # Load model-specific configurations
                if 'models' in config_data:
                    for model_id, model_config in config_data['models'].items():
                        # Extract the model letter (e.g., 'B' from 'B_language')
                        model_letter = model_id.split('_')[0] if '_' in model_id else model_id
                        
                        if model_letter in self.model_type_mapping:
                            # Map to the format expected by existing code
                            provider_name = self._determine_provider(model_config.get('api_url', ''))
                            
                            if provider_name:
                                self.model_external_configs[model_letter] = {
                                    'provider': provider_name,
                                    'model': model_config.get('model_name', ''),
                                    'enabled': model_config.get('enabled', False),
                                    'use_external': model_config.get('use_external', False),
                                    'timeout': model_config.get('timeout', self.global_settings['timeout'])
                                }
                                
                                # Store API key if provided
                                if model_config.get('api_key'):
                                    self.api_keys[provider_name] = model_config['api_key']
                
                logger.info(f"Loaded external API configuration from {self.config_file_path}")
            else:
                logger.info(f"No existing configuration found at {self.config_file_path}, using defaults")
                # Check for legacy config and migrate if needed
                if os.path.exists(self.legacy_config_file_path):
                    self._migrate_legacy_config()
                else:
                    # Create default config file
                    self.save_config()
                    
        except Exception as e:
            logger.error(f"Failed to load external API configuration: {str(e)}")
    
    def _migrate_legacy_config(self):
        """Migrate configuration from legacy file to new format"""
        try:
            with open(self.legacy_config_file_path, 'r', encoding='utf-8') as f:
                legacy_config = json.load(f)
            
            # Create new config structure
            new_config = {
                'global_settings': self.global_settings,
                'models': {}
            }
            
            # Migrate model configurations
            for model_id, config in legacy_config.get('model_configs', {}).items():
                if config.get('enabled', False) and config.get('provider'):
                    provider_config = self.get_provider_config(config['provider'])
                    model_name = f"{model_id}_{self.model_type_mapping.get(model_id, 'model')}"
                    
                    new_config['models'][model_name] = {
                        'use_external': True,
                        'api_url': provider_config.get('base_url', ''),
                        'api_key': legacy_config.get('api_keys', {}).get(config['provider'], ''),
                        'model_name': config.get('model', ''),
                        'headers': {},
                        'timeout': self.global_settings['timeout'],
                        'enabled': True
                    }
            
            # Save new config
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, indent=2)
            
            logger.info(f"Migrated legacy configuration to {self.config_file_path}")
        except Exception as e:
            logger.error(f"Failed to migrate legacy configuration: {str(e)}")
    
    def _determine_provider(self, api_url):
        """Determine provider based on API URL"""
        if not api_url:
            return None
        
        for provider, config in self.providers.items():
            if config['base_url'] in api_url:
                return provider
        
        # Default to openai if not matched
        return 'openai'
    
    def save_config(self):
        """Save external API configuration to file"""
        try:
            # First load existing config to preserve manual edits
            existing_config = {}
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
            
            # Update with current settings
            if 'global_settings' not in existing_config:
                existing_config['global_settings'] = {}
            existing_config['global_settings'].update(self.global_settings)
            
            # Update model configurations
            if 'models' not in existing_config:
                existing_config['models'] = {}
            
            for model_letter, config in self.model_external_configs.items():
                if config.get('enabled', False) and config.get('provider'):
                    provider_config = self.get_provider_config(config['provider'])
                    model_name = f"{model_letter}_{self.model_type_mapping.get(model_letter, 'model')}"
                    
                    if model_name not in existing_config['models']:
                        existing_config['models'][model_name] = {
                            'use_external': config.get('use_external', False),
                            'api_url': provider_config.get('base_url', ''),
                            'api_key': self.api_keys.get(config['provider'], ''),
                            'model_name': config.get('model', ''),
                            'headers': {},
                            'timeout': config.get('timeout', self.global_settings['timeout']),
                            'enabled': True
                        }
            
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_config, f, indent=2)
                
            logger.info(f"Saved external API configuration to {self.config_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save external API configuration: {str(e)}")
            return False
    
    def get_supported_providers(self):
        """Get list of supported API providers"""
        return list(self.providers.keys())
    
    def get_provider_config(self, provider_name):
        """Get configuration for a specific provider"""
        return self.providers.get(provider_name, {})
    
    def get_supported_models(self, provider_name):
        """Get list of supported models for a specific provider"""
        provider_config = self.get_provider_config(provider_name)
        return provider_config.get('supported_models', [])
    
    def set_api_key(self, provider_name, api_key):
        """Set API key for a specific provider"""
        if provider_name in self.providers:
            self.api_keys[provider_name] = api_key
            self.save_config()
            logger.info(f"Set API key for provider {provider_name}")
            return True
        else:
            logger.error(f"Provider {provider_name} not supported")
            return False
    
    def get_api_key(self, provider_name):
        """Get API key for a specific provider"""
        return self.api_keys.get(provider_name, '')
    
    def configure_model_external_api(self, model_id, provider_name, model_name, api_key=None):
        """Configure external API for a specific model"""
        try:
            # Validate model_id
            if model_id not in self.model_type_mapping:
                logger.error(f"Invalid model ID: {model_id}")
                return False
            
            # Validate provider
            if provider_name not in self.providers:
                logger.error(f"Unsupported provider: {provider_name}")
                return False
            
            # Validate model name
            supported_models = self.get_supported_models(provider_name)
            if model_name not in supported_models:
                logger.error(f"Unsupported model {model_name} for provider {provider_name}")
                return False
            
            # Update model configuration
            self.model_external_configs[model_id] = {
                'provider': provider_name,
                'model': model_name,
                'enabled': True,
                'use_external': True
            }
            
            # Update API key if provided
            if api_key:
                self.set_api_key(provider_name, api_key)
            
            # Save configuration
            self.save_config()
            
            # Test the connection
            connection_success = self.test_connection(model_id)
            
            logger.info(f"Configured model {model_id} to use external API: {provider_name} - {model_name}")
            return connection_success
            
        except Exception as e:
            logger.error(f"Failed to configure external API for model {model_id}: {str(e)}")
            return False
    
    def get_model_external_config(self, model_id):
        """Get external API configuration for a specific model"""
        return self.model_external_configs.get(model_id, {})
    
    def disable_model_external_api(self, model_id):
        """Disable external API for a specific model"""
        if model_id in self.model_external_configs:
            self.model_external_configs[model_id]['enabled'] = False
            self.model_external_configs[model_id]['use_external'] = False
            self.save_config()
            logger.info(f"Disabled external API for model {model_id}")
            return True
        return False
    
    def enable_model_external_api(self, model_id):
        """Enable external API for a specific model"""
        if model_id in self.model_external_configs:
            self.model_external_configs[model_id]['enabled'] = True
            self.model_external_configs[model_id]['use_external'] = True
            self.save_config()
            logger.info(f"Enabled external API for model {model_id}")
            # Test the connection
            return self.test_connection(model_id)
        return False
    
    def is_model_using_external_api(self, model_id):
        """Check if a model is configured to use external API"""
        config = self.get_model_external_config(model_id)
        return config.get('enabled', False) and config.get('use_external', False)
    
    def test_connection(self, model_id):
        """Test connection to external API for a specific model"""
        try:
            config = self.get_model_external_config(model_id)
            
            if not config.get('enabled', False) or not config.get('use_external', False):
                logger.info(f"External API is not enabled for model {model_id}")
                return False
            
            provider_name = config.get('provider')
            model_name = config.get('model')
            api_key = self.get_api_key(provider_name)
            
            if not api_key:
                logger.error(f"No API key found for provider {provider_name}")
                self.update_connection_status(model_id, False, "No API key provided")
                return False
            
            # Test the connection based on provider type
            provider_config = self.get_provider_config(provider_name)
            success, message = self._test_provider_connection(provider_config, api_key, model_name)
            
            # Update connection status
            self.update_connection_status(model_id, success, message)
            
            if success:
                logger.info(f"Connection test successful for model {model_id} with {provider_name} - {model_name}")
            else:
                logger.warning(f"Connection test failed for model {model_id}: {message}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error testing connection for model {model_id}: {str(e)}")
            self.update_connection_status(model_id, False, str(e))
            return False
    
    def _test_provider_connection(self, provider_config, api_key, model_name):
        """Test connection to a specific provider"""
        try:
            headers = {}
            
            # Set up authentication based on provider type
            if provider_config.get('auth_type') == 'bearer':
                headers['Authorization'] = f'Bearer {api_key}'
            elif provider_config.get('auth_type') == 'header':
                header_name = provider_config.get('api_key_header', 'api-key')
                headers[header_name] = api_key
            
            # For Anthropic, add content-type and Anthropic-Version headers
            if provider_config.get('base_url', '').find('anthropic') != -1:
                headers['Content-Type'] = 'application/json'
                headers['Anthropic-Version'] = '2023-06-01'
            
            # Try to list models or make a simple request
            if 'models_endpoint' in provider_config:
                endpoint = provider_config['base_url'] + provider_config['models_endpoint']
                
                # For Google, API key is in query params
                if provider_config.get('auth_type') == 'query_param':
                    param_name = provider_config.get('api_key_param', 'key')
                    endpoint = f"{endpoint}?{param_name}={api_key}"
                    response = requests.get(endpoint, timeout=10)
                else:
                    response = requests.get(endpoint, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    return True, "Connection successful"
                else:
                    return False, f"Failed to list models: {response.status_code} {response.text}"
            else:
                # Fallback: make a simple chat completion request
                if 'chat_endpoint' in provider_config:
                    endpoint = provider_config['base_url'] + provider_config['chat_endpoint']
                    
                    # Handle Google's endpoint format with model placeholder
                    if provider_config.get('auth_type') == 'query_param':
                        param_name = provider_config.get('api_key_param', 'key')
                        endpoint = endpoint.replace('{model}', model_name)
                        endpoint = f"{endpoint}?{param_name}={api_key}"
                        
                        # Google Gemini request format
                        data = {
                            "contents": [{
                                "parts": [{"text": "Hello"}]
                            }]
                        }
                    elif provider_config.get('base_url', '').find('anthropic') != -1:
                        # Anthropic request format
                        data = {
                            "model": model_name,
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 10
                        }
                    else:
                        # Standard OpenAI-compatible format
                        data = {
                            "model": model_name,
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 10
                        }
                    
                    # Make the request
                    if provider_config.get('auth_type') == 'query_param':
                        response = requests.post(endpoint, json=data, timeout=10)
                    else:
                        response = requests.post(endpoint, headers=headers, json=data, timeout=10)
                    
                    if response.status_code == 200:
                        return True, "Connection successful"
                    else:
                        return False, f"Failed chat request: {response.status_code} {response.text}"
                
            return False, "No suitable endpoint found for testing"
            
        except requests.exceptions.Timeout:
            return False, "Connection timed out"
        except requests.exceptions.ConnectionError:
            return False, "Connection error"
        except Exception as e:
            return False, str(e)
    
    def update_connection_status(self, model_id, is_connected, message):
        """Update the connection status for a model"""
        self.connection_status[model_id] = {
            'is_connected': is_connected,
            'message': message,
            'last_checked': datetime.now().isoformat()
        }
        self.save_config()
    
    def get_connection_status(self, model_id):
        """Get the connection status for a model"""
        return self.connection_status.get(model_id, {
            'is_connected': False,
            'message': 'Not configured',
            'last_checked': None
        })
    
    def start_monitoring(self):
        """Start monitoring connection statuses"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Started external API connection monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring connection statuses"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Stopped external API connection monitoring")
    
    def _monitoring_loop(self):
        """Main loop for monitoring connection statuses"""
        while self.is_monitoring:
            try:
                # Test connections for all enabled models
                for model_id, config in self.model_external_configs.items():
                    if config.get('enabled', False) and config.get('use_external', False):
                        # Test connection every 5 minutes
                        last_checked = self.connection_status.get(model_id, {}).get('last_checked')
                        if not last_checked or self._should_check_connection(last_checked):
                            self.test_connection(model_id)
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                # Sleep for 30 seconds before retrying
                time.sleep(30)
    
    def _should_check_connection(self, last_checked_str):
        """Check if it's time to re-test the connection"""
        try:
            last_checked = datetime.fromisoformat(last_checked_str)
            time_diff = datetime.now() - last_checked
            # Check every 5 minutes (300 seconds)
            return time_diff.total_seconds() > 300
        except:
            return True
    
    def call_external_api(self, model_id, request_data):
        """Call the external API for a specific model with the given request data"""
        try:
            # Check if model is configured to use external API
            if not self.is_model_using_external_api(model_id):
                logger.warning(f"Model {model_id} is not configured to use external API")
                return None, "Model not configured to use external API"
            
            # Get model configuration
            config = self.get_model_external_config(model_id)
            provider_name = config.get('provider')
            model_name = config.get('model')
            
            # Check connection status
            connection_status = self.get_connection_status(model_id)
            if not connection_status.get('is_connected', False):
                logger.warning(f"Connection to external API for model {model_id} is not active")
                # Try to re-establish connection
                if not self.test_connection(model_id):
                    return None, f"Connection not active: {connection_status.get('message', 'Unknown error')}"
            
            # Get provider configuration and API key
            provider_config = self.get_provider_config(provider_name)
            api_key = self.get_api_key(provider_name)
            
            # Make the API call
            response, error = self._make_api_call(
                provider_config, 
                api_key, 
                model_name, 
                request_data, 
                model_id
            )
            
            if error:
                logger.error(f"API call failed for model {model_id}: {error}")
                return None, error
            
            return response, None
            
        except Exception as e:
            logger.error(f"Error calling external API for model {model_id}: {str(e)}")
            return None, str(e)
    
    def _make_api_call(self, provider_config, api_key, model_name, request_data, model_id):
        """Make an API call to the specified provider"""
        try:
            headers = {'Content-Type': 'application/json'}
            endpoint = provider_config['base_url']
            
            # Set up authentication based on provider type
            if provider_config.get('auth_type') == 'bearer':
                headers['Authorization'] = f'Bearer {api_key}'
            elif provider_config.get('auth_type') == 'header':
                header_name = provider_config.get('api_key_header', 'api-key')
                headers[header_name] = api_key
            
            # Handle specific provider requirements
            if provider_config.get('base_url', '').find('anthropic') != -1:
                headers['Anthropic-Version'] = '2023-06-01'
                endpoint += provider_config['chat_endpoint']
                
                # Format request data for Anthropic
                formatted_data = {
                    "model": model_name,
                    "messages": request_data.get('messages', []),
                    "max_tokens": request_data.get('max_tokens', 1000),
                    "temperature": request_data.get('temperature', 0.7)
                }
            elif provider_config.get('auth_type') == 'query_param':
                # For Google Gemini
                param_name = provider_config.get('api_key_param', 'key')
                endpoint += provider_config['chat_endpoint'].replace('{model}', model_name)
                endpoint = f"{endpoint}?{param_name}={api_key}"
                
                # Format request data for Google Gemini
                messages = request_data.get('messages', [])
                contents = []
                for msg in messages:
                    parts = [{"text": msg.get('content', '')}]
                    # Handle images if present
                    if 'image' in msg:
                        parts.append({"inline_data": {"mime_type": msg['image']['mime_type'], "data": msg['image']['data']}})
                    contents.append({"parts": parts})
                
                formatted_data = {
                    "contents": contents,
                    "generationConfig": {
                        "temperature": request_data.get('temperature', 0.7),
                        "maxOutputTokens": request_data.get('max_tokens', 1000)
                    }
                }
            else:
                # Standard OpenAI-compatible format
                endpoint += provider_config['chat_endpoint']
                formatted_data = {
                    "model": model_name,
                    "messages": request_data.get('messages', []),
                    "max_tokens": request_data.get('max_tokens', 1000),
                    "temperature": request_data.get('temperature', 0.7),
                    "top_p": request_data.get('top_p', 1.0),
                    "n": request_data.get('n', 1)
                }
            
            # Make the request
            logger.debug(f"Making API call to {endpoint} with data: {formatted_data}")
            response = requests.post(
                endpoint, 
                headers=headers, 
                json=formatted_data, 
                timeout=60  # Increased timeout for external API calls
            )
            
            # Check response status
            if response.status_code == 200:
                response_json = response.json()
                
                # Parse response based on provider
                if provider_config.get('base_url', '').find('anthropic') != -1:
                    # Anthropic response format
                    return {
                        'content': response_json.get('content', [{}])[0].get('text', ''),
                        'model': model_name,
                        'provider': provider_config.get('base_url'),
                        'usage': response_json.get('usage', {})
                    }, None
                elif provider_config.get('auth_type') == 'query_param':
                    # Google Gemini response format
                    candidates = response_json.get('candidates', [])
                    if candidates:
                        content = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                        return {
                            'content': content,
                            'model': model_name,
                            'provider': provider_config.get('base_url'),
                            'usage': response_json.get('usageMetadata', {})
                        }, None
                    else:
                        return None, "No candidates in response"
                else:
                    # Standard OpenAI-compatible format
                    choices = response_json.get('choices', [])
                    if choices:
                        return {
                            'content': choices[0].get('message', {}).get('content', ''),
                            'model': model_name,
                            'provider': provider_config.get('base_url'),
                            'usage': response_json.get('usage', {})
                        }, None
                    else:
                        return None, "No choices in response"
            else:
                return None, f"API request failed: {response.status_code} {response.text}"
            
        except requests.exceptions.Timeout:
            return None, "API request timed out"
        except requests.exceptions.ConnectionError:
            return None, "API connection error"
        except Exception as e:
            return None, str(e)
    
    def get_all_model_configs(self):
        """Get all model configurations"""
        result = {}
        for model_id, config in self.model_external_configs.items():
            result[model_id] = {
                **config,
                'status': self.get_connection_status(model_id)
            }
        return result
    
    def get_model_type(self, model_id):
        """Get the type of a model based on its ID"""
        return self.model_type_mapping.get(model_id, 'unknown')
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()

# Singleton instance of the external API config manager
global_api_config = None

def get_external_api_config():
    """Get the singleton instance of the external API config manager"""
    global global_api_config
    if global_api_config is None:
        global_api_config = ExternalAPIConfig()
    return global_api_config

# Initialize the config manager when this module is loaded
if __name__ == "__main__":
    # For testing purposes
    api_config = ExternalAPIConfig()
    print("External API Config Manager initialized successfully")
    print(f"Supported providers: {api_config.get_supported_providers()}")
    
    # Example configuration (would be done through the UI in production)
    # api_config.configure_model_external_api('B', 'openai', 'gpt-3.5-turbo', 'your-api-key-here')