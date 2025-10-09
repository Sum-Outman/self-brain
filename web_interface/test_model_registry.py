#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for model registry and external API integration
This script tests if the system can properly load model registry and handle external API calls.
"""

import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelRegistryTest")


def load_model_registry():
    """Test loading the model registry from config file"""
    try:
        # Get the path to model_registry.json
        model_registry_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'model_registry.json')
        logger.info(f"Attempting to load model registry from: {model_registry_path}")
        
        # Check if file exists
        if not os.path.exists(model_registry_path):
            logger.error(f"Model registry file not found: {model_registry_path}")
            return None
        
        # Read and parse the registry file
        with open(model_registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
            
        logger.info(f"Successfully loaded model registry with {len(registry)} models")
        
        # Print basic info about each model
        for model_name, model_info in registry.items():
            model_type = model_info.get('model_type', 'Unknown')
            model_source = model_info.get('model_source', 'Unknown')
            api_url = model_info.get('api_url', 'None')
            
            logger.info(f"- Model: {model_name}")
            logger.info(f"  Type: {model_type}")
            logger.info(f"  Source: {model_source}")
            logger.info(f"  API URL: {api_url}")
            
        return registry
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse model registry JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading model registry: {str(e)}")
    
    return None


def check_external_models(registry):
    """Check external models configuration in the registry"""
    if not registry:
        logger.warning("No registry provided, skipping external models check")
        return
    
    external_models = []
    
    for model_name, model_info in registry.items():
        if model_info.get('model_source') == 'external':
            external_models.append((model_name, model_info))
    
    if not external_models:
        logger.info("No external models found in registry")
        return
    
    logger.info(f"Found {len(external_models)} external models in registry")
    
    # Check each external model's configuration
    for model_name, model_info in external_models:
        # Check required fields for external models
        provider = model_info.get('provider', 'Unknown')
        api_url = model_info.get('api_url', '')
        api_key = model_info.get('api_key', '')
        api_model = model_info.get('api_model', '')
        
        logger.info(f"- External Model: {model_name}")
        logger.info(f"  Provider: {provider}")
        logger.info(f"  API URL: {api_url}")
        logger.info(f"  API Model: {api_model}")
        
        # Check for missing required configuration
        missing_fields = []
        if not api_url:
            missing_fields.append('api_url')
        if not api_model:
            missing_fields.append('api_model')
        
        if missing_fields:
            logger.warning(f"  WARNING: Missing required fields: {', '.join(missing_fields)}")
        
        # Check if API key is provided (masked for security)
        has_api_key = bool(api_key)
        logger.info(f"  Has API Key: {has_api_key}")


def simulate_external_api_call():
    """Simulate an external API call similar to how it's done in the app"""
    try:
        from app import call_external_api_model
        
        # Create a mock external API configuration
        mock_config = {
            'provider': 'openai',
            'api_key': 'mock-api-key',  # This is just for testing
            'model': 'gpt-3.5-turbo',
            'base_url': 'https://api.openai.com/v1',
            'timeout': 30
        }
        
        logger.info("Simulating external API call with mock configuration")
        logger.info(f"Provider: {mock_config['provider']}")
        logger.info(f"Model: {mock_config['model']}")
        logger.info(f"Base URL: {mock_config['base_url']}")
        
        # Note: This will fail since we're using a mock API key, but it's expected
        try:
            response = call_external_api_model("Hello, world!", mock_config)
            logger.info(f"API call succeeded, response: {response[:100]}...")
        except Exception as e:
            logger.info(f"API call failed as expected: {str(e)}")
            logger.info("This is normal in testing environment with mock credentials")
            
    except ImportError:
        logger.warning("Could not import call_external_api_model function")
    except Exception as e:
        logger.error(f"Error during simulated API call: {str(e)}")


def main():
    """Main test function"""
    logger.info("=== Model Registry and External API Integration Test ===")
    logger.info(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test loading the model registry
    registry = load_model_registry()
    
    # Check external models if registry was loaded
    if registry:
        check_external_models(registry)
    
    # Simulate an external API call
    simulate_external_api_call()
    
    logger.info("=== Test Completed ===")


if __name__ == "__main__":
    main()