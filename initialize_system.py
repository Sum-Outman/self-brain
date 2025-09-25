#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self Brain System Initialization Script
This script initializes the Self Brain AGI system, ensuring all models are properly configured
for training from scratch and all features are enabled.
"""

import os
import sys
import json
import shutil
import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system_init.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SelfBrainInit")

# Base directory
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = BASE_DIR / "web_interface" / "config.json"

# Check Python version
def check_python_version():
    """Check if the Python version is compatible with the system"""
    required_version = (3, 8)
    current_version = sys.version_info
    
    logger.info(f"Current Python version: {current_version.major}.{current_version.minor}.{current_version.micro}")
    
    if current_version < required_version:
        logger.error(f"Python version {required_version[0]}.{required_version[1]} or higher is required")
        logger.error("Please upgrade your Python installation and try again")
        sys.exit(1)
    
    logger.info("Python version check passed")
    return True

# Create configuration files
def create_config_files():
    """Create default configuration files if they don't exist"""
    logger.info("Checking configuration files...")
    
    # Create web interface config
    if not CONFIG_PATH.exists():
        logger.warning("Web interface configuration not found, creating default config")
        # Ensure directory exists
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Create default config
        default_config = {
            "web_port": 5000,
            "manager_port": 5015,
            "debug_mode": False,
            "log_level": "INFO",
            "max_upload_size": 50,
            "model_timeout": 300
        }
        
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        logger.info("Default web interface configuration created")
    
    # Check model registry
    model_registry_path = BASE_DIR / "config" / "model_registry.json"
    if not model_registry_path.exists():
        logger.warning("Model registry not found, creating default registry")
        # Ensure directory exists
        model_registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create default model registry
        default_registry = {
            "models": [
                {
                    "id": "a_manager",
                    "type": "management",
                    "path": "manager_model",
                    "description": "System management and coordination model",
                    "version": "1.0.0"
                },
                {
                    "id": "b_language",
                    "type": "language",
                    "path": "sub_models/B_language",
                    "description": "Natural language processing model",
                    "version": "1.0.0"
                }
            ]
        }
        
        with open(model_registry_path, 'w', encoding='utf-8') as f:
            json.dump(default_registry, f, indent=2, ensure_ascii=False)
        
        logger.info("Default model registry created")

# Initialize model directories
def initialize_model_directories():
    """Create necessary model directories"""
    logger.info("Initializing model directories...")
    
    # Create submodels directory
    submodels_dir = BASE_DIR / "sub_models"
    submodels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manager model directory
    manager_dir = BASE_DIR / "manager_model"
    manager_dir.mkdir(parents=True, exist_ok=True)
    
    # Create web interface directory
    web_dir = BASE_DIR / "web_interface"
    web_dir.mkdir(parents=True, exist_ok=True)
    
    # Create templates directory
    templates_dir = web_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config directory
    config_dir = BASE_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

# Create requirements file
def create_requirements_file():
    """Create requirements.txt if it doesn't exist"""
    requirements_path = BASE_DIR / "requirements.txt"
    
    if not requirements_path.exists():
        logger.warning("requirements.txt not found, creating default requirements")
        
        default_requirements = """
# Core dependencies
flask>=2.0.0,<3.0.0
fastapi>=0.68.0,<1.0.0
uvicorn>=0.15.0,<1.0.0
psutil>=5.8.0,<6.0.0
python-multipart>=0.0.5
requests>=2.25.0,<3.0.0
pydantic>=1.8.0,<2.0.0

# Optional dependencies
numpy>=1.20.0,<2.0.0
pandas>=1.3.0,<2.0.0
sentence-transformers>=2.2.0,<3.0.0
scikit-learn>=1.0.0,<2.0.0
"""
        
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write(default_requirements.strip())
        
        logger.info("Default requirements.txt created")

# Main initialization function
def main():
    """Main initialization function"""
    logger.info("===== Self Brain AGI System Initialization =====")
    
    # Check Python version
    check_python_version()
    
    # Create configuration files
    create_config_files()
    
    # Initialize model directories
    initialize_model_directories()
    
    # Create requirements file
    create_requirements_file()
    
    logger.info("System initialization completed successfully!")
    logger.info("You can now start the system with start_system.py")

if __name__ == "__main__":
    main()