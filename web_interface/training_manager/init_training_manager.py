#!/usr/bin/env python
# Self Brain AGI System - Training Manager Initialization Script
# Copyright 2025 AGI System Team

import os
import sys
import json
import logging
import shutil
import subprocess
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SelfBrainInit")

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Required directories
REQUIRED_DIRECTORIES = [
    "data/models",
    "data/checkpoints",
    "data/training",
    "data/validation",
    "data/testing",
    "data/raw",
    "data/processed",
    "logs/training",
    "logs/api",
    "logs/models",
    "temp",
    "config"
]

# Configuration files to create
CONFIG_FILES = [
    "config/training_config.json",
    "config/api_config.json",
    "config/model_config.json"
]

class TrainingManagerInitializer:
    """Class to initialize the Training Manager environment"""
    
    def __init__(self, force=False):
        """Initialize with options"""
        self.force = force
    
    def create_directories(self):
        """Create required directories"""
        logger.info("Creating required directories...")
        
        for dir_path in REQUIRED_DIRECTORIES:
            full_path = os.path.join(PROJECT_ROOT, dir_path)
            
            if os.path.exists(full_path):
                if self.force:
                    logger.warning(f"Directory {dir_path} already exists, overwriting...")
                    shutil.rmtree(full_path)
                else:
                    logger.info(f"Directory {dir_path} already exists, skipping...")
                    continue
            
            try:
                os.makedirs(full_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {str(e)}")
                return False
        
        return True
    
    def create_config_files(self):
        """Create configuration files"""
        logger.info("Creating configuration files...")
        
        # Training configuration
        training_config = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "default_training_params": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "loss_function": "cross_entropy",
                "metrics": ["accuracy"],
                "early_stopping_patience": 10,
                "checkpoint_interval": 5,
                "log_interval": 10
            },
            "model_specific_params": {
                "A_management_model": {
                    "epochs": 150,
                    "batch_size": 16,
                    "learning_rate": 0.0005,
                    "special_params": {
                        "emotion_weight": 0.3
                    }
                },
                "B_language_model": {
                    "epochs": 200,
                    "batch_size": 8,
                    "learning_rate": 0.0001,
                    "special_params": {
                        "max_sequence_length": 512,
                        "vocab_size": 30000
                    }
                },
                "C_audio_model": {
                    "epochs": 100,
                    "batch_size": 16,
                    "learning_rate": 0.0003,
                    "special_params": {
                        "sample_rate": 16000,
                        "n_mfcc": 13
                    }
                },
                "D_image_model": {
                    "epochs": 100,
                    "batch_size": 8,
                    "learning_rate": 0.0002,
                    "special_params": {
                        "image_size": [224, 224],
                        "channels": 3
                    }
                },
                "E_video_model": {
                    "epochs": 80,
                    "batch_size": 4,
                    "learning_rate": 0.0001,
                    "special_params": {
                        "frames_per_clip": 16,
                        "frame_size": [112, 112]
                    }
                },
                "F_spatial_model": {
                    "epochs": 120,
                    "batch_size": 16,
                    "learning_rate": 0.0005,
                    "special_params": {
                        "grid_resolution": 0.1
                    }
                },
                "G_sensor_model": {
                    "epochs": 50,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "special_params": {
                        "window_size": 100
                    }
                },
                "H_computer_model": {
                    "epochs": 80,
                    "batch_size": 16,
                    "learning_rate": 0.0005,
                    "special_params": {
                        "max_command_length": 256
                    }
                },
                "I_motion_model": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.0003,
                    "special_params": {
                        "max_path_points": 50
                    }
                },
                "J_knowledge_model": {
                    "epochs": 200,
                    "batch_size": 8,
                    "learning_rate": 0.0001,
                    "special_params": {
                        "embedding_dim": 768,
                        "context_window": 2048
                    }
                },
                "K_programming_model": {
                    "epochs": 150,
                    "batch_size": 8,
                    "learning_rate": 0.0002,
                    "special_params": {
                        "max_code_length": 1024,
                        "language_support": ["python", "javascript", "java", "c++", "sql"]
                    }
                }
            }
        }
        
        # API configuration
        api_config = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "server": {
                "host": "0.0.0.0",
                "port": 5001,
                "debug": False,
                "threaded": True,
                "max_workers": 10,
                "timeout": 600
            },
            "security": {
                "enable_cors": True,
                "allowed_origins": ["*"],
                "api_key_required": False,
                "rate_limiting": {
                    "enabled": False,
                    "requests_per_minute": 100
                }
            },
            "logging": {
                "level": "INFO",
                "file": "logs/api/api.log",
                "rotation": "daily"
            }
        }
        
        # Model configuration
        model_config = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "models": {
                "A_management_model": {
                    "type": "management",
                    "description": "Main management model to coordinate all other models",
                    "architecture": "transformer",
                    "input_types": ["text", "emotion", "context"],
                    "output_types": ["text", "model_calls", "emotion"],
                    "local_model_path": "data/models/A_management_model",
                    "external_api_support": True,
                    "active": True
                },
                "B_language_model": {
                    "type": "language",
                    "description": "General purpose language model with multilingual support",
                    "architecture": "llama-like",
                    "input_types": ["text"],
                    "output_types": ["text"],
                    "local_model_path": "data/models/B_language_model",
                    "external_api_support": True,
                    "active": True
                },
                "C_audio_model": {
                    "type": "audio",
                    "description": "Audio processing model for speech and sound analysis",
                    "architecture": "cnn-transformer",
                    "input_types": ["audio"],
                    "output_types": ["text", "audio_features"],
                    "local_model_path": "data/models/C_audio_model",
                    "external_api_support": True,
                    "active": True
                },
                "D_image_model": {
                    "type": "vision",
                    "description": "Image processing model for recognition and generation",
                    "architecture": "vit",
                    "input_types": ["image"],
                    "output_types": ["text", "image"],
                    "local_model_path": "data/models/D_image_model",
                    "external_api_support": True,
                    "active": True
                },
                "E_video_model": {
                    "type": "vision",
                    "description": "Video processing model for analysis and generation",
                    "architecture": "3d-cnn",
                    "input_types": ["video"],
                    "output_types": ["text", "video"],
                    "local_model_path": "data/models/E_video_model",
                    "external_api_support": True,
                    "active": True
                },
                "F_spatial_model": {
                    "type": "vision",
                    "description": "Spatial perception and 3D modeling model",
                    "architecture": "pointnet",
                    "input_types": ["depth_map", "stereo_images"],
                    "output_types": ["3d_model", "spatial_coordinates"],
                    "local_model_path": "data/models/F_spatial_model",
                    "external_api_support": False,
                    "active": True
                },
                "G_sensor_model": {
                    "type": "sensor",
                    "description": "Sensor data processing and fusion model",
                    "architecture": "lstm",
                    "input_types": ["sensor_data"],
                    "output_types": ["sensor_readings", "predictions"],
                    "local_model_path": "data/models/G_sensor_model",
                    "external_api_support": False,
                    "active": True
                },
                "H_computer_model": {
                    "type": "control",
                    "description": "Computer system control and automation model",
                    "architecture": "rule-based-transformer",
                    "input_types": ["commands"],
                    "output_types": ["system_commands"],
                    "local_model_path": "data/models/H_computer_model",
                    "external_api_support": False,
                    "active": True
                },
                "I_motion_model": {
                    "type": "control",
                    "description": "Motion and actuator control model",
                    "architecture": "mpc",
                    "input_types": ["target_positions", "current_state"],
                    "output_types": ["control_signals"],
                    "local_model_path": "data/models/I_motion_model",
                    "external_api_support": False,
                    "active": True
                },
                "J_knowledge_model": {
                    "type": "knowledge",
                    "description": "Comprehensive knowledge base and reasoning model",
                    "architecture": "retrieval-augmented-transformer",
                    "input_types": ["queries"],
                    "output_types": ["answers", "explanations"],
                    "local_model_path": "data/models/J_knowledge_model",
                    "external_api_support": True,
                    "active": True
                },
                "K_programming_model": {
                    "type": "code",
                    "description": "Code generation and analysis model",
                    "architecture": "code-llama",
                    "input_types": ["code", "descriptions"],
                    "output_types": ["code", "explanations"],
                    "local_model_path": "data/models/K_programming_model",
                    "external_api_support": True,
                    "active": True
                }
            }
        }
        
        # Write configuration files
        configs = {
            CONFIG_FILES[0]: training_config,
            CONFIG_FILES[1]: api_config,
            CONFIG_FILES[2]: model_config
        }
        
        for config_path, config_content in configs.items():
            full_path = os.path.join(PROJECT_ROOT, config_path)
            
            if os.path.exists(full_path):
                if self.force:
                    logger.warning(f"Configuration file {config_path} already exists, overwriting...")
                else:
                    logger.info(f"Configuration file {config_path} already exists, skipping...")
                    continue
            
            try:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                
                # Write the configuration file
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(config_content, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Created configuration file: {config_path}")
                
            except Exception as e:
                logger.error(f"Failed to create configuration file {config_path}: {str(e)}")
                return False
        
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("Installing Python dependencies...")
        
        requirements_path = os.path.join(PROJECT_ROOT, "web_interface", "training_manager", "requirements.txt")
        
        if not os.path.exists(requirements_path):
            logger.error(f"Requirements file not found: {requirements_path}")
            return False
        
        try:
            # Install dependencies using pip
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', requirements_path],
                check=True
            )
            
            logger.info("Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {str(e)}")
            return False
        
        return True
    
    def initialize(self):
        """Run the complete initialization process"""
        logger.info("Starting Training Manager initialization...")
        
        # Step 1: Create directories
        if not self.create_directories():
            logger.error("Failed to create directories, initialization aborted")
            return False
        
        # Step 2: Create configuration files
        if not self.create_config_files():
            logger.error("Failed to create configuration files, initialization aborted")
            return False
        
        # Step 3: Install dependencies
        if not self.install_dependencies():
            logger.warning("Failed to install some dependencies, but continuing with initialization")
        
        # Final message
        logger.info("\nTraining Manager initialization completed successfully!\n")
        logger.info("Next steps:")
        logger.info("1. Start the Training API server with: python web_interface/training_manager/start_training_api.py")
        logger.info("2. Upload training data via the API or place it in the data/raw directory")
        logger.info("3. Train models using the API or directly through the ModelTrainer classes")
        
        return True

def main():
    """Main function to parse arguments and run initialization"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Self Brain Training Manager Initialization')
    parser.add_argument('--force', action='store_true', help='Force overwrite of existing files and directories')
    parser.add_argument('--skip-dirs', action='store_true', help='Skip creating directories')
    parser.add_argument('--skip-config', action='store_true', help='Skip creating configuration files')
    parser.add_argument('--skip-deps', action='store_true', help='Skip installing dependencies')
    
    args = parser.parse_args()
    
    # Create initializer
    initializer = TrainingManagerInitializer(force=args.force)
    
    # Run initialization with selected options
    if args.skip_dirs and args.skip_config and args.skip_deps:
        logger.warning("All initialization steps skipped. Nothing to do.")
        return
    
    # Create directories if not skipped
    if not args.skip_dirs:
        if not initializer.create_directories():
            logger.error("Failed to create directories, initialization aborted")
            sys.exit(1)
    
    # Create configuration files if not skipped
    if not args.skip_config:
        if not initializer.create_config_files():
            logger.error("Failed to create configuration files, initialization aborted")
            sys.exit(1)
    
    # Install dependencies if not skipped
    if not args.skip_deps:
        if not initializer.install_dependencies():
            logger.warning("Failed to install some dependencies")
    
    logger.info("Initialization completed with selected options")

if __name__ == '__main__':
    main()