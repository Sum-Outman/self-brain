#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self Brain AGI System Startup Script (Fixed for Flask)
This script initializes and starts the Self Brain AGI system components with Flask support.
"""

import os
import sys
import json
import logging
import time
import subprocess
import threading
import webbrowser
import signal
import socket
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SelfBrainSystem")

# Define base directory
BASE_DIR = Path(__file__).parent

# Global process tracking dictionary
processes = {}

# Graceful shutdown handler
def signal_handler(sig, frame):
    """Handle system signals for graceful shutdown"""
    logger.info("Received shutdown signal, stopping all services...")
    stop_all_services()
    sys.exit(0)

# Load configuration
def load_config(config_path="config/system_config.yaml"):
    """Load system configuration"""
    try:
        config_file = BASE_DIR / config_path
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    try:
                        import yaml
                        return yaml.safe_load(f)
                    except ImportError:
                        logger.warning("PyYAML not installed, cannot load YAML configuration")
                        return {}
                elif config_path.endswith('.json'):
                    return json.load(f)
                else:
                    logger.error(f"Unsupported config file format: {config_path}")
                    return {}
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        # Return default configuration if loading fails
        return {
            "ports": {
                "manager": 5000,           # Main Web Interface
                "management_model": 5001,  # A Management Model
                "language": 5002,          # Language Model
                "audio": 5003,             # Audio Processing Model
                "image": 5004,             # Image Processing Model
                "video": 5005,             # Video Processing Model
                "spatial": 5006,           # Spatial Perception Model
                "sensor": 5007,            # Sensor Model
                "computer_control": 5008,  # Computer Control Model
                "actuator": 5009,          # Motion & Actuator Control
                "expert": 5010,            # Knowledge Expert Model
                "programming": 5011,       # Programming Model
                "training_manager": 5012,  # Training Manager
                "device_manager": 5013,    # Device Manager
                "standalone_manager": 5014,# Standalone Manager
                "manager_api": 5015        # Manager Model API
            },
            "models": {
                "local_models": {
                    "A_management": True,
                    "B_language": True,
                    "C_audio": True,
                    "D_image": True,
                    "E_video": True,
                    "F_spatial": True,
                    "G_sensor": True,
                    "H_computer_control": True,
                    "I_motion_control": True,
                    "J_knowledge": True,
                    "K_programming": True
                }
            }
        }

# Check if port is in use
def check_port_in_use(port):
    """Check if a port is currently in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            return s.connect_ex(('localhost', port)) == 0
        except Exception as e:
            logger.error(f"Error checking port {port}: {str(e)}")
            return False

# Initialize system
def initialize_system():
    """Initialize the system environment"""
    logger.info("Initializing system environment...")
    initialize_script = BASE_DIR / "initialize_system.py"
    if initialize_script.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(initialize_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            if result.returncode == 0:
                logger.info("System initialization completed successfully")
            else:
                logger.error(f"System initialization failed with exit code {result.returncode}")
                if result.stderr:
                    logger.error(f"Initialization error: {result.stderr}")
        except Exception as e:
            logger.error(f"Error during system initialization: {str(e)}")
    else:
        logger.warning("System initialization script not found, skipping initialization")

# Start Flask submodel
def start_flask_submodel(model_name, port):
    """启动Flask子模型服务 | Start Flask submodel service"""
    model_dir = BASE_DIR / "sub_models" / model_name
    app_file = model_dir / "app.py"
    
    if not model_dir.exists() or not app_file.exists():
        logger.error(f"Model directory or app.py not found for {model_name}")
        return False
    
    try:
        # Check if port is in use
        if check_port_in_use(port):
            logger.error(f"Port {port} is already in use, cannot start {model_name}")
            return False
        
        # Set environment variables for Flask
        env = os.environ.copy()
        env["PORT"] = str(port)
        env["HOST"] = "0.0.0.0"
        env["FLASK_APP"] = str(app_file)
        env["FLASK_ENV"] = "development"
        
        # Start the Flask application
        process = subprocess.Popen(
            [sys.executable, str(app_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            universal_newlines=True,
            cwd=model_dir,
            env=env
        )
        
        # Store process information
        processes[f"model_{model_name}"] = {
            "process": process,
            "port": port,
            "start_time": time.time()
        }
        
        logger.info(f"{model_name} started on port {port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start {model_name}: {str(e)}")
        return False

# Start all models
def start_all_models():
    """Start all models sequentially"""
    logger.info("Starting all models...")
    config = load_config()
    
    # Get ports from config
    ports = {
        'management_model': config.get('ports', {}).get('management_model', 5001),
        'language': config.get('ports', {}).get('language', 5002),
        'audio': config.get('ports', {}).get('audio', 5003),
        'image': config.get('ports', {}).get('image', 5004),
        'video': config.get('ports', {}).get('video', 5005),
        'spatial': config.get('ports', {}).get('spatial', 5006),
        'sensor': config.get('ports', {}).get('sensor', 5007),
        'computer_control': config.get('ports', {}).get('computer_control', 5008),
        'actuator': config.get('ports', {}).get('actuator', 5009),
        'expert': config.get('ports', {}).get('expert', 5010),
        'programming': config.get('ports', {}).get('programming', 5011)
    }
    
    # Map model names to ports based on system_config.yaml
    model_ports = {
        "A_management": ports.get('management_model'),
        "B_language": ports.get('language'),
        "C_audio": ports.get('audio'),
        "D_image": ports.get('image'),
        "E_video": ports.get('video'),
        "F_spatial": ports.get('spatial'),
        "G_sensor": ports.get('sensor'),
        "H_computer_control": ports.get('computer_control'),
        "I_motion_control": ports.get('actuator'),
        "J_knowledge": ports.get('expert'),
        "K_programming": ports.get('programming')
    }
    
    # Start each model with delay to avoid port conflicts
    for model_name, port in model_ports.items():
        if start_flask_submodel(model_name, port):
            time.sleep(2)  # Wait 2 seconds between starting models
        else:
            logger.warning(f"Failed to start {model_name}")
    
    logger.info("All submodels startup processes initiated")

# Start web interface
def start_web_interface():
    """Start web interface service"""
    logger.info("Starting web interface...")
    web_interface_dir = BASE_DIR / "web_interface"
    config = load_config()
    web_port = config.get('ports', {}).get('manager', 5000)
    
    # Skip if already running
    if "web_interface" in processes:
        logger.warning("Web interface is already running, skipping start")
        return True
    
    # Check if port is in use
    if check_port_in_use(web_port):
        logger.error(f"Port {web_port} is already in use, cannot start web interface")
        return False
    
    # Check if web interface directory exists
    if not web_interface_dir.exists():
        logger.error("Web interface directory not found")
        return False
    
    try:
        # Get absolute path to app.py
        app_py_path = os.path.join(web_interface_dir, "app.py")
        
        # Set environment variables for web interface
        env = os.environ.copy()
        env["PORT"] = str(web_port)
        env["HOST"] = "0.0.0.0"
        env["FLASK_APP"] = "app.py"
        env["FLASK_ENV"] = "development"
        env["SUPPORT_MULTI_CAMERA"] = "true"
        env["SUPPORT_EXTERNAL_DEVICES"] = "true"
        
        process = subprocess.Popen(
            [sys.executable, app_py_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            universal_newlines=True,
            cwd=web_interface_dir,
            env=env
        )
        
        # Store process information
        processes["web_interface"] = {
            "process": process,
            "port": web_port,
            "start_time": time.time()
        }
        
        logger.info(f"Web interface started on port {web_port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start web interface: {str(e)}")
        return False

# Monitor processes
def monitor_processes():
    """Monitor system processes and restart if needed"""
    logger.info("Starting process monitoring...")
    while True:
        time.sleep(10)  # Check every 10 seconds instead of 5
        for name, proc_info in list(processes.items()):
            process = proc_info["process"]
            if process.poll() is not None:
                # Process has exited
                logger.warning(f"Process {name} has exited with code {process.returncode}")
                
                # Restart web interface if it exits
                if name == "web_interface":
                    logger.info("Attempting to restart web interface...")
                    start_web_interface()
                elif name.startswith("model_"):
                    # Extract model name from process key
                    model_name = name.replace("model_", "")
                    port = proc_info["port"]
                    logger.info(f"Attempting to restart model {model_name}...")
                    start_flask_submodel(model_name, port)
        
        # Sleep before next check

# Stop all services
def stop_all_services():
    """Stop all running services"""
    logger.info("Stopping all services...")
    
    # Stop all processes
    for name, proc_info in list(processes.items()):
        process = proc_info["process"]
        try:
            process.terminate()
            process.wait(timeout=5)  # Wait for process to terminate
            logger.info(f"Stopped {name}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout waiting for {name} to terminate, killing forcefully")
            process.kill()
        except Exception as e:
            logger.error(f"Error stopping {name}: {str(e)}")
    
    # Clear process list
    processes.clear()
    logger.info("All services stopped")

# Verify system status
def verify_system_status():
    """Verify that all critical components are running"""
    logger.info("Verifying system status...")
    
    # Check if web interface is running
    web_interface_running = "web_interface" in processes
    
    # Check if at least one model service is running
    model_services_running = any(name.startswith("model_") for name in processes)
    
    # Log status
    logger.info(f"Web interface running: {web_interface_running}")
    logger.info(f"Model services running: {model_services_running}")
    logger.info(f"Total services running: {len(processes)}")
    
    # Provide access information
    if web_interface_running:
        web_port = processes["web_interface"]["port"]
        logger.info(f"System is accessible at http://localhost:{web_port}")
    
    return web_interface_running

# Initialize training directories
def init_training_directories():
    """Initialize training data directories for trainable models"""
    training_root = BASE_DIR / "training_data"
    
    # Load config to get model information
    config = load_config()
    
    # Create training directories for each model
    model_names = ["A_management", "B_language", "C_audio", "D_image", "E_video", 
                  "F_spatial", "G_sensor", "H_computer_control", "I_motion_control", 
                  "J_knowledge", "K_programming"]
    
    for model_name in model_names:
        model_dir = BASE_DIR / "sub_models" / model_name
        training_dir = model_dir / "training_data"
        
        # Create training directory if it doesn't exist
        if not training_dir.exists():
            training_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created training directory: {training_dir}")
            
            # For models that support training from scratch, create initial config
            from_scratch_config = training_dir / "from_scratch_config.json"
            if not from_scratch_config.exists():
                with open(from_scratch_config, 'w', encoding='utf-8') as f:
                    json.dump({
                        "model_id": model_name,
                        "from_scratch": True,
                        "training_params": {
                            "epochs": 100,
                            "batch_size": 32,
                            "learning_rate": 0.001
                        },
                        "created_at": datetime.now().isoformat()
                    }, f, ensure_ascii=False, indent=2)
                logger.info(f"Created from-scratch training config for {model_name}")

# Wait for services to be ready
def wait_for_service(port, timeout=30):
    """Wait for a service to be ready on the specified port"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_port_in_use(port):
            return True
        time.sleep(1)
    return False

# Main function
def main():
    """主启动函数 | Main startup function"""
    logger.info("===== Self Brain AGI System Startup (Fixed) =====")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize the system
    initialize_system()
    
    # Initialize training directories
    init_training_directories()
    
    # Check and install dependencies
    logger.info("Checking and installing dependencies...")
    try:
        requirements_file = BASE_DIR / "requirements.txt"
        if requirements_file.exists():
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            logger.info("Dependencies installed successfully")
        else:
            logger.warning("requirements.txt not found, skipping dependency installation")
    except Exception as e:
        logger.error(f"Error during dependency installation: {str(e)}")
    
    # Start web interface first
    web_started = start_web_interface()
    if not web_started:
        logger.error("Failed to start web interface, cannot continue")
        sys.exit(1)
    
    # Wait for web interface to be ready
    config = load_config()
    web_port = config.get('ports', {}).get('manager', 5000)
    if wait_for_service(web_port):
        logger.info("Web interface is ready")
    else:
        logger.warning("Web interface may not be fully ready")
    
    # Start all submodels
    start_all_models()
    
    # Start process monitor in a separate thread
    monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
    monitor_thread.start()
    
    # Verify system status
    verify_system_status()
    
    # Open browser after a delay
    def open_browser():
        time.sleep(8)  # Wait for services to start
        logger.info("Opening browser to access Self Brain AGI system...")
        try:
            webbrowser.open(f'http://localhost:{web_port}')
        except Exception as e:
            logger.error(f"Failed to open browser: {str(e)}")
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.start()
    
    logger.info("="*50)
    logger.info("Self Brain AGI System startup completed!")
    logger.info(f"Web interface available at http://localhost:{web_port}")
    logger.info("="*50)
    
    # Keep main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nSystem shutting down...")
    finally:
        stop_all_services()

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
