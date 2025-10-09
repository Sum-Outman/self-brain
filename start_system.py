#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self Brain AGI System Startup Script
This script initializes and starts the Self Brain AGI system components.
"""

import os
import sys
import json
import logging
import time
import subprocess
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
import signal
<<<<<<< HEAD
import traceback

# Debug flag and additional logging
DEBUG_MODE = True

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SelfBrainSystem")

# Add debug print function
def debug_print(msg):
    if DEBUG_MODE:
        print("[DEBUG] %s - %s" % (time.strftime('%H:%M:%S'), msg))
        logger.debug(msg)
=======
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SelfBrainSystem")

# Define base directory as absolute path
BASE_DIR = Path(__file__).parent.absolute()

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
            # Create default config file
            default_config = {
                "ports": {
                    "manager": 5000,
                    "language": 5001,
                    "audio": 5002,
                    "image": 5003,
                    "video": 5004,
                    "spatial": 5005,
                    "sensor": 5006,
                    "computer_control": 5007,
                    "knowledge": 5008,
                    "motion": 5009,
                    "programming": 5010,
                    "web_backend": 8000,
                    "web_frontend": 8080,
                    "manager_api": 5015,
                    "agi_core": 5014,
                    "device_manager": 5013
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
                        "I_knowledge": True,
<<<<<<< HEAD
                        "J_motion": {
                "enabled": True,
                "priority": 1
            },
=======
                        "J_motion": True,
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                        "K_programming": True
                    }
                }
            }
            # Ensure config directory exists
            config_dir = config_file.parent
            config_dir.mkdir(parents=True, exist_ok=True)
            # Save default config
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Created default configuration file at {config_file}")
            return default_config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        # Return default configuration if loading fails
        return {
            "ports": {
                "manager": 5000,
                "language": 5001,
                "audio": 5002,
                "image": 5003,
                "video": 5004,
                "spatial": 5005,
                "sensor": 5006,
                "computer_control": 5007,
                "knowledge": 5008,
                "motion": 5009,
                "programming": 5010,
                "web_backend": 8000,
                "web_frontend": 8080,
                "manager_api": 5015,
                "agi_core": 5014,
                "device_manager": 5013
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
                    "I_knowledge": True,
                    "J_motion": True,
                    "K_programming": True
                }
            }
        }

# Check if port is in use
def check_port_in_use(port):
    """Check if a port is currently in use"""
    import socket
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
    
    # Create necessary directories
    required_dirs = [
        BASE_DIR / "data" / "training" / "language",
        BASE_DIR / "data" / "training" / "audio",
        BASE_DIR / "data" / "training" / "image",
        BASE_DIR / "data" / "training" / "video",
        BASE_DIR / "data" / "training" / "knowledge",
        BASE_DIR / "data" / "cache",
        BASE_DIR / "logs",
        BASE_DIR / "models"
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
        
    # Create initialize_system.py if not exists
    initialize_script = BASE_DIR / "initialize_system.py"
    if not initialize_script.exists():
        with open(initialize_script, 'w', encoding='utf-8') as f:
            f.write('''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self Brain AGI System Initialization Script
This script initializes the system environment for Self Brain AGI."""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SelfBrainInitializer")

# Define base directory
BASE_DIR = Path(__file__).parent.absolute()

# Generate sample training data
logger.info("Generating sample training data...")

# Sample language training data
def generate_language_data():
    """Generate sample language training data"""
    data_dir = BASE_DIR / "data" / "training" / "language"
    sample_data = [
        {"text": "Hello, how can I help you today?", "intent": "greeting"},
        {"text": "What is the weather like?", "intent": "weather_query"},
        {"text": "Tell me a joke", "intent": "request_joke"},
        {"text": "Explain quantum computing", "intent": "request_explanation"},
        {"text": "Goodbye", "intent": "farewell"}
    ]
    
    # Generate 100 samples
    all_samples = []
    for i in range(100):
        base = random.choice(sample_data)
        # Make slight variations
        if random.random() > 0.7:
            text = base["text"] + "?" if not base["text"].endswith("?") else base["text"][:-1]
        else:
            text = base["text"]
        all_samples.append({"text": text, "intent": base["intent"]})
    
    with open(data_dir / "sample_data.json", 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2)
    logger.info(f"Generated language training data: {data_dir / 'sample_data.json'}")

# Sample audio training data
def generate_audio_data():
    """Generate placeholder for audio training data"""
    data_dir = BASE_DIR / "data" / "training" / "audio"
    # Create placeholder file
    with open(data_dir / "audio_placeholder.txt", 'w', encoding='utf-8') as f:
        f.write("Audio training data will be stored here.\\n")
        f.write("This includes WAV files and corresponding transcriptions.")
    logger.info(f"Created audio training data placeholder: {data_dir / 'audio_placeholder.txt'}")

# Sample image training data
def generate_image_data():
    """Generate placeholder for image training data"""
    data_dir = BASE_DIR / "data" / "training" / "image"
    # Create placeholder file
    with open(data_dir / "image_placeholder.txt", 'w', encoding='utf-8') as f:
        f.write("Image training data will be stored here.\\n")
        f.write("This includes image files and corresponding labels.")
    logger.info(f"Created image training data placeholder: {data_dir / 'image_placeholder.txt'}")

# Sample video training data
def generate_video_data():
    """Generate placeholder for video training data"""
    data_dir = BASE_DIR / "data" / "training" / "video"
    # Create placeholder file
    with open(data_dir / "video_placeholder.txt", 'w', encoding='utf-8') as f:
        f.write("Video training data will be stored here.\\n")
        f.write("This includes video files and corresponding annotations.")
    logger.info(f"Created video training data placeholder: {data_dir / 'video_placeholder.txt'}")

# Sample knowledge training data
def generate_knowledge_data():
    """Generate sample knowledge training data"""
    data_dir = BASE_DIR / "data" / "training" / "knowledge"
    sample_knowledge = [
        {"domain": "physics", "content": "Newton's laws of motion describe the relationship between a physical object and the forces acting upon it."},
        {"domain": "mathematics", "content": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides."},
        {"domain": "chemistry", "content": "Water is composed of two hydrogen atoms and one oxygen atom, with the chemical formula H2O."},
        {"domain": "biology", "content": "DNA is a molecule that carries genetic instructions for the development, functioning, growth and reproduction of all known organisms."},
        {"domain": "computer science", "content": "Machine learning is a method of data analysis that automates analytical model building."}
    ]
    
    with open(data_dir / "sample_knowledge.json", 'w', encoding='utf-8') as f:
        json.dump(sample_knowledge, f, indent=2)
    logger.info(f"Generated knowledge training data: {data_dir / 'sample_knowledge.json'}")

# Create models registry
def create_models_registry():
    """Create models registry file"""
    registry_path = BASE_DIR / "training_manager" / "models_registry.json"
    registry_dir = registry_path.parent
    registry_dir.mkdir(parents=True, exist_ok=True)
    
    models_registry = {
        "models": [
<<<<<<< HEAD
            {"id": "A_management", "name": "Management Model", "type": "manager", "port": 5000, "priority": 1},
=======
            {"id": "A_management", "name": "Management Model", "type": "manager", "port": 5000},
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            {"id": "B_language", "name": "Language Model", "type": "language", "port": 5001},
            {"id": "C_audio", "name": "Audio Model", "type": "audio", "port": 5002},
            {"id": "D_image", "name": "Image Model", "type": "vision", "port": 5003},
            {"id": "E_video", "name": "Video Model", "type": "vision", "port": 5004},
            {"id": "F_spatial", "name": "Spatial Model", "type": "spatial", "port": 5005},
            {"id": "G_sensor", "name": "Sensor Model", "type": "sensor", "port": 5006},
            {"id": "H_computer_control", "name": "Computer Control Model", "type": "control", "port": 5007},
            {"id": "I_knowledge", "name": "Knowledge Model", "type": "knowledge", "port": 5008},
            {"id": "J_motion", "name": "Motion Control Model", "type": "control", "port": 5009},
            {"id": "K_programming", "name": "Programming Model", "type": "programming", "port": 5010}
        ]
    }
    
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(models_registry, f, indent=2)
    logger.info(f"Created models registry: {registry_path}")

# Main initialization
if __name__ == "__main__":
    try:
        generate_language_data()
        generate_audio_data()
        generate_image_data()
        generate_video_data()
        generate_knowledge_data()
        create_models_registry()
        
        logger.info("Self Brain AGI System initialization completed successfully!")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        sys.exit(1)
''')
        
    # Run initialization script
    try:
        result = subprocess.run(
            [sys.executable, str(initialize_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8'
        )
        if result.returncode == 0:
            logger.info("System initialization completed successfully")
        else:
            logger.error(f"System initialization failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"Initialization error: {result.stderr}")
    except Exception as e:
        logger.error(f"Error during system initialization: {str(e)}")
    
    logger.info("System environment initialization completed")
# Start submodel
def start_submodel(model_name, port):
    """启动子模型服务 | Start submodel service"""
    model_dir = BASE_DIR / "sub_models" / model_name
    
    # Check if model directory exists
    if not model_dir.exists():
        logger.error(f"Model directory for {model_name} does not exist at {model_dir}")
        return False
    
    # Check if app.py exists in the model directory
    app_file = model_dir / "app.py"
    if not app_file.exists():
        logger.error(f"app.py not found in {model_name} directory")
        return False
    
    try:
        # Check if port is in use
        if check_port_in_use(port):
            logger.error(f"Port {port} is already in use, cannot start {model_name}")
            return False
        
        # Change to model directory
        os.chdir(model_dir)
        
<<<<<<< HEAD
        # Start the model service with explicit port parameter
        cmd = [sys.executable, "app.py", "--port", str(port)]
=======
        # Start the model service - use the existing Flask app
        cmd = [sys.executable, "app.py"]
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            encoding='utf-8'
        )
        
        # Restore current directory
        os.chdir(BASE_DIR)
        
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
        # Restore current directory in case of error
        os.chdir(BASE_DIR)
        return False

# Start AGI core
def start_agi_core():
    """Start AGI core system"""
    logger.info("Starting AGI core system...")
    agi_core_dir = BASE_DIR / "agi_core"
    config = load_config()
    agi_core_port = config.get('ports', {}).get('agi_core', 5014)
    
    # Skip if already running
    if "agi_core" in processes:
        logger.warning("AGI core is already running, skipping start")
        return True
    
    # Check if port is in use
    if check_port_in_use(agi_core_port):
        logger.error(f"Port {agi_core_port} is already in use, cannot start AGI core")
        return False
    
    # Create AGI core directory if it doesn't exist
    if not agi_core_dir.exists():
        logger.warning("AGI core directory not found, creating...")
        agi_core_dir.mkdir(parents=True, exist_ok=True)
        
        # Create basic app.py file
        app_file = agi_core_dir / "app.py"
        with open(app_file, 'w', encoding='utf-8') as f:
            content = '''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AGICore")

app = FastAPI(title="Self Brain AGI Core")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "AGI Core", 
        "port": os.environ.get("PORT"),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5014))
    logger.info(f"Starting AGI Core service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
            f.write(content)
    
    try:
        # Change to AGI core directory
        os.chdir(agi_core_dir)
        
        # Start the AGI core
        cmd = [sys.executable, "app.py"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            encoding='utf-8'
        )
        
        # Restore current directory
        os.chdir(BASE_DIR)
        
        # Store process information
        processes["agi_core"] = {
            "process": process,
            "port": agi_core_port,
            "start_time": time.time()
        }
        
        logger.info(f"AGI core started on port {agi_core_port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start AGI core: {str(e)}")
        # Restore current directory in case of error
        os.chdir(BASE_DIR)
        return False

# Start device manager
def start_device_manager():
    """Start device manager service"""
    logger.info("Starting device manager...")
    device_manager_dir = BASE_DIR / "device_manager"
    config = load_config()
    device_manager_port = config.get('ports', {}).get('device_manager', 5013)
    
    # Skip if already running
    if "device_manager" in processes:
        logger.warning("Device manager is already running, skipping start")
        return True
    
    # Check if port is in use
    if check_port_in_use(device_manager_port):
        logger.error(f"Port {device_manager_port} is already in use, cannot start device manager")
        return False
    
    # Create device manager directory if it doesn't exist
    if not device_manager_dir.exists():
        logger.warning("Device manager directory not found, creating...")
        device_manager_dir.mkdir(parents=True, exist_ok=True)
        
        # Create basic app.py file
        app_file = device_manager_dir / "app.py"
        with open(app_file, 'w', encoding='utf-8') as f:
            content = '''
from fastapi import FastAPI
import uvicorn
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeviceManager")

app = FastAPI(title="Self Brain Device Manager")

@app.get("/api/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Device Manager",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/devices")
def list_devices():
    """List connected devices"""
    # This is a placeholder implementation
    return {"devices": [], "count": 0}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5013))
    logger.info(f"Starting Device Manager on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
            f.write(content)
    
    try:
        # Change to device manager directory
        os.chdir(device_manager_dir)
        
        # Start the device manager
        cmd = [sys.executable, "app.py"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            encoding='utf-8'
        )
        
        # Restore current directory
        os.chdir(BASE_DIR)
        
        # Store process information
        processes["device_manager"] = {
            "process": process,
            "port": device_manager_port,
            "start_time": time.time()
        }
        
        logger.info(f"Device manager started on port {device_manager_port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start device manager: {str(e)}")
        # Restore current directory in case of error
        os.chdir(BASE_DIR)
        return False

# Start web interface
def start_web_interface():
    """Start web interface service"""
    logger.info("Starting web interface...")
    web_interface_dir = BASE_DIR / "web_interface"
    config = load_config()
    web_port = config.get('ports', {}).get('web_frontend', 8080)
    
    # Skip if already running
    if "web_interface" in processes:
        logger.warning("Web interface is already running, skipping start")
        return True
    
    # Check if port is in use
    if check_port_in_use(web_port):
        logger.error(f"Port {web_port} is already in use, cannot start web interface")
        return False
    
    # Check if web interface directory exists
    if not os.path.isdir(str(web_interface_dir)):
        logger.error(f"Web interface directory not found at {web_interface_dir.absolute()}")
        logger.error(f"Current working directory: {os.getcwd()}")
        return False
    
    try:
        # Change to web interface directory
        os.chdir(web_interface_dir)
        
        # Start the web interface
        cmd = [sys.executable, "app.py"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            encoding='utf-8'
        )
        
        # Restore current directory
        os.chdir(BASE_DIR)
        
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
        # Restore current directory in case of error
        os.chdir(BASE_DIR)
        return False

# Start manager model
<<<<<<< HEAD

=======
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
def start_manager_model():
    """Start manager model service"""
    logger.info("Starting manager model...")
    manager_dir = BASE_DIR / "manager_model"
    config = load_config()
<<<<<<< HEAD
    # Use the manager port (5000) instead of manager_api port (5015)
    manager_port = config.get('ports', {}).get('manager', 5000)
=======
    manager_port = config.get('ports', {}).get('manager_api', 5015)
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    
    # Skip if already running
    if "manager_model" in processes:
        logger.warning("Manager model is already running, skipping start")
        return True
    
    # Check if port is in use
    if check_port_in_use(manager_port):
        logger.error(f"Port {manager_port} is already in use, cannot start manager model")
        return False
    
    # Create manager directory if it doesn't exist
    if not manager_dir.exists():
        logger.warning("Manager model directory not found, creating...")
        manager_dir.mkdir(parents=True, exist_ok=True)
        
<<<<<<< HEAD
        # Create basic app.py file - Using Flask framework compatible with existing code
        app_file = manager_dir / "app.py"
        with open(app_file, 'w', encoding='utf-8') as f:
            content = '''
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
=======
        # Create basic app.py file
        app_file = manager_dir / "app.py"
        with open(app_file, 'w', encoding='utf-8') as f:
            content = '''
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import logging
import time
from datetime import datetime
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
<<<<<<< HEAD
logger = logging.getLogger('A_Management_Model')

# Application initialization
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Simple login authentication decorator
def login_required(f):
    """Decorator to verify if user is logged in
    In production, replace with more secure authentication mechanism
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if authentication is enabled (read from config or use default)
        auth_enabled = False
        
        if not auth_enabled:
            return f(*args, **kwargs)
        
        # Simple token-based authentication
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({"error": "Unauthorized access"}), 401
            
        # In a real system, validate the token against a database or authentication service
        valid_tokens = ["test_token"]
        if token[7:] not in valid_tokens:
            return jsonify({"error": "Invalid token"}), 401
        
        return f(*args, **kwargs)
    return decorated_function

# ManagementModel class with proper initialization parameters
class ManagementModel:
    """
    Main management model for Self Brain system
    Manages all sub-models and coordinates system operations
    """
    def __init__(self, submodel_registry=None):
        # Initialize submodel registry
        self.submodel_registry = submodel_registry or {}
        
        # System information
        self.system_name = "Self Brain"
        self.version = "1.0.0"
        self.team_email = "silencecrowtom@qq.com"
        
        # System status
        self.system_status = "online"
        self.last_health_check = datetime.now()
        
        # Initialize request queue and history
        self.request_queue = deque(maxlen=100)
        self.request_history = []
        
        # Initialize emotion engine
        self.emotion_engine = None
        
        logger.info("Manager model initialized successfully")
    
    def process_user_input(self, user_input, context=None):
        """Process user input and generate response"""
        # This is a placeholder implementation
        # In a real system, this would analyze the input and coordinate with appropriate submodels
        
        # Simple echo response for testing
        response = {
            "status": "success",
            "message": f"Received: {user_input}",
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log the request
        self.log_request(user_input, response)
        
        return response
    
    def log_request(self, request_data, response_data):
        """Log request and response data"""
        request_info = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "request_data": request_data,
            "response_data": response_data
        }
        
        self.request_queue.append(request_info)
        self.request_history.append(request_info)
=======
logger = logging.getLogger("ManagerModel")

app = FastAPI(title="Self Brain Manager API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ManagerModel:
    """
    Manager model for Self Brain system
    Handles system-wide management operations
    """
    def __init__(self):
        self.system_name = "Self Brain"
        self.version = "1.0.0"
        self.team_email = "silencecrowtom@qq.com"
        self.models = {}
        self.system_status = "online"
        logger.info("Manager model initialized")
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    
    def get_system_info(self):
        """Get system information"""
        return {
            "system_name": self.system_name,
            "version": self.version,
            "team_email": self.team_email,
            "status": self.system_status,
<<<<<<< HEAD
            "last_health_check": self.last_health_check.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "active_models": len(self.submodel_registry)
        }

# Global manager model instance
manager_model = None

# Initialize manager model
def initialize_management_model():
    """Initialize the management model"""
    global manager_model
    
    try:
        # Create empty submodel registry
        submodel_registry = {}
        
        # Initialize the management model
        manager_model = ManagementModel(submodel_registry)
        
        logger.info("Management model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize management model: {str(e)}")
        return False

# Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Manager Model",
        "port": os.environ.get("PORT", "5000"),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get system information"""
    if manager_model:
        return jsonify(manager_model.get_system_info())
    else:
        return jsonify({"error": "Manager model not initialized"}), 503

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint for user interaction"""
    if not manager_model:
        return jsonify({"error": "Manager model not initialized"}), 503
    
    try:
        data = request.json
        user_input = data.get('text', '')
        context = data.get('context', None)
        
        if not user_input:
            return jsonify({"error": "No text provided"}), 400
        
        response = manager_model.process_user_input(user_input, context)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Main entry point
if __name__ == '__main__':
    # Initialize management model
    initialize_management_model()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Start the Flask server
    logger.info(f"Starting Manager Model service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
=======
            "timestamp": datetime.now().isoformat()
        }

# Create manager instance
manager = ManagerModel()

@app.get("/api/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Manager Model", 
        "port": os.environ.get("PORT"),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/system/info")
def system_info():
    """Get system information"""
    return manager.get_system_info()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5015))
    logger.info(f"Starting Manager Model service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
'''
            f.write(content)
    
    try:
        # Change to manager directory
        os.chdir(manager_dir)
        
        # Start the manager model
        cmd = [sys.executable, "app.py"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            text=True
        )
        
        # Restore current directory
        os.chdir(BASE_DIR)
        
        # Store process information
        processes["manager_model"] = {
            "process": process,
            "port": manager_port,
            "start_time": time.time()
        }
        
        logger.info(f"Manager model started on port {manager_port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start manager model: {str(e)}")
        # Restore current directory in case of error
        os.chdir(BASE_DIR)
        return False

# Start all models
def start_all_models():
    """Start all submodels"""
    logger.info("Starting all submodels...")
    
    # Load config to get model ports
    config = load_config()
    ports = config.get('ports', {})
    model_config = config.get('models', {})
    local_models = model_config.get('local_models', {})
    
    # Map model names to ports based on system_config.yaml
    model_ports = {
<<<<<<< HEAD
        "A_management": ports.get('manager', 5000),
=======
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        "B_language": ports.get('language', 5001),
        "C_audio": ports.get('audio', 5002),
        "D_image": ports.get('image', 5003),
        "E_video": ports.get('video', 5004),
        "F_spatial": ports.get('spatial', 5005),
        "G_sensor": ports.get('sensor', 5006),
        "H_computer_control": ports.get('computer_control', 5007),
        "I_knowledge": ports.get('knowledge', 5008),
        "J_motion": ports.get('motion', 5009),
        "K_programming": ports.get('programming', 5010)
    }
    
    # Start each model
<<<<<<< HEAD
    # Start each model if enabled in config
    for model_name, port in model_ports.items():
        if local_models.get(model_name, False):
            start_submodel(model_name, port)
        else:
            logger.info(f"Skipping {model_name} as it's disabled in config")
=======
    for model_name, port in model_ports.items():
        start_submodel(model_name, port)
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    
    logger.info("All submodels startup processes initiated")

# Monitor processes
def monitor_processes():
    """Monitor system processes and restart if needed"""
    logger.info("Starting process monitoring...")
    while True:
        for name, proc_info in list(processes.items()):
            process = proc_info["process"]
            if process.poll() is not None:
                # Process has exited
                logger.warning(f"Process {name} has exited with code {process.returncode}")
                
                # Restart web interface or manager model if they exit
                if name == "web_interface":
                    logger.info("Attempting to restart web interface...")
                    start_web_interface()
                elif name == "manager_model":
                    logger.info("Attempting to restart manager model...")
                    start_manager_model()
                elif name.startswith("model_"):
                    # Extract model name from process key
                    model_name = name.replace("model_", "")
                    port = proc_info["port"]
                    logger.info(f"Attempting to restart model {model_name}...")
                    start_submodel(model_name, port)
                elif name == "device_manager":
                    logger.info("Attempting to restart device manager...")
                    start_device_manager()
                elif name == "agi_core":
                    logger.info("Attempting to restart AGI core...")
                    start_agi_core()
        
        # Sleep before next check
        time.sleep(5)

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
    
    # Check if manager model is running
    manager_running = "manager_model" in processes
    
    # Check if at least one model service is running
    model_services_running = any(name.startswith("model_") for name in processes)
    
    # Check if AGI core is running
    agi_core_running = "agi_core" in processes
    
    # Log status
    logger.info(f"Web interface running: {web_interface_running}")
    logger.info(f"Manager model running: {manager_running}")
    logger.info(f"Model services running: {model_services_running}")
    logger.info(f"AGI core running: {agi_core_running}")
    logger.info(f"Device manager running: {'device_manager' in processes}")
    
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
    ports = config.get('ports', {})
    
    # Map model names to ports based on system_config.yaml
    model_ports = {
        "B_language": ports.get('language', 5001),
        "C_audio": ports.get('audio', 5002),
        "D_image": ports.get('image', 5003),
        "E_video": ports.get('video', 5004),
        "F_spatial": ports.get('spatial', 5005),
        "G_sensor": ports.get('sensor', 5006),
        "H_computer_control": ports.get('computer_control', 5007),
        "I_knowledge": ports.get('knowledge', 5008),
        "J_motion": ports.get('motion', 5009),
        "K_programming": ports.get('programming', 5010)
    }
    
    # Create training directories for each model
    for model_name in model_ports.keys():
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

# Main function
def main():
    """主启动函数 | Main startup function"""
<<<<<<< HEAD
    try:
        debug_print("=== SYSTEM STARTUP INITIATED ===")
        
        # Initialize the system
        debug_print("1. Initializing system environment...")
        initialize_system()
        debug_print("   - System environment initialized")
        
        # Initialize training directories
        debug_print("2. Initializing training directories...")
        init_training_directories()
        debug_print("   - Training directories initialized")
        
        # Check and install dependencies
        debug_print("3. Installing dependencies...")
        try:
            requirements_file = BASE_DIR / "requirements.txt"
            if requirements_file.exists():
                debug_print(f"   - Found requirements.txt: {requirements_file}")
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding='utf-8'
                )
                debug_print("   - Dependencies installed successfully")
            else:
                debug_print("   - requirements.txt not found, skipping dependency installation")
        except Exception as e:
            debug_print(f"   - Error during dependency installation: {str(e)}")
        
        # Start all submodels
        debug_print("4. Starting all submodels...")
        start_all_models()
        debug_print("   - Submodels startup processes initiated")
        
        # Start AGI core system
        debug_print("5. Starting AGI core...")
        agi_core_started = start_agi_core()
        debug_print(f"   - AGI core started: {agi_core_started}")
        
        # Start device manager
        debug_print("6. Starting device manager...")
        device_manager_started = start_device_manager()
        debug_print(f"   - Device manager started: {device_manager_started}")
        
        # Start web interface with multi-camera support
        debug_print("7. Configuring web interface environment...")
        web_interface_dir = BASE_DIR / "web_interface"
        web_env = os.environ.copy()
        web_env['SUPPORT_MULTI_CAMERA'] = "true"
        web_env['SUPPORT_EXTERNAL_DEVICES'] = "true"
        os.environ.update(web_env)
        debug_print("   - Web interface environment configured")
        
        debug_print("8. Starting web interface...")
        web_started = start_web_interface()
        debug_print(f"   - Web interface started: {web_started}")
        
        if not web_started:
            debug_print("   - FAILED to start web interface, cannot continue")
            stop_all_services()
            debug_print("Exiting with code 1")
            sys.exit(1)
        
        # Skip starting manager model as A_management is already running
        debug_print("9. A_management model already running as the system manager")
        debug_print("   - Skipping redundant manager model startup")
        
        # Start process monitor in a separate thread
        debug_print("10. Starting process monitoring thread...")
        monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
        monitor_thread.start()
        debug_print("   - Process monitoring thread started")
        
        # Verify system status
        debug_print("11. Verifying system status...")
        verify_system_status()
        debug_print("   - System status verification completed")
        
        # Open browser after a delay
        debug_print("12. Preparing to open browser...")
        def open_browser():
            time.sleep(5)  # Wait for services to start
            debug_print("   - Opening browser to access Self Brain AGI system...")
            try:
                # Get web port from config
                config = load_config()
                web_port = config.get('ports', {}).get('web_frontend', 8080)
                debug_print(f"   - Opening browser at http://localhost:{web_port}")
                webbrowser.open(f'http://localhost:{web_port}')
            except Exception as e:
                debug_print(f"   - Failed to open browser: {str(e)}")
        
        # Start browser thread
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.start()
        debug_print("   - Browser thread started")
        
        # Get web port from config for logging
        config = load_config()
        web_port = config.get('ports', {}).get('web_frontend', 8080)
        
        debug_print("="*50)
        debug_print("Self Brain AGI System startup completed!")
        debug_print(f"Web interface available at http://localhost:{web_port}")
        debug_print("="*50)
        
        # Keep main thread running
        debug_print("Entering main loop...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        debug_print("Keyboard interrupt detected, shutting down...")
    except Exception as e:
        debug_print(f"SYSTEM ERROR: {str(e)}")
        debug_print("Stack trace:")
        traceback.print_exc(file=sys.stdout)
    finally:
        debug_print("Stopping all services...")
        stop_all_services()
        debug_print("All services stopped, exiting.")
=======
    logger.info("===== Self Brain AGI System Startup =====")
    
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
                encoding='utf-8'
            )
            logger.info("Dependencies installed successfully")
        else:
            logger.warning("requirements.txt not found, skipping dependency installation")
    except Exception as e:
        logger.error(f"Error during dependency installation: {str(e)}")
    
    # Start all submodels
    start_all_models()
    
    # Start AGI core system
    start_agi_core()
    
    # Start device manager
    start_device_manager()
    
    # Start web interface with multi-camera support
    web_interface_dir = BASE_DIR / "web_interface"
    web_env = os.environ.copy()
    web_env['SUPPORT_MULTI_CAMERA'] = "true"
    web_env['SUPPORT_EXTERNAL_DEVICES'] = "true"
    os.environ.update(web_env)
    
    web_started = start_web_interface()
    if not web_started:
        logger.error("Failed to start web interface, cannot continue")
        stop_all_services()
        sys.exit(1)
    
    # Start manager model
    start_manager_model()
    
    # Start process monitor in a separate thread
    monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
    monitor_thread.start()
    
    # Verify system status
    verify_system_status()
    
    # Open browser after a delay
    def open_browser():
        time.sleep(5)  # Wait for services to start
        logger.info("Opening browser to access Self Brain AGI system...")
        try:
            # Get web port from config
            config = load_config()
            web_port = config.get('ports', {}).get('web_frontend', 8080)
            webbrowser.open(f'http://localhost:{web_port}')
        except Exception as e:
            logger.error(f"Failed to open browser: {str(e)}")
    
    # Start browser thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.start()
    
    # Get web port from config for logging
    config = load_config()
    web_port = config.get('ports', {}).get('web_frontend', 8080)
    
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
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
