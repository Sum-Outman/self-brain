
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import sys
import json
import logging
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add sub-models path to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../sub_models')))

# Import model management components
from unified_model_manager import get_model_manager, get_model
from A_management.app import ManagementApp
from manager_model.data_bus import DataBus, get_data_bus
from manager_model.model_registry import ModelRegistry, get_model_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AGICore")

# Initialize FastAPI app
app = FastAPI(title="Self Brain AGI Core")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../web_interface/static'))
templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../web_interface/templates'))

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Initialize core components
data_bus = get_data_bus()
model_registry = get_model_registry()
model_manager = get_model_manager(os.path.abspath(os.path.join(os.path.dirname(__file__), '../sub_models/unified_models_config.json')))

# Initialize management model (Model A)
management_app = None

def initialize_management_model():
    """Initialize the management model (Model A)"""
    global management_app
    try:
        management_app = ManagementApp()
        logger.info("Management model (Model A) initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize management model: {str(e)}")
        # Create a fallback management app
        class FallbackManagementApp:
            def predict(self, data):
                return {"response": "Management model is not available. Please check the logs."}
            def train(self, data):
                return {"status": "error", "message": "Management model is not available"}
            def evaluate(self, data):
                return {"status": "error", "message": "Management model is not available"}
        management_app = FallbackManagementApp()

@app.get("/")
def root(request: Request):
    """Root endpoint serving the main chat interface"""
    return templates.TemplateResponse("ai_chat.html", {"request": request})

@app.get("/api/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Self Brain AGI Core", 
        "version": "1.0",
        "port": os.environ.get("PORT", 5014),
        "timestamp": datetime.now().isoformat(),
        "models": model_manager.get_available_models()
    }

@app.get("/api/models")
def list_models():
    """List all available models"""
    return {
        "models": model_manager.get_available_models(),
        "status": model_manager.get_model_status()
    }

@app.post("/api/chat/enhanced/send")
async def send_enhanced_chat(request: Request):
    """Handle enhanced chat messages with multi-modal support"""
    try:
        data = await request.json()
        message = data.get('message', '')
        model_selection = data.get('model_selection', 'management')
        files = data.get('files', [])
        temperature = data.get('temperature', 0.7)
        style = data.get('style', 'default')
        session_id = data.get('session_id', 'default')
        context = data.get('context', [])

        if not message and not files:
            raise HTTPException(status_code=400, detail="Message or files are required")

        logger.info(f"Received chat request: model={model_selection}, message_length={len(message)}, files_count={len(files)}")

        # Prepare request data
        request_data = {
            'message': message,
            'files': files,
            'temperature': temperature,
            'style': style,
            'session_id': session_id,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }

        # Process the request through the management model
        response = management_app.predict(request_data)

        # Add additional information
        response['model_used'] = model_selection
        response['timestamp'] = datetime.now().isoformat()

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/{model_name}/train")
async def train_model(model_name: str, request: Request):
    """Train a specific model"""
    try:
        data = await request.json()
        training_data = data.get('data', [])
        epochs = data.get('epochs', 5)
        batch_size = data.get('batch_size', 16)
        learning_rate = data.get('learning_rate', 0.001)

        if not training_data:
            raise HTTPException(status_code=400, detail="Training data is required")

        logger.info(f"Starting training for model {model_name}: epochs={epochs}, batch_size={batch_size}")

        # Prepare training parameters
        training_params = {
            'data': training_data,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }

        # Train the model
        if model_name == 'management' and management_app:
            result = management_app.train(training_params)
        else:
            model = model_manager.get_model(model_name)
            if not model or not hasattr(model, 'train'):
                raise HTTPException(status_code=404, detail=f"Model {model_name} does not support training")
            result = model.train(**training_params)

        return JSONResponse(content=result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error training model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/self_learning/{action}")
def control_self_learning(action: str):
    """Control self-learning process for knowledge model"""
    try:
        if action not in ['start', 'stop', 'status']:
            raise HTTPException(status_code=400, detail="Action must be 'start', 'stop', or 'status'")

        # Send command to data bus
        data_bus.publish("self_learning_control", {'action': action})

        if action == 'status':
            # For status, we need to wait for a response
            status_response = {}
            def status_handler(data):
                nonlocal status_response
                status_response = data

            # Subscribe to status response
            subscription_id = data_bus.subscribe("self_learning_status", status_handler)
            time.sleep(1)  # Give time for response
            data_bus.unsubscribe(subscription_id)

            return JSONResponse(content=status_response or {"status": "no_response"})

        return JSONResponse(content={"status": "success", "action": action})

    except Exception as e:
        logger.error(f"Error controlling self-learning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/add")
async def add_knowledge(request: Request):
    """Add new knowledge to the knowledge base"""
    try:
        data = await request.json()
        content = data.get('content', '')
        domain = data.get('domain', 'general')
        source = data.get('source', 'unknown')

        if not content:
            raise HTTPException(status_code=400, detail="Content is required")

        # Publish knowledge update
        data_bus.publish("knowledge_update", {
            'content': content,
            'domain': domain,
            'source': source
        })

        return JSONResponse(content={"status": "success", "message": "Knowledge added successfully"})

    except Exception as e:
        logger.error(f"Error adding knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings")
def get_settings():
    """Get current system settings"""
    try:
        # Get model configurations
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sub_models/unified_models_config.json'))
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Get model status
        model_status = model_manager.get_model_status()

        return JSONResponse(content={
            "config": config,
            "model_status": model_status
        })

    except Exception as e:
        logger.error(f"Error getting settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings")
async def update_settings(request: Request):
    """Update system settings"""
    try:
        data = await request.json()
        new_config = data.get('config', {})

        # Save updated configuration
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sub_models/unified_models_config.json'))
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)

        # Reinitialize model manager with new config
        global model_manager
        model_manager = get_model_manager(config_path)

        return JSONResponse(content={"status": "success", "message": "Settings updated successfully"})

    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hardware/cameras")
def get_cameras():
    """List available cameras"""
    try:
        # In a real implementation, this would use a library like OpenCV to list cameras
        # For now, we'll return mock data
        cameras = [
            {"id": 0, "name": "Integrated Camera", "resolution": [1920, 1080]},
            {"id": 1, "name": "External Webcam", "resolution": [1280, 720]}
        ]
        return JSONResponse(content={"cameras": cameras})

    except Exception as e:
        logger.error(f"Error listing cameras: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hardware/sensors")
def get_sensors():
    """List connected sensors"""
    try:
        # In a real implementation, this would query the sensor model for connected sensors
        # For now, we'll return mock data
        sensors = [
            {"id": "temp1", "name": "Temperature Sensor", "type": "temperature", "unit": "°C"},
            {"id": "hum1", "name": "Humidity Sensor", "type": "humidity", "unit": "%"},
            {"id": "motion1", "name": "Motion Sensor", "type": "motion", "unit": ""}
        ]
        return JSONResponse(content={"sensors": sensors})

    except Exception as e:
        logger.error(f"Error listing sensors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/external/model/connect")
async def connect_external_model(request: Request):
    """Connect to an external API model"""
    try:
        data = await request.json()
        model_name = data.get('model_name')
        api_key = data.get('api_key')
        api_url = data.get('api_url')
        external_model_name = data.get('external_model_name')

        if not all([model_name, api_key, api_url, external_model_name]):
            raise HTTPException(status_code=400, detail="Missing required parameters")

        logger.info(f"Connecting external model: {external_model_name} for {model_name}")

        # In a real implementation, this would validate the connection and update the model configuration
        # For now, we'll simulate a successful connection
        time.sleep(1)  # Simulate network delay

        # Update the model configuration to use the external API
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sub_models/unified_models_config.json'))
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if model_name not in config:
            config[model_name] = {"mode": "external", "config": {}}
        else:
            config[model_name]["mode"] = "external"
            
        config[model_name]["config"]["api_key"] = "********"  # Don't store the actual key in config
        config[model_name]["config"]["api_url"] = api_url
        config[model_name]["config"]["external_model_name"] = external_model_name

        # Save the updated configuration (without the actual API key)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Here you would typically store the API key securely
        # For this example, we'll just return success

        return JSONResponse(content={"status": "success", "message": "External model connected successfully"})

    except Exception as e:
        logger.error(f"Error connecting external model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize the system
def initialize_system():
    """Initialize the entire AGI system"""
    logger.info("Starting Self Brain AGI Core initialization...")
    
    # Start model initialization in a separate thread
    model_thread = threading.Thread(target=initialize_management_model)
    model_thread.daemon = True
    model_thread.start()
    
    # Wait for management model to initialize (with timeout)
    start_time = time.time()
    timeout = 30  # 30 seconds timeout
    while management_app is None and time.time() - start_time < timeout:
        time.sleep(0.5)
    
    if management_app is None:
        logger.warning("Management model initialization timed out, continuing with limited functionality")
    
    logger.info("Self Brain AGI Core initialization completed")

# Initialize the system when the module is loaded
initialize_system()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5014))
    logger.info(f"Starting Self Brain AGI Core service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
