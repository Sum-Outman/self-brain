import os
import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestWebInterface")

# Change to project root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger.info("Testing Web Interface configuration...")

# Import necessary modules to test configuration
from web_interface.app import app, socketio
import yaml

# Check port configuration
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'system_config.yaml')
port = 8080  # Default port

if os.path.exists(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            port = config.get('ports', {}).get('web_frontend', 8080)
            logger.info(f"Successfully loaded system config. Web frontend port: {port}")
    except Exception as e:
        logger.error(f"Failed to load system config: {str(e)}")
else:
    logger.warning(f"System config file not found at {config_path}")

logger.info(f"Web interface should be accessible at http://localhost:{port}")
logger.info("Test completed. You can now start the web interface with 'python web_interface/app.py'")