import os
import cv2
import numpy as np
import base64
import datetime
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CameraManager')

# Determine snapshot directory
snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_interface', 'snapshots')
os.makedirs(snapshot_dir, exist_ok=True)

class CameraManager: