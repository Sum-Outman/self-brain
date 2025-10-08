
from fastapi import FastAPI
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
    
    def get_system_info(self):
        """Get system information"""
        return {
            "system_name": self.system_name,
            "version": self.version,
            "team_email": self.team_email,
            "status": self.system_status,
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
