
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
