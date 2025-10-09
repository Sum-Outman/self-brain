
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
