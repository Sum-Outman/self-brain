#!/usr/bin/env python3
"""
Fix socketio configuration to ensure routes work properly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import app, socketio
import threading
import time
import requests

def test_endpoint():
    """Test the endpoint"""
    time.sleep(3)
    try:
        print("Testing POST to /api/knowledge/optimize...")
        response = requests.post('http://localhost:5009/api/knowledge/optimize', timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
            print(response.json())
        else:
            print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting Flask app with socketio...")
    
    # Start test in background
    threading.Thread(target=test_endpoint, daemon=True).start()
    
    # Run Flask app with socketio
    socketio.run(app, 
                host='0.0.0.0', 
                port=8080, 
                debug=True, 
                use_reloader=False,
                allow_unsafe_werkzeug=True)