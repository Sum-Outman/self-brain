#!/usr/bin/env python3
"""
Debug script to check Flask routes and test endpoint
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import app, socketio
import requests
import threading
import time

def check_routes():
    """Check all registered routes"""
    with app.app_context():
        print("=== Registered Routes ===")
        for rule in app.url_map.iter_rules():
            methods = ','.join(sorted(rule.methods))
            print(f"{rule.endpoint}: {rule.rule} [{methods}]")
        
        # Check specifically for optimize endpoint
        optimize_rules = [r for r in app.url_map.iter_rules() if 'optimize' in str(r.rule)]
        if optimize_rules:
            print(f"\n=== Optimize Endpoint Found ===")
            for rule in optimize_rules:
                print(f"  {rule.endpoint}: {rule.rule} [{','.join(sorted(rule.methods))}]")
        else:
            print("\n=== Optimize Endpoint NOT Found ===")

def test_endpoint():
    """Test the endpoint"""
    time.sleep(2)
    try:
        print("\n=== Testing Endpoint ===")
        response = requests.post('http://localhost:5000/api/knowledge/optimize', timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check routes first
    check_routes()
    
    # Start a test server
    print("\n=== Starting Test Server ===")
    threading.Thread(target=lambda: app.run(host='127.0.0.1', port=5002, debug=False), daemon=True).start()
    
    # Test the endpoint
    test_endpoint()