#!/usr/bin/env python3
"""
Simple debug script to check Flask routes
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import app

def check_routes():
    """Check all registered routes"""
    print("=== Registered Routes ===")
    with app.app_context():
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

if __name__ == "__main__":
    check_routes()